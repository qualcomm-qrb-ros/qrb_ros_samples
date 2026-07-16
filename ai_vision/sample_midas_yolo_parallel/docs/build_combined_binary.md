# Building midas_yolo_combined_int8_split.bin

This document describes how to build `midas_yolo_combined_int8_split.bin` —
the combined MiDaS + YOLO11n-seg INT8 context binary for the QCS9075
(Dragonwing IQ-9075 EVK).

## Background

The combined binary contains two named QNN graphs in a single shared context:

| Graph name    | Model                  | Input                    | Output                                                    |
|---------------|------------------------|---------------------------|-----------------------------------------------------------|
| `midas`       | MiDaS depth estimation | `image` float32 1×3×256×256 | `output_0` (depth map, 1×256×256)                       |
| `yolov11_seg` | YOLO11n-seg detection  | `images` float32 1×3×640×640 | `boxes`, `scores`, `coeffs`, `proto` (separate outputs) |

The YOLO graph uses a **split-output ONNX** (`yolo11n-seg-split.onnx`) — the
final `Concat` that assembles boxes/scores/mask-coefficients into one
`output0` tensor is removed, exposing them as four separate graph outputs.
This is required to prevent the QNN INT8 compiler from eliding the Sigmoid
in the class-score branch.

Neither `midas_256.onnx` nor `yolo11n-seg-split.onnx` are checked into this
repo — see [Model preparation](#model-preparation) below for how to produce
both ONNX files (and the calibration datasets referenced in the compile jobs)
from public checkpoints.

## Model preparation

Both ONNX inputs to the compile jobs below are derived from public,
off-the-shelf checkpoints — nothing here is specific to this repo or
requires files from elsewhere. You'll need:

```bash
pip install torch timm ultralytics onnx opencv-python-headless numpy qai-hub
qai-hub configure --api_token <YOUR_TOKEN>
```

### MiDaS → `midas_256.onnx`

`MiDaS_small` is loaded from `torch.hub`, traced, and exported to ONNX at a
fixed 256×256 input resolution:

```python
import torch

model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
model.eval()

input_shape = (1, 3, 256, 256)
dummy = torch.rand(input_shape)

# check_trace=False: MiDaS_small's efficientnet backbone uses dynamic "SAME"
# padding that triggers false-positive trace consistency warnings. The trace
# is correct for our fixed input size.
traced = torch.jit.trace(model, dummy, check_trace=False)

torch.onnx.export(
    traced,
    dummy,
    "midas_256.onnx",
    input_names=["image"],
    output_names=["depth"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
```

### YOLO11n-seg → `yolo11n-seg-split.onnx`

First export the stock Ultralytics checkpoint to ONNX at 640×640:

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")   # auto-downloaded by ultralytics if not present
model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
# writes yolo11n-seg.onnx
```

Then remove the final `Concat` node that assembles `output0` (boxes + scores
+ mask-coefficients) so the three branches — plus the mask-prototype output —
become four separate graph outputs. This is what prevents the QNN INT8
compiler from eliding the Sigmoid in the class-score branch (see the table
in "Background" above):

```python
import onnx
from onnx import helper, TensorProto

m = onnx.load("yolo11n-seg.onnx")

# Identify the branch tensors feeding the final Concat that produces output0.
# Exact tensor names depend on the exported opset/graph; inspect the model
# (e.g. with Netron) to confirm these for your checkpoint.
BOXES  = "/model.23/Mul_2_output_0"      # (1,  4, 8400)
SCORES = "/model.23/Sigmoid_output_0"    # (1, 80, 8400)
COEFFS = "/model.23/Concat_2_output_0"   # (1, 32, 8400)

concat_node = next(n for n in m.graph.node
                   if "output0" in n.output and n.op_type == "Concat")
m.graph.node.remove(concat_node)

# Rename the branch tensors to clean output names.
for n in m.graph.node:
    n.output[:] = ["boxes"  if o == BOXES  else
                   "scores" if o == SCORES else
                   "coeffs" if o == COEFFS else
                   "proto"  if o == "output1" else o
                   for o in n.output]
    n.input[:]  = ["boxes"  if i == BOXES  else
                   "scores" if i == SCORES else
                   "coeffs" if i == COEFFS else
                   "proto"  if i == "output1" else i
                   for i in n.input]

del m.graph.output[:]
m.graph.output.extend([
    helper.make_tensor_value_info("boxes",  TensorProto.FLOAT, [1,  4, 8400]),
    helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 80, 8400]),
    helper.make_tensor_value_info("coeffs", TensorProto.FLOAT, [1, 32, 8400]),
    helper.make_tensor_value_info("proto",  TensorProto.FLOAT, [1, 32, 160, 160]),
])

onnx.checker.check_model(m)
onnx.save(m, "yolo11n-seg-split.onnx")
```

> **Note on box coordinate space.** The split model outputs box coordinates
> already in pixel space (0–640), not normalised `[0, 1]`. Any downstream
> post-processor must detect this (e.g. treat `cx > 2.0` as pixel-space) and
> skip the usual `× input_width` scaling step.

### Calibration datasets (`midas_calib`, `yolo_calib`)

The compile jobs in [Build process](#build-process) require real-image
calibration data — omitting it (or falling back to synthetic/default
calibration) produces near-zero YOLO class scores after INT8 quantization.
Point these functions at a folder of your own representative images (for
YOLO, natural/COCO-like photos containing the object classes you care about
work well; for MiDaS, any photos are fine since it's class-agnostic depth
estimation).

```python
import pathlib
import cv2
import numpy as np
import qai_hub

def build_midas_calibration(image_dir: str, num_samples: int = 128,
                            size: int = 256) -> qai_hub.Dataset:
    """Real-image calibration inputs for MiDaS (ImageNet-normalised RGB)."""
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(sorted(pathlib.Path(image_dir).glob(ext)))

    rng = np.random.default_rng(42)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    samples = []
    while len(samples) < num_samples:
        p = image_paths[int(rng.integers(0, len(image_paths)))]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
        chw = (chw - mean) / std
        samples.append(chw[np.newaxis, ...])

    return qai_hub.upload_dataset({"image": samples}, name="midas_calib")


def build_yolo_calibration(image_dir: str, num_samples: int = 128) -> qai_hub.Dataset:
    """Real-image calibration inputs for YOLO, with light augmentation to
    diversify lighting/scale/crop across the calibration set."""
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(sorted(pathlib.Path(image_dir).glob(ext)))

    rng = np.random.default_rng(1337)
    samples = []
    while len(samples) < num_samples:
        p = image_paths[int(rng.integers(0, len(image_paths)))]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Random crop
        if img.shape[0] > 32 and img.shape[1] > 32:
            scale = float(rng.uniform(0.6, 1.0))
            h, w = img.shape[:2]
            ch, cw = max(32, int(h * scale)), max(32, int(w * scale))
            y0 = int(rng.integers(0, h - ch + 1))
            x0 = int(rng.integers(0, w - cw + 1))
            img = img[y0:y0 + ch, x0:x0 + cw]
        # Random flip
        if rng.random() < 0.5:
            img = cv2.flip(img, 1)
        # Random blur
        if rng.random() < 0.3:
            k = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        # HSV colour jitter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(rng.uniform(0.7, 1.3)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * float(rng.uniform(0.7, 1.3)), 0, 255)
        hsv[..., 0] = (hsv[..., 0] + float(rng.uniform(-8, 8))) % 180.0
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
        samples.append(chw[np.newaxis, ...])

    return qai_hub.upload_dataset({"images": samples}, name="yolo_calib")
```

With `midas_256.onnx`, `yolo11n-seg-split.onnx`, `midas_calib`, and
`yolo_calib` in hand, proceed to the compile/link jobs below.

## Build process

The binary is built as three separate QAI Hub jobs — two
`submit_compile_job` calls (one per model) followed by one
`submit_link_job` call.

### Step 1 — Compile MiDaS to a QNN DLC

```python
midas_job = qai_hub.submit_compile_job(
    model="midas_256.onnx",
    device=qai_hub.Device("Dragonwing IQ-9075 EVK"),
    name="midas (merge)",
    input_specs={"image": (1, 3, 256, 256)},
    options="--quantize_full_type int8 --quantize_io --target_runtime qnn_dlc"
            " --qnn_options context_enable_graphs=midas",
    calibration_data=midas_calib,   # uploaded qai_hub.Dataset of real images,
                                     # resized to 256x256, ImageNet-normalised
)
```

### Step 2 — Compile split-output YOLO11n-seg to a QNN DLC

```python
yolo_job = qai_hub.submit_compile_job(
    model="yolo11n-seg-split.onnx",
    device=qai_hub.Device("Dragonwing IQ-9075 EVK"),
    name="yolov11_seg split int8 dlc",
    input_specs={"images": (1, 3, 640, 640)},
    options="--quantize_full_type int8 --target_runtime qnn_dlc"
            " --qnn_options context_enable_graphs=yolov11_seg",
    calibration_data=yolo_calib,   # uploaded qai_hub.Dataset of real images,
                                    # resized/augmented to 640x640
)
```

> **YOLO does not need `--quantize_io`.** Only MiDaS uses `--quantize_io`
> (for float32 I/O compatibility with the ROS inference node). For YOLO, the
> split-output ONNX alone (removing the final Concat) is sufficient to
> prevent Sigmoid elision.

### Step 3 — Link the two DLCs into one shared-context binary

```python
link_job = qai_hub.submit_link_job(
    models=[midas_job.get_target_model(), yolo_job.get_target_model()],
    device=qai_hub.Device("Dragonwing IQ-9075 EVK"),
    name="midas+yolov11_seg split int8",
)
```

Download the linked binary and save it locally as
`midas_yolo_combined_int8_split.bin`.

### Verification (on-device, native binary)

```bash
cd ~/midas/build-native
LD_LIBRARY_PATH=. ./midas_depth bus.jpg out.jpg \
    midas_yolo_combined_int8_split.bin \
    midas_yolo_combined_int8_split.bin \
    --midas-graph midas --seg-graph yolov11_seg
```

Confirm the run reports non-trivial detections with confidences well above
the confidence threshold — near-zero scores across all classes indicate a
calibration or quantization issue (see "Compile options reference" below).

## Compile options reference

| Model | Option | Reason |
|-------|--------|--------|
| MiDaS | `--quantize_full_type int8` | INT8 weights and activations |
| MiDaS | `--quantize_io` | Float32 I/O for the ROS inference node |
| MiDaS | `--qnn_options context_enable_graphs=midas` | Sets the graph name in the DLC for later linking |
| YOLO  | `--quantize_full_type int8` | INT8 weights and activations |
| YOLO  | `--qnn_options context_enable_graphs=yolov11_seg` | Sets the graph name in the DLC for later linking |
| Both  | `--target_runtime qnn_dlc` | Intermediate DLC format required for the link job |
| Both  | real-image `calibration_data` (uploaded `qai_hub.Dataset`) | **Required** — omitting this produces near-zero YOLO class scores |
