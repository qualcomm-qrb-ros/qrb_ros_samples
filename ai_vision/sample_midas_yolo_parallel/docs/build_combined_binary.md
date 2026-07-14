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
