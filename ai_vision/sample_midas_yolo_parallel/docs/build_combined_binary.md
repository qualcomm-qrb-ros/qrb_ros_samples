# Building midas_yolo_combined_int8_split.bin

This document describes how `midas_yolo_combined_int8_split.bin` — the
**working** combined MiDaS + YOLO11n-seg INT8 context binary for the QCS9075
(Dragonwing IQ-9075 EVK) — was actually produced, reconstructed from the QAI
Hub job history (job IDs below can be opened at
`https://workbench.aihub.qualcomm.com/jobs/<id>/`).

> **This is not the same binary as `midas_yolo_combined_int8_split_v2.bin`.**
> The `_v2` binary (built by `compile_int8_split.py`) reproduces the
> zero-score bug (see "Known bad process" below). `_v2` is currently
> deployed to `/opt/model/midas_yolo_combined.bin` on-device and needs to be
> replaced with a rebuild following the process in this document.

## Background

The combined binary contains two named QNN graphs in a single shared context:

| Graph name    | Model                  | Input                    | Output                                                    |
|---------------|------------------------|---------------------------|-----------------------------------------------------------|
| `midas`       | MiDaS depth estimation | `image` float32 1×3×256×256 | `output_0` (depth map, 1×256×256)                       |
| `yolov11_seg` | YOLO11n-seg detection  | `images` float32 1×3×640×640 | `boxes`, `scores`, `coeffs`, `proto` (separate outputs) |

The YOLO graph uses the **split-output ONNX** (`yolo11n-seg-split.onnx`, see
`docs/yolo_int8_fix.md`) — the final `Concat` that assembles
boxes/scores/mask-coefficients into one `output0` tensor is removed, exposing
them as four separate graph outputs. This is required to prevent the QNN
INT8 compiler from eliding the Sigmoid in the class-score branch.

## Actual build process (reconstructed from QAI Hub job history)

The binary was built as **three separate QAI Hub jobs** — two
`submit_compile_job` calls (one per model) followed by one
`submit_link_job` call — the same pattern used by `merge_models.py`, **not**
via `compile_int8_split.py`'s `submit_compile_and_link_jobs` helper (see
"Known bad process" below for why that path produces a broken binary).

### Step 1 — Compile MiDaS to a QNN DLC

```python
midas_job = qai_hub.submit_compile_job(
    model="midas_256.onnx",
    device=qai_hub.Device("Dragonwing IQ-9075 EVK"),
    name="midas (merge)",
    input_specs={"image": (1, 3, 256, 256)},
    options="--quantize_full_type int8 --quantize_io --target_runtime qnn_dlc"
            " --qnn_options context_enable_graphs=midas",
    calibration_data=midas_calib,   # uploaded qai_hub.Dataset, name "midas_calib"
)
```

- Job ID: `jgn7e69rp` (2026-07-06 10:38:53, SUCCESS)
- Calibration dataset: `midas_calib` (`d74njwj02`) — built by
  `build_midas_calibration()` in `merge_models.py` (real images from repo
  root, resized to 256×256, ImageNet-normalised).
- Output DLC: `mqpyxr00q`

### Step 2 — Compile split-output YOLO11n-seg to a QNN DLC

```python
yolo_job = qai_hub.submit_compile_job(
    model="yolo11n-seg-split.onnx",
    device=qai_hub.Device("Dragonwing IQ-9075 EVK"),
    name="yolov11_seg split int8 dlc",
    input_specs={"images": (1, 3, 640, 640)},
    options="--quantize_full_type int8 --target_runtime qnn_dlc"
            " --qnn_options context_enable_graphs=yolov11_seg",
    calibration_data=yolo_calib,   # uploaded qai_hub.Dataset, name "yolo_split_int8_calib"
)
```

- Job ID: `jgjwl1q85` (2026-07-07 09:29:43, SUCCESS)
- Calibration dataset: `yolo_split_int8_calib` (`d26qgo1z7`) — real images
  (bus.jpg etc.) from the repo root, resized/augmented to 640×640, matching
  the augmentation logic in `build_yolo_calibration()` in `merge_models.py`.
- Output DLC: `mqkxdrx7n`

> **Important — `--quantize_io` is NOT used for YOLO in the working build.**
> This directly contradicts `compile_int8_split.py`'s docstring claim that
> `--quantize_io` is "required even for the split ONNX." The working job
> (`jgjwl1q85`) omits `--quantize_io` entirely for the YOLO compile step and
> produces correct, non-zero class scores. Only MiDaS uses `--quantize_io`
> (for float32 I/O compatibility with the ROS inference node).

### Step 3 — Link the two DLCs into one shared-context binary

```python
link_job = qai_hub.submit_link_job(
    models=[midas_job.get_target_model(), yolo_job.get_target_model()],
    device=qai_hub.Device("Dragonwing IQ-9075 EVK"),
    name="midas+yolov11_seg split int8",
)
```

- Job ID: `jp1v8312p` (2026-07-07 09:38:10, SUCCESS)
- Inputs: `job_jgn7e69rp_optimized_dlc` (`mqpyxr00q`) +
  `job_jgjwl1q85_optimized_dlc` (`mqkxdrx7n`)
- Output: `job_jp1v8312p_linked_bin` (`mqyw7gw5n`) — downloaded locally as
  `midas_yolo_combined_int8_split.bin`.

### Verification (on-device, native binary)

```bash
cd ~/midas/build-native
LD_LIBRARY_PATH=. ./midas_depth bus.jpg out.jpg \
    midas_yolo_combined_int8_split.bin \
    midas_yolo_combined_int8_split.bin \
    --midas-graph midas --seg-graph yolov11_seg
```

Confirmed output on `bus.jpg`: `detections=5` — person 85.3%, person 85.3%,
person 83.3%, bus 75.7%, person 38.3% (matches the fp16 reference binary's
detections and confidences).

## Known bad process: why `_v2.bin` is broken

`midas_yolo_combined_int8_split_v2.bin` was built ~2 hours later by
`compile_int8_split.py`, using `qai_hub.submit_compile_and_link_jobs()`
instead of separate `submit_compile_job` + `submit_link_job` calls:

```python
calib_ds = build_calibration_dataset(calib_samples)   # built, but never passed to compile

compile_jobs, link_job = qai_hub.submit_compile_and_link_jobs(
    models=[midas_onnx, yolo_onnx],
    device=device,
    name="midas+yolov11_seg_int8_split_v2",
    input_specs=[{"image": MIDAS_INPUT_SHAPE}, {"images": YOLO_INPUT_SHAPE}],
    graph_names=[MIDAS_GRAPH_NAME, YOLO_GRAPH_NAME],
    compile_options=[MIDAS_COMPILE_OPTS, YOLO_COMPILE_OPTS],
)
```

**Root cause: `qai_hub.submit_compile_and_link_jobs()` has no
`calibration_data` parameter.** (Confirmed via
`inspect.signature(qai_hub.submit_compile_and_link_jobs)` — it only accepts
`models, device, name, input_specs, graph_names, compile_options,
link_options, retry`.) `compile_int8_split.py` builds and uploads a
calibration dataset (`calib_ds`) but then never passes it into the call — it
compiles with **no calibration data at all**, falling back to QAI Hub's
default calibration behaviour.

This is confirmed directly from the job records — the resulting compile jobs
show `calibration_dataset: None`:

| Job | ID | Date | `calibration_dataset` |
|-----|----|------|------------------------|
| MiDaS compile | `jpy7d7qrp` | 2026-07-07 11:34:23 | `None` |
| YOLO compile  | `jp0vrvd2g` | 2026-07-07 11:34:24 | `None` |
| Link          | `jgl1k1oe5` | 2026-07-07 11:35:25 | n/a |

Without real-image calibration data, the YOLO INT8 quantizer picks a
different (bad) dynamic range for the class-score tensor. The result: all
class scores collapse to a narrow near-zero band (observed max score across
8400 anchors × 80 classes ≈ 0.04 on `bus.jpg`, vs. ≈ 0.85 for the correctly
calibrated build), well below the 0.25 confidence threshold — reproducing a
variant of the original Sigmoid-elision symptom, but from a different cause
(missing calibration, not the Concat/Sigmoid fusion this binary's split ONNX
was specifically designed to avoid).

**Fix direction:** Use `submit_compile_job` + `submit_link_job` directly (as
`merge_models.py` already does, and as documented in "Actual build process"
above) instead of `submit_compile_and_link_jobs`, so `calibration_data` is
actually applied. `compile_int8_split.py` should be corrected or retired in
favor of a `merge_models.py`-style two-step compile.

## Compile options reference (working build)

| Model | Option | Reason |
|-------|--------|--------|
| MiDaS | `--quantize_full_type int8` | INT8 weights and activations |
| MiDaS | `--quantize_io` | Float32 I/O for the ROS inference node |
| MiDaS | `--qnn_options context_enable_graphs=midas` | Sets the graph name in the DLC for later linking |
| YOLO  | `--quantize_full_type int8` | INT8 weights and activations |
| YOLO  | `--qnn_options context_enable_graphs=yolov11_seg` | Sets the graph name in the DLC for later linking |
| Both  | `--target_runtime qnn_dlc` | Intermediate DLC format required for the link job |
| Both  | real-image `calibration_data` (uploaded `qai_hub.Dataset`) | **Required** — omitting this (as `compile_int8_split.py` inadvertently does) produces near-zero YOLO class scores |

> **YOLO does NOT need `--quantize_io`.** The working compile job
> (`jgjwl1q85`) omits it. `docs/yolo_int8_fix.md`'s claim that
> `--quantize_io` is "required even for the split ONNX" is not supported by
> the job that actually produced correct scores — the split-output ONNX
> alone (removing the final Concat) was sufficient to prevent Sigmoid
> elision. `--quantize_io` may still be worth keeping for I/O type
> consistency with the ROS pipeline, but is not what fixes the zero-score
> bug.

## ROS pipeline dets=0

As of July 2026, the ROS pipeline (`qrb_ros_nn_inference` shared inference
node) reports `dets=0` even with a correctly-calibrated binary, while the
native `midas_depth` binary produces correct detections from the same
binary. See prior investigation notes: the QNN HTP runtime does not appear
to write to the scores output buffer when called through the ROS inference
node's `graphExecute` path (raw bytes all `0x00`), while boxes/coeffs/proto
buffers are written correctly. Suspected cause: buffer allocation strategy
(`malloc` vs. `rpcmem`/ION) in the ROS inference node. Not yet fixed;
workaround is to use the native `midas_depth` binary.

## QAI Hub job IDs (working `midas_yolo_combined_int8_split.bin` build)

| Job | ID | Description |
|-----|----|-------------|
| MiDaS compile | `jgn7e69rp` | `midas_256.onnx` -> DLC, INT8+quantize_io, calib=`midas_calib` (`d74njwj02`) |
| YOLO compile  | `jgjwl1q85` | `yolo11n-seg-split.onnx` -> DLC, INT8 (no quantize_io), calib=`yolo_split_int8_calib` (`d26qgo1z7`) |
| Link          | `jp1v8312p` | Combined context binary, graph names: midas + yolov11_seg |

## QAI Hub job IDs (broken `_v2.bin` build — for reference, do not repeat)

| Job | ID | Description |
|-----|----|-------------|
| Compile (via `submit_compile_and_link_jobs`) | `jpy7d7qrp` | `midas_256.onnx`, `calibration_dataset=None` |
| Compile (via `submit_compile_and_link_jobs`) | `jp0vrvd2g` | `yolo11n-seg-split.onnx`, `calibration_dataset=None` |
| Link          | `jgl1k1oe5` | Combined context binary from the two uncalibrated DLCs above |
