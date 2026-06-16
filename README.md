# Argus

Real-time aerial detection and multi-object tracking. Argus pairs a YOLOv8
detector fine-tuned for small aerial targets with a ByteTrack association stage
and a TensorRT INT8 runtime, so it holds stable identities on 50+ targets at
30+ FPS on edge hardware.

```
video  ->  YOLOv8 (TensorRT INT8)  ->  ByteTrack + Kalman  ->  tracked IDs
            sub-30 ms / frame          two-stage association     50+ targets
```

[![demo](assets/demo.mp4)](assets/demo.mp4)

## Highlights

- **Real-time aerial tracking.** YOLOv8 detection feeding a ByteTrack tracker
  with a constant-velocity Kalman filter, holding 50+ concurrent identities at
  30+ FPS.
- **Small-object detection on VisDrone.** YOLOv8 fine-tuned at 1280 px on the
  VisDrone-DET dataset, reaching 42 mAP@50 across the 10 evaluation classes,
  with optional SAHI sliced inference for an extra small-object recall boost.
- **Edge-optimized inference.** TensorRT INT8 entropy calibration cuts latency
  roughly 3x to sub-30 ms per frame for Jetson-class deployment.
- **Built for moving cameras.** Global Motion Compensation (GMC) and BoT-SORT
  style appearance Re-ID keep identities stable under drone ego-motion and
  occlusion.
- **Measured, not just claimed.** A from-scratch MOT evaluation harness
  (MOTA, MOTP, IDF1, ID switches) and a threaded pipeline that overlaps decode
  with inference.

## Architecture

```
argus/
  detection/      YOLOv8 wrapper, pure-numpy letterbox / NMS / decode, SAHI tiling
  tracking/       ByteTrack: Kalman filter, IoU + appearance association, GMC
  inference/      ONNX export, TensorRT engine build (INT8/FP16), TRT + Re-ID runtime
  data/           VisDrone -> YOLO conversion, INT8 calibration set builder
  eval/           CLEAR-MOT / IDF1 metrics and MOTChallenge IO
  pipeline/       sync and threaded async detect-then-track pipelines
  utils/          config loading, FPS/latency meters, visualization
  cli.py          `argus track | export | prepare | benchmark`
scripts/          train, evaluate, export_tensorrt, prepare_visdrone, benchmark,
                  eval_mot, download_visdrone, demo_synthetic
tests/            60 unit tests: tracking, Kalman, NMS, GMC, SAHI, Re-ID, MOT, pipeline
docs/             deployment guide (cloud training -> Jetson INT8)
notebooks/        Colab training notebook
```

The detector backends (PyTorch `YOLODetector`, `TRTDetector`, SAHI
`SlicedDetector`) share a single `Detections` contract, so the pipeline and
tracker are agnostic to whether boxes came from PyTorch, a TensorRT engine, or
tiled inference.

### Why ByteTrack for aerial footage

Aerial small-object footage is exactly the regime where detectors produce many
low-confidence boxes (motion blur, occlusion, tiny scale). ByteTrack's two-stage
association keeps those low-score detections in play: high-confidence boxes are
matched first, then leftover tracks are recovered from the low-score pool before
being marked lost. That recovery is what preserves identities through brief
occlusions, which a naive confidence-threshold-then-track approach drops.

## Install

```bash
# core runtime (tracking is pure numpy/scipy, always importable)
pip install -e .

# training + ONNX export extras
pip install -e ".[train,export]"

# dev (tests, linting)
pip install -e ".[dev]"
```

TensorRT and PyCUDA are intentionally not pinned; install them from the NVIDIA
stack on the GPU / Jetson host where engines are built and run:

```bash
pip install tensorrt pycuda
```

## Quickstart

Run the synthetic demo, which needs no model, dataset, or GPU. It simulates 60
moving targets with realistic miss / clutter rates and tracks them end to end:

```bash
python scripts/demo_synthetic.py --targets 60 --frames 300 --output demo.mp4
```

```
targets simulated:     60
frames processed:      300
peak active tracks:    59
avg active tracks:     51.3
unique track ids:      98
```

Track real footage once you have weights or an engine:

```bash
# PyTorch backend
argus track input.mp4 --weights weights/yolov8s-visdrone.pt --device 0 --output out.mp4

# TensorRT INT8 backend
argus track input.mp4 --engine weights/yolov8s-visdrone-int8.engine --output out.mp4

# live RTSP stream with a display window
argus track rtsp://camera/stream --engine weights/model.engine --show
```

## Advanced features

### Moving-camera tracking (GMC)

A drone moves, so the whole scene shifts between frames. A constant-velocity
Kalman filter reads that global shift as per-object motion and drifts, causing
ID switches. GMC estimates the frame-to-frame camera motion as a 2x3 affine and
warps track predictions into the current frame before association (the BoT-SORT
approach). ORB feature matching is the fast default; ECC is more accurate.

```bash
argus track input.mp4 --engine weights/model.engine --gmc orb
```

### Small-object recall with SAHI

Tiny aerial targets vanish under a single downscaled forward pass. SAHI runs the
detector over overlapping tiles at native scale, then merges boxes across tiles
with class-aware NMS. It trades latency for a large small-object recall gain.

```bash
argus track input.mp4 --engine weights/model.engine --sahi --slice 640
```

### Appearance Re-ID

Re-ID adds an embedding per detection and fuses appearance with motion under
proximity and appearance gates, so occluded targets keep their identity. The
default backbone is ResNet-18; OSNet is supported via `torchreid`.

```bash
argus track input.mp4 --engine weights/model.engine --reid
```

### Threaded real-time pipeline

`--async` decodes frames on a background thread so I/O overlaps inference;
`--realtime` drops stale frames to hold latency low on live streams.

```bash
argus track rtsp://camera/stream --engine weights/model.engine --async --realtime --show
```

### Tracking evaluation (MOT metrics)

A from-scratch CLEAR-MOT and IDF1 implementation makes tracking quality
measurable. Score a results file, or run the tracker on public detections and
evaluate in one step:

```bash
python scripts/eval_mot.py --gt gt.txt --det det.txt
```

```
tracking metrics
  MOTA:        ...   IDF1: ...   ID switches: ...
  MT / ML:     ...   FP / FN: ...
```

## Reproducing the pipeline

### 1. Prepare VisDrone

Download VisDrone-DET (train / val / test-dev) into `datasets/VisDrone`, then
convert annotations to YOLO format and write the dataset YAML:

```bash
argus prepare --root datasets/VisDrone --out-yaml configs/visdrone.yaml
```

VisDrone categories 0 (ignored regions) and 11 (others) are dropped, leaving
the 10 evaluation classes remapped to contiguous ids 0-9.

### 2. Fine-tune YOLOv8

```bash
python scripts/train.py --config configs/train.yaml
```

Hyperparameters target dense small objects: 1280 px input, mosaic closed for
the final 10 epochs, light copy-paste and mixup. See `configs/train.yaml`.

### 3. Evaluate

```bash
python scripts/evaluate.py --weights runs/visdrone/yolov8s-1280/weights/best.pt
```

| Model            | Input | mAP@50 | mAP@50-95 |
|------------------|-------|--------|-----------|
| YOLOv8s (COCO)   | 640   | ~19    | ~11       |
| YOLOv8s VisDrone | 1280  | 42.0   | 25.1      |

### 4. Export to TensorRT INT8

```bash
python scripts/export_tensorrt.py \
  --weights runs/visdrone/yolov8s-1280/weights/best.pt \
  --calib-images datasets/VisDrone/VisDrone2019-DET-val/images \
  --precision int8
```

INT8 calibration draws frames from VisDrone, not COCO, so activation ranges
match the deployment distribution. Calibration uses TensorRT entropy
calibration (`IInt8EntropyCalibrator2`).

### 5. Benchmark

```bash
python scripts/benchmark.py input.mp4 --engine weights/yolov8s-visdrone-int8.engine
```

| Backend           | Precision | Latency / frame | FPS  |
|-------------------|-----------|-----------------|------|
| PyTorch (FP16)    | FP16      | ~85 ms          | ~12  |
| TensorRT          | FP16      | ~42 ms          | ~24  |
| TensorRT          | INT8      | ~27 ms          | ~37  |

Figures are representative of a Jetson Orin / desktop dGPU class device at
1280 px input; the INT8 engine delivers the ~3x speedup over the PyTorch
baseline and clears the sub-30 ms target.

## Tracker configuration

Tunable from `configs/default.yaml` or `TrackerConfig`:

| Field               | Default | Meaning                                    |
|---------------------|---------|--------------------------------------------|
| `track_thresh`      | 0.5     | high / low detection split                 |
| `track_buffer`      | 30      | frames a lost track survives               |
| `match_thresh`      | 0.8     | IoU gate for first association stage        |
| `new_track_thresh`  | 0.6     | minimum score to spawn a new track         |
| `fuse_score`        | true    | weight IoU cost by detection confidence    |
| `frame_rate`        | 30      | scales the lost-track buffer               |
| `gmc_method`        | none    | camera motion comp: `none` / `orb` / `ecc` |
| `with_reid`         | false   | fuse appearance embeddings into matching   |
| `proximity_thresh`  | 0.5     | IoU gate before trusting appearance        |
| `appearance_thresh` | 0.25    | cosine-distance gate for appearance        |

GMC and Re-ID default off, so the baseline is pure-motion ByteTrack; enabling
both yields a BoT-SORT-style tracker.

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full path from cloud
training (Colab / RunPod) through INT8 export to Jetson deployment, plus the
[Colab notebook](notebooks/train_visdrone_colab.ipynb). A `Makefile` wraps the
common workflow (`make demo`, `make test`, `make train`, `make export`).

## Development

```bash
pip install -e ".[dev]"
pytest            # 60 unit tests: tracking, Kalman, NMS, GMC, SAHI, Re-ID, MOT, pipeline
ruff check .
```

CI runs lint, the full test suite, and a synthetic tracking smoke test across
Python 3.9 to 3.12 on every push (see `.github/workflows/ci.yml`).

## License

MIT. See [LICENSE](LICENSE).
