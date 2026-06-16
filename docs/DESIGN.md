# Design notes

Why Argus is built the way it is. This complements the README, which covers
usage; this file covers the engineering decisions.

## Goals

1. Real-time multi-object tracking on aerial (moving-camera) video.
2. Small-object accuracy on the VisDrone class set.
3. Edge deployability (Jetson-class, INT8).
4. Reproducible and verifiable without specialised hardware.

Goal 4 shapes a lot of the structure: the tracking, evaluation and analytics
stacks are pure numpy/scipy and fully tested on CPU, while the GPU-only pieces
(PyTorch, TensorRT) sit behind lazy imports so the package installs and the
test suite runs anywhere.

## Layering

```
            +-------------------+
 video ---> |    Detector       |  YOLODetector | ORTDetector | TRTDetector
            +---------+---------+   (optionally wrapped by SlicedDetector / SAHI)
                      | Detections (N x [x1,y1,x2,y2,score,cls])
                      v
            +-------------------+
            |    ByteTracker     |  Kalman predict -> GMC warp -> 2-stage match
            +---------+---------+   (+ optional appearance Re-ID)
                      | list[STrack]
                      v
            +-------------------+
            | Analytics / Eval   |  counting, zones, speed | MOTA/IDF1
            +-------------------+
```

The single `Detections` contract is the key seam: every detector backend
produces it and the tracker only consumes it, so PyTorch, ONNX Runtime,
TensorRT and SAHI are fully interchangeable. The pipeline duck-types the
detector (`detect(frame) -> Detections`), so no backend is special-cased.

## Tracking: why ByteTrack, plus GMC and Re-ID

**ByteTrack.** Aerial small-object detection produces many low-confidence boxes
(blur, occlusion, tiny scale). ByteTrack's two-stage association keeps those in
play: confident boxes match first, then leftover tracks are recovered from the
low-score pool before being dropped. That recovery is the main defence against
ID fragmentation in this regime.

**Global Motion Compensation.** A constant-velocity Kalman filter assumes a
static camera. A moving drone violates that, and the global scene shift gets
absorbed as per-object motion, causing drift and ID switches. GMC estimates the
frame-to-frame affine and warps the track state (position and velocity blocks of
the Kalman mean, and the covariance) into the current frame before association.
ORB feature matching is the fast default; ECC is the accurate fallback.

**Appearance Re-ID.** Motion alone cannot re-link a track after a long
occlusion. An embedding per detection, fused with IoU under proximity and
appearance gates (the BoT-SORT rule), lets identities survive gaps. It is
opt-in because it needs an embedding model and adds latency.

Together, GMC + Re-ID on top of ByteTrack is essentially BoT-SORT, assembled
from interchangeable parts rather than a fork.

## Detection: SAHI

A single 1280 px forward pass still downsamples a 4K drone frame enough to lose
the smallest targets. SAHI runs the detector on overlapping native-scale tiles
and merges with class-aware NMS. It is a wrapper, not a new model, so it
composes with any backend.

## Inference path

Three backends, one contract:

- `YOLODetector` (PyTorch) for training-time and GPU inference.
- `ORTDetector` (ONNX Runtime) for dependency-light CPU inference. This is what
  lets the whole pipeline run on real video with no GPU and no PyTorch.
- `TRTDetector` (TensorRT) for INT8 edge deployment.

INT8 calibration uses VisDrone frames, not COCO, so activation ranges match the
deployment distribution. Engines are architecture-specific, so they are built on
(or for) the target device.

## Evaluation

Metrics are implemented from scratch (CLEAR-MOT, IDF1) rather than pulling a
heavy dependency, both to keep the install light and to make the definitions
auditable. A synthetic MOT generator produces hard sequences (births/deaths,
misses, clutter) so the tracker and metrics can be exercised end to end on CPU,
and the same `evaluate()` runs against real MOTChallenge files.

## Performance

The sync pipeline is simple and deterministic. The async pipeline adds a
background decode thread feeding a bounded queue, so frame decode (often the
hidden serial cost on H.264/RTSP) overlaps inference. A realtime mode drops
stale frames so live latency does not grow under load.

## Testing strategy

Every pure-Python component is unit tested, and the integration points
(detector to tracker to analytics) are covered with fake backends so they run
without models. The synthetic MOT regression test locks in tracker quality
floors (MOTA/IDF1) so association changes cannot silently regress.
