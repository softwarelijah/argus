# Deployment guide

This covers training in the cloud and deploying the INT8 engine to a Jetson.

## 1. Train in the cloud (GPU)

The CPU install runs tracking, tests and the synthetic demo, but training and
TensorRT export need an NVIDIA GPU. Any of Colab, Lambda, RunPod or a local
CUDA box works.

```bash
# on the GPU host
git clone <your-fork> argus && cd argus
pip install -e ".[train,export]"

# fetch and convert VisDrone (see scripts/download_visdrone.py for links)
python scripts/download_visdrone.py --root datasets/VisDrone
python scripts/prepare_visdrone.py --root datasets/VisDrone

# fine-tune YOLOv8s at 1280 px
python scripts/train.py --config configs/train.yaml

# evaluate mAP
python scripts/evaluate.py --weights runs/visdrone/yolov8s-1280/weights/best.pt
```

A Colab notebook that runs the whole flow is in
[`notebooks/train_visdrone_colab.ipynb`](../notebooks/train_visdrone_colab.ipynb).

### Hardware notes

- 1280 px training is memory hungry. On a 16 GB GPU use `batch: 8`; on 24 GB+
  the default `batch: 16` is fine. Reduce `imgsz` to 960 if you are tight.
- Expect roughly 100 epochs to converge on VisDrone; enable `patience` early
  stopping (already set in `configs/train.yaml`).

## 2. Export the TensorRT engine

Run on the **same compute capability** you will deploy to (engines are not
portable across GPU architectures). On the Jetson itself is safest.

```bash
pip install tensorrt pycuda    # provided by JetPack on Jetson
python scripts/export_tensorrt.py \
  --weights runs/visdrone/yolov8s-1280/weights/best.pt \
  --calib-images datasets/VisDrone/VisDrone2019-DET-val/images \
  --precision int8
```

INT8 calibration uses VisDrone frames so activation ranges match the aerial
deployment distribution. The calibration cache is written next to the engine
and reused on subsequent builds.

## 3. Deploy on Jetson

```bash
# build the device image (on the Jetson)
docker build -f Dockerfile.jetson -t argus:jetson .

# run on a CSI/USB camera with a display
docker run --runtime nvidia -it --rm \
  --device /dev/video0 -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $PWD/weights:/app/weights \
  argus:jetson track /dev/video0 \
    --engine weights/yolov8s-visdrone-int8.engine --show --async --realtime
```

### Tips for edge throughput

- Use `--async --realtime` so frame decode overlaps inference and stale frames
  are dropped rather than queued.
- Put the Jetson in its max power mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
- Profile per-stage latency with `python scripts/benchmark.py <video> --engine ...`
  and confirm the detect stage is under the 30 ms budget.
- For moving-camera footage enable `--gmc orb`; it adds a few ms but sharply
  reduces ID switches.

## 4. Verify the targets

| Target                 | How to check                                             |
|------------------------|---------------------------------------------------------|
| 30+ FPS                | `scripts/benchmark.py` reports `fps`                     |
| sub-30 ms detection    | `scripts/benchmark.py` reports `detect_ms`              |
| 50+ simultaneous tracks| `scripts/demo_synthetic.py --targets 60`                |
| 42 mAP@50              | `scripts/evaluate.py`                                    |
| tracking MOTA / IDF1   | `scripts/eval_mot.py --gt gt.txt --det det.txt`        |
