#!/usr/bin/env bash
#
# One-shot Argus GPU pipeline for a RunPod (or any x86 CUDA) box.
# Runs: install -> download VisDrone -> train -> evaluate -> export INT8 ->
# benchmark, writing every result into results/.
#
# Quick validation first (recommended, ~20 min, a few cents):
#     SMOKE=1 bash scripts/cloud/runpod_bootstrap.sh
#
# Full run for the real numbers:
#     bash scripts/cloud/runpod_bootstrap.sh
#
# Tunables (env vars):
#     SMOKE=1        run 10 epochs at 960px for a fast end-to-end check
#     EPOCHS=100     override training epochs
#     BATCH=16       override batch size (drop to 8 on a 16 GB GPU)
#     IMGSZ=1280     training/inference resolution
#     MODEL=yolov8s.pt  base checkpoint
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
mkdir -p results weights

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-1280}"
MODEL="${MODEL:-yolov8s.pt}"
RUN_NAME="yolov8s-${IMGSZ}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo ">> SMOKE mode: short run to validate the whole pipeline"
  EPOCHS=10
  IMGSZ=960
  RUN_NAME="smoke"
fi

echo "==> 1/6 environment"
nvidia-smi || { echo "no GPU visible; is this a GPU pod?"; exit 1; }
python -m pip install --upgrade pip
pip install -e ".[train,export]"
pip install tensorrt || echo "WARN: tensorrt pip install failed; INT8 export step may be skipped"

echo "==> 2/6 dataset"
python scripts/download_visdrone.py --root datasets/VisDrone
python scripts/prepare_visdrone.py --root datasets/VisDrone

echo "==> 3/6 train (${EPOCHS} epochs @ ${IMGSZ}px, batch ${BATCH})"
python scripts/train.py --config configs/train.yaml \
  --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ" --device 0 --name "$RUN_NAME" \
  2>&1 | tee results/train.log

BEST="runs/visdrone/${RUN_NAME}/weights/best.pt"
echo "best weights: $BEST"

echo "==> 4/6 evaluate mAP"
python scripts/evaluate.py --weights "$BEST" --imgsz "$IMGSZ" 2>&1 | tee results/eval.log

echo "==> 5/6 export INT8 TensorRT engine"
if python scripts/export_tensorrt.py --weights "$BEST" \
     --calib-images datasets/VisDrone/VisDrone2019-DET-val/images \
     --precision int8 --imgsz "$IMGSZ" 2>&1 | tee results/export.log; then
  ENGINE=$(ls -t weights/*int8*.engine 2>/dev/null | head -1 || true)
else
  echo "custom export failed; falling back to ultralytics native engine export"
  yolo export model="$BEST" format=engine int8=True imgsz="$IMGSZ" \
    data=configs/visdrone.yaml 2>&1 | tee -a results/export.log
  ENGINE=$(ls -t "runs/visdrone/${RUN_NAME}/weights/"*.engine 2>/dev/null | head -1 || true)
fi
echo "engine: ${ENGINE:-<none>}"

echo "==> 6/6 benchmark (synthetic ${IMGSZ}-input frames)"
python scripts/benchmark.py --synthetic 300 --weights "$BEST" --imgsz "$IMGSZ" --device 0 \
  2>&1 | tee results/benchmark_pytorch.log
if [[ -n "${ENGINE:-}" ]]; then
  python scripts/benchmark.py --synthetic 300 --engine "$ENGINE" --imgsz "$IMGSZ" \
    2>&1 | tee results/benchmark_tensorrt.log
fi

echo
echo "==================  RESULTS SUMMARY  =================="
echo "-- mAP --";        grep -E "mAP@50" results/eval.log || true
echo "-- PyTorch FPS --"; grep -E "fps|detect_ms" results/benchmark_pytorch.log || true
if [[ -f results/benchmark_tensorrt.log ]]; then
  echo "-- TensorRT INT8 FPS --"; grep -E "fps|detect_ms" results/benchmark_tensorrt.log || true
fi
echo "======================================================"
echo "Done. Copy back: $BEST, ${ENGINE:-(no engine)}, and the results/ folder."
echo "Pull with runpodctl, or: tar czf argus_results.tgz results $BEST ${ENGINE:-}"
