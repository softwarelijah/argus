"""Export YOLOv8 weights to ONNX and build INT8 / FP16 TensorRT engines.

The INT8 path uses entropy calibration over a representative set of VisDrone
frames. Calibration data drives the per-tensor activation ranges, so it should
be drawn from the deployment distribution (aerial imagery), not COCO.
"""

from __future__ import annotations

import os

import numpy as np


def export_onnx(
    weights: str,
    out_path: str | None = None,
    imgsz: int = 1280,
    opset: int = 12,
    simplify: bool = True,
) -> str:
    """Export an Ultralytics checkpoint to ONNX and return the file path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("ultralytics is required to export ONNX.") from exc

    model = YOLO(weights)
    exported = model.export(
        format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=False
    )
    if out_path and exported != out_path:
        os.replace(exported, out_path)
        return out_path
    return exported


class _EntropyCalibrator:
    """INT8 entropy calibrator feeding letterboxed VisDrone frames.

    Defined as a factory so the heavy ``tensorrt`` base class is only resolved
    when TensorRT is actually present.
    """

    def __new__(cls, *args, **kwargs):  # pragma: no cover - GPU-only path
        import pycuda.driver as cuda
        import tensorrt as trt

        class Impl(trt.IInt8EntropyCalibrator2):
            def __init__(self, calib_frames, input_shape, cache_file):
                super().__init__()
                self.cache_file = cache_file
                self.frames = calib_frames
                self.batch_size = input_shape[0]
                self.input_shape = input_shape
                self.index = 0
                self.device_input = cuda.mem_alloc(int(np.prod(input_shape)) * 4)

            def get_batch_size(self):
                return self.batch_size

            def get_batch(self, names):
                if self.index + self.batch_size > len(self.frames):
                    return None
                batch = self.frames[self.index : self.index + self.batch_size]
                batch = np.ascontiguousarray(batch, dtype=np.float32)
                cuda.memcpy_htod(self.device_input, batch)
                self.index += self.batch_size
                return [int(self.device_input)]

            def read_calibration_cache(self):
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

        return Impl(*args, **kwargs)


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "int8",
    imgsz: int = 1280,
    workspace_gb: int = 4,
    calib_frames: np.ndarray | None = None,
    calib_cache: str = "calibration.cache",
) -> str:
    """Build a TensorRT engine from ONNX.

    ``precision`` is one of ``"fp32"``, ``"fp16"`` or ``"int8"``. For INT8,
    ``calib_frames`` must be an ``(N, 3, imgsz, imgsz)`` float32 array of
    preprocessed calibration images.
    """
    try:  # pragma: no cover - GPU-only path
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("tensorrt is required to build an engine.") from exc

    logger = trt.Logger(trt.Logger.INFO)  # pragma: no cover
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"failed to parse ONNX:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("this platform does not support fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_frames is None:
            raise ValueError("INT8 build requires calib_frames")
        config.int8_calibrator = _EntropyCalibrator(
            calib_frames, (1, 3, imgsz, imgsz), calib_cache
        )

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path
