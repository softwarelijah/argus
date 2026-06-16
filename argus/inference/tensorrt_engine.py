"""Minimal TensorRT runtime wrapper.

Loads a serialized ``.engine`` file and runs synchronous inference over a CUDA
stream. ``tensorrt`` and ``pycuda`` are imported lazily so the package installs
and imports cleanly on a CPU-only machine; they are only needed on the GPU /
Jetson host where the engine actually runs.
"""

from __future__ import annotations

import numpy as np


class TRTEngine:
    """Wraps a deserialized TensorRT engine and its execution context."""

    def __init__(self, engine_path: str) -> None:
        try:
            import pycuda.autoinit  # noqa: F401  (initialises the CUDA context)
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - GPU-only path
            raise ImportError(
                "tensorrt and pycuda are required to run a TensorRT engine. "
                "Install them on the GPU/Jetson host."
            ) from exc

        self._cuda = cuda
        self._trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.bindings: list[int] = []
        self.stream = cuda.Stream()
        self._allocate()

    def _allocate(self) -> None:
        cuda, trt = self._cuda, self._trt
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            binding = {"name": name, "shape": shape, "dtype": dtype,
                       "host": host_mem, "device": device_mem}
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def infer(self, blob: np.ndarray) -> list[np.ndarray]:
        """Run a forward pass and return a list of output arrays."""
        cuda = self._cuda
        inp = self.inputs[0]
        np.copyto(inp["host"], blob.ravel())
        cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        results = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        for out in self.outputs:
            results.append(out["host"].reshape(out["shape"]).copy())
        return results

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            del self.context
            del self.engine
        except Exception:
            pass
