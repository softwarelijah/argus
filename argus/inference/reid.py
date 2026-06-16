"""Appearance embedding extractor for Re-ID.

Crops each detection from the frame and encodes it into an L2-normalised
feature vector that the tracker fuses with motion (see
:func:`argus.tracking.matching.fuse_motion_appearance`). The torch backbone is
imported lazily so the rest of the library stays CPU/import friendly.

A lightweight ResNet-18 backbone (ImageNet features, penultimate layer) is the
default so the project has no extra model dependency; for best accuracy swap in
a dedicated Re-ID network such as OSNet via ``backbone="osnet"`` with
``torchreid`` installed.
"""

from __future__ import annotations

import numpy as np


class ReIDExtractor:
    """Extract appearance embeddings for a batch of detection crops."""

    def __init__(
        self,
        backbone: str = "resnet18",
        device: str | None = None,
        input_size: tuple[int, int] = (128, 256),  # (w, h)
        weights: str | None = None,
    ) -> None:
        try:
            import torch
            import torchvision
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "torch and torchvision are required for ReIDExtractor. "
                "Install with `pip install argus-tracker[train]`."
            ) from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        if backbone == "osnet":  # pragma: no cover - requires torchreid
            import torchreid

            model = torchreid.models.build_model("osnet_x1_0", num_classes=1000, pretrained=True)
            model.classifier = torch.nn.Identity()
        else:
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if not weights else None)
            net.fc = torch.nn.Identity()  # expose the 512-d pooled features
            model = net

        if weights:
            model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model = model.to(self.device).eval()

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        return self.extract(frame, boxes)

    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Return an ``(N, D)`` array of L2-normalised embeddings for ``boxes``.

        ``boxes`` are ``(N, 4)`` xyxy in frame coordinates.
        """
        import cv2

        torch = self._torch
        if len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)

        h, w = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes.astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((self.input_size[1], self.input_size[0], 3), np.uint8))
                continue
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, self.input_size)
            crops.append(crop[:, :, ::-1])  # BGR -> RGB

        batch = np.ascontiguousarray(np.stack(crops)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        tensor = (tensor - self._mean) / self._std

        with torch.no_grad():
            feats = self.model(tensor)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)
