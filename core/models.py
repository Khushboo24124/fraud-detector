"""Model registry: every pre-trained model is loaded once, lazily, from a local cache.

Revisions are pinned in settings.yaml so scores are reproducible. Run
`python scripts/download_models.py` before a demo so nothing is fetched over venue Wi-Fi.
"""
from __future__ import annotations

import threading
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .config import MODEL_DIR, ensure_dirs, settings

_lock = threading.Lock()  # torch modules are shared singletons; serialise inference


def _hf_kwargs(name: str) -> dict:
    m = settings()["models"][name]
    return {"revision": m["revision"], "cache_dir": str(MODEL_DIR / "hf")}


class AIImageDetector:
    """Common interface: list of PIL images -> list of synthetic probabilities."""
    key: str          # settings.yaml models.<key>
    name: str
    version: str

    def predict(self, images: list[Image.Image]) -> list[float]:
        raise NotImplementedError


class CommunityForensicsDetector(AIImageDetector):
    """ViT-S trained on 4,803 generators (CVPR 2025). Single sigmoid logit = P(fake)."""
    key = "community_forensics"

    def __init__(self) -> None:
        import torch
        from transformers import AutoImageProcessor, ViTForImageClassification

        m = settings()["models"]["community_forensics"]
        self.name = "CommunityForensics-ViT-S/384"
        self.version = m["revision"][:8]
        # Do NOT override size: the processor does shortest_edge=440 + center-crop 384.
        self.processor = AutoImageProcessor.from_pretrained(m["repo"], **_hf_kwargs("community_forensics"))
        self.model = ViTForImageClassification.from_pretrained(m["repo"], **_hf_kwargs("community_forensics")).eval()
        self._torch = torch

    def predict(self, images):
        with _lock, self._torch.no_grad():
            inputs = self.processor(images=[i.convert("RGB") for i in images], return_tensors="pt")
            logits = self.model(**inputs).logits.reshape(-1)
            return self._torch.sigmoid(logits).tolist()


class SiglipAIDetector(AIImageDetector):
    """SigLIP fine-tuned on 120k AI vs human images. Labels {0: ai, 1: hum}."""
    key = "siglip_ai_vs_human"

    def __init__(self) -> None:
        import torch
        from transformers import AutoImageProcessor, SiglipForImageClassification

        m = settings()["models"]["siglip_ai_vs_human"]
        self.name = "SigLIP-ai-vs-human"
        self.version = m["revision"][:8]
        self.processor = AutoImageProcessor.from_pretrained(m["repo"], **_hf_kwargs("siglip_ai_vs_human"))
        self.model = SiglipForImageClassification.from_pretrained(m["repo"], **_hf_kwargs("siglip_ai_vs_human")).eval()
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ai_index = next(i for i, l in labels.items() if l.startswith("ai"))
        self._torch = torch

    def predict(self, images):
        with _lock, self._torch.no_grad():
            inputs = self.processor(images=[i.convert("RGB") for i in images], return_tensors="pt")
            probs = self._torch.softmax(self.model(**inputs).logits, dim=-1)
            return probs[:, self.ai_index].tolist()


@lru_cache(maxsize=1)
def ai_detectors() -> tuple[AIImageDetector, ...]:
    return (CommunityForensicsDetector(), SiglipAIDetector())


def _opencv_zoo_file(name: str) -> Path:
    ensure_dirs()
    m = settings()["models"][name]
    path = MODEL_DIR / m["file"]
    if not path.exists():
        urllib.request.urlretrieve(m["url"], path)
    return path


class FaceEngine:
    """OpenCV Zoo YuNet (detection) + SFace (128-d embedding). Pure OpenCV, no compiler needed."""

    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(_opencv_zoo_file("face_detector")), "", (320, 320), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(_opencv_zoo_file("face_recognizer")), "")
        self.version = "yunet2023mar+sface2021dec"

    def faces(self, image: Image.Image, max_side: int = 1280) -> list[dict]:
        """Return faces sorted by area: bbox, score, aligned crop and L2-normalised embedding."""
        cv2 = self._cv2
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        scale = min(1.0, max_side / max(bgr.shape[:2]))
        if scale < 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        with _lock:
            self.detector.setInputSize((w, h))
            _, found = self.detector.detect(bgr)
            out = []
            for f in (found if found is not None else []):
                aligned = self.recognizer.alignCrop(bgr, f)
                emb = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
                emb /= np.linalg.norm(emb) + 1e-9
                x, y, fw, fh = (f[:4] / scale).astype(int).tolist()
                out.append({"bbox": [x, y, fw, fh], "score": float(f[14]), "embedding": emb,
                            "crop": Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))})
        return sorted(out, key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)


@lru_cache(maxsize=1)
def face_engine() -> FaceEngine:
    return FaceEngine()


@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr(image: Image.Image, max_side: int = 2000) -> list[dict]:
    """Run OCR; returns lines with text, confidence and box [x0, y0, x1, y1] in original pixels."""
    rgb = np.asarray(image.convert("RGB"))
    scale = min(1.0, max_side / max(rgb.shape[:2]))
    if scale < 1.0:
        import cv2
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    with _lock:
        result, _ = ocr_engine()(rgb)
    lines = []
    for box, text, conf in result or []:
        pts = np.array(box, dtype=np.float32) / scale
        lines.append({"text": text, "conf": float(conf),
                      "box": [float(pts[:, 0].min()), float(pts[:, 1].min()),
                              float(pts[:, 0].max()), float(pts[:, 1].max())]})
    return lines
