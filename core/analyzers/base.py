"""Analyzer plug-in contract and the shared per-claim context."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..config import BLOB_DIR, ensure_dirs
from ..schemas import ClaimInfo, EvidenceItem, Kind, Signal


@dataclass
class Context:
    """Everything an analyzer may read. Expensive decodes are cached per item."""
    claim_id: str
    info: ClaimInfo
    items: list[EvidenceItem]
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def slot(self, item: EvidenceItem) -> dict[str, Any]:
        return self.cache.setdefault(item.id, {})

    def image(self, item: EvidenceItem) -> Image.Image:
        """Decoded RGB image with EXIF orientation applied (images only)."""
        s = self.slot(item)
        if "image" not in s:
            img = Image.open(item.path)
            s["raw_image"] = img
            s["image"] = ImageOps.exif_transpose(img).convert("RGB")
        return s["image"]

    def raw_image(self, item: EvidenceItem) -> Image.Image:
        self.image(item)
        return self.slot(item)["raw_image"]

    def pages(self, item: EvidenceItem, dpi: int = 150, max_pages: int = 10) -> list[Image.Image]:
        """Page images for any document: the image itself, or rendered PDF pages."""
        if item.kind is Kind.IMAGE:
            return [self.image(item)]
        s = self.slot(item)
        key = f"pages@{dpi}"
        if key not in s:
            import pymupdf as fitz
            with fitz.open(item.path) as doc:
                s[key] = [
                    Image.open(io.BytesIO(p.get_pixmap(dpi=dpi).tobytes("png"))).convert("RGB")
                    for p in list(doc)[:max_pages]
                ]
        return s[key]


class Analyzer:
    """Per-item analyzer. Subclasses set name/version and implement applies_to + analyze."""
    name: str = "analyzer"
    version: str = "1"

    def applies_to(self, item: EvidenceItem) -> bool:
        raise NotImplementedError

    def analyze(self, item: EvidenceItem, ctx: Context) -> list[Signal]:
        raise NotImplementedError

    def signal(self, item: EvidenceItem | None, code: str, fraud_prob: float | None,
               reliability: float = 1.0, **evidence: Any) -> Signal:
        return Signal(analyzer=self.name, version=self.version, item_id=item.id if item else None,
                      code=code, fraud_prob=None if fraud_prob is None else float(fraud_prob),
                      reliability=float(np.clip(reliability, 0.0, 1.0)), evidence=evidence)


class ClaimAnalyzer(Analyzer):
    """Runs once per claim, after all per-item analyzers (cross-checks, identity, reuse)."""

    def applies_to(self, item: EvidenceItem) -> bool:
        return False

    def analyze_claim(self, ctx: Context) -> list[Signal]:
        raise NotImplementedError


def save_artifact(item: EvidenceItem, name: str, image: Image.Image) -> str:
    ensure_dirs()
    path = BLOB_DIR / f"{item.sha256[:16]}_{name}.png"
    image.save(path)
    item.artifacts[name] = str(path)
    return str(path)
