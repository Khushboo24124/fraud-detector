"""Intake: validate bytes, store content-addressed copies, measure input quality.

Quality is turned into a `reliability` in [0, 1] that later scales how much an
item's forensic signals may move the score. A WhatsApp-recompressed, blurry,
low-resolution photo must not be judged as confidently as a camera original.
"""
from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import BLOB_DIR, ensure_dirs, settings
from .schemas import EvidenceItem, Kind, Role


class IntakeError(ValueError):
    """Raised for files that must be rejected before analysis."""


_MAGIC = {
    b"\xff\xd8\xff": ("image", ".jpg"),
    b"\x89PNG\r\n\x1a\n": ("image", ".png"),
    b"RIFF": ("image", ".webp"),       # verified further below
    b"BM": ("image", ".bmp"),
    b"%PDF-": ("pdf", ".pdf"),
}

# IJG standard luminance quantisation table (quality 50), used to estimate JPEG quality.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def sniff(data: bytes) -> tuple[Kind, str]:
    """Identify the file by its magic bytes, never by its name or browser MIME type."""
    for magic, (kind, ext) in _MAGIC.items():
        if data.startswith(magic):
            if magic == b"RIFF" and data[8:12] != b"WEBP":
                continue
            return Kind(kind), ext
    raise IntakeError("Unsupported file type (accepted: JPEG, PNG, WEBP, BMP, PDF).")


def store_blob(data: bytes, ext: str) -> tuple[str, Path]:
    ensure_dirs()
    digest = hashlib.sha256(data).hexdigest()
    path = BLOB_DIR / f"{digest}{ext}"
    if not path.exists():
        path.write_bytes(data)
    return digest, path


def guess_role(filename: str, kind: Kind) -> Role:
    """Filename hint only; analyzers refine the role from content (OCR text, faces)."""
    name = filename.lower()
    hints = [
        (("selfie",), Role.SELFIE),
        (("licen", "dl_", "dl-", "driving"), Role.LICENCE),
        (("rc_", "rc-", "registration"), Role.RC),
        (("invoice", "bill", "estimate", "receipt"), Role.INVOICE),
        (("claim_form", "claimform", "intimation"), Role.CLAIM_FORM),
    ]
    for keys, role in hints:
        if any(k in name for k in keys):
            return role
    return Role.OTHER_DOC if kind is Kind.PDF else Role.DAMAGE_PHOTO


def estimate_jpeg_quality(img: Image.Image) -> int | None:
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables:
        return None
    q = np.array(tables[0], dtype=np.float64)
    if q.size != 64:
        return None
    scale = float(np.mean(q / _STD_LUMA) * 100.0)
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return int(np.clip(round(quality), 1, 100))


def assess_image_quality(img: Image.Image) -> tuple[float, dict]:
    cfg = settings()["quality"]
    w, h = img.size
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    jpeg_q = estimate_jpeg_quality(img)

    issues, reliability = [], 1.0
    if min(w, h) < cfg["min_side_px"]:
        issues.append(f"low resolution {w}x{h}")
        reliability *= 0.5
    elif min(w, h) < 2 * cfg["min_side_px"]:
        reliability *= 0.85
    if blur < cfg["blur_laplacian_min"]:
        issues.append("blurry")
        reliability *= 0.7
    if jpeg_q is not None and jpeg_q < cfg["jpeg_quality_low"]:
        issues.append(f"heavily compressed (JPEG q≈{jpeg_q})")
        reliability *= 0.7
    return round(reliability, 3), {
        "width": w, "height": h, "blur": round(blur, 1), "jpeg_quality": jpeg_q,
        "format": img.format, "issues": issues,
    }


def ingest(filename: str, data: bytes, role: Role | None = None) -> EvidenceItem:
    limits = settings()["limits"]
    if not data:
        raise IntakeError(f"{filename}: file is empty.")
    if len(data) > limits["max_file_mb"] * 1024 * 1024:
        raise IntakeError(f"{filename}: larger than {limits['max_file_mb']} MB.")

    kind, ext = sniff(data)
    digest, path = store_blob(data, ext)
    item = EvidenceItem(
        id=uuid.uuid4().hex[:12], filename=filename, kind=kind,
        role=role or guess_role(filename, kind), sha256=digest, path=str(path),
    )

    if kind is Kind.IMAGE:
        Image.MAX_IMAGE_PIXELS = limits["max_image_pixels"]
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as exc:  # corrupt, truncated or decompression bomb
            raise IntakeError(f"{filename}: image could not be decoded ({exc}).") from exc
        item.reliability, item.quality = assess_image_quality(img)
    return item
