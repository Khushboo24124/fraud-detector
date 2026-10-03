"""Generator-independent forensics: ELA, noise residual, EXIF and provenance metadata.

These signals do not depend on any particular AI generator, which is what keeps the
system useful when a new generator appears that the classifiers have never seen.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime

import cv2
import numpy as np
from PIL import ExifTags, Image

from ..config import settings
from ..schemas import EvidenceItem, Kind, Role
from .base import Analyzer, Context, save_artifact

BLOCK = 16
WORK_SIDE = 1024

EDIT_SOFTWARE = ["photoshop", "gimp", "snapseed", "lightroom", "picsart", "canva", "pixlr",
                 "facetune", "affinity", "paint.net", "fotor", "remini", "photoroom", "inpaint"]
GENERATOR_MARKERS = [
    (rb"trainedAlgorithmicMedia", "IPTC DigitalSourceType = trainedAlgorithmicMedia"),
    (rb"compositeWithTrainedAlgorithmicMedia", "IPTC DigitalSourceType = composite with AI"),
    (rb"c2pa", "C2PA content-credentials manifest"),
    (rb"Stable Diffusion", "Stable Diffusion"),
    (rb"ComfyUI", "ComfyUI workflow"),
    (rb"Midjourney", "Midjourney"),
    (rb"DALL[\-\xb7]?E", "DALL-E"),
    (rb"Adobe Firefly", "Adobe Firefly"),
    (rb"Imagen", "Google Imagen"),
    (rb"\"prompt\"\s*:", "embedded generation prompt"),
]
PNG_GEN_KEYS = {"parameters", "prompt", "workflow", "Dream", "sd-metadata", "invokeai_metadata"}


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def _block_mean(a: np.ndarray, block: int = BLOCK) -> np.ndarray:
    h, w = (a.shape[0] // block) * block, (a.shape[1] // block) * block
    a = a[:h, :w]
    return a.reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def _work_gray(img: Image.Image) -> tuple[Image.Image, np.ndarray]:
    scale = min(1.0, WORK_SIDE / max(img.size))
    work = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS) if scale < 1 else img
    return work, cv2.cvtColor(np.asarray(work), cv2.COLOR_RGB2GRAY).astype(np.float32)


def ela_map(img: Image.Image, quality: int = 90) -> np.ndarray:
    """Block z-scores of re-compression error, normalised by local texture.

    Raw ELA is high on every sharp edge; dividing by texture energy keeps only
    regions whose *compression history* differs from the rest of the image.
    """
    work, gray = _work_gray(img)
    buf = io.BytesIO()
    work.save(buf, "JPEG", quality=quality)
    recompressed = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    err = np.abs(np.asarray(work, dtype=np.float32) - recompressed).mean(axis=2)
    texture = _block_mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))
    ratio = _block_mean(err) / (texture + 4.0)
    # Flat blocks (blank paper, clear sky) carry no compression evidence; exclude them
    # from both the statistics and the result, otherwise they dominate the z-scores.
    textured = texture > 2.0
    if textured.sum() < 16:
        return np.zeros_like(ratio)
    z = np.zeros_like(ratio)
    vals = ratio[textured]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    z[textured] = (vals - med) / mad
    return z


def noise_map(img: Image.Image) -> np.ndarray:
    """Block z-scores of sensor-noise level. Pasted or generated patches carry different noise."""
    _, gray = _work_gray(img)
    residual = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    std = np.sqrt(_block_mean(residual ** 2))
    return np.abs(_robust_z(np.log(std + 1e-3)))


def anomaly_regions(z: np.ndarray, threshold: float, min_blocks: int = 4) -> list[dict]:
    mask = (z > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_blocks:
            regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                            "blocks": int(area), "z": float(z[labels == i].max())})
    return sorted(regions, key=lambda r: r["z"], reverse=True)


def render_heatmap(img: Image.Image, layers: list[np.ndarray], boxes: list[tuple] | None = None) -> Image.Image:
    """Blend normalised evidence layers into one JET overlay on the photo."""
    w, h = img.size
    acc = np.zeros((h, w), np.float32)
    for layer in layers:
        if layer is None or layer.size == 0:
            continue
        lay = cv2.resize(layer.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        lay = np.clip(lay, 0, None)
        if lay.max() > 0:
            acc += lay / lay.max()
    if acc.max() > 0:
        acc /= acc.max()
    colour = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(base, 0.55, colour, 0.45, 0)
    for (x0, y0, x1, y1) in boxes or []:
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), max(2, w // 300))
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def _exif(raw: Image.Image) -> dict:
    out = {}
    try:
        ex = raw.getexif()
        tags = dict(ex)
        tags.update(ex.get_ifd(0x8769))  # Exif sub-IFD holds DateTimeOriginal
        for k, v in tags.items():
            name = ExifTags.TAGS.get(k, str(k))
            if name in {"Software", "Make", "Model", "DateTimeOriginal", "DateTime", "ProcessingSoftware"}:
                out[name] = str(v).strip("\x00 ")
    except Exception:
        pass
    return out


def _parse_exif_date(s: str) -> date | None:
    try:
        return datetime.strptime(s[:19], "%Y:%m:%d %H:%M:%S").date()
    except ValueError:
        return None


class ImageForensicsAnalyzer(Analyzer):
    name = "image_forensics"
    version = "1.0"

    def applies_to(self, item: EvidenceItem) -> bool:
        # Selfies included: a generator signature or editing software in a selfie's metadata
        # is strong evidence of a synthetic identity.
        return item.kind is Kind.IMAGE

    def analyze(self, item, ctx: Context):
        th = settings()["thresholds"]
        img, raw = ctx.image(item), ctx.raw_image(item)
        is_doc = item.role.is_document
        signals = []

        # --- Error level analysis (meaningful for JPEG; lossless files have no history) ---
        z_ela = ela_map(img)
        regions = anomaly_regions(z_ela, th["ela_zscore"])
        jpeg_q = item.quality.get("jpeg_quality")
        ela_rel = item.reliability * (0.9 if raw.format == "JPEG" else 0.4)
        if jpeg_q is not None and jpeg_q < 75:
            ela_rel *= 0.6  # heavy recompression smears local differences
        # A lone hot region is suspicious; hot blocks everywhere is just texture/noise.
        # Not scored on document images: our eval measured identical ELA z-scores on genuine
        # scans and pasted forgeries (median 8.1 vs 8.1), so it would only add noise there.
        if not is_doc and regions and sum(r["blocks"] for r in regions) < 0.25 * z_ela.size:
            signals.append(self.signal(item, "IMG-ELA-01", None, ela_rel, regions=len(regions),
                                       zmax=regions[0]["z"], top_regions=regions[:3]))

        # --- Heatmap: tiles (from AI analyzer) + ELA + noise, with suspicious boxes ---
        layers = [np.clip(z_ela, 0, None), noise_map(img)]
        boxes = []
        if "tile_map" in ctx.slot(item):
            tb, tp = ctx.slot(item)["tile_map"]
            tile_layer = np.zeros((img.height, img.width), np.float32)
            for (x0, y0, x1, y1), p in zip(tb, tp):
                tile_layer[y0:y1, x0:x1] = np.maximum(tile_layer[y0:y1, x0:x1], p)
            layers.append(tile_layer * 2)  # model evidence weighs more than classic signals
        scale = img.width / z_ela.shape[1]
        for r in regions[:3]:
            boxes.append(tuple(int(v * scale) for v in (r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"])))
        save_artifact(item, "heatmap", render_heatmap(img, layers, boxes))

        # --- Provenance metadata: explicit generator / AI signatures ---
        data = open(item.path, "rb").read()
        marker = next((label for pat, label in GENERATOR_MARKERS if re.search(pat, data)), None)
        if marker is None and raw.format == "PNG":
            keys = set((raw.info or {}).keys()) & PNG_GEN_KEYS
            marker = f"PNG text chunk '{sorted(keys)[0]}'" if keys else None
        if marker:
            signals.append(self.signal(item, "IMG-META-03", 0.95, 1.0, marker=marker))

        # --- EXIF: editing software and capture timeline ---
        exif = _exif(raw)
        item.quality["exif"] = exif or None
        software = (exif.get("Software", "") + " " + exif.get("ProcessingSoftware", "")).lower()
        hit = next((s for s in EDIT_SOFTWARE if s in software), None)
        if hit:
            signals.append(self.signal(item, "IMG-META-01", 0.75, 1.0, software=software.strip()))

        taken = _parse_exif_date(exif.get("DateTimeOriginal", "")) if exif else None
        # Only damage photos must be taken around the accident; a selfie can be from any day.
        if taken and ctx.info.accident_date and item.role is Role.DAMAGE_PHOTO:
            acc = date.fromisoformat(ctx.info.accident_date)
            days = (taken - acc).days
            if days < -1:
                signals.append(self.signal(item, "IMG-META-02", 0.8, 0.9, taken=str(taken),
                                           days=abs(days), direction="before"))
            elif days > 30:
                signals.append(self.signal(item, "IMG-META-02", 0.62, 0.8, taken=str(taken),
                                           days=days, direction="after"))
        return signals
