"""AI-generation detection: calibrated model ensemble on the whole photo + tile scan for local edits.

Raw detector probabilities are not on a common scale (CommunityForensics outputs tiny
probabilities for the newest generators even when it ranks them correctly), so they are
fused by a stacked logistic regression trained on our labelled set
(scripts/train_image_calibrator.py, cross-validated). Without a calibrator the raw mean is used.

Selfies are scored too (a deepfake selfie is the core of a synthetic identity), with their
own "face" profile: see ai_profiles in settings.yaml for why the detector set differs.

Whole-image classifiers average over every pixel, so a real photo with one AI-inpainted
region looks "real". The tile scan compares each crop's log-odds with the whole image's.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import MODEL_DIR, settings
from ..models import ai_detectors
from ..schemas import EvidenceItem, Kind, Role
from .base import Analyzer, Context

GRID = 3          # 3x3 overlapping tiles
TILE_FRAC = 0.5   # each tile spans half the image side
PROFILE_BY_ROLE = {Role.DAMAGE_PHOTO: "scene", Role.SELFIE: "face"}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def tile_boxes(w: int, h: int, grid: int = GRID, frac: float = TILE_FRAC) -> list[tuple[int, int, int, int]]:
    tw, th = int(w * frac), int(h * frac)
    xs = np.linspace(0, w - tw, grid).astype(int)
    ys = np.linspace(0, h - th, grid).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


class _Profile:
    """Detectors + calibrator for one kind of photo (scene = damage photo, face = selfie)."""

    def __init__(self, name: str, cfg: dict, all_detectors) -> None:
        self.name = name
        self.detectors = [d for d in all_detectors if d.key in cfg["detectors"]]
        self.calibrator = None
        path = MODEL_DIR / cfg["calibrator"]
        if path.exists():
            import joblib
            bundle = joblib.load(path)
            if bundle["models"] == [d.name for d in sorted(self.detectors, key=lambda d: d.name)]:
                self.calibrator = bundle

    def fuse(self, per_model: dict[str, float]) -> float:
        if self.calibrator is None:
            return float(np.mean(list(per_model.values())))
        x = np.array([[_logit(per_model[m]) for m in self.calibrator["models"]]])
        return float(self.calibrator["model"].predict_proba(x)[0, 1])


class AIImageAnalyzer(Analyzer):
    name = "ai_ensemble"

    def __init__(self) -> None:
        detectors = ai_detectors()
        self.detectors = list(detectors)
        self.profiles = {n: _Profile(n, c, detectors) for n, c in settings()["ai_profiles"].items()}
        self.version = "+".join(f"{d.name}@{d.version}" for d in detectors) + "|" + ",".join(
            f"{n}:{'calibrated' if p.calibrator else 'mean'}" for n, p in self.profiles.items())

    @property
    def calibrator(self):  # backwards compatible: the damage-photo calibrator
        return self.profiles["scene"].calibrator

    def applies_to(self, item: EvidenceItem) -> bool:
        return item.kind is Kind.IMAGE and item.role in PROFILE_BY_ROLE

    def fuse(self, per_model: dict[str, float], profile: str = "scene") -> float:
        return self.profiles[profile].fuse(per_model)

    def analyze(self, item, ctx: Context):
        th = settings()["thresholds"]
        img = ctx.image(item)
        profile = self.profiles[PROFILE_BY_ROLE[item.role]]

        per_model = {d.name: d.predict([img])[0] for d in profile.detectors}
        prob = profile.fuse(per_model)
        raw = np.array(list(per_model.values()))
        agree = int((raw >= 0.5).sum()) if prob >= th["ai_flag"] else int((raw < 0.5).sum())

        common = dict(per_model={k: round(v, 4) for k, v in per_model.items()}, models_total=len(raw),
                      prob=prob, spread=float(np.ptp(raw)), calibrated=profile.calibrator is not None,
                      profile=profile.name, threshold=th["ai_flag"])
        signals = []
        if th["ai_borderline"] <= prob < th["ai_flag"]:
            signals.append(self.signal(item, "IMG-AI-02", prob, item.reliability * 0.5,
                                       inconclusive=True, models_agree=agree, **common))
        else:
            code = "IMG-AI-01" if prob >= th["ai_flag"] else "IMG-AI-00"
            signals.append(self.signal(item, code, prob, item.reliability, models_agree=agree, **common))

        if th.get("tile_scan_enabled", False) and profile.name == "scene":
            signals += self._tile_scan(item, ctx, img, per_model, prob)
        return signals

    def _tile_scan(self, item, ctx: Context, img, per_model: dict[str, float], prob: float):
        """Compare each crop's log-odds with the whole image's (generalist detector only)."""
        th = settings()["thresholds"]
        primary = self.profiles["scene"].detectors[0]
        boxes = tile_boxes(*img.size)
        tile_probs = np.array(primary.predict([img.crop(b) for b in boxes]))
        tile_logit = np.array([_logit(p) for p in tile_probs])
        ctx.slot(item)["tile_map"] = (boxes, (tile_logit - tile_logit.min()) / (np.ptp(tile_logit) + 1e-6))
        global_logit = _logit(per_model[primary.name])
        k = int(tile_logit.argmax())
        gap = tile_logit[k] - global_logit
        if gap >= th["tile_logit_gap"] and tile_probs[k] >= th["tile_min_prob"] and prob < th["ai_flag"]:
            return [self.signal(item, "IMG-LOC-01", float(np.clip(0.5 + gap / 20, 0.6, 0.85)),
                                item.reliability * 0.8, tile_prob=float(tile_probs[k]),
                                global_prob=per_model[primary.name], box=list(map(int, boxes[k])))]
        return []
