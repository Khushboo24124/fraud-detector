"""Learn how to turn raw AI-detector outputs into one calibrated P(AI) (logistic regression on log-odds).

Why: raw detector probabilities are not on a common scale. CommunityForensics ranks new
generators well but outputs tiny probabilities for them; SigLIP is confident but noisier.
Averaging raw outputs flagged only 8% of fakes and sent half to INCONCLUSIVE.

Two profiles (settings.yaml -> ai_profiles):
  scene  damage photos, both detectors        input: reports/eval_results_images.csv
  face   selfies, CommunityForensics only     input: reports/eval_results_faces.csv

Honesty:
  * 5-fold GroupKFold: every quality variant / framing of one source photo stays in one fold.
  * scene also gets a leave-one-generator-out test: the calibrator never sees the generator
    it is tested on (nor the real photos it is tested on), which is what "unseen fakes" means.
  * Rates are reported at the single threshold in settings.yaml (thresholds.ai_flag).

Usage:  python scripts/eval.py --skip-docs                 (writes the CSVs)
        python scripts/train_image_calibrator.py           (both profiles)
        python scripts/train_image_calibrator.py --profile face
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import MODEL_DIR, ensure_dirs, settings  # noqa: E402
from core.models import ai_detectors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# (csv files, group column). External files are optional: used when scripts/eval_external.py has run.
SOURCES = {"scene": (["reports/eval_results_images.csv", "reports/eval_external_images.csv"], "file"),
           "face": (["reports/eval_results_faces.csv", "reports/eval_external_faces.csv"], "group")}
SOURCE_NAMES = {("image", 0): "DrBimmer car damage (real)", ("image", 1): "frontier-synthetic-2026 (AI)",
                ("cardd", 0): "CarDD car damage (real)", ("ai_vs_human", 0): "AI-vs-Human (real)",
                ("ai_vs_human", 1): "AI-vs-Human (AI)", ("face", 0): "face_recognition examples (real)",
                ("face", 1): "frontier AI faces", ("selfies_id", 0): "Selfies & ID dataset (real selfies)"}


def load(profile: str) -> pd.DataFrame:
    files, _ = SOURCES[profile]
    parts = []
    for f in files:
        if (ROOT / f).exists():
            d = pd.read_csv(ROOT / f)
            d["set"] = d["set"] if "set" in d else "image"
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["source"] = [SOURCE_NAMES.get((s, int(l)), f"{s} ({'AI' if l else 'real'})") for s, l in zip(d.set, d.label)]
    return d


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def new_model(y: np.ndarray, real_weight: float = 1.0) -> LogisticRegression:
    """Balanced classes, then real photos up-weighted by the profile's prior (settings.yaml)."""
    n_ai, n_real = int(y.sum()), int((y == 0).sum())
    return LogisticRegression(class_weight={0: real_weight * n_ai / max(n_real, 1), 1: 1.0})


def rates(y: np.ndarray, p: np.ndarray) -> dict:
    th = settings()["thresholds"]
    flag, lo = th["ai_flag"], th["ai_borderline"]
    return {"auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
            "tpr": float((p[y == 1] >= flag).mean()), "fpr": float((p[y == 0] >= flag).mean()),
            "ai_inconclusive": float(((p[y == 1] >= lo) & (p[y == 1] < flag)).mean()),
            "real_inconclusive": float(((p[y == 0] >= lo) & (p[y == 0] < flag)).mean()),
            "threshold": flag, "n": int(len(y)), "n_ai": int(y.sum()), "n_real": int((y == 0).sum())}


def grouped_cv(X, y, groups, w: float) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        oof[te] = new_model(y[tr], w).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def leave_one_generator_out(d: pd.DataFrame, X, y, w: float) -> tuple[np.ndarray, dict]:
    """Hold out one generator's fakes plus one fold of real photos; train on everything else."""
    real_files = sorted(d.loc[y == 0, "file"].unique())
    real_fold = {f: k % 5 for k, f in enumerate(real_files)}
    gens = sorted(d.loc[y == 1, "generator"].unique())
    oof = np.full(len(y), np.nan)
    for k, g in enumerate(gens):
        test = ((y == 1) & (d.generator == g)).to_numpy() | \
               ((y == 0) & d.file.map(real_fold).eq(k % 5)).to_numpy()
        model = new_model(y[~test], w).fit(X[~test], y[~test])
        oof[test] = model.predict_proba(X[test])[:, 1]   # a real photo is held out at least once
    th = settings()["thresholds"]["ai_flag"]
    per_gen = {g: float((oof[((y == 1) & (d.generator == g)).to_numpy()] >= th).mean()) for g in gens}
    return oof, per_gen


def train(profile: str) -> None:
    _, group_col = SOURCES[profile]
    d = load(profile)
    if "variant" in d:
        d = d[d.variant != "splice_ai_patch"].reset_index(drop=True)
    if group_col not in d:
        d[group_col] = d["file"]
    d[group_col] = d[group_col].fillna(d["file"])
    wanted = {det.name for det in ai_detectors() if det.key in settings()["ai_profiles"][profile]["detectors"]}
    model_cols = sorted(c for c in d.columns if c.startswith("m:") and c[2:] in wanted)
    d = d.dropna(subset=model_cols).reset_index(drop=True)
    X = np.column_stack([logit(d[c].to_numpy()) for c in model_cols])
    y, groups = d.label.to_numpy().astype(int), d[group_col].to_numpy()

    w = float(settings()["ai_profiles"][profile].get("real_weight", 1))
    oof = grouped_cv(X, y, groups, w)
    cv = rates(y, oof)
    variant_col = "variant" if "variant" in d else None
    if variant_col:
        cv["by_variant"] = {v: rates(y[d[variant_col] == v], oof[d[variant_col] == v])
                            for v in d[variant_col].unique()}
    th = settings()["thresholds"]["ai_flag"]
    cv["by_source"] = {src: {"n": int((d.source == src).sum()), "label": int(d.label[d.source == src].iloc[0]),
                             "flag_rate": float((oof[(d.source == src).to_numpy()] >= th).mean())}
                       for src in sorted(d.source.unique())}
    if profile == "scene":
        lo, per_gen = leave_one_generator_out(d, X, y, w)
        cv["logo"] = rates(y, lo)
        cv["logo_by_generator"] = per_gen

    model = new_model(y, w).fit(X, y)
    ensure_dirs()
    path = MODEL_DIR / settings()["ai_profiles"][profile]["calibrator"]
    joblib.dump({"model": model, "models": [c[2:] for c in model_cols], "cv": cv, "profile": profile,
                 "real_weight": w}, path)
    print(f"[{profile}] saved {path}")
    print(f"[{profile}] grouped CV @ {cv['threshold']}: AUC {cv['auc']:.3f}, "
          f"AI flagged {cv['tpr']:.0%}, real flagged {cv['fpr']:.0%} (n={cv['n']})")
    for src, v in cv["by_source"].items():
        print(f"[{profile}]   {src:40s} n={v['n']:4d}  {'flagged' if v['label'] else 'false alarms'} {v['flag_rate']:.0%}")
    if "logo" in cv:
        lg = cv["logo"]
        print(f"[{profile}] unseen generator @ {lg['threshold']}: AUC {lg['auc']:.3f}, "
              f"AI flagged {lg['tpr']:.0%}, real flagged {lg['fpr']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["scene", "face", "all"], default="all")
    args = ap.parse_args()
    for p in (["scene", "face"] if args.profile == "all" else [args.profile]):
        if (ROOT / SOURCES[p][0][0]).exists():
            train(p)
        else:
            print(f"[{p}] skipped: {SOURCES[p][0][0]} not found (run scripts/eval.py first)")


if __name__ == "__main__":
    main()
