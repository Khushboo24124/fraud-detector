"""Numbers shown on the website, read live from the backend's own evaluation outputs.

Nothing is typed in by hand: if the models are retrained or the evaluation is re-run, the
home page, accuracy page and FAQ update automatically, so the UI can never disagree with
reports/eval_report.md. A fallback is used only when a file is missing.
"""
from __future__ import annotations

import csv
import statistics
from functools import lru_cache

from core.config import MODEL_DIR, ROOT, settings


def _cv(profile: str) -> dict:
    try:
        import joblib
        return joblib.load(MODEL_DIR / settings()["ai_profiles"][profile]["calibrator"]).get("cv", {})
    except Exception:
        return {}


def _rows(name: str) -> list[dict]:
    path = ROOT / "reports" / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


@lru_cache(maxsize=1)
def facts() -> dict:
    scene, face = _cv("scene"), _cv("face")
    src = scene.get("by_source", {})
    fsrc = face.get("by_source", {})
    f: dict = {"threshold": settings()["thresholds"]["ai_flag"],
               "borderline": settings()["thresholds"]["ai_borderline"]}

    def rate(d: dict, key: str):
        return d.get(key, {}).get("flag_rate")

    f["cardd_fp"], f["cardd_n"] = rate(src, "CarDD car damage (real)"), src.get("CarDD car damage (real)", {}).get("n")
    f["drb_fp"], f["drb_n"] = rate(src, "DrBimmer car damage (real)"), src.get("DrBimmer car damage (real)", {}).get("n")
    f["aivh_tp"], f["aivh_n"] = rate(src, "AI-vs-Human (AI)"), src.get("AI-vs-Human (AI)", {}).get("n")
    f["stock_fp"] = rate(src, "AI-vs-Human (real)")
    f["frontier_tp"] = rate(src, "frontier-synthetic-2026 (AI)")
    f["selfie_tp"], f["selfie_inc"], f["selfie_fp"] = face.get("tpr"), face.get("ai_inconclusive"), face.get("fpr")
    f["selfie_real_n"] = sum(v["n"] for v in fsrc.values() if v.get("label") == 0) or None

    ident = _rows("eval_external_identity.csv")
    th_match, th_mis = settings()["thresholds"]["face_match"], settings()["thresholds"]["face_mismatch"]
    same = [float(r["similarity"]) for r in ident if r["same"] == "1"]
    diff = [float(r["similarity"]) for r in ident if r["same"] == "0"]
    f["face_same_match"] = sum(s >= th_match for s in same) / len(same) if same else None
    f["face_same_manual"] = sum(th_mis <= s < th_match for s in same) / len(same) if same else None
    f["face_diff_match"] = sum(s >= th_match for s in diff) / len(diff) if diff else None
    f["face_diff_match_n"] = sum(s >= th_match for s in diff)
    f["face_pairs"] = len(diff) or None
    f["face_people"] = len({r["id_person"] for r in ident}) or None

    docs = _rows("eval_results_documents.csv")
    forged = [r for r in docs if r["label"] == "1"]
    genuine = [r for r in docs if r["label"] == "0"]
    flagged = lambda r: r["tier"] in ("MEDIUM", "HIGH")  # noqa: E731
    f["doc_caught"] = f"{sum(map(flagged, forged))}/{len(forged)}" if forged else "—"
    f["doc_false"] = f"{sum(map(flagged, genuine))}/{len(genuine)}" if genuine else "—"
    f["doc_ms"] = int(statistics.median(int(r["ms"]) for r in docs)) if docs else None

    rec = _rows("eval_external_docs.csv")
    rf = [r for r in rec if r["label"] == "1"]
    f["receipt_caught"] = sum(map(flagged, rf)) / len(rf) if rf else None

    imgs = _rows("eval_results_images.csv")
    f["photo_s"] = statistics.median(int(r["ms"]) for r in imgs) / 1000 if imgs else None
    return f
