"""Evaluate ClaimGuard on public external datasets (the ones the team downloaded from Kaggle).

Expected layout (built by scripts/sample_external.py from data/raw/*):
  data/external/manifest.csv        set,path,label,source
  data/external/cardd/              real car-damage photos (CarDD)               label 0
  data/external/ai_vs_human/        real (0_*) and AI (1_*) images                label 0/1
  data/external/casia/{real,tampered,ground_truth}/   CASIA v2 splices + masks
  data/external/findit/             receipts, genuine and forged by people (Find it again!)
  data/external/selfies_id/<person>/{ID_1,Selfie_1..4}.jpg

Outputs (reports/):
  eval_external_images.csv   raw detector scores, used by train_image_calibrator.py
  eval_external_faces.csv    real selfies scored as selfies, used by the face calibrator
  eval_external_identity.csv licence-vs-selfie similarities (genuine and impostor pairs)
  eval_external_docs.csv     pipeline results on Find it again! receipts
  eval_external_casia.csv    heatmap (ELA) behaviour on CASIA splices vs authentic photos

Usage:  python scripts/eval_external.py [--only images,faces,identity,docs,casia]
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from core.models import ai_detectors, face_engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data/external"
OUT = ROOT / "reports"


def whatsapp(img: Image.Image) -> Image.Image:
    s = min(1.0, 1280 / max(img.size))
    img = img.convert("RGB").resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=60)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def save_csv(rows: list[dict], name: str) -> None:
    keys = sorted({k for r in rows for k in r})
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"  wrote reports/{name} ({len(rows)} rows)")


def manifest() -> list[dict]:
    return list(csv.DictReader(open(EXT / "manifest.csv", encoding="utf-8")))


# ------------------------------------------------------------------ AI-image detectors
def run_images() -> None:
    dets = ai_detectors()
    rows = []
    items = [r for r in manifest() if r["set"] in ("cardd", "ai_vs_human")]
    t = time.time()
    for k, r in enumerate(items, 1):
        img = Image.open(EXT / r["path"]).convert("RGB")
        for variant, im in (("original", img), ("whatsapp", whatsapp(img))):
            rows.append({"set": r["set"], "file": r["path"], "label": int(r["label"]), "variant": variant,
                         "generator": r["set"] if r["label"] == "1" else "camera",
                         **{f"m:{d.name}": d.predict([im])[0] for d in dets}})
        if k % 100 == 0:
            print(f"  images {k}/{len(items)} ({time.time() - t:.0f}s)")
    save_csv(rows, "eval_external_images.csv")


# ------------------------------------------------------------------ real selfies for the face calibrator
def run_faces() -> None:
    dets = ai_detectors()
    rows = []
    for person in sorted(p for p in (EXT / "selfies_id").iterdir() if p.is_dir()):
        for f in sorted(person.glob("Selfie_*.jpg")):
            img = Image.open(f).convert("RGB")
            for variant, im in (("original", img), ("whatsapp", whatsapp(img))):
                rows.append({"set": "selfies_id", "file": f"{person.name}/{f.name}", "group": person.name,
                             "label": 0, "variant": variant, "framing": "selfie", "generator": "camera",
                             **{f"m:{d.name}": d.predict([im])[0] for d in dets}})
    save_csv(rows, "eval_external_faces.csv")


# ------------------------------------------------------------------ identity: licence vs selfie
def run_identity() -> None:
    fe = face_engine()
    emb = {}
    for person in sorted(p for p in (EXT / "selfies_id").iterdir() if p.is_dir()):
        for f in sorted(person.glob("*.jpg")):
            faces = fe.faces(Image.open(f).convert("RGB"))
            if faces:
                emb[(person.name, f.stem)] = faces[0]["embedding"]
    people = sorted({p for p, _ in emb})
    rows = []
    for p in people:
        ids = [k for k in emb if k[0] == p and k[1].startswith("ID")]
        for idk in ids:
            for q in people:
                for sk in [k for k in emb if k[0] == q and k[1].startswith("Selfie")]:
                    rows.append({"id_person": p, "id_file": idk[1], "selfie_person": q, "selfie_file": sk[1],
                                 "same": int(p == q), "similarity": float(np.dot(emb[idk], emb[sk]))})
    print(f"  faces found in {len(emb)} images of {len(people)} people")
    save_csv(rows, "eval_external_identity.csv")


# ------------------------------------------------------------------ documents: Find it again!
def run_docs() -> None:
    from core.pipeline import analyze_claim
    rows = []
    items = [r for r in manifest() if r["set"] == "findit"]
    t = time.time()
    for k, r in enumerate(items, 1):
        rep = analyze_claim([(Path(r["path"]).name, (EXT / r["path"]).read_bytes(), None)], persist=False)
        doc_sigs = [s for s in rep.signals if s.fraud_prob and s.fraud_prob > 0.5]
        rows.append({"set": "findit", "file": r["path"], "label": int(r["label"]), "split": r["source"],
                     "risk": rep.domains["document"].risk, "overall": rep.overall_risk, "tier": rep.tier.value,
                     "role": rep.items[0].role.value if rep.items else "",
                     "codes": " ".join(sorted({s.code for s in doc_sigs}))})
        if k % 50 == 0:
            print(f"  docs {k}/{len(items)} ({time.time() - t:.0f}s)")
    save_csv(rows, "eval_external_docs.csv")


# ------------------------------------------------------------------ CASIA: does the heatmap find splices?
def run_casia() -> None:
    from core.analyzers.image_forensics import anomaly_regions, ela_map
    from core.config import settings
    th = settings()["thresholds"]["ela_zscore"]
    gt_dir = EXT / "casia/ground_truth"
    rows = []
    for r in [r for r in manifest() if r["set"] == "casia"]:
        img = Image.open(EXT / r["path"]).convert("RGB")
        z = ela_map(img)
        regions = anomaly_regions(z, th)
        fired = bool(regions) and sum(x["blocks"] for x in regions) < 0.25 * z.size
        hit = None
        if r["label"] == "1":
            num = Path(r["path"]).stem.split("_")[-1]
            gt = next(iter(sorted(gt_dir.glob(f"gt_{num}.*"))), None)
            if gt is not None and regions:
                mask = np.asarray(Image.open(gt).convert("L").resize((z.shape[1], z.shape[0]))) > 127
                hit = any(mask[x["y"]:x["y"] + x["h"], x["x"]:x["x"] + x["w"]].any() for x in regions[:3])
            elif gt is not None:
                hit = False
        rows.append({"set": "casia", "file": r["path"], "label": int(r["label"]), "ela_fired": fired,
                     "regions": len(regions), "top3_overlaps_mask": hit})
    save_csv(rows, "eval_external_casia.csv")


STEPS = {"images": run_images, "faces": run_faces, "identity": run_identity, "docs": run_docs, "casia": run_casia}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STEPS))
    args = ap.parse_args()
    if not (EXT / "manifest.csv").exists():
        sys.exit("data/external/manifest.csv not found: run scripts/sample_external.py first")
    OUT.mkdir(exist_ok=True)
    for name in args.only.split(","):
        print(f"[{name}]")
        STEPS[name.strip()]()


if __name__ == "__main__":
    main()
