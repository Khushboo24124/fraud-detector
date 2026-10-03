"""Evaluation harness: runs the full pipeline on labelled data and writes reports/eval_report.md.

Image set: real car-damage photos vs recent-generator AI images, each also degraded the way
claim uploads are (JPEG q70, half resolution, WhatsApp-style, sensor noise), plus partial
manipulations (an AI region spliced into a real photo).
Face set: real portraits vs AI faces scored as selfies (scripts/make_face_set.py output).
Document set: scripts/make_documents.py output.

Usage:  python scripts/eval.py [--limit 20] [--skip-images] [--skip-faces] [--skip-docs] [--report-only]
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

from core.config import settings  # noqa: E402
from core.pipeline import analyze_claim  # noqa: E402
from core.schemas import Role, Tier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FLAGGED = {Tier.MEDIUM, Tier.HIGH}
VARIANTS = ("original", "jpeg70", "half_res", "whatsapp", "noise")


# ------------------------------------------------------------------ image variants
def _jpeg(img: Image.Image, q: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    return buf.getvalue()


def degrade(img: Image.Image, how: str) -> bytes:
    if how == "original":
        return _jpeg(img, 95)
    if how == "jpeg70":
        return _jpeg(img, 70)
    if how == "half_res":
        return _jpeg(img.resize((img.width // 2, img.height // 2), Image.LANCZOS), 90)
    if how == "whatsapp":  # max side 1280 then q≈60, like messenger recompression
        s = min(1.0, 1280 / max(img.size))
        return _jpeg(img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS), 60)
    if how == "noise":  # low-light phone sensor noise (Gaussian, sigma 8 of 255), then JPEG
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        noisy = arr + np.random.default_rng(7).normal(0, 8, arr.shape)
        return _jpeg(Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)), 90)
    raise ValueError(how)


def splice(real: Image.Image, ai: Image.Image, rng: random.Random) -> bytes:
    """Paste an AI-generated patch (~30% of the side) into a real photo with a feathered edge."""
    real = real.convert("RGB")
    side = int(min(real.size) * rng.uniform(0.3, 0.4))
    patch = ai.convert("RGB").resize((side, side), Image.LANCZOS)
    mask = Image.new("L", (side, side), 0)
    mask.paste(255, (side // 10, side // 10, side - side // 10, side - side // 10))
    mask = mask.filter(ImageFilter.GaussianBlur(side // 20))
    x = rng.randint(0, real.width - side); y = rng.randint(0, real.height - side)
    out = real.copy(); out.paste(patch, (x, y), mask)
    return _jpeg(out, 92)


def run_images(limit: int | None) -> list[dict]:
    src = ROOT / "data/eval/images/originals"
    rows = list(csv.DictReader(open(src / "manifest.csv")))
    real = [r for r in rows if r["label"] == "0"][:limit]
    ai = [r for r in rows if r["label"] == "1"][:limit]
    rng = random.Random(3)
    samples = []
    for r in real + ai:
        img = Image.open(src / r["path"])
        for how in VARIANTS:
            samples.append((r["path"], int(r["label"]), how, r["generator"], degrade(img, how)))
    for r, a in zip(real, ai):  # partial manipulation: real photo + AI patch
        samples.append((r["path"], 1, "splice_ai_patch", a["generator"],
                        splice(Image.open(src / r["path"]), Image.open(src / a["path"]), rng)))
    results = []
    for k, (name, label, how, gen, data) in enumerate(samples, 1):
        t = time.perf_counter()
        rep = analyze_claim([(f"{how}_{name}", data, Role.DAMAGE_PHOTO)], persist=False)
        ai_sig = next((s for s in rep.signals if s.code.startswith("IMG-AI")), None)
        per_model = ai_sig.evidence.get("per_model", {}) if ai_sig else {}
        results.append({"set": "image", "file": name, "label": label, "variant": how, "generator": gen,
                        "risk": rep.domains["image"].risk, "overall": rep.overall_risk, "tier": rep.tier.value,
                        "local_flag": any(s.code == "IMG-LOC-01" for s in rep.signals),
                        "ms": int((time.perf_counter() - t) * 1000),
                        **{f"m:{m}": p for m, p in per_model.items()}})
        if k % 20 == 0:
            print(f"  images {k}/{len(samples)}")
    return results


def run_faces(limit: int | None) -> list[dict]:
    """Every face image scored as a selfie, as uploaded (original) and after WhatsApp recompression."""
    src = ROOT / "data/eval/faces"
    if not (src / "manifest.csv").exists():
        print("  faces skipped: run scripts/make_face_set.py first")
        return []
    rows = list(csv.DictReader(open(src / "manifest.csv")))[:limit]
    results = []
    for r in rows:
        img = Image.open(src / r["path"])
        for how in ("original", "whatsapp"):
            t = time.perf_counter()
            rep = analyze_claim([(f"{how}_{r['path']}", degrade(img, how), Role.SELFIE)], persist=False)
            ai_sig = next((s for s in rep.signals if s.code.startswith("IMG-AI")), None)
            per_model = ai_sig.evidence.get("per_model", {}) if ai_sig else {}
            results.append({"set": "face", "file": r["path"], "group": r["group"], "label": int(r["label"]),
                            "variant": how, "framing": r["framing"], "generator": r["generator"],
                            "risk": rep.domains["image"].risk, "overall": rep.overall_risk,
                            "tier": rep.tier.value, "ms": int((time.perf_counter() - t) * 1000),
                            **{f"m:{m}": p for m, p in per_model.items()}})
    return results


def run_docs(limit: int | None) -> list[dict]:
    src = ROOT / "data/eval/docs"
    rows = list(csv.DictReader(open(src / "manifest.csv")))[: (limit * 3 if limit else None)]
    results = []
    for r in rows:
        t = time.perf_counter()
        rep = analyze_claim([(r["path"], (src / r["path"]).read_bytes(), Role.INVOICE)], persist=False)
        results.append({"set": "document", "file": r["path"], "label": int(r["label"]), "variant": r["variant"],
                        "generator": "", "risk": rep.domains["document"].risk, "overall": rep.overall_risk,
                        "tier": rep.tier.value, "codes": " ".join(sorted({s.code for s in rep.signals
                                                                         if s.fraud_prob and s.fraud_prob > 0.5})),
                        "ms": int((time.perf_counter() - t) * 1000)})
    return results


# ------------------------------------------------------------------ reporting
def auc(rows: list[dict], key: str) -> str:
    y = [r["label"] for r in rows if r.get(key) is not None]
    s = [r[key] for r in rows if r.get(key) is not None]
    return f"{roc_auc_score(y, s):.3f}" if len(set(y)) == 2 else "n/a"


def rate(rows: list[dict], cond) -> str:
    return f"{np.mean([cond(r) for r in rows]):.0%} ({sum(cond(r) for r in rows)}/{len(rows)})" if rows else "n/a"


def flagged(r: dict) -> bool:
    return r["tier"] in ("MEDIUM", "HIGH")


def _cal_cv(name: str) -> dict | None:
    path = ROOT / "data/models" / settings()["ai_profiles"][name]["calibrator"]
    if not path.exists():
        return None
    import joblib
    return joblib.load(path).get("cv")


def _cal_weight(name: str) -> str:
    return f"{settings()['ai_profiles'][name].get('real_weight', 1):g}"


def _pct(x: float) -> str:
    return f"{x:.0%}"


def report(img: list[dict], doc: list[dict], faces: list[dict] | None = None) -> str:
    th = settings()["thresholds"]
    T = th["ai_flag"]
    L = ["# ClaimGuard evaluation report", "",
         f"One AI-image threshold everywhere: calibrated P(AI) **≥ {T}** is flagged; "
         f"{th['ai_borderline']}–{T} is held as *inconclusive* (sent to manual review, never auto-cleared).", "",
         "Claim-level *flagged* = tier MEDIUM or HIGH. INCONCLUSIVE counts as *not* flagged.", ""]
    cv = _cal_cv("scene")
    if cv and "tpr" in cv:
        L += ["## Damage photos: AI-generated detection (headline)", "",
              f"n = {cv['n']} image scores ({cv['n_real']} real, {cv['n_ai']} AI) from all image datasets, each photo "
              "also as quality variants. Calibrator scores are out-of-fold: no image is scored by a calibrator that "
              "saw it. **The per-dataset table further down matters more than this average.**", "",
              f"| Test | AUC | AI images flagged @{T} | Real photos flagged @{T} | AI → inconclusive | Real → inconclusive |",
              "|---|---|---|---|---|---|",
              f"| 5-fold CV, grouped by source photo | {cv['auc']:.3f} | {_pct(cv['tpr'])} | {_pct(cv['fpr'])} | "
              f"{_pct(cv['ai_inconclusive'])} | {_pct(cv['real_inconclusive'])} |"]
        if "logo" in cv:
            lg = cv["logo"]
            L.append(f"| **Unseen generator** (leave-one-generator-out) | {lg['auc']:.3f} | {_pct(lg['tpr'])} | "
                     f"{_pct(lg['fpr'])} | {_pct(lg['ai_inconclusive'])} | {_pct(lg['real_inconclusive'])} |")
        L += ["", "Robustness, AUC by upload quality: " + ", ".join(
            f"{k} {v['auc']:.3f}" for k, v in cv["by_variant"].items()), ""]
        if "logo_by_generator" in cv:
            L += ["### Unseen-generator test, per generator (all quality variants)", "",
                  "Each row is scored by a calibrator that never saw that generator. The detectors themselves are "
                  "pre-trained and were never tuned on any of these images; generators such as FLUX.2, "
                  "GPT-Image-1.5 and Seedream-4.5 were released after the CVPR 2025 CommunityForensics model.", "",
                  f"| Generator | AI images flagged @{T} |", "|---|---|"]
            L += [f"| {g} | {_pct(v)} |" for g, v in sorted(cv["logo_by_generator"].items(), key=lambda kv: -kv[1])]
            L.append("")
    if img:
        L += ["## Damage photos: end-to-end claim tiers", "",
              "Whole pipeline on single-photo claims (calibrator fitted on all images, so this is in-sample; "
              "use the table above for honest rates).", ""]
        models = sorted({k for r in img for k in r if k.startswith("m:")})
        L += ["| Variant | AUC (image risk) | " + " | ".join(f"AUC {m[2:]}" for m in models)
              + " | Fakes flagged | Real flagged (false positive) | Real → HIGH |",
              "|---|---|" + "---|" * len(models) + "---|---|---|"]
        for v in VARIANTS:
            rows = [r for r in img if r["variant"] == v]
            if not rows:
                continue
            fakes, reals = [r for r in rows if r["label"] == 1], [r for r in rows if r["label"] == 0]
            L.append(f"| {v} | {auc(rows, 'risk')} | " + " | ".join(auc(rows, m) for m in models)
                     + f" | {rate(fakes, flagged)} | {rate(reals, flagged)} | {rate(reals, lambda r: r['tier'] == 'HIGH')} |")
        sp = [r for r in img if r["variant"] == "splice_ai_patch"]
        L += ["", f"**Partial manipulation (AI patch spliced into real photo):** flagged {rate(sp, flagged)}, "
              f"localized-region signal fired {rate(sp, lambda r: r['local_flag'])}. Known limitation.", "",
              f"Median latency per photo: {int(np.median([r['ms'] for r in img]))} ms (CPU).", ""]
    fcv = _cal_cv("face")
    if fcv and "tpr" in fcv:
        L += ["## Selfies: AI-generated / deepfake face detection", "",
              f"n = {fcv['n']} selfie images ({fcv['n_real']} real, {fcv['n_ai']} AI; full photo + selfie crop, "
              "as uploaded and WhatsApp-recompressed). 5-fold CV grouped by source photo.", "",
              f"| AUC | AI faces flagged @{T} | Real faces flagged @{T} | AI → inconclusive | Real → inconclusive |",
              "|---|---|---|---|---|",
              f"| {fcv['auc']:.3f} | {_pct(fcv['tpr'])} | {_pct(fcv['fpr'])} | {_pct(fcv['ai_inconclusive'])} | "
              f"{_pct(fcv['real_inconclusive'])} |", "",
              f"So {_pct(fcv['tpr'] + fcv['ai_inconclusive'])} of AI selfies reach a human (flagged or inconclusive), "
              f"while {_pct(fcv['fpr'])} of genuine selfies are flagged. The face calibrator weights genuine selfies "
              f"{_cal_weight('face')}x because deepfakes are the rare case (unweighted, 3x more real selfies were "
              "flagged).", "",
              "Why selfies use only CommunityForensics: on real portraits the SigLIP detector answered "
              "'AI' for most photos (AUC 0.58 on this set), so averaging it in made real selfies look fake.", ""]
    if faces:
        reals = [r for r in faces if r["label"] == 0]; fakes = [r for r in faces if r["label"] == 1]
        L += [f"End-to-end single-selfie claims: AI faces flagged {rate(fakes, flagged)}, "
              f"real faces flagged {rate(reals, flagged)}.", ""]
    if doc:
        L += ["## Document tampering (on our synthetic document set)", "", f"AUC (document risk): **{auc(doc, 'risk')}**", "",
              "| Variant | n | Flagged | Most common reason codes |", "|---|---|---|---|"]
        for v in sorted({r["variant"] for r in doc}):
            rows = [r for r in doc if r["variant"] == v]
            codes: dict[str, int] = {}
            for r in rows:
                for c in (r.get("codes") or "").split():
                    codes[c] = codes.get(c, 0) + 1
            top = ", ".join(f"{c}×{n}" for c, n in sorted(codes.items(), key=lambda x: -x[1])[:3])
            L.append(f"| {v} | {len(rows)} | {rate(rows, flagged)} | {top} |")
        L += ["", f"Median latency per document: {int(np.median([r['ms'] for r in doc]))} ms.", ""]
    L += external_section()
    L += ["## Caveats", "",
          "- Before the public datasets were added, the calibrator had seen only 40 real photos from one dataset. "
          "It then flagged 28% of real CarDD photos and 60% of AI-vs-Human real photos as AI; with them, 1% and 16%. "
          "The price is honest: the newest generators (frontier set) are caught 38% of the time, not the 74% that "
          "the narrow test suggested.",
          "- Real photos: public car-damage dataset; AI images: recent generators (frontier-synthetic-images-2026). "
          "Many AI images show people or scenes rather than car damage, so scene content differs between classes.",
          "- Real faces: 17 public example photos plus the Selfies & ID Images sample (29 people). AI faces are "
          "crops of frontier-synthetic images, so the AI-face set is small (38 faces).",
          "- The AI-vs-Human real photos are stock photography (often retouched); 16% of them are flagged, which is "
          "why the damage-photo profile should be judged on the car-damage rows (CarDD, DrBimmer).",
          "- Documents and tampering are synthetic (scripts/make_documents.py) and made by the same generator as the "
          "genuine ones, so 100% on documents means *on our synthetic set*; real forgeries will be more varied.",
          "- Small sample sizes: treat rates as indicative, not as production accuracy claims."]
    return "\n".join(L)


def external_section() -> list[str]:
    """Results on public datasets (scripts/eval_external.py), shown per dataset."""
    import pandas as pd
    th = settings()["thresholds"]
    T = th["ai_flag"]
    L = []
    cv, fcv = _cal_cv("scene"), _cal_cv("face")
    by = {**(cv or {}).get("by_source", {}), **(fcv or {}).get("by_source", {})}
    if by:
        L += ["## Results per dataset (public data, out-of-fold)", "",
              f"Each image is scored by a calibrator that never saw it (5-fold CV grouped by source photo). "
              f"Real sets show the **false-alarm** rate, AI sets the **detection** rate, at the {T} threshold.", "",
              "| Dataset | Kind | n | Flagged as AI |", "|---|---|---|---|"]
        for src, v in sorted(by.items(), key=lambda kv: (kv[1]["label"], kv[0])):
            L.append(f"| {src} | {'AI' if v['label'] else 'real'} | {v['n']} | {_pct(v['flag_rate'])} |")
        L.append("")
    ident = ROOT / "reports/eval_external_identity.csv"
    if ident.exists():
        d = pd.read_csv(ident)
        g, imp = d[d.same == 1].similarity, d[d.same == 0].similarity
        m, mm = th["face_match"], th["face_mismatch"]
        L += ["## Identity: licence vs selfie (Selfies & ID Images)", "",
              f"{d.id_person.nunique()} people, {len(g)} genuine pairs (own ID vs own selfie), {len(imp)} impostor pairs "
              "(ID vs someone else's selfie).", "",
              "| Pair type | Match (≥ %.2f) | Uncertain (%.2f–%.2f) | Mismatch (< %.2f) | Median similarity |" % (m, mm, m, mm),
              "|---|---|---|---|---|",
              f"| Same person | {_pct((g >= m).mean())} | {_pct(((g >= mm) & (g < m)).mean())} | {_pct((g < mm).mean())} | {g.median():.2f} |",
              f"| Different person | {_pct((imp >= m).mean())} | {_pct(((imp >= mm) & (imp < m)).mean())} | {_pct((imp < mm).mean())} | {imp.median():.2f} |",
              ""]
    docs = ROOT / "reports/eval_external_docs.csv"
    if docs.exists():
        d = pd.read_csv(docs).fillna("")
        forged, genuine = d[d.label == 1], d[d.label == 0]
        fl = lambda x: x.tier.isin(["MEDIUM", "HIGH"])
        from sklearn.metrics import roc_auc_score
        codes: dict[str, int] = {}
        for c in " ".join(forged.codes).split():
            codes[c] = codes.get(c, 0) + 1
        L += ["## Documents: real forgeries made by people (Find it again!)", "",
              f"{len(forged)} receipts forged by hand (GIMP, Paint, ...) and {len(genuine)} genuine ones, as images.", "",
              "| AUC (document risk) | Forged flagged | Genuine flagged | Most common reasons on forged |",
              "|---|---|---|---|",
              f"| {roc_auc_score(d.label, d.risk):.3f} | {_pct(fl(forged).mean())} | {_pct(fl(genuine).mean())} | "
              + ", ".join(f"{c}×{n}" for c, n in sorted(codes.items(), key=lambda x: -x[1])[:3]) + " |", "",
              "Compare with 100% on our synthetic PDFs: hand-made forgeries on photographed receipts are much harder, "
              "because there is no PDF structure to inspect and the edits are only a few pixels wide.", ""]
    casia = ROOT / "reports/eval_external_casia.csv"
    if casia.exists():
        d = pd.read_csv(casia)
        t, r = d[d.label == 1], d[d.label == 0]
        hit = t.top3_overlaps_mask.dropna().astype(str).eq("True")
        L += ["## Partial edits / heatmap check (CASIA v2 splices)", "",
              f"| | Authentic ({len(r)}) | Spliced ({len(t)}) |", "|---|---|---|",
              f"| Compression view highlights a region | {_pct(r.ela_fired.mean())} | {_pct(t.ela_fired.mean())} |", "",
              f"On spliced images, a highlighted box overlaps the true edited area in {_pct(hit.mean())} of cases. "
              "This is why the compression view is shown to investigators but not scored.", ""]
    return L


def save_csv(rows: list[dict], path: Path) -> None:
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        for k, v in list(r.items()):
            if k in ("label", "ms"):
                r[k] = int(v)
            elif k in ("risk", "overall") or k.startswith("m:"):
                r[k] = float(v) if v != "" else None
            elif k == "local_flag":
                r[k] = v == "True"
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-faces", action="store_true")
    ap.add_argument("--skip-docs", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from saved CSVs")
    args = ap.parse_args()
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    img_csv, doc_csv = out / "eval_results_images.csv", out / "eval_results_documents.csv"
    face_csv = out / "eval_results_faces.csv"
    if not args.report_only:
        if not args.skip_images:
            save_csv(run_images(args.limit), img_csv)
        if not args.skip_faces:
            rows = run_faces(args.limit)
            if rows:
                save_csv(rows, face_csv)
        if not args.skip_docs:
            save_csv(run_docs(args.limit), doc_csv)
    md = report(load_csv(img_csv), load_csv(doc_csv), load_csv(face_csv))
    (out / "eval_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
