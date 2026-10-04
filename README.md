# 🛡️ ClaimGuard: synthetic identity & deepfake claim detection

ClaimGuard checks the evidence attached to a **motor-insurance claim** (damage photos, repair
invoices, RC, driving licence and selfie) **as one connected case**. It flags AI-generated or
manipulated content and returns an **explainable fraud score**: LOW, MEDIUM, HIGH or INCONCLUSIVE.
Every point of that score is traceable to a reason code an investigator can verify.

> A screening signal, not proof. A human makes the final decision.

## Quick start (Windows, CPU only)

```bash
py -3.11 -m venv %USERPROFILE%\.venvs\claimguard
%USERPROFILE%\.venvs\claimguard\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/download_models.py        # one-time; afterwards everything runs offline
python scripts/train_invoice_model.py    # invoice anomaly model (seconds)
python scripts/make_documents.py         # synthetic genuine + tampered documents
python scripts/fetch_images.py           # real car-damage photos + recent-generator AI images
python scripts/make_face_set.py          # selfie set: real portraits + AI faces
python scripts/eval.py --skip-docs       # score damage photos and selfies with the detectors
python scripts/sample_external.py        # optional: sample the Kaggle datasets in data/raw/ (see Data)
python scripts/eval_external.py          # optional: score them (AI images, selfies, ID pairs, receipts, CASIA)
python scripts/train_image_calibrator.py # learn the detector fusion per profile (cross-validated)
python scripts/eval.py                   # full accuracy report with the new calibrators
python scripts/make_demo_claims.py       # ready-made demo claims for the UI
streamlit run app.py
```

Tests: `python -m pytest -q`. Accuracy report: `python scripts/eval.py`, which writes `reports/eval_report.md`.

## What it detects

| Module | Technique | Reason codes |
|---|---|---|
| AI-generated photos **and selfies** | **CommunityForensics ViT** (CVPR 2025, 4,803 generators) and a **SigLIP** AI-vs-human classifier, fused by a **learned calibrator** (logistic regression, cross-validated). Damage photos and selfies have separate profiles (`ai_profiles` in settings.yaml): selfies use CommunityForensics only, because SigLIP calls most real portraits AI. One threshold everywhere: ≥ 0.65 flagged, 0.40–0.65 *inconclusive*, not a guess. | `IMG-AI-*` |
| Visual forensics | Heatmap of texture-normalised **ELA** and a **noise residual** for the investigator to inspect (shown, not scored; see Measured results) | `IMG-ELA-01` |
| Provenance | C2PA / IPTC *trainedAlgorithmicMedia* / SD / ComfyUI markers; EXIF editing software (photos and selfies); capture date vs accident date (damage photos) | `IMG-META-*` |
| PDF tampering | Cover-up boxes over hidden text, stacked text runs, font or size change inside a row, incremental saves, editor metadata | `DOC-OVL-*`, `DOC-FONT-01`, `DOC-META-*` |
| Scans and photos of documents | OCR (RapidOCR) character-height geometry (ELA on document scans was measured and dropped, see below) | `DOC-FONT-02` |
| Invoice logic | Items → subtotal → tax → total arithmetic; **IsolationForest** trained on genuine invoices only | `DOC-MATH-01`, `DOC-ANOM-01` |
| Cross-document | Vehicle number, owner name, invoice-vs-accident date, number plate seen in the photo | `DOC-XCHK-*`, `IMG-XCHK-01` |
| Identity (bonus) | OpenCV **YuNet + SFace** face match, licence vs selfie, with an uncertain band; a clear mismatch is HIGH by rule. A deepfake selfie is caught by the AI-image check above | `ID-FACE-*` |
| Cross-claim reuse | SHA-256, perceptual hash and face embeddings in SQLite | `XCLM-*` |

## Web app

`streamlit run app.py` opens the website: a home page, four menus (Technology, Solutions, Resources,
About) and a step-by-step **Analyze a claim** page (details → upload → run → verdict → evidence).
UI code lives in `ui/` (`theme.py` holds all colours and CSS). Numbers shown on the site are read
live from `reports/` and `data/models/`, so the UI always matches the evaluation. To change the
pictures, replace `assets/hero_placeholder.png` and `assets/inspection_placeholder.jpg`.

## Architecture

```
Streamlit UI ──► core.pipeline.analyze_claim(files, info) ──► ClaimReport
                    │
   ingest ─ magic-byte sniffing · SHA-256 blob store · quality gate → reliability
   analyzers/ (plug-ins, each emits Signals)
      image_ai · image_forensics · documents · claim_level (cross-doc, identity, reuse)
   risk ─ log-odds fusion · group pooling · overrides · INCONCLUSIVE · plain-English summary
   store ─ SQLite (claims, photo hashes, face embeddings)
   config/ ─ settings.yaml (weights, thresholds, pinned model revisions)
             reason_codes.yaml (every explanation) · domain_vehicle.yaml (field rules)
```

* **One seam.** Analyzers only produce `Signal`s and the risk engine only consumes them. Swapping a
  model, adding a check or replacing the UI never crosses it.
* **Explainable by construction.** Each signal adds `weight × reliability × logit(p)` to the score.
  The UI shows these as a waterfall, so the explanation *is* the computation.
* **No double counting.** Correlated signals share a group, and a group keeps its strongest fraud-side
  value. Genuine evidence can't cancel a fake photo.
* **Robust to bad inputs.** The quality gate lowers the reliability of blurry, tiny or
  heavily-recompressed files, so they move the score less.
* **Failure isolation.** If an analyzer crashes, the claim still completes and the gap is shown (`SYS-FAIL-01`).
* **Domain as config.** Motor insurance lives in `domain_vehicle.yaml`; health or property would be a new rules file.
* **Private.** All models run locally, and no claim data leaves the machine. Model revisions are pinned and
  weights are safetensors/ONNX only.

## Data

Real fraud data is scarce, so ClaimGuard is validated on:
* **Images:** real car-damage photos (`DrBimmer/comprehensive-car-damage`) and AI images from about 20
  recent generators (`Thermostatic/frontier-synthetic-images-2026`). Each is degraded the way claim
  uploads are (JPEG q70, half resolution, WhatsApp-style, sensor noise), plus AI patches spliced into real photos.
* **Public datasets (Kaggle, in `data/raw/`, sampled by `scripts/sample_external.py`):** CarDD (374 real
  car-damage photos), AI vs Human-Generated Images (300 real + 300 AI), Selfies & ID Images (29 people:
  ID + selfies), Find it again! (100 receipts forged by people + 100 genuine), CASIA v2 (50 authentic +
  50 spliced, with masks). Check each dataset's license before redistributing; they are not in the zip.
* **Selfies:** 17 real portrait photos (MIT-licensed examples from `ageitgey/face_recognition`, in
  `data/raw/faces_real/`; add your own) and face crops of the AI images that show a person, each as the
  full photo and a selfie-style crop, original and WhatsApp-recompressed.
* **Documents:** generated garage invoices with 5 tampering methods (retype over a cover box, clean
  redaction, line-item inflation, print-and-paste JPEG, editor re-save) against genuine PDFs and scans.

## Measured results (`reports/eval_report.md`)

All rates at the one threshold (0.65), out-of-fold (no image is scored by a calibrator that saw it), per dataset.

| Check | Result |
|---|---|
| Real car-damage photos wrongly flagged as AI | **1%** CarDD (748), **0%** DrBimmer (200) |
| AI images caught | **92%** AI-vs-Human (600); **38%** frontier-synthetic-2026, the newest generators (200) |
| Real stock photos wrongly flagged (out of domain) | 16% AI-vs-Human real (600) |
| AI-generated selfies | 46% flagged + 29% inconclusive, so **75% reach a human**; **4%** of real selfies flagged (300) |
| Licence vs selfie, Selfies & ID Images (17 people) | same person: **82%** match, 16% "manual check", 2% mismatch; different person: **0%** match, 99% mismatch |
| Document tampering, our synthetic PDFs and scans (120) | 40/40 forged flagged, 0/80 genuine |
| Hand-made forgeries on real receipts, Find it again! (200) | **3%** forged flagged, 1% genuine flagged: a known gap |
| Compression view on CASIA splices (100) | highlights a region on 52% of authentic vs 50% of spliced images, so it stays unscored |

**Decisions driven by measurement:**
* With only 40 real photos from one dataset, the calibrator flagged 28% of real CarDD photos as AI (a team member's real phone photo was flagged at 66%). Adding CarDD and AI-vs-Human cut that to 1%, at the cost of catching fewer images from the newest generators (38%). We chose fewer false alarms on genuine claims.
* Arithmetic and anomaly checks are tuned on Indian GST invoices; on foreign receipts they fired on 44–71% of *genuine* ones. They now run only on documents with GST/₹ markers (`domain_markers`), and the skip is shown as a note.
* Averaging raw detector outputs flagged only 8% of fakes, because the newest generators get tiny raw scores even when ranked correctly. A learned calibrator raised that to 74% on our first (narrow) test set.
* On real portraits SigLIP answered "AI" for most photos (AUC 0.58), so selfies use CommunityForensics alone. Their calibrator weights genuine selfies 2× (deepfakes are the rare case): unweighted it flagged 12% of real selfies, weighted 4%.
* The AI threshold is one number in `settings.yaml` (0.65), read by the analyzer and the evaluation and quoted here. At 0.5 the false-positive rate on real photos was 11%.
* ELA fired on 90% of *genuine* photos (vs 75% of AI ones), and its z-scores were identical on genuine and forged document scans, so it is shown on the heatmap but **not scored** (`DOC-IMG-01` is reserved, not emitted).
* A tile scan for partial edits fired equally on real and spliced photos (35% vs 35%), so it is **disabled** in config.

## Limitations (stated openly)

* **Partial manipulations** (an AI patch inpainted into a real photo) are **not reliably detected**. Whole-image detectors and the tile scan both missed them. The next step would be a localization model (e.g. TruFor-style).

* Hand-made edits on photographed receipts (a few pixels changed in GIMP/Paint) are almost never caught (3%): there is no PDF structure to inspect. A pixel-level forgery-localization model would be the next step.
* The newest image generators (FLUX.2, GPT-Image-1.5, Seedream-4.5) are caught less than half the time by the pre-trained detectors.
* Deepfake selfies: about half are flagged outright and the rest of the catch is *inconclusive*; a face-specific detector would raise this. The real-selfie set is small (17 people); add your own photos to `data/raw/faces_real/` and rerun.
* Detector accuracy figures come from small, partly synthetic test sets. They're indicative, not production claims.
* Clean, same-font PDF edits are caught only through arithmetic or anomaly checks.
* Video, liveness detection and registry (RTO/hospital) verification are out of scope for the prototype.
