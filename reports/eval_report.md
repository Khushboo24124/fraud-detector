# ClaimGuard evaluation report

One AI-image threshold everywhere: calibrated P(AI) **≥ 0.65** is flagged; 0.4–0.65 is held as *inconclusive* (sent to manual review, never auto-cleared).

Claim-level *flagged* = tier MEDIUM or HIGH. INCONCLUSIVE counts as *not* flagged.

## Damage photos: AI-generated detection (headline)

n = 2348 image scores (1548 real, 800 AI) from all image datasets, each photo also as quality variants. Calibrator scores are out-of-fold: no image is scored by a calibrator that saw it. **The per-dataset table further down matters more than this average.**

| Test | AUC | AI images flagged @0.65 | Real photos flagged @0.65 | AI → inconclusive | Real → inconclusive |
|---|---|---|---|---|---|
| 5-fold CV, grouped by source photo | 0.915 | 78% | 7% | 6% | 9% |
| **Unseen generator** (leave-one-generator-out) | 0.919 | 79% | 7% | 8% | 10% |

Robustness, AUC by upload quality: original 0.952, jpeg70 0.900, half_res 0.899, whatsapp 0.945, noise 0.922

### Unseen-generator test, per generator (all quality variants)

Each row is scored by a calibrator that never saw that generator. The detectors themselves are pre-trained and were never tuned on any of these images; generators such as FLUX.2, GPT-Image-1.5 and Seedream-4.5 were released after the CVPR 2025 CommunityForensics model.

| Generator | AI images flagged @0.65 |
|---|---|
| flux-1.1-pro | 100% |
| janus-7b | 100% |
| ai_vs_human | 93% |
| frames-23-1-25 | 80% |
| lumina-17-2-25 | 80% |
| Imagen-4.0-Ultra | 67% |
| stable-diffusion-3 | 60% |
| FLUX.2-pro | 53% |
| imagen-4-ultra-24-7-25 | 50% |
| imagen-4.0-ultra-generate-exp-05-20 | 40% |
| Qwen-Image-2512 | 33% |
| seedream-3-24-7-25 | 33% |
| 4o-26-3-25 | 0% |
| GPT-Image-1.5 | 0% |
| Seedream-4.5 | 0% |
| black-forest-labs/FLUX.2-klein-4B | 0% |
| flux-2-pro-25-11-25 | 0% |
| kling_v2_1 | 0% |
| recraft-v3-24-7-25 | 0% |

## Damage photos: end-to-end claim tiers

Whole pipeline on single-photo claims (calibrator fitted on all images, so this is in-sample; use the table above for honest rates).

| Variant | AUC (image risk) | AUC CommunityForensics-ViT-S/384 | AUC SigLIP-ai-vs-human | Fakes flagged | Real flagged (false positive) | Real → HIGH |
|---|---|---|---|---|---|---|
| original | 0.902 | 0.896 | 0.739 | 35% (14/40) | 0% (0/40) | 0% (0/40) |
| jpeg70 | 0.908 | 0.877 | 0.749 | 25% (10/40) | 0% (0/40) | 0% (0/40) |
| half_res | 0.867 | 0.904 | 0.767 | 38% (15/40) | 0% (0/40) | 0% (0/40) |
| whatsapp | 0.879 | 0.868 | 0.738 | 28% (11/40) | 0% (0/40) | 0% (0/40) |
| noise | 0.956 | 0.892 | 0.758 | 18% (7/40) | 0% (0/40) | 0% (0/40) |

**Partial manipulation (AI patch spliced into real photo):** flagged 0% (0/40), localized-region signal fired 0% (0/40). Known limitation.

Median latency per photo: 612 ms (CPU).

## Selfies: AI-generated / deepfake face detection

n = 376 selfie images (300 real, 76 AI; full photo + selfie crop, as uploaded and WhatsApp-recompressed). 5-fold CV grouped by source photo.

| AUC | AI faces flagged @0.65 | Real faces flagged @0.65 | AI → inconclusive | Real → inconclusive |
|---|---|---|---|---|
| 0.848 | 46% | 4% | 29% | 10% |

So 75% of AI selfies reach a human (flagged or inconclusive), while 4% of genuine selfies are flagged. The face calibrator weights genuine selfies 2x because deepfakes are the rare case (unweighted, 3x more real selfies were flagged).

Why selfies use only CommunityForensics: on real portraits the SigLIP detector answered 'AI' for most photos (AUC 0.58 on this set), so averaging it in made real selfies look fake.

End-to-end single-selfie claims: AI faces flagged 36% (27/76), real faces flagged 3% (2/68).

## Document tampering (on our synthetic document set)

AUC (document risk): **1.000**

| Variant | n | Flagged | Most common reason codes |
|---|---|---|---|
| T1_cover_and_retype | 8 | 100% (8/8) | DOC-FONT-01×8, DOC-MATH-01×8, DOC-META-03×8 |
| T2_redact_clean | 8 | 100% (8/8) | DOC-MATH-01×8, DOC-OVL-01×8, DOC-ANOM-01×1 |
| T3_inflate_item | 8 | 100% (8/8) | DOC-FONT-01×8, DOC-MATH-01×8, DOC-OVL-01×8 |
| T4_image_paste | 8 | 100% (8/8) | DOC-MATH-01×8 |
| T5_metadata_edit | 8 | 100% (8/8) | DOC-META-01×8, DOC-META-03×8 |
| genuine_pdf | 40 | 0% (0/40) |  |
| genuine_scan | 40 | 0% (0/40) |  |

Median latency per document: 32 ms.

## Results per dataset (public data, out-of-fold)

Each image is scored by a calibrator that never saw it (5-fold CV grouped by source photo). Real sets show the **false-alarm** rate, AI sets the **detection** rate, at the 0.65 threshold.

| Dataset | Kind | n | Flagged as AI |
|---|---|---|---|
| AI-vs-Human (real) | real | 600 | 16% |
| CarDD car damage (real) | real | 748 | 1% |
| DrBimmer car damage (real) | real | 200 | 0% |
| Selfies & ID dataset (real selfies) | real | 232 | 4% |
| face_recognition examples (real) | real | 68 | 4% |
| AI-vs-Human (AI) | AI | 600 | 92% |
| frontier AI faces | AI | 76 | 46% |
| frontier-synthetic-2026 (AI) | AI | 200 | 38% |

## Identity: licence vs selfie (Selfies & ID Images)

17 people, 92 genuine pairs (own ID vs own selfie), 1798 impostor pairs (ID vs someone else's selfie).

| Pair type | Match (≥ 0.45) | Uncertain (0.30–0.45) | Mismatch (< 0.30) | Median similarity |
|---|---|---|---|---|
| Same person | 82% | 16% | 2% | 0.53 |
| Different person | 0% | 1% | 99% | 0.07 |

## Documents: real forgeries made by people (Find it again!)

100 receipts forged by hand (GIMP, Paint, ...) and 100 genuine ones, as images.

| AUC (document risk) | Forged flagged | Genuine flagged | Most common reasons on forged |
|---|---|---|---|
| 0.596 | 3% | 1% | DOC-FONT-02×24, DOC-ANOM-01×3, DOC-MATH-01×3 |

Compare with 100% on our synthetic PDFs: hand-made forgeries on photographed receipts are much harder, because there is no PDF structure to inspect and the edits are only a few pixels wide.

## Partial edits / heatmap check (CASIA v2 splices)

| | Authentic (50) | Spliced (50) |
|---|---|---|
| Compression view highlights a region | 52% | 50% |

On spliced images, a highlighted box overlaps the true edited area in 28% of cases. This is why the compression view is shown to investigators but not scored.

## Caveats

- Before the public datasets were added, the calibrator had seen only 40 real photos from one dataset. It then flagged 28% of real CarDD photos and 60% of AI-vs-Human real photos as AI; with them, 1% and 16%. The price is honest: the newest generators (frontier set) are caught 38% of the time, not the 74% that the narrow test suggested.
- Real photos: public car-damage dataset; AI images: recent generators (frontier-synthetic-images-2026). Many AI images show people or scenes rather than car damage, so scene content differs between classes.
- Real faces: 17 public example photos plus the Selfies & ID Images sample (29 people). AI faces are crops of frontier-synthetic images, so the AI-face set is small (38 faces).
- The AI-vs-Human real photos are stock photography (often retouched); 16% of them are flagged, which is why the damage-photo profile should be judged on the car-damage rows (CarDD, DrBimmer).
- Documents and tampering are synthetic (scripts/make_documents.py) and made by the same generator as the genuine ones, so 100% on documents means *on our synthetic set*; real forgeries will be more varied.
- Small sample sizes: treat rates as indicative, not as production accuracy claims.