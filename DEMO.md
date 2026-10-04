# 10-minute demo script

Before going on stage:
```bash
streamlit run app.py      # warm the models: run demo claim 1 once, then on the Analyze page click 🗑 Clear history → Yes, clear everything
# Demo 3 needs demo 1 to have run first (it reuses claim 1's photo). Demos 4-5 use public example
# faces: for the stage, put a teammate's photos in data/raw/faces/ and run scripts/make_demo_claims.py.
```
Close other heavy apps. Everything runs offline, so venue Wi-Fi doesn't matter.

| Min | PS demo step | What to click | What to say |
|---|---|---|---|
| 0:00 | Problem | (title screen) | "Generative AI makes fake damage photos and edited bills cheap. Legacy fraud tools score *behaviour*; nobody checks whether the *evidence itself* is real. Investigators need a verdict they can trust *and* verify." |
| 1:00 | ① Image + confidence | Demo **1 genuine claim** → Run | "Two real damage photos and two clean documents: **LOW**. Both AI detectors rate the photos as camera-captured, and every document passed 5–7 integrity checks. Genuine claims flow through." |
| 2:30 | ① + ② + ③ | Demo **2 fraud claim** → Run | "**HIGH**, and it says exactly why." Walk through the reason cards top to bottom: AI-generated photo (2/2 detectors, 99%), cover-up box hiding the original total, font change, arithmetic, vehicle-number and owner mismatch with the RC, invoice dated before the accident. |
| 4:00 | ② Document flags | Evidence tab → `repair_invoice.pdf` | Red boxes on the page show the retyped total. "The original ₹ amount is still underneath; we read it out." |
| 4:45 | ③ Composite score | Waterfall chart | "Every bar is one evidence group in log-odds. Contributions add, so the explanation **is** the computation. Green bars are evidence *for* the claimant; a genuine side photo can't cancel a fake front photo." |
| 5:30 | Novelty | Demo **3 reused photo** → Run | "A clean-looking claim, but the photo was cropped and recompressed from claim 1. A perceptual hash catches it: **HIGH** by rule. Synthetic-identity rings recycle evidence; single-file detectors never see it." |
| 6:00 | ⑤ Identity | Demo **4 identity match** → Run, then Demo **5 identity mismatch** → Run | "Same licence. Matching selfie: **LOW**. Someone else's selfie: **HIGH** by rule, even though the selfie itself is a real photo. Between the two thresholds we say 'manual check' instead of guessing." |
| 7:00 | ⑤ Deepfake selfie | Demo **6 deepfake selfie** → Run (or upload any AI portrait on its own) | "One AI-generated selfie with nothing else: the AI-image check runs on selfies too, so it is **HIGH** on its own. That is the synthetic-identity case." |
| 7:30 | ④ Architecture | Sidebar → **How scoring works** | Plug-in analyzers → one `Signal` contract → risk engine. Config-driven weights, pinned models, all local. Domain rules live in one YAML file. |
| 8:15 | Accuracy | Same page → *Measured accuracy* | Show the per-dataset table. "On 948 real car-damage photos from two public datasets, about 1% are wrongly flagged as AI. Older and mid-2025 generators are caught over 90%; the very newest, about 40%, so those cases go to review. Face match: 0% of different people matched across 1,800 pairs." Then say what *didn't* work, honestly: hand-made receipt edits are caught only 3% (no PDF structure to inspect); ELA/heatmap fired equally on real and spliced CASIA images, so it is unscored. |
| 9:15 | Close | — | "Screening signal, not proof. ClaimGuard tells an investigator *where to look and why*, and says INCONCLUSIVE instead of guessing." |

## Backup plan
* If the live run is slow: the history page keeps every analysed claim, and the HTML case reports in `reports/` open offline.
* If the projector cuts off the layout: browser zoom 80%.
