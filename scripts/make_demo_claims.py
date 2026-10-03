"""Build ready-to-load demo claims in data/samples/<claim>/ (files + info.json).

  1_genuine_claim     real damage photos + consistent invoice and RC         -> expect LOW
  2_fraud_claim       AI-generated photo, retyped invoice total, RC of another
                      owner/vehicle, invoice dated before the accident          -> expect HIGH
  3_reused_photo      genuine-looking claim that recycles claim 1's photo      -> expect HIGH (run after 1)
  4_identity_match    licence + selfie of the same person                    -> expect LOW
  5_identity_mismatch same licence + a different person's selfie             -> expect HIGH
  6_deepfake_selfie   one AI-generated selfie, uploaded on its own             -> expect HIGH

Faces for 4/5 come from data/raw/faces/{licence,selfie_same,selfie_other}.jpg. The shipped ones are
public example photos; for the live demo, swap in a teammate's licence photo and selfies.

Usage:  python scripts/make_demo_claims.py   (needs data/eval/images and the AI models)
"""
from __future__ import annotations

import csv
import io
import json
import random
import shutil
import sys
from datetime import timedelta
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.make_documents import (make_case, render_invoice, render_licence, render_rc,  # noqa: E402
                                    t1_cover_and_retype)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/eval/images/originals"
OUT = ROOT / "data/samples"


def pick_ai_image() -> Path:
    """The AI photo the ensemble is most confident about: the demo should show a clear case."""
    from core.models import ai_detectors
    rows = [r for r in csv.DictReader(open(SRC / "manifest.csv")) if r["label"] == "1"]
    best, best_p = None, -1.0
    for r in rows[:25]:
        img = Image.open(SRC / r["path"]).convert("RGB")
        p = min(d.predict([img])[0] for d in ai_detectors())
        if p > best_p:
            best, best_p = SRC / r["path"], p
    return best


def pick_ai_face() -> Path:
    """The selfie-like AI crop (face fills >= 8% of a >= 400 px frame) the face profile is surest about."""
    from core.analyzers.image_ai import AIImageAnalyzer
    from core.models import face_engine
    faces = ROOT / "data/eval/faces"
    rows = [r for r in csv.DictReader(open(faces / "manifest.csv")) if r["label"] == "1" and r["framing"] == "crop"]
    profile = AIImageAnalyzer().profiles["face"]
    scored = []
    for r in rows:
        img = Image.open(faces / r["path"]).convert("RGB")
        found = face_engine().faces(img)
        if min(img.size) < 400 or not found or \
                found[0]["bbox"][2] * found[0]["bbox"][3] < 0.08 * img.width * img.height:
            continue
        scored.append((profile.fuse({det.name: det.predict([img])[0] for det in profile.detectors}), faces / r["path"]))
    return max(scored)[1]


def passport_crop(img: Image.Image) -> Image.Image:
    """Licence-style head shot (25:32) around the largest face."""
    from core.models import face_engine
    img = img.convert("RGB")
    found = face_engine().faces(img)
    if not found:
        return img
    x, y, w, h = found[0]["bbox"]
    cw = int(w * 1.9); ch = int(cw * 32 / 25)
    cx, cy = x + w // 2, y + int(h * 0.45)
    return img.crop((max(0, cx - cw // 2), max(0, cy - ch // 2), min(img.width, cx + cw // 2), min(img.height, cy + ch // 2)))


def as_phone_jpeg(src: Path, dst: Path, max_side: int = 1600, quality: int = 88) -> None:
    img = Image.open(src).convert("RGB")
    img.thumbnail((max_side, max_side))
    img.save(dst, "JPEG", quality=quality)


def write_info(folder: Path, **info) -> None:
    (folder / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    rng = random.Random(2026)
    if OUT.exists():
        shutil.rmtree(OUT)
    real = sorted(SRC.glob("real_*.jpg"))

    # 1. genuine
    case = make_case(rng)
    d = OUT / "1_genuine_claim"; d.mkdir(parents=True)
    for k, src in enumerate(real[3:5], 1):
        as_phone_jpeg(src, d / f"damage_photo_{k}.jpg")
    render_invoice(case, d / "repair_invoice.pdf")
    render_rc(case, d / "rc_certificate.pdf")
    write_info(d, claimant_name=case["owner"], policy_no="POL-MTR-884213",
               vehicle_no=case["plate"], accident_date=case["accident_date"].isoformat())

    # 2. fraud
    case2 = make_case(rng)
    d = OUT / "2_fraud_claim"; d.mkdir(parents=True)
    ai = pick_ai_image()
    as_phone_jpeg(ai, d / "damage_photo_front.jpg")
    as_phone_jpeg(real[9], d / "damage_photo_side.jpg")
    case2["invoice_date"] = case2["accident_date"] - timedelta(days=6)   # billed before the accident
    tmp = d / "_clean_invoice.pdf"
    render_invoice(case2, tmp)
    t1_cover_and_retype(tmp, d / "repair_invoice.pdf", case2, rng)
    tmp.unlink()
    other = make_case(rng)
    render_rc(case2, d / "rc_certificate.pdf", plate_override=other["plate"], owner_override=other["owner"])
    write_info(d, claimant_name=case2["owner"], policy_no="POL-MTR-990417",
               vehicle_no=case2["plate"], accident_date=case2["accident_date"].isoformat())

    # 3. recycled photo from claim 1 (cropped + recompressed so the file hash differs)
    case3 = make_case(rng)
    d = OUT / "3_reused_photo"; d.mkdir(parents=True)
    img = Image.open(OUT / "1_genuine_claim/damage_photo_1.jpg")
    w, h = img.size
    img.crop((int(w * .03), int(h * .03), int(w * .97), int(h * .97))).save(d / "damage_photo.jpg", quality=80)
    render_invoice(case3, d / "repair_invoice.pdf")
    render_rc(case3, d / "rc_certificate.pdf")
    write_info(d, claimant_name=case3["owner"], policy_no="POL-MTR-551902",
               vehicle_no=case3["plate"], accident_date=case3["accident_date"].isoformat())

    # 4-5. identity: same licence, matching vs non-matching selfie (same claimant name, so the
    # cross-claim face index does not flag claim 5 as face reuse; the mismatch must stand alone)
    faces_dir = ROOT / "data/raw/faces"
    case4 = make_case(rng)
    licence_face = passport_crop(Image.open(faces_dir / "licence.jpg"))
    for folder, selfie, note in [("4_identity_match", "selfie_same.jpg", "Licence and selfie show the same person."),
                                 ("5_identity_mismatch", "selfie_other.jpg", "Selfie shows a different person.")]:
        d = OUT / folder; d.mkdir(parents=True)
        render_licence(case4, d / "driving_licence.jpg", licence_face)
        if folder.startswith("5"):  # same card, re-photographed: different bytes, so no file-reuse hit
            Image.open(d / "driving_licence.jpg").save(d / "driving_licence.jpg", quality=89)
        as_phone_jpeg(faces_dir / selfie, d / "selfie.jpg")
        write_info(d, claimant_name=case4["owner"], policy_no="POL-MTR-310078",
                   vehicle_no=case4["plate"], accident_date=case4["accident_date"].isoformat(), note=note)

    # 6. deepfake selfie: one AI-generated face, uploaded alone (PS demo step 1 for identity)
    d = OUT / "6_deepfake_selfie"; d.mkdir(parents=True)
    as_phone_jpeg(pick_ai_face(), d / "selfie.jpg")
    write_info(d, note="AI-generated selfie with no other evidence: the image check alone must catch it.")
    print("demo claims written to", OUT)


if __name__ == "__main__":
    main()
