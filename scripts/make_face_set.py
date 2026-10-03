"""Build the face (selfie) evaluation set in data/eval/faces/.

  real : portrait photos in data/raw/faces_real/ (add your own team's selfies here; the shipped ones
         are the MIT-licensed example photos from github.com/ageitgey/face_recognition)
  ai   : face-centred crops of the AI images in data/eval/images/originals that contain a person

Every image is saved twice, as the full photo and as a selfie-style crop around the largest face,
so the set covers both framings. Variants of one source share a `group` for grouped CV.

Usage:  python scripts/make_face_set.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import face_engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data/raw/faces_real"
AI_SRC = ROOT / "data/eval/images/originals"
OUT = ROOT / "data/eval/faces"


def selfie_crop(img: Image.Image, face: dict) -> Image.Image:
    """Head-and-shoulders crop like a phone selfie: face ~ 15-25% of the frame."""
    x, y, w, h = face["bbox"]
    cx, cy, side = x + w // 2, y + h // 2, max(int(max(w, h) * 2.4), 320)
    return img.crop((max(0, cx - side // 2), max(0, cy - side // 2),
                     min(img.width, cx + side // 2), min(img.height, cy + int(side * 0.6))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fe = face_engine()
    rows = []

    def add(img: Image.Image, label: int, group: str, generator: str) -> None:
        faces = fe.faces(img)
        if not faces:
            return
        for framing, im in (("full", img), ("crop", selfie_crop(img, faces[0]))):
            if min(im.size) < 160:
                continue
            name = f"{'ai' if label else 'real'}_{len(rows):03d}_{framing}.jpg"
            im.convert("RGB").save(OUT / name, quality=92)
            rows.append({"path": name, "label": label, "group": group, "framing": framing, "generator": generator})

    for p in sorted(REAL.glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            add(Image.open(p).convert("RGB"), 0, p.stem, "camera")
    for r in csv.DictReader(open(AI_SRC / "manifest.csv")):
        if r["label"] == "1":
            add(Image.open(AI_SRC / r["path"]).convert("RGB"), 1, Path(r["path"]).stem, r["generator"])

    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "group", "framing", "generator"])
        w.writeheader(); w.writerows(rows)
    n_ai = sum(r["label"] for r in rows)
    print(f"{len(rows)} face images ({len(rows) - n_ai} real, {n_ai} AI) written to {OUT}")


if __name__ == "__main__":
    main()
