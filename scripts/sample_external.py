"""Build a small, reproducible evaluation sample from the Kaggle datasets in data/raw/ -> data/external/.

Expected downloads (kaggle datasets download ... -p data/raw/<name> --unzip):
  data/raw/cardd/test2017/*.jpg                         CarDD (real car damage)
  data/raw/ai_vs_human/train.csv + train_data/          AI vs Human-Generated Images (label 1 = AI)
  data/raw/casia_v2/{real,tampered,ground truth}/       CASIA v2 splicing + masks
  data/raw/find_it_again/findit2/{train,val,test}[.txt] Find it again! receipts (forged by people)
  data/raw/selfies_id/selfies-id-images-dataset.zip     Selfies & ID Images

Sampling (seeded): all CarDD test photos; 300 real + 300 AI from AI-vs-Human; first 50 CASIA
authentic + 50 tampered (+ masks); 100 forged + 100 genuine receipts; per person ID_1, ID_2 and
Selfie_1..4. Large photos are resized to <= 1600 px (claims arrive resized anyway); receipts and
CASIA files keep their pixels (PNG re-saved losslessly, CASIA copied byte-for-byte).

Usage:  python scripts/sample_external.py
"""
from __future__ import annotations

import csv
import io
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/external"


def save_jpeg(img: Image.Image, path: Path, max_side: int = 1600) -> None:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=93)


def main() -> None:
    rng = random.Random(42)
    rows = []

    for f in sorted((RAW / "cardd/test2017").glob("*")):
        dst = OUT / "cardd" / f"{f.stem}.jpg"
        if not dst.exists():
            save_jpeg(Image.open(f), dst)
        rows.append(("cardd", f"cardd/{dst.name}", 0, "CarDD"))

    av = list(csv.DictReader(open(RAW / "ai_vs_human/train.csv")))
    for lab in ("0", "1"):
        for r in rng.sample([r for r in av if r["label"] == lab], 300):
            dst = OUT / "ai_vs_human" / f"{lab}_{Path(r['file_name']).stem}.jpg"
            if not dst.exists():
                save_jpeg(Image.open(RAW / "ai_vs_human" / r["file_name"]), dst)
            rows.append(("ai_vs_human", f"ai_vs_human/{dst.name}", int(lab), "AI-vs-Human"))

    for sub, pre, lab in (("real", "au", 0), ("tampered", "tp", 1), ("ground truth", "gt", None)):
        for k in range(1, 51):
            src = next(iter(sorted((RAW / "casia_v2" / sub).glob(f"{pre}_{k:03d}.*"))), None)
            if src is None:
                continue
            dst = OUT / "casia" / sub.replace(" ", "_") / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            if lab is not None:
                rows.append(("casia", f"casia/{dst.parent.name}/{src.name}", lab, "CASIA v2"))

    fi = []
    for split in ("train", "val", "test"):
        for r in csv.DictReader(open(RAW / f"find_it_again/findit2/{split}.txt")):
            if (RAW / f"find_it_again/findit2/{split}/{r['image']}").exists():
                fi.append((split, r["image"], int(r["forged"])))
    pick = rng.sample([x for x in fi if x[2] == 1], 100) + rng.sample([x for x in fi if x[2] == 0], 100)
    (OUT / "findit").mkdir(parents=True, exist_ok=True)
    for split, img, lab in pick:
        dst = OUT / "findit" / img
        if not dst.exists():
            im = Image.open(RAW / f"find_it_again/findit2/{split}/{img}"); im.load()
            im.save(dst, compress_level=6)              # lossless: same pixels, smaller file
        rows.append(("findit", f"findit/{img}", lab, split))

    z = zipfile.ZipFile(RAW / "selfies_id/selfies-id-images-dataset.zip")
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) < 4 or not parts[-1]:
            continue
        person, base = parts[2].split("--")[-1].strip().replace(" ", "_"), Path(parts[-1]).stem
        if base in ("ID_1", "ID_2", "Selfie_1", "Selfie_2", "Selfie_3", "Selfie_4"):
            dst = OUT / "selfies_id" / person / f"{base}.jpg"
            if not dst.exists():
                save_jpeg(Image.open(io.BytesIO(z.read(n))), dst, 1280)
            rows.append(("selfies_id", f"selfies_id/{person}/{base}.jpg", 0, "Selfies & ID"))

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["set", "path", "label", "source"]); w.writerows(rows)
    print(f"{len(rows)} files sampled into {OUT}")


if __name__ == "__main__":
    main()
