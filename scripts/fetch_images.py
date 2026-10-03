"""Fetch a small labelled image test set from public Hugging Face datasets.

real : DrBimmer/comprehensive-car-damage  (real photos of damaged cars)
ai   : Thermostatic/frontier-synthetic-images-2026, rows whose prompt mentions a vehicle
       (recent generators: GPT-Image, FLUX, Imagen, Seedream, Qwen-Image, ...)

Usage:  python scripts/fetch_images.py --n 40 --out data/eval/images
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co"


def get(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as exc:  # datasets-server is flaky (502/timeouts) under load
            print(f"  retry {attempt + 1}/{retries}: {exc}")
            time.sleep(3 * (attempt + 1))
    return {"rows": []}


def download(url: str, path: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
        return True
    except Exception as exc:
        print("  skip", path.name, exc)
        return False


def image_url(cell) -> str | None:
    if isinstance(cell, dict):
        return cell.get("src") or cell.get("url")
    return None


def fetch_real(n: int, out: Path) -> list[dict]:
    rows, offset = [], 0
    while len(rows) < n:
        data = get(f"{API}/rows?dataset=DrBimmer/comprehensive-car-damage&config=default&split=train"
                   f"&offset={offset}&length=50")
        for r in data["rows"]:
            row = r["row"]
            if len(rows) >= n:
                break
            if r["row_idx"] % 7:  # spread across the dataset's classes
                continue
            p = out / f"real_{r['row_idx']:05d}.jpg"
            if download(image_url(row["image"]), p):
                rows.append({"path": p.name, "label": 0, "source": "DrBimmer/comprehensive-car-damage",
                             "generator": "camera"})
        offset += 50
    return rows


VEHICLE_WORDS = ("car", "vehicle", "truck", "bumper", "accident", "road", "street", "garage",
                 "parking", "motor", "taxi", "van", "suv", "traffic")


def fetch_ai(n: int, out: Path, total_rows: int = 40290, probes: int = 14) -> list[dict]:
    """Rows are grouped by generator, so probe evenly spaced offsets for a diverse mix."""
    rows = []
    per_probe = max(2, -(-n // probes))
    for k in range(probes):
        offset = int(k * (total_rows - 100) / max(1, probes - 1))
        data = get(f"{API}/rows?dataset=Thermostatic/frontier-synthetic-images-2026&config=default"
                   f"&split=train&offset={offset}&length=100")
        cands = [r["row"] for r in data["rows"] if r["row"].get("label") == 1]
        cands.sort(key=lambda r: not any(w in (r.get("prompt") or "").lower() for w in VEHICLE_WORDS))
        for row in cands[:per_probe]:
            if len(rows) >= n:
                break
            gen = row.get("generator") or "unknown"
            ext = ".png" if "png" in (row.get("mime") or "") else ".jpg"
            p = out / f"ai_{len(rows):04d}_{gen.replace('/', '_').replace(' ', '_')[:30]}{ext}"
            url = image_url(row["image"])
            if url and download(url, p):
                rows.append({"path": p.name, "label": 1, "source": "Thermostatic/frontier-synthetic-images-2026",
                             "generator": gen})
        print(f"  probe {k + 1}/{probes}: {len(rows)} AI images")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/eval/images")
    args = ap.parse_args()
    out = Path(args.out) / "originals"
    out.mkdir(parents=True, exist_ok=True)
    real = sorted(out.glob("real_*.jpg"))
    if len(real) >= args.n:  # resume: keep already-downloaded real photos
        real_rows = [{"path": p.name, "label": 0, "source": "DrBimmer/comprehensive-car-damage",
                      "generator": "camera"} for p in real[:args.n]]
    else:
        real_rows = fetch_real(args.n, out)
    rows = real_rows + fetch_ai(args.n, out)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "generator"])
        w.writeheader(); w.writerows(rows)
    print(f"real={sum(r['label'] == 0 for r in rows)} ai={sum(r['label'] == 1 for r in rows)} -> {out}")


if __name__ == "__main__":
    main()
