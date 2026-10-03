"""Pre-fetch every model into data/models so the demo runs fully offline.

Usage:  python scripts/download_models.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from core import models  # noqa: E402


def main() -> None:
    probe = Image.new("RGB", (512, 512), (120, 120, 120))
    for label, loader in [("AI-image detectors", models.ai_detectors),
                          ("Face engine (YuNet + SFace)", models.face_engine),
                          ("OCR (RapidOCR)", models.ocr_engine)]:
        t = time.time()
        obj = loader()
        print(f"[ok] {label} loaded in {time.time() - t:.1f}s")
        if label.startswith("AI"):
            for d in obj:
                t = time.time()
                p = d.predict([probe])[0]
                print(f"     {d.name}@{d.version}: probe P(ai)={p:.3f}  ({(time.time() - t) * 1000:.0f} ms)")
    print("All models cached under data/models. The app can now run offline.")


if __name__ == "__main__":
    main()
