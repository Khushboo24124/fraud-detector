"""Train the invoice anomaly model (IsolationForest) on genuine repair-invoice profiles.

Unsupervised on purpose: we only ever need examples of *normal* invoices, so the
model can flag inflation patterns it has never seen (no fraud labels required).

Usage:  python scripts/train_invoice_model.py --n 600
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzers.documents import IFOREST_PATH  # noqa: E402
from core.config import ensure_dirs  # noqa: E402
from core.fields import invoice_features  # noqa: E402
from scripts.make_documents import make_case  # noqa: E402


def case_to_fields(case: dict) -> dict:
    return {"items": [{"amount": i["amount"]} for i in case["items"]], "total": case["total"],
            "subtotal": case["subtotal"], "taxes": [case["cgst"], case["sgst"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)  # different seed from the eval set
    args = ap.parse_args()
    rng = random.Random(args.seed)
    feats = [invoice_features(case_to_fields(make_case(rng))) for _ in range(args.n)]
    names = list(feats[0])
    X = np.array([[f[k] for k in names] for f in feats])
    model = IsolationForest(n_estimators=300, contamination=0.02, random_state=0).fit(X)
    ensure_dirs()
    joblib.dump({"model": model, "features": names, "mean": X.mean(0), "std": X.std(0),
                 "trained_on": f"{args.n} synthetic genuine invoices (seed {args.seed})"}, IFOREST_PATH)
    print(f"saved {IFOREST_PATH} | features={names}")


if __name__ == "__main__":
    main()
