"""Turn positioned text (PDF spans or OCR boxes) into rows and structured claim fields.

Both PDF text and OCR output are normalised into `TextEl` so every downstream rule
works identically for digital PDFs, scans and phone photos of documents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from .config import domain_rules

AMOUNT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d+)\.(\d{2})(?!\d)")
DATE_LABELS = {
    "invoice_date": ["invoice date", "bill date", "date of invoice", "inv. date", "inv date"],
    "accident_date": ["date of accident", "accident date", "date of loss", "loss date"],
    "registration_date": ["date of registration", "regn. date", "reg. date", "registration date"],
    "issue_date": ["date of issue", "issue date", "doi"],
}


@dataclass
class TextEl:
    text: str
    box: tuple[float, float, float, float]   # x0, y0, x1, y1
    page: int = 0
    meta: dict[str, Any] = field(default_factory=dict)  # font, size, conf, seqno ...

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2


def build_rows(els: list[TextEl]) -> list[list[TextEl]]:
    """Group elements into visual rows (same page, overlapping vertical centre)."""
    rows: list[list[TextEl]] = []
    for el in sorted(els, key=lambda e: (e.page, e.cy, e.box[0])):
        if rows:
            last = rows[-1]
            ref_h = float(np.median([e.height for e in last]))
            if last[0].page == el.page and abs(el.cy - np.mean([e.cy for e in last])) <= 0.5 * max(ref_h, 1):
                last.append(el)
                continue
        rows.append([el])
    return [sorted(r, key=lambda e: e.box[0]) for r in rows]


def row_text(row: list[TextEl]) -> str:
    return "  ".join(e.text.strip() for e in row if e.text.strip())


def parse_amounts(s: str) -> list[float]:
    return [float(m.group(1).replace(",", "") + "." + m.group(2)) for m in AMOUNT_RE.finditer(s)]


def parse_date(s: str) -> date | None:
    m = re.search(domain_rules()["fields"]["date"], s)
    if not m:
        return None
    d, mth, y = (int(g) for g in m.groups())
    y = y + 2000 if y < 100 else y
    try:
        return date(y, mth, d)
    except ValueError:
        return None


def normalise_plate(s: str) -> str:
    return re.sub(r"[\s\-]", "", s.upper())


def find_plates(text: str) -> list[str]:
    pat = domain_rules()["fields"]["vehicle_no"]
    cleaned = text.upper().replace("O", "0").replace("I", "1")  # OCR confusions in digit groups
    out = []
    for src in (text.upper(), cleaned):
        for m in re.finditer(pat, src):
            plate = "".join(m.groups())
            # state code and series must be letters in the original text
            if plate[:2].isalpha():
                out.append(plate)
    return list(dict.fromkeys(out))


def classify_document(text: str) -> str | None:
    lower = text.lower()
    scores = {t: sum(lower.count(k) for k in kws) for t, kws in domain_rules()["document_types"].items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def label_rx(label: str) -> str:
    """OCR often drops spaces ('CustomerName', 'GrandTotal'), so spaces are optional."""
    return re.escape(label).replace(r"\ ", r"\s*")


def _label_value(rows_text: list[str], labels: list[str], pattern: str) -> str | None:
    for label in sorted(labels, key=len, reverse=True):
        rx = re.compile(rf"\b{label_rx(label)}\b\s*[:\-]?\s*{pattern}", re.I)
        for r in rows_text:
            m = rx.search(r)
            if m:
                return m.group(1).strip()
    return None


def extract_fields(rows: list[list[TextEl]]) -> dict[str, Any]:
    rules = domain_rules()["fields"]
    texts = [row_text(r) for r in rows]
    full = "\n".join(texts)
    f: dict[str, Any] = {"vehicle_numbers": find_plates(full)}

    inv = re.search(rules["invoice_no"], full, re.I)
    f["invoice_no"] = inv.group(1) if inv else None
    name = _label_value(texts, rules["name_labels"], r"([A-Za-z][A-Za-z .]{2,40}?)(?:\s{2,}|$)")
    f["owner_name"] = re.sub(r"\s+", " ", name).strip(" .").title() if name else None

    for key, labels in DATE_LABELS.items():
        val = _label_value(texts, labels, r"(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})")
        f[key] = parse_date(val).isoformat() if val and parse_date(val) else None

    # Money: label rows vs line-item rows
    def is_label(t: str, labels: list[str]) -> bool:
        return any(re.search(rf"\b{label_rx(l)}\b", t, re.I) for l in labels)

    total = subtotal = None
    taxes, items = [], []
    for t, row in zip(texts, rows):
        amts = parse_amounts(t)
        if not amts:
            continue
        if is_label(t, rules["subtotal_labels"]):
            subtotal = amts[-1]
        elif is_label(t, rules["total_labels"]):
            total = amts[-1]
        elif is_label(t, rules["tax_labels"]):
            taxes.append(amts[-1])
        elif re.search(r"[A-Za-z]{3,}", t) and not re.search(r"\b(date|phone|mobile|gstin|policy|invoice)\b", t, re.I):
            items.append({"text": t, "amount": amts[-1], "row": row})
    f.update(total=total, subtotal=subtotal, taxes=taxes,
             items=[{"text": i["text"], "amount": i["amount"]} for i in items])
    f["_item_rows"] = [i["row"] for i in items]
    return f


def math_checks(f: dict[str, Any], tol: float) -> list[dict[str, Any]]:
    """Arithmetic consistency of an invoice: items -> subtotal -> + tax -> total."""
    issues = []
    items_sum = round(sum(i["amount"] for i in f.get("items", [])), 2)
    sub, total, taxes = f.get("subtotal"), f.get("total"), f.get("taxes", [])
    if f.get("items") and sub is not None and abs(items_sum - sub) > tol:
        issues.append({"label": "Line items vs subtotal", "expected": items_sum, "stated": sub})
    elif f.get("items") and sub is None and not taxes and total is not None and abs(items_sum - total) > tol:
        issues.append({"label": "Line items vs total", "expected": items_sum, "stated": total})
    if sub is not None and taxes and total is not None:
        expected = round(sub + sum(taxes), 2)
        if abs(expected - total) > tol:
            issues.append({"label": "Subtotal + tax vs grand total", "expected": expected, "stated": total})
    for i in issues:
        i["diff"] = abs(i["expected"] - i["stated"])
    return issues


def invoice_features(f: dict[str, Any]) -> dict[str, float] | None:
    """Numeric profile of an invoice for the IsolationForest anomaly model."""
    items = [i["amount"] for i in f.get("items", [])]
    total = f.get("total")
    if not items or not total:
        return None
    sub = f.get("subtotal") or sum(items)
    return {
        "log_total": float(np.log10(max(total, 1))),
        "n_items": float(len(items)),
        "max_item_share": float(max(items) / max(sum(items), 1)),
        "tax_rate": float(sum(f.get("taxes", [])) / max(sub, 1)),
        "round_share": float(np.mean([a % 100 == 0 for a in items])),
        "log_max_item": float(np.log10(max(max(items), 1))),
    }
