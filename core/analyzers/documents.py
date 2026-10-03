"""Document tampering: PDF structure/fonts/overlays, OCR geometry, arithmetic, anomaly model."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import MODEL_DIR, domain_rules, settings
from ..fields import (TextEl, build_rows, classify_document, extract_fields, invoice_features,
                      math_checks, row_text)
from ..models import ocr
from ..schemas import EvidenceItem, Kind, Role
from .base import Analyzer, Context

EDIT_TOOLS = ["ilovepdf", "smallpdf", "sejda", "pdfescape", "pdf-xchange", "phantompdf", "sodapdf",
              "pdffiller", "dochub", "pdf editor", "pdfelement", "canva", "photoshop", "nitro pro",
              "online2pdf", "pdf24", "foxit pdf editor", "inkscape", "libreoffice draw"]
ROLE_BY_TYPE = {"invoice": Role.INVOICE, "rc": Role.RC, "licence": Role.LICENCE, "claim_form": Role.CLAIM_FORM}
IFOREST_PATH = MODEL_DIR / "invoice_iforest.joblib"


def font_family(font: str) -> str:
    """'ABCDEF+TimesNewRomanPS-BoldMT' -> 'timesnewromanps' (ignore subset prefix and weight)."""
    name = font.split("+")[-1]
    return re.split(r"[-,]", name)[0].lower()


def _pdf_date(s: str | None) -> datetime | None:
    m = re.match(r"D:(\d{14})", s or "")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def _inside(inner, outer, pad=1.0) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] - pad <= cx <= outer[2] + pad and outer[1] - pad <= cy <= outer[3] + pad


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ocr_elements(ctx: Context, item: EvidenceItem) -> list[TextEl]:
    """OCR every page once per item; cached for the plate reader and role classifier too."""
    s = ctx.slot(item)
    if "ocr" not in s:
        s["ocr"] = [TextEl(l["text"], tuple(l["box"]), page=i, meta={"conf": l["conf"]})
                    for i, page in enumerate(ctx.pages(item, dpi=200)) for l in ocr(page)]
    return s["ocr"]


class DocumentAnalyzer(Analyzer):
    name = "document_forensics"
    version = "1.0"

    def __init__(self) -> None:
        self.iforest = None
        if IFOREST_PATH.exists():
            import joblib
            self.iforest = joblib.load(IFOREST_PATH)

    def applies_to(self, item: EvidenceItem) -> bool:
        return item.kind is Kind.PDF or item.role.is_document

    # ------------------------------------------------------------------ entry
    def analyze(self, item, ctx: Context):
        signals: list = []
        if item.kind is Kind.PDF:
            els, reliability = self._pdf(item, signals)
            if sum(len(e.text) for e in els) < 30:          # scanned PDF: no text layer
                els, reliability = ocr_elements(ctx, item), 0.8
        else:
            els = ocr_elements(ctx, item)
            confs = [e.meta.get("conf", 0.0) for e in els]
            reliability = item.reliability * (float(np.mean(confs)) if confs else 0.0)
            signals += self._ocr_geometry(item, els, reliability)

        rows = build_rows(els)
        item.text = "\n".join(row_text(r) for r in rows)
        if len(item.text) < 30:
            signals.append(self.signal(item, "DOC-Q-01", None, chars=len(item.text)))
            return signals

        doc_type = classify_document(item.text)
        if doc_type and item.role in (Role.OTHER_DOC, Role.DAMAGE_PHOTO):
            item.role = ROLE_BY_TYPE[doc_type]
        fields = extract_fields(rows)
        fields.pop("_item_rows", None)
        fields["doc_type"] = doc_type
        item.fields = fields

        markers = domain_rules().get("domain_markers", [])
        in_domain = not markers or any(m in item.text.lower() for m in markers)
        if in_domain:
            tol = domain_rules()["consistency"]["amount_tolerance"]
            for issue in math_checks(fields, tol):
                signals.append(self.signal(item, "DOC-MATH-01", 0.85, reliability, **issue))
            signals += self._anomaly(item, fields, reliability)
        else:
            signals.append(self.signal(item, "DOC-DOM-00", None, markers=", ".join(markers)))

        # Clean documents must produce positive evidence too, or "no flags" would be
        # indistinguishable from "nothing was checked".
        if not any(s.fraud_prob and s.fraud_prob > 0.5 for s in signals):
            passed = (["metadata", "revision history", "font consistency", "cover-up boxes", "overlapping text"]
                      if item.kind is Kind.PDF else ["character geometry"])
            passed += ["arithmetic"] if fields.get("total") and in_domain else []
            passed += ["anomaly model"] if self.iforest is not None and item.role is Role.INVOICE and in_domain else []
            signals.append(self.signal(item, "DOC-OK-00", 0.3, reliability * 0.8,
                                       checks=len(passed), passed=", ".join(passed)))
        return signals

    # ------------------------------------------------------------------ PDF
    def _pdf(self, item: EvidenceItem, signals: list) -> tuple[list[TextEl], float]:
        import pymupdf as fitz

        raw = Path(item.path).read_bytes()
        max_pages = settings()["limits"]["max_pdf_pages"]
        els: list[TextEl] = []
        with fitz.open(item.path) as doc:
            meta = doc.metadata or {}
            producer = f"{meta.get('producer', '')} {meta.get('creator', '')}".strip()
            tool = next((t for t in EDIT_TOOLS if t in producer.lower()), None)
            if tool:
                signals.append(self.signal(item, "DOC-META-03", 0.7, 1.0, tool=producer))
            created, modified = _pdf_date(meta.get("creationDate")), _pdf_date(meta.get("modDate"))
            if created and modified and (modified - created).total_seconds() > 3600:
                days = max(1, (modified - created).days)
                signals.append(self.signal(item, "DOC-META-01", 0.68, 0.9, days=days,
                                           producer=producer or "unknown"))
            revisions = raw.count(b"%%EOF") - (1 if getattr(doc, "is_fast_webaccess", False) else 0)
            if revisions > 1:
                signals.append(self.signal(item, "DOC-META-02", 0.72, 0.9, revisions=revisions))

            for pno, page in enumerate(list(doc)[:max_pages]):
                page_els = []
                for sp in page.get_texttrace():
                    text = "".join(chr(c[0]) for c in sp["chars"])
                    if not text.strip() or sp.get("type") == 3:   # skip invisible OCR layers
                        continue
                    page_els.append(TextEl(text, tuple(sp["bbox"]), pno,
                                           {"font": sp["font"], "size": sp["size"], "seqno": sp["seqno"]}))
                found, hidden = self._overlays(item, page, pno, page_els)
                signals += found
                # Extract fields from what a reader actually sees, not from covered-up text.
                els += [e for e in page_els if id(e) not in hidden]
        signals += self._fonts(item, els)
        return els, 1.0

    def _overlays(self, item, page, pno, els: list[TextEl]):
        """Cover-up boxes (white rectangle drawn over text) and stacked text runs.

        Returns the signals plus the ids of text elements that are hidden from view.
        """
        out, hidden_ids = [], set()
        pw, ph = page.rect.width, page.rect.height
        for d in page.get_drawings():
            fill, r = d.get("fill"), d["rect"]
            if not fill or min(fill) < 0.9 or r.width > 0.6 * pw or r.height > 0.1 * ph or r.width < 4:
                continue
            box = (r.x0, r.y0, r.x1, r.y1)
            under = [e for e in els if _inside(e.box, box) and e.meta["seqno"] < d["seqno"]]
            over = [e for e in els if _inside(e.box, box) and e.meta["seqno"] > d["seqno"]]
            hidden_ids.update(id(e) for e in under)
            if over and under:
                out.append(self.signal(item, "DOC-OVL-01", 0.92, 1.0, page=pno + 1,
                                       text=" ".join(e.text for e in over),
                                       hidden=" ".join(e.text for e in under), box=list(box)))
            elif over and r.width * r.height < 4 * sum((e.box[2] - e.box[0]) * e.height for e in over):
                out.append(self.signal(item, "DOC-OVL-01", 0.62, 0.8, page=pno + 1,
                                       text=" ".join(e.text for e in over), box=list(box)))
        for i, a in enumerate(els):
            for b in els[i + 1:]:
                if a.text.strip() != b.text.strip() and _iou(a.box, b.box) > 0.5:
                    hidden, visible = sorted((a, b), key=lambda e: e.meta["seqno"])
                    hidden_ids.add(id(hidden))
                    out.append(self.signal(item, "DOC-OVL-02", 0.88, 1.0, page=pno + 1,
                                           hidden=hidden.text, visible=visible.text))
        return out, hidden_ids

    def _fonts(self, item, els: list[TextEl]):
        """A value whose font family or size differs from its own row is a classic edit trace."""
        out, flagged = [], set()
        for row in build_rows(els):
            weights: dict[str, int] = {}
            for e in row:
                weights[font_family(e.meta["font"])] = weights.get(font_family(e.meta["font"]), 0) + len(e.text)
            base_family = max(weights, key=weights.get)
            base = [e for e in row if font_family(e.meta["font"]) == base_family]
            base_size = float(np.median([e.meta["size"] for e in base]))
            for e in row:
                if not re.search(r"\d", e.text) or id(e) in flagged:
                    continue
                fam = font_family(e.meta["font"])
                size_ratio = e.meta["size"] / base_size if base_size else 1.0
                if fam != base_family or not 0.88 <= size_ratio <= 1.14:
                    flagged.add(id(e))
                    out.append(self.signal(item, "DOC-FONT-01", 0.8, 1.0, page=e.page + 1,
                                           text=e.text.strip(), font=e.meta["font"], size=e.meta["size"],
                                           base_font=base[0].meta["font"], base_size=base_size,
                                           box=list(e.box)))
        # document-wide: a font family used almost nowhere else, only on numbers
        total = sum(len(e.text) for e in els) or 1
        usage: dict[str, int] = {}
        for e in els:
            usage[font_family(e.meta["font"])] = usage.get(font_family(e.meta["font"]), 0) + len(e.text)
        common = max(usage, key=usage.get) if usage else ""
        for e in els:
            fam = font_family(e.meta["font"])
            if id(e) not in flagged and usage[fam] / total < 0.03 and re.search(r"\d", e.text):
                flagged.add(id(e))
                out.append(self.signal(item, "DOC-FONT-01", 0.7, 0.9, page=e.page + 1,
                                       text=e.text.strip(), font=e.meta["font"], size=e.meta["size"],
                                       base_font=common, base_size=e.meta["size"], box=list(e.box)))
        return out[:6]

    # ------------------------------------------------------------------ images of documents
    def _ocr_geometry(self, item, els: list[TextEl], reliability: float):
        out = []
        for row in build_rows(els):
            if len(row) < 2:
                continue
            for e in row:
                if not re.search(r"\d", e.text):
                    continue
                peers = [p.height for p in row if p is not e]
                ratio = e.height / float(np.median(peers))
                if ratio < 0.72 or ratio > 1.38:
                    out.append(self.signal(item, "DOC-FONT-02", 0.66, reliability * 0.8,
                                           text=e.text, ratio=ratio, box=list(e.box)))
        return out[:4]

    def _anomaly(self, item, fields, reliability):
        if self.iforest is None or item.role is not Role.INVOICE:
            return []
        feats = invoice_features(fields)
        if feats is None:
            return []
        bundle = self.iforest
        x = np.array([[feats[k] for k in bundle["features"]]])
        score = float(bundle["model"].decision_function(x)[0])
        if score >= 0:
            return []
        z = (x[0] - bundle["mean"]) / (bundle["std"] + 1e-9)
        worst = [bundle["features"][i] for i in np.argsort(-np.abs(z))[:2]]
        prob = float(np.clip(0.55 + 2 * abs(score), 0.55, 0.85))
        return [self.signal(item, "DOC-ANOM-01", prob, reliability * 0.8, score=abs(score),
                            features=", ".join(worst), values=feats)]
