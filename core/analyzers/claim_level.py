"""Claim-level analyzers: consistency across documents, identity match, cross-claim reuse.

These look at the claim as one connected case rather than isolated files.
"""
from __future__ import annotations

from datetime import date
from difflib import SequenceMatcher

import imagehash
import numpy as np

from .. import store
from ..config import domain_rules, settings
from ..fields import find_plates
from ..models import face_engine
from ..schemas import Kind, Role
from .base import ClaimAnalyzer, Context, save_artifact
from .documents import ocr_elements


class CrossDocumentAnalyzer(ClaimAnalyzer):
    name = "cross_document"
    version = "1.0"

    def analyze_claim(self, ctx: Context):
        rules = domain_rules()["consistency"]
        th = settings()["thresholds"]
        docs = [i for i in ctx.items if i.fields]
        out = []

        # Vehicle number: typed at intake + every document that names one
        plates = {f"{i.filename}": i.fields["vehicle_numbers"][0] for i in docs if i.fields.get("vehicle_numbers")}
        if ctx.info.vehicle_no:
            plates["claim intake"] = ctx.info.vehicle_no.upper().replace(" ", "").replace("-", "")
        if rules["vehicle_no_must_match"] and len(set(plates.values())) > 1:
            out.append(self.signal(None, "DOC-XCHK-01", 0.85, 0.9,
                                   values="; ".join(f"{k}: {v}" for k, v in plates.items())))

        # Owner name: fuzzy compare (OCR noise, initials)
        names = {i.filename: i.fields["owner_name"] for i in docs if i.fields.get("owner_name")}
        if ctx.info.claimant_name:
            names["claim intake"] = ctx.info.claimant_name.title()
        vals = list(names.values())
        if rules["owner_name_must_match"] and len(vals) > 1:
            worst = min(SequenceMatcher(None, a.lower(), b.lower()).ratio()
                        for k, a in enumerate(vals) for b in vals[k + 1:])
            if worst < th["name_similarity"]:
                out.append(self.signal(None, "DOC-XCHK-02", 0.8, 0.85,
                                       values="; ".join(f"{k}: {v}" for k, v in names.items())))

        # Invoice date vs accident date
        accident = ctx.info.accident_date or next(
            (i.fields.get("accident_date") for i in docs if i.fields.get("accident_date")), None)
        if accident and rules["invoice_after_accident"]:
            acc = date.fromisoformat(accident)
            for i in docs:
                if i.role is Role.INVOICE and i.fields.get("invoice_date"):
                    d = date.fromisoformat(i.fields["invoice_date"])
                    days = (d - acc).days
                    if days < 0 or days > rules["max_days_accident_to_invoice"]:
                        out.append(self.signal(i, "DOC-XCHK-03", 0.85 if days < 0 else 0.65, 0.9,
                                               invoice_date=str(d), accident_date=accident, days=abs(days),
                                               direction="before" if days < 0 else "after"))

        # Number plate visible in a damage photo vs the documents
        doc_plates = set(plates.values())
        if doc_plates:
            for i in ctx.items:
                if i.role is Role.DAMAGE_PHOTO and i.kind is Kind.IMAGE:
                    seen = find_plates(" ".join(e.text for e in ocr_elements(ctx, i)))
                    if seen and not set(seen) & doc_plates:
                        out.append(self.signal(i, "IMG-XCHK-01", 0.8, 0.75, plate=seen[0],
                                               doc_plate=", ".join(sorted(doc_plates))))
        return out


class IdentityAnalyzer(ClaimAnalyzer):
    name = "identity"

    def __init__(self) -> None:
        self.engine = face_engine()
        self.version = self.engine.version

    def best_face(self, ctx: Context, item):
        s = ctx.slot(item)
        if "faces" not in s:
            s["faces"] = self.engine.faces(ctx.pages(item, dpi=200)[0])
        return s["faces"][0] if s["faces"] else None

    def analyze_claim(self, ctx: Context):
        th = settings()["thresholds"]
        ids = [i for i in ctx.items if i.role is Role.LICENCE]
        selfies = [i for i in ctx.items if i.role is Role.SELFIE]
        if not ids or not selfies:
            return []
        id_item, selfie = ids[0], selfies[0]
        f_id, f_self = self.best_face(ctx, id_item), self.best_face(ctx, selfie)
        missing = [it.filename for it, f in ((id_item, f_id), (selfie, f_self)) if f is None]
        if missing:
            return [self.signal(None, "ID-FACE-03", None, where=", ".join(missing))]

        sim = float(np.dot(f_id["embedding"], f_self["embedding"]))
        rel = min(id_item.reliability, selfie.reliability)
        ev = dict(similarity=sim, match_t=th["face_match"], id_item=id_item.id, selfie_item=selfie.id)
        save_artifact(id_item, "face", f_id["crop"])
        save_artifact(selfie, "face", f_self["crop"])
        if sim >= th["face_match"]:
            return [self.signal(selfie, "ID-FACE-00", 0.15, rel, **ev)]
        if sim < th["face_mismatch"]:
            return [self.signal(selfie, "ID-FACE-01", 0.9, rel, **ev)]
        return [self.signal(selfie, "ID-FACE-02", 0.6, rel * 0.6, **ev)]


class ReuseAnalyzer(ClaimAnalyzer):
    """Has this exact file, this photo (perceptually) or this face been seen in another claim?"""
    name = "cross_claim_reuse"
    version = "1.0"

    def analyze_claim(self, ctx: Context):
        th = settings()["thresholds"]
        out, phashes, faces = [], {}, []
        for item in ctx.items:
            hit = store.find_file(item.sha256, ctx.claim_id)
            if hit:
                # Exact same bytes: usually a re-upload of the same claim or a test run, so it is a note,
                # not evidence. Near-duplicates (cropped / recompressed copies) are still scored below.
                out.append(self.signal(item, "XCLM-FILE-REUSE", None, 1.0, other_claim=hit["claim_id"]))
                continue
            if item.role is Role.DAMAGE_PHOTO and item.kind is Kind.IMAGE:
                h = int(str(imagehash.phash(ctx.image(item))), 16)
                phashes[item.id] = h
                near = store.nearest_photo(h, ctx.claim_id)
                if near and near[0] <= th["phash_reuse"]:
                    out.append(self.signal(item, "XCLM-PHOTO-REUSE", 0.9, 1.0, distance=near[0],
                                           other_claim=near[1], other_date=near[2][:10]))
            if item.role in (Role.SELFIE, Role.LICENCE):
                face = ctx.slot(item).get("faces")
                if face:
                    emb = face[0]["embedding"]
                    faces.append((item.id, emb))
                    near = store.nearest_face(emb, ctx.claim_id)
                    other_name = (near[2] or "").strip().lower() if near else ""
                    this_name = ctx.info.claimant_name.strip().lower()
                    if near and near[0] >= th["face_reuse"] and other_name != this_name:
                        out.append(self.signal(item, "XCLM-FACE-REUSE", 0.9, 0.9, similarity=near[0],
                                               other_claim=near[1], other_name=near[2] or "unknown"))
        ctx.cache["_index"] = {"phashes": phashes, "faces": faces}
        return out
