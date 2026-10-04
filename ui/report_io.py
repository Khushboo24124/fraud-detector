"""Rebuild a ClaimReport from the JSON saved in SQLite, so a past analysis reopens exactly as it was scored.

Uses only the backend's own dataclasses (core.schemas); the scores are never recomputed here.
"""
from __future__ import annotations

from core import store
from core.schemas import (ClaimInfo, ClaimReport, DomainScore, EvidenceItem, Kind, Reason, Role, Signal,
                          Tier)


def from_dict(d: dict) -> ClaimReport:
    items = [EvidenceItem(**{**i, "kind": Kind(i["kind"]), "role": Role(i["role"])}) for i in d["items"]]
    return ClaimReport(
        claim_id=d["claim_id"], created_at=d["created_at"], info=ClaimInfo(**d["info"]), items=items,
        signals=[Signal(**s) for s in d["signals"]],
        domains={k: DomainScore(**v) for k, v in d["domains"].items()},
        overall_risk=d["overall_risk"], tier=Tier(d["tier"]),
        reasons=[Reason(**r) for r in d["reasons"]], summary=d["summary"],
        inconclusive_why=d.get("inconclusive_why", []), overrides=d.get("overrides", []),
        timings_ms=d.get("timings_ms", {}), model_versions=d.get("model_versions", {}))


def load(claim_id: str) -> ClaimReport | None:
    d = store.load_report_json(claim_id)
    return from_dict(d) if d else None
