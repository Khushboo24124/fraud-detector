"""Explainable risk engine: calibrated signals -> additive log-odds -> tiers and reasons.

Why log-odds: contributions add up exactly, so the UI can show a waterfall of how
every point of the score was earned, and no signal is hidden inside an opaque blend.

Pooling rules (to stop double counting and to keep fraud from being "averaged away"):
  * signals in the same group are correlated -> the group keeps its strongest
    fraud-side contribution (+ a small bonus for independent repeat hits);
  * genuine-side evidence only counts in a group with no fraud-side evidence,
    so three real photos cannot cancel one fake photo.
"""
from __future__ import annotations

import math
import string
from collections import defaultdict

import numpy as np

from .config import reason_catalog, settings
from .schemas import DomainScore, EvidenceItem, Reason, Signal, Tier

TIER_ORDER = [Tier.LOW, Tier.MEDIUM, Tier.HIGH]


def _logit(p: float) -> float:
    lo, hi = settings()["risk"]["prob_clip"]
    p = min(max(p, lo), hi)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class _SafeDict(dict):
    def __missing__(self, key):
        return "?"


def render_text(code: str, evidence: dict) -> str:
    template = reason_catalog().get(code, {}).get("template", code)
    try:
        return string.Formatter().vformat(template, (), _SafeDict(evidence))
    except (ValueError, TypeError):
        # a numeric format spec met a missing value; fall back to raw fields
        return template.split("{")[0] + " " + ", ".join(f"{k}={v}" for k, v in evidence.items()
                                                         if not isinstance(v, (dict, list)))


def contribution(sig: Signal) -> float:
    group = reason_catalog().get(sig.code, {}).get("group")
    if sig.fraud_prob is None or group is None:
        return 0.0
    weight = settings()["risk"]["group_weights"].get(group, 0.0)
    return weight * sig.reliability * _logit(sig.fraud_prob)


def pool_groups(signals: list[Signal]) -> dict[str, float]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for s in signals:
        g = reason_catalog().get(s.code, {}).get("group")
        c = contribution(s)
        if g and c != 0.0:
            by_group[g].append(c)
    pooled = {}
    for g, cs in by_group.items():
        pos = sorted((c for c in cs if c > 0), reverse=True)
        if pos:
            pooled[g] = pos[0] + min(0.6, 0.2 * (len(pos) - 1))
        else:
            pooled[g] = float(np.mean(cs))
    return pooled


def tier_for(p: float) -> Tier:
    t = settings()["risk"]["tiers"]
    return Tier.HIGH if p >= t["high"] else Tier.MEDIUM if p >= t["medium"] else Tier.LOW


def assess(signals: list[Signal], items: list[EvidenceItem]):
    cfg = settings()["risk"]
    prior = cfg["prior_logit"]
    pooled = pool_groups(signals)

    domains = {}
    for name, groups in cfg["domains"].items():
        contribs = {g: pooled[g] for g in groups if g in pooled}
        logit = prior + sum(contribs.values())
        risk = _sigmoid(logit)
        domains[name] = DomainScore(name=name, risk=round(risk, 4), authenticity=round(1 - risk, 4),
                                    evaluated=bool(contribs), contributions=contribs)

    overall = _sigmoid(prior + sum(pooled.values()))
    tier = tier_for(overall)

    # Hard overrides raise the tier floor; they never lower it.
    overrides = []
    for s in signals:
        floor = cfg["overrides"].get(s.code)
        if floor and TIER_ORDER.index(Tier(floor)) > TIER_ORDER.index(tier):
            tier = Tier(floor)
            overrides.append(f"{s.code} sets the minimum tier to {floor}")
    if overrides:
        floor_p = cfg["tiers"]["high" if tier is Tier.HIGH else "medium"]
        overall = max(overall, floor_p)

    # INCONCLUSIVE: refuse to guess when the evidence is weak or contradictory.
    why = []
    scored = [s for s in signals if s.fraud_prob is not None]
    if not scored:
        why.append("No analyzable evidence was found in the submitted files.")
    if any(s.evidence.get("inconclusive") for s in signals):
        why.append("AI-image evidence is borderline on at least one photo.")
    if scored and max(s.reliability for s in scored) < cfg["inconclusive"]["min_reliability"]:
        why.append("All evidence is low quality (low resolution, blur or heavy compression).")
    if tier is not Tier.HIGH and why:
        tier = Tier.INCONCLUSIVE

    names = {i.id: i.filename for i in items}
    reasons = []
    for s in signals:
        meta = reason_catalog().get(s.code, {})
        reasons.append(Reason(code=s.code, title=meta.get("title", s.code), text=render_text(s.code, s.evidence),
                              group=meta.get("group"), item_id=s.item_id, filename=names.get(s.item_id),
                              contribution=round(contribution(s), 3), fraud_prob=s.fraud_prob,
                              evidence=s.evidence))
    reasons.sort(key=lambda r: (-(r.contribution > 0), -abs(r.contribution)))
    return domains, round(overall, 4), tier, reasons, why, overrides


def summarise(tier: Tier, overall: float, reasons: list[Reason], why: list[str], overrides: list[str]) -> str:
    """Plain-English explanation, generated only from reason codes (never free-form)."""
    drivers = [r for r in reasons if r.contribution > 0.05][:3]
    genuine = [r for r in reasons if r.contribution < -0.05][:2]
    if tier is Tier.INCONCLUSIVE:
        head = f"The claim could not be scored with confidence (estimated fraud likelihood {overall:.0%}). "
        head += " ".join(why) + " Request better evidence or route to manual review."
    else:
        action = {Tier.HIGH: "Hold payment and send to a fraud investigator.",
                  Tier.MEDIUM: "Review the flagged items before approving.",
                  Tier.LOW: "No significant tampering indicators; proceed with standard processing."}[tier]
        head = f"Fraud likelihood is {tier.value} ({overall:.0%}). {action}"
    parts = [head]
    if drivers:
        lead = "Minor indicators (outweighed by other evidence): " if tier is Tier.LOW else "Main reasons: "
        parts.append(lead + "; ".join(
            f"{r.title.lower()}{f' in {r.filename}' if r.filename else ''}" for r in drivers) + ".")
    if genuine:
        parts.append("Evidence supporting authenticity: " + "; ".join(
            f"{r.title.lower()}{f' ({r.filename})' if r.filename else ''}" for r in genuine) + ".")
    if overrides:
        parts.append("Rule applied: " + "; ".join(overrides) + ".")
    return " ".join(parts)
