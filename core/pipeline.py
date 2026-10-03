"""The single public entry point: analyze_claim(files, info) -> ClaimReport.

UI, API, evaluation script and tests all call this function and nothing deeper.
Every analyzer is isolated: if one crashes, the claim still completes and the
failure itself becomes visible evidence (SYS-FAIL-01) instead of a silent gap.
"""
from __future__ import annotations

import time
import traceback
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Callable

from . import risk, store
from .analyzers.base import Analyzer, ClaimAnalyzer, Context
from .analyzers.claim_level import CrossDocumentAnalyzer, IdentityAnalyzer, ReuseAnalyzer
from .analyzers.documents import DocumentAnalyzer, ocr_elements
from .analyzers.image_ai import AIImageAnalyzer
from .analyzers.image_forensics import ImageForensicsAnalyzer
from .ingest import IntakeError, ingest
from .models import face_engine
from .schemas import ClaimInfo, ClaimReport, EvidenceItem, Kind, Role, Signal

Progress = Callable[[float, str], None]


@lru_cache(maxsize=1)
def item_analyzers() -> tuple[Analyzer, ...]:
    # Order matters: the AI analyzer leaves a tile map that forensics folds into the heatmap.
    return (AIImageAnalyzer(), ImageForensicsAnalyzer(), DocumentAnalyzer())


@lru_cache(maxsize=1)
def claim_analyzers() -> tuple[ClaimAnalyzer, ...]:
    # Reuse runs last: it indexes faces found by the identity analyzer.
    return (CrossDocumentAnalyzer(), IdentityAnalyzer(), ReuseAnalyzer())


def _refine_role(item: EvidenceItem, ctx: Context) -> None:
    """Images default to 'damage photo'; content decides if it is really a document or a selfie."""
    els = ocr_elements(ctx, item)
    chars = sum(len(e.text) for e in els)
    if chars >= 80 and len(els) >= 5:
        item.role = Role.OTHER_DOC                  # DocumentAnalyzer narrows it to invoice/rc/licence
        return
    faces = face_engine().faces(ctx.image(item))
    ctx.slot(item)["faces"] = faces
    if faces:
        x, y, w, h = faces[0]["bbox"]
        if w * h > 0.06 * item.quality["width"] * item.quality["height"]:
            item.role = Role.SELFIE


def _run(analyzer: Analyzer, fn, timings: dict, *args) -> list[Signal]:
    t = time.perf_counter()
    try:
        return fn(*args)
    except Exception as exc:  # isolate: one broken analyzer must not sink the claim
        traceback.print_exc()
        item = args[0] if args and isinstance(args[0], EvidenceItem) else None
        return [analyzer.signal(item, "SYS-FAIL-01", None, analyzer=analyzer.name,
                                error=f"{type(exc).__name__}: {exc}")]
    finally:
        timings[analyzer.name] = timings.get(analyzer.name, 0) + int((time.perf_counter() - t) * 1000)


def analyze_claim(files: list[tuple[str, bytes, Role | None]], info: ClaimInfo | None = None,
                  progress: Progress | None = None, persist: bool = True) -> ClaimReport:
    progress = progress or (lambda f, m: None)
    info = info or ClaimInfo()
    claim_id = "CLM-" + uuid.uuid4().hex[:8].upper()
    t0 = time.perf_counter()
    timings: dict[str, int] = {}

    items, signals, explicit = [], [], set()
    for name, data, role in files:
        try:
            item = ingest(name, data, role)
        except IntakeError as exc:
            signals.append(Signal("intake", "1", None, "SYS-FAIL-01", None,
                                  evidence={"analyzer": f"intake of {name}", "error": str(exc)}))
            continue
        items.append(item)
        if role is not None:
            explicit.add(item.id)
    ctx = Context(claim_id=claim_id, info=info, items=items)

    progress(0.05, "Classifying evidence")
    for item in items:
        if item.kind is Kind.IMAGE and item.id not in explicit and item.role is Role.DAMAGE_PHOTO:
            _refine_role(item, ctx)

    analyzers = item_analyzers()
    steps = max(1, len(items) * len(analyzers))
    done = 0
    for item in items:
        for a in analyzers:
            done += 1
            if a.applies_to(item):
                progress(0.1 + 0.7 * done / steps, f"{a.name} → {item.filename}")
                signals += _run(a, a.analyze, timings, item, ctx)

    for a in claim_analyzers():
        progress(0.85, f"Claim-level check: {a.name}")
        signals += _run(a, a.analyze_claim, timings, ctx)

    progress(0.95, "Scoring and explaining")
    domains, overall, tier, reasons, why, overrides = risk.assess(signals, items)
    report = ClaimReport(
        claim_id=claim_id, created_at=datetime.now().isoformat(timespec="seconds"), info=info,
        items=items, signals=signals, domains=domains, overall_risk=overall, tier=tier,
        reasons=reasons, summary=risk.summarise(tier, overall, reasons, why, overrides),
        inconclusive_why=why, overrides=overrides,
        model_versions={a.name: a.version for a in (*analyzers, *claim_analyzers())},
    )
    timings["total"] = int((time.perf_counter() - t0) * 1000)
    report.timings_ms = timings
    if persist:
        index = ctx.cache.get("_index", {})
        store.save_report(report, index.get("phashes", {}), index.get("faces", []))
    progress(1.0, "Done")
    return report
