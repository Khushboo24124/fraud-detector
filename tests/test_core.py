"""Fast unit tests (no ML models needed):  python -m pytest -q"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import risk  # noqa: E402
from core.fields import TextEl, build_rows, extract_fields, find_plates, math_checks, parse_amounts  # noqa: E402
from core.ingest import IntakeError, estimate_jpeg_quality, sniff  # noqa: E402
from core.schemas import EvidenceItem, Kind, Role, Signal, Tier  # noqa: E402


def sig(code, p, rel=1.0, item="a", **ev):
    return Signal("test", "1", item, code, p, rel, ev)


def items():
    return [EvidenceItem("a", "photo.jpg", Kind.IMAGE, Role.DAMAGE_PHOTO, "x" * 64, "p"),
            EvidenceItem("b", "photo2.jpg", Kind.IMAGE, Role.DAMAGE_PHOTO, "y" * 64, "p")]


# ---------------------------------------------------------------- risk engine invariants
def test_every_flag_has_a_reason():
    _, overall, tier, reasons, _, _ = risk.assess([sig("IMG-AI-01", 0.95, prob=0.95, models_agree=2, models_total=2)], items())
    assert tier in (Tier.MEDIUM, Tier.HIGH)
    assert any(r.contribution > 0 and r.text for r in reasons)


def test_genuine_photo_cannot_cancel_fake_photo():
    fake_only = risk.assess([sig("IMG-AI-01", 0.95, item="a")], items())[1]
    fake_plus_real = risk.assess([sig("IMG-AI-01", 0.95, item="a"), sig("IMG-AI-00", 0.05, item="b")], items())[1]
    assert fake_plus_real == pytest.approx(fake_only)


def test_correlated_signals_not_double_counted():
    one = risk.assess([sig("DOC-FONT-01", 0.8)], items())[1]
    three = risk.assess([sig("DOC-FONT-01", 0.8)] * 3, items())[1]
    assert three > one
    assert three < risk.assess([sig("DOC-FONT-01", 0.8), sig("DOC-MATH-01", 0.8), sig("DOC-OVL-02", 0.8)], items())[1]


def test_low_reliability_moves_score_less():
    sharp = risk.assess([sig("IMG-AI-01", 0.8, rel=1.0)], items())[1]
    blurry = risk.assess([sig("IMG-AI-01", 0.8, rel=0.3)], items())[1]
    assert blurry < sharp


def test_unscored_signals_do_not_move_the_score():
    base = risk.assess([sig("DOC-OK-00", 0.3)], items())[1]
    with_info = risk.assess([sig("DOC-OK-00", 0.3), sig("IMG-ELA-01", None, regions=3, zmax=9.0)], items())[1]
    assert with_info == base


def test_override_sets_floor():
    _, overall, tier, _, _, overrides = risk.assess([sig("XCLM-PHOTO-REUSE", 0.6)], items())
    assert tier is Tier.HIGH and overrides


def test_inconclusive_when_detectors_disagree():
    s = sig("IMG-AI-02", 0.5, rel=0.4, inconclusive=True)
    assert risk.assess([s], items())[2] is Tier.INCONCLUSIVE


def test_inconclusive_when_no_evidence():
    assert risk.assess([], items())[2] is Tier.INCONCLUSIVE


def test_deterministic():
    s = [sig("IMG-AI-01", 0.7), sig("DOC-MATH-01", 0.85, item=None)]
    assert risk.assess(s, items())[1] == risk.assess(s, items())[1]


# ---------------------------------------------------------------- field extraction
def test_plates_and_amounts():
    assert find_plates("Vehicle No: MH 12 AB 1234 / ka-05-mn-4321") == ["MH12AB1234", "KA05MN4321"]
    assert parse_amounts("Rate 1,250.00 Amount 12,500.00 Qty 10") == [1250.0, 12500.0]


def _rows(lines):
    return build_rows([TextEl(t, (0, 20 * k, 400, 20 * k + 10)) for k, t in enumerate(lines)])


def test_invoice_math_detects_inflated_total():
    f = extract_fields(_rows(["CustomerName: Rahul Sharma", "1 Front Bumper 1 4,000.00 4,000.00",
                              "2 Labour Charges 1 1,000.00 1,000.00", "Sub Total 5,000.00",
                              "CGST @ 9% 450.00", "SGST @ 9% 450.00", "Grand Total 9,900.00"]))
    assert f["owner_name"] == "Rahul Sharma"
    assert f["subtotal"] == 5000 and f["total"] == 9900
    issues = math_checks(f, 1.0)
    assert len(issues) == 1 and issues[0]["expected"] == 5900


def test_invoice_math_passes_when_consistent():
    f = extract_fields(_rows(["1 Front Bumper 1 4,000.00 4,000.00", "Sub Total 4,000.00",
                              "CGST @ 9% 360.00", "SGST @ 9% 360.00", "Grand Total 4,720.00"]))
    assert math_checks(f, 1.0) == []


# ---------------------------------------------------------------- intake
def test_sniff_rejects_disguised_files():
    with pytest.raises(IntakeError):
        sniff(b"MZ\x90\x00 this is an exe renamed to .jpg")
    assert sniff(b"%PDF-1.7 ...")[0] is Kind.PDF


def test_jpeg_quality_estimate():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (90, 120, 200)).save(buf, "JPEG", quality=60)
    q = estimate_jpeg_quality(Image.open(io.BytesIO(buf.getvalue())))
    assert 55 <= q <= 65


# ---------------------------------------------------------------- selfie / deepfake coverage
def test_ai_detection_covers_selfies_and_damage_photos():
    from core.analyzers.image_ai import AIImageAnalyzer
    from core.analyzers.image_forensics import ImageForensicsAnalyzer
    ai = object.__new__(AIImageAnalyzer)          # applies_to needs no models
    forensics = object.__new__(ImageForensicsAnalyzer)
    for role in (Role.DAMAGE_PHOTO, Role.SELFIE):
        item = EvidenceItem("a", "x.jpg", Kind.IMAGE, role, "x" * 64, "p")
        assert ai.applies_to(item) and forensics.applies_to(item)
    pdf = EvidenceItem("b", "x.pdf", Kind.PDF, Role.INVOICE, "y" * 64, "p")
    assert not ai.applies_to(pdf)


def test_single_ai_threshold_is_consistent():
    from core.config import settings
    th = settings()["thresholds"]
    assert 0 < th["ai_borderline"] < th["ai_flag"] < 1
    assert set(settings()["ai_profiles"]) == {"scene", "face"}


def test_flagged_selfie_alone_is_high_risk():
    sel = [EvidenceItem("a", "selfie.jpg", Kind.IMAGE, Role.SELFIE, "x" * 64, "p")]
    _, overall, tier, _, _, _ = risk.assess([sig("IMG-AI-01", 0.97, prob=0.97, models_agree=1, models_total=1)], sel)
    assert tier is Tier.HIGH


# ---------------------------------------------------------------- UI wiring (no dead menu links)
def test_every_menu_link_has_a_page():
    from ui import nav, pages_about, pages_resources, pages_solutions, pages_tech
    pages = {**pages_tech.PAGES, **pages_solutions.PAGES, **pages_resources.PAGES, **pages_about.PAGES}
    linked = {pid for items in nav.MENUS.values() for pid, _, _ in items}
    assert linked == set(pages)


def test_ui_report_roundtrip_keeps_scores():
    from ui.report_io import from_dict
    from core.schemas import ClaimInfo, ClaimReport, DomainScore
    sigs = [sig("IMG-AI-01", 0.95, prob=0.95, models_agree=2, models_total=2)]
    domains, overall, tier, reasons, why, overrides = risk.assess(sigs, items())
    rep = ClaimReport("CLM-T", "2026-01-01T00:00:00", ClaimInfo(), items(), sigs, domains, overall, tier, reasons,
                      "s", why, overrides)
    back = from_dict(rep.to_dict())
    assert back.tier is rep.tier and back.overall_risk == rep.overall_risk
    assert [r.code for r in back.reasons] == [r.code for r in rep.reasons]


def test_any_photo_id_counts_as_identity_document():
    from core.ingest import guess_role
    for name in ("aadhaar_front.jpg", "PAN-card.png", "voter_id.jpg", "passport.jpg"):
        assert guess_role(name, Kind.IMAGE) is Role.ID_CARD
    assert Role.ID_CARD.is_identity and Role.LICENCE.is_identity and Role.ID_CARD.is_document
    assert not Role.SELFIE.is_identity
