"""Solutions pages, with honest status labels."""
from __future__ import annotations

from .components import bullets, cards, md, page_footer_buttons, page_hero, section


def _status(label: str, cls: str) -> None:
    md(f'<span class="cg-badge {cls}" style="font-size:.85rem;margin:0 0 10px 0">{label}</span>')


def motor() -> None:
    page_hero("Motor claims", "Our main use case, working today.")
    _status("Live", "cg-live")
    section("What it catches")
    cards([("🤖", "AI-made damage photos", "Photos created or changed by AI tools."),
           ("♻️", "Reused photos", "The same accident photo used again, even if cropped or recompressed."),
           ("🧾", "Edited garage bills", "Raised totals, cover-up boxes, font changes, sums that don't add up."),
           ("🪪", "Mismatched details", "Vehicle number or owner name different on RC, bill and claim; bill dated "
            "before the accident."),
           ("🤳", "Fake identity", "Selfie that doesn't match the photo ID, or an AI-made selfie.")], cols=3)
    section("Files it accepts")
    bullets(["Damage photos (JPG, PNG, WEBP, BMP)", "Repair invoice / estimate (PDF or photo)",
             "Registration certificate (RC)", "Driving licence or any photo ID (Aadhaar, PAN, voter ID) and claimant selfie", "Claim form"])
    page_footer_buttons()


def bills() -> None:
    page_hero("Bills & receipts", "Edited bills are checked today; hand-edited photos of receipts are still hard.")
    _status("Beta", "cg-beta")
    section("Works well")
    bullets(["Digital PDF bills: cover-up boxes, hidden text, font changes, edit history, wrong totals.",
             "Indian GST invoices: arithmetic and anomaly checks."])
    section("Still weak")
    bullets(["Photos of paper receipts where a few digits were changed in Paint or GIMP.",
             "Foreign receipts: arithmetic and anomaly checks are skipped (a note explains why)."])
    page_footer_buttons()


def health() -> None:
    page_hero("Health claims", "Next on our roadmap.")
    _status("Coming next", "cg-soon")
    section("What it would check")
    bullets(["Edited hospital bills and discharge summaries (same document checks).",
             "AI-made or reused injury photos (same photo checks).",
             "Patient name and dates matching across documents."])
    section("What is needed")
    bullets(["A new rules file for health documents (fields, document types, totals).",
             "Real hospital bill samples to test on. The engine itself stays the same."])
    page_footer_buttons()


def property_() -> None:
    page_hero("Property claims", "Next on our roadmap.")
    _status("Coming next", "cg-soon")
    section("What it would check")
    bullets(["Exaggerated or AI-made damage photos (fire, water, storm).",
             "Photos reused from old claims or the internet.",
             "Repair estimates and invoices (same document checks)."])
    section("What is needed")
    bullets(["A rules file for property documents.",
             "Internet reverse-image search would need an online service, so it would be an optional add-on."])
    page_footer_buttons()


PAGES = {"sol-motor": motor, "sol-bills": bills, "sol-health": health, "sol-property": property_}
