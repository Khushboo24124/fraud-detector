"""Data contracts shared by every module.

The `Signal` is the single seam of the system: analyzers only produce Signals,
the risk engine only consumes Signals, and the UI only renders a ClaimReport.
Swapping a model, adding an analyzer or replacing the UI never crosses that seam.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Kind(str, Enum):
    IMAGE = "image"
    PDF = "pdf"


class Role(str, Enum):
    """What an uploaded file is, from the claim's point of view."""
    DAMAGE_PHOTO = "damage_photo"
    INVOICE = "invoice"
    RC = "rc"
    LICENCE = "licence"
    ID_CARD = "id_card"          # any photo ID: Aadhaar, PAN, voter ID, passport
    SELFIE = "selfie"
    CLAIM_FORM = "claim_form"
    OTHER_DOC = "other_doc"

    @property
    def is_document(self) -> bool:
        return self in {Role.INVOICE, Role.RC, Role.LICENCE, Role.ID_CARD, Role.CLAIM_FORM, Role.OTHER_DOC}

    @property
    def is_identity(self) -> bool:
        """Documents whose face photo can be matched against the selfie."""
        return self in {Role.LICENCE, Role.ID_CARD}


class Tier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ClaimInfo:
    """Facts typed in by the claims handler at intake (all optional)."""
    claimant_name: str = ""
    policy_no: str = ""
    vehicle_no: str = ""
    accident_date: str = ""  # ISO yyyy-mm-dd


@dataclass
class EvidenceItem:
    id: str
    filename: str
    kind: Kind
    role: Role
    sha256: str
    path: str                       # stored copy (content-addressed)
    reliability: float = 1.0        # 0..1, set by the quality gate
    quality: dict[str, Any] = field(default_factory=dict)
    text: str = ""                  # extracted / OCR text, filled by document analyzers
    fields: dict[str, Any] = field(default_factory=dict)  # extracted structured fields
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> blob path (heatmaps, crops)


@dataclass
class Signal:
    """One measured finding. `fraud_prob` > 0.5 pushes toward fraud, < 0.5 toward genuine."""
    analyzer: str
    version: str
    item_id: str | None             # None for claim-level findings
    code: str                       # key into config/reason_codes.yaml
    fraud_prob: float | None        # None = informational only, never scored
    reliability: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reason:
    code: str
    title: str
    text: str
    group: str | None
    item_id: str | None
    filename: str | None
    contribution: float             # signed log-odds added to the overall score
    fraud_prob: float | None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainScore:
    name: str
    risk: float                     # 0..1 fraud likelihood from this domain's evidence
    authenticity: float             # 1 - risk, what the UI calls "authenticity score"
    evaluated: bool                 # False if no evidence of this domain was present
    contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class ClaimReport:
    claim_id: str
    created_at: str
    info: ClaimInfo
    items: list[EvidenceItem]
    signals: list[Signal]
    domains: dict[str, DomainScore]
    overall_risk: float
    tier: Tier
    reasons: list[Reason]           # sorted by |contribution|, strongest first
    summary: str                    # plain-English explanation
    inconclusive_why: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
