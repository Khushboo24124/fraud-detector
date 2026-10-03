"""Self-contained HTML case report (images embedded) that an investigator can file or print to PDF."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image

from .schemas import ClaimReport, Kind, Tier

TIER_COLOUR = {Tier.LOW: "#15803d", Tier.MEDIUM: "#b45309", Tier.HIGH: "#b91c1c", Tier.INCONCLUSIVE: "#475569"}


def _img_tag(path: str, max_side: int = 520) -> str:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return ""
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def build_html(r: ClaimReport) -> str:
    e = html.escape
    colour = TIER_COLOUR[r.tier]
    doms = "".join(
        f"<div class='kpi'><div class='lbl'>{e(d.name.title())} authenticity</div>"
        f"<div class='val'>{'—' if not d.evaluated else f'{d.authenticity:.0%}'}</div></div>"
        for d in r.domains.values())
    reasons = "".join(
        f"<tr><td><code>{e(x.code)}</code></td><td><b>{e(x.title)}</b><br>{e(x.text)}</td>"
        f"<td>{e(x.filename or 'claim')}</td><td class='num'>{x.contribution:+.2f}</td></tr>"
        for x in r.reasons)
    items = ""
    for it in r.items:
        pics = _img_tag(it.path) if it.kind is Kind.IMAGE else ""
        if "heatmap" in it.artifacts:
            pics += _img_tag(it.artifacts["heatmap"])
        flds = {k: v for k, v in it.fields.items() if k not in ("items",) and v not in (None, [], "")}
        items += (f"<div class='item'><h3>{e(it.filename)} <span class='role'>{e(it.role.value)}</span></h3>"
                  f"<div class='pics'>{pics}</div>"
                  f"<p class='muted'>SHA-256 {it.sha256[:16]}… · reliability {it.reliability:.0%}"
                  f"{' · ' + e(', '.join(it.quality.get('issues', []))) if it.quality.get('issues') else ''}</p>"
                  + (f"<pre>{e(str(flds))}</pre>" if flds else "") + "</div>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ClaimGuard report {e(r.claim_id)}</title>
<style>
body{{font-family:Segoe UI,system-ui,sans-serif;margin:32px;color:#0f172a;max-width:1000px}}
.banner{{border-left:8px solid {colour};background:#f8fafc;padding:16px 20px;border-radius:6px}}
.tier{{color:{colour};font-size:28px;font-weight:700}} .kpis{{display:flex;gap:16px;margin:18px 0}}
.kpi{{flex:1;background:#f1f5f9;border-radius:8px;padding:12px}} .lbl{{font-size:12px;color:#475569}}
.val{{font-size:22px;font-weight:600}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #e2e8f0;padding:8px;text-align:left;vertical-align:top;font-size:14px}}
.num{{text-align:right;font-variant-numeric:tabular-nums}} .item{{margin:20px 0}} .pics img{{max-width:48%;margin-right:8px;border-radius:6px}}
.role{{font-size:12px;background:#e2e8f0;border-radius:4px;padding:2px 6px}} .muted{{color:#64748b;font-size:12px}}
pre{{background:#f8fafc;padding:8px;white-space:pre-wrap;font-size:12px}}
</style></head><body>
<h1>Claim {e(r.claim_id)}</h1><p class="muted">Generated {e(r.created_at)} · claimant {e(r.info.claimant_name or '—')}
 · vehicle {e(r.info.vehicle_no or '—')} · accident {e(r.info.accident_date or '—')}</p>
<div class="banner"><div class="tier">{r.tier.value} · {r.overall_risk:.0%} fraud likelihood</div><p>{e(r.summary)}</p></div>
<div class="kpis">{doms}</div>
<h2>Reasons</h2><table><tr><th>Code</th><th>Finding</th><th>Evidence</th><th class="num">Log-odds</th></tr>{reasons}</table>
<h2>Evidence</h2>{items}
<p class="muted">Screening signal, not proof: a human investigator makes the final decision.
Models: {e(', '.join(f'{k}={v}' for k, v in r.model_versions.items()))}</p>
</body></html>"""


def save_html(r: ClaimReport, path: Path) -> Path:
    path.write_text(build_html(r), encoding="utf-8")
    return path
