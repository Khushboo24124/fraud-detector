"""All colours and CSS for the ClaimGuard UI live here (one place, fully offline: no web fonts or CDNs)."""
from __future__ import annotations

import streamlit as st

from core.schemas import Tier

GRADIENT = "linear-gradient(120deg, #4B1FA8 0%, #2F5BEA 45%, #22B5F0 70%, #FF7A45 100%)"
NAVY, BODY, BG = "#0F1B3D", "#475569", "#F7F8FB"

# Tier -> (main colour, light background, icon). Same tiers the risk engine returns.
TIER_STYLE = {
    Tier.HIGH: ("#DC2626", "#FEF2F2", "🚨"),
    Tier.MEDIUM: ("#D97706", "#FFFBEB", "⚠️"),
    Tier.LOW: ("#16A34A", "#F0FDF4", "✅"),
    Tier.INCONCLUSIVE: ("#64748B", "#F1F5F9", "❔"),
}

CSS = f"""
<style>
:root {{ --navy:{NAVY}; --body:{BODY}; --bg:{BG}; --grad:{GRADIENT}; }}
html, body, [class*="css"], .stApp {{
  font-family: "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
}}
.stApp {{ background: var(--bg); }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 5.2rem !important; max-width: 1240px; }}
h1, h2, h3, h4 {{ color: var(--navy); letter-spacing: -0.02em; }}
p, li {{ color: var(--body); }}

/* ---------- top bar ---------- */
.cg-topbar {{ position: fixed; top: 0; left: 0; right: 0; z-index: 1000002; height: 62px;
  display:flex; align-items:center; gap: 28px; padding: 0 28px; background: rgba(255,255,255,.97);
  border-bottom: 1px solid #E5E7EB; box-shadow: 0 1px 8px rgba(15,27,61,.06); }}
.cg-topbar::after {{ content:""; position:absolute; left:0; right:0; bottom:-1px; height:3px; background: var(--grad); }}
.cg-logo {{ font-weight: 800; font-size: 1.25rem; color: var(--navy) !important; text-decoration: none !important; white-space: nowrap; }}
.cg-menus {{ display:flex; gap: 6px; flex: 1; }}
.cg-menu {{ position: relative; }}
.cg-menu > span {{ display:inline-block; padding: 20px 12px; font-weight: 600; color: var(--navy); cursor: default; white-space: nowrap; }}
.cg-menu > span::after {{ content:" ▾"; font-size:.75em; opacity:.6; }}
.cg-drop {{ display:none; position:absolute; top: 58px; left: 0; min-width: 270px; background:#fff; border-radius: 12px;
  box-shadow: 0 12px 32px rgba(15,27,61,.16); padding: 8px; border: 1px solid #EEF0F4; }}
.cg-menu:hover .cg-drop {{ display:block; }}
.cg-drop a {{ display:block; padding: 9px 12px; border-radius: 8px; color: var(--navy) !important; text-decoration:none !important; font-size:.94rem; }}
.cg-drop a:hover {{ background: #F1F4FF; }}
.cg-drop a small {{ display:block; color:#64748B; font-size:.78rem; }}
.cg-cta {{ background: var(--grad); color:#fff !important; padding: 9px 18px; border-radius: 999px; font-weight:700;
  text-decoration:none !important; white-space: nowrap; box-shadow: 0 4px 14px rgba(47,91,234,.35); }}
@media (max-width: 900px) {{ .cg-menu > span {{ padding: 20px 6px; font-size: .9rem; }} .cg-topbar {{ gap: 10px; padding: 0 12px; }} }}
@media (max-width: 700px) {{ .cg-menus {{ display:none; }} }}

/* ---------- generic pieces ---------- */
.cg-card {{ background:#fff; border-radius: 12px; padding: 20px 22px; box-shadow: 0 2px 10px rgba(15,27,61,.06);
  border: 1px solid #EEF0F4; height: 100%; }}
.cg-card h4 {{ margin: 6px 0 6px 0; font-size: 1.05rem; }}
.cg-card p {{ margin: 0; font-size: .93rem; line-height: 1.5; }}
.cg-icon {{ width: 44px; height: 44px; border-radius: 10px; background: #F1F4FF; display:flex; align-items:center;
  justify-content:center; font-size: 1.35rem; }}
.cg-badge {{ display:inline-block; font-size:.72rem; font-weight:700; padding: 2px 10px; border-radius: 999px; margin-left:6px; vertical-align: middle; }}
.cg-live {{ background:#DCFCE7; color:#166534; }} .cg-beta {{ background:#FEF3C7; color:#92400E; }} .cg-soon {{ background:#E2E8F0; color:#334155; }}
.cg-section-title {{ font-size: 2.1rem; font-weight: 800; color: var(--navy); margin: 2.6rem 0 .4rem 0; letter-spacing:-.03em; line-height:1.1; }}
.cg-section-sub {{ color: var(--body); font-size: 1.05rem; margin-bottom: 1.2rem; max-width: 820px; }}
.cg-btn {{ display:inline-block; padding: 12px 22px; border-radius: 999px; font-weight: 700; text-decoration:none !important; margin: 6px 8px 0 0; }}
.cg-btn-white {{ background:#fff; color: var(--navy) !important; }}
.cg-btn-ghost {{ border: 2px solid rgba(255,255,255,.85); color:#fff !important; }}
.cg-btn-grad {{ background: var(--grad); color:#fff !important; box-shadow: 0 4px 14px rgba(47,91,234,.3); }}

/* ---------- hero ---------- */
.cg-hero {{ background: var(--grad); border-radius: 22px; padding: 56px 48px; display:flex; gap: 36px; align-items:center;
  position: relative; overflow: hidden; }}
.cg-hero::before {{ content:""; position:absolute; inset:0; background: radial-gradient(circle at 80% 20%, rgba(255,255,255,.18), transparent 45%); }}
.cg-hero-text {{ flex: 1.1; position: relative; }}
.cg-hero-text h1 {{ color:#fff; font-size: 3.1rem; line-height: 1.04; font-weight: 900; margin: 0 0 14px 0; letter-spacing:-.035em; }}
.cg-hero-text p {{ color: rgba(255,255,255,.92); font-size: 1.15rem; max-width: 540px; }}
.cg-hero-img {{ flex: 1; position: relative; }}
.cg-hero-img img {{ width:100%; border-radius: 14px; box-shadow: 0 18px 40px rgba(0,0,0,.28); }}
@media (max-width: 900px) {{ .cg-hero {{ flex-direction: column; padding: 36px 24px; }} .cg-hero-text h1 {{ font-size: 2.2rem; }} }}

/* ---------- stats ---------- */
.cg-stat {{ background:#fff; border-radius: 14px; padding: 20px; border:1px solid #EEF0F4; box-shadow: 0 2px 10px rgba(15,27,61,.05); height:100%; }}
.cg-stat b {{ display:block; font-size: 2.1rem; font-weight: 900; background: var(--grad); -webkit-background-clip:text; background-clip:text; color: transparent; }}
.cg-stat span {{ color: var(--body); font-size: .92rem; }}

/* ---------- inspection photo ---------- */
.cg-inspect {{ display:grid; grid-template-columns: 1fr 1.5fr 1fr; gap: 18px; align-items:center; }}
.cg-inspect-col {{ display:flex; flex-direction:column; gap: 14px; }}
.cg-inspect-img {{ position:relative; }}
.cg-inspect-img img {{ width:100%; border-radius: 16px; display:block; box-shadow: 0 12px 30px rgba(15,27,61,.2); }}
.cg-dot {{ position:absolute; width: 30px; height: 30px; margin:-15px 0 0 -15px; border-radius:50%; background:#DC2626; color:#fff;
  font-weight:800; font-size:.85rem; display:flex; align-items:center; justify-content:center; border: 3px solid #fff;
  box-shadow: 0 0 0 4px rgba(220,38,38,.25); cursor: help; }}
.cg-dot .cg-tip {{ display:none; position:absolute; bottom: 38px; left: 50%; transform: translateX(-50%); width: 220px; background: var(--navy);
  color:#fff; font-weight: 500; font-size:.8rem; padding: 8px 10px; border-radius: 8px; z-index: 5; }}
.cg-dot:hover .cg-tip {{ display:block; }}
.cg-callout {{ background:#fff; border-radius: 12px; padding: 12px 14px; border: 1px solid #F3D3D3; box-shadow: 0 2px 10px rgba(15,27,61,.06); }}
.cg-callout b {{ color: var(--navy); font-size: .95rem; }} .cg-callout p {{ margin: 4px 0 0 0; font-size: .86rem; }}
.cg-num {{ display:inline-flex; width: 22px; height: 22px; border-radius: 50%; background:#DC2626; color:#fff; font-size:.75rem; font-weight:800;
  align-items:center; justify-content:center; margin-right: 6px; }}
@media (max-width: 900px) {{ .cg-inspect {{ grid-template-columns: 1fr; }} }}

/* ---------- steps / stepper ---------- */
.cg-step {{ text-align:center; }}
.cg-step .n {{ width: 46px; height: 46px; margin: 0 auto 8px auto; border-radius: 50%; background: var(--grad); color:#fff; font-weight: 800;
  display:flex; align-items:center; justify-content:center; font-size: 1.1rem; }}
.cg-stepper {{ display:flex; gap: 6px; margin: 0 0 18px 0; flex-wrap: wrap; }}
.cg-stepper div {{ flex:1; min-width: 120px; padding: 10px 12px; border-radius: 10px; background:#fff; border:1px solid #E5E7EB; font-size:.88rem;
  color:#94A3B8; font-weight:600; }}
.cg-stepper div.done {{ color:#166534; border-color:#BBF7D0; background:#F0FDF4; }}
.cg-stepper div.now {{ color:#fff; background: var(--grad); border-color: transparent; }}

/* ---------- analyze page ---------- */
.cg-verdict {{ border-radius: 16px; padding: 22px 26px; border: 2px solid; display:flex; gap: 18px; align-items:flex-start; }}
.cg-verdict .ic {{ font-size: 2.2rem; line-height: 1; }}
.cg-verdict h2 {{ margin: 0 0 4px 0; font-size: 1.7rem; }}
.cg-verdict .next {{ margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(15,27,61,.1); font-size: .95rem; color: var(--navy); }}
.cg-verdict .meta {{ font-size:.82rem; color:#64748B; margin-top: 6px; }}
.cg-tile {{ background:#fff; border-radius: 12px; padding: 16px; border:1px solid #EEF0F4; text-align:center; }}
.cg-tile .v {{ font-size: 1.9rem; font-weight: 800; color: var(--navy); }} .cg-tile .l {{ font-size:.85rem; color: var(--body); }}
.cg-tile .h {{ font-size:.75rem; color:#94A3B8; }}
.cg-reason {{ border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; border-left: 6px solid; color: var(--navy); }}
.cg-reason .code {{ font-family: Consolas, monospace; font-size: .74rem; opacity: .7; }}
.cg-chip {{ display:inline-block; font-size:.73rem; padding: 1px 8px; border-radius: 10px; background:#E2E8F0; color: var(--navy); margin-left: 6px; }}
.cg-check {{ background:#fff; border-radius: 10px; padding: 10px 12px; border:1px solid #E5E7EB; margin-bottom: 10px; }}
.cg-check b {{ color: var(--navy); font-size: .92rem; }} .cg-check p {{ margin: 3px 0 0 0; font-size: .82rem; }}
.cg-check.flag {{ border-color:#FCA5A5; background:#FEF2F2; }} .cg-check.ok {{ border-color:#BBF7D0; background:#F0FDF4; }}
.cg-check.warn {{ border-color:#FCD34D; background:#FFFBEB; }} .cg-check.na {{ opacity: .75; }}
.cg-live-row {{ font-size: .95rem; padding: 3px 0; color: var(--navy); }}
.cg-cta-band {{ background: var(--grad); border-radius: 20px; padding: 40px; text-align:center; margin: 40px 0 10px 0; }}
.cg-cta-band h2 {{ color:#fff; margin: 0 0 8px 0; }} .cg-cta-band p {{ color: rgba(255,255,255,.9); }}
.cg-avatar {{ width: 60px; height: 60px; border-radius: 50%; background: var(--grad); color:#fff; font-weight: 800; font-size: 1.3rem;
  display:flex; align-items:center; justify-content:center; }}
.cg-page-hero {{ background: var(--grad); border-radius: 18px; padding: 34px 36px; margin-bottom: 18px; }}
.cg-page-hero h1 {{ color:#fff; margin: 0 0 6px 0; font-size: 2.3rem; }} .cg-page-hero p {{ color: rgba(255,255,255,.92); margin:0; font-size: 1.08rem; }}
section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #EEF0F4; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{ background: var(--grad) !important; border: none !important; color:#fff !important; }}
button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {{ color:#fff !important; font-weight: 700; }}
[data-testid="stHeaderActionElements"], .cg-verdict h2 a, .cg-page-hero h1 a, .cg-hero h1 a {{ display:none !important; }}
/* sidebar starts below the top bar */
section[data-testid="stSidebar"] {{ top: 62px !important; height: calc(100vh - 62px) !important; }}
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {{ top: 72px !important; z-index: 999991; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
