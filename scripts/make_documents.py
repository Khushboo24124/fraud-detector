"""Synthetic motor-claim documents: genuine garage invoices / RCs / licences + tampered variants.

Real fraud data is not available, so we manufacture it the way a fraudster would:
  T1 cover_and_retype  white box over the total, new value typed in another font (PDF editor style)
  T2 redact_clean      original removed cleanly, same font re-inserted, metadata kept clean
  T3 inflate_item      one line item inflated, totals left untouched
  T4 image_paste       invoice printed/scanned as JPEG and a number pasted in an image editor
  T5 metadata_edit     content unchanged, re-saved by an online editor days later

Usage:  python scripts/make_documents.py --n 40 --out data/eval/docs
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohan", "Kavya", "Arjun", "Meera",
         "Suresh", "Pooja", "Karan", "Neha", "Aditya", "Ishita"]
LAST = ["Sharma", "Patel", "Iyer", "Reddy", "Singh", "Gupta", "Nair", "Joshi", "Kulkarni", "Mehta"]
STATES = ["MH", "KA", "DL", "TN", "GJ", "RJ", "UP", "KL", "TS", "WB"]
GARAGES = ["Shree Ganesh Auto Works", "Metro Car Care", "Speedline Motors Service Centre",
           "Royal Auto Garage", "Prime Wheels Workshop", "City Motors Body Shop"]
PARTS = [("Front Bumper Replacement", 3500, 9500), ("Rear Bumper Repair", 1500, 5000),
         ("Headlamp Assembly (LH)", 2500, 12000), ("Bonnet Denting & Painting", 2000, 7000),
         ("Door Panel Denting", 1500, 6000), ("Windshield Glass", 4000, 15000),
         ("Side Mirror Assembly", 1200, 4500), ("Radiator Support", 1800, 6500),
         ("Fender Replacement", 2500, 8000), ("Labour Charges", 800, 4000),
         ("Wheel Alignment", 400, 1200), ("Tail Lamp Assembly", 1500, 6000)]
MAKES = ["Maruti Suzuki Swift", "Hyundai i20", "Tata Nexon", "Honda City", "Mahindra XUV300",
         "Kia Seltos", "Toyota Glanza", "Hyundai Creta"]


def money(x: float) -> str:
    return f"{x:,.2f}"


def plate(rng: random.Random) -> str:
    return f"{rng.choice(STATES)} {rng.randint(1, 50):02d} {rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}" \
           f"{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')} {rng.randint(1000, 9999)}"


def make_case(rng: random.Random) -> dict:
    accident = date(2026, rng.randint(1, 8), rng.randint(1, 28))
    items = []
    for name, lo, hi in rng.sample(PARTS, rng.randint(3, 7)):
        qty = 1 if "Labour" in name or "Alignment" in name else rng.choice([1, 1, 1, 2])
        rate = round(rng.uniform(lo, hi) / 50) * 50
        items.append({"desc": name, "qty": qty, "rate": float(rate), "amount": float(rate * qty)})
    subtotal = sum(i["amount"] for i in items)
    cgst = round(subtotal * 0.09, 2)
    return {
        "owner": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "plate": plate(rng),
        "make": rng.choice(MAKES), "garage": rng.choice(GARAGES),
        "invoice_no": f"INV-2026-{rng.randint(1000, 9999)}",
        "accident_date": accident, "invoice_date": accident + timedelta(days=rng.randint(2, 20)),
        "items": items, "subtotal": subtotal, "cgst": cgst, "sgst": cgst,
        "total": round(subtotal + 2 * cgst, 2),
    }


def render_invoice(case: dict, path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(f"Invoice {case['invoice_no']}")
    c.setAuthor(case["garage"])
    W, H = A4
    y = H - 60
    c.setFont("Helvetica-Bold", 16); c.drawString(50, y, case["garage"])
    c.setFont("Helvetica", 9); y -= 16
    c.drawString(50, y, "Plot 14, Industrial Estate, Near Highway Junction   |   GSTIN: 27ABCDE1234F1Z5")
    y -= 30; c.setFont("Helvetica-Bold", 13); c.drawString(50, y, "TAX INVOICE")
    c.setFont("Helvetica", 10); y -= 22
    rows = [("Invoice No:", case["invoice_no"]), ("Invoice Date:", case["invoice_date"].strftime("%d/%m/%Y")),
            ("Customer Name:", case["owner"]), ("Vehicle No:", case["plate"]), ("Make / Model:", case["make"])]
    for label, val in rows:
        c.setFont("Helvetica-Bold", 10); c.drawString(50, y, label)
        c.setFont("Helvetica", 10); c.drawString(150, y, val); y -= 16
    y -= 14
    cols = [50, 85, 360, 400, 480]
    c.setFont("Helvetica-Bold", 10)
    for x, h in zip(cols, ["S.No", "Description", "Qty", "Rate", "Amount"]):
        c.drawString(x, y, h)
    c.line(50, y - 4, 550, y - 4); y -= 20
    c.setFont("Helvetica", 10)
    for n, it in enumerate(case["items"], 1):
        c.drawString(cols[0], y, str(n)); c.drawString(cols[1], y, it["desc"])
        c.drawString(cols[2], y, str(it["qty"])); c.drawRightString(460, y, money(it["rate"]))
        c.drawRightString(550, y, money(it["amount"])); y -= 18
    c.line(50, y + 6, 550, y + 6); y -= 8
    for label, val, bold in [("Sub Total", case["subtotal"], False), ("CGST @ 9%", case["cgst"], False),
                             ("SGST @ 9%", case["sgst"], False), ("Grand Total", case["total"], True)]:
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.drawString(360, y, label); c.drawRightString(550, y, money(val)); y -= 18
    c.setFont("Helvetica", 8)
    c.drawString(50, 60, "This is a computer generated invoice. Subject to local jurisdiction.")
    c.save()


def render_rc(case: dict, path: Path, plate_override: str | None = None, owner_override: str | None = None) -> None:
    c = canvas.Canvas(str(path), pagesize=(500, 300))
    c.setFont("Helvetica-Bold", 13); c.drawString(30, 265, "CERTIFICATE OF REGISTRATION")
    c.setFont("Helvetica", 9); c.drawString(30, 250, "Transport Department - Form 23")
    reg = case["accident_date"] - timedelta(days=900)
    rows = [("Regn. No:", plate_override or case["plate"]), ("Owner Name:", owner_override or case["owner"]),
            ("Maker / Model:", case["make"]), ("Chassis No:", "MA3EWDE1S00" + case["invoice_no"][-4:]),
            ("Engine No:", "K12MN" + case["invoice_no"][-4:]),
            ("Date of Registration:", reg.strftime("%d/%m/%Y"))]
    y = 220
    for label, val in rows:
        c.setFont("Helvetica-Bold", 10); c.drawString(30, y, label)
        c.setFont("Helvetica", 10); c.drawString(160, y, val); y -= 22
    c.save()


def render_licence(case: dict, path: Path, face: Image.Image | None = None, name_override: str | None = None) -> None:
    img = Image.new("RGB", (1012, 638), (236, 242, 250))
    d = ImageDraw.Draw(img)
    try:
        f_big, f = ImageFont.truetype("arialbd.ttf", 40), ImageFont.truetype("arial.ttf", 30)
    except OSError:
        f_big = f = ImageFont.load_default()
    d.rectangle([0, 0, 1012, 90], fill=(30, 64, 140))
    d.text((30, 22), "DRIVING LICENCE", fill="white", font=f_big)
    lines = [f"DL No: {case['plate'][:2]}12 2019{case['invoice_no'][-4:]}567",
             f"Name: {name_override or case['owner']}", "Date of Birth: 14/03/1991",
             "Valid Till: 13/03/2041", "Class: LMV, MCWG"]
    for i, t in enumerate(lines):
        d.text((330, 130 + i * 60), t, fill=(20, 20, 20), font=f)
    d.rectangle([40, 130, 290, 450], outline=(80, 80, 80), width=3)
    if face is not None:
        img.paste(face.convert("RGB").resize((250, 320)), (40, 130))
    img.save(path, quality=92)


# ------------------------------------------------------------------ tampering
def _find(page, text: str):
    hits = page.search_for(text)
    return hits[-1] if hits else None


def t1_cover_and_retype(src: Path, dst: Path, case: dict, rng: random.Random) -> dict:
    new_total = round(case["total"] * rng.uniform(1.3, 2.2), -2)
    doc = pymupdf.open(src)
    page = doc[0]
    r = _find(page, money(case["total"]))
    cover = pymupdf.Rect(r.x0 - 2, r.y0 - 1, r.x1 + 2, r.y1 + 1)
    page.draw_rect(cover, color=None, fill=(1, 1, 1))
    page.insert_text((r.x0 - 8, r.y1 - 2), money(new_total), fontname="tiro", fontsize=10.5)
    doc.set_metadata({**doc.metadata, "modDate": pymupdf.get_pdf_now(), "producer": "PDF Editor Online"})
    doc.save(dst)
    return {"field": "Grand Total", "original": case["total"], "tampered": new_total}


def t2_redact_clean(src: Path, dst: Path, case: dict, rng: random.Random) -> dict:
    new_total = round(case["total"] * rng.uniform(1.2, 1.8), -2)
    doc = pymupdf.open(src)
    page = doc[0]
    r = _find(page, money(case["total"]))
    page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions()
    width = pymupdf.get_text_length(money(new_total), fontname="helv", fontsize=10)
    page.insert_text((r.x1 - width, r.y1 - 2.2), money(new_total), fontname="helv", fontsize=10)
    doc.save(dst, garbage=4, deflate=True)  # full rewrite: no incremental trace, metadata untouched
    return {"field": "Grand Total", "original": case["total"], "tampered": new_total}


def t3_inflate_item(src: Path, dst: Path, case: dict, rng: random.Random) -> dict:
    it = max(case["items"], key=lambda i: i["amount"])
    new_amt = round(it["amount"] * rng.uniform(1.5, 2.5), -2)
    doc = pymupdf.open(src)
    page = doc[0]
    hits = page.search_for(money(it["amount"]))
    r = hits[-1]
    page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions()
    width = pymupdf.get_text_length(money(new_amt), fontname="cour", fontsize=10)
    page.insert_text((r.x1 - width, r.y1 - 2.2), money(new_amt), fontname="cour", fontsize=10)
    doc.save(dst)
    return {"field": it["desc"], "original": it["amount"], "tampered": new_amt}


def t4_image_paste(src: Path, dst: Path, case: dict, rng: random.Random) -> dict:
    """Print-to-image, then paste a new total in an image editor and re-save as JPEG."""
    new_total = round(case["total"] * rng.uniform(1.3, 2.0), -2)
    doc = pymupdf.open(src)
    page = doc[0]
    r = _find(page, money(case["total"]))
    zoom = 200 / 72
    img = Image.open(io.BytesIO(page.get_pixmap(dpi=200).tobytes("png"))).convert("RGB")
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=88); img = Image.open(buf).convert("RGB")
    d = ImageDraw.Draw(img)
    box = [int(r.x0 * zoom) - 6, int(r.y0 * zoom) - 4, int(r.x1 * zoom) + 6, int(r.y1 * zoom) + 4]
    d.rectangle(box, fill=(255, 255, 255))
    try:
        font = ImageFont.truetype("arial.ttf", int(rng.uniform(30, 38)))
    except OSError:
        font = ImageFont.load_default()
    text = money(new_total)
    tw = d.textlength(text, font=font)
    d.text((box[2] - 6 - tw, box[1]), text, fill=(10, 10, 10), font=font)
    img.save(dst, "JPEG", quality=90)
    return {"field": "Grand Total", "original": case["total"], "tampered": new_total}


def t5_metadata_edit(src: Path, dst: Path, case: dict, rng: random.Random) -> dict:
    doc = pymupdf.open(src)
    created = datetime.strptime(doc.metadata["creationDate"][2:16], "%Y%m%d%H%M%S")
    later = created + timedelta(days=rng.randint(5, 40))
    doc.set_metadata({**doc.metadata, "producer": "iLovePDF", "modDate": later.strftime("D:%Y%m%d%H%M%S")})
    doc.save(dst)
    return {"field": "metadata", "original": "", "tampered": "re-saved by editor"}


TAMPERS = {"T1_cover_and_retype": (t1_cover_and_retype, ".pdf"), "T2_redact_clean": (t2_redact_clean, ".pdf"),
           "T3_inflate_item": (t3_inflate_item, ".pdf"), "T4_image_paste": (t4_image_paste, ".jpg"),
           "T5_metadata_edit": (t5_metadata_edit, ".pdf")}


def genuine_scan(src: Path, dst: Path) -> None:
    """A genuine invoice photographed/scanned to JPEG: the honest counterpart of T4."""
    page = pymupdf.open(src)[0]
    img = Image.open(io.BytesIO(page.get_pixmap(dpi=200).tobytes("png"))).convert("RGB")
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=88)
    Image.open(buf).convert("RGB").save(dst, "JPEG", quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/eval/docs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(args.n):
        case = make_case(rng)
        g = out / f"inv_{k:03d}_genuine.pdf"
        render_invoice(case, g)
        rows.append({"path": g.name, "label": 0, "variant": "genuine_pdf", "detail": ""})
        gs = out / f"inv_{k:03d}_genuine_scan.jpg"
        genuine_scan(g, gs)
        rows.append({"path": gs.name, "label": 0, "variant": "genuine_scan", "detail": ""})
        name = list(TAMPERS)[k % len(TAMPERS)]
        fn, ext = TAMPERS[name]
        t = out / f"inv_{k:03d}_{name}{ext}"
        detail = fn(g, t, case, rng)
        rows.append({"path": t.name, "label": 1, "variant": name, "detail": str(detail)})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "variant", "detail"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} documents to {out}")


if __name__ == "__main__":
    main()
