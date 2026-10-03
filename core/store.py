"""SQLite persistence: claim history plus the photo/face index used for cross-claim reuse.

At hackathon scale (<10^4 rows) brute-force Hamming and cosine search in numpy takes
milliseconds, so no vector database is needed. Swap this module for Postgres +
pgvector when the index grows past ~10^5 rows; nothing else changes.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator

import numpy as np

from .config import DB_PATH, ensure_dirs
from .schemas import ClaimReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY, created_at TEXT, claimant_name TEXT, policy_no TEXT,
    vehicle_no TEXT, accident_date TEXT, overall_risk REAL, tier TEXT, report_json TEXT);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY, claim_id TEXT REFERENCES claims(id), filename TEXT,
    sha256 TEXT, role TEXT, path TEXT);
CREATE INDEX IF NOT EXISTS idx_items_sha ON items(sha256);
CREATE TABLE IF NOT EXISTS photo_hashes (
    item_id TEXT PRIMARY KEY, claim_id TEXT, phash INTEGER);
CREATE TABLE IF NOT EXISTS faces (
    item_id TEXT, claim_id TEXT, claimant_name TEXT, embedding BLOB);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def save_report(report: ClaimReport, phashes: dict[str, int], faces: list[tuple[str, np.ndarray]]) -> None:
    i = report.info
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?)",
                    (report.claim_id, report.created_at, i.claimant_name, i.policy_no, i.vehicle_no,
                     i.accident_date, report.overall_risk, report.tier.value,
                     json.dumps(report.to_dict(), default=str)))
        con.executemany("INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?)",
                        [(it.id, report.claim_id, it.filename, it.sha256, it.role.value, it.path)
                         for it in report.items])
        con.executemany("INSERT OR REPLACE INTO photo_hashes VALUES (?,?,?)",
                        [(item_id, report.claim_id, h - (1 << 64) if h >= (1 << 63) else h)
                         for item_id, h in phashes.items()])
        con.executemany("INSERT INTO faces VALUES (?,?,?,?)",
                        [(item_id, report.claim_id, i.claimant_name, emb.astype(np.float32).tobytes())
                         for item_id, emb in faces])


def find_file(sha256: str, exclude_claim: str) -> sqlite3.Row | None:
    with connect() as con:
        return con.execute(
            "SELECT i.claim_id, c.created_at FROM items i JOIN claims c ON c.id = i.claim_id "
            "WHERE i.sha256 = ? AND i.claim_id != ? LIMIT 1", (sha256, exclude_claim)).fetchone()


def nearest_photo(phash: int, exclude_claim: str) -> tuple[int, str, str] | None:
    """Smallest Hamming distance to any stored photo of another claim."""
    with connect() as con:
        rows = con.execute("SELECT p.phash, p.claim_id, c.created_at FROM photo_hashes p "
                           "JOIN claims c ON c.id = p.claim_id WHERE p.claim_id != ?",
                           (exclude_claim,)).fetchall()
    if not rows:
        return None
    hashes = np.array([r["phash"] for r in rows], dtype=np.int64).view(np.uint64)
    xor = np.bitwise_xor(hashes, np.uint64(phash & ((1 << 64) - 1)))
    dist = np.array([bin(int(v)).count("1") for v in xor])
    k = int(dist.argmin())
    return int(dist[k]), rows[k]["claim_id"], rows[k]["created_at"]


def nearest_face(embedding: np.ndarray, exclude_claim: str) -> tuple[float, str, str] | None:
    with connect() as con:
        rows = con.execute("SELECT claim_id, claimant_name, embedding FROM faces WHERE claim_id != ?",
                           (exclude_claim,)).fetchall()
    if not rows:
        return None
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = mat @ embedding.astype(np.float32)
    k = int(sims.argmax())
    return float(sims[k]), rows[k]["claim_id"], rows[k]["claimant_name"]


def list_claims(limit: int = 50) -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT id, created_at, claimant_name, vehicle_no, overall_risk, tier "
                           "FROM claims ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()


def load_report_json(claim_id: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT report_json FROM claims WHERE id = ?", (claim_id,)).fetchone()
    return json.loads(row["report_json"]) if row else None


def clear_all() -> None:
    with connect() as con:
        for t in ("faces", "photo_hashes", "items", "claims"):
            con.execute(f"DELETE FROM {t}")
