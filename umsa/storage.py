"""Persistencia local de auditorías académicas."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from umsa import PIPELINE_VERSION


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


class AuditStore:
    def __init__(self, root: str | Path = "runtime"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "umsa_auditor.sqlite3"
        with self.connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS audits (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    pipeline_version INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )

    def connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def save_audit(self, audit: dict) -> str:
        payload = _json_safe(audit)
        ident = hashlib.sha256(
            (str(payload.get("archivo")) + json.dumps(payload, sort_keys=True, ensure_ascii=False)).encode()
        ).hexdigest()[:24]
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO audits VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ident,
                    datetime.now(UTC).isoformat(),
                    str(payload.get("archivo", "documento.pdf")),
                    float(payload.get("puntaje", 0.0)),
                    str(payload.get("veredicto", "")),
                    PIPELINE_VERSION,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        return ident

    def audits(self) -> list[dict]:
        with self.connect() as db:
            records = db.execute(
                "SELECT id, created_at, filename, score, verdict FROM audits ORDER BY rowid DESC"
            ).fetchall()
        return [
            {
                "id": ident,
                "created_at": created_at,
                "filename": filename,
                "score": score,
                "verdict": verdict,
            }
            for ident, created_at, filename, score, verdict in records
        ]

    def audit(self, ident: str) -> dict:
        with self.connect() as db:
            record = db.execute("SELECT payload FROM audits WHERE id = ?", (ident,)).fetchone()
        if not record:
            raise ValueError("La auditoría no existe en el registro local.")
        return json.loads(record[0])

    def delete_audit(self, ident: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM audits WHERE id = ?", (ident,))
