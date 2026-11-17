"""Persistencia local transaccional de catálogos y experimentos reproducibles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
import sklearn

from sismo import PIPELINE_VERSION

from sismo.data import clean_catalog


class Store:
    def __init__(self, root="runtime"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(exist_ok=True)
        self.database = self.root / "sismolab.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS catalogs (name TEXT PRIMARY KEY, content TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, metadata TEXT NOT NULL, checksum TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS validations (
                id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, created_at TEXT NOT NULL,
                summary TEXT NOT NULL, report TEXT NOT NULL)""")

    def connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def save_catalog(self, frame):
        # JSON estándar de Python conserva la representación round-trip de float64.
        records = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
        content = json.dumps(records, default=lambda value: value.isoformat(), allow_nan=False)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO catalogs VALUES (?, ?)", ("active", content))

    def load_catalog(self):
        with self.connect() as db:
            record = db.execute("SELECT content FROM catalogs WHERE name = ?", ("active",)).fetchone()
        return clean_catalog(pd.DataFrame(json.loads(record[0]))).frame if record else None

    def save_experiment(self, bundle):
        ident = bundle["metadata"]["id"]
        # Identificador generado por el backend; no se aceptan nombres de archivo externos.
        import uuid

        ident = str(uuid.UUID(ident))
        target = self.root / "models" / (ident + ".joblib")
        temporary = target.with_suffix(".tmp")
        joblib.dump(bundle, temporary, compress=3)
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(target)
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?)",
                (ident, json.dumps(bundle["metadata"], ensure_ascii=False), checksum),
            )

    def experiments(self):
        with self.connect() as db:
            records = db.execute("SELECT metadata FROM experiments ORDER BY rowid DESC").fetchall()
        return [json.loads(r[0]) for r in records]

    def load_experiment(self, ident):
        import uuid

        ident = str(uuid.UUID(ident))
        with self.connect() as db:
            record = db.execute(
                "SELECT metadata, checksum FROM experiments WHERE id = ?", (ident,)
            ).fetchone()
        if not record:
            raise ValueError("El experimento no existe en el registro local.")
        metadata = json.loads(record[0])
        if metadata.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("El experimento pertenece a otra versión del pipeline; reentrena con esta versión.")
        if metadata["sklearn"] != sklearn.__version__:
            raise ValueError("Versión de scikit-learn diferente: restaura uv.lock o reentrena.")
        path = self.root / "models" / (ident + ".joblib")
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record[1]:
            raise ValueError("El artefacto no existe o su checksum cambió.")
        # Sólo artefactos creados localmente. Nunca deserializar uploads joblib/pickle de usuarios.
        return joblib.load(path)

    def save_validation(self, bundle, new_catalog, rows, metrics, drift):
        from datetime import datetime, timezone
        from sismo.data import fingerprint

        experiment_id = bundle["metadata"]["id"]
        ident = hashlib.sha256((experiment_id + fingerprint(new_catalog)).encode()).hexdigest()
        summary = {
            "eventos": len(rows),
            "evaluables": int(rows.observable.sum()),
            "pendientes": int((~rows.observable).sum()),
            "metrics": metrics,
            "catalog_sha256": fingerprint(new_catalog),
        }
        report = {
            "predictions": json.loads(
                rows.drop(columns="future_max").to_json(orient="records", date_format="iso")
            ),
            "drift": json.loads(drift.to_json(orient="records")),
        }
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO validations VALUES (?, ?, ?, ?, ?)",
                (
                    ident,
                    experiment_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(summary),
                    json.dumps(report),
                ),
            )
        return ident

    def validations(self):
        with self.connect() as db:
            records = db.execute(
                "SELECT id, experiment_id, created_at, summary FROM validations ORDER BY rowid DESC"
            ).fetchall()
        return [
            {"id": ident, "experiment_id": experiment_id, "created_at": created_at, **json.loads(summary)}
            for ident, experiment_id, created_at, summary in records
        ]

    def validation_report(self, ident):
        with self.connect() as db:
            record = db.execute("SELECT report FROM validations WHERE id = ?", (ident,)).fetchone()
        if not record:
            raise ValueError("Validación no encontrada.")
        return json.loads(record[0])
