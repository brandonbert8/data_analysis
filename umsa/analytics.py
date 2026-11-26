"""Consultas de sólo lectura sobre todos los datasets, con procedencia verificable."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from umsa.data import DATASET_DIR
from umsa.notebook import notebook_cells


def normalized_text(value: str) -> str:
    return " ".join("".join(c for c in unicodedata.normalize("NFD", value.lower())
                            if unicodedata.category(c) != "Mn").split())


def teacher_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Separa personas/roles y unifica sólo prefijos de nombre sin ambigüedad."""
    rows = []
    for _, row in frame.iterrows():
        for person in str(row.get("tutores", "")).split(";"):
            match = re.search(r"[,\s]+(tutor[ae]?|asesor[ae]?)\.?\s*$", person, re.IGNORECASE)
            name = (person[:match.start()] if match else person).strip(" ,")
            if not name or name.lower() == "nan":
                continue
            role = match.group(1).lower() if match else "sin especificar"
            role = "tutor" if role.startswith("tutor") else "asesor" if role.startswith("asesor") else role
            rows.append({"docente": name, "nombre_original": name, "rol": role,
                         "archivo_pdf": row["archivo_pdf"], "titulo": row.get("titulo", ""),
                         "fecha": row.get("fecha"), "autores": row.get("autores", "")})
    columns = ["docente", "nombre_original", "rol", "archivo_pdf", "titulo", "fecha", "autores"]
    result = pd.DataFrame(rows, columns=columns)
    names = {normalized_text(name): name for name in result["docente"]}
    canonical = {}
    for key, display in names.items():
        longer = [other for other in names if other.startswith(key + " ")]
        longest = [other for other in longer if not any(x.startswith(other + " ") for x in longer)]
        canonical[key] = names[longest[0]] if len(longest) == 1 else display
    result["docente"] = result["docente"].map(lambda name: canonical[normalized_text(name)])
    return result.drop_duplicates(["docente", "rol", "archivo_pdf"]).reset_index(drop=True)


class DatasetQueryError(ValueError):
    pass


class DatasetCatalog:
    def __init__(self, directory: Path = DATASET_DIR):
        self.directory = directory
        self.tables: dict[str, pd.DataFrame] = {}
        self.origins: dict[str, str] = {}
        for path in sorted(directory.iterdir()):
            if path.suffix not in {".csv", ".parquet"}:
                continue
            name = "ds_" + re.sub(r"\W", "_", path.stem)
            self.tables[name] = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
            self.origins[name] = f"dataset/{path.name}"
        aliases = {
            "metadatos": "ds_metadatos_informatica",
            "corpus": "ds_df_maestro_final_modelos_70" if "ds_df_maestro_final_modelos_70" in self.tables
            else "ds_resultados_modelos_umsa_70",
            "benchmark": "ds_benchmark_ml_vs_dl_umsa",
        }
        for alias, table in aliases.items():
            if table in self.tables:
                self.tables[alias] = self.tables[table]
                self.origins[alias] = self.origins[table]
        for alias, table in (("docentes_original", "metadatos"), ("docentes_final", "corpus")):
            if table in self.tables:
                self.tables[alias] = teacher_rows(self.tables[table])
                self.origins[alias] = self.origins[table] + " · docentes separados y normalizados"
        if "metadatos" in self.tables:
            projects = self.tables["metadatos"].copy()
            if "corpus" in self.tables:
                final = self.tables["corpus"]
                extra = [column for column in final if column not in projects or column == "archivo_pdf"]
                projects = projects.merge(final[extra], on="archivo_pdf", how="left", validate="one_to_one")
                projects["en_corpus_final"] = projects["archivo_pdf"].isin(final["archivo_pdf"])
            projects["pdf_disponible"] = projects["archivo_pdf"].map(lambda name: (directory / "PDFs" / name).is_file())
            removed = {}
            for cell in notebook_cells():
                for name in re.findall(r"\['archivo_pdf'\]\s*!=\s*'([^']+)'", cell["source"]):
                    removed[name] = f"Retiro explícito por duplicidad en repositorioUMSA.ipynb · celda {cell['cell']}"
            projects["motivo_exclusion"] = projects.apply(
                lambda row: None if row.get("en_corpus_final") else removed.get(row["archivo_pdf"],
                    "Motivo individual no documentado en los artefactos; el registro está ausente del corpus final."), axis=1)
            self.tables["proyectos"] = projects
            self.origins["proyectos"] = self.origins["metadatos"] + " + métricas del corpus final (unión por archivo_pdf)"
            subjects = []
            for row in projects.itertuples():
                for subject in str(getattr(row, "materias", "")).split(";"):
                    if subject.strip() and subject.lower() != "nan":
                        subjects.append({"archivo_pdf": row.archivo_pdf, "materia": subject.strip(), "fecha": row.fecha})
            self.tables["materias"] = pd.DataFrame(subjects, columns=["archivo_pdf", "materia", "fecha"]).drop_duplicates()
            self.origins["materias"] = self.origins["metadatos"] + " · materias separadas"

    def project(self, filename: str, selected_filename: str = "") -> dict:
        if selected_filename and filename != selected_filename:
            raise DatasetQueryError("El ámbito está limitado al documento seleccionado.")
        rows = self.tables["proyectos"].loc[self.tables["proyectos"]["archivo_pdf"].eq(filename)]
        if rows.empty:
            raise DatasetQueryError("Documento no encontrado. Usa buscar_proyectos para obtener su archivo_pdf.")
        # Pandas serializa NaN como null y conserva tipos JSON simples.
        return json.loads(rows.drop(columns=["texto_completo", "conclusiones_limpias"], errors="ignore")
                          .to_json(orient="records", force_ascii=False))[0]

    def schema(self, filename: str = "") -> str:
        entries = []
        for name, frame in self.tables.items():
            if filename and "archivo_pdf" not in frame:
                continue
            rows = frame if not filename else frame.loc[frame["archivo_pdf"].eq(filename)]
            entries.append({"tabla": name, "filas": len(rows), "fuente": self.origins[name],
                            "columnas": {col: str(dtype) for col, dtype in frame.dtypes.items()}})
        return json.dumps(entries, ensure_ascii=False)

    def query(self, sql: str, filename: str = "") -> dict:
        if not isinstance(sql, str) or len(sql) > 8000 or not re.match(r"\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
            raise DatasetQueryError("Sólo se admite una consulta SELECT de lectura.")
        try:
            parsed = parse_one(sql, read="sqlite")
            # SQLite omite algunos eventos READ en JOIN USING; el AST conserva ambas fuentes.
            accessed = {table.name.lower() for table in parsed.find_all(exp.Table) if table.name.lower() in self.tables}
        except ParseError:
            raise DatasetQueryError("La consulta SQL no se pudo interpretar. Usa sintaxis SQLite SELECT.") from None
        allowed_functions = {"count", "sum", "avg", "min", "max", "round", "coalesce", "ifnull", "nullif",
                             "lower", "upper", "length", "substr", "replace", "trim", "like", "abs",
                             "total", "group_concat", "printf", "instr", "norm", "row_number", "rank", "dense_rank",
                             "median", "stddev"}

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_READ:
                if arg1 not in self.tables or (filename and "archivo_pdf" not in self.tables[arg1]):
                    return sqlite3.SQLITE_DENY
                accessed.add(arg1)
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION:
                return sqlite3.SQLITE_OK if (arg2 or "").lower() in allowed_functions else sqlite3.SQLITE_DENY
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        with sqlite3.connect(":memory:") as connection:
            connection.create_function("norm", 1, lambda value: normalized_text(str(value or "")))
            connection.create_aggregate("median", 1, Median)
            connection.create_aggregate("stddev", 1, StandardDeviation)
            for name, frame in self.tables.items():
                if filename:
                    if "archivo_pdf" not in frame:
                        continue
                    frame = frame.loc[frame["archivo_pdf"].eq(filename)]
                frame.to_sql(name, connection, index=False)
            connection.set_authorizer(authorize)
            deadline = time.monotonic() + 2
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            try:
                cursor = connection.execute(sql)
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchmany(101)
            except sqlite3.Error as exc:
                raise DatasetQueryError(f"Consulta no válida: {exc}. Revisa las tablas y columnas del catálogo.") from None
        if not accessed:
            raise DatasetQueryError("La consulta debe leer al menos una tabla del catálogo para aportar evidencia.")
        records = [{col: value[:1200] if isinstance(value, str) else value for col, value in zip(columns, row)}
                   for row in rows[:100]]
        return {"sql": sql, "rows": records, "returned_rows": len(records), "truncated": len(rows) > 100,
                "tables": sorted(accessed), "files": sorted({self.origins[name] for name in accessed}),
                "population": [{"table": name,
                                "base": f"{len(self.tables['metadatos'])} registros originales" if name in ("metadatos", "docentes_original", "proyectos", "materias")
                                else "70 tesis únicas finales" if name in ("corpus", "docentes_final")
                                else f"{len(self.tables[name])} filas del archivo de origen"}
                               for name in sorted(accessed)],
                "note": "Máximo 100 filas; textos de resultados limitados a 1200 caracteres por valor."}


class Median:
    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            try:
                number = float(value)
                if np.isfinite(number):
                    self.values.append(number)
            except (ValueError, TypeError):
                pass

    def finalize(self):
        return float(np.median(self.values)) if self.values else None


class StandardDeviation(Median):
    def finalize(self):
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else None
