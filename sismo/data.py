"""Contrato, conversiones y auditoría del catálogo de eventos."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = ["time", "latitude", "longitude", "depth", "mag"]
QUALITY = ["nst", "gap", "dmin", "rms", "horizontalError", "depthError", "magError", "magNst"]
NUMERIC = REQUIRED[1:] + QUALITY
OPTIONAL_TEXT = ["magType", "place", "magSource"]
MAX_ROWS = 50_000


@dataclass
class CleanResult:
    frame: pd.DataFrame
    rejected: pd.DataFrame
    audit: dict
    schema: pd.DataFrame


def read_csv(content: bytes) -> pd.DataFrame:
    """UTF-8/Latin-1 y separador coma/punto y coma, sin interpretar código."""
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("El CSV supera 20 MB.")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
            first = text.splitlines()[0] if text.splitlines() else ""
            frame = pd.read_csv(io.StringIO(text), sep=";" if first.count(";") > first.count(",") else ",")
            if len(frame) > MAX_ROWS:
                raise ValueError(f"Máximo {MAX_ROWS:,} filas por catálogo en esta versión.")
            return frame
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"CSV vacío o mal formado: {exc}") from exc
    raise ValueError("No se pudo leer el archivo.")


def country(place: str) -> str:
    text = str(place)
    names = ["Chile", "Bolivia", "Peru", "Argentina", "Paraguay", "Brazil"]
    text = text.replace("Perú", "Peru").replace("Brasil", "Brazil")
    match = re.search(r",\s*([^,]+)$", text)
    if match and match.group(1).strip() in names:
        return match.group(1).strip()
    found = [p for p in names if p.lower() in text.lower()]
    return found[0] if len(found) == 1 else "Frontera" if len(found) > 1 else "Sin clasificar"


def clean_catalog(raw: pd.DataFrame) -> CleanResult:
    if raw.empty:
        raise ValueError("El catálogo está vacío.")
    if len(raw) > MAX_ROWS:
        raise ValueError(f"El catálogo combinado supera {MAX_ROWS:,} filas.")
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        raise ValueError("Hay nombres de columnas duplicados.")
    missing = sorted(set(REQUIRED) - set(df))
    if missing:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(missing))
    before = {c: str(df[c].dtype) for c in df}
    df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce", utc=True)
    if "updated" in df:
        df["updated"] = pd.to_datetime(df["updated"], format="mixed", errors="coerce", utc=True)
    conversions = {}
    for col in NUMERIC:
        if col not in df:
            df[col] = np.nan
        original_present = df[col].notna()
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col in QUALITY:
            df.loc[df[col] < 0, col] = np.nan
        conversions[col] = int((original_present & df[col].isna()).sum())
    invalid = pd.DataFrame(
        {
            "fecha inválida": df.time.isna(),
            "latitud inválida": ~df.latitude.between(-90, 90),
            "longitud inválida": ~df.longitude.between(-180, 180),
            "profundidad inválida": ~df.depth.between(-10, 800),
            "magnitud inválida": ~df.mag.between(-2, 10),
        },
        index=df.index,
    )
    bad = invalid.any(axis=1)
    rejected = raw.loc[bad].copy()
    rejected["motivo_rechazo"] = (
        invalid.loc[bad].apply(lambda row: "; ".join(row.index[row]), axis=1)
        if bad.any()
        else pd.Series(dtype=str)
    )
    df = df.loc[~bad].copy()
    if df.empty:
        raise ValueError("No hay eventos válidos: revisa fechas UTC y valores físicos.")
    for col in OPTIONAL_TEXT:
        if col not in df:
            df[col] = "Sin dato"
        df[col] = df[col].fillna("Sin dato").astype(str).str.strip().replace("", "Sin dato")
    if "id" not in df:
        df["id"] = pd.NA
    no_id = df["id"].isna() | df["id"].astype(str).str.strip().eq("")
    generated = (
        df[REQUIRED]
        .astype(str)
        .agg("|".join, axis=1)
        .map(lambda x: "local-" + hashlib.sha256(x.encode()).hexdigest()[:20])
    )
    df["id"] = df["id"].where(~no_id, generated).astype(str).str.strip()
    n_valid = len(df)
    # Para revisiones de un mismo ID se conserva la última fecha de actualización conocida.
    if "updated" in df:
        df = df.sort_values("updated", kind="stable", na_position="first")
    df = df.drop_duplicates("id", keep="last").drop_duplicates(REQUIRED, keep="last")
    df = df.sort_values(["time", "id"], kind="stable").reset_index(drop=True)
    df["country"] = df["place"].map(country)
    df["is_mag_guc"] = df["magSource"].str.lower().eq("guc").astype(int)
    schema = pd.DataFrame(
        [
            {
                "columna": c,
                "tipo_original": before.get(c, "ausente"),
                "tipo_final": str(df[c].dtype),
                "nulos": int(df[c].isna().sum()),
                "conversiones_a_nulo": conversions.get(c, 0),
            }
            for c in df
        ]
    )
    return CleanResult(
        df,
        rejected,
        {
            "entrada": len(raw),
            "válidos": len(df),
            "rechazados": int(bad.sum()),
            "duplicados": n_valid - len(df),
            "ids_generados": int(no_id.sum()),
            "constantes": [c for c in df if df[c].nunique(dropna=False) == 1],
        },
        schema,
    )


def fingerprint(df: pd.DataFrame) -> str:
    """Identidad reproducible del contenido utilizado, incluidos tipos y columnas."""
    stable = df.sort_values(["time", "id"]).reindex(sorted(df.columns), axis=1).reset_index(drop=True)
    digest = hashlib.sha256("|".join(f"{c}:{stable[c].dtype}" for c in stable).encode())
    digest.update(pd.util.hash_pandas_object(stable, index=False).values.tobytes())
    return digest.hexdigest()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    # Evita fórmulas activas al abrir exports con textos aportados por usuarios en Excel.
    output = frame.copy()
    for col in output.select_dtypes(include=["object", "string"]):
        output[col] = output[col].map(
            lambda x: "'" + x if isinstance(x, str) and x.startswith(("=", "+", "-", "@")) else x
        )
    return output.to_csv(index=False).encode("utf-8-sig")
