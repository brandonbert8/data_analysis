"""Variables causales y etiquetas con censura temporal explícita."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

F19 = [
    "latitude",
    "longitude",
    "depth",
    "nst",
    "gap",
    "dmin",
    "rms",
    "horizontalError",
    "depthError",
    "magError",
    "magNst",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "events_last_7d",
    "events_last_30d",
    "time_since_last",
    "is_mag_guc",
]
F25 = F19 + ["mag", "cell_n7", "cell_n30", "cell_maxmag30", "glob_maxmag_30d", "days_since_last_cell"]
CATS = ["magType", "country"]


@dataclass(frozen=True)
class FeatureConfig:
    horizon: int = 7
    magnitude: float = 5.0
    grid: float = 0.5

    def __post_init__(self):
        if not 1 <= self.horizon <= 30 or not 3 <= self.magnitude <= 8 or not 0.1 <= self.grid <= 2:
            raise ValueError("Horizonte, magnitud o tamaño de celda fuera de rango.")


def past_stats(t, mag, window):
    """Ventana [t-window, t), excluye TODOS los eventos simultáneos."""
    low = np.searchsorted(t, t - window, side="left")
    high = np.searchsorted(t, t, side="left")
    maximum = np.full(len(t), np.nan)
    elapsed = np.full(len(t), np.nan)
    for i, (a, b) in enumerate(zip(low, high)):
        if b > a:
            maximum[i] = mag[a:b].max()
        if b:
            elapsed[i] = t[i] - t[b - 1]
    return high - low, maximum, elapsed


def build_features(catalog: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    df = catalog.sort_values(["time", "id"]).reset_index(drop=True).copy()
    t = df.time.astype("int64").to_numpy() / 86_400_000_000_000
    mag = df.mag.to_numpy()
    df["lat_grid"] = np.floor(df.latitude / config.grid) * config.grid
    df["lon_grid"] = np.floor(df.longitude / config.grid) * config.grid
    for label, values, period in [("month", df.time.dt.month, 12), ("hour", df.time.dt.hour, 24)]:
        df[label + "_sin"] = np.sin(2 * np.pi * values / period)
        df[label + "_cos"] = np.cos(2 * np.pi * values / period)
    n7, _, _ = past_stats(t, mag, 7)
    n30, mx30, last = past_stats(t, mag, 30)
    df["events_last_7d"], df["events_last_30d"] = n7 + 1, n30 + 1
    df["glob_maxmag_30d"] = mx30
    df["time_since_last"] = np.log1p(last * 24)
    for c in ["cell_n7", "cell_n30", "cell_maxmag30", "days_since_last_cell", "future_max"]:
        df[c] = np.nan
    for idx in df.groupby(["lat_grid", "lon_grid"]).indices.values():
        tc, mc = t[idx], mag[idx]
        c7, _, _ = past_stats(tc, mc, 7)
        c30, cm, elapsed = past_stats(tc, mc, 30)
        df.loc[idx, "cell_n7"] = c7
        df.loc[idx, "cell_n30"] = c30
        df.loc[idx, "cell_maxmag30"] = cm
        df.loc[idx, "days_since_last_cell"] = np.log1p(elapsed)
        low = np.searchsorted(tc, tc, side="right")
        high = np.searchsorted(tc, tc + config.horizon, side="right")
        df.loc[idx, "future_max"] = [mc[a:b].max() if b > a else -np.inf for a, b in zip(low, high)]
    # La cobertura se supone continua hasta el último evento del catálogo, no hasta el reloj actual.
    df["label_end"] = df.time + pd.Timedelta(days=config.horizon)
    df["observable"] = df.label_end <= df.time.max()
    df["warmup_complete"] = df.time >= df.time.min() + pd.Timedelta(days=30)
    df["target"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[df.observable, "target"] = (df.loc[df.observable, "future_max"] >= config.magnitude).astype(int)
    return df


def temporal_partitions(frame: pd.DataFrame, horizon: int, train_fraction=0.6, test_fraction=0.2):
    """Bloques por fecha con embargo estricto en AMBAS fronteras."""
    eligible = frame.loc[frame.observable & frame.warmup_complete].sort_values("time").copy()
    times = eligible.time.drop_duplicates().sort_values().reset_index(drop=True)
    if len(times) < 50:
        raise ValueError(
            "Se necesitan al menos 50 fechas elegibles, 30 días de historia y horizonte observable."
        )
    train_end = times.iloc[int(len(times) * train_fraction)]
    test_start = times.iloc[int(len(times) * (1 - test_fraction))]
    delta = pd.Timedelta(days=horizon)
    train = eligible.loc[(eligible.time < train_end) & (eligible.time + delta < train_end)]
    validation = eligible.loc[(eligible.time >= train_end) & (eligible.time + delta < test_start)]
    test = eligible.loc[eligible.time >= test_start]
    embargo = eligible.loc[~eligible.index.isin(train.index.union(validation.index).union(test.index))]
    if min(len(train), len(validation), len(test)) < 10:
        raise ValueError("Bloques demasiado pequeños tras el embargo; usa un catálogo más extenso.")
    return {"Entrenamiento": train, "Validación": validation, "Prueba": test, "Embargo": embargo}


def temporal_folds(frame: pd.DataFrame, horizon: int, n_splits=3):
    """Origen expansivo por fechas, sin confundir número de filas con días."""
    times = frame.time.drop_duplicates().sort_values().reset_index(drop=True)
    boundaries = np.linspace(0, len(times), n_splits + 2, dtype=int)
    for i in range(1, n_splits + 1):
        start = times.iloc[boundaries[i]]
        stop = (
            times.iloc[boundaries[i + 1]]
            if boundaries[i + 1] < len(times)
            else times.iloc[-1] + pd.Timedelta("1ns")
        )
        train = np.flatnonzero((frame.time + pd.Timedelta(days=horizon) < start).to_numpy())
        valid = np.flatnonzero(((frame.time >= start) & (frame.time < stop)).to_numpy())
        yield train, valid
