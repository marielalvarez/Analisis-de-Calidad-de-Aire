
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.structural import UnobservedComponents


# ----------------------------- Helpers generales ----------------------------- #

def sanitize_colnames(cols: List[str]) -> List[str]:
    """Normaliza nombres de columnas a [0-9a-zA-Z_]."""
    return [re.sub(r'[^0-9a-zA-Z_]+', '', c) for c in cols]

def infer_date_and_station_cols(
    df: pd.DataFrame,
    date_regex: str = r"(?:^|_)(date|fecha|time|datetime)(?:_|$)",
    station_regex: str = r"(?:^|_)(estacion|station|site|sitie|ubic|planta)(?:_|$)"
) -> Tuple[str, Optional[str]]:
    """Intenta inferir columnas de fecha y de estación por regex; la fecha cae a la primera columna si no hay match."""
    candidates_date = [c for c in df.columns if re.search(date_regex, c, flags=re.IGNORECASE)]
    date_col = candidates_date[0] if candidates_date else df.columns[0]

    candidates_station = [c for c in df.columns if re.search(station_regex, c, flags=re.IGNORECASE)]
    station_col = candidates_station[0] if candidates_station else None
    return date_col, station_col

def infer_freq_from_index(idx: pd.Index) -> pd.Timedelta:
    """Estima la frecuencia modal a partir del índice (DatetimeIndex)."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return pd.Timedelta("0s")
    d = pd.Series(idx).diff().dropna()
    return d.mode().iloc[0] if not d.empty else pd.Timedelta("0s")


# -------------------------- Perfilado / PAR 1 compacto ----------------------- #

def load_and_basic_profile(
    data: Union[str, pd.DataFrame],
    *,
    date_col: Optional[str] = None,
    station_col: Optional[str] = None,
    normalize_colnames: bool = True,
    date_regex: str = r"(?:^|_)(date|fecha|time|datetime)(?:_|$)",
    station_regex: str = r"(?:^|_)(estacion|station|site|sitie|ubic|planta)(?:_|$)",
    set_datetime_index: bool = True,
    show_na_map: bool = True,
    na_map_max_rows: int = 5000,
    top_missing_k: int = 10,
    bad_range_quantile: float = 0.99,
    bad_range_multiplier: float = 3.0,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Carga (o recibe) un DataFrame, estandariza nombres, detecta columnas clave, setea índice temporal,
    perfila columnas (numéricas/categóricas), duplica, frecuencia, NA heatmap y checa valores extremos.
    Devuelve artefactos útiles para el siguiente paso.
    """
    # 1) Carga o copia
    if isinstance(data, str):
        df = pd.read_excel(data)
    else:
        df = data.copy()

    # 2) Normaliza nombres
    if normalize_colnames:
        df.columns = sanitize_colnames(df.columns.tolist())

    # 3) Detecta date/station si no vienen
    if date_col is None or (station_col is None and station_regex):
        auto_date, auto_station = infer_date_and_station_cols(df, date_regex, station_regex)
        date_col = date_col or auto_date
        station_col = station_col or auto_station

    # 4) Parse fecha y orden
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    if set_datetime_index:
        df = df.set_index(date_col)

    # 5) Numéricas vs categóricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # 6) Diccionario de datos
    dict_rows = []
    for c in df.columns:
        s = df[c]
        row = {
            "columna": c,
            "tipo": str(s.dtype),
            "nulos": int(s.isna().sum()),
            "%nulos": round(100 * s.isna().mean(), 2),
        }
        if s.dtype == "O" or getattr(s.dtype, "name", "") == "category":
            uniq = s.nunique(dropna=True)
            row["distintos"] = int(uniq)
            # muestra valores
            vals = s.dropna().unique()[:8]
            row["muestra_valores"] = ", ".join(map(str, vals)) + ("..." if uniq > 8 else "")
        else:
            row["min"] = pd.to_numeric(s, errors="coerce").min()
            row["p1"] = s.quantile(0.01)
            row["p50"] = s.median()
            row["p99"] = s.quantile(0.99)
            row["max"] = pd.to_numeric(s, errors="coerce").max()
        dict_rows.append(row)
    data_dictionary = pd.DataFrame(dict_rows).sort_values("columna").reset_index(drop=True)

    # 7) Duplicados por (fecha, estación?) en el espacio original
    if set_datetime_index:
        base = df.reset_index()
    else:
        base = df.copy()
    if station_col and station_col in base.columns:
        dups = base.duplicated(subset=[date_col, station_col]).sum()
    else:
        dups = base.duplicated(subset=[date_col]).sum()

    # 8) Inferencia de frecuencia
    freq_guess = infer_freq_from_index(df.index if set_datetime_index else pd.to_datetime(df[date_col], errors="coerce"))

    # 9) NA por columna (top-k) y NA heatmap opcional
    missing_counts = df[num_cols].isna().sum().sort_values(ascending=False)
    if show_na_map and len(df) > 0 and len(num_cols) > 0:
        sample = df[num_cols].iloc[:na_map_max_rows]
        plt.figure(figsize=(10, 4))
        sns.heatmap(sample.isna(), cbar=False)
        plt.title("Mapa de NA (muestra)")
        plt.tight_layout()
        plt.show()

    # 10) Rango “malo” (negativos y > multiplier * p99)
    bad_ranges = {}
    for c in num_cols:
        s = df[c]
        try:
            neg = int((s < 0).sum())
            hi = s.quantile(bad_range_quantile) * bad_range_multiplier
            big = int((s > hi).sum())
            bad_ranges[c] = {"negativos": neg, f"mayores_{bad_range_multiplier}x_p{int(bad_range_quantile*100)}(>{hi:.2f})": big}
        except Exception:
            bad_ranges[c] = {"negativos": np.nan, f"mayores_{bad_range_multiplier}x_p{int(bad_range_quantile*100)}": np.nan}
    bad_ranges_df = pd.DataFrame(bad_ranges).T.reset_index().rename(columns={"index": "columna"})

    if verbose:
        print("Dimensión (filas, columnas):", df.shape)
        print("\nColumnas numéricas:", num_cols)
        print("\nColumnas categóricas:", cat_cols)
        print(f"\nDuplicados detectados: {dups}")
        print("\nAproximación de frecuencia temporal:", freq_guess)
        print("\nFaltantes por columna (top):")
        print(missing_counts.head(top_missing_k))

    return {
        "df": df,
        "date_col": date_col,
        "station_col": station_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "duplicates": int(dups),
        "freq_guess": freq_guess,
        "data_dictionary": data_dictionary,
        "bad_ranges": bad_ranges_df,
        "missing_counts": missing_counts,
    }


# --------------------- Preprocesamiento + Imputación / PAR 2 ----------------- #

def _hampel(series: pd.Series, window: int = 7*24, n_sigmas: float = 3.0) -> pd.Series:
    """Filtro Hampel: marca outliers respecto a mediana/ MAD local y los reemplaza con NaN."""
    s = series.copy()
    k = int(window)
    if k < 3 or s.isna().all():
        return s
    med = s.rolling(k, center=True, min_periods=max(3, k // 3)).median()
    mad = (s - med).abs().rolling(k, center=True, min_periods=max(3, k // 3)).median() * 1.4826
    mask = (mad > 0) & ((s - med).abs() > n_sigmas * mad)
    s[mask] = np.nan
    return s

def _impute_series(
    ts: pd.Series,
    period_hint_hours: int = 1,
    small_gap: int = 6,
    method: str = "kalman"
) -> pd.Series:
    """Imputación: interpolate(time) limitada + Kalman (UCM) -> STL -> interpolate(time)."""
    s = ts.copy()
    s_interp = s.interpolate(method="time", limit=small_gap, limit_direction="both")

    if method == "none":
        return s_interp

    # Periodo para estacionalidad (horario/diario/semanal)
    if period_hint_hours <= 1:
        period = 24            # horario -> día
    elif period_hint_hours >= 24:
        period = 7             # diario/semanal
    else:
        period = 24            # fallback: día

    try:
        # Nota: algunos statsmodels prefieren trend=True para local linear trend;
        # mantenemos la firma original y capturamos excepciones si falla.
        mod = UnobservedComponents(s_interp, level="local linear trend",
                                   seasonal=period if period >= 2 else None)
        res = mod.fit(disp=False)
        fitted = res.predict(start=s.index.min(), end=s.index.max())
        s_interp[s.isna()] = fitted[s.isna()]
    except Exception:
        try:
            stl = STL(s_interp, period=max(2, period), robust=True).fit()
            resid = s_interp - stl.seasonal
            resid = resid.interpolate(method="time", limit_direction="both")
            s_interp[s.isna()] = (resid + stl.seasonal)[s.isna()]
        except Exception:
            s_interp = s_interp.interpolate(method="time", limit_direction="both")

    return s_interp

def prepare_clean_impute(
    df: pd.DataFrame,
    *,
    date_col: str,
    num_cols: List[str],
    cat_cols: List[str],
    station_col: Optional[str] = None,
    freq_guess: Optional[pd.Timedelta] = None,
    coverage_threshold: float = 0.80,
    q_low: float = 0.01,
    q_hi: float = 0.99,
    hampel_days: int = 7,
    hampel_sigmas: float = 3.0,
    impute_method: str = "kalman",
    small_gap: int = 6,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Replica el flujo del PAR 2:
    - Filtra por cobertura de columnas numéricas
    - Dedup por (fecha, estación?)
    - Clipping por cuantiles (no negativo)
    - Filtro Hampel con ventana basada en frecuencia inferida
    - Imputación por Kalman→STL→interpolate y flags *_imputed

    Devuelve: (df_prep, target_cols)
    """
    # Asegurar índice temporal
    if not isinstance(df.index, pd.DatetimeIndex):
        df_work = df.copy().set_index(pd.to_datetime(df[date_col], errors="coerce"))
    else:
        df_work = df.copy()
    df_work = df_work[~df_work.index.isna()]

    # Cobertura
    keep_cols = [c for c in num_cols if c in df_work.columns and df_work[c].isna().mean() < coverage_threshold]
    # Preservar categóricas disponibles
    cats_present = [c for c in cat_cols if c in df_work.columns]
    df_prep = df_work[cats_present + keep_cols]

    if verbose:
        print("Columnas incluidas (por cobertura):", keep_cols)

    target_cols = keep_cols.copy()

    # Deduplicación por fecha/(estación)
    if station_col and station_col in df_prep.columns:
        df_prep = (
            df_prep.reset_index()
                   .drop_duplicates(subset=[date_col, station_col])
                   .set_index(date_col)
        )
    else:
        df_prep = (
            df_prep.reset_index()
                   .drop_duplicates(subset=[date_col])
                   .set_index(date_col)
        )

    # Clipping por cuantiles
    def _clip_by_quantiles(g: pd.DataFrame, cols: List[str], ql: float, qh: float) -> pd.DataFrame:
        g = g.copy()
        for c in cols:
            s = g[c]
            lo, hi = s.quantile(ql), s.quantile(qh)
            g[c] = s.clip(lower=max(lo, 0), upper=hi)
        return g

    if station_col and station_col in df_prep.columns:
        df_prep = df_prep.groupby(station_col, group_keys=False).apply(
            _clip_by_quantiles, cols=target_cols, ql=q_low, qh=q_hi
        )
    else:
        df_prep = _clip_by_quantiles(df_prep, cols=target_cols, ql=q_low, qh=q_hi)

    # Hampel: ventana en horas según frecuencia
    if freq_guess is None or freq_guess <= pd.Timedelta("0s"):
        freq_guess = infer_freq_from_index(df_prep.index)
    hours = max(int(freq_guess.total_seconds() // 3600), 1)
    win = max(1, (hampel_days * 24) // hours)

    for c in target_cols:
        if station_col and station_col in df_prep.columns:
            df_prep[c] = df_prep.groupby(station_col, group_keys=False)[c].apply(lambda s: _hampel(s, window=win, n_sigmas=hampel_sigmas))
        else:
            df_prep[c] = _hampel(df_prep[c], window=win, n_sigmas=hampel_sigmas)

    # Imputación + flags
    for c in target_cols:
        flag = f"{c}_imputed"
        if station_col and station_col in df_prep.columns:
            def _imp(g: pd.DataFrame) -> pd.DataFrame:
                orig_na = g[c].isna()
                g[c] = _impute_series(g[c], period_hint_hours=hours, small_gap=small_gap, method=impute_method)
                g[flag] = orig_na & g[c].notna()
                return g
            df_prep = df_prep.groupby(station_col, group_keys=False).apply(_imp)
        else:
            orig_na = df_prep[c].isna()
            df_prep[c] = _impute_series(df_prep[c], period_hint_hours=hours, small_gap=small_gap, method=impute_method)
            df_prep[flag] = orig_na & df_prep[c].notna()

    return df_prep, target_cols


__all__ = [
    "load_and_basic_profile",
    "prepare_clean_impute",
    "sanitize_colnames",
    "infer_date_and_station_cols",
    "infer_freq_from_index",
]
