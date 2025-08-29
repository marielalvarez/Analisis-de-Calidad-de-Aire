
import pandas as pd
from typing import Tuple, Dict, Optional

# -------------------------
# Helpers (shared)
# -------------------------

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]

def _coerce_numeric(df: pd.DataFrame, exclude=("date",)) -> pd.DataFrame:
    for c in df.columns:
        if c not in exclude:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _find_date_col_in_multiindex(df: pd.DataFrame):
    for col in df.columns:
        if isinstance(col, tuple):
            if any(isinstance(x, str) and x.strip().lower() == "date" for x in col):
                return col
        elif isinstance(col, str) and col.strip().lower() == "date":
            return col
    return df.columns[0]

def _normalize_date_and_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Common cleanup: strip, rename first col to 'date' if needed, cast date, drop NaT,
    replace 'NULL'/'null' with NA, dedup and numeric coercion; then sort by date.
    """
    df = _strip_cols(df)
    if isinstance(df.columns[0], str) and df.columns[0].strip().lower() != "date":
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df[~df["date"].isna()].reset_index(drop=True)
    # Replace common null strings
    df = df.replace(to_replace=r"^\s*(NULL|null|NaN|nan)\s*$", value=pd.NA, regex=True)
    df = _dedup_columns(df)
    df = _coerce_numeric(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def _maybe_drop_units_row(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a units row (e.g., 'ppm', 'ppb', 'ug/m^3') sitting right under headers.
    Heuristic: if first row's date is not parseable -> drop it.
    """
    c0 = df.columns[0]
    first_date = pd.to_datetime(df[c0].iloc[0], dayfirst=True, errors="coerce")
    if pd.isna(first_date):
        return df.iloc[1:].reset_index(drop=True)
    return df

def _extract_station_from_multi(raw_df: pd.DataFrame, station: str) -> pd.DataFrame:
    """Extract station subframe from a wide MultiIndex-like dataframe and attach 'date'."""
    raw = raw_df.copy()
    date_col = _find_date_col_in_multiindex(raw)
    ser_date = raw[date_col].squeeze()

    station_df: Optional[pd.DataFrame] = None

    # Case 1: pandas MultiIndex with top level == station
    if isinstance(raw.columns, pd.MultiIndex):
        if station in raw.columns.get_level_values(0):
            station_df = raw[station].copy()
        else:
            # Try columns where first element equals station
            cols = [c for c in raw.columns if isinstance(c, tuple) and c[0] == station]
            if cols:
                station_df = raw.loc[:, cols].copy()
    else:
        # Fallback: columns that are tuples but stored as Python tuples in object dtype
        cols = [c for c in raw.columns if isinstance(c, tuple) and c[0] == station]
        if cols:
            station_df = raw.loc[:, cols].copy()

    if station_df is None:
        raise KeyError("Estacion no encontrada: %s" % station)

    # Flatten columns to first level (variable names) if MultiIndex
    if hasattr(station_df.columns, "get_level_values"):
        try:
            station_df.columns = station_df.columns.get_level_values(0)
        except Exception:
            station_df.columns = [c[0] if isinstance(c, tuple) else c for c in station_df.columns]

    station_df.insert(0, "date", ser_date)
    station_df = _normalize_date_and_numeric(station_df)
    return station_df

# -------------------------
# Year-specific cleaners
# -------------------------

def _clean_2021(df_sheet_2021: pd.DataFrame) -> pd.DataFrame:
    """2021: each sheet is a station; first row are headers; no units row."""
    df = _normalize_date_and_numeric(df_sheet_2021)
    mask = (df["date"] >= "2021-01-01") & (df["date"] < "2022-01-01")
    return df.loc[mask].reset_index(drop=True)

def _clean_2022(df_sheet_2022: pd.DataFrame) -> pd.DataFrame:
    """2022: sheet looks like 2021; filter to 2022 only."""
    df = _normalize_date_and_numeric(df_sheet_2022)
    return df[df["date"].dt.year == 2022].reset_index(drop=True)

def _clean_2023(raw_2023_2024: pd.DataFrame, station: str) -> pd.DataFrame:
    """2023: extract station from wide MultiIndex df and filter 2023."""
    df = _extract_station_from_multi(raw_2023_2024, station)
    return df[df["date"].dt.year == 2023].reset_index(drop=True)

def _clean_2024(raw_2023_2024: pd.DataFrame, station: str) -> pd.DataFrame:
    """2024: extract station from wide MultiIndex df and filter 2024."""
    df = _extract_station_from_multi(raw_2023_2024, station)
    return df[df["date"].dt.year == 2024].reset_index(drop=True)

def _clean_2025(df_sheet_2025: pd.DataFrame) -> pd.DataFrame:
    """2025: sheet per station; extra units row under header -> drop it; filter 2025."""
    df = _maybe_drop_units_row(df_sheet_2025)
    df = _normalize_date_and_numeric(df)
    return df[df["date"].dt.year == 2025].reset_index(drop=True)

# -------------------------
# Public API
# -------------------------

def station_to_5dfs(
    df_2021_sheet: pd.DataFrame,
    df_2022_sheet: pd.DataFrame,
    raw_2023_2024: pd.DataFrame,
    df_2025_sheet: pd.DataFrame,
    station: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return 5 cleaned DataFrames for the given station (2021..2025)."""
    df21 = _clean_2021(df_2021_sheet)
    df22 = _clean_2022(df_2022_sheet)
    df23 = _clean_2023(raw_2023_2024, station)
    df24 = _clean_2024(raw_2023_2024, station)
    df25 = _clean_2025(df_2025_sheet)
    return df21, df22, df23, df24, df25

def station_to_5dfs_dict(
    df_2021_sheet: pd.DataFrame,
    df_2022_sheet: pd.DataFrame,
    raw_2023_2024: pd.DataFrame,
    df_2025_sheet: pd.DataFrame,
    station: str,
) -> Dict[int, pd.DataFrame]:
    """Same as station_to_5dfs but returns a dict {2021: df, ..., 2025: df}."""
    df21, df22, df23, df24, df25 = station_to_5dfs(
        df_2021_sheet, df_2022_sheet, raw_2023_2024, df_2025_sheet, station
    )
    return {2021: df21, 2022: df22, 2023: df23, 2024: df24, 2025: df25}
