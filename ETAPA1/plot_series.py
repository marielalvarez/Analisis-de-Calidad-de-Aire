# ¿cómo usarlo?
# from plot_series import faltantes, plotserie
#
# 1) Faltantes:
#    por_col, total = faltantes(df)
#
# 2) Graficar recibe:
#    # df cuya primera columna es la fecha 'YYYY-MM-DD HH:MM:SS'
#    # string que representa la columna del nombre del contaminante, p.ej. "PM10"
#    # escala: "mes" -> te va a dar 12 subplots (enero...diciembre) del año del df
#    #         "anno" -> 1 gráfico con toda la serie del año del df
#    fig, axes = plotserie(df, columna="Contaminante 1", escala="mes")   # o escala="anno"
#
# Nota:
# - Si el df contuviera más de un año, se tomará el menor y se avisará por consola.
# - Devuelve (fig, axes). Para "mes" axes es una matriz 3x4; para "anno" es un solo Axes.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

def faltantes(df: pd.DataFrame):
    """
    USO:
        por_col, total = faltantes(df)

    Devuelve:
        - por_col: Series con conteo de NaN por columna (desc)
        - total:   Entero con NaN totales en el DataFrame
    """
    por_col = df.isna().sum().sort_values(ascending=False)
    total = int(por_col.sum())
    print(f"Faltantes totales en el DataFrame: {total}")
    return por_col, total


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que la primera columna sea datetime y sea el índice (ordenado)."""
    _df = df.copy()
    if not isinstance(_df.index, pd.DatetimeIndex):
        fecha_col = _df.columns[0]
        _df[fecha_col] = pd.to_datetime(_df[fecha_col], errors="coerce")
        _df = _df.set_index(fecha_col)
    return _df.sort_index()

def plotserie(df: pd.DataFrame, columna: str, escala: str = "mes"):
    """
    USO:
        fig, axes = plotserie(df, columna="Contaminante 1", escala="mes")  # o "anno"

    Parámetros:
        - df: DataFrame con la PRIMERA columna como fecha 'YYYY-MM-DD HH:MM:SS'.
        - columna: nombre de la columna a graficar.
        - escala: "mes" -> 12 subplots (enero...diciembre) del año del df
                  "anno"/"año"/"anio" -> 1 gráfico con toda la serie del año.

    Devuelve:
        - (fig, axes): Figure y Axes (matriz 3x4 si "mes"; Axes único si "anno").
    """
    _df = _ensure_datetime_index(df)

    if columna not in _df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    s = pd.to_numeric(_df[columna], errors="coerce")

    years = s.index.year.unique()
    if len(years) == 0:
        raise ValueError("No hay fechas válidas en el índice.")
    if len(years) > 1:
        y = int(years.min())
        print(f"Múltiples años detectados ({list(map(int, years))}). Se graficará el {y}.")
        s = s[s.index.year == y]
    year = int(s.index.year.min())

    esc = (escala or "").strip().lower()
    if esc in {"mes", "mensual", "m"}:
        fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True, constrained_layout=True)
        axes = axes.ravel()

        for m in range(1, 13):
            ax = axes[m - 1]
            sm = s[s.index.month == m]
            ax.grid(True, alpha=0.3)
            ax.set_title(pd.Timestamp(year=year, month=m, day=1).strftime("%b %Y"))

            if sm.empty:
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                continue

            ax.plot(sm.index, sm.values, linewidth=1.2)
            locator = AutoDateLocator()
            formatter = AutoDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        fig.suptitle(f"{columna} — Serie por mes ({year})", fontsize=14)
        return fig, axes.reshape(3, 4)

    elif esc in {"anno", "año", "anio", "anual", "y", "year"}:
        fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
        ax.grid(True, alpha=0.3)
        ax.plot(s.index, s.values, linewidth=1.5)
        ax.set_title(f"{columna} — Serie anual {year}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Concentración")

        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        return fig, ax

    else:
        raise ValueError("Parámetro 'escala' debe ser 'mes' o 'anno' (acepta 'año'/'anio').")

