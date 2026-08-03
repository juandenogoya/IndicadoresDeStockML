"""
perfil_metricas.py
Metricas de comportamiento por ticker para el perfilado de carteras.

Orquesta modulos PUROS:
    - volatilidad_mtf: ATR% Diario / Semanal / Mensual.
    - ft_metricas:     beta (vs benchmark) y max_drawdown.

PURO: recibe DataFrames de precios (ticker + benchmark) y devuelve un dict.
SIN config, SIN database, SIN side effects. La lectura de la DB, la eleccion del
benchmark (futuros ES) y la persistencia son responsabilidad del script de
Fase 3 (scripts/compute_perfiles_carteras.py).

Diseno: docs/perfiles_carteras.md (secciones 5 y 13, Fase 1).
"""

import pandas as pd

from src.utils import volatilidad_mtf
from src.utils import ft_metricas

# Dias habiles ~ 1 anio (homogeneo con ft_metricas.DIAS_HABILES_ANIO).
VENTANA_1A = ft_metricas.DIAS_HABILES_ANIO  # 252


def _retornos_alineados(df_ticker: pd.DataFrame, df_bench: pd.DataFrame):
    """
    Retornos diarios del ticker y del benchmark ALINEADOS por fecha (inner
    join): la beta necesita ambas series en las mismas ruedas. Devuelve
    (r_ticker, r_bench) como listas del mismo largo, o ([], []) si no hay solape.
    """
    t = df_ticker[["fecha", "close"]].rename(columns={"close": "c_t"}).copy()
    b = df_bench[["fecha", "close"]].rename(columns={"close": "c_b"}).copy()
    t["fecha"] = pd.to_datetime(t["fecha"])
    b["fecha"] = pd.to_datetime(b["fecha"])
    m = pd.merge(t, b, on="fecha", how="inner").sort_values("fecha")
    if len(m) < 2:
        return [], []
    m["r_t"] = m["c_t"].astype(float).pct_change()
    m["r_b"] = m["c_b"].astype(float).pct_change()
    m = m.dropna(subset=["r_t", "r_b"])
    return m["r_t"].tolist(), m["r_b"].tolist()


def beta_ticker(df_ticker: pd.DataFrame, df_bench: pd.DataFrame,
                ventana: int = VENTANA_1A):
    """
    Beta del ticker vs benchmark sobre la ULTIMA `ventana` de retornos diarios
    alineados. None si no hay solape suficiente.
    """
    r_t, r_b = _retornos_alineados(df_ticker, df_bench)
    if len(r_t) < 2:
        return None
    return ft_metricas.beta(r_t[-ventana:], r_b[-ventana:])


def _max_dd_pct(closes):
    """max_drawdown de ft_metricas -> solo el % (o None)."""
    dd = ft_metricas.max_drawdown(closes)
    return dd["max_dd_pct"] if dd else None


def metricas_ticker(df_ticker: pd.DataFrame, df_bench: pd.DataFrame,
                    period_d: int = volatilidad_mtf.ATR_PERIOD_D,
                    period_w: int = volatilidad_mtf.ATR_PERIOD_W,
                    period_m: int = volatilidad_mtf.ATR_PERIOD_M,
                    ventana_beta: int = VENTANA_1A,
                    ventana_dd: int = VENTANA_1A) -> dict:
    """
    Metricas de comportamiento de un ticker para el perfilado.

    Args:
        df_ticker: OHLC diario del ticker [fecha, open, high, low, close].
        df_bench:  precios diarios del benchmark [fecha, close] (futuros ES).
                   Si es None, la beta queda en None.
        period_*:  ventanas del ATR por timeframe.
        ventana_beta: dias de retornos para la beta (default 1 anio).
        ventana_dd:   dias para el drawdown reciente (default 1 anio).

    Returns:
        dict con:
            atr_pct_d / atr_pct_w / atr_pct_m  volatilidad multi-TF (%)
            beta                                sensibilidad al benchmark
            max_dd_1a                           peor caida del ultimo anio (%)
            max_dd_hist                         peor caida de toda la historia (%)
        Cualquier clave es None si no hay historia suficiente.
    """
    vol = volatilidad_mtf.volatilidad_multi_tf(
        df_ticker, period_d=period_d, period_w=period_w, period_m=period_m
    )

    df = df_ticker.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    closes = df.sort_values("fecha")["close"].astype(float).tolist()

    beta_val = (beta_ticker(df_ticker, df_bench, ventana=ventana_beta)
                if df_bench is not None else None)

    return {
        "atr_pct_d":   vol["atr_pct_d"],
        "atr_pct_w":   vol["atr_pct_w"],
        "atr_pct_m":   vol["atr_pct_m"],
        "beta":        beta_val,
        "max_dd_1a":   _max_dd_pct(closes[-ventana_dd:]) if len(closes) > 1 else None,
        "max_dd_hist": _max_dd_pct(closes) if len(closes) > 1 else None,
    }
