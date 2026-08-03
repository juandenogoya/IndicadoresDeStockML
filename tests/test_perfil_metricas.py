"""
Tests del modulo puro src.utils.perfil_metricas (metricas de comportamiento
por ticker para el perfilado de carteras).

Sin DB: orquesta volatilidad_mtf + ft_metricas, ambos puros. Cubre el alineado
de retornos para la beta, los casos borde y el dict de salida.
"""

from datetime import date, timedelta

import pandas as pd

from src.utils import perfil_metricas


def _ohlc_diario(n_dias: int, inicio: date | None = None,
                 start: float = 100.0, step: float = 1.0) -> list[dict]:
    if inicio is None:
        inicio = date.today() - timedelta(days=int(n_dias * 1.5) + 10)
    filas = []
    d = inicio
    close = start
    while len(filas) < n_dias:
        if d.weekday() < 5:
            o = close
            close = round(close + step, 2)
            c = close
            filas.append({
                "fecha": d, "open": o,
                "high": max(o, c) + 0.5, "low": min(o, c) - 0.5, "close": c,
            })
        d += timedelta(days=1)
    return filas


# ── beta ───────────────────────────────────────────────────────────────────

def test_beta_contra_si_mismo_es_uno():
    df = pd.DataFrame(_ohlc_diario(300))
    bench = df[["fecha", "close"]].copy()
    b = perfil_metricas.beta_ticker(df, bench)
    assert b is not None and abs(b - 1.0) < 1e-6


def test_beta_sin_solape_es_none():
    df = pd.DataFrame(_ohlc_diario(100, inicio=date(2020, 1, 6)))
    bench = pd.DataFrame(_ohlc_diario(100, inicio=date(2023, 1, 2)))[["fecha", "close"]]
    assert perfil_metricas.beta_ticker(df, bench) is None


# ── metricas_ticker ────────────────────────────────────────────────────────

def test_metricas_ticker_claves_y_tipos():
    df = pd.DataFrame(_ohlc_diario(400))
    bench = df[["fecha", "close"]].copy()
    m = perfil_metricas.metricas_ticker(df, bench)
    assert set(m) == {
        "atr_pct_d", "atr_pct_w", "atr_pct_m",
        "beta", "max_dd_1a", "max_dd_hist",
    }
    for k in ("atr_pct_d", "atr_pct_w", "atr_pct_m", "beta"):
        assert m[k] is not None


def test_metricas_serie_creciente_drawdown_cero():
    """Serie estrictamente creciente -> nunca cae -> max_dd = 0."""
    df = pd.DataFrame(_ohlc_diario(400))
    bench = df[["fecha", "close"]].copy()
    m = perfil_metricas.metricas_ticker(df, bench)
    assert m["max_dd_hist"] == 0.0
    assert m["max_dd_1a"] == 0.0


def test_metricas_con_caida_drawdown_positivo():
    """Serie que sube y despues cae -> max_dd_hist > 0."""
    filas = _ohlc_diario(200)
    # Forzar una caida al final: bajar el close de las ultimas 20 barras.
    for i in range(-20, 0):
        c = filas[i]["close"] - 40
        filas[i]["close"] = c
        filas[i]["low"] = c - 0.5
        filas[i]["high"] = c + 0.5
    df = pd.DataFrame(filas)
    bench = df[["fecha", "close"]].copy()
    m = perfil_metricas.metricas_ticker(df, bench)
    assert m["max_dd_hist"] is not None and m["max_dd_hist"] > 0


def test_metricas_sin_benchmark_beta_none():
    df = pd.DataFrame(_ohlc_diario(300))
    m = perfil_metricas.metricas_ticker(df, None)
    assert m["beta"] is None
    assert m["atr_pct_d"] is not None
