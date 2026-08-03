"""
Tests del modulo puro src.utils.volatilidad_mtf (ATR% multi-timeframe).

Sin DB ni mocks: volatilidad_mtf solo usa pandas + ta. Cubre el resample OHLC
por calendario (W-FRI y M), la exclusion de la barra en curso, la agregacion
OHLC (open/high/low/close) y el ATR% por timeframe con sus casos borde.
"""

from datetime import date, timedelta

import pandas as pd

from src.utils import volatilidad_mtf


def _ohlc_diario(n_dias: int, inicio: date | None = None,
                 start: float = 100.0, step: float = 1.0) -> list[dict]:
    """n_dias de barras OHLC (solo dias habiles L-V), close creciente."""
    if inicio is None:
        inicio = date.today() - timedelta(days=int(n_dias * 1.5) + 10)
    filas = []
    d = inicio
    close = start
    while len(filas) < n_dias:
        if d.weekday() < 5:  # L-V
            o = close
            close = round(close + step, 2)
            c = close
            filas.append({
                "fecha": d, "open": o,
                "high": max(o, c) + 0.5, "low": min(o, c) - 0.5, "close": c,
            })
        d += timedelta(days=1)
    return filas


# ── resample_ohlc: agregacion ──────────────────────────────────────────────

def test_resample_agrega_ohlc_una_semana():
    """Una semana Lun-Vie -> una barra: open=Lun, high=max, low=min, close=Vie."""
    # 2024-01-01 es lunes; toda la semana cae en el mismo periodo W-FRI.
    filas = [
        {"fecha": date(2024, 1, 1), "open": 10, "high": 12, "low": 9,  "close": 11},
        {"fecha": date(2024, 1, 2), "open": 11, "high": 13, "low": 10, "close": 12},
        {"fecha": date(2024, 1, 3), "open": 12, "high": 14, "low": 11, "close": 13},
        {"fecha": date(2024, 1, 4), "open": 13, "high": 15, "low": 12, "close": 14},
        {"fecha": date(2024, 1, 5), "open": 14, "high": 16, "low": 13, "close": 15},
    ]
    b = volatilidad_mtf.resample_ohlc(pd.DataFrame(filas), "W-FRI")
    assert len(b) == 1
    row = b.iloc[0]
    assert row["open"] == 10.0
    assert row["high"] == 16.0
    assert row["low"] == 9.0
    assert row["close"] == 15.0
    assert isinstance(row["fecha"], date)


def test_resample_semana_corta_por_feriado_es_una_barra():
    """4 dias (feriado el lunes) siguen siendo UNA barra semanal valida."""
    filas = [
        {"fecha": date(2024, 1, 2), "open": 11, "high": 13, "low": 10, "close": 12},
        {"fecha": date(2024, 1, 3), "open": 12, "high": 14, "low": 11, "close": 13},
        {"fecha": date(2024, 1, 4), "open": 13, "high": 15, "low": 12, "close": 14},
        {"fecha": date(2024, 1, 5), "open": 14, "high": 16, "low": 13, "close": 15},
    ]
    b = volatilidad_mtf.resample_ohlc(pd.DataFrame(filas), "W-FRI")
    assert len(b) == 1
    assert b.iloc[0]["open"] == 11.0 and b.iloc[0]["close"] == 15.0


# ── resample_ohlc: exclusion de la barra en curso ──────────────────────────

def test_resample_excluye_mes_en_curso():
    hoy = date.today()
    filas = _ohlc_diario(120)
    filas.append({"fecha": hoy, "open": 1, "high": 1, "low": 1, "close": 1})
    b = volatilidad_mtf.resample_ohlc(pd.DataFrame(filas), "M")
    periodo_actual = pd.Timestamp(hoy).to_period("M")
    assert all(pd.Timestamp(f).to_period("M") < periodo_actual for f in b["fecha"])


def test_resample_excluye_semana_en_curso():
    hoy = date.today()
    lunes_actual = hoy - timedelta(days=hoy.weekday())
    filas = _ohlc_diario(60)
    filas.append({"fecha": lunes_actual, "open": 1, "high": 1, "low": 1, "close": 1})
    b = volatilidad_mtf.resample_ohlc(pd.DataFrame(filas), "W-FRI")
    assert all(f < lunes_actual for f in b["fecha"])


def test_resample_vacio():
    cols = ["fecha", "open", "high", "low", "close"]
    assert volatilidad_mtf.resample_ohlc(pd.DataFrame(columns=cols), "W-FRI").empty
    assert volatilidad_mtf.resample_ohlc(None, "M").empty


# ── atr_pct ────────────────────────────────────────────────────────────────

def test_atr_pct_historia_insuficiente():
    df = pd.DataFrame(_ohlc_diario(10))
    assert volatilidad_mtf.atr_pct(df, 14) is None


def test_atr_pct_positivo_con_historia():
    df = pd.DataFrame(_ohlc_diario(40))
    v = volatilidad_mtf.atr_pct(df, 14)
    assert v is not None and v > 0


# ── volatilidad_multi_tf ───────────────────────────────────────────────────

def test_multi_tf_todos_los_timeframes():
    """Con ~1.5 anios de historia, los tres TF devuelven ATR% > 0."""
    out = volatilidad_mtf.volatilidad_multi_tf(pd.DataFrame(_ohlc_diario(400)))
    assert set(out) == {"atr_pct_d", "atr_pct_w", "atr_pct_m"}
    for k in out:
        assert out[k] is not None and out[k] > 0, f"{k} deberia ser > 0"


def test_multi_tf_vacio():
    cols = ["fecha", "open", "high", "low", "close"]
    out = volatilidad_mtf.volatilidad_multi_tf(pd.DataFrame(columns=cols))
    assert out == {"atr_pct_d": None, "atr_pct_w": None, "atr_pct_m": None}


def test_multi_tf_mensual_none_si_falta_historia():
    """Pocas semanas -> diario/semanal pueden salir, mensual no (< 7 barras)."""
    out = volatilidad_mtf.volatilidad_multi_tf(pd.DataFrame(_ohlc_diario(30)))
    assert out["atr_pct_m"] is None
