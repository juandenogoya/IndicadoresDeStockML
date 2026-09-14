"""
Tests de la lectura de features_sector en src/pipeline/feature_calculator.py.

BUG (13/9/2026): el scanner tomaba la ULTIMA fila de features_sector sin mirar
la fecha, y la tabla no se actualizaba a diario. El modelo ML recibio durante
meses 11 de sus 53 features de semanas atras, sin un solo error. Estos tests
fijan el contrato nuevo: se lee la fila de la rueda de la ultima barra, y si
no esta, NaN (el imputer del modelo lo cubre) -- nunca un valor de otra rueda.
"""

import inspect
from datetime import date

import numpy as np
import pandas as pd

from src.pipeline import feature_calculator as fc
from src.utils.contexto_sectorial import FEATURES_SECTORIALES


def _query_con(respuesta, llamadas):
    def query_df(sql, params=None):
        llamadas.append((sql, params))
        return respuesta.copy()
    return query_df


def test_lee_la_fila_de_la_rueda_pedida(monkeypatch):
    llamadas = []
    fila = pd.DataFrame([{c: 1.5 for c in FEATURES_SECTORIALES}])
    monkeypatch.setattr(fc, "query_df", _query_con(fila, llamadas))

    out = fc._obtener_zscore_sectorial("AAPL", date(2026, 9, 11))

    sql, params = llamadas[0]
    assert params == {"ticker": "AAPL", "fecha": date(2026, 9, 11)}
    assert "fecha = :fecha" in sql
    # La forma del bug: "la ultima que haya".
    assert "ORDER BY" not in sql.upper() and "LIMIT" not in sql.upper()
    assert out == {c: 1.5 for c in FEATURES_SECTORIALES}


def test_sin_fila_de_esa_rueda_devuelve_nan(monkeypatch):
    monkeypatch.setattr(fc, "query_df", _query_con(pd.DataFrame(), []))
    out = fc._obtener_zscore_sectorial("AAPL", date(2026, 9, 11))
    assert set(out) == set(FEATURES_SECTORIALES)
    assert all(np.isnan(v) for v in out.values())


def test_la_fecha_es_obligatoria():
    p = inspect.signature(fc._obtener_zscore_sectorial).parameters["fecha"]
    assert p.default is inspect.Parameter.empty


def _ohlcv(n=320):
    rng = np.random.default_rng(11)
    fechas = pd.date_range("2025-06-02", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    abre = close * (1 + rng.normal(0, 0.004, n))
    return pd.DataFrame({
        "fecha": fechas, "ticker": "AAA",
        "open": abre, "close": close,
        "high": np.maximum(abre, close) * (1 + rng.uniform(0, 0.01, n)),
        "low": np.minimum(abre, close) * (1 - rng.uniform(0, 0.01, n)),
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


def test_calcular_features_pide_la_rueda_de_la_ultima_barra(monkeypatch):
    llamadas = []
    fila = pd.DataFrame([{c: 0.25 for c in FEATURES_SECTORIALES}])
    monkeypatch.setattr(fc, "query_df", _query_con(fila, llamadas))
    df = _ohlcv()

    calc = fc.calcular_features_completas(df, "AAA", "Technology")

    assert calc["ok"], calc.get("error")
    assert llamadas[0][1]["fecha"] == calc["meta"]["precio_fecha"]
    assert calc["meta"]["precio_fecha"] == df["fecha"].iloc[-1].date()
    assert calc["meta"]["sector_rueda_ok"] is True
    assert all(calc["features_v3"][c] == 0.25 for c in FEATURES_SECTORIALES)


def test_calcular_features_sin_la_rueda_marca_y_deja_nan(monkeypatch):
    monkeypatch.setattr(fc, "query_df", _query_con(pd.DataFrame(), []))
    calc = fc.calcular_features_completas(_ohlcv(), "AAA", "Technology")

    assert calc["ok"], calc.get("error")
    assert calc["meta"]["sector_rueda_ok"] is False
    assert all(np.isnan(calc["features_v3"][c]) for c in FEATURES_SECTORIALES)
