"""
Tests del modo incremental de procesar_scoring_ticker (13/9/2026).

scoring_tecnico es insumo de features_sector (pct_long_sector) y el Paso 2
diario lo recalcula solo para las ultimas ruedas. El calculo es fila a fila,
asi que esas ruedas tienen que dar exactamente lo mismo que la corrida completa.
"""

import numpy as np
import pandas as pd
import pytest

import src.data.database as database
from src.scoring import rule_based as rb


@pytest.fixture
def db_falsa(monkeypatch):
    rng = np.random.default_rng(3)
    fechas = pd.date_range("2026-08-03", periods=30, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, len(fechas)))
    ind = pd.DataFrame({
        "fecha": fechas, "rsi14": rng.uniform(20, 80, len(fechas)),
        "macd_hist": rng.normal(0, 1, len(fechas)),
        "sma21": close + rng.normal(0, 1, len(fechas)),
        "sma50": close + rng.normal(0, 2, len(fechas)),
        "sma200": close + rng.normal(0, 3, len(fechas)),
        "momentum": rng.normal(0, 1, len(fechas)),
    })
    pre = pd.DataFrame({"fecha": fechas, "close": close})

    def query_df(sql, params=None):
        return (ind if "indicadores_tecnicos" in sql else pre).copy()

    escritos = []
    monkeypatch.setattr(rb, "query_df", query_df)
    monkeypatch.setattr(database, "upsert_scoring", lambda d: escritos.append(d.copy()))
    return fechas, escritos


def test_desde_da_lo_mismo_que_la_corrida_completa(db_falsa):
    fechas, _ = db_falsa
    completo = rb.procesar_scoring_ticker("AAA", guardar_db=False, verbose=False)
    desde = fechas[20].date()
    inc = rb.procesar_scoring_ticker("AAA", guardar_db=False, desde=desde, verbose=False)

    esperado = completo[completo["fecha"] >= pd.Timestamp(desde)].reset_index(drop=True)
    assert len(inc) == 10
    pd.testing.assert_frame_equal(inc, esperado)


def test_desde_persiste_solo_las_ruedas_recalculadas(db_falsa):
    fechas, escritos = db_falsa
    rb.procesar_scoring_ticker("AAA", guardar_db=True, desde=fechas[25].date(), verbose=False)
    assert len(escritos) == 1
    assert escritos[0]["fecha"].min() == fechas[25]


def test_sin_sesiones_desde_no_escribe(db_falsa):
    fechas, escritos = db_falsa
    out = rb.procesar_scoring_ticker("AAA", guardar_db=True,
                                     desde=(fechas[-1] + pd.Timedelta(days=7)).date())
    assert out.empty
    assert escritos == []


def test_verbose_false_no_imprime_la_linea_por_ticker(db_falsa, capsys):
    rb.procesar_scoring_ticker("AAA", guardar_db=False, verbose=False)
    assert "sesiones" not in capsys.readouterr().out
