"""
Tests de src/indicators/estructura.py (swings confirmados, estructura, BOS/CHoCH).

Lo que importa fijar (docs/estructura_velas.md sec. 4):
  - INVARIANCIA: la fila de una fecha calculada con los datos hasta esa fecha es
    igual a la calculada con toda la historia. Es el test que le falto a
    market_structure.py, cuya historia mira N ruedas al futuro;
  - un swing existe recien N barras despues de su barra, y el evento se fecha ahi;
  - sin las N barras de la derecha no hay swing (nada provisional en la ultima barra);
  - la clasificacion de estructura y BOS/CHoCH es la del modulo viejo.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import estructura as est


def _df(closes, ancho=0.5, fechas=None):
    closes = np.asarray(closes, dtype=float)
    if fechas is None:
        fechas = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({
        "fecha": fechas,
        "open": closes,
        "high": closes + ancho,
        "low": closes - ancho,
        "close": closes,
    })


def _random_walk(n, semilla):
    rng = np.random.default_rng(semilla)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    alto = close * (1 + rng.uniform(0, 0.02, n))
    bajo = close * (1 - rng.uniform(0, 0.02, n))
    return pd.DataFrame({
        "fecha": pd.bdate_range("2023-01-02", periods=n),
        "open": close, "high": alto, "low": bajo, "close": close,
    })


def _iguales(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-9))


# Zigzag alcista (N=2): swing highs en 2, 6, 10 (confirmados en 4, 8, 12) y swing
# lows en 4, 8, 12 (confirmados en 6, 10, 14).
ZIGZAG_ALCISTA = [10, 11, 12, 11, 10, 11, 13, 12, 11, 12, 14, 13, 12, 13, 15, 16, 17]


# -- contrato -----------------------------------------------------------------------

def test_columnas_iguales_a_market_structure():
    from src.indicators.market_structure import FEATURE_COLS_MS
    assert est.COLUMNAS == FEATURE_COLS_MS


def test_no_modifica_la_entrada_y_ordena_por_fecha():
    df = _df(ZIGZAG_ALCISTA)
    desordenado = df.iloc[::-1].copy()
    copia = desordenado.copy()
    out = est.calcular_estructura(desordenado, ventanas=(2,))
    pd.testing.assert_frame_equal(desordenado, copia)
    assert list(out["fecha"]) == list(df["fecha"])


def test_conserva_ticker_si_viene():
    df = _df(ZIGZAG_ALCISTA)
    df["ticker"] = "AAA"
    out = est.calcular_estructura(df, ventanas=(2,))
    assert list(out.columns[:2]) == ["ticker", "fecha"]


# -- invariancia ----------------------------------------------------------------------

@pytest.mark.parametrize("semilla", [1, 7])
def test_invariancia_la_historia_es_lo_que_se_sabia_ese_dia(semilla):
    df = _random_walk(180, semilla)
    completo = est.calcular_estructura(df)
    for t in range(len(df)):
        parcial = est.calcular_estructura(df.iloc[: t + 1]).iloc[-1]
        for col in est.COLUMNAS:
            assert _iguales(parcial[col], completo.loc[t, col]), (t, col)


def test_invariancia_semanal_con_tope_semanal():
    df = _random_walk(120, 3)
    completo = est.calcular_estructura(df, tope_dias=est.TOPE_DIAS_SEMANAL)
    for t in range(0, len(df), 7):
        parcial = est.calcular_estructura(df.iloc[: t + 1],
                                          tope_dias=est.TOPE_DIAS_SEMANAL).iloc[-1]
        for col in est.COLUMNAS:
            assert _iguales(parcial[col], completo.loc[t, col]), (t, col)


# -- swings ---------------------------------------------------------------------------

def test_el_swing_se_confirma_n_barras_despues():
    highs = [1, 2, 3, 10, 3, 2, 1, 1, 1]
    df = _df(highs, ancho=0.0)
    out = est.calcular_estructura(df, ventanas=(2,))
    assert out["is_sh_2"].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert pd.isna(out.loc[4, "dist_sh_2_pct"])       # el dia 4 el swing no existe
    assert out.loc[5, "dist_sh_2_pct"] == pytest.approx((2 - 10) / 10 * 100)
    assert out.loc[5, "dias_sh_2"] == 2                # desde la barra del swing
    assert out.loc[6, "dias_sh_2"] == 3


def test_sin_barras_a_la_derecha_no_hay_swing():
    df = _df([1, 2, 3, 4, 5, 6, 7], ancho=0.0)
    out = est.calcular_estructura(df, ventanas=(2,))
    assert out["is_sh_2"].sum() == 0
    assert out["dist_sh_2_pct"].isna().all()


def test_maximos_iguales_dan_un_solo_swing():
    df = _df([1, 2, 5, 5, 2, 1, 1], ancho=0.0)
    sh, _ = est.swings_en_barra(df["high"], df["low"], 2)
    assert sh.tolist() == [False, False, True, False, False, False, False]


def test_tope_de_dias_diario_y_semanal():
    closes = [1, 2, 3, 4, 5, 20] + list(np.linspace(19, 1, 300))
    df = _df(closes, ancho=0.0)
    diario = est.calcular_estructura(df, ventanas=(5,))
    semanal = est.calcular_estructura(df, ventanas=(5,), tope_dias=est.TOPE_DIAS_SEMANAL)
    assert diario["dias_sh_5"].iloc[-1] == 252
    assert semanal["dias_sh_5"].iloc[-1] == 260


# -- estructura y BOS / CHoCH -----------------------------------------------------------

def test_estructura_alcista_y_bos_en_el_cruce():
    out = est.calcular_estructura(_df(ZIGZAG_ALCISTA), ventanas=(2,))
    assert out["is_sh_2"].to_numpy().nonzero()[0].tolist() == [4, 8, 12]
    assert out["is_sl_2"].to_numpy().nonzero()[0].tolist() == [6, 10, 14]
    # Con un solo swing low confirmado no hay estructura; desde t=10 hay HH y HL.
    assert out.loc[:9, "estructura_2"].eq(0).all()
    assert out.loc[10:, "estructura_2"].eq(1).all()
    # BOS alcista en el cruce del close sobre el swing high ya confirmado.
    assert out["bos_bull_2"].to_numpy().nonzero()[0].tolist() == [6, 10, 14]
    assert out[["choch_bull_2", "bos_bear_2", "choch_bear_2"]].to_numpy().sum() == 0


def test_espejo_bajista():
    espejo = [30 - c for c in ZIGZAG_ALCISTA]
    out = est.calcular_estructura(_df(espejo), ventanas=(2,))
    assert out.loc[10:, "estructura_2"].eq(-1).all()
    assert out["bos_bear_2"].to_numpy().nonzero()[0].tolist() == [6, 10, 14]
    assert out["bos_bull_2"].sum() == 0


def test_choch_alcista_rompe_una_estructura_bajista():
    closes = [30 - c for c in ZIGZAG_ALCISTA] + [16, 17, 19]
    out = est.calcular_estructura(_df(closes), ventanas=(2,))
    assert out.loc[18, "estructura_2"] == -1
    assert out.loc[19, "choch_bull_2"] == 1
    assert out.loc[19, "bos_bull_2"] == 0
    assert out["choch_bull_2"].sum() == 1


def test_un_swing_confirmado_por_encima_del_close_no_genera_evento_retroactivo():
    # Al confirmarse, el close siempre esta <= al swing high (es el maximo de las
    # N barras de la derecha): el primer cruce posible es posterior.
    df = _random_walk(250, 11)
    out = est.calcular_estructura(df, ventanas=(5,))
    confirmados = out.index[out["is_sh_5"] == 1]
    assert (out.loc[confirmados, "dist_sh_5_pct"] <= 1e-9).all()
