"""
Tests de scripts/forward_testing/ft_scoring_estructura.py (insumos de FT_SMC_v3).

Lo que importa fijar:
  - el ALIASING de la ventana: las columnas de N se sirven con los nombres _10 que
    espera ft_scoring.calcular_score_estructura. Un error de renombre aca no falla:
    devuelve otra columna, o None, y el bot decide con otro numero. Paso una vez en
    el loader del backtest (renombrar _5 -> _10 borraba las _5 que usa COMBO);
  - la ventana pedida tiene que estar PERSISTIDA en features_estructura: pedir N=7
    tiene que explotar, no devolver vacio;
  - el score se calcula con el MISMO codigo que la v1 sobre las claves que el
    loader promete (CLAVES_SCORE);
  - el trailing SL solo sube.

Sin DB: se testean los builders de SQL y las funciones puras.
"""

import re

import pytest

from scripts.forward_testing import ft_scoring_estructura as fse
from scripts.forward_testing.ft_scoring import calcular_score_estructura
from src.indicators.estructura import VENTANAS_TABLA

# Columnas de estructura que el score y la gestion de posiciones leen con nombre _10.
_CON_ALIAS = [
    ("estructura_10", "estructura_{n}"),
    ("choch_bear_10", "choch_bear_{n}"),
    ("dist_sl_10_pct", "dist_sl_{n}_pct"),
    ("dist_sh_10_pct", "dist_sh_{n}_pct"),
]


def _alias(sql: str) -> dict:
    """{nombre expuesto: columna real} a partir de los 'X AS Y' del SQL."""
    return {expuesto: real for real, expuesto in re.findall(r"(\w+)\s+AS\s+(\w+)", sql)}


@pytest.mark.parametrize("n", VENTANAS_TABLA)
def test_features_hoy_aliasa_la_ventana(n):
    sql = fse.sql_features_hoy(n)
    al = _alias(sql)
    for expuesto, patron in _CON_ALIAS:
        assert al.get(expuesto) == patron.format(n=n), (n, expuesto, al.get(expuesto))
    # El evento del lookback tambien sale de la ventana pedida.
    assert f"choch_bull_{n}" in sql and f"bos_bull_{n}" in sql


@pytest.mark.parametrize("n", VENTANAS_TABLA)
def test_estructura_tickers_aliasa_la_ventana(n):
    al = _alias(fse.sql_estructura_tickers(n))
    for expuesto, patron in _CON_ALIAS:
        assert al.get(expuesto) == patron.format(n=n), (n, expuesto, al.get(expuesto))
    assert al.get("bos_bear_10") == f"bos_bear_{n}"


@pytest.mark.parametrize("n", VENTANAS_TABLA)
def test_ninguna_columna_de_otra_ventana(n):
    """No se filtra una ventana distinta a la pedida (el bug que busca este test)."""
    sql = fse.sql_features_hoy(n) + fse.sql_estructura_tickers(n)
    sufijos = re.findall(r"es\.\w+?_(\d+)(?:_pct)?\b", sql)
    assert set(sufijos) == {str(n)}, set(sufijos)


def test_lookback_anclado_al_dato_no_al_reloj():
    sql = fse.sql_features_hoy(5)
    assert "CURRENT_DATE" not in sql
    assert "MAX(fecha)" in sql and "u.fecha - INTERVAL" in sql


def test_ventana_no_persistida_explota():
    with pytest.raises(ValueError, match="no esta en features_estructura"):
        fse._validar(7)
    for n in VENTANAS_TABLA:
        assert fse._validar(n) == n


def test_claves_del_contrato_alcanzan_para_el_score():
    """Una fila armada SOLO con CLAVES_SCORE tiene que dar un score valido."""
    fila = {
        "tuvo_choch_bull": 1, "tuvo_bos_bull": 0,
        "estructura_10": 1, "choch_bear_10": 0,
        "dist_sl_10_pct": 3.0, "dist_sh_10_pct": -2.0,
        "es_alcista": 1, "vol_spike": 1,
        "patron_engulfing_bull": 0, "patron_hammer": 0,
        "close": 100.0,
    }
    assert set(fila) == set(fse.CLAVES_SCORE)
    score, detalle = calcular_score_estructura(fila)
    assert score == 3.0                      # CHoCH + confirmacion + estructura solida
    assert detalle["swing_low"] == pytest.approx(100.0 / 1.03, rel=1e-6)

    # Un filtro obligatorio que no cumple devuelve -1 (no abre).
    assert calcular_score_estructura({**fila, "es_alcista": 0})[0] == -1.0
    assert calcular_score_estructura({**fila, "estructura_10": -1})[0] == -1.0
    assert calcular_score_estructura({**fila, "dist_sl_10_pct": 12.0})[0] == -1.0


def test_trailing_sl_solo_sube(monkeypatch):
    datos = {
        "SUBE":  {"ticker": "SUBE",  "close": 110.0, "dist_sl_10_pct": 5.0},
        "BAJA":  {"ticker": "BAJA",  "close": 100.0, "dist_sl_10_pct": 5.0},
        "SINSL": {"ticker": "SINSL", "close": 100.0, "dist_sl_10_pct": 0.0},
    }
    monkeypatch.setattr(fse, "obtener_estructura_tickers",
                        lambda tickers, ventana: datos)

    posiciones = [
        {"id": 1, "ticker": "SUBE",  "stop_loss": 100.0},   # nuevo 104,76 -> sube
        {"id": 2, "ticker": "BAJA",  "stop_loss": 99.0},    # nuevo 95,24  -> no baja
        {"id": 3, "ticker": "SINSL", "stop_loss": 90.0},    # sin swing -> no toca
        {"id": 4, "ticker": "OTRO",  "stop_loss": 90.0},    # sin dato -> no toca
    ]
    ups = fse.calcular_actualizaciones_sl(posiciones, 5)
    assert [u["id"] for u in ups] == [1]
    assert ups[0]["nuevo_sl"] == pytest.approx(110.0 / 1.05, abs=1e-4)
    assert ups[0]["nuevo_sl"] > 100.0

    assert fse.calcular_actualizaciones_sl([], 5) == []
