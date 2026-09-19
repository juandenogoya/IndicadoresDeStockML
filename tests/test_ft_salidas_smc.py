"""Tests de src/utils/ft_salidas_smc.py (regla de salida de FT_SMC_v1 y grilla)."""

import os
import re

import numpy as np
import pytest

from src.utils import ft_salidas_smc as fsm
from src.utils.ft_salidas_smc import Regla

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAN = float("nan")


def serie(close, dist10=None, dist5=None, choch=None, estr=None, datos=None, balance=None, dia=None):
    n = len(close)
    return dict(
        dia=dia if dia is not None else list(range(n)),
        close=close,
        dist10=dist10 if dist10 is not None else [5.0] * n,
        dist5=dist5 if dist5 is not None else [3.0] * n,
        choch10=choch if choch is not None else [0] * n,
        estr10=estr if estr is not None else [0] * n,
        datos=datos if datos is not None else [True] * n,
        balance=balance if balance is not None else [False] * n,
    )


def salida(s, regla, i=0):
    return fsm.primera_salida(s["dia"], s["close"], s["dist10"], s["dist5"], s["choch10"],
                              s["estr10"], s["datos"], s["balance"], i, regla)


# --- guardas contra el bot ----------------------------------------------------------

def _src(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def test_guarda_parametros_del_bot():
    bot = _src("scripts/forward_testing/ft_bot_smc.py")
    assert re.search(r"^DIAS_MAX_POS\s*=\s*20\b", bot, re.M)
    assert re.search(r"^MAX_POSICIONES\s*=\s*5\b", bot, re.M)
    assert fsm.DIAS_TIME_STOP_V1 == 20 and fsm.MAX_POSICIONES_V1 == 5


def test_guarda_orden_de_prioridades_del_bot():
    bot = _src("scripts/forward_testing/ft_bot_smc.py")
    cuerpo = bot[bot.index("def evaluar_cierres"):bot.index("def procesar_trailing_sl")]
    pos = [cuerpo.index(m) for m in ('"EARNINGS_MANANA"', '"TRAILING_SL"', '"CHOCH_BEAR"',
                                     '"ESTRUCTURA_ROTA"', "TIME_STOP_")]
    assert pos == sorted(pos)
    # el trailing se actualiza ANTES de evaluar cierres
    run = bot[bot.index("def run("):]
    assert run.index("procesar_trailing_sl(") < run.index("evaluar_cierres(")


def test_swing_low_es_la_formula_del_bot():
    sc = _src("scripts/forward_testing/ft_scoring.py")
    assert "close / (1.0 + dist_sl_pct / 100.0)" in sc
    assert fsm.swing_low(110.0, 10.0) == pytest.approx(100.0)
    assert fsm.swing_low(100.0, 0.0) == 0.0
    assert fsm.swing_low(100.0, -2.0) == 0.0
    assert fsm.swing_low(100.0, NAN) == 0.0
    assert fsm.swing_low(NAN, 5.0) == 0.0


# --- la grilla ----------------------------------------------------------------------

def test_grilla_96_con_la_actual_primero():
    g = fsm.grilla()
    assert len(g) == 96 == len(set(g))
    assert g[0] == fsm.REGLA_ACTUAL == Regla("trail10", True, True, 20)
    assert {r.stop for r in g} == set(fsm.STOPS)
    assert {r.tiempo for r in g} == set(fsm.TIEMPOS)


def test_etiqueta():
    assert fsm.etiqueta(fsm.REGLA_ACTUAL) == "stop trail10 choch si estr si tiempo 20"
    assert fsm.etiqueta(Regla("sin", False, False, None)).endswith("tiempo sin")


def test_stop_inicial():
    assert fsm.stop_inicial(Regla("sin", True, True, 20), 110, 10, 5) is None
    assert fsm.stop_inicial(Regla("trail10", True, True, 20), 110, 10, 5) == pytest.approx(100)
    assert fsm.stop_inicial(Regla("fijo", True, True, 20), 110, 10, 5) == pytest.approx(100)
    assert fsm.stop_inicial(Regla("trail5", True, True, 20), 105, 10, 5) == pytest.approx(100)
    # trail5 sin swing low de 5 barras arranca en el de 10
    assert fsm.stop_inicial(Regla("trail5", True, True, 20), 110, 10, 0) == pytest.approx(100)


# --- la regla, rueda por rueda --------------------------------------------------------

def test_balance_va_primero_aunque_toque_el_stop():
    s = serie([110, 90, 90], balance=[False, True, False])
    assert salida(s, fsm.REGLA_ACTUAL)[:2] == (1, "BALANCE")


def test_balance_sale_aunque_no_haya_fila_de_estructura():
    s = serie([110, 90, 90], datos=[True, False, True], balance=[False, True, False])
    assert salida(s, fsm.REGLA_ACTUAL)[:2] == (1, "BALANCE")


def test_sin_fila_de_estructura_solo_mira_el_balance():
    s = serie([110, 90, 90], datos=[True, False, True])
    assert salida(s, fsm.REGLA_ACTUAL)[:2] == (2, "STOP")


def test_close_faltante_no_evalua_nada():
    s = serie([110, NAN, 90], balance=[False, True, False])
    assert salida(s, fsm.REGLA_ACTUAL)[:2] == (2, "STOP")


def test_stop_cierra_con_close_menor_o_igual():
    sl = fsm.swing_low(110, 10)                        # ~100
    s = serie([110, sl + 0.01, sl], dist10=[10, 10, 10])
    assert salida(s, Regla("fijo", False, False, None))[:2] == (2, "STOP")


def test_trailing_sube_antes_de_evaluar_y_nunca_baja():
    # dia 1: close 132, dist 10 -> stop sube a 120; dia 2: dist enorme -> no baja; dia 3 cae a 119
    s = serie([110, 132, 125, 119], dist10=[10, 10, 50, 10])
    j, mot, sl = salida(s, Regla("trail10", False, False, None))
    assert (j, mot) == (3, "STOP") and sl == pytest.approx(120)
    # con stop fijo la misma serie no sale
    assert salida(s, Regla("fijo", False, False, None))[:2] == (None, None)


def test_trailing_no_sube_sin_fila_de_estructura():
    s = serie([110, 132, 119], dist10=[10, 10, 10], datos=[True, False, True])
    assert salida(s, Regla("trail10", False, False, None))[:2] == (None, None)


def test_trail5_sigue_el_swing_low_de_5_barras():
    s = serie([105, 126, 119], dist10=[10, 50, 50], dist5=[5, 5, 5])
    assert salida(s, Regla("trail5", False, False, None))[:2] == (2, "STOP")    # stop 120
    assert salida(s, Regla("trail10", False, False, None))[:2] == (None, None)  # stop ~95


def test_choch_y_estructura_solo_si_estan_activas():
    s = serie([110, 108, 108], choch=[0, 1, 0], estr=[0, 0, -1])
    assert salida(s, Regla("sin", True, True, None))[:2] == (1, "CHOCH")
    assert salida(s, Regla("sin", False, True, None))[:2] == (2, "ESTRUCTURA")
    assert salida(s, Regla("sin", False, False, None))[:2] == (None, None)


def test_stop_va_antes_que_choch_la_misma_rueda():
    s = serie([110, 99], choch=[0, 1])
    assert salida(s, fsm.REGLA_ACTUAL)[:2] == (1, "STOP")


def test_time_stop_por_dias_corridos():
    s = serie([110, 111, 112, 113], dia=[0, 7, 19, 21])
    assert salida(s, Regla("sin", False, False, 20))[:2] == (3, "TIEMPO")
    assert salida(s, Regla("sin", False, False, 19))[:2] == (2, "TIEMPO")
    assert salida(s, Regla("sin", False, False, None))[:2] == (None, None)


def test_entrada_en_la_ultima_rueda_es_censura():
    s = serie([110, 111])
    assert salida(s, fsm.REGLA_ACTUAL, i=1)[:2] == (None, None)


# --- vectorizada = referencia ---------------------------------------------------------

def _aleatoria(rng, n=60):
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    close[rng.random(n) < 0.03] = np.nan
    return dict(
        dia=np.cumsum(rng.integers(1, 4, n)).tolist(),
        close=close.tolist(),
        dist10=np.where(rng.random(n) < 0.1, np.nan, rng.uniform(-2, 9, n)).tolist(),
        dist5=rng.uniform(-1, 6, n).tolist(),
        choch10=(rng.random(n) < 0.04).astype(int).tolist(),
        estr10=rng.choice([-1, 0, 1], n, p=[0.05, 0.6, 0.35]).tolist(),
        datos=(rng.random(n) > 0.05).tolist(),
        balance=(rng.random(n) < 0.02).tolist(),
    )


@pytest.mark.parametrize("semilla", range(40))
def test_vectorizada_igual_a_la_referencia(semilla):
    rng = np.random.default_rng(semilla)
    s = _aleatoria(rng)
    reglas = fsm.grilla()
    for i in (0, 5, 20):
        if not s["close"][i] == s["close"][i]:
            continue
        idx, mot = fsm.salidas_reglas(s["dia"], s["close"], s["dist10"], s["dist5"], s["choch10"],
                                      s["estr10"], s["datos"], s["balance"], i, reglas)
        for k, r in enumerate(reglas):
            j, m, _ = salida(s, r, i)
            assert idx[k] == (j if j is not None else -1), (semilla, i, r)
            assert mot[k] == (fsm.MOTIVOS.index(m) if m else -1), (semilla, i, r)


# --- entradas -------------------------------------------------------------------------

def test_entradas_por_ticker_una_posicion_a_la_vez():
    senal = [True, True, False, True, True, True, False]
    salidas = {0: 3, 3: 5, 5: None}
    assert fsm.entradas_por_ticker(senal, 0, salidas.get) == [0, 3, 5]   # reentra en 3 (salio en 3)


def test_entradas_por_ticker_desde():
    senal = [True, False, True, False]
    assert fsm.entradas_por_ticker(senal, 1, lambda i: i + 1) == [2]


def test_cartera_con_tope_prioriza_score_y_libera_cupo_al_salir():
    cand = {1: [(1, "B", "b1"), (3, "A", "a1"), (2, "C", "c1")],
            2: [(3, "D", "d2")],
            4: [(1, "E", "e4"), (1, "D", "d4")]}
    sal = {"a1": 4, "b1": None, "c1": 10, "d2": 9, "e4": None, "d4": None}
    ent = fsm.cartera_con_tope([1, 2, 3, 4], cand, sal.get, max_pos=2)
    # dia 1 entran A (3) y C (2); dia 2 no hay cupo; dia 4 sale A y entra D (desempate D < E)
    assert ent == ["a1", "c1", "d4"]


def test_cartera_con_tope_no_repite_ticker_abierto():
    cand = {1: [(1, "A", "a1")], 2: [(3, "A", "a2"), (1, "B", "b2")]}
    ent = fsm.cartera_con_tope([1, 2], cand, lambda c: None, max_pos=5)
    assert ent == ["a1", "b2"]


def test_familia_ft():
    assert fsm.familia_ft("EARNINGS_MANANA") == "BALANCE"
    assert fsm.familia_ft("TRAILING_SL") == "STOP"
    assert fsm.familia_ft("CHOCH_BEAR") == "CHOCH"
    assert fsm.familia_ft("ESTRUCTURA_ROTA") == "ESTRUCTURA"
    assert fsm.familia_ft("TIME_STOP_21D") == "TIEMPO"
