"""Tests de src/utils/ft_salidas.py (analisis de salidas de FT)."""

import math
import os
import re

import pytest

from src.strategies.scoring import calcular_score_tecnico
from src.utils import ft_salidas as fs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _cond(sma200=True, sma50=True, sma21=True, macd=True, rsi=True, rsi_valor=55.0):
    return {"sma200": sma200, "sma50": sma50, "sma21": sma21, "macd": macd,
            "rsi": rsi, "rsi_valor": rsi_valor}


# --- la regla es una regla de si/no --------------------------------------------

def test_valores_posibles_no_hay_nada_entre_3_5_y_4():
    v = fs.valores_posibles()
    assert v == [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.5]
    assert not [x for x in v if 3.5 < x < 4.0]


@pytest.mark.parametrize("c", fs.combinaciones())
def test_entrada_equivale_a_sma200_sma50_y_dos_de_tres(c):
    dos_de_tres = sum([c["sma21"], c["macd"], c["rsi"]]) >= 2
    assert fs.cumple_entrada(c) == (c["sma200"] and c["sma50"] and dos_de_tres)


@pytest.mark.parametrize("c", fs.combinaciones())
def test_salida_actual_es_la_entrada_negada(c):
    # sin histeresis: salir con <= 3,5 es exactamente dejar de cumplir la entrada
    assert fs.sale(c) == (not fs.cumple_entrada(c))


@pytest.mark.parametrize("c", fs.combinaciones())
def test_variantes_forma_booleana(c):
    base = c["sma200"] and c["sma50"]
    assert fs.sale(c, "sin_sma21") == (not (base and (c["macd"] or c["rsi"])))
    assert fs.sale(c, "sin_macd") == (not (base and (c["sma21"] or c["rsi"])))
    assert fs.sale(c, "sin_rsi") == (not (base and (c["sma21"] or c["macd"])))


@pytest.mark.parametrize("c", fs.combinaciones())
def test_sin_x_nunca_sale_mas_que_la_actual(c):
    for v in ("sin_sma21", "sin_macd", "sin_rsi"):
        if fs.sale(c, v):
            assert fs.sale(c)


def test_sin_sma21_no_sale_por_perder_solo_sma21():
    c = _cond(sma21=False, macd=True, rsi=False)       # SMA50 + MACD = 3,5
    assert fs.sale(c)                                   # la regla actual sale
    assert not fs.sale(c, "sin_sma21")                  # la variante no


def test_variante_desconocida():
    with pytest.raises(ValueError):
        fs.sale(_cond(), "sin_sma50")


def test_score_coincide_con_scoring():
    fila = {"close": 110, "sma21": 105, "sma50": 100, "sma200": 90,
            "rsi14": 70, "macd": 1.0, "macd_signal": 0.5}
    esperado, _ = calcular_score_tecnico(fila)
    c = fs.condiciones_desde_fila(fila)
    assert fs.score(c) == esperado == 4.5
    assert c["rsi"] is False and c["rsi_valor"] == 70.0


def test_umbrales_iguales_al_bot():
    ruta = os.path.join(ROOT, "scripts", "forward_testing", "ft_bot_tech_sectorial.py")
    txt = open(ruta, encoding="utf-8").read()
    ent = float(re.search(r"^SCORE_ENTRADA\s*=\s*([\d.]+)", txt, re.M).group(1))
    sal = float(re.search(r"^SCORE_SALIDA\s*=\s*([\d.]+)", txt, re.M).group(1))
    assert (ent, sal) == (fs.SCORE_ENTRADA_V1, fs.SCORE_SALIDA_V1)


# --- gatillos ------------------------------------------------------------------

def test_gatillos_rsi_por_arriba_y_por_abajo():
    prev = _cond()
    assert fs.gatillos(prev, _cond(rsi=False, rsi_valor=72)) == ["RSI sale por arriba (>68)"]
    assert fs.gatillos(prev, _cond(rsi=False, rsi_valor=40)) == ["RSI sale por abajo (<45)"]


def test_gatillos_varios_y_ninguno():
    prev = _cond()
    assert fs.gatillos(prev, _cond(sma50=False, sma21=False)) == ["pierde SMA50", "pierde SMA21"]
    assert fs.gatillos(prev, prev) == ["sin cambio en la ultima rueda"]
    # una condicion que ya estaba perdida no es gatillo
    assert fs.gatillos(_cond(sma21=False), _cond(sma21=False)) == ["sin cambio en la ultima rueda"]


# --- despues de la salida -------------------------------------------------------

@pytest.mark.parametrize("motivo,familia", [
    ("SCORE_DEGRADADO_3.0", "SCORE"), ("SCORE_DEGRADADO_COMPRA_67", "SCORE"),
    ("SCORE_DEGRADADO_0.0_SPLIT_FIX", "SPLIT_FIX"), ("STOP_LOSS_ATR", "STOP"),
    ("TRAILING_SL", "STOP"), ("TAKE_PROFIT_ATR", "TAKE_PROFIT"),
    ("TIME_STOP_21D", "TIME_STOP"), ("EARNINGS_MANANA", "BALANCE"),
    ("ROTACION_SECTORIAL", "ROTACION"),
])
def test_familia_salida(motivo, familia):
    assert fs.familia_salida(motivo) == familia


def test_exceso_z_y_clasificacion():
    ex, z = fs.exceso_z(0.05, 0.01, 0.02, 4)          # exceso 4%, sigma*sqrt(4) = 4%
    assert ex == pytest.approx(0.04) and z == pytest.approx(1.0)
    assert fs.clasificar(z) == "temprano"
    assert fs.clasificar(-1.0) == "a_tiempo"
    assert fs.clasificar(0.3) == "indiferente"
    assert fs.clasificar(None) is None
    assert fs.exceso_z(0.05, 0.01, 0.0, 4) == (pytest.approx(0.04), None)
    assert fs.exceso_z(float("nan"), 0.01, 0.02, 4) == (None, None)


def test_ic95_por_dia_el_dia_pesa_uno():
    # 3 operaciones el dia A (+1) y 1 el dia B (-1): por operacion +0,5, por dia 0
    r = fs.ic95_por_dia([1, 1, 1, -1], ["A", "A", "A", "B"])
    assert r["ops"] == 4 and r["dias"] == 2
    assert r["media_ops"] == pytest.approx(0.5)
    assert r["media"] == pytest.approx(0.0)
    assert math.isnan(r["lo"])                          # con 2 dias no hay intervalo


def test_ic95_por_dia_intervalo():
    r = fs.ic95_por_dia([1.0, 2.0, 3.0, float("nan")], ["a", "b", "c", "d"])
    assert r["dias"] == 3 and r["media"] == pytest.approx(2.0)
    assert r["lo"] < 2.0 < r["hi"]


# --- re-simulacion y regla de lectura (paso 1) -----------------------------------

def test_ruedas_de_balance_marca_la_rueda_anterior():
    from datetime import date
    f = [date(2026, 5, d) for d in (4, 5, 6, 7, 8, 11)]      # lun..vie + lun
    b = fs.ruedas_de_balance(f, [date(2026, 5, 7), date(2026, 5, 10), date(2026, 5, 11),
                                 date(2026, 5, 4), date(2026, 6, 30)])
    # 7/5 -> sale el 6/5; 10/5 (domingo) -> sale el 8/5; 11/5 -> sale el 8/5;
    # 4/5 es la primera rueda (no hay anterior); 30/6 esta despues del final (no se marca)
    assert b == [False, False, True, False, True, False]


def test_primera_salida_prioridad_y_censura():
    close = [100, 101, 99, 97, 110]
    nada = [False] * 5
    # stop en 98: la rueda 3 (97) lo toca
    assert fs.primera_salida(close, nada, nada, 0, 98, 120) == (3, "STOP")
    # el score va antes que el stop
    sc = [False, False, False, True, False]
    assert fs.primera_salida(close, sc, nada, 0, 98, 120) == (3, "SCORE")
    # el balance va antes que todo
    bal = [False, False, False, True, False]
    assert fs.primera_salida(close, sc, bal, 0, 98, 120) == (3, "BALANCE")
    # take profit
    assert fs.primera_salida(close, nada, nada, 0, 90, 105) == (4, "TAKE_PROFIT")
    # sin salida -> censura; y la rueda de entrada nunca cuenta
    assert fs.primera_salida(close, nada, nada, 0, 50, 200) == (None, None)
    assert fs.primera_salida([100, 90], [True, False], nada, 0, 95, 200) == (1, "STOP")


def test_primera_salida_saltea_ruedas_sin_precio():
    close = [100, float("nan"), 97]
    assert fs.primera_salida(close, [False, True, False], [False] * 3, 0, 98, 200) == (2, "STOP")


def test_score_por_rueda_sin_datos_no_sale():
    malo = _cond(sma50=False)
    assert fs.score_por_rueda([None, malo, _cond()], "actual") == [False, True, False]


def test_evaluar_variante_pasa_y_no_pasa():
    dias = [f"d{i}" for i in range(12)]
    anios = [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024, 2025, 2025, 2026, 2026]
    ret = [1.0] * 12
    buena = fs.evaluar_variante([0.5, 0.6] * 6, dias, anios, ret, ret)
    assert buena["pasa"] and buena["anios_positivos"] == 6
    # misma mejora pero la cola empeora 2 puntos -> no pasa
    peor_cola = fs.evaluar_variante([0.5, 0.6] * 6, dias, anios, [-1.0] * 12, ret)
    assert peor_cola["c1_ic_sobre_cero"] and not peor_cola["c3_cola"] and not peor_cola["pasa"]
    # diferencia nula -> no pasa
    nula = fs.evaluar_variante([0.1, -0.1] * 6, dias, anios, ret, ret)
    assert not nula["c1_ic_sobre_cero"] and not nula["pasa"]


# --- paso 2: grilla de pesos ------------------------------------------------------

def test_indice_estado_ida_y_vuelta():
    for i in range(fs.N_ESTADOS):
        assert fs.indice_estado(fs.estado_desde_indice(i)) == i


def test_pesos_actuales_reproducen_la_regla_actual():
    assert fs.mascara_pesos(fs.PESOS_ACTUALES) == fs.mascara_variante("actual")


def test_grilla_589_reglas_y_la_actual_4_veces():
    g = fs.reglas_grilla()
    assert sum(len(v) for v in g.values()) == 3750
    assert len(g) == 589
    assert len(g[fs.mascara_variante("actual")]) == 4


def test_macd_1_da_las_mismas_salidas():
    p = dict(fs.PESOS_ACTUALES, macd=1)
    assert fs.mascara_pesos(p) == fs.mascara_variante("actual")


def test_sma50_3_deja_de_salir_en_3_combinaciones():
    d = fs.diferencias_de_regla(fs.mascara_pesos(dict(fs.PESOS_ACTUALES, sma50=3)),
                                fs.mascara_variante("actual"))
    assert d["empieza_a_salir"] == []
    assert sorted(d["deja_de_salir"]) == sorted([
        "SMA200 SMA50 . . RSI", "SMA200 SMA50 . MACD .", "SMA200 SMA50 SMA21 . ."])


def test_peso_cero_sale_mas_y_sin_x_sale_menos():
    act = fs.mascara_variante("actual")
    cero = fs.mascara_pesos(dict(fs.PESOS_ACTUALES, sma21=0))
    assert cero & act == act and cero != act            # peso 0: sale en MAS combinaciones
    sin = fs.mascara_variante("sin_sma21")
    assert sin & act == sin and sin != act              # sin_X: sale en MENOS


def test_sma200_con_peso_puede_mantener_debajo_de_la_sma200():
    p = dict(fs.PESOS_ACTUALES, rsi=2)
    m = fs.mascara_pesos(p, sma200=1)
    debajo = fs.indice_estado({"sma200": False, "sma50": True, "sma21": True,
                               "macd": True, "rsi": True})
    assert not (m >> debajo) & 1                        # 2+1+1,5+2 = 6,5 > 3,5: no sale
    assert (fs.mascara_variante("actual") >> debajo) & 1


def test_bits_reglas_columna_sin_datos():
    b = fs.bits_reglas([fs.mascara_variante("actual"), 0])
    assert b.shape == (2, 33) and not b[:, fs.ESTADO_SIN_DATOS].any()
    assert b[0].sum() == bin(fs.mascara_variante("actual")).count("1")


def test_es_candidata():
    assert fs.es_candidata(-0.2, 0.0, -6.0, -6.5)
    assert not fs.es_candidata(0.1, 0.5, -6.0, -6.0)     # post no mejora
    assert not fs.es_candidata(-0.2, -0.1, -6.0, -6.0)   # pierde en el tramo
    assert not fs.es_candidata(-0.2, 0.1, -8.0, -6.0)    # cola 2 puntos peor
    assert not fs.es_candidata(float("nan"), 0.1, -6.0, -6.0)
