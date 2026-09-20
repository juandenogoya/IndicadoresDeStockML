"""Tests de src/utils/ft_entradas.py (analisis de ENTRADAS de FT).

Lo que estos tests protegen, en orden de importancia:
  1. que la regla escrita sin pesos sea EXACTAMENTE la del bot (si no, toda grilla
     construida encima mide otra cosa);
  2. que la frontera de las zonas en 0 separe igual que la condicion binaria;
  3. que la version vectorizada de la clasificacion coincida con la escalar.
"""

import math
import os
import re

import pandas as pd
import pytest

from src.strategies.scoring import RSI_MAX, RSI_MIN, calcular_score_tecnico
from src.utils import ft_entradas as fe

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _fila(close=100.0, sma21=98.0, sma50=95.0, sma200=90.0,
          rsi14=55.0, macd=1.0, macd_signal=0.5):
    return {"close": close, "sma21": sma21, "sma50": sma50, "sma200": sma200,
            "rsi14": rsi14, "macd": macd, "macd_signal": macd_signal}


# --- la regla actual, enumerada -----------------------------------------------

def test_valores_posibles_del_score():
    assert fe.valores_posibles_score() == [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.5]


@pytest.mark.parametrize("c", fe.combinaciones_binarias())
def test_regla_booleana_es_identica_al_score(c):
    """La afirmacion central del doc: score >= 4,0 es SMA200 y SMA50 y 2 de 3."""
    assert fe.regla_v1_booleana(c) == fe.cumple_entrada_por_score(c)


@pytest.mark.parametrize("c", fe.combinaciones_binarias())
def test_sin_sma50_nunca_entra(c):
    """SMA50 no pondera: es obligatoria de hecho (sin ella el maximo es 3,5)."""
    if not c["sma50"]:
        assert fe.score(c) <= 3.5
        assert not fe.cumple_entrada_por_score(c)


def test_solo_cuatro_de_dieciseis_estados_califican():
    califican = [c for c in fe.combinaciones_binarias()
                 if c["sma200"] and fe.cumple_entrada_por_score(c)]
    assert len(califican) == 4


def test_el_umbral_no_es_continuo():
    """Todo el rango de umbrales produce unas pocas reglas distintas."""
    reglas = fe.reglas_por_umbral()
    assert len(set(reglas.values())) < len(fe.valores_posibles_score())
    # el umbral vigente y el siguiente valor posible dan reglas DISTINTAS
    assert reglas[4.0] != reglas[4.5]


def test_umbral_igual_al_bot():
    ruta = os.path.join(ROOT, "scripts", "forward_testing", "ft_bot_tech_sectorial.py")
    with open(ruta, encoding="utf-8") as f:
        txt = f.read()
    ent = float(re.search(r"^SCORE_ENTRADA\s*=\s*([\d.]+)", txt, re.M).group(1))
    assert fe.SCORE_ENTRADA_V1 == ent


# --- zonas: bordes ------------------------------------------------------------

@pytest.mark.parametrize("dist,esperada", [
    (-30.0, "S50_BAJA"), (-10.0, "S50_BAJA"), (-9.99, "S50_MEDIA_NEG"),
    (-5.0, "S50_MEDIA_NEG"), (-4.99, "S50_CERCA_NEG"),
    (0.0, "S50_CERCA_NEG"), (0.0001, "S50_CERCA"),
    (5.0, "S50_CERCA"), (5.01, "S50_MEDIA"),
    (10.0, "S50_MEDIA"), (10.01, "S50_ALTA"), (80.0, "S50_ALTA"),
])
def test_zona_s50_bordes(dist, esperada):
    assert fe.zona_s50(dist) == esperada


@pytest.mark.parametrize("dist,esperada", [
    (-7.0, "S21_BAJA"), (-6.5, "S21_BAJA"), (-3.5, "S21_MEDIA_NEG"),
    (0.0, "S21_CERCA_NEG"), (0.01, "S21_CERCA"), (3.5, "S21_CERCA"),
    (3.6, "S21_MEDIA"), (6.5, "S21_MEDIA"), (6.6, "S21_ALTA"),
])
def test_zona_s21_bordes(dist, esperada):
    assert fe.zona_s21(dist) == esperada


def test_frontera_cero_separa_igual_que_la_condicion_binaria():
    """La razon de usar intervalos (a, b]: dist > 0 <=> close > sma."""
    for dist in (-12.0, -7.0, -0.001, 0.0, 0.001, 3.0, 7.0, 12.0):
        assert (fe.zona_s50(dist) in fe.ZONAS_S50_OK) == (dist > 0)
        assert (fe.zona_s21(dist) in fe.ZONAS_S21_OK) == (dist > 0)


def test_zonas_devuelven_none_sin_dato():
    assert fe.zona_s50(None) is None
    assert fe.zona_s21(float("nan")) is None
    assert fe.zona_macd(None, 1.0) is None
    assert fe.zona_rsi(None) is None
    assert fe.distancia_pct(100.0, 0.0) is None


@pytest.mark.parametrize("rsi,esperada", [
    (10.0, "RSI_LOW"), (44.99, "RSI_LOW"), (RSI_MIN, "RSI_IN"), (55.0, "RSI_IN"),
    (RSI_MAX, "RSI_IN"), (68.01, "RSI_HIGH"), (95.0, "RSI_HIGH"),
])
def test_zona_rsi(rsi, esperada):
    assert fe.zona_rsi(rsi) == esperada
    assert fe.rsi_ok(rsi) == (esperada == "RSI_IN")


def test_macd_el_and_hist_es_redundante():
    """`macd > signal and hist > 0` con hist = macd - signal es una sola condicion."""
    for macd, signal in ((1.0, 0.5), (0.5, 1.0), (0.0, 0.0), (-2.0, -3.0), (-3.0, -2.0)):
        fila = _fila(macd=macd, macd_signal=signal)
        _, d = calcular_score_tecnico(fila)
        assert (fe.zona_macd(macd, signal) == "MACD_UP") == bool(d["cond_macd"])


# --- coherencia zonas <-> condiciones binarias --------------------------------

@pytest.mark.parametrize("close,sma21,sma50,sma200,rsi14,macd,sig", [
    (100.0, 98.0, 95.0, 90.0, 55.0, 1.0, 0.5),
    (100.0, 101.0, 105.0, 90.0, 30.0, -1.0, 0.5),
    (100.0, 100.0, 100.0, 100.0, 68.0, 0.0, 0.0),
    (50.0, 55.0, 60.0, 70.0, 72.0, -2.0, -1.0),
    (120.0, 100.0, 90.0, 80.0, 80.0, 3.0, 1.0),
])
def test_condiciones_desde_estado_reproducen_a_scoring(close, sma21, sma50, sma200,
                                                       rsi14, macd, sig):
    fila = _fila(close, sma21, sma50, sma200, rsi14, macd, sig)
    estado = fe.estado_desde_fila(fila)
    assert fe.condiciones_desde_estado(estado) == fe.condiciones_desde_fila(fila)


def test_clave_estado_y_enumeracion():
    estado = fe.estado_desde_fila(_fila())
    clave = fe.clave_estado(estado)
    # close 100 / sma50 95 -> +5,26% (S50_MEDIA); / sma21 98 -> +2,04% (S21_CERCA)
    assert clave == "S50_MEDIA|S21_CERCA|MACD_UP|RSI_IN"
    posibles = fe.estados_posibles()
    assert len(posibles) == 6 * 6 * 2 * 3      # RSI desglosado para medir
    assert len(set(posibles)) == len(posibles)
    assert clave in posibles


def test_clave_none_si_falta_un_eje():
    estado = fe.estado_desde_fila(_fila(macd=None, macd_signal=None))
    assert fe.clave_estado(estado) is None


# --- vectorizada == escalar ---------------------------------------------------

def test_clasificar_serie_igual_a_la_escalar():
    valores = [-25.0, -10.0, -9.99, -5.0, -0.001, 0.0, 0.001, 5.0, 5.01,
               10.0, 10.01, 42.0, float("nan")]
    serie = fe.clasificar_serie(pd.Series(valores), fe.CORTES_S50, fe.ZONAS_S50)
    for valor, obtenido in zip(valores, serie):
        assert obtenido == fe.zona_s50(valor)


def test_cortes_rsi_vectorizados_coinciden_con_zona_rsi():
    valores = [10.0, 44.99, 45.0, 55.0, 68.0, 68.01, 95.0, float("nan")]
    serie = fe.clasificar_serie(pd.Series(valores), fe.CORTES_RSI, fe.ZONAS_RSI)
    for valor, obtenido in zip(valores, serie):
        assert obtenido == fe.zona_rsi(valor)


def test_clasificar_serie_valida_longitudes():
    with pytest.raises(ValueError):
        fe.clasificar_serie(pd.Series([1.0]), fe.CORTES_S50, fe.ZONAS_S21[:3])


# --- el espacio de reglas booleanas -------------------------------------------

def test_dedekind_m4():
    """168 funciones monotonas de 4 variables, 166 sin las triviales."""
    assert len(fe.reglas_monotonas(incluir_triviales=True)) == 168
    assert len(fe.reglas_monotonas()) == 166


def test_indice_y_estado_son_inversos():
    for i in range(fe.N_ESTADOS):
        e = fe.estado_de_indice(i)
        assert fe.indice_estado(e["sma50"], e["sma21"], e["macd"], e["rsi"]) == i


def test_la_v1_esta_en_el_espacio_y_es_la_regla_conocida():
    v1 = fe.mascara_v1()
    assert v1 in fe.reglas_monotonas()
    assert bin(v1).count("1") == 4
    for i in range(fe.N_ESTADOS):
        cond = dict(fe.estado_de_indice(i), sma200=True)
        assert bool((v1 >> i) & 1) == fe.regla_v1_booleana(cond)


def test_monotonia_detecta_la_no_monotona():
    # entra con {sma50} pero no con {sma50, rsi}: agregar una condicion saca -> no monotona
    m = (1 << fe.indice_estado(1, 0, 0, 0))
    assert not fe.es_monotona(m)
    assert fe.es_monotona(m | (1 << fe.indice_estado(1, 0, 0, 1))
                          | (1 << fe.indice_estado(1, 1, 0, 0))
                          | (1 << fe.indice_estado(1, 0, 1, 0))
                          | (1 << fe.indice_estado(1, 1, 1, 0))
                          | (1 << fe.indice_estado(1, 1, 0, 1))
                          | (1 << fe.indice_estado(1, 0, 1, 1))
                          | (1 << fe.indice_estado(1, 1, 1, 1)))


def test_hay_reglas_que_ningun_score_ponderado_alcanza():
    """Las 18 que justifican enumerar reglas en vez de barrer pesos."""
    no_umbral = [m for m in fe.reglas_monotonas() if not fe.es_de_umbral(m)]
    assert len(no_umbral) == 18
    assert fe.mascara_v1() not in no_umbral


@pytest.mark.parametrize("m", fe.reglas_monotonas())
def test_implicantes_primos_describen_la_regla_sin_perdida(m):
    """Entra si y solo si se cumple alguno de los implicantes primos entero."""
    primos = fe.implicantes_primos(m)
    for i in range(fe.N_ESTADOS):
        e = fe.estado_de_indice(i)
        cubierto = any(all(e[k] for k in grupo) for grupo in primos)
        assert cubierto == bool((m >> i) & 1)


def test_texto_regla_de_la_v1():
    t = fe.texto_regla(fe.mascara_v1())
    assert t.count("|") == 2
    assert all("SMA50" in parte for parte in t.split(" | "))


def test_evaluar_mascara_vectorizado_igual_al_bit():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 500
    c = {k: rng.random(n) > 0.5 for k in fe.CONDICIONES_PUNTUADAS}
    for m in (fe.mascara_v1(), fe.reglas_monotonas()[10], fe.reglas_monotonas()[-1]):
        got = fe.evaluar_mascara(m, c["sma50"], c["sma21"], c["macd"], c["rsi"])
        for j in range(n):
            i = fe.indice_estado(c["sma50"][j], c["sma21"][j], c["macd"][j], c["rsi"][j])
            assert bool(got[j]) == bool((m >> i) & 1)


def test_mascara_desde_pesos_coincide_con_estados_que_califican():
    for nombre, (pesos, umbral) in fe.JUEGOS_PESOS.items():
        m = fe.mascara_desde_pesos(pesos, umbral)
        etiquetas = fe.estados_que_califican(pesos, umbral)
        assert bin(m).count("1") == len(etiquetas)


# --- juegos de pesos ----------------------------------------------------------

def test_pesos_v1_reproducen_la_regla_actual():
    pesos, umbral = fe.JUEGOS_PESOS["v1"]
    for c in fe.combinaciones_binarias():
        entra = c["sma200"] and fe.score_ponderado(c, pesos) >= umbral
        assert entra == fe.regla_v1_booleana(c)


def test_estados_que_califican_de_la_v1_son_cuatro():
    assert len(fe.estados_que_califican()) == 4


def test_los_juegos_pre_declarados_no_son_duplicados():
    """Dos juegos con el mismo conjunto de estados son la misma estrategia."""
    grupos = fe.reglas_distintas()
    duplicados = {tuple(v) for v in grupos.values() if len(v) > 1}
    assert not duplicados, f"juegos que producen la misma regla: {duplicados}"
    assert len(grupos) == len(fe.JUEGOS_PESOS)


def test_escalar_los_pesos_y_el_umbral_da_la_misma_regla():
    """Lo que define la decision es la relacion pesos/umbral, no su escala."""
    pesos, umbral = fe.JUEGOS_PESOS["v1"]
    dobles = tuple(2 * p for p in pesos)
    assert fe.estados_que_califican(dobles, 2 * umbral) == fe.estados_que_califican(
        pesos, umbral)


def test_ponderar_el_fallo_equivale_a_mover_el_umbral():
    """Dar peso al 'no cumple' no agrega ninguna regla: es una reparametrizacion."""
    w_mas, w_menos = (2.0, 1.0, 1.5, 1.0), (0.5, 0.5, 0.5, 0.5)
    # score_con_fallo = sum(w_menos) + sum((w_mas - w_menos) * x)
    pesos_dif = tuple(a - b for a, b in zip(w_mas, w_menos))
    constante = sum(w_menos)
    for umbral in (3.0, 3.5, 4.0, 4.5, 5.5):
        for c in fe.combinaciones_binarias():
            if not c["sma200"]:
                continue
            con_fallo = sum(
                (w_mas[i] if c[k] else w_menos[i])
                for i, k in enumerate(("sma50", "sma21", "macd", "rsi")))
            assert (con_fallo >= umbral) == (
                fe.score_ponderado(c, pesos_dif) >= umbral - constante)


# --- criterio a priori de los cortes de SMA21 ---------------------------------

def test_cortes_s21_salen_de_escalar_los_de_s50():
    """Criterio declarado antes de mirar la distribucion: raiz(50/21)."""
    escalados = fe.cortes_escalados(fe.CORTES_S50, fe.FACTOR_S21)
    for corte_declarado, corte_escalado in zip(fe.CORTES_S21, escalados):
        assert abs(corte_declarado - corte_escalado) <= 0.3
    assert math.isclose(fe.FACTOR_S21, math.sqrt(50.0 / 21.0))


# --- control de insumos -------------------------------------------------------

def test_discrepancias_borde_ignora_el_redondeo():
    """El redondeo a 4 decimales NO es una discrepancia; una columna que no corresponde
    al close, si."""
    assert not fe.discrepancias_borde(1.23456, 1.2346)
    assert not fe.discrepancias_borde(0.00005, 0.0)
    assert fe.discrepancias_borde(5.0, 4.5)
    assert fe.discrepancias_borde(1.0, None)
    assert not fe.discrepancias_borde(None, None)


def test_cambia_de_zona_detecta_el_borde_del_cero():
    """El caso que motiva recomputar la distancia: close apenas encima de la media,
    dist guardada 0.0000. La condicion binaria dice True y la zona guardada, negativa."""
    assert fe.cambia_de_zona(0.00005, 0.0, fe.CORTES_S50, fe.ZONAS_S50)
    assert fe.zona_s50(0.00005) in fe.ZONAS_S50_OK
    assert fe.zona_s50(0.0) not in fe.ZONAS_S50_OK
    # lejos de un corte, el redondeo no cambia nada
    assert not fe.cambia_de_zona(7.12345, 7.1235, fe.CORTES_S50, fe.ZONAS_S50)
