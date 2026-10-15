"""Tests de src/utils/carteras_perfil.py (backtest de carteras por perfil)."""

import random
from datetime import date, timedelta

from src.utils import carteras_perfil as cp

D0 = date(2024, 1, 2)


def _ruedas(closes: list, ticker="AAA", candidatos_dia0=("AAA",), balance=None):
    """Una serie de ruedas de un solo ticker; candidato solo el primer dia."""
    out = []
    for i, c in enumerate(closes):
        out.append(cp.Rueda(fecha=D0 + timedelta(days=i), close={ticker: c},
                            balance=(balance or {}).get(i, set()),
                            candidatos=list(candidatos_dia0) if i == 0 else []))
    return out


# --- seleccion -----------------------------------------------------------------

def test_desempate_no_es_alfabetico():
    # 20 empatados en el score maximo: con distintas semillas el top 3 cambia y no
    # es siempre el principio del alfabeto
    el = [(f"T{i:02d}", 5.5, f"S{i}") for i in range(20)]
    tops = {tuple(cp.top_familia(el, random.Random(s))) for s in range(20)}
    assert len(tops) > 10
    assert ("T00", "T01", "T02") not in tops or len(tops) > 1


def test_misma_semilla_mismo_resultado():
    el = [(f"T{i:02d}", 5.5, "S") for i in range(20)]
    a = cp.top_familia(el, random.Random(7), max_por_sector=None)
    b = cp.top_familia(list(reversed(el)), random.Random(7), max_por_sector=None)
    assert a == b                      # el orden de entrada no influye


def test_score_manda_sobre_el_sorteo():
    el = [("BAJO", 4.0, "A"), ("ALTO", 5.5, "B"), ("MEDIO", 5.0, "C")]
    assert cp.top_familia(el, random.Random(1)) == ["ALTO", "MEDIO", "BAJO"]


def test_tope_por_sector_en_tecnica():
    el = [("A1", 5.5, "Tech"), ("A2", 5.5, "Tech"), ("A3", 5.5, "Tech"), ("B1", 4.0, "Energy")]
    sel = cp.seleccionar({"Tecnica": el}, random.Random(3))
    assert len(sel) == 2
    assert "B1" in sel


def test_prioridad_mas_familias_primero():
    elegidos = {"X": {"familias": {"Tecnica"}, "puesto": 1},
                "Y": {"familias": {"Tecnica", "SMC"}, "puesto": 3}}
    out = cp.por_cartera(elegidos, {"X": "Moderada", "Y": "Moderada"}, random.Random(1))
    assert out["Moderada"] == ["Y", "X"]


def test_ticker_sin_perfil_queda_afuera():
    elegidos = {"X": {"familias": {"SMC"}, "puesto": 1}}
    out = cp.por_cartera(elegidos, {}, random.Random(1))
    assert all(not v for v in out.values())


# --- simulacion ----------------------------------------------------------------

def test_stop_loss_sobre_el_close_con_gap():
    # entra a 100; cae a 95 (-5%, nada) y despues gap a 90 (-10%): sale a 90, no a 93
    sim = cp.simular(_ruedas([100, 95, 90, 120]), "Conservadora")
    op = sim["operaciones"][0]
    assert op["motivo_salida"] == cp.MOTIVO_SL
    assert op["precio_salida"] == 90


def test_take_profit_por_perfil():
    # +12% alcanza el TP de Conservadora pero no el de Moderada (15%)
    serie = [100, 112, 112]
    c = cp.simular(_ruedas(serie), "Conservadora")["operaciones"][0]
    m = cp.simular(_ruedas(serie), "Moderada")["operaciones"][0]
    assert c["motivo_salida"] == cp.MOTIVO_TP
    assert m["motivo_salida"] == cp.MOTIVO_FIN


def test_balance_tiene_prioridad_y_bloquea_la_entrada():
    # dia 1: balance cerca -> sale aunque este en TP
    sim = cp.simular(_ruedas([100, 130], balance={1: {"AAA"}}), "Conservadora")
    assert sim["operaciones"][0]["motivo_salida"] == cp.MOTIVO_EARN
    # balance el dia de la senal -> no entra
    sim = cp.simular(_ruedas([100, 130], balance={0: {"AAA"}}), "Conservadora")
    assert sim["operaciones"] == []


def test_tamano_15pct_del_capital_actual_y_cupo():
    tks = [f"T{i}" for i in range(8)]
    r = [cp.Rueda(D0, {t: 100.0 for t in tks}, set(), tks)]
    sim = cp.simular(r, "Moderada")
    ops = sim["operaciones"]
    assert len(ops) == cp.MAX_POSICIONES                 # 8 candidatos, entran 6
    assert all(o["qty"] == 150 for o in ops)             # 15% de 100k / 100


def test_no_entra_otro_hasta_vender_y_no_reentra_la_misma_rueda():
    tks = [f"T{i}" for i in range(7)]
    r0 = cp.Rueda(D0, {t: 100.0 for t in tks}, set(), tks)
    c1 = {t: 100.0 for t in tks}
    c1["T0"] = 80.0                                       # T0 toca el SL
    r1 = cp.Rueda(D0 + timedelta(days=1), c1, set(), ["T0", "T6"])
    sim = cp.simular([r0, r1], "Moderada")
    entradas = {(o["ticker"], o["fecha_entrada"]) for o in sim["operaciones"]}
    assert ("T6", r1.fecha) in entradas                  # ocupa el lugar que libero T0
    assert ("T0", r1.fecha) not in entradas              # no reentra la misma rueda


def test_capital_actual_crece_con_las_ganancias():
    # gana 12% con la primera (TP) y la segunda compra se calcula sobre el capital nuevo
    r = [cp.Rueda(D0, {"A": 100.0}, set(), ["A"]),
         cp.Rueda(D0 + timedelta(days=1), {"A": 112.0, "B": 100.0}, set(), []),
         cp.Rueda(D0 + timedelta(days=2), {"B": 100.0}, set(), ["B"])]
    ops = cp.simular(r, "Conservadora")["operaciones"]
    b = next(o for o in ops if o["ticker"] == "B")
    assert b["qty"] == int(0.15 * (100_000 + 150 * 12) / 100)


# --- lectura -------------------------------------------------------------------

def _res(total, tramos):
    return {"retorno_total_pct": total, "tramos": tramos}


def test_leer_pasa_y_no_pasa():
    sel = [_res(20, {a: 5 for a in range(2021, 2027)})] * 3
    azar = [_res(5, {a: 1 for a in range(2021, 2027)})] * 3
    assert cp.leer(sel, azar)["pasa"] is True
    # gana en 3 de 6 tramos -> no pasa aunque gane plata
    sel2 = [_res(20, {2021: 5, 2022: 5, 2023: 5, 2024: 0, 2025: 0, 2026: 0})] * 3
    assert cp.leer(sel2, azar)["pasa"] is False
    # pierde plata -> no pasa aunque gane todos los tramos al azar
    sel3 = [_res(-1, {a: 5 for a in range(2021, 2027)})] * 3
    azar3 = [_res(-30, {a: -10 for a in range(2021, 2027)})] * 3
    assert cp.leer(sel3, azar3)["pasa"] is False


# --- variantes de salida (diagnostico de la Conservadora) ----------------------

def test_default_igual_que_salida_explicita():
    r = _ruedas([100, 95, 90, 120])
    a = cp.simular(r, "Conservadora")
    b = cp.simular(r, "Conservadora", salida=cp.Salida(balance=True, sl=0.07, tp=0.12,
                                                        usar_default=False))
    assert a == b


def test_sin_balance_mantiene_la_posicion():
    sal = cp.Salida(balance=False, sl=0.07, tp=0.12, usar_default=False)
    sim = cp.simular(_ruedas([100, 101, 102], balance={1: {"AAA"}}), "Conservadora", salida=sal)
    assert sim["operaciones"][0]["motivo_salida"] == cp.MOTIVO_FIN


def test_sin_balance_igual_bloquea_la_entrada():
    # la variante solo toca la SALIDA: con balance cerca no se compra
    sal = cp.Salida(balance=False, sl=0.07, tp=0.12, usar_default=False)
    sim = cp.simular(_ruedas([100, 130], balance={0: {"AAA"}}), "Conservadora", salida=sal)
    assert sim["operaciones"] == []


def test_sin_sl_ni_tp():
    sal = cp.Salida(balance=True, sl=None, tp=None, usar_default=False)
    sim = cp.simular(_ruedas([100, 50, 200]), "Conservadora", salida=sal)
    assert sim["operaciones"][0]["motivo_salida"] == cp.MOTIVO_FIN


def test_time_stop_en_ruedas():
    sal = cp.Salida(balance=False, sl=None, tp=None, time_stop=2, usar_default=False)
    sim = cp.simular(_ruedas([100, 100, 100, 100]), "Conservadora", salida=sal)
    op = sim["operaciones"][0]
    assert op["motivo_salida"] == cp.MOTIVO_TIME
    assert op["fecha_salida"] == D0 + timedelta(days=2)
