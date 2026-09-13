"""
test_precio_referencia.py -- regla del precio de referencia del subyacente.

El close de precios_diarios manda, llevado a la escala de la rueda con los splits
reales posteriores (registro polygon_splits); el precio de la captura de opciones
solo tapa el hueco. Casos sinteticos con los numeros reales medidos el 10/9/2026.
Sin DB.
"""

from datetime import date
from decimal import Decimal

import pytest

from src.utils import precio_referencia as pr

KLAC = [(date(2026, 6, 12), Decimal("10.000"))]
SCCO = [(date(2026, 5, 13), Decimal("1.010")), (date(2026, 8, 11), Decimal("1.012"))]


# ── Resolucion basica ─────────────────────────────────────────────────────────

def test_el_close_diario_manda_aunque_haya_captura():
    assert pr.resolver_precio(100.0, 100.04) == (100.0, pr.FUENTE_DIARIO)


def test_la_captura_tapa_el_hueco_del_close():
    assert pr.resolver_precio(None, 97.0) == (97.0, pr.FUENTE_SNAPSHOT)


def test_la_captura_que_tapa_el_hueco_no_se_escala():
    # La captura ya esta en la escala de su dia: el factor no le aplica.
    assert pr.resolver_precio(None, 1829.47, factor=10.0) == (1829.47, pr.FUENTE_SNAPSHOT)


def test_sin_ninguno_no_inventa_precio():
    # Caso 2026-09-09 sin precios_diarios cargado: hueco visible, no un numero.
    assert pr.resolver_precio(None, None) == (None, None)


def test_valores_invalidos_cuentan_como_ausentes():
    for malo in (0, -5, float("nan"), "abc"):
        assert pr.resolver_precio(malo, 50.0) == (50.0, pr.FUENTE_SNAPSHOT)
        assert pr.resolver_precio(80.0, malo) == (80.0, pr.FUENTE_DIARIO)
    assert pr.resolver_precio(0, float("nan")) == (None, None)
    assert pr.resolver_precio(80.0, None, factor=0) == (80.0, pr.FUENTE_DIARIO)


def test_acepta_decimal_de_la_db():
    precio, fuente = pr.resolver_precio(Decimal("212.3400"), None)
    assert isinstance(precio, float)
    assert precio == 212.34
    assert fuente == pr.FUENTE_DIARIO


# ── Escala de split ───────────────────────────────────────────────────────────

def test_factor_escala_klac_antes_y_despues_del_split():
    assert pr.factor_escala(KLAC, date(2026, 5, 20)) == 10.0
    # El dia del split ya rige la escala nueva.
    assert pr.factor_escala(KLAC, date(2026, 6, 12)) == 1.0
    assert pr.factor_escala(KLAC, date(2026, 9, 9)) == 1.0


def test_factor_escala_ignora_ajustes_chicos():
    # SCCO 1,01 / 1,012 no son splits de precio (con ellos el cruce empeora).
    assert pr.factor_escala(SCCO, date(2026, 5, 1)) == 1.0


def test_factor_escala_multiplica_y_respeta_hasta():
    splits = [(date(2026, 3, 1), 2), (date(2026, 7, 1), 3), (date(2026, 12, 1), 4)]
    assert pr.factor_escala(splits, date(2026, 1, 5)) == 24.0
    # Un split posterior a lo reflejado en precios_diarios no cuenta.
    assert pr.factor_escala(splits, date(2026, 1, 5), hasta=date(2026, 9, 9)) == 6.0
    # Split reverso 1:10.
    assert pr.factor_escala([(date(2026, 7, 1), 0.1)], date(2026, 6, 1)) == pytest.approx(0.1)


def test_factores_por_ticker_omite_los_neutros():
    filas = [("KLAC", date(2026, 6, 12), Decimal("10")),
             ("SCCO", date(2026, 8, 11), Decimal("1.012")),
             ("CRWD", date(2026, 7, 2), Decimal("4"))]
    assert pr.factores_por_ticker(filas, date(2026, 5, 20)) == {"KLAC": 10.0, "CRWD": 4.0}
    assert pr.factores_por_ticker(filas, date(2026, 6, 20)) == {"CRWD": 4.0}


def test_close_en_escala_con_captura_fresca():
    # KLAC 2026-05-20: close corregido 182,947 x 10 = captura del dia.
    precio, fuente = pr.resolver_precio(182.947, 1829.47, factor=10.0)
    assert precio == pytest.approx(1829.47)
    assert fuente == pr.FUENTE_DIARIO_ESCALA


def test_con_captura_rancia_el_valor_sale_del_close_no_de_la_captura():
    # KLAC 2026-04-27: captura rancia de Railway 1935 vs close 190 (x10 = 1900).
    # El caso que rompio el primer diseno (ratio 10,18 no es split exacto).
    precio, fuente = pr.resolver_precio(190.0, 1935.0, factor=10.0)
    assert precio == pytest.approx(1900.0)
    assert fuente == pr.FUENTE_DIARIO_ESCALA


# ── Validacion ────────────────────────────────────────────────────────────────

def test_ratio_de_split_tolerancia_medida():
    assert pr.ratio_de_split(100.0, 1000.0) == 10.0
    assert pr.ratio_de_split(100.0, 400.9999) == 4.0
    assert pr.ratio_de_split(100.0, 25.0) == 0.25
    assert pr.ratio_de_split(100.0, 1018.0) is None     # rancio, no split exacto
    assert pr.ratio_de_split(100.0, 101.2) is None
    assert pr.ratio_de_split(None, 1000.0) is None


def test_escala_sin_registro_avisa_si_falta_el_split():
    closes = {"KLAC": 182.947, "AAPL": 302.25}
    snaps = {"KLAC": 1829.47, "AAPL": 302.25}
    assert pr.escalas_sin_registro(closes, snaps) == [("KLAC", 182.947, 1829.47, 10.0)]
    assert pr.escalas_sin_registro(closes, snaps, factores={"KLAC": 10.0}) == []


def test_divergencias_en_escala():
    closes = {"OK": 100.0, "KLAC": 190.0, "RANCIO": 100.0, "SIN_REG": 50.0, "SOLO_CLOSE": 5.0}
    snaps = {"OK": 100.04, "KLAC": 1935.0, "RANCIO": 94.0, "SIN_REG": 500.0, "SOLO_SNAP": 7.0}
    div = pr.medir_divergencias(closes, snaps, factores={"KLAC": 10.0})
    # KLAC en escala (1900 vs 1935) es captura rancia 1,8%; SIN_REG va por
    # escalas_sin_registro, no aca.
    assert [d[0] for d in div] == ["RANCIO", "KLAC"]
    assert div[1][1] == pytest.approx(1900.0)


def test_divergencias_respeta_la_tolerancia_y_ordena():
    closes = {"A": 100.0, "B": 100.0, "C": 100.0}
    # Lejos del borde a proposito: 99.5/100-1 da 0.0050000000000000044 en float,
    # que es > 0.005 -> un caso "justo en la tolerancia" no es testeable.
    snaps = {"A": 101.0, "B": 107.0, "C": 99.8}   # 1% / 7% / 0,2%
    assert [d[0] for d in pr.medir_divergencias(closes, snaps, tol=0.005)] == ["B", "A"]
    assert [d[0] for d in pr.medir_divergencias(closes, snaps, tol=0.02)] == ["B"]


# ── Mapas, conteos y cobertura ───────────────────────────────────────────────

def test_resolver_mapa_con_factores_y_huecos():
    closes = {"AAA": 10.0, "KLAC": 182.947}
    snaps = {"KLAC": 1829.47, "CCC": 30.0}
    res = pr.resolver_mapa(closes, snaps, factores={"KLAC": 10.0})
    assert res["AAA"] == (10.0, pr.FUENTE_DIARIO)
    assert res["KLAC"][1] == pr.FUENTE_DIARIO_ESCALA
    assert res["CCC"] == (30.0, pr.FUENTE_SNAPSHOT)

    res = pr.resolver_mapa(closes, snaps, tickers=["AAA", "ZZZ"])
    assert set(res) == {"AAA", "ZZZ"}
    assert res["ZZZ"] == (None, None)


def test_contar_fuentes():
    res = {"A": (1.0, pr.FUENTE_DIARIO), "B": (2.0, pr.FUENTE_DIARIO),
           "C": (3.0, pr.FUENTE_SNAPSHOT), "D": (None, None),
           "E": (1829.47, pr.FUENTE_DIARIO_ESCALA)}
    assert pr.contar_fuentes(res) == {
        pr.FUENTE_DIARIO: 2, pr.FUENTE_DIARIO_ESCALA: 1,
        pr.FUENTE_SNAPSHOT: 1, "sin_precio": 1}


def test_cobertura_baja():
    assert pr.cobertura_baja(0, 200) is True          # 2026-09-09
    assert pr.cobertura_baja(179, 200) is True
    assert pr.cobertura_baja(180, 200) is False       # justo 90%
    assert pr.cobertura_baja(200, 200) is False
    assert pr.cobertura_baja(0, 0) is False           # sin tickers no es anomalia
