"""
test_valuacion_implicita.py -- tests del modulo PURO
src/utils/valuacion_implicita.py.

Casos sinteticos con numeros redondos: no tocan DB ni red. Cada test fija una
decision del modulo, no un detalle de implementacion.
"""

import pytest

from src.utils import valuacion_implicita as V


# Empresa de juguete con numeros que salen exactos a mano:
#   100 acciones a 10 -> market cap 1.000 | deuda neta 200 -> EV 1.200
#   resultado 50 -> PER 20 | ventas 500 -> P/S 2 | ebitda 120 -> EV/EBITDA 10
BASE = {
    "close": 10.0, "shares": 100.0, "net_debt": 200.0,
    "net_income_ttm": 50.0, "revenue_ttm": 500.0, "ebitda_ttm": 120.0,
    "equity": 400.0, "fcf_ttm": 40.0,
}


def test_los_multiplos_salen_de_agregados():
    m = V.multiplos(BASE, 10.0)
    assert m["market_cap"] == 1000.0
    assert m["enterprise_value"] == 1200.0
    assert m["pe_ratio"] == 20.0
    assert m["ps_ratio"] == 2.0
    assert m["pb_ratio"] == 2.5
    assert m["ev_ebitda"] == 10.0
    assert m["fcf_yield"] == pytest.approx(0.04)


def test_al_mover_el_precio_la_deuda_neta_no_se_mueve():
    """
    El escenario es sobre el PRECIO, no sobre el negocio. Duplicar el precio
    duplica el market cap (1.000 -> 2.000) pero el EV sube solo lo mismo en
    ABSOLUTO (1.200 -> 2.200), porque la deuda sigue siendo 200. Confundir eso
    -- duplicar el EV -- es el error clasico del calculo.
    """
    m = V.multiplos(BASE, 20.0)
    assert m["market_cap"] == 2000.0
    assert m["enterprise_value"] == 2200.0
    assert m["pe_ratio"] == 40.0            # el PER si duplica
    assert m["ev_ebitda"] == pytest.approx(2200.0 / 120.0)   # el EV/EBITDA no


def test_un_denominador_negativo_no_produce_multiplo():
    """
    Un PER con resultado negativo no es "barato", es otra categoria. Emitirlo
    ensuciaria la comparacion contra una historia hecha de numeros positivos.
    """
    base = dict(BASE, net_income_ttm=-50.0)
    assert V.multiplos(base, 10.0)["pe_ratio"] is None


def test_el_fcf_yield_si_admite_numerador_negativo():
    """Quemar caja es informacion, y el market cap siempre es > 0."""
    base = dict(BASE, fcf_ttm=-40.0)
    assert V.multiplos(base, 10.0)["fcf_yield"] == pytest.approx(-0.04)


def test_sin_acciones_no_hay_nada():
    assert V.multiplos(dict(BASE, shares=None), 10.0)["market_cap"] is None


# ------------------------------------------------------------- la inversa --
def test_la_inversa_devuelve_el_precio_que_produce_ese_multiplo():
    """Ida y vuelta: pedir el PER actual tiene que devolver el precio actual."""
    assert V.precio_para(BASE, "pe_ratio", 20.0) == pytest.approx(10.0)
    assert V.precio_para(BASE, "ps_ratio", 2.0) == pytest.approx(10.0)
    assert V.precio_para(BASE, "ev_ebitda", 10.0) == pytest.approx(10.0)
    assert V.precio_para(BASE, "fcf_yield", 0.04) == pytest.approx(10.0)


def test_la_inversa_sobre_EV_descuenta_la_deuda_neta():
    """
    EV/EBITDA de 15 -> EV 1.800 -> equity 1.600 -> precio 16. Sin restar la
    deuda daria 18: es exactamente la diferencia entre lo que vale la empresa
    y lo que vale su equity.
    """
    assert V.precio_para(BASE, "ev_ebitda", 15.0) == pytest.approx(16.0)


def test_la_inversa_no_inventa_precios_negativos():
    """
    Un EV objetivo por debajo de la deuda neta implicaria equity negativo. No
    es un precio bajo: no es un precio.
    """
    assert V.precio_para(BASE, "ev_ebitda", 1.0) is None      # EV 120 < deuda 200


def test_la_inversa_necesita_la_deuda_para_las_metricas_de_EV():
    assert V.precio_para(dict(BASE, net_debt=None), "ev_ebitda", 10.0) is None
    # pero el PER no la necesita
    assert V.precio_para(dict(BASE, net_debt=None), "pe_ratio", 20.0) == 10.0


# ------------------------------------------------ ubicacion en la historia --
def test_el_percentil_es_la_fraccion_de_historia_por_debajo():
    """Se lee como "estuvo mas caro que el N% de su historia"."""
    assert V.percentil_de([10, 20, 30, 40], 30) == 0.75
    assert V.percentil_de([10, 20, 30, 40], 5) == 0.0
    assert V.percentil_de([10, 20, 30, 40], 100) == 1.0


def test_el_percentil_descarta_los_huecos_en_vez_de_contarlos_como_cero():
    """Una rueda sin multiplo no es un multiplo de cero."""
    assert V.percentil_de([10, None, 30, None], 30) == 1.0


def test_el_cuantil_interpola():
    assert V.cuantil([10, 20, 30, 40], 0.5) == pytest.approx(25.0)
    assert V.cuantil([10, 20, 30, 40], 0.0) == 10.0
    assert V.cuantil([10, 20, 30, 40], 1.0) == 40.0


def test_serie_vacia_no_rompe():
    assert V.percentil_de([], 10) is None
    assert V.cuantil([], 0.5) is None
    assert V.cuantil(None, 0.5) is None


# ------------------------------------------------------------- escenarios --
def test_escenario_por_variacion_porcentual():
    r = V.escenario(BASE, {"pe_ratio": [10, 20, 30]}, variacion=0.50)
    assert r["precio"] == pytest.approx(15.0)
    assert r["multiplos"]["pe_ratio"] == pytest.approx(30.0)
    assert r["percentiles"]["pe_ratio"] == pytest.approx(1.0)


def test_escenario_por_precio_objetivo_le_gana_a_la_variacion():
    r = V.escenario(BASE, {}, precio_objetivo=12.0, variacion=0.50)
    assert r["precio"] == 12.0
    assert r["variacion"] == pytest.approx(0.20)


def test_avisa_cuando_la_tesis_implica_un_multiplo_nunca_visto():
    """
    No es un veredicto sobre la tesis: es el aviso de que para sostenerla hay
    que creer algo que todavia no paso, y eso merece argumento aparte.
    """
    r = V.escenario(BASE, {"pe_ratio": [10, 15, 20]}, precio_objetivo=50.0)
    assert "pe_ratio" in r["fuera_de_rango"]

    r2 = V.escenario(BASE, {"pe_ratio": [10, 15, 20, 25]}, precio_objetivo=10.0)
    assert "pe_ratio" not in r2["fuera_de_rango"]


def test_precios_de_referencia_convierte_percentiles_en_precios():
    """
    La vista util: "para volver a su PER mediano el precio tendria que ser X".
    Historia de PER [10,20,30,40] -> mediana 25 -> precio 25*50/100 = 12,5.
    """
    ref = V.precios_de_referencia(BASE, {"pe_ratio": [10, 20, 30, 40]},
                                  percentiles=(0.5,))
    assert ref["pe_ratio"][0.5] == pytest.approx(12.5)


def test_precios_de_referencia_deja_None_donde_no_hay_historia():
    ref = V.precios_de_referencia(BASE, {}, percentiles=(0.5,))
    assert ref["pe_ratio"][0.5] is None


def test_ida_y_vuelta_entre_escenario_y_referencia():
    """
    Invariante que amarra las dos direcciones: el precio que precios_de_
    referencia da para el percentil 50 tiene que caer, al pasarlo por
    escenario(), en el percentil 50 de esa misma historia.
    """
    historia = {"pe_ratio": [10.0, 20.0, 30.0, 40.0]}
    precio = V.precios_de_referencia(BASE, historia, percentiles=(0.5,))["pe_ratio"][0.5]
    r = V.escenario(BASE, historia, precio_objetivo=precio)
    assert r["multiplos"]["pe_ratio"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# La direccion inversa: que tiene que hacer el NEGOCIO
# ---------------------------------------------------------------------------

def test_denominador_para_es_el_espejo_de_multiplos():
    """
    Round-trip: si con el precio en 15 el PER da X, entonces pedir el
    denominador que hace falta para un PER de X a 15 tiene que devolver el
    resultado original. Si el despeje esta mal, esto no cierra.
    """
    m = V.multiplos(BASE, 15.0)
    den = V.denominador_para(BASE, "pe_ratio", m["pe_ratio"], 15.0)
    assert den == pytest.approx(BASE["net_income_ttm"])


def test_denominador_para_resta_la_deuda_del_lado_correcto():
    """
    En las metricas sobre EV el numerador es market cap MAS deuda neta, no el
    market cap. A 15: mc 1.500, EV 1.700; para un EV/EBITDA de 10 hace falta
    un ebitda de 170, no de 150. Usar el market cap aca es el error clasico.
    """
    assert V.denominador_para(BASE, "ev_ebitda", 10.0, 15.0) == pytest.approx(170.0)
    m = V.multiplos(BASE, 15.0)
    vuelta = V.denominador_para(BASE, "ev_ebitda", m["ev_ebitda"], 15.0)
    assert vuelta == pytest.approx(BASE["ebitda_ttm"])


def test_sin_deuda_conocida_no_hay_denominador_sobre_ev():
    """Misma regla que en precio_para: un EV sin deuda neta no existe."""
    sin_deuda = dict(BASE, net_debt=None)
    assert V.denominador_para(sin_deuda, "ev_ebitda", 10.0, 15.0) is None
    assert V.denominador_para(sin_deuda, "pe_ratio", 20.0, 15.0) is not None


def test_el_fcf_requerido_admite_ser_negativo():
    """
    fcf_yield es el unico que se lee al reves y el unico que acepta objetivo
    negativo: quemar caja es informacion, no un dato invalido.
    """
    assert V.denominador_para(BASE, "fcf_yield", -0.02, 10.0) == pytest.approx(-20.0)


def test_un_multiplo_objetivo_no_positivo_no_produce_denominador():
    assert V.denominador_para(BASE, "pe_ratio", 0.0, 10.0) is None
    assert V.denominador_para(BASE, "pe_ratio", -5.0, 10.0) is None


def test_la_exigencia_es_la_razon_entre_el_multiplo_implicito_y_el_objetivo():
    """
    Identidad que amarra la inversa a la directa: el crecimiento que hace falta
    es exactamente multiplo_implicito / multiplo_objetivo - 1. Se verifica por
    separado porque exigencia() lo calcula por la via larga (despejando el
    denominador) y las dos cuentas tienen que coincidir.
    """
    historia = {"pe_ratio": [10.0, 20.0, 30.0, 40.0]}     # mediana 25
    ex = V.exigencia(BASE, historia, 15.0)["pe_ratio"]
    assert ex["multiplo_objetivo"] == pytest.approx(25.0)
    assert ex["multiplo_implicito"] == pytest.approx(30.0)   # 1.500 / 50
    assert ex["actual"] == pytest.approx(50.0)
    assert ex["requerido"] == pytest.approx(60.0)            # 1.500 / 25
    assert ex["crecimiento"] == pytest.approx(30.0 / 25.0 - 1.0)


def test_la_identidad_tambien_vale_sobre_ev():
    """
    La deuda neta se cancela al dividir, asi que la razon vale igual para las
    metricas sobre EV. Es lo que justifica leer la columna 'crece' sin
    preguntarse si el multiplo era sobre equity o sobre EV.
    """
    historia = {"ev_ebitda": [8.0, 10.0, 12.0]}              # mediana 10
    ex = V.exigencia(BASE, historia, 15.0)["ev_ebitda"]
    assert ex["crecimiento"] == pytest.approx(
        ex["multiplo_implicito"] / ex["multiplo_objetivo"] - 1.0)


def test_sin_historia_no_hay_exigencia():
    ex = V.exigencia(BASE, {}, 15.0)["pe_ratio"]
    assert ex["multiplo_objetivo"] is None
    assert ex["requerido"] is None
    assert ex["crecimiento"] is None


def test_el_crecimiento_historico_compara_contra_el_mismo_trimestre():
    """
    pasos=4 significa "el mismo trimestre del anio pasado". Con 100 en los 4
    primeros y 110/120 despues, los crecimientos son +10% y +20%.
    """
    serie = [("2023-03-31", 100.0), ("2023-06-30", 100.0),
             ("2023-09-30", 100.0), ("2023-12-31", 100.0),
             ("2024-03-31", 110.0), ("2024-06-30", 120.0)]
    assert V.crecimiento_historico(serie) == pytest.approx([0.10, 0.20])


def test_el_crecimiento_historico_saltea_huecos_y_bases_no_positivas():
    """
    Un None no es un cero y una base <= 0 no produce porcentaje: dividir por
    un patrimonio negativo devolveria un numero con signo invertido que se
    leeria como caida.
    """
    serie = [("q1", 100.0), ("q2", None), ("q3", -50.0), ("q4", 100.0),
             ("q5", 110.0), ("q6", 50.0), ("q7", 10.0), ("q8", 200.0)]
    assert V.crecimiento_historico(serie) == pytest.approx([0.10, 1.0])


def test_el_crecimiento_historico_necesita_un_anio_entero():
    serie = [("q1", 100.0), ("q2", 110.0), ("q3", 120.0)]
    assert V.crecimiento_historico(serie) == []


def test_el_roe_implicito_usa_el_patrimonio_de_hoy():
    """Resultado 100 contra patrimonio 400 -> 25%."""
    assert V.roe_implicito(BASE, 100.0) == pytest.approx(0.25)


def test_un_patrimonio_no_positivo_no_produce_roe():
    """
    Con patrimonio negativo el ROE cambia de signo y deja de significar lo que
    dice su nombre. Vale None antes que un numero que se lee al reves.
    """
    assert V.roe_implicito(dict(BASE, equity=-400.0), 100.0) is None
    assert V.roe_implicito(dict(BASE, equity=None), 100.0) is None
