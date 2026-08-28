"""
test_sec_xbrl.py -- tests del modulo PURO src/utils/sec_xbrl.py.

Casos SINTETICOS: no tocan la DB, no salen a la red y no dependen del cache
de companyfacts. Cada test construye a mano el minimo de hechos XBRL que
reproduce un problema real observado en los datos de SEC.

Los numeros de los casos "reales" salen de mediciones documentadas en
docs/fuentes_fundamentales.md.
"""

import pytest

from src.utils import sec_xbrl


# --------------------------------------------------------------- helpers --
def _hecho(val, end, start=None, filed="2026-01-01", tag=None, accn="x"):
    h = {"val": val, "end": end, "filed": filed, "accn": accn, "form": "10-Q"}
    if start:
        h["start"] = start
    return h


def _facts(**por_tag):
    """
    _facts(NetIncomeLoss=[hecho, ...], Assets=[...]) -> dict companyfacts.
    La unidad no importa para la normalizacion; se usa USD para todos.
    """
    return {"entityName": "TEST CORP", "cik": 1,
            "facts": {"us-gaap": {tag: {"units": {"USD": hechos}}
                                  for tag, hechos in por_tag.items()}}}


def _anio(tag, q1, h1, m9, fy, ini="2025-01-01",
          fines=("2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31")):
    """Cadena de acumulados de un ejercicio calendario: 3m, 6m, 9m, 12m."""
    return [_hecho(q1, fines[0], ini), _hecho(h1, fines[1], ini),
            _hecho(m9, fines[2], ini), _hecho(fy, fines[3], ini)]


def _periodo(res, fecha):
    for p in res["periodos"]:
        if p["period_end"] == fecha:
            return p
    return None


# ------------------------------------------------ 1. desacumulacion YTD --
def test_desacumula_trimestres_desde_acumulados():
    """
    El estado de flujo se informa ACUMULADO en el ejercicio. Sin desacumular,
    cfo/capex/d_and_a quedaban con ~la mitad de trimestres que el resto
    (mediana 36 vs 70). Q2 = H1 - Q1, Q3 = 9M - H1, Q4 = FY - 9M.
    """
    facts = _facts(NetCashProvidedByUsedInOperatingActivities=_anio(
        "cfo", q1=100, h1=250, m9=420, fy=600))
    r = sec_xbrl.normalizar(facts)

    assert _periodo(r, "2025-03-31")["cfo"] == 100      # publicado directo
    assert _periodo(r, "2025-06-30")["cfo"] == 150      # 250 - 100
    assert _periodo(r, "2025-09-30")["cfo"] == 170      # 420 - 250
    assert _periodo(r, "2025-12-31")["cfo"] == 180      # 600 - 420
    assert _periodo(r, "2025-12-31")["cfo__derivado"] is True
    assert _periodo(r, "2025-03-31")["cfo__derivado"] is False


def test_q4_se_deriva_del_anual():
    """El 10-K trae el anio entero y nunca desagrega el Q4."""
    facts = _facts(Revenues=[_hecho(10, "2025-03-31", "2025-01-01"),
                             _hecho(30, "2025-09-30", "2025-01-01"),
                             _hecho(45, "2025-12-31", "2025-01-01")])
    r = sec_xbrl.normalizar(facts)
    assert _periodo(r, "2025-12-31")["revenue"] == 15   # 45 - 30
    assert _periodo(r, "2025-12-31")["revenue__derivado"] is True


def test_el_trimestre_publicado_le_gana_al_derivado():
    """Si la empresa publico el trimestre suelto, ese manda."""
    facts = _facts(Revenues=_anio("rev", 10, 25, 42, 60) +
                   [_hecho(99, "2025-06-30", "2025-04-01")])   # Q2 directo
    r = sec_xbrl.normalizar(facts)
    p = _periodo(r, "2025-06-30")
    assert p["revenue"] == 99
    assert p["revenue__derivado"] is False


# ----------------------------------------------------- 2. restatements --
def test_gana_el_filed_mas_reciente():
    """Ante el mismo periodo reexpresado, se toma la ultima presentacion."""
    facts = _facts(NetIncomeLoss=[
        _hecho(100, "2025-03-31", "2025-01-01", filed="2025-05-01"),
        _hecho(105, "2025-03-31", "2025-01-01", filed="2025-11-01")])
    r = sec_xbrl.normalizar(facts)
    assert _periodo(r, "2025-03-31")["net_income"] == 105


def test_point_in_time_devuelve_lo_que_se_sabia_entonces():
    """
    hasta_filed ignora lo presentado despues: es la unica capacidad que SEC
    tiene y las fuentes comerciales no.
    """
    facts = _facts(NetIncomeLoss=[
        _hecho(100, "2025-03-31", "2025-01-01", filed="2025-05-01"),
        _hecho(105, "2025-03-31", "2025-01-01", filed="2025-11-01")])
    r = sec_xbrl.normalizar(facts, hasta_filed="2025-08-01")
    assert _periodo(r, "2025-03-31")["net_income"] == 100


# ----------------------------------------------------- 3. sinonimos/tags --
def test_gana_el_tag_de_mayor_prioridad():
    """El primero de la lista de candidatos es el preferido."""
    facts = _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=[
            _hecho(10, "2025-03-31", "2025-01-01")],
        Revenues=[_hecho(20, "2025-03-31", "2025-01-01")])
    r = sec_xbrl.normalizar(facts)
    p = _periodo(r, "2025-03-31")
    assert p["revenue"] == 10
    assert p["revenue__tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_avisa_cuando_la_serie_cambia_de_tag():
    """
    Red de seguridad contra el error SILENCIOSO. Los tres casos reales que se
    colaron (ProfitLoss, ...Domestic, EPS por resta) se manifestaron igual:
    dos tags distintos dentro de la misma serie.
    """
    facts = _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=[
            _hecho(10, "2025-03-31", "2025-01-01")],
        Revenues=[_hecho(20, "2025-06-30", "2025-04-01")])
    r = sec_xbrl.normalizar(facts)
    avisos = [a for a in r["avisos"]
              if a["tipo"] == "cambio_de_tag" and a["concepto"] == "revenue"]
    assert len(avisos) == 1
    assert len(avisos[0]["tags"]) == 2


def test_profit_loss_no_se_usa_como_net_income():
    """
    ProfitLoss incluye minoritarios: NO es sinonimo de NetIncomeLoss.
    Usarlo daba FCX +41% y CARR +11% contra la fuente de contraste.
    """
    assert "ProfitLoss" not in sec_xbrl.FLUJO_ADITIVO["net_income"]
    facts = _facts(ProfitLoss=[_hecho(999, "2025-03-31", "2025-01-01")])
    r = sec_xbrl.normalizar(facts)
    assert all("net_income" not in p for p in r["periodos"])


def test_pretax_no_toma_el_segmento_domestico():
    """'...BeforeIncomeTaxesDomestic' es un SEGMENTO, no el total."""
    assert not any("Domestic" in t
                   for t in sec_xbrl.FLUJO_ADITIVO["pretax_income"])


# --------------------------------------- 4. ponderados: acciones vs EPS --
def test_acciones_usan_algebra_de_promedio_ponderado():
    """
    El numero de acciones SI es un promedio ponderado: el acumulado de 9 meses
    es el promedio de esos 9 meses. Q4 = 4*FY - 3*9M.
    Con acciones constantes en 100, todos los trimestres deben dar 100.
    """
    facts = _facts(WeightedAverageNumberOfDilutedSharesOutstanding=_anio(
        "sh", q1=100, h1=100, m9=100, fy=100))
    r = sec_xbrl.normalizar(facts)
    assert _periodo(r, "2025-12-31")["shares_diluted"] == pytest.approx(100)


def test_eps_se_deriva_por_resta_no_por_promedio():
    """
    A pesar del nombre, el EPS acumulado NO es un promedio: se comporta como
    la SUMA de los trimestrales. Caso real AAPL FY2025: EPS anual 7.46, EPS de
    9 meses 5.62, Q4 real 1.85. La resta da 1.84; la formula ponderada
    (4*7.46 - 3*5.62) daria 12.98.
    """
    facts = _facts(
        EarningsPerShareDiluted=[_hecho(5.62, "2025-09-30", "2025-01-01"),
                                 _hecho(7.46, "2025-12-31", "2025-01-01")])
    r = sec_xbrl.normalizar(facts)
    assert _periodo(r, "2025-12-31")["eps_diluted"] == pytest.approx(1.84, abs=1e-9)


def test_eps_se_descarta_si_no_concuerda_con_resultado_sobre_acciones():
    """
    Control cruzado: la resta de acumulados se contrasta contra el metodo
    independiente resultado/acciones. Si discrepan, hueco + aviso.
    Preferible un hueco a un numero equivocado que se propague al PER.
    """
    facts = _facts(
        EarningsPerShareDiluted=[_hecho(5.0, "2025-09-30", "2025-01-01"),
                                 _hecho(7.0, "2025-12-31", "2025-01-01")],
        # resta -> 2.0, pero resultado/acciones -> 500/100 = 5.0
        NetIncomeLoss=_anio("ni", 100, 200, 300, 800),
        WeightedAverageNumberOfDilutedSharesOutstanding=_anio(
            "sh", 100, 100, 100, 100))
    r = sec_xbrl.normalizar(facts)
    assert "eps_diluted" not in _periodo(r, "2025-12-31")
    assert any(a["tipo"] == "ponderado_discordante" and
               a["concepto"] == "eps_diluted" for a in r["avisos"])


def test_acciones_fuera_de_banda_se_descartan():
    """Un split rompe la equivalencia del promedio: mejor hueco que numero malo."""
    facts = _facts(WeightedAverageNumberOfDilutedSharesOutstanding=[
        _hecho(100, "2025-09-30", "2025-01-01"),
        _hecho(1000, "2025-12-31", "2025-01-01")])   # 4*1000-3*100 = 3700
    r = sec_xbrl.normalizar(facts)
    assert all("shares_diluted" not in p for p in r["periodos"])
    assert any(a["tipo"] == "ponderado_implausible" for a in r["avisos"])


def test_no_resta_acumulados_no_adyacentes():
    """
    Si falta un tramo intermedio, la diferencia abarca DOS trimestres y
    quedaria imputada a uno solo. Sin Q3 (9M), el Q4 no se puede derivar del
    semestre: 60 - 25 son dos trimestres, no uno.
    """
    facts = _facts(Revenues=[_hecho(10, "2025-03-31", "2025-01-01"),
                             _hecho(25, "2025-06-30", "2025-01-01"),
                             _hecho(60, "2025-12-31", "2025-01-01")])
    r = sec_xbrl.normalizar(facts)
    assert _periodo(r, "2025-06-30")["revenue"] == 15      # H1 - Q1, adyacente
    assert all("revenue" not in p for p in r["periodos"]
               if p["period_end"] == "2025-12-31")


def test_eps_usa_resultado_de_accionistas_comunes():
    """
    El EPS se calcula sobre el resultado atribuible a los COMUNES. Usar el
    total hacia fallar el control cruzado en todo el sector financiero
    (JPM: 36 de 73 trimestres descartados por los dividendos preferidos).
    """
    facts = _facts(
        EarningsPerShareDiluted=[_hecho(3.0, "2025-09-30", "2025-01-01"),
                                 _hecho(5.0, "2025-12-31", "2025-01-01")],
        NetIncomeLoss=_anio("ni", 100, 200, 300, 700),          # total: Q4=400
        NetIncomeLossAvailableToCommonStockholdersBasic=_anio(
            "nic", 100, 200, 300, 500),                          # comunes: Q4=200
        WeightedAverageNumberOfDilutedSharesOutstanding=_anio(
            "sh", 100, 100, 100, 100))
    r = sec_xbrl.normalizar(facts)
    # resta -> 2.0 ; comunes/acciones -> 200/100 = 2.0 (concuerdan)
    # si usara el total: 400/100 = 4.0 y lo habria descartado
    assert _periodo(r, "2025-12-31")["eps_diluted"] == pytest.approx(2.0)


# ------------------------------------------------- 5. calendario fiscal --
def test_etiqueta_trimestres_de_ejercicio_no_calendario():
    """
    El ejercicio no se puede asumir por mes: AAPL cierra en septiembre, WMT el
    31 de enero, NVDA a fines de enero. Se deduce de los hechos anuales.
    """
    fines = ("2025-04-30", "2025-07-31", "2025-10-31", "2026-01-31")
    facts = _facts(Revenues=_anio("rev", 10, 25, 42, 60,
                                  ini="2025-02-01", fines=fines))
    r = sec_xbrl.normalizar(facts)
    assert (_periodo(r, "2025-04-30")["fiscal_year"],
            _periodo(r, "2025-04-30")["fiscal_quarter"]) == (2026, 1)
    assert _periodo(r, "2025-10-31")["fiscal_quarter"] == 3
    assert _periodo(r, "2026-01-31")["fiscal_quarter"] == 4


def test_etiqueta_el_ejercicio_en_curso():
    """
    El ejercicio en curso todavia no tiene hecho anual (sale con el 10-K).
    Sin proyectarlo, los trimestres mas recientes -- los que interesan --
    quedaban sin etiquetar.
    """
    facts = _facts(Revenues=_anio("rev", 10, 25, 42, 60) +
                   [_hecho(12, "2026-03-31", "2026-01-01")])
    r = sec_xbrl.normalizar(facts)
    p = _periodo(r, "2026-03-31")
    assert p["fiscal_year"] == 2026
    assert p["fiscal_quarter"] == 1


# ------------------------------------------------------- 6. instantaneos --
def test_instantaneo_no_crea_periodos_fantasma():
    """
    EntityCommonStockSharesOutstanding viene fechado en la PORTADA del filing,
    semanas despues del cierre. Si definiera periodos, generaria una fila por
    filing. El indice lo fijan solo los conceptos de duracion.
    """
    facts = _facts(Revenues=[_hecho(10, "2025-03-31", "2025-01-01")])
    facts["facts"]["dei"] = {"EntityCommonStockSharesOutstanding":
                             {"units": {"shares": [_hecho(500, "2025-04-18")]}}}
    r = sec_xbrl.normalizar(facts)
    assert [p["period_end"] for p in r["periodos"]] == ["2025-03-31"]
    # ...pero el valor si se engancha al trimestre por cercania
    assert _periodo(r, "2025-03-31")["shares_out"] == 500


def test_instantaneo_lejano_no_se_engancha():
    """Mas alla de la tolerancia no se asocia: no inventar un balance ajeno."""
    facts = _facts(Revenues=[_hecho(10, "2025-03-31", "2025-01-01")],
                   Assets=[_hecho(999, "2025-09-30")])
    r = sec_xbrl.normalizar(facts)
    assert "assets" not in _periodo(r, "2025-03-31")


# ------------------------------------------------------------ 7. bordes --
def test_companyfacts_vacio_no_explota():
    r = sec_xbrl.normalizar({})
    assert r["periodos"] == []
    assert r["avisos"] == []


def test_desde_recorta_la_salida():
    facts = _facts(Revenues=_anio("rev", 10, 25, 42, 60))
    r = sec_xbrl.normalizar(facts, desde="2025-07-01")
    assert [p["period_end"] for p in r["periodos"]] == ["2025-09-30", "2025-12-31"]


def test_no_rellena_conceptos_ausentes():
    """Un concepto que no esta NO aparece como clave: nunca se pone cero."""
    facts = _facts(Revenues=[_hecho(10, "2025-03-31", "2025-01-01")])
    p = _periodo(sec_xbrl.normalizar(facts), "2025-03-31")
    assert "net_income" not in p
    assert "assets" not in p
