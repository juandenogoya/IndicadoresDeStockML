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
def _hecho(val, end, start=None, filed="2026-01-01", tag=None, accn="x",
           form="10-Q"):
    h = {"val": val, "end": end, "filed": filed, "accn": accn, "form": form}
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


def test_profit_loss_no_se_usa_como_sinonimo_de_net_income():
    """
    ProfitLoss incluye minoritarios: NO es sinonimo de NetIncomeLoss.
    Usarlo como tal daba FCX +41% y CARR +11% contra la fuente de contraste.
    Puede entrar SOLO por la identidad ProfitLoss - minoritarios (abajo).
    """
    assert "ProfitLoss" not in sec_xbrl.FLUJO_ADITIVO["net_income"]
    facts = _facts(
        ProfitLoss=[_hecho(999, "2025-03-31", "2025-01-01")],
        NetIncomeLossAttributableToNoncontrollingInterest=[
            _hecho(99, "2025-03-31", "2025-01-01")])
    p = _periodo(sec_xbrl.normalizar(facts), "2025-03-31")
    assert p["net_income"] == 900          # 999 - 99, NO 999


def test_net_income_real_le_gana_a_la_identidad():
    """La derivacion solo rellena huecos; donde hay NetIncomeLoss, gana."""
    facts = _facts(
        NetIncomeLoss=[_hecho(800, "2025-03-31", "2025-01-01")],
        ProfitLoss=[_hecho(999, "2025-03-31", "2025-01-01")],
        NetIncomeLossAttributableToNoncontrollingInterest=[
            _hecho(99, "2025-03-31", "2025-01-01")])
    p = _periodo(sec_xbrl.normalizar(facts), "2025-03-31")
    assert p["net_income"] == 800
    assert p["net_income__tag"] == "NetIncomeLoss"
    assert p["net_income__derivado"] is False


def test_sin_tag_de_minoritarios_el_total_es_el_resultado():
    """
    Filer sin participaciones no controlantes: ProfitLoss ES el resultado.
    Medido sobre el cache: 0 tickers usan una variante del tag de minoritarios
    sin tener tambien el estandar, asi que su ausencia total significa que no
    hay minoritarios, no que estan escondidos con otro nombre.
    """
    facts = _facts(ProfitLoss=[_hecho(999, "2025-03-31", "2025-01-01")])
    p = _periodo(sec_xbrl.normalizar(facts), "2025-03-31")
    assert p["net_income"] == 999
    assert p["net_income__derivado"] is True


def test_minoritarios_ausentes_en_el_periodo_no_se_asumen_cero():
    """
    EL caso AVGO: tiene el tag de minoritarios pero no lo declara en casi
    ningun trimestre. Asumir cero ahi daba 11% de error contra el control.
    Un hueco se ve; un resultado inflado 11% se propaga al PER.
    """
    facts = _facts(
        ProfitLoss=[_hecho(999, "2025-03-31", "2025-01-01"),
                    _hecho(1000, "2025-06-30", "2025-04-01")],
        NetIncomeLossAttributableToNoncontrollingInterest=[
            _hecho(99, "2025-03-31", "2025-01-01")])
    r = sec_xbrl.normalizar(facts)
    assert _periodo(r, "2025-03-31")["net_income"] == 900
    # El Q2 no queda con un numero inventado. Ni siquiera llega a existir como
    # periodo: era el unico concepto de duracion que lo habria creado.
    assert all(p.get("net_income") is None
               for p in r["periodos"] if p["period_end"] == "2025-06-30")
    assert any(a["tipo"] == "net_income_sin_minoritarios" for a in r["avisos"])


def test_la_identidad_se_cruza_contra_el_resultado_para_comunes():
    """
    Control cruzado: ...AvailableToCommonStockholders mide el mismo renglon
    por otro camino. Si la identidad no da, no se emite y queda el aviso.
    """
    facts = _facts(
        ProfitLoss=[_hecho(999, "2025-03-31", "2025-01-01")],
        NetIncomeLossAttributableToNoncontrollingInterest=[
            _hecho(99, "2025-03-31", "2025-01-01")],
        NetIncomeLossAvailableToCommonStockholdersBasic=[
            _hecho(500, "2025-03-31", "2025-01-01")])     # 900 vs 500: no da
    r = sec_xbrl.normalizar(facts)
    assert "net_income" not in _periodo(r, "2025-03-31")
    assert any(a["tipo"] == "net_income_derivado_sin_control"
               for a in r["avisos"])


def test_la_identidad_tolera_los_dividendos_preferidos():
    """
    El control no mide EXACTAMENTE lo mismo: descuenta los preferidos. Una
    diferencia chica es esperable y no invalida la derivacion.
    """
    facts = _facts(
        ProfitLoss=[_hecho(1000, "2025-03-31", "2025-01-01")],
        NetIncomeLossAttributableToNoncontrollingInterest=[
            _hecho(0, "2025-03-31", "2025-01-01")],
        NetIncomeLossAvailableToCommonStockholdersBasic=[
            _hecho(980, "2025-03-31", "2025-01-01")])     # -2% de preferidos
    p = _periodo(sec_xbrl.normalizar(facts), "2025-03-31")
    assert p["net_income"] == 1000


def test_la_identidad_desacumula_como_cualquier_flujo():
    """
    ProfitLoss y los minoritarios se informan ACUMULADOS igual que el resto.
    Si no se desacumularan, el Q4 saldria con el valor del ejercicio entero.
    """
    facts = _facts(
        ProfitLoss=_anio("pl", q1=100, h1=250, m9=420, fy=600),
        NetIncomeLossAttributableToNoncontrollingInterest=_anio(
            "nci", q1=10, h1=25, m9=42, fy=60))
    r = sec_xbrl.normalizar(facts)
    assert _periodo(r, "2025-03-31")["net_income"] == 90     # 100 - 10
    assert _periodo(r, "2025-12-31")["net_income"] == 162    # (600-420)-(60-42)


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
                                 _hecho(7.46, "2025-12-31", "2025-01-01")],
        # Insumos del control cruzado: Q4 -> 184/100 = 1.84, concuerda con la
        # resta. Sin ellos el valor derivado no se emite (ver el test de
        # ponderado_sin_control).
        NetIncomeLoss=_anio("ni", 100, 250, 562, 746),
        WeightedAverageNumberOfDilutedSharesOutstanding=_anio(
            "sh", 100, 100, 100, 100))
    r = sec_xbrl.normalizar(facts)
    assert _periodo(r, "2025-12-31")["eps_diluted"] == pytest.approx(1.84, abs=1e-9)


def test_eps_derivado_sin_con_que_cruzarlo_no_se_emite():
    """
    La resta de acumulados es fragil: los dos tramos vienen de filings
    distintos y pueden estar en BASES DE SPLIT distintas (el mas nuevo se
    re-expresa, el viejo no). Caso real KLAC FY2025Q4: FY post-split 3.18 menos
    9M pre-split 21.4 dio eps=-18.28 en un trimestre que gano ~9 por accion.
    Ese valor paso porque las acciones se habian descartado por implausibles y
    el control se degradaba en silencio a "no control".
    """
    facts = _facts(
        EarningsPerShareDiluted=[_hecho(21.4, "2025-09-30", "2025-01-01"),
                                 _hecho(3.18, "2025-12-31", "2025-01-01")])
    r = sec_xbrl.normalizar(facts)
    # Descartado el unico concepto de duracion, no queda ni el periodo: el
    # indice lo definen los conceptos de duracion, no los avisos.
    assert all("eps_diluted" not in p for p in r["periodos"])
    assert any(a["tipo"] == "ponderado_sin_control" and
               a["concepto"] == "eps_diluted" for a in r["avisos"])


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


# ------------------------------------------ 8. fechas de publicacion --
def test_expone_desde_cuando_el_periodo_fue_publico():
    """
    Un trimestre no esta disponible el dia que cierra: se publica con el 10-Q,
    semanas despues (mediana medida 31-35 dias). Armar una serie historica con
    period_end en vez de filed_primero adelantaria cada trimestre mas de un mes.
    filed_primero es el filed MAS VIEJO, no el del hecho elegido -- que es la
    ultima reexpresion y puede ser anios posterior.
    """
    facts = _facts(Revenues=[
        _hecho(10, "2025-03-31", "2025-01-01", filed="2025-05-05"),
        _hecho(11, "2025-03-31", "2025-01-01", filed="2026-05-05")])   # reexpresion
    p = _periodo(sec_xbrl.normalizar(facts), "2025-03-31")
    assert p["revenue"] == 11              # el valor: la ultima presentacion
    assert p["filed_primero"] == "2025-05-05"
    assert p["filed_ultimo"] == "2026-05-05"


def test_descarta_publicacion_anterior_al_cierre():
    """
    INVARIANTE: un periodo no puede ser publico antes de terminar. Hay hechos
    con fecha de cierre futura en filings anteriores (proyecciones); tomarlos
    daria lag negativo -- hasta -309 dias en un caso real -- y eso es
    lookahead.
    """
    facts = _facts(Revenues=[
        _hecho(10, "2025-03-31", "2025-01-01", filed="2024-02-01"),   # imposible
        _hecho(10, "2025-03-31", "2025-01-01", filed="2025-05-05")])
    p = _periodo(sec_xbrl.normalizar(facts), "2025-03-31")
    assert p["filed_primero"] == "2025-05-05"


# ------------------------------- 8b. de que formulario sale cada numero --
def test_el_reporte_periodico_le_gana_al_proxy_aunque_sea_mas_viejo():
    """
    "Gana la presentacion mas reciente" es correcto entre reportes periodicos
    y desastroso fuera de ellos. El DEF 14A -- el proxy de compensacion --
    taguea NetIncomeLoss en su tabla Pay Versus Performance y se presenta
    DESPUES del 10-K, asi que ganaba.

    Caso real: Schwab publica su resultado anual en MILES en el proxy. Se
    tomaba 8,9 MM en vez de 8.852,0 MM, en los cinco ejercicios, y como el Q4
    se deriva restando 9M al anual, el Q4 salia -6.384 MM cuando el real fue
    +2.459 MM.

    Los numeros son los de Schwab 2025: 1.909 + 2.126 + 2.358 acumulan 6.393
    a los nueve meses, y el anual real es 8.852.
    """
    facts = _facts(NetIncomeLoss=[
        _hecho(1909.0, "2025-03-31", "2025-01-01", filed="2025-05-05"),
        _hecho(4035.0, "2025-06-30", "2025-01-01", filed="2025-08-05"),
        _hecho(6393.0, "2025-09-30", "2025-01-01", filed="2025-11-05"),
        _hecho(8852.0, "2025-12-31", "2025-01-01",
               filed="2026-02-25", form="10-K"),
        _hecho(8.9, "2025-12-31", "2025-01-01",
               filed="2026-04-06", form="DEF 14A")])
    r = sec_xbrl.normalizar(facts)
    # Con el proxy ganando, este Q4 daba 8,9 - 6.393 = -6.384.
    assert _periodo(r, "2025-12-31")["net_income"] == pytest.approx(2459.0)


def test_si_el_periodo_SOLO_esta_en_un_8K_igual_se_usa():
    """
    El rango de formulario decide entre COMPETIDORES; no descarta datos. Un
    trimestre que solo existe en un comunicado de resultados sigue valiendo:
    la alternativa es un hueco, y el 8-K es la misma empresa informando.
    """
    facts = _facts(Revenues=[_hecho(42, "2025-03-31", "2025-01-01", form="8-K")])
    assert _periodo(sec_xbrl.normalizar(facts), "2025-03-31")["revenue"] == 42


def test_entre_dos_periodicos_sigue_ganando_el_mas_reciente():
    """La regla de la reexpresion no se toca donde si corresponde."""
    facts = _facts(Revenues=[
        _hecho(10, "2025-03-31", "2025-01-01", filed="2025-05-05", form="10-Q"),
        _hecho(11, "2025-03-31", "2025-01-01", filed="2026-02-20", form="10-K")])
    assert _periodo(sec_xbrl.normalizar(facts), "2025-03-31")["revenue"] == 11


# ------------------------------------- 9. tags curados y mezcla de tags --
def _mezclado():
    """
    Un ejercicio armado con dos tags: los tres 10-Q publican el trimestre bajo
    RevenueFromContractWithCustomer* y el Q4 sale del 10-K, que solo trae la
    cadena de Revenues. Es la forma exacta en que se manifiesta el defecto en
    los datos reales.
    """
    return _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=[
            _hecho(10, "2025-03-31", "2025-01-01"),
            _hecho(20, "2025-06-30", "2025-04-01"),
            _hecho(30, "2025-09-30", "2025-07-01")],
        Revenues=_anio("rev", 100, 200, 300, 400))


def test_sin_curar_gana_la_prioridad_del_sinonimo():
    """Punto de partida: el tag moderno va primero en FLUJO_ADITIVO y gana."""
    r = sec_xbrl.normalizar(_mezclado())
    assert _periodo(r, "2025-03-31")["revenue"] == 10
    assert _periodo(r, "2025-12-31")["revenue"] == 100      # 400 - 300, otro tag


def test_tag_curado_reemplaza_la_lista_de_sinonimos():
    """
    Con dos tags presentes gana el CURADO aunque el otro tenga mas prioridad.
    Es el caso de AMT: Revenues son las ventas totales (10.645 MM en 2025) y
    RevenueFromContractWithCustomerExcludingAssessedTax es un renglon chico
    (936 MM), pero este ultimo va primero por ser el tag moderno.
    """
    r = sec_xbrl.normalizar(_mezclado(), tags_curados={"revenue": "Revenues"})
    assert _periodo(r, "2025-03-31")["revenue"] == 100
    assert _periodo(r, "2025-03-31")["revenue__tag"] == "Revenues"


def _avisos_de(res, *tipos):
    return [a for a in res["avisos"] if a["tipo"] in tipos]


def test_curar_silencia_el_aviso_de_mezcla():
    """
    El aviso desaparece porque desaparece la CAUSA, no porque se suprima: con
    el tag curado los 4 trimestres salen del mismo concepto.

    En este armado solo Revenues publica anual, asi que la mezcla no se puede
    comprobar y cae en el aviso debil. Ver los tests de abajo para los tres
    casos de la clasificacion.
    """
    sin_curar = sec_xbrl.normalizar(_mezclado())
    debiles = _avisos_de(sin_curar, "mezcla_no_verificable")
    assert len(debiles) == 1
    assert debiles[0]["concepto"] == "revenue"
    assert debiles[0]["fiscal_year"] == 2025

    curado = sec_xbrl.normalizar(_mezclado(), tags_curados={"revenue": "Revenues"})
    assert _avisos_de(curado, "mezcla_en_ejercicio", "mezcla_no_verificable") == []


# Los tres casos de la clasificacion se prueban sobre la funcion directamente:
# armar por el pipeline entero un ejercicio donde DOS tags publiquen anual y
# ademas se mezclen en los trimestres exige una combinacion de cadenas que
# oscurece lo que se esta probando, que es la comparacion de los anuales.
def _entrada(*anuales):
    """(periodos, anuales) de un ejercicio 2025 mezclado entre dos tags."""
    periodos = [
        {"period_end": "2025-03-31", "fiscal_year": 2025, "revenue__tag": "Revenues"},
        {"period_end": "2025-12-31", "fiscal_year": 2025,
         "revenue__tag": "RevenueFromContractWithCustomerExcludingAssessedTax"},
    ]
    tags = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
    hechos = {t: {"val": v, "filed": "2026-02-01"}
              for t, v in zip(tags, anuales) if v is not None}
    return periodos, {"revenue": {"2025-12-31": hechos}}


def test_la_mezcla_avisa_cuando_los_anuales_difieren():
    """
    El defecto real: los dos tags miden cosas distintas, asi que los 4
    trimestres ya no son sumables. Es el caso de AMT (10.645 vs 936 MM).
    """
    periodos, anuales = _entrada(10645.0, 936.0)
    avisos = sec_xbrl._avisos_mezcla_en_ejercicio(periodos, ["revenue"], anuales)
    assert [a["tipo"] for a in avisos] == ["mezcla_en_ejercicio"]
    assert avisos[0]["fiscal_year"] == 2025


def test_la_mezcla_entre_sinonimos_redundantes_no_avisa():
    """
    Si los dos tags publican el mismo anual son sinonimos y la suma no cambia.
    Medido: 109 de 126 ejercicios que mezclan tags caen aca (JPM, GOOG, COST).
    Avisar de todos condenaria el aviso a que nadie lo lea, que es justamente
    lo que le paso a `cambio_de_tag`.
    """
    periodos, anuales = _entrada(10645.0, 10645.0)
    assert sec_xbrl._avisos_mezcla_en_ejercicio(periodos, ["revenue"], anuales) == []


def test_la_mezcla_sin_dos_anuales_avisa_mas_debil():
    """
    Un solo tag publica anual: no hay con que comparar. No se calla (podria
    ser el defecto real) pero tampoco se declara error.
    """
    periodos, anuales = _entrada(10645.0, None)
    avisos = sec_xbrl._avisos_mezcla_en_ejercicio(periodos, ["revenue"], anuales)
    assert [a["tipo"] for a in avisos] == ["mezcla_no_verificable"]


def test_el_tag_curado_deja_hueco_donde_falta_en_vez_de_mezclar():
    """
    El curado REEMPLAZA la lista, no se antepone. Si el tag elegido no cubre
    un trimestre, ese trimestre queda vacio. Anteponerlo dejaria a los otros
    de respaldo y volveria a mezclar justo donde el elegido falta -- que es el
    caso que la curacion viene a resolver. El hueco se ve; el numero mezclado
    no.
    """
    facts = _facts(
        Revenues=[_hecho(100, "2025-03-31", "2025-01-01")],
        RevenueFromContractWithCustomerExcludingAssessedTax=[
            _hecho(7, "2025-06-30", "2025-04-01")])
    r = sec_xbrl.normalizar(facts, tags_curados={"revenue": "Revenues"})
    assert _periodo(r, "2025-03-31")["revenue"] == 100
    assert all(p.get("revenue") is None for p in r["periodos"]
               if p["period_end"] == "2025-06-30")


def test_el_curado_acepta_una_lista_para_migraciones_legitimas():
    """
    Con ASC 606 (2019) medio universo migro de SalesRevenueNet al tag nuevo:
    ningun tag solo cubre toda la ventana y la lista es la salida correcta.
    El orden de la lista sigue mandando -- Revenues no entra por no estar.
    """
    facts = _facts(
        SalesRevenueNet=[_hecho(50, "2018-03-31", "2018-01-01")],
        Revenues=[_hecho(999, "2018-03-31", "2018-01-01")],
        RevenueFromContractWithCustomerExcludingAssessedTax=[
            _hecho(60, "2019-03-31", "2019-01-01")])
    r = sec_xbrl.normalizar(facts, tags_curados={"revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet"]})
    assert _periodo(r, "2018-03-31")["revenue"] == 50       # no 999
    assert _periodo(r, "2019-03-31")["revenue"] == 60


def test_el_cambio_de_tag_entre_ejercicios_no_es_mezcla():
    """
    Migrar de taxonomia es legitimo: cada ejercicio queda armado con UN tag y
    sus 4 trimestres siguen siendo sumables. Se emite cambio_de_tag, que es
    informativo, pero NO mezcla_en_ejercicio, que senala un error. Distinguir
    los dos casos es el punto entero del aviso nuevo.
    """
    facts = _facts(
        SalesRevenueNet=_anio("rev", 10, 20, 30, 40, ini="2018-01-01",
                              fines=("2018-03-31", "2018-06-30",
                                     "2018-09-30", "2018-12-31")),
        Revenues=_anio("rev", 100, 200, 300, 400))
    r = sec_xbrl.normalizar(facts)
    tipos = {a["tipo"] for a in r["avisos"] if a.get("concepto") == "revenue"}
    assert "cambio_de_tag" in tipos
    assert "mezcla_en_ejercicio" not in tipos


def test_la_mezcla_no_se_mira_en_los_conceptos_instantaneos():
    """
    Un instantaneo (balance) no se suma entre trimestres, asi que dos tags
    distintos ahi no producen el error que este aviso persigue: no hay
    ninguna suma que quede armada con dos magnitudes.
    """
    facts = _facts(
        Assets=[_hecho(1000, "2025-03-31"), _hecho(1100, "2025-06-30")],
        Revenues=_anio("rev", 100, 200, 300, 400))
    r = sec_xbrl.normalizar(facts)
    assert not [a for a in r["avisos"]
                if a["tipo"] == "mezcla_en_ejercicio" and a["concepto"] == "assets"]


# ------------------------------------- 12. los tags de deuda como fallback --
#
# El orden de INSTANTE["debt_long"] esta al reves de lo intuitivo porque
# _elegir se queda con el `pri` MAS GRANDE. Estos dos tests fijan la relacion
# para que, si alguien arregla _elegir sin dar vuelta la lista, falle en vez
# de invertir el comportamiento en silencio.

def _con_deuda(**tags_de_deuda):
    """Un ejercicio calendario minimo mas los instantes de deuda que se pidan."""
    base = {"Revenues": _anio("Revenues", 10, 20, 30, 40)}
    for tag, val in tags_de_deuda.items():
        base[tag] = [_hecho(val, "2025-12-31")]
    return _facts(**base)


def test_el_tag_preferido_le_gana_al_fallback():
    """
    Con los dos presentes gana LongTermDebtNoncurrent, no el fallback nuevo.
    Si este test empieza a fallar es porque se toco _elegir: hay que DAR
    VUELTA la lista de INSTANTE["debt_long"], no cambiar el test.
    """
    r = sec_xbrl.normalizar(_con_deuda(
        LongTermDebtNoncurrent=500,
        LongTermDebtAndCapitalLeaseObligations=900))
    p = _periodo(r, "2025-12-31")
    assert p["debt_long"] == 500
    assert p["debt_long__tag"] == "LongTermDebtNoncurrent"


def test_el_fallback_entra_cuando_no_esta_el_preferido():
    """
    Es la razon de ser del agregado: VZ y T publican su deuda consolidada con
    este tag y quedaban sin deuda neta, y por lo tanto sin EV ni EV/EBITDA.
    """
    r = sec_xbrl.normalizar(_con_deuda(
        LongTermDebtAndCapitalLeaseObligations=900))
    p = _periodo(r, "2025-12-31")
    assert p["debt_long"] == 900
    assert p["debt_long__tag"] == "LongTermDebtAndCapitalLeaseObligations"


def test_la_deuda_total_combinada_no_cuenta_como_deuda_larga():
    """
    DebtLongtermAndShorttermCombinedAmount es la deuda TOTAL. Si entrara como
    sinonimo de debt_long, net_debt = debt_short + debt_long - cash contaria
    el corto plazo DOS VECES. Debe quedar en None.
    """
    r = sec_xbrl.normalizar(_con_deuda(
        DebtLongtermAndShorttermCombinedAmount=1500))
    p = _periodo(r, "2025-12-31")
    # Un concepto sin ningun hecho ni siquiera aparece como clave, que es lo
    # que se quiere: hueco visible, no un cero silencioso.
    assert p.get("debt_long") is None


def test_el_tag_corto_preferido_le_gana_al_fallback():
    """
    Mismo contrato que en debt_long: con los dos presentes gana DebtCurrent.
    Si falla, se toco _elegir y hay que dar vuelta INSTANTE["debt_short"].
    """
    r = sec_xbrl.normalizar(_con_deuda(
        DebtCurrent=100,
        LongTermDebtAndCapitalLeaseObligationsCurrent=700))
    p = _periodo(r, "2025-12-31")
    assert p["debt_short"] == 100


def test_el_fallback_corto_entra_cuando_no_esta_el_preferido():
    """HD, KO, LOW, TGT y CVS tenian la deuda larga y no la corta."""
    r = sec_xbrl.normalizar(_con_deuda(
        LongTermDebtAndCapitalLeaseObligationsCurrent=700))
    p = _periodo(r, "2025-12-31")
    assert p["debt_short"] == 700
