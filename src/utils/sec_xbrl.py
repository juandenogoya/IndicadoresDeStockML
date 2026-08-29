"""
sec_xbrl.py
Normalizador de SEC XBRL (companyfacts) -> serie TRIMESTRAL por ticker.

Funcion PURA: sin DB, sin red, sin config, solo stdlib. Recibe el dict de
companyfacts ya descargado y devuelve periodos trimestrales normalizados.
ASCII en strings (regla del proyecto).

Contexto y evidencia: docs/fuentes_fundamentales.md

--------------------------------------------------------------------------
POR QUE ESTO ES NECESARIO
--------------------------------------------------------------------------
SEC XBRL no publica estados contables: publica HECHOS sueltos
(taxonomia -> tag -> unidad -> [hechos]). JPM declara 918 tags. Armar una
serie trimestral comparable exige resolver cuatro problemas:

  1. SINONIMOS. Cada empresa elige su tag. Ninguno cubre el universo: la
     union de 3 tags de revenue cubre 118 de 123 empresas medidas.
  2. LOS TAGS CAMBIAN DENTRO DE LA MISMA EMPRESA. NVDA dejo de usar
     RevenueFromContractWithCustomer... en 2022, UNH cambio StockholdersEquity
     en 2015, WMT CommonStockSharesOutstanding en 2012. No siguieron un tag:
     cambiaron. Por eso el candidato se resuelve POR PERIODO, no por ticker.
  3. EL TRIMESTRE NO SIEMPRE SE PUBLICA SUELTO. El estado de resultados y el
     de flujo se informan ACUMULADOS en el ejercicio (3m, 6m, 9m, 12m), y el
     10-K trae el anio entero sin desagregar el Q4. Hay que desacumular.
  4. RESTATEMENTS. El mismo trimestre aparece en varios filings. Hay que
     elegir uno -- y esa multiplicidad ES lo que habilita el point-in-time.

--------------------------------------------------------------------------
REGLA APRENDIDA A LOS GOLPES (no relajarla)
--------------------------------------------------------------------------
La lista de candidatos debe contener SOLO SINONIMOS VERDADEROS. Un tag
parecido devuelve un numero PLAUSIBLE DE OTRA COSA, y eso no falla ni avisa.
Tres casos reales detectados solo por cruce contra otra fuente:

  - ProfitLoss como sinonimo de NetIncomeLoss  -> incluye minoritarios
    (FCX +41%, CARR +11%). NO es el mismo renglon.
  - IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic como pretax
    -> es el SEGMENTO domestico, no el total (CRWD, SNOW, PATH, AVAV).
  - EPS y acciones del Q4 derivados por resta -> son promedios PONDERADOS.

Por eso `normalizar()` devuelve `avisos`: cambios de tag dentro de una serie
y derivaciones que no superaron su control cruzado. Un aviso NO es un error;
es "esto merece que alguien lo mire".
"""

from collections import defaultdict
from datetime import date

# ---------------------------------------------------------------------------
# Clasificacion de conceptos. El tipo determina COMO se desacumula.
# ---------------------------------------------------------------------------
# ADITIVO: el trimestre se obtiene restando acumulados (Q2 = H1 - Q1, etc.)
FLUJO_ADITIVO = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenuesNetOfInterestExpense", "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet", "InterestAndDividendIncomeOperating"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold",
                        "CostOfGoodsSold", "CostOfServices"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "operating_expense": ["OperatingExpenses", "CostsAndExpenses"],
    "sga": ["SellingGeneralAndAdministrativeExpense",
            "GeneralAndAdministrativeExpense"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    # OJO: '...BeforeIncomeTaxesDomestic' NO va aca (es el segmento domestico).
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax_provision": ["IncomeTaxExpenseBenefit"],
    # OJO: ProfitLoss NO va aca (incluye minoritarios).
    "net_income": ["NetIncomeLoss"],
    # Diluted primero: es la base del EPS diluido, que es contra lo que se
    # cruza. Basic queda como respaldo -- difieren solo por el efecto dilutivo,
    # y cualquiera de los dos es mucho mejor que el resultado total, que no
    # descuenta los dividendos preferidos. Medido: Basic 66/147, Diluted 54/147.
    "net_income_common": ["NetIncomeLossAvailableToCommonStockholdersDiluted",
                          "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating"],
    # net_interest_income (margen financiero de bancos) NO esta aca a
    # proposito. El tag mas frecuente es InterestIncomeExpenseNonoperatingNet
    # (34/147), pero eso es el neto de intereses de una empresa NO financiera
    # -- caja e intereses de deuda -- y no el margen financiero de un banco.
    # Mezclarlos seria exactamente el error de ProfitLoss: un numero plausible
    # de otra cosa. El bloque bancario esta diferido (ver doc, seccion 10).
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
    # OJO: 'Depreciation' a secas NO va aca. Es la amortizacion de bienes de
    # uso SOLA, sin la de intangibles, y por lo tanto no es un sinonimo de D&A
    # sino un subconjunto -- el mismo error de categoria que ProfitLoss para
    # net_income. Medido sobre 337 ejercicios de las 60 empresas que publican
    # los dos tags, Depreciation es la MEDIANA del 73% de la D&A completa, y en
    # 28 de esas 60 esta por debajo del 70% (SPGI 10%, AMGN 17%, WBD 17%,
    # HL 0%). Como fallback entraba en 429 de las 476 mezclas de tags del
    # concepto, subestimaba la D&A, y por lo tanto el EBITDA, y por lo tanto
    # SOBREESTIMABA el EV/EBITDA en silencio.
    # Sacarlo cuesta que 17 tickers se queden sin d_and_a y sin EV/EBITDA. Es
    # la decision correcta segun la regla del modulo: un hueco visible vale
    # mas que un numero plausible de otra cosa.
    "d_and_a": ["DepreciationDepletionAndAmortization",
                "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization"],
}

# PONDERADO: promedios ponderados y magnitudes por accion. NO se suman ni se
# restan. Se derivan con su propia algebra y con control cruzado (ver abajo).
FLUJO_PONDERADO = {
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "shares_basic": ["WeightedAverageNumberOfSharesOutstandingBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
}

# INSTANTE: foto a una fecha (balance). No se desacumula nada.
INSTANTE = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "inventory": ["InventoryNet"],
    "debt_short": ["ShortTermBorrowings", "DebtCurrent", "LongTermDebtCurrent",
                   "OtherShortTermBorrowings"],
    "debt_long": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill",
                    "FiniteLivedIntangibleAssetsNet"],
    "shares_out": ["EntityCommonStockSharesOutstanding",
                   "CommonStockSharesOutstanding"],
}

CONCEPTOS = {}
CONCEPTOS.update({k: "aditivo" for k in FLUJO_ADITIVO})
CONCEPTOS.update({k: "ponderado" for k in FLUJO_PONDERADO})
CONCEPTOS.update({k: "instante" for k in INSTANTE})

# Tolerancia del control cruzado de los conceptos ponderados (2%).
# Por debajo de esto los dos metodos se consideran de acuerdo.
TOL_PONDERADO = 0.02

# ---------------------------------------------------------------------------
# Reconstruccion de net_income cuando la empresa dejo de tagear NetIncomeLoss
# ---------------------------------------------------------------------------
# Ocho tickers del universo (MA, CAT, SCCO, AVAV, AVGO, F, FCX, AMT) declaran
# el resultado bajo ProfitLoss y no bajo NetIncomeLoss. MA y CAT no tagean
# NetIncomeLoss desde 2014 y 2011. Como ProfitLoss incluye a los minoritarios,
# esta prohibido usarlo como sinonimo (ver la nota del encabezado), y el
# resultado era que esos ocho quedaban SIN resultado neto -- y por lo tanto sin
# PER -- sin que nada avisara. Ese es el modo de falla silencioso que este
# modulo existe para evitar.
#
# La salida no es un sinonimo nuevo sino una IDENTIDAD contable:
#
#     NetIncomeLoss = ProfitLoss - NetIncomeLossAttributableToNoncontrolling...
#
# Condiciones, ninguna relajable:
#   1. Solo rellena HUECOS. Donde NetIncomeLoss existe, gana NetIncomeLoss.
#   2. El hecho de minoritarios tiene que existir PARA ESE PERIODO. No se
#      asume cero: si la empresa tiene el tag pero no lo declaro en el
#      trimestre, no se sabe cuanto es. Medido en AVGO, asumir cero da 11% de
#      error. Solo se toma NCI=0 cuando la empresa NO tiene el tag en ningun
#      lado (filer sin participaciones no controlantes).
#   3. Control cruzado contra NetIncomeLossAvailableToCommonStockholders*,
#      que es una medicion INDEPENDIENTE del mismo renglon. Si difieren mas de
#      la tolerancia, no se emite y queda un aviso.
#
# Medido sobre los 8 tickers desde 2020: 193 periodos coinciden dentro de
# 0,5%; 3 no (AVGO por la condicion 2, FCX en 2 trimestres).
TAG_RESULTADO_TOTAL = "ProfitLoss"
TAG_MINORITARIOS = "NetIncomeLossAttributableToNoncontrollingInterest"

# Mas laxa que TOL_PONDERADO porque el control no mide exactamente lo mismo:
# ...AvailableToCommonStockholders descuenta ademas los dividendos preferidos.
# Una diferencia chica es esperable; una grande significa que la identidad no
# se cumple y ahi hay que abstenerse.
TOL_NET_INCOME = 0.05

# Dos tags que publican el mismo anual por debajo de esto son sinonimos
# redundantes y su mezcla no cambia ninguna suma. Ver
# _avisos_mezcla_en_ejercicio.
TOL_MEZCLA = 0.01

# Limites de dias para clasificar la duracion de un hecho. Los trimestres
# fiscales reales van de 84 a 98 dias; los ejercicios, de 358 a 371.
_LIM = ((100, "Q"), (196, "H"), (290, "9M"), (380, "FY"))

# Cuantos trimestres abarca cada tramo acumulado. Se usa para desacumular:
# el "k" tiene que salir de la DURACION del hecho, no de su posicion en la
# lista de tramos disponibles -- si a una empresa le falta el Q1, el hecho de
# 9 meses seria el segundo de la lista y se lo trataria como si fuera un
# semestre.
_QS = {"Q": 1, "H": 2, "9M": 3, "FY": 4}


# --------------------------------------------------------------- utilidades --
def _dias(inicio, fin):
    return (date.fromisoformat(fin) - date.fromisoformat(inicio)).days


def _tramo(inicio, fin):
    """Clasifica un hecho de duracion en Q / H / 9M / FY, o None si no encaja."""
    n = _dias(inicio, fin)
    if n <= 0:
        return None
    for lim, nombre in _LIM:
        if n <= lim:
            return nombre
    return None


def _hechos(facts, tags):
    """
    Todos los hechos de los tags candidatos, con su prioridad (posicion en la
    lista). Busca en us-gaap y en dei (shares_out vive en dei).
    """
    out = []
    for pri, tag in enumerate(tags):
        for taxonomia in ("us-gaap", "dei"):
            grupo = facts.get(taxonomia, {})
            if tag not in grupo:
                continue
            for unidad, arr in grupo[tag].get("units", {}).items():
                for h in arr:
                    if h.get("val") is None:
                        continue
                    out.append({"val": h["val"], "start": h.get("start"),
                                "end": h.get("end"), "filed": h.get("filed", ""),
                                "form": h.get("form", ""), "accn": h.get("accn", ""),
                                "tag": tag, "pri": pri, "unidad": unidad})
    return out


def _rango_forma(hecho):
    """
    1 si el hecho viene de un reporte periodico (10-K / 10-Q y sus enmiendas),
    0 si viene de cualquier otro formulario.

    Existe porque la regla "gana la presentacion mas reciente" es correcta
    entre reportes periodicos y desastrosa fuera de ellos. El DEF 14A -- el
    proxy de compensacion -- taguea NetIncomeLoss en su tabla "Pay Versus
    Performance", se presenta DESPUES del 10-K y por lo tanto ganaba. Medido
    sobre el universo: 295 hechos de DEF 14A ganaban la eleccion en 59
    tickers, y en 8 de ellos cambiaban el numero. El peor, Schwab, publica su
    resultado anual en miles en el proxy: 8,9 MM contra los 8.852,0 MM del
    10-K, y eso se arrastraba a los cinco ejercicios. El Q4, que se deriva
    como FY menos 9M, salia en -6.384 MM cuando el real fue +2.459 MM.

    Un 8-K (comunicado de resultados) tiene el mismo problema al reves: llega
    ANTES que el 10-Q y trae cifras preliminares. Si un periodo SOLO existe en
    un 8-K, se sigue usando -- este rango solo decide cuando hay competencia.
    """
    return 1 if hecho.get("form", "").split("/")[0] in ("10-K", "10-Q") else 0


def _elegir(cands, hasta_filed=None):
    """
    Elige un hecho entre varios que describen el MISMO periodo.

    Se ordena ASCENDENTE y se toma el ultimo, asi que cada clave esta escrita
    de modo que "mas grande" signifique "mejor":

      1. el reporte periodico le gana a cualquier otro formulario;
      2. despues `pri`, la posicion del tag en la lista de sinonimos;
      3. a igualdad, el 'filed' mas reciente -- la ultima reexpresion conocida.

    hasta_filed habilita el POINT-IN-TIME: descarta lo presentado despues de
    esa fecha, devolviendo lo que se sabia entonces.

    OJO -- DEFECTO CONOCIDO Y NO CORREGIDO ACA: al tomar el ultimo del orden
    ascendente, `pri` grande le gana a `pri` chico, o sea que entre dos
    sinonimos GANA EL MENOS PREFERIDO. Es lo contrario de lo que hace la otra
    rama de seleccion, el bloque `directos` de _serie_aditiva, que compara
    `h["pri"] < prev["pri"]`. Las dos ramas se contradicen. Se deja como estaba
    a proposito: cambiarlo mueve numeros en todo el universo y merece su
    propia medicion, no venir de arrastre con el arreglo del formulario.
    """
    if hasta_filed:
        cands = [c for c in cands if c["filed"] and c["filed"] <= hasta_filed]
    if not cands:
        return None
    return sorted(cands, key=lambda h: (_rango_forma(h), h["pri"], h["filed"]))[-1]


# ------------------------------------------------------- calendario fiscal --
def _ejercicios(facts):
    """
    Deduce los ejercicios fiscales [(inicio, fin)] a partir de los hechos de
    duracion anual. Se usan varios conceptos como ancla porque ninguno esta
    en el 100% de las empresas.

    El calendario NO se puede asumir por mes: AAPL cierra a fines de
    septiembre, WMT el 31 de enero, NVDA a fines de enero. Se deduce del dato.
    """
    anclas = ("revenue", "net_income", "cfo", "tax_provision", "operating_income")
    pares = set()
    for con in anclas:
        for h in _hechos(facts, FLUJO_ADITIVO[con]):
            if h["start"] and h["end"] and _tramo(h["start"], h["end"]) == "FY":
                pares.add((h["start"], h["end"]))
    return sorted(pares)


def _proyectar_ejercicios(ejercicios, hasta):
    """
    Extiende el calendario fiscal hacia adelante.

    El ejercicio EN CURSO todavia no tiene hecho anual (se publica recien con
    el 10-K), asi que sin esto los trimestres mas recientes -- justo los que
    interesan -- quedan sin etiquetar. Se proyecta repitiendo el mismo cierre
    un anio despues.
    """
    if not ejercicios or not hasta:
        return ejercicios
    out = list(ejercicios)
    ini, fin = out[-1]
    while fin < hasta:
        try:
            nini, nfin = fin, date.fromisoformat(fin).replace(
                year=date.fromisoformat(fin).year + 1).isoformat()
        except ValueError:          # 29 de febrero
            nini, nfin = fin, (date.fromisoformat(fin).replace(day=28).replace(
                year=date.fromisoformat(fin).year + 1)).isoformat()
        out.append((nini, nfin))
        fin = nfin
    return out


def _etiquetar(period_end, ejercicios):
    """
    (fiscal_year, fiscal_quarter) de un cierre trimestral.

    fiscal_year = anio calendario del CIERRE del ejercicio (convencion
    habitual: el FY2026 de AAPL termina en septiembre de 2026).
    fiscal_quarter = posicion 1-4 dentro del ejercicio, por dias transcurridos
    desde el inicio -- no por orden de aparicion, que fallaria si falta un Q.
    """
    for ini, fin in ejercicios:
        if ini < period_end <= fin:
            transcurrido = _dias(ini, period_end)
            q = min(4, max(1, int(round(transcurrido / 91.31))))
            return int(fin[:4]), q
    return None, None


# ------------------------------------------------------------ desacumulacion --
def _primer_filed(hechos, end, tramo=None):
    """
    Fecha en que el periodo se publico POR PRIMERA VEZ.

    No es lo mismo que el `filed` del hecho elegido: `_elegir` se queda con la
    ultima reexpresion, asi que su fecha puede ser anios posterior. Para armar
    una serie historica SIN MIRAR HACIA ADELANTE hace falta saber desde cuando
    el dato era publico, y eso es el filed MAS VIEJO de ese periodo.

    Un trimestre cerrado el 27/9 no estuvo disponible el 27/9: se publico con
    el 10-Q, semanas despues. Usarlo antes seria lookahead.

    INVARIANTE: un periodo no puede ser publico ANTES de terminar. Hay hechos
    con fecha de cierre futura que aparecen en filings anteriores
    (proyecciones, compromisos contractuales); tomarlos daria un lag negativo
    -- hasta -309 dias en un caso real -- y eso es exactamente la direccion
    que produce lookahead. Se descartan ANTES de tomar el minimo.
    """
    fechas = [h["filed"] for h in hechos
              if h.get("end") == end and h.get("filed") and h["filed"] >= end
              and (tramo is None or not h.get("start")
                   or _tramo(h["start"], h["end"]) == tramo)]
    return min(fechas) if fechas else None


def _cadenas(hechos):
    """
    Agrupa los hechos de duracion por INICIO de ejercicio. Dentro de un mismo
    ejercicio los acumulados comparten el mismo 'start' y se diferencian por
    el 'end' (3m, 6m, 9m, 12m). Devuelve {inicio: {tramo: hecho}}.
    """
    por_inicio = defaultdict(lambda: defaultdict(list))
    for h in hechos:
        if not h["start"] or not h["end"]:
            continue
        t = _tramo(h["start"], h["end"])
        if t:
            por_inicio[h["start"]][t].append(h)
    salida = {}
    for ini, tramos in por_inicio.items():
        salida[ini] = {t: _elegir(c) for t, c in tramos.items()}
    return salida


def _serie_aditiva(hechos):
    """
    Serie trimestral de un concepto ADITIVO.

    Dos fuentes, en este orden:
      1. hechos de 3 meses publicados directamente (se usan tal cual)
      2. desacumulacion: Q2 = H1 - Q1, Q3 = 9M - H1, Q4 = FY - 9M

    El paso 2 es imprescindible: el estado de flujo se informa acumulado, y el
    10-K nunca desagrega el Q4. Sin esto, cfo/capex/d_and_a quedan con ~la
    mitad de los trimestres que el resto.
    """
    directos = {}
    for h in hechos:
        if not h["start"] or not h["end"]:
            continue
        if _tramo(h["start"], h["end"]) != "Q":
            continue
        prev = directos.get(h["end"])
        if prev is None or h["pri"] < prev["pri"] or \
           (h["pri"] == prev["pri"] and h["filed"] > prev["filed"]):
            directos[h["end"]] = h

    serie = {e: {"val": h["val"], "tag": h["tag"], "derivado": False,
                 "filed": h["filed"],
                 "filed_primero": _primer_filed(hechos, h["end"], "Q")}
             for e, h in directos.items()}

    ORDEN = ["Q", "H", "9M", "FY"]
    for ini, tramos in _cadenas(hechos).items():
        acum = [(t, tramos[t]) for t in ORDEN if tramos.get(t)]
        for i in range(1, len(acum)):
            t_act, actual = acum[i]
            t_prev, previo = acum[i - 1]
            # Solo se resta entre acumulados ADYACENTES. Si falta un tramo
            # intermedio, la diferencia abarcaria dos trimestres y quedaria
            # imputada a uno solo -- un error que no se nota.
            if _QS[t_act] - _QS[t_prev] != 1:
                continue
            fin = actual["end"]
            if fin in serie and not serie[fin]["derivado"]:
                continue          # ya hay un trimestre publicado: gana ese
            serie[fin] = {"val": actual["val"] - previo["val"],
                          "tag": actual["tag"], "derivado": True,
                          "filed": actual["filed"],
                          "filed_primero": _primer_filed(hechos, actual["end"])}
    return serie


def _completar_net_income(serie_ni, serie_total, serie_minoritarios,
                          serie_control, tiene_tag_minoritarios):
    """
    Rellena los huecos de `net_income` con ProfitLoss - minoritarios.

    Ver la nota de TAG_RESULTADO_TOTAL para el por que y las condiciones.
    Devuelve (serie_completada, avisos). No muta la serie recibida.

    El valor derivado queda con tag "ProfitLoss-NCI" y derivado=True, asi que
    _avisos_cambio_tag lo va a reportar como cambio de tag en las empresas que
    usaron los dos: es correcto que lo haga, ahi hay algo para mirar.
    """
    avisos = []
    salida = dict(serie_ni or {})
    for fin, tot in (serie_total or {}).items():
        if fin in salida:
            continue                       # NetIncomeLoss real: gana siempre
        nci = (serie_minoritarios or {}).get(fin)
        if nci is None:
            if tiene_tag_minoritarios:
                # La empresa tiene minoritarios pero no los declaro en este
                # periodo. Asumir cero seria inventar el numero.
                avisos.append({
                    "tipo": "net_income_sin_minoritarios", "concepto": "net_income",
                    "period_end": fin,
                    "detalle": "hay %s pero falta %s en el periodo; no se deriva"
                               % (TAG_RESULTADO_TOTAL, TAG_MINORITARIOS)})
                continue
            valor = tot["val"]             # filer sin participaciones: NCI = 0
        else:
            valor = tot["val"] - nci["val"]

        ctrl = (serie_control or {}).get(fin)
        if ctrl is not None:
            base = max(abs(ctrl["val"]), 1.0)
            if abs(valor - ctrl["val"]) / base > TOL_NET_INCOME:
                avisos.append({
                    "tipo": "net_income_derivado_sin_control", "concepto": "net_income",
                    "period_end": fin,
                    "detalle": "%s-%s da %.0f pero el control da %.0f"
                               % (TAG_RESULTADO_TOTAL, "NCI", valor, ctrl["val"])})
                continue

        salida[fin] = {"val": valor, "tag": "%s-NCI" % TAG_RESULTADO_TOTAL,
                       "derivado": True, "filed": tot["filed"],
                       "filed_primero": tot.get("filed_primero")}
    return salida, avisos


def _serie_ponderada(hechos, concepto, serie_ni=None, serie_acciones=None):
    """
    Serie trimestral de un concepto PONDERADO (por accion o promedio ponderado).

    Estos NO se restan: el promedio ponderado de 12 meses no es la suma de los
    de cada trimestre. La algebra correcta para el tramo k es
        valor_Qk = k * acumulado_k - (k-1) * acumulado_(k-1)

    Medida contra datos reales, esa formula acierta en ~2 de cada 3 casos y
    falla feo en el resto (HON -119%, KLAC -73%, CRWD +541%), porque splits,
    emisiones y reexpresiones rompen la equivalencia. Por eso cada valor
    derivado pasa un CONTROL, y el que no lo pasa se descarta con un aviso.

    OJO -- la derivacion NO es la misma para los dos tipos, y confundirlos da
    numeros absurdos:

      - ACCIONES (shares_*): son promedios ponderados de verdad. El acumulado
        de 9 meses es el promedio de esos 9 meses, no la suma. Se derivan con
        valor_Qk = k * acumulado_k - (k-1) * acumulado_(k-1).
        Control: PLAUSIBILIDAD. El promedio del trimestre tiene que caer en una
        banda razonable alrededor del promedio del acumulado; un split o una
        emision grande lo saca de la banda, que es justo cuando no hay que
        confiar en la derivacion.

      - POR ACCION (eps_*): a pesar del nombre NO son promedios. El EPS de 9
        meses es resultado_9m / acciones_promedio_9m, que se comporta como la
        SUMA de los EPS trimestrales. Se derivan por RESTA simple.
        Verificado en AAPL FY2025: EPS anual 7.46, EPS de 9 meses 5.62, Q4 real
        1.85 -- la resta da 1.84 y la formula ponderada daria 12.98.
        Control: se compara contra el metodo INDEPENDIENTE
        resultado_del_trimestre / acciones_del_trimestre.

    Es preferible un hueco a un numero equivocado: un hueco se ve, un EPS mal
    calculado se propaga al PER y de ahi a todo lo que dependa de el.
    """
    avisos = []
    es_acciones = concepto.startswith("shares")
    directos = {}
    for h in hechos:
        if not h["start"] or not h["end"]:
            continue
        if _tramo(h["start"], h["end"]) != "Q":
            continue
        prev = directos.get(h["end"])
        if prev is None or h["pri"] < prev["pri"] or \
           (h["pri"] == prev["pri"] and h["filed"] > prev["filed"]):
            directos[h["end"]] = h

    serie = {e: {"val": h["val"], "tag": h["tag"], "derivado": False,
                 "filed": h["filed"],
                 "filed_primero": _primer_filed(hechos, h["end"], "Q")}
             for e, h in directos.items()}

    ORDEN = ["Q", "H", "9M", "FY"]
    for ini, tramos in _cadenas(hechos).items():
        acum = [(t, tramos[t]) for t in ORDEN if tramos.get(t)]
        for i in range(1, len(acum)):
            t_act, actual = acum[i]
            t_prev, previo = acum[i - 1]
            if _QS[t_act] - _QS[t_prev] != 1:   # ver nota en _serie_aditiva
                continue
            fin = actual["end"]
            if fin in serie and not serie[fin]["derivado"]:
                continue
            k = _QS[t_act]                  # posicion del trimestre (2, 3 o 4)
            if es_acciones:
                valor = k * actual["val"] - (k - 1) * previo["val"]
            else:
                valor = actual["val"] - previo["val"]

            if es_acciones:
                ref = actual["val"]         # promedio del acumulado
                if ref <= 0 or not (0.3 * ref <= valor <= 3.0 * ref):
                    avisos.append({
                        "tipo": "ponderado_implausible", "concepto": concepto,
                        "period_end": fin,
                        "detalle": "acciones del trimestre %.6g fuera de banda "
                                   "respecto del acumulado %.6g (split o emision?)"
                                   % (valor, ref)})
                    continue
            else:
                ni = (serie_ni or {}).get(fin)
                ac = (serie_acciones or {}).get(fin)
                if not (ni and ac and ac["val"]):
                    # SIN CONTROL NO SE EMITE. La resta de acumulados es fragil:
                    # los dos tramos vienen de filings distintos y pueden estar
                    # en BASES DE SPLIT distintas, porque el mas nuevo se re-
                    # expresa y el viejo no. KLAC FY2025Q4 dio eps=-18.28
                    # (FY post-split 3.18 menos 9M pre-split 21.4) en un
                    # trimestre que gano ~9 por accion. Ese valor paso porque
                    # las acciones del trimestre se habian descartado por
                    # implausibles, y sin denominador el control se degradaba en
                    # silencio a "no control". Un hueco se ve; un EPS al reves
                    # se propaga al PER.
                    avisos.append({
                        "tipo": "ponderado_sin_control", "concepto": concepto,
                        "period_end": fin,
                        "detalle": "resta de acumulados %.6g sin resultado/acciones "
                                   "con que cruzarla, se descarta" % (valor,)})
                    continue
                alterno = ni["val"] / ac["val"]
                base = max(abs(valor), abs(alterno))
                if base > 0 and abs(valor - alterno) / base > TOL_PONDERADO:
                    avisos.append({
                        "tipo": "ponderado_discordante", "concepto": concepto,
                        "period_end": fin,
                        "detalle": "resta de acumulados %.6g vs resultado/acciones "
                                   "%.6g: no concuerdan, se descarta"
                                   % (valor, alterno)})
                    continue
            serie[fin] = {"val": valor, "tag": actual["tag"],
                          "derivado": True, "filed": actual["filed"],
                          "filed_primero": _primer_filed(hechos, actual["end"])}
    return serie, avisos


def _serie_instante(hechos):
    """Serie de un concepto INSTANTE: un valor por fecha, sin derivar nada."""
    por_fecha = defaultdict(list)
    for h in hechos:
        if h["start"] or not h["end"]:
            continue                        # los instantaneos no tienen start
        por_fecha[h["end"]].append(h)
    salida = {}
    for fecha, cands in por_fecha.items():
        h = _elegir(cands)
        if h:
            salida[fecha] = {"val": h["val"], "tag": h["tag"],
                             "derivado": False, "filed": h["filed"],
                             "filed_primero": _primer_filed(hechos, fecha)}
    return salida


# ------------------------------------------------------------------ avisos --
def _instante_cercano(serie, period_end, tolerancia=45):
    """
    Valor instantaneo correspondiente a un cierre trimestral.

    Los del balance caen exactamente en el cierre; el numero de acciones de dei
    viene fechado en la portada del filing, semanas despues. Como los cierres
    estan a ~91 dias entre si, tomar el mas cercano dentro de 45 dias no puede
    agarrar el trimestre equivocado.
    """
    if period_end in serie:
        return serie[period_end]
    mejor, mejor_d = None, None
    for fecha, d in serie.items():
        dist = abs(_dias(min(fecha, period_end), max(fecha, period_end)))
        if dist <= tolerancia and (mejor_d is None or dist < mejor_d):
            mejor, mejor_d = d, dist
    return mejor


def _avisos_cambio_tag(concepto, serie, desde=None):
    """
    Detecta que la serie de un concepto se armo con MAS DE UN TAG.

    Es la red de seguridad contra el error silencioso: los tres casos reales
    que se colaron (ProfitLoss, ...Domestic, EPS por resta) se manifestaron
    todos igual -- dos tags distintos dentro de la misma serie. Un cambio de
    tag puede ser legitimo (la empresa migro de taxonomia) o puede significar
    que se esta mezclando otro renglon del estado contable. El modulo no puede
    distinguirlo: avisa.
    """
    items = [(f, d) for f, d in serie.items() if desde is None or f >= desde]
    tags = sorted({d["tag"] for _, d in items})
    if len(tags) <= 1:
        return []
    return [{"tipo": "cambio_de_tag", "concepto": concepto,
             "detalle": "la serie usa %d tags distintos: %s"
                        % (len(tags), ", ".join(tags)),
             "tags": tags}]


def _anuales_por_tag(hechos):
    """
    {fin_de_ejercicio: {tag: hecho}} con los hechos de duracion ANUAL,
    quedandose con la ultima reexpresion de cada tag. Es la evidencia que
    permite decidir si una mezcla de tags importa o no.
    """
    out = defaultdict(dict)
    for h in hechos:
        if not h["start"] or not h["end"]:
            continue
        if _tramo(h["start"], h["end"]) != "FY":
            continue
        prev = out[h["end"]].get(h["tag"])
        if prev is None or h["filed"] > prev["filed"]:
            out[h["end"]][h["tag"]] = h
    return out


def _avisos_mezcla_en_ejercicio(periodos, conceptos, anuales):
    """
    Detecta que los trimestres de un MISMO ejercicio se armaron con tags
    distintos Y que eso CAMBIA EL NUMERO.

    Hay que leerlo distinto que `cambio_de_tag`. Un cambio a lo largo de la
    serie suele ser legitimo: la empresa migro de taxonomia (medio universo
    paso de SalesRevenueNet a RevenueFromContractWithCustomer* con ASC 606) y
    de ahi en mas usa el nuevo. Una mezcla DENTRO de un ejercicio es otra
    cosa: los 4 trimestres tienen que medir la misma magnitud para poder
    sumarse, y si no la miden, el TTM y el anual reconstruido son la suma de
    dos renglones distintos. Es el modo de falla que dejo revenue en 87,8% de
    reconciliacion mientras net_income daba 99,6%.

    PERO la mezcla sola no alcanza como senal. Medido sobre los 147 tickers:
    de 126 ejercicios que mezclan tags, 109 son INOCUOS -- los dos tags son
    sinonimos y publican el mismo anual, asi que la suma no cambia. Avisar de
    los 126 seria condenar el aviso a que nadie lo lea, que es exactamente el
    problema que este aviso viene a resolver.

    Por eso se cruza contra el anual de cada tag y se separan tres casos:
      - los anuales DIFIEREN  -> "mezcla_en_ejercicio", el defecto real.
      - los anuales COINCIDEN -> sinonimos redundantes, silencio.
      - un solo tag publica anual -> no se puede verificar, y se emite
        "mezcla_no_verificable", que es mas debil y no exige accion.

    La cura del primer caso es el mapeo curado ticker -> tag (ver
    src/data/sec/tags_curados.py). Este aviso existe para que esa curacion no
    quede vieja EN SILENCIO cuando entre un ticker nuevo al universo o una
    empresa cambie de taxonomia.
    """
    # Ultimo cierre de cada ejercicio: es la fecha con la que estan indexados
    # los hechos anuales.
    cierre = {}
    por_concepto = defaultdict(lambda: defaultdict(set))
    for p in periodos:
        fy = p.get("fiscal_year")
        if fy is None:
            continue
        cierre[fy] = max(cierre.get(fy, ""), p["period_end"])
        for concepto in conceptos:
            tag = p.get(concepto + "__tag")
            if tag:
                por_concepto[concepto][fy].add(tag)

    avisos = []
    for concepto in sorted(por_concepto):
        for fy in sorted(por_concepto[concepto]):
            tags = sorted(por_concepto[concepto][fy])
            if len(tags) <= 1:
                continue
            candidatos = (anuales.get(concepto) or {}).get(cierre.get(fy), {})
            vals = [h["val"] for h in candidatos.values() if h["val"]]
            comun = "el ejercicio %s mezcla %d tags: %s" % (fy, len(tags),
                                                            ", ".join(tags))
            if len(vals) < 2:
                avisos.append({
                    "tipo": "mezcla_no_verificable", "concepto": concepto,
                    "detalle": comun + " -- un solo tag publica anual, no se "
                                       "puede comprobar si cambia el numero",
                    "fiscal_year": fy, "tags": tags})
                continue
            lo, hi = min(vals), max(vals)
            if abs(hi - lo) / max(abs(hi), 1.0) > TOL_MEZCLA:
                avisos.append({
                    "tipo": "mezcla_en_ejercicio", "concepto": concepto,
                    "detalle": comun + " -- y sus anuales difieren (%.4g vs "
                                       "%.4g)" % (hi, lo),
                    "fiscal_year": fy, "tags": tags})
    return avisos


def _tags_de(concepto, tags, curados):
    """
    Lista de tags candidatos para un concepto, con la curacion aplicada.

    Un tag curado REEMPLAZA la lista de sinonimos, no se antepone. Anteponerlo
    dejaria a los demas como respaldo, y entonces volveria a mezclar
    exactamente en los trimestres donde el tag elegido falta -- que es el caso
    que la curacion viene a resolver. Es preferible el hueco VISIBLE al numero
    mezclado invisible.

    Se acepta una LISTA cuando la empresa migro de taxonomia de verdad y
    ningun tag solo cubre toda la ventana. En ese caso sigue siendo posible
    mezclar dentro de un ejercicio, y `mezcla_en_ejercicio` lo avisa.
    """
    if not curados or concepto not in curados:
        return tags
    elegido = curados[concepto]
    return [elegido] if isinstance(elegido, str) else list(elegido)


# ------------------------------------------------------------------- API --
def normalizar(companyfacts, hasta_filed=None, desde=None, tags_curados=None):
    """
    companyfacts : dict crudo de data.sec.gov/api/xbrl/companyfacts/CIK...json
    hasta_filed  : 'YYYY-MM-DD'. Point-in-time: ignora lo presentado despues,
                   devolviendo lo que se sabia en esa fecha. None = todo.
    desde        : 'YYYY-MM-DD'. Recorta la salida a periodos posteriores.
    tags_curados : {concepto: tag} o {concepto: [tags]} para ESTE ticker. El
                   tag curado reemplaza la lista de sinonimos del concepto.
                   Existe porque "revenue" no es un renglon en XBRL y en 23 de
                   147 tickers hay dos tags que valen cosas distintas: elegir
                   entre ellos es una decision contable, no algoritmica. El
                   mapeo vive en src/data/sec/tags_curados.py.

    Devuelve:
      {
        "entidad": str, "cik": int|None,
        "periodos": [ {period_end, fiscal_year, fiscal_quarter,
                       <concepto>: valor, <concepto>__tag, <concepto>__derivado} ],
        "avisos":   [ {tipo, concepto, ...} ],
        "meta":     { concepto: {n, derivados, tags} },
      }

    Los periodos vienen ordenados por fecha. Un concepto ausente en un periodo
    simplemente no aparece como clave: nunca se rellena con cero.
    """
    facts = (companyfacts or {}).get("facts", {})
    series, avisos, meta = {}, [], {}

    def _traer(tags):
        h = _hechos(facts, tags)
        if hasta_filed:
            h = [x for x in h if x["filed"] and x["filed"] <= hasta_filed]
        return h

    anuales = {}
    for concepto, tags in FLUJO_ADITIVO.items():
        hechos = _traer(_tags_de(concepto, tags, tags_curados))
        series[concepto] = _serie_aditiva(hechos)
        # Se guardan para el control de mezcla de tags, que necesita saber si
        # dos tags publican el mismo anual o dos numeros distintos.
        anuales[concepto] = _anuales_por_tag(hechos)

    # ORDEN IMPORTANTE: net_income se completa ANTES del EPS, porque el control
    # cruzado del EPS lo usa como respaldo de net_income_common.
    hechos_total = _traer([TAG_RESULTADO_TOTAL])
    if hechos_total:
        series["net_income"], av = _completar_net_income(
            series["net_income"],
            _serie_aditiva(hechos_total),
            _serie_aditiva(_traer([TAG_MINORITARIOS])),
            series.get("net_income_common"),
            TAG_MINORITARIOS in facts.get("us-gaap", {}))
        avisos.extend(av)

    # ORDEN IMPORTANTE: las acciones se calculan ANTES que el EPS, porque el
    # control cruzado del EPS necesita resultado/acciones del mismo trimestre.
    for concepto in ("shares_diluted", "shares_basic"):
        s, av = _serie_ponderada(
            _traer(_tags_de(concepto, FLUJO_PONDERADO[concepto], tags_curados)),
            concepto)
        series[concepto] = s
        avisos.extend(av)
    # El EPS se calcula sobre el resultado atribuible a los accionistas
    # COMUNES, no sobre el total: en una empresa con acciones preferidas los
    # dividendos preferidos ya estan descontados. Usar net_income a secas hace
    # fallar el control cruzado en todo el sector financiero (JPM: 36 de 73
    # trimestres descartados). Se prefiere net_income_common y se cae a
    # net_income solo si no existe.
    ni_para_eps = dict(series.get("net_income") or {})
    ni_para_eps.update(series.get("net_income_common") or {})
    for concepto, acciones in (("eps_diluted", "shares_diluted"),
                               ("eps_basic", "shares_basic")):
        s, av = _serie_ponderada(
            _traer(_tags_de(concepto, FLUJO_PONDERADO[concepto], tags_curados)),
            concepto, serie_ni=ni_para_eps,
            serie_acciones=series.get(acciones))
        series[concepto] = s
        avisos.extend(av)

    for concepto, tags in INSTANTE.items():
        series[concepto] = _serie_instante(
            _traer(_tags_de(concepto, tags, tags_curados)))

    for concepto, serie in series.items():
        avisos.extend(_avisos_cambio_tag(concepto, serie, desde))
        meta[concepto] = {
            "n": len(serie),
            "derivados": sum(1 for d in serie.values() if d["derivado"]),
            "tags": sorted({d["tag"] for d in serie.values()}),
        }

    # El indice de periodos lo definen SOLO los conceptos de duracion: son los
    # unicos cuyo 'end' es un cierre trimestral real. Los instantaneos se
    # enganchan despues por cercania. Sin esto aparecen periodos fantasma: el
    # EntityCommonStockSharesOutstanding de dei viene fechado en la PORTADA del
    # filing (semanas despues del cierre) y generaria una fila por cada filing.
    duracion = [c for c in series if CONCEPTOS[c] != "instante"]
    fechas = sorted({f for c in duracion for f in series[c]})
    if not fechas:
        fechas = sorted({f for s in series.values() for f in s})

    ejercicios = _proyectar_ejercicios(_ejercicios(facts),
                                       fechas[-1] if fechas else None)

    periodos = []
    for f in fechas:
        if desde is not None and f < desde:
            continue
        fy, fq = _etiquetar(f, ejercicios)
        fila = {"period_end": f, "fiscal_year": fy, "fiscal_quarter": fq}
        primeros, ultimos = [], []
        for concepto in duracion:
            d = series[concepto].get(f)
            if d is None:
                continue
            fila[concepto] = d["val"]
            fila[concepto + "__tag"] = d["tag"]
            fila[concepto + "__derivado"] = d["derivado"]
            # El invariante se aplica al RECOLECTAR: un instantaneo se engancha
            # por cercania (+-45 dias) y trae su propia fecha, que puede ser
            # anterior a este cierre. Filtrar despues del min dejaria el
            # periodo sin fecha en vez de con la correcta.
            if d.get("filed_primero") and d["filed_primero"] >= f:
                primeros.append(d["filed_primero"])
            if d.get("filed"):
                ultimos.append(d["filed"])
        for concepto in INSTANTE:
            d = _instante_cercano(series[concepto], f)
            if d is None:
                continue
            fila[concepto] = d["val"]
            fila[concepto + "__tag"] = d["tag"]
            fila[concepto + "__derivado"] = False
            # El invariante se aplica al RECOLECTAR: un instantaneo se engancha
            # por cercania (+-45 dias) y trae su propia fecha, que puede ser
            # anterior a este cierre. Filtrar despues del min dejaria el
            # periodo sin fecha en vez de con la correcta.
            if d.get("filed_primero") and d["filed_primero"] >= f:
                primeros.append(d["filed_primero"])
            if d.get("filed"):
                ultimos.append(d["filed"])
        # filed_primero = desde cuando el trimestre fue PUBLICO. Es la fecha que
        # hay que usar para armar series historicas sin mirar hacia adelante: un
        # trimestre cerrado el 27/9 no estuvo disponible el 27/9.
        # filed_ultimo  = de que presentacion viene el valor guardado hoy
        # (procedencia).
        # El invariante "no puede ser publico antes de terminar" se aplica en
        # _primer_filed, ANTES de tomar el minimo -- filtrarlo aca dejaria el
        # periodo sin fecha en vez de con la correcta.
        fila["filed_primero"] = min(primeros) if primeros else None
        fila["filed_ultimo"] = max(ultimos) if ultimos else None
        periodos.append(fila)

    # Va DESPUES de armar los periodos porque necesita el fiscal_year, que se
    # etiqueta recien aca. Mira solo los conceptos ADITIVOS: son los unicos
    # que se suman entre trimestres, que es de donde sale el dano. Un
    # instantaneo (balance) no se suma nunca, y los ponderados (EPS, acciones)
    # tienen su propio control cruzado en _serie_ponderada.
    avisos.extend(_avisos_mezcla_en_ejercicio(periodos, list(FLUJO_ADITIVO),
                                              anuales))

    return {"entidad": companyfacts.get("entityName", ""),
            "cik": companyfacts.get("cik"),
            "periodos": periodos, "avisos": avisos, "meta": meta}
