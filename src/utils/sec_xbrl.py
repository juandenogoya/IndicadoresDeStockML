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
    "net_income_common": ["NetIncomeLossAvailableToCommonStockholdersBasic"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating"],
    "net_interest_income": ["InterestIncomeExpenseNet",
                            "InterestIncomeExpenseAfterProvisionForLoanLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
    "d_and_a": ["DepreciationDepletionAndAmortization",
                "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization", "Depreciation"],
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


def _elegir(cands, hasta_filed=None):
    """
    Elige un hecho entre varios que describen el MISMO periodo.

    Criterio: primero la prioridad del tag (el sinonimo preferido), despues el
    'filed' mas reciente -- que es la ultima reexpresion conocida.

    hasta_filed habilita el POINT-IN-TIME: descarta lo presentado despues de
    esa fecha, devolviendo lo que se sabia entonces.
    """
    if hasta_filed:
        cands = [c for c in cands if c["filed"] and c["filed"] <= hasta_filed]
    if not cands:
        return None
    return sorted(cands, key=lambda h: (h["pri"], h["filed"]))[-1]


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
                 "filed": h["filed"]} for e, h in directos.items()}

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
                          "filed": actual["filed"]}
    return serie


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
                 "filed": h["filed"]} for e, h in directos.items()}

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
                if ni and ac and ac["val"]:
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
                          "derivado": True, "filed": actual["filed"]}
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
                             "derivado": False, "filed": h["filed"]}
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


# ------------------------------------------------------------------- API --
def normalizar(companyfacts, hasta_filed=None, desde=None):
    """
    companyfacts : dict crudo de data.sec.gov/api/xbrl/companyfacts/CIK...json
    hasta_filed  : 'YYYY-MM-DD'. Point-in-time: ignora lo presentado despues,
                   devolviendo lo que se sabia en esa fecha. None = todo.
    desde        : 'YYYY-MM-DD'. Recorta la salida a periodos posteriores.

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

    for concepto, tags in FLUJO_ADITIVO.items():
        series[concepto] = _serie_aditiva(_traer(tags))

    # ORDEN IMPORTANTE: las acciones se calculan ANTES que el EPS, porque el
    # control cruzado del EPS necesita resultado/acciones del mismo trimestre.
    for concepto in ("shares_diluted", "shares_basic"):
        s, av = _serie_ponderada(_traer(FLUJO_PONDERADO[concepto]), concepto)
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
        s, av = _serie_ponderada(_traer(FLUJO_PONDERADO[concepto]), concepto,
                                 serie_ni=ni_para_eps,
                                 serie_acciones=series.get(acciones))
        series[concepto] = s
        avisos.extend(av)

    for concepto, tags in INSTANTE.items():
        series[concepto] = _serie_instante(_traer(tags))

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
        for concepto in duracion:
            d = series[concepto].get(f)
            if d is None:
                continue
            fila[concepto] = d["val"]
            fila[concepto + "__tag"] = d["tag"]
            fila[concepto + "__derivado"] = d["derivado"]
        for concepto in INSTANTE:
            d = _instante_cercano(series[concepto], f)
            if d is None:
                continue
            fila[concepto] = d["val"]
            fila[concepto + "__tag"] = d["tag"]
            fila[concepto + "__derivado"] = False
        periodos.append(fila)

    return {"entidad": companyfacts.get("entityName", ""),
            "cik": companyfacts.get("cik"),
            "periodos": periodos, "avisos": avisos, "meta": meta}
