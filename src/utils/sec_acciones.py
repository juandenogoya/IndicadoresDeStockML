"""
sec_acciones.py
Serie POINT-IN-TIME de acciones en circulacion, desde la PORTADA de cada filing.

Funcion PURA: sin DB, sin red, sin config. Solo stdlib. ASCII en strings.

Por que existe este modulo aparte del normalizador
--------------------------------------------------
`fundamentales_sec_q.shares_out` no sirve para armar series historicas contra
precio, y el motivo no es un bug del normalizador sino como publica SEC:

1. SEC RE-EXPRESA retroactivamente todo lo "por accion" cuando hay un split.
   El mismo period_end aparece con dos `filed` distintos y valores 10x aparte:

       us-gaap:CommonStockSharesOutstanding  (KLAC)
         end=2025-06-30  filed=2025-08-08      132.023.000   <- base pre-split
         end=2025-06-30  filed=2026-08-06    1.320.227.000   <- re-expresado

   El normalizador se queda con el `filed` mas nuevo (que es lo correcto para
   "cual es el valor vigente de ese trimestre"), asi que el conteo post-split
   aterriza sobre un trimestre de 2025. `precios_diarios` NO se re-ajusta hacia
   atras (regla documentada del proyecto): el precio de 2025 sigue en base
   pre-split. Cruzarlos parte el market cap por el factor del split.

2. Los dos tags candidatos NO miden lo mismo. En BA,
   us-gaap:CommonStockSharesOutstanding da 1.012.261.159 constante (acciones
   EMITIDAS) contra 754-790M de dei (acciones EN CIRCULACION, netas de
   autocartera). Alternar entre ellos por disponibilidad mete saltos de 1.8x.

3. Medido sobre el universo: 39 de 147 tickers (26,5%) tienen saltos de conteo
   entre trimestres adyacentes por alguna de estas causas, mezcladas con
   splits reales, IPOs y errores de unidad de 1e3/1e6.

La solucion es usar SOLO `dei:EntityCommonStockSharesOutstanding`, que es un
hecho de la PORTADA: cada filing declara su propio conteo a su propia fecha, y
por eso NO se re-expresa nunca. Los hechos pre-split quedan pre-split para
siempre. Eso lo convierte en una funcion escalon point-in-time, que es
exactamente lo que hace falta para multiplicar por el precio de ese dia.

    KLAC dei  end=2026-04-27 filed=2026-04-30    130.627.521   pre-split
             end=2026-08-03 filed=2026-08-06  1.306.546.783   post-split (real)

Consecuencia de diseno, y conviene tenerla explicita: los multiplos historicos
se calculan desde AGREGADOS (market_cap / net_income_ttm), nunca desde
magnitudes por accion (close / eps_ttm). Los agregados son INVARIANTES al
split -- un 10:1 no cambia el resultado ni las ventas -- y el unico termino por
accion que queda, el precio, se multiplica por el conteo de portada vigente ese
dia, que esta en su misma base.
"""

TAG_PORTADA = "EntityCommonStockSharesOutstanding"
TAXONOMIA = "dei"

# Respaldo para los filers de CLASES MULTIPLES. En esas empresas SEC declara el
# conteo de portada POR CLASE, con dimensiones XBRL, y la API companyfacts
# descarta los hechos dimensionales -> el tag de portada directamente no
# aparece. Medido: 20 de 147 tickers (GOOG, META, V, MA, F, UPS, NKE, HSY,
# DELL, SNAP, PINS, PLTR, RBLX, RIVN, ABNB, HOOD, ZM, ASAN, AI, PATH) -- todos
# de clase dual o triple.
#
# El respaldo es el promedio ponderado del TRIMESTRE, tomado en su PRIMER
# `filed`, que es la version contemporanea al precio de ese trimestre. NO se usa
# us-gaap:CommonStockSharesOutstanding: se re-expresa igual que todo lo demas y
# ademas mide acciones EMITIDAS, no en circulacion.
#
# Es una magnitud distinta (promedio del trimestre vs conteo a una fecha), asi
# que cada punto queda etiquetado con su `fuente` y el consumidor puede verlo.
TAGS_PROMEDIO = ["WeightedAverageNumberOfDilutedSharesOutstanding",
                 "WeightedAverageNumberOfSharesOutstandingBasic"]
TAXONOMIA_PROMEDIO = "us-gaap"

# Duracion (en dias) que se acepta como "un trimestre". Los acumulados (H, 9M,
# FY) quedan afuera: su promedio ponderado no es el del trimestre.
DIAS_TRIMESTRE = (60, 110)

# Un conteo de acciones plausible. Descarta los errores de unidad (valores
# reportados en miles o millones en vez de unidades) sin descartar empresas
# chicas de verdad.
MIN_ACCIONES = 100_000
MAX_ACCIONES = 1e13

# Cuanto puede alejarse un punto de la mediana de su propia serie antes de que
# sea un error de dato y no un hecho. Ver _fuera_de_escala().
FACTOR_ESCALA = 100.0


def serie_portada(companyfacts, desde=None):
    """
    companyfacts : dict crudo de data.sec.gov/api/xbrl/companyfacts/CIK...json
    desde        : 'YYYY-MM-DD'. Recorta a portadas posteriores.

    Devuelve [{fecha, shares, accn, filed, form}] ordenado por fecha
    ascendente, una entrada por PORTADA (no por trimestre).

    `fecha` es el `end` del hecho, o sea la fecha a la que la empresa declara
    ese conteo en la caratula -- suele caer unos dias ANTES del filing.

    Deduplicacion: si dos filings declaran la misma fecha de portada, gana el
    de `filed` mas TEMPRANO. Es al reves que en el normalizador, y a proposito:
    aca no se busca el valor vigente sino el que estaba publicado entonces, y
    una re-declaracion posterior de la misma fecha ya viene en otra base.
    """
    facts = (companyfacts or {}).get("facts", {})
    tag = facts.get(TAXONOMIA, {}).get(TAG_PORTADA)
    if not tag:
        return []

    return _recolectar(tag, desde, "portada")


def _recolectar(tag, desde, fuente, solo_trimestre=False):
    """
    Nucleo comun de las dos series: recorre los hechos de un tag, aplica los
    filtros e invariantes, y deduplica por fecha quedandose con el `filed` mas
    TEMPRANO.
    """
    por_fecha = {}
    for hechos in tag.get("units", {}).values():
        for h in hechos:
            fecha, val, filed = h.get("end"), h.get("val"), h.get("filed")
            if not fecha or val is None or not filed:
                continue
            # INVARIANTE: una portada no puede estar fechada DESPUES de su
            # propio filing. Caso real: un 10-Q de AAL declara portada
            # 2027-07-17 habiendose presentado el 2026-07-23 -- un anio mal
            # tipeado. Sin este filtro esa fecha se vuelve el fin de la serie y
            # el conteo queda vigente para siempre.
            if fecha > filed:
                continue
            if desde is not None and fecha < desde:
                continue
            if not (MIN_ACCIONES <= val <= MAX_ACCIONES):
                continue
            if solo_trimestre:
                d = _dias(h.get("start"), fecha)
                if d is None or not (DIAS_TRIMESTRE[0] <= d <= DIAS_TRIMESTRE[1]):
                    continue
            prev = por_fecha.get(fecha)
            if prev is None or filed < prev["filed"]:
                por_fecha[fecha] = {"fecha": fecha, "shares": float(val),
                                    "accn": h.get("accn"), "filed": filed,
                                    "form": h.get("form"), "fuente": fuente}
    return [por_fecha[f] for f in sorted(por_fecha)]


def _dias(inicio, fin):
    """Dias entre dos fechas ISO. None si falta alguna o no parsean."""
    if not inicio or not fin:
        return None
    try:
        from datetime import date
        a = date(*(int(x) for x in str(inicio)[:10].split("-")))
        b = date(*(int(x) for x in str(fin)[:10].split("-")))
    except (ValueError, TypeError):
        return None
    return (b - a).days


def serie_promedio(companyfacts, desde=None):
    """
    Respaldo para los filers de clases multiples: promedio ponderado de
    acciones del TRIMESTRE, en su PRIMER `filed`.

    `fecha` es el cierre del trimestre (no una portada), asi que el punto
    empieza a regir el dia del cierre y no el dia en que se publico. Es una
    aproximacion consciente: se prefiere tener conteo a no tenerlo, y queda
    etiquetado con fuente='promedio_diluido' para que se vea.

    Se prefiere el diluido; el basico entra solo donde el diluido falta.
    """
    facts = (companyfacts or {}).get("facts", {}).get(TAXONOMIA_PROMEDIO, {})
    por_fecha = {}
    for nombre in TAGS_PROMEDIO:
        tag = facts.get(nombre)
        if not tag:
            continue
        for punto in _recolectar(tag, desde, "promedio_diluido",
                                 solo_trimestre=True):
            por_fecha.setdefault(punto["fecha"], punto)
    return [por_fecha[f] for f in sorted(por_fecha)]


def _mediana(vals):
    v = sorted(vals)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def _fuera_de_escala(serie, factor=FACTOR_ESCALA):
    """
    Descarta lo que esta a ORDENES DE MAGNITUD de la mediana de la serie.

    Hace falta ademas del despicado local porque los errores de unidad vienen
    de a RACHAS: HL tiene dos trimestres seguidos en ~6,3e11 contra una serie
    de ~620M, y con dos puntos malos consecutivos los vecinos de cada uno ya no
    concuerdan, asi que el filtro local no los ve.

    El umbral es deliberadamente flojo (x100). El split mas grande del universo
    es 20:1 y la deriva de una empresa en 8 anios no llega a un orden de
    magnitud, asi que nada legitimo se acerca; los errores de unidad son
    factores de 1000.
    """
    med = _mediana([p["shares"] for p in serie])
    if not med:
        return list(serie)
    return [p for p in serie
            if (1.0 / factor) <= (p["shares"] / med) <= factor]


def despicar(serie, factor=1.6):
    """
    Saca los PICOS AISLADOS: un punto que se va y vuelve.

    La distincion que hace esto posible: un split desplaza el nivel de forma
    PERMANENTE (el punto anterior y el posterior NO concuerdan entre si), y un
    error de dato es una excursion de un solo punto (anterior y posterior SI
    concuerdan). Por eso se puede limpiar lo segundo sin tocar lo primero.

    Casos reales que motivan la funcion:
      HL   612.636.803 -> 617.339.547.000 -> 618.232.871   (error de unidad x1000)
      WFC  3.752.223.519 -> 1.823.028.137 -> 3.631.639.714 (conteo parcial)
    y los que NO se tocan, porque el nivel queda desplazado:
      AAPL 4.275.634.000 -> 17.001.802.000 -> 17.0xx       (split 4:1 real)

    Los extremos de la serie se dejan como estan: sin dos vecinos no hay con
    que juzgarlos, e inventar un criterio para ellos seria peor que el pico.
    """
    if len(serie) < 3:
        return list(serie)
    out = [serie[0]]
    for prev, act, sig in zip(serie, serie[1:], serie[2:]):
        a, b, c = prev["shares"], act["shares"], sig["shares"]
        vecinos_concuerdan = a > 0 and c > 0 and (1.0 / factor) <= (c / a) <= factor
        if vecinos_concuerdan:
            ref = (a + c) / 2.0
            if ref > 0 and not ((1.0 / factor) <= (b / ref) <= factor):
                continue                 # pico aislado: se descarta
        out.append(act)
    out.append(serie[-1])
    return out


def serie_acciones(companyfacts, desde=None):
    """
    La serie a usar. Portada si la hay; si no, el respaldo del promedio
    ponderado. NO se mezclan: son magnitudes distintas y alternar entre ellas
    meteria escalones que no ocurrieron.
    """
    s = serie_portada(companyfacts, desde=desde)
    if not s:
        s = serie_promedio(companyfacts, desde=desde)
    # Primero la escala (saca rachas), despues los picos aislados.
    return despicar(_fuera_de_escala(s))


def acciones_asof(serie, fecha):
    """
    Conteo vigente en `fecha`: el de la ultima portada anterior o igual.
    None si en esa fecha no habia ninguna todavia.

    Es una funcion ESCALON: entre dos filings el conteo se mantiene. No se
    interpola -- una recompra progresiva es real, pero inventarle una pendiente
    seria fabricar dato que SEC no publica.
    """
    if not serie:
        return None
    f = str(fecha)[:10]
    ultimo = None
    for punto in serie:
        if punto["fecha"] > f:
            break
        ultimo = punto
    return ultimo


def recorrer_asof(serie, fechas):
    """
    acciones_asof() sobre una lista ORDENADA de fechas, en una sola pasada.
    Devuelve [(fecha, punto_o_None), ...].
    """
    salida, j, actual = [], 0, None
    for fecha in fechas:
        f = str(fecha)[:10]
        while j < len(serie) and serie[j]["fecha"] <= f:
            actual = serie[j]
            j += 1
        salida.append((f, actual))
    return salida


def saltos(serie, umbral=1.6):
    """
    Cambios bruscos de conteo entre portadas consecutivas. Diagnostico, no
    correccion: un salto puede ser un split REAL (y entonces el precio salta
    igual y el market cap queda bien) o un problema de dato.

    Devuelve [{desde, hasta, ratio, shares_antes, shares_despues}].
    """
    out = []
    for a, b in zip(serie, serie[1:]):
        if not a["shares"]:
            continue
        r = b["shares"] / a["shares"]
        if r > umbral or r < 1.0 / umbral:
            out.append({"desde": a["fecha"], "hasta": b["fecha"],
                        "ratio": r, "shares_antes": a["shares"],
                        "shares_despues": b["shares"]})
    return out
