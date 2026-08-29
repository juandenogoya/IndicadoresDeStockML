"""
valuacion_implicita.py -- escenarios de valuacion implicita.

Modulo PURO: stdlib, sin DB, sin red, sin config. Recibe dicts y listas.

Que responde
------------
Las dos direcciones de la misma pregunta:

  DIRECTA   "si el precio fuera 250, que PER / EV-EBITDA / P-S implica,
            y donde cae eso en la propia historia de la empresa?"
  INVERSA   "que precio hace falta para que su PER vuelva a la mediana de
            los ultimos 3 anios?"

La inversa suele ser la util: convierte un multiplo, que es abstracto, en un
precio, que es accionable, sin pedirle al que lee que haga la cuenta de cabeza.

Las tres reglas del calculo
---------------------------
1. TODO SALE DE AGREGADOS. market_cap = precio * acciones; EV = market_cap +
   deuda_neta. Nunca se pasa por magnitudes "por accion" (EPS, BVPS): SEC las
   re-expresa ante un split y precios_diarios tambien, pero con horizontes
   distintos. Es la misma regla que gobierna fundamentales_sec_multiplos_d y
   esta explicada en src/utils/sec_acciones.py.

2. AL MOVER EL PRECIO SOLO SE MUEVE EL EQUITY. La deuda neta y los
   denominadores TTM se mantienen: el escenario es sobre el PRECIO, no sobre el
   negocio. Por eso EV = market_cap_implicito + net_debt con el MISMO net_debt.
   Un escenario que ademas cambie las ventas o el margen es otra herramienta.

3. UN DENOMINADOR NEGATIVO NO PRODUCE MULTIPLO. Un PER con resultado negativo
   no es "barato", es otra categoria; emitirlo ensuciaria la comparacion contra
   la historia, que esta hecha de numeros positivos. Se devuelve None, igual que
   en la capa diaria.

Lo que este modulo NO hace, y conviene tenerlo escrito
-----------------------------------------------------
El percentil dice DONDE cae el multiplo dentro de su propio rango. No dice si
ese rango esta justificado. Parte de el viene del REGIMEN DE TASAS y no de la
empresa: un PER de 20 en 2021 con la tasa en cero no significa lo mismo que un
PER de 20 hoy. "Barato contra si misma" es una observacion, no una tesis.
"""

# Metricas soportadas y de que se componen. La tupla es
#   (denominador TTM, si el numerador es el EV en vez del market cap)
# Un solo lugar donde esta escrito como se arma cada multiplo, para que la
# version directa y la inversa no puedan divergir.
METRICAS = {
    "pe_ratio": ("net_income_ttm", False),
    "pb_ratio": ("equity", False),
    "ps_ratio": ("revenue_ttm", False),
    "ev_ebitda": ("ebitda_ttm", True),
}

# fcf_yield va aparte: es el unico que se invierte (numerador y denominador
# cambian de lugar) y el unico que admite numerador negativo -- quemar caja es
# informacion, y la escala no se rompe porque el market cap siempre es > 0.
FCF = "fcf_ttm"


def _num(v):
    """A float, o None. Acepta Decimal de psycopg2 sin que el llamador lo sepa."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # NaN


def market_cap(precio, acciones):
    p, a = _num(precio), _num(acciones)
    if p is None or a is None or p <= 0 or a <= 0:
        return None
    return p * a


def multiplos(base, precio):
    """
    Multiplos implicitos por `precio`, con los denominadores TTM de `base`.

    base: dict con acciones/shares, net_debt y los *_ttm. Es exactamente una
    fila de fundamentales_sec_multiplos_d, para que el llamador no tenga que
    traducir nada.
    """
    acciones = base.get("shares", base.get("acciones"))
    mc = market_cap(precio, acciones)
    if mc is None:
        return {k: None for k in list(METRICAS) + ["market_cap",
                                                   "enterprise_value",
                                                   "fcf_yield"]}
    nd = _num(base.get("net_debt"))
    ev = (mc + nd) if nd is not None else None

    out = {"market_cap": mc, "enterprise_value": ev}
    for metrica, (campo, usa_ev) in METRICAS.items():
        den = _num(base.get(campo))
        num = ev if usa_ev else mc
        out[metrica] = (num / den) if (num is not None and den is not None
                                       and den > 0) else None
    fcf = _num(base.get(FCF))
    out["fcf_yield"] = (fcf / mc) if fcf is not None else None
    return out


def precio_para(base, metrica, objetivo):
    """
    INVERSA: que precio hace que `metrica` valga `objetivo`.

    Despeja el precio de la misma formula que usa multiplos(), no de una
    aproximacion. Para las metricas sobre EV hay que restar la deuda neta antes
    de dividir por las acciones: es la diferencia entre lo que vale la empresa
    y lo que vale su equity, y saltearla es el error clasico de este calculo.

    Devuelve None cuando el despeje no tiene sentido economico (denominador no
    positivo, o un EV objetivo menor que la deuda neta -- que implicaria un
    equity negativo).
    """
    if metrica == "fcf_yield":
        fcf, acciones = _num(base.get(FCF)), _num(base.get("shares"))
        obj = _num(objetivo)
        if fcf is None or not obj or acciones is None or acciones <= 0:
            return None
        mc = fcf / obj
        return (mc / acciones) if mc > 0 else None

    if metrica not in METRICAS:
        return None
    campo, usa_ev = METRICAS[metrica]
    den = _num(base.get(campo))
    obj = _num(objetivo)
    acciones = _num(base.get("shares", base.get("acciones")))
    if den is None or den <= 0 or obj is None or obj <= 0:
        return None
    if acciones is None or acciones <= 0:
        return None

    objetivo_num = obj * den                      # market cap o EV objetivo
    if usa_ev:
        nd = _num(base.get("net_debt"))
        if nd is None:
            return None
        objetivo_num -= nd                        # de EV a market cap
    return (objetivo_num / acciones) if objetivo_num > 0 else None


# ---------------------------------------------------------------------------
# Ubicacion en la historia propia
# ---------------------------------------------------------------------------

def percentil_de(serie, valor):
    """
    En que percentil de `serie` cae `valor`, entre 0 y 1.

    Definicion: la FRACCION de observaciones historicas menores o iguales. Se
    elige esta y no una interpolacion porque el resultado se lee como "estuvo
    mas caro que el 80% de su historia", que es literalmente esta cuenta.

    Las observaciones None se descartan: son ruedas sin multiplo (denominador
    negativo, TTM incompleto), no ceros.
    """
    vals = [v for v in (_num(x) for x in (serie or [])) if v is not None]
    v = _num(valor)
    if not vals or v is None:
        return None
    return sum(1 for x in vals if x <= v) / len(vals)


def cuantil(serie, p):
    """
    Valor de `serie` en el percentil p (0 a 1), por interpolacion lineal.

    Es la operacion inversa de percentil_de y se usa para la pregunta util:
    "que precio implica volver a la mediana". Se interpola porque aca el
    resultado alimenta una cuenta, no una lectura.
    """
    vals = sorted(v for v in (_num(x) for x in (serie or [])) if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    p = min(max(_num(p) or 0.0, 0.0), 1.0)
    pos = p * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


# ---------------------------------------------------------------------------
# API de escenario
# ---------------------------------------------------------------------------

def escenario(base, historia, precio_objetivo=None, variacion=None):
    """
    base     : fila de multiplos vigente (shares, net_debt, *_ttm, close).
    historia : {metrica: [valores historicos del propio ticker]}.
    precio_objetivo / variacion: la tesis. `variacion` es una fraccion
               (0.20 = +20%) y se aplica sobre base["close"]. Se usa una u
               otra; si vienen las dos gana precio_objetivo.

    Devuelve {precio, variacion, multiplos: {...}, percentiles: {...},
              fuera_de_rango: [...]}.

    `fuera_de_rango` lista las metricas cuyo valor implicito nunca se vio en la
    historia disponible. No es un veredicto sobre la tesis: es el aviso de que
    para sostenerla hay que creer algo que todavia no paso, y eso merece
    argumento aparte.
    """
    close = _num(base.get("close"))
    precio = _num(precio_objetivo)
    if precio is None and variacion is not None and close is not None:
        precio = close * (1.0 + (_num(variacion) or 0.0))
    if precio is None:
        precio = close

    m = multiplos(base, precio)
    historia = historia or {}
    pcts, fuera = {}, []
    for metrica in list(METRICAS) + ["fcf_yield"]:
        serie = historia.get(metrica)
        pcts[metrica] = percentil_de(serie, m.get(metrica))
        vals = [v for v in (_num(x) for x in (serie or [])) if v is not None]
        v = m.get(metrica)
        if vals and v is not None and (v > max(vals) or v < min(vals)):
            fuera.append(metrica)

    return {
        "precio": precio,
        "precio_actual": close,
        "variacion": ((precio / close - 1.0)
                      if (close and precio is not None) else None),
        "multiplos": m,
        "percentiles": pcts,
        "fuera_de_rango": fuera,
    }


def precios_de_referencia(base, historia, percentiles=(0.10, 0.25, 0.50,
                                                       0.75, 0.90)):
    """
    Para cada metrica y cada percentil, que precio implica.

    Es la vista que convierte "esta en el percentil 85 de su PER" en "para
    volver a su mediana el precio tendria que ser 178". Devuelve
    {metrica: {percentil: precio}}, con None donde el despeje no aplica.
    """
    salida = {}
    for metrica in list(METRICAS) + ["fcf_yield"]:
        serie = (historia or {}).get(metrica)
        fila = {}
        for p in percentiles:
            objetivo = cuantil(serie, p)
            fila[p] = (precio_para(base, metrica, objetivo)
                       if objetivo is not None else None)
        salida[metrica] = fila
    return salida


# ---------------------------------------------------------------------------
# La direccion inversa: que tiene que hacer el NEGOCIO
# ---------------------------------------------------------------------------
#
# Todo lo de arriba mueve el precio y deja el negocio quieto. Esta seccion hace
# exactamente lo contrario: fija el precio en la tesis, fija el multiplo en una
# referencia de la propia historia, y despeja cuanto tiene que valer el
# denominador.
#
# Son dos preguntas opuestas y las dos son legitimas. Lo que NO es legitimo es
# mezclarlas: un escenario que mueve el precio Y el negocio al mismo tiempo
# tiene dos incognitas y una sola ecuacion, y cualquier numero que devuelva es
# una eleccion disfrazada de calculo.
#
# Por que esto importa: un +20% de precio mueve TODOS los multiplos de precio
# exactamente +20%. Eso es aritmetica, no informacion. Lo que convierte "PER
# 43" en una respuesta es contra que se lo compara -- su propia historia -- y
# que hace falta que pase para llegar ahi.


def denominador_para(base, metrica, objetivo, precio):
    """
    Cuanto tiene que valer el denominador TTM para que `metrica` valga
    `objetivo` con el precio en `precio`.

    Espejo exacto de precio_para(): alla se despeja el precio dejando el
    negocio quieto, aca se despeja el negocio dejando el precio quieto. Sale de
    la misma formula que multiplos(), no de una aproximacion.

    Devuelve un valor en la MONEDA Y ESCALA de los *_ttm de `base` (millones,
    en la fuente SEC), no un ratio.
    """
    p, obj = _num(precio), _num(objetivo)
    acciones = _num(base.get("shares", base.get("acciones")))
    if p is None or p <= 0 or obj is None:
        return None
    if acciones is None or acciones <= 0:
        return None
    mc = p * acciones

    # fcf_yield va primero porque es el unico que se lee al reves (el fcf esta
    # en el numerador) y el unico que admite objetivo negativo.
    if metrica == "fcf_yield":
        return mc * obj

    if metrica not in METRICAS or obj <= 0:
        return None
    campo, usa_ev = METRICAS[metrica]
    num = mc
    if usa_ev:
        nd = _num(base.get("net_debt"))
        if nd is None:
            return None                       # EV sin deuda conocida no existe
        num = mc + nd
    return (num / obj) if num > 0 else None


def exigencia(base, historia, precio, referencia=0.50):
    """
    Que tiene que hacer el negocio para que `precio` sea consistente con un
    multiplo NORMAL de la propia empresa.

    Fija el multiplo en el percentil `referencia` de su historia (la mediana
    por defecto) y despeja el denominador. Por metrica devuelve:

      multiplo_objetivo   el de referencia, sacado de su propia historia
      multiplo_implicito  el que resulta de `precio` sin tocar el negocio
      actual              denominador TTM de hoy
      requerido           denominador que haria falta
      crecimiento         requerido/actual - 1

    El crecimiento es identicamente multiplo_implicito/multiplo_objetivo - 1
    (vale tambien para las metricas sobre EV, porque la deuda neta se cancela
    al dividir). Igual se calcula por la via larga: asi la formula de cada
    multiplo vive en UN solo lugar y la directa y la inversa no pueden
    divergir en silencio.
    """
    implicitos = multiplos(base, precio)
    salida = {}
    for metrica in list(METRICAS) + ["fcf_yield"]:
        campo = FCF if metrica == "fcf_yield" else METRICAS[metrica][0]
        objetivo = cuantil((historia or {}).get(metrica), referencia)
        actual = _num(base.get(campo))
        requerido = (denominador_para(base, metrica, objetivo, precio)
                     if objetivo is not None else None)
        crecimiento = None
        if requerido is not None and actual is not None and actual > 0:
            crecimiento = requerido / actual - 1.0
        salida[metrica] = {"multiplo_objetivo": objetivo,
                           "multiplo_implicito": implicitos.get(metrica),
                           "actual": actual,
                           "requerido": requerido,
                           "crecimiento": crecimiento}
    return salida


def crecimiento_historico(serie, pasos=4):
    """
    Distribucion del crecimiento interanual de una serie TTM TRIMESTRAL.

    serie: [(period_end, valor)] ordenada por period_end y SIN repetidos.

    La serie DIARIA no sirve aca, y es la trampa de esta cuenta: el TTM es una
    escalera que solo cambia cuando sale un balance, asi que medir su
    crecimiento en cada rueda multiplica por ~63 las observaciones sin agregar
    una sola. El n quedaria inflado y la dispersion, aplastada.

    `pasos` = 4 compara contra el mismo trimestre del anio anterior, que es lo
    que hace comparable el numero contra el crecimiento requerido por una
    tesis anual.

    Devuelve la lista ORDENADA de crecimientos, lista para cuantil().
    """
    vals = [_num(v) for _, v in (serie or [])]
    out = []
    for i in range(pasos, len(vals)):
        previo, actual = vals[i - pasos], vals[i]
        if previo is not None and actual is not None and previo > 0:
            out.append(actual / previo - 1.0)
    return sorted(out)


def roe_implicito(base, net_income):
    """
    ROE que implica `net_income` contra el patrimonio de HOY.

    El patrimonio se deja quieto a proposito, y eso convierte el resultado en
    un TECHO: si la empresa gana mas y no reparte todo, el patrimonio crece y
    el ROE que hace falta es menor que este. Sirve para descartar lo imposible,
    no para proyectar.

    Se usa el equity contable tal cual. ROTCE (sobre patrimonio tangible) y
    ROIC (sobre capital invertido, que necesita NOPAT y por lo tanto una tasa
    impositiva) no salen de esta tabla: pedirian columnas que la capa derivada
    no tiene, y aproximarlos aca seria inventar un numero con cara de dato.
    """
    eq = _num(base.get("equity"))
    ni = _num(net_income)
    if eq is None or eq <= 0 or ni is None:
        return None
    return ni / eq
