"""
acciones_series.py
Arma la serie de acciones en circulacion combinando dos fuentes, y VALIDA que
esten en la misma base antes de mezclarlas.

Funcion PURA: sin DB, sin red, sin config. Solo stdlib. ASCII en strings.

El problema que resuelve
------------------------
Para multiplicar un precio por un conteo de acciones, los dos tienen que estar
en la MISMA base de split. `precios_diarios` se corrige retroactivamente (toda
la historia queda en base actual), asi que el conteo tambien tiene que estarlo.

    yahooquery OrdinarySharesNumber -> base ACTUAL siempre (Yahoo re-expresa),
                                       pero solo ~4 puntos anuales + 5
                                       trimestrales: llega a 2022/2023.
    SEC portada (dei)               -> base DE SU MOMENTO, pero llega a 2018.

Ninguna de las dos alcanza sola. La combinacion es valida SOLO si el ticker no
cambio de base en el tramo que se quiere usar, y eso NO se asume: se COMPRUEBA
comparando las dos series donde se solapan. Si ahi coinciden, la serie SEC esta
en base actual en ese tramo; si ademas no tiene saltos mas atras, sigue estando
en base actual hasta el principio y se puede usar.

Reglas de construccion, y por que
---------------------------------
1. ESCALON, NUNCA INTERPOLACION. Entre dos puntos el conteo se mantiene. Un
   escalon es un dato viejo y se puede etiquetar con su antiguedad; una
   interpolacion es un dato inventado e indistinguible despues. Ademas
   suavizaria justo las discontinuidades reales (una fusion, una emision), que
   son la cola del error -- medido: mediana 0,24% pero p99 11,35%, con casos de
   30% (BG, CAR, SPGI en su fusion con IHS Markit).

2. YAHOO MANDA DONDE LLEGA. Es base actual por construccion; SEC solo extiende
   hacia atras, nunca pisa a Yahoo.

3. LA DISCREPANCIA AVISA, NO CORRIGE. Si las dos fuentes difieren fuera de
   tolerancia, no se elige una: se marca el ticker y se usa solo Yahoo. Una
   correccion automatica sobre una discrepancia que no se entendio es como se
   fabrican los errores silenciosos.
"""

# Cuanto pueden diferir las dos fuentes y seguir considerandose la misma base.
# Amplio a proposito: los dos conteos estan fechados con semanas de diferencia
# (Yahoo al cierre fiscal, SEC a la portada del filing) y entre medio hay
# recompras reales. Lo que tiene que separar es un SPLIT, y el mas chico del
# universo es 1.5:1 -- o sea 50% de desvio. 12% deja margen de sobra.
#
# El 0.12 no es a ojo. Barrido sobre los 200 tickers a 0.10 / 0.12 / 0.15 / 0.20:
#     0.10 -> 135 extendidos
#     0.12 -> 137  (+TRIP 10.2%, +RKLB 11.2%; ninguno perdido)
#     0.15 -> 137  (nada nuevo)
#     0.20 -> 137  (nada nuevo)
# Hay una BANDA VACIA entre 11.2% y 20%: los tickers caen o en deriva ordinaria
# o muy lejos, nunca en el medio. Por eso el valor exacto no es un filo de
# navaja, y por eso subirlo mas no compra nada -- lo que falta despues de 0.12
# no falla por tolerancia (HON esta 50% afuera, V no tiene con que solapar).
#
# RIESGO QUE QUEDA, dicho explicito: esta guarda NUNCA protegio contra usar una
# CLASE de acciones en vez del total. Un filer multiclase cuya clase menor pese
# ~12% del total pasaria; con 0.10 pasaba uno que pesara ~10%. El agujero es el
# mismo, apenas mas ancho. Lo que lo cierra es que las series salgan de tags sin
# dimension o de Polygon (weighted_shares_outstanding), no este numero.
TOL_BASE = 0.12

# Salto entre puntos consecutivos que se considera cambio de base.
UMBRAL_SALTO = 1.5

# Cuanto puede cambiar el conteo entre el ULTIMO punto SEC que se agrega y el
# PRIMER punto de Yahoo. Son un trimestre vecinos: recompras y emisiones mueven
# unidades porcentuales, no multiplos.
#
# Esta guarda existe porque las otras dos NO alcanzan, y se descubrio con datos:
# validar_base() compara los tramos que se SOLAPAN, y sin_saltos() mira la
# serie SEC por dentro. Si la serie SEC es internamente coherente pero esta en
# otra base que Yahoo -- porque el split ocurrio DESPUES del ultimo punto SEC --
# las dos dan el visto bueno y el escalon aparece justo en la juntura. Medido:
# AVGO empalmaba con x9,72 (split 10:1) y WMT con x2,99 (3:1), o sea market cap
# 10 y 3 veces mas chico en todo el tramo agregado.
#
# 1,25 separa limpio: los empalmes sanos dan mediana 0,49% y el peor legitimo
# 13,8% (ASAN, que diluye fuerte por stock based compensation).
UMBRAL_EMPALME = 1.25

# Ventana para aparear un punto de una serie con el de la otra.
DIAS_APAREO = 60


def _fecha(v):
    from datetime import date
    if v is None:
        return None
    if hasattr(v, "year"):
        return v
    try:
        p = str(v)[:10].split("-")
        return date(int(p[0]), int(p[1]), int(p[2]))
    except (ValueError, IndexError):
        return None


def _iso(v):
    f = _fecha(v)
    return f.isoformat() if f is not None else None


def _dias(a, b):
    fa, fb = _fecha(a), _fecha(b)
    return None if fa is None or fb is None else abs((fb - fa).days)


def aparear(serie_a, serie_b, dias=DIAS_APAREO):
    """
    Pares (punto_a, punto_b) de fechas cercanas. Cada punto de `a` se aparea
    con el de `b` mas proximo dentro de `dias`. Sin par -> no entra.
    """
    pares = []
    for pa in serie_a:
        mejor, mejor_d = None, None
        for pb in serie_b:
            d = _dias(pa["fecha"], pb["fecha"])
            if d is None or d > dias:
                continue
            if mejor_d is None or d < mejor_d:
                mejor, mejor_d = pb, d
        if mejor is not None:
            pares.append((pa, mejor))
    return pares


def validar_base(serie_yahoo, serie_sec, tol=TOL_BASE, dias=DIAS_APAREO):
    """
    Compara las dos fuentes donde se solapan.

    Devuelve {ok, n_pares, ratio_min, ratio_max, motivo}:
      ok=True   las dos coinciden en todo el solapamiento -> misma base
      ok=False  difieren (hubo split entre medio) o no hay con que comparar

    `ok=False` por falta de solapamiento NO es un error: es que no se puede
    afirmar nada, y el criterio es no extender.
    """
    pares = aparear(serie_yahoo, serie_sec, dias)
    if not pares:
        return {"ok": False, "n_pares": 0, "ratio_min": None,
                "ratio_max": None, "motivo": "sin solapamiento"}
    ratios = []
    for py, ps in pares:
        if not ps["shares"]:
            continue
        ratios.append(py["shares"] / ps["shares"])
    if not ratios:
        return {"ok": False, "n_pares": len(pares), "ratio_min": None,
                "ratio_max": None, "motivo": "sin valores comparables"}
    rmin, rmax = min(ratios), max(ratios)
    ok = (1 - tol) <= rmin and rmax <= (1 + tol)
    motivo = None if ok else f"ratio yahoo/sec entre {rmin:.3f} y {rmax:.3f}"
    return {"ok": ok, "n_pares": len(ratios), "ratio_min": rmin,
            "ratio_max": rmax, "motivo": motivo}


def sin_saltos(serie, desde=None, umbral=UMBRAL_SALTO):
    """
    True si la serie no cambia de base en el tramo. Un salto de >=umbral entre
    puntos consecutivos es un split o un evento de capital: en cualquiera de los
    dos casos no se puede afirmar que el tramo previo este en la base de hoy.
    """
    pts = [p for p in serie if desde is None or _iso(p["fecha"]) >= _iso(desde)]
    for a, b in zip(pts, pts[1:]):
        if not a["shares"]:
            continue
        r = b["shares"] / a["shares"]
        if r >= umbral or r <= 1.0 / umbral:
            return False, f"salto x{r:.2f} en {_iso(b['fecha'])}"
    return True, None


def rebasar(serie, splits):
    """
    Lleva una serie de conteos, cada uno en la base de SU momento, a la base
    de HOY. Devuelve (serie_nueva, factores_aplicados).

    POR QUE HACE FALTA
    Las dos fuentes point-in-time de acciones -- la portada de SEC y el
    conteo de Polygon -- publican lo que era cierto ESE dia. `precios_diarios`,
    en cambio, se corrige retroactivamente por divisor cuando aparece un split
    (ver "Splits" en CLAUDE.md), o sea que esta en la base de hoy. Multiplicar
    un precio de hoy por un conteo de ayer da un market cap dividido por el
    factor del split.

    Medido: 4.218 de las 6.396 ruedas sin market cap tienen esta unica causa.
    Las guardas las rechazan bien -- solas no pueden distinguir un split de un
    error de dato -- pero con una lista AUTORITATIVA de splits el ajuste deja
    de ser una adivinanza y pasa a ser una multiplicacion.

    `splits` : [{fecha, ratio}] con ratio = split_to / split_from. Un 10:1 da
               10: las acciones se multiplican por 10 y el precio se divide.

    Cada punto se multiplica por el producto de los splits POSTERIORES a el.
    Los posteriores y no los anteriores: un conteo de 2021 todavia no "sabe"
    del split de 2022, asi que hay que aplicarselo.

    EL CORTE ES `filed`, NO `fecha`, y esa distincion vale toda la funcion.
    Un numero esta en la base vigente cuando se PRESENTO, no cuando cerro el
    trimestre que describe. El nivel balance lo hace evidente: GOOG declara
    658.763.000 acciones al 2022-03-31 (10-Q de abril) y 13.078.000.000 al
    2022-06-30 (10-Q de julio, ya con el split del 18/7 aplicado). Las dos
    fechas de periodo son PREVIAS al split; una de las cifras ya esta
    re-expresada y la otra no. Cortando por `fecha` se rebasan las dos, queda
    un salto de x19,85 y la serie se rechaza entera. Cortando por `filed`
    cada una recibe lo que le toca y la serie queda continua.
    (`_recolectar` guarda el `filed` MAS TEMPRANO de cada punto -- la
    presentacion original -- que es justo el momento cuya base rige.)
    Si un punto no trae `filed`, se cae a `fecha`.

    EL BORDE ES ESTRICTO (`>`, no `>=`): la fecha que publica Polygon es la de
    EJECUCION y puede caer uno o dos dias despues de la ex-date que registra
    el proyecto (KLAC figura 2026-06-12 y en CLAUDE.md quedo el 11/6).

    ESTA FUNCION NO SABE SI EL EVENTO ES UN SPLIT DE VERDAD, y no puede
    saberlo. Polygon mezcla en el mismo endpoint los splits reales con los
    ajustes de PRECIO por spinoff, que no mueven el conteo de acciones: HON
    1,061 es la escision de Solstice, IBM 1,046 la de Kyndryl, MMM 1,196 la
    de Solventum, DELL 1,973 la de VMware, GSK 0,8 la de Haleon. Aplicarles
    el factor CORROMPE la serie (HON empeoro de 0,500 a 0,471). Filtrar por
    "ratio plausible" tampoco alcanza: BBD 1,1 e ITUB 1,03 son bonificaciones
    brasileras que SI cambian el conteo. Por eso el que decide no es esta
    funcion sino el que la llama, corriendo la validacion con y sin rebase y
    quedandose con la que pasa. Adivinar la accion corporativa es un problema
    que no tenemos como resolver; medir cual de las dos series valida, si.
    """
    if not splits:
        return list(serie), []
    ss = sorted(({"fecha": _iso(x["fecha"]), "ratio": float(x["ratio"])}
                 for x in splits if x.get("ratio")), key=lambda x: x["fecha"])
    salida, usados = [], []
    for p in serie:
        f = _iso(p.get("filed") or p["fecha"])
        factor = 1.0
        for x in ss:
            if x["fecha"] > f:
                factor *= x["ratio"]
        q = dict(p)
        if factor != 1.0 and p.get("shares"):
            q["shares"] = float(p["shares"]) * factor
            q["rebase"] = factor
            usados.append((f, factor))
        salida.append(q)
    return salida, usados


def construir(serie_yahoo, serie_sec, desde, tol=TOL_BASE,
              umbral=UMBRAL_SALTO, dias=DIAS_APAREO, etiqueta="sec_portada"):
    """
    Serie final de acciones para un ticker, y el veredicto de la validacion.

    Devuelve (serie, diagnostico).
      serie       : [{fecha, shares, periodo, fuente}] ordenada, sin duplicados
                    de fecha, con Yahoo teniendo prioridad donde llega.

    `etiqueta` es el valor que va a la columna `fuente` de los puntos que
    aporta SEC. Hay tres niveles de serie SEC (portada / balance / promedio
    ponderado) y no son la misma medicion, asi que el nombre tiene que decir
    cual entro: si todos se guardan como "sec_portada", un promedio del
    trimestre pasa por un conteo a una fecha y nadie se entera.
      diagnostico : {extendido, desde_efectivo, validacion, motivo, n_yahoo,
                     n_sec_usados}

    La extension con SEC hacia atras requiere TRES condiciones, las tres
    verificadas y ninguna asumida:
      1. las fuentes coinciden en el solapamiento (misma base ahi),
      2. la serie SEC no tiene saltos en el tramo que se quiere agregar, y
      3. el EMPALME es continuo: el ultimo punto SEC y el primero de Yahoo no
         pueden diferir por un multiplo.
    Si falla cualquiera, se devuelve solo Yahoo y el diagnostico dice por que.

    La tercera no es redundante con las otras dos: cuando el split ocurre
    DESPUES del ultimo punto SEC, la serie SEC es coherente consigo misma y el
    solapamiento que se compara cae todo del mismo lado, asi que 1 y 2 aprueban
    y el escalon entra igual.
    """
    yah = sorted((p for p in serie_yahoo if p.get("shares")),
                 key=lambda p: _iso(p["fecha"]))
    sec = sorted((p for p in serie_sec if p.get("shares")),
                 key=lambda p: _iso(p["fecha"]))
    diag = {"extendido": False, "desde_efectivo": None, "validacion": None,
            "motivo": None, "n_yahoo": len(yah), "n_sec_usados": 0}
    if not yah:
        diag["motivo"] = "sin datos de yahoo"
        return [], diag

    salida = [{"fecha": _iso(p["fecha"]), "shares": float(p["shares"]),
               "periodo": p.get("periodo"), "fuente": "yahooquery"}
              for p in yah]
    diag["desde_efectivo"] = salida[0]["fecha"]

    val = validar_base(yah, sec, tol, dias)
    diag["validacion"] = val
    if not val["ok"]:
        diag["motivo"] = val["motivo"]
        return salida, diag

    # Tramo de SEC anterior al primer punto de Yahoo.
    corte = salida[0]["fecha"]
    previos = [p for p in sec if _iso(p["fecha"]) < corte
               and _iso(p["fecha"]) >= _iso(desde)]
    if not previos:
        diag["motivo"] = "yahoo ya cubre desde el inicio pedido"
        return salida, diag

    # El chequeo de saltos incluye el primer punto posterior al corte: un split
    # justo entre el ultimo SEC previo y el arranque de Yahoo quedaria afuera.
    tramo = previos + [p for p in sec if _iso(p["fecha"]) >= corte][:1]
    limpio, motivo = sin_saltos(tramo, umbral=umbral)
    if not limpio:
        diag["motivo"] = motivo
        return salida, diag

    # El empalme mismo tiene que ser continuo. Ver UMBRAL_EMPALME.
    r_emp = salida[0]["shares"] / previos[-1]["shares"] if previos[-1]["shares"] else None
    if r_emp is None or r_emp >= UMBRAL_EMPALME or r_emp <= 1.0 / UMBRAL_EMPALME:
        diag["motivo"] = (f"empalme x{r_emp:.2f} entre {_iso(previos[-1]['fecha'])} "
                          f"y {salida[0]['fecha']}" if r_emp else "empalme sin valor")
        return salida, diag

    extra = [{"fecha": _iso(p["fecha"]), "shares": float(p["shares"]),
              "periodo": "q", "fuente": etiqueta} for p in previos]
    diag.update({"extendido": True, "n_sec_usados": len(extra),
                 "desde_efectivo": extra[0]["fecha"]})
    return extra + salida, diag


def asof(serie, fecha):
    """Punto vigente en `fecha` (escalon). None si no hay ninguno anterior."""
    f = _iso(fecha)
    ultimo = None
    for p in serie:
        if _iso(p["fecha"]) > f:
            break
        ultimo = p
    return ultimo


def recorrer_asof(serie, fechas):
    """asof() sobre una lista ORDENADA de fechas, en una pasada."""
    salida, j, actual = [], 0, None
    for fecha in fechas:
        f = _iso(fecha)
        while j < len(serie) and _iso(serie[j]["fecha"]) <= f:
            actual = serie[j]
            j += 1
        salida.append((f, actual))
    return salida
