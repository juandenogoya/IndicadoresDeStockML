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
# universo es 1.5:1 -- 10% deja un margen holgado contra eso.
TOL_BASE = 0.10

# Salto entre puntos consecutivos que se considera cambio de base.
UMBRAL_SALTO = 1.5

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


def construir(serie_yahoo, serie_sec, desde, tol=TOL_BASE,
              umbral=UMBRAL_SALTO, dias=DIAS_APAREO):
    """
    Serie final de acciones para un ticker, y el veredicto de la validacion.

    Devuelve (serie, diagnostico).
      serie       : [{fecha, shares, periodo, fuente}] ordenada, sin duplicados
                    de fecha, con Yahoo teniendo prioridad donde llega.
      diagnostico : {extendido, desde_efectivo, validacion, motivo, n_yahoo,
                     n_sec_usados}

    La extension con SEC hacia atras requiere DOS condiciones, las dos
    verificadas y ninguna asumida:
      1. las fuentes coinciden en el solapamiento (misma base ahi), y
      2. la serie SEC no tiene saltos en el tramo que se quiere agregar.
    Si falla cualquiera, se devuelve solo Yahoo y el diagnostico dice por que.
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

    extra = [{"fecha": _iso(p["fecha"]), "shares": float(p["shares"]),
              "periodo": "q", "fuente": "sec_portada"} for p in previos]
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
