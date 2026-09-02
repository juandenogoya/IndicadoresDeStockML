"""
dashboard/hoy.py
View model de la vista "Hoy": la pantalla de arranque del dashboard.

Por que existe:
    El dashboard abria en un menu de 7 vistas mas un selector de 200 tickers.
    Antes de ver un solo dato habia que tomar dos decisiones, y ninguna de las
    dos se puede tomar bien sin haber visto antes que paso en el mercado. La
    vista Hoy invierte eso: entra con la respuesta ("esto cambio, esto conviene
    mirar") y desde ahi se navega al detalle.

Que muestra, y por que en ese orden:
    1. Clima   -- como quedo repartido el universo (alcista/neutral/bajista).
    2. Inusual -- que se salio de su comportamiento historico en opciones.
    3. Cartera -- como viene el dinero en Forward Testing.
    4. Agenda  -- que balances vienen, marcando los que ya estan en cartera.
    De lo general a lo propio: primero el entorno, despues lo que te toca.
    El cruce de (4) con las posiciones abiertas es el unico dato de la vista
    que no existe en ninguna otra pantalla.

PURO: sin Streamlit, sin HTML, sin DB. Recibe lo que cargaron los loaders de
sintesis_data y devuelve estructuras listas para renderizar. Misma disciplina
que view.py -- es lo que permite que el informe tenga dos renderers (pantalla
y JPG) sin duplicar contenido.
"""

from datetime import date, datetime

DASH = "-"


def _pct(parte: int, total: int):
    return round(parte * 100.0 / total, 1) if total else None


def _num(v):
    """float() tolerante: la DB devuelve Decimal y puede haber None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _a_fecha(v):
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _cuando(fecha_str, hoy: date) -> str:
    """Proximidad en dias CORRIDOS. Un balance se anuncia en una fecha de
    calendario, asi que aca NO corresponde el calendario NYSE."""
    f = _a_fecha(fecha_str)
    if f is None:
        return DASH
    d = (f - hoy).days
    if d <= 0:
        return "hoy"
    if d == 1:
        return "manana"
    return f"en {d} dias"


# -- 1. Clima del universo ---------------------------------------------------

_ORDEN = ["ALCISTA", "NEUTRAL", "BAJISTA"]


def construir_clima(reparto: list) -> dict:
    """
    reparto: formato largo [{fecha, veredicto, n}] de cargar_reparto_veredictos.

    La frase compara contra la rueda ANTERIOR, no contra un absoluto: "55
    alcistas" no dice nada sin saber si la rueda pasada habia 30 o 90. Si
    todavia no hay dia anterior (la tabla se acaba de estrenar), lo dice en
    vez de inventar una tendencia.
    """
    if not reparto:
        return {"fecha": None, "total": 0, "filas": [], "delta": None,
                "frase": "Todavia no hay veredictos precomputados."}

    fechas = sorted({r["fecha"] for r in reparto})
    ult = fechas[-1]
    prev = fechas[-2] if len(fechas) > 1 else None

    def _conteo(f):
        return {r["veredicto"]: int(r["n"]) for r in reparto if r["fecha"] == f}

    conteo = _conteo(ult)
    total = sum(conteo.values())
    filas = [{"veredicto": v, "n": conteo.get(v, 0),
              "pct": _pct(conteo.get(v, 0), total)} for v in _ORDEN]

    delta = None
    if prev:
        anterior = _conteo(prev)
        delta = {v: conteo.get(v, 0) - anterior.get(v, 0) for v in _ORDEN}

    alc = conteo.get("ALCISTA", 0)
    baj = conteo.get("BAJISTA", 0)
    p_alc = _pct(alc, total)

    if delta is None:
        frase = (f"{alc} de {total} tickers ({p_alc}%) con veredicto alcista y "
                 f"{baj} bajista. Es el primer dia de historia: desde la "
                 f"proxima rueda se puede comparar contra la anterior.")
    else:
        d = delta["ALCISTA"]
        if d > 0:
            mov = f"{d} mas que la rueda anterior"
        elif d < 0:
            mov = f"{abs(d)} menos que la rueda anterior"
        else:
            mov = "los mismos que la rueda anterior"
        frase = (f"{alc} de {total} tickers ({p_alc}%) con veredicto alcista, "
                 f"{mov}. Bajistas: {baj}.")

    return {"fecha": ult, "total": total, "filas": filas,
            "delta": delta, "frase": frase}


# -- 2. Lo inusual del dia ---------------------------------------------------

def construir_inusual(filas_radar: list, fecha=None, top: int = 5) -> dict:
    """filas_radar: salida de dashboard.radar.construir_radar, ya ordenada por
    magnitud. Aca solo se recorta y se resume: la vista Hoy es un indice, el
    analisis vive en Radar del dia."""
    if not filas_radar:
        return {"fecha": fecha, "filas": [],
                "frase": "Sin actividad inusual en opciones con el umbral por defecto."}
    top_filas = filas_radar[:top]
    tickers = ", ".join(f["ticker"] for f in top_filas)
    frase = (f"{len(filas_radar)} tickers con actividad inusual en opciones "
             f"(z>=2). Los mas marcados: {tickers}.")
    return {"fecha": fecha, "filas": top_filas, "frase": frase}


# -- 3. Forward Testing ------------------------------------------------------

def construir_ft(resumen: dict) -> dict:
    """
    resumen: salida de cargar_ft_resumen (equity MARCADA A MERCADO).

    Se reportan mejor y peor estrategia, no un promedio: son 10 estrategias
    corriendo en paralelo justamente para compararlas, y el promedio esconde
    el unico dato que interesa, que es cual funciona.
    """
    if not resumen or not resumen.get("estrategias"):
        return {"fecha": None, "filas": [], "kpis": [],
                "frase": "Sin equity calculada. Correr ft_run_diario.bat."}

    tot = resumen["total"]
    ests = resumen["estrategias"]
    mejor = ests[0]
    peor = ests[-1]
    ret = _num(tot.get("retorno_acum_pct"))
    eq = _num(tot.get("equity"))

    kpis = [
        # "USD" va en la etiqueta y no en el valor: con 4 KPIs en 4 columnas,
        # el prefijo empuja el numero y Streamlit lo trunca con ellipsis.
        {"label": "Equity total (USD)", "valor": f"{eq:,.0f}" if eq else DASH},
        {"label": "Retorno acumulado",
         "valor": f"{ret:+.2f}%" if ret is not None else DASH},
        {"label": "Posiciones abiertas", "valor": str(tot.get("n_posiciones", 0))},
        {"label": "Estrategias", "valor": str(len(ests))},
    ]

    r_mejor = _num(mejor.get("retorno_acum_pct"))
    r_peor = _num(peor.get("retorno_acum_pct"))
    frase = (f"{len(ests)} estrategias en paralelo, {ret:+.2f}% agregado. "
             f"Mejor: {mejor['nombre']} ({r_mejor:+.2f}%). "
             f"Peor: {peor['nombre']} ({r_peor:+.2f}%).")
    if tot.get("stale"):
        frase += " Atencion: la equity se calculo con precios marcados como viejos."

    filas = []
    for e in ests:
        eq_e = _num(e["equity"])
        acum = _num(e["retorno_acum_pct"])
        dia = _num(e["retorno_dia_pct"])
        expo = _num(e["exposicion_pct"])
        filas.append({
            "Estrategia": e["nombre"],
            "Equity": f"{eq_e:,.0f}" if eq_e is not None else DASH,
            "Acum %": f"{acum:+.2f}" if acum is not None else DASH,
            "Dia %": f"{dia:+.2f}" if dia is not None else DASH,
            "Posic.": int(e["n_posiciones"] or 0),
            # OJO con las unidades: pese al nombre, exposicion_pct se guarda
            # como FRACCION (ft_compute_equity: valor_mercado / equity), no
            # como porcentaje. Sin el x100 un 0,9594 se mostraba como "1".
            # Los retorno_*_pct, en cambio, YA vienen multiplicados por 100.
            "Exposic. %": f"{expo * 100:.0f}" if expo is not None else DASH,
        })

    return {"fecha": resumen.get("fecha"), "filas": filas, "kpis": kpis,
            "frase": frase, "mejor": mejor["nombre"], "peor": peor["nombre"]}


# -- 4. Agenda: balances que vienen ------------------------------------------

def construir_agenda(earnings: list, posiciones: list, hoy=None) -> dict:
    """
    Cruza los balances proximos con las posiciones FT abiertas.

    Ese cruce es la razon de ser del bloque: "AVGO reporta" es informacion
    publica y esta en cualquier lado; "AVGO reporta HOY y lo tenes abierto en
    3 estrategias" es lo que obliga a hacer algo, y no aparece en ninguna otra
    pantalla del dashboard.
    """
    hoy = hoy or date.today()
    if not earnings:
        return {"filas": [], "n_en_cartera": 0,
                "frase": "Sin balances anunciados en la ventana."}

    abiertas = {p["ticker"]: p for p in (posiciones or [])}
    filas = []
    en_cartera = []
    for e in earnings:
        tk = e["ticker"]
        pos = abiertas.get(tk)
        filas.append({
            "Ticker": tk,
            "Sector": e.get("sector") or DASH,
            "Reporta": e["fecha"],
            "Cuando": _cuando(e["fecha"], hoy),
            "En cartera FT": (f"si ({int(pos['n_estrategias'])} estrat.)"
                              if pos else "no"),
        })
        if pos:
            en_cartera.append(tk)

    if en_cartera:
        frase = (f"{len(earnings)} balances en los proximos dias. "
                 f"{len(en_cartera)} con posicion FT abierta: "
                 f"{', '.join(en_cartera)}.")
    else:
        frase = (f"{len(earnings)} balances en los proximos dias, "
                 f"ninguno con posicion FT abierta.")

    return {"filas": filas, "n_en_cartera": len(en_cartera), "frase": frase}
