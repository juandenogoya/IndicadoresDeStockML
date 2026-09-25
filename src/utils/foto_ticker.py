"""
foto_ticker.py
"Foto al cierre" de un ticker para el mensaje diario de Telegram (25/9/2026).

QUE ES Y QUE NO ES
    Es el ESTADO del ticker en la rueda de datos y como llego ahi en las 5
    ruedas previas: cierre, RSI, MACD, volumen contra su mediana de 52 semanas,
    posicion contra SMA50/SMA200 y dias hasta el proximo balance. Sirve para
    decidir si vale la pena abrir el chart y revisar a mano.
    NO es un pronostico ni una senal: no lleva direccion esperada, ni score,
    ni probabilidad. Lo medido en el proyecto (FT contra control, ML v3 con
    AUC 0,51, alertas inusuales sin direccion a 5 ruedas) no respalda leerlo
    de otra forma.

QUE TICKERS
    Los que las estrategias FT activas marcaron como candidatos en su ultima
    corrida (`ft_candidatos_diarios`), filtrados de dos maneras:
      1. CONFLUENCIA: el ticker aparece en 2+ FAMILIAS de estrategias. Se
         cuenta por familia y no por instancia: las 5 instancias de TECH_SECTOR
         comparten la regla de entrada, y un ticker que aparece en las 5 es
         una sola coincidencia, no cinco.
      2. TOP POR FAMILIA: los N de mayor score dentro de cada familia. El score
         solo es comparable DENTRO de una familia (cada logica tiene su escala),
         nunca entre familias.

REGLAS DE CLASIFICACION
    RSI y MACD salen de src/utils/clasificacion_tecnica.py (los mismos umbrales
    que el dashboard y el MCP: RSI <35 Sobreventa / >65 Sobrecompra; MACD
    Compra/Venta segun la linea contra la senal). Asi el mensaje no dice algo
    distinto que el dashboard sobre el mismo ticker.
    Volumen: el de la rueda dividido la MEDIANA de las 252 ruedas ANTERIORES
    (sin incluir la propia). Mediana y no promedio: los dias de balance inflan
    el promedio. Con menos de 126 ruedas de historia no se informa.

Modulo PURO (stdlib): sin DB, sin red, sin pandas. Lo usa
scripts/manual/telegram_foto.py. ASCII puro en los textos (regla cp1252).
"""

from datetime import date, timedelta
from statistics import median
from typing import Optional

from src.utils.clasificacion_tecnica import clasificar_rsi, clasificar_macd
from src.utils.trading_calendar import trading_days_between

# ── Parametros ────────────────────────────────────────────────────────────────

RUEDAS_PREVIAS = 5          # ruedas que se muestran antes de la de datos
VENTANA_VOL = 252           # ~52 semanas
MIN_RUEDAS_VOL = 126        # por debajo, la mediana no se informa
TOP_POR_FAMILIA = 3
MAX_CONFLUENCIA = 6
MAX_TICKERS = 15            # tope duro de fotos por mensaje diario
MAX_LEN_MENSAJE = 4000      # Telegram corta en 4096

# Orden de presentacion y etiqueta de cada familia
FAMILIAS = ["ML", "TECH", "TECH_SECTOR", "SMC"]
ETIQUETA_FAMILIA = {
    "ML":          "ML Scanner",
    "TECH":        "Tecnico",
    "TECH_SECTOR": "Tec. sectorial",
    "SMC":         "SMC",
    "COMBO":       "Combo",
}
CORTA_FAMILIA = {"ML": "ML", "TECH": "TEC", "TECH_SECTOR": "TS", "SMC": "SMC",
                 "COMBO": "CMB"}

_ABREV_RSI = {"Sobreventa": "SV", "Neutral": "N", "Sobrecompra": "SC"}
_ABREV_MACD = {"Compra": "C", "Venta": "V", "Neutral": "-"}
_ESTADO_RSI = {"Sobreventa": "Sobreventa", "Neutral": "Neutro",
               "Sobrecompra": "Sobrecompra"}
_DIAS = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]


# ── Familias y seleccion ──────────────────────────────────────────────────────

def familia_de(logica: Optional[str]) -> Optional[str]:
    """
    Familia de una estrategia a partir de `ft_estrategias.logica`.
    tecnico_sectorial* va antes que tecnico (el prefijo lo contiene).
    """
    if not logica:
        return None
    lg = logica.lower()
    if lg.startswith("ml_scanner"):
        return "ML"
    if lg.startswith("tecnico_sectorial"):
        return "TECH_SECTOR"
    if lg.startswith("tecnico"):
        return "TECH"
    if lg.startswith("smc"):
        return "SMC"
    if lg.startswith("combo"):
        return "COMBO"
    return lg.upper()


def orden_familia(f: str) -> int:
    return FAMILIAS.index(f) if f in FAMILIAS else len(FAMILIAS)


def agrupar(candidatos: list) -> dict:
    """
    candidatos: dicts {ticker, estrategia, familia, score}.
    Devuelve {ticker: {"familias": {familia: max_score}, "estrategias": set}}.
    """
    por_ticker = {}
    for c in candidatos:
        fam = c.get("familia")
        if not fam:
            continue
        t = por_ticker.setdefault(c["ticker"], {"familias": {}, "estrategias": set()})
        score = c.get("score")
        score = float(score) if score is not None else float("-inf")
        t["familias"][fam] = max(t["familias"].get(fam, float("-inf")), score)
        t["estrategias"].add(c["estrategia"])
    return por_ticker


def seleccionar(candidatos: list,
                top_n: int = TOP_POR_FAMILIA,
                max_confluencia: int = MAX_CONFLUENCIA,
                max_tickers: int = MAX_TICKERS) -> dict:
    """
    Elige que tickers llevan foto.

    Devuelve:
        confluencia: [{ticker, familias: [..], n_estrategias}]  (2+ familias)
        top:         {familia: [{ticker, score, n_estrategias}]}
        orden:       tickers unicos a fotografiar (confluencia primero, despues
                     el top de cada familia en el orden de FAMILIAS), con tope
                     max_tickers
    """
    grupos = agrupar(candidatos)

    confl = []
    for tk, g in grupos.items():
        if len(g["familias"]) >= 2:
            confl.append({
                "ticker": tk,
                "familias": sorted(g["familias"], key=orden_familia),
                "n_estrategias": len(g["estrategias"]),
            })
    confl.sort(key=lambda x: (-len(x["familias"]), -x["n_estrategias"], x["ticker"]))
    confl = confl[:max_confluencia]

    top = {}
    familias = sorted({f for g in grupos.values() for f in g["familias"]},
                      key=orden_familia)
    for fam in familias:
        filas = []
        for tk, g in grupos.items():
            if fam not in g["familias"]:
                continue
            n_fam = sum(1 for c in candidatos
                        if c["ticker"] == tk and c.get("familia") == fam)
            filas.append({"ticker": tk, "score": g["familias"][fam],
                          "n_estrategias": n_fam})
        filas.sort(key=lambda x: (-x["score"], -x["n_estrategias"], x["ticker"]))
        top[fam] = filas[:top_n]

    orden = []
    for c in confl:
        orden.append(c["ticker"])
    for fam in familias:
        for f in top[fam]:
            if f["ticker"] not in orden:
                orden.append(f["ticker"])
    return {"confluencia": confl, "top": top, "orden": orden[:max_tickers]}


# ── Foto de un ticker ─────────────────────────────────────────────────────────

def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x    # NaN -> None


def _pct(a, b) -> Optional[float]:
    a, b = _f(a), _f(b)
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1.0) * 100.0


def _lado(close, sma) -> Optional[str]:
    c, s = _f(close), _f(sma)
    if c is None or s is None:
        return None
    return "sobre" if c > s else "bajo"


def vol_relativo(volumenes: list, i: int,
                 ventana: int = VENTANA_VOL,
                 minimo: int = MIN_RUEDAS_VOL) -> Optional[float]:
    """
    Volumen de la posicion i dividido la mediana de las `ventana` ruedas
    ANTERIORES (sin la propia). None si hay menos de `minimo` ruedas previas
    con dato o si la mediana es 0.
    """
    v = _f(volumenes[i])
    previos = [x for x in (_f(y) for y in volumenes[max(0, i - ventana):i])
               if x is not None]
    if v is None or len(previos) < minimo:
        return None
    med = median(previos)
    if med <= 0:
        return None
    return v / med


def cambios(prev: Optional[dict], act: dict) -> list:
    """
    Cambios de estado de una rueda contra la anterior, en texto corto (va
    dentro de un <pre> que se lee en el celular):
        MACD->C / MACD->V      cruce de la linea contra la senal
        RSI->SC / RSI->SV / RSI->N   cambio de zona
        SMA50->sobre / SMA50->bajo   el cierre cruzo la media (idem SMA200)
    """
    if prev is None:
        return []
    out = []
    if prev.get("macd") and act.get("macd") and prev["macd"] != act["macd"]:
        out.append("MACD->" + _ABREV_MACD.get(act["macd"], act["macd"]))
    if prev.get("rsi_estado") and act.get("rsi_estado") \
            and prev["rsi_estado"] != act["rsi_estado"]:
        out.append("RSI->" + _ABREV_RSI.get(act["rsi_estado"], act["rsi_estado"]))
    for k, nombre in (("lado50", "SMA50"), ("lado200", "SMA200")):
        if prev.get(k) and act.get(k) and prev[k] != act[k]:
            out.append(nombre + "->" + act[k])
    return out


def construir_foto(filas: list,
                   n_previas: int = RUEDAS_PREVIAS,
                   ventana_vol: int = VENTANA_VOL,
                   min_vol: int = MIN_RUEDAS_VOL) -> Optional[dict]:
    """
    filas: historia diaria ORDENADA ascendente del ticker, dicts con
        fecha, close, volume, rsi14, macd, macd_signal, sma50, sma200.
    La ultima fila es la rueda de datos. Devuelve None si no hay al menos
    n_previas + 2 filas (hace falta el cierre previo a la primera mostrada).
    """
    if len(filas) < n_previas + 2:
        return None
    vols = [r.get("volume") for r in filas]
    ini = len(filas) - (n_previas + 1)

    ruedas = []
    anterior = None
    for i in range(ini - 1, len(filas)):
        r = filas[i]
        est = {
            "fecha": r["fecha"],
            "close": _f(r.get("close")),
            "var_pct": _pct(r.get("close"), filas[i - 1].get("close")) if i > 0 else None,
            "rsi": _f(r.get("rsi14")),
            "rsi_estado": clasificar_rsi(_f(r.get("rsi14"))),
            "macd": clasificar_macd(_f(r.get("macd")), _f(r.get("macd_signal"))),
            "vol_x": vol_relativo(vols, i, ventana_vol, min_vol),
            "lado50": _lado(r.get("close"), r.get("sma50")),
            "lado200": _lado(r.get("close"), r.get("sma200")),
        }
        est["cambios"] = cambios(anterior, est)
        anterior = est
        if i >= ini:
            ruedas.append(est)

    hoy = filas[-1]
    ult = ruedas[-1]
    return {
        "fecha": hoy["fecha"],
        "close": ult["close"],
        "var_pct": ult["var_pct"],
        "rsi": ult["rsi"],
        "rsi_estado": ult["rsi_estado"],
        "macd": ult["macd"],
        "vol_x": ult["vol_x"],
        "dist50": _pct(hoy.get("close"), hoy.get("sma50")),
        "dist200": _pct(hoy.get("close"), hoy.get("sma200")),
        "lado50": ult["lado50"],
        "lado200": ult["lado200"],
        "var_n_pct": _pct(hoy.get("close"), filas[-1 - n_previas].get("close")),
        "n_previas": n_previas,
        "ruedas": ruedas,
    }


def ruedas_hasta(desde: date, evento: Optional[date]) -> Optional[int]:
    """
    Ruedas habiles NYSE despues de `desde` hasta `evento` inclusive.
    0 = el evento cae en la misma rueda de datos. None si no hay fecha o si
    ya paso (earnings_calendar guarda solo la PROXIMA fecha y se refresca
    semanal: una fecha pasada es dato viejo, no un balance proximo).
    """
    if evento is None or desde is None:
        return None
    if evento < desde:
        return None
    return len(trading_days_between(desde + timedelta(days=1), evento))


# ── Texto ─────────────────────────────────────────────────────────────────────

def _num(x: Optional[float], dec: int = 2) -> str:
    """Numero con coma decimal y punto de miles (formato AR)."""
    if x is None:
        return "s/d"
    s = f"{x:,.{dec}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def _sg(x: Optional[float], dec: int = 1) -> str:
    if x is None:
        return "s/d"
    return ("+" if x >= 0 else "") + _num(x, dec) + "%"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _dm(d: date) -> str:
    return f"{d.day:02d}/{d.month:02d}"


def texto_earnings(n: Optional[int], fecha: Optional[date]) -> str:
    if n is None:
        return "Earnings: sin fecha proxima"
    if n == 0:
        return f"Earnings: en la rueda de datos ({_dm(fecha)})"
    return f"Earnings: en {n} rueda{'s' if n != 1 else ''} ({_dm(fecha)})"


def bloque_ticker(ticker: str, foto: dict, *,
                  familias: Optional[list] = None,
                  sector: Optional[str] = None,
                  earnings_fecha: Optional[date] = None,
                  rueda_ref: Optional[date] = None) -> str:
    """Bloque HTML (parse_mode=HTML de Telegram) de un ticker."""
    cab = f"<b>{_esc(ticker)}</b>  {_num(foto['close'])} ({_sg(foto['var_pct'])})"
    extras = []
    if familias:
        extras.append(", ".join(CORTA_FAMILIA.get(f, f) for f in familias))
    if sector:
        extras.append(_esc(sector))
    if extras:
        cab += "  [" + " | ".join(extras) + "]"
    lineas = [cab]
    if rueda_ref is not None and foto["fecha"] != rueda_ref:
        lineas.append(f"OJO: ultimo dato del {_dm(foto['fecha'])}, no de la rueda {_dm(rueda_ref)}")

    rsi = "s/d" if foto["rsi"] is None else f"{foto['rsi']:.0f} {_ESTADO_RSI.get(foto['rsi_estado'], '')}"
    lineas.append(f"RSI {rsi} | MACD {foto['macd'] or 's/d'}")
    vol = "s/d (historia corta)" if foto["vol_x"] is None else _num(foto["vol_x"], 1) + "x"
    lineas.append(f"Vol {vol} mediana 52s")
    s50 = "s/d" if foto["dist50"] is None else f"{_sg(foto['dist50'])} {foto['lado50']}"
    s200 = "s/d" if foto["dist200"] is None else f"{_sg(foto['dist200'])} {foto['lado200']}"
    lineas.append(f"SMA50 {s50} | SMA200 {s200}")
    n_e = ruedas_hasta(foto["fecha"], earnings_fecha)
    lineas.append(f"{foto['n_previas']} ruedas: {_sg(foto['var_n_pct'])} | "
                  + texto_earnings(n_e, earnings_fecha))

    tabla = ["fecha   var%    RSI    MACD  vol"]
    for r in foto["ruedas"]:
        rsi_c = "s/d   " if r["rsi"] is None else \
            f"{r['rsi']:>3.0f} {_ABREV_RSI.get(r['rsi_estado'], '?'):<2}"
        vol_c = "s/d " if r["vol_x"] is None else _num(r["vol_x"], 1) + "x"
        fila = (f"{_dm(r['fecha'])}  {_sg(r['var_pct']):>6}  {rsi_c}  "
                f"{_ABREV_MACD.get(r['macd'], '?'):<4}  {vol_c:>4}")
        if r["cambios"]:
            fila += "  " + ", ".join(r["cambios"])
        tabla.append(fila)
    lineas.append("<pre>" + _esc("\n".join(tabla)) + "</pre>")
    return "\n".join(lineas)


def encabezado(rueda: date, fecha_candidatos: Optional[date], seleccion: dict,
               n_candidatos: int, n_estrategias: int) -> str:
    """Encabezado HTML con la seleccion (confluencia + top por familia)."""
    dia = _DIAS[rueda.weekday()]
    lineas = [f"<b>FOTO AL CIERRE</b> - rueda {rueda.isoformat()} ({dia})"]
    if fecha_candidatos is not None:
        lineas.append(f"Corrida FT del {fecha_candidatos.isoformat()}: "
                      f"{n_candidatos} tickers candidatos en {n_estrategias} "
                      "estrategias. Estado, no pronostico.")
    lineas.append("Tabla: RSI SV/N/SC = sobreventa/neutro/sobrecompra, "
                  "MACD C/V = compra/venta, vol = x mediana 52s.")
    if seleccion["confluencia"]:
        lineas.append("")
        lineas.append("<b>En 2+ familias</b>")
        for c in seleccion["confluencia"]:
            fams = ", ".join(ETIQUETA_FAMILIA.get(f, f) for f in c["familias"])
            lineas.append(f"{_esc(c['ticker'])}: {fams} ({c['n_estrategias']} estrategias)")
    if seleccion["top"]:
        lineas.append("")
        lineas.append("<b>Top por familia</b>")
        for fam, filas in seleccion["top"].items():
            tks = ", ".join(_esc(f["ticker"]) for f in filas)
            lineas.append(f"{ETIQUETA_FAMILIA.get(fam, fam)}: {tks}")
    return "\n".join(lineas)


def empaquetar(bloques: list, max_len: int = MAX_LEN_MENSAJE) -> list:
    """
    Junta bloques en mensajes de hasta max_len sin partir ningun bloque
    (partir un <pre> deja HTML invalido y Telegram rechaza el mensaje).
    Un bloque que solo ya excede max_len va en un mensaje propio.
    """
    mensajes, actual = [], ""
    for b in bloques:
        cand = b if not actual else actual + "\n\n" + b
        if len(cand) <= max_len or not actual:
            actual = cand
        else:
            mensajes.append(actual)
            actual = b
    if actual:
        mensajes.append(actual)
    return mensajes
