"""
cierre_grupos.py
"Cierre por grupo" de una rueda para Telegram (29/9/2026, EN EVALUACION).

QUE ES Y QUE NO ES
    El cierre de la rueda leido POR SECTOR: que grupo quedo mas fuerte y mas
    debil, que tickers cruzaron un nivel (muro de OI, swing confirmado, SMA) y
    que conviene vigilar en la proxima rueda (cerca de un nivel, balance
    proximo, cambio de veredicto). Es ESTADO, no pronostico: no lleva
    direccion esperada. Definiciones, criterios pre-registrados y resultados:
    docs/cierre_grupos.md.

ENTRADA
    Una lista de filas, una por ticker con dato en la rueda D, ya alineadas por
    el script contra la rueda ANTERIOR del universo (no un shift por ticker):

        ticker, sector, close, close_ant, close_5,
        sma50, sma50_ant, sma200, sma200_ant,
        sop_ant, res_ant          muros de OI (ventana corto) de la rueda ANTERIOR
        sop_dist, res_dist        distancia % a los muros de D (para "vigilar")
        swing_arriba, swing_abajo bos/choch N5 en D (0/1/None)
        dist_sh, dist_sl          distancia % al ultimo swing N5 confirmado en D
        veredicto, veredicto_ant
        earnings_ruedas, earnings_fecha   proximo balance en ruedas NYSE (o None)

Modulo PURO (stdlib): sin DB, sin red, sin pandas. ASCII en los textos
(regla cp1252). Lo usan scripts/manual/telegram_cierre.py y el auditor.
"""

from datetime import date
from statistics import median
from typing import Optional

from src.utils.foto_ticker import _num, _sg, _esc, _dm, empaquetar, MAX_LEN_MENSAJE

# ── Parametros (pre-registrados en docs/cierre_grupos.md) ─────────────────────

MIN_N_RANKING = 5           # sectores con menos tickers no compiten en el ranking
UMBRAL_CERCA_PCT = 1.0      # "cerca de un nivel" = a menos de este % en D
RUEDAS_EARNINGS = 5         # balances en las proximas N ruedas NYSE
MAX_LISTA = 8               # tickers por lista en el mensaje (se informa el total)
RUEDAS_VAR_N = 5            # la variacion de "5r"

TIPOS_CRUCE = ("OI", "SWING", "SMA")

# Etiquetas cortas de sector (la tabla va en un <pre> que se lee en el celular)
CORTO_SECTOR = {
    "Technology":             "Tecnologia",
    "Consumer Cyclical":      "Cons. ciclico",
    "Financial Services":     "Financiero",
    "Industrials":            "Industrial",
    "Basic Materials":        "Mat. basicos",
    "Healthcare":             "Salud",
    "Energy":                 "Energia",
    "Communication Services": "Comunicacion",
    "Consumer Defensive":     "Cons. defens.",
    "Real Estate":            "Real Estate",
    "Utilities":              "Utilities",
}
_DIAS = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]
_VERED_CORTO = {"ALCISTA": "A", "NEUTRAL": "N", "BAJISTA": "B"}


# ── Utilidades ────────────────────────────────────────────────────────────────

def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x    # NaN -> None


def _var(a, b) -> Optional[float]:
    """a / b - 1 en %. None si falta alguno o b <= 0."""
    a, b = _f(a), _f(b)
    if a is None or b is None or b <= 0:
        return None
    return (a / b - 1.0) * 100.0


def _lado(close, sma) -> Optional[str]:
    """Misma regla que foto_ticker._lado: close > sma = sobre, si no bajo."""
    c, s = _f(close), _f(sma)
    if c is None or s is None:
        return None
    return "sobre" if c > s else "bajo"


def _flag(v) -> bool:
    return v is not None and _f(v) is not None and _f(v) != 0


def _mediana(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def sector_corto(s: Optional[str]) -> str:
    if not s:
        return "Sin sector"
    return CORTO_SECTOR.get(s, s[:13])


# ── 1. Grupos ─────────────────────────────────────────────────────────────────

def _agregado(filas: list) -> dict:
    vd = [_var(r.get("close"), r.get("close_ant")) for r in filas]
    v5 = [_var(r.get("close"), r.get("close_5")) for r in filas]
    lados = [_lado(r.get("close"), r.get("sma50")) for r in filas]
    con_lado = [x for x in lados if x is not None]
    vered = {"ALCISTA": 0, "NEUTRAL": 0, "BAJISTA": 0}
    n_vered = 0
    for r in filas:
        v = r.get("veredicto")
        if v in vered:
            vered[v] += 1
            n_vered += 1
    return {
        "n": len(filas),
        "var_dia": _mediana(vd),
        "var_5r": _mediana(v5),
        "suben": sum(1 for x in vd if x is not None and x > 0),
        "bajan": sum(1 for x in vd if x is not None and x < 0),
        "sobre_sma50_pct": (100.0 * sum(1 for x in con_lado if x == "sobre") / len(con_lado)
                            if con_lado else None),
        "veredictos": vered if n_vered else None,
    }


def resumen_grupos(filas: list, min_n: int = MIN_N_RANKING) -> dict:
    """
    Agregados por sector y del universo. `grupos` va ordenado: primero los
    rankeables por var_dia descendente, despues los de n chico.
    """
    por_sector = {}
    for r in filas:
        por_sector.setdefault(r.get("sector") or "Sin sector", []).append(r)

    grupos = []
    for sector, fs in por_sector.items():
        g = _agregado(fs)
        g["sector"] = sector
        g["rankeable"] = g["n"] >= min_n and g["var_dia"] is not None
        grupos.append(g)

    rank = sorted([g for g in grupos if g["rankeable"]],
                  key=lambda g: (-g["var_dia"], g["sector"]))
    chicos = sorted([g for g in grupos if not g["rankeable"]], key=lambda g: g["sector"])
    return {
        "universo": _agregado(filas),
        "grupos": rank + chicos,
        "fuerte": rank[0]["sector"] if rank else None,
        "debil": rank[-1]["sector"] if len(rank) > 1 else None,
    }


# ── 2. Cruces de nivel ────────────────────────────────────────────────────────

def cruces(filas: list) -> dict:
    """
    {tipo: {"arriba": [(ticker, detalle)], "abajo": [(ticker, detalle)]}}
      OI    : close_D contra el muro de la rueda ANTERIOR (detalle = strike)
      SWING : bos/choch N5 en D (detalle = None)
      SMA   : cambio de lado de SMA50 / SMA200 (detalle = "SMA50" | "SMA200")
    Listas ordenadas por ticker.
    """
    out = {t: {"arriba": [], "abajo": []} for t in TIPOS_CRUCE}
    for r in sorted(filas, key=lambda x: x["ticker"]):
        tk = r["ticker"]
        c = _f(r.get("close"))

        res, sop = _f(r.get("res_ant")), _f(r.get("sop_ant"))
        if c is not None and res is not None and c > res:
            out["OI"]["arriba"].append((tk, res))
        if c is not None and sop is not None and c < sop:
            out["OI"]["abajo"].append((tk, sop))

        if _flag(r.get("swing_arriba")):
            out["SWING"]["arriba"].append((tk, None))
        if _flag(r.get("swing_abajo")):
            out["SWING"]["abajo"].append((tk, None))

        for k, nombre in (("sma50", "SMA50"), ("sma200", "SMA200")):
            antes = _lado(r.get("close_ant"), r.get(k + "_ant"))
            ahora = _lado(r.get("close"), r.get(k))
            if antes and ahora and antes != ahora:
                out["SMA"]["arriba" if ahora == "sobre" else "abajo"].append((tk, nombre))
    return out


def tickers_con_cruce(cr: dict, tipo: str) -> set:
    """Tickers distintos con algun cruce del tipo (arriba o abajo)."""
    return {tk for lado in ("arriba", "abajo") for tk, _ in cr[tipo][lado]}


# ── 3. Vigilar ────────────────────────────────────────────────────────────────

def vigilar(filas: list, umbral: float = UMBRAL_CERCA_PCT,
            ruedas_earn: int = RUEDAS_EARNINGS) -> dict:
    """
    cerca_oi / cerca_swing: [(ticker, "sop"|"res"|"SL"|"SH", dist_pct)] con
        |dist| < umbral en D, ordenado por |dist|. Un ticker entra una vez por
        tipo (el nivel mas cercano).
    earnings: [(ticker, ruedas, fecha)] con 1 <= ruedas <= ruedas_earn.
    veredictos: [(ticker, antes, ahora)] cuando cambio.
    veredictos_comparables: tickers con veredicto en D Y en la rueda anterior.
        En 0 la lista vacia NO significa "ningun cambio": falta la rueda anterior
        en veredictos_universo_diario (pasa cuando FT no corrio esa rueda, como
        el 13/10/2026) y no hay contra que comparar.
    """
    cerca_oi, cerca_sw, earn, vered = [], [], [], []
    comparables = 0
    for r in filas:
        tk = r["ticker"]
        for destino, pares in ((cerca_oi, (("sop", r.get("sop_dist")), ("res", r.get("res_dist")))),
                               (cerca_sw, (("SL", r.get("dist_sl")), ("SH", r.get("dist_sh"))))):
            cands = [(abs(_f(d)), nom, _f(d)) for nom, d in pares
                     if _f(d) is not None and abs(_f(d)) < umbral]
            if cands:
                _, nom, d = min(cands)
                destino.append((tk, nom, d))
        n = r.get("earnings_ruedas")
        if n is not None and 1 <= n <= ruedas_earn:
            earn.append((tk, n, r.get("earnings_fecha")))
        a, b = r.get("veredicto_ant"), r.get("veredicto")
        if a and b:
            comparables += 1
            if a != b:
                vered.append((tk, a, b))
    cerca_oi.sort(key=lambda x: (abs(x[2]), x[0]))
    cerca_sw.sort(key=lambda x: (abs(x[2]), x[0]))
    earn.sort(key=lambda x: (x[1], x[0]))
    vered.sort(key=lambda x: x[0])
    return {"cerca_oi": cerca_oi, "cerca_swing": cerca_sw,
            "earnings": earn, "veredictos": vered,
            "veredictos_comparables": comparables}


# ── Resultado completo ────────────────────────────────────────────────────────

def construir(rueda: date, rueda_ant: date, filas: list, *,
              n_universo: Optional[int] = None,
              oi_disponible: bool = True,
              ruedas_salteadas: Optional[list] = None,
              min_n: int = MIN_N_RANKING) -> dict:
    """
    ruedas_salteadas: dias habiles NYSE entre rueda_ant y rueda que no son
    rueda del universo (hueco en precios_diarios). Con un hueco, "dia" compara
    contra una rueda mas vieja y el mensaje tiene que decirlo.
    """
    return {
        "rueda": rueda,
        "rueda_ant": rueda_ant,
        "ruedas_salteadas": list(ruedas_salteadas or []),
        "n_con_dato": len(filas),
        "n_universo": n_universo,
        "oi_disponible": oi_disponible,
        "grupos": resumen_grupos(filas, min_n),
        "cruces": cruces(filas),
        "vigilar": vigilar(filas),
    }


# ── Texto (HTML de Telegram) ──────────────────────────────────────────────────

def _lista(items: list, fmt, maximo: int = MAX_LISTA) -> str:
    if not items:
        return "ninguno"
    txt = ", ".join(fmt(x) for x in items[:maximo])
    if len(items) > maximo:
        txt += f" y {len(items) - maximo} mas"
    return txt


def _vered_txt(v: Optional[dict]) -> str:
    if not v:
        return "s/d"
    return f"{v['ALCISTA']}/{v['NEUTRAL']}/{v['BAJISTA']}"


def bloque_grupos(res: dict) -> str:
    g = res["grupos"]
    u = g["universo"]
    rueda = res["rueda"]
    lineas = [f"<b>CIERRE POR SECTOR</b> - rueda {rueda.isoformat()} ({_DIAS[rueda.weekday()]})"]
    cob = f"{res['n_con_dato']} tickers"
    if res.get("n_universo") and res["n_universo"] != res["n_con_dato"]:
        cob = f"{res['n_con_dato']} de {res['n_universo']} tickers con dato"
    lineas.append(f"Universo ({cob}): mediana {_sg(u['var_dia'])} dia, "
                  f"{_sg(u['var_5r'])} en {RUEDAS_VAR_N} ruedas. "
                  f"Suben {u['suben']} / bajan {u['bajan']}.")
    if res.get("ruedas_salteadas"):
        faltan = ", ".join(_dm(d) for d in res["ruedas_salteadas"])
        lineas.append(f"OJO: hueco en precios_diarios ({faltan}); 'dia' y los cruces "
                      f"comparan contra la rueda del {_dm(res['rueda_ant'])}.")
    if g["fuerte"]:
        gf = next(x for x in g["grupos"] if x["sector"] == g["fuerte"])
        linea = f"Mas fuerte: {_esc(sector_corto(g['fuerte']))} ({_sg(gf['var_dia'])})"
        if g["debil"]:
            gd = next(x for x in g["grupos"] if x["sector"] == g["debil"])
            linea += f" | Mas debil: {_esc(sector_corto(g['debil']))} ({_sg(gd['var_dia'])})"
        lineas.append(linea)

    tabla = ["sector          n    dia     5r  s/b  >S50 A/N/B"]
    for x in g["grupos"]:
        nombre = sector_corto(x["sector"]) + ("" if x["rankeable"] else "*")
        s50 = "s/d" if x["sobre_sma50_pct"] is None else f"{x['sobre_sma50_pct']:.0f}%"
        tabla.append(f"{nombre:<14} {x['n']:>3} {_sg(x['var_dia']):>6} {_sg(x['var_5r']):>6} "
                     f"{x['suben']:>2}/{x['bajan']:<2} {s50:>4} {_vered_txt(x['veredictos'])}")
    lineas.append("<pre>" + _esc("\n".join(tabla)) + "</pre>")
    notas = ["dia/5r = mediana del sector; s/b = suben/bajan; >S50 = % sobre SMA50; "
             "A/N/B = veredictos alcista/neutral/bajista."]
    if any(not x["rankeable"] for x in g["grupos"]):
        notas.append(f"* menos de {MIN_N_RANKING} tickers: fuera del ranking.")
    lineas.append(" ".join(notas))
    return "\n".join(lineas)


def bloque_cruces(res: dict) -> str:
    cr = res["cruces"]
    ant = _dm(res["rueda_ant"])
    lineas = [f"<b>Cruces de nivel</b> (nivel de la rueda anterior, {ant})"]

    tk = lambda x: _esc(x[0])
    if res.get("oi_disponible", True):
        lineas.append(f"Muro OI corto - perdieron soporte ({len(cr['OI']['abajo'])}): "
                      f"{_lista(cr['OI']['abajo'], tk)}")
        lineas.append(f"Muro OI corto - superaron resistencia ({len(cr['OI']['arriba'])}): "
                      f"{_lista(cr['OI']['arriba'], tk)}")
    else:
        lineas.append(f"Muro OI: sin muros calculados para el {ant} (hueco en opciones).")
    lineas.append(f"Swing N5 - perdieron el ultimo minimo ({len(cr['SWING']['abajo'])}): "
                  f"{_lista(cr['SWING']['abajo'], tk)}")
    lineas.append(f"Swing N5 - superaron el ultimo maximo ({len(cr['SWING']['arriba'])}): "
                  f"{_lista(cr['SWING']['arriba'], tk)}")
    sma = lambda x: f"{_esc(x[0])} {x[1][3:]}"
    lineas.append(f"SMA - cruzaron hacia abajo ({len(cr['SMA']['abajo'])}): "
                  f"{_lista(cr['SMA']['abajo'], sma)}")
    lineas.append(f"SMA - cruzaron hacia arriba ({len(cr['SMA']['arriba'])}): "
                  f"{_lista(cr['SMA']['arriba'], sma)}")
    return "\n".join(lineas)


def bloque_vigilar(res: dict) -> str:
    v = res["vigilar"]
    lineas = ["<b>Vigilar en la proxima rueda</b>"]
    # 2 decimales: con 1, un 0,99 se lee "1,0%" en una lista "a menos de 1,0%"
    cerca = lambda x: f"{_esc(x[0])} {x[1]} {_sg(x[2], 2)}"
    lineas.append(f"A menos de {_num(UMBRAL_CERCA_PCT, 1)}% de un muro OI "
                  f"({len(v['cerca_oi'])}): {_lista(v['cerca_oi'], cerca)}")
    lineas.append(f"A menos de {_num(UMBRAL_CERCA_PCT, 1)}% de un swing N5 "
                  f"({len(v['cerca_swing'])}): {_lista(v['cerca_swing'], cerca)}")
    earn = lambda x: f"{_esc(x[0])} {x[1]}r" + (f" ({_dm(x[2])})" if x[2] else "")
    lineas.append(f"Balance en {RUEDAS_EARNINGS} ruedas ({len(v['earnings'])}): "
                  f"{_lista(v['earnings'], earn)}")
    ver = lambda x: (f"{_esc(x[0])} {_VERED_CORTO.get(x[1], x[1])}->"
                     f"{_VERED_CORTO.get(x[2], x[2])}")
    if v.get("veredictos_comparables", 1):
        lineas.append(f"Cambio de veredicto ({len(v['veredictos'])}): "
                      f"{_lista(v['veredictos'], ver)}")
    else:
        lineas.append(f"Cambio de veredicto: sin veredictos de la rueda anterior "
                      f"({_dm(res['rueda_ant'])}), no hay contra que comparar.")
    lineas.append("sop/res = muro OI, SL/SH = swing low/high; % = distancia del cierre. "
                  "Estado, no pronostico.")
    return "\n".join(lineas)


def mensajes(res: dict, max_len: int = MAX_LEN_MENSAJE) -> list:
    return empaquetar([bloque_grupos(res), bloque_cruces(res), bloque_vigilar(res)], max_len)


# ── Evaluacion (criterios 2-4 de docs/cierre_grupos.md) ───────────────────────

def evaluar(resultados: list, max_len: int = MAX_LEN_MENSAJE) -> dict:
    """
    resultados: salidas de construir() de varias ruedas.
      cruces_por_dia[tipo]  = tickers distintos con cruce por rueda
      mediana_cruces[tipo]  = mediana de lo anterior
      fuerte / debil        = sector de cada rueda + el mas repetido y su share
      mensajes              = (cantidad, largo maximo) por rueda
    """
    out = {"n_ruedas": len(resultados), "cruces_por_dia": {}, "mediana_cruces": {}}
    for t in TIPOS_CRUCE:
        serie = [len(tickers_con_cruce(r["cruces"], t)) for r in resultados
                 if t != "OI" or r.get("oi_disponible", True)]
        out["cruces_por_dia"][t] = serie
        out["mediana_cruces"][t] = median(serie) if serie else None

    for clave in ("fuerte", "debil"):
        serie = [r["grupos"][clave] for r in resultados]
        conteo = {}
        for s in serie:
            if s is not None:
                conteo[s] = conteo.get(s, 0) + 1
        top = max(conteo.items(), key=lambda kv: (kv[1], kv[0])) if conteo else (None, 0)
        out[clave] = {"serie": serie, "mas_repetido": top[0],
                      "share": top[1] / len(serie) if serie else None}

    msgs = [mensajes(r, max_len) for r in resultados]
    out["mensajes"] = [(len(m), max(len(x) for x in m)) for m in msgs]
    return out


def veredicto_criterios(ev: dict, rango=(2, 30), share_max: float = 0.80,
                        max_mensajes: int = 2, max_len: int = MAX_LEN_MENSAJE) -> dict:
    """Aplica las reglas pre-registradas a la salida de evaluar()."""
    c2 = {t: (m is not None and rango[0] <= m <= rango[1])
          for t, m in ev["mediana_cruces"].items()}
    c3 = all(ev[k]["share"] is not None and ev[k]["share"] <= share_max
             for k in ("fuerte", "debil"))
    c4 = all(n <= max_mensajes and largo <= max_len for n, largo in ev["mensajes"])
    return {"c2_por_tipo": c2, "c2": any(c2.values()), "c3": c3, "c4": c4}
