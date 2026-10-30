"""
vigilancia_intradia.py -- seguimiento intradiario de la lista de vigilancia.

Modulo PURO (stdlib): no importa DB, config ni pipeline. Etapa 2 de la linea
intradiaria (docs/intradia.md sec. 9).

QUE HACE. La rutina de la rueda D ya dice que tickers vigilar: la foto (2+
familias de FT y top por familia, docs/telegram_foto.md) y el cierre por sector
(cruces de nivel y "vigilar": cerca de un nivel, cambio de veredicto,
docs/cierre_grupos.md). Este modulo:
    1. arma con eso la LISTA DE VIGILANCIA de D: por ticker, los niveles que
       se siguen y de que lado quedo el close de D (`armar_lista`);
    2. en cada corte de la rueda siguiente compara el precio del corte contra
       esos niveles y contra la volatilidad del ticker (`eventos`);
    3. a las 15:00 ET informa solo lo nuevo respecto de las 11:00, y lo que se
       deshizo (`novedades`);
    4. arma el mensaje de Telegram (`mensaje`).

No recorre el universo ni inventa senales: sigue lo que la rutina ya marco. Es
ESTADO, no pronostico (mismo criterio que la foto y el cierre): sin direccion
esperada ni "oportunidad".

NIVELES (todos congelados con los datos de D):
    OI_SOP / OI_RES     muro de OI de D, ventana corta (opciones_pcr_plazo_diario)
    SW_SL / SW_SH       ultimo swing N5 confirmado conocido en D. La tabla guarda
                        la distancia y no el precio; sale exacto del close:
                        nivel = close / (1 + dist/100) (dist es DOUBLE)
    SMA50 / SMA200      de D (indicadores_tecnicos)
    OI_SOP_ANT / ...    el nivel que se CRUZO en D. Para OI es el muro de D-1: el
                        de D se recalcula con el close de D y queda debajo del
                        precio por construccion (misma razon que el cierre usa
                        "el muro de la rueda anterior"). Para swing, el de D-1.
                        En las SMA el cruce de D se marca en la propia fila.

EVENTOS en el corte (pre-registrados el 30/10/2026):
    cruce       el precio del corte quedo del otro lado del nivel que el close de D
    revierte    idem, sobre un nivel que se habia cruzado EN D: deshace el cruce
                de ayer (ej. "perdio el soporte de OI" y vuelve arriba)
    movimiento  |z| >= UMBRAL_Z, z = ln(precio_corte / close_D) / sigma, con sigma
                el desvio de los retornos diarios (log) de las VENTANA_SIGMA ruedas
                hasta D; con menos de MIN_RUEDAS_SIGMA no se informa. A las 11:00
                es exigente (media rueda contra el desvio de un dia entero):
                limitacion declarada, no se ajusta mirando resultados.
"""

import math
from datetime import date, datetime
from typing import Optional

from src.utils.foto_ticker import MAX_LEN_MENSAJE, _dm, _esc, _num, _sg

UMBRAL_Z = 2.0
VENTANA_SIGMA = 252
MIN_RUEDAS_SIGMA = 126
MAX_LINEAS = 10                 # lineas por seccion del mensaje (se informa el total)

ORIGEN_PUSH = "push"
ORIGEN_RECONSTRUIDA = "reconstruida"

TIPOS = ("cruce", "revierte", "movimiento")

NIVELES = {
    "OI_SOP": "soporte OI",
    "OI_RES": "resistencia OI",
    "SW_SL": "swing low N5",
    "SW_SH": "swing high N5",
    "SMA50": "SMA50",
    "SMA200": "SMA200",
    "OI_SOP_ANT": "soporte OI perdido ayer",
    "OI_RES_ANT": "resistencia OI superada ayer",
    "SW_SL_ANT": "swing low N5 roto ayer",
    "SW_SH_ANT": "swing high N5 roto ayer",
}
# Nivel cruzado en D -> nivel "de D" del mismo tipo (para no duplicar si coinciden)
_BASE = {"OI_SOP_ANT": "OI_SOP", "OI_RES_ANT": "OI_RES",
         "SW_SL_ANT": "SW_SL", "SW_SH_ANT": "SW_SH"}

_VERED = {"ALCISTA": "A", "NEUTRAL": "N", "BAJISTA": "B"}
_CORTE_TXT = {"apertura_90": "11:00 ET", "cierre_60": "15:00 ET"}

TABLA_LISTAS = "vigilancia_listas"
TABLA_TICKERS = "vigilancia_tickers"
TABLA_NIVELES = "vigilancia_niveles"
TABLA_ENVIOS = "vigilancia_envios"

DDL_TABLAS = f"""
CREATE TABLE IF NOT EXISTS {TABLA_LISTAS} (
    fecha        DATE         NOT NULL,      -- rueda D de la lista (fecha de DATOS)
    n_tickers    INTEGER      NOT NULL,      -- 0 = lista VACIA, que no es lo mismo que no subida
    origen       VARCHAR(12)  NOT NULL,      -- push | reconstruida
    nota         TEXT,                       -- de donde salio la foto (corrida de FT)
    generado_en  TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (fecha)
);
CREATE TABLE IF NOT EXISTS {TABLA_TICKERS} (
    fecha        DATE              NOT NULL,   -- rueda D de la lista (fecha de DATOS)
    ticker       VARCHAR(20)       NOT NULL,
    close_d      DOUBLE PRECISION  NOT NULL,
    sigma_d      DOUBLE PRECISION,             -- desvio de retornos log diarios hasta D
    motivos      TEXT              NOT NULL,   -- por que se vigila, separado por '; '
    origen       VARCHAR(12)       NOT NULL,   -- push | reconstruida
    generado_en  TIMESTAMPTZ       NOT NULL,
    PRIMARY KEY (fecha, ticker)
);
CREATE TABLE IF NOT EXISTS {TABLA_NIVELES} (
    fecha         DATE              NOT NULL,
    ticker        VARCHAR(20)       NOT NULL,
    nivel         VARCHAR(16)       NOT NULL,  -- OI_SOP | SW_SL | SMA50 | OI_SOP_ANT | ...
    valor         DOUBLE PRECISION  NOT NULL,
    lado_d        VARCHAR(5)        NOT NULL,  -- sobre | bajo: el close de D contra el nivel
    cruzado_en_d  BOOLEAN           NOT NULL,
    PRIMARY KEY (fecha, ticker, nivel)
);
CREATE TABLE IF NOT EXISTS {TABLA_ENVIOS} (
    fecha        DATE         NOT NULL,        -- rueda del corte
    corte        VARCHAR(16)  NOT NULL,
    fecha_lista  DATE,                         -- rueda D de la lista usada (NULL: no habia)
    n_vigilados  INTEGER      NOT NULL,
    n_eventos    INTEGER      NOT NULL,
    texto        TEXT         NOT NULL,
    enviado_en   TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (fecha, corte)
);
"""

COLS_LISTAS = ["fecha", "n_tickers", "origen", "nota", "generado_en"]
COLS_TICKERS = ["fecha", "ticker", "close_d", "sigma_d", "motivos", "origen", "generado_en"]
COLS_NIVELES = ["fecha", "ticker", "nivel", "valor", "lado_d", "cruzado_en_d"]
COLS_ENVIOS = ["fecha", "corte", "fecha_lista", "n_vigilados", "n_eventos", "texto", "enviado_en"]


# ── Utilidades ────────────────────────────────────────────────────────────────

def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def lado(precio, valor) -> Optional[str]:
    """'sobre' | 'bajo' del precio contra el nivel; None si falta o es igual."""
    p, v = _f(precio), _f(valor)
    if p is None or v is None or p == v:
        return None
    return "sobre" if p > v else "bajo"


def nivel_desde_dist(close, dist_pct) -> Optional[float]:
    """features_estructura guarda dist = (close - nivel) / nivel * 100."""
    c, d = _f(close), _f(dist_pct)
    if c is None or d is None or d <= -100:
        return None
    return c / (1.0 + d / 100.0)


def sigma_log(closes: list) -> Optional[float]:
    """
    Desvio (muestral) de los retornos log diarios de las ultimas VENTANA_SIGMA
    ruedas de `closes` (ascendente, terminando en D). None con menos de
    MIN_RUEDAS_SIGMA retornos.
    """
    cs = [_f(c) for c in closes]
    cs = [c for c in cs if c is not None and c > 0]
    rets = [math.log(b / a) for a, b in zip(cs, cs[1:])][-VENTANA_SIGMA:]
    if len(rets) < MIN_RUEDAS_SIGMA:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) if var > 0 else None


# ── 1. Lista de vigilancia ────────────────────────────────────────────────────

def motivos_foto(seleccion: dict) -> dict:
    """{ticker: [motivo]} de la seleccion de foto_ticker.seleccionar (solo `orden`)."""
    orden = set(seleccion.get("orden", []))
    out = {}
    for c in seleccion.get("confluencia", []):
        if c["ticker"] in orden:
            out.setdefault(c["ticker"], []).append(
                "foto " + "+".join(c["familias"]))
    for fam, filas in seleccion.get("top", {}).items():
        for f in filas:
            if f["ticker"] in orden:
                out.setdefault(f["ticker"], []).append(f"foto top {fam}")
    return out


def motivos_cierre(res: dict) -> dict:
    """{ticker: [motivo]} de cierre_grupos.construir, SIN balances (decision 30/10/2026)."""
    out = {}

    def add(tk, m):
        out.setdefault(tk, []).append(m)

    cr = res["cruces"]
    for tk, _ in cr["OI"]["abajo"]:
        add(tk, "perdio sop OI")
    for tk, _ in cr["OI"]["arriba"]:
        add(tk, "supero res OI")
    for tk, _ in cr["SWING"]["abajo"]:
        add(tk, "rompio SL N5")
    for tk, _ in cr["SWING"]["arriba"]:
        add(tk, "rompio SH N5")
    for lado_cr, verbo in (("abajo", "bajo"), ("arriba", "sobre")):
        for tk, sma in cr["SMA"][lado_cr]:
            add(tk, f"paso {verbo} {sma}")
    v = res["vigilar"]
    for tk, nom, _ in v["cerca_oi"]:
        add(tk, f"cerca {nom} OI")
    for tk, nom, _ in v["cerca_swing"]:
        add(tk, f"cerca {nom} N5")
    for tk, a, b in v["veredictos"]:
        add(tk, f"veredicto {_VERED.get(a, a)}>{_VERED.get(b, b)}")
    return out


def armar_lista(rueda: date, filas_cierre: list, res_cierre: dict, motivos: dict,
                strikes_d: dict, estr_ant: dict, sigmas: dict,
                generado_en: datetime, origen: str = ORIGEN_PUSH) -> tuple[list, list]:
    """
    Lista de vigilancia de la rueda D.

    filas_cierre: filas de telegram_cierre.armar_filas para D (close, close_ant,
                  sma50, sma200, dist_sl, dist_sh, ...), una por ticker.
    res_cierre:   cierre_grupos.construir para D (de ahi salen los cruces de D).
    motivos:      {ticker: [motivo]} = motivos_foto + motivos_cierre.
    strikes_d:    {ticker: (soporte_strike, resistencia_strike)} de D.
    estr_ant:     {ticker: (dist_sl_5_pct, dist_sh_5_pct)} de D-1.
    sigmas:       {ticker: sigma_log}.

    Devuelve (tickers, niveles): filas para TABLA_TICKERS y TABLA_NIVELES.
    Un ticker sin close en D queda afuera (no hay contra que comparar).
    """
    por_tk = {r["ticker"]: r for r in filas_cierre}
    cr = res_cierre["cruces"]
    oi_ant = {"OI_SOP_ANT": dict(cr["OI"]["abajo"]), "OI_RES_ANT": dict(cr["OI"]["arriba"])}
    sw_cruzo = {"SW_SL_ANT": {tk for tk, _ in cr["SWING"]["abajo"]},
                "SW_SH_ANT": {tk for tk, _ in cr["SWING"]["arriba"]}}
    sma_cruzo = {(tk, sma) for lado_cr in ("arriba", "abajo") for tk, sma in cr["SMA"][lado_cr]}

    tickers, niveles = [], []
    for tk in sorted(motivos):
        r = por_tk.get(tk)
        close = _f(r.get("close")) if r else None
        if close is None:
            continue
        cand = {}
        sop, res = strikes_d.get(tk, (None, None))
        cand["OI_SOP"], cand["OI_RES"] = (_f(sop), False), (_f(res), False)
        cand["SW_SL"] = (nivel_desde_dist(close, r.get("dist_sl")), False)
        cand["SW_SH"] = (nivel_desde_dist(close, r.get("dist_sh")), False)
        for sma, col in (("SMA50", "sma50"), ("SMA200", "sma200")):
            cand[sma] = (_f(r.get(col)), (tk, sma) in sma_cruzo)
        for nom, d in oi_ant.items():
            if tk in d:
                cand[nom] = (_f(d[tk]), True)
        dsl, dsh = estr_ant.get(tk, (None, None))
        if tk in sw_cruzo["SW_SL_ANT"]:
            cand["SW_SL_ANT"] = (nivel_desde_dist(r.get("close_ant"), dsl), True)
        if tk in sw_cruzo["SW_SH_ANT"]:
            cand["SW_SH_ANT"] = (nivel_desde_dist(r.get("close_ant"), dsh), True)
        # El nivel cruzado en D manda si coincide con el de D (mismo strike / swing).
        for ant, base in _BASE.items():
            if ant in cand and cand[ant][0] is not None and cand.get(base, (None,))[0] is not None \
                    and abs(cand[ant][0] - cand[base][0]) < 1e-9:
                del cand[base]

        for nom in NIVELES:
            if nom not in cand:
                continue
            valor, cruzado = cand[nom]
            ld = lado(close, valor)
            if ld is None:
                continue
            niveles.append({"fecha": rueda, "ticker": tk, "nivel": nom, "valor": valor,
                            "lado_d": ld, "cruzado_en_d": bool(cruzado)})
        tickers.append({"fecha": rueda, "ticker": tk, "close_d": close,
                        "sigma_d": sigmas.get(tk), "motivos": "; ".join(motivos[tk]),
                        "origen": origen, "generado_en": generado_en})
    return tickers, niveles


# ── 2. Eventos en el corte ────────────────────────────────────────────────────

def eventos(tickers: list, niveles: list, precios: dict) -> list:
    """
    Eventos de un corte. precios = {ticker: precio del corte} (close de la barra
    que termina en el corte; un ticker sin esa barra no evalua).
    """
    info = {t["ticker"]: t for t in tickers}
    out = []
    for n in niveles:
        tk = n["ticker"]
        p = _f(precios.get(tk))
        if p is None or tk not in info:
            continue
        lc = lado(p, n["valor"])
        if lc is None or lc == n["lado_d"]:
            continue
        out.append({"ticker": tk, "tipo": "revierte" if n["cruzado_en_d"] else "cruce",
                    "nivel": n["nivel"], "valor": _f(n["valor"]), "lado": lc,
                    "precio": p, "dist_pct": (p / _f(n["valor"]) - 1.0) * 100.0})
    for tk, t in info.items():
        p, c, s = _f(precios.get(tk)), _f(t["close_d"]), _f(t.get("sigma_d"))
        if p is None or c is None or s is None or p <= 0 or c <= 0:
            continue
        z = math.log(p / c) / s
        if abs(z) >= UMBRAL_Z:
            out.append({"ticker": tk, "tipo": "movimiento", "nivel": None, "valor": None,
                        "lado": None, "precio": p, "z": z,
                        "var_pct": (p / c - 1.0) * 100.0})
    out.sort(key=lambda e: (TIPOS.index(e["tipo"]), e["ticker"], e["nivel"] or ""))
    return out


def clave(ev: dict) -> tuple:
    return (ev["ticker"], ev["tipo"], ev["nivel"])


def novedades(ev_actual: list, ev_previo: Optional[list]) -> tuple[list, list]:
    """
    (nuevos, deshechos) del corte actual contra el anterior. Sin corte previo
    (None) todo es nuevo y no hay deshechos.
    """
    if ev_previo is None:
        return list(ev_actual), []
    prev = {clave(e): e for e in ev_previo}
    act = {clave(e) for e in ev_actual}
    nuevos = [e for e in ev_actual if clave(e) not in prev]
    deshechos = [e for k, e in prev.items() if k not in act]
    deshechos.sort(key=lambda e: (TIPOS.index(e["tipo"]), e["ticker"], e["nivel"] or ""))
    return nuevos, deshechos


# ── 3. Mensaje (HTML de Telegram) ─────────────────────────────────────────────

def _motivo_corto(motivos: str) -> str:
    return _esc(motivos.split("; ")[0]) + (" +" if "; " in motivos else "")


def _linea(ev: dict, motivos: dict, prefijo: str = "") -> str:
    # Distancia al nivel con 2 decimales: con 1, un cruce por 0,04% se lee "-0,0%"
    # (mismo ajuste que hizo el cierre por sector, docs/cierre_grupos.md).
    tk = f"<b>{_esc(ev['ticker'])}</b>{prefijo}"
    m = f" <i>({_motivo_corto(motivos.get(ev['ticker'], ''))})</i>" if motivos.get(ev["ticker"]) else ""
    if ev["tipo"] == "movimiento":
        return (f"{tk} {_sg(ev['var_pct'])} desde el cierre "
                f"(z {'+' if ev['z'] >= 0 else ''}{_num(ev['z'], 1)}){m}")
    verbo = "vuelve sobre" if ev["tipo"] == "revierte" and ev["lado"] == "sobre" else \
            "vuelve bajo" if ev["tipo"] == "revierte" else \
            "pasa sobre" if ev["lado"] == "sobre" else "pasa bajo"
    return (f"{tk} {verbo} {NIVELES[ev['nivel']]} {_num(ev['valor'])} "
            f"({_sg(ev['dist_pct'], 2)}){m}")


def _seccion(titulo: str, evs: list, motivos: dict, maximo: int, prefijo: str = "") -> list:
    if not evs:
        return []
    lineas = [f"<b>{titulo} ({len(evs)})</b>"]
    lineas += [_linea(e, motivos, prefijo) for e in evs[:maximo]]
    if len(evs) > maximo:
        lineas.append(f"... y {len(evs) - maximo} mas")
    return lineas + [""]


def mensaje(rueda: date, corte: str, rueda_lista: date, tickers: list,
            nuevos: list, deshechos: list, sin_dato: list,
            hay_previo: Optional[bool] = None) -> str:
    """
    hay_previo: None en el corte de las 11:00; True/False a las 15:00 segun si
    hubo corte de las 11:00 para comparar.

    Un solo mensaje por corte: si no entra en MAX_LEN_MENSAJE con MAX_LINEAS por
    seccion, se achica la lista de cada seccion (el total se informa igual). No
    se corta el texto: partir una etiqueta HTML hace que Telegram rechace todo.
    """
    for maximo in (MAX_LINEAS, 6, 3, 1):
        txt = _armar(rueda, corte, rueda_lista, tickers, nuevos, deshechos,
                     sin_dato, hay_previo, maximo)
        if len(txt) <= MAX_LEN_MENSAJE:
            break
    return txt


def _armar(rueda, corte, rueda_lista, tickers, nuevos, deshechos, sin_dato,
           hay_previo, maximo) -> str:
    motivos = {t["ticker"]: t["motivos"] for t in tickers}
    n = len(tickers)
    lineas = [f"<b>Intradia {_CORTE_TXT.get(corte, corte)}</b> | rueda {_dm(rueda)} | "
              f"lista del {_dm(rueda_lista)}",
              f"{n} vigilados" + (f" | sin dato del corte: {len(sin_dato)}" if sin_dato else ""),
              ""]
    if hay_previo is False:
        lineas += ["<i>Sin el corte de las 11:00 para comparar: se informa todo.</i>", ""]
    pref = "Nuevos: " if hay_previo else ""
    cuerpo = []
    cuerpo += _seccion(pref + "cruzan un nivel",
                       [e for e in nuevos if e["tipo"] == "cruce"], motivos, maximo)
    cuerpo += _seccion(pref + "revierten el cruce de ayer",
                       [e for e in nuevos if e["tipo"] == "revierte"], motivos, maximo)
    cuerpo += _seccion(pref + f"movimiento grande (|z| >= {_num(UMBRAL_Z, 1)})",
                       [e for e in nuevos if e["tipo"] == "movimiento"], motivos, maximo)
    # Lo deshecho se describe como estaba a las 11:00, y se aclara.
    cuerpo += _seccion("Se deshicieron desde las 11:00", deshechos, motivos, maximo,
                       prefijo=" (a las 11:00)")
    if not cuerpo:
        cuerpo = [f"{n} vigilados, sin novedades" +
                  (" desde las 11:00." if hay_previo else "."), ""]
    lineas += cuerpo
    lineas.append("<i>Estado, no pronostico.</i>")
    return "\n".join(lineas)


def mensaje_sin_lista(rueda: date, corte: str, esperada: date,
                      disponible: Optional[date]) -> str:
    disp = f"la ultima es del {_dm(disponible)}" if disponible else "no hay ninguna"
    return (f"<b>Intradia {_CORTE_TXT.get(corte, corte)}</b> | rueda {_dm(rueda)}\n"
            f"Lista de vigilancia de la rueda {_dm(esperada)} no subida ({disp}): "
            f"no se evalua.\nSe sube con intradia_post_rutina.bat despues de la rutina.")


def lineas_por_tipo(nuevos: list, deshechos: list) -> dict:
    """Cuantas lineas lleva el mensaje por tipo (criterio 2 de la evaluacion)."""
    out = {t: sum(1 for e in nuevos if e["tipo"] == t) for t in TIPOS}
    out["deshechos"] = len(deshechos)
    return out
