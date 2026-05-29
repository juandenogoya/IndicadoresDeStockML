"""
dashboard_sintesis.py
El "cerebro" del Dashboard (informe descriptivo por ticker). Cruza tecnico +
opciones + estructura (SMC) + sector y produce un VEREDICTO de 3 valores
(ALCISTA / NEUTRAL / BAJISTA) + una frase, mas la lista de reglas activadas.

Funcion PURA: no toca DB ni tiene side effects. Recibe dicts de datos ya
traidos por la capa de fetch del dashboard y devuelve la sintesis. Asi es
testeable sin UI ni base de datos, e importable tanto por el dashboard como
(a futuro) por la tool MCP get_ticker_sintesis.

Diseno cerrado en dashboard/README.md (fuente de verdad). Resumen del arbol:

  Capa 1 - cada dimension del TICKER vota su sesgo (igual peso):
    A (tecnico)    : MACD diario + semanal (concordancia). RSI/SMA/ADX = matiz.
    B (opciones)   : PCR_oi por plazo, SOLO por concordancia de los 3 plazos.
                     Divergencia -> B no vota (Mixto) + matiz B3.
    F (estructura) : SMC ventana estrategica (10). CHoCH/BOS/estructura_10.

  Capa 2 - consenso:
    - 3 dimensiones presentes: gana la direccion con >=2 votos y mas que la
      contraria; repartido -> NEUTRAL.
    - 2 presentes: exigir 2/2 (ambas en la misma direccion).
    - 1 presente: vota esa dimension (baja confianza).
    - dimension sin datos se OMITE.

  Capa 3 - complementos (modulan la FRASE, NO el estado):
    C (muros de OI + price action), D (sector via opciones).

Las reglas son heuristicas DESCRIPTIVAS (no senales de trading validadas).
ASCII puro en strings (regla de encoding del proyecto).
"""

from typing import Optional

from src.utils.clasificacion_tecnica import (
    clasificar_rsi,
    clasificar_macd,
    clasificar_adx,
    clasificar_tendencia_sma,
    clasificar_estructura_smc,
)

# Umbrales de los complementos (Capa 3)
WALL_CERCANO_PCT = 3.0   # muro "cercano" al precio (|dist_pct| <= 3%)
Z_SECTOR_INUSUAL = 1.5   # |z| del PCR_vol sectorial para "inusual"

# Ventana de opciones usada para las reglas de muros (mayor OI = mas confiable)
_VENTANA_MUROS = "medio"

ALCISTA, BAJISTA, MIXTO, NEUTRAL = "Alcista", "Bajista", "Mixto", "Neutral"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _signo(voto: Optional[str]) -> Optional[int]:
    """Mapea el voto de una dimension a +1 / -1 / 0 (presente sin direccion) /
    None (sin datos -> se omite del consenso)."""
    if voto == ALCISTA:
        return 1
    if voto == BAJISTA:
        return -1
    if voto in (MIXTO, NEUTRAL):
        return 0
    return None


# ── Capa 1: votos por dimension ─────────────────────────────────────────────────

def votar_tecnico(diario: dict, semanal: dict) -> dict:
    """
    Dimension A. Vota por concordancia del MACD diario y semanal.
      A1: Compra en ambos -> Alcista
      A2: Venta en ambos  -> Bajista
      A3: difieren         -> Mixto
    Si solo hay un timeframe, vota ese. Sin MACD en ninguno -> None.
    RSI/SMA/ADX no votan (son matices de la frase).

    Args:
        diario, semanal: dicts con claves rsi, macd, macd_signal (y para
            diario opcionalmente adx, close, sma21/50/200).
    Returns:
        {"voto": str|None, "macd_d": str|None, "macd_w": str|None}
    """
    diario = diario or {}
    semanal = semanal or {}
    macd_d = clasificar_macd(diario.get("macd"), diario.get("macd_signal"))
    macd_w = clasificar_macd(semanal.get("macd"), semanal.get("macd_signal"))

    pres = [m for m in (macd_d, macd_w) if m in ("Compra", "Venta")]
    if not pres:
        voto = None
    elif len(pres) == 1:
        voto = ALCISTA if pres[0] == "Compra" else BAJISTA
    elif macd_d == macd_w == "Compra":
        voto = ALCISTA
    elif macd_d == macd_w == "Venta":
        voto = BAJISTA
    else:
        voto = MIXTO
    return {"voto": voto, "macd_d": macd_d, "macd_w": macd_w}


def votar_opciones(plazos: dict) -> dict:
    """
    Dimension B. Vota SOLO por concordancia de los 3 plazos (decision 27/5):
      B1: veredicto_oi Alcista en los 3 plazos con liquidez -> Alcista
      B2: veredicto_oi Bajista en los 3 plazos con liquidez -> Bajista
      B3: divergencia entre plazos -> Mixto (B no vota; matiz en la frase)
    Si no hay liquidez en los 3 plazos -> None (sin datos suficientes).
    Nunca se agrega/promedia un PCR unico.

    Args:
        plazos: {"corto": {...}, "medio": {...}, "largo": {...}} donde cada
            bloque tiene "veredicto_oi" en {"Alcista","Bajista",None}.
    Returns:
        {"voto": str|None, "veredictos": {ventana: str|None}, "divergencia": bool}
    """
    plazos = plazos or {}
    vers = {}
    for v in ("corto", "medio", "largo"):
        bloque = plazos.get(v) or {}
        vers[v] = bloque.get("veredicto_oi")  # "Alcista"/"Bajista"/None

    con_liquidez = [x for x in vers.values() if x in (ALCISTA, BAJISTA)]
    divergencia = (ALCISTA in con_liquidez) and (BAJISTA in con_liquidez)

    if len(con_liquidez) == 3 and all(x == ALCISTA for x in con_liquidez):
        voto = ALCISTA
    elif len(con_liquidez) == 3 and all(x == BAJISTA for x in con_liquidez):
        voto = BAJISTA
    elif divergencia:
        voto = MIXTO            # presente pero sin direccion (omite del consenso)
    else:
        voto = None             # sin liquidez suficiente -> se omite
    return {"voto": voto, "veredictos": vers, "divergencia": divergencia}


def votar_estructura(smc: dict) -> dict:
    """
    Dimension F. Vota la estructura SMC (ventana estrategica 10).
    Delega en clasificar_estructura_smc (CHoCH manda; sino estructura_10).

    Args:
        smc: dict con estructura_10, choch_bull_10, choch_bear_10,
             bos_bull_10, bos_bear_10.
    Returns:
        {"voto": str|None, "etiqueta": str|None}
    """
    smc = smc or {}
    if not any(k in smc for k in ("estructura_10", "choch_bull_10", "choch_bear_10",
                                  "bos_bull_10", "bos_bear_10")):
        return {"voto": None, "etiqueta": None}
    r = clasificar_estructura_smc(
        smc.get("estructura_10"),
        smc.get("choch_bull_10"), smc.get("choch_bear_10"),
        smc.get("bos_bull_10"), smc.get("bos_bear_10"),
    )
    return {"voto": r["voto"], "etiqueta": r["etiqueta"]}


# ── Capa 2: consenso ─────────────────────────────────────────────────────────────

def consensuar(voto_a: Optional[str], voto_b: Optional[str], voto_f: Optional[str]) -> dict:
    """
    Combina los votos de las 3 dimensiones en un ESTADO de 3 valores.
    Dimensiones sin datos (voto None) se omiten.

    Returns:
        {"estado": "ALCISTA"|"NEUTRAL"|"BAJISTA",
         "n_dim": int, "alcistas": int, "bajistas": int, "confianza": str}
    """
    signos = {"A": _signo(voto_a), "B": _signo(voto_b), "F": _signo(voto_f)}
    presentes = {d: s for d, s in signos.items() if s is not None}
    n = len(presentes)
    alc = sum(1 for s in presentes.values() if s == 1)
    baj = sum(1 for s in presentes.values() if s == -1)

    if n == 0:
        estado, conf = "NEUTRAL", "sin_datos"
    elif n == 1:
        # Una sola dimension: vota su direccion, baja confianza.
        s = next(iter(presentes.values()))
        estado = "ALCISTA" if s == 1 else "BAJISTA" if s == -1 else "NEUTRAL"
        conf = "baja"
    elif n == 2:
        # Exigir 2/2 en la misma direccion.
        if alc == 2:
            estado = "ALCISTA"
        elif baj == 2:
            estado = "BAJISTA"
        else:
            estado = "NEUTRAL"
        conf = "media"
    else:  # n == 3: consenso 2/3
        if alc >= 2 and alc > baj:
            estado = "ALCISTA"
        elif baj >= 2 and baj > alc:
            estado = "BAJISTA"
        else:
            estado = "NEUTRAL"
        conf = "alta"

    return {"estado": estado, "n_dim": n, "alcistas": alc,
            "bajistas": baj, "confianza": conf}


# ── Capa 3: reglas que modulan la frase (NO el estado) ──────────────────────────

def _es_reversion_alcista(pa: dict) -> bool:
    pa = pa or {}
    return bool(pa.get("patron_hammer") or pa.get("patron_engulfing_bull"))


def _es_reversion_bajista(pa: dict) -> bool:
    pa = pa or {}
    return bool(pa.get("patron_shooting_star") or pa.get("patron_engulfing_bear"))


def _reglas_tecnico(votos_a: dict, diario: dict, reglas: list) -> None:
    """Familia A: confluencia/divergencia MACD (vota) + matices RSI/SMA/ADX."""
    macd_d, macd_w = votos_a.get("macd_d"), votos_a.get("macd_w")
    if macd_d and macd_w:
        if macd_d == macd_w == "Compra":
            reglas.append(_r("A1", "tecnico", "MACD en Compra en diario y semanal: momentum alcista alineado."))
        elif macd_d == macd_w == "Venta":
            reglas.append(_r("A2", "tecnico", "MACD en Venta en diario y semanal: momentum bajista alineado."))
        else:
            reglas.append(_r("A3", "tecnico", "MACD difiere entre diario (%s) y semanal (%s): senal poco confiable." % (macd_d, macd_w)))

    diario = diario or {}
    rsi_d = clasificar_rsi(diario.get("rsi"))
    if rsi_d == "Sobrecompra":
        reglas.append(_r("A4", "matiz", "RSI diario en sobrecompra: riesgo de correccion de corto plazo."))
    elif rsi_d == "Sobreventa":
        reglas.append(_r("A5", "matiz", "RSI diario en sobreventa: posible rebote tecnico."))

    tend = clasificar_tendencia_sma(diario.get("close"), diario.get("sma21"),
                                    diario.get("sma50"), diario.get("sma200"))
    adx = clasificar_adx(diario.get("adx"))
    if tend == "Alcista" and adx == "Fuerte":
        reglas.append(_r("A6", "refuerzo", "Tendencia alcista por medias con ADX fuerte: tendencia con fuerza."))
    elif tend == "Bajista" and adx == "Fuerte":
        reglas.append(_r("A6", "refuerzo", "Tendencia bajista por medias con ADX fuerte: tendencia bajista con fuerza."))


def _reglas_opciones(votos_b: dict, diario: dict, reglas: list) -> None:
    """Familia B: posicionamiento y divergencia por plazo."""
    voto = votos_b.get("voto")
    if voto == ALCISTA:
        reglas.append(_r("B1", "opciones", "PCR_oi alcista en los 3 plazos: posicionamiento alcista consistente."))
    elif voto == BAJISTA:
        reglas.append(_r("B2", "opciones", "PCR_oi bajista en los 3 plazos: posicionamiento bajista consistente."))
    if votos_b.get("divergencia"):
        vers = votos_b.get("veredictos") or {}
        detalle = ", ".join("%s %s" % (v, (vers.get(v) or "s/d").lower()) for v in ("corto", "medio", "largo"))
        reglas.append(_r("B3", "matiz", "Opciones divergen por plazo (%s): expectativa distinta segun horizonte; las opciones no fijan un sesgo unico." % detalle))

    # B4: cruce tecnico x opciones (refuerzo bajista)
    macd_d = clasificar_macd((diario or {}).get("macd"), (diario or {}).get("macd_signal"))
    vers = votos_b.get("veredictos") or {}
    if macd_d == "Venta" and vers.get("medio") == BAJISTA:
        reglas.append(_r("B4", "refuerzo", "MACD diario en Venta + PCR_oi del plazo medio bajista: presion bajista confirmada por opciones."))


def _reglas_muros(diario: dict, plazos: dict, pa: dict, reglas: list) -> None:
    """Familia C: muros de OI como techo/piso, reforzados por price action."""
    diario = diario or {}
    rsi_d = clasificar_rsi(diario.get("rsi"))
    medio = (plazos or {}).get(_VENTANA_MUROS) or {}
    call_w = medio.get("call_wall")
    put_w = medio.get("put_wall")

    cerca_arriba = (call_w and call_w.get("posicion") == "arriba"
                    and abs(call_w.get("dist_pct", 99)) <= WALL_CERCANO_PCT)
    cerca_abajo = (put_w and put_w.get("posicion") == "debajo"
                   and abs(put_w.get("dist_pct", 99)) <= WALL_CERCANO_PCT)

    if rsi_d == "Sobrecompra" and cerca_arriba:
        msg = "RSI en sobrecompra + call wall a +%.1f%% (strike %s): techo de opciones cerca." % (call_w["dist_pct"], call_w.get("strike"))
        if _es_reversion_bajista(pa) and (pa or {}).get("vol_spike"):
            msg += " Patron de reversion bajista con volumen lo refuerza."
        reglas.append(_r("C1", "riesgo", msg))

    if rsi_d == "Sobreventa" and cerca_abajo:
        msg = "RSI en sobreventa + put wall a %.1f%% (strike %s): piso de opciones cerca, zona de rebote." % (put_w["dist_pct"], put_w.get("strike"))
        if _es_reversion_alcista(pa) and (pa or {}).get("vol_spike"):
            msg += " Patron de reversion alcista con volumen lo refuerza."
        reglas.append(_r("C2", "oportunidad", msg))

    # C3: precio acotado entre soporte y resistencia cercanos
    if cerca_arriba and cerca_abajo:
        reglas.append(_r("C3", "matiz", "Precio entre put wall y call wall cercanos: rango acotado de corto plazo."))


def _reglas_sector(estado: str, sector_plazo: dict, reglas: list) -> None:
    """Familia D: el sector acompana o no (contexto, no cambia el estado)."""
    sec_medio = (sector_plazo or {}).get(_VENTANA_MUROS) or {}
    z = sec_medio.get("pcr_vol_sector_zscore")
    ver_sec = sec_medio.get("veredicto_oi")

    if estado == "ALCISTA":
        if ver_sec == ALCISTA:
            reglas.append(_r("D1", "contexto", "El sector tambien se posiciona alcista: acompana."))
        elif z is not None and z >= Z_SECTOR_INUSUAL:
            reglas.append(_r("D2", "contexto", "El sector muestra cobertura inusual (PCR_vol z=%+.1f): el sector no acompana, cautela." % z))
    if z is not None and abs(z) >= Z_SECTOR_INUSUAL:
        reglas.append(_r("D3", "contexto", "PCR_vol sectorial inusual (z=%+.1f): posible rotacion sectorial." % z))


def _r(rid: str, tipo: str, mensaje: str) -> dict:
    return {"id": rid, "tipo": tipo, "mensaje": mensaje}


# ── Frase del veredicto ─────────────────────────────────────────────────────────

# Pistas cortas para la frase del veredicto, por id de regla. La frase es de
# UNA linea; el mensaje completo de cada regla vive en la lista `reglas`.
_PISTAS_FRASE = {
    "A4": "con RSI diario en sobrecompra",
    "A5": "con RSI diario en sobreventa",
    "A6": "con la tendencia confirmada por ADX",
    "B3": "con las opciones divergiendo por plazo",
    "B4": "con presion bajista en el plazo medio de opciones",
    "C1": "con un techo de opciones cerca",
    "C2": "con un piso de opciones cerca",
    "C3": "en un rango acotado por muros de opciones",
    "D1": "y el sector acompana",
    "D2": "pero el sector no acompana",
    "D3": "con posible rotacion sectorial",
}


def _frase_veredicto(estado: str, cons: dict, votos_b: dict, reglas: list) -> str:
    """Arma la frase de 1 linea (corta) que acompana al estado en el encabezado."""
    if cons["n_dim"] == 0:
        return "sin datos suficientes para una lectura."

    base = {
        "ALCISTA": "consenso alcista",
        "BAJISTA": "consenso bajista",
        "NEUTRAL": "lectura mixta sin direccion clara",
    }[estado]

    # Hasta 2 pistas, en el orden en que se dispararon, sin repetir.
    pistas, vistos = [], set()
    for r in reglas:
        p = _PISTAS_FRASE.get(r["id"])
        if p and p not in vistos:
            pistas.append(p)
            vistos.add(p)
        if len(pistas) == 2:
            break

    frase = base
    if pistas:
        frase += " " + ", ".join(pistas)
    if cons["confianza"] == "baja":
        frase += " (solo 1 dimension con datos)"
    return frase + "."


# ── Entrada principal ────────────────────────────────────────────────────────────

def sintetizar(datos: dict) -> dict:
    """
    Sintesis completa del ticker. Funcion pura.

    Args:
        datos: {
          "tecnico":   {"diario": {rsi, macd, macd_signal, adx, close,
                                    sma21, sma50, sma200},
                        "semanal": {rsi, macd, macd_signal}},
          "opciones_plazo": {"corto"|"medio"|"largo":
                        {pcr_vol, pcr_oi, veredicto_oi, put_wall, call_wall}},
          "sector_plazo":   {"corto"|"medio"|"largo":
                        {pcr_vol_sector, veredicto_oi, pcr_vol_sector_zscore}},
          "price_action":   {patron_*, vol_spike, es_alcista},
          "estructura":     {estructura_10, choch_bull_10, choch_bear_10,
                             bos_bull_10, bos_bear_10},
        }
        Cualquier seccion puede faltar o venir vacia (se omite del voto).

    Returns:
        {"estado", "frase", "votos": {A,B,F}, "consenso": {...},
         "reglas": [{id,tipo,mensaje}], "etiqueta_smc": str|None}
    """
    datos = datos or {}
    tecnico = datos.get("tecnico") or {}
    diario = tecnico.get("diario") or {}
    semanal = tecnico.get("semanal") or {}
    plazos = datos.get("opciones_plazo") or {}
    sector_plazo = datos.get("sector_plazo") or {}
    pa = datos.get("price_action") or {}
    smc = datos.get("estructura") or {}

    votos_a = votar_tecnico(diario, semanal)
    votos_b = votar_opciones(plazos)
    votos_f = votar_estructura(smc)

    cons = consensuar(votos_a["voto"], votos_b["voto"], votos_f["voto"])
    estado = cons["estado"]

    reglas: list = []
    _reglas_tecnico(votos_a, diario, reglas)
    _reglas_opciones(votos_b, diario, reglas)
    _reglas_muros(diario, plazos, pa, reglas)
    _reglas_sector(estado, sector_plazo, reglas)

    frase = _frase_veredicto(estado, cons, votos_b, reglas)

    return {
        "estado": estado,
        "frase": frase,
        "votos": {"A": votos_a["voto"], "B": votos_b["voto"], "F": votos_f["voto"]},
        "consenso": cons,
        "reglas": reglas,
        "etiqueta_smc": votos_f.get("etiqueta"),
        "_detalle": {"tecnico": votos_a, "opciones": votos_b},
    }
