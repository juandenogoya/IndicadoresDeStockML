"""
telegram_notifier.py
Formatea y envia el resumen de alertas del scanner via Telegram.

Dos mensajes por ejecucion:
    1. Scanner ML   : senales no-NEUTRAL con score y ML%
    2. Setups Vela  : patrones hammer / engulfing / shooting star detectados.
                      Desde el 16/10/2026 salen de features_velas (definicion
                      clasica, src/indicators/velas.py), leida por ticker +
                      precio_fecha. NO de los patron_* de features_pa: esos son
                      los de precio_accion.py (el 71% de sus envolventes no
                      envuelve) y siguen alimentando el score del scanner, que
                      este cambio no toca. Ver docs/estructura_velas.md sec. 12.5.

Requiere TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env.
"""

import requests
from typing import List, Dict
from datetime import datetime

from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.utils.contexto_sectorial import marca


# ─────────────────────────────────────────────────────────────
# Emojis y etiquetas por nivel de alerta
# ─────────────────────────────────────────────────────────────

_EMOJI = {
    "COMPRA_FUERTE": "\U0001F7E2\U0001F7E2",   # verde verde
    "COMPRA":        "\U0001F7E2",              # verde
    "NEUTRAL":       "\U0001F7E1",              # amarillo
    "VENTA":         "\U0001F534",              # rojo
    "VENTA_FUERTE":  "\U0001F534\U0001F534",    # rojo rojo
}

_LABEL = {
    "COMPRA_FUERTE": "COMPRA FUERTE",
    "COMPRA":        "COMPRA",
    "NEUTRAL":       "NEUTRAL",
    "VENTA":         "VENTA",
    "VENTA_FUERTE":  "VENTA FUERTE",
}

_PATRON_NOMBRE = {
    "patron_hammer":         "Hammer",
    "patron_engulfing_bull": "Envolvente Alc.",
    "patron_shooting_star":  "Shooting Star",
    "patron_engulfing_bear": "Envolvente Baj.",
}

_PATRONES_ALCISTAS = ["patron_hammer", "patron_engulfing_bull"]
_PATRONES_BAJISTAS = ["patron_shooting_star", "patron_engulfing_bear"]
_PATRONES_TELEGRAM = _PATRONES_ALCISTAS + _PATRONES_BAJISTAS


# ─────────────────────────────────────────────────────────────
# Envio raw via Bot API
# ─────────────────────────────────────────────────────────────

def _send(text: str) -> bool:
    """
    Envia un mensaje de texto a Telegram.
    Retorna True si fue exitoso.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text":    text,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return True
        print(f"  [WARN] Telegram error {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"  [ERROR] Telegram: {e}")
        return False


def _send_long(texto: str) -> bool:
    """
    Envia un texto largo dividiendolo en chunks de 4000 chars si es necesario.
    """
    MAX_LEN = 4000
    if len(texto) <= MAX_LEN:
        return _send(texto)

    chunks = []
    current = ""
    for line in texto.split("\n"):
        if len(current) + len(line) + 1 > MAX_LEN:
            chunks.append(current)
            current = line + "\n"
        else:
            current += line + "\n"
    if current:
        chunks.append(current)

    ok = True
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}/{len(chunks)}] " if len(chunks) > 1 else ""
        ok = ok and _send(header + chunk)
    return ok


# ─────────────────────────────────────────────────────────────
# Mensaje 1: Scanner ML (sin NEUTRAL)
# ─────────────────────────────────────────────────────────────

def formatear_mensaje1_scanner(resultados: List[Dict]) -> str:
    """
    Genera el mensaje de scanner ML.

    - Omite NEUTRAL (evita mensajes de 124 tickers).
    - COMPRA_FUERTE / VENTA_FUERTE: una linea detallada por ticker.
    - COMPRA / VENTA: formato compacto (hasta 5 tickers por linea).
    """
    now     = datetime.now().strftime("%Y-%m-%d")
    n_total = len(resultados)
    n_ok    = sum(1 for r in resultados if not r.get("error"))
    n_err   = n_total - n_ok

    orden  = ["COMPRA_FUERTE", "COMPRA", "VENTA", "VENTA_FUERTE"]
    grupos = {n: [] for n in orden}

    for r in resultados:
        if r.get("error"):
            continue
        nivel = r.get("alert_nivel", "NEUTRAL")
        if nivel in grupos:
            grupos[nivel].append(r)

    n_alcistas = len(grupos["COMPRA_FUERTE"]) + len(grupos["COMPRA"])
    n_bajistas = len(grupos["VENTA"]) + len(grupos["VENTA_FUERTE"])

    lines = []
    lines.append(f"<b>Scanner ML \u2014 {now}</b>")
    lines.append(
        f"<i>{n_ok} tickers | {n_alcistas} alcistas | {n_bajistas} bajistas</i>"
    )

    for nivel in orden:
        items = grupos.get(nivel, [])
        if not items:
            continue

        emoji = _EMOJI.get(nivel, "")
        label = _LABEL.get(nivel, nivel)
        items_sorted = sorted(items, key=lambda x: x.get("alert_score", 0), reverse=True)

        lines.append("")
        lines.append(f"<b>{emoji} {label} ({len(items)})</b>")

        if "FUERTE" in nivel:
            # Una linea por ticker con detalle completo
            for r in items_sorted:
                ticker = r["ticker"]
                score  = r.get("alert_score", 0)
                prob   = r.get("ml_prob_ganancia", 0)
                precio = r.get("precio_cierre", 0) or 0
                scope  = (r.get("ml_modelo_usado") or "?")[:3].upper()
                # Indicadores MTF
                t1w = r.get("tendencia_1w", "neutral")
                t1m = r.get("tendencia_1m", "neutral")
                _colores = {"alcista": "\U0001f7e2", "bajista": "\U0001f534", "neutral": "\u26ab"}
                _nd = "\u26ab"
                mtf_str = f"{_colores.get(t1w, _nd)}1W {_colores.get(t1m, _nd)}1M"
                # La marca va al final y con el texto completo: este es el
                # aviso que se lee en el celular y decide si vale la pena
                # mirar el ticker. COMPRA_FUERTE es ademas el nivel con el que
                # abren posicion los bots, asi que es donde mas importa saber
                # que el score se calculo con 11 features vacias.
                lines.append(
                    f"  <b>{ticker}</b> score={score:.0f} ml={prob:.0%} "
                    f"${precio:.2f} [{scope}] {mtf_str}"
                    f"{marca(r.get('sector'))}"
                )
        else:
            # Compacto: hasta 5 tickers por linea
            CHUNK = 5
            for i in range(0, len(items_sorted), CHUNK):
                chunk = items_sorted[i:i + CHUNK]
                # Aca la marca va abreviada: son hasta 5 tickers por linea y la
                # etiqueta completa partiria el renglon en el celular.
                parts = [
                    f"<b>{r['ticker']}</b>{marca(r.get('sector'), corta=True)}"
                    f" s={r.get('alert_score', 0):.0f}"
                    for r in chunk
                ]
                lines.append("  " + " | ".join(parts))

    lines.append("")
    if n_err > 0:
        lines.append(f"<i>{n_err} errores en {n_total} tickers</i>")
    else:
        lines.append(f"<i>{n_ok} tickers procesados sin errores</i>")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Mensaje 2: Setups de Vela
# ─────────────────────────────────────────────────────────────

def _fecha_datos(r: Dict):
    """Rueda de la barra del resultado (precio_fecha) como date, o None."""
    f = r.get("precio_fecha")
    if f is None:
        return None
    return f.date() if hasattr(f, "date") and callable(f.date) else f


def cargar_velas_clasicas(resultados: List[Dict]) -> Dict[tuple, Dict[str, int]]:
    """
    Lee de features_velas los patrones de la rueda de cada resultado.

    La clave es (ticker, precio_fecha): precio_fecha es la rueda sobre la que
    se calculo la alerta. scan_fecha es cuando corrio el scanner y con ella se
    leeria el dia equivocado (patron critico "alertas_scanner: scan_fecha NO es
    la fecha de datos").

    Returns:
        {(ticker, date): {patron: 0/1}}. Un par sin fila en features_velas no
        aparece: ese ticker sale sin patron (y se avisa cuantos fueron).
    """
    pares = {(r["ticker"], _fecha_datos(r)) for r in resultados
             if not r.get("error") and _fecha_datos(r) is not None}
    if not pares:
        return {}

    from src.data.database import query_df
    cols = ", ".join(_PATRONES_TELEGRAM)
    df = query_df(
        f"SELECT ticker, fecha, {cols} FROM features_velas "
        "WHERE ticker = ANY(:tickers) AND fecha = ANY(:fechas)",
        {"tickers": sorted({t for t, _ in pares}),
         "fechas": sorted({f for _, f in pares})},
    )
    velas = {}
    for row in df.to_dict("records"):
        f = row["fecha"]
        f = f.date() if hasattr(f, "date") and callable(f.date) else f
        clave = (row["ticker"], f)
        if clave in pares:
            velas[clave] = {p: int(row[p] or 0) for p in _PATRONES_TELEGRAM}

    faltan = len(pares) - len(velas)
    if faltan:
        print(f"  [WARN] {faltan} tickers sin fila en features_velas para su "
              f"rueda: salen sin patron en Setups de Vela (correr el Paso 2c).")
    return velas


def formatear_mensaje2_velas(resultados: List[Dict],
                             velas: Dict[tuple, Dict[str, int]] = None) -> str:
    """
    Genera el mensaje de setups de vela.

    Los patrones (hammer, envolventes, shooting star) salen de `velas`
    (cargar_velas_clasicas: features_velas por ticker + precio_fecha), NUNCA de
    los patron_* que trae cada resultado -- esos son los de precio_accion.py.
    Sin `velas` no hay patrones. vol_spike si sale del resultado (bien calculado).
    Muestra el nivel ML como contexto y agrega campanita si vol_spike.
    Retorna string vacio si no hay patrones detectados.
    """
    now = datetime.now().strftime("%Y-%m-%d")
    velas = velas or {}

    # (nivel, ticker, patron_nombre, vol_spike, alert_score)
    alcistas: list = []
    bajistas: list = []

    for r in resultados:
        if r.get("error"):
            continue

        nivel  = r.get("alert_nivel", "NEUTRAL")
        ticker = r["ticker"]
        vol    = bool(r.get("vol_spike"))
        score  = r.get("alert_score", 0)
        patrones = velas.get((ticker, _fecha_datos(r)), {})

        for p in _PATRONES_ALCISTAS:
            if patrones.get(p):
                alcistas.append((nivel, ticker, _PATRON_NOMBRE[p], vol, score))
                break

        for p in _PATRONES_BAJISTAS:
            if patrones.get(p):
                bajistas.append((nivel, ticker, _PATRON_NOMBRE[p], vol, score))
                break

    if not alcistas and not bajistas:
        return ""

    n_total = len(alcistas) + len(bajistas)
    lines = []
    lines.append(f"<b>Setups de Vela \u2014 {now}</b>")
    lines.append(f"<i>{n_total} patrones detectados</i>")

    if alcistas:
        lines.append("")
        lines.append("<b>Alcistas</b>")
        for nivel, ticker, patron, vol, score in sorted(alcistas, key=lambda x: x[4], reverse=True):
            emoji   = _EMOJI.get(nivel, " ")
            vol_str = " \U0001F514" if vol else ""
            lines.append(f"  {emoji} <b>{ticker}</b>  {patron}{vol_str}")

    if bajistas:
        lines.append("")
        lines.append("<b>Bajistas</b>")
        for nivel, ticker, patron, vol, score in sorted(bajistas, key=lambda x: x[4]):
            emoji   = _EMOJI.get(nivel, " ")
            vol_str = " \U0001F514" if vol else ""
            lines.append(f"  {emoji} <b>{ticker}</b>  {patron}{vol_str}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Funcion publica principal
# ─────────────────────────────────────────────────────────────

def enviar_resumen(resultados: List[Dict]) -> bool:
    """
    Formatea y envia el resumen de alertas por Telegram (2 mensajes).

    Mensaje 1: Scanner ML (solo senales no-NEUTRAL).
    Mensaje 2: Setups de vela (si existen), con los patrones de features_velas
    leidos por ticker + precio_fecha (cargar_velas_clasicas). De cada resultado
    se usan ticker, precio_fecha, alert_nivel, alert_score y vol_spike.

    Args:
        resultados: lista de dicts con resultados del scanner.

    Returns:
        True si el mensaje 1 fue enviado exitosamente.
    """
    if not resultados:
        print("  [WARN] Sin resultados para enviar por Telegram.")
        return False

    # Mensaje 1: Scanner ML (siempre se envia)
    texto1 = formatear_mensaje1_scanner(resultados)
    ok1 = _send_long(texto1)

    # Mensaje 2: Setups de vela (solo si hay patrones)
    try:
        velas = cargar_velas_clasicas(resultados)
    except Exception as e:
        # Sin features_velas no se inventan patrones: el mensaje 2 no sale.
        print(f"  [WARN] No se pudo leer features_velas ({e}): sin Setups de Vela.")
        velas = {}
    texto2 = formatear_mensaje2_velas(resultados, velas)
    ok2 = True
    if texto2:
        ok2 = _send_long(texto2)

    return ok1
