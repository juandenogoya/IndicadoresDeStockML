"""
telegram_notifier.py
Formatea y envia el resumen de alertas del scanner via Telegram.

Dos mensajes por ejecucion:
    1. Scanner ML   : senales no-NEUTRAL con score y ML%
    2. Setups Vela  : patrones hammer / engulfing / shooting star detectados

Requiere TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env.
"""

import requests
from typing import List, Dict
from datetime import datetime

from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


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
                lines.append(
                    f"  <b>{ticker}</b> score={score:.0f} ml={prob:.0%} "
                    f"${precio:.2f} [{scope}]"
                )
        else:
            # Compacto: hasta 5 tickers por linea
            CHUNK = 5
            for i in range(0, len(items_sorted), CHUNK):
                chunk = items_sorted[i:i + CHUNK]
                parts = [
                    f"<b>{r['ticker']}</b> s={r.get('alert_score', 0):.0f}"
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

def formatear_mensaje2_velas(resultados: List[Dict]) -> str:
    """
    Genera el mensaje de setups de vela.

    Filtra tickers con patron_hammer, patron_engulfing_bull,
    patron_shooting_star o patron_engulfing_bear.
    Muestra el nivel ML como contexto y agrega campanita si vol_spike.
    Retorna string vacio si no hay patrones detectados.
    """
    now = datetime.now().strftime("%Y-%m-%d")

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

        for p in _PATRONES_ALCISTAS:
            if r.get(p):
                alcistas.append((nivel, ticker, _PATRON_NOMBRE[p], vol, score))
                break

        for p in _PATRONES_BAJISTAS:
            if r.get(p):
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
    Mensaje 2: Setups de vela con patrones PA (si existen).

    Los resultados deben incluir los campos de patrones de vela:
        patron_hammer, patron_shooting_star, patron_engulfing_bull,
        patron_engulfing_bear, es_alcista, vol_spike.
    Estos se agregan en cron_diario.paso_scanner() desde calc["features_pa"].

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
    texto2 = formatear_mensaje2_velas(resultados)
    ok2 = True
    if texto2:
        ok2 = _send_long(texto2)

    return ok1
