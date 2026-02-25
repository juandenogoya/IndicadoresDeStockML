"""
telegram_notifier.py
Formatea y envia el resumen de alertas del scanner via Telegram.

Requiere TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env.
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime

from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


# ─────────────────────────────────────────────────────────────
# Emojis por nivel de alerta (ASCII-safe labels como fallback)
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


# ─────────────────────────────────────────────────────────────
# Formateo del resumen
# ─────────────────────────────────────────────────────────────

def formatear_resumen(resultados: List[Dict]) -> str:
    """
    Genera el texto del mensaje Telegram a partir de la lista de resultados.

    Cada elemento de resultados es un dict con keys:
        ticker, sector, alert_nivel, alert_score, alert_detalle,
        precio_cierre, precio_fecha, ml_prob_ganancia, ml_modelo_usado,
        pa_ev1, pa_ev2, pa_ev3, pa_ev4,
        bear_bos10, bear_choch10, bear_estructura,
        score_ponderado, error (solo si fallo)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_total = len(resultados)
    n_ok    = sum(1 for r in resultados if not r.get("error"))
    n_err   = n_total - n_ok

    lines = []
    lines.append(f"<b>Scanner de Alertas ML</b>")
    lines.append(f"<i>{now}</i>  |  {n_ok}/{n_total} tickers procesados")
    lines.append("")

    # ── Agrupar por nivel ─────────────────────────────────────
    orden = ["COMPRA_FUERTE", "COMPRA", "NEUTRAL", "VENTA", "VENTA_FUERTE"]
    grupos = {n: [] for n in orden}

    for r in resultados:
        if r.get("error"):
            continue
        nivel = r.get("alert_nivel", "NEUTRAL")
        grupos.setdefault(nivel, []).append(r)

    for nivel in orden:
        items = grupos.get(nivel, [])
        if not items:
            continue

        emoji = _EMOJI.get(nivel, "")
        label = _LABEL.get(nivel, nivel)
        lines.append(f"{emoji} <b>{label}</b> ({len(items)})")

        for r in sorted(items, key=lambda x: x.get("alert_score", 0), reverse=True):
            ticker = r["ticker"]
            score  = r.get("alert_score", 0)
            prob   = r.get("ml_prob_ganancia", 0)
            precio = r.get("precio_cierre", 0)
            sector = r.get("sector") or "?"
            scope  = r.get("ml_modelo_usado", "?")

            # Flags PA y bajistas compactos
            pa_flags  = "".join([
                "1" if r.get("pa_ev1") else ".",
                "2" if r.get("pa_ev2") else ".",
                "3" if r.get("pa_ev3") else ".",
                "4" if r.get("pa_ev4") else ".",
            ])
            bear_flags = "".join([
                "B" if r.get("bear_bos10") else ".",
                "C" if r.get("bear_choch10") else ".",
                "E" if r.get("bear_estructura") else ".",
            ])

            lines.append(
                f"  <b>{ticker}</b> ({sector[:3].upper()}) "
                f"score={score:.0f} ml={prob:.0%} "
                f"${precio:.2f} | EV={pa_flags} bear={bear_flags} | [{scope[:3].upper()}]"
            )

        lines.append("")

    # ── Errores ───────────────────────────────────────────────
    errores = [r for r in resultados if r.get("error")]
    if errores:
        lines.append(f"<b>Errores ({len(errores)}):</b>")
        for r in errores:
            lines.append(f"  {r['ticker']}: {str(r.get('error','?'))[:60]}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Funcion publica principal
# ─────────────────────────────────────────────────────────────

def enviar_resumen(resultados: List[Dict]) -> bool:
    """
    Formatea y envia el resumen de alertas por Telegram.

    Args:
        resultados: lista de dicts con los resultados del scanner

    Returns:
        True si el mensaje fue enviado exitosamente
    """
    if not resultados:
        print("  [WARN] Sin resultados para enviar por Telegram.")
        return False

    texto = formatear_resumen(resultados)

    # Telegram tiene limite de 4096 chars; dividir si es necesario
    MAX_LEN = 4000
    if len(texto) <= MAX_LEN:
        return _send(texto)

    # Partir en chunks por newline
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
