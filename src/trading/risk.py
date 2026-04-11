"""
risk.py
Gestion de riesgo: position sizing, stop loss, limites de portafolio.

Parametros configurables via variables de entorno o defaults conservadores.
"""

import os


# ── Parametros de riesgo (defaults conservadores) ─────────────
MAX_POSICIONES      = int(os.getenv("BOT_MAX_POSICIONES", "5"))
RIESGO_POR_TRADE    = float(os.getenv("BOT_RIESGO_POR_TRADE", "0.02"))   # 2% del equity por trade
MAX_EXPOSICION      = float(os.getenv("BOT_MAX_EXPOSICION", "0.80"))      # max 80% del equity invertido
STOP_LOSS_PCT       = float(os.getenv("BOT_STOP_LOSS_PCT", "0.05"))       # stop loss 5%
TAKE_PROFIT_PCT     = float(os.getenv("BOT_TAKE_PROFIT_PCT", "0.10"))     # take profit 10%
SCORE_MIN_COMPRA    = int(os.getenv("BOT_SCORE_MIN_COMPRA", "65"))        # score ML minimo para entrar
NIVEL_MIN_COMPRA    = os.getenv("BOT_NIVEL_MIN_COMPRA", "COMPRA_FUERTE")  # nivel minimo de alerta


def calcular_qty(equity: float, precio: float) -> int:
    """
    Calcula la cantidad de acciones a comprar segun riesgo por trade.
    Retorna entero (acciones enteras, no fracciones).

    equity: capital total de la cuenta
    precio: precio actual del ticker
    """
    capital_a_invertir = equity * RIESGO_POR_TRADE
    qty = int(capital_a_invertir / precio)
    return max(qty, 1)  # minimo 1 accion


def puede_abrir_posicion(
    n_posiciones_actuales: int,
    equity: float,
    capital_invertido: float,
    precio: float,
) -> tuple[bool, str]:
    """
    Verifica si se puede abrir una nueva posicion.
    Retorna (puede_abrir, motivo).
    """
    # Limite de posiciones simultaneas
    if n_posiciones_actuales >= MAX_POSICIONES:
        return False, f"Maximo de posiciones alcanzado ({MAX_POSICIONES})"

    # Limite de exposicion total
    capital_nuevo = precio * calcular_qty(equity, precio)
    exposicion_nueva = (capital_invertido + capital_nuevo) / equity
    if exposicion_nueva > MAX_EXPOSICION:
        return False, f"Exposicion maxima alcanzada ({MAX_EXPOSICION*100:.0f}%)"

    # Capital minimo por operacion
    if precio * 1 > equity * MAX_EXPOSICION:
        return False, "Capital insuficiente para 1 accion"

    return True, "OK"


def calcular_stop_loss(precio_entrada: float) -> float:
    """Precio de stop loss dado el precio de entrada."""
    return round(precio_entrada * (1 - STOP_LOSS_PCT), 2)


def calcular_take_profit(precio_entrada: float) -> float:
    """Precio de take profit dado el precio de entrada."""
    return round(precio_entrada * (1 + TAKE_PROFIT_PCT), 2)


def cumple_filtros_entrada(
    alert_nivel: str,
    score: float,
    tendencia_1w: str = None,
    tendencia_1m: str = None,
    usar_filtro_mtf: bool = False,
) -> tuple[bool, str]:
    """
    Evalua si una senal cumple los filtros de entrada.
    Retorna (cumple, motivo).
    """
    # Filtro de nivel minimo
    niveles_validos = {
        "COMPRA_FUERTE": 4,
        "COMPRA":        3,
        "NEUTRAL":       2,
        "VENTA":         1,
        "VENTA_FUERTE":  0,
    }
    nivel_requerido = niveles_validos.get(NIVEL_MIN_COMPRA, 4)
    nivel_actual    = niveles_validos.get(alert_nivel, 0)

    if nivel_actual < nivel_requerido:
        return False, f"Nivel {alert_nivel} < minimo {NIVEL_MIN_COMPRA}"

    # Filtro de score ML
    if score < SCORE_MIN_COMPRA:
        return False, f"Score {score} < minimo {SCORE_MIN_COMPRA}"

    # Filtro MTF opcional
    if usar_filtro_mtf:
        if tendencia_1w == "bajista":
            return False, "Tendencia 1W bajista (filtro MTF activo)"
        if tendencia_1m == "bajista":
            return False, "Tendencia 1M bajista (filtro MTF activo)"

    return True, "OK"
