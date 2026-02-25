"""
strategies.py
Definición de las estrategias de entrada y salida del backtesting.

ESTRATEGIAS DE ENTRADA:
    E1 — Score Ponderado    : score_ponderado >= 0.60
    E2 — Minimo Condiciones : condiciones_ok >= 4
    E3 — Score Estricto     : score_ponderado >= 0.75
    E4 — Confluence Core    : RSI + MACD + SMA200 obligatorios + 1 mas

ESTRATEGIAS DE SALIDA:
    S1 — Fijo Simple        : SL -3%  / TP +6%  / Timeout 20 dias
    S2 — ATR Dinamico       : SL -1.5xATR / TP +3xATR / Timeout 20 dias
    S3 — Trailing Stop      : trailing 1.5xATR desde max cierre / Timeout 30 dias
    S4 — Señal Contraria    : cierra cuando score <= 0.30 / Timeout 30 dias
"""

import pandas as pd
from src.utils.config import (
    SCORE_ENTRADA_UMBRAL, SCORE_SALIDA_UMBRAL,
    TIMEOUT_DIAS, UMBRAL_NEUTRO,
)


# ─────────────────────────────────────────────────────────────
# Parámetros de cada estrategia
# ─────────────────────────────────────────────────────────────

PARAMS_SALIDA = {
    "S1": {"sl_pct": 0.03,  "tp_pct": 0.06,  "timeout": 20, "trailing": False, "por_senal": False},
    "S2": {"sl_atr": 1.5,   "tp_atr": 3.0,   "timeout": 20, "trailing": False, "por_senal": False},
    "S3": {"sl_atr": 1.5,   "tp_atr": None,  "timeout": 30, "trailing": True,  "por_senal": False},
    "S4": {"sl_pct": 0.05,  "tp_pct": None,  "timeout": 30, "trailing": False, "por_senal": True},
}


# ─────────────────────────────────────────────────────────────
# Evaluadores de Entrada
# ─────────────────────────────────────────────────────────────

def check_entrada(row: pd.Series, estrategia: str) -> bool:
    """
    Evalúa si una fila del dataset cumple la condición de entrada.

    Args:
        row:        fila del DataFrame combinado (scoring + indicadores)
        estrategia: "E1" | "E2" | "E3" | "E4"

    Returns:
        True si se genera señal de entrada LONG
    """
    if estrategia == "E1":
        return row["score_ponderado"] >= SCORE_ENTRADA_UMBRAL

    elif estrategia == "E2":
        return row["condiciones_ok"] >= 4

    elif estrategia == "E3":
        return row["score_ponderado"] >= 0.75

    elif estrategia == "E4":
        # RSI + MACD + SMA200 obligatorios + al menos 1 de los restantes
        core_ok = (
            bool(row["cond_rsi"]) and
            bool(row["cond_macd"]) and
            bool(row["cond_sma200"])
        )
        if not core_ok:
            return False
        adicionales = (
            bool(row["cond_sma21"]) +
            bool(row["cond_sma50"]) +
            bool(row["cond_momentum"])
        )
        return adicionales >= 1

    return False


# ─────────────────────────────────────────────────────────────
# Inicialización de stops al entrar en posición
# ─────────────────────────────────────────────────────────────

def calcular_stops_iniciales(precio_entrada: float, atr: float,
                              estrategia: str) -> tuple:
    """
    Calcula el stop loss y take profit iniciales al abrir posición.

    Returns:
        (stop_loss_precio, take_profit_precio)
        None en take_profit si la estrategia no lo usa.
    """
    p = PARAMS_SALIDA[estrategia]

    if estrategia == "S1":
        sl = precio_entrada * (1 - p["sl_pct"])
        tp = precio_entrada * (1 + p["tp_pct"])

    elif estrategia == "S2":
        sl = precio_entrada - p["sl_atr"] * atr
        tp = precio_entrada + p["tp_atr"] * atr

    elif estrategia == "S3":
        # Trailing: stop inicial = entrada - 1.5×ATR, TP libre
        sl = precio_entrada - p["sl_atr"] * atr
        tp = None

    elif estrategia == "S4":
        # Stop de seguridad fijo al -5%, cierre principal por señal contraria
        sl = precio_entrada * (1 - p["sl_pct"])
        tp = None

    else:
        sl = precio_entrada * (1 - 0.05)
        tp = None

    return sl, tp


# ─────────────────────────────────────────────────────────────
# Evaluadores de Salida (en cada sesión dentro de la posición)
# ─────────────────────────────────────────────────────────────

def check_salida(row: pd.Series, estrategia: str, stop_loss: float,
                 take_profit: float, dias_en_posicion: int) -> tuple:
    """
    Evalúa si la posición debe cerrarse en la sesión actual.

    Args:
        row:              fila del DataFrame combinado
        estrategia:       "S1" | "S2" | "S3" | "S4"
        stop_loss:        precio de stop loss actual (puede ser trailing)
        take_profit:      precio objetivo (None si no aplica)
        dias_en_posicion: cantidad de días que lleva abierta la operación

    Returns:
        (cerrar: bool, motivo: str, precio_salida: float)
        Si no cierra: (False, None, None)
    """
    p = PARAMS_SALIDA[estrategia]
    low   = row["low"]
    high  = row["high"]
    close = row["close"]

    # 1. Stop Loss (hit intraday: low <= stop)
    if low <= stop_loss:
        return True, "STOP_LOSS", round(stop_loss, 4)

    # 2. Take Profit (hit intraday: high >= target)
    if take_profit is not None and high >= take_profit:
        return True, "TAKE_PROFIT", round(take_profit, 4)

    # 3. Señal contraria (S4): score cae bajo umbral de salida
    if p["por_senal"] and row["score_ponderado"] <= SCORE_SALIDA_UMBRAL:
        return True, "SENAL", round(close, 4)

    # 4. Timeout: máximo de días alcanzado
    if dias_en_posicion >= p["timeout"]:
        return True, "TIMEOUT", round(close, 4)

    return False, None, None


def actualizar_trailing_stop(stop_actual: float, high_water_mark: float,
                              close: float, atr: float,
                              estrategia: str) -> tuple:
    """
    Actualiza el trailing stop y el high water mark si corresponde (S3).

    Returns:
        (nuevo_stop, nuevo_high_water_mark)
    """
    if estrategia != "S3":
        return stop_actual, high_water_mark

    p = PARAMS_SALIDA[estrategia]
    nuevo_hwm = max(high_water_mark, close)
    nuevo_stop = nuevo_hwm - p["sl_atr"] * atr

    # El stop solo sube, nunca baja
    nuevo_stop = max(nuevo_stop, stop_actual)

    return round(nuevo_stop, 4), round(nuevo_hwm, 4)


# ─────────────────────────────────────────────────────────────
# Clasificación del resultado
# ─────────────────────────────────────────────────────────────

def clasificar_resultado(retorno_pct: float) -> str:
    """
    Clasifica la operación según el retorno.
        GANANCIA si retorno > +UMBRAL_NEUTRO
        PERDIDA  si retorno < -UMBRAL_NEUTRO
        NEUTRO   en caso contrario
    """
    umbral = UMBRAL_NEUTRO * 100   # convertir a porcentaje (1% default)
    if retorno_pct > umbral:
        return "GANANCIA"
    elif retorno_pct < -umbral:
        return "PERDIDA"
    return "NEUTRO"
