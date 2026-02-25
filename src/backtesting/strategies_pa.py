"""
strategies_pa.py
Definicion de estrategias de entrada y salida del backtesting PA challenger.

ENTRADAS:
    EV1 -- Estructura + Patron + Volumen (confluencia):
        estructura_10==+1 AND dias_sl_10<=15
        AND (patron_engulfing_bull==1 OR patron_hammer==1)
        AND (vol_spike==1 OR up_vol_5d>0.55)

    EV2 -- BOS Confirmado + Volumen:
        bos_bull_10==1 AND vol_price_confirm==1 AND estructura_10>=0

    EV3 -- CHoCH Alcista (reversion):
        choch_bull_10==1 AND vol_spike==1 AND dist_sl_10_pct>-4.0

    EV4 -- Pullback a Estructura:
        estructura_10==+1 AND dist_sl_10_pct BETWEEN 0 AND +4%
        AND es_alcista==1 AND dias_sl_10<=10

SALIDAS:
    SV1 -- Resistencia estructural:
        Safety SL -5% | dist_sh_10_pct >= -1.0% | Timeout 20d

    SV2 -- Rotura de estructura bajista:
        Safety SL -5% | bos_bear_10==1 OR choch_bear_10==1 | Timeout 20d

    SV3 -- Score deteriorado + estructura:
        Safety SL -5% | score_ponderado<0.30 OR estructura_10==-1 | Timeout 20d

    SV4 -- ATR Stop + Target:
        SL = entry - 1.5*ATR14 | TP = entry + 2.5*ATR14 | Timeout 20d
"""

import pandas as pd
from src.utils.config import UMBRAL_NEUTRO


# ─────────────────────────────────────────────────────────────
# Parametros de cada estrategia de salida
# ─────────────────────────────────────────────────────────────

PARAMS_SALIDA_PA = {
    "SV1": {"safety_sl_pct": 0.05, "timeout": 20},
    "SV2": {"safety_sl_pct": 0.05, "timeout": 20},
    "SV3": {"safety_sl_pct": 0.05, "timeout": 20},
    "SV4": {"sl_atr": 1.5, "tp_atr": 2.5, "timeout": 20},
}


# ─────────────────────────────────────────────────────────────
# Helper: lectura segura de columnas (NaN -> None)
# ─────────────────────────────────────────────────────────────

def _v(row: pd.Series, col: str):
    """Retorna el valor de la columna o None si no existe o es NaN."""
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    return val


# ─────────────────────────────────────────────────────────────
# Evaluadores de Entrada
# ─────────────────────────────────────────────────────────────

def check_entrada_pa(row: pd.Series, estrategia: str) -> bool:
    """
    Evalua si la fila cumple la condicion de entrada PA.
    Retorna False si alguna feature requerida es None/NaN.

    Args:
        row:        fila del DataFrame combinado (OHLCV + indicadores + PA + MS)
        estrategia: "EV1" | "EV2" | "EV3" | "EV4"

    Returns:
        True si se genera senal de entrada LONG
    """
    if estrategia == "EV1":
        est10    = _v(row, "estructura_10")
        dias_sl  = _v(row, "dias_sl_10")
        eng_bull = _v(row, "patron_engulfing_bull")
        hammer   = _v(row, "patron_hammer")
        v_spike  = _v(row, "vol_spike")
        up_vol   = _v(row, "up_vol_5d")
        if any(x is None for x in [est10, dias_sl, eng_bull, hammer, v_spike, up_vol]):
            return False
        patron_ok = (int(eng_bull) == 1 or int(hammer) == 1)
        vol_ok    = (int(v_spike) == 1 or float(up_vol) > 0.55)
        return int(est10) == 1 and int(dias_sl) <= 15 and patron_ok and vol_ok

    elif estrategia == "EV2":
        bos   = _v(row, "bos_bull_10")
        vpc   = _v(row, "vol_price_confirm")
        est10 = _v(row, "estructura_10")
        if any(x is None for x in [bos, vpc, est10]):
            return False
        return int(bos) == 1 and int(vpc) == 1 and int(est10) >= 0

    elif estrategia == "EV3":
        choch   = _v(row, "choch_bull_10")
        v_spike = _v(row, "vol_spike")
        dsl     = _v(row, "dist_sl_10_pct")
        if any(x is None for x in [choch, v_spike, dsl]):
            return False
        return int(choch) == 1 and int(v_spike) == 1 and float(dsl) > -4.0

    elif estrategia == "EV4":
        est10 = _v(row, "estructura_10")
        dsl   = _v(row, "dist_sl_10_pct")
        alc   = _v(row, "es_alcista")
        dias  = _v(row, "dias_sl_10")
        if any(x is None for x in [est10, dsl, alc, dias]):
            return False
        dsl_f = float(dsl)
        return int(est10) == 1 and 0.0 <= dsl_f <= 4.0 and int(alc) == 1 and int(dias) <= 10

    return False


# ─────────────────────────────────────────────────────────────
# Inicializacion de stops al entrar en posicion
# ─────────────────────────────────────────────────────────────

def calcular_stops_iniciales_pa(precio_entrada: float, atr: float,
                                 estrategia: str) -> tuple:
    """
    Calcula SL y TP iniciales para estrategias PA.

    SV1, SV2, SV3: SL de seguridad -5%, TP estructural (None = cierre por condicion)
    SV4:           SL = entry - 1.5*ATR, TP = entry + 2.5*ATR

    Returns:
        (stop_loss, take_profit)  -- take_profit puede ser None
    """
    p = PARAMS_SALIDA_PA[estrategia]

    if estrategia == "SV4":
        sl = precio_entrada - p["sl_atr"] * atr
        tp = precio_entrada + p["tp_atr"] * atr
    else:
        sl = precio_entrada * (1 - p["safety_sl_pct"])
        tp = None

    return round(sl, 4), (round(tp, 4) if tp is not None else None)


# ─────────────────────────────────────────────────────────────
# Evaluadores de Salida
# ─────────────────────────────────────────────────────────────

def check_salida_pa(row: pd.Series, estrategia: str, stop_loss: float,
                    take_profit, dias_en_posicion: int) -> tuple:
    """
    Evalua si la posicion PA debe cerrarse en la sesion actual.

    Orden de prioridad:
        1. Stop Loss (intraday: low <= stop_loss)
        2. Take Profit fijo (SV4: intraday high >= tp)
        3. Condicion estructural especifica de la estrategia (al close)
        4. Timeout (al close)

    Args:
        row:              fila del DataFrame combinado
        estrategia:       "SV1" | "SV2" | "SV3" | "SV4"
        stop_loss:        precio de stop loss actual
        take_profit:      precio objetivo (None para SV1/SV2/SV3)
        dias_en_posicion: dias que lleva abierta la operacion

    Returns:
        (cerrar: bool, motivo: str, precio_salida: float)
        Si no cierra: (False, None, None)
    """
    p     = PARAMS_SALIDA_PA[estrategia]
    low   = float(row["low"])
    high  = float(row["high"])
    close = float(row["close"])

    # 1. Stop Loss de seguridad (todos los tipos)
    if low <= stop_loss:
        return True, "STOP_LOSS", round(stop_loss, 4)

    # 2. Take Profit fijo (solo SV4)
    if take_profit is not None and high >= take_profit:
        return True, "TAKE_PROFIT", round(take_profit, 4)

    # 3. Condicion estructural especifica
    if estrategia == "SV1":
        # Cierre por proximidad al swing high: dist_sh_10_pct >= -1.0%
        dsh = _v(row, "dist_sh_10_pct")
        if dsh is not None and float(dsh) >= -1.0:
            return True, "TARGET", round(close, 4)

    elif estrategia == "SV2":
        # Cierre por rotura bajista: BOS o CHoCH bearish en ventana 10
        bos_b   = _v(row, "bos_bear_10")
        choch_b = _v(row, "choch_bear_10")
        if (bos_b is not None and int(bos_b) == 1) or \
           (choch_b is not None and int(choch_b) == 1):
            return True, "ESTRUCTURA", round(close, 4)

    elif estrategia == "SV3":
        # Cierre por deterioro de score o estructura bajista
        score = _v(row, "score_ponderado")
        est10 = _v(row, "estructura_10")
        senal_score = (score is not None and float(score) < 0.30)
        senal_est   = (est10 is not None and int(est10) == -1)
        if senal_score or senal_est:
            return True, "SENAL", round(close, 4)

    # 4. Timeout
    if dias_en_posicion >= p["timeout"]:
        return True, "TIMEOUT", round(close, 4)

    return False, None, None


# ─────────────────────────────────────────────────────────────
# Clasificacion del resultado
# ─────────────────────────────────────────────────────────────

def clasificar_resultado_pa(retorno_pct: float) -> str:
    """Clasifica la operacion PA segun el retorno (mismos umbrales que el original)."""
    umbral = UMBRAL_NEUTRO * 100
    if retorno_pct > umbral:
        return "GANANCIA"
    elif retorno_pct < -umbral:
        return "PERDIDA"
    return "NEUTRO"
