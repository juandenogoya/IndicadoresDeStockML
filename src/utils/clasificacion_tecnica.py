"""
clasificacion_tecnica.py
Clasificacion interpretativa de osciladores/indicadores en estados legibles.

Funciones PURAS (sin DB, sin side effects): reciben valores crudos y devuelven
un estado en texto. Pensadas para alimentar el tablero de sintesis (tool MCP)
sin volcar numeros crudos al LLM.

Umbrales homogeneos con config.py (NO hardcodear otros):
    RSI_OVERSOLD  = 35   -> RSI < 35 = Sobreventa
    RSI_OVERBOUGHT = 65  -> RSI > 65 = Sobrecompra
    entre 35 y 65        -> Neutral

MACD (convencion estandar acordada):
    linea MACD > signal (hist > 0)  -> Compra
    linea MACD < signal (hist < 0)  -> Venta
"""

from typing import Optional

from src.utils.config import RSI_OVERSOLD, RSI_OVERBOUGHT


def clasificar_rsi(rsi: Optional[float]) -> Optional[str]:
    """
    Clasifica el RSI en Sobreventa / Neutral / Sobrecompra segun los umbrales
    de config (35 / 65). Retorna None si rsi es None.
    """
    if rsi is None:
        return None
    try:
        r = float(rsi)
    except (TypeError, ValueError):
        return None
    if r < RSI_OVERSOLD:
        return "Sobreventa"
    if r > RSI_OVERBOUGHT:
        return "Sobrecompra"
    return "Neutral"


def clasificar_macd(macd: Optional[float], signal: Optional[float]) -> Optional[str]:
    """
    Clasifica el MACD en Compra / Venta segun su posicion respecto a la linea
    de senal. Retorna None si falta algun valor.

    Compra : linea MACD > signal  (histograma positivo)
    Venta  : linea MACD < signal  (histograma negativo)
    Neutral: macd == signal (cruce exacto, raro)
    """
    if macd is None or signal is None:
        return None
    try:
        m = float(macd)
        s = float(signal)
    except (TypeError, ValueError):
        return None
    if m > s:
        return "Compra"
    if m < s:
        return "Venta"
    return "Neutral"


def clasificar_timeframe(rsi, macd, signal) -> dict:
    """
    Clasifica un timeframe completo (diario o semanal). Devuelve dict con
    los estados de RSI y MACD listos para el tablero.
    """
    return {
        "rsi": clasificar_rsi(rsi),
        "macd": clasificar_macd(macd, signal),
    }
