"""
clasificacion_tecnica.py
Clasificacion interpretativa de osciladores/indicadores en estados legibles.

Funciones PURAS (sin DB, sin side effects): reciben valores crudos y devuelven
un estado en texto. Pensadas para alimentar el tablero de sintesis (tool MCP)
sin volcar numeros crudos al LLM.

Umbrales homogeneos con config.py (RSI_OVERSOLD=35 / RSI_OVERBOUGHT=65).
Se replican aca como constantes locales en lugar de importar config porque
este modulo debe ser AUTONOMO y puro: el MCP server lo consume y tiene
prohibido importar config.py (que ejecuta load_dotenv() al importarse).
Si los umbrales cambian en config.py, actualizar tambien aca.

    RSI < 35 = Sobreventa  |  35-65 = Neutral  |  RSI > 65 = Sobrecompra

MACD (convencion estandar acordada):
    linea MACD > signal (hist > 0)  -> Compra
    linea MACD < signal (hist < 0)  -> Venta

ADX (fuerza de tendencia, sin direccion):
    ADX >= 25 = Fuerte  |  20-25 = Moderada  |  ADX < 20 = Debil

Tendencia SMA (alineacion de medias + posicion del close):
    close por encima de las medias y SMA21>=SMA50>=SMA200 = Alcista
    close por debajo y SMA21<=SMA50<=SMA200               = Bajista
    cualquier mezcla                                       = Lateral

Estructura SMC (ventana estrategica 10):
    CHoCH alcista/bajista (giro reciente) tiene prioridad sobre la estructura.
    Sin CHoCH: vota la estructura_10 (1=Alcista, -1=Bajista, 0=Neutral).
"""

from typing import Optional

# Homogeneo con config.RSI_OVERSOLD / config.RSI_OVERBOUGHT (ver docstring)
RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65

# Umbrales ADX (estandar de Wilder)
ADX_DEBIL  = 20
ADX_FUERTE = 25


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


def clasificar_adx(adx: Optional[float]) -> Optional[str]:
    """
    Clasifica la FUERZA de la tendencia segun ADX (sin direccion).
    Fuerte (>=25) / Moderada (20-25) / Debil (<20). None si adx es None.
    """
    if adx is None:
        return None
    try:
        a = float(adx)
    except (TypeError, ValueError):
        return None
    if a >= ADX_FUERTE:
        return "Fuerte"
    if a < ADX_DEBIL:
        return "Debil"
    return "Moderada"


def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def clasificar_tendencia_sma(close, sma21, sma50, sma200) -> Optional[str]:
    """
    Clasifica la tendencia por alineacion de medias + posicion del close.

    Alcista : close por encima de TODAS las medias disponibles y
              alineacion ascendente (SMA21 >= SMA50 >= SMA200).
    Bajista : close por debajo de TODAS y alineacion descendente.
    Lateral : cualquier otra combinacion.

    Las medias None se ignoran (ej. SMA200 sin 200 dias de historia). Si no
    hay ninguna media o falta el close, retorna None.
    """
    c = _to_float(close)
    if c is None:
        return None
    pares = [(21, _to_float(sma21)), (50, _to_float(sma50)), (200, _to_float(sma200))]
    avail = [(p, v) for p, v in pares if v is not None]
    if not avail:
        return None

    smas = [v for _, v in sorted(avail)]  # por periodo asc: [s21, s50, s200]
    asc  = all(smas[i] >= smas[i + 1] for i in range(len(smas) - 1))
    desc = all(smas[i] <= smas[i + 1] for i in range(len(smas) - 1))
    n_above = sum(1 for _, v in avail if c > v)

    if n_above == len(avail) and asc:
        return "Alcista"
    if n_above == 0 and desc:
        return "Bajista"
    return "Lateral"


def _flag(val) -> bool:
    """
    Normaliza una columna flag SMC que puede venir como smallint(0/1),
    boolean, None o str. True solo si es un 1/True real.
    """
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    try:
        return int(val) != 0
    except (TypeError, ValueError):
        return False


def clasificar_estructura_smc(estructura, choch_bull, choch_bear,
                              bos_bull, bos_bear) -> dict:
    """
    Voto y etiqueta de la estructura de mercado (SMC, ventana 10).

    El CHoCH (cambio de caracter = giro) tiene PRIORIDAD: anticipa el cambio
    de sesgo aunque la estructura aun no haya virado. Sin CHoCH, vota la
    estructura (1=Alcista, -1=Bajista, 0/None=Neutral). El BOS refuerza la
    etiqueta pero no cambia el voto (ya implicito en la estructura).

    Args:
        estructura: estructura_10 (1 / 0 / -1) o None.
        choch_bull, choch_bear, bos_bull, bos_bear: flags (0/1, bool o None).

    Returns:
        {"voto": "Alcista"|"Bajista"|"Neutral", "etiqueta": str}
    """
    cb, cs = _flag(choch_bull), _flag(choch_bear)
    bb, bs = _flag(bos_bull), _flag(bos_bear)
    est = None
    try:
        est = int(estructura) if estructura is not None else None
    except (TypeError, ValueError):
        est = None

    # CHoCH (giro reciente) manda. Conflicto (ambos) -> sin voto claro.
    if cb and not cs:
        return {"voto": "Alcista", "etiqueta": "CHoCH alcista"}
    if cs and not cb:
        return {"voto": "Bajista", "etiqueta": "CHoCH bajista"}

    if est == 1:
        return {"voto": "Alcista", "etiqueta": "BOS alcista" if bb else "Estructura alcista"}
    if est == -1:
        return {"voto": "Bajista", "etiqueta": "BOS bajista" if bs else "Estructura bajista"}
    return {"voto": "Neutral", "etiqueta": "Lateral"}
