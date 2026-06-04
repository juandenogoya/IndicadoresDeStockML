"""
scoring.py -- Scoring tecnico rule-based (SMA / MACD / RSI).

Logica COMPARTIDA FT <-> Alpaca (Plan B, Tarea 16). Modulo PURO: sin DB,
sin side effects. Replica la logica de src/trading/strategy_technical.py.

Movido desde scripts/forward_testing/ft_scoring.py (1/6/2026) para invertir la
dependencia: ahora vive en src/ y scripts/ lo importa (scripts -> src, nunca al
reves). ft_scoring.py lo re-exporta para retrocompatibilidad de sus importadores.

Contrato de entrada (dict `row`), claves que calzan con senales_bot_diaria:
    close, sma21, sma50, sma200, rsi14, macd, macd_signal
"""

# Puntos por condicion
PTS_SMA50    = 2.0
PTS_SMA21    = 1.0
PTS_MACD     = 1.5
PTS_RSI      = 1.0
SCORE_MAXIMO_TECH = PTS_SMA50 + PTS_SMA21 + PTS_MACD + PTS_RSI  # 5.5

# Filtros RSI (sin sobrecompra)
RSI_MIN = 45.0
RSI_MAX = 68.0


def calcular_score_tecnico(row: dict) -> tuple:
    """
    Calcula el score tecnico (3 capas, max 5.5 pts).

    Capa 1 (obligatoria): close > SMA200 -> si falla, score = 0
    Capa 2 (tendencia)  : close > SMA50 (+2.0), close > SMA21 (+1.0)
    Capa 3 (momentum)   : MACD > Signal y hist>0 (+1.5), RSI 45-68 (+1.0)

    Retorna (score: float, detalle: dict)
    """
    close  = float(row.get("close",       0) or 0)
    sma21  = float(row.get("sma21",       0) or 0)
    sma50  = float(row.get("sma50",       0) or 0)
    sma200 = float(row.get("sma200",      0) or 0)
    rsi    = float(row.get("rsi14",       0) or 0)
    macd   = float(row.get("macd",        0) or 0)
    signal = float(row.get("macd_signal", 0) or 0)
    hist   = macd - signal

    detalle = {
        "filtro_sma200": close > sma200,
        "cond_sma50":    close > sma50,
        "cond_sma21":    close > sma21,
        "cond_macd":     macd > signal and hist > 0,
        "cond_rsi":      RSI_MIN <= rsi <= RSI_MAX,
        "rsi":           round(rsi,  2),
        "macd_hist":     round(hist, 4),
        "close":         close,
        "sma50":         round(sma50,  2),
        "sma200":        round(sma200, 2),
    }

    # Capa 1: filtro obligatorio
    if not detalle["filtro_sma200"]:
        return 0.0, detalle

    # Capas 2 y 3
    score = 0.0
    if detalle["cond_sma50"]: score += PTS_SMA50
    if detalle["cond_sma21"]: score += PTS_SMA21
    if detalle["cond_macd"]:  score += PTS_MACD
    if detalle["cond_rsi"]:   score += PTS_RSI

    return round(score, 2), detalle
