"""
bt_scoring.py
Funciones de scoring puras para el motor de backtesting historico.

Adaptacion de ft_scoring.py: sin queries a DB, sin dependencias de Railway.
Trabaja sobre los dicts que BtDataLoader expone por fecha.

Funciones publicas:
    calcular_score_tecnico(row)     -> (score, detalle)
    calcular_score_estructura(row)  -> (score, detalle)
    swing_low_precio(close, dist)   -> float
    SCORE_ENTRADA_TECH              (4.0)
    SCORE_SALIDA_TECH               (3.5)
    SCORE_ENTRADA_MIN_SMC           (1)
    MIN_SL_DISTANCE_PCT             (1.0)
    MAX_SL_DISTANCE_PCT             (8.0)
"""

# ── Constantes TECH_SECTOR_v1 / COMBO_v1 ──────────────────────────────────────
PTS_SMA50    = 2.0
PTS_SMA21    = 1.0
PTS_MACD     = 1.5
PTS_RSI      = 1.0
RSI_MIN      = 45.0
RSI_MAX      = 68.0

SCORE_ENTRADA_TECH = 4.0
SCORE_SALIDA_TECH  = 3.5
ATR_MULT_SL        = 2.0
ATR_MULT_TP        = 4.0

SECTOR_BUDGET      = 11_111.0
MAX_POS_SECTOR     = 5
POSITION_PCT       = 0.20    # 20% del budget de sector

CANDLE_SCORE_EXCLUIR_COMBO = -3.0  # COMBO: skip si candle_score_5d <= este valor

# ── Constantes SMC_v1 ─────────────────────────────────────────────────────────
MIN_SL_DISTANCE_PCT  = 1.0
MAX_SL_DISTANCE_PCT  = 8.0
LOOKBACK_DIAS        = 12

SCORE_ENTRADA_MIN_SMC = 1     # score >= 1 para entrar en SMC
MAX_POSICIONES_SMC    = 5
CAPITAL_INICIAL       = 100_000.0
MAX_DEPLOY_PCT        = 0.80  # no deployar mas del 80% del capital
RIESGO_POR_TRADE_SMC  = 0.15  # 15% del capital disponible por trade SMC
DIAS_MAX_SMC          = 20


# ── Scoring tecnico ────────────────────────────────────────────────────────────

def calcular_score_tecnico(row: dict) -> tuple:
    """
    Score tecnico (3 capas, max 5.5 pts). Identico a ft_scoring.calcular_score_tecnico.
    Capa 1 (obligatoria): close > SMA200 -> si falla, score = 0
    Capa 2 (tendencia)  : close > SMA50 (+2.0), close > SMA21 (+1.0)
    Capa 3 (momentum)   : MACD > Signal y hist>0 (+1.5), RSI 45-68 (+1.0)
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

    if not detalle["filtro_sma200"]:
        return 0.0, detalle

    score = 0.0
    if detalle["cond_sma50"]: score += PTS_SMA50
    if detalle["cond_sma21"]: score += PTS_SMA21
    if detalle["cond_macd"]:  score += PTS_MACD
    if detalle["cond_rsi"]:   score += PTS_RSI

    return round(score, 2), detalle


# ── Scoring estructural SMC ────────────────────────────────────────────────────

def swing_low_precio(close: float, dist_sl_pct: float) -> float:
    """Precio absoluto del swing low estructural. SL solo puede subir."""
    if dist_sl_pct <= 0 or close <= 0:
        return 0.0
    return round(close / (1.0 + dist_sl_pct / 100.0), 4)


def calcular_score_estructura(row: dict) -> tuple:
    """
    Evalua condiciones de entrada estructural SMC. Identico a ft_scoring.calcular_score_estructura.
    score = -1 si no cumple condiciones obligatorias
    score = 0-3 para ranking de calidad
    """
    tuvo_choch_bull = int(row.get("tuvo_choch_bull", 0) or 0)
    tuvo_bos_bull   = int(row.get("tuvo_bos_bull",   0) or 0)
    estructura_10   = int(row.get("estructura_10",   0) or 0)
    choch_bear_10   = int(row.get("choch_bear_10",   0) or 0)
    dist_sl_pct     = float(row.get("dist_sl_10_pct", 0) or 0)
    dist_sh_pct     = float(row.get("dist_sh_10_pct", 0) or 0)
    es_alcista      = int(row.get("es_alcista",      0) or 0)
    vol_spike       = int(row.get("vol_spike",       0) or 0)
    eng_bull        = int(row.get("patron_engulfing_bull", 0) or 0)
    hammer          = int(row.get("patron_hammer",   0) or 0)
    close           = float(row.get("close",         0) or 0)

    sl_precio = swing_low_precio(close, dist_sl_pct) if close else 0.0

    detalle = {
        "tuvo_choch_bull":    bool(tuvo_choch_bull),
        "tuvo_bos_bull":      bool(tuvo_bos_bull),
        "estructura_10":      estructura_10,
        "choch_bear_10":      bool(choch_bear_10),
        "dist_sl_pct":        round(dist_sl_pct, 2),
        "dist_sh_pct":        round(dist_sh_pct, 2),
        "swing_low":          sl_precio,
        "es_alcista":         bool(es_alcista),
        "vol_spike":          bool(vol_spike),
        "eng_bull":           bool(eng_bull),
        "hammer":             bool(hammer),
        "cond_evento":        bool(tuvo_choch_bull or tuvo_bos_bull),
        "cond_estructura":    estructura_10 >= 0,
        "cond_no_choch_bear": not bool(choch_bear_10),
        "cond_vela":          bool(es_alcista),
        "cond_sl_rango":      MIN_SL_DISTANCE_PCT <= dist_sl_pct <= MAX_SL_DISTANCE_PCT,
    }

    if not detalle["cond_evento"]:        return -1.0, detalle
    if not detalle["cond_estructura"]:    return -1.0, detalle
    if not detalle["cond_no_choch_bear"]: return -1.0, detalle
    if not detalle["cond_vela"]:          return -1.0, detalle
    if not detalle["cond_sl_rango"]:      return -1.0, detalle

    score = 0.0
    if tuvo_choch_bull:                 score += 1.0
    if vol_spike or eng_bull or hammer: score += 1.0
    if estructura_10 == 1:              score += 1.0

    return round(score, 2), detalle
