"""
ft_scoring.py
Logica de scoring y queries a Railway DB para el motor de Forward Testing.

Este modulo es COMPLETAMENTE AUTONOMO: sin dependencias de src.trading,
sin alpaca, sin bots en vivo. Solo Railway DB + logica pura.

Funciones disponibles:

  Scoring tecnico (FT_TECH_v1 / FT_TECH_SECTOR_v1):
    calcular_score_tecnico(row)   -> (score, detalle)
    obtener_indicadores_hoy()     -> list[dict]

  Scoring estructural SMC (FT_SMC_v1):
    _swing_low_precio(close, dist_sl_pct)       -> float
    calcular_score_estructura(row)               -> (score, detalle)
    obtener_features_hoy()                       -> list[dict]
    obtener_estructura_tickers(tickers)          -> dict[str, dict]
    calcular_actualizaciones_sl(posiciones)      -> list[dict]
"""

from sqlalchemy import text
from src.data.database import get_engine


# ═══════════════════════════════════════════════════════════════
# SCORING TECNICO  (SMA / MACD / RSI)
# Replica la logica de src/trading/strategy_technical.py
# ═══════════════════════════════════════════════════════════════

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


def obtener_indicadores_hoy() -> list:
    """
    Lee el ultimo registro de indicadores tecnicos disponibles
    para todos los tickers (JOIN con precios_diarios para el close).
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (i.ticker)
                i.ticker,
                i.fecha,
                p.close,
                i.sma21,
                i.sma50,
                i.sma200,
                i.rsi14,
                i.atr14,
                i.macd,
                i.macd_signal,
                (i.macd - i.macd_signal) AS macd_hist
            FROM indicadores_tecnicos i
            JOIN precios_diarios p
              ON p.ticker = i.ticker AND p.fecha = i.fecha
            ORDER BY i.ticker, i.fecha DESC
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


# ═══════════════════════════════════════════════════════════════
# SCORING ESTRUCTURAL SMC  (CHoCH / BOS / Trailing SL)
# Replica la logica de src/trading/strategy_structure.py
# ═══════════════════════════════════════════════════════════════

# Limites del SL estructural
MIN_SL_DISTANCE_PCT = 1.0   # % minimo de distancia SL desde close
MAX_SL_DISTANCE_PCT = 8.0   # % maximo de distancia SL desde close
LOOKBACK_DIAS       = 12    # dias calendario para buscar CHoCH/BOS (~10 habiles)
SCORE_MAXIMO_SMC    = 3


def _swing_low_precio(close: float, dist_sl_pct: float) -> float:
    """
    Calcula el precio absoluto del ultimo swing low estructural.
    Formula: swing_low = close / (1 + dist_sl_pct / 100)
    """
    if dist_sl_pct <= 0 or close <= 0:
        return 0.0
    return round(close / (1.0 + dist_sl_pct / 100.0), 4)


def calcular_score_estructura(row: dict) -> tuple:
    """
    Evalua condiciones de entrada estructural para un ticker.

    Retorna (score, detalle):
        score = -1  si no cumple alguna condicion obligatoria
        score = 0-3 si cumple todas (para rankear calidad del setup)

    Condiciones OBLIGATORIAS (todas deben cumplirse):
        1. CHoCH bull O BOS bull en lookback
        2. estructura_10 >= 0  (no rota al baja)
        3. choch_bear_10 = 0   (sin cambio bajista hoy)
        4. es_alcista = 1      (vela alcista hoy)
        5. dist_sl en rango    (1% - 8%)

    Score de calidad (0-3):
        +1 CHoCH bull detectado
        +1 confirmacion vela/vol (vol_spike o eng_bull o hammer)
        +1 estructura solida (estructura_10 = +1)
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

    swing_low = _swing_low_precio(close, dist_sl_pct) if close else 0.0

    detalle = {
        "tuvo_choch_bull":    bool(tuvo_choch_bull),
        "tuvo_bos_bull":      bool(tuvo_bos_bull),
        "estructura_10":      estructura_10,
        "choch_bear_10":      bool(choch_bear_10),
        "dist_sl_pct":        round(dist_sl_pct, 2),
        "dist_sh_pct":        round(dist_sh_pct, 2),
        "swing_low":          swing_low,
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

    # Filtros obligatorios
    if not detalle["cond_evento"]:        return -1.0, detalle
    if not detalle["cond_estructura"]:    return -1.0, detalle
    if not detalle["cond_no_choch_bear"]: return -1.0, detalle
    if not detalle["cond_vela"]:          return -1.0, detalle
    if not detalle["cond_sl_rango"]:      return -1.0, detalle

    # Score de calidad para ranking
    score = 0.0
    if tuvo_choch_bull:                 score += 1.0
    if vol_spike or eng_bull or hammer: score += 1.0
    if estructura_10 == 1:              score += 1.0

    return round(score, 2), detalle


def obtener_features_hoy() -> list:
    """
    Carga features estructurales del dia mas reciente para todos los tickers
    que tuvieron CHoCH o BOS bull en el lookback definido.
    Une con features_precio_accion, indicadores_tecnicos y precios_diarios.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            WITH eventos AS (
                SELECT
                    ticker,
                    MAX(CASE WHEN choch_bull_10 = 1 THEN 1 ELSE 0 END) AS tuvo_choch_bull,
                    MAX(CASE WHEN bos_bull_10   = 1 THEN 1 ELSE 0 END) AS tuvo_bos_bull
                FROM features_market_structure
                WHERE fecha >= CURRENT_DATE - INTERVAL '{LOOKBACK_DIAS} days'
                GROUP BY ticker
                HAVING MAX(CASE WHEN choch_bull_10 = 1 THEN 1 ELSE 0 END) = 1
                    OR MAX(CASE WHEN bos_bull_10   = 1 THEN 1 ELSE 0 END) = 1
            ),
            estado_actual AS (
                SELECT DISTINCT ON (ms.ticker)
                    ms.ticker,
                    ms.fecha,
                    ms.estructura_10,
                    ms.choch_bull_10,
                    ms.choch_bear_10,
                    ms.bos_bull_10,
                    ms.dist_sh_10_pct,
                    ms.dist_sl_10_pct,
                    ms.dias_sh_10,
                    ms.dias_sl_10
                FROM features_market_structure ms
                ORDER BY ms.ticker, ms.fecha DESC
            )
            SELECT
                ea.*,
                ev.tuvo_choch_bull,
                ev.tuvo_bos_bull,
                pa.es_alcista,
                pa.patron_engulfing_bull,
                pa.patron_hammer,
                pa.vol_spike,
                pa.up_vol_5d,
                i.rsi14,
                i.atr14,
                i.sma200,
                p.close
            FROM estado_actual ea
            JOIN eventos ev
              ON ev.ticker = ea.ticker
            JOIN features_precio_accion pa
              ON pa.ticker = ea.ticker AND pa.fecha = ea.fecha
            JOIN indicadores_tecnicos i
              ON i.ticker = ea.ticker AND i.fecha = ea.fecha
            JOIN precios_diarios p
              ON p.ticker = ea.ticker AND p.fecha = ea.fecha
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_estructura_tickers(tickers: list) -> dict:
    """
    Retorna el ultimo dia de features_market_structure + precio
    para los tickers dados (usada para gestionar posiciones abiertas).
    """
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (ms.ticker)
                ms.ticker,
                ms.fecha,
                ms.estructura_10,
                ms.choch_bear_10,
                ms.bos_bear_10,
                ms.dist_sl_10_pct,
                ms.dist_sh_10_pct,
                p.close
            FROM features_market_structure ms
            JOIN precios_diarios p
              ON p.ticker = ms.ticker AND p.fecha = ms.fecha
            WHERE ms.ticker = ANY(:tickers)
            ORDER BY ms.ticker, ms.fecha DESC
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


def calcular_actualizaciones_sl(posiciones: list) -> list:
    """
    Recalcula el trailing SL estructural para cada posicion abierta.

    Filosofia: el SL solo sube, nunca baja.
    Formula: nuevo_sl = close / (1 + dist_sl_10_pct / 100)

    Retorna lista de dicts con:
        id, ticker, nuevo_sl, sl_anterior, delta_pct
    Solo incluye posiciones donde nuevo_sl > sl_actual.
    """
    if not posiciones:
        return []

    tickers   = [p["ticker"] for p in posiciones]
    datos_map = obtener_estructura_tickers(tickers)
    updates   = []

    for pos in posiciones:
        ticker    = pos["ticker"]
        datos_hoy = datos_map.get(ticker)
        if not datos_hoy:
            continue

        sl_actual = float(pos.get("stop_loss") or 0)
        if sl_actual <= 0:
            continue

        close       = float(datos_hoy.get("close", 0) or 0)
        dist_sl_pct = float(datos_hoy.get("dist_sl_10_pct", 0) or 0)

        if close <= 0 or dist_sl_pct <= 0:
            continue

        nuevo_sl = _swing_low_precio(close, dist_sl_pct)

        # Solo actualizamos si el nuevo SL es mas alto (proteccion sube, nunca baja)
        if nuevo_sl <= sl_actual:
            continue

        delta_pct = round((nuevo_sl - sl_actual) / sl_actual * 100, 2)
        updates.append({
            "id":          pos["id"],
            "ticker":      ticker,
            "nuevo_sl":    nuevo_sl,
            "sl_anterior": sl_actual,
            "delta_pct":   delta_pct,
        })

    return updates


# ═══════════════════════════════════════════════════════════════
# CANDLE SCORE  (ventana 5 dias de trading)
# Scoring de velas acumulado — se usa como desempate secundario
# despues del score tecnico primario.
# ═══════════════════════════════════════════════════════════════

def obtener_candle_score_5d() -> dict:
    """
    Calcula el score de velas acumulado de los ultimos 5 dias de trading
    para todos los tickers en una sola query (un viaje a Railway DB).

    Logica de scoring por dia:
      Base direccional:
        es_alcista = 1   ->  +0.5
        es_alcista = 0   ->  -0.5

      Patron bonus (se suma al base, sin solapamiento):
        patron_engulfing_bull    ->  +1.5   (absorbe oferta previa)
        patron_engulfing_bear    ->  -1.5   (absorbe demanda previa)
        patron_hammer            ->  +1.0   (rechazo de zona baja)
        patron_shooting_star     ->  -1.0   (rechazo de zona alta)
        patron_marubozu alcista  ->  +1.0   (fuerza total compradora)
        patron_marubozu bajista  ->  -0.5   (fuerza vendedora)

      Volumen:
        vol_price_confirm        ->  +1.0   (volumen confirma movimiento)
        vol_price_diverge        ->  -0.5   (volumen contradice movimiento)
        vol_spike solo           ->  +0.5   (atencion elevada sin direccion)

    Estructura de mercado (se agrega una sola vez, del ultimo dia):
        bos_bull_5   ->  +2.0   (break estructural alcista reciente)
        choch_bull_5 ->  +1.5   (cambio de caracter alcista)
        bos_bear_5   ->  -2.0   (break estructural bajista reciente)
        choch_bear_5 ->  -3.0   (cambio de caracter bajista — senhal fuerte)

    Retorna dict {ticker: candle_score_5d}
    Rango tipico: -12 a +17 (score 0 = neutral, positivo = momentum alcista)
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            WITH dias_recientes AS (
                -- Los 5 dias de trading mas recientes con datos
                SELECT DISTINCT fecha
                FROM features_precio_accion
                ORDER BY fecha DESC
                LIMIT 5
            ),
            candle_sum AS (
                SELECT
                    pa.ticker,
                    COUNT(*)                                                AS dias_analizados,
                    SUM(
                        -- Base direccional
                        CASE WHEN COALESCE(pa.es_alcista, 0) = 1
                             THEN  0.5 ELSE -0.5 END

                        -- Patron bonus alcista
                        + COALESCE(pa.patron_engulfing_bull,  0) *  1.5
                        + COALESCE(pa.patron_hammer,          0) *  1.0
                        + CASE WHEN COALESCE(pa.patron_marubozu, 0) = 1
                                    AND COALESCE(pa.es_alcista,   0) = 1
                               THEN  1.0 ELSE 0 END

                        -- Patron bonus bajista
                        + COALESCE(pa.patron_engulfing_bear,  0) * -1.5
                        + COALESCE(pa.patron_shooting_star,   0) * -1.0
                        + CASE WHEN COALESCE(pa.patron_marubozu, 0) = 1
                                    AND COALESCE(pa.es_alcista,   0) = 0
                               THEN -0.5 ELSE 0 END

                        -- Volumen
                        + COALESCE(pa.vol_price_confirm, 0) *  1.0
                        + COALESCE(pa.vol_price_diverge, 0) * -0.5
                        + CASE WHEN COALESCE(pa.vol_spike,         0) = 1
                                    AND COALESCE(pa.vol_price_confirm, 0) = 0
                                    AND COALESCE(pa.vol_price_diverge, 0) = 0
                               THEN 0.5 ELSE 0 END
                    )                                                       AS base_score
                FROM features_precio_accion pa
                JOIN dias_recientes dr ON dr.fecha = pa.fecha
                GROUP BY pa.ticker
            ),
            estructura AS (
                -- Ultima fila de market structure por ticker (para senales _5)
                SELECT DISTINCT ON (ms.ticker)
                    ms.ticker,
                    COALESCE(ms.bos_bull_5,   0) *  2.0
                    + COALESCE(ms.choch_bull_5, 0) *  1.5
                    + COALESCE(ms.bos_bear_5,   0) * -2.0
                    + COALESCE(ms.choch_bear_5, 0) * -3.0  AS struct_score
                FROM features_market_structure ms
                ORDER BY ms.ticker, ms.fecha DESC
            )
            SELECT
                cs.ticker,
                ROUND(
                    CAST(cs.base_score + COALESCE(ea.struct_score, 0) AS NUMERIC),
                    2
                )                   AS candle_score_5d,
                cs.dias_analizados
            FROM candle_sum cs
            LEFT JOIN estructura ea ON ea.ticker = cs.ticker
        """)).fetchall()

    return {r.ticker: float(r.candle_score_5d) for r in rows}
