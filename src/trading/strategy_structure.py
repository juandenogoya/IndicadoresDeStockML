"""
strategy_structure.py
Estrategia SMC (Smart Money Concepts) basada en estructura de mercado.

Entrada: CHoCH/BOS alcista en ultimos ~10 dias habiles + estructura sostenida
         + vela de confirmacion alcista hoy.

Condiciones OBLIGATORIAS para entrar (todas deben cumplirse):
    1. Evento alcista en lookback (~10 dias habiles):
         choch_bull_10=1  O  bos_bull_10=1
    2. Estructura vigente hoy: estructura_10 >= 0 (no rota al baja)
    3. NO hay CHoCH bajista hoy: choch_bear_10 = 0
    4. Vela alcista hoy: es_alcista = 1 (close > open)
    5. SL estructural valido: MIN_SL_DIST <= dist_sl_10_pct <= MAX_SL_DIST (1%-8%)

Scoring para RANKEAR candidatos (0-3 pts):
    +1  CHoCH bull detectado (cambio de caracter — señal mas fuerte que BOS)
    +1  Confirmacion vela/vol hoy: vol_spike=1 O eng_bull=1 O hammer=1
    +1  Estructura solida hoy: estructura_10 = +1 (tendencia HH/HL confirmada)

Stop Loss Trailing (Filosofia B):
    SL_precio = close / (1 + dist_sl_10_pct / 100)   [ultimo swing low estructural]
    Se actualiza SOLO al alza: si nuevo_sl > sl_actual → actualizar (nunca bajar)

Salidas (orden de prioridad):
    1. SL trailing roto  : precio_actual <= stop_loss_actual
    2. CHoCH bajista hoy : choch_bear_10 = 1
    3. Estructura rota   : estructura_10 = -1
    4. Time stop         : >= DIAS_MAX_POS dias calendario en posicion
"""

from datetime import date
from sqlalchemy import text
from src.data.database import get_engine
from src.trading import alpaca_client, risk

# ── Parametros ────────────────────────────────────────────────
SCORE_ENTRADA       = 1      # score minimo (0-3) para abrir posicion
SCORE_MAXIMO        = 3      # score maximo posible

MAX_SL_DISTANCE_PCT = 8.0    # % maximo de distancia SL desde close
MIN_SL_DISTANCE_PCT = 1.0    # % minimo de distancia SL (descarta SL degenerado)
LOOKBACK_DIAS       = 12     # dias calendario para buscar CHoCH/BOS (~10 habiles)
DIAS_MAX_POS        = 20     # time-stop: 20 dias calendario maximos


# ── Helper ───────────────────────────────────────────────────

def _swing_low_precio(close: float, dist_sl_pct: float) -> float:
    """
    Recupera el precio absoluto del ultimo swing low estructural.

    Formula: dist_sl_10_pct = (close - swing_low) / swing_low * 100
    Despejando: swing_low = close / (1 + dist_sl_pct / 100)
    """
    if dist_sl_pct <= 0 or close <= 0:
        return 0.0
    return round(close / (1.0 + dist_sl_pct / 100.0), 4)


# ── Scoring ─────────────────────────────────────────────────

def calcular_score_estructura(row: dict) -> tuple[float, dict]:
    """
    Evalua condiciones de entrada estructural para un ticker.

    Retorna (score, detalle):
        score = -1  si no cumple alguna condicion obligatoria
        score = 0-3 si cumple todas (para rankear calidad del setup)
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
        # Evento lookback
        "tuvo_choch_bull":     bool(tuvo_choch_bull),
        "tuvo_bos_bull":       bool(tuvo_bos_bull),
        # Estado estructural hoy
        "estructura_10":       estructura_10,
        "choch_bear_10":       bool(choch_bear_10),
        "dist_sl_pct":         round(dist_sl_pct, 2),
        "dist_sh_pct":         round(dist_sh_pct, 2),
        "swing_low":           swing_low,
        # Vela/volumen hoy
        "es_alcista":          bool(es_alcista),
        "vol_spike":           bool(vol_spike),
        "eng_bull":            bool(eng_bull),
        "hammer":              bool(hammer),
        # Condiciones obligatorias (para diagnostico)
        "cond_evento":         bool(tuvo_choch_bull or tuvo_bos_bull),
        "cond_estructura":     estructura_10 >= 0,
        "cond_no_choch_bear":  not bool(choch_bear_10),
        "cond_vela":           bool(es_alcista),
        "cond_sl_rango":       MIN_SL_DISTANCE_PCT <= dist_sl_pct <= MAX_SL_DISTANCE_PCT,
    }

    # ── Filtros obligatorios ─────────────────────────────────
    if not detalle["cond_evento"]:         return -1.0, detalle
    if not detalle["cond_estructura"]:     return -1.0, detalle
    if not detalle["cond_no_choch_bear"]:  return -1.0, detalle
    if not detalle["cond_vela"]:           return -1.0, detalle
    if not detalle["cond_sl_rango"]:       return -1.0, detalle

    # ── Score de calidad para ranking (0-3) ──────────────────
    score = 0.0
    if tuvo_choch_bull:                    score += 1.0   # CHoCH > BOS
    if vol_spike or eng_bull or hammer:    score += 1.0   # confirmacion vela/vol
    if estructura_10 == 1:                 score += 1.0   # estructura fuerte HH/HL

    return round(score, 2), detalle


# ── Datos ───────────────────────────────────────────────────

def obtener_features_hoy() -> list[dict]:
    """
    Carga features estructurales del dia mas reciente para todos los tickers,
    filtrando los que tuvieron CHoCH/BOS bull en el lookback definido.
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


def obtener_estructura_tickers(tickers: list[str]) -> dict[str, dict]:
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


# ── Trailing SL ─────────────────────────────────────────────

def calcular_actualizaciones_sl(posiciones: list[dict]) -> list[dict]:
    """
    Recalcula el SL estructural para posiciones abiertas.
    Retorna SOLO las posiciones donde el nuevo SL es mas alto (trail up).
    Nunca baja el SL.

    Retorna lista de dicts: {id, ticker, nuevo_sl, sl_anterior, delta_pct}
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

        close_hoy   = float(datos_hoy.get("close", 0) or 0)
        dist_sl_hoy = float(datos_hoy.get("dist_sl_10_pct", 0) or 0)
        sl_actual   = float(pos.get("stop_loss", 0) or 0)

        if not close_hoy or dist_sl_hoy <= 0:
            continue

        nuevo_sl = _swing_low_precio(close_hoy, dist_sl_hoy)

        # Solo actualizar si el SL sube (trailing up, nunca down)
        if nuevo_sl > sl_actual:
            delta_pct = round((nuevo_sl - sl_actual) / sl_actual * 100, 2) if sl_actual > 0 else 0
            updates.append({
                "id":          pos["id"],
                "ticker":      ticker,
                "nuevo_sl":    nuevo_sl,
                "sl_anterior": sl_actual,
                "delta_pct":   delta_pct,
            })

    return updates


# ── Entradas ─────────────────────────────────────────────────

def evaluar_entradas_estructura(
    posiciones_actuales: list[dict],
    equity: float,
) -> list[dict]:
    """
    Escanea todos los tickers con CHoCH/BOS reciente y retorna
    candidatos para abrir posicion (ordenados por score descendente).
    """
    tickers_abiertos = {p["ticker"] for p in posiciones_actuales}
    features         = obtener_features_hoy()
    candidatos       = []

    for row in features:
        ticker = row["ticker"]
        if ticker in tickers_abiertos:
            continue

        score, detalle = calcular_score_estructura(row)

        if score < SCORE_ENTRADA:   # incluye score=-1 (filtro obligatorio falló)
            continue

        if len(posiciones_actuales) + len(candidatos) >= risk.MAX_POSICIONES:
            break

        candidatos.append({
            "ticker":  ticker,
            "score":   score,
            "detalle": detalle,
            "row":     row,
        })

    if not candidatos:
        return []

    candidatos.sort(key=lambda x: x["score"], reverse=True)

    a_abrir = []
    for c in candidatos:
        ticker = c["ticker"]
        row    = c["row"]
        det    = c["detalle"]

        try:
            precio = alpaca_client.get_latest_price(ticker, suffix="_3")
        except Exception:
            precio = float(row.get("close") or 0)
            if not precio:
                continue

        # Recalcular swing_low con el precio real de mercado
        dist_sl_pct = det["dist_sl_pct"]
        swing_low   = _swing_low_precio(precio, dist_sl_pct)

        # Verificar que el SL sigue siendo valido con el precio real
        if swing_low <= 0:
            continue
        dist_real = (precio - swing_low) / swing_low * 100
        if not (MIN_SL_DISTANCE_PCT <= dist_real <= MAX_SL_DISTANCE_PCT + 2.0):
            continue

        atr = float(row.get("atr14") or 0)
        qty = risk.calcular_qty(equity, precio, equity * risk.RIESGO_POR_TRADE)
        if qty < 1:
            continue

        evento = "CHoCH_BULL" if det["tuvo_choch_bull"] else "BOS_BULL"
        nivel  = f"{evento}_{c['score']:.0f}/{SCORE_MAXIMO:.0f}"

        a_abrir.append({
            "ticker":        ticker,
            "precio":        precio,
            "qty":           qty,
            "capital":       round(precio * qty, 2),
            "pct_equity":    round(precio * qty / equity * 100, 1),
            "stop_loss":     swing_low,
            "take_profit":   None,          # Filosofia B: sin TP fijo
            "atr":           round(atr, 4),
            "score":         c["score"],
            "nivel":         nivel,
            "evento":        evento,
            "dist_sl_pct":   round(dist_sl_pct, 2),
            "dist_sh_pct":   round(det["dist_sh_pct"], 2),
            "estructura_10": det["estructura_10"],
            "vol_spike":     det["vol_spike"],
            "eng_bull":      det["eng_bull"],
            "hammer":        det["hammer"],
        })

    return a_abrir


# ── Salidas ──────────────────────────────────────────────────

def evaluar_cierres_estructura(posiciones: list[dict]) -> list[dict]:
    """
    Evalua posiciones abiertas y detecta condiciones de salida.

    Flujo esperado del runner:
        1. Actualizar trailing SL (calcular_actualizaciones_sl + DB update)
        2. Llamar esta funcion con posiciones ya actualizadas en memoria
        3. Cerrar las posiciones retornadas

    Prioridades de salida:
        1. SL trailing roto  : precio_actual <= stop_loss
        2. CHoCH bajista hoy : choch_bear_10 = 1
        3. Estructura rota   : estructura_10 = -1
        4. Time stop         : >= DIAS_MAX_POS dias calendario
    """
    if not posiciones:
        return []

    tickers   = [p["ticker"] for p in posiciones]
    datos_map = obtener_estructura_tickers(tickers)
    a_cerrar  = []

    for pos in posiciones:
        ticker    = pos["ticker"]
        datos_hoy = datos_map.get(ticker)
        if not datos_hoy:
            continue

        try:
            precio_actual = alpaca_client.get_latest_price(ticker, suffix="_3")
        except Exception:
            precio_actual = float(datos_hoy.get("close") or 0)
            if not precio_actual:
                continue

        stop_loss     = float(pos.get("stop_loss", 0) or 0)
        fecha_entrada = pos["fecha_entrada"]
        dias_abierta  = (date.today() - fecha_entrada).days if fecha_entrada else 0

        estructura_10 = int(datos_hoy.get("estructura_10", 0) or 0)
        choch_bear_10 = int(datos_hoy.get("choch_bear_10", 0) or 0)

        motivo = None

        # ── P1: SL trailing roto ─────────────────────────────
        if stop_loss and precio_actual <= stop_loss:
            motivo = "TRAILING_SL"

        # ── P2: CHoCH bajista (cambio de caracter negativo) ───
        elif choch_bear_10 == 1:
            motivo = "CHOCH_BEAR"

        # ── P3: Estructura rota (LH/LL confirmados) ───────────
        elif estructura_10 == -1:
            motivo = "ESTRUCTURA_ROTA"

        # ── P4: Time stop ─────────────────────────────────────
        elif dias_abierta >= DIAS_MAX_POS:
            motivo = f"TIME_STOP_{dias_abierta}D"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar
