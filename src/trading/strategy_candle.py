"""
strategy_candle.py
Estrategia de reversión basada en patrones de velas (Price Action).

Logica:
    Entrada : patron de reversión alcista + contexto bajista + confirmación volumen
    Salida  : condicional D+1, SL/TP dinamicos (ATR), patron contrario, time-stop 5d

Scoring (0-4 pts, minimo 3 para operar):
    +1  Contexto bajista previo   : tendencia_velas <= -1 OR pos_rango_20d < 0.35
    +1  Patron de reversión        : patron_engulfing_bull=1 OR patron_hammer=1
    +1  Volumen confirma           : vol_spike=1 OR up_vol_5d >= 0.60
    +1  RSI en zona neutral-baja   : rsi14 < 52

Filtro obligatorio: es_alcista=1 (la vela del dia es verde — cierre > apertura)

Salidas (orden de prioridad):
    1. SL dinamico  : entrada - 1.5 x ATR14
    2. TP dinamico  : entrada + 3.0 x ATR14  (ratio 1:2)
    3. D+1 falla    : si al dia siguiente precio_cierre < precio_entrada  → salir
    4. Patron contrario: patron_engulfing_bear=1 OR patron_shooting_star=1 → salir
    5. Time stop    : 5 dias habiles maximos en posicion
"""

from datetime import date
from sqlalchemy import text
from src.data.database import get_engine
from src.trading import alpaca_client, risk

# ── Parametros de scoring ─────────────────────────────────────
PTS_CONTEXTO   = 1.0
PTS_PATRON     = 1.0
PTS_VOLUMEN    = 1.0
PTS_RSI        = 1.0
SCORE_MAXIMO   = PTS_CONTEXTO + PTS_PATRON + PTS_VOLUMEN + PTS_RSI  # 4.0
SCORE_ENTRADA  = 3.0

# Parametros de salida
ATR_MULT_SL    = 1.5   # SL = entrada - 1.5 x ATR
ATR_MULT_TP    = 3.0   # TP = entrada + 3.0 x ATR  (ratio 1:2)
RSI_UMBRAL     = 52.0  # no entrar si momentum ya es alto
DIAS_MAX_POS   = 5     # time-stop: maximo dias habiles en posicion

# Thresholds de contexto
TENDENCIA_BAJISTA = -1    # tendencia_velas <= este valor
POS_RANGO_BAJO    = 0.35  # pos_rango_20d < este valor (precio cerca de minimos)
UP_VOL_UMBRAL     = 0.60  # up_vol_5d >= este valor


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────

def calcular_score_vela(row: dict) -> tuple[float, dict]:
    """
    Calcula el score de reversión para un ticker.
    Retorna (score, detalle). Score=0 si filtro obligatorio falla.
    """
    es_alcista        = int(row.get("es_alcista", 0) or 0)
    tendencia_velas   = float(row.get("tendencia_velas", 0) or 0)
    pos_rango_20d     = float(row.get("pos_rango_20d", 1) or 1)
    patron_eng_bull   = int(row.get("patron_engulfing_bull", 0) or 0)
    patron_hammer     = int(row.get("patron_hammer", 0) or 0)
    patron_eng_bear   = int(row.get("patron_engulfing_bear", 0) or 0)
    patron_shooting   = int(row.get("patron_shooting_star", 0) or 0)
    vol_spike         = int(row.get("vol_spike", 0) or 0)
    up_vol_5d         = float(row.get("up_vol_5d", 0) or 0)
    rsi               = float(row.get("rsi14", 99) or 99)

    detalle = {
        "filtro_alcista":    bool(es_alcista),
        "cond_contexto":     tendencia_velas <= TENDENCIA_BAJISTA or pos_rango_20d < POS_RANGO_BAJO,
        "cond_patron":       bool(patron_eng_bull or patron_hammer),
        "cond_volumen":      bool(vol_spike or up_vol_5d >= UP_VOL_UMBRAL),
        "cond_rsi":          rsi < RSI_UMBRAL,
        "patron_eng_bull":   bool(patron_eng_bull),
        "patron_hammer":     bool(patron_hammer),
        "patron_eng_bear":   bool(patron_eng_bear),
        "patron_shooting":   bool(patron_shooting),
        "tendencia_velas":   tendencia_velas,
        "pos_rango_20d":     round(pos_rango_20d, 4),
        "vol_spike":         bool(vol_spike),
        "up_vol_5d":         round(up_vol_5d, 4),
        "rsi":               round(rsi, 2),
    }

    # Filtro obligatorio: vela alcista
    if not detalle["filtro_alcista"]:
        return 0.0, detalle

    score = 0.0
    if detalle["cond_contexto"]: score += PTS_CONTEXTO
    if detalle["cond_patron"]:   score += PTS_PATRON
    if detalle["cond_volumen"]:  score += PTS_VOLUMEN
    if detalle["cond_rsi"]:      score += PTS_RSI

    return round(score, 2), detalle


# ─────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────

def obtener_features_hoy() -> list[dict]:
    """
    Lee las ultimas features de precio accion disponibles para todos los tickers,
    uniendo con indicadores_tecnicos para ATR y RSI.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (pa.ticker)
                pa.ticker,
                pa.fecha,
                pa.es_alcista,
                pa.tendencia_velas,
                pa.pos_rango_20d,
                pa.patron_engulfing_bull,
                pa.patron_hammer,
                pa.patron_engulfing_bear,
                pa.patron_shooting_star,
                pa.vol_spike,
                pa.up_vol_5d,
                pa.vol_price_confirm,
                pa.dist_min_20d,
                pa.dist_max_20d,
                i.rsi14,
                i.atr14,
                i.sma50,
                i.sma200,
                p.close
            FROM features_precio_accion pa
            JOIN indicadores_tecnicos i
              ON i.ticker = pa.ticker AND i.fecha = pa.fecha
            JOIN precios_diarios p
              ON p.ticker = pa.ticker AND p.fecha = pa.fecha
            ORDER BY pa.ticker, pa.fecha DESC
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_features_ticker(ticker: str) -> dict | None:
    """Features del ultimo dia disponible para un ticker especifico."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT DISTINCT ON (pa.ticker)
                pa.ticker,
                pa.fecha,
                pa.es_alcista,
                pa.tendencia_velas,
                pa.pos_rango_20d,
                pa.patron_engulfing_bull,
                pa.patron_hammer,
                pa.patron_engulfing_bear,
                pa.patron_shooting_star,
                pa.vol_spike,
                pa.up_vol_5d,
                pa.vol_price_confirm,
                i.rsi14,
                i.atr14,
                i.sma50,
                i.sma200,
                p.close
            FROM features_precio_accion pa
            JOIN indicadores_tecnicos i
              ON i.ticker = pa.ticker AND i.fecha = pa.fecha
            JOIN precios_diarios p
              ON p.ticker = pa.ticker AND p.fecha = pa.fecha
            WHERE pa.ticker = :ticker
            ORDER BY pa.ticker, pa.fecha DESC
        """), {"ticker": ticker}).fetchone()
    return dict(row._mapping) if row else None


# ─────────────────────────────────────────────────────────────
# Logica de entradas
# ─────────────────────────────────────────────────────────────

def evaluar_entradas_candle(
    posiciones_actuales: list[dict],
    equity: float,
) -> list[dict]:
    """
    Evalua todos los tickers y retorna candidatos para abrir posicion
    basados en patrones de vela + contexto + volumen.
    """
    tickers_abiertos = {p["ticker"] for p in posiciones_actuales}
    features         = obtener_features_hoy()
    candidatos       = []

    for row in features:
        ticker = row["ticker"]
        if ticker in tickers_abiertos:
            continue

        score, detalle = calcular_score_vela(row)

        if score < SCORE_ENTRADA:
            continue

        if len(posiciones_actuales) + len(candidatos) >= risk.MAX_POSICIONES:
            break

        candidatos.append({"ticker": ticker, "score": score, "detalle": detalle, "row": row})

    if not candidatos:
        return []

    # Ordenar por score descendente
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

        atr = float(row.get("atr14") or 0)
        if not atr:
            continue

        stop_loss   = round(precio - ATR_MULT_SL * atr, 4)
        take_profit = round(precio + ATR_MULT_TP * atr, 4)

        qty = risk.calcular_qty(equity, precio, equity * risk.RIESGO_POR_TRADE)
        if qty < 1:
            continue

        patron_nombre = "Engulfing" if det["patron_eng_bull"] else "Hammer"
        nivel = f"VELA_{patron_nombre.upper()}_{c['score']:.0f}/{SCORE_MAXIMO:.0f}"

        a_abrir.append({
            "ticker":       ticker,
            "precio":       precio,
            "qty":          qty,
            "capital":      round(precio * qty, 2),
            "pct_equity":   round(precio * qty / equity * 100, 1),
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "atr":          round(atr, 4),
            "score":        c["score"],
            "nivel":        nivel,
            "patron":       patron_nombre,
            "tendencia_velas": det["tendencia_velas"],
            "pos_rango_20d":   det["pos_rango_20d"],
            "vol_spike":    det["vol_spike"],
            "up_vol_5d":    det["up_vol_5d"],
            "rsi":          det["rsi"],
            "cond_contexto": det["cond_contexto"],
            "cond_patron":   det["cond_patron"],
            "cond_volumen":  det["cond_volumen"],
            "cond_rsi":      det["cond_rsi"],
        })

    return a_abrir


# ─────────────────────────────────────────────────────────────
# Logica de salidas
# ─────────────────────────────────────────────────────────────

def evaluar_cierres_candle(posiciones: list[dict]) -> list[dict]:
    """
    Evalua posiciones abiertas y detecta condiciones de salida.

    Prioridades:
        1. SL dinamico  (precio <= stop_loss)
        2. TP dinamico  (precio >= take_profit)
        3. D+1 falla    (precio_actual < precio_entrada al dia siguiente de entrada)
        4. Patron contrario detectado hoy
        5. Time stop    (>= DIAS_MAX_POS dias habiles)
    """
    if not posiciones:
        return []

    features_map = {
        row["ticker"]: row
        for row in obtener_features_hoy()
    }

    a_cerrar = []

    for pos in posiciones:
        ticker          = pos["ticker"]
        feat            = features_map.get(ticker)
        if not feat:
            continue

        try:
            precio_actual = alpaca_client.get_latest_price(ticker, suffix="_3")
        except Exception:
            precio_actual = float(feat.get("close") or 0)
            if not precio_actual:
                continue

        precio_entrada = float(pos["precio_entrada"])
        stop_loss      = float(pos["stop_loss"])   if pos.get("stop_loss")   else None
        take_profit    = float(pos["take_profit"]) if pos.get("take_profit") else None
        fecha_entrada  = pos["fecha_entrada"]
        dias_abierta   = (date.today() - fecha_entrada).days if fecha_entrada else 0

        patron_contrario = (
            int(feat.get("patron_engulfing_bear", 0) or 0) or
            int(feat.get("patron_shooting_star",  0) or 0)
        )

        motivo = None

        # ── Prioridad 1: SL ──────────────────────────────────
        if stop_loss and precio_actual <= stop_loss:
            motivo = "STOP_LOSS"

        # ── Prioridad 2: TP ──────────────────────────────────
        elif take_profit and precio_actual >= take_profit:
            motivo = "TAKE_PROFIT"

        # ── Prioridad 3: D+1 falla ───────────────────────────
        # Si ya paso al menos 1 dia y precio < entrada: tesis invalida
        elif dias_abierta >= 1 and precio_actual < precio_entrada:
            motivo = "D1_FALLA"

        # ── Prioridad 4: Patron contrario ────────────────────
        elif patron_contrario:
            motivo = "PATRON_CONTRARIO"

        # ── Prioridad 5: Time stop ───────────────────────────
        elif dias_abierta >= DIAS_MAX_POS:
            motivo = f"TIME_STOP_{dias_abierta}D"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar
