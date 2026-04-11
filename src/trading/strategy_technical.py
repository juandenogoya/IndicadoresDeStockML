"""
strategy_technical.py
Estrategia basada en 3 capas de indicadores tecnicos.

Capa 1 — Filtro macro (obligatorio):
    Precio > SMA200  →  si no se cumple, skip

Capa 2 — Alineacion de tendencia:
    Precio > SMA50   →  2.0 pts
    Precio > SMA21   →  1.0 pt

Capa 3 — Momentum (timing):
    MACD > Signal Y histograma positivo  →  1.5 pts
    RSI entre 45 y 68                   →  1.0 pt   (no sobrecomprado)

Entrada : score >= 4.0 sobre 5.5
Salida  : precio < SMA50
          precio < SMA200
          MACD histograma negativo 2 dias seguidos
          RSI < 40
          SL 5% / TP 10%
"""

from datetime import date
from sqlalchemy import text
from src.data.database import get_engine
from src.trading import alpaca_client, risk

# ── Parametros de scoring ─────────────────────────────────────
PTS_SMA50      = 2.0
PTS_SMA21      = 1.0
PTS_MACD       = 1.5
PTS_RSI        = 1.0
SCORE_MAXIMO   = PTS_SMA50 + PTS_SMA21 + PTS_MACD + PTS_RSI   # 5.5
SCORE_ENTRADA  = float(4.0)
SCORE_SALIDA   = float(2.5)   # si cae a <= 2.5 salimos

RSI_MIN        = 45.0    # no entrar si momentum es debil
RSI_MAX        = 68.0    # no entrar si esta sobrecomprado
RSI_SALIDA     = 40.0    # salir si RSI cae bajo este valor
MACD_DIAS_NEG  = 2       # dias consecutivos de histograma negativo para salir


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────

def calcular_score_tecnico(row: dict) -> tuple[float, dict]:
    """
    Calcula el score tecnico para un ticker dado un dict de indicadores.
    Retorna (score, detalle).
    """
    close  = float(row.get("close",  0) or 0)
    sma21  = float(row.get("sma21",  0) or 0)
    sma50  = float(row.get("sma50",  0) or 0)
    sma200 = float(row.get("sma200", 0) or 0)
    rsi    = float(row.get("rsi14",  0) or 0)
    macd   = float(row.get("macd",   0) or 0)
    signal = float(row.get("macd_signal", 0) or 0)
    hist   = macd - signal

    detalle = {
        "filtro_sma200": close > sma200,
        "cond_sma50":    close > sma50,
        "cond_sma21":    close > sma21,
        "cond_macd":     macd > signal and hist > 0,
        "cond_rsi":      RSI_MIN <= rsi <= RSI_MAX,
        "rsi":           round(rsi, 2),
        "macd_hist":     round(hist, 4),
        "close":         close,
        "sma50":         round(sma50, 2),
        "sma200":        round(sma200, 2),
    }

    # Capa 1: filtro obligatorio
    if not detalle["filtro_sma200"]:
        return 0.0, detalle

    # Capas 2 y 3
    score = 0.0
    if detalle["cond_sma50"]:  score += PTS_SMA50
    if detalle["cond_sma21"]:  score += PTS_SMA21
    if detalle["cond_macd"]:   score += PTS_MACD
    if detalle["cond_rsi"]:    score += PTS_RSI

    return round(score, 2), detalle


# ─────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────

def obtener_indicadores_hoy() -> list[dict]:
    """
    Lee los ultimos indicadores disponibles para todos los tickers.
    Une precios_diarios con indicadores_tecnicos en la ultima fecha.
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
                i.macd,
                i.macd_signal,
                (i.macd - i.macd_signal) AS macd_hist
            FROM indicadores_tecnicos i
            JOIN precios_diarios p
              ON p.ticker = i.ticker AND p.fecha = i.fecha
            ORDER BY i.ticker, i.fecha DESC
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_historial_macd(ticker: str, dias: int = 3) -> list[float]:
    """
    Retorna los ultimos N valores del histograma MACD para un ticker.
    Orden: mas reciente primero.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT (macd - macd_signal) AS hist
            FROM indicadores_tecnicos
            WHERE ticker = :ticker
            ORDER BY fecha DESC
            LIMIT :dias
        """), {"ticker": ticker, "dias": dias}).fetchall()
    return [float(r.hist) for r in rows]


# ─────────────────────────────────────────────────────────────
# Logica de entradas
# ─────────────────────────────────────────────────────────────

def evaluar_entradas_tech(
    posiciones_actuales: list[dict],
    equity: float,
) -> list[dict]:
    """
    Evalua todos los tickers y retorna candidatos para abrir posicion.
    Aplica las 3 capas de scoring.
    """
    tickers_abiertos = {p["ticker"] for p in posiciones_actuales}
    indicadores      = obtener_indicadores_hoy()
    candidatos       = []

    for row in indicadores:
        ticker = row["ticker"]
        if ticker in tickers_abiertos:
            continue

        score, detalle = calcular_score_tecnico(row)

        if score < SCORE_ENTRADA:
            continue

        if len(posiciones_actuales) + len(candidatos) >= risk.MAX_POSICIONES:
            break

        candidatos.append({
            "ticker":         ticker,
            "score_tecnico":  score,
            "score_maximo":   SCORE_MAXIMO,
            "detalle":        detalle,
        })

    if not candidatos:
        return []

    # Distribucion de capital — fixed 15% por trade
    a_abrir = []
    for c in candidatos:
        ticker = c["ticker"]
        try:
            precio = alpaca_client.get_latest_price(ticker, suffix="_2")
        except Exception:
            precio = float(c["detalle"]["close"]) if c["detalle"]["close"] else None
            if not precio:
                continue

        qty = risk.calcular_qty(equity, precio, equity * risk.RIESGO_POR_TRADE)
        if qty < 1:
            continue

        det = c["detalle"]
        a_abrir.append({
            "ticker":        ticker,
            "precio":        precio,
            "qty":           qty,
            "capital":       round(precio * qty, 2),
            "pct_equity":    round(precio * qty / equity * 100, 1),
            "stop_loss":     risk.calcular_stop_loss(precio),
            "take_profit":   risk.calcular_take_profit(precio),
            "score":         c["score_tecnico"],
            "nivel":         f"TECH_{score:.1f}/{SCORE_MAXIMO}",
            "cond_sma50":    det["cond_sma50"],
            "cond_sma21":    det["cond_sma21"],
            "cond_macd":     det["cond_macd"],
            "cond_rsi":      det["cond_rsi"],
            "rsi":           det["rsi"],
        })

    return a_abrir


# ─────────────────────────────────────────────────────────────
# Logica de salidas
# ─────────────────────────────────────────────────────────────

def evaluar_cierres_tech(posiciones: list[dict]) -> list[dict]:
    """
    Evalua posiciones abiertas y detecta condiciones de salida:
    1. Precio rompe SMA200 (critico)
    2. Precio rompe SMA50
    3. MACD histograma negativo 2 dias consecutivos
    4. RSI < 40
    5. SL / TP
    """
    if not posiciones:
        return []

    # Indicadores actuales
    indicadores_map = {
        row["ticker"]: row
        for row in obtener_indicadores_hoy()
    }

    a_cerrar = []

    for pos in posiciones:
        ticker = pos["ticker"]
        ind    = indicadores_map.get(ticker)
        if not ind:
            continue

        try:
            precio_actual = alpaca_client.get_latest_price(ticker, suffix="_2")
        except Exception:
            precio_actual = float(ind.get("close") or 0)
            if not precio_actual:
                continue

        close  = float(ind.get("close",  precio_actual))
        sma50  = float(ind.get("sma50",  0) or 0)
        sma200 = float(ind.get("sma200", 0) or 0)
        rsi    = float(ind.get("rsi14",  50) or 50)
        hist   = float(ind.get("macd_hist", 0) or 0)

        stop_loss   = float(pos["stop_loss"])   if pos.get("stop_loss")   else None
        take_profit = float(pos["take_profit"]) if pos.get("take_profit") else None

        motivo = None

        # ── Prioridad 1: SL / TP ──────────────────────────────
        if stop_loss and precio_actual <= stop_loss:
            motivo = "STOP_LOSS"
        elif take_profit and precio_actual >= take_profit:
            motivo = "TAKE_PROFIT"

        # ── Prioridad 2: Precio bajo SMA200 (critico) ─────────
        elif sma200 and close < sma200:
            motivo = "ROMPE_SMA200"

        # ── Prioridad 3: Precio bajo SMA50 ────────────────────
        elif sma50 and close < sma50:
            motivo = "ROMPE_SMA50"

        # ── Prioridad 4: RSI debil ────────────────────────────
        elif rsi < RSI_SALIDA:
            motivo = f"RSI_DEBIL_{rsi:.0f}"

        # ── Prioridad 5: MACD histograma negativo consecutivo ─
        else:
            historial = obtener_historial_macd(ticker, MACD_DIAS_NEG)
            if len(historial) >= MACD_DIAS_NEG and all(h < 0 for h in historial):
                motivo = f"MACD_NEG_{MACD_DIAS_NEG}D"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar
