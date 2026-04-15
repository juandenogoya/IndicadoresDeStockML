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

Entrada : score >= SCORE_ENTRADA (4.0 sobre 5.5)

Salida  : EXIT PRIMARIO — score <= SCORE_SALIDA (indicadores se desalinean)
          EXIT EMERGENCIA — precio <= SL (ATR-based) o precio >= TP (ATR-based)

Principio: entradas y salidas usan el mismo sistema de scoring.
           Si score <= SCORE_SALIDA → imposible que score >= SCORE_ENTRADA
           → contradiccion exit/entry eliminada por diseno matematico.
           Adicionalmente: ticker cerrado en la misma corrida queda excluido
           de entradas (misma data no puede dar decisiones opuestas).
"""

import os
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

SCORE_ENTRADA  = float(os.getenv("BOT_SCORE_ENTRADA_TECH", "4.0"))
SCORE_SALIDA   = float(os.getenv("BOT_SCORE_SALIDA_TECH",  "3.5"))

RSI_MIN        = float(os.getenv("BOT_RSI_MIN", "45.0"))   # no entrar si momentum debil
RSI_MAX        = float(os.getenv("BOT_RSI_MAX", "68.0"))   # no entrar si sobrecomprado

# ── Parametros de SL/TP de emergencia (ATR-based) ─────────────
ATR_MULT_SL    = float(os.getenv("BOT_ATR_MULT_SL", "2.0"))   # SL = entrada - 2x ATR14
ATR_MULT_TP    = float(os.getenv("BOT_ATR_MULT_TP", "4.0"))   # TP = entrada + 4x ATR14


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
    Incluye atr14 para calcular SL/TP de emergencia al entrar.
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


# ─────────────────────────────────────────────────────────────
# Logica de entradas
# ─────────────────────────────────────────────────────────────

def evaluar_entradas_tech(
    posiciones_actuales: list[dict],
    equity: float,
    tickers_excluidos: set = None,
) -> list[dict]:
    """
    Evalua todos los tickers y retorna candidatos para abrir posicion.
    Aplica las 3 capas de scoring.

    tickers_excluidos: set de tickers cerrados en la misma corrida.
    Garantiza que la misma data no genere decision de cierre Y apertura.
    """
    tickers_abiertos  = {p["ticker"] for p in posiciones_actuales}
    tickers_bloqueados = (tickers_excluidos or set()) | tickers_abiertos
    indicadores        = obtener_indicadores_hoy()
    candidatos         = []

    for row in indicadores:
        ticker = row["ticker"]
        if ticker in tickers_bloqueados:
            continue

        score, detalle = calcular_score_tecnico(row)

        if score < SCORE_ENTRADA:
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
        det    = c["detalle"]
        row    = c["row"]

        try:
            precio = alpaca_client.get_latest_price(ticker, suffix="_2")
        except Exception:
            precio = float(det["close"]) if det["close"] else None
            if not precio:
                continue

        atr = float(row.get("atr14") or 0)
        if not atr:
            continue

        qty = risk.calcular_qty(equity, precio, equity * risk.RIESGO_POR_TRADE)
        if qty < 1:
            continue

        score = c["score"]
        a_abrir.append({
            "ticker":      ticker,
            "precio":      precio,
            "qty":         qty,
            "capital":     round(precio * qty, 2),
            "pct_equity":  round(precio * qty / equity * 100, 1),
            "stop_loss":   round(precio - ATR_MULT_SL * atr, 4),
            "take_profit": round(precio + ATR_MULT_TP * atr, 4),
            "atr":         round(atr, 4),
            "score":       score,
            "nivel":       f"TECH_{score:.1f}/{SCORE_MAXIMO}",
            "cond_sma50":  det["cond_sma50"],
            "cond_sma21":  det["cond_sma21"],
            "cond_macd":   det["cond_macd"],
            "cond_rsi":    det["cond_rsi"],
            "rsi":         det["rsi"],
        })

    return a_abrir


# ─────────────────────────────────────────────────────────────
# Logica de salidas
# ─────────────────────────────────────────────────────────────

def evaluar_cierres_tech(posiciones: list[dict]) -> list[dict]:
    """
    Evalua posiciones abiertas y detecta condiciones de salida.

    Prioridades:
        1. EXIT PRIMARIO  — score <= SCORE_SALIDA (indicadores se desalinean)
           El mismo sistema de scoring que decide la entrada decide la salida.
           Garantia matematica: score <= SCORE_SALIDA < SCORE_ENTRADA
           → imposible que la misma data genere exit + entry simultaneamente.

        2. EXIT EMERGENCIA — precio <= stop_loss  (SL ATR-based)
        3. EXIT EMERGENCIA — precio >= take_profit (TP ATR-based)
    """
    if not posiciones:
        return []

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

        stop_loss   = float(pos["stop_loss"])   if pos.get("stop_loss")   else None
        take_profit = float(pos["take_profit"]) if pos.get("take_profit") else None

        score_actual, _ = calcular_score_tecnico(ind)

        motivo = None

        # ── P1: Score degradado (exit primario) ───────────────
        if score_actual <= SCORE_SALIDA:
            motivo = f"SCORE_DEGRADADO_{score_actual:.1f}"

        # ── P2: SL de emergencia (ATR-based) ──────────────────
        elif stop_loss and precio_actual <= stop_loss:
            motivo = "STOP_LOSS_EMERGENCIA"

        # ── P3: TP de emergencia (ATR-based) ──────────────────
        elif take_profit and precio_actual >= take_profit:
            motivo = "TAKE_PROFIT"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
                "score_cierre":  score_actual,
            })

    return a_cerrar
