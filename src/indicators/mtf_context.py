"""
mtf_context.py
Calcula tendencia en timeframes superiores (1W y 1M) para usar como
contexto de confirmacion en el analisis tecnico diario.

No modifica el score ML. Aporta informacion visual para Sheets y Telegram.

Funciones:
    get_tendencia_1w(ticker) -> str  : "alcista" | "bajista" | "neutral"
    get_tendencia_1m(ticker) -> str  : "alcista" | "bajista" | "neutral"
    get_contexto_mtf(ticker) -> dict : {"tendencia_1w": ..., "tendencia_1m": ...}
    get_contexto_mtf_batch(tickers)  : dict de dicts, una sola query para todos
"""

from __future__ import annotations
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Etiquetas
# ─────────────────────────────────────────────────────────────
ALCISTA = "alcista"
BAJISTA = "bajista"
NEUTRAL = "neutral"


def _estructura_a_tendencia(estructura: Optional[float]) -> str:
    """Convierte estructura_10 (-1/0/+1) a etiqueta legible."""
    if estructura is None:
        return NEUTRAL
    try:
        v = float(estructura)
    except (TypeError, ValueError):
        return NEUTRAL
    if v >= 1:
        return ALCISTA
    if v <= -1:
        return BAJISTA
    return NEUTRAL


def get_contexto_mtf_batch(tickers: list[str]) -> dict[str, dict]:
    """
    Obtiene tendencia_1w y tendencia_1m para una lista de tickers
    en dos queries (una por timeframe).

    Returns:
        {
            "JPM": {"tendencia_1w": "alcista", "tendencia_1m": "bajista"},
            "KO":  {"tendencia_1w": "neutral",  "tendencia_1m": "alcista"},
            ...
        }
    """
    from src.data.database import query_df

    resultado: dict[str, dict] = {t: {"tendencia_1w": NEUTRAL, "tendencia_1m": NEUTRAL}
                                   for t in tickers}

    if not tickers:
        return resultado

    placeholders = ", ".join(f"'{t}'" for t in tickers)

    # ── Tendencia 1W: estructura_10 de la ultima semana disponible ──
    sql_1w = f"""
        SELECT DISTINCT ON (ticker)
            ticker,
            estructura_10
        FROM features_market_structure_1w
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, fecha DESC
    """
    try:
        df_1w = query_df(sql_1w)
        for _, row in df_1w.iterrows():
            t = row["ticker"]
            if t in resultado:
                resultado[t]["tendencia_1w"] = _estructura_a_tendencia(row.get("estructura_10"))
    except Exception:
        pass  # tabla no disponible — queda NEUTRAL

    # ── Tendencia 1M: comparar cierre actual vs cierre de 4 semanas atras ──
    # Usamos precios_semanales para no necesitar pipeline mensual adicional.
    # 4 semanas ~ 1 mes. Alcista si retorno > +2%, bajista si < -2%, neutral si intermedio.
    sql_1m = f"""
        WITH ultimas AS (
            SELECT
                ticker,
                fecha,
                close,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
            FROM precios_semanales
            WHERE ticker IN ({placeholders})
        ),
        cierre_actual AS (
            SELECT ticker, close AS close_now  FROM ultimas WHERE rn = 1
        ),
        cierre_4s    AS (
            SELECT ticker, close AS close_4s   FROM ultimas WHERE rn = 4
        )
        SELECT
            ca.ticker,
            ROUND(((ca.close_now - c4.close_4s) / NULLIF(c4.close_4s, 0) * 100)::numeric, 2)
                AS retorno_4s_pct
        FROM cierre_actual ca
        JOIN cierre_4s c4 USING (ticker)
    """
    UMBRAL_PCT = 2.0  # >+2% alcista, <-2% bajista
    try:
        df_1m = query_df(sql_1m)
        for _, row in df_1m.iterrows():
            t = row["ticker"]
            if t not in resultado:
                continue
            try:
                ret = float(row["retorno_4s_pct"])
            except (TypeError, ValueError):
                continue
            if ret >= UMBRAL_PCT:
                resultado[t]["tendencia_1m"] = ALCISTA
            elif ret <= -UMBRAL_PCT:
                resultado[t]["tendencia_1m"] = BAJISTA
            else:
                resultado[t]["tendencia_1m"] = NEUTRAL
    except Exception:
        pass  # tabla no disponible — queda NEUTRAL

    return resultado


def get_contexto_mtf(ticker: str) -> dict:
    """
    Version single-ticker. Util para pruebas o uso puntual.
    Para el scanner usar get_contexto_mtf_batch (mas eficiente).
    """
    return get_contexto_mtf_batch([ticker]).get(ticker, {
        "tendencia_1w": NEUTRAL,
        "tendencia_1m": NEUTRAL,
    })
