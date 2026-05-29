"""
mtf_context.py
Calcula tendencia en timeframes superiores (1W y 1M) para usar como
contexto de confirmacion en el analisis tecnico diario.

No modifica el score ML. Aporta informacion visual para Telegram (mensaje
del scanner). NO se persiste en alertas_scanner.

AL VUELO desde precios_diarios (28/5/2026, Plan C): antes leia las tablas
precios_semanales / features_market_structure_1w (pipeline 1W deprecado,
congeladas). Ahora resamplea W-FRI y calcula la estructura SMC semanal en el
momento, reusando la logica pura de src (igual que el dashboard). Asi el
contexto MTF del Telegram queda SIEMPRE fresco sin mantener tablas 1W.

Funciones:
    get_contexto_mtf(ticker) -> dict : {"tendencia_1w": ..., "tendencia_1m": ...}
    get_contexto_mtf_batch(tickers)  : dict de dicts (una sola query bulk de precios)
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Etiquetas
# ─────────────────────────────────────────────────────────────
ALCISTA = "alcista"
BAJISTA = "bajista"
NEUTRAL = "neutral"

UMBRAL_1M_PCT = 2.0   # tendencia_1m: >+2% alcista, <-2% bajista
_DIAS_HISTORIA = 540  # ~75 semanas: suficiente para SMC semanal ventana 10
_MIN_SEMANAS_SMC = 21 # 2*10+1: minimo para pivots de la ventana 10


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
    Obtiene tendencia_1w (estructura SMC semanal) y tendencia_1m (retorno 4
    semanas) para una lista de tickers, calculadas AL VUELO desde precios_diarios.

    Una sola query bulk de precios (ultimos ~540 dias) + resample W-FRI por
    ticker. tendencia_1w usa estructura_10 sobre barras semanales; tendencia_1m
    compara el cierre semanal actual vs el de 4 semanas atras.

    Returns:
        {"JPM": {"tendencia_1w": "alcista", "tendencia_1m": "bajista"}, ...}
        Tickers sin datos suficientes quedan en NEUTRAL/NEUTRAL.
    """
    resultado: dict[str, dict] = {t: {"tendencia_1w": NEUTRAL, "tendencia_1m": NEUTRAL}
                                   for t in tickers}
    if not tickers:
        return resultado

    from src.data.database import query_df
    from src.data.resample_weekly import resample_a_semanal
    from src.indicators.market_structure_1w import _calcular_ticker_1w

    desde = date.today() - timedelta(days=_DIAS_HISTORIA)
    placeholders = ", ".join(f"'{t}'" for t in tickers)
    try:
        df = query_df(f"""
            SELECT ticker, fecha, open, high, low, close, volume, adj_close
            FROM   precios_diarios
            WHERE  ticker IN ({placeholders})
              AND  fecha >= :desde
            ORDER  BY ticker, fecha
        """, params={"desde": desde})
    except Exception:
        return resultado  # sin precios -> todo NEUTRAL

    if df.empty:
        return resultado

    for ticker, grp in df.groupby("ticker", sort=False):
        if ticker not in resultado:
            continue
        try:
            sem = resample_a_semanal(grp)
        except Exception:
            continue
        if sem.empty:
            continue

        # tendencia_1m: retorno de 4 semanas (close[-1] vs close[-4])
        if len(sem) >= 4:
            c_now = float(sem["close"].iloc[-1])
            c_4s = float(sem["close"].iloc[-4])
            if c_4s:
                ret = (c_now - c_4s) / c_4s * 100
                if ret >= UMBRAL_1M_PCT:
                    resultado[ticker]["tendencia_1m"] = ALCISTA
                elif ret <= -UMBRAL_1M_PCT:
                    resultado[ticker]["tendencia_1m"] = BAJISTA

        # tendencia_1w: estructura SMC (ventana 10) sobre barras semanales
        if len(sem) >= _MIN_SEMANAS_SMC:
            try:
                df_w = sem.rename(columns={"fecha_semana": "fecha"})[
                    ["fecha", "open", "high", "low", "close", "volume"]
                ].copy()
                ms = _calcular_ticker_1w(df_w)
                resultado[ticker]["tendencia_1w"] = _estructura_a_tendencia(
                    ms["estructura_10"].iloc[-1]
                )
            except Exception:
                pass

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
