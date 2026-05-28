"""
dashboard/sintesis_data.py
Capa de fetch del Dashboard. Lee la DB LOCAL (Plan C: fuente de verdad) y
arma el dict de entrada que consume src.utils.dashboard_sintesis.sintetizar().

Decisiones de origen de datos (verificadas 27/5/2026):
  - Todo sale de LOCAL (precios_diarios, indicadores_tecnicos, features_*,
    opciones_pcr_plazo_diario, opciones_sector_pcr_plazo_diario). Railway esta
    congelado para precios bajo Plan C.
  - El TECNICO SEMANAL se calcula AL VUELO desde precios_diarios local
    (resample W-FRI + RSI/MACD), NO desde indicadores_tecnicos_1w (tabla de un
    pipeline deprecado, congelada en 2026-04-02 / 124 tickers). Asi cubre los
    199 tickers y no arrastra data vieja.
  - Solo se computan RSI14 + MACD semanal (lo que vota/ muestra el informe), sin
    el warmup de SMA200 que descartaria tickers con < 200 semanas de historia.

Requiere correr con engine LOCAL: que DATABASE_URL NO este seteada (si lo esta,
get_engine cae a Railway). El launcher del dashboard se encarga de eso.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import ta

from src.data.database import query_df
from src.data.resample_weekly import resample_a_semanal

# Periodos estandar (homogeneo con config.py; replicados como en
# clasificacion_tecnica.py para mantener el modulo simple)
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

_VER_MAP = {"A": "Alcista", "B": "Bajista"}


def _f(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _muro_recalc(strike, oi, close) -> dict | None:
    """
    Recalcula el muro de OI con el close diario real (no con precio_sub del
    snapshot, que es el cierre del dia anterior). dist_pct negativo = debajo
    (soporte); positivo = arriba (resistencia).
    """
    s, c = _f(strike), _f(close)
    if s is None or c is None or c == 0:
        return None
    dist = (s - c) / c * 100
    return {
        "strike":   round(s, 2),
        "oi":       int(oi) if oi is not None else None,
        "dist_pct": round(dist, 2),
        "posicion": "debajo" if dist < 0 else "arriba",
    }


# ── Bloques de datos ─────────────────────────────────────────────────────────

def _perfil(ticker: str) -> dict:
    df = query_df(
        "SELECT sector, industry FROM activos WHERE ticker = :t",
        params={"t": ticker},
    )
    if df.empty:
        return {}
    return {"sector": df.iloc[0]["sector"], "industry": df.iloc[0]["industry"]}


def _precio(ticker: str) -> dict:
    df = query_df(
        """
        SELECT fecha, close, adj_close
        FROM   precios_diarios
        WHERE  ticker = :t
        ORDER  BY fecha DESC
        LIMIT  1
        """,
        params={"t": ticker},
    )
    if df.empty:
        return {}
    row = df.iloc[0]
    close = _f(row["close"]) if row["close"] is not None else _f(row["adj_close"])
    return {"fecha": row["fecha"], "close": close}


def _tecnico_diario(ticker: str) -> dict:
    df = query_df(
        """
        SELECT fecha, rsi14, macd, macd_signal, adx, sma21, sma50, sma200
        FROM   indicadores_tecnicos
        WHERE  ticker = :t
        ORDER  BY fecha DESC
        LIMIT  1
        """,
        params={"t": ticker},
    )
    if df.empty:
        return {}
    r = df.iloc[0]
    return {
        "fecha":       r["fecha"],
        "rsi":         _f(r["rsi14"]),
        "macd":        _f(r["macd"]),
        "macd_signal": _f(r["macd_signal"]),
        "adx":         _f(r["adx"]),
        "sma21":       _f(r["sma21"]),
        "sma50":       _f(r["sma50"]),
        "sma200":      _f(r["sma200"]),
    }


def _tecnico_semanal(ticker: str) -> dict:
    """
    Calcula RSI14 + MACD semanal AL VUELO desde precios_diarios local.
    Resamplea a W-FRI (semana en curso excluida) y aplica ta sobre el close
    semanal. Devuelve la ultima semana CERRADA. {} si no hay historia suficiente.
    """
    df = query_df(
        """
        SELECT fecha, open, high, low, close, volume, adj_close
        FROM   precios_diarios
        WHERE  ticker = :t
        ORDER  BY fecha ASC
        """,
        params={"t": ticker},
    )
    if df.empty:
        return {}

    sem = resample_a_semanal(df)
    if sem.empty or len(sem) < MACD_SLOW + MACD_SIGNAL:
        # Sin suficientes semanas para un MACD estable -> semanal "sin datos".
        return {}

    close = sem["close"].astype(float)
    rsi = ta.momentum.RSIIndicator(close=close, window=RSI_PERIOD).rsi()
    macd_ind = ta.trend.MACD(close=close, window_fast=MACD_FAST,
                             window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    macd = macd_ind.macd()
    signal = macd_ind.macd_signal()

    return {
        "fecha":       sem["fecha_semana"].iloc[-1],
        "rsi":         _f(rsi.iloc[-1]),
        "macd":        _f(macd.iloc[-1]),
        "macd_signal": _f(signal.iloc[-1]),
    }


def _opciones_plazo(ticker: str, close) -> dict:
    """PCR + muros por ventana del ultimo snapshot del ticker. Muros recalc vs close."""
    df = query_df(
        """
        SELECT ventana, pcr_vol, pcr_oi, veredicto_oi, precio_sub,
               call_vol, put_vol, call_oi, put_oi,
               soporte_strike, soporte_oi, resistencia_strike, resistencia_oi
        FROM   opciones_pcr_plazo_diario
        WHERE  ticker = :t
          AND  fecha = (SELECT MAX(fecha) FROM opciones_pcr_plazo_diario WHERE ticker = :t)
        """,
        params={"t": ticker},
    )
    out = {}
    for _, r in df.iterrows():
        out[r["ventana"]] = {
            "pcr_vol":      _f(r["pcr_vol"]),
            "pcr_oi":       _f(r["pcr_oi"]),
            "veredicto_oi": _VER_MAP.get(r["veredicto_oi"]),
            "put_wall":     _muro_recalc(r["soporte_strike"], r["soporte_oi"], close),
            "call_wall":    _muro_recalc(r["resistencia_strike"], r["resistencia_oi"], close),
            # Crudos (para el papel de trabajo)
            "call_vol":     int(r["call_vol"]) if r["call_vol"] is not None else None,
            "put_vol":      int(r["put_vol"])  if r["put_vol"]  is not None else None,
            "call_oi":      int(r["call_oi"])  if r["call_oi"]  is not None else None,
            "put_oi":       int(r["put_oi"])   if r["put_oi"]   is not None else None,
        }
    return out


def _sector_plazo(sector: str) -> dict:
    """PCR sectorial por ventana del ultimo snapshot del sector."""
    if not sector:
        return {}
    df = query_df(
        """
        SELECT ventana, pcr_vol_sector, pcr_oi_sector, veredicto_oi,
               pcr_vol_sector_zscore, pcr_vol_sector_media, pcr_vol_sector_std,
               n_tickers
        FROM   opciones_sector_pcr_plazo_diario
        WHERE  sector = :s
          AND  fecha = (SELECT MAX(fecha) FROM opciones_sector_pcr_plazo_diario WHERE sector = :s)
        """,
        params={"s": sector},
    )
    out = {}
    for _, r in df.iterrows():
        out[r["ventana"]] = {
            "pcr_vol_sector":        _f(r["pcr_vol_sector"]),
            "pcr_oi_sector":         _f(r["pcr_oi_sector"]),
            "veredicto_oi":          _VER_MAP.get(r["veredicto_oi"]),
            "pcr_vol_sector_zscore": _f(r["pcr_vol_sector_zscore"]),
            "pcr_vol_sector_media":  _f(r["pcr_vol_sector_media"]),
            "pcr_vol_sector_std":    _f(r["pcr_vol_sector_std"]),
            "n_tickers":             int(r["n_tickers"]) if r["n_tickers"] is not None else None,
        }
    return out


def _price_action(ticker: str) -> dict:
    df = query_df(
        """
        SELECT es_alcista, vol_spike,
               patron_engulfing_bull, patron_engulfing_bear,
               patron_hammer, patron_shooting_star
        FROM   features_precio_accion
        WHERE  ticker = :t
        ORDER  BY fecha DESC
        LIMIT  1
        """,
        params={"t": ticker},
    )
    if df.empty:
        return {}
    r = df.iloc[0]
    return {k: r[k] for k in df.columns}


def _estructura(ticker: str) -> dict:
    df = query_df(
        """
        SELECT estructura_10, choch_bull_10, choch_bear_10, bos_bull_10, bos_bear_10
        FROM   features_market_structure
        WHERE  ticker = :t
        ORDER  BY fecha DESC
        LIMIT  1
        """,
        params={"t": ticker},
    )
    if df.empty:
        return {}
    r = df.iloc[0]
    return {k: r[k] for k in df.columns}


# ── Entrada principal ────────────────────────────────────────────────────────

def cargar_datos_ticker(ticker: str) -> dict:
    """
    Arma el dict completo de un ticker para el dashboard:
    secciones de display (perfil, precio) + las que consume el cerebro
    (tecnico, opciones_plazo, sector_plazo, price_action, estructura).
    """
    ticker = (ticker or "").strip().upper()
    perfil = _perfil(ticker)
    precio = _precio(ticker)
    close = precio.get("close")

    # El close vive en precios_diarios (no en indicadores_tecnicos). Lo inyectamos
    # en el tecnico diario porque la clasificacion de tendencia SMA y la regla A6
    # lo necesitan (close vs medias).
    diario = _tecnico_diario(ticker)
    if diario is not None:
        diario["close"] = close

    return {
        "ticker":         ticker,
        "perfil":         perfil,
        "precio":         precio,
        "tecnico": {
            "diario":  diario,
            "semanal": _tecnico_semanal(ticker),
        },
        "opciones_plazo": _opciones_plazo(ticker, close),
        "sector_plazo":   _sector_plazo(perfil.get("sector")),
        "price_action":   _price_action(ticker),
        "estructura":     _estructura(ticker),
    }


def listar_tickers() -> list:
    """Tickers disponibles (con precio en local), orden alfabetico."""
    df = query_df("SELECT DISTINCT ticker FROM precios_diarios ORDER BY ticker")
    return df["ticker"].tolist() if not df.empty else []


def cargar_radar() -> dict:
    """
    Datos para el Radar (familia E): z-scores de actividad de opciones de TODO el
    universo en la ultima fecha disponible, + z-score de volumen por sector.

    percentil_vol viene en escala 0-100 (usado como guarda de liquidez).
    stock_vol_z = z de volumen de la ACCION (ticker_zscore_diario) en la misma
    fecha; habilita el cruce accion+opciones. None si la tabla no tiene esa fecha
    (ticker_zscore_diario se recalcula en local; ver memory/dashboard.md).

    Returns:
        {"fecha": str, "tickers": [ {ticker, sector, vol_total, vol_z, iv_z,
          pcr_z, pcr_vol, iv_avg, percentil_vol, stock_vol_z} ], "sector_z": {...}}
    """
    df = query_df(
        """
        SELECT fecha, ticker, sector, vol_total,
               vol_total_zscore, iv_zscore, pcr_vol_zscore,
               pcr_vol, iv_avg, percentil_vol
        FROM   opciones_zscore_diario
        WHERE  fecha = (SELECT MAX(fecha) FROM opciones_zscore_diario)
        """
    )
    fecha_opc = df["fecha"].iloc[0] if not df.empty else None

    sec = query_df(
        """
        SELECT sector, vol_total_sector_zscore
        FROM   opciones_sector_zscore_diario
        WHERE  fecha = (SELECT MAX(fecha) FROM opciones_sector_zscore_diario)
        """
    )
    sector_z = {}
    for _, r in sec.iterrows():
        if r["sector"] is not None:
            sector_z[r["sector"]] = _f(r["vol_total_sector_zscore"])

    # z de volumen de la accion en la MISMA fecha (cruce accion+opciones).
    stock_z = {}
    if fecha_opc is not None:
        sk = query_df(
            "SELECT ticker, vol_zscore FROM ticker_zscore_diario WHERE fecha = :f",
            params={"f": fecha_opc},
        )
        for _, r in sk.iterrows():
            stock_z[r["ticker"]] = _f(r["vol_zscore"])

    tickers = []
    fecha = None
    for _, r in df.iterrows():
        fecha = str(r["fecha"])
        tickers.append({
            "ticker":        r["ticker"],
            "sector":        r["sector"],
            "vol_total":     int(r["vol_total"]) if r["vol_total"] is not None else None,
            "vol_z":         _f(r["vol_total_zscore"]),
            "iv_z":          _f(r["iv_zscore"]),
            "pcr_z":         _f(r["pcr_vol_zscore"]),
            "pcr_vol":       _f(r["pcr_vol"]),
            "iv_avg":        _f(r["iv_avg"]),
            "percentil_vol": _f(r["percentil_vol"]),
            "stock_vol_z":   stock_z.get(r["ticker"]),
        })
    return {"fecha": fecha, "tickers": tickers, "sector_z": sector_z}
