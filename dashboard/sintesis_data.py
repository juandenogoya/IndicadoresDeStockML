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

from src.data.database import query_df
from src.data.resample_weekly import resample_a_semanal
from src.utils.weekly_tf import rsi_macd_semanal

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


# Umbral de retorno 4 semanas para clasificar tendencia mensual (homogeneo con
# src/indicators/mtf_context.py, que el scanner usa para tendencia_1m).
_TEND_1M_UMBRAL_PCT = 2.0


def _semanal_bundle(ticker: str) -> dict:
    """
    Calcula TODO lo del timeframe superior AL VUELO desde precios_diarios local,
    resampleando a W-FRI una sola vez (semana en curso excluida):
      - tecnico semanal: RSI14 + MACD (la ultima semana cerrada)
      - smc_semanal: estructura SMC sobre barras semanales (tendencia_1w; reusa
        _calcular_ticker_1w de market_structure_1w, sin tocar las tablas _1w)
      - mensual: retorno de 4 semanas -> tendencia_1m (Alcista/Bajista/Neutral)

    No depende de indicadores_tecnicos_1w / features_market_structure_1w /
    precios_semanales (tablas congeladas bajo Plan C). Todo se deriva de
    precios_diarios LOCAL.

    Returns:
        {"tecnico": {fecha, rsi, macd, macd_signal} | {},
         "smc_semanal": {fecha, estructura_10, choch_bull_10, choch_bear_10,
                         bos_bull_10, bos_bear_10} | {},
         "mensual": {fecha, retorno_4s_pct, tendencia} | {}}
    """
    vacio = {"tecnico": {}, "smc_semanal": {}, "mensual": {}}
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
        return vacio

    sem = resample_a_semanal(df)
    if sem.empty:
        return vacio

    out = {"tecnico": {}, "smc_semanal": {}, "mensual": {}}
    fecha_sem = sem["fecha_semana"].iloc[-1]

    # 1. Tecnico semanal (RSI/MACD): reusa weekly_tf (fuente unica del semanal
    # al vuelo, compartida con el MCP). Devuelve {} si faltan semanas.
    out["tecnico"] = rsi_macd_semanal(sem["close"], fecha_sem)

    # 2. SMC semanal (tendencia_1w): estructura sobre barras semanales.
    # _calcular_ticker_1w espera columnas fecha/open/high/low/close/volume.
    if len(sem) >= 21:  # ventana 10 necesita 2*10+1 barras para pivots
        try:
            from src.indicators.market_structure_1w import _calcular_ticker_1w
            df_w = sem.rename(columns={"fecha_semana": "fecha"})[
                ["fecha", "open", "high", "low", "close", "volume"]
            ].copy()
            ms = _calcular_ticker_1w(df_w)
            ult = ms.iloc[-1]
            out["smc_semanal"] = {
                "fecha":         fecha_sem,
                "estructura_10": int(ult["estructura_10"]) if ult["estructura_10"] is not None else None,
                "choch_bull_10": int(ult["choch_bull_10"]),
                "choch_bear_10": int(ult["choch_bear_10"]),
                "bos_bull_10":   int(ult["bos_bull_10"]),
                "bos_bear_10":   int(ult["bos_bear_10"]),
            }
        except Exception:
            out["smc_semanal"] = {}

    # 3. Mensual (tendencia_1m): retorno de 4 semanas (close[-1] vs close[-4]).
    if len(sem) >= 4:
        c_now = _f(sem["close"].iloc[-1])
        c_4s = _f(sem["close"].iloc[-4])
        if c_now is not None and c_4s and c_4s != 0:
            ret = (c_now - c_4s) / c_4s * 100
            if ret >= _TEND_1M_UMBRAL_PCT:
                tend = "Alcista"
            elif ret <= -_TEND_1M_UMBRAL_PCT:
                tend = "Bajista"
            else:
                tend = "Neutral"
            out["mensual"] = {
                "fecha":          fecha_sem,
                "retorno_4s_pct": round(ret, 2),
                "tendencia":      tend,
            }

    return out


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
    (tecnico, opciones_plazo, sector_plazo, price_action, estructura) +
    tendencia_superior (contexto TF semanal/mensual, NO vota el veredicto).
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

    sem = _semanal_bundle(ticker)

    return {
        "ticker":         ticker,
        "perfil":         perfil,
        "precio":         precio,
        "tecnico": {
            "diario":  diario,
            "semanal": sem["tecnico"],
        },
        "opciones_plazo": _opciones_plazo(ticker, close),
        "sector_plazo":   _sector_plazo(perfil.get("sector")),
        "price_action":   _price_action(ticker),
        "estructura":     _estructura(ticker),
        # Contexto de timeframe superior (al vuelo; no modifica el veredicto):
        "tendencia_superior": {
            "smc_semanal": sem["smc_semanal"],
            "mensual":     sem["mensual"],
        },
    }


def listar_tickers() -> list:
    """Tickers disponibles (con precio en local), orden alfabetico."""
    df = query_df("SELECT DISTINCT ticker FROM precios_diarios ORDER BY ticker")
    return df["ticker"].tolist() if not df.empty else []


# ── Analisis Financiero (fundamentales) ──────────────────────────────────────

def listar_sectores_fundamentales() -> list:
    """Sectores presentes en fundamentales_ratios_q (para el screener)."""
    df = query_df(
        "SELECT DISTINCT sector FROM fundamentales_ratios_q "
        "WHERE sector IS NOT NULL ORDER BY sector"
    )
    return df["sector"].tolist() if not df.empty else []


def listar_regiones_fundamentales() -> list:
    """Regiones disponibles en ticker_pais (para el filtro del screener)."""
    df = query_df(
        "SELECT DISTINCT region FROM ticker_pais "
        "WHERE region IS NOT NULL ORDER BY region"
    )
    return df["region"].tolist() if not df.empty else []


def cargar_financiero_ticker(ticker: str) -> dict:
    """
    Foto fundamental del ULTIMO Q de un ticker: ratios + comparacion vs sector.

    Returns:
        {ratios: {col: val}, vs_sector: {metric: {...}}, peer_meta: {...},
         fiscal_period_end, reporting_currency}
    """
    ticker = (ticker or "").strip().upper()
    r = query_df(
        """
        SELECT * FROM fundamentales_ratios_q
        WHERE ticker = :t
        ORDER BY fiscal_period_end DESC
        LIMIT 1
        """,
        params={"t": ticker},
    )
    if r.empty:
        return {"ratios": {}, "vs_sector": {}, "peer_meta": {}}
    ratios = r.iloc[0].to_dict()
    # Valuacion AL CIERRE del dia: pisar los multiplos de Yahoo con los *_px
    # (precio actual). El comparativo (vs_sector) ya viene recalculado con *_px.
    # FALLBACK: en ADRs no-USD los *_px son NULL (mezcla precio USD / denominador
    # en moneda local) -> se conserva el multiplo de Yahoo, que viene bien en la
    # moneda del ADR. Solo se pisa si el *_px tiene valor.
    for _base, _px in (("pe_ratio", "pe_ratio_px"), ("pb_ratio", "pb_ratio_px"),
                       ("ps_ratio", "ps_ratio_px"), ("ev_ebitda", "ev_ebitda_px")):
        if _px in ratios and ratios.get(_px) is not None:
            ratios[_base] = ratios.get(_px)

    vs = query_df(
        """
        SELECT metric, peer_median, peer_p25, peer_p75, vs_median_pct,
               percentile, peer_n, peer_basis, peer_region, low_sample
        FROM fundamentales_ticker_vs_sector
        WHERE ticker = :t
          AND fiscal_period_end = (
              SELECT MAX(fiscal_period_end) FROM fundamentales_ticker_vs_sector
              WHERE ticker = :t
          )
        """,
        params={"t": ticker},
    )
    vs_map = {}
    peer_meta = {}
    for _, row in vs.iterrows():
        vs_map[row["metric"]] = {
            "peer_median":   _f(row["peer_median"]),
            "vs_median_pct": _f(row["vs_median_pct"]),
            "percentile":    _f(row["percentile"]),
            "peer_n":        int(row["peer_n"]) if row["peer_n"] is not None else None,
            "peer_basis":    row["peer_basis"],
            "low_sample":    bool(row["low_sample"]),
        }
        # peer_meta: tomar el de cualquier fila (es el mismo basis/region/n por ticker)
        if not peer_meta:
            peer_meta = {
                "peer_basis":  row["peer_basis"],
                "peer_region": row["peer_region"],
                "peer_n":      int(row["peer_n"]) if row["peer_n"] is not None else None,
            }

    # Precio de cierre del ultimo dia en precios_diarios (dato informativo:
    # precio de mercado en USD vs valor libro/BPA que estan en moneda de reporte).
    precio = _precio(ticker)

    return {
        "ratios":             ratios,
        "vs_sector":          vs_map,
        "peer_meta":          peer_meta,
        "fiscal_period_end":  ratios.get("fiscal_period_end"),
        "reporting_currency": ratios.get("reporting_currency"),
        "precio":             precio,
    }


def cargar_screener_sector(sector: str, region: str | None = None) -> list:
    """
    Ultimo Q de cada ticker de un sector (opcionalmente filtrado por region),
    con sus ratios clave. region=None o 'Todas' => sin filtro de region.

    Returns: lista de dicts (un ticker c/u) con columnas de ratios.
    """
    params = {"sector": sector}
    region_clause = ""
    if region and region not in ("Todas", "Todas las regiones"):
        region_clause = "AND p.region = :region"
        params["region"] = region

    df = query_df(
        f"""
        WITH ult AS (
          SELECT DISTINCT ON (ticker) *
          FROM fundamentales_ratios_q
          WHERE sector = :sector
          ORDER BY ticker, fiscal_period_end DESC
        )
        -- Valuacion al CIERRE del dia (*_px); fallback a Yahoo si *_px es NULL
        -- (ADRs no-USD: el *_px no aplica por mezcla de monedas).
        SELECT u.ticker, u.reporting_currency, p.region, u.profile,
               COALESCE(u.pe_ratio_px, u.pe_ratio) AS pe_ratio,
               COALESCE(u.pb_ratio_px, u.pb_ratio) AS pb_ratio,
               COALESCE(u.ps_ratio_px, u.ps_ratio) AS ps_ratio,
               COALESCE(u.ev_ebitda_px, u.ev_ebitda) AS ev_ebitda,
               u.roe_ttm, u.roa_ttm, u.roic_ttm, u.rotce_ttm,
               u.net_margin_ttm, u.operating_margin_ttm, u.efficiency_ratio_ttm,
               u.revenue_yoy_pct, u.revenue_qoq_pct
        FROM ult u
        LEFT JOIN ticker_pais p ON p.ticker = u.ticker
        WHERE TRUE {region_clause}
        ORDER BY u.ticker
        """,
        params=params,
    )
    return df.to_dict("records") if not df.empty else []


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


def fecha_datos() -> str:
    """Fecha maxima de precios_diarios (ISO). Sirve de key de cache del screener
    de veredictos: cambia solo cuando entran datos nuevos."""
    df = query_df("SELECT MAX(fecha) AS f FROM precios_diarios")
    if df.empty or df.iloc[0]["f"] is None:
        return ""
    return str(df.iloc[0]["f"])


def listar_sectores() -> list:
    """Sectores del universo (tabla activos), para el filtro del screener de
    veredictos. Query barata -> puebla el multiselect antes de calcular nada."""
    df = query_df(
        "SELECT DISTINCT sector FROM activos WHERE sector IS NOT NULL ORDER BY sector"
    )
    return df["sector"].tolist() if not df.empty else []


def cargar_veredictos_universo() -> list:
    """
    Veredicto sintetico (ALCISTA / NEUTRAL / BAJISTA) de TODOS los tickers con
    datos, via cargar_datos_ticker + sintetizar. Recorre los ~199 -> ~2 min;
    pensada para cachear (1 calculo por fecha de datos). Funcion pura, sin Streamlit.

    Returns:
        [{ticker, sector, veredicto, frase}] en orden alfabetico de ticker.
    """
    from src.utils.dashboard_sintesis import sintetizar

    filas = []
    for tk in listar_tickers():
        datos = cargar_datos_ticker(tk)
        if not (datos.get("perfil") or (datos.get("precio") or {}).get("close") is not None):
            continue
        s = sintetizar(datos)
        filas.append({
            "ticker":    tk,
            "sector":    (datos.get("perfil") or {}).get("sector") or "?",
            "veredicto": s["estado"],
            "frase":     s.get("frase", ""),
        })
    return filas
