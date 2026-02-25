"""
feature_calculator.py
Calcula los 53 features del modelo V3 para cualquier ticker en tiempo real.

Flujo:
    1. OHLCV  ->  calcular_indicadores()       (29 indicadores tecnicos)
    2. indicadores + precios -> calcular_scoring()  (score rule-based)
    3. OHLCV  ->  _calcular_ticker()           (24 features market structure)
    4. OHLCV + indicadores -> _calcular_grupo() (32 features precio/accion)
    5. feature_engineering()                   (bb_posicion, atr14_pct, momentum_pct)
    6. Z-scores sectoriales desde DB (o NaN si ticker nuevo sin sector)
    7. Retorna el ultimo row como dict con las 53 features listas para el modelo
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict

from src.indicators.technical import calcular_indicadores
from src.scoring.rule_based import calcular_scoring
from src.indicators.market_structure import _calcular_ticker, FEATURE_COLS_MS
from src.indicators.precio_accion import _calcular_grupo
from src.ml.trainer import feature_engineering, FEATURE_COLS, _BOOL_COLS
from src.ml.trainer_v3 import FEATURE_COLS_V3
from src.data.database import query_df


# ─────────────────────────────────────────────────────────────
# Columnas PA relevantes para el scanner (no para el modelo V3,
# sino para evaluar condiciones EV1-EV4 y senales bajistas)
# ─────────────────────────────────────────────────────────────

PA_COLS_EXTRA = [
    "es_alcista", "patron_hammer", "patron_engulfing_bull",
    "vol_spike", "up_vol_5d", "vol_price_confirm",
    "patron_engulfing_bear", "patron_shooting_star",
]


# ─────────────────────────────────────────────────────────────
# Z-scores sectoriales desde DB
# ─────────────────────────────────────────────────────────────

_SECTOR_ZCOLS = [
    "z_rsi_sector", "z_retorno_1d_sector", "z_retorno_5d_sector",
    "z_vol_sector", "z_dist_sma50_sector", "z_adx_sector",
    "pct_long_sector", "rank_retorno_sector",
    "rsi_sector_avg", "adx_sector_avg", "retorno_1d_sector_avg",
]


def _obtener_zscore_sectorial(ticker: str,
                               fecha: Optional[object] = None) -> Dict[str, Optional[float]]:
    """
    Obtiene los z-scores sectoriales de la ultima fecha disponible en DB
    para el ticker. Retorna dict con NaN si no hay datos.
    """
    empty = {c: np.nan for c in _SECTOR_ZCOLS}

    sql = """
        SELECT z_rsi_sector, z_retorno_1d_sector, z_retorno_5d_sector,
               z_vol_sector, z_dist_sma50_sector, z_adx_sector,
               pct_long_sector, rank_retorno_sector,
               rsi_sector_avg, adx_sector_avg, retorno_1d_sector_avg
        FROM features_sector
        WHERE ticker = :ticker
        ORDER BY fecha DESC
        LIMIT 1
    """
    try:
        df = query_df(sql, params={"ticker": ticker})
        if df.empty:
            return empty
        row = df.iloc[0]
        return {c: (float(row[c]) if not pd.isna(row[c]) else np.nan) for c in _SECTOR_ZCOLS}
    except Exception:
        return empty


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def calcular_features_completas(df_ohlcv: pd.DataFrame,
                                  ticker: str,
                                  sector: Optional[str] = None) -> Dict:
    """
    Calcula los 53 features V3 + features PA adicionales para la barra mas reciente.

    Args:
        df_ohlcv: DataFrame con columnas fecha/open/high/low/close/volume/ticker
                  Debe tener >= 250 barras para features confiables.
        ticker:   codigo del activo
        sector:   nombre del sector (None si desconocido)

    Returns:
        dict con:
            'features_v3'  : dict[str, float] con las 53 features del modelo
            'features_pa'  : dict con features PA (EV1-EV4) + senales bajistas
            'meta'         : dict con precio_cierre, precio_fecha, atr14
            'ok'           : bool, False si hubo error critico
            'error'        : str con descripcion del error (si ok==False)
    """
    try:
        df = df_ohlcv.copy()
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values("fecha").reset_index(drop=True)

        if len(df) < 50:
            return {"ok": False, "error": f"Solo {len(df)} barras disponibles (min=50)"}

        # ── 1. Indicadores tecnicos ───────────────────────────
        df_ind = calcular_indicadores(df, ticker)
        df_ind["fecha"] = pd.to_datetime(df_ind["fecha"])

        # ── 2. Scoring rule-based ─────────────────────────────
        df_precios_simple = df[["fecha", "close"]].copy()
        df_precios_simple["fecha"] = pd.to_datetime(df_precios_simple["fecha"])

        df_sc = calcular_scoring(df_ind, df_precios_simple, ticker)
        df_sc["fecha"] = pd.to_datetime(df_sc["fecha"])

        # ── 3. Merge: OHLCV + indicadores + scoring ───────────
        df_base = df.merge(df_ind, on="fecha", how="left", suffixes=("", "_ind"))
        # Eliminar columnas duplicadas de indicadores que ya existen
        dup_cols = [c for c in df_base.columns if c.endswith("_ind")]
        df_base.drop(columns=dup_cols, inplace=True, errors="ignore")

        df_base = df_base.merge(
            df_sc[["fecha", "score_ponderado", "condiciones_ok",
                   "cond_rsi", "cond_macd", "cond_sma21",
                   "cond_sma50", "cond_sma200", "cond_momentum"]],
            on="fecha", how="left"
        )

        # ── 4. Feature engineering (bb_posicion, atr14_pct, momentum_pct)
        df_base = feature_engineering(df_base)

        # ── 5. Market structure (24 features) ─────────────────
        df_ms = _calcular_ticker(df[["fecha", "open", "high", "low", "close", "volume"]].copy())
        df_ms["fecha"] = pd.to_datetime(df_ms["fecha"])
        df_base = df_base.merge(df_ms[["fecha"] + FEATURE_COLS_MS], on="fecha", how="left",
                                 suffixes=("", "_ms"))

        # ── 6. Features precio/accion (PA) ─────────────────────
        # _calcular_grupo necesita ticker, fecha, open, high, low, close, volume, atr14, vol_relativo
        df_pa_input = df_base[["fecha", "open", "high", "low", "close", "volume",
                                "atr14", "vol_relativo"]].copy()
        df_pa_input.insert(0, "ticker", ticker)
        df_pa_result = _calcular_grupo(df_pa_input)

        pa_extra_cols = [c for c in PA_COLS_EXTRA if c in df_pa_result.columns]
        df_base = df_base.merge(df_pa_result[["fecha"] + pa_extra_cols], on="fecha", how="left",
                                 suffixes=("", "_pa"))

        # ── 7. Z-scores sectoriales ──────────────────────────
        zscores = _obtener_zscore_sectorial(ticker)
        for col, val in zscores.items():
            df_base[col] = val

        # ── 8. Extraer la ultima barra ─────────────────────────
        # Buscar la ultima fila con datos tecnicos validos
        mask_valida = df_base["rsi14"].notna() & df_base["atr14"].notna()
        if not mask_valida.any():
            return {"ok": False, "error": "No hay barras con indicadores calculados"}

        last_idx = df_base[mask_valida].index[-1]
        row = df_base.loc[last_idx]

        # ── 9. Armar dict de features V3 ──────────────────────
        features_v3 = {}
        for col in FEATURE_COLS_V3:
            val = row.get(col, np.nan)
            if pd.isna(val):
                features_v3[col] = np.nan
            else:
                # Convertir booleanos de scoring a float
                if col in _BOOL_COLS:
                    features_v3[col] = float(bool(val))
                else:
                    features_v3[col] = float(val)

        # ── 10. Armar dict de features PA ─────────────────────
        def _safe_int(col):
            v = row.get(col, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return int(v)

        def _safe_float(col):
            v = row.get(col, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)

        features_pa = {
            # Condiciones alcistas PA
            "es_alcista":            _safe_int("es_alcista"),
            "patron_hammer":         _safe_int("patron_hammer"),
            "patron_engulfing_bull": _safe_int("patron_engulfing_bull"),
            "vol_spike":             _safe_int("vol_spike"),
            "up_vol_5d":             _safe_float("up_vol_5d"),
            "vol_price_confirm":     _safe_int("vol_price_confirm"),
            # Condiciones bajistas PA
            "patron_engulfing_bear": _safe_int("patron_engulfing_bear"),
            "patron_shooting_star":  _safe_int("patron_shooting_star"),
            # Market structure (para EV1-EV4 y senales bajistas)
            "estructura_10":   _safe_int("estructura_10"),
            "dias_sl_10":      _safe_int("dias_sl_10"),
            "dias_sl_5":       _safe_int("dias_sl_5"),
            "dist_sl_10_pct":  _safe_float("dist_sl_10_pct"),
            "dist_sh_10_pct":  _safe_float("dist_sh_10_pct"),
            "bos_bull_10":     _safe_int("bos_bull_10"),
            "choch_bull_10":   _safe_int("choch_bull_10"),
            "bos_bear_10":     _safe_int("bos_bear_10"),
            "choch_bear_10":   _safe_int("choch_bear_10"),
            "dist_sl_5_pct":   _safe_float("dist_sl_5_pct"),
            "dias_sh_10":      _safe_int("dias_sh_10"),
        }

        # ── 11. Meta ──────────────────────────────────────────
        meta = {
            "precio_cierre": _safe_float("close"),
            "precio_fecha":  row["fecha"].date() if hasattr(row["fecha"], "date") else row["fecha"],
            "atr14":         _safe_float("atr14"),
            "score_ponderado": _safe_float("score_ponderado"),
            "condiciones_ok":  _safe_int("condiciones_ok"),
        }

        return {
            "ok":          True,
            "features_v3": features_v3,
            "features_pa": features_pa,
            "meta":        meta,
        }

    except Exception as e:
        import traceback
        return {"ok": False, "error": f"Excepcion: {e}\n{traceback.format_exc()}"}
