"""
volatilidad_mtf.py
Volatilidad multi-timeframe (ATR% por timeframe) desde precios diarios.

Modulo PURO: solo pandas + ta. SIN config, SIN database, SIN side effects.
Mismo criterio que weekly_tf.py: importable sin tocar la DB, testeable sin mocks.

Motor: ATR% = ATR(period) / close * 100, sobre las barras del timeframe.
    - El ATR es de la libreria `ta` (ta.volatility.AverageTrueRange), homogeneo
      con src/indicators/technical.py.
    - Normalizado en % del precio -> comparable entre tickers de precio muy
      distinto (base del perfilado de carteras; ver docs/perfiles_carteras.md).

Timeframes y periodos (default; configurables por argumento):
    Diario   ATR 14  sobre las barras diarias tal cual
    Semanal  ATR 8   sobre barras W-FRI (semana de calendario Lun-Vie)
    Mensual  ATR 6   sobre barras de fin de mes

Resample (homogeneo con weekly_tf.resample_close_semanal):
    - Barras de CALENDARIO, no solapadas. Una semana corta por feriado es una
      barra valida igual (se agrupa por periodo, no se cuentan dias).
    - OHLC de cada barra: open=primero, high=max, low=min, close=ultimo.
    - La barra EN CURSO (semana/mes incompleto) se EXCLUYE para no medir sobre
      una barra parcial. El diario recibe cierres ya completos (precios_diarios),
      por eso no se le aplica exclusion.

Los periodos se replican aca (modulo autonomo, igual que weekly_tf). El diario
usa 14 para ser homogeneo con config.ATR_PERIOD; 8 semanal y 6 mensual son las
elecciones del perfilado (docs/perfiles_carteras.md, Fase 0).
"""

import pandas as pd
import ta

ATR_PERIOD_D = 14
ATR_PERIOD_W = 8
ATR_PERIOD_M = 6

_OHLC = ["open", "high", "low", "close"]


def _f(val):
    """Valor -> float o None (descarta NaN)."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def resample_ohlc(df_diario: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """
    Resamplea OHLC diario a barras de calendario del `anchor` dado, excluyendo
    la barra en curso (incompleta).

    Args:
        df_diario: DataFrame con columnas [fecha, open, high, low, close].
        anchor:    periodo pandas: "W-FRI" (semanal) o "M" (mensual).

    Returns:
        DataFrame [fecha (date), open, high, low, close] ASC. Vacio si no hay
        datos o si todo cae en la barra en curso.
    """
    cols = ["fecha"] + _OHLC
    if df_diario is None or len(df_diario) == 0:
        return pd.DataFrame(columns=cols)

    df = df_diario[cols].copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["_period"] = df["fecha"].dt.to_period(anchor)

    barras = (
        df.groupby("_period", sort=True)
        .agg(
            fecha=("fecha", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .reset_index()
    )

    # Excluir la barra en curso (el periodo que contiene hoy).
    periodo_actual = pd.Timestamp.today().normalize().to_period(anchor)
    barras = barras[barras["_period"] < periodo_actual].copy()

    barras["fecha"] = pd.to_datetime(barras["fecha"]).dt.date
    for c in _OHLC:
        barras[c] = barras[c].astype(float)
    return barras.drop(columns="_period").sort_values("fecha").reset_index(drop=True)


def atr_pct(df_ohlc: pd.DataFrame, period: int):
    """
    ATR% = ATR(period) / close * 100 en la ULTIMA barra de df_ohlc.

    Args:
        df_ohlc: DataFrame [high, low, close] (barras del timeframe, ASC).
        period:  ventana del ATR.

    Returns:
        float (ATR% de la ultima barra) o None si faltan barras (< period+1) o
        el resultado no es valido.
    """
    if df_ohlc is None or len(df_ohlc) < period + 1:
        return None
    atr_ind = ta.volatility.AverageTrueRange(
        high=df_ohlc["high"], low=df_ohlc["low"], close=df_ohlc["close"],
        window=period,
    )
    atr = _f(atr_ind.average_true_range().iloc[-1])
    close = _f(df_ohlc["close"].iloc[-1])
    if atr is None or close is None or close == 0:
        return None
    return round(atr / close * 100, 4)


def volatilidad_multi_tf(df_diario: pd.DataFrame, period_d: int = ATR_PERIOD_D,
                         period_w: int = ATR_PERIOD_W,
                         period_m: int = ATR_PERIOD_M) -> dict:
    """
    ATR% en los tres timeframes (Diario / Semanal / Mensual) para un ticker.

    Args:
        df_diario: DataFrame OHLC diario [fecha, open, high, low, close]
                   (se ordena internamente).
        period_*:  ventanas del ATR por timeframe.

    Returns:
        {"atr_pct_d", "atr_pct_w", "atr_pct_m"}. Cada valor es float o None si
        no hay historia suficiente en ese timeframe.
    """
    if df_diario is None or len(df_diario) == 0:
        return {"atr_pct_d": None, "atr_pct_w": None, "atr_pct_m": None}

    df = df_diario.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    semanal = resample_ohlc(df, "W-FRI")
    mensual = resample_ohlc(df, "M")

    return {
        "atr_pct_d": atr_pct(df, period_d),
        "atr_pct_w": atr_pct(semanal, period_w),
        "atr_pct_m": atr_pct(mensual, period_m),
    }
