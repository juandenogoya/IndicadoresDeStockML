"""
weekly_tf.py
Tecnico SEMANAL al vuelo desde precios diarios. Fuente UNICA del RSI/MACD
semanal para el MCP (get_ticker_sintesis) y el dashboard.

Funciones PURAS: solo pandas + ta. SIN config, SIN database, SIN side effects.
Pensado para ser importable por el MCP server, que tiene prohibido importar
config.py (ejecuta load_dotenv) o cualquier modulo con DB. Por eso replica el
resample W-FRI en vez de reusar src.data.resample_weekly (que importa la DB al
nivel de modulo).

Resample (homogeneo con src/data/resample_weekly.resample_a_semanal):
    - Anchor viernes (W-FRI): cada semana va de sabado a viernes.
    - Close = ultimo cierre de la semana; fecha_semana = ultimo dia habil real.
    - Semana en curso (incompleta) EXCLUIDA para no calcular sobre parciales.

Indicadores (mismos periodos que config.py / dashboard):
    RSI 14 | MACD 12/26/9 sobre el close semanal.

Si los periodos cambian en config.py, actualizar tambien aca (modulo autonomo).
"""

import pandas as pd
import ta

# Periodos estandar (homogeneo con config.py; replicados para mantener el
# modulo autonomo, igual que clasificacion_tecnica.py).
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# Minimo de semanas cerradas para que MACD (slow + signal) tenga sentido.
_MIN_SEMANAS = MACD_SLOW + MACD_SIGNAL  # 35


def _f(val):
    """Valor -> float o None (descarta NaN)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def resample_close_semanal(df_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Resamplea precios diarios a cierres semanales (W-FRI), excluyendo la semana
    en curso. Solo necesita columnas [fecha, close] (el RSI/MACD usan el close).

    Returns:
        DataFrame [fecha_semana (date), close (float)] ordenado ASC.
        Vacio si no hay datos o todo cae en la semana en curso.
    """
    if df_diario is None or len(df_diario) == 0:
        return pd.DataFrame(columns=["fecha_semana", "close"])

    df = df_diario[["fecha", "close"]].copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["_week"] = df["fecha"].dt.to_period("W-FRI")

    weekly = (
        df.groupby("_week", sort=True)
        .agg(fecha_semana=("fecha", "last"), close=("close", "last"))
        .reset_index(drop=True)
    )

    # Excluir semana en curso (incompleta).
    hoy = pd.Timestamp.today().normalize()
    lunes_actual = hoy - pd.Timedelta(days=hoy.weekday())
    weekly = weekly[pd.to_datetime(weekly["fecha_semana"]) < lunes_actual].copy()

    weekly["fecha_semana"] = pd.to_datetime(weekly["fecha_semana"]).dt.date
    weekly["close"] = weekly["close"].astype(float)
    return weekly.sort_values("fecha_semana").reset_index(drop=True)


def rsi_macd_semanal(close, fecha_semana=None) -> dict:
    """
    Calcula RSI14 + MACD de la ULTIMA semana a partir de una serie de cierres
    semanales (ascendente). Reusa ta (no duplica indicadores).

    Args:
        close:        serie/lista de cierres semanales en orden ASC.
        fecha_semana: fecha de la ultima semana (se devuelve tal cual).

    Returns:
        {fecha, rsi, macd, macd_signal} de la ultima semana, o {} si no hay
        suficientes semanas. Claves homogeneas con lo que consume el dashboard
        (dashboard_sintesis.votar_tecnico).
    """
    if close is None:
        return {}
    c = pd.Series(list(close), dtype="float64").reset_index(drop=True)
    if len(c) < _MIN_SEMANAS:
        return {}

    rsi = ta.momentum.RSIIndicator(close=c, window=RSI_PERIOD).rsi()
    macd_ind = ta.trend.MACD(
        close=c, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL
    )
    return {
        "fecha":       fecha_semana,
        "rsi":         _f(rsi.iloc[-1]),
        "macd":        _f(macd_ind.macd().iloc[-1]),
        "macd_signal": _f(macd_ind.macd_signal().iloc[-1]),
    }


def tecnico_semanal(filas) -> dict:
    """
    Entrada de alto nivel: recibe filas OHLC diarias (cualquier secuencia de
    mappings con al menos las claves 'fecha' y 'close'), resamplea a semanal
    W-FRI (excluye la semana en curso) y devuelve el RSI/MACD de la ultima
    semana cerrada.

    Pensado para el MCP, que tiene las filas diarias crudas (asyncpg Records ->
    dict). El dashboard, que ya resamplea para SMC/mensual, puede llamar
    directamente a rsi_macd_semanal sobre su close semanal.

    Returns:
        {fecha, rsi, macd, macd_signal} o {} si la historia es insuficiente.
    """
    if filas is None:
        return {}
    df = pd.DataFrame([dict(r) for r in filas])
    if df.empty or "fecha" not in df.columns or "close" not in df.columns:
        return {}
    sem = resample_close_semanal(df)
    if sem.empty:
        return {}
    return rsi_macd_semanal(sem["close"], sem["fecha_semana"].iloc[-1])
