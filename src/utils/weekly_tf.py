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
    - Semana incompleta EXCLUIDA para no calcular sobre parciales.

Semana completa = POR DATO, no por reloj (17/9/2026, docs/estructura_velas.md sec. 6):
    Antes se excluia la semana que contiene HOY. Con la rutina corriendo el viernes a
    la noche con el dato del viernes, la semana recien cerrada quedaba afuera y el
    semanal atrasaba una semana; un sabado con el viernes faltante tomaba como
    cerrada una semana parcial. Ahora la ultima semana cuenta si su ultimo dato es
    la ultima rueda habil NYSE de esa semana (trading_calendar: un Viernes Santo la
    cierra el jueves). trading_calendar cubre 2025-2027: fuera de ese rango un
    feriado de viernes solo atrasa la semana hasta que llega el dato siguiente.
    Es la unica regla: la usa tambien resample_weekly.

Indicadores (mismos periodos que config.py / dashboard):
    RSI 14 | MACD 12/26/9 sobre el close semanal.

Si los periodos cambian en config.py, actualizar tambien aca (modulo autonomo).
"""

from datetime import date, timedelta

import pandas as pd
import ta

from src.utils.trading_calendar import is_trading_day

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


def ultima_rueda_de_semana(d) -> date:
    """Ultimo dia habil NYSE de la semana W-FRI (sabado a viernes) que contiene `d`."""
    d = pd.Timestamp(d).date()
    dia = d + timedelta(days=(4 - d.weekday()) % 7)   # el viernes de esa semana
    while not is_trading_day(dia):
        dia -= timedelta(days=1)
    return dia


def excluir_semana_incompleta(weekly: pd.DataFrame, col: str = "fecha_semana") -> pd.DataFrame:
    """
    Saca la ULTIMA semana si su ultimo dato no es la ultima rueda habil de esa semana.
    `weekly` ordenado ascendente, con `col` = ultimo dia con dato de cada semana.
    Las semanas anteriores ya terminaron: no se tocan aunque les falte un dia.
    """
    if weekly is None or len(weekly) == 0:
        return weekly
    ultima = pd.Timestamp(weekly[col].iloc[-1]).date()
    if ultima < ultima_rueda_de_semana(ultima):
        return weekly.iloc[:-1].copy()
    return weekly


def resample_close_semanal(df_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Resamplea precios diarios a cierres semanales (W-FRI), excluyendo la ultima
    semana si esta incompleta en los datos (excluir_semana_incompleta). Solo
    necesita columnas [fecha, close] (el RSI/MACD usan el close).

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

    weekly = excluir_semana_incompleta(weekly).copy()

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
