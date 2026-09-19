"""
Tests del modulo puro src.utils.weekly_tf (semanal al vuelo).

Sin DB ni mocks: weekly_tf solo usa pandas + ta. Cubre el resample W-FRI,
la exclusion de la semana en curso, el calculo RSI/MACD y los casos borde
(historia insuficiente, entrada vacia).
"""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from src.utils import weekly_tf


def _serie_diaria(n_dias: int, inicio: date | None = None) -> list[dict]:
    """Genera n_dias de filas diarias (solo dias habiles L-V) con close creciente."""
    if inicio is None:
        # Arrancar suficientemente atras para tener >35 semanas cerradas.
        inicio = date.today() - timedelta(days=int(n_dias * 1.5) + 14)
    filas = []
    d = inicio
    close = 100.0
    while len(filas) < n_dias:
        if d.weekday() < 5:  # L-V
            close += 1.0
            filas.append({"fecha": d, "close": round(close, 2)})
        d += timedelta(days=1)
    return filas


def test_resample_excluye_semana_en_curso():
    """La semana en curso (incompleta) no debe aparecer en el resample."""
    hoy = date.today()
    lunes_actual = hoy - timedelta(days=hoy.weekday())
    # Filas que incluyen dias de la semana actual.
    filas = _serie_diaria(60)
    filas.append({"fecha": lunes_actual, "close": 999.0})  # dia de la semana en curso
    df = pd.DataFrame(filas)
    sem = weekly_tf.resample_close_semanal(df)
    assert not sem.empty
    assert all(f < lunes_actual for f in sem["fecha_semana"])


def test_resample_fecha_semana_es_date():
    df = pd.DataFrame(_serie_diaria(40))
    sem = weekly_tf.resample_close_semanal(df)
    assert isinstance(sem["fecha_semana"].iloc[-1], date)
    assert sem["close"].dtype == float


def test_resample_vacio():
    assert weekly_tf.resample_close_semanal(pd.DataFrame(columns=["fecha", "close"])).empty
    assert weekly_tf.resample_close_semanal(None).empty


def test_rsi_macd_historia_suficiente():
    """Con >35 semanas, devuelve rsi/macd/macd_signal numericos."""
    filas = _serie_diaria(300)  # ~60 semanas
    out = weekly_tf.tecnico_semanal(filas)
    assert out, "esperaba bloque semanal no vacio"
    assert set(out) == {"fecha", "rsi", "macd", "macd_signal"}
    assert isinstance(out["rsi"], float) and not math.isnan(out["rsi"])
    # Serie estrictamente creciente -> RSI alto y MACD positivo.
    assert out["rsi"] > 60
    assert out["macd"] > out["macd_signal"]
    assert isinstance(out["fecha"], date)


def test_rsi_macd_historia_insuficiente():
    """Menos de 35 semanas -> {} (no se inventa el semanal)."""
    filas = _serie_diaria(40)  # ~8 semanas
    assert weekly_tf.tecnico_semanal(filas) == {}


def test_tecnico_semanal_entrada_invalida():
    assert weekly_tf.tecnico_semanal(None) == {}
    assert weekly_tf.tecnico_semanal([]) == {}
    assert weekly_tf.tecnico_semanal([{"fecha": date.today()}]) == {}  # sin close


def test_rsi_macd_semanal_directo():
    """rsi_macd_semanal acepta una serie de cierres y respeta el minimo."""
    assert weekly_tf.rsi_macd_semanal([1, 2, 3]) == {}
    closes = list(range(1, 60))
    out = weekly_tf.rsi_macd_semanal(closes, fecha_semana=date(2026, 1, 2))
    assert out["fecha"] == date(2026, 1, 2)
    assert out["rsi"] is not None and out["macd"] is not None


# -- semana completa por DATO, no por reloj (17/9/2026) ---------------------------------

def _dias_habiles(desde: date, hasta: date) -> list[dict]:
    filas, d, close = [], desde, 100.0
    while d <= hasta:
        if d.weekday() < 5:
            close += 1.0
            filas.append({"fecha": d, "close": close})
        d += timedelta(days=1)
    return filas


def test_ultima_rueda_de_semana():
    assert weekly_tf.ultima_rueda_de_semana(date(2026, 9, 14)) == date(2026, 9, 18)  # lunes
    assert weekly_tf.ultima_rueda_de_semana(date(2026, 9, 19)) == date(2026, 9, 25)  # sabado -> semana siguiente
    assert weekly_tf.ultima_rueda_de_semana(date(2026, 4, 1)) == date(2026, 4, 2)    # Viernes Santo 3/4


def test_semana_con_dato_del_viernes_cuenta_aunque_hoy_sea_viernes():
    # La rutina del viernes a la noche: antes se excluia por reloj.
    sem = weekly_tf.resample_close_semanal(pd.DataFrame(
        _dias_habiles(date(2026, 8, 3), date(2026, 9, 11))))
    assert sem["fecha_semana"].iloc[-1] == date(2026, 9, 11)


def test_semana_sin_el_viernes_se_excluye():
    sem = weekly_tf.resample_close_semanal(pd.DataFrame(
        _dias_habiles(date(2026, 8, 3), date(2026, 9, 10))))
    assert sem["fecha_semana"].iloc[-1] == date(2026, 9, 4)


def test_semana_de_viernes_santo_cierra_el_jueves():
    filas = [f for f in _dias_habiles(date(2026, 3, 2), date(2026, 4, 2))]
    sem = weekly_tf.resample_close_semanal(pd.DataFrame(filas))
    assert sem["fecha_semana"].iloc[-1] == date(2026, 4, 2)


def test_semanas_anteriores_no_se_tocan_aunque_les_falte_un_dia():
    filas = [f for f in _dias_habiles(date(2026, 8, 3), date(2026, 9, 11))
             if f["fecha"] != date(2026, 8, 21)]          # falta un viernes en el medio
    sem = weekly_tf.resample_close_semanal(pd.DataFrame(filas))
    assert date(2026, 8, 20) in set(sem["fecha_semana"])
