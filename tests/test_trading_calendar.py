"""
test_trading_calendar.py -- calendario NYSE (src/utils/trading_calendar.py).

Casos anclados en datos reales: la auditoria de huecos de precios_diarios del
12/9/2026 encontro los 200 tickers sin barra el 2025-01-09. NYSE cerro por el
duelo nacional de Jimmy Carter y el calendario no lo tenia.
"""

from datetime import date

from src.utils import trading_calendar as tc


def test_cierre_extraordinario_carter_2025():
    d = date(2025, 1, 9)
    assert not tc.is_trading_day(d)
    assert tc.holiday_name(d) is not None
    assert tc.prev_trading_day(date(2025, 1, 10)) == date(2025, 1, 8)
    assert tc.next_trading_day(date(2025, 1, 8)) == date(2025, 1, 10)
    semana = tc.trading_days_between(date(2025, 1, 6), date(2025, 1, 10))
    assert d not in semana
    assert len(semana) == 4


def test_feriado_y_fin_de_semana():
    assert not tc.is_trading_day(date(2026, 9, 7))    # Labor Day
    assert not tc.is_trading_day(date(2026, 9, 12))   # sabado
    assert tc.prev_trading_day(date(2026, 9, 8)) == date(2026, 9, 4)


def test_un_hueco_de_datos_no_es_feriado():
    # 2026-08-28: faltaban 157 tickers en precios_diarios, pero NYSE opero.
    assert tc.is_trading_day(date(2026, 8, 28))


def test_describe_date_marca_el_cierre():
    assert "FERIADO" in tc.describe_date(date(2025, 1, 9))
