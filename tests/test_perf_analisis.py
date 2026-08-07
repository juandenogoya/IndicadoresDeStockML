"""
test_perf_analisis.py -- tests del modulo PURO src/utils/perf_analisis.py.

Casos sinteticos: no tocan la DB. Verifican alineacion por fecha, equity
base 100, cartera equiponderada, y que las metricas de riesgo salgan de
ft_metricas (coherencia de signo/valor), sin numeros magicos.
"""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from src.utils import perf_analisis


def _serie(ticker, precios, f0=date(2024, 1, 1)):
    """DataFrame [ticker, fecha, close] con una rueda por dia calendario."""
    filas = [{"ticker": ticker, "fecha": f0 + timedelta(days=i), "close": p}
             for i, p in enumerate(precios)]
    return pd.DataFrame(filas)


def _bench(precios, f0=date(2024, 1, 1)):
    filas = [{"fecha": f0 + timedelta(days=i), "close": p}
             for i, p in enumerate(precios)]
    return pd.DataFrame(filas)


def test_insuficiente_sin_tickers():
    res = perf_analisis.construir_analisis(pd.DataFrame(), _bench([1, 2]), [])
    assert res["insuficiente"] is True


def test_ticker_sin_precios_va_a_descartados():
    precios = _serie("AAA", [100, 101, 102])
    res = perf_analisis.construir_analisis(precios, _bench([10, 11, 12]),
                                           ["AAA", "ZZZ"])
    assert res["insuficiente"] is False
    assert res["tickers"] == ["AAA"]
    assert res["descartados"] == ["ZZZ"]


def test_ventana_comun_es_interseccion_de_fechas():
    # AAA arranca 1/1, BBB arranca 3/1 -> la ventana comun arranca 3/1.
    aaa = _serie("AAA", [100, 101, 102, 103, 104], f0=date(2024, 1, 1))
    bbb = _serie("BBB", [50, 51, 52], f0=date(2024, 1, 3))
    precios = pd.concat([aaa, bbb], ignore_index=True)
    bench = _bench([10, 10, 10, 10, 10], f0=date(2024, 1, 1))
    res = perf_analisis.construir_analisis(precios, bench, ["AAA", "BBB"])
    assert res["desde"] == date(2024, 1, 3)
    assert res["hasta"] == date(2024, 1, 5)
    assert res["n_dias"] == 3


def test_equity_base100_un_ticker():
    # +10% cada dia sobre 2 pasos -> base 100 termina en 121.
    precios = _serie("AAA", [100, 110, 121])
    bench = _bench([100, 100, 100])
    res = perf_analisis.construir_analisis(precios, bench, ["AAA"], rf_anual=0.0)
    serie = res["series"]
    eq = serie[serie["serie"] == "AAA"].sort_values("fecha")["valor"].tolist()
    assert eq[0] == pytest.approx(100.0)
    assert eq[-1] == pytest.approx(121.0, abs=1e-6)
    assert res["metricas"]["retorno_cartera_pct"] == pytest.approx(21.0, abs=1e-3)


def test_cartera_equiponderada_promedia_retornos():
    # AAA +10%/dia, BBB 0%/dia -> cartera 50/50 ~ +5%/dia.
    aaa = _serie("AAA", [100, 110, 121])
    bbb = _serie("BBB", [100, 100, 100])
    precios = pd.concat([aaa, bbb], ignore_index=True)
    bench = _bench([100, 100, 100])
    res = perf_analisis.construir_analisis(precios, bench, ["AAA", "BBB"],
                                           rf_anual=0.0)
    assert res["label"] == "Cartera"
    assert res["pesos"]["AAA"] == pytest.approx(0.5)
    # retorno diario de la cartera = (0.10 + 0.0)/2 = 0.05 -> 2 pasos = 10.25%
    assert res["metricas"]["retorno_cartera_pct"] == pytest.approx(10.25, abs=1e-2)


def test_rf_line_crece_a_la_tasa():
    precios = _serie("AAA", [100, 100, 100, 100])
    bench = _bench([100, 100, 100, 100])
    res = perf_analisis.construir_analisis(precios, bench, ["AAA"], rf_anual=0.05)
    serie = res["series"]
    rf = serie[serie["serie"] == perf_analisis.RF_LABEL].sort_values("fecha")["valor"].tolist()
    # cada paso crece (1.05)^(1/252)-1; monotona creciente y > 100 al final.
    assert rf[0] == pytest.approx(100.0)
    assert rf[-1] > 100.0
    assert all(b >= a for a, b in zip(rf, rf[1:]))


def test_beta_uno_cuando_ticker_igual_a_benchmark():
    # Ticker identico al benchmark -> beta ~ 1, exceso vs bench ~ 0.
    px = [100, 105, 103, 108, 110]
    precios = _serie("AAA", px)
    bench = _bench(px)
    res = perf_analisis.construir_analisis(precios, bench, ["AAA"], rf_anual=0.0)
    assert res["metricas"]["beta"] == pytest.approx(1.0, abs=1e-6)
    assert res["metricas"]["exceso_vs_bench_pct"] == pytest.approx(0.0, abs=1e-3)


def test_series_tiene_las_tres_lineas():
    precios = _serie("AAA", [100, 101, 102])
    bench = _bench([100, 100, 100])
    res = perf_analisis.construir_analisis(precios, bench, ["AAA"])
    series_presentes = set(res["series"]["serie"].unique())
    assert series_presentes == {"AAA", perf_analisis.BENCH_LABEL, perf_analisis.RF_LABEL}
