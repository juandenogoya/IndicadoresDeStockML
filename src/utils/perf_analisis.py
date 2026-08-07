"""
perf_analisis.py
Performance de un ticker o cartera (buy-and-hold) vs benchmark + tasa libre.

Modulo PURO: pandas + ft_metricas. SIN config, SIN database, SIN side effects.
Recibe DataFrames de precios y devuelve series (para graficar) + metricas.

Modelo:
    - Buy-and-hold. La cartera es EQUIPONDERADA con REBALANCEO DIARIO: el retorno
      de la cartera cada dia es el promedio (ponderado) de los retornos de sus
      tickers. Es la eleccion mas simple y honesta para la v1.
    - Alineacion por fecha con INNER JOIN sobre TODOS los tickers + benchmark: la
      ventana efectiva arranca cuando TODOS los seleccionados tienen dato (un
      ticker que listo mas tarde acorta la ventana comun). Se reporta desde/hasta.
    - Todo el calculo de riesgo se delega a ft_metricas.resumen_riesgo (fuente
      unica de Sharpe/Sortino/beta/IR/maxDD). Aca solo se arma la equity base 100,
      la linea de la tasa libre y el alfa de Jensen.

Descriptivo, NO predictivo: las conclusiones valen para la ventana observada.
"""

import pandas as pd

from src.utils import ft_metricas

BENCH_LABEL = "Benchmark (ES)"
RF_LABEL = "Tasa libre"


def _media(xs):
    return sum(xs) / len(xs) if xs else None


def _equity_base100(retornos):
    """Serie de equity base 100 a partir de una lista de retornos diarios."""
    eq = [100.0]
    for r in retornos:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def construir_analisis(precios: pd.DataFrame, bench: pd.DataFrame,
                       tickers, rf_anual: float = 0.05, pesos=None) -> dict:
    """
    Args:
        precios: DataFrame [ticker, fecha, close] de los tickers seleccionados.
        bench:   DataFrame [fecha, close] del benchmark (ES).
        tickers: lista de tickers a incluir en la cartera.
        rf_anual: tasa libre de riesgo anual (fraccion, ej. 0.05 = 5%).
        pesos:   dict {ticker: peso} opcional. Default equiponderado.

    Returns:
        dict. Si no hay ventana suficiente: {"insuficiente": True, "motivo": ...}.
        Si OK: series (long para altair), metricas (de resumen_riesgo + alfa),
        ventana (desde/hasta/n_dias), tickers usados/descartados, label, pesos.
    """
    tickers = list(dict.fromkeys(tickers))   # dedup preservando orden
    if not tickers:
        return {"insuficiente": True, "motivo": "sin tickers"}

    px = precios[["ticker", "fecha", "close"]].copy()
    px["fecha"] = pd.to_datetime(px["fecha"])
    px["close"] = px["close"].astype(float)
    disponibles = set(px["ticker"].unique())
    presentes = [t for t in tickers if t in disponibles]
    descartados = [t for t in tickers if t not in disponibles]
    if not presentes:
        return {"insuficiente": True, "motivo": "ningun ticker con precios",
                "descartados": descartados}

    wide = (px[px["ticker"].isin(presentes)]
            .pivot(index="fecha", columns="ticker", values="close")
            .sort_index())

    b = bench[["fecha", "close"]].copy()
    b["fecha"] = pd.to_datetime(b["fecha"])
    b["close"] = b["close"].astype(float)
    b = b.rename(columns={"close": "_bench"}).set_index("fecha").sort_index()

    df = wide.join(b, how="inner").dropna().sort_index()
    if len(df) < 2:
        return {"insuficiente": True, "motivo": "ventana comun < 2 ruedas",
                "tickers": presentes, "descartados": descartados}

    # Pesos (equiponderado por default; se normalizan a suma 1 sobre presentes).
    if pesos:
        w = {t: float(pesos.get(t, 0.0)) for t in presentes}
        s = sum(w.values())
        w = ({t: v / s for t, v in w.items()} if s > 0
             else {t: 1.0 / len(presentes) for t in presentes})
    else:
        w = {t: 1.0 / len(presentes) for t in presentes}

    # Retornos diarios alineados.
    rets = df[presentes].pct_change()
    r_port = sum(rets[t] * w[t] for t in presentes).iloc[1:]      # dropna del 1er dia
    r_bench = df["_bench"].pct_change().iloc[1:]

    r_port_list = r_port.tolist()
    r_bench_list = r_bench.tolist()

    equity_port = _equity_base100(r_port_list)
    equity_bench = _equity_base100(r_bench_list)
    rf_d = ft_metricas.rf_diaria(rf_anual)
    equity_rf = [100.0 * (1.0 + rf_d) ** i for i in range(len(df))]

    # Metricas de riesgo (fuente unica) sobre la equity de la cartera.
    resumen = ft_metricas.resumen_riesgo(
        equity_port, retornos_bench=r_bench_list, rf_anual=rf_anual)

    # Alfa de Jensen anualizado (en %): exceso que no explica la beta.
    beta = resumen.get("beta")
    alfa_anual_pct = None
    if beta is not None:
        alfa_d = (_media(r_port_list) - rf_d) - beta * (_media(r_bench_list) - rf_d)
        alfa_anual_pct = round(alfa_d * ft_metricas.DIAS_HABILES_ANIO * 100, 4)

    ret_cartera = resumen["retorno_total_pct"]
    ret_bench = resumen.get("benchmark_retorno_pct")
    ret_rf = round(equity_rf[-1] - 100.0, 4)

    label = presentes[0] if len(presentes) == 1 else "Cartera"
    fechas = [d.date() for d in df.index]

    # Series en formato LONG para altair.
    filas = []
    for f, v in zip(fechas, equity_port):
        filas.append({"fecha": f, "serie": label, "valor": round(v, 4)})
    for f, v in zip(fechas, equity_bench):
        filas.append({"fecha": f, "serie": BENCH_LABEL, "valor": round(v, 4)})
    for f, v in zip(fechas, equity_rf):
        filas.append({"fecha": f, "serie": RF_LABEL, "valor": round(v, 4)})
    series = pd.DataFrame(filas)

    metricas = {
        "retorno_cartera_pct": ret_cartera,
        "retorno_bench_pct": ret_bench,
        "retorno_rf_pct": ret_rf,
        "exceso_vs_bench_pct": (round(ret_cartera - ret_bench, 4)
                                if ret_bench is not None else None),
        "exceso_vs_rf_pct": round(ret_cartera - ret_rf, 4),
        "alfa_jensen_anual_pct": alfa_anual_pct,
        "volatilidad_anual_pct": resumen.get("volatilidad_anual"),
        "max_dd_pct": resumen.get("max_dd_pct"),
        "sharpe": resumen.get("sharpe"),          # dict con IC95% + concluyente
        "sortino": resumen.get("sortino"),
        "beta": beta,
        "information_ratio": resumen.get("information_ratio"),
    }

    return {
        "insuficiente": False,
        "label": label,
        "tickers": presentes,
        "descartados": descartados,
        "pesos": w,
        "rf_anual": rf_anual,
        "desde": fechas[0],
        "hasta": fechas[-1],
        "n_dias": len(df),
        "series": series,
        "metricas": metricas,
    }
