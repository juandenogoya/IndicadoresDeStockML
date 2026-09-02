"""
performance.py -- Vista "Performance" del dashboard.

Compara la performance buy-and-hold de un ticker o cartera contra el benchmark
(futuros ES) y contra la tasa libre de riesgo, sobre la ventana comun de datos.

SOLO lectura: lee precios_diarios + futuros_diarios y delega TODO el calculo al
modulo PURO src/utils/perf_analisis.py (que a su vez usa ft_metricas). No recalcula
metricas aca.

Descriptivo, NO predictivo: las conclusiones valen para la ventana observada.
"""

import pandas as pd
import altair as alt
import streamlit as st

from src.data.database import query_df
from src.utils import perf_analisis

BENCH_TICKER = "ES=F"

COLOR_DOM = None  # se arma dinamico segun el label de la cartera


@st.cache_data(ttl=600)
def _cargar_precios(tickers):
    """Cierres diarios de los tickers seleccionados."""
    if not tickers:
        return pd.DataFrame(columns=["ticker", "fecha", "close"])
    return query_df(
        "SELECT ticker, fecha, close FROM precios_diarios "
        "WHERE ticker = ANY(:tks) ORDER BY ticker, fecha",
        params={"tks": list(tickers)})


@st.cache_data(ttl=600)
def _cargar_bench():
    """Cierres diarios del benchmark (futuros ES)."""
    return query_df(
        "SELECT fecha, close FROM futuros_diarios WHERE ticker = :b ORDER BY fecha",
        params={"b": BENCH_TICKER})


@st.cache_data(ttl=600)
def _listar_tickers_precios():
    df = query_df("SELECT DISTINCT ticker FROM precios_diarios ORDER BY ticker")
    return df["ticker"].tolist()


def _grafico(series, label):
    """Lineas de equity base 100: cartera vs benchmark vs tasa libre."""
    dom = [label, perf_analisis.BENCH_LABEL, perf_analisis.RF_LABEL]
    rng = ["#4c8ac3", "#e0a44c", "#8a8a8a"]
    chart = (
        alt.Chart(series)
        .mark_line()
        .encode(
            x=alt.X("fecha:T", title=None),
            y=alt.Y("valor:Q", title="Base 100",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("serie:N", title=None,
                            scale=alt.Scale(domain=dom, range=rng), sort=dom),
            tooltip=["fecha:T", "serie:N", alt.Tooltip("valor:Q", format=".1f")],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def _fmt_pct(v):
    """Con signo (+/-): para retornos, excesos y alfa."""
    return "-" if v is None else f"{v:+.1f}%"


def _fmt_mag(v):
    """Sin signo: para magnitudes no direccionales (volatilidad)."""
    return "-" if v is None else f"{v:.1f}%"


def _fmt_dd(v):
    """Drawdown: se guarda positivo pero ES una caida -> mostrar negativo."""
    return "-" if v is None else f"-{v:.1f}%"


def _tabla_metricas(m, label):
    """Tabla de metricas ajustadas por riesgo (2 columnas: metrica | valor)."""
    sh = m.get("sharpe") or {}
    sharpe_txt = "-"
    if sh.get("sharpe") is not None:
        ic = ""
        if sh.get("ic95_lo") is not None:
            marca = "" if sh.get("concluyente") else "  (IC cruza 0: no concluyente)"
            ic = f"  [IC95% {sh['ic95_lo']:.2f} a {sh['ic95_hi']:.2f}]{marca}"
        sharpe_txt = f"{sh['sharpe']:.2f}{ic}"

    filas = [
        ("Retorno " + label, _fmt_pct(m["retorno_cartera_pct"])),
        ("Retorno benchmark (ES)", _fmt_pct(m["retorno_bench_pct"])),
        ("Retorno tasa libre", _fmt_pct(m["retorno_rf_pct"])),
        ("Exceso vs benchmark", _fmt_pct(m["exceso_vs_bench_pct"])),
        ("Exceso vs tasa libre", _fmt_pct(m["exceso_vs_rf_pct"])),
        ("Alfa de Jensen (anual)", _fmt_pct(m["alfa_jensen_anual_pct"])),
        ("Volatilidad anual", _fmt_mag(m["volatilidad_anual_pct"])),
        ("Max drawdown", _fmt_dd(m["max_dd_pct"])),
        ("Sharpe (anual)", sharpe_txt),
        ("Sortino (anual)", "-" if m["sortino"] is None else f"{m['sortino']:.2f}"),
        ("Beta vs ES", "-" if m["beta"] is None else f"{m['beta']:.2f}"),
        ("Information ratio", "-" if m["information_ratio"] is None
         else f"{m['information_ratio']:.2f}"),
    ]
    st.dataframe(pd.DataFrame(filas, columns=["Metrica", "Valor"]),
                 hide_index=True, width="stretch")


def _veredicto(m, label):
    """Lectura textual honesta, siempre acotada a la ventana."""
    partes = []
    ex_b = m["exceso_vs_bench_pct"]
    ex_rf = m["exceso_vs_rf_pct"]
    if ex_b is not None:
        partes.append(f"le {'gano' if ex_b >= 0 else 'perdio'} al benchmark "
                      f"por {abs(ex_b):.1f} pp")
    partes.append(f"{'supero' if ex_rf >= 0 else 'quedo debajo de'} la tasa libre "
                  f"por {abs(ex_rf):.1f} pp")
    sh = m.get("sharpe") or {}
    if sh.get("sharpe") is not None and not sh.get("concluyente"):
        partes.append("el Sharpe NO es concluyente (su IC95% cruza cero)")
    return f"En la ventana, {label} " + "; ".join(partes) + "."


def construir_performance(tickers=None):
    st.subheader("Performance -- ticker/cartera vs benchmark + tasa libre")
    st.caption("Buy-and-hold sobre la ventana comun de datos. Cartera = "
               "equiponderada, rebalanceo diario. Benchmark = futuros ES. "
               "Descriptivo, no una recomendacion: las conclusiones valen SOLO "
               "para el periodo mostrado.")

    universo = _listar_tickers_precios()
    if not universo:
        st.error("No hay tickers en precios_diarios.")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        default = [t for t in ["AAPL"] if t in universo] or universo[:1]
        sel = st.multiselect("Tickers (1 o varios = cartera)", universo,
                             default=default, key="perf_tks")
    with c2:
        rf_pct = st.number_input("Tasa libre anual (%)", min_value=0.0,
                                 max_value=25.0, value=5.0, step=0.5, key="perf_rf")

    if not sel:
        st.info("Elegi al menos un ticker.")
        return

    precios = _cargar_precios(tuple(sorted(sel)))
    bench = _cargar_bench()
    if bench.empty:
        st.error(f"Sin datos de benchmark {BENCH_TICKER} en futuros_diarios.")
        return

    res = perf_analisis.construir_analisis(precios, bench, sel, rf_anual=rf_pct / 100.0)

    if res.get("insuficiente"):
        st.warning(f"Ventana insuficiente: {res.get('motivo')}.")
        return

    label = res["label"]
    if res["descartados"]:
        st.caption(f"Sin precios (excluidos): {', '.join(res['descartados'])}.")
    if len(res["tickers"]) > 1:
        st.caption(f"Cartera equiponderada de {len(res['tickers'])}: "
                   f"{', '.join(res['tickers'])}.")

    st.markdown(f"**Ventana:** {res['desde']} -> {res['hasta']}  "
                f"({res['n_dias']} ruedas)  |  tasa libre {rf_pct:.1f}% anual")

    _grafico(res["series"], label)

    col_a, col_b = st.columns([2, 3])
    with col_a:
        _tabla_metricas(res["metricas"], label)
    with col_b:
        st.markdown("**Lectura**")
        st.write(_veredicto(res["metricas"], label))
        st.caption("Alfa de Jensen: exceso de retorno que NO explica la exposicion "
                   "al mercado (beta). Sharpe con IC95%: si el intervalo cruza cero, "
                   "el ratio no distingue la seleccion del azar en esta ventana.")
