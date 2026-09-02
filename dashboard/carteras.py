"""
carteras.py -- Vista "Carteras" del dashboard (perfiles de riesgo del universo).

Consume la tabla perfiles_ticker (Fase 3). SOLO lectura, no recalcula: el
computo vive en scripts/compute_perfiles_carteras.py (mensual). Tres vistas:

  - Mapa:                  scatter beta vs ATR% mensual, coloreado por perfil.
  - Por cartera:           tickers de un perfil, rankeados intra-caja.
  - Excepciones / Validacion: los que se despegan de su sector + chequeo de las
                           anclas del usuario (Fase 4, validacion).

Modelo (docs/perfiles_carteras.md): perfil = comportamiento cuantitativo puro
(percentil composite -> cuartil); sector = contexto + flag de excepcion.
"""

import pandas as pd
import altair as alt
import streamlit as st

from src.data.database import query_df

PERFILES = ["Conservadora", "Moderada", "Arriesgada", "Especulativa"]

# Escala de color consistente (tranquilo -> caliente).
COLOR_DOM = PERFILES
COLOR_RNG = ["#4cc38a", "#4c8ac3", "#e0a44c", "#e05d5d"]

# Anclas del usuario = conjunto de validacion (docs seccion 3). NO alimentan la
# clasificacion; solo se chequea que caigan donde deben.
ANCLAS = {
    "Conservadora": ["KO", "WMT", "PG", "JNJ", "MCD"],
    "Moderada":     ["JPM", "BAC", "CAT", "DE", "LMT"],
    "Arriesgada":   ["AAPL", "MSFT", "META", "AMZN"],
    "Especulativa": ["NVDA", "AMD", "AVGO", "ARM", "MU"],
}

_COLS_TABLA = ["rank_en_caja", "ticker", "sector", "industry", "score_riesgo",
               "atr_pct_w", "atr_pct_m", "beta", "max_dd_1a", "excepcion"]


@st.cache_data(ttl=600)
def _cargar():
    """Ultimo snapshot de perfiles_ticker (o vacio si la tabla no existe)."""
    try:
        return query_df(
            "SELECT * FROM perfiles_ticker "
            "WHERE fecha = (SELECT MAX(fecha) FROM perfiles_ticker)")
    except Exception:
        return pd.DataFrame()


def _color():
    return alt.Color("perfil:N", title="Perfil",
                     scale=alt.Scale(domain=COLOR_DOM, range=COLOR_RNG),
                     sort=PERFILES)


def _mapa(df):
    st.markdown("**Mapa del universo** -- cada punto es un ticker; el color es su "
                "perfil (comportamiento). Eje X: sensibilidad al mercado (beta). "
                "Eje Y: volatilidad mensual (ATR%).")

    d = df.dropna(subset=["beta", "atr_pct_m"]).copy()
    sin_beta = len(df) - len(d)

    scatter = (
        alt.Chart(d)
        .mark_circle(size=90, opacity=0.75)
        .encode(
            x=alt.X("beta:Q", title="Beta (vs futuros ES)"),
            y=alt.Y("atr_pct_m:Q", title="ATR% mensual"),
            color=_color(),
            tooltip=["ticker", "sector", "perfil", "score_riesgo",
                     "beta", "atr_pct_m", "atr_pct_w", "max_dd_1a"],
        )
        .properties(height=460)
        .interactive()
    )
    st.altair_chart(scatter, width="stretch")
    if sin_beta:
        st.caption(f"{sin_beta} ticker(s) sin beta/vol suficiente quedan fuera del grafico.")

    # Distribucion por perfil.
    dist = (df["perfil"].value_counts()
            .reindex(PERFILES).fillna(0).astype(int).reset_index())
    dist.columns = ["perfil", "n"]
    barras = (
        alt.Chart(dist).mark_bar().encode(
            x=alt.X("perfil:N", sort=PERFILES, title=None),
            y=alt.Y("n:Q", title="Tickers"),
            color=_color(),
            tooltip=["perfil", "n"],
        ).properties(height=200)
    )
    st.altair_chart(barras, width="stretch")


def _tabla_perfil(sub):
    cols = [c for c in _COLS_TABLA if c in sub.columns]
    t = sub.sort_values("rank_en_caja", na_position="last")[cols].copy()
    st.dataframe(
        t, width="stretch", hide_index=True,
        column_config={
            "rank_en_caja": st.column_config.NumberColumn("rank", format="%d"),
            "score_riesgo": st.column_config.NumberColumn("score", format="%.1f"),
            "atr_pct_w": st.column_config.NumberColumn("ATR%w", format="%.1f"),
            "atr_pct_m": st.column_config.NumberColumn("ATR%m", format="%.1f"),
            "beta": st.column_config.NumberColumn("beta", format="%.2f"),
            "max_dd_1a": st.column_config.NumberColumn("DD 1a", format="%.1f"),
            "excepcion": st.column_config.CheckboxColumn("exc."),
        })


def _por_cartera(df):
    c1, c2 = st.columns([1, 3])
    with c1:
        perfil = st.radio("Cartera", PERFILES, key="cart_perfil")
    sub = df[df["perfil"] == perfil]
    with c2:
        st.markdown(f"**{perfil}** -- {len(sub)} tickers, ordenados por riesgo "
                    "dentro de la caja (rank 1 = mas caliente).")
    if sub.empty:
        st.info("Sin tickers en esta caja.")
        return
    _tabla_perfil(sub)


def _validacion(df):
    st.markdown("**Validacion de anclas** -- las referencias del usuario deberian "
                "caer en su caja. Un desajuste NO es error: refleja el "
                "comportamiento de la ventana (~15 meses).")
    by_tk = df.set_index("ticker")["perfil"].to_dict()
    filas = []
    for esperado, tks in ANCLAS.items():
        for tk in tks:
            obt = by_tk.get(tk)
            if obt is None:
                continue
            filas.append({"ticker": tk, "esperado": esperado, "obtenido": obt,
                          "coincide": "OK" if obt == esperado else "revisar"})
    val = pd.DataFrame(filas)
    n_ok = (val["coincide"] == "OK").sum()
    st.caption(f"Coinciden {n_ok} de {len(val)} anclas.")
    st.dataframe(val, width="stretch", hide_index=True)


def _excepciones(df):
    exc = df[df["excepcion"]].copy()
    st.markdown(f"**Excepciones** -- {len(exc)} tickers cuyo comportamiento se "
                "despega 2+ cajas de su sector (la senal mas interesante).")
    if not exc.empty:
        exc = exc.sort_values("score_riesgo", ascending=False)
        cols = ["ticker", "sector", "industry", "caja_base", "perfil", "movio",
                "score_riesgo", "atr_pct_m", "beta"]
        cols = [c for c in cols if c in exc.columns]
        show = exc[cols].copy()
        show["caja_base"] = show["caja_base"].map(
            {0: "Conservadora", 1: "Moderada", 2: "Arriesgada", 3: "Especulativa"})
        st.dataframe(
            show, width="stretch", hide_index=True,
            column_config={
                "industry": st.column_config.TextColumn("industria"),
                "caja_base": st.column_config.TextColumn("sector base"),
                "perfil": st.column_config.TextColumn("perfil (comportamiento)"),
                "movio": st.column_config.NumberColumn("despegue", format="%+d"),
                "score_riesgo": st.column_config.NumberColumn("score", format="%.1f"),
                "atr_pct_m": st.column_config.NumberColumn("ATR%m", format="%.1f"),
                "beta": st.column_config.NumberColumn("beta", format="%.2f"),
            })
    st.divider()
    _validacion(df)


def construir_carteras(tickers=None):
    st.subheader("Carteras -- perfil de riesgo del universo")

    df = _cargar()
    if df.empty:
        st.info("No hay perfiles calculados todavia. Corre "
                "`scripts/manual/compute_perfiles_carteras.bat` para poblar "
                "`perfiles_ticker`.")
        return

    fecha = pd.to_datetime(df["fecha"].iloc[0]).date()
    st.caption(f"Snapshot {fecha}. El PERFIL es el comportamiento cuantitativo "
               "(ATR% multi-TF + beta + drawdown, por percentil del universo). "
               "El SECTOR es contexto y marca las excepciones. Descriptivo, no "
               "una recomendacion de compra.")

    # segmented_control ya es horizontal por definicion: no acepta `horizontal`.
    modo = st.segmented_control("Vista",
                                ["Mapa", "Por cartera", "Excepciones / Validacion"],
                                default="Mapa", key="cart_modo")
    st.divider()
    if modo == "Mapa":
        _mapa(df)
    elif modo == "Por cartera":
        _por_cartera(df)
    else:
        _excepciones(df)
