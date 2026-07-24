"""
earnings_reaccion.py -- vista "Reaccion a balances" del dashboard.

Muestra, para un ticker, la reaccion REAL del precio y el volumen en las ruedas
posteriores a cada presentacion de balance (desde 2020). Sin estimaciones ni
sorpresa: solo el hecho duro (cuando reporto) cruzado con el comportamiento del
precio.

Fuentes (LOCAL):
    earnings_historico  -> fecha de anuncio + report_time (pre/post-market).
    precios_diarios     -> close + volumen (las ruedas de la ventana).

DIA 0 (primera rueda en que el mercado pudo reaccionar):
    - report_time 'post-market'  -> la rueda habil SIGUIENTE al anuncio
      (cuando se publico, el mercado ya estaba cerrado).
    - 'pre-market' / desconocido -> la rueda del propio anuncio (o la siguiente
      si ese dia no operó).
    Se resuelve contra las filas REALES de precios_diarios (que ya son dias
    habiles), sin depender del calendario: la reaccion es el primer close cuya
    fecha cumple la condicion.

Base del % : el close de la rueda ANTERIOR al dia 0 (ultimo precio "limpio"
    pre-reaccion). Asi el gap de apertura queda dentro de la ventana medida.
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.data.database import query_df

VENTANA_POST = 7    # ruedas a mostrar desde el dia 0 (inclusive)
VOL_BASE_N   = 20   # ruedas previas para el volumen promedio de referencia


# ── Datos ─────────────────────────────────────────────────────────────────────

def _eventos(ticker: str) -> pd.DataFrame:
    return query_df(
        "SELECT fiscal_period_end, announcement_date, report_time "
        "FROM earnings_historico WHERE ticker = :t "
        "ORDER BY announcement_date DESC",
        {"t": ticker},
    )


def _precios(ticker: str) -> pd.DataFrame:
    df = query_df(
        "SELECT fecha, close, volume FROM precios_diarios "
        "WHERE ticker = :t ORDER BY fecha",
        {"t": ticker},
    )
    if not df.empty:
        df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def _etiqueta_q(fpe: pd.Timestamp) -> str:
    """Etiqueta legible del trimestre por su cierre fiscal (ej '2025-Q4')."""
    q = (fpe.month - 1) // 3 + 1
    return f"{fpe.year}-Q{q}"


def construir_series(ticker: str, n_eventos: int) -> tuple:
    """
    Devuelve (serie_long, resumen):
      serie_long: 1 fila por (evento, offset) con precio_pct y vol_rel.
      resumen   : 1 fila por evento con gap y retorno acumulado de la ventana.
    """
    ev = _eventos(ticker)
    px = _precios(ticker)
    if ev.empty or px.empty:
        return pd.DataFrame(), pd.DataFrame()

    ev["fiscal_period_end"] = pd.to_datetime(ev["fiscal_period_end"])
    ev["announcement_date"] = pd.to_datetime(ev["announcement_date"])

    fechas = px["fecha"].tolist()
    filas, resumen = [], []

    for _, e in ev.head(n_eventos).iterrows():
        ann = e["announcement_date"]
        post = (e["report_time"] or "").lower() == "post-market"
        # indice de la primera rueda de reaccion
        idx = None
        for i, f in enumerate(fechas):
            if (f > ann) if post else (f >= ann):
                idx = i
                break
        if idx is None or idx == 0:
            continue                      # sin rueda previa o sin datos posteriores

        base_close = float(px.iloc[idx - 1]["close"])
        if base_close <= 0:
            continue
        # volumen promedio de referencia (ruedas previas al dia 0)
        prev = px.iloc[max(0, idx - VOL_BASE_N):idx]
        vol_avg = float(prev["volume"].mean()) if not prev.empty else None

        etiqueta = _etiqueta_q(e["fiscal_period_end"])
        ventana = px.iloc[idx: idx + VENTANA_POST]
        for k, (_, row) in enumerate(ventana.iterrows()):
            pct = float(row["close"]) / base_close - 1.0
            vol_rel = (float(row["volume"]) / vol_avg) if vol_avg else None
            filas.append({
                "quarter": etiqueta,
                "announcement_date": ann.date().isoformat(),
                "offset": k,
                "fecha": row["fecha"].date().isoformat(),
                "precio_pct": pct * 100.0,
                "volume": float(row["volume"]),
                "vol_rel": vol_rel,
            })

        if len(ventana) >= 1:
            gap = float(ventana.iloc[0]["close"]) / base_close - 1.0
            fin = float(ventana.iloc[-1]["close"]) / base_close - 1.0
            resumen.append({
                "quarter": etiqueta,
                "anuncio": ann.date().isoformat(),
                "timing": e["report_time"],
                "gap_dia0_%": round(gap * 100, 2),
                f"acum_{len(ventana)}r_%": round(fin * 100, 2),
            })

    return pd.DataFrame(filas), pd.DataFrame(resumen)


# ── Vista ─────────────────────────────────────────────────────────────────────

def construir_reaccion(tickers: list):
    st.title("Reaccion a balances")
    st.caption("Comportamiento REAL del precio y el volumen en las ruedas "
               "posteriores a cada balance. Dia 0 = primera rueda en que el "
               "mercado pudo reaccionar (ajustado por pre/post-market).")

    ticker = st.sidebar.selectbox("Ticker", tickers, key="er_ticker")
    n_ev = st.sidebar.slider("Trimestres a comparar", 2, 12, 4, key="er_nev")

    ev = _eventos(ticker)
    if ev.empty:
        st.warning(
            f"No hay historia local de balances para {ticker}.\n\n"
            "El backfill inicial corre en Oracle -> Railway; local se completa "
            "con el sync final. Para traer solo este ticker ya:\n"
            f"`python scripts/refresh_earnings_historico.py --ticker {ticker}`"
        )
        return

    serie, resumen = construir_series(ticker, n_ev)
    if serie.empty:
        st.warning("Hay fechas de balance pero faltan precios en la ventana "
                   "para cruzarlas.")
        return

    # ── Resumen de eventos ───────────────────────────────────────────────────
    st.subheader(f"{ticker} -- ultimos {len(resumen)} balances")
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    # ── Panel de PRECIO: % acumulado desde el close previo al dia 0 ──────────
    st.markdown("**Precio** -- variacion % desde el cierre previo al balance")
    base = alt.Chart(serie).encode(
        x=alt.X("offset:O", title="Rueda desde el dia 0"),
        color=alt.Color("quarter:N", title="Trimestre"),
    )
    linea_px = base.mark_line(point=True).encode(
        y=alt.Y("precio_pct:Q", title="% vs cierre previo"),
        tooltip=["quarter", "fecha", alt.Tooltip("precio_pct:Q", format="+.2f"),
                 alt.Tooltip("vol_rel:Q", format=".2f", title="vol x prom")],
    )
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(y="y:Q")
    st.altair_chart((linea_px + cero).properties(height=320),
                    use_container_width=True)

    # ── Panel de VOLUMEN: multiplo del promedio previo ───────────────────────
    st.markdown("**Volumen** -- multiplo del promedio de las 20 ruedas previas")
    if serie["vol_rel"].notna().any():
        linea_vol = base.mark_line(point=True, strokeDash=[2, 2]).encode(
            y=alt.Y("vol_rel:Q", title="volumen / promedio previo"),
            tooltip=["quarter", "fecha",
                     alt.Tooltip("vol_rel:Q", format=".2f", title="vol x prom"),
                     alt.Tooltip("volume:Q", format=",.0f")],
        )
        uno = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((linea_vol + uno).properties(height=260),
                        use_container_width=True)
    else:
        st.info("Sin volumen de referencia suficiente para el multiplo.")

    st.caption("Lectura: el panel de precio muestra si el ticker suele seguir o "
               "revertir el gap; el de volumen, si el interes se sostiene o se "
               "apaga en los dias siguientes.")
