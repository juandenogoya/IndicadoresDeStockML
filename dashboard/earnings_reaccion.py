"""
earnings_reaccion.py -- vista "Reaccion a balances" del dashboard.

Muestra, para un ticker, la reaccion REAL del precio y el volumen en una ventana
SIMETRICA alrededor de cada presentacion de balance (desde 2020): N ruedas ANTES
(el run-up hacia el balance) y N ruedas DESDE el dia 0 (la reaccion y el
follow-through). Sin estimaciones ni sorpresa: solo el hecho duro (cuando
reporto) cruzado con el comportamiento del precio.

Fuentes (LOCAL):
    earnings_historico  -> fecha de anuncio + report_time (pre/post-market).
    precios_diarios     -> close + volumen (las ruedas de la ventana).

DIA 0 (primera rueda en que el mercado pudo reaccionar):
    - report_time 'post-market'  -> la rueda habil SIGUIENTE al anuncio
      (cuando se publico, el mercado ya estaba cerrado).
    - 'pre-market' / desconocido -> la rueda del propio anuncio (o la siguiente
      si ese dia no opero).
    Se resuelve contra las filas REALES de precios_diarios (que ya son dias
    habiles), sin depender del calendario: la reaccion es el primer close cuya
    fecha cumple la condicion.

VENTANA (parametrizable, N de 1 a 10):
    - PRE : offsets -N .. -1  (las N ruedas anteriores al dia 0).
    - POST: offsets  0 .. N-1 (el dia 0 CUENTA como post, mas las N-1 siguientes).
    El dia 0 es el pivote; una linea vertical lo marca en los graficos.

Filtro por ANIO (multi-seleccion, por cierre fiscal del trimestre): permite ver
un anio, varios, o todos, para no mezclar demasiados periodos en el overlay.

TRES paneles:
    1. Precio (USD)  -- el cierre REAL en dolares (sin normalizar). Cada trimestre
       se ubica en su banda de precio; sirve para ver el movimiento en crudo.
    2. Precio (%)    -- variacion acumulada vs el cierre previo al dia 0 (0% = el
       ultimo precio limpio antes de la reaccion). Normaliza para comparar
       trimestres en la misma escala.
    3. Volumen       -- multiplo del promedio de VOL_BASE_N ruedas previas a la
       ventana. "1.0" = volumen normal previo al evento.
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.data.database import query_df

N_DEFAULT  = 7    # ruedas por lado (pre y post) por defecto; slider 1..10
VOL_BASE_N = 50   # ruedas previas a la ventana para el volumen promedio de ref.


# -- Datos --------------------------------------------------------------------

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


def construir_series(ticker: str, anios: list, trimestres: list,
                     n_ruedas: int) -> tuple:
    """
    Devuelve (serie_long, resumen) para los balances cuyo cierre fiscal cae en
    alguno de los `anios` Y en alguno de los `trimestres` (1..4) seleccionados
    (ambos filtros combinan con AND; ej. Q1 de varios anios = estacionalidad):
      serie_long: 1 fila por (evento, offset) con close, precio_pct y vol_rel,
                  offsets de -n_ruedas a n_ruedas-1 (pre + post, dia 0 = offset 0).
      resumen   : 1 fila por evento con run-up pre, gap dia 0 y acumulado post.
    """
    ev = _eventos(ticker)
    px = _precios(ticker)
    if ev.empty or px.empty:
        return pd.DataFrame(), pd.DataFrame()

    ev["fiscal_period_end"] = pd.to_datetime(ev["fiscal_period_end"])
    ev["announcement_date"] = pd.to_datetime(ev["announcement_date"])
    if anios:
        ev = ev[ev["fiscal_period_end"].dt.year.isin(anios)]
    if trimestres:
        q_col = (ev["fiscal_period_end"].dt.month - 1) // 3 + 1
        ev = ev[q_col.isin(trimestres)]
    if ev.empty:
        return pd.DataFrame(), pd.DataFrame()

    fechas = px["fecha"].tolist()
    n = len(px)
    N = int(n_ruedas)
    filas, resumen = [], []

    for _, e in ev.iterrows():
        ann = e["announcement_date"]
        post = (e["report_time"] or "").lower() == "post-market"
        # indice de la primera rueda de reaccion (dia 0)
        idx = None
        for i, f in enumerate(fechas):
            if (f > ann) if post else (f >= ann):
                idx = i
                break
        if idx is None or idx == 0:
            continue                      # sin rueda previa (base) o sin datos

        base_close = float(px.iloc[idx - 1]["close"])
        if base_close <= 0:
            continue

        # volumen de referencia: VOL_BASE_N ruedas ANTES de la ventana pre.
        ini_pre = max(0, idx - N)          # primer indice de la ventana pre
        base_vol = px.iloc[max(0, ini_pre - VOL_BASE_N):ini_pre]
        vol_avg = float(base_vol["volume"].mean()) if not base_vol.empty else None

        etiqueta = _etiqueta_q(e["fiscal_period_end"])

        # ventana simetrica: offsets -N .. N-1 (los que existan en la serie).
        j0, j1 = max(0, idx - N), min(n, idx + N)
        for j in range(j0, j1):
            row = px.iloc[j]
            pct = float(row["close"]) / base_close - 1.0
            vol_rel = (float(row["volume"]) / vol_avg) if vol_avg else None
            filas.append({
                "quarter": etiqueta,
                "announcement_date": ann.date().isoformat(),
                "offset": j - idx,        # <0 pre, 0 dia 0, >0 post
                "fase": "pre" if j < idx else "post",
                "fecha": row["fecha"].date().isoformat(),
                "close": float(row["close"]),
                "precio_pct": pct * 100.0,
                "volume": float(row["volume"]),
                "vol_rel": vol_rel,
            })

        # resumen del evento
        run_up = (base_close / float(px.iloc[ini_pre]["close"]) - 1.0) \
            if ini_pre < idx else None
        gap = float(px.iloc[idx]["close"]) / base_close - 1.0
        fin_idx = min(n - 1, idx + N - 1)
        post_fin = float(px.iloc[fin_idx]["close"]) / base_close - 1.0
        resumen.append({
            "quarter": etiqueta,
            "anuncio": ann.date().isoformat(),
            "timing": e["report_time"],
            "run_up_pre_%": round(run_up * 100, 2) if run_up is not None else None,
            "gap_dia0_%": round(gap * 100, 2),
            "acum_post_%": round(post_fin * 100, 2),
        })

    return pd.DataFrame(filas), pd.DataFrame(resumen)


# -- Vista --------------------------------------------------------------------

def _divisor():
    """Linea vertical punteada que separa pre (offset<0) de post (offset>=0)."""
    return alt.Chart(pd.DataFrame({"x": [-0.5]})).mark_rule(
        strokeDash=[6, 3], color="#888").encode(x="x:Q")


def construir_reaccion(tickers: list):
    st.title("Reaccion a balances")
    st.caption("Comportamiento REAL del precio y el volumen en una ventana "
               "simetrica alrededor de cada balance. Dia 0 = primera rueda en "
               "que el mercado pudo reaccionar (ajustado por pre/post-market); "
               "cuenta como la primera rueda post. La linea punteada separa el "
               "antes del despues.")

    # key `ticker` UNICA en todo el dashboard + bind a la URL: el ticker elegido
    # en el informe o en el radar llega ya seleccionado aca (y viceversa).
    ticker = st.sidebar.selectbox("Ticker", tickers, key="ticker",
                                  bind="query-params")

    ev = _eventos(ticker)
    if ev.empty:
        st.warning(
            f"No hay historia local de balances para {ticker}.\n\n"
            "Para traer solo este ticker:\n"
            f"`python scripts/refresh_earnings_historico.py --ticker {ticker}`"
        )
        return

    anios_disp = sorted(
        {pd.to_datetime(d).year for d in ev["fiscal_period_end"]}, reverse=True)
    anios_sel = st.sidebar.multiselect(
        "Anios (por cierre fiscal)", anios_disp, default=anios_disp[:1],
        key="er_anios",
        help="Elegi uno o varios anios. Vacio = ninguno; seleccionalos todos "
             "para ver la historia completa.")
    q_sel = st.sidebar.pills(
        "Trimestres", [1, 2, 3, 4], selection_mode="multi",
        default=[1, 2, 3, 4], format_func=lambda q: f"Q{q}", key="er_q",
        help="Toggle por trimestre (cierre fiscal). Combina con el anio: ej. "
             "solo Q1 en varios anios para ver estacionalidad de la reaccion.")
    n_r = st.sidebar.slider("Ruedas por lado (pre y post)", 1, 10, N_DEFAULT,
                            key="er_nrue")

    if not anios_sel:
        st.info("Elegi al menos un anio en la barra lateral.")
        return
    if not q_sel:
        st.info("Elegi al menos un trimestre (Q1-Q4) en la barra lateral.")
        return

    serie, resumen = construir_series(ticker, anios_sel, q_sel, n_r)
    if serie.empty:
        st.warning("Hay fechas de balance pero faltan precios en la ventana "
                   "para cruzarlas (o no hay balances en los anios elegidos).")
        return

    anios_txt = ", ".join(str(a) for a in sorted(anios_sel))
    q_txt = "" if len(q_sel) == 4 else \
        " | " + ", ".join(f"Q{q}" for q in sorted(q_sel))
    st.subheader(f"{ticker} -- {len(resumen)} balances ({anios_txt}{q_txt}) "
                 f"| +/- {n_r} ruedas")
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    base = alt.Chart(serie).encode(
        x=alt.X("offset:Q",
                axis=alt.Axis(title="Rueda relativa al dia 0", tickMinStep=1,
                              format="d"),
                scale=alt.Scale(nice=False)),
        color=alt.Color("quarter:N", title="Trimestre"),
    )

    # -- PRECIO (USD): el cierre real, sin normalizar -------------------------
    st.markdown("**Precio (USD)** -- cierre real en la ventana (cada trimestre "
                "en su banda de precio)")
    linea_usd = base.mark_line(point=True).encode(
        y=alt.Y("close:Q", title="Precio (USD)", scale=alt.Scale(zero=False)),
        tooltip=["quarter", "fase", "fecha",
                 alt.Tooltip("offset:Q", title="rueda"),
                 alt.Tooltip("close:Q", format="$,.2f", title="precio"),
                 alt.Tooltip("vol_rel:Q", format=".2f", title="vol x prom")],
    )
    st.altair_chart((linea_usd + _divisor()).properties(height=320),
                    use_container_width=True)

    # -- PRECIO (%): acumulado desde el cierre previo al dia 0 ----------------
    st.markdown("**Precio (%)** -- variacion vs el cierre previo al balance "
                "(0% = ultimo precio antes de la reaccion)")
    linea_px = base.mark_line(point=True).encode(
        y=alt.Y("precio_pct:Q", title="% vs cierre previo"),
        tooltip=["quarter", "fase", "fecha",
                 alt.Tooltip("offset:Q", title="rueda"),
                 alt.Tooltip("precio_pct:Q", format="+.2f", title="% precio"),
                 alt.Tooltip("vol_rel:Q", format=".2f", title="vol x prom")],
    )
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(y="y:Q")
    st.altair_chart((linea_px + cero + _divisor()).properties(height=320),
                    use_container_width=True)

    # -- VOLUMEN: multiplo del promedio previo --------------------------------
    st.markdown(f"**Volumen** -- multiplo del promedio de {VOL_BASE_N} ruedas "
                "previas a la ventana")
    if serie["vol_rel"].notna().any():
        linea_vol = base.mark_line(point=True, strokeDash=[2, 2]).encode(
            y=alt.Y("vol_rel:Q", title="volumen / promedio previo"),
            tooltip=["quarter", "fase", "fecha",
                     alt.Tooltip("offset:Q", title="rueda"),
                     alt.Tooltip("vol_rel:Q", format=".2f", title="vol x prom"),
                     alt.Tooltip("volume:Q", format=",.0f")],
        )
        uno = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((linea_vol + uno + _divisor()).properties(height=280),
                        use_container_width=True)
    else:
        st.info("Sin volumen de referencia suficiente para el multiplo.")

    st.caption("Lectura: a la izquierda del divisor, como venia el precio y el "
               "volumen ANTES del balance (run-up); a la derecha, la reaccion "
               "post. El panel USD muestra el movimiento en crudo; el de % lo "
               "normaliza para comparar trimestres; el de volumen, si el interes "
               "se sostiene o se apaga.")
