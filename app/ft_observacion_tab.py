"""
ft_observacion_tab.py
Capa de observacion diaria del forward testing.

Sub-tabs:
  - Posiciones Abiertas  : analisis de ft_posiciones_diarias (estado='abierta')
  - Oportunidades NO Abiertas: analisis de ft_candidatos_diarios (entro=False)

Secciones por sub-tab:
  1. Momentum              (tech_score, candle_score_5d, up_vol_5d)
  2. Laterales/Acumulacion (lateral_ratio, rango_5d_pct, atr14)
  3. Retornos vs Tiempo    (retorno_pct / retorno_5d/10d/20d, dias_abierta)

Regla: tablas por defecto, checkbox por sub-tab para habilitar graficos.
"""

import pandas as pd
import streamlit as st


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fmt_pct(v, dec=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"+{v:.{dec}f}%" if v >= 0 else f"{v:.{dec}f}%"


def _fmt_score(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{float(v):.1f}"


def _lateral_label(ratio):
    if ratio is None or (isinstance(ratio, float) and pd.isna(ratio)):
        return "-"
    r = float(ratio)
    if r < 0.5:
        return "Lateral / Acumulacion"
    if r < 1.0:
        return "Rango estrecho"
    return "En movimiento"


def _safe_fecha(val):
    """Convierte un valor de pandas a date de Python, o None."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        return pd.Timestamp(val).date()
    except Exception:
        return None


# ─── Secciones: Posiciones Abiertas ──────────────────────────────────────────

def _sec_momentum_pos(df, show_charts):
    st.markdown("#### 1. Momentum")
    st.caption(
        "Tech Score: indicadores tecnicos (0-5.5). "
        "Candle 5d: estructura de velas ultimas 5 sesiones. "
        "Alc+Vol 5d: dias alcistas con volumen confirmado."
    )

    cols = ["ticker", "estrategia", "tech_score", "candle_score_5d",
            "up_vol_5d", "dias_abierta", "retorno_pct"]
    df_m = df[[c for c in cols if c in df.columns]].copy()
    df_m = df_m.sort_values("tech_score", ascending=False, na_position="last")

    df_show = pd.DataFrame()
    df_show["Ticker"]      = df_m["ticker"]
    df_show["Estrategia"]  = df_m["estrategia"] if "estrategia" in df_m.columns else "-"
    df_show["Tech Score"]  = df_m["tech_score"].map(_fmt_score)
    df_show["Candle 5d"]   = df_m["candle_score_5d"].map(_fmt_score)
    df_show["Alc+Vol 5d"]  = df_m["up_vol_5d"].fillna(0).astype(int).astype(str)
    df_show["Dias Abiert"] = df_m["dias_abierta"].fillna(0).astype(int).astype(str)
    df_show["Retorno"]     = df_m["retorno_pct"].map(lambda v: _fmt_pct(v))

    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker":      st.column_config.TextColumn(width=75),
            "Estrategia":  st.column_config.TextColumn(width=210),
            "Tech Score":  st.column_config.TextColumn(width=100, help="Score tecnico maximo 5.5"),
            "Candle 5d":   st.column_config.TextColumn(width=100, help="Score estructura velas 5 dias"),
            "Alc+Vol 5d":  st.column_config.TextColumn(width=105, help="Dias alcistas con vol confirmado (ultimos 5d)"),
            "Dias Abiert": st.column_config.TextColumn(width=105),
            "Retorno":     st.column_config.TextColumn(width=90),
        },
    )

    if show_charts:
        st.markdown("**Tech Score vs Candle Score 5d**")
        chart = df_m[["ticker", "tech_score", "candle_score_5d"]].dropna().set_index("ticker")
        st.bar_chart(chart)


def _sec_laterales_pos(df, show_charts):
    st.markdown("#### 2. Deteccion de Laterales / Acumulacion")
    st.caption(
        "Lateral Ratio = Rango 5d / ATR14. "
        "< 0.5 = lateral o acumulacion. "
        "0.5-1.0 = rango estrecho. "
        "> 1.0 = en movimiento / tendencia."
    )

    cols = ["ticker", "estrategia", "rango_5d_pct", "atr14", "lateral_ratio"]
    df_l = df[[c for c in cols if c in df.columns]].copy()
    if "lateral_ratio" in df_l.columns:
        df_l = df_l.sort_values("lateral_ratio", ascending=True, na_position="last")

    df_show = pd.DataFrame()
    df_show["Ticker"]        = df_l["ticker"]
    df_show["Estrategia"]    = df_l["estrategia"] if "estrategia" in df_l.columns else "-"
    df_show["Rango 5d %"]    = df_l["rango_5d_pct"].map(
        lambda v: f"{v:.2f}%" if pd.notna(v) else "-"
    )
    df_show["ATR 14"]        = df_l["atr14"].map(
        lambda v: f"{v:.4f}" if pd.notna(v) else "-"
    )
    df_show["Lat. Ratio"]    = df_l["lateral_ratio"].map(
        lambda v: f"{v:.3f}" if pd.notna(v) else "-"
    )
    df_show["Clasificacion"] = df_l["lateral_ratio"].map(_lateral_label)

    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker":        st.column_config.TextColumn(width=75),
            "Estrategia":    st.column_config.TextColumn(width=210),
            "Rango 5d %":    st.column_config.TextColumn(width=110,
                             help="Variacion porcentual (max-min) en ultimas 5 sesiones"),
            "ATR 14":        st.column_config.TextColumn(width=90,
                             help="Average True Range 14 periodos"),
            "Lat. Ratio":    st.column_config.TextColumn(width=100,
                             help="Rango5d/ATR: <0.5=lateral, 0.5-1=estrecho, >1=tendencia"),
            "Clasificacion": st.column_config.TextColumn(width=185),
        },
    )

    if "lateral_ratio" in df_l.columns:
        lr = df_l["lateral_ratio"].dropna()
        if not lr.empty:
            n_lat = int((lr < 0.5).sum())
            n_rng = int(((lr >= 0.5) & (lr < 1.0)).sum())
            n_mov = int((lr >= 1.0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Lateral / Acumulacion", n_lat)
            c2.metric("Rango estrecho", n_rng)
            c3.metric("En movimiento", n_mov)

    if show_charts and "lateral_ratio" in df_l.columns:
        st.markdown("**Lateral Ratio por ticker**")
        chart = df_l[["ticker", "lateral_ratio"]].dropna().set_index("ticker")
        st.bar_chart(chart)


def _sec_retornos_pos(df, show_charts):
    st.markdown("#### 3. Retornos vs Tiempo")
    st.caption("Ret/Dia = retorno_pct / dias_abierta (rendimiento diario promedio).")

    cols = ["ticker", "estrategia", "dias_abierta", "retorno_pct"]
    df_r = df[[c for c in cols if c in df.columns]].copy()
    df_r["dias_abierta"] = df_r["dias_abierta"].fillna(0).astype(int)
    df_r = df_r.sort_values("retorno_pct", ascending=False, na_position="last")

    df_show = pd.DataFrame()
    df_show["Ticker"]      = df_r["ticker"]
    df_show["Estrategia"]  = df_r["estrategia"] if "estrategia" in df_r.columns else "-"
    df_show["Dias Abiert"] = df_r["dias_abierta"].astype(str)
    df_show["Retorno %"]   = df_r["retorno_pct"].map(lambda v: _fmt_pct(v))
    df_show["Ret/Dia %"]   = df_r.apply(
        lambda row: _fmt_pct(row["retorno_pct"] / row["dias_abierta"], dec=3)
        if row["dias_abierta"] > 0 and pd.notna(row.get("retorno_pct"))
        else "-",
        axis=1,
    )

    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker":      st.column_config.TextColumn(width=75),
            "Estrategia":  st.column_config.TextColumn(width=210),
            "Dias Abiert": st.column_config.TextColumn(width=105),
            "Retorno %":   st.column_config.TextColumn(width=100),
            "Ret/Dia %":   st.column_config.TextColumn(width=105,
                           help="Retorno promedio por dia habil"),
        },
    )

    if show_charts:
        st.markdown("**Retorno % por ticker**")
        chart = df_r[["ticker", "retorno_pct"]].dropna().set_index("ticker")
        st.bar_chart(chart)


# ─── Secciones: Oportunidades NO Abiertas ────────────────────────────────────

def _sec_momentum_op(df, show_charts):
    st.markdown("#### 1. Momentum")
    st.caption("Score: puntuacion tecnica del candidato al momento de la identificacion.")

    cols = ["ticker", "estrategia", "score", "motivo_skip"]
    df_m = df[[c for c in cols if c in df.columns]].copy()
    df_m = df_m.sort_values("score", ascending=False, na_position="last")

    df_show = pd.DataFrame()
    df_show["Ticker"]      = df_m["ticker"]
    df_show["Estrategia"]  = df_m["estrategia"] if "estrategia" in df_m.columns else "-"
    df_show["Score"]       = df_m["score"].map(_fmt_score)
    df_show["Motivo Skip"] = df_m["motivo_skip"].fillna("-") if "motivo_skip" in df_m.columns else "-"

    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker":      st.column_config.TextColumn(width=75),
            "Estrategia":  st.column_config.TextColumn(width=210),
            "Score":       st.column_config.TextColumn(width=80,
                           help="Score tecnico del candidato"),
            "Motivo Skip": st.column_config.TextColumn(width=270),
        },
    )

    if show_charts:
        st.markdown("**Score por ticker**")
        chart = df_m[["ticker", "score"]].dropna().set_index("ticker")
        st.bar_chart(chart)


def _sec_laterales_op(df, show_charts):
    st.markdown("#### 2. Deteccion de Laterales / Acumulacion")
    st.caption(
        "Lateral Ratio calculado en tiempo real desde precios_diarios e indicadores_tecnicos. "
        "< 0.5 = lateral o acumulacion. 0.5-1 = rango estrecho. > 1 = tendencia."
    )

    cols = ["ticker", "estrategia", "rango_5d_pct", "atr14", "lateral_ratio"]
    df_l = df[[c for c in cols if c in df.columns]].copy()
    if "lateral_ratio" in df_l.columns:
        df_l = df_l.sort_values("lateral_ratio", ascending=True, na_position="last")

    df_show = pd.DataFrame()
    df_show["Ticker"]     = df_l["ticker"]
    df_show["Estrategia"] = df_l["estrategia"] if "estrategia" in df_l.columns else "-"

    if "rango_5d_pct" in df_l.columns:
        df_show["Rango 5d %"] = df_l["rango_5d_pct"].map(
            lambda v: f"{v:.2f}%" if pd.notna(v) else "-"
        )
    if "atr14" in df_l.columns:
        df_show["ATR 14"] = df_l["atr14"].map(
            lambda v: f"{v:.4f}" if pd.notna(v) else "-"
        )
    if "lateral_ratio" in df_l.columns:
        df_show["Lat. Ratio"]    = df_l["lateral_ratio"].map(
            lambda v: f"{v:.3f}" if pd.notna(v) else "-"
        )
        df_show["Clasificacion"] = df_l["lateral_ratio"].map(_lateral_label)

    col_cfg = {
        "Ticker":        st.column_config.TextColumn(width=75),
        "Estrategia":    st.column_config.TextColumn(width=210),
        "Rango 5d %":    st.column_config.TextColumn(width=110,
                         help="Variacion max-min en ultimas 5 sesiones"),
        "ATR 14":        st.column_config.TextColumn(width=90),
        "Lat. Ratio":    st.column_config.TextColumn(width=100,
                         help="Rango5d/ATR: <0.5=lateral, >1=tendencia"),
        "Clasificacion": st.column_config.TextColumn(width=185),
    }
    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={k: v for k, v in col_cfg.items() if k in df_show.columns},
    )

    if "lateral_ratio" in df_l.columns:
        lr = df_l["lateral_ratio"].dropna()
        if not lr.empty:
            n_lat = int((lr < 0.5).sum())
            n_rng = int(((lr >= 0.5) & (lr < 1.0)).sum())
            n_mov = int((lr >= 1.0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Lateral / Acumulacion", n_lat)
            c2.metric("Rango estrecho", n_rng)
            c3.metric("En movimiento", n_mov)

    if show_charts and "lateral_ratio" in df_l.columns:
        st.markdown("**Lateral Ratio por ticker**")
        chart = df_l[["ticker", "lateral_ratio"]].dropna().set_index("ticker")
        st.bar_chart(chart)


def _sec_retornos_op(df, show_charts):
    st.markdown("#### 3. Retornos vs Tiempo (Contrafactual)")
    st.caption(
        "Retornos calculados a 5, 10 y 20 dias habiles desde la fecha de identificacion. "
        "Se completan automaticamente cuando se cumplen los plazos."
    )

    cols = ["ticker", "estrategia", "precio_apertura",
            "retorno_5d", "retorno_10d", "retorno_20d"]
    df_r = df[[c for c in cols if c in df.columns]].copy()

    ret_cols = [c for c in ["retorno_5d", "retorno_10d", "retorno_20d"] if c in df_r.columns]
    has_data  = any(df_r[c].notna().any() for c in ret_cols) if ret_cols else False

    df_show = pd.DataFrame()
    df_show["Ticker"]     = df_r["ticker"]
    df_show["Estrategia"] = df_r["estrategia"] if "estrategia" in df_r.columns else "-"

    if "precio_apertura" in df_r.columns:
        df_show["Precio Ref."] = df_r["precio_apertura"].map(
            lambda v: f"${v:.2f}" if pd.notna(v) else "-"
        )

    if "retorno_5d" in df_r.columns:
        df_show["Ret 5d %"] = df_r["retorno_5d"].map(
            lambda v: _fmt_pct(v) if pd.notna(v) else "pendiente"
        )
    if "retorno_10d" in df_r.columns:
        df_show["Ret 10d %"] = df_r["retorno_10d"].map(
            lambda v: _fmt_pct(v) if pd.notna(v) else "pendiente"
        )
    if "retorno_20d" in df_r.columns:
        df_show["Ret 20d %"] = df_r["retorno_20d"].map(
            lambda v: _fmt_pct(v) if pd.notna(v) else "pendiente"
        )

    if not has_data:
        st.info(
            "Aun no hay retornos contrafactuales disponibles. "
            "Se calculan automaticamente a los 5, 10 y 20 dias habiles "
            "desde que el candidato fue identificado."
        )

    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker":      st.column_config.TextColumn(width=75),
            "Estrategia":  st.column_config.TextColumn(width=210),
            "Precio Ref.": st.column_config.TextColumn(width=95,
                           help="Precio de cierre al dia de identificacion"),
            "Ret 5d %":    st.column_config.TextColumn(width=95),
            "Ret 10d %":   st.column_config.TextColumn(width=95),
            "Ret 20d %":   st.column_config.TextColumn(width=95),
        },
    )

    if show_charts and has_data:
        st.markdown("**Retornos contrafactuales por ticker**")
        chart_cols = ["ticker"] + [c for c in ret_cols if c in df_r.columns]
        chart = (
            df_r[chart_cols]
            .dropna(subset=ret_cols, how="all")
            .set_index("ticker")
        )
        st.bar_chart(chart)


# ─── Sub-tab: Posiciones Abiertas ────────────────────────────────────────────

def _render_posiciones_abiertas(query_fn, eid, fecha):
    sql = """
        SELECT
            p.ticker,
            p.operacion_id,
            p.tech_score,
            p.candle_score_5d,
            p.up_vol_5d,
            p.dias_abierta,
            p.retorno_pct,
            p.rango_5d_pct,
            p.atr14,
            p.lateral_ratio,
            p.vol_price_confirm,
            p.vol_price_diverge,
            e.nombre AS estrategia
        FROM ft_posiciones_diarias p
        JOIN ft_estrategias e ON e.id = p.estrategia_id
        WHERE p.estado = 'abierta'
          AND p.fecha = :fecha
          AND (:eid = 0 OR p.estrategia_id = :eid)
        ORDER BY p.tech_score DESC NULLS LAST, p.ticker
    """
    df = query_fn(sql, {"fecha": str(fecha), "eid": eid})

    if df.empty:
        st.info(f"Sin posiciones abiertas registradas para el {fecha}.")
        return

    st.caption(f"{len(df)} posiciones abiertas al {fecha}")

    show_charts = st.checkbox(
        "Mostrar graficos",
        key="charts_pos_ab",
        value=False,
    )

    st.divider()
    _sec_momentum_pos(df, show_charts)
    st.divider()
    _sec_laterales_pos(df, show_charts)
    st.divider()
    _sec_retornos_pos(df, show_charts)


# ─── Sub-tab: Oportunidades NO Abiertas ──────────────────────────────────────

def _render_oportunidades_no_abiertas(query_fn, eid, fecha):
    sql = """
        WITH precios_5d AS (
            SELECT
                ticker,
                MAX(close) AS max_5d,
                MIN(close) AS min_5d
            FROM (
                SELECT ticker, close,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker ORDER BY fecha DESC
                       ) AS rn
                FROM precios_diarios
                WHERE fecha <= :fecha
            ) ranked
            WHERE rn <= 5
            GROUP BY ticker
        ),
        atr_latest AS (
            SELECT DISTINCT ON (ticker) ticker, atr14
            FROM indicadores_tecnicos
            WHERE fecha <= :fecha
            ORDER BY ticker, fecha DESC
        )
        SELECT
            c.ticker,
            e.nombre           AS estrategia,
            c.score,
            c.motivo_skip,
            c.precio_apertura,
            c.retorno_5d,
            c.retorno_10d,
            c.retorno_20d,
            a.atr14,
            CASE
                WHEN COALESCE(c.precio_apertura, 0) > 0
                THEN (p5.max_5d - p5.min_5d) / c.precio_apertura * 100
                ELSE NULL
            END AS rango_5d_pct,
            CASE
                WHEN COALESCE(a.atr14, 0) > 0
                THEN (p5.max_5d - p5.min_5d) / a.atr14
                ELSE NULL
            END AS lateral_ratio
        FROM ft_candidatos_diarios c
        JOIN ft_estrategias e ON e.id = c.estrategia_id
        LEFT JOIN atr_latest a ON a.ticker = c.ticker
        LEFT JOIN precios_5d p5 ON p5.ticker = c.ticker
        WHERE c.entro = FALSE
          AND c.fecha = :fecha
          AND (:eid = 0 OR c.estrategia_id = :eid)
        ORDER BY c.score DESC NULLS LAST, c.ticker
    """
    df = query_fn(sql, {"fecha": str(fecha), "eid": eid})

    if df.empty:
        st.info(f"Sin oportunidades NO abiertas registradas para el {fecha}.")
        return

    st.caption(f"{len(df)} oportunidades identificadas y NO abiertas al {fecha}")

    show_charts = st.checkbox(
        "Mostrar graficos",
        key="charts_op_nab",
        value=False,
    )

    st.divider()
    _sec_momentum_op(df, show_charts)
    st.divider()
    _sec_laterales_op(df, show_charts)
    st.divider()
    _sec_retornos_op(df, show_charts)


# ─── Entry point ─────────────────────────────────────────────────────────────

def render_ft_observacion_tab(query_fn):
    """Punto de entrada del tab. Llamar desde streamlit_app.py."""
    st.subheader("Observacion Diaria")
    st.caption(
        "Seguimiento diario de posiciones abiertas y oportunidades no capturadas. "
        "Datos cargados por los bots al final de cada sesion."
    )

    # Estrategias disponibles
    df_est = query_fn("SELECT id, nombre FROM ft_estrategias ORDER BY id")
    if df_est.empty:
        st.warning("No hay estrategias registradas en ft_estrategias.")
        return

    # Fechas disponibles
    df_fp = query_fn(
        "SELECT MAX(fecha) AS ultima FROM ft_posiciones_diarias WHERE estado = 'abierta'"
    )
    df_fc = query_fn(
        "SELECT MAX(fecha) AS ultima FROM ft_candidatos_diarios WHERE entro = FALSE"
    )

    fechas_validas = []
    for df_tmp in [df_fp, df_fc]:
        if not df_tmp.empty:
            val = _safe_fecha(df_tmp.iloc[0]["ultima"])
            if val is not None:
                fechas_validas.append(val)

    from datetime import date as _date
    ultima_global = max(fechas_validas) if fechas_validas else _date.today()

    # Controles
    col1, col2 = st.columns([2, 1])
    with col1:
        opciones_est = {"Todas las estrategias": 0}
        for _, row in df_est.iterrows():
            opciones_est[str(row["nombre"])] = int(row["id"])
        sel_nombre = st.selectbox(
            "Estrategia",
            list(opciones_est.keys()),
            key="obs_estrategia",
        )
        eid = opciones_est[sel_nombre]

    with col2:
        fecha_sel = st.date_input(
            "Fecha",
            value=ultima_global,
            key="obs_fecha",
        )

    st.divider()

    subtab1, subtab2 = st.tabs([
        "Posiciones Abiertas",
        "Oportunidades NO Abiertas",
    ])

    with subtab1:
        _render_posiciones_abiertas(query_fn, eid, fecha_sel)

    with subtab2:
        _render_oportunidades_no_abiertas(query_fn, eid, fecha_sel)
