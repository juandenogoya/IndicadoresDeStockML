"""
forward_testing_tab.py
Tab "Forward Testing" para streamlit_app.py

Sub-tabs: una por estrategia activa en ft_estrategias.
Cada sub-tab muestra:
    - Header: resumen de capital (invertido, saldo, variacion)
    - Tabla de operaciones: abiertas y cerradas
"""

import pandas as pd
import streamlit as st


# ── Helpers de formato ────────────────────────────────────────────────────────

def _fmt_usd(v, decimals=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"${v:,.{decimals}f}"


def _fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    signo = "+" if v >= 0 else ""
    return f"{signo}{v:.{decimals}f}%"


def _fmt_pnl(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    signo = "+" if v >= 0 else ""
    return f"{signo}${v:,.2f}"


def _color_pnl(val: str) -> str:
    """Devuelve color de celda segun signo del PnL."""
    try:
        clean = val.replace("$", "").replace(",", "").replace("+", "")
        num = float(clean)
        if num > 0:
            return "color: #4caf50; font-weight: bold"
        if num < 0:
            return "color: #ef5350; font-weight: bold"
    except Exception:
        pass
    return ""


# ── Query de operaciones ──────────────────────────────────────────────────────

def _cargar_operaciones(query_fn, estrategia_id: int) -> pd.DataFrame:
    """
    Carga todas las operaciones de una estrategia (abiertas y cerradas).
    Para posiciones abiertas, trae el precio de cierre mas reciente
    de precios_diarios como precio_actual y calcula PnL no realizado.
    """
    return query_fn("""
        SELECT
            o.id,
            o.ticker,
            o.cantidad,
            o.capital_entrada,
            o.fecha_entrada,
            o.precio_entrada,
            o.stop_loss,
            o.take_profit,
            o.score_entrada,
            o.fecha_salida,
            o.precio_salida,
            o.pnl,
            o.pnl_pct,
            o.motivo_salida,
            -- Precio actual para posiciones abiertas
            p.close AS precio_actual
        FROM ft_operaciones o
        LEFT JOIN (
            SELECT DISTINCT ON (ticker) ticker, close
            FROM precios_diarios
            ORDER BY ticker, fecha DESC
        ) p ON p.ticker = o.ticker AND o.fecha_salida IS NULL
        WHERE o.estrategia_id = :eid
        ORDER BY
            o.fecha_salida IS NOT NULL,   -- abiertas primero
            o.fecha_entrada DESC
    """, {"eid": estrategia_id})


# ── Header de capital ─────────────────────────────────────────────────────────

def _render_header(est: dict, df_ops: pd.DataFrame):
    """
    Muestra el resumen de capital de la estrategia en la parte superior.
    Calcula el PnL no realizado de posiciones abiertas usando precio_actual.
    """
    capital_inicial      = float(est["capital_inicial"])
    cash_disponible      = float(est["cash_disponible"])
    capital_inmovilizado = float(est["capital_inmovilizado"])
    capital_actual       = float(est["capital_actual"])

    # PnL no realizado: suma de (precio_actual - precio_entrada) * cantidad
    df_abiertas = df_ops[df_ops["fecha_salida"].isna()].copy()
    pnl_no_realizado = 0.0
    valor_mercado     = 0.0

    if not df_abiertas.empty and "precio_actual" in df_abiertas.columns:
        df_abiertas = df_abiertas.dropna(subset=["precio_actual"])
        if not df_abiertas.empty:
            df_abiertas["pnl_nr"] = (
                (df_abiertas["precio_actual"].astype(float) -
                 df_abiertas["precio_entrada"].astype(float)) *
                df_abiertas["cantidad"].astype(float)
            )
            pnl_no_realizado = float(df_abiertas["pnl_nr"].sum())
            valor_mercado    = float(
                (df_abiertas["precio_actual"].astype(float) *
                 df_abiertas["cantidad"].astype(float)).sum()
            )

    # PnL realizado acumulado
    df_cerradas = df_ops[df_ops["fecha_salida"].notna()].copy()
    pnl_realizado = float(df_cerradas["pnl"].sum()) if not df_cerradas.empty else 0.0

    # Saldo total estimado = cash + valor de mercado actual de posiciones
    saldo_estimado   = cash_disponible + valor_mercado
    variacion_dol    = saldo_estimado - capital_inicial
    variacion_pct    = (variacion_dol / capital_inicial * 100) if capital_inicial else 0.0

    # Metricas
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Capital Inicial",
        _fmt_usd(capital_inicial),
    )
    c2.metric(
        "Saldo Estimado",
        _fmt_usd(saldo_estimado),
        help="Cash disponible + valor de mercado actual de posiciones abiertas",
    )
    c3.metric(
        "Variacion $",
        _fmt_pnl(variacion_dol),
        delta=f"{variacion_pct:+.2f}%",
    )
    c4.metric(
        "Capital Invertido",
        _fmt_usd(capital_inmovilizado),
        help="Capital en posiciones abiertas al precio de entrada",
    )
    c5.metric(
        "PnL No Realizado",
        _fmt_pnl(pnl_no_realizado),
        help="Ganancia/perdida latente de posiciones abiertas vs precio actual",
    )
    c6.metric(
        "PnL Realizado",
        _fmt_pnl(pnl_realizado),
        help="Ganancia/perdida de operaciones ya cerradas",
    )

    # Barra de estado
    n_abiertas  = len(df_abiertas)
    n_cerradas  = len(df_cerradas)
    fecha_inicio = str(est.get("fecha_inicio", "-"))[:10]

    st.caption(
        f"Activa desde: {fecha_inicio}  |  "
        f"Posiciones abiertas: {n_abiertas}  |  "
        f"Operaciones cerradas: {n_cerradas}  |  "
        f"Cash libre: {_fmt_usd(cash_disponible)}"
    )


# ── Tabla de operaciones ──────────────────────────────────────────────────────

def _render_tabla(df_ops: pd.DataFrame):
    """
    Tabla unificada de operaciones abiertas y cerradas.
    Las abiertas aparecen primero con precio actual y PnL no realizado.
    """
    if df_ops.empty:
        st.info("Sin operaciones registradas todavia.")
        return

    filas = []
    for _, row in df_ops.iterrows():
        abierta = pd.isna(row["fecha_salida"])

        # Precio y PnL segun estado
        if abierta:
            precio_ref = float(row["precio_actual"]) if pd.notna(row.get("precio_actual")) else None
            pnl_val    = (
                (precio_ref - float(row["precio_entrada"])) * int(row["cantidad"])
                if precio_ref else None
            )
            pnl_pct_val = (
                (precio_ref / float(row["precio_entrada"]) - 1) * 100
                if precio_ref else None
            )
            fecha_salida_str  = "Abierta"
            precio_salida_str = _fmt_usd(precio_ref) if precio_ref else "-"
            estado = "🟢 Abierta"
        else:
            precio_ref        = float(row["precio_salida"]) if pd.notna(row["precio_salida"]) else None
            pnl_val           = float(row["pnl"])    if pd.notna(row["pnl"])    else None
            pnl_pct_val       = float(row["pnl_pct"]) if pd.notna(row["pnl_pct"]) else None
            fecha_salida_str  = str(row["fecha_salida"])[:10]
            precio_salida_str = _fmt_usd(precio_ref)
            motivo            = str(row.get("motivo_salida") or "")
            estado = f"🔴 {motivo}" if motivo else "🔴 Cerrada"

        filas.append({
            "Estado":          estado,
            "Ticker":          row["ticker"],
            "Cant.":           int(row["cantidad"]),
            "Capital ($)":     _fmt_usd(float(row["capital_entrada"])),
            "F. Entrada":      str(row["fecha_entrada"])[:10],
            "P. Entrada":      _fmt_usd(float(row["precio_entrada"])),
            "F. Salida":       fecha_salida_str,
            "P. Salida/Actual": precio_salida_str,
            "P&L ($)":         _fmt_pnl(pnl_val),
            "P&L (%)":         _fmt_pct(pnl_pct_val),
            "Score":           f"{float(row['score_entrada']):.1f}" if pd.notna(row.get("score_entrada")) else "-",
            "_pnl_raw":        pnl_val or 0.0,   # para colorear
        })

    df_show = pd.DataFrame(filas)

    # Estilo: colorear columna P&L
    def _style(row):
        styles = [""] * len(row)
        try:
            idx_pnl  = list(row.index).index("P&L ($)")
            idx_ppct = list(row.index).index("P&L (%)")
            pnl_r    = row["_pnl_raw"]
            if pnl_r > 0:
                color = "color: #4caf50; font-weight: bold"
            elif pnl_r < 0:
                color = "color: #ef5350; font-weight: bold"
            else:
                color = ""
            styles[idx_pnl]  = color
            styles[idx_ppct] = color
        except Exception:
            pass
        return styles

    cols_visible = [c for c in df_show.columns if c != "_pnl_raw"]
    styled = df_show[cols_visible + ["_pnl_raw"]].style.apply(_style, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Estado":           st.column_config.TextColumn(width=160),
            "Ticker":           st.column_config.TextColumn(width=70),
            "Cant.":            st.column_config.NumberColumn(width=60),
            "Capital ($)":      st.column_config.TextColumn(width=110),
            "F. Entrada":       st.column_config.TextColumn(width=95),
            "P. Entrada":       st.column_config.TextColumn(width=95),
            "F. Salida":        st.column_config.TextColumn(width=95),
            "P. Salida/Actual": st.column_config.TextColumn(width=115,
                                help="Cerradas: precio de salida. Abiertas: ultimo cierre disponible."),
            "P&L ($)":          st.column_config.TextColumn(width=100),
            "P&L (%)":          st.column_config.TextColumn(width=85),
            "Score":            st.column_config.TextColumn(width=60),
            "_pnl_raw":         None,   # ocultar columna auxiliar
        },
    )


# ── Tabla de oportunidades ────────────────────────────────────────────────────

def _render_oportunidades(query_fn, estrategia_id: int, df_ops: pd.DataFrame):
    """
    Tabla de oportunidades: candidatos que cumplieron condiciones de entrada
    en los ultimos 5 dias pero no entraron por limite de capital o posiciones.
    Muestra el score de cada dia para ver la consistencia de la senal.
    """
    st.subheader("Oportunidades")
    st.caption(
        "Activos con senal de entrada qualifying que no pudieron abrirse "
        "por limite de capital (80%) o posiciones (max 5). "
        "Mas dias consecutivos con score alto = senal mas robusta."
    )

    # Tickers actualmente abiertos — no son oportunidades
    tickers_abiertos = set(
        df_ops[df_ops["fecha_salida"].isna()]["ticker"].tolist()
    )

    # Ultimas 5 fechas con datos de candidatos para esta estrategia
    df_cand = query_fn("""
        SELECT ticker, fecha, score, entro
        FROM ft_candidatos_diarios
        WHERE estrategia_id = :eid
          AND fecha IN (
              SELECT DISTINCT fecha
              FROM ft_candidatos_diarios
              WHERE estrategia_id = :eid
              ORDER BY fecha DESC
              LIMIT 5
          )
        ORDER BY ticker, fecha DESC
    """, {"eid": estrategia_id})

    if df_cand.empty:
        st.info("Sin datos de candidatos todavia. Se registran automaticamente al correr los bots.")
        return

    # Filtrar: solo tickers que tuvieron al menos un dia sin entrar
    tickers_oport = (
        df_cand[df_cand["entro"] == False]["ticker"].unique().tolist()
    )
    # Excluir los que ya estan abiertos (ya capturamos esa oportunidad)
    tickers_oport = [t for t in tickers_oport if t not in tickers_abiertos]

    if not tickers_oport:
        st.info("Todos los candidatos tienen posicion abierta o no hay oportunidades pendientes.")
        return

    # Pivot: ticker x fecha, valor = score
    df_pivot = df_cand[df_cand["ticker"].isin(tickers_oport)].pivot_table(
        index="ticker", columns="fecha", values="score", aggfunc="first"
    )

    # Ordenar columnas: mas reciente primero
    df_pivot = df_pivot[sorted(df_pivot.columns, reverse=True)]

    # Calcular score promedio de los dias disponibles (para ordenar filas)
    df_pivot["_avg"] = df_pivot.iloc[:, :].mean(axis=1, skipna=True)
    df_pivot = df_pivot.sort_values("_avg", ascending=False)
    df_pivot = df_pivot.drop(columns=["_avg"])

    # Renombrar columnas a formato legible (dd/mm)
    df_pivot.columns = [
        pd.to_datetime(c).strftime("%d/%m") if not isinstance(c, str)
        else str(c)[5:10].replace("-", "/")
        for c in df_pivot.columns
    ]

    # Formatear valores: mostrar score con 1 decimal o "-" si falta
    df_show = df_pivot.copy()
    for col in df_show.columns:
        df_show[col] = df_show[col].map(
            lambda v: f"{v:.1f}" if pd.notna(v) else "-"
        )

    df_show = df_show.reset_index()
    df_show.columns.name = None

    # Estilo: colorear scores por nivel
    def _style_score(val: str) -> str:
        try:
            v = float(val)
            if v >= 80:
                return "color: #4caf50; font-weight: bold"
            if v >= 65:
                return "color: #8bc34a"
            if v >= 4.5:
                return "color: #4caf50; font-weight: bold"
            if v >= 4.0:
                return "color: #8bc34a"
            if v >= 2:
                return "color: #ffb300"
        except Exception:
            pass
        return ""

    cols_fecha = [c for c in df_show.columns if c != "ticker"]
    col_cfg    = {"ticker": st.column_config.TextColumn("Ticker", width=80)}
    for c in cols_fecha:
        col_cfg[c] = st.column_config.TextColumn(c, width=80)

    st.dataframe(
        df_show.style.applymap(_style_score, subset=cols_fecha),
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
    )
    st.caption(
        f"{len(tickers_oport)} activos con senal qualifying sin posicion abierta. "
        "Verde intenso = score muy alto. Verde claro = qualifying. "
        "'-' = sin senal ese dia (senal perdio fuerza)."
    )


# ── Sub-tab por estrategia ────────────────────────────────────────────────────

def _render_estrategia(query_fn, est: dict):
    """Renderiza el contenido completo de una estrategia."""
    eid = int(est["id"])

    with st.spinner("Cargando datos..."):
        df_ops = _cargar_operaciones(query_fn, eid)

    _render_header(est, df_ops)
    st.divider()
    _render_tabla(df_ops)
    st.divider()
    _render_oportunidades(query_fn, eid, df_ops)


# ── Entry point ───────────────────────────────────────────────────────────────

def render_forward_testing_tab(query_fn):
    """
    Renderiza el tab completo de Forward Testing.
    Recibe query_fn(sql, params=None) -> pd.DataFrame
    """
    # Cargar estrategias activas
    df_est = query_fn("""
        SELECT id, nombre, descripcion, logica,
               capital_inicial, capital_actual,
               cash_disponible, capital_inmovilizado,
               activa, fecha_inicio
        FROM ft_estrategias
        WHERE activa = TRUE
        ORDER BY id
    """)

    if df_est.empty:
        st.warning("No hay estrategias activas en ft_estrategias.")
        st.info("Ejecuta: python scripts/forward_testing/ft_setup_estrategias.py")
        return

    estrategias = [dict(r) for _, r in df_est.iterrows()]

    # Una sub-tab por estrategia
    iconos = ["🤖", "📈", "🕯️", "🔬", "⚡", "🎯"]
    labels = [
        f"{iconos[i % len(iconos)]} {est['nombre'].replace('FT_', '')}"
        for i, est in enumerate(estrategias)
    ]

    sub_tabs = st.tabs(labels)

    for tab, est in zip(sub_tabs, estrategias):
        with tab:
            logica_label = {
                "ml_scanner":    "ML Scanner",
                "tecnico":       "Tecnico SMA/MACD/RSI",
                "smc_estructura":"SMC CHoCH/BOS",
            }.get(est["logica"], est["logica"])

            st.caption(
                f"**Logica:** {logica_label}  |  "
                f"**Descripcion:** {est.get('descripcion', '-')}"
            )
            _render_estrategia(query_fn, est)
