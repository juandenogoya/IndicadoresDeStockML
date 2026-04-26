"""
opciones_tab.py
Sub-tab "Opciones" para streamlit_app.py

Sub-tabs:
    1. Universo  -- PCR ranking de todos los tickers (--vol-ticker)
    2. Sectores  -- PCR agrupado por sector (--vol-sector)
    3. Ticker    -- Analisis macro L0+L1 + drill strikes L2 (--analisis / --drill)
"""

import pandas as pd
import streamlit as st
from datetime import date

# ── constantes ────────────────────────────────────────────────────────────────
IV_MIN = 0.05   # 5%
IV_MAX = 2.00   # 200%
ATM_PCT_DEFAULT = 3.0

DTE_BUCKETS = ["0-7d", "8-14d", "15-30d", "31-60d", "61-90d"]
DTE_LIMITS  = [7, 14, 30, 60, 90]

# Colores PCR para fondo de celdas
def _pcr_bg(pcr):
    """Devuelve color de fondo segun PCR (bajista si alto = mas puts)."""
    if pd.isna(pcr):
        return "background-color:#2a2a2a"
    if pcr > 1.5:
        return "background-color:#5c0000"
    if pcr > 1.0:
        return "background-color:#7a2800"
    if pcr > 0.7:
        return "background-color:#4a4a00"
    if pcr > 0.4:
        return "background-color:#1a6e2a"
    return "background-color:#0d4f0d"


def _pcr_label(pcr):
    if pd.isna(pcr):
        return "-"
    if pcr > 1.5:
        return "MUY BAJISTA"
    if pcr > 1.0:
        return "BAJISTA"
    if pcr > 0.7:
        return "NEUTRO"
    if pcr > 0.4:
        return "ALCISTA"
    return "MUY ALCISTA"


def _dte_bucket(dte: int) -> str:
    for limit, label in zip(DTE_LIMITS, DTE_BUCKETS):
        if dte <= limit:
            return label
    return "61-90d"


def _moneyness(strike, precio, tipo, atm_pct):
    dist = abs(strike - precio) / precio * 100
    if dist <= atm_pct:
        return "ATM"
    if tipo == "call":
        return "ITM" if strike < precio else "OTM"
    else:
        return "ITM" if strike > precio else "OTM"


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 1: UNIVERSO
# ─────────────────────────────────────────────────────────────────────────────

def _subtab_universo(query_fn):
    st.subheader("Universo de Opciones — PCR por Ticker")

    # Fechas disponibles
    df_fechas = query_fn(
        "SELECT DISTINCT fecha FROM opciones_resumen_diario ORDER BY fecha DESC LIMIT 30"
    )
    if df_fechas.empty:
        st.warning("No hay datos en opciones_resumen_diario todavia.")
        return

    fechas = [str(f) for f in df_fechas["fecha"].tolist()]
    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        fecha_sel = st.selectbox("Fecha snapshot", fechas, key="univ_fecha")
    with col2:
        top_n = st.selectbox("Mostrar top", [20, 50, 100, 0], index=0,
                             format_func=lambda x: "Todos" if x == 0 else f"Top {x}",
                             key="univ_top")

    df = query_fn(
        """
        SELECT r.ticker, a.sector,
               r.call_vol, r.put_vol, r.pcr_vol,
               r.call_oi,  r.put_oi,  r.pcr_oi,
               r.iv_call_avg, r.iv_put_avg,
               r.n_contratos, r.max_oi_strike, r.precio_sub
        FROM opciones_resumen_diario r
        LEFT JOIN activos a ON r.ticker = a.ticker
        WHERE r.fecha = :fecha
        ORDER BY (r.call_vol + r.put_vol) DESC
        """,
        {"fecha": fecha_sel},
    )

    if df.empty:
        st.info(f"Sin datos para {fecha_sel}.")
        return

    if top_n > 0:
        df = df.head(top_n)

    # Columnas de display
    df["PCR_vol"] = df["pcr_vol"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    df["PCR_oi"]  = df["pcr_oi"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    df["IV_call"] = df["iv_call_avg"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df["IV_put"]  = df["iv_put_avg"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df["Precio"]  = df["precio_sub"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
    df["Sentimiento"] = df["pcr_vol"].apply(_pcr_label)
    df["MaxOI_Strike"] = df["max_oi_strike"].apply(lambda x: f"${x:.0f}" if pd.notna(x) else "-")

    vol_total = df["call_vol"] + df["put_vol"]
    df["Vol_total"] = vol_total

    display = df[["ticker", "sector", "Precio", "call_vol", "put_vol", "PCR_vol",
                  "Sentimiento", "call_oi", "put_oi", "PCR_oi",
                  "IV_call", "IV_put", "MaxOI_Strike"]].copy()
    display.columns = ["Ticker", "Sector", "Precio", "Call Vol", "Put Vol", "PCR Vol",
                       "Sentimiento", "Call OI", "Put OI", "PCR OI",
                       "IV Call", "IV Put", "Max OI Strike"]
    display = display.reset_index(drop=True)

    # Estilo por PCR
    def _style_row(row):
        pcr_str = row["PCR Vol"]
        try:
            pcr = float(pcr_str)
        except Exception:
            pcr = float("nan")
        bg = _pcr_bg(pcr)
        return [bg if col in ("PCR Vol", "Sentimiento") else "" for col in row.index]

    styled = display.style.apply(_style_row, axis=1)
    st.dataframe(styled, use_container_width=True, height=600)

    # Grafico barras Call vs Put vol (top 20)
    st.markdown("---")
    st.markdown("**Volumen Call vs Put — Top 20 por volumen total**")
    chart_df = df.head(20)[["ticker", "call_vol", "put_vol"]].set_index("ticker")
    chart_df.columns = ["Call Vol", "Put Vol"]
    st.bar_chart(chart_df)

    # Metricas resumidas
    st.markdown("---")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    pcr_median = df["pcr_vol"].median()
    total_call = df["call_vol"].sum()
    total_put  = df["put_vol"].sum()
    with col_m1:
        st.metric("Tickers analizados", len(df))
    with col_m2:
        st.metric("PCR Vol mediano", f"{pcr_median:.2f}")
    with col_m3:
        st.metric("Call Vol total", f"{total_call:,.0f}")
    with col_m4:
        st.metric("Put Vol total", f"{total_put:,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 2: SECTORES
# ─────────────────────────────────────────────────────────────────────────────

# ── helpers de sentimiento / tendencia ────────────────────────────────────────

def _semaforo(pcr):
    """Devuelve (emoji, etiqueta) segun PCR_vol."""
    if pcr is None or pd.isna(pcr):
        return "⚪", "NEUTRO"
    if pcr > 1.0:
        return "🔴", "BAJISTA"
    if pcr > 0.7:
        return "🟡", "NEUTRO"
    return "🟢", "ALCISTA"


def _tendencia_label(avg_new, avg_old, sentimiento):
    """
    Compara promedio PCR ultimas 2 fechas (avg_new) vs primeras 3 (avg_old).
    PCR bajando = mas alcista = sentimiento alcista fortaleciendose.
    PCR subiendo = mas bajista = sentimiento bajista fortaleciendose.
    """
    if avg_new is None or avg_old is None:
        return "sin cambio"
    delta = avg_new - avg_old   # positivo = mas bajista, negativo = mas alcista
    if abs(delta) < 0.05:
        return "sin cambio"
    if sentimiento == "ALCISTA":
        return "fortaleciendose" if delta < 0 else "debilitandose"
    if sentimiento == "BAJISTA":
        return "fortaleciendose" if delta > 0 else "debilitandose"
    # NEUTRO: umbral mas alto para evitar ruido
    if abs(delta) < 0.10:
        return "sin cambio"
    return "fortaleciendose" if delta < 0 else "debilitandose"


def _flow_signal(pcr_vol):
    """Etiqueta de flujo en lenguaje de mercado."""
    if pcr_vol is None or pd.isna(pcr_vol):
        return "Sin direccion"
    if pcr_vol < 0.7:
        return "Calls dominando"
    if pcr_vol > 1.0:
        return "Puts dominando"
    return "Sin direccion"


def _subtab_sectores(query_fn):
    st.subheader("Opciones por Sector")

    # Ultimas 5 fechas disponibles (mas reciente primero)
    df_fechas = query_fn(
        "SELECT DISTINCT fecha FROM opciones_resumen_diario ORDER BY fecha DESC LIMIT 5"
    )
    if df_fechas.empty:
        st.warning("No hay datos disponibles.")
        return

    fechas_desc = [str(f) for f in df_fechas["fecha"].tolist()]   # newest first
    fechas_asc  = list(reversed(fechas_desc))                      # oldest to newest

    # Cargar datos de sectores para todas las fechas disponibles
    fecha_list = "', '".join(fechas_desc)
    df_all = query_fn(
        f"""
        SELECT a.sector,
               r.fecha::text AS fecha,
               SUM(r.call_vol) AS call_vol,
               SUM(r.put_vol)  AS put_vol,
               CASE WHEN SUM(r.call_vol) > 0
                    THEN ROUND(SUM(r.put_vol)::numeric / SUM(r.call_vol), 3)
                    ELSE NULL END AS pcr_vol,
               SUM(r.call_oi)  AS call_oi,
               SUM(r.put_oi)   AS put_oi,
               CASE WHEN SUM(r.call_oi) > 0
                    THEN ROUND(SUM(r.put_oi)::numeric / SUM(r.call_oi), 3)
                    ELSE NULL END AS pcr_oi
        FROM opciones_resumen_diario r
        JOIN activos a ON r.ticker = a.ticker
        WHERE r.fecha::text IN ('{fecha_list}')
          AND a.sector IS NOT NULL
        GROUP BY a.sector, r.fecha
        ORDER BY a.sector, r.fecha
        """
    )

    if df_all.empty:
        st.info("Sin datos de sector para las ultimas 5 fechas.")
        return

    # Ordenar sectores por volumen total descendente
    vol_por_sector = (
        df_all.groupby("sector")
        .apply(lambda g: float(g["call_vol"].sum() + g["put_vol"].sum()))
        .sort_values(ascending=False)
    )
    sectores = vol_por_sector.index.tolist()

    # ── TABLA PIVOT: sector x fecha, PCR_vol ────────────────────────────────
    st.markdown("### PCR Vol por Sector — Ultimas 5 fechas")

    pivot_rows = []
    for sector in sectores:
        sdf = df_all[df_all["sector"] == sector]
        row = {"Sector": sector}
        pcr_vol_vals, pcr_oi_vals = [], []

        for f in fechas_asc:
            match = sdf[sdf["fecha"] == f]
            if not match.empty and pd.notna(match["pcr_vol"].iloc[0]):
                v = float(match["pcr_vol"].iloc[0])
                row[f] = v
                pcr_vol_vals.append(v)
            else:
                row[f] = None

            if not match.empty and pd.notna(match["pcr_oi"].iloc[0]):
                pcr_oi_vals.append(float(match["pcr_oi"].iloc[0]))

        row["_avg_vol"] = sum(pcr_vol_vals) / len(pcr_vol_vals) if pcr_vol_vals else None
        row["_avg_oi"]  = sum(pcr_oi_vals)  / len(pcr_oi_vals)  if pcr_oi_vals  else None
        pivot_rows.append(row)

    df_pivot_raw = pd.DataFrame(pivot_rows)

    # Renombrar columnas: fechas cortas + promedios legibles
    rename_map = {f: f[5:] for f in fechas_asc}   # "2026-04-24" -> "04-24"
    rename_map["_avg_vol"] = "PCR_vol avg"
    rename_map["_avg_oi"]  = "PCR_oi avg"
    df_pivot = df_pivot_raw.rename(columns=rename_map)

    date_cols_short = [f[5:] for f in fechas_asc]
    color_cols = date_cols_short + ["PCR_vol avg", "PCR_oi avg"]

    def _color_pivot(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in color_cols:
            if col in df.columns:
                styles[col] = df[col].apply(
                    lambda v: _pcr_bg(v) if (v is not None and pd.notna(v))
                    else "background-color:#1e1e1e"
                )
        return styles

    def _fmt_pcr(v):
        return f"{v:.2f}" if (v is not None and pd.notna(v)) else "-"

    fmt_map = {col: _fmt_pcr for col in color_cols if col in df_pivot.columns}

    styled_pivot = (
        df_pivot.style
        .apply(_color_pivot, axis=None)
        .format(fmt_map, na_rep="-")
    )
    st.dataframe(styled_pivot, use_container_width=True, hide_index=True)
    st.caption(
        "Verde = Alcista (PCR < 0.7)  |  "
        "Amarillo = Neutro (0.7-1.0)  |  "
        "Rojo = Bajista (> 1.0)  |  "
        "avg = promedio de las 5 fechas"
    )

    # ── TARJETAS RESUMEN POR SECTOR ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Resumen por Sector")
    st.caption(
        "Tendencia: promedio ultimas 2 fechas vs promedio primeras 3 fechas del periodo."
    )

    card_style_base = (
        "border:1px solid #333; border-radius:8px; padding:14px 16px; "
        "margin:4px 0; min-height:130px;"
    )

    n_per_row = 3
    for i in range(0, len(sectores), n_per_row):
        group = sectores[i : i + n_per_row]
        cols  = st.columns(n_per_row)

        for j, sector in enumerate(group):
            sdf = df_all[df_all["sector"] == sector]

            # PCR por fecha disponible (en orden cronologico)
            pcr_by_date = {}
            for f in fechas_asc:
                match = sdf[sdf["fecha"] == f]
                if not match.empty and pd.notna(match["pcr_vol"].iloc[0]):
                    pcr_by_date[f] = float(match["pcr_vol"].iloc[0])

            dates_ok = [f for f in fechas_asc if f in pcr_by_date]

            # Dividir: first 3 / last 2
            if len(dates_ok) >= 2:
                last2  = dates_ok[-2:]
                first3 = dates_ok[:-2] if len(dates_ok) > 2 else []
                avg_new = sum(pcr_by_date[f] for f in last2) / len(last2)
                avg_old = (sum(pcr_by_date[f] for f in first3) / len(first3)
                           if first3 else None)
                current_pcr = avg_new
            elif len(dates_ok) == 1:
                current_pcr = pcr_by_date[dates_ok[0]]
                avg_new = current_pcr
                avg_old = None
            else:
                current_pcr = avg_new = avg_old = None

            emoji, sentimiento = _semaforo(current_pcr)
            tendencia = _tendencia_label(avg_new, avg_old, sentimiento)
            flow      = _flow_signal(current_pcr)
            pcr_str   = f"{current_pcr:.2f}" if current_pcr is not None else "-"

            # Color de sentimiento para el texto
            color_sent = {"ALCISTA": "#4CAF50", "BAJISTA": "#f44336", "NEUTRO": "#FFC107"}
            color_tend = {"fortaleciendose": "#81C784", "debilitandose": "#e57373",
                          "sin cambio": "#90A4AE"}
            sent_color = color_sent.get(sentimiento, "#ccc")
            tend_color = color_tend.get(tendencia, "#ccc")

            html = (
                f"<div style='{card_style_base}'>"
                f"  <div style='font-size:1.0em; font-weight:bold; margin-bottom:6px;'>"
                f"    {emoji} {sector}"
                f"  </div>"
                f"  <div style='font-size:1.3em; font-weight:bold; color:{sent_color};'>"
                f"    {sentimiento}"
                f"  </div>"
                f"  <div style='color:{tend_color}; font-size:0.85em; margin-top:4px;'>"
                f"    Tendencia: {tendencia}"
                f"  </div>"
                f"  <div style='color:#aaa; font-size:0.85em;'>"
                f"    Flujo: {flow}"
                f"  </div>"
                f"  <div style='color:#666; font-size:0.75em; margin-top:6px;'>"
                f"    PCR Vol actual: {pcr_str}"
                f"  </div>"
                f"</div>"
            )

            with cols[j]:
                st.markdown(html, unsafe_allow_html=True)

        # Relleno de columnas vacias en la ultima fila
        for j in range(len(group), n_per_row):
            with cols[j]:
                st.empty()

    # ── GLOSARIO ─────────────────────────────────────────────────────────────
    st.markdown("")
    with st.expander("Glosario — Como leer el resumen por sector"):
        glosario = pd.DataFrame({
            "Sentimiento": [
                "ALCISTA", "ALCISTA",
                "NEUTRO",  "NEUTRO",
                "BAJISTA", "BAJISTA",
            ],
            "Tendencia": [
                "fortaleciendose", "debilitandose",
                "fortaleciendose", "debilitandose",
                "fortaleciendose", "debilitandose",
            ],
            "Significado": [
                "Calls dominan y aumentan: flujo comprador creciente",
                "Calls dominan pero se reducen: posible toma de ganancias",
                "Flujo mixto moviendose hacia calls: sesgo alcista emergente",
                "Flujo mixto moviendose hacia puts: cautela creciente",
                "Puts dominan y aumentan: cobertura o presion vendedora creciente",
                "Puts dominan pero se reducen: posible relajacion de cobertura",
            ],
        })
        st.dataframe(glosario, use_container_width=True, hide_index=True)
        st.caption(
            "PCR Vol < 0.7 = Alcista  |  0.7 - 1.0 = Neutro  |  > 1.0 = Bajista  |  "
            "Tendencia compara promedio ultimas 2 fechas vs primeras 3 del periodo de 5 dias."
        )

    # ── DRILL-DOWN POR SECTOR ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Drill-down por sector")

    col_d1, col_d2 = st.columns([2, 2])
    with col_d1:
        sector_drill = st.selectbox(
            "Sector", ["(elegir)"] + sectores, key="sect_drill"
        )
    with col_d2:
        fecha_drill = st.selectbox(
            "Fecha", fechas_desc, key="sect_drill_fecha"
        )

    if sector_drill != "(elegir)":
        df_drill = query_fn(
            """
            SELECT r.ticker,
                   r.call_vol, r.put_vol, r.pcr_vol,
                   r.call_oi,  r.put_oi,  r.pcr_oi,
                   r.iv_call_avg, r.iv_put_avg, r.precio_sub
            FROM opciones_resumen_diario r
            JOIN activos a ON r.ticker = a.ticker
            WHERE r.fecha = :fecha AND a.sector = :sector
            ORDER BY (r.call_vol + r.put_vol) DESC
            """,
            {"fecha": fecha_drill, "sector": sector_drill},
        )
        if df_drill.empty:
            st.info(f"Sin datos para {sector_drill} en {fecha_drill}.")
        else:
            df_drill["PCR Vol"]    = df_drill["pcr_vol"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
            df_drill["PCR OI"]     = df_drill["pcr_oi"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
            df_drill["IV Call"]    = df_drill["iv_call_avg"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
            df_drill["IV Put"]     = df_drill["iv_put_avg"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
            df_drill["Precio"]     = df_drill["precio_sub"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
            df_drill["Sentimiento"] = df_drill["pcr_vol"].apply(_pcr_label)

            show = df_drill[[
                "ticker", "Precio", "call_vol", "put_vol", "PCR Vol",
                "Sentimiento", "call_oi", "put_oi", "PCR OI", "IV Call", "IV Put",
            ]].copy()
            show.columns = [
                "Ticker", "Precio", "Call Vol", "Put Vol", "PCR Vol",
                "Sentimiento", "Call OI", "Put OI", "PCR OI", "IV Call", "IV Put",
            ]

            def _style_drill_sector(row):
                pcr_str = row.get("PCR Vol", "-")
                try:
                    pcr = float(pcr_str)
                except Exception:
                    pcr = float("nan")
                bg = _pcr_bg(pcr)
                return [bg if col in ("PCR Vol", "Sentimiento") else "" for col in row.index]

            styled_drill = show.reset_index(drop=True).style.apply(_style_drill_sector, axis=1)
            st.dataframe(styled_drill, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 3: TICKER — analisis L0+L1 + drill L2
# ─────────────────────────────────────────────────────────────────────────────

def _build_grid(df_snap, atm_pct, fecha):
    """
    Construye el grid DTE-bucket x moneyness en memoria.
    Retorna dos dicts: calls[bucket][moneyness] y puts[bucket][moneyness]
    con claves: vol, oi, iv_oi_sum, iv_oi_weight, n, n_iv_plaus
    """
    def _cell():
        return {"vol": 0, "oi": 0, "iv_oi_sum": 0.0, "iv_oi_weight": 0,
                "n": 0, "n_iv_plaus": 0}

    grid = {"call": {}, "put": {}}
    for bucket in DTE_BUCKETS:
        grid["call"][bucket] = {m: _cell() for m in ["ITM", "ATM", "OTM"]}
        grid["put"][bucket]  = {m: _cell() for m in ["ITM", "ATM", "OTM"]}

    for _, row in df_snap.iterrows():
        dte   = (pd.Timestamp(row["vencimiento"]) - pd.Timestamp(fecha)).days
        if dte < 0:
            continue
        bucket = _dte_bucket(dte)
        if bucket not in DTE_BUCKETS:
            continue
        tipo    = str(row["tipo"]).lower()
        if tipo not in ("call", "put"):
            continue
        mon     = _moneyness(float(row["strike"]), float(row["precio_subyacente"]),
                             tipo, atm_pct)
        vol     = int(row["volumen"]) if pd.notna(row["volumen"]) else 0
        oi      = int(row["open_interest"]) if pd.notna(row["open_interest"]) else 0
        iv_raw  = float(row["iv"]) if pd.notna(row["iv"]) else None
        iv      = iv_raw if (iv_raw is not None and IV_MIN <= iv_raw <= IV_MAX) else None

        c = grid[tipo][bucket][mon]
        c["vol"] += vol
        c["oi"]  += oi
        c["n"]   += 1
        if iv is not None and oi > 0:
            c["iv_oi_sum"]    += iv * oi
            c["iv_oi_weight"] += oi
            c["n_iv_plaus"]   += 1

    return grid["call"], grid["put"]


def _grid_to_df(calls, puts, df_snap, fecha):
    """Convierte los dicts de grilla en DataFrames para display."""
    total_cvol = sum(calls[b][m]["vol"] for b in DTE_BUCKETS for m in ["ITM","ATM","OTM"])
    total_pvol = sum(puts[b][m]["vol"]  for b in DTE_BUCKETS for m in ["ITM","ATM","OTM"])
    total_coi  = sum(calls[b][m]["oi"]  for b in DTE_BUCKETS for m in ["ITM","ATM","OTM"])
    total_poi  = sum(puts[b][m]["oi"]   for b in DTE_BUCKETS for m in ["ITM","ATM","OTM"])

    # HV_20d
    hv20 = None
    if "hv_20d" in df_snap.columns and not df_snap["hv_20d"].isna().all():
        hv20 = df_snap["hv_20d"].dropna().median()

    rows = []
    for bucket in DTE_BUCKETS:
        for mon in ["ITM", "ATM", "OTM"]:
            c = calls[bucket][mon]
            p = puts[bucket][mon]

            def _iv_w(cell):
                if cell["iv_oi_weight"] > 0:
                    return cell["iv_oi_sum"] / cell["iv_oi_weight"]
                return None

            def _iv_hv(iv_w):
                if iv_w and hv20 and hv20 > 0:
                    return iv_w / hv20
                return None

            def _cov(cell):
                if cell["n"] > 0:
                    return cell["n_iv_plaus"] / cell["n"] * 100
                return 0.0

            c_iv  = _iv_w(c)
            p_iv  = _iv_w(p)

            rows.append({
                "Bucket": bucket,
                "Mon":    mon,
                "C_VOL%": f"{c['vol']/total_cvol*100:.1f}%" if total_cvol else "-",
                "C_OI%":  f"{c['oi']/total_coi*100:.1f}%"  if total_coi  else "-",
                "C_IVw":  f"{c_iv*100:.1f}%" if c_iv else "N/D",
                "C_IVHV": f"{_iv_hv(c_iv):.2f}x" if _iv_hv(c_iv) else "-",
                "C_N":    c["n"],
                "C_cov":  _cov(c),
                "P_VOL%": f"{p['vol']/total_pvol*100:.1f}%" if total_pvol else "-",
                "P_OI%":  f"{p['oi']/total_poi*100:.1f}%"  if total_poi  else "-",
                "P_IVw":  f"{p_iv*100:.1f}%" if p_iv else "N/D",
                "P_IVHV": f"{_iv_hv(p_iv):.2f}x" if _iv_hv(p_iv) else "-",
                "P_N":    p["n"],
                "P_cov":  _cov(p),
                # vol/oi raw para calculos
                "c_vol": c["vol"], "c_oi": c["oi"],
                "p_vol": p["vol"], "p_oi": p["oi"],
            })

    return pd.DataFrame(rows), total_cvol, total_pvol, total_coi, total_poi


def _render_nivel0(df_res, df_snap, fecha):
    """Nivel 0: header con PCR, IV Skew, sesgo."""
    st.markdown("#### Nivel 0 — Resumen Macro")

    pcr_vol = float(df_res["pcr_vol"].iloc[0]) if pd.notna(df_res["pcr_vol"].iloc[0]) else None
    pcr_oi  = float(df_res["pcr_oi"].iloc[0])  if pd.notna(df_res["pcr_oi"].iloc[0])  else None
    cv      = int(df_res["call_vol"].iloc[0])
    pv      = int(df_res["put_vol"].iloc[0])
    coi     = int(df_res["call_oi"].iloc[0])
    poi     = int(df_res["put_oi"].iloc[0])
    precio  = float(df_res["precio_sub"].iloc[0]) if pd.notna(df_res["precio_sub"].iloc[0]) else None
    max_oi  = float(df_res["max_oi_strike"].iloc[0]) if pd.notna(df_res["max_oi_strike"].iloc[0]) else None
    iv_call = float(df_res["iv_call_avg"].iloc[0]) if pd.notna(df_res["iv_call_avg"].iloc[0]) else None
    iv_put  = float(df_res["iv_put_avg"].iloc[0])  if pd.notna(df_res["iv_put_avg"].iloc[0])  else None

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        pcr_str = f"{pcr_vol:.2f}" if pcr_vol else "-"
        st.metric("PCR Vol", pcr_str, help="put_vol / call_vol. >1 = sesgo bajista")
    with col2:
        pcr_oi_str = f"{pcr_oi:.2f}" if pcr_oi else "-"
        st.metric("PCR OI", pcr_oi_str, help="put_OI / call_OI. OI = dia anterior (T-1)")
    with col3:
        skew = None
        if iv_call and iv_put:
            skew = iv_put / iv_call
        st.metric("IV Skew (Put/Call)", f"{skew:.2f}" if skew else "-",
                  help="IV_put / IV_call ATM. >1 = mercado pagando mas por puts (proteccion)")
    with col4:
        precio_str = f"${precio:.2f}" if precio else "-"
        st.metric("Precio subyacente", precio_str)
    with col5:
        max_str = f"${max_oi:.0f}" if max_oi else "-"
        st.metric("Max OI Strike", max_str, help="Strike con mayor OI combinado call+put")

    # Semaforo de sesgo
    sesgo = _pcr_label(pcr_vol)
    bg    = _pcr_bg(pcr_vol)
    st.markdown(
        f"<div style='{bg}; padding:6px 12px; border-radius:6px; display:inline-block; "
        f"font-weight:bold; color:white;'>Sentimiento: {sesgo}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown(f"**Call Vol:** {cv:,}  |  **Put Vol:** {pv:,}")
    with col_v2:
        st.markdown(f"**Call OI:** {coi:,}  |  **Put OI:** {poi:,} *(T-1)*")


def _render_nivel1(calls, puts, df_snap, fecha):
    """Nivel 1: grid DTE-bucket x moneyness con calls y puts."""
    st.markdown("#### Nivel 1 — Grid DTE x Moneyness")

    df_grid, total_cvol, total_pvol, total_coi, total_poi = _grid_to_df(
        calls, puts, df_snap, fecha
    )

    # Pivotear para display
    rows_display = []
    for bucket in DTE_BUCKETS:
        sub = df_grid[df_grid["Bucket"] == bucket]
        if sub["C_N"].sum() == 0 and sub["P_N"].sum() == 0:
            continue
        for _, row in sub.iterrows():
            mon = row["Mon"]
            c_cov = row["C_cov"]
            p_cov = row["P_cov"]
            c_ivw = row["C_IVw"] + ("!" if (row["C_N"] > 0 and c_cov < 30) else "")
            p_ivw = row["P_IVw"] + ("!" if (row["P_N"] > 0 and p_cov < 30) else "")

            rows_display.append({
                "DTE Bucket": bucket,
                "Mon":        mon,
                "C VOL%":  row["C_VOL%"],
                "C OI%":   row["C_OI%"],
                "C IV_w":  c_ivw,
                "C IV/HV": row["C_IVHV"],
                "C N":     row["C_N"],
                "P VOL%":  row["P_VOL%"],
                "P OI%":   row["P_OI%"],
                "P IV_w":  p_ivw,
                "P IV/HV": row["P_IVHV"],
                "P N":     row["P_N"],
                "_c_vol":  row["c_vol"],
                "_p_vol":  row["p_vol"],
                "_c_oi":   row["c_oi"],
                "_p_oi":   row["p_oi"],
            })

    if not rows_display:
        st.info("Sin datos para armar el grid.")
        return

    df_disp = pd.DataFrame(rows_display)

    # Colorear por moneyness (fondo suave)
    mon_colors = {"ITM": "#1a1a4a", "ATM": "#1a3a1a", "OTM": "#2a1a00"}

    def _style_grid(row):
        bg = f"background-color:{mon_colors.get(row['Mon'], '')}"
        return [bg] * len(row)

    # Calcular PCR por bucket para footer
    pcr_rows = []
    for bucket in DTE_BUCKETS:
        sub = df_grid[df_grid["Bucket"] == bucket]
        if sub["C_N"].sum() == 0 and sub["P_N"].sum() == 0:
            continue
        cv_b = sub["c_vol"].sum()
        pv_b = sub["p_vol"].sum()
        coi_b = sub["c_oi"].sum()
        poi_b = sub["p_oi"].sum()
        pcr_v = pv_b / cv_b if cv_b > 0 else None
        pcr_o = poi_b / coi_b if coi_b > 0 else None
        pcr_rows.append({
            "DTE Bucket": bucket,
            "PCR Vol":  f"{pcr_v:.2f}" if pcr_v else "-",
            "PCR OI":   f"{pcr_o:.2f}" if pcr_o else "-",
            "Sentimiento": _pcr_label(pcr_v),
        })

    cols_show = ["DTE Bucket", "Mon",
                 "C VOL%", "C OI%", "C IV_w", "C IV/HV", "C N",
                 "P VOL%", "P OI%", "P IV_w", "P IV/HV", "P N"]

    styled = df_disp[cols_show].style.apply(_style_grid, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("! = cobertura IV < 30% (baja confianza). IV ponderado por OI. IV filtro 5%-200%.")

    # PCR por bucket
    if pcr_rows:
        st.markdown("**PCR por DTE Bucket:**")
        df_pcr = pd.DataFrame(pcr_rows)
        st.dataframe(df_pcr, use_container_width=True, hide_index=True)


def _render_drill(df_snap, tipo, dte_max, top_n, atm_pct, fecha):
    """Nivel 2: tabla de strikes individuales."""
    st.markdown(f"#### Nivel 2 — Drill Strikes ({tipo.upper()}, DTE <= {dte_max}d)")

    mask = df_snap["tipo"].str.lower() == tipo.lower()
    df_t = df_snap[mask].copy()
    df_t["dte"] = (pd.to_datetime(df_t["vencimiento"]) - pd.Timestamp(fecha)).dt.days
    df_t = df_t[df_t["dte"] >= 0]
    df_t = df_t[df_t["dte"] <= dte_max]

    if df_t.empty:
        st.info(f"Sin contratos {tipo.upper()} con DTE <= {dte_max}.")
        return

    # Moneyness y IV filtrado
    df_t["mon"] = df_t.apply(
        lambda r: _moneyness(float(r["strike"]), float(r["precio_subyacente"]), tipo, atm_pct),
        axis=1,
    )
    df_t["iv_filt"] = df_t["iv"].apply(
        lambda x: float(x) if (pd.notna(x) and IV_MIN <= float(x) <= IV_MAX) else None
    )
    df_t["prima"] = ((df_t["bid"].fillna(0) + df_t["ask"].fillna(0)) / 2)
    df_t["voi"]   = df_t.apply(
        lambda r: round(float(r["volumen"]) / float(r["open_interest"]), 2)
        if (pd.notna(r["open_interest"]) and r["open_interest"] > 0
            and pd.notna(r["volumen"])) else None,
        axis=1,
    )
    df_t["dist_pct"] = ((df_t["strike"] - df_t["precio_subyacente"]) / df_t["precio_subyacente"] * 100).round(1)

    # Ordenar por vencimiento luego por moneyness y strike
    mon_order = {"ITM": 0, "ATM": 1, "OTM": 2}
    df_t["mon_order"] = df_t["mon"].map(mon_order)
    df_t = df_t.sort_values(["vencimiento", "mon_order", "strike"]).reset_index(drop=True)

    # Agrupar por vencimiento
    venc_groups = df_t.groupby("vencimiento", sort=False)

    all_rows = []
    for venc, grp in venc_groups:
        dte_v  = grp["dte"].iloc[0]
        vol_v  = grp["volumen"].sum()
        oi_v   = grp["open_interest"].sum()

        if top_n > 0:
            grp = grp.nlargest(top_n, "volumen")

        for _, r in grp.iterrows():
            iv_str = f"{r['iv_filt']*100:.1f}%" if r["iv_filt"] else "-"
            hv = float(r["hv_20d"]) if r["hv_20d"] and r["hv_20d"] > 0 else None
            ivhv = f"{r['iv_filt']/hv:.2f}x" if (r["iv_filt"] and hv) else "-"
            voi_str = f"{r['voi']:.2f}" if r["voi"] else "-"
            all_rows.append({
                "Venc":    str(venc)[:10],
                "DTE":     int(dte_v),
                "Mon":     r["mon"],
                "Strike":  float(r["strike"]),
                "Dist%":   f"{r['dist_pct']:+.1f}%",
                "Prima":   f"${r['prima']:.2f}",
                "IV":      iv_str,
                "Vol":     int(r["volumen"]) if pd.notna(r["volumen"]) else 0,
                "OI":      int(r["open_interest"]) if pd.notna(r["open_interest"]) else 0,
                "IV/HV":   ivhv,
                "V/OI":    voi_str,
                "_sep_label": f"{venc}  DTE={dte_v}  Vol={vol_v:,}  OI={oi_v:,}",
            })

    if not all_rows:
        st.info("Sin datos con los filtros aplicados.")
        return

    df_drill = pd.DataFrame(all_rows)

    # Colorear por moneyness
    mon_colors = {"ITM": "#1a1a4a", "ATM": "#1a3a1a", "OTM": "#2a1a00"}

    def _style_drill(row):
        bg = f"background-color:{mon_colors.get(row['Mon'], '')}"
        # V/OI alto en naranja
        styles = []
        for col in row.index:
            if col == "Mon":
                styles.append(bg)
            elif col == "V/OI":
                try:
                    v = float(row["V/OI"])
                    if v > 1.0:
                        styles.append("background-color:#7a3800; font-weight:bold")
                    elif v > 0.3:
                        styles.append("background-color:#4a3000")
                    else:
                        styles.append("")
                except Exception:
                    styles.append("")
            else:
                styles.append("")
        return styles

    cols_show = ["Venc", "DTE", "Mon", "Strike", "Dist%", "Prima",
                 "IV", "Vol", "OI", "IV/HV", "V/OI"]
    styled = df_drill[cols_show].style.apply(_style_drill, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    n_strikes = len(df_drill)
    st.caption(
        f"Strikes mostrados: {n_strikes} | "
        f"ATM umbral: +/-{atm_pct:.0f}% | "
        f"IV filtro 5%-200% | "
        f"OI = dia anterior (T-1) | "
        f"V/OI > 0.3 = flujo nuevo significativo"
    )


def _subtab_ticker(query_fn):
    st.subheader("Analisis por Ticker")

    # Controles
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        df_tickers = query_fn(
            """
            SELECT DISTINCT r.ticker FROM opciones_resumen_diario r
            JOIN activos a ON r.ticker = a.ticker
            ORDER BY r.ticker
            """
        )
        tickers = df_tickers["ticker"].tolist() if not df_tickers.empty else []
        if not tickers:
            st.warning("No hay tickers en opciones_resumen_diario.")
            return
        ticker = st.selectbox("Ticker", tickers, key="tick_ticker")

    with col2:
        df_fechas = query_fn(
            "SELECT DISTINCT fecha FROM opciones_resumen_diario WHERE ticker = :t ORDER BY fecha DESC",
            {"t": ticker},
        )
        fechas = [str(f) for f in df_fechas["fecha"].tolist()] if not df_fechas.empty else []
        if not fechas:
            st.info("Sin fechas disponibles para este ticker.")
            return
        fecha_sel = st.selectbox("Fecha snapshot", fechas, key="tick_fecha")

    with col3:
        atm_pct = st.slider("Umbral ATM (%)", min_value=1, max_value=10,
                            value=int(ATM_PCT_DEFAULT), key="tick_atm")

    # Cargar resumen diario
    df_res = query_fn(
        """
        SELECT call_vol, put_vol, pcr_vol, call_oi, put_oi, pcr_oi,
               iv_call_avg, iv_put_avg, n_contratos, max_oi_strike, precio_sub
        FROM opciones_resumen_diario
        WHERE ticker = :ticker AND fecha = :fecha
        """,
        {"ticker": ticker, "fecha": fecha_sel},
    )

    if df_res.empty:
        st.info("Sin datos de resumen para el ticker/fecha seleccionados.")
        return

    # Nivel 0
    _render_nivel0(df_res, None, fecha_sel)

    st.markdown("---")

    # Cargar snapshot completo para este ticker/fecha
    with st.spinner("Cargando contratos..."):
        df_snap = query_fn(
            """
            SELECT vencimiento, tipo, strike, volumen, open_interest,
                   iv, bid, ask, precio_subyacente, hv_20d
            FROM opciones_snapshot
            WHERE ticker = :ticker AND fecha_snapshot = :fecha
            ORDER BY tipo, vencimiento, strike
            """,
            {"ticker": ticker, "fecha": fecha_sel},
        )

    if df_snap.empty:
        st.info("Sin contratos en opciones_snapshot para este ticker/fecha.")
        return

    # Nivel 1
    calls, puts = _build_grid(df_snap, atm_pct, fecha_sel)
    _render_nivel1(calls, puts, df_snap, fecha_sel)

    st.markdown("---")

    # Nivel 2: drill
    st.markdown("#### Nivel 2 — Drill por Strikes")
    col_d1, col_d2, col_d3 = st.columns([2, 2, 2])
    with col_d1:
        tipo_drill = st.selectbox("Tipo", ["call", "put"], key="tick_tipo")
    with col_d2:
        dte_max = st.selectbox("DTE maximo", [7, 14, 30, 60, 90], index=2, key="tick_dte")
    with col_d3:
        top_n = st.selectbox("Top N por vencimiento",
                             [0, 5, 10, 20, 50],
                             index=0,
                             format_func=lambda x: "Todos" if x == 0 else f"Top {x}",
                             key="tick_topn")

    _render_drill(df_snap, tipo_drill, dte_max, top_n, atm_pct, fecha_sel)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_opciones_tab(query_fn):
    """
    Renderiza el tab completo de Opciones.
    Recibe query_fn(sql, params=None) -> pd.DataFrame
    """
    st_u, st_s, st_t = st.tabs(["🌐 Universo", "🏭 Sectores", "🔬 Ticker"])

    with st_u:
        _subtab_universo(query_fn)

    with st_s:
        _subtab_sectores(query_fn)

    with st_t:
        _subtab_ticker(query_fn)
