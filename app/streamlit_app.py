"""
streamlit_app.py
Dashboard interactivo para el Sistema Scanner ML de Activos.

Secciones:
    1. Dashboard  -- alertas del dia con tabla coloreada
    2. Agregar Ticker -- incorporar nuevo ticker y asignar modelo champion
    3. Historial  -- alertas pasadas filtrables por fecha y ticker
"""

import sys
import os
import importlib.util
import pathlib

# Asegurar que el directorio raiz del proyecto este en el path
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from datetime import date, timedelta


# ─────────────────────────────────────────────────────────────
# Config de pagina
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Scanner ML Activos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Conexion a DB — directo desde st.secrets (no depende de config.py)
# ─────────────────────────────────────────────────────────────

def _get_database_url() -> str:
    """Obtiene DATABASE_URL desde st.secrets o variable de entorno."""
    # 1. st.secrets (Streamlit Cloud)
    try:
        url = st.secrets.get("DATABASE_URL") or st.secrets["DATABASE_URL"]
        if url:
            return str(url)
    except Exception:
        pass
    # 2. Variable de entorno (local con .env)
    return os.getenv("DATABASE_URL", "")


@st.cache_resource
def _get_engine():
    """Crea el engine SQLAlchemy una sola vez (cacheado)."""
    from sqlalchemy import create_engine
    db_url = _get_database_url()
    if not db_url:
        raise ValueError("DATABASE_URL no configurada. Verifica los Secrets en Streamlit Cloud.")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, connect_args={"sslmode": "require"})


def query(sql: str, params: dict = None) -> pd.DataFrame:
    """Ejecuta un SELECT y retorna un DataFrame."""
    from sqlalchemy import text
    engine = _get_engine()
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params or {})


# Inyectar DATABASE_URL al entorno para que src.pipeline pueda usarla (tab Agregar Ticker)
_db_url = _get_database_url()
if _db_url and "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = _db_url


# ─────────────────────────────────────────────────────────────
# Colores y emojis por nivel de alerta
# ─────────────────────────────────────────────────────────────

_BG_COLOR = {
    "COMPRA_FUERTE": "#0d4f0d",
    "COMPRA":        "#1a6e2a",
    "NEUTRAL":       "#4a4a00",
    "VENTA":         "#7a2800",
    "VENTA_FUERTE":  "#5c0000",
}
_EMOJIS = {
    "COMPRA_FUERTE": "🟢🟢",
    "COMPRA":        "🟢",
    "NEUTRAL":       "🟡",
    "VENTA":         "🔴",
    "VENTA_FUERTE":  "🔴🔴",
}


# ─────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ─────────────────────────────────────────────────────────────

def tab_dashboard():
    st.subheader("Alertas del Dia")

    # Obtener ultima fecha disponible
    df_f = query("SELECT MAX(DATE(scan_fecha)) AS ultima FROM alertas_scanner")
    ultima_fecha = df_f.iloc[0]["ultima"] if not df_f.empty else None

    if ultima_fecha is None:
        st.warning("No hay datos en alertas_scanner todavia.")
        return

    # Selector de fecha
    fecha_sel = st.date_input(
        "Fecha de escaneo",
        value=ultima_fecha,
        max_value=date.today(),
    )

    sql = """
        SELECT * FROM (
            SELECT DISTINCT ON (ticker)
               ticker, sector, alert_nivel, alert_score,
               ml_prob_ganancia, ml_modelo_usado,
               pa_ev1, pa_ev2, pa_ev3, pa_ev4,
               bear_bos10, bear_choch10, bear_estructura,
               precio_cierre, score_ponderado
            FROM alertas_scanner
            WHERE DATE(scan_fecha) = :fecha
            ORDER BY ticker, scan_fecha DESC
        ) sub
        ORDER BY alert_score DESC
    """
    df = query(sql, {"fecha": str(fecha_sel)})

    if df.empty:
        st.warning(f"Sin datos para {fecha_sel}.")
        return

    # ── Metricas resumen ──────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Tickers", len(df))
    c2.metric("🟢🟢 Compra Fuerte", int((df.alert_nivel == "COMPRA_FUERTE").sum()))
    c3.metric("🟢 Compra",          int((df.alert_nivel == "COMPRA").sum()))
    c4.metric("🟡 Neutral",         int((df.alert_nivel == "NEUTRAL").sum()))
    c5.metric("🔴 Venta/Fuerte",    int(df.alert_nivel.isin(["VENTA", "VENTA_FUERTE"]).sum()))

    st.divider()

    # ── Filtros ───────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sectores_disp = ["Todos"] + sorted(df["sector"].dropna().unique().tolist())
        sector_sel = st.selectbox("Sector", sectores_disp)
    with col_f2:
        niveles_disp = ["Todos", "COMPRA_FUERTE", "COMPRA", "NEUTRAL", "VENTA", "VENTA_FUERTE"]
        nivel_sel = st.selectbox("Nivel", niveles_disp)
    with col_f3:
        prob_min = st.slider("ML Prob. minima (%)", 0, 100, 0, 5)

    # ── Aplicar filtros ───────────────────────────────────────
    df_f = df.copy()
    if sector_sel != "Todos":
        df_f = df_f[df_f.sector == sector_sel]
    if nivel_sel != "Todos":
        df_f = df_f[df_f.alert_nivel == nivel_sel]
    df_f = df_f[df_f.ml_prob_ganancia >= prob_min / 100]

    if df_f.empty:
        st.info("Ningun ticker cumple los filtros seleccionados.")
        st.caption(f"Fecha: {fecha_sel} | 0 de {len(df)} tickers mostrados")
        return

    # ── Formatear columnas para display ──────────────────────
    df_show = df_f.copy()
    df_show["Nivel"]   = df_show["alert_nivel"].map(lambda x: f"{_EMOJIS.get(x, '')} {x}")
    df_show["Score"]   = df_show["alert_score"].map(lambda x: f"{x:.0f}")
    df_show["ML %"]    = df_show["ml_prob_ganancia"].map(lambda x: f"{x:.0%}")
    df_show["Precio"]  = df_show["precio_cierre"].map(lambda x: f"${x:.2f}" if x else "-")
    df_show["Modelo"]  = df_show["ml_modelo_usado"].map(
        lambda x: {"global": "GLO", "Financials": "FIN",
                   "Consumer Staples": "CON", "Consumer Discretionary": "CON"}.get(x, x[:3].upper())
        if x else "-"
    )
    df_show["EV"] = df_show.apply(
        lambda r: "".join([
            str(int(r.pa_ev1 or 0)), str(int(r.pa_ev2 or 0)),
            str(int(r.pa_ev3 or 0)), str(int(r.pa_ev4 or 0)),
        ]), axis=1
    )
    df_show["Bear"] = df_show.apply(
        lambda r: ("B" if r.bear_bos10 else ".") +
                  ("C" if r.bear_choch10 else ".") +
                  ("E" if r.bear_estructura else "."),
        axis=1
    )

    cols_display = ["ticker", "sector", "Nivel", "Score", "ML %",
                    "Precio", "EV", "Bear", "Modelo"]
    st.dataframe(
        df_show[cols_display].rename(columns={"ticker": "Ticker", "sector": "Sector"}),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(width=80),
            "Score":  st.column_config.TextColumn(width=65),
            "ML %":   st.column_config.TextColumn(width=70),
            "Precio": st.column_config.TextColumn(width=90),
            "EV":     st.column_config.TextColumn(width=55, help="EV1 EV2 EV3 EV4"),
            "Bear":   st.column_config.TextColumn(width=60, help="BOS CHoCH Estructura"),
        },
    )
    st.caption(f"Fecha: {fecha_sel} | {len(df_f)} de {len(df)} tickers mostrados")


# ─────────────────────────────────────────────────────────────
# TAB 2: AGREGAR TICKER
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def _cargar_mod_19():
    """Carga 19_incorporar_ticker.py via importlib (solo una vez)."""
    script_path = ROOT / "scripts" / "19_incorporar_ticker.py"
    spec = importlib.util.spec_from_file_location("mod19", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tab_agregar():
    st.subheader("Incorporar Nuevo Ticker")
    st.write(
        "Descarga 5 anos de historial, evalua los 4 modelos champion V3 "
        "y asigna el mejor al ticker."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker_input = st.text_input(
            "Ticker",
            placeholder="Ej: AAPL, MSFT, NVDA",
            max_chars=10,
        ).upper().strip()
    with col2:
        umbral_pct = st.number_input(
            "Umbral retorno 20d (%)",
            min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        )

    forzar = st.checkbox("Re-evaluar si ya tiene modelo asignado")

    if not ticker_input:
        st.info("Ingresa un ticker para comenzar la evaluacion.")
        return

    # Verificar asignacion actual
    df_actual = query(
        "SELECT modelo_asignado FROM activos WHERE ticker = :t",
        {"t": ticker_input}
    )
    if not df_actual.empty and df_actual.iloc[0]["modelo_asignado"]:
        st.info(f"**{ticker_input}** ya tiene modelo asignado: "
                f"**{df_actual.iloc[0]['modelo_asignado']}**")

    if st.button("Evaluar Modelos", type="primary"):
        with st.spinner(f"Procesando {ticker_input}... puede tardar 1-2 minutos"):
            try:
                mod19 = _cargar_mod_19()
                resultado = mod19.incorporar_ticker(
                    ticker_input,
                    umbral=umbral_pct / 100,
                    forzar=forzar,
                )
            except Exception as e:
                st.error(f"Error inesperado: {e}")
                return

        if resultado.get("error"):
            msg = resultado['error']
            if "0 barras" in msg or "barras disponibles" in msg:
                st.warning(
                    f"**yfinance no puede descargar datos en Streamlit Cloud** "
                    f"(restriccion de red del servidor gratuito).\n\n"
                    f"Ejecuta este comando localmente desde tu PC:\n\n"
                    f"```\npython scripts/19_incorporar_ticker.py {ticker_input}\n```\n\n"
                    f"Una vez asignado el modelo, el scanner lo incluira automaticamente."
                )
            else:
                st.error(f"Error: {msg}")
            return

        if resultado.get("ya_existia") and not forzar:
            st.info(
                f"**{ticker_input}** ya tenia modelo asignado: "
                f"**{resultado['modelo_asignado']}**. "
                f"Activa 'Re-evaluar' para forzar una nueva evaluacion."
            )
            return

        # ── Resultados ────────────────────────────────────────
        st.success(
            f"Modelo asignado: **{resultado['modelo_asignado']}**  "
            f"| F1 TEST = {resultado.get('f1_ganador', 0):.4f}"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Filas TRAIN", resultado.get("n_train", "-"))
        c2.metric("Filas TEST",  resultado.get("n_test", "-"))
        c3.metric("Filas BACKTEST", resultado.get("n_backtest", "-"))

        # Tabla comparativa
        f1_todos = resultado.get("f1_todos", {})
        if f1_todos:
            rows = []
            ganador = resultado["modelo_asignado"]
            for scope, vals in sorted(f1_todos.items(), key=lambda x: -x[1]["test"]):
                rows.append({
                    "Modelo":       scope,
                    "F1 TEST":      f"{vals['test']:.4f}",
                    "F1 BACKTEST":  f"{vals['backtest']:.4f}" if vals.get("backtest") is not None else "N/A",
                    "Ganador":      "SI" if scope == ganador else "",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.divider()
        st.info(
            f"**Siguiente paso:** Para que el cron diario escanee **{ticker_input}** "
            f"automaticamente, agregalo a `ACTIVOS` en `src/utils/config.py` y hace "
            f"`git push`. El cron de GitHub Actions lo incluira desde el proximo dia habil."
        )


# ─────────────────────────────────────────────────────────────
# TAB 3: HISTORIAL
# ─────────────────────────────────────────────────────────────

def tab_historial():
    st.subheader("Historial de Alertas")

    # ── Filtros ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        fecha_desde = st.date_input("Desde", value=date.today() - timedelta(days=30))
    with col2:
        fecha_hasta = st.date_input("Hasta", value=date.today())
    with col3:
        df_tickers = query("SELECT DISTINCT ticker FROM alertas_scanner ORDER BY ticker")
        tickers_disp = ["Todos"] + (df_tickers["ticker"].tolist() if not df_tickers.empty else [])
        ticker_sel = st.selectbox("Ticker", tickers_disp)

    col4, col5 = st.columns(2)
    with col4:
        nivel_opts = ["Todos", "COMPRA_FUERTE", "COMPRA", "NEUTRAL", "VENTA", "VENTA_FUERTE"]
        nivel_sel = st.selectbox("Nivel de alerta", nivel_opts)
    with col5:
        prob_min_h = st.slider("ML Prob. minima (%)", 0, 100, 0, 5, key="prob_hist")

    # ── Query ─────────────────────────────────────────────────
    filtros = ["DATE(scan_fecha) BETWEEN :desde AND :hasta"]
    params  = {"desde": str(fecha_desde), "hasta": str(fecha_hasta)}

    if ticker_sel != "Todos":
        filtros.append("ticker = :ticker")
        params["ticker"] = ticker_sel
    if nivel_sel != "Todos":
        filtros.append("alert_nivel = :nivel")
        params["nivel"] = nivel_sel
    if prob_min_h > 0:
        filtros.append("ml_prob_ganancia >= :prob_min")
        params["prob_min"] = prob_min_h / 100

    sql = f"""
        SELECT DATE(scan_fecha) AS fecha, ticker, sector,
               alert_nivel, alert_score, ml_prob_ganancia,
               ml_modelo_usado, precio_cierre
        FROM alertas_scanner
        WHERE {' AND '.join(filtros)}
        ORDER BY scan_fecha DESC, alert_score DESC
        LIMIT 500
    """
    df = query(sql, params)

    if df.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    st.metric("Registros encontrados", len(df))

    # ── Formatear ─────────────────────────────────────────────
    df_show = df.copy()
    df_show["Nivel"]  = df_show["alert_nivel"].map(lambda x: f"{_EMOJIS.get(x, '')} {x}")
    df_show["ML %"]   = df_show["ml_prob_ganancia"].map(lambda x: f"{x:.0%}" if x else "-")
    df_show["Precio"] = df_show["precio_cierre"].map(lambda x: f"${x:.2f}" if x else "-")
    df_show["Score"]  = df_show["alert_score"].map(lambda x: f"{x:.0f}" if x else "-")

    st.dataframe(
        df_show[["fecha", "ticker", "sector", "Nivel", "Score", "ML %",
                 "Precio", "ml_modelo_usado"]].rename(columns={
            "fecha":            "Fecha",
            "ticker":           "Ticker",
            "sector":           "Sector",
            "ml_modelo_usado":  "Modelo",
        }),
        hide_index=True,
        use_container_width=True,
    )

    # ── Grafico de evolucion (si se filtra por ticker) ────────
    if ticker_sel != "Todos" and len(df) > 1:
        st.divider()
        st.subheader(f"Evolucion de {ticker_sel}")
        df_plot = df[["fecha", "ml_prob_ganancia", "alert_score"]].copy()
        df_plot = df_plot.sort_values("fecha").set_index("fecha")
        df_plot.columns = ["ML Prob.", "Alert Score"]
        st.line_chart(df_plot)


# ─────────────────────────────────────────────────────────────
# TAB 4: ANALISIS TECNICO
# ─────────────────────────────────────────────────────────────

def tab_analisis():
    st.subheader("Indicadores Tecnicos — Vista Rapida")

    # ── Query principal: ultima fila por ticker ────────────────
    df = query("""
        SELECT
            i.ticker,
            p.sector,
            i.fecha,
            pd.close                            AS precio,
            ROUND(i.dist_sma21::numeric, 2)     AS vs_sma21_pct,
            ROUND(i.dist_sma50::numeric, 2)     AS vs_sma50_pct,
            ROUND(i.dist_sma200::numeric, 2)    AS vs_sma200_pct,
            ROUND(i.rsi14::numeric, 1)          AS rsi,
            ROUND(i.macd::numeric, 3)           AS macd,
            ROUND(i.macd_signal::numeric, 3)    AS macd_signal,
            ROUND(i.macd_hist::numeric, 3)      AS macd_hist,
            ROUND(i.adx::numeric, 1)            AS adx,
            ROUND(i.vol_relativo::numeric, 2)   AS vol_rel,
            ROUND(i.atr14::numeric, 2)          AS atr,
            ms.estructura_5,
            ms.estructura_10,
            CASE
                WHEN ms.bos_bull_5=1  OR ms.bos_bull_10=1  THEN 'BOS+'
                WHEN ms.bos_bear_5=1  OR ms.bos_bear_10=1  THEN 'BOS-'
                WHEN ms.choch_bull_5=1 OR ms.choch_bull_10=1 THEN 'CHoCH+'
                WHEN ms.choch_bear_5=1 OR ms.choch_bear_10=1 THEN 'CHoCH-'
                ELSE ''
            END                                 AS evento_ms,
            a.alert_nivel,
            ROUND(a.alert_score::numeric)       AS score,
            ROUND(a.ml_prob_ganancia::numeric * 100, 1) AS ml_pct
        FROM (
            SELECT DISTINCT ON (ticker) ticker, fecha,
                   dist_sma21, dist_sma50, dist_sma200,
                   rsi14, macd, macd_signal, macd_hist,
                   adx, vol_relativo, atr14
            FROM indicadores_tecnicos
            ORDER BY ticker, fecha DESC
        ) i
        JOIN (
            SELECT DISTINCT ON (ticker) ticker, sector
            FROM alertas_scanner
            ORDER BY ticker, scan_fecha DESC
        ) p ON p.ticker = i.ticker
        JOIN (
            SELECT DISTINCT ON (ticker) ticker, fecha, close
            FROM precios_diarios
            ORDER BY ticker, fecha DESC
        ) pd ON pd.ticker = i.ticker
        LEFT JOIN (
            SELECT DISTINCT ON (ticker) ticker,
                   estructura_5, estructura_10,
                   bos_bull_5, bos_bear_5, choch_bull_5, choch_bear_5,
                   bos_bull_10, bos_bear_10, choch_bull_10, choch_bear_10
            FROM features_market_structure
            ORDER BY ticker, fecha DESC
        ) ms ON ms.ticker = i.ticker
        LEFT JOIN (
            SELECT DISTINCT ON (ticker) ticker, alert_nivel, alert_score, ml_prob_ganancia
            FROM alertas_scanner
            ORDER BY ticker, scan_fecha DESC
        ) a ON a.ticker = i.ticker
        ORDER BY p.sector, i.ticker
    """)

    if df.empty:
        st.warning("Sin datos disponibles.")
        return

    # ── Filtro por sector ──────────────────────────────────────
    sectores = ["Todos"] + sorted(df["sector"].dropna().unique().tolist())
    sector_sel = st.selectbox("Sector", sectores, key="at_sector")
    if sector_sel != "Todos":
        df = df[df["sector"] == sector_sel]

    # ── Formatear columnas ─────────────────────────────────────
    def _fmt_pct(v):
        if pd.isna(v):
            return "-"
        return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"

    def _fmt_rsi(v):
        if pd.isna(v):
            return "-"
        return f"{v:.0f}"

    def _fmt_macd(v):
        if pd.isna(v):
            return "-"
        return "Alcista" if v > 0 else "Bajista"

    def _fmt_adx(v):
        if pd.isna(v):
            return "-"
        if v >= 25:
            return f"{v:.0f} (Tend)"
        return f"{v:.0f} (Deb)"

    def _fmt_est(v):
        if pd.isna(v):
            return "-"
        return {1: "Alcista", -1: "Bajista", 0: "Neutral"}.get(int(v), "-")

    def _fmt_volrel(v):
        if pd.isna(v):
            return "-"
        return f"{v:.2f}x"

    df_show = pd.DataFrame({
        "Ticker":    df["ticker"],
        "Sector":    df["sector"],
        "Precio":    df["precio"].map(lambda x: f"${x:.2f}" if pd.notna(x) else "-"),
        "vs SMA21":  df["vs_sma21_pct"].map(_fmt_pct),
        "vs SMA50":  df["vs_sma50_pct"].map(_fmt_pct),
        "vs SMA200": df["vs_sma200_pct"].map(_fmt_pct),
        "RSI":       df["rsi"].map(_fmt_rsi),
        "MACD":      df["macd_hist"].map(_fmt_macd),
        "ADX":       df["adx"].map(_fmt_adx),
        "Vol Rel":   df["vol_rel"].map(_fmt_volrel),
        "ATR":       df["atr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        "Est.5":     df["estructura_5"].map(_fmt_est),
        "Est.10":    df["estructura_10"].map(_fmt_est),
        "Evento MS": df["evento_ms"].fillna(""),
        "Senal":     df["alert_nivel"].map(lambda x: f"{_EMOJIS.get(x,'')} {x}" if pd.notna(x) else "-"),
        "Score":     df["score"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "-"),
        "ML%":       df["ml_pct"].map(lambda x: f"{x:.0f}%" if pd.notna(x) else "-"),
    })

    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker":    st.column_config.TextColumn(width=65),
            "Sector":    st.column_config.TextColumn(width=160),
            "Precio":    st.column_config.TextColumn(width=80),
            "vs SMA21":  st.column_config.TextColumn(width=80, help="% distancia precio vs SMA 21"),
            "vs SMA50":  st.column_config.TextColumn(width=80, help="% distancia precio vs SMA 50"),
            "vs SMA200": st.column_config.TextColumn(width=90, help="% distancia precio vs SMA 200"),
            "RSI":       st.column_config.TextColumn(width=55, help="RSI 14 periodos"),
            "MACD":      st.column_config.TextColumn(width=80, help="Histograma MACD positivo=Alcista"),
            "ADX":       st.column_config.TextColumn(width=95, help="ADX: >25 tendencia definida"),
            "Vol Rel":   st.column_config.TextColumn(width=75, help="Volumen relativo vs promedio 20d"),
            "ATR":       st.column_config.TextColumn(width=60, help="Average True Range 14"),
            "Est.5":     st.column_config.TextColumn(width=80, help="Estructura market structure 5 barras"),
            "Est.10":    st.column_config.TextColumn(width=80, help="Estructura market structure 10 barras"),
            "Evento MS": st.column_config.TextColumn(width=85, help="Ultimo evento BOS/CHoCH"),
            "Senal":     st.column_config.TextColumn(width=150, help="Senal ML mas reciente"),
            "Score":     st.column_config.TextColumn(width=60),
            "ML%":       st.column_config.TextColumn(width=60),
        },
    )

    st.caption(
        "vs SMA: % por encima (+) o por debajo (-) de la media. "
        "MACD: positivo del histograma. "
        "ADX >25 = tendencia definida. "
        "Vol Rel: volumen actual / promedio 20d. "
        "Est: estructura de mercado (5 y 10 barras). "
        "Evento MS: ultimo BOS o CHoCH registrado."
    )


# ─────────────────────────────────────────────────────────────
# TAB 5: ESTRUCTURA DE VELAS
# ─────────────────────────────────────────────────────────────

def tab_velas():
    st.subheader("Estructura de Velas — Ultimos 5 Dias")

    # ── Selector ticker ────────────────────────────────────────
    df_tickers = query("SELECT DISTINCT ticker FROM precios_diarios ORDER BY ticker")
    tickers = df_tickers["ticker"].tolist() if not df_tickers.empty else []
    if not tickers:
        st.warning("Sin tickers disponibles.")
        return
    ticker_sel = st.selectbox("Ticker", tickers, key="velas_ticker")

    # ── OHLCV ─────────────────────────────────────────────────
    df_p = query("""
        SELECT fecha, open, high, low, close, volume
        FROM precios_diarios
        WHERE ticker = :t
        ORDER BY fecha DESC
        LIMIT 5
    """, {"t": ticker_sel})

    if df_p.empty:
        st.warning(f"Sin datos de precio para {ticker_sel}.")
        return

    # ── Patrones PA ────────────────────────────────────────────
    df_pat = query("""
        SELECT fecha,
               COALESCE(patron_hammer, 0)         AS patron_hammer,
               COALESCE(patron_engulfing_bull, 0) AS patron_engulfing_bull,
               COALESCE(patron_engulfing_bear, 0) AS patron_engulfing_bear,
               COALESCE(patron_shooting_star, 0)  AS patron_shooting_star
        FROM features_precio_accion
        WHERE ticker = :t
        ORDER BY fecha DESC
        LIMIT 5
    """, {"t": ticker_sel})

    # ── Vol relativo ───────────────────────────────────────────
    df_vol = query("""
        SELECT fecha, vol_relativo
        FROM indicadores_tecnicos
        WHERE ticker = :t
        ORDER BY fecha DESC
        LIMIT 5
    """, {"t": ticker_sel})

    # ── Merge ──────────────────────────────────────────────────
    df_p["fecha"] = pd.to_datetime(df_p["fecha"])
    df = df_p.copy()

    if not df_pat.empty:
        df_pat["fecha"] = pd.to_datetime(df_pat["fecha"])
        df = df.merge(df_pat, on="fecha", how="left")
    else:
        for c in ["patron_hammer", "patron_engulfing_bull",
                  "patron_engulfing_bear", "patron_shooting_star"]:
            df[c] = 0

    if not df_vol.empty:
        df_vol["fecha"] = pd.to_datetime(df_vol["fecha"])
        df = df.merge(df_vol[["fecha", "vol_relativo"]], on="fecha", how="left")
    else:
        df["vol_relativo"] = float("nan")

    df = df.sort_values("fecha", ascending=False).reset_index(drop=True)

    # ── Calculos de estructura ─────────────────────────────────
    df["rango"]      = (df["high"] - df["low"]).replace(0, float("nan"))
    df["cuerpo"]     = (df["close"] - df["open"]).abs()
    df["sombra_sup"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["sombra_inf"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["cuerpo_pct"]     = (df["cuerpo"]     / df["rango"] * 100).round(1)
    df["sombra_sup_pct"] = (df["sombra_sup"] / df["rango"] * 100).round(1)
    df["sombra_inf_pct"] = (df["sombra_inf"] / df["rango"] * 100).round(1)
    df["cierre_rango"]   = ((df["close"] - df["low"]) / df["rango"] * 100).round(1)
    df["es_alcista"]     = df["close"] >= df["open"]

    # ── Helpers de etiqueta ────────────────────────────────────
    def _lbl_cuerpo(v):
        if pd.isna(v): return "n/d"
        if v > 70: return f"Grande ({v:.0f}%)"
        if v > 40: return f"Medio ({v:.0f}%)"
        if v > 20: return f"Pequeno ({v:.0f}%)"
        return f"Doji ({v:.0f}%)"

    def _lbl_ssup(v):
        if pd.isna(v): return "n/d"
        if v > 60: return f"Larga ({v:.0f}%)"
        if v > 30: return f"Media ({v:.0f}%)"
        return f"Corta ({v:.0f}%)"

    def _lbl_sinf(v):
        if pd.isna(v): return "n/d"
        if v > 60: return f"Larga ({v:.0f}%)"
        if v > 30: return f"Media ({v:.0f}%)"
        return f"Corta ({v:.0f}%)"

    def _lbl_cr(v):
        if pd.isna(v): return "n/d"
        if v > 80: return f"Muy alto ({v:.0f}%)"
        if v > 60: return f"Alto ({v:.0f}%)"
        if v > 40: return f"Neutro ({v:.0f}%)"
        if v > 20: return f"Bajo ({v:.0f}%)"
        return f"Muy bajo ({v:.0f}%)"

    def _patron(r):
        if r.get("patron_hammer", 0):         return "Hammer"
        if r.get("patron_engulfing_bull", 0): return "Engulfing Bull"
        if r.get("patron_engulfing_bear", 0): return "Engulfing Bear"
        if r.get("patron_shooting_star", 0):  return "Shooting Star"
        return "-"

    # ── Resumen de los 5 dias ──────────────────────────────────
    n_alc   = int(df["es_alcista"].sum())
    n_baj   = len(df) - n_alc
    avg_cr  = df["cierre_rango"].mean()
    avg_vol = df["vol_relativo"].mean() if df["vol_relativo"].notna().any() else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Alcistas / Bajistas (5d)", f"{n_alc}  /  {n_baj}")
    c2.metric("Cierre en rango prom.", f"{avg_cr:.0f}%" if pd.notna(avg_cr) else "-",
              help="0% = cierra en minimo, 100% = cierra en maximo")
    c3.metric("Volumen relativo prom.", f"{avg_vol:.2f}x" if pd.notna(avg_vol) else "-",
              help="vs promedio 20 dias")

    st.divider()

    # ── Tabla principal ────────────────────────────────────────
    rows = []
    for _, r in df.iterrows():
        alc = bool(r["es_alcista"])
        tipo_emoji = "🟢" if alc else "🔴"

        ss  = r["sombra_sup_pct"]
        si  = r["sombra_inf_pct"]
        cr  = r["cierre_rango"]
        vol = r.get("vol_relativo", float("nan"))

        # Sombra sup larga = presion vendedora = rojo
        ss_e = "🔴" if (pd.notna(ss) and ss > 60) else ("🟡" if (pd.notna(ss) and ss > 30) else "🟢")
        # Sombra inf larga = soporte = verde
        si_e = "🟢" if (pd.notna(si) and si > 60) else ("🟡" if (pd.notna(si) and si > 30) else "🔴")
        # Cierre alto = compradores = verde
        cr_e = "🟢" if (pd.notna(cr) and cr > 60) else ("🟡" if (pd.notna(cr) and cr > 40) else "🔴")

        if pd.notna(vol):
            vol_e   = "🟢" if vol > 1.3 else ("🟡" if vol >= 0.9 else "🔴")
            vol_txt = f"{vol_e} {vol:.2f}x"
        else:
            vol_txt = "-"

        rows.append({
            "Fecha":        r["fecha"].strftime("%d/%m"),
            "Cierre":       f"${float(r['close']):.2f}",
            "Tipo":         f"{tipo_emoji} {'Alcista' if alc else 'Bajista'}",
            "Cuerpo":       _lbl_cuerpo(r["cuerpo_pct"]),
            "S. Superior":  f"{ss_e} {_lbl_ssup(ss)}",
            "S. Inferior":  f"{si_e} {_lbl_sinf(si)}",
            "Cierre/Rango": f"{cr_e} {_lbl_cr(cr)}",
            "Patron":       _patron(r),
            "Volumen":      vol_txt,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Fecha":        st.column_config.TextColumn(width=65),
            "Cierre":       st.column_config.TextColumn(width=85),
            "Tipo":         st.column_config.TextColumn(width=110),
            "Cuerpo":       st.column_config.TextColumn(width=145),
            "S. Superior":  st.column_config.TextColumn(width=165,
                            help="Sombra superior como % del rango total. Larga = rechazo en maximos."),
            "S. Inferior":  st.column_config.TextColumn(width=165,
                            help="Sombra inferior como % del rango total. Larga = soporte en minimos."),
            "Cierre/Rango": st.column_config.TextColumn(width=180,
                            help="Donde cerro dentro del rango del dia. 100% = cerro en maximo."),
            "Patron":       st.column_config.TextColumn(width=145),
            "Volumen":      st.column_config.TextColumn(width=110,
                            help="Volumen del dia vs promedio 20 dias."),
        },
    )

    # ── Lectura rapida — un bloque por dia ─────────────────────
    st.divider()
    st.markdown("**Lectura rapida:**")

    for _, r in df.iterrows():
        alc      = bool(r["es_alcista"])
        tipo_str = "ALCISTA" if alc else "BAJISTA"
        fecha_lbl = r["fecha"].strftime("%d/%m/%Y")
        partes   = []

        cp = r["cuerpo_pct"]
        if pd.notna(cp):
            if cp > 70:
                partes.append(f"cuerpo grande ({cp:.0f}%), conviccion {tipo_str.lower()} fuerte")
            elif cp < 20:
                partes.append(f"vela tipo doji ({cp:.0f}%), indecision, sin conviccion clara")
            else:
                partes.append(f"cuerpo {_lbl_cuerpo(cp).lower()}")

        ss = r["sombra_sup_pct"]
        if pd.notna(ss):
            if ss > 60:
                partes.append(f"sombra superior larga ({ss:.0f}%): rechazo en maximos, presion vendedora")
            elif ss > 30:
                partes.append(f"sombra superior media ({ss:.0f}%): resistencia moderada")

        si = r["sombra_inf_pct"]
        if pd.notna(si):
            if si > 60:
                partes.append(f"sombra inferior larga ({si:.0f}%): soporte activo, compradores absorbieron")
            elif si > 30:
                partes.append(f"sombra inferior media ({si:.0f}%): soporte moderado")

        cr = r["cierre_rango"]
        if pd.notna(cr):
            if cr > 80:
                partes.append(f"cerro en el {cr:.0f}% del rango: compradores dominaron al cierre")
            elif cr < 20:
                partes.append(f"cerro en el {cr:.0f}% del rango: vendedores dominaron al cierre")
            elif cr > 60:
                partes.append(f"cierre alto ({cr:.0f}%): precio sostenido")
            elif cr < 40:
                partes.append(f"cierre bajo ({cr:.0f}%): precio debil")

        pat = _patron(r)
        if pat != "-":
            partes.append(f"patron {pat}")

        vol = r.get("vol_relativo", float("nan"))
        if pd.notna(vol):
            if vol > 1.5:
                partes.append(f"volumen {vol:.1f}x: confirma el movimiento")
            elif vol < 0.8:
                partes.append(f"volumen bajo ({vol:.1f}x): movimiento sin respaldo")

        texto  = f"<b>{fecha_lbl} ({tipo_str}):</b> " + " | ".join(partes) + "."
        bg     = "#1a3a1a" if alc else "#3a1a1a"
        border = "#26a69a" if alc else "#ef5350"
        st.markdown(
            f'<div style="background:{bg};padding:10px 14px;border-radius:6px;'
            f'border-left:3px solid {border};margin-bottom:8px;font-size:0.88rem;">'
            f'{texto}</div>',
            unsafe_allow_html=True,
        )

    # ── Guia de interpretacion (expandible) ───────────────────
    st.divider()
    with st.expander("Guia de interpretacion"):
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("**Sombra Superior**")
            st.markdown("🔴 **Larga (>60%)** — Rechazo en maximos. Vendedores tomaron control. Resistencia activa.")
            st.markdown("🟡 **Media (30-60%)** — Resistencia moderada, no dominante.")
            st.markdown("🟢 **Corta (<30%)** — Sin resistencia. Precio cerro sostenido cerca del maximo.")
            st.markdown("")
            st.markdown("**Sombra Inferior**")
            st.markdown("🟢 **Larga (>60%)** — Rechazo en minimos. Compradores absorbieron la caida. Soporte activo.")
            st.markdown("🟡 **Media (30-60%)** — Soporte moderado.")
            st.markdown("🔴 **Corta (<30%)** — Sin defensa en los minimos. Precio cayo sin resistencia.")
        with gc2:
            st.markdown("**Cuerpo**")
            st.markdown("**Grande (>70%)** — Fuerte conviccion. Dominio claro de compradores o vendedores.")
            st.markdown("**Medio (40-70%)** — Conviccion moderada. Direccion clara pero sin dominio total.")
            st.markdown("**Pequeno (20-40%)** — Poca conviccion. Equilibrio con leve sesgo.")
            st.markdown("**Doji (<20%)** — Indecision. Compradores y vendedores empatados.")
            st.markdown("")
            st.markdown("**Cierre en Rango** *(0% = cerro en minimo | 100% = cerro en maximo)*")
            st.markdown("🟢 **>80%** — Compradores dominaron al cierre. Sesgo alcista fuerte.")
            st.markdown("🟢 **60-80%** — Cierre alto, precio sostenido.")
            st.markdown("🟡 **40-60%** — Neutro, sin conviccion al cierre.")
            st.markdown("🔴 **20-40%** — Cierre bajo, sesgo bajista.")
            st.markdown("🔴 **<20%** — Vendedores dominaron al cierre. Sesgo bajista fuerte.")
        st.markdown("")
        st.markdown("**Patrones**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Hammer** — Sombra inferior larga tras caida. Rechazo fuerte en minimos. "
                        "Posible reversion alcista. Mas valido con volumen alto.")
            st.markdown("**Shooting Star** — Sombra superior larga tras subida. Rechazo en maximos. "
                        "Posible reversion bajista.")
        with col_p2:
            st.markdown("**Engulfing Bull** — Vela alcista envuelve completamente la bajista anterior. "
                        "Reversion alcista fuerte.")
            st.markdown("**Engulfing Bear** — Vela bajista envuelve completamente la alcista anterior. "
                        "Reversion bajista fuerte.")


# ─────────────────────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────────────────────

st.title("📈 Scanner ML — Activos Bursatiles")
st.caption("Senales basadas en ML V3 + Price Action + Market Structure")

# ── Info de entrenamiento de modelos ──────────────────────────
try:
    df_modelos = query("""
        SELECT scope, algoritmo, f1_test, created_at
        FROM modelos_produccion
        WHERE modelo_version = 'v3'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    if not df_modelos.empty:
        ultima_fecha_train = pd.to_datetime(df_modelos.iloc[0]["created_at"]).strftime("%Y-%m-%d")
        df_v3 = query("""
            SELECT scope, algoritmo, f1_test
            FROM modelos_produccion
            WHERE modelo_version = 'v3' AND f1_test IS NOT NULL
            ORDER BY scope
        """)
        resumen_f1 = " | ".join(
            f"{r['scope']}: {r['f1_test']:.4f}"
            for _, r in df_v3.iterrows()
        ) if not df_v3.empty else ""
        st.caption(
            f"**Ultimo entrenamiento de modelos V3:** {ultima_fecha_train}   "
            + (f"— F1 TEST: {resumen_f1}" if resumen_f1 else "")
        )
except Exception:
    pass

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "➕ Agregar Ticker",
    "📋 Historial",
    "📈 Analisis Tecnico",
    "🕯️ Velas",
])

with tab1:
    tab_dashboard()

with tab2:
    tab_agregar()

with tab3:
    tab_historial()

with tab4:
    tab_analisis()

with tab5:
    tab_velas()
