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
# TAB 5: GRAFICO DE VELAS
# ─────────────────────────────────────────────────────────────

def tab_velas():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader("Grafico de Velas Diarias")

    # ── Controles ─────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        df_tickers = query("SELECT DISTINCT ticker FROM precios_diarios ORDER BY ticker")
        tickers = df_tickers["ticker"].tolist() if not df_tickers.empty else []
        ticker_sel = st.selectbox("Ticker", tickers, key="velas_ticker")
    with col2:
        periodo = st.selectbox(
            "Periodo",
            ["1 mes", "3 meses", "6 meses", "1 ano", "2 anos"],
            index=2,
            key="velas_periodo",
        )
        periodo_dias = {"1 mes": 30, "3 meses": 90, "6 meses": 180, "1 ano": 365, "2 anos": 730}[periodo]
    with col3:
        mostrar_bb = st.checkbox("Bollinger Bands", value=True, key="velas_bb")

    params = {"t": ticker_sel, "dias": periodo_dias}

    # ── Datos de precio ───────────────────────────────────────
    df_p = query("""
        SELECT fecha, open, high, low, close, volume
        FROM precios_diarios
        WHERE ticker = :t AND fecha >= CURRENT_DATE - :dias
        ORDER BY fecha
    """, params)

    # ── Indicadores ───────────────────────────────────────────
    df_i = query("""
        SELECT fecha, sma21, sma50, sma200,
               bb_upper, bb_lower, bb_middle,
               rsi14, macd, macd_signal, macd_hist,
               adx, atr14
        FROM indicadores_tecnicos
        WHERE ticker = :t AND fecha >= CURRENT_DATE - :dias
        ORDER BY fecha
    """, params)

    # ── Market structure (solo eventos BOS/CHoCH) ─────────────
    df_ms = query("""
        SELECT fecha,
               bos_bull_5, bos_bear_5, choch_bull_5, choch_bear_5,
               bos_bull_10, bos_bear_10, choch_bull_10, choch_bear_10,
               estructura_5, estructura_10
        FROM features_market_structure
        WHERE ticker = :t AND fecha >= CURRENT_DATE - :dias
          AND (bos_bull_5=1 OR bos_bear_5=1 OR choch_bull_5=1 OR choch_bear_5=1
               OR bos_bull_10=1 OR bos_bear_10=1 OR choch_bull_10=1 OR choch_bear_10=1)
        ORDER BY fecha
    """, params)

    # ── Ultima alerta ─────────────────────────────────────────
    df_al = query("""
        SELECT alert_nivel, alert_score, ml_prob_ganancia
        FROM alertas_scanner
        WHERE ticker = :t
        ORDER BY scan_fecha DESC
        LIMIT 1
    """, {"t": ticker_sel})

    if df_p.empty:
        st.warning(f"Sin datos de precio para {ticker_sel}.")
        return

    df_p["fecha"] = pd.to_datetime(df_p["fecha"])
    df_i["fecha"] = pd.to_datetime(df_i["fecha"])
    df = df_p.merge(df_i, on="fecha", how="left")

    if not df_ms.empty:
        df_ms["fecha"] = pd.to_datetime(df_ms["fecha"])

    # ── Metricas resumen ──────────────────────────────────────
    ult = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else ult

    delta_precio = float(ult["close"]) - float(prev["close"])
    delta_pct    = delta_precio / float(prev["close"]) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precio", f"${ult['close']:.2f}", f"{delta_pct:+.2f}%")
    c2.metric("RSI 14", f"{ult['rsi14']:.1f}" if pd.notna(ult['rsi14']) else "-")
    c3.metric("ADX",    f"{ult['adx']:.1f}"   if pd.notna(ult['adx'])   else "-")
    c4.metric("ATR",    f"${ult['atr14']:.2f}" if pd.notna(ult['atr14']) else "-")

    if not df_al.empty:
        al    = df_al.iloc[0]
        nivel = al["alert_nivel"] or "NEUTRAL"
        c5.metric("Senal ML", f"{_EMOJIS.get(nivel, '')} {nivel}")
    else:
        c5.metric("Senal ML", "-")

    # ── Figura: 4 subplots ────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=(f"{ticker_sel} — Velas Diarias", "Volumen", "RSI 14", "MACD"),
    )

    # Velas
    fig.add_trace(go.Candlestick(
        x=df["fecha"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Precio",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # SMAs
    for col_nm, color, label in [
        ("sma21",  "#f6c90e", "SMA 21"),
        ("sma50",  "#fb8c00", "SMA 50"),
        ("sma200", "#e040fb", "SMA 200"),
    ]:
        if col_nm in df.columns:
            fig.add_trace(go.Scatter(
                x=df["fecha"], y=df[col_nm], name=label,
                line=dict(color=color, width=1.3),
            ), row=1, col=1)

    # Bollinger Bands (opcional)
    if mostrar_bb and "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["fecha"], y=df["bb_upper"], name="BB+",
            line=dict(color="rgba(150,150,150,0.5)", width=0.8, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["fecha"], y=df["bb_lower"], name="BB-",
            line=dict(color="rgba(150,150,150,0.5)", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.06)",
            showlegend=False,
        ), row=1, col=1)

    # Marcadores BOS / CHoCH
    if not df_ms.empty:
        fecha_close = df.set_index("fecha")["close"]

        def _get_precio(f):
            try:
                return float(fecha_close.loc[f])
            except KeyError:
                idx = fecha_close.index.get_indexer([f], method="nearest")[0]
                return float(fecha_close.iloc[idx])

        _markers = [
            ("bos_bull_5",   "bos_bull_10",   "BOS+",    "triangle-up",   "lime",   0.994),
            ("bos_bear_5",   "bos_bear_10",   "BOS-",    "triangle-down", "#ef5350", 1.006),
            ("choch_bull_5", "choch_bull_10", "CHoCH+",  "star",          "cyan",   0.987),
            ("choch_bear_5", "choch_bear_10", "CHoCH-",  "star",          "orange", 1.013),
        ]
        for col5, col10, label, symbol, color, offset in _markers:
            mask = df_ms[col5].astype(bool) | df_ms[col10].astype(bool)
            eventos = df_ms[mask]
            if eventos.empty:
                continue
            xs = eventos["fecha"].tolist()
            ys = [_get_precio(f) * offset for f in xs]
            pos = "bottom center" if offset < 1 else "top center"
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                text=[label] * len(xs),
                textposition=pos,
                marker=dict(symbol=symbol, color=color, size=9),
                name=label,
                showlegend=True,
            ), row=1, col=1)

    # Volumen coloreado
    colors_vol = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df["fecha"], y=df["volume"],
        marker_color=colors_vol, name="Volumen", showlegend=False,
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["rsi14"], name="RSI",
        line=dict(color="#2196F3", width=1.5), showlegend=False,
    ), row=3, col=1)
    for lvl, clr in [(70, "rgba(239,83,80,0.4)"), (30, "rgba(38,166,154,0.4)"), (50, "rgba(180,180,180,0.25)")]:
        fig.add_shape(
            type="line",
            x0=df["fecha"].iloc[0], x1=df["fecha"].iloc[-1],
            y0=lvl, y1=lvl,
            line=dict(color=clr, width=0.8, dash="dash"),
            row=3, col=1,
        )

    # MACD
    colors_hist = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df["fecha"], y=df["macd_hist"],
        marker_color=colors_hist, name="Histograma", showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["macd"], name="MACD",
        line=dict(color="#2196F3", width=1.2),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["macd_signal"], name="Signal",
        line=dict(color="#FF5722", width=1.2),
    ), row=4, col=1)

    # ── Layout ────────────────────────────────────────────────
    fig.update_layout(
        height=800,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(size=10),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_yaxes(title_text="USD",  row=1, col=1)
    fig.update_yaxes(title_text="Vol",  row=2, col=1)
    fig.update_yaxes(title_text="RSI",  row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "BOS+/- = Break of Structure alcista/bajista  |  "
        "CHoCH+/- = Change of Character alcista/bajista  |  "
        "Datos: precios_diarios + indicadores_tecnicos + features_market_structure (Railway PostgreSQL)"
    )


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
