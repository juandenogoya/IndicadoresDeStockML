"""
iol_tab.py
Tab "Argentina (IOL)" para streamlit_app.py

Sub-tabs:
    1. Scanner AR    -- alertas diarias desde alertas_scanner_ar (DB)
    2. Analisis Live -- fetch IOL live, indicadores + velas por ticker
    3. Opciones AR   -- chain con Greeks live + PCR + historial IV (DB)
    4. Estrategias   -- resultados de backtest AR + lanzar BT desde la UI
    5. Historial     -- historial de alertas_scanner_ar filtrable

Fuente de datos:
    - Precios e indicadores: IOL REST API (live, sin tocar DB)
    - Alertas y BT results:  Railway DB (alertas_scanner_ar, resultados_bt_ar)
    - Greeks opciones:       IOL live + opciones_ar_gregas (DB) para historico

Todos los calculos en ARS original (sin conversion de moneda).
"""

import os
import time
import warnings
import traceback
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paleta compartida con el resto del app ────────────────────────────────────
_BG = {
    "COMPRA_FUERTE": "#0d4f0d",
    "COMPRA":        "#1a6e2a",
    "NEUTRAL":       "#4a4a00",
    "VENTA":         "#7a2800",
}
_EMO = {
    "COMPRA_FUERTE": "🟢🟢",
    "COMPRA":        "🟢",
    "NEUTRAL":       "🟡",
    "VENTA":         "🔴",
}


# ── Cliente IOL (cacheado) ────────────────────────────────────────────────────

@st.cache_resource
def _get_iol_client():
    """Inicializa el cliente IOL una sola vez por sesion."""
    import sys, pathlib
    root = pathlib.Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Credenciales: st.secrets (Cloud) o .env (local)
    username = None
    password = None
    try:
        username = st.secrets.get("IOL_USERNAME")
        password = st.secrets.get("IOL_PASSWORD")
    except Exception:
        pass
    if not username:
        username = os.getenv("IOL_USERNAME")
        password = os.getenv("IOL_PASSWORD")

    if not username or not password:
        return None

    os.environ["IOL_USERNAME"] = username
    os.environ["IOL_PASSWORD"] = password

    from src.data.iol_client import IOLClient
    return IOLClient()


def _client_ok() -> bool:
    try:
        c = _get_iol_client()
        return c is not None
    except Exception:
        return False


def _client_error_msg() -> str:
    """Retorna mensaje de error de inicializacion del cliente, o '' si esta OK."""
    try:
        c = _get_iol_client()
        if c is None:
            return "IOL_USERNAME / IOL_PASSWORD no configuradas en .env o st.secrets."
        return ""
    except Exception as e:
        return str(e)


# ── Calculos de indicadores (standalone, sin dependencias externas) ───────────

def _sma(s, n):
    return s.rolling(n).mean()


def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    ms = m.ewm(span=sig, adjust=False).mean()
    return m, ms, m - ms


def _atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _adx(df, n=14):
    pdm = df["high"].diff().clip(lower=0)
    mdm = (-df["low"].diff()).clip(lower=0)
    pdm[pdm < mdm.values] = 0
    mdm[mdm < pdm.values] = 0
    atr = _atr(df, n)
    pdi = 100 * pdm.rolling(n).mean() / atr.replace(0, np.nan)
    mdi = 100 * mdm.rolling(n).mean() / atr.replace(0, np.nan)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.rolling(n).mean()


def _bb(s, n=20, k=2):
    m = s.rolling(n).mean()
    std = s.rolling(n).std()
    return m + k * std, m, m - k * std


def _calcular_todos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["sma20"]  = _sma(c, 20)
    df["sma50"]  = _sma(c, 50)
    df["sma200"] = _sma(c, 200)
    df["dist_sma20"]  = (c - df["sma20"])  / df["sma20"]  * 100
    df["dist_sma50"]  = (c - df["sma50"])  / df["sma50"]  * 100
    df["dist_sma200"] = (c - df["sma200"]) / df["sma200"] * 100
    df["rsi14"] = _rsi(c, 14)
    df["macd"], df["macd_sig"], df["macd_hist"] = _macd(c)
    df["atr14"] = _atr(df, 14)
    df["adx"]   = _adx(df, 14)
    df["bb_up"], df["bb_mid"], df["bb_lo"] = _bb(c, 20)
    df["vol_rel"] = df["volume"] / df["volume"].rolling(20).mean()
    return df


def _swings(highs, lows, n=10):
    sh, sl = [], []
    for i in range(n, len(highs) - n):
        if highs[i] == max(highs[i-n:i+n+1]):
            sh.append(i)
        if lows[i]  == min(lows[i-n:i+n+1]):
            sl.append(i)
    return sh, sl


def _estructura(df: pd.DataFrame, n=10) -> dict:
    h, l = df["high"].values, df["low"].values
    c = df["close"].values
    sh, sl = _swings(h, l, n)
    res = {"estructura": 0, "bos_bull": 0, "bos_bear": 0,
           "choch_bull": 0, "choch_bear": 0}
    if len(sh) < 2 or len(sl) < 2:
        return res
    sh1, sh2 = h[sh[-1]], h[sh[-2]]
    sl1, sl2 = l[sl[-1]], l[sl[-2]]
    if sh1 > sh2 and sl1 > sl2:
        res["estructura"] = 1
    elif sh1 < sh2 and sl1 < sl2:
        res["estructura"] = -1
    if c[-1] > h[sh[-1]]:
        res["bos_bull"] = 1
    if c[-1] < l[sl[-1]]:
        res["bos_bear"] = 1
    if res["estructura"] == -1 and res["bos_bull"]:
        res["choch_bull"] = 1
    if res["estructura"] == 1  and res["bos_bear"]:
        res["choch_bear"] = 1
    return res


@st.cache_data(ttl=300)
def _fetch_historia(ticker: str):
    """Fetch de precios IOL cacheado 5 minutos."""
    client = _get_iol_client()
    if client is None:
        return None
    desde = (date.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    hasta = date.today().strftime("%Y-%m-%d")
    df = client.get_price_history(ticker, from_date=desde, to_date=hasta)
    if df is None or df.empty:
        return None
    return _calcular_todos(df)


@st.cache_data(ttl=60)
def _fetch_quote(ticker: str):
    client = _get_iol_client()
    if client is None:
        return None
    try:
        return client.get_quote(ticker)
    except Exception:
        return None


@st.cache_data(ttl=120)
def _fetch_opciones(ticker: str):
    client = _get_iol_client()
    if client is None:
        return None, None, "Credenciales IOL no configuradas."
    try:
        raw  = client.get_options_chain_raw(ticker)
        df   = client.get_options_chain_df(ticker)
        return raw, df, None
    except Exception as e:
        return None, None, str(e)


# ── Helpers de formato ────────────────────────────────────────────────────────

def _fmt_pct(v, decimals=1):
    if pd.isna(v):
        return "-"
    return f"+{v:.{decimals}f}%" if v >= 0 else f"{v:.{decimals}f}%"


def _fmt_ars(v):
    if pd.isna(v):
        return "-"
    return f"$ {v:,.2f}"


def _est_label(v):
    return {1: "Alcista", -1: "Bajista", 0: "Neutral"}.get(int(v) if not pd.isna(v) else 0, "-")


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 1: SCANNER AR
# ─────────────────────────────────────────────────────────────────────────────

def _subtab_scanner(query_fn):
    st.subheader("Scanner Argentina — Alertas Diarias")

    # Ultima fecha disponible en DB
    df_f = query_fn("SELECT MAX(DATE(scan_fecha)) AS ult FROM alertas_scanner_ar")
    ultima = df_f.iloc[0]["ult"] if not df_f.empty else None

    if ultima is None:
        st.info("Sin datos aun. Ejecuta `scripts/37_scanner_ar.py` para generar la primera corrida.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        fecha_sel = st.date_input("Fecha", value=ultima, max_value=date.today(),
                                  key="sc_fecha")
    with col2:
        df_sec = query_fn("SELECT DISTINCT sector FROM activos_ar ORDER BY sector")
        sectores = ["Todos"] + df_sec["sector"].tolist()
        sector_sel = st.selectbox("Sector", sectores, key="sc_sector")
    with col3:
        nivel_opts = ["Todos", "COMPRA_FUERTE", "COMPRA", "NEUTRAL", "VENTA"]
        nivel_sel = st.selectbox("Nivel", nivel_opts, key="sc_nivel")

    sql = """
        SELECT ticker, sector, alert_nivel, alert_score,
               precio_cierre, precio_fecha,
               rsi14, macd_hist, dist_sma50, dist_sma200,
               adx, estructura_10, bos_bull_10, choch_bull_10,
               alert_detalle
        FROM alertas_scanner_ar
        WHERE DATE(scan_fecha) = :fecha
        ORDER BY alert_score DESC
    """
    df = query_fn(sql, {"fecha": str(fecha_sel)})

    if df.empty:
        st.warning(f"Sin datos para {fecha_sel}.")
        return

    if sector_sel != "Todos":
        df = df[df.sector == sector_sel]
    if nivel_sel != "Todos":
        df = df[df.alert_nivel == nivel_sel]

    # Metricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("🟢🟢 Compra Fuerte", int((df.alert_nivel == "COMPRA_FUERTE").sum()))
    c3.metric("🟢 Compra",          int((df.alert_nivel == "COMPRA").sum()))
    c4.metric("🟡 Neutral / 🔴 Venta",
              int(df.alert_nivel.isin(["NEUTRAL", "VENTA"]).sum()))

    st.divider()

    # Tabla
    df_show = pd.DataFrame({
        "Ticker":      df["ticker"],
        "Sector":      df["sector"],
        "Nivel":       df["alert_nivel"].map(lambda x: f"{_EMO.get(x,'')} {x}"),
        "Score":       df["alert_score"].map(lambda x: f"{x:.0f}"),
        "Precio ARS":  df["precio_cierre"].map(_fmt_ars),
        "RSI":         df["rsi14"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-"),
        "MACD Hist":   df["macd_hist"].map(lambda x: "Alc" if pd.notna(x) and x > 0 else "Baj"),
        "vs SMA50":    df["dist_sma50"].map(_fmt_pct),
        "vs SMA200":   df["dist_sma200"].map(_fmt_pct),
        "ADX":         df["adx"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "-"),
        "Estructura":  df["estructura_10"].map(_est_label),
        "BOS+":        df["bos_bull_10"].map(lambda x: "Si" if x == 1 else ""),
        "CHoCH+":      df["choch_bull_10"].map(lambda x: "Si" if x == 1 else ""),
        "Detalle":     df["alert_detalle"].fillna(""),
    })

    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker":     st.column_config.TextColumn(width=70),
            "Nivel":      st.column_config.TextColumn(width=155),
            "Score":      st.column_config.TextColumn(width=60),
            "Precio ARS": st.column_config.TextColumn(width=110),
            "RSI":        st.column_config.TextColumn(width=55),
            "MACD Hist":  st.column_config.TextColumn(width=70),
            "vs SMA50":   st.column_config.TextColumn(width=80),
            "vs SMA200":  st.column_config.TextColumn(width=90),
            "ADX":        st.column_config.TextColumn(width=50),
            "Estructura": st.column_config.TextColumn(width=85),
            "BOS+":       st.column_config.TextColumn(width=50),
            "CHoCH+":     st.column_config.TextColumn(width=60),
            "Detalle":    st.column_config.TextColumn(width=280),
        },
    )
    st.caption(f"Fecha escaneo: {fecha_sel} | {len(df)} tickers | precios en ARS")


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 2: ANALISIS LIVE
# ─────────────────────────────────────────────────────────────────────────────

def _subtab_analisis(query_fn):
    st.subheader("Analisis Live por Ticker")

    if not _client_ok():
        msg = _client_error_msg()
        st.error(msg or "No se pudo inicializar el cliente IOL.")
        return

    # Selector
    df_ar = query_fn("SELECT ticker, sector FROM activos_ar WHERE activo=true ORDER BY ticker")
    tickers_ar = df_ar["ticker"].tolist() if not df_ar.empty else []

    col1, col2 = st.columns([2, 4])
    with col1:
        if tickers_ar:
            ticker_sel = st.selectbox("Ticker AR", tickers_ar, key="live_ticker")
        else:
            ticker_sel = st.text_input("Ticker AR", value="GGAL", key="live_ticker_txt").upper()
    with col2:
        st.markdown("")
        st.markdown("")
        st.caption("Datos traidos en tiempo real desde IOL. Calculo en ARS original.")

    if not ticker_sel:
        return

    # Quote en tiempo real
    with st.spinner("Cargando cotizacion..."):
        q = _fetch_quote(ticker_sel)

    if q:
        trade = q.get("trade", {}) or {}
        precio = trade.get("lot_price", {}).get("value") or trade.get("unit_price")
        var    = trade.get("variation")
        maxi   = trade.get("maximum", {}).get("value")
        mini   = trade.get("minimum", {}).get("value")
        apertura = trade.get("opening_lot_price", {}).get("value")
        vol_nom  = trade.get("total_nominal_volume")

        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        qc1.metric("Precio ARS", _fmt_ars(precio),
                   delta=f"{var:+.2f}%" if var is not None else None)
        qc2.metric("Apertura",   _fmt_ars(apertura))
        qc3.metric("Maximo",     _fmt_ars(maxi))
        qc4.metric("Minimo",     _fmt_ars(mini))
        qc5.metric("Volumen",    f"{int(vol_nom):,}" if vol_nom else "-")
    else:
        st.caption("Quote no disponible fuera de horario BCBA (10:30-17:00 ART)")

    st.divider()

    # Historico + indicadores
    with st.spinner(f"Calculando indicadores de {ticker_sel}..."):
        df = _fetch_historia(ticker_sel)

    if df is None or df.empty:
        st.error(f"No se pudieron traer datos historicos para {ticker_sel}.")
        return

    ultima = df.iloc[-1]
    n_barras = len(df)
    fecha_ini = df.iloc[0]["fecha"]
    fecha_fin = df.iloc[-1]["fecha"]

    st.caption(f"{n_barras} barras  |  {fecha_ini} al {fecha_fin}  |  precios en ARS")

    # ── Indicadores principales ───────────────────────────────────
    st.markdown("#### Indicadores Tecnicos — Ultima Sesion")

    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("RSI 14",
               f"{ultima['rsi14']:.1f}" if pd.notna(ultima["rsi14"]) else "-",
               help="<35 oversold | >65 overbought")
    ic2.metric("MACD Hist",
               f"{ultima['macd_hist']:+.4f}" if pd.notna(ultima["macd_hist"]) else "-",
               help=">0 momentum alcista")
    ic3.metric("ADX",
               f"{ultima['adx']:.1f}" if pd.notna(ultima["adx"]) else "-",
               help=">20 tendencia presente")
    ic4.metric("Vol Relativo",
               f"{ultima['vol_rel']:.2f}x" if pd.notna(ultima["vol_rel"]) else "-",
               help="vs promedio 20 dias")

    ic5, ic6, ic7 = st.columns(3)
    ic5.metric("vs SMA 20",  _fmt_pct(ultima["dist_sma20"]))
    ic6.metric("vs SMA 50",  _fmt_pct(ultima["dist_sma50"]))
    ic7.metric("vs SMA 200", _fmt_pct(ultima["dist_sma200"]))

    # ── Bollinger ─────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Posicion Bollinger Bands (20,2)")
    precio_actual = float(ultima["close"])
    bb_up = float(ultima["bb_up"]) if pd.notna(ultima["bb_up"]) else None
    bb_lo = float(ultima["bb_lo"]) if pd.notna(ultima["bb_lo"]) else None
    bb_mid = float(ultima["bb_mid"]) if pd.notna(ultima["bb_mid"]) else None

    if bb_up and bb_lo:
        rango_bb = bb_up - bb_lo
        pos_bb   = (precio_actual - bb_lo) / rango_bb * 100 if rango_bb > 0 else 50
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Banda Superior", _fmt_ars(bb_up))
        bc2.metric("Banda Media",    _fmt_ars(bb_mid))
        bc3.metric("Banda Inferior", _fmt_ars(bb_lo))
        bc4.metric("Posicion en BB", f"{pos_bb:.0f}%",
                   help="0%=banda inf | 100%=banda sup")

    # ── Estructura de mercado ─────────────────────────────────────
    st.divider()
    st.markdown("#### Estructura de Mercado (N=10)")
    est = _estructura(df, n=10)
    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    ec1.metric("Estructura",  _est_label(est["estructura"]))
    ec2.metric("BOS Alcista", "Si" if est["bos_bull"] else "No")
    ec3.metric("BOS Bajista", "Si" if est["bos_bear"] else "No")
    ec4.metric("CHoCH Bull",  "Si" if est["choch_bull"] else "No")
    ec5.metric("CHoCH Bear",  "Si" if est["choch_bear"] else "No")

    # ── Velas — ultimos 5 dias ─────────────────────────────────────
    st.divider()
    st.markdown("#### Velas — Ultimas 5 Sesiones")
    df5 = df.tail(5).copy()
    velas = []
    for _, r in df5.iterrows():
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        rng  = h - l if h != l else 1
        cuerpo = abs(c - o)
        ssup   = h - max(o, c)
        sinf   = min(o, c) - l
        alc    = c >= o
        velas.append({
            "Fecha":     str(r["fecha"]),
            "Dir":       "Alc" if alc else "Baj",
            "Cierre ARS": _fmt_ars(c),
            "Cuerpo %":  f"{cuerpo/rng*100:.1f}%",
            "S.Sup %":   f"{ssup/rng*100:.1f}%",
            "S.Inf %":   f"{sinf/rng*100:.1f}%",
            "Vol Rel":   f"{r['vol_rel']:.2f}x" if pd.notna(r["vol_rel"]) else "-",
            "RSI":       f"{r['rsi14']:.1f}" if pd.notna(r["rsi14"]) else "-",
        })
    st.dataframe(pd.DataFrame(velas), hide_index=True, use_container_width=True)

    # ── Serie de precios (chart) ──────────────────────────────────
    st.divider()
    st.markdown("#### Cierre historico (ARS)")
    df_chart = df[["fecha", "close", "sma20", "sma50", "sma200"]].copy()
    df_chart = df_chart.set_index("fecha")
    df_chart.columns = ["Cierre", "SMA 20", "SMA 50", "SMA 200"]
    st.line_chart(df_chart, height=320)


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 3: OPCIONES AR
# ─────────────────────────────────────────────────────────────────────────────

def _subtab_opciones(query_fn):
    st.subheader("Opciones Argentina — Chain Live + Historial")
    st.caption("Greeks calculados por IOL via Black-Scholes. Solo disponible en horario BCBA (10:30-17:00 ART).")

    if not _client_ok():
        msg = _client_error_msg()
        st.error(msg or "No se pudo inicializar el cliente IOL.")
        return

    # Tickers con opciones
    df_ar = query_fn(
        "SELECT ticker FROM activos_ar WHERE activo=true AND tiene_opciones=true ORDER BY ticker"
    )
    tickers_opc = df_ar["ticker"].tolist() if not df_ar.empty else [
        "GGAL", "YPFD", "BMA", "PAM", "TGS", "EDN", "CEPU", "TXAR", "ALU"
    ]

    ticker_sel = st.selectbox("Ticker subyacente", tickers_opc, key="opc_ticker")

    col_sub1, col_sub2 = st.tabs(["Chain Live", "Historial IV / PCR"])

    with col_sub1:
        # Boton para forzar refresco (limpia cache si habia error previo)
        if st.button("Recargar chain", key="btn_reload_chain"):
            _fetch_opciones.clear()
            st.rerun()

        with st.spinner(f"Cargando chain de {ticker_sel}..."):
            raw, df_chain, err = _fetch_opciones(ticker_sel)

        if raw is None:
            if err:
                st.error(f"Error al obtener el chain: {err}")
            else:
                st.warning("No se pudo obtener el chain.")
            st.caption("Posibles causas: (1) mercado BCBA cerrado (10:30-17:00 ART), "
                       "(2) credenciales IOL incorrectas, (3) ticker sin opciones activas.")
            return

        spot       = raw.get("precioSubyacente") or raw.get("spot_price", 0)
        risk_free  = raw.get("tasaLibreDeRiesgo") or raw.get("risk_free_rate", 0)
        n_total    = len(raw.get("opciones") or raw.get("options") or [])

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Spot ARS",    _fmt_ars(spot))
        mc2.metric("Tasa libre",  f"{risk_free*100:.1f}% anual" if risk_free else "-")
        mc3.metric("Strikes total", str(n_total))
        mc4.metric("Strikes liquidos", str(len(df_chain)) if df_chain is not None else "0")

        if df_chain is None or df_chain.empty:
            st.info("Sin Greeks disponibles ahora. Los Greeks se calculan durante horario de mercado.")
            return

        # PCR por volumen
        vol_calls = df_chain[df_chain.option_type == "C"]["volume"].fillna(0).sum()
        vol_puts  = df_chain[df_chain.option_type == "V"]["volume"].fillna(0).sum()
        pcr = vol_puts / vol_calls if vol_calls > 0 else None

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("PCR Volumen",
                   f"{pcr:.3f}" if pcr else "-",
                   help="Put/Call ratio. >1 = mas puts (sesgo bajista)")
        pc2.metric("Vol Calls", f"{int(vol_calls):,}")
        pc3.metric("Vol Puts",  f"{int(vol_puts):,}")

        st.divider()

        # Filtro por vencimiento
        vencimientos = sorted(df_chain["expiration"].unique())
        venc_sel = st.selectbox("Vencimiento", vencimientos, key="opc_venc")
        df_v = df_chain[df_chain.expiration == venc_sel].copy()

        # Separar calls y puts
        calls = df_v[df_v.option_type == "C"].sort_values("strike_price")
        puts  = df_v[df_v.option_type == "V"].sort_values("strike_price")

        # IV ATM (strikes mas cercanos al spot)
        atm_c = calls.iloc[(calls["strike_price"] - spot).abs().argsort()].head(3)
        iv_atm = atm_c["implied_volatility"].mean()

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Expiracion",  venc_sel)
        sc2.metric("IV ATM (Calls)", f"{iv_atm*100:.1f}%" if pd.notna(iv_atm) else "-")
        dias_exp = (pd.to_datetime(venc_sel) - pd.Timestamp.today()).days
        sc3.metric("Dias al venc.", str(dias_exp))

        st.markdown("**Calls**")
        _mostrar_chain(calls, spot)

        st.markdown("**Puts**")
        _mostrar_chain(puts, spot)

    with col_sub2:
        _historial_iv_pcr(query_fn, ticker_sel)


def _mostrar_chain(df: pd.DataFrame, spot: float):
    if df.empty:
        st.caption("Sin datos.")
        return
    rows = []
    for _, r in df.iterrows():
        dist = (r["strike_price"] - spot) / spot * 100
        rows.append({
            "Symbol":      r["symbol"],
            "Strike":      _fmt_ars(r["strike_price"]),
            "Dist Spot %": _fmt_pct(dist),
            "Bid":         _fmt_ars(r["bid_price"]),
            "Ask":         _fmt_ars(r["ask_price"]),
            "Teor.":       _fmt_ars(r["theoretical_price"]),
            "IV %":        f"{r['implied_volatility']*100:.1f}%" if pd.notna(r["implied_volatility"]) else "-",
            "Delta":       f"{r['delta']:.3f}" if pd.notna(r["delta"]) else "-",
            "Gamma":       f"{r['gamma']:.5f}" if pd.notna(r["gamma"]) else "-",
            "Theta":       f"{r['theta']:.2f}" if pd.notna(r["theta"]) else "-",
            "Vega":        f"{r['vega']:.2f}" if pd.notna(r["vega"]) else "-",
            "Vol":         f"{int(r['volume']):,}" if pd.notna(r["volume"]) and r["volume"] else "-",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={
                     "Symbol":    st.column_config.TextColumn(width=110),
                     "Strike":    st.column_config.TextColumn(width=100),
                     "Dist Spot %": st.column_config.TextColumn(width=90),
                     "IV %":     st.column_config.TextColumn(width=70),
                     "Delta":    st.column_config.TextColumn(width=70),
                     "Gamma":    st.column_config.TextColumn(width=80),
                     "Theta":    st.column_config.TextColumn(width=70),
                     "Vega":     st.column_config.TextColumn(width=65),
                 })


def _historial_iv_pcr(query_fn, ticker: str):
    st.markdown("#### Historial IV ATM y PCR")
    df_iv = query_fn("""
        SELECT fecha,
               ROUND(AVG(CASE WHEN option_type='C' THEN implied_volatility END)::numeric, 4) AS iv_calls,
               ROUND(AVG(CASE WHEN option_type='V' THEN implied_volatility END)::numeric, 4) AS iv_puts,
               ROUND(
                   SUM(CASE WHEN option_type='V' THEN COALESCE(volume,0) ELSE 0 END)::numeric
                   / NULLIF(SUM(CASE WHEN option_type='C' THEN COALESCE(volume,0) ELSE 0 END), 0)
               , 3) AS pcr
        FROM opciones_ar_gregas
        WHERE ticker_subyacente = :t
        GROUP BY fecha
        ORDER BY fecha DESC
        LIMIT 60
    """, {"t": ticker})

    if df_iv.empty:
        st.info("Sin historial de opciones en DB. El snapshot diario construye esta serie con el tiempo.")
        st.caption("Configura el cron de GH Actions para correr 38_opciones_ar_snapshot.py durante horario BCBA.")
        return

    df_iv = df_iv.sort_values("fecha").set_index("fecha")
    st.line_chart(df_iv[["iv_calls", "iv_puts"]].rename(columns={
        "iv_calls": "IV Calls", "iv_puts": "IV Puts"
    }), height=250)

    st.line_chart(df_iv[["pcr"]].rename(columns={"pcr": "PCR"}), height=200)

    st.dataframe(
        df_iv.reset_index().sort_values("fecha", ascending=False),
        hide_index=True, use_container_width=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 4: ESTRATEGIAS
# ─────────────────────────────────────────────────────────────────────────────

def _subtab_estrategias(query_fn):
    st.subheader("Estrategias AR — Backtest y Forward Testing")

    st_tabs = st.tabs(["Resultados BT", "Ejecutar BT", "Forward Testing AR"])

    with st_tabs[0]:
        _mostrar_resultados_bt(query_fn)

    with st_tabs[1]:
        _ejecutar_bt(query_fn)

    with st_tabs[2]:
        _forward_testing_ar(query_fn)


def _mostrar_resultados_bt(query_fn):
    st.markdown("#### Resultados de Backtest AR guardados")

    df_r = query_fn("""
        SELECT estrategia_entrada, estrategia_salida, ticker,
               segmento, total_operaciones, ganancias, perdidas,
               win_rate, retorno_promedio_pct, retorno_total_pct,
               max_drawdown_pct, profit_factor, dias_promedio_posicion
        FROM resultados_bt_ar
        ORDER BY win_rate DESC, profit_factor DESC
    """)

    if df_r.empty:
        st.info("Sin resultados de backtest AR todavia. Usa 'Ejecutar BT' para correr el primer analisis.")
        return

    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        estrategias = ["Todas"] + sorted(df_r["estrategia_entrada"].unique().tolist())
        est_sel = st.selectbox("Estrategia entrada", estrategias, key="bt_est")
    with col2:
        tickers_bt = ["Todos"] + sorted(df_r["ticker"].unique().tolist())
        ticker_bt  = st.selectbox("Ticker", tickers_bt, key="bt_ticker")

    df_f = df_r.copy()
    if est_sel != "Todas":
        df_f = df_f[df_f.estrategia_entrada == est_sel]
    if ticker_bt != "Todos":
        df_f = df_f[df_f.ticker == ticker_bt]

    df_show = df_f.copy()
    df_show["Win Rate"] = df_show["win_rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    df_show["Ret Prom"] = df_show["retorno_promedio_pct"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
    df_show["Ret Total"] = df_show["retorno_total_pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
    df_show["MaxDD"]    = df_show["max_drawdown_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    df_show["PF"]       = df_show["profit_factor"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    df_show["Dias"]     = df_show["dias_promedio_posicion"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    st.dataframe(
        df_show[["estrategia_entrada", "estrategia_salida", "ticker", "segmento",
                 "total_operaciones", "ganancias", "perdidas",
                 "Win Rate", "Ret Prom", "Ret Total", "MaxDD", "PF", "Dias"]].rename(columns={
            "estrategia_entrada": "Entrada", "estrategia_salida": "Salida",
            "ticker": "Ticker", "segmento": "Segmento",
            "total_operaciones": "Ops", "ganancias": "Win", "perdidas": "Loss",
        }),
        hide_index=True, use_container_width=True,
    )


def _ejecutar_bt(query_fn):
    st.markdown("#### Ejecutar Backtest sobre historial IOL")
    st.caption("Fetch de 3 anos de historia desde IOL -> calculo de indicadores -> simulacion de trades -> guardado en DB")

    if not _client_ok():
        msg = _client_error_msg()
        st.error(msg or "No se pudo inicializar el cliente IOL.")
        return

    df_ar = query_fn("SELECT ticker, sector FROM activos_ar WHERE activo=true ORDER BY ticker")
    tickers_ar = df_ar["ticker"].tolist() if not df_ar.empty else []

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_bt = st.selectbox("Ticker a backtestear", ["Todos"] + tickers_ar, key="run_bt_ticker")
    with col2:
        est_entrada = st.selectbox("Estrategia entrada",
                                   ["EAR1 (Score > 60)", "EAR2 (Score > 75)", "EAR3 (BOS Bull)"],
                                   key="run_bt_est")
    with col3:
        est_salida = st.selectbox("Estrategia salida",
                                  ["SAR1 (Stop 5% / Take 10%)", "SAR2 (Stop 8% / Take 15%)",
                                   "SAR3 (CHoCH Bear)", "SAR4 (20 dias max)"],
                                  key="run_bt_sal")

    st.info(
        "El backtest corre sobre los datos historicos que devuelve IOL (~800 barras). "
        "Los resultados se guardan en `resultados_bt_ar` y `operaciones_bt_ar`."
    )

    if st.button("Ejecutar Backtest", type="primary", key="run_bt_btn"):
        st.warning("Backtest AR en desarrollo. Proximamente disponible en esta UI.")
        st.caption("Mientras tanto: `python scripts/37_scanner_ar.py` genera el scanner diario, "
                   "y el modulo de BT AR se construye sobre ese mismo motor.")


def _forward_testing_ar(query_fn):
    st.markdown("#### Forward Testing Argentina")
    st.caption("Tracking de senales en tiempo real. Historia construida por el cron diario.")

    df_hist = query_fn("""
        SELECT DATE(scan_fecha) AS fecha, ticker, alert_nivel, alert_score,
               precio_cierre, rsi14, estructura_10
        FROM alertas_scanner_ar
        WHERE scan_fecha >= NOW() - INTERVAL '30 days'
        ORDER BY scan_fecha DESC, alert_score DESC
        LIMIT 200
    """)

    if df_hist.empty:
        st.info("Sin historial de senales AR. El cron diario construye esta serie con el tiempo.")
        return

    # Evolucion de scores
    df_pivot = df_hist.pivot_table(
        index="fecha", columns="ticker", values="alert_score", aggfunc="max"
    )
    st.markdown("**Evolucion de Alert Score — 30 dias**")
    st.line_chart(df_pivot, height=280)

    # Tabla resumen
    df_resumen = df_hist.groupby("ticker").agg(
        dias_en_compra=("alert_nivel", lambda x: (x.isin(["COMPRA_FUERTE", "COMPRA"])).sum()),
        score_promedio=("alert_score", "mean"),
        ultima_senal=("alert_nivel", "last"),
        ultimo_precio=("precio_cierre", "last"),
    ).reset_index().sort_values("score_promedio", ascending=False)

    df_resumen["Score Prom"] = df_resumen["score_promedio"].map(lambda x: f"{x:.1f}")
    df_resumen["Dias Compra"] = df_resumen["dias_en_compra"].astype(str)
    df_resumen["Ultimo Precio"] = df_resumen["ultimo_precio"].map(_fmt_ars)
    df_resumen["Ultima Senal"] = df_resumen["ultima_senal"].map(
        lambda x: f"{_EMO.get(x,'')} {x}"
    )

    st.markdown("**Resumen por ticker — ultimos 30 dias**")
    st.dataframe(
        df_resumen[["ticker", "Ultima Senal", "Score Prom", "Dias Compra", "Ultimo Precio"]].rename(
            columns={"ticker": "Ticker"}
        ),
        hide_index=True, use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 5: HISTORIAL
# ─────────────────────────────────────────────────────────────────────────────

def _subtab_historial(query_fn):
    st.subheader("Historial de Alertas AR")

    col1, col2, col3 = st.columns(3)
    with col1:
        desde = st.date_input("Desde", value=date.today() - timedelta(days=60), key="hist_desde")
    with col2:
        hasta = st.date_input("Hasta", value=date.today(), key="hist_hasta")
    with col3:
        df_t = query_fn("SELECT DISTINCT ticker FROM alertas_scanner_ar ORDER BY ticker")
        tickers_h = ["Todos"] + (df_t["ticker"].tolist() if not df_t.empty else [])
        ticker_h  = st.selectbox("Ticker", tickers_h, key="hist_ticker")

    col4, col5 = st.columns(2)
    with col4:
        nivel_h = st.selectbox("Nivel", ["Todos", "COMPRA_FUERTE", "COMPRA", "NEUTRAL", "VENTA"],
                               key="hist_nivel")
    with col5:
        score_min = st.slider("Score minimo", 0, 100, 0, 5, key="hist_score")

    filtros = ["DATE(scan_fecha) BETWEEN :desde AND :hasta"]
    params  = {"desde": str(desde), "hasta": str(hasta)}
    if ticker_h != "Todos":
        filtros.append("ticker = :ticker")
        params["ticker"] = ticker_h
    if nivel_h != "Todos":
        filtros.append("alert_nivel = :nivel")
        params["nivel"] = nivel_h
    if score_min > 0:
        filtros.append("alert_score >= :score_min")
        params["score_min"] = score_min

    df = query_fn(f"""
        SELECT DATE(scan_fecha) AS fecha, ticker, sector,
               alert_nivel, alert_score, precio_cierre,
               rsi14, dist_sma50, dist_sma200, estructura_10, alert_detalle
        FROM alertas_scanner_ar
        WHERE {' AND '.join(filtros)}
        ORDER BY scan_fecha DESC, alert_score DESC
        LIMIT 500
    """, params)

    if df.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        return

    st.metric("Registros", len(df))

    df_show = df.copy()
    df_show["Nivel"]    = df_show["alert_nivel"].map(lambda x: f"{_EMO.get(x,'')} {x}")
    df_show["Score"]    = df_show["alert_score"].map(lambda x: f"{x:.0f}")
    df_show["Precio"]   = df_show["precio_cierre"].map(_fmt_ars)
    df_show["RSI"]      = df_show["rsi14"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    df_show["vs SMA50"] = df_show["dist_sma50"].map(_fmt_pct)
    df_show["Est.10"]   = df_show["estructura_10"].map(_est_label)

    st.dataframe(
        df_show[["fecha", "ticker", "sector", "Nivel", "Score", "Precio",
                 "RSI", "vs SMA50", "Est.10", "alert_detalle"]].rename(columns={
            "fecha": "Fecha", "ticker": "Ticker", "sector": "Sector",
            "alert_detalle": "Detalle",
        }),
        hide_index=True, use_container_width=True,
        column_config={
            "Detalle": st.column_config.TextColumn(width=300),
        }
    )

    # Grafico evolucion si se filtra por ticker
    if ticker_h != "Todos" and len(df) > 1:
        st.divider()
        st.markdown(f"**Evolucion de {ticker_h}**")
        df_plot = df[["fecha", "alert_score", "precio_cierre"]].copy()
        df_plot = df_plot.sort_values("fecha").set_index("fecha")
        df_plot.columns = ["Alert Score", "Precio ARS"]

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.line_chart(df_plot[["Alert Score"]], height=200)
        with col_g2:
            st.line_chart(df_plot[["Precio ARS"]], height=200)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_iol_tab(query_fn):
    """Punto de entrada desde streamlit_app.py."""
    st.markdown(
        "Mercado **Argentina (BCBA)** | Datos via IOL REST API | Precios en **ARS**"
    )

    if not _client_ok():
        st.warning(
            "IOL_USERNAME / IOL_PASSWORD no configuradas. "
            "Agregalas en `.env` (local) o `st.secrets` (Streamlit Cloud)."
        )

    tabs = st.tabs([
        "📡 Scanner AR",
        "🔍 Analisis Live",
        "📊 Opciones AR",
        "🧪 Estrategias",
        "📋 Historial",
    ])

    with tabs[0]:
        try:
            _subtab_scanner(query_fn)
        except Exception as e:
            st.error(f"Error en Scanner AR: {e}")
            st.code(traceback.format_exc())

    with tabs[1]:
        try:
            _subtab_analisis(query_fn)
        except Exception as e:
            st.error(f"Error en Analisis Live: {e}")
            st.code(traceback.format_exc())

    with tabs[2]:
        try:
            _subtab_opciones(query_fn)
        except Exception as e:
            st.error(f"Error en Opciones AR: {e}")
            st.code(traceback.format_exc())

    with tabs[3]:
        try:
            _subtab_estrategias(query_fn)
        except Exception as e:
            st.error(f"Error en Estrategias: {e}")
            st.code(traceback.format_exc())

    with tabs[4]:
        try:
            _subtab_historial(query_fn)
        except Exception as e:
            st.error(f"Error en Historial: {e}")
            st.code(traceback.format_exc())
