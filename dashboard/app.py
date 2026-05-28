"""
dashboard/app.py
Dashboard — Informe descriptivo por ticker (Streamlit, LOCAL).

Cruza tecnico (D/W) + opciones por plazo + sentimiento sectorial y muestra un
VEREDICTO de 3 valores + cuadros de detalle, para que el usuario entienda la
descripcion. Herramienta DESCRIPTIVA (no predictiva). Spec: dashboard/README.md.

El contenido se arma en dashboard/view.py (fuente unica, compartida con la
exportacion a JPG). Correr con engine LOCAL via dashboard/run_dashboard.bat.
"""

import os
# Forzar engine local ANTES de cualquier query (Plan C: local es la verdad).
os.environ.pop("DATABASE_URL", None)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from dashboard.sintesis_data import cargar_datos_ticker, listar_tickers, cargar_radar
from dashboard.view import construir_vista
from dashboard.metricas import construir_papel
from dashboard.radar import construir_radar
from dashboard.export_jpg import generar_jpg, generar_papel_pdf
from src.utils.dashboard_sintesis import sintetizar


def _tabla(filas, columnas):
    st.dataframe(pd.DataFrame(filas, columns=columnas),
                 hide_index=True, use_container_width=True)


def _encabezado(enc: dict):
    linea = f"**{enc['ticker']}** | {enc['sector']}"
    if enc["industry"]:
        linea += f" / {enc['industry']}"
    linea += f" | ${enc['close']} | cierre {enc['fecha']}"
    st.markdown(linea)
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:8px;background:{enc['color']};"
        f"color:white;font-size:1.1rem;'><b>VEREDICTO: {enc['estado']}</b> - {enc['frase']}</div>",
        unsafe_allow_html=True,
    )


def _render(vista: dict):
    _encabezado(vista["encabezado"])
    st.divider()

    tec = vista["tecnico"]
    st.subheader("Tecnico")
    st.caption(f"Diario al {tec['fecha_d']}  |  Semanal (W-FRI cerrada) al {tec['fecha_w']}")
    _tabla(tec["filas"], ["Indicador", "Diario", "Semanal"])
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Opciones - Sesgo (ticker)")
        if vista["opciones"]:
            _tabla(vista["opciones"], ["Plazo", "PCR_vol", "PCR_oi", "Sesgo"])
        else:
            st.caption("Sin datos de opciones por plazo.")
    with c2:
        st.subheader("Niveles - Muros de OI (ticker)")
        if vista["muros"]:
            _tabla(vista["muros"], ["Plazo", "Soporte", "Resistencia"])
        else:
            st.caption("Sin datos de opciones por plazo.")
    st.divider()

    sec = vista["sector"]
    st.subheader(f"Sentimiento - Sector ({sec['sector']})")
    if sec["filas"]:
        _tabla(sec["filas"], ["Plazo", "PCR_vol_sec", "Sesgo", "Inusual?"])
        st.caption("z = desvios del PCR_vol del sector vs su media historica. "
                   "z>+1 cobertura inusual (bajista atipico); z<-1 optimismo inusual; |z|<1 normal.")
    else:
        st.caption("Sin datos sectoriales.")
    st.divider()

    conc = vista["conclusion"]
    st.subheader("Conclusion")
    st.markdown("**Rapida**")
    for b in conc["rapida"]:
        st.markdown(f"- {b}")
    st.markdown("**Detallada**")
    st.write(conc["detallada"])


def _render_papel(papel: list):
    st.caption("Trazabilidad de cada metrica del informe: valor, dato crudo, como se "
               "calcula, ventana, fuente (tabla.columna) y umbral de interpretacion.")
    cols = {"metrica": "Metrica", "valor": "Valor", "crudo": "Crudo",
            "formula": "Como se calcula", "ventana": "Ventana",
            "fuente": "Fuente", "umbral": "Umbral"}
    for sec in papel:
        st.subheader(sec["seccion"])
        df = pd.DataFrame(sec["filas"])[list(cols.keys())].rename(columns=cols)
        st.dataframe(df, hide_index=True, use_container_width=True)


def _sidebar_export(ticker, datos, sintesis):
    st.sidebar.divider()
    st.sidebar.markdown("**Exportar**")

    if st.sidebar.button("Informe (JPG)", use_container_width=True):
        with st.spinner("Generando imagen..."):
            try:
                path = generar_jpg(ticker, datos, sintesis)
                with open(path, "rb") as f:
                    st.session_state["jpg_bytes"] = f.read()
                st.session_state["jpg_name"] = path.name
                st.session_state["jpg_ticker"] = ticker
            except Exception as exc:
                st.sidebar.error(f"Error generando JPG: {exc}")

    if st.session_state.get("jpg_ticker") == ticker and st.session_state.get("jpg_bytes"):
        st.sidebar.download_button(
            f"Descargar {st.session_state['jpg_name']}",
            data=st.session_state["jpg_bytes"],
            file_name=st.session_state["jpg_name"],
            mime="image/jpeg",
            use_container_width=True,
        )

    if st.sidebar.button("Papel de trabajo (PDF)", use_container_width=True):
        with st.spinner("Generando PDF..."):
            try:
                path = generar_papel_pdf(ticker, datos, sintesis)
                with open(path, "rb") as f:
                    st.session_state["pdf_bytes"] = f.read()
                st.session_state["pdf_name"] = path.name
                st.session_state["pdf_ticker"] = ticker
            except Exception as exc:
                st.sidebar.error(f"Error generando PDF: {exc}")

    if st.session_state.get("pdf_ticker") == ticker and st.session_state.get("pdf_bytes"):
        st.sidebar.download_button(
            f"Descargar {st.session_state['pdf_name']}",
            data=st.session_state["pdf_bytes"],
            file_name=st.session_state["pdf_name"],
            mime="application/pdf",
            use_container_width=True,
        )


def _vista_informe(tickers):
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "AAPL" if "AAPL" in tickers else tickers[0]
    ticker = st.sidebar.selectbox("Ticker", tickers, key="ticker")

    datos = cargar_datos_ticker(ticker)
    if not (datos.get("perfil") or datos.get("precio", {}).get("close") is not None):
        st.warning(f"Sin datos suficientes para {ticker}.")
        return

    sintesis = sintetizar(datos)
    vista = construir_vista(datos, sintesis)
    papel = construir_papel(datos, sintesis)

    tab_informe, tab_papel = st.tabs(["Informe", "Papel de trabajo"])
    with tab_informe:
        _render(vista)
    with tab_papel:
        _render_papel(papel)

    _sidebar_export(ticker, datos, sintesis)


def _vista_radar():
    st.subheader("Radar del dia - actividad inusual en opciones")
    data = cargar_radar()
    st.caption(f"Anomalias del {data.get('fecha')}. z = desvios del valor de hoy vs la "
               "historia del propio ticker; filtra ruido por percentil de volumen (>=50).")
    z = st.slider("Umbral z (inusual)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    filas = construir_radar(data, z=z)
    if not filas:
        st.info("Sin anomalias con el umbral actual.")
        return

    cols = {"ticker": "Ticker", "sector": "Sector", "tipo": "Tipo",
            "magnitud": "Magnitud (z)", "vol_z": "vol z", "iv_z": "IV z",
            "pcr_z": "PCR z", "sector_acompana": "Sector acompana?"}
    df = pd.DataFrame(filas)[list(cols.keys())].rename(columns=cols)
    st.caption(f"{len(df)} tickers con actividad inusual (orden por magnitud). "
               "Tags: Volumen inusual / IV en alza / Sesgo a calls / Cobertura (puts).")

    event = st.dataframe(df, hide_index=True, use_container_width=True,
                         on_select="rerun", selection_mode="single-row")
    rows = event.selection.rows if event and getattr(event, "selection", None) else []
    if rows:
        tk = df.iloc[rows[0]]["Ticker"]
        if st.button(f"Abrir informe de {tk}", use_container_width=False):
            st.session_state["_ir_informe"] = True
            st.session_state["_nav_ticker"] = tk
            st.rerun()
    else:
        st.caption("Tip: seleccionar una fila para abrir su informe descriptivo.")


def main():
    st.set_page_config(page_title="Informe descriptivo por ticker", layout="wide")

    # Navegacion pendiente desde el radar (antes de instanciar los widgets).
    if st.session_state.pop("_ir_informe", False):
        st.session_state["modo"] = "Informe por ticker"
        if "_nav_ticker" in st.session_state:
            st.session_state["ticker"] = st.session_state.pop("_nav_ticker")

    st.title("Informe descriptivo por ticker")
    st.caption("Descriptivo, no predictivo. Complemento de TradingView. "
               "Cruza tecnico + opciones + sector con reglas trazables.")

    tickers = listar_tickers()
    if not tickers:
        st.error("No hay tickers en la DB local (precios_diarios vacia).")
        return

    modo = st.sidebar.radio("Vista", ["Informe por ticker", "Radar del dia"], key="modo")
    st.sidebar.divider()
    if modo == "Radar del dia":
        _vista_radar()
    else:
        _vista_informe(tickers)


if __name__ == "__main__":
    main()
