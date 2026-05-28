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

from dashboard.sintesis_data import cargar_datos_ticker, listar_tickers
from dashboard.view import construir_vista
from dashboard.export_jpg import generar_jpg
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


def _sidebar_export(ticker, datos, sintesis):
    st.sidebar.divider()
    if st.sidebar.button("Generar JPG", use_container_width=True):
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


def main():
    st.set_page_config(page_title="Informe descriptivo por ticker", layout="wide")
    st.title("Informe descriptivo por ticker")
    st.caption("Descriptivo, no predictivo. Complemento de TradingView. "
               "Cruza tecnico + opciones + sector con reglas trazables.")

    tickers = listar_tickers()
    if not tickers:
        st.error("No hay tickers en la DB local (precios_diarios vacia).")
        return

    default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0
    ticker = st.sidebar.selectbox("Ticker", tickers, index=default_idx)

    datos = cargar_datos_ticker(ticker)
    if not (datos.get("perfil") or datos.get("precio", {}).get("close") is not None):
        st.warning(f"Sin datos suficientes para {ticker}.")
        return

    sintesis = sintetizar(datos)
    vista = construir_vista(datos, sintesis)
    _render(vista)
    _sidebar_export(ticker, datos, sintesis)


if __name__ == "__main__":
    main()
