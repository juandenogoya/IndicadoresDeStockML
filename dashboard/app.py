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

from datetime import date

import pandas as pd
import streamlit as st

from dashboard.sintesis_data import (
    cargar_datos_ticker, listar_tickers, cargar_radar,
    cargar_financiero_ticker, cargar_screener_sector,
    listar_sectores_fundamentales, listar_regiones_fundamentales,
    cargar_veredictos_universo, fecha_datos, listar_sectores,
)
from dashboard.view import construir_vista
from dashboard.metricas import construir_papel
from dashboard.radar import construir_radar
from dashboard.financiero import (
    construir_bloques_ticker, texto_peer_basis, construir_screener,
)
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
        st.subheader("Opciones por plazo (ticker)")
        if vista["opciones"]:
            _tabla(vista["opciones"], ["Plazo", "PCR_vol (dia)", "PCR_oi (posic.)", "Sesgo (OI)"])
            st.caption("Sesgo = lectura del PCR_oi (posicionamiento acumulado; <1 = sesgo a "
                       "calls). PCR_vol = actividad del dia, direccionalmente ambigua. Si vol y "
                       "oi cruzan el 1.0 en lados opuestos, leer ambas (divergencia dia vs OI).")
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
        _tabla(sec["filas"], ["Plazo", "PCR_vol_sec", "Sesgo (OI)", "Inusual?"])
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


def _nombre_descarga(ticker: str, fecha, sufijo: str, ext: str) -> str:
    """Nombre con el que el navegador guarda la descarga: YYYYMMDD_TICKER[_suf].ext.

    El archivo en disco conserva el nombre que le pone su generador; esto es
    solo la etiqueta de la bajada. Hace falta calcularlo ANTES de generar
    porque download_button pide file_name al dibujarse, no al hacer clic.
    """
    try:
        f = fecha.strftime("%Y%m%d")
    except AttributeError:
        f = str(fecha or "").replace("-", "")[:8] or date.today().strftime("%Y%m%d")
    suf = f"_{sufijo}" if sufijo else ""
    return f"{f}_{ticker.upper()}{suf}.{ext}"


def _infografia_simple_bytes(ticker: str) -> bytes:
    # Import diferido: el modulo trae weasyprint. Solo al pedirlo.
    from scripts.reports.make_infografia_simple import generar_infografia_simple
    return generar_infografia_simple(ticker).read_bytes()


def _infografia_fundamental_bytes(ticker: str) -> bytes:
    # Import diferido: el modulo hace os.environ.pop(DATABASE_URL) al cargar
    # (inocuo en local) y trae weasyprint. Solo al pedirlo.
    from scripts.reports.make_infografia_fundamental import generar_infografia
    return generar_infografia(ticker).read_bytes()


@st.fragment
def _sidebar_export(ticker, datos, sintesis):
    """Exportaciones del informe.

    `download_button(data=<callable>)` genera el archivo AL HACER CLIC, en un
    thread aparte: UN clic en vez de dos (antes era generar -> rerun -> aparece
    el boton de descargar) y sin las 9 llaves de session_state que oficiaban de
    buffer. El @st.fragment aisla el bloque: bajar un archivo ya no re-renderiza
    el informe entero.

    OJO con el callable: no recibe argumentos y los comandos de Streamlit que
    se ejecuten adentro se IGNORAN (limitacion de la API), asi que no puede
    haber un st.error ahi. Si el generador falla -- tipicamente weasyprint o
    pymupdf ausentes, es decir el dashboard corriendo FUERA del venv -- lo que
    falla es la descarga, no la pagina. Es un modo de falla permanente y de
    entorno, no transitorio: por eso se acepta a cambio de sacar el doble clic.

    Se llama dentro de un `with st.sidebar:` y usa los comandos SIN prefijo: un
    fragment no puede escribir via `st.sidebar.*` (StreamlitAPIException).
    """
    fecha_cierre = (datos.get("precio") or {}).get("fecha")
    st.divider()
    st.markdown("**Exportar**")

    st.download_button(
        "Informe (JPG)",
        data=lambda: generar_jpg(ticker, datos, sintesis).read_bytes(),
        file_name=_nombre_descarga(ticker, fecha_cierre, "", "jpg"),
        mime="image/jpeg", width="stretch", key="dl_jpg",
    )
    st.download_button(
        "Papel de trabajo (PDF)",
        data=lambda: generar_papel_pdf(ticker, datos, sintesis).read_bytes(),
        file_name=_nombre_descarga(ticker, fecha_cierre, "papel", "pdf"),
        mime="application/pdf", width="stretch", key="dl_pdf",
    )
    st.download_button(
        "Infografia simple (PNG)",
        data=lambda: _infografia_simple_bytes(ticker),
        file_name=_nombre_descarga(ticker, fecha_cierre, "simple", "png"),
        mime="image/png", width="stretch", key="dl_ig_simple",
    )


def _vista_informe(tickers):
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "AAPL" if "AAPL" in tickers else tickers[0]
    # bind="query-params": el ticker viaja en la URL (?ticker=NVDA), asi que el
    # informe se puede marcar como favorito, compartir y sobrevive a un F5. La
    # key `ticker` es UNICA en todo el dashboard (informe / financiero /
    # earnings) -> cambiar de vista ya no obliga a re-elegir el ticker.
    ticker = st.sidebar.selectbox("Ticker", tickers, key="ticker",
                                  bind="query-params")

    datos = _datos_ticker_cache(ticker, fecha_datos())
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

    with st.sidebar:
        _sidebar_export(ticker, datos, sintesis)


def _nav_a_informe(tk):
    """Navega al informe descriptivo con el ticker tk.

    El TICKER se escribe directo en session_state: su selectbox no se instancio
    en esta vista, y con la key unificada (`ticker`, bindeada a query-params) es
    la via soportada para moverlo -- un parametro bindeado no se puede tocar via
    st.query_params.

    El MODO no se puede escribir asi: su radio YA se creo en main() en esta misma
    corrida y Streamlit prohibe reescribir la key de un widget ya instanciado.
    Por eso sigue siendo un flag diferido que main() consume al inicio de la
    corrida siguiente.
    """
    st.session_state["ticker"] = tk
    st.session_state["_ir_informe"] = True
    st.rerun()


@st.cache_data(show_spinner=False)
def _veredictos_cache(fecha_key: str):
    """Veredictos del universo, cacheados por fecha de datos (1 calculo/dia).
    fecha_key entra solo como clave de cache; el calculo lee la DB local."""
    return cargar_veredictos_universo()


@st.cache_data(show_spinner=False)
def _radar_cache(fecha_key: str):
    """cargar_radar() cacheada por fecha de datos.

    El wrapper vive ACA y no en sintesis_data.py a proposito: ese modulo no
    importa Streamlit -- lo comparten la exportacion a JPG y scripts/reports --
    y queremos que siga asi. Es el mismo activo que haria barata una eventual
    migracion de frontend: no ensuciarlo con el framework de turno.
    """
    return cargar_radar()


@st.cache_data(show_spinner=False)
def _datos_ticker_cache(ticker: str, fecha_key: str):
    """cargar_datos_ticker() cacheada por (ticker, fecha de datos). Misma razon
    que _radar_cache para que el decorador viva aca."""
    return cargar_datos_ticker(ticker)


@st.fragment
def _radar_anomalias(data):
    """Bloque de anomalias del radar.

    Fragment: mover el slider z re-filtra SOLO esta tabla. Antes re-ejecutaba
    la vista entera -- incluida la consulta del radar a la DB, que no estaba
    cacheada -- y redibujaba el screener de abajo.
    """
    st.caption(f"Anomalias del {data.get('fecha')}. z = desvios del valor de hoy vs la "
               "historia del propio ticker; filtra ruido por percentil de volumen (>=50).")
    z = st.slider("Umbral z (inusual)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    filas = construir_radar(data, z=z)
    if not filas:
        st.info("Sin anomalias con el umbral actual.")
        return

    cols = {"ticker": "Ticker", "sector": "Sector", "tipo": "Tipo",
            "magnitud": "Magnitud (z)", "vol_z": "vol z", "iv_z": "IV z",
            "pcr_z": "PCR z", "accion_z": "Accion z", "sector_acompana": "Sector acompana?"}
    df = pd.DataFrame(filas)[list(cols.keys())].rename(columns=cols)
    st.caption(f"{len(df)} tickers con actividad inusual (orden por magnitud). "
               "Tags: Volumen inusual / IV en alza / Sesgo a calls / Cobertura (puts).")
    event = st.dataframe(df, hide_index=True, use_container_width=True,
                         on_select="rerun", selection_mode="single-row", key="radar_tbl")
    rows = event.selection.rows if event and getattr(event, "selection", None) else []
    if rows:
        tk = df.iloc[rows[0]]["Ticker"]
        if st.button(f"Abrir informe de {tk}", key="radar_abrir"):
            # st.rerun() dentro de _nav_a_informe usa scope="app" (default),
            # asi que sale del fragment y recarga la app entera. Es lo que
            # queremos: estamos cambiando de vista.
            _nav_a_informe(tk)
    else:
        st.caption("Tip: seleccionar una fila para abrir su informe descriptivo.")


@st.fragment
def _radar_screener():
    """Screener por veredicto. Fragment: cambiar los filtros o seleccionar una
    fila ya no redibuja el bloque de anomalias de arriba."""
    st.subheader("Screener por veredicto")
    st.caption("Veredicto sintetico (tecnico + opciones + estructura) de cada ticker. "
               "El primer Buscar del dia calcula el universo (~2 min); luego es instantaneo "
               "hasta que entren datos nuevos.")
    cs1, cs2 = st.columns(2)
    with cs1:
        sel = st.multiselect("Veredictos", ["ALCISTA", "NEUTRAL", "BAJISTA"],
                             default=["ALCISTA"], key="scr_sel")
    with cs2:
        sel_sec = st.multiselect("Sectores (vacio = todos)", listar_sectores(),
                                 default=[], key="scr_sec")
    if st.button("Buscar", key="scr_buscar"):
        st.session_state["scr_run"] = True

    if not st.session_state.get("scr_run"):
        return
    if not sel:
        st.info("Elegi al menos un veredicto.")
        return
    with st.spinner("Calculando veredictos del universo (~2 min la primera vez)..."):
        universo = _veredictos_cache(fecha_datos())
    res = [u for u in universo
           if u["veredicto"] in sel and (not sel_sec or u["sector"] in sel_sec)]
    if not res:
        filtro = ", ".join(sel) + (f" en {', '.join(sel_sec)}" if sel_sec else "")
        st.info(f"Ningun ticker con veredicto {filtro}.")
        return
    dfv = pd.DataFrame(res).rename(columns={
        "ticker": "Ticker", "sector": "Sector", "veredicto": "Veredicto", "frase": "Lectura"})
    dfv = dfv[["Ticker", "Sector", "Veredicto", "Lectura"]]
    _sec_txt = f" | sectores: {', '.join(sel_sec)}" if sel_sec else ""
    st.caption(f"{len(dfv)} tickers con veredicto {', '.join(sel)}{_sec_txt} "
               f"(de {len(universo)} evaluados).")
    ev = st.dataframe(dfv, hide_index=True, use_container_width=True,
                      on_select="rerun", selection_mode="single-row", key="scr_tbl")
    r2 = ev.selection.rows if ev and getattr(ev, "selection", None) else []
    if r2:
        tk2 = dfv.iloc[r2[0]]["Ticker"]
        if st.button(f"Abrir informe de {tk2}", key="scr_abrir"):
            _nav_a_informe(tk2)
    else:
        st.caption("Tip: seleccionar una fila para abrir su informe descriptivo.")


def _vista_radar():
    # -- 1. Radar de anomalias de opciones --------------------------------
    st.subheader("Radar del dia - actividad inusual en opciones")
    _radar_anomalias(_radar_cache(fecha_datos()))

    # -- 2. Screener por veredicto ----------------------------------------
    st.divider()
    _radar_screener()


def _tabla_financiera(filas, columnas_orden):
    """Tabla con coloreo good/bad en la columna vs_mediana."""
    def _estilo(row):
        color = row.get("_color")
        css = ""
        if color == "good":
            css = "color: #1a7f37; font-weight: 600;"
        elif color == "bad":
            css = "color: #cf222e; font-weight: 600;"
        # Styler.apply(axis=1) exige un estilo por CADA columna del DataFrame
        # (incluye la columna oculta _color) -> iterar sobre row.index, no sobre
        # columnas_orden (que tiene 5 y el df 6).
        return [css if c == "vs Sector" else "" for c in row.index]

    df = pd.DataFrame(filas)
    # Mapear nombres internos -> etiquetas de columna
    df_disp = pd.DataFrame({
        "Metrica":    df["metrica"],
        "Valor":      df["valor"],
        "vs Sector":  df["vs_mediana"],
        "Mediana":    df["mediana"],
        "Percentil":  df["percentil"],
        "_color":     df["color"],
    })
    styler = df_disp.style.apply(_estilo, axis=1)
    st.dataframe(styler, hide_index=True, use_container_width=True,
                 column_config={"_color": None})


@st.fragment
def _sidebar_export_infografia(tk):
    """Export de la infografia fundamental en el SIDEBAR (homogeneo con el
    informe tecnico, que usa _sidebar_export). Mismo patron: un solo clic,
    generacion diferida al callable. Ver la nota de _sidebar_export -- incluido
    el `with st.sidebar:` obligatorio en el llamador."""
    st.divider()
    st.markdown("**Exportar**")
    st.download_button(
        "Infografia (PNG)",
        data=lambda: _infografia_fundamental_bytes(tk),
        file_name=_nombre_descarga(tk, fecha_datos(), "fundamental", "png"),
        mime="image/png", width="stretch", key="fin_ig_btn",
    )


@st.fragment
def _ficha_empresa_boton(tk):
    """Boton (area principal) para generar la ficha de empresa (PNG) del ticker
    EN PANTALLA: presentacion contra si misma (ultimo Q + variacion interanual).
    Import diferido (weasyprint). Muestra la ficha inline + descarga."""
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(f"Generar ficha de {tk} (PNG)", key="ficha_emp_btn",
                     use_container_width=True):
            with st.spinner("Generando ficha..."):
                try:
                    from scripts.reports.make_ficha_empresa import generar_ficha_empresa
                    path = generar_ficha_empresa(tk)
                    with open(path, "rb") as f:
                        st.session_state["ficha_emp_bytes"] = f.read()
                    st.session_state["ficha_emp_name"] = path.name
                    st.session_state["ficha_emp_ticker"] = tk
                except Exception as exc:
                    st.error(f"Error generando ficha: {exc}")
    if (st.session_state.get("ficha_emp_ticker") == tk
            and st.session_state.get("ficha_emp_bytes")):
        with c2:
            st.download_button(
                f"Descargar {st.session_state['ficha_emp_name']}",
                data=st.session_state["ficha_emp_bytes"],
                file_name=st.session_state["ficha_emp_name"],
                mime="image/png", key="ficha_emp_dl", use_container_width=True)
        st.image(st.session_state["ficha_emp_bytes"], use_container_width=True)
    st.caption(f"Ficha de presentacion (PNG, fondo oscuro) SOLO de {tk}: el "
               "ultimo trimestre reportado + variacion interanual, la empresa "
               "contra si misma (sin pares). Descriptivo, no recomienda.")


def _vista_financiera(tickers):
    st.subheader("Analisis Financiero")
    st.caption("Foto fundamental del ultimo trimestre reportado: valuacion, "
               "calidad, crecimiento y solvencia, comparada con la mediana de "
               "pares de la misma region. Descriptivo, no recomienda.")

    modo = st.radio("Modo", ["Por ticker", "Screener sectorial"],
                    horizontal=True, key="fin_modo")

    if modo == "Por ticker":
        tk = st.selectbox("Ticker", tickers, key="ticker", bind="query-params")
        data = cargar_financiero_ticker(tk)
        if not data.get("ratios"):
            st.warning(f"Sin datos fundamentales para {tk}. "
                       "Correr scripts/manual/refresh_fundamentales.bat.")
            return

        with st.sidebar:
            _sidebar_export_infografia(tk)

        ratios = data["ratios"]
        cur = data.get("reporting_currency") or "?"
        fpe = data.get("fiscal_period_end")
        sector = ratios.get("sector") or "?"
        precio = data.get("precio") or {}
        cierre = precio.get("close")
        fecha_cierre = precio.get("fecha")
        linea = f"**{tk}** | {sector} | ult. Q: {fpe} | moneda: {cur}"
        if cierre is not None:
            linea += f" | cierre USD {cierre:,.2f} ({fecha_cierre})"
        st.markdown(linea)
        st.caption(texto_peer_basis(data))

        with st.expander(f"Ficha de empresa (PNG) -- solo {tk}", expanded=False):
            _ficha_empresa_boton(tk)

        bloques = construir_bloques_ticker(data)
        for b in bloques:
            st.markdown(f"**{b['bloque']}**")
            _tabla_financiera(b["filas"],
                              ["Metrica", "Valor", "vs Sector", "Mediana", "Percentil"])
        st.caption("vs Sector: distancia a la mediana de pares (verde=mejor, "
                   "rojo=peor, segun la metrica). Percentil: posicion dentro del "
                   "sector. Absolutos (BPA, valor libro) en moneda de reporte: no "
                   "comparables cross-ticker. Bancos: ROIC/margen oper./liquidez "
                   "pueden ser '-' (sin estructura aplicable).")

    else:  # Screener sectorial
        sectores = listar_sectores_fundamentales()
        if not sectores:
            st.error("No hay datos en fundamentales_ratios_q.")
            return
        regiones = ["Todas"] + listar_regiones_fundamentales()
        c1, c2 = st.columns(2)
        with c1:
            sector = st.selectbox("Sector", sectores, key="fin_sector")
        with c2:
            region = st.selectbox("Region", regiones, key="fin_region")

        if st.button("Generar", use_container_width=False):
            st.session_state["fin_run"] = True

        if st.session_state.get("fin_run"):
            rows = cargar_screener_sector(sector, region)
            if not rows:
                st.info("Sin tickers para ese sector/region.")
                return
            screener = construir_screener(rows)
            st.caption(f"{screener['n']} tickers en {sector}"
                       f"{'' if region == 'Todas' else ' / ' + region}. "
                       "PER/P-B mas bajo = mas barato; ROE/ROIC/margen/crecimiento "
                       "mas alto = mejor. Orden por PER asc.")
            df = pd.DataFrame(screener["filas"], columns=screener["columnas"])
            # Anexar fila mediana
            df_med = pd.DataFrame([screener["mediana"]], columns=screener["columnas"])
            df_full = pd.concat([df, df_med], ignore_index=True)
            st.dataframe(df_full, hide_index=True, use_container_width=True)
            st.caption("Ultima fila = mediana del sector (sobre valores "
                       "disponibles). '-' = metrica no disponible para ese ticker.")


def _chat_meta(res: dict) -> str:
    """Linea de metadata bajo cada respuesta: tokens entrada/salida + tools."""
    parts = []
    if res.get("tokens_in") or res.get("tokens_out"):
        parts.append(f"Entrada ~{res.get('tokens_in', 0)} | "
                     f"Salida ~{res.get('tokens_out', 0)} tok")
    tools = list(dict.fromkeys(res.get("tools_used") or []))
    if tools:
        parts.append("tools: " + ", ".join(tools))
    return "  |  ".join(parts)


def _render_registro_consumo():
    """Expander con el consumo de tokens acumulado por fecha (tabla llm_uso_tokens)."""
    from src.agent import uso_tokens
    with st.expander("Registro de consumo (tokens por dia)"):
        filas = uso_tokens.resumen_por_fecha(usuario=uso_tokens.USUARIO_DEFAULT, dias=30)
        if not filas:
            st.caption("Sin registros todavia.")
            return
        df = pd.DataFrame(filas).rename(columns={
            "fecha": "Fecha", "consultas": "Consultas",
            "entrada": "Entrada", "salida": "Salida", "total": "Total",
        })
        st.dataframe(df[["Fecha", "Consultas", "Entrada", "Salida", "Total"]],
                     hide_index=True, use_container_width=True)
        tot = df[["Entrada", "Salida", "Total"]].sum()
        st.caption(f"Ultimos 30 dias: Entrada {int(tot['Entrada'])} | "
                   f"Salida {int(tot['Salida'])} | Total {int(tot['Total'])} tokens.")


def _vista_chat():
    st.subheader("Consultas en lenguaje natural")
    st.caption("Pregunta sobre precios, indicadores, opciones, alertas ML o "
               "fundamentales del universo (199 tickers). Responde con datos de "
               "la DB local (solo lectura) via el LLM. Tip: consultas simples y "
               "concretas gastan menos tokens.")

    if "chat_msgs" not in st.session_state:
        st.session_state["chat_msgs"] = []      # [{role, content, meta}] para render
        st.session_state["chat_contents"] = []  # historial para el modelo (podado)

    if st.session_state["chat_msgs"]:
        if st.sidebar.button("Limpiar conversacion", use_container_width=True):
            st.session_state["chat_msgs"] = []
            st.session_state["chat_contents"] = []
            st.rerun()
    else:
        st.markdown(
            "**Ejemplos:**\n"
            "- Cual es el ultimo cierre y RSI de NVDA?\n"
            "- Cuantos tickers hay en el sector Technology?\n"
            "- Como viene el PCR de opciones de AAPL?\n"
            "- Dame el veredicto fundamental de JPM"
        )

    for m in st.session_state["chat_msgs"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])

    pregunta = st.chat_input("Escribi tu consulta...")
    if pregunta:
        from src.agent.orchestrator import answer_sync
        from src.agent import uso_tokens

        st.session_state["chat_msgs"].append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Consultando datos..."):
                res = answer_sync(pregunta, contents=st.session_state["chat_contents"])
            if res.get("error"):
                st.error(res["text"])
            else:
                st.markdown(res["text"])
            meta = _chat_meta(res)
            if meta:
                st.caption(meta)

        st.session_state["chat_contents"] = res["contents"]
        st.session_state["chat_msgs"].append(
            {"role": "assistant", "content": res["text"], "meta": meta}
        )

        # Registro de consumo (best-effort: si falla, el chat sigue funcionando).
        if not res.get("error"):
            uso_tokens.registrar_uso(
                usuario=uso_tokens.USUARIO_DEFAULT,
                modelo=res.get("model"),
                tokens_in=res.get("tokens_in", 0),
                tokens_out=res.get("tokens_out", 0),
                tokens_total=res.get("tokens", 0),
                n_rondas=res.get("n_rondas", 0),
                tools=", ".join(dict.fromkeys(res.get("tools_used") or [])) or None,
                pregunta=pregunta,
            )

    _render_registro_consumo()


def main():
    st.set_page_config(page_title="Informe descriptivo por ticker", layout="wide")

    # Navegacion pendiente desde el radar: se consume ANTES de instanciar el
    # radio de vistas (su key no se puede reescribir una vez creado). El ticker
    # ya no viaja por aca -- lo escribe _nav_a_informe directo en su key unica.
    if st.session_state.pop("_ir_informe", False):
        st.session_state["modo"] = "Informe por ticker"

    st.title("Informe descriptivo por ticker")
    st.caption("Descriptivo, no predictivo. Complemento de TradingView. "
               "Cruza tecnico + opciones + sector con reglas trazables.")

    tickers = listar_tickers()
    if not tickers:
        st.error("No hay tickers en la DB local (precios_diarios vacia).")
        return

    modo = st.sidebar.radio("Vista",
                            ["Informe por ticker", "Radar del dia",
                             "Analisis Financiero", "Reaccion a balances",
                             "Carteras", "Performance", "Consultas (IA)"],
                            key="modo")
    st.sidebar.divider()
    if modo == "Radar del dia":
        _vista_radar()
    elif modo == "Analisis Financiero":
        _vista_financiera(tickers)
    elif modo == "Reaccion a balances":
        from dashboard.earnings_reaccion import construir_reaccion
        construir_reaccion(tickers)
    elif modo == "Carteras":
        from dashboard.carteras import construir_carteras
        construir_carteras(tickers)
    elif modo == "Performance":
        from dashboard.performance import construir_performance
        construir_performance(tickers)
    elif modo == "Consultas (IA)":
        _vista_chat()
    else:
        _vista_informe(tickers)


if __name__ == "__main__":
    main()
