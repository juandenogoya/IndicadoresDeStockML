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
    fecha_datos, listar_sectores,
    estado_datos, cargar_veredictos_precomputados,
    cargar_reparto_veredictos, cargar_ft_resumen,
    cargar_posiciones_ft_abiertas, cargar_earnings_proximos,
)
from dashboard.view import construir_vista
from dashboard.metricas import construir_papel
from dashboard.radar import construir_radar
from dashboard.hoy import (
    construir_clima, construir_inusual, construir_ft, construir_agenda,
)
from dashboard.financiero import (
    construir_bloques_ticker, texto_peer_basis, construir_screener,
)
from dashboard.export_jpg import generar_jpg, generar_papel_pdf
from src.utils.dashboard_sintesis import sintetizar


def _tabla(filas, columnas):
    st.dataframe(pd.DataFrame(filas, columns=columnas),
                 hide_index=True, width="stretch")


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
        st.dataframe(df, hide_index=True, width="stretch")


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


@st.cache_data(show_spinner=False, ttl=300)
def _veredictos_cache():
    """Veredictos del universo, leidos de veredictos_universo_diario.

    ANTES esto llamaba a cargar_veredictos_universo(), que recorre los ~200
    tickers en vivo: ~2 minutos la primera vez de cada dia, y el cache vivia en
    memoria del proceso -> se perdia al reiniciar Streamlit y se volvia a pagar.
    Los dos minutos eran la razon practica por la que el screener casi no se
    usaba.

    Ahora el calculo corre una vez de noche (scripts/compute_veredictos_universo.py,
    encadenado a ft_run_diario.bat) y aca solo se LEE. El TTL corto alcanza:
    la tabla cambia una vez por rutina.
    """
    return cargar_veredictos_precomputados()


@st.cache_data(show_spinner=False, ttl=60)
def _estado_datos_cache():
    """Frescura cacheada 60s.

    Se dibuja en TODAS las vistas y en cada rerun, pero solo cambia cuando
    corre la rutina nocturna. No se keyea por fecha de datos como los otros
    caches porque esa key saldria de la misma consulta que queremos evitar.
    """
    return estado_datos()


def _enumerar(items: list) -> str:
    """'A' | 'A y B' | 'A, B y C'. Los mensajes de estado los lee una persona
    apurada: una lista separada por comas se lee como enumeracion truncada."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} y {items[-1]}"


def _banda_frescura():
    """Estado del dato, visible en todas las vistas.

    Antes el dashboard servia el informe con lo que hubiera en la DB sin decir
    de cuando era. Como la rutina nocturna es MANUAL, las tablas se
    desincronizan con normalidad: el 2/9/2026 precios_diarios estaba al 1/9 y
    opciones_pcr_plazo_diario al 31/8, y el veredicto se computaba igual
    cruzando dos ruedas distintas, sin avisar.

    Tres estados, en orden de gravedad:
      rojo    -- precios_diarios atrasado vs el ultimo cierre: falta correr el
                 recovery. Todo lo que se muestre es viejo.
      naranja -- precios al dia pero alguna tabla critica quedo atras: lo que
                 se muestra MEZCLA ruedas. Es el mas traicionero de los tres,
                 porque cada numero por separado parece correcto.
      verde   -- alineado.
    """
    e = _estado_datos_cache()
    if not e["ancla"]:
        st.error("precios_diarios esta vacia: no hay datos que mostrar.")
        return

    if not e["ancla_al_dia"]:
        atraso = e["atraso_ancla"]
        ruedas = "rueda" if atraso == 1 else "ruedas"
        st.badge("Datos atrasados", color="red")
        st.caption(
            f"Precios al {e['ancla']}, {atraso} {ruedas} detras del ultimo "
            f"cierre ({e['esperado']}). TODO el dashboard esta mostrando esa "
            f"rueda vieja. Correr scripts/manual/recovery_incremental.bat."
        )
    elif e["rezagadas"]:
        st.badge("Tablas desalineadas", color="orange")
        st.caption(
            f"Cierre {e['ancla']}, pero {_enumerar(e['rezagadas'])} "
            f"{'va' if len(e['rezagadas']) == 1 else 'van'} atras: lo que se "
            f"muestra cruza ruedas distintas. Correr "
            f"scripts/manual/ft_run_diario.bat (sincroniza y recomputa opciones)."
        )
    else:
        st.badge(f"Datos al {e['ancla']}", color="green")

    with st.expander("Detalle por tabla", expanded=False):
        filas = [{
            "Tabla": t["etiqueta"],
            "Ultima fecha": t["fecha"] or "-",
            "vs precios": ("al dia" if t["dias_vs_ancla"] == 0
                           else (f"{t['dias_vs_ancla']} "
                                 f"{'rueda' if t['dias_vs_ancla'] == 1 else 'ruedas'} atras"
                                 if t["dias_vs_ancla"] is not None else "-")),
            "Entra en el veredicto": "si" if t["critica"] else "no",
        } for t in e["tablas"]]
        st.table(pd.DataFrame(filas))
        st.caption(f"Ultimo dia habil NYSE cerrado: {e['esperado']} "
                   "(via src/utils/trading_calendar, no aritmetica de fechas).")


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
    event = st.dataframe(df, hide_index=True, width="stretch",
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
    st.caption("Veredicto sintetico (tecnico + opciones + estructura) de cada ticker, "
               "precomputado por la rutina nocturna. La busqueda es instantanea.")
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

    data = _veredictos_cache()
    universo = data["filas"]
    if not universo:
        st.warning("La tabla veredictos_universo_diario esta vacia. "
                   "La puebla la rutina nocturna (ft_run_diario.bat); "
                   "para hacerlo ahora a mano:")
        st.code("python scripts/compute_veredictos_universo.py", language="bash")
        return
    # Los veredictos pueden ser de una rueda anterior si la rutina nocturna no
    # corrio. Decirlo, en vez de presentarlos como de hoy.
    ancla = _estado_datos_cache().get("ancla")
    if data["fecha"] and ancla and data["fecha"] != ancla:
        st.warning(f"Veredictos del {data['fecha']}, pero los precios estan al "
                   f"{ancla}: falta correr la rutina nocturna. Estan calculados "
                   "sobre una rueda anterior.")

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
               f"(de {len(universo)} evaluados al {data['fecha']}).")
    ev = st.dataframe(dfv, hide_index=True, width="stretch",
                      on_select="rerun", selection_mode="single-row", key="scr_tbl")
    r2 = ev.selection.rows if ev and getattr(ev, "selection", None) else []
    if r2:
        tk2 = dfv.iloc[r2[0]]["Ticker"]
        if st.button(f"Abrir informe de {tk2}", key="scr_abrir"):
            _nav_a_informe(tk2)
    else:
        st.caption("Tip: seleccionar una fila para abrir su informe descriptivo.")


@st.cache_data(show_spinner=False)
def _hoy_cache(fecha_key: str):
    """Insumos de la vista Hoy, cacheados por fecha de datos.

    Se agrupan en UN cache y no en cuatro porque se consumen juntos y siempre
    en la misma pantalla: cuatro entradas serian cuatro claves que invalidan a
    la vez. Los decoradores viven en app.py por la misma razon que los otros
    (sintesis_data.py no importa Streamlit).
    """
    return {
        "reparto":    cargar_reparto_veredictos(),
        "radar":      cargar_radar(),
        "ft":         cargar_ft_resumen(),
        "posiciones": cargar_posiciones_ft_abiertas(),
        "earnings":   cargar_earnings_proximos(),
    }


def _vista_hoy():
    """
    Pantalla de arranque: que paso y que conviene mirar, sin elegir nada.

    Cada bloque abre con una FRASE que ya trae la conclusion, y recien despues
    muestra la tabla que la sostiene. Es la inversion del patron anterior del
    dashboard (tabla primero, interpretacion en un caption al pie): el numero
    suelto obliga a que el lector haga la cuenta cada vez.
    """
    d = _hoy_cache(fecha_datos())

    # -- 1. Clima del universo -------------------------------------------
    clima = construir_clima(d["reparto"])
    st.subheader("Clima del universo")
    st.markdown(clima["frase"])
    if clima["filas"]:
        cols = st.columns(len(clima["filas"]))
        for col, f in zip(cols, clima["filas"]):
            delta = None
            if clima["delta"] is not None:
                dv = clima["delta"].get(f["veredicto"], 0)
                delta = f"{dv:+d} vs rueda previa" if dv else "sin cambios"
            with col:
                st.metric(f["veredicto"].capitalize(),
                          f"{f['n']}  ({f['pct']}%)", delta=delta,
                          delta_color=("normal" if f["veredicto"] == "ALCISTA"
                                       else "inverse" if f["veredicto"] == "BAJISTA"
                                       else "off"))
    st.caption(f"Veredicto sintetico al {clima['fecha']} (tecnico + opciones + "
               "estructura). Detalle y filtros en Radar del dia.")

    st.divider()

    # -- 2. Lo inusual del dia -------------------------------------------
    inusual = construir_inusual(construir_radar(d["radar"], z=2.0),
                                d["radar"].get("fecha"))
    st.subheader("Lo inusual de la rueda")
    st.markdown(inusual["frase"])
    if inusual["filas"]:
        df = pd.DataFrame(inusual["filas"])[
            ["ticker", "sector", "tipo", "magnitud"]].rename(columns={
                "ticker": "Ticker", "sector": "Sector",
                "tipo": "Que se salio de lo normal", "magnitud": "Magnitud (z)"})
        st.dataframe(df, hide_index=True, width="stretch")
        c1, _ = st.columns([1, 3])
        with c1:
            if st.button("Ver el radar completo", width="stretch", key="hoy_radar"):
                _nav_a("radar")
    st.caption(f"Anomalias de opciones al {inusual['fecha']}. z = desvios "
               "respecto de la propia historia del ticker.")

    st.divider()

    # -- 3. Forward Testing ----------------------------------------------
    ft = construir_ft(d["ft"])
    st.subheader("Forward Testing")
    st.markdown(ft["frase"])
    if ft["kpis"]:
        cols = st.columns(len(ft["kpis"]))
        for col, k in zip(cols, ft["kpis"]):
            with col:
                st.metric(k["label"], k["valor"])
        with st.expander("Detalle por estrategia", expanded=False):
            # Altura explicita: dentro de un expander, st.dataframe se virtualiza
            # y colapsa a 2 filas visibles (verificado en el navegador). Son 10
            # estrategias fijas y se quieren ver todas de una.
            st.dataframe(pd.DataFrame(ft["filas"]), hide_index=True,
                         width="stretch",
                         height=(len(ft["filas"]) + 1) * 35 + 3)
            st.caption("Equity MARCADA A MERCADO (ft_equity_diaria): incluye el "
                       "resultado de las posiciones abiertas, que en la curva a "
                       "costo de entrada es invisible.")
    st.caption(f"Al {ft['fecha']}." if ft["fecha"] else "")

    st.divider()

    # -- 4. Agenda --------------------------------------------------------
    agenda = construir_agenda(d["earnings"], d["posiciones"])
    st.subheader("Balances que vienen")
    st.markdown(agenda["frase"])
    if agenda["filas"]:
        st.dataframe(pd.DataFrame(agenda["filas"]), hide_index=True,
                     width="stretch")
    st.caption("Cruce de earnings_calendar con las posiciones FT abiertas. Un "
               "balance con posicion abierta es el unico caso de esta pantalla "
               "que pide una decision antes de que abra el mercado.")


def _vista_radar():
    # -- 1. Radar de anomalias de opciones --------------------------------
    st.subheader("Radar del dia - actividad inusual en opciones")
    _radar_anomalias(_radar_cache(fecha_datos()))

    # -- 2. Screener por veredicto ----------------------------------------
    st.divider()
    _radar_screener()


def _tabla_financiera(filas, columnas_orden):
    """Tabla del analisis fundamental.

    Dos cosas que no hacia antes:
      - El percentil dentro del sector se dibuja como BARRA (column_config.
        ProgressColumn) y no como texto. Es el numero que se lee de un vistazo
        -- "esta arriba o abajo de sus pares" -- y como texto obligaba a leer
        las 20 filas una por una para ubicar los extremos.
      - Los colores good/bad pasan a los del tema oscuro. Los anteriores
        (#1a7f37 / #cf222e) estaban elegidos para fondo claro y sobre el fondo
        actual quedaban casi ilegibles.
    """
    def _estilo(row):
        color = row.get("_color")
        css = ""
        if color == "good":
            css = "color: #4ade80; font-weight: 600;"
        elif color == "bad":
            css = "color: #f87171; font-weight: 600;"
        # Styler.apply(axis=1) exige un estilo por CADA columna del DataFrame
        # (incluye las ocultas) -> iterar sobre row.index, no sobre
        # columnas_orden.
        return [css if c == "vs Sector" else "" for c in row.index]

    df = pd.DataFrame(filas)
    df_disp = pd.DataFrame({
        "Metrica":   df["metrica"],
        "Valor":     df["valor"],
        "vs Sector": df["vs_mediana"],
        "Mediana":   df["mediana"],
        "Percentil": df["percentil_num"],
        "_color":    df["color"],
    })
    styler = df_disp.style.apply(_estilo, axis=1)
    st.dataframe(
        styler, hide_index=True, width="stretch",
        column_config={
            "_color": None,
            "Percentil": st.column_config.ProgressColumn(
                "Percentil en el sector", min_value=0, max_value=100,
                format="%.0f%%",
                help="Fraccion de pares de la misma region con un valor menor "
                     "o igual. 50% = justo en la mediana del sector."),
        },
    )


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
                     width="stretch"):
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
                mime="image/png", key="ficha_emp_dl", width="stretch")
        st.image(st.session_state["ficha_emp_bytes"], width="stretch")
    st.caption(f"Ficha de presentacion (PNG, fondo oscuro) SOLO de {tk}: el "
               "ultimo trimestre reportado + variacion interanual, la empresa "
               "contra si misma (sin pares). Descriptivo, no recomienda.")


def _vista_financiera(tickers):
    st.subheader("Analisis Financiero")
    st.caption("Foto fundamental del ultimo trimestre reportado: valuacion, "
               "calidad, crecimiento y solvencia, comparada con la mediana de "
               "pares de la misma region. Descriptivo, no recomienda.")

    modo = st.segmented_control("Modo", ["Por ticker", "Screener sectorial"],
                                default="Por ticker", key="fin_modo")

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
        with st.popover("Como leer esta tabla", icon=":material/help:"):
            st.markdown("**vs Sector**: distancia a la mediana de pares. "
                        "Verde = mejor, rojo = peor, segun lo que signifique la "
                        "metrica (un PER bajo es bueno; un ROE bajo no).")
            st.markdown("**Percentil**: posicion dentro del sector. "
                        "50% = justo en la mediana.")
            st.markdown("Los ABSOLUTOS (BPA, valor libro) van en moneda de "
                        "reporte: no son comparables entre tickers sin pasar "
                        "por FX.")
            st.markdown("En bancos, ROIC / margen operativo / liquidez pueden "
                        "aparecer como guion: no tienen estructura contable "
                        "aplicable.")

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

        if st.button("Generar", width="content"):
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
            st.dataframe(df_full, hide_index=True, width="stretch")
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
                     hide_index=True, width="stretch")
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
        if st.sidebar.button("Limpiar conversacion", width="stretch"):
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


# -- Navegacion --------------------------------------------------------------
#
# st.navigation en vez del st.radio de 8 items planos. Tres ventajas concretas:
#
#   1. Agrupa por proposito (Panorama / Analisis / Cartera / Herramientas). Con
#      8 items sueltos hay que leer la lista entera para ubicarse; con grupos
#      se salta al que corresponde.
#   2. Cada vista tiene su URL (/hoy, /informe, ...), asi que se puede marcar
#      como favorita -- y combinada con ?ticker=NVDA, un informe concreto es un
#      link compartible.
#   3. Habilita st.switch_page, que reemplaza el mecanismo de flags diferidos
#      (_ir_informe / _ir_radar). Ese mecanismo existia SOLO porque la key de un
#      widget ya instanciado no se puede reescribir: con el radio en pantalla,
#      mover "modo" desde una vista tiraba excepcion, y habia que dejar una
#      marca para consumirla al inicio de la corrida siguiente. Con navigation
#      el salto es directo.

_PAGINAS = {}


def _nav_a(clave: str, **estado):
    """Salta a otra vista, fijando estado antes si hace falta.

    _PAGINAS lo puebla main() en cada corrida ANTES de ejecutar la pagina, asi
    que siempre esta cargado cuando una vista llama aca.
    """
    for k, v in estado.items():
        st.session_state[k] = v
    st.switch_page(_PAGINAS[clave])


def _nav_a_informe(tk):
    """Abre el informe descriptivo de tk.

    El ticker se escribe directo en session_state: su selectbox no se instancio
    en la vista de origen, y con la key unificada (`ticker`, bindeada a
    query-params) es la via soportada para moverlo -- un parametro bindeado no
    se puede tocar via st.query_params.
    """
    _nav_a("informe", ticker=tk)


@st.cache_data(show_spinner=False, ttl=300)
def _tickers_cache():
    """El universo se pide en casi todas las vistas y cambia con un alta/baja
    manual, no durante la sesion."""
    return listar_tickers()


# Envoltorios de pagina: st.Page recibe callables SIN argumentos, y varias
# vistas necesitan el universo. Cada una lo pide al cache en vez de recibirlo.

def _pg_hoy():
    _vista_hoy()


def _pg_radar():
    _vista_radar()


def _pg_informe():
    _vista_informe(_tickers_cache())


def _pg_financiero():
    _vista_financiera(_tickers_cache())


def _pg_balances():
    from dashboard.earnings_reaccion import construir_reaccion
    construir_reaccion(_tickers_cache())


def _pg_carteras():
    from dashboard.carteras import construir_carteras
    construir_carteras(_tickers_cache())


def _pg_performance():
    from dashboard.performance import construir_performance
    construir_performance(_tickers_cache())


def main():
    st.set_page_config(page_title="Panel de analisis", layout="wide")

    # Cromo comun a todas las vistas: va ANTES de pg.run(), que es lo que hace
    # que el entrypoint funcione como marco y no como pagina.
    st.title("Panel de analisis")
    st.caption("Descriptivo, no predictivo. Complemento de TradingView. "
               "Cruza tecnico + opciones + sector con reglas trazables.")
    _banda_frescura()

    if not _tickers_cache():
        st.error("No hay tickers en la DB local (precios_diarios vacia).")
        return

    paginas = {
        "hoy": st.Page(_pg_hoy, title="Hoy", url_path="hoy",
                       icon=":material/today:", default=True),
        "radar": st.Page(_pg_radar, title="Radar del dia", url_path="radar",
                         icon=":material/radar:"),
        "informe": st.Page(_pg_informe, title="Informe por ticker",
                           url_path="informe", icon=":material/description:"),
        "financiero": st.Page(_pg_financiero, title="Analisis Financiero",
                              url_path="financiero",
                              icon=":material/account_balance:"),
        "balances": st.Page(_pg_balances, title="Reaccion a balances",
                            url_path="balances", icon=":material/event:"),
        "carteras": st.Page(_pg_carteras, title="Carteras", url_path="carteras",
                            icon=":material/donut_small:"),
        "performance": st.Page(_pg_performance, title="Performance",
                               url_path="performance",
                               icon=":material/trending_up:"),
        "chat": st.Page(_vista_chat, title="Consultas (IA)",
                        url_path="consultas", icon=":material/chat:"),
    }
    _PAGINAS.clear()
    _PAGINAS.update(paginas)

    # Los grupos ordenan de lo general a lo propio, igual que la vista Hoy:
    # que pasa en el mercado -> que pasa con un papel -> que pasa con mi plata.
    st.navigation({
        "Panorama": [paginas["hoy"], paginas["radar"]],
        "Analisis": [paginas["informe"], paginas["financiero"],
                     paginas["balances"]],
        "Cartera": [paginas["carteras"], paginas["performance"]],
        "Herramientas": [paginas["chat"]],
    }).run()


if __name__ == "__main__":
    main()
