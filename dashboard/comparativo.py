"""
comparativo.py -- Vista "Comparativo Fundamental" del dashboard.

Que hace
--------
Toma un ticker y una TESIS DE PRECIO ("si subiera 20%") y muestra en que se
convierten sus multiplos. Es la version visual de scripts/manual/
valuacion_implicita.py: mismo motor puro (src/utils/valuacion_implicita.py),
misma fuente (fundamentales_sec_multiplos_d), sin recalcular nada aca.

Por que existe
--------------
La fuente SEC se mergeo a main sin un solo consumidor de interfaz: el unico
lector era un CLI. Una fuente que hay que consultar acordandose de un comando
no se consulta. Este tab es la superficie donde el dato se mira -- y por lo
tanto donde se nota si se rompe.

Las dos cosas que la vista tiene que dejar ver
----------------------------------------------
1. LOS MULTIPLOS NO SE MUEVEN IGUAL. PER y P/S escalan LINEAL con el precio:
   +10% de precio son +10% de PER. EV/EBITDA NO, porque el EV es
   market cap + deuda neta y al mover el precio solo se mueve la primera
   parte. En una empresa endeudada +10% de precio puede ser +6% de
   EV/EBITDA; en una con caja neta (deuda negativa) el efecto es al reves y
   AMPLIFICA. Por eso cada multiplo muestra su propia variacion al lado: la
   asimetria se ve sola, sin explicarla.

2. DONDE CAE ESO EN SU PROPIA HISTORIA. Un PER de 28 no dice nada; "percentil
   85 de los ultimos 3 anios" si. El percentil viene del motor y es
   ESTRICTO -- si la ventana de 756 ruedas no esta llena en tiempo, no hay
   percentil y se dice, en vez de mostrar uno calculado sobre poca historia
   que parece valido y no lo es.

Huecos que se muestran como huecos
----------------------------------
EV/EBITDA hoy llega a 80 de 144 tickers, por DOS causas distintas: 48 no
tienen EBITDA (falta la D&A) y 29 no tienen deuda neta (companyfacts descarta
los hechos dimensionados). La vista dice "sin dato" y CUAL DE LAS DOS es, en
vez de dejar una celda vacia que se lee como cero.
"""

import pandas as pd
import streamlit as st

from src.data.database import query_df
from src.utils import valuacion_implicita as V

TABLA = "fundamentales_sec_multiplos_d"

# Los tres que pidio la vista. P/B y fcf_yield quedan afuera a proposito:
# fcf_yield se lee AL REVES que los demas (mas alto = mas barato) y mezclarlo
# en la misma grilla invita a leerlo mal.
VISTA = [
    ("pe_ratio", "PER", "resultado neto (TTM)"),
    ("ps_ratio", "P/S", "ventas (TTM)"),
    ("ev_ebitda", "EV/EBITDA", "EBITDA (TTM)"),
]

# Campos que necesita el motor para armar los multiplos. Son exactamente las
# columnas de la tabla: el motor recibe la fila sin traducir nada.
CAMPOS = ["fecha", "close", "shares", "net_debt", "equity",
          "net_income_ttm", "revenue_ttm", "ebitda_ttm", "fcf_ttm",
          "period_end", "filed_primero", "shares_fuente", "shares_dias",
          "net_debt_q"]

METRICAS = [m for m, _, _ in VISTA]
ETIQUETA = {m: et for m, et, _ in VISTA}


# --------------------------------------------------------------------- datos

@st.cache_data(ttl=600)
def _base(ticker):
    """Ultima rueda del ticker. None si no esta en la fuente SEC."""
    df = query_df(
        "SELECT %s FROM %s WHERE ticker = :t ORDER BY fecha DESC LIMIT 1"
        % (", ".join(CAMPOS), TABLA), {"t": ticker})
    return None if df.empty else df.iloc[0].to_dict()


@st.cache_data(ttl=600)
def _historia(ticker):
    """
    {metrica: [valores]} de toda la historia del ticker, para el percentil.

    Se traen los nulos incluidos y los filtra el motor: recortarlos aca
    cambiaria silenciosamente el denominador del percentil.
    """
    df = query_df(
        "SELECT %s FROM %s WHERE ticker = :t ORDER BY fecha"
        % (", ".join(METRICAS), TABLA), {"t": ticker})
    if df.empty:
        return {}
    return {m: df[m].tolist() for m in METRICAS}


@st.cache_data(ttl=600)
def _tickers_con_fuente():
    """Solo los que estan en la fuente SEC. No tiene sentido ofrecer el resto."""
    df = query_df("SELECT DISTINCT ticker FROM %s ORDER BY ticker" % TABLA)
    return df["ticker"].tolist() if not df.empty else []


# ------------------------------------------------------------------ formato

def _n(v, dec=2):
    return "--" if v is None or pd.isna(v) else f"{float(v):,.{dec}f}"


def _pct(v, dec=1):
    return "--" if v is None or pd.isna(v) else f"{float(v) * 100:,.{dec}f}%"


def _delta(actual, nuevo):
    """Variacion del multiplo, para que la asimetria del EV se vea sola."""
    if actual is None or nuevo is None or pd.isna(actual) or pd.isna(nuevo):
        return None
    a = float(actual)
    return None if a == 0 else (float(nuevo) / a - 1.0)


def _motivo_sin_dato(metrica, base):
    """
    Por que falta un multiplo. Un 'sin dato' explicado se puede accionar; una
    celda vacia se lee como cero.

    El hueco de EV/EBITDA tiene DOS causas y conviene distinguirlas porque se
    arreglan distinto. Medido sobre la ultima rueda, 144 tickers:
        48 sin EBITDA      -> falta la D&A. 25 de 147 tickers no tienen NINGUNA
                              D&A en su ultimo Q, y no es un hueco puntual:
                              MSFT tiene operating_income en los 10 trimestres
                              y D&A nulo en los 10. Usa un tag que no esta en
                              los sinonimos. Se cura ampliando el mapeo, con
                              cuidado: `Depreciation` a secas se saco a
                              proposito por ser un SUBCONJUNTO (mediana 73% de
                              la D&A completa) que inflaba el multiplo.
        29 sin deuda neta  -> la empresa tagea parte de su deuda con
                              dimensiones y companyfacts las descarta. El EV se
                              apaga en vez de calcularse incompleto.
    Se chequea el EBITDA primero porque es la causa mas frecuente.
    """
    if metrica == "ev_ebitda":
        if not _positivo(base.get("ebitda_ttm")):
            return ("sin EBITDA: no hay D&A para este ticker, o el EBITDA TTM "
                    "no es positivo. En bancos es lo esperable -- el EV no "
                    "aplica cuando los depositos entran como deuda")
        if base.get("net_debt") is None or pd.isna(base.get("net_debt")):
            return ("sin deuda neta: la empresa tagea parte de su deuda con "
                    "dimensiones y companyfacts las descarta, asi que el EV "
                    "se apaga en vez de calcularse incompleto")
    if metrica == "pe_ratio" and not _positivo(base.get("net_income_ttm")):
        return "resultado neto TTM no positivo: no es 'barato', es otra categoria"
    if metrica == "ps_ratio" and not _positivo(base.get("revenue_ttm")):
        return "ventas TTM no positivas"
    return "sin dato"


def _positivo(v):
    return v is not None and not pd.isna(v) and float(v) > 0


# -------------------------------------------------------------------- vista

def construir_comparativo(tickers=None):
    st.header("Comparativo Fundamental")
    st.caption(
        "Multiplos sobre la fuente SEC XBRL y que pasa con ellos bajo una "
        "tesis de precio. Solo lectura: el computo vive en "
        "`scripts/compute_sec_multiplos.py`."
    )

    disponibles = _tickers_con_fuente()
    if not disponibles:
        st.warning(
            "La tabla `fundamentales_sec_multiplos_d` esta vacia o no existe. "
            "Correr `scripts/manual/refresh_fundamentales_sec.bat`."
        )
        return

    ticker = st.selectbox(
        "Ticker", disponibles, key="comp_ticker",
        help="Se puede escribir para filtrar. Solo aparecen los %d tickers "
             "que tienen fuente SEC: los ADR extranjeros presentan 20-F "
             "anual y no tienen XBRL trimestral." % len(disponibles),
    )
    if not ticker:
        return

    base = _base(ticker)
    if not base:
        st.warning("Sin datos para %s." % ticker)
        return

    historia = _historia(ticker)
    actual = V.escenario(base, historia)          # sin tesis: el estado de hoy

    # ---------------------------------------------------------------- hoy --
    st.subheader("Hoy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", _n(base.get("close")))
    for col, (metrica, etiqueta, _) in zip((c2, c3, c4), VISTA):
        val = actual["multiplos"].get(metrica)
        pct = actual["percentiles"].get(metrica)
        col.metric(etiqueta, _n(val),
                   help=None if val is not None
                   else _motivo_sin_dato(metrica, base))
        col.caption(("percentil %s" % _n(pct * 100, 0))
                    if pct is not None else "sin percentil")

    _pie_de_datos(base)

    st.divider()

    # -------------------------------------------------------------- tesis --
    st.subheader("Tesis 1")
    st.caption("Mueve SOLO el precio. La deuda neta y los denominadores TTM "
               "quedan como estan: es una tesis sobre la cotizacion, no sobre "
               "el negocio.")

    col_in, col_res = st.columns([1, 3])
    with col_in:
        variacion = st.number_input(
            "Incremento del precio (%)", min_value=-95.0, max_value=500.0,
            value=20.0, step=5.0, key="comp_var",
        )
    tesis = V.escenario(base, historia, variacion=variacion / 100.0)

    with col_res:
        st.metric("Precio de la tesis", _n(tesis["precio"]),
                  delta=_pct(tesis.get("variacion")))

    filas = []
    for metrica, etiqueta, denom in VISTA:
        hoy = actual["multiplos"].get(metrica)
        nuevo = tesis["multiplos"].get(metrica)
        d = _delta(hoy, nuevo)
        pct_new = tesis["percentiles"].get(metrica)
        filas.append({
            "Multiplo": etiqueta,
            "Denominador": denom,
            "Hoy": _n(hoy),
            "Con la tesis": _n(nuevo),
            "Variacion": _pct(d),
            "Percentil propio": (_n(pct_new * 100, 0) if pct_new is not None
                                 else "--"),
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True,
                 use_container_width=True)

    _nota_asimetria(base, actual, tesis, variacion)

    fuera = [ETIQUETA[m] for m in tesis.get("fuera_de_rango", [])
             if m in ETIQUETA]
    if fuera:
        st.warning(
            "Con esta tesis, %s quedaria en un valor que **nunca se vio** en "
            "la historia disponible del ticker. No invalida la tesis: dice "
            "que para sostenerla hay que creer algo que todavia no paso."
            % ", ".join(fuera)
        )

    faltan = [et for m, et, _ in VISTA
              if actual["multiplos"].get(m) is None]
    if faltan:
        st.info(
            "Sin dato para %s. Motivo: %s"
            % (", ".join(faltan),
               "; ".join(_motivo_sin_dato(m, base)
                         for m, et, _ in VISTA
                         if actual["multiplos"].get(m) is None))
        )



def _nota_asimetria(base, actual, tesis, variacion):
    """
    Explica el numero SOLO cuando hay algo que explicar: si el EV/EBITDA se
    movio distinto que el PER, se dice por que. Sin dato, no se dice nada --
    una nota que aparece siempre deja de leerse.
    """
    d_pe = _delta(actual["multiplos"].get("pe_ratio"),
                  tesis["multiplos"].get("pe_ratio"))
    d_ev = _delta(actual["multiplos"].get("ev_ebitda"),
                  tesis["multiplos"].get("ev_ebitda"))
    if d_pe is None or d_ev is None:
        return
    if abs(d_ev - d_pe) < 0.005:
        return

    nd = base.get("net_debt")
    caja_neta = nd is not None and not pd.isna(nd) and float(nd) < 0
    st.caption(
        "El EV/EBITDA se movio %s (%s) que el PER (%s): el EV es "
        "%s, y al mover el precio solo cambia el market cap. %s"
        % ("menos" if abs(d_ev) < abs(d_pe) else "mas",
           _pct(d_ev), _pct(d_pe),
           "market cap MENOS la caja neta" if caja_neta
           else "market cap MAS la deuda neta",
           "Tiene caja neta, asi que el efecto se amplifica en vez de "
           "amortiguarse." if caja_neta else
           "Es lo que hace al EV/EBITDA mas comparable entre empresas con "
           "deudas distintas.")
    )


def _pie_de_datos(base):
    """De cuando es el dato. Un multiplo sin su fecha invita a confundirlo."""
    partes = []
    if base.get("fecha") is not None:
        partes.append("rueda %s" % base["fecha"])
    if base.get("period_end") is not None:
        partes.append("ultimo balance %s" % base["period_end"])
    if base.get("shares_fuente"):
        dias = base.get("shares_dias")
        antig = ("" if dias is None or pd.isna(dias)
                 else ", conteo de hace %d dias" % int(dias))
        partes.append("acciones via `%s`%s" % (base["shares_fuente"], antig))
    if partes:
        st.caption(" | ".join(partes))

    # El arrastre de deuda se avisa SIEMPRE que ocurre. Es la contrapartida de
    # haberlo permitido: usar el ultimo valor conocido es legitimo, hacerlo sin
    # decirlo no. Ver MAX_ARRASTRE_Q en src/utils/fundamentales_ttm.py.
    q = base.get("net_debt_q")
    if q is not None and not pd.isna(q) and int(q) > 0:
        st.caption(
            ":orange[Deuda arrastrada: uno de los dos componentes no viene en "
            "el ultimo balance y se uso el de hace %d trimestre%s.] El EV "
            "queda aproximado -- la deuda es un stock, se mueve una mediana "
            "de 2,7%% por trimestre." % (int(q), "s" if int(q) != 1 else ""))
