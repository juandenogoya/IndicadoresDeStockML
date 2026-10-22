"""
status_balances.py -- STATUS de los balances fundamentales. SOLO LEE.

Responde dos preguntas y las cruza:
  1. (local) que trimestre tenemos de cada ticker y a quien le FALTA el ultimo.
  2. (red)   que trimestre tiene yahooquery HOY, y si trae los IMPORTES.

POR QUE EXISTE. El metodo de deteccion estaba documentado (docs/fuentes_fundamentales.md
sec. 2) pero no habia herramienta: cada vez que se queria saber "cuan atrasados
estamos" habia que rearmar el analisis a mano. Medido el 21/10/2026: el refresh es
MANUAL y llevaba 55 dias sin correr, con 64 de 199 tickers sin su ultimo balance y
nadie avisando. El dato no grita: `fundamentales_ratios_q`, los multiplos `*_px`, la
vista "Analisis Financiero" y `get_fundamentals` del MCP siguen contestando con el
trimestre viejo, y las medianas sectoriales mezclan tickers en trimestres distintos.

EL METODO (cadencia propia; los calendarios fiscales desplazados se acomodan solos):

    proximo_Q_esperado = ultimo_Q_con_income + 3 meses (fin de mes)
    anuncio_esperado   = proximo_Q_esperado + lag_mediano_del_ticker
    falta si             anuncio_esperado < hoy - TOLERANCIA_DIAS

El lag mediano (cierre fiscal -> anuncio) sale de `earnings_historico`, que tiene ~26
trimestres por ticker. Los calendarios de FT (`earnings_calendar`) NO sirven para
esto: solo guardan la proxima fecha.

TRES COSAS QUE NO SE PUEDEN MEDIR A LA LIGERA, cada una aprendida de un error:

1. "Ultimo Q" NO es la ultima fila. Existen FILAS STUB: el trimestre entra con la
   fila creada y los importes en NULL. Se mide el ultimo Q con income REAL.
2. "yahooquery lo tiene" NO es que el periodo aparezca en la respuesta. Para ABT,
   AMZN, BA, BAC y MDLZ la fuente devuelve el periodo con `DilutedEPS` y el resto
   NULL: hay fila y no hay balance. Por eso la fase 2 chequea los IMPORTES. Leer solo
   el periodo hace concluir "ya esta disponible" sobre un ticker que sigue vacio
   (paso el 21/10/2026).
3. Un ticker SIN NINGUNA fila trimestral no es un atraso: HMY/RIO/UL/VOD reportan
   semestral y la fuente no tiene trimestral para ellos. Se informan aparte y NO
   cuentan como faltantes; con la proyeccion por trimestres darian falso positivo
   para siempre.

Uso:
    python scripts/manual/status_balances.py                 local + yahooquery
    python scripts/manual/status_balances.py --solo-local    sin red (instantaneo)
    python scripts/manual/status_balances.py --csv           deja el detalle por ticker

Salida: 0 = al dia | 2 = hay faltantes recuperables | 1 = error.
Si hay faltantes, el arreglo es `scripts/manual/refresh_fundamentales.bat`.
Ver docs/fuentes_fundamentales.md.
"""

import argparse
import os
import sys
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.environ.pop("DATABASE_URL", None)          # LOCAL manda (Plan C)

import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine

TOLERANCIA_DIAS = 5     # margen sobre el anuncio esperado antes de decir "falta"
RECIENTE_DIAS = 20      # vencido por menos que esto = reporto hace poco, la fuente tarda
CHUNK = 20              # mismo tamano que usa el refresh
SEP = "=" * 78


def log(msg=""):
    print(msg, flush=True)


# ── Fase 1: que tenemos ──────────────────────────────────────────────────────

def cargar_local():
    with get_engine().connect() as c:
        universo = [r[0] for r in c.execute(text(
            "SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker")).fetchall()]
        inc = pd.read_sql(text("""
            SELECT ticker, fiscal_period_end, total_revenue, net_income, fetched_at
            FROM fundamentales_income_q
        """), c)
        eh = pd.read_sql(text("""
            SELECT ticker, fiscal_period_end, announcement_date
            FROM earnings_historico
        """), c)
    inc["fiscal_period_end"] = pd.to_datetime(inc["fiscal_period_end"])
    eh["fiscal_period_end"] = pd.to_datetime(eh["fiscal_period_end"])
    eh["announcement_date"] = pd.to_datetime(eh["announcement_date"])
    return universo, inc, eh


def lags_por_ticker(eh):
    """Mediana de dias entre cierre fiscal y anuncio. Devuelve (serie, default)."""
    eh = eh.copy()
    eh["lag"] = (eh["announcement_date"] - eh["fiscal_period_end"]).dt.days
    por_tk = eh[eh["lag"].between(1, 180)].groupby("ticker")["lag"].median()
    return por_tk, float(por_tk.median())


def estado_local(universo, inc, eh, hoy):
    """Un DataFrame por ticker con su ultimo Q real y si le falta el siguiente."""
    con_dato = inc[inc["total_revenue"].notna() | inc["net_income"].notna()]
    ult_fila = inc.groupby("ticker")["fiscal_period_end"].max()
    ult_real = con_dato.groupby("ticker")["fiscal_period_end"].max()
    lag_tk, lag_def = lags_por_ticker(eh)

    filas = []
    for tk in universo:
        uq = ult_real.get(tk, pd.NaT)
        # sin NINGUNA fila trimestral: no es atraso (semestral / la fuente no lo da)
        if tk not in ult_fila.index:
            filas.append({"ticker": tk, "ultimo_Q": pd.NaT, "proximo_Q": None,
                          "anuncio_esp": None, "vencido_dias": None,
                          "clase": "sin trimestral", "falta": False})
            continue
        if pd.isna(uq):
            filas.append({"ticker": tk, "ultimo_Q": pd.NaT, "proximo_Q": None,
                          "anuncio_esp": None, "vencido_dias": None,
                          "clase": "solo filas stub", "falta": True})
            continue
        prox = (uq + pd.DateOffset(months=3) + pd.offsets.MonthEnd(0)).date()
        esp = prox + timedelta(days=int(float(lag_tk.get(tk, lag_def))))
        venc = ((hoy - timedelta(days=TOLERANCIA_DIAS)) - esp).days
        falta = venc > 0
        if not falta:
            clase = "al dia"
        elif ult_fila[tk] > uq:
            clase = "stub en la fuente"      # la fila del Q nuevo existe, sin importes
        elif venc <= RECIENTE_DIAS:
            clase = "reporto hace poco"
        else:
            clase = "vencido"
        filas.append({"ticker": tk, "ultimo_Q": uq, "proximo_Q": prox,
                      "anuncio_esp": esp, "vencido_dias": venc,
                      "clase": clase, "falta": falta})
    return pd.DataFrame(filas), ult_real, lag_def


# ── Fase 2: que tiene la fuente ──────────────────────────────────────────────

def consultar_yahooquery(universo):
    """
    {ticker: (ultimo_periodo, ultimo_periodo_CON_IMPORTES)} desde yahooquery, con la
    misma llamada que el refresh. El segundo valor es el que importa: la fuente
    devuelve periodos con EPS y sin estado contable (ver docstring, punto 2).
    """
    from yahooquery import Ticker

    from src.utils.yfinance_lock import acquire as acquire_yf_lock
    acquire_yf_lock("status_balances.py (solo lectura)")

    out, sin_respuesta = {}, []
    total = -(-len(universo) // CHUNK)
    for i in range(0, len(universo), CHUNK):
        lote = universo[i:i + CHUNK]
        try:
            df = Ticker(lote, asynchronous=True).income_statement(frequency="q")
        except Exception as e:
            log(f"  chunk {i//CHUNK+1}/{total}: ERROR {str(e)[:60]}")
            sin_respuesta += lote
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            log(f"  chunk {i//CHUNK+1}/{total}: sin datos")
            sin_respuesta += lote
            continue
        d = df.reset_index()
        col_tk = "symbol" if "symbol" in d.columns else d.columns[0]
        d["asOfDate"] = pd.to_datetime(d["asOfDate"])
        if "periodType" in d.columns:
            d = d[d["periodType"].isin(["3M", "quarterly", "q"])]
        # los nombres son los de INCOME_MAP del refresh
        cols = [c for c in ("TotalRevenue", "NetIncome") if c in d.columns]
        for tk, g in d.groupby(col_tk):
            periodo = g["asOfDate"].max()
            con_imp = pd.NaT
            if cols:
                g2 = g.dropna(subset=cols, how="all")
                if len(g2):
                    con_imp = g2["asOfDate"].max()
            out[tk] = (periodo, con_imp)
        sin_respuesta += [t for t in lote if t not in out]
        log(f"  chunk {i//CHUNK+1}/{total}: {len(out)} tickers acumulados")
    return out, sin_respuesta


# ── Informe ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Status de los balances fundamentales")
    ap.add_argument("--solo-local", action="store_true",
                    help="no consulta yahooquery (instantaneo)")
    ap.add_argument("--csv", action="store_true",
                    help="escribe el detalle por ticker en reportes/")
    args = ap.parse_args()
    hoy = date.today()

    try:
        universo, inc, eh = cargar_local()
    except Exception as e:
        log(f"[ERROR] DB: {e}")
        return 1

    est, ult_real, lag_def = estado_local(universo, inc, eh, hoy)
    ult_fetch = inc["fetched_at"].max()

    log()
    log(SEP)
    log(f"  STATUS DE BALANCES  |  hoy {hoy}  |  universo activo: {len(universo)}")
    log(SEP)
    log(f"\nQUE TENEMOS (fundamentales_income_q, fuente yahooquery)")
    log(f"  ultimo trimestre cargado : {ult_real.max():%Y-%m-%d}")
    log(f"  ultimo refresh           : {ult_fetch:%Y-%m-%d} "
        f"({(hoy - ult_fetch.date()).days} dias)")
    log(f"  lag mediano del universo : {lag_def:.0f} dias (cierre fiscal -> anuncio)")
    log("\n  reparto por ultimo trimestre CON IMPORTES:")
    for q, n in ult_real.dt.date.value_counts().sort_index(ascending=False).items():
        log(f"    {q}  {n:3d} tickers")

    log("\n  por clase:")
    for clase, n in est["clase"].value_counts().items():
        log(f"    {clase:22s} {n:3d}")

    faltan = est[est["falta"]]
    log(f"\n  >>> LE FALTA EL ULTIMO BALANCE A {len(faltan)} de {len(universo)} "
        f"({len(faltan)/len(universo):.0%})")
    sin_tri = est[est["clase"] == "sin trimestral"]
    if len(sin_tri):
        log(f"      (aparte: {len(sin_tri)} sin trimestral, NO cuentan como atraso: "
            f"{sorted(sin_tri['ticker'])})")
    for clase in ("vencido", "stub en la fuente", "reporto hace poco"):
        g = faltan[faltan["clase"] == clase].sort_values("vencido_dias", ascending=False)
        if not len(g):
            continue
        log(f"\n      {clase} ({len(g)}):")
        for r in g.head(12).itertuples():
            log(f"        {r.ticker:6s} tenemos {r.ultimo_Q:%Y-%m-%d}  "
                f"esperado ~{r.anuncio_esp}  vencido {int(r.vencido_dias)} d")
        if len(g) > 12:
            log(f"        ... y {len(g)-12} mas")

    if args.solo_local:
        log("\n(--solo-local: no se consulto yahooquery)")
        return 2 if len(faltan) else 0

    log(f"\nQUE TIENE YAHOOQUERY HOY ({len(universo)} tickers, chunks de {CHUNK})")
    try:
        yq, sin_resp = consultar_yahooquery(universo)
    except SystemExit:
        raise
    except Exception as e:
        log(f"[ERROR] yahooquery: {e}")
        return 1
    log(f"\n  respondio por {len(yq)} de {len(universo)} | sin respuesta: {len(sin_resp)}")

    cmp_ = []
    for r in est.itertuples():
        periodo, con_imp = yq.get(r.ticker, (pd.NaT, pd.NaT))
        nuestro = r.ultimo_Q
        if pd.isna(periodo):
            sit = "la fuente no responde"
        elif pd.notna(con_imp) and (pd.isna(nuestro) or con_imp > nuestro):
            sit = "RECUPERABLE con un refresh"
        elif pd.notna(periodo) and (pd.isna(nuestro) or periodo > nuestro):
            sit = "la fuente lo tiene VACIO (EPS solo)"
        else:
            sit = "la fuente no tiene nada mas nuevo"
        cmp_.append({"ticker": r.ticker, "clase": r.clase, "nuestro": nuestro,
                     "yq_periodo": periodo, "yq_con_importes": con_imp,
                     "situacion": sit})
    cmp_ = pd.DataFrame(cmp_)

    log("\n" + SEP)
    log("  EL CRUCE: nuestro ultimo Q contra el de la fuente")
    log(SEP)
    for sit, n in cmp_["situacion"].value_counts().items():
        log(f"  {sit:38s} {n:3d}")

    rec = cmp_[cmp_["situacion"] == "RECUPERABLE con un refresh"]
    if len(rec):
        log(f"\n  >>> {len(rec)} se recuperan AHORA con "
            f"scripts/manual/refresh_fundamentales.bat")
        for r in rec.head(12).itertuples():
            tenemos = f"{r.nuestro:%Y-%m-%d}" if pd.notna(r.nuestro) else "nada"
            log(f"    {r.ticker:6s} tenemos {tenemos} -> "
                f"yahooquery {r.yq_con_importes:%Y-%m-%d}")
        if len(rec) > 12:
            log(f"    ... y {len(rec)-12} mas")

    vacio = cmp_[cmp_["situacion"] == "la fuente lo tiene VACIO (EPS solo)"]
    if len(vacio):
        log(f"\n  {len(vacio)} con el periodo PRESENTE y sin importes en la fuente: "
            f"{sorted(vacio['ticker'])}")
        log("  No se arreglan con un refresh ni esperando (medido: siguen igual 2 "
            "meses despues). Requieren otra fuente.")

    if args.csv:
        destino = os.path.join(ROOT, "reportes", f"status_balances_{hoy:%Y%m%d}.csv")
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        cmp_.to_csv(destino, index=False)
        log(f"\n  detalle por ticker: {destino}")

    return 2 if len(rec) or len(faltan) else 0


if __name__ == "__main__":
    sys.exit(main())
