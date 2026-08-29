"""
compute_sec_multiplos.py
Motor de fundamentales_sec_multiplos_d: serie DIARIA de multiplos de valuacion
sobre la fuente SEC XBRL, con percentil "caro vs si misma".

Capa DERIVADA y recomputable: funcion de fundamentales_sec_q +
fundamentales_sec_acciones + precios_diarios, sin salir a la red. Se puede
borrar la tabla y reconstruirla. LOCAL-only.

Las tres decisiones que gobiernan el calculo
--------------------------------------------
1. TODO SE CALCULA DESDE AGREGADOS, nunca desde magnitudes por accion.
       PER = market_cap / net_income_ttm      (no close / eps_ttm)
   SEC re-expresa retroactivamente lo "por accion" cuando hay un split y
   `precios_diarios` no se re-ajusta hacia atras. Cruzar un eps_ttm de hoy con
   un precio de 2023 da un PER partido por el factor del split, en silencio, en
   los 25 tickers del universo que tuvieron split o evento de capital. Los
   agregados son invariantes: un 10:1 no cambia el resultado ni las ventas.
   El unico termino por accion que queda es el precio, y se multiplica por el
   conteo de portada VIGENTE ESE DIA, que esta en su misma base.
   Detalle y evidencia: src/utils/sec_acciones.py.

2. EL ANCLA ES `filed_primero`, NO `period_end`. Cada rueda usa el TTM que
   estaba PUBLICO ese dia. Un trimestre cerrado el 30/9 no estuvo disponible el
   30/9 (AAPL publica a los ~34 dias, JPM entre 31 y 44); anclarlo por
   period_end adelantaria cada trimestre mas de un mes y meteria lookahead en
   toda la serie. `lag_dias` deja la distancia a la vista.

3. EL PERCENTIL ES TRAILING. La ventana mira solo hacia atras, asi que el valor
   de una rueda no depende de lo que paso despues. Sirve como senal sin
   trucarse a si misma.

CAVEAT que no resuelve esta tabla, y conviene tenerlo escrito: el percentil
compara el multiplo de hoy contra su propia historia, pero parte de ese rango
viene del REGIMEN DE TASAS, no de la empresa. Un PER de 20 en 2021 con la tasa
en cero no significa lo mismo que un PER de 20 hoy. El percentil dice donde
esta parado el multiplo, no si eso esta justificado.

Uso:
    python scripts/compute_sec_multiplos.py
    python scripts/compute_sec_multiplos.py --tickers AAPL,JPM --verbose
    python scripts/compute_sec_multiplos.py --desde 2022-01-01 --rebuild
    python scripts/compute_sec_multiplos.py --dry-run
"""

import argparse
import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import psycopg2
import psycopg2.extras

from scripts.oneshot.create_fundamentales_tables import _parse_env_file
from src.utils import acciones_series as A
from src.utils import fundamentales_ttm as T

SEP = "=" * 64
TABLA = "fundamentales_sec_multiplos_d"

# Arranque objetivo. El techo real lo pone `acciones_circulacion`: 101 tickers
# llegan a 2021, 83 a 2022 y 16 a 2023 (ver acciones_circulacion_validacion).
DESDE_DEFECTO = "2021-01-01"

# Ventana del percentil "vs si misma": ~3 anios de ruedas. Es un LOOKBACK FIJO
# y no toda la historia disponible a proposito -- con ventana expansiva, una
# lectura de 2026 se compara contra una muestra el doble de larga que una de
# 2022 y los percentiles dejan de ser comparables entre si.
VENTANA_PCT = 756
MIN_OBS_PCT = 250

METRICAS_PCT = [("pe_ratio", "pe_pct"), ("pb_ratio", "pb_pct"),
                ("ps_ratio", "ps_pct"), ("ev_ebitda", "ev_ebitda_pct")]

COLS = ["ticker", "fecha", "close", "period_end", "filed_primero", "lag_dias",
        "n_periodos", "revenue_ttm", "net_income_ttm", "ebitda_ttm", "fcf_ttm",
        "equity", "net_debt", "shares", "shares_fuente", "market_cap",
        "enterprise_value", "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
        "fcf_yield", "pe_pct", "pb_pct", "ps_pct", "ev_ebitda_pct", "n_obs_pct",
        "shares_dias"]
PK = ["ticker", "fecha"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _conn(env):
    return psycopg2.connect(
        host=env.get("DB_HOST", "localhost"), port=int(env.get("DB_PORT", 5432)),
        dbname=env.get("DB_NAME", "activos_ml"), user=env.get("DB_USER", "postgres"),
        password=env.get("DB_PASSWORD", ""))


# ------------------------------------------------------------- lectura --
def tickers_disponibles(env, tickers=None):
    with _conn(env) as cx:
        with cx.cursor() as cur:
            if tickers:
                cur.execute("SELECT DISTINCT ticker FROM fundamentales_sec_q "
                            "WHERE ticker = ANY(%s) ORDER BY ticker", (tickers,))
            else:
                cur.execute("SELECT DISTINCT ticker FROM fundamentales_sec_q "
                            "ORDER BY ticker")
            return [r[0] for r in cur.fetchall()]


def leer_trimestres(cur, ticker):
    cur.execute("SELECT * FROM fundamentales_sec_q WHERE ticker=%s "
                "ORDER BY period_end", (ticker,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def leer_acciones(cur, ticker):
    """
    Serie de `acciones_circulacion`: base de split ACTUAL, validada ticker por
    ticker contra la serie SEC (ver refresh_acciones_circulacion.py). Por eso
    aca ya no hace falta ninguna guarda de base: el problema se resolvio aguas
    arriba, en la fuente, en vez de detectarse aguas abajo con heuristicas.
    """
    cur.execute("SELECT fecha, shares, fuente FROM acciones_circulacion "
                "WHERE ticker=%s ORDER BY fecha", (ticker,))
    return [{"fecha": f.isoformat(), "shares": float(s), "fuente": u}
            for f, s, u in cur.fetchall()]


def leer_precios(cur, ticker, desde):
    cur.execute("SELECT fecha, close FROM precios_diarios "
                "WHERE ticker=%s AND fecha >= %s AND close IS NOT NULL "
                "ORDER BY fecha", (ticker, desde))
    return [(f, float(c)) for f, c in cur.fetchall()]


# ------------------------------------------------------------- calculo --
def _div(num, den, exigir_positivo=True):
    """None si el denominador falta, es cero, o es negativo cuando importa."""
    if num is None or den is None:
        return None
    if den == 0 or (exigir_positivo and den <= 0):
        return None
    return num / den


def multiplos(close, shares, ttm):
    """
    Multiplos de una rueda a partir del cierre, el conteo point-in-time y la
    fila TTM anclada. Todos los denominadores son AGREGADOS.

    Un multiplo con denominador negativo no se emite: un PER con resultado
    negativo no es "barato", es una categoria distinta, y dejarlo entrar
    ensuciaria el percentil con numeros que no viven en la misma escala.
    """
    vacio = {"market_cap": None, "enterprise_value": None, "pe_ratio": None,
             "pb_ratio": None, "ps_ratio": None, "ev_ebitda": None,
             "fcf_yield": None}
    if not close or not shares or close <= 0 or shares <= 0:
        return vacio
    mc = close * shares
    nd = ttm.get("net_debt")
    ev = (mc + nd) if nd is not None else None
    return {
        "market_cap": mc,
        "enterprise_value": ev,
        "pe_ratio": _div(mc, ttm.get("net_income_ttm")),
        "pb_ratio": _div(mc, ttm.get("equity")),
        "ps_ratio": _div(mc, ttm.get("revenue_ttm")),
        "ev_ebitda": _div(ev, ttm.get("ebitda_ttm")),
        # El FCF yield SI admite numerador negativo: quemar caja es informacion,
        # y la escala no se rompe porque el denominador (market cap) es > 0.
        "fcf_yield": _div(ttm.get("fcf_ttm"), mc, exigir_positivo=False),
    }


def serie_diaria(precios, filas_ttm, serie_acc):
    """
    Una fila por rueda. Devuelve [] si el ticker no tiene con que calcular.
    """
    fechas = [f for f, _ in precios]
    anclas = dict(T.recorrer_asof(filas_ttm, [f.isoformat() for f in fechas]))
    conteos = dict(A.recorrer_asof(serie_acc, [f.isoformat() for f in fechas]))

    out = []
    for fecha, close in precios:
        iso = fecha.isoformat()
        ttm = anclas.get(iso) or {}
        acc = conteos.get(iso) or {}
        shares = acc.get("shares")
        fila = {"fecha": fecha, "close": close,
                "period_end": ttm.get("period_end"),
                "filed_primero": ttm.get("filed_primero"),
                "lag_dias": None, "n_periodos": ttm.get("n_periodos"),
                "revenue_ttm": ttm.get("revenue_ttm"),
                "net_income_ttm": ttm.get("net_income_ttm"),
                "ebitda_ttm": ttm.get("ebitda_ttm"),
                "fcf_ttm": ttm.get("fcf_ttm"),
                "equity": ttm.get("equity"), "net_debt": ttm.get("net_debt"),
                "shares": shares, "shares_fuente": acc.get("fuente"),
                # Antiguedad del conteo. Entre puntos se mantiene (ESCALON, no
                # se interpola), y el error de ese escalon crece con los dias:
                # medido mediana 0,24% pero p99 11,35%. Sin esta columna el
                # riesgo es invisible en la fila.
                "shares_dias": None}
        if acc.get("fecha"):
            fila["shares_dias"] = (fecha - _fecha(acc["fecha"])).days
        if ttm.get("filed_primero"):
            fila["lag_dias"] = (fecha - _fecha(ttm["filed_primero"])).days
        fila.update(multiplos(close, shares, ttm))
        out.append(fila)
    return out


def _fecha(iso):
    from datetime import date
    p = str(iso)[:10].split("-")
    return date(int(p[0]), int(p[1]), int(p[2]))


def percentiles(filas, ventana=VENTANA_PCT, min_obs=MIN_OBS_PCT, estricto=True):
    """
    Percentil TRAILING de cada multiplo dentro de su propia historia reciente.
    Incluye la observacion del dia (es "donde cae hoy en su rango"), pero NADA
    posterior.

    Las ruedas sin multiplo no cuentan como observacion: rolling().rank() sobre
    NaN no los computa, asi que la ventana son las ultimas `ventana` ruedas y
    `n_obs` cuenta cuantas de ellas tenian dato.

    `estricto` (default) exige ademas que la ventana este llena EN TIEMPO: no
    alcanza con tener min_obs observaciones, tiene que haber `ventana` ruedas de
    historia detras. Son dos cosas distintas y la diferencia importa: con 250
    dias de historia el percentil sale de un solo regimen de tasas y de una sola
    fase del ciclo del ticker, y despues se lee como si fuera "su rango
    historico". Ese es el riesgo de proyectar hacia adelante desde una
    distribucion sin forma -- preferimos un NULL honesto a un numero que invita
    a decidir. Con --percentil-permisivo se afloja a min_obs.
    """
    if not filas:
        return filas
    df = pd.DataFrame(filas)
    historia = pd.Series(range(1, len(df) + 1), index=df.index)
    ventana_llena = historia >= ventana
    for origen, destino in METRICAS_PCT:
        s = pd.to_numeric(df[origen], errors="coerce")
        r = s.rolling(ventana, min_periods=min_obs).rank(pct=True)
        if estricto:
            r = r.where(ventana_llena)
        df[destino] = r
    n_obs = pd.to_numeric(df["pe_ratio"], errors="coerce") \
              .rolling(ventana, min_periods=1).count()
    df["n_obs_pct"] = n_obs
    return df.where(pd.notnull(df), None).to_dict("records")


# ----------------------------------------------------------- escritura --
def upsert(env, ticker, filas, rebuild):
    if rebuild:
        with _conn(env) as cx:
            with cx.cursor() as cur:
                cur.execute(f"DELETE FROM {TABLA} WHERE ticker=%s", (ticker,))
            cx.commit()
    if not filas:
        return 0
    ph = ", ".join(f"%({c})s" for c in COLS)
    upd = [c for c in COLS if c not in PK]
    setc = ", ".join(f"{c}=EXCLUDED.{c}" for c in upd) + ", computed_at=NOW()"
    sql = (f"INSERT INTO {TABLA} ({', '.join(COLS)}) VALUES ({ph}) "
           f"ON CONFLICT (ticker, fecha) DO UPDATE SET {setc}")
    with _conn(env) as cx:
        with cx.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, filas, page_size=1000)
        cx.commit()
    return len(filas)


def _limpiar(fila, ticker):
    """Normaliza tipos para psycopg2: NaN de pandas -> None."""
    out = {"ticker": ticker}
    for c in COLS:
        if c == "ticker":
            continue
        v = fila.get(c)
        if v is not None and isinstance(v, float) and v != v:
            v = None
        out[c] = v
    return out


# ---------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(
        description="Computa fundamentales_sec_multiplos_d (LOCAL-only)")
    p.add_argument("--tickers", help="CSV; default: todos los de la fuente SEC")
    p.add_argument("--desde", default=DESDE_DEFECTO,
                   help=f"Primera rueda a calcular (default {DESDE_DEFECTO})")
    p.add_argument("--ventana-pct", type=int, default=VENTANA_PCT,
                   help=f"Ruedas de la ventana del percentil (default {VENTANA_PCT})")
    p.add_argument("--min-obs", type=int, default=MIN_OBS_PCT,
                   help=f"Minimo de observaciones para emitir percentil "
                        f"(default {MIN_OBS_PCT})")
    p.add_argument("--rebuild", action="store_true",
                   help="Borra la serie del ticker antes de reescribirla")
    p.add_argument("--percentil-permisivo", action="store_true",
                   help="Emite percentil sin exigir la ventana llena. Ver la "
                        "advertencia en percentiles()")
    p.add_argument("--dry-run", action="store_true", help="No escribe en la DB")
    p.add_argument("--verbose", action="store_true", help="Detalle por ticker")
    args = p.parse_args()

    env = _parse_env_file(os.path.join(ROOT, ".env"))
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    print()
    print(SEP)
    print(f"  COMPUTE MULTIPLOS SEC{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    lista = tickers_disponibles(env, tickers)
    log(f"tickers: {len(lista)}  |  desde: {args.desde}  |  "
        f"ventana percentil: {args.ventana_pct} ruedas (min {args.min_obs})")

    n_filas = n_ok = 0
    sin_acciones, sin_ttm, sin_precios = [], [], []
    with _conn(env) as cx:
        with cx.cursor() as cur:
            for ticker in lista:
                precios = leer_precios(cur, ticker, args.desde)
                if not precios:
                    sin_precios.append(ticker)
                    continue
                acc = leer_acciones(cur, ticker)
                if not acc:
                    sin_acciones.append(ticker)
                    continue
                ttm = T.enriquecer(T.serie_ttm(leer_trimestres(cur, ticker)))
                if not any(f["ventana_ok"] for f in ttm):
                    sin_ttm.append(ticker)
                    continue
                filas = serie_diaria(precios, ttm, acc)
                filas = percentiles(filas, args.ventana_pct, args.min_obs,
                                    estricto=not args.percentil_permisivo)
                filas = [_limpiar(f, ticker) for f in filas]
                if not args.dry_run:
                    n_filas += upsert(env, ticker, filas, args.rebuild)
                else:
                    n_filas += len(filas)
                n_ok += 1
                if args.verbose:
                    con_pe = sum(1 for f in filas if f["pe_ratio"] is not None)
                    log(f"  {ticker:<6} {len(filas):>5} ruedas  "
                        f"PER en {con_pe:>5}  ({filas[0]['fecha']} -> "
                        f"{filas[-1]['fecha']})")

    print()
    print(SEP)
    print(f"  OK  |  tickers: {n_ok}  |  filas: {n_filas}")
    for etiqueta, xs in (("sin serie de acciones", sin_acciones),
                         ("sin TTM completo", sin_ttm),
                         ("sin precios", sin_precios)):
        if xs:
            print(f"  {etiqueta}: {len(xs)}  ->  {', '.join(xs)}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
