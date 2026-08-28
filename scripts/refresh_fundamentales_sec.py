"""
refresh_fundamentales_sec.py
Motor de la fuente SEC XBRL: descarga incremental -> normaliza -> UPSERT.

Fuente PARALELA a yahooquery. Las dos conviven; ningun consumidor actual lee
las tablas fundamentales_sec_*. LOCAL-only.

Este script es el UNICO lugar de la fuente SEC que toca la base y el universo.
El normalizador (src/utils/sec_xbrl.py) y el cliente (src/data/sec/client.py)
no saben que existe una DB -- ver la regla de dependencia en
src/data/sec/__init__.py.

Alcance: los tickers de region USA. Los ~53 ADR extranjeros presentan 20-F
ANUAL y no tienen XBRL trimestral: quedan fuera por construccion, no por
configuracion. Ver docs/fuentes_fundamentales.md seccion 4.

Incremental: se consulta `submissions` (~164 KB) y solo se baja companyfacts
(~4 MB) si el accession del ultimo 10-Q/10-K cambio. Sin balances nuevos, un
refresh completo mueve ~24 MB en vez de ~522 MB.

Uso:
    python scripts/refresh_fundamentales_sec.py
    python scripts/refresh_fundamentales_sec.py --dry-run
    python scripts/refresh_fundamentales_sec.py --tickers AAPL,JPM
    python scripts/refresh_fundamentales_sec.py --solo-normalizar   # usa cache
    python scripts/refresh_fundamentales_sec.py --forzar            # re-baja todo
"""

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import psycopg2
import psycopg2.extras

from scripts.oneshot.create_fundamentales_tables import _parse_env_file
from src.data.sec import client
from src.utils.sec_xbrl import CONCEPTOS, normalizar

SEP = "=" * 64
CACHE_DEFECTO = os.path.join(ROOT, "data", "sec_cache")

CONCEPTOS_ORD = list(CONCEPTOS)
COLS = (["ticker", "cik", "period_end", "fiscal_year", "fiscal_quarter"]
        + CONCEPTOS_ORD + ["origen", "filed_max"])
PK = ["ticker", "period_end"]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _conn(env):
    return psycopg2.connect(
        host=env.get("DB_HOST", "localhost"), port=int(env.get("DB_PORT", 5432)),
        dbname=env.get("DB_NAME", "activos_ml"), user=env.get("DB_USER", "postgres"),
        password=env.get("DB_PASSWORD", ""))


# ------------------------------------------------------------- universo --
def user_agent(env):
    """
    SEC EXIGE un User-Agent que identifique a un responsable, con mail de
    contacto; sin eso devuelve 403 en todos los endpoints. Se toma de .env o
    del ambiente. Se falla temprano y con instrucciones en vez de dejar que
    reviente a mitad de una corrida.
    """
    ua = env.get("SEC_USER_AGENT") or os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit(
            "\nFALTA SEC_USER_AGENT.\n"
            "SEC rechaza (403) los pedidos sin un User-Agent identificable.\n"
            "Agregar al .env del proyecto:\n"
            "    SEC_USER_AGENT=tu-mail@dominio.com IndicadoresDeStockML\n")
    return ua


def universo_usa(env, tickers=None):
    """
    [(ticker, cik)] de los tickers USA. El CIK sale del listado oficial de SEC.

    Se lee ticker_pais (no `activos`) porque el corte que importa aca es la
    REGION: SEC solo tiene XBRL trimestral de los filers estadounidenses.
    """
    with _conn(env) as cx:
        with cx.cursor() as cur:
            if tickers:
                cur.execute("SELECT ticker FROM ticker_pais WHERE ticker = ANY(%s)",
                            (tickers,))
            else:
                cur.execute("SELECT ticker FROM ticker_pais WHERE region='USA' "
                            "ORDER BY ticker")
            de_db = [r[0] for r in cur.fetchall()]

    s = client.sesion(user_agent(env))
    client.verificar_acceso(s)
    mapa = client.mapa_cik(s)
    pares = [(t, mapa[t]) for t in de_db if t in mapa]
    faltan = [t for t in de_db if t not in mapa]
    if faltan:
        log(f"  sin CIK en el listado de SEC ({len(faltan)}): {', '.join(faltan)}")
    return pares, s


def accn_conocidos(env):
    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute("SELECT ticker, ultimo_accn FROM fundamentales_sec_ingesta")
            return {t: a for t, a in cur.fetchall()}


# ------------------------------------------------------ armado de filas --
def filas_de(ticker, cik, resultado):
    """Periodos normalizados -> filas listas para el UPSERT."""
    filas = []
    for p in resultado["periodos"]:
        origen = {}
        for c in CONCEPTOS_ORD:
            if c in p:
                origen[c] = {"tag": p.get(c + "__tag"),
                             "derivado": bool(p.get(c + "__derivado"))}
        if not origen:
            continue                     # periodo sin ningun concepto: se omite
        fila = {"ticker": ticker, "cik": cik, "period_end": p["period_end"],
                "fiscal_year": p.get("fiscal_year"),
                "fiscal_quarter": p.get("fiscal_quarter"),
                "origen": json.dumps(origen), "filed_max": None}
        for c in CONCEPTOS_ORD:
            fila[c] = p.get(c)
        filas.append(fila)
    return filas


def filas_avisos(ticker, resultado):
    return [{"ticker": ticker, "tipo": a["tipo"], "concepto": a.get("concepto", ""),
             "period_end": a.get("period_end"), "detalle": a.get("detalle", ""),
             "tags": json.dumps(a.get("tags")) if a.get("tags") else None}
            for a in resultado["avisos"]]


# ------------------------------------------------------------- escritura --
def upsert_serie(env, filas):
    if not filas:
        return 0
    ph = ", ".join(f"%({c})s" for c in COLS)
    upd = [c for c in COLS if c not in PK]
    setc = ", ".join(f"{c}=EXCLUDED.{c}" for c in upd) + ", computed_at=NOW()"
    sql = (f"INSERT INTO fundamentales_sec_q ({', '.join(COLS)}) VALUES ({ph}) "
           f"ON CONFLICT (ticker, period_end) DO UPDATE SET {setc}")
    with _conn(env) as cx:
        with cx.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, filas, page_size=500)
        cx.commit()
    return len(filas)


def reemplazar_avisos(env, ticker, filas):
    """
    Los avisos se REGENERAN enteros por ticker: son una foto del ultimo
    normalizado, no un historico. Si un aviso desaparecio (la empresa dejo de
    cambiar de tag), tiene que desaparecer de la tabla.
    """
    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute("DELETE FROM fundamentales_sec_avisos WHERE ticker=%s",
                        (ticker,))
            if filas:
                psycopg2.extras.execute_batch(cur, (
                    "INSERT INTO fundamentales_sec_avisos "
                    "(ticker, tipo, concepto, period_end, detalle, tags) VALUES "
                    "(%(ticker)s, %(tipo)s, %(concepto)s, %(period_end)s, "
                    "%(detalle)s, %(tags)s)"), filas, page_size=500)
        cx.commit()
    return len(filas)


def upsert_ingesta(env, fila):
    cols = ["ticker", "cik", "ultimo_accn", "ultimo_form", "ultimo_filed",
            "periodos", "periodo_min", "periodo_max", "avisos",
            "bytes_descargados", "error", "fetched_at"]
    ph = ", ".join(f"%({c})s" for c in cols)
    setc = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "ticker")
    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute(
                f"INSERT INTO fundamentales_sec_ingesta ({', '.join(cols)}) "
                f"VALUES ({ph}) ON CONFLICT (ticker) DO UPDATE SET {setc}, "
                f"updated_at=NOW()", fila)
        cx.commit()


# ------------------------------------------------------------------ main --
def main():
    p = argparse.ArgumentParser(description="Refresh de la fuente SEC XBRL (LOCAL)")
    p.add_argument("--tickers", help="CSV de tickers; default: todos los USA")
    p.add_argument("--cache", default=CACHE_DEFECTO, help="Directorio del cache")
    p.add_argument("--desde", help="Recorta los periodos a >= esta fecha")
    p.add_argument("--forzar", action="store_true",
                   help="Re-baja companyfacts aunque no haya filing nuevo")
    p.add_argument("--solo-normalizar", action="store_true",
                   help="No sale a la red: usa el cache tal como esta")
    p.add_argument("--dry-run", action="store_true", help="No escribe en la DB")
    args = p.parse_args()

    env = _parse_env_file(os.path.join(ROOT, ".env"))
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    print()
    print(SEP)
    print(f"  REFRESH FUNDAMENTALES SEC XBRL"
          f"{'  [DRY-RUN]' if args.dry_run else ''}"
          f"{'  [SOLO CACHE]' if args.solo_normalizar else ''}")
    print(SEP)
    print()
    log(f"cache: {args.cache}")

    pares, s = universo_usa(env, tickers)
    log(f"tickers a procesar: {len(pares)}")

    if args.solo_normalizar:
        estados = {t: {"estado": "sin_cambios", "accn": None, "form": None,
                       "filed": None, "bytes": 0, "error": None} for t, _ in pares}
    else:
        log("sincronizando cache (submissions -> companyfacts si cambio)...")
        estados = client.sincronizar(pares, args.cache,
                                     accn_conocidos=accn_conocidos(env),
                                     forzar=args.forzar, s=s)
        resumen = {}
        for v in estados.values():
            resumen[v["estado"]] = resumen.get(v["estado"], 0) + 1
        mb = sum(v["bytes"] for v in estados.values()) / 1e6
        log(f"  {resumen}  |  descargado: {mb:.1f} MB")

    n_filas = n_avisos = n_ok = n_sin = 0
    for ticker, cik in pares:
        datos = client.leer_cache(args.cache, ticker)
        if datos is None:
            n_sin += 1
            continue
        r = normalizar(datos, desde=args.desde)
        filas = filas_de(ticker, cik, r)
        avisos = filas_avisos(ticker, r)
        est = estados.get(ticker, {})
        if not args.dry_run:
            n_filas += upsert_serie(env, filas)
            n_avisos += reemplazar_avisos(env, ticker, avisos)
            upsert_ingesta(env, {
                "ticker": ticker, "cik": cik,
                "ultimo_accn": est.get("accn"), "ultimo_form": est.get("form"),
                "ultimo_filed": est.get("filed"), "periodos": len(filas),
                "periodo_min": filas[0]["period_end"] if filas else None,
                "periodo_max": filas[-1]["period_end"] if filas else None,
                "avisos": len(avisos), "bytes_descargados": est.get("bytes", 0),
                "error": est.get("error"),
                "fetched_at": datetime.now() if est.get("estado") == "actualizado" else None,
            })
        else:
            n_filas += len(filas)
            n_avisos += len(avisos)
        n_ok += 1

    print()
    print(SEP)
    print(f"  OK  |  tickers: {n_ok}  |  filas: {n_filas}  |  avisos: {n_avisos}"
          f"{'  |  sin cache: ' + str(n_sin) if n_sin else ''}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
