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
from src.data.sec import tags_curados
from src.utils.sec_acciones import serie_acciones
from src.utils.sec_xbrl import CONCEPTOS, normalizar

SEP = "=" * 64
CACHE_DEFECTO = os.path.join(ROOT, "data", "sec_cache")

# VENTANA DE RETENCION. SEC tiene ~19 anios, pero el techo real de cualquier
# analisis contra precio es `precios_diarios`, que arranca en 2020-01-02 (y en
# 2024 para 60 de los 147 tickers). Guardar 2007 en adelante seria peso muerto.
#
# El corte NO es 2020 sino 2018: el TTM necesita PISTA DE DESPEGUE. El primer
# dia con precio (2/1/2020) necesita el TTM del ultimo trimestre publico en esa
# fecha -- para un calendario normal, Q3-2019 -- y ese TTM se extiende hasta
# Q4-2018. Cortar en 2020 dejaria el primer anio de la serie sin TTM valido.
#
# El cache en disco conserva TODO, asi que la historia previa es re-derivable
# en cualquier momento con --desde.
DESDE_DEFECTO = "2018-01-01"

CONCEPTOS_ORD = list(CONCEPTOS)
COLS = (["ticker", "cik", "period_end", "fiscal_year", "fiscal_quarter"]
        + CONCEPTOS_ORD + ["origen", "filed_primero", "filed_ultimo"])
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
                "origen": json.dumps(origen),
                "filed_primero": p.get("filed_primero"),
                "filed_ultimo": p.get("filed_ultimo")}
        for c in CONCEPTOS_ORD:
            fila[c] = p.get(c)
        filas.append(fila)
    return filas


def filas_acciones(ticker, serie):
    """
    Serie de acciones de PORTADA -> filas. Grano distinto al de la serie
    trimestral: una fila por FILING, no por trimestre. Ver
    src/utils/sec_acciones.py para por que no puede vivir en la misma tabla.
    """
    return [{"ticker": ticker, "fecha": p["fecha"], "shares": p["shares"],
             "accn": p.get("accn"), "filed": p.get("filed"),
             "form": p.get("form"), "fuente": p.get("fuente")}
            for p in serie]


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


def reemplazar_acciones(env, ticker, filas):
    """
    La serie se REGENERA entera por ticker. Es una proyeccion pura del cache,
    no un historico acumulado: si un punto deja de calificar (una portada
    fechada despues de su propio filing, un valor con error de unidad), tiene
    que DESAPARECER. Con UPSERT sobreviviria para siempre -- fue exactamente lo
    que paso con la portada 2027-07-17 de AAL en la primera corrida.
    """
    cols = ["ticker", "fecha", "shares", "accn", "filed", "form", "fuente"]
    ph = ", ".join(f"%({c})s" for c in cols)
    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute("DELETE FROM fundamentales_sec_acciones WHERE ticker=%s",
                        (ticker,))
            if filas:
                psycopg2.extras.execute_batch(cur, (
                    f"INSERT INTO fundamentales_sec_acciones "
                    f"({', '.join(cols)}) VALUES ({ph})"), filas, page_size=500)
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


def purgar_fuera_de_ventana(env, tickers, desde):
    """
    Borra los periodos anteriores a la ventana de retencion.

    Scopeado a los tickers procesados: una corrida con --tickers no toca al
    resto. Es seguro porque el cache en disco conserva la historia completa --
    volver a traerla es correr con --desde mas viejo, sin salir a la red.
    """
    if not tickers:
        return 0
    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute("DELETE FROM fundamentales_sec_q "
                        "WHERE ticker = ANY(%s) AND period_end < %s",
                        (list(tickers), desde))
            n = cur.rowcount
            cur.execute("DELETE FROM fundamentales_sec_acciones "
                        "WHERE ticker = ANY(%s) AND fecha < %s",
                        (list(tickers), desde))
            n += cur.rowcount
        cx.commit()
    return n


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
    p.add_argument("--desde", default=DESDE_DEFECTO,
                   help=f"Recorta los periodos a >= esta fecha (default: "
                        f"{DESDE_DEFECTO}; ver VENTANA DE RETENCION)")
    p.add_argument("--sin-purga", action="store_true",
                   help="No borra los periodos anteriores a --desde")
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

    log(f"ventana de retencion: periodos >= {args.desde}")
    n_filas = n_avisos = n_ok = n_sin = n_acciones = 0
    procesados = []
    for ticker, cik in pares:
        datos = client.leer_cache(args.cache, ticker)
        if datos is None:
            n_sin += 1
            continue
        # El mapeo curado resuelve la ambiguedad de revenue en los 23 tickers
        # donde dos tags XBRL valen cosas distintas. Para los otros 124
        # devuelve {} y el normalizador se comporta igual que antes.
        r = normalizar(datos, desde=args.desde,
                       tags_curados=tags_curados.para(ticker))
        filas = filas_de(ticker, cik, r)
        avisos = filas_avisos(ticker, r)
        acciones = filas_acciones(ticker, serie_acciones(datos, desde=args.desde))
        est = estados.get(ticker, {})
        if not args.dry_run:
            n_filas += upsert_serie(env, filas)
            n_acciones += reemplazar_acciones(env, ticker, acciones)
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
            n_acciones += len(acciones)
            n_avisos += len(avisos)
        procesados.append(ticker)
        n_ok += 1

    n_purga = 0
    if not args.dry_run and not args.sin_purga:
        n_purga = purgar_fuera_de_ventana(env, procesados, args.desde)
        if n_purga:
            log(f"purgados {n_purga} periodos anteriores a {args.desde}")

    print()
    print(SEP)
    print(f"  OK  |  tickers: {n_ok}  |  filas: {n_filas}"
          f"  |  acciones: {n_acciones}  |  avisos: {n_avisos}"
          f"{'  |  purgados: ' + str(n_purga) if n_purga else ''}"
          f"{'  |  sin cache: ' + str(n_sin) if n_sin else ''}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
