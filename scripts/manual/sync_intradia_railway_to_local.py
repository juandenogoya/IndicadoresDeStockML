"""
sync_intradia_railway_to_local.py -- baja precios_intradia_30m de Railway a LOCAL
y purga Railway dejando las ultimas ~30 ruedas.

Mismo esquema que el crudo de opciones (Plan C): Oracle escribe en Railway, que
es un BUFFER DE TRANSITO; LOCAL es la fuente de verdad historica. Separado de
sync_opciones_railway_to_local.bat a proposito: la linea intradiaria esta en
prueba en la rama main-y9tr06 y no toca nada de lo que ya funciona
(docs/intradia.md). Durante la prueba se corre a mano.

Sync INCREMENTAL e idempotente: trae de Railway las ruedas desde la ultima que
ya tiene local menos SOLAPE_DIAS (un corte que se reconstruyo tarde queda
cubierto) y hace upsert por la PK (ticker, corte, inicio_barra).

PURGA, mismo principio que retencion_opciones_railway.py: se borra UNICAMENTE
lo que esta PROBADAMENTE replicado en local (COUNT por rueda, local >= Railway).
Sin VACUUM FULL: en un buffer rotativo las paginas libres las reusa el INSERT del
dia siguiente, y reescribir la tabla cada vez solo genera WAL y page cache, que
es lo que factura Railway (docs/consumo_railway.md sec. 6). Y si el sync falla,
la purga no corre.

Uso:
    python scripts/manual/sync_intradia_railway_to_local.py
    python scripts/manual/sync_intradia_railway_to_local.py --dry-run
    python scripts/manual/sync_intradia_railway_to_local.py --keep-days 40
    python scripts/manual/sync_intradia_railway_to_local.py --sin-purga

Salida: 0 OK / 2 sync OK pero la purga dejo ruedas sin borrar o fallo / 1 error.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, text

from src.utils.intradia import COLUMNAS, SQL_UPSERT, TABLA

KEEP_DAYS_DEFAULT = 30   # ruedas distintas que Railway conserva
SOLAPE_DIAS = 5


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Conexiones (patron dual de retencion_opciones_railway) ────────────────────

def _leer_database_url_local() -> str:
    """Lee DATABASE_URL de .env.local SIN cargarla en os.environ."""
    env_path = os.path.join(ROOT, ".env.local")
    if not os.path.exists(env_path):
        raise RuntimeError(".env.local no encontrado (DATABASE_URL de Railway).")
    with open(env_path, "r", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if linea.startswith("DATABASE_URL"):
                return linea.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DATABASE_URL no esta definida en .env.local.")


def _engine_railway():
    url = _leer_database_url_local().replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url, connect_args={"sslmode": "require"}, pool_pre_ping=True)


def _engine_local():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    os.environ.pop("DATABASE_URL", None)
    from src.data.database import get_engine
    return get_engine()


def _existe_tabla(eng) -> bool:
    with eng.connect() as c:
        return bool(c.execute(text("SELECT to_regclass(:t)"), {"t": TABLA}).scalar())


def _conteos(eng, desde=None) -> dict:
    """{fecha: filas} por rueda."""
    sql = f"SELECT fecha, COUNT(*) FROM {TABLA}"
    params = {}
    if desde is not None:
        sql += " WHERE fecha >= :d"
        params["d"] = desde
    with eng.connect() as c:
        return dict(c.execute(text(sql + " GROUP BY fecha"), params).fetchall())


# ── Sync ──────────────────────────────────────────────────────────────────────

def sincronizar(eng_rail, eng_loc, dry_run: bool) -> int:
    with eng_loc.connect() as c:
        max_local = c.execute(text(f"SELECT MAX(fecha) FROM {TABLA}")).scalar()
    desde = (max_local - timedelta(days=SOLAPE_DIAS)) if max_local else None
    log(f"  Ultima rueda en local: {max_local or '(vacia)'} -> trae desde {desde or 'el principio'}")

    sql = f"SELECT {', '.join(COLUMNAS)} FROM {TABLA}"
    params = {}
    if desde is not None:
        sql += " WHERE fecha >= :d"
        params["d"] = desde
    with eng_rail.connect() as c:
        filas = [tuple(r) for r in c.execute(text(sql + " ORDER BY fecha, corte, ticker"),
                                             params).fetchall()]
    log(f"  Filas en Railway a sincronizar: {len(filas):,}")
    if dry_run or not filas:
        return len(filas)

    import psycopg2.extras
    raw = eng_loc.raw_connection()
    try:
        with raw.cursor() as cur:
            psycopg2.extras.execute_values(cur, SQL_UPSERT, filas, page_size=1000)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()

    # Verificacion: cada rueda traida tiene en local al menos lo de Railway.
    rail = _conteos(eng_rail, desde)
    loc = _conteos(eng_loc, desde)
    faltan = {f: (n, loc.get(f, 0)) for f, n in rail.items() if loc.get(f, 0) < n}
    if faltan:
        raise RuntimeError(f"verificacion fallida, local con menos filas que Railway: {faltan}")
    log(f"  Upsert OK y verificado: {len(rail)} ruedas.")
    return len(filas)


# ── Purga ─────────────────────────────────────────────────────────────────────

def purgar(eng_rail, eng_loc, keep_days: int, dry_run: bool) -> int:
    rail = _conteos(eng_rail)
    fechas = sorted(rail, reverse=True)
    candidatas = fechas[keep_days:]
    if not candidatas:
        log(f"  Purga: Railway tiene {len(fechas)} ruedas (<= {keep_days}). Nada que purgar.")
        return 0

    loc = _conteos(eng_loc, min(candidatas))
    seguras = [f for f in candidatas if loc.get(f, 0) >= rail[f]]
    retenidas = [f for f in candidatas if f not in seguras]
    log(f"  Purga: {len(candidatas)} ruedas fuera de la retencion de {keep_days} "
        f"({candidatas[-1]} -> {candidatas[0]}); {len(seguras)} verificadas en local.")
    for f in retenidas:
        log(f"  [WARN] {f} NO se borra: Railway {rail[f]} filas, local {loc.get(f, 0)}.")
    if dry_run or not seguras:
        return 2 if retenidas else 0

    with eng_rail.begin() as c:
        borradas = c.execute(text(f"DELETE FROM {TABLA} WHERE fecha = ANY(:fs)"),
                             {"fs": seguras}).rowcount
    log(f"  Purga OK: {borradas:,} filas borradas de Railway.")
    return 2 if retenidas else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync precios_intradia_30m Railway -> local + purga")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-days", type=int, default=KEEP_DAYS_DEFAULT)
    ap.add_argument("--sin-purga", action="store_true")
    args = ap.parse_args()

    log("=" * 60)
    log(f"  SYNC INTRADIA  Railway -> Local  |  {'DRY RUN' if args.dry_run else 'REAL'}")
    log("=" * 60)
    try:
        eng_rail = _engine_railway()
        eng_loc = _engine_local()
        for nombre, eng in (("Railway", eng_rail), ("local", eng_loc)):
            if not _existe_tabla(eng):
                log(f"  [ERROR] {TABLA} no existe en {nombre}: correr "
                    f"scripts/oneshot/create_precios_intradia_table.py --target ambos")
                return 1
        sincronizar(eng_rail, eng_loc, args.dry_run)
    except Exception as e:
        log(f"  [ERROR] sync: {type(e).__name__}: {e}")
        log("  La purga NO corre: sin sync verificado no se borra.")
        return 1

    if args.sin_purga:
        return 0
    try:
        return purgar(eng_rail, eng_loc, args.keep_days, args.dry_run)
    except Exception as e:
        log(f"  [ERROR] purga: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
