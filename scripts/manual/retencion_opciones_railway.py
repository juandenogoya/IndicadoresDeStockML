"""
retencion_opciones_railway.py -- purga de opciones_snapshot en Railway.

CAUSA RAIZ del incidente 2026-07-20 (Railway detenido por limite de consumo):
opciones_snapshot crecia sin retencion (~100k filas = ~19 MB/dia => +580 MB/mes).
Al 20/7 pesaba 923 MB de los 938 MB totales de la DB (98%).

Bajo Plan C, Railway es un BUFFER DE TRANSITO: el cron Oracle escribe ahi el
crudo (irrecuperable) y el sync manual lo baja a LOCAL, que es la fuente de
verdad historica. Railway no necesita la historia: solo el colchon de dias
recientes que aun podrian no estar sincronizados.

PRINCIPIO DE SEGURIDAD: se borra UNICAMENTE lo que esta PROBADAMENTE replicado
en local. Para cada fecha candidata se compara COUNT(*) local vs Railway; si
local tiene menos filas, esa fecha NO se toca (y se reporta). La seguridad
viene de la verificacion, no de confiar en que el sync anduvo.

Ademas del DELETE, hace VACUUM FULL: en PostgreSQL el DELETE solo marca las
paginas como reusables pero NO devuelve el disco (el archivo sigue ocupando lo
mismo y Railway factura por volumen). VACUUM FULL reescribe la tabla compacta.
OJO: toma un lock exclusivo (~1-3 min para esta tabla); correr fuera de las
ventanas del cron de opciones (23:00 / 02:00 / 06:00 UTC).

Uso:
    python scripts/manual/retencion_opciones_railway.py --dry-run   # muestra plan
    python scripts/manual/retencion_opciones_railway.py             # purga + vacuum
    python scripts/manual/retencion_opciones_railway.py --keep-days 15
    python scripts/manual/retencion_opciones_railway.py --no-vacuum # solo delete

Encadenado como paso final de sync_opciones_railway_to_local.bat (el momento
natural: local recien verificado fresco). Ahi corre con --quiet-skip para no
ensuciar el output si no hay nada que purgar.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime

from sqlalchemy import create_engine, text

KEEP_DAYS_DEFAULT = 10   # ultimos N dias de snapshot que Railway conserva


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Conexiones (patron dual de push_senales_bot: lee local, opera Railway) ────

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


# ── Purga ─────────────────────────────────────────────────────────────────────

def _tam_tabla(eng) -> str:
    with eng.connect() as c:
        return c.execute(text(
            "SELECT pg_size_pretty(pg_total_relation_size('opciones_snapshot'))"
        )).scalar()


def run(keep_days: int, dry_run: bool, vacuum: bool, quiet_skip: bool) -> int:
    eng_rail = _engine_railway()
    eng_loc = _engine_local()

    # Fechas en Railway, de mas nueva a mas vieja
    with eng_rail.connect() as c:
        fechas_rail = dict(c.execute(text(
            "SELECT fecha_snapshot, COUNT(*) FROM opciones_snapshot "
            "GROUP BY 1 ORDER BY 1 DESC"
        )).fetchall())

    todas = list(fechas_rail.keys())          # DESC
    a_conservar = todas[:keep_days]
    candidatas = todas[keep_days:]            # las que superan la retencion

    if not candidatas:
        if not quiet_skip:
            log(f"Railway tiene {len(todas)} dias (<= retencion de {keep_days}). "
                f"Nada que purgar.")
        return 0

    log("=" * 64)
    log(f"  RETENCION opciones_snapshot en RAILWAY  |  keep={keep_days} dias")
    log(f"  Modo: {'DRY RUN' if dry_run else 'REAL'}")
    log("=" * 64)
    log(f"  Dias en Railway : {len(todas)}  ({todas[-1]} -> {todas[0]})")
    log(f"  Conservar       : {len(a_conservar)}  (desde {a_conservar[-1]})")
    log(f"  Candidatas purga: {len(candidatas)}")
    log(f"  Tamano actual   : {_tam_tabla(eng_rail)}")

    # ── Verificacion: local debe tener AL MENOS las mismas filas por fecha ──
    with eng_loc.connect() as c:
        fechas_loc = dict(c.execute(text(
            "SELECT fecha_snapshot, COUNT(*) FROM opciones_snapshot "
            "WHERE fecha_snapshot = ANY(:fs) GROUP BY 1"
        ), {"fs": candidatas}).fetchall())

    verificadas, bloqueadas = [], []
    for f in candidatas:
        if fechas_loc.get(f, 0) >= fechas_rail[f]:
            verificadas.append(f)
        else:
            bloqueadas.append((f, fechas_rail[f], fechas_loc.get(f, 0)))

    if bloqueadas:
        log("")
        log(f"  [!] {len(bloqueadas)} fechas NO replicadas en local -> NO se tocan:")
        for f, nr, nl in bloqueadas:
            log(f"      {f}  railway={nr:,}  local={nl:,}  -> correr sync primero")

    if not verificadas:
        log("  Ninguna fecha verificada como replicada. No se borra nada.")
        return 1

    filas_a_borrar = sum(fechas_rail[f] for f in verificadas)
    log("")
    log(f"  Verificadas OK  : {len(verificadas)} fechas "
        f"({verificadas[-1]} -> {verificadas[0]}) = {filas_a_borrar:,} filas")

    if dry_run:
        log("  [DRY RUN] no se borra nada.")
        return 0

    # ── DELETE por fecha (transacciones cortas; un fallo no invalida el resto) ──
    borradas = 0
    for f in sorted(verificadas):
        with eng_rail.begin() as c:
            n = c.execute(text(
                "DELETE FROM opciones_snapshot WHERE fecha_snapshot = :f"
            ), {"f": f}).rowcount
        borradas += n
        log(f"    - {f}: {n:,} filas borradas")

    log(f"  Total borrado   : {borradas:,} filas")

    # ── VACUUM FULL: devolver el disco (DELETE solo no reduce el volumen) ────
    if vacuum:
        log("  VACUUM FULL opciones_snapshot (lock exclusivo, puede tardar)...")
        with eng_rail.connect().execution_options(
                isolation_level="AUTOCOMMIT") as c:
            c.execute(text("VACUUM FULL opciones_snapshot"))
        log(f"  Tamano final    : {_tam_tabla(eng_rail)}")
    else:
        log("  --no-vacuum: el espacio queda marcado reusable pero NO se "
            "devuelve al volumen (correr VACUUM FULL luego).")

    log("=" * 64)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Purga opciones_snapshot en Railway (solo fechas replicadas en local)")
    ap.add_argument("--keep-days", type=int, default=KEEP_DAYS_DEFAULT,
                    help=f"Dias de snapshot a conservar en Railway (default {KEEP_DAYS_DEFAULT})")
    ap.add_argument("--dry-run", action="store_true", help="Muestra el plan, no borra")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="Salta el VACUUM FULL post-delete (no recomendado)")
    ap.add_argument("--quiet-skip", action="store_true",
                    help="Silencioso si no hay nada que purgar (uso encadenado al sync)")
    args = ap.parse_args()

    if args.keep_days < 5:
        print("  [ERROR] keep-days < 5 elimina el colchon de seguridad. Abortando.")
        sys.exit(1)

    sys.exit(run(args.keep_days, args.dry_run, not args.no_vacuum, args.quiet_skip))


if __name__ == "__main__":
    main()
