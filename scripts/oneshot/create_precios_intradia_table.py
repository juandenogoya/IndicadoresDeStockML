"""
create_precios_intradia_table.py -- crea precios_intradia_30m (one-shot).

La tabla vive en RAILWAY (la escribe el snapshot de Oracle, como buffer de
transito con ~30 ruedas de retencion) y en LOCAL (fuente de verdad historica,
la llena sync_intradia_railway_to_local.py). Es la primera excepcion al Plan C
("Railway = SOLO opciones_snapshot"), decidida el 30/10/2026: ver
docs/intradia.md. El DDL vive en src/utils/intradia.py (una sola definicion).

Idempotente (CREATE ... IF NOT EXISTS).

Uso:
    python scripts/oneshot/create_precios_intradia_table.py --target ambos
    python scripts/oneshot/create_precios_intradia_table.py --target local
    python scripts/oneshot/create_precios_intradia_table.py --target railway --dry-run
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, text

from src.utils.intradia import DDL_TABLA, TABLA


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


def engine_railway():
    url = _leer_database_url_local().replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url, connect_args={"sslmode": "require"}, pool_pre_ping=True)


def engine_local():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    os.environ.pop("DATABASE_URL", None)
    from src.data.database import get_engine
    return get_engine()


def crear(nombre: str, eng, dry_run: bool):
    print(f"\n[{nombre}]")
    if dry_run:
        print(DDL_TABLA)
        print("  DRY RUN: no se ejecuto.")
        return
    with eng.begin() as c:
        c.execute(text(DDL_TABLA))
        cols = c.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = :t ORDER BY ordinal_position"), {"t": TABLA}).fetchall()
        idx = c.execute(text(
            "SELECT indexname FROM pg_indexes WHERE tablename = :t ORDER BY 1"),
            {"t": TABLA}).fetchall()
    print(f"  {TABLA}: {len(cols)} columnas")
    for col in cols:
        print(f"    {col.column_name:<14} {col.data_type}")
    print(f"  indices: {', '.join(i.indexname for i in idx)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["railway", "local", "ambos"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.target in ("railway", "ambos"):
        crear("RAILWAY", engine_railway(), args.dry_run)
    if args.target in ("local", "ambos"):
        crear("LOCAL", engine_local(), args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
