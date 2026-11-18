"""
create_precios_intradia_table.py -- crea las tablas de la linea intradiaria (one-shot).

    precios_intradia_30m   barras de 30 min de los cortes (etapa 1)
    vigilancia_listas      cabecera de la lista de vigilancia de la rueda D (etapa 2)
    vigilancia_tickers     tickers de la lista
    vigilancia_niveles     niveles que se siguen de cada ticker de la lista
    vigilancia_envios      alertas enviadas por corte (evita repetirlas)

Las cinco viven en RAILWAY (buffer de transito con ~30 ruedas de retencion:
Oracle escribe las barras y los envios, y lee la lista) y en LOCAL (fuente de
verdad historica: el sync baja barras y envios, push_vigilancia.py escribe la
lista en los dos lados). Es la primera excepcion al Plan C
("Railway = SOLO opciones_snapshot"), decidida el 30/10/2026: ver
docs/intradia.md. Los DDL viven en src/utils/intradia.py y
src/utils/vigilancia_intradia.py (una sola definicion).

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

from src.utils import intradia, vigilancia_intradia as vi

DDL = intradia.DDL_TABLA + vi.DDL_TABLAS
TABLAS = [intradia.TABLA, vi.TABLA_LISTAS, vi.TABLA_TICKERS, vi.TABLA_NIVELES, vi.TABLA_ENVIOS]


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
        print(DDL)
        print("  DRY RUN: no se ejecuto.")
        return
    with eng.begin() as c:
        c.execute(text(DDL))
        for tabla in TABLAS:
            cols = c.execute(text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = :t ORDER BY ordinal_position"), {"t": tabla}).fetchall()
            idx = c.execute(text(
                "SELECT indexname FROM pg_indexes WHERE tablename = :t ORDER BY 1"),
                {"t": tabla}).fetchall()
            print(f"  {tabla}: {len(cols)} columnas | indices: "
                  f"{', '.join(i.indexname for i in idx)}")
            for col in cols:
                print(f"    {col.column_name:<14} {col.data_type}")


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
