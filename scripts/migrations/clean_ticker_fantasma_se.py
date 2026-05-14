"""
clean_ticker_fantasma_se.py
Limpia el ticker fantasma 'SE\\r' (SE con \\r CR Windows) de todas las
tablas donde aparezca.

Origen del bug: en algun momento (probablemente carga via CSV o copy/paste)
una fila con ticker contaminado con \\r entro a activos. Desde entonces:
  - Aparece como "duplicado" de SE en COUNT(DISTINCT ticker)
  - yfinance NO puede descargar para 'SE\\r' (ticker invalido) -> "sin datos"
  - Tablas relacionadas (precios, indicadores, etc.) acumulan filas
    historicas huerfanas con ese ticker

Estrategia:
  1. Identificar TODAS las tablas con columna 'ticker'
  2. Para cada una, contar filas con ticker = 'SE' || CHR(13)
  3. Mostrar el resumen
  4. Si --confirm, ejecutar DELETE en cada tabla

Uso:
    python scripts/migrations/clean_ticker_fantasma_se.py
        -> DRY-RUN contra LOCAL (default)
    python scripts/migrations/clean_ticker_fantasma_se.py --target railway
        -> DRY-RUN contra RAILWAY
    python scripts/migrations/clean_ticker_fantasma_se.py --confirm
        -> Ejecuta DELETE en LOCAL
    python scripts/migrations/clean_ticker_fantasma_se.py --target railway --confirm
        -> Ejecuta DELETE en RAILWAY
"""
import os
import sys
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
from sqlalchemy import create_engine, text


def _parse_env_file(path: str) -> dict:
    result = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                result[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return result


def get_engine(target: str):
    env_general = _parse_env_file(os.path.join(ROOT, ".env"))
    if target == "local":
        host = env_general.get("DB_HOST", "localhost")
        port = env_general.get("DB_PORT", "5432")
        name = env_general.get("DB_NAME", "activos_ml")
        user = env_general.get("DB_USER", "postgres")
        pwd  = env_general.get("DB_PASSWORD", "")
        url  = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
        return create_engine(url, pool_pre_ping=True)
    # railway
    env_local = _parse_env_file(os.path.join(ROOT, ".env.local"))
    db_url = env_local.get("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError("DATABASE_URL no encontrado en .env.local")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, connect_args={"sslmode": "require"}, pool_pre_ping=True)


GHOST_TICKER_SQL = "'SE' || CHR(13)"   # 'SE\r'

SEP = "=" * 65


def main():
    parser = argparse.ArgumentParser(
        description="Limpia el ticker fantasma 'SE\\r' de todas las tablas con columna ticker."
    )
    parser.add_argument("--target", choices=["local", "railway"], default="local",
                        help="DB destino (default: local)")
    parser.add_argument("--confirm", action="store_true",
                        help="Sin este flag es DRY-RUN. Con --confirm ejecuta DELETE.")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  CLEAN TICKER FANTASMA 'SE\\r'  |  TARGET: {args.target.upper()}  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  MODO: {'EJECUCION DELETE' if args.confirm else 'DRY-RUN (sin escribir)'}")
    print(SEP)
    print()

    engine = get_engine(args.target)

    # 1. Descubrir tablas con columna 'ticker'
    sql_descubrir = text("""
        SELECT table_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND column_name = 'ticker'
        ORDER BY table_name
    """)
    with engine.connect() as c:
        tablas = [r[0] for r in c.execute(sql_descubrir).fetchall()]

    print(f"  Tablas con columna 'ticker': {len(tablas)}")
    for t in tablas:
        print(f"    - {t}")
    print()

    # 2. Por cada tabla, contar filas con ticker contaminado
    print(f"  Filas con ticker = 'SE\\r':")
    print(f"  {'-' * 50}")
    print(f"  {'tabla':<35s}  {'filas':>10}")

    counts = {}
    with engine.connect() as c:
        for t in tablas:
            try:
                sql = text(f"SELECT COUNT(*) FROM {t} WHERE ticker = {GHOST_TICKER_SQL}")
                n = c.execute(sql).scalar() or 0
                counts[t] = n
                if n > 0:
                    print(f"  {t:<35s}  {n:>10,}")
            except Exception as e:
                print(f"  {t:<35s}  ERROR: {str(e)[:30]}")

    total_filas = sum(counts.values())
    tablas_afectadas = [t for t, n in counts.items() if n > 0]

    print()
    print(f"  Tablas afectadas: {len(tablas_afectadas)}")
    print(f"  Total filas a borrar: {total_filas:,}")
    print()

    if total_filas == 0:
        print("  Nada que limpiar. La DB no tiene el ticker fantasma.")
        sys.exit(0)

    if not args.confirm:
        print(f"  DRY-RUN: para ejecutar DELETE, agregar --confirm")
        sys.exit(0)

    # 3. Ejecutar DELETEs (cada tabla en su propia transaccion)
    print(f"  Ejecutando DELETE en {len(tablas_afectadas)} tablas...")
    total_borradas = 0
    with engine.begin() as c:
        for t in tablas_afectadas:
            try:
                res = c.execute(text(f"DELETE FROM {t} WHERE ticker = {GHOST_TICKER_SQL}"))
                n_borradas = res.rowcount or 0
                total_borradas += n_borradas
                print(f"    {t:<35s}  {n_borradas:>10,} borradas")
            except Exception as e:
                print(f"    {t:<35s}  ERROR: {str(e)[:50]}")

    print()
    print(SEP)
    print(f"  LIMPIEZA COMPLETADA  ({args.target.upper()})")
    print(f"  Total borradas: {total_borradas:,}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
