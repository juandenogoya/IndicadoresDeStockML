"""
clean_railway_may12.py
Script one-shot para limpiar las 8 filas basura de precios_diarios en Railway,
con fecha=2026-05-12, originadas por pipeline_automatico.bat corriendo
pre-mercado el 12/05/2026 (OHLCV del 11 etiquetados como del 12).

Uso:
    python scripts/migrations/clean_railway_may12.py              # dry-run
    python scripts/migrations/clean_railway_may12.py --confirm    # ejecuta el DELETE

El script muestra primero las filas que va a borrar, espera confirmacion
del flag --confirm para hacer el DELETE. Solo borra fecha exacta 2026-05-12.
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


def get_railway_engine():
    env = _parse_env_file(os.path.join(ROOT, ".env.local"))
    db_url = env.get("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError("DATABASE_URL no encontrado en .env.local")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, connect_args={"sslmode": "require"})


FECHA_TARGET = "2026-05-12"
SEP = "=" * 65


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="Sin este flag es DRY-RUN. Con el flag ejecuta el DELETE.")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  CLEAN RAILWAY  --  fecha={FECHA_TARGET}  --  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  MODO: {'EJECUCION' if args.confirm else 'DRY-RUN (sin escribir)'}")
    print(SEP)
    print()

    engine = get_railway_engine()

    with engine.connect() as c:
        df = pd.read_sql(
            text(f"SELECT ticker, fecha, open, high, low, close, volume "
                 f"FROM precios_diarios WHERE fecha = '{FECHA_TARGET}' ORDER BY ticker"),
            c
        )

    if df.empty:
        print(f"  No hay filas con fecha={FECHA_TARGET} en Railway. Nada que limpiar.")
        sys.exit(0)

    print(f"  Filas a borrar: {len(df)}")
    print()
    print(f"  {'ticker':<8} {'open':>10} {'high':>10} {'low':>10} {'close':>10} {'volume':>14}")
    print("  " + "-" * 60)
    for _, r in df.iterrows():
        print(f"  {r['ticker']:<8} {r['open']:>10.2f} {r['high']:>10.2f} "
              f"{r['low']:>10.2f} {r['close']:>10.2f} {r['volume']:>14,}")

    print()
    if not args.confirm:
        print("  DRY-RUN: para ejecutar el DELETE, correr con --confirm")
        print()
        sys.exit(0)

    # Ejecutar DELETE
    print("  Ejecutando DELETE...")
    with engine.connect() as c:
        res = c.execute(
            text(f"DELETE FROM precios_diarios WHERE fecha = '{FECHA_TARGET}'")
        )
        c.commit()
        print(f"  OK -- filas eliminadas: {res.rowcount}")

    print()
    print(SEP)
    print("  LIMPIEZA COMPLETADA")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
