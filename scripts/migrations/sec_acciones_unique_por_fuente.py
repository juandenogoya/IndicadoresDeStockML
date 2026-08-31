"""
sec_acciones_unique_por_fuente.py -- UNIQUE (ticker, fecha) -> (ticker, fecha, fuente)

POR QUE: `fundamentales_sec_acciones` guardaba UN solo nivel por ticker, el que
sec_acciones.serie_acciones() elegia por preferencia. Elegir ahi es elegir a
CIEGAS: el unico que puede saber que nivel sirve es quien lo valida contra
yahooquery, y eso ocurre despues, en acciones_series.construir().

Medido: con la eleccion a ciegas, TRIP y LYFT se quedaban con una portada de 4-6
puntos que arranca en 2025 y perdian 930 ruedas de market cap; y al invertir la
regla por cobertura se rompian UPST y SNOW, que ya andaban. Ninguna regla ciega
gana. Con los tres niveles persistidos, el refresh los prueba en orden y se
queda con el primero que VALIDA.

La tabla es DERIVADA y se regenera entera desde el cache, asi que la migracion
no puede perder informacion: solo habilita guardar mas.

Idempotente. LOCAL-only.
"""
import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

TABLA = "fundamentales_sec_acciones"
VIEJA = f"{TABLA}_uniq"
NUEVA = f"{TABLA}_uniq_fuente"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    os.environ.pop("DATABASE_URL", None)          # LOCAL, nunca Railway
    dsn = dict(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
               user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
               dbname=os.getenv("DB_NAME"))

    with psycopg2.connect(**dsn) as cx:
        with cx.cursor() as cur:
            cur.execute("""SELECT conname FROM pg_constraint
                           WHERE conrelid = %s::regclass AND contype = 'u'""", (TABLA,))
            actuales = {r[0] for r in cur.fetchall()}
            print(f"  constraints UNIQUE hoy: {sorted(actuales) or '(ninguna)'}")

            if NUEVA in actuales:
                print("  ya migrada, nada que hacer")
                return 0
            if args.dry_run:
                print(f"  [DRY-RUN] DROP {VIEJA}  ->  ADD {NUEVA} (ticker, fecha, fuente)")
                return 0

            if VIEJA in actuales:
                cur.execute(f"ALTER TABLE {TABLA} DROP CONSTRAINT {VIEJA}")
                print(f"  DROP {VIEJA}")
            cur.execute(f"ALTER TABLE {TABLA} ADD CONSTRAINT {NUEVA} "
                        f"UNIQUE (ticker, fecha, fuente)")
            print(f"  ADD  {NUEVA} (ticker, fecha, fuente)")
        cx.commit()
    print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
