"""
multiplos_net_debt_q.py -- agrega fundamentales_sec_multiplos_d.net_debt_q

POR QUE: la deuda neta pasa a admitir ARRASTRE. Si a un trimestre le falta uno
de los dos componentes de deuda pero la empresa lo tagea en otros periodos, se
usa el ultimo valor conocido en vez de apagar el EV entero. Antes esos
trimestres perdian el EV; el trimestre actual SI trae uno de los dos en 14 de
los 15 tickers afectados, y el que falta esta publicado uno o dos trimestres
antes.

La columna guarda CUANTOS TRIMESTRES de antiguedad tiene el componente mas
viejo que se uso (0 = todo del periodo). Existe por la misma razon que
`shares_dias`: un dato arrastrado sigue siendo un dato, pero uno arrastrado EN
SILENCIO es una trampa. Con la edad a la vista, el consumidor decide.

Medido: arrastrando solo el componente ausente, el error del EV tiene mediana
0,05% y 11 de 15 quedan por debajo de 0,30%. Los expuestos son pocos y grandes
-- TMUS 6,2%, DE 3,1%, CVX 1,9% -- y son justo los que esta columna delata.

Idempotente. LOCAL-only.
"""
import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

TABLA = "fundamentales_sec_multiplos_d"
COL = "net_debt_q"


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
            cur.execute("""SELECT 1 FROM information_schema.columns
                           WHERE table_name = %s AND column_name = %s""",
                        (TABLA, COL))
            if cur.fetchone():
                print(f"  {TABLA}.{COL} ya existe, nada que hacer")
                return 0
            if args.dry_run:
                print(f"  [DRY-RUN] ALTER TABLE {TABLA} ADD COLUMN {COL} smallint")
                return 0
            cur.execute(f"ALTER TABLE {TABLA} ADD COLUMN {COL} smallint")
            print(f"  ADD {TABLA}.{COL} smallint")
        cx.commit()
    print("  OK -- recomputar con scripts/compute_sec_multiplos.py para poblarla")
    return 0


if __name__ == "__main__":
    sys.exit(main())
