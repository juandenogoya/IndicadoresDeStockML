"""
acciones_fuente_mas_ancha.py -- acciones_circulacion.fuente varchar(20) -> (40)

POR QUE: la etiqueta de `fuente` dejo de nombrar solo el nivel SEC y ahora
tambien dice si la serie fue REBASADA a la base de split de hoy. El sufijo
`_rb` empuja la etiqueta mas larga fuera del limite:

    sec_promedio_diluido      20  <- justo en el borde
    sec_promedio_diluido_rb   23  <- no entra

El refresh se caia con StringDataRightTruncation despues de bajar los 200
tickers de yahooquery, o sea que el error costaba la corrida entera.

La alternativa era acortar la etiqueta (`spd_rb` y companeras). No: la columna
existe para que un humano lea de donde salio cada punto seis meses despues, y
40 caracteres en una tabla de ~2.500 filas no cuestan nada. Ensanchar un
varchar no reescribe la tabla ni toca los datos.

Idempotente. LOCAL-only.
"""
import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

TABLA = "acciones_circulacion"
COL = "fuente"
ANCHO = 40


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
            cur.execute("""SELECT character_maximum_length
                           FROM information_schema.columns
                           WHERE table_name = %s AND column_name = %s""",
                        (TABLA, COL))
            fila = cur.fetchone()
            if not fila:
                print(f"  {TABLA}.{COL} no existe -- nada que hacer")
                return 1
            actual = fila[0]
            print(f"  {TABLA}.{COL} hoy: varchar({actual})")

            if actual is None or actual >= ANCHO:
                print("  ya alcanza, nada que hacer")
                return 0
            if args.dry_run:
                print(f"  [DRY-RUN] ALTER ... TYPE varchar({ANCHO})")
                return 0

            cur.execute(f"ALTER TABLE {TABLA} ALTER COLUMN {COL} "
                        f"TYPE varchar({ANCHO})")
            print(f"  ALTER {TABLA}.{COL} -> varchar({ANCHO})")
        cx.commit()
    print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
