"""
create_polygon_tables.py -- las 3 tablas de la fuente Polygon. LOCAL-only.

  polygon_splits     historial de splits por ticker. La razon principal por la
                     que se sumo Polygon: hoy `splits.py` los detecta con una
                     heuristica de dos etapas que ya dio falsos positivos
                     (ORCL, DELL), y las guardas de acciones_series rechazan
                     series enteras por no poder distinguir un split de un
                     error. Con una lista autoritativa el rebase deja de ser
                     heuristica y pasa a ser aritmetica.

  polygon_acciones   conteo point-in-time por (ticker, fecha). Se guardan los
                     DOS campos, no uno: `share_class` es solo la clase de ese
                     ticker y `weighted` es el total de todas. Guardar solo el
                     total perderia el desglose por clase, que es justamente
                     lo que companyfacts descarta y no se consigue gratis en
                     ningun otro lado.
                     OJO: `weighted` viene en la BASE DE SU MOMENTO.

  polygon_ingesta    log por (ticker, tarea). Existe para que las corridas
                     sean REANUDABLES: a 4 pedidos por minuto una pasada
                     completa son horas, y que la maten a la mitad no puede
                     costar la mitad del trabajo. Sin este log no se puede
                     distinguir "todavia no lo pedi" de "lo pedi y no tiene
                     splits", que en la tabla se ven igual: ninguna fila.

Idempotente.
"""
import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

DDL = """
CREATE TABLE IF NOT EXISTS polygon_splits (
    id              SERIAL      PRIMARY KEY,
    ticker          TEXT        NOT NULL,
    execution_date  DATE        NOT NULL,
    split_from      NUMERIC     NOT NULL,
    split_to        NUMERIC     NOT NULL,
    -- ratio = split_to / split_from. Un 10:1 da 10: las acciones se
    -- MULTIPLICAN por 10 y el precio se divide. Se guarda calculado para que
    -- nadie tenga que acordarse de la orientacion.
    ratio           NUMERIC     NOT NULL,
    polygon_id      TEXT,
    fetched_at      TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT polygon_splits_uniq UNIQUE (ticker, execution_date)
);
CREATE INDEX IF NOT EXISTS idx_polygon_splits_ticker
    ON polygon_splits (ticker, execution_date);

CREATE TABLE IF NOT EXISTS polygon_acciones (
    id                  SERIAL      PRIMARY KEY,
    ticker              TEXT        NOT NULL,
    fecha               DATE        NOT NULL,
    share_class_shares  NUMERIC,
    weighted_shares     NUMERIC,
    market_cap          NUMERIC,
    fetched_at          TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT polygon_acciones_uniq UNIQUE (ticker, fecha)
);
CREATE INDEX IF NOT EXISTS idx_polygon_acciones_ticker
    ON polygon_acciones (ticker, fecha);

CREATE TABLE IF NOT EXISTS polygon_ingesta (
    id          SERIAL      PRIMARY KEY,
    ticker      TEXT        NOT NULL,
    tarea       TEXT        NOT NULL,
    existe      BOOLEAN,
    n_filas     INTEGER,
    detalle     TEXT,
    fetched_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT polygon_ingesta_uniq UNIQUE (ticker, tarea)
);
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    os.environ.pop("DATABASE_URL", None)          # LOCAL, nunca Railway
    dsn = dict(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
               user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
               dbname=os.getenv("DB_NAME"))
    if args.dry_run:
        print(DDL)
        return 0
    with psycopg2.connect(**dsn) as cx:
        with cx.cursor() as cur:
            cur.execute(DDL)
        cx.commit()
    print("  OK  polygon_splits / polygon_acciones / polygon_ingesta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
