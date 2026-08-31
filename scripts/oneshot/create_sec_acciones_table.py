"""
create_sec_acciones_table.py
Crea fundamentales_sec_acciones: serie POINT-IN-TIME de acciones en circulacion
tomada de la PORTADA de cada filing (dei:EntityCommonStockSharesOutstanding).

Por que una tabla aparte de fundamentales_sec_q:
    Distinto GRANO y distinta regla de desempate. fundamentales_sec_q tiene una
    fila por TRIMESTRE con el valor VIGENTE (el `filed` mas nuevo gana); esta
    tabla tiene una fila por FILING con el valor que estaba publicado ENTONCES
    (el `filed` mas viejo gana ante una re-declaracion de la misma fecha).
    Meterlas en la misma tabla obligaria a elegir una sola de las dos reglas.

Para que sirve:
    Es el unico conteo de acciones que se puede multiplicar por un precio
    HISTORICO sin romperlo. SEC re-expresa retroactivamente lo "por accion"
    cuando hay un split y `precios_diarios` no se re-ajusta hacia atras; el
    hecho de portada no se re-expresa nunca porque cada filing declara el suyo.
    Detalle y evidencia: src/utils/sec_acciones.py y
    docs/fuentes_fundamentales.md.

Uso:
    python scripts/oneshot/create_sec_acciones_table.py
    python scripts/oneshot/create_sec_acciones_table.py --dry-run
"""

import sys
import os
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import text

from scripts.oneshot.create_fundamentales_tables import get_local_engine

SEP = "=" * 64
TABLA = "fundamentales_sec_acciones"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    id           SERIAL        PRIMARY KEY,
    ticker       VARCHAR(20)   NOT NULL,
    -- fecha de PORTADA: a que fecha la empresa declara el conteo en la
    -- caratula del filing. Cae unos dias ANTES del filed.
    fecha        DATE          NOT NULL,
    shares       NUMERIC(20,2) NOT NULL,
    accn         VARCHAR(30),
    filed        DATE,
    form         VARCHAR(10),
    -- 'portada' = dei:EntityCommonStockSharesOutstanding (conteo a una fecha).
    -- 'promedio_diluido' = respaldo para filers de CLASES MULTIPLES, donde la
    -- portada viene con dimensiones XBRL y companyfacts la descarta. Es el
    -- promedio ponderado del trimestre, no un conteo a una fecha: magnitud
    -- parecida pero distinta, por eso queda etiquetada.
    fuente       VARCHAR(20),
    computed_at  TIMESTAMP     NOT NULL DEFAULT NOW(),

    -- UNIQUE FULL (sin WHERE): requisito de ON CONFLICT. Ver CLAUDE.md.
    CONSTRAINT {TABLA}_uniq UNIQUE (ticker, fecha)
)
"""

ALTERS = [
    f"ALTER TABLE {TABLA} ADD COLUMN IF NOT EXISTS fuente VARCHAR(20)",
]

INDICES = [
    f"CREATE INDEX IF NOT EXISTS idx_sec_acc_ticker_fecha "
    f"ON {TABLA} (ticker, fecha)",
]


def crear(engine, dry_run: bool):
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_schema='public' AND table_name=:t)
        """), {"t": TABLA}).scalar()
        if existe:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLA}")).scalar()
            log(f"{TABLA}: ya existe ({n} filas).")
        elif dry_run:
            log(f"{TABLA}: DRY-RUN, se crearia.")
            return
        else:
            conn.execute(text(DDL.strip()))
            log(f"{TABLA}: creada.")
        if not dry_run:
            for stmt in ALTERS:
                conn.execute(text(stmt))
            for stmt in INDICES:
                conn.execute(text(stmt))
            conn.commit()
            log(f"indices verificados ({len(INDICES)}).")


def main():
    p = argparse.ArgumentParser(description=f"Crea {TABLA} (LOCAL-only)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mostrar-ddl", action="store_true")
    args = p.parse_args()

    if args.mostrar_ddl:
        print(DDL)
        return

    print()
    print(SEP)
    print(f"  CREATE {TABLA}{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()
    crear(get_local_engine(), args.dry_run)
    print()
    print(SEP)
    print("  OK")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
