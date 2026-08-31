"""
create_acciones_circulacion_table.py
Crea `acciones_circulacion`: serie de acciones en circulacion por ticker, en
BASE DE SPLIT ACTUAL, apareable con precios_diarios.

Por que es una tabla propia y no una columna de las fundamentales:
    Es un dato BASICO y transversal -- market cap, screener, dashboard e
    infografias lo necesitan, no solo los multiplos SEC. Y su grano no es el
    trimestre fiscal sino "la fecha en que alguien reporto un conteo", que no
    coincide con ningun period_end.

Por que en base actual:
    `precios_diarios` se corrige retroactivamente ante un split (splits.py
    corregir divide toda la historia), asi que el precio esta en base actual en
    todas las ruedas. yahooquery re-expresa igual -- KLAC figura con 1.367M en
    2023, ya post-split 10:1 -- asi que las dos series aparean sin conversion.
    La serie SEC de portada NO sirve para esto: esta en la base de su momento.
    Ver src/utils/acciones_series.py.

Cobertura: 200 tickers (todo el universo, no solo los 147 de SEC).

Uso:
    python scripts/oneshot/create_acciones_circulacion_table.py
    python scripts/oneshot/create_acciones_circulacion_table.py --dry-run
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
TABLA = "acciones_circulacion"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    id           SERIAL         PRIMARY KEY,
    ticker       VARCHAR(20)    NOT NULL,
    -- fecha a la que corresponde el conteo (cierre fiscal en yahooquery,
    -- portada del filing en sec_portada)
    fecha        DATE           NOT NULL,
    shares       NUMERIC(20,2)  NOT NULL,
    -- 'q' trimestral | 'a' anual. yahooquery sirve ~5 puntos trimestrales
    -- (ultimo anio) y ~4 anuales: entre 2023 y 2025 hay UN punto por anio.
    periodo      VARCHAR(2),
    -- 'yahooquery'  = OrdinarySharesNumber, base actual por construccion.
    -- 'sec_portada' = dei:EntityCommonStockSharesOutstanding, usado SOLO para
    --                 extender hacia atras y SOLO si se valido que ese ticker
    --                 no cambio de base (ver acciones_series.construir).
    fuente       VARCHAR(20)    NOT NULL,
    computed_at  TIMESTAMP      NOT NULL DEFAULT NOW(),

    -- UNIQUE FULL (sin WHERE): requisito de ON CONFLICT. Ver CLAUDE.md.
    CONSTRAINT {TABLA}_uniq UNIQUE (ticker, fecha)
)
"""

# El veredicto de la validacion por ticker. Se guarda porque es la respuesta a
# "hasta cuando puedo confiar en la serie de este ticker", y esa pregunta se
# hace despues, no durante el refresh.
DDL_VALIDACION = f"""
CREATE TABLE IF NOT EXISTS {TABLA}_validacion (
    ticker         VARCHAR(20)   PRIMARY KEY,
    extendido      BOOLEAN       NOT NULL,
    desde_efectivo DATE,
    n_yahoo        SMALLINT,
    n_sec_usados   SMALLINT,
    n_pares        SMALLINT,
    ratio_min      NUMERIC(12,4),
    ratio_max      NUMERIC(12,4),
    motivo         TEXT,
    updated_at     TIMESTAMP     NOT NULL DEFAULT NOW()
)
"""

INDICES = [
    f"CREATE INDEX IF NOT EXISTS idx_acc_circ_ticker_fecha "
    f"ON {TABLA} (ticker, fecha)",
]

TABLAS = [(TABLA, DDL), (f"{TABLA}_validacion", DDL_VALIDACION)]


def crear(engine, dry_run):
    with engine.connect() as conn:
        for tabla, ddl in TABLAS:
            existe = conn.execute(text("""
                SELECT EXISTS (SELECT 1 FROM information_schema.tables
                               WHERE table_schema='public' AND table_name=:t)
            """), {"t": tabla}).scalar()
            if existe:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()
                log(f"{tabla}: ya existe ({n} filas).")
                continue
            if dry_run:
                log(f"{tabla}: DRY-RUN, se crearia.")
                continue
            conn.execute(text(ddl.strip()))
            log(f"{tabla}: creada.")
        if not dry_run:
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
        print(DDL); print(DDL_VALIDACION); return
    print(); print(SEP)
    print(f"  CREATE {TABLA}{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP); print()
    crear(get_local_engine(), args.dry_run)
    print(); print(SEP); print("  OK"); print(SEP); print()


if __name__ == "__main__":
    main()
