"""
create_fundamentales_sector_table.py
Crea la tabla fundamentales_ticker_vs_sector (comparativa ticker vs su sector).

Contexto:
    Un ratio aislado (PER=32) no dice si la empresa esta cara o barata. Esta
    tabla situa cada metrica del ticker contra la mediana de sus PARES de la
    MISMA region (para no mezclar prima de riesgo pais). Responde: "el PER de
    X esta por encima/debajo de la media del sector, y cuan lejos".

Formato LONG (1 fila por ticker x metrica):
    Permite agregar metricas sin tocar schema y filtrar facil por metrica.
    Las 10 metricas v2: pe_ratio, pb_ratio, ps_ratio, ev_ebitda, roe_ttm,
    roa_ttm, roic_ttm, net_margin_ttm, operating_margin_ttm, revenue_yoy_pct.

Politica de peer-set (3 niveles, decidida por tamano del bucket sector x region):
    1. bucket (sector, region_del_ticker) con n>=5  -> peer_basis='region'
    2. ticker no-USA en bucket chico, pero su sector USA tiene n>=5
       -> peer_basis='usa_fallback' (flag honesto: comparado vs pares USA)
    3. resto (sector chico tambien en USA, o sin region) -> peer_basis='none'
       (leyenda "pocas empresas en seguimiento", sin benchmark)

    El umbral N=5 esta validado contra los datos (14 buckets limpios). El
    motor (compute_fundamentales_sector.py) es PARAMETRIZABLE por region para
    habilitar curaduria del usuario downstream (elegir que paises incluir).

Snapshot:
    Compara el ULTIMO Q de cada ticker contra el ultimo Q de cada par. Como
    los earnings estan escalonados, las fechas pueden diferir unos dias -- es
    una foto "estado actual", no un corte sincronico. fiscal_period_end = el
    ultimo Q del ticker. UPSERT por (ticker, fiscal_period_end, metric) ->
    acumula historia por Q.

Estadisticos por fila:
    value          : valor del ticker
    peer_median/p25/p75 : sobre valores NO-NULL del peer set (incluye al ticker)
    vs_median_pct  : (value-median)/|median|  (fraccional)
    percentile     : fraccion de pares con valor <= value (0..1, neutral)
    peer_n         : pares no-null usados en esta metrica
    low_sample     : peer_basis='none' o peer_n<3 (comparacion no confiable)

Uso:
    python scripts/oneshot/create_fundamentales_sector_table.py --target local
    python scripts/oneshot/create_fundamentales_sector_table.py --target local --dry-run
"""

import sys
import os
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import text
from scripts.oneshot.create_fundamentales_tables import (
    get_railway_engine, get_local_engine,
)

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


DDL = """
CREATE TABLE IF NOT EXISTS fundamentales_ticker_vs_sector (
    id                  SERIAL        PRIMARY KEY,
    ticker              VARCHAR(20)   NOT NULL,
    fiscal_period_end   DATE          NOT NULL,
    sector              VARCHAR(100),
    ticker_region       VARCHAR(20),
    peer_region         VARCHAR(20),   -- region del peer-set (='USA' si fallback)
    peer_basis          VARCHAR(20)   NOT NULL,  -- region | usa_fallback | none
    metric              VARCHAR(40)   NOT NULL,
    value               NUMERIC(18,6),
    peer_median         NUMERIC(18,6),
    peer_p25            NUMERIC(18,6),
    peer_p75            NUMERIC(18,6),
    vs_median_pct       NUMERIC(18,6),
    percentile          NUMERIC(6,4),
    peer_n              SMALLINT,
    low_sample          BOOLEAN       NOT NULL DEFAULT FALSE,
    computed_at         TIMESTAMP     NOT NULL DEFAULT NOW(),
    CONSTRAINT fundamentales_tvs_uniq UNIQUE (ticker, fiscal_period_end, metric)
)
"""

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_fund_tvs_ticker ON fundamentales_ticker_vs_sector (ticker)",
    "CREATE INDEX IF NOT EXISTS idx_fund_tvs_metric ON fundamentales_ticker_vs_sector (metric)",
    "CREATE INDEX IF NOT EXISTS idx_fund_tvs_sector ON fundamentales_ticker_vs_sector (sector)",
]


def crear_en(engine, etiqueta: str, dry_run: bool):
    tabla = "fundamentales_ticker_vs_sector"
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_schema='public' AND table_name=:t)
        """), {"t": tabla}).scalar()
        if existe:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()
            log(f"[{etiqueta}] {tabla}: ya existe ({n} filas). Verificando indices...")
            if dry_run:
                log(f"[{etiqueta}]   DRY-RUN: verificaria {len(INDICES)} indices.")
                return
            for stmt in INDICES:
                conn.execute(text(stmt))
            conn.commit()
            log(f"[{etiqueta}]   indices OK.")
            return
        if dry_run:
            log(f"[{etiqueta}] {tabla}: DRY-RUN, crearia tabla + {len(INDICES)} indices.")
            return
        conn.execute(text(DDL.strip()))
        for stmt in INDICES:
            conn.execute(text(stmt))
        conn.commit()
        log(f"[{etiqueta}] {tabla}: creada (0 filas) + {len(INDICES)} indices.")


def main():
    parser = argparse.ArgumentParser(
        description="Crea fundamentales_ticker_vs_sector en Railway y/o local")
    parser.add_argument("--target", choices=["railway", "local", "both"],
                        default="local")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  CREATE fundamentales_ticker_vs_sector  |  target={args.target}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    if args.target in ("railway", "both"):
        crear_en(get_railway_engine(), "RAILWAY", args.dry_run)
        print()
    if args.target in ("local", "both"):
        crear_en(get_local_engine(), "LOCAL", args.dry_run)
        print()

    log("Completado.")
    print()


if __name__ == "__main__":
    main()
