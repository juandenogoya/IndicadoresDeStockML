"""
create_fundamentales_ratios_table.py
Crea la tabla fundamentales_ratios_q (capa derivada) en local.

Contexto:
    Las 4 tablas raw (income/balance/cashflow/valuation) guardan magnitudes
    crudas. Esta tabla DERIVADA precomputa ratios + crecimiento QoQ/YoY para
    leer conclusiones, no recalcular. Es funcion PURA de las 4 raw -> se
    recomputa cuando se quiera sin re-fetch (la pobla
    scripts/compute_fundamentales_ratios.py).

Rol en el proyecto (acordado 31/5/2026):
    Vista PARALELA y DESCRIPTIVA del fundamental -- NO se mezcla con el score
    tecnico ni con los bots. Responde 3 preguntas sobre la empresa:
      P1) como la ve el mercado vs su valor real (PER, P/B, BVPS, BPA...)
      P2) como hace las cosas (rentabilidad/calidad/crecimiento Q/Q y Y/Y)
      P3) cash: FCF (el resto de capital-allocation -> v2)
    + estructura de solvencia (pasivo corriente: current_ratio, etc).

Bases de calculo:
    - Crecimiento: TRIMESTRAL (Q vs Q-1 = QoQ; Q vs Q-4 = YoY).
    - Rentabilidad / retornos / margenes: TTM (suma 4Q), estandar de industria.

Inmunidad de escala/moneda:
    Los RATIOS son adimensionales -> inmunes a moneda y a escala (reales vs
    "en miles"). El analisis comparable cross-ticker vive ahi. Los pocos
    ABSOLUTOS (book_value_per_share, eps_*, fcf_ttm, working_capital, net_debt,
    *_ttm crudos) quedan en reporting_currency y NO son comparables cross-ticker
    sin FX. El compute corre dos cross-checks de escala (ver ese script).

NULLs:
    Bancos (~17) sin gross_profit/operating/EBIT -> margenes/ROIC/current_ratio
    NULL. ~22 tickers sin EBIT -> ROIC NULL. NULL = "no aplica / no disponible",
    nunca rompe. La leyenda explicativa va en doc/dashboard, no por fila.

Sector:
    sector/industry denormalizados desde la tabla activos -> habilita
    GROUP BY sector para medianas por sector sin joins.

Uso:
    python scripts/oneshot/create_fundamentales_ratios_table.py --target local
    python scripts/oneshot/create_fundamentales_ratios_table.py --target local --dry-run
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
CREATE TABLE IF NOT EXISTS fundamentales_ratios_q (
    id                          SERIAL        PRIMARY KEY,
    ticker                      VARCHAR(20)   NOT NULL,
    fiscal_period_end           DATE          NOT NULL,
    period_type                 VARCHAR(5)    NOT NULL DEFAULT '3M',
    reporting_currency          VARCHAR(5),
    sector                      VARCHAR(100),
    industry                    VARCHAR(100),

    -- P1) Mercado vs valor real
    pe_ratio                    NUMERIC(18,6),
    pb_ratio                    NUMERIC(18,6),
    ps_ratio                    NUMERIC(18,6),
    ev_ebitda                   NUMERIC(18,6),
    pe_yoy_pct                  NUMERIC(18,6),
    pb_yoy_pct                  NUMERIC(18,6),
    book_value_per_share        NUMERIC(20,6),   -- absoluto (moneda reporte)
    eps_q                       NUMERIC(20,6),   -- BPA trimestral (absoluto)
    eps_ttm                     NUMERIC(20,6),   -- BPA TTM (absoluto)

    -- P2) Como hace las cosas (rentabilidad/calidad TTM + crecimiento Q)
    roe_ttm                     NUMERIC(18,6),
    roa_ttm                     NUMERIC(18,6),
    roic_ttm                    NUMERIC(18,6),
    gross_margin_ttm            NUMERIC(18,6),
    operating_margin_ttm        NUMERIC(18,6),
    net_margin_ttm              NUMERIC(18,6),
    net_margin_yoy_delta        NUMERIC(18,6),   -- ttm vs ttm hace 4Q (fraccion)
    operating_margin_yoy_delta  NUMERIC(18,6),
    opex_to_revenue_ttm         NUMERIC(18,6),   -- gastos operativos / revenue
    revenue_qoq_pct             NUMERIC(18,6),
    revenue_yoy_pct             NUMERIC(18,6),
    net_income_qoq_pct          NUMERIC(18,6),
    net_income_yoy_pct          NUMERIC(18,6),
    eps_qoq_pct                 NUMERIC(18,6),
    eps_yoy_pct                 NUMERIC(18,6),

    -- P3) Cash (solo FCF en v1)
    fcf_ttm                     NUMERIC(20,2),   -- absoluto (moneda reporte)
    fcf_margin_ttm              NUMERIC(18,6),
    fcf_qoq_pct                 NUMERIC(18,6),
    fcf_yoy_pct                 NUMERIC(18,6),

    -- Estructura / solvencia (pasivo corriente)
    current_ratio               NUMERIC(18,6),   -- activo corr / pasivo corr
    working_capital             NUMERIC(20,2),   -- CA - CL (absoluto)
    debt_to_equity              NUMERIC(18,6),
    net_debt                    NUMERIC(20,2),   -- absoluto
    net_debt_to_ebitda_ttm      NUMERIC(18,6),

    -- Agregados TTM crudos (contexto, moneda de reporte)
    revenue_ttm                 NUMERIC(20,2),
    net_income_ttm              NUMERIC(20,2),
    ebitda_ttm                  NUMERIC(20,2),

    -- META
    n_quarters_available        SMALLINT,
    computed_at                 TIMESTAMP     NOT NULL DEFAULT NOW(),

    CONSTRAINT fundamentales_ratios_q_uniq UNIQUE (ticker, fiscal_period_end)
)
"""

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_fund_ratios_ticker_fecha "
    "ON fundamentales_ratios_q (ticker, fiscal_period_end)",
    "CREATE INDEX IF NOT EXISTS idx_fund_ratios_fecha "
    "ON fundamentales_ratios_q (fiscal_period_end)",
    "CREATE INDEX IF NOT EXISTS idx_fund_ratios_sector "
    "ON fundamentales_ratios_q (sector)",
]


def crear_en(engine, etiqueta: str, dry_run: bool):
    tabla = "fundamentales_ratios_q"
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema='public' AND table_name=:t
            )
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
        description="Crea la tabla fundamentales_ratios_q en Railway y/o local"
    )
    parser.add_argument("--target", choices=["railway", "local", "both"],
                        default="local", help="Donde crear (default: local)")
    parser.add_argument("--dry-run", action="store_true", help="Simula sin crear")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  CREATE fundamentales_ratios_q  |  target={args.target}"
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
