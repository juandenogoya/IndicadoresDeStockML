"""
create_fundamentales_tables.py
Crea las 4 tablas de analisis fundamental en Railway y/o local.

Contexto:
    Complemento del analisis tecnico/ML. Permite comparar Q actual vs Q-1
    (QoQ) y vs Q-4 (YoY, interanual) usando metricas crudas de los 3
    statements + ratios de valuacion pre-calculados por Yahoo.

Fuente de datos: yahooquery (gratis, cobertura completa del universo).
    Verificado: cubre US + ADRs de China/Brasil/Europa/LatAm en USD o
    moneda de reporte. FMP descartado: el plan free no cubre ADRs
    internacionales y `period=quarter` requiere plan Premium (USD 69/mo).

Tablas creadas (mismo schema en Railway y local):
    fundamentales_income_q     -- income statement trimestral
    fundamentales_balance_q    -- balance sheet trimestral
    fundamentales_cashflow_q   -- cash flow statement trimestral
    fundamentales_valuation_q  -- PE/PB/PS/PEG/EV ratios pre-calculados (3M + TTM)

Diseno:
    - PK natural: (ticker, fiscal_period_end) [valuation suma period_type]
    - Schema "wide": ~12-15 columnas dedicadas por tabla con las metricas
      mas usadas para ratios + columna raw_json (JSONB) con la respuesta
      completa de yahooquery. Esto evita re-fetch si mañana queremos una
      metrica nueva que hoy esta en JSONB.
    - Todas las metricas son NULLABLE: bancos no traen current_assets/
      current_liabilities/current_debt, empresas con perdida no tienen
      pe_ratio, etc. NULL es la señal honesta de "no aplica / no disponible".
    - reporting_currency por fila: ADRs (BABA, PBR, NIO, BIDU...) reportan
      en su moneda local, no en USD del ADR. Yahoo da `summaryDetail.currency`
      durante el refresh -- se copia tal cual aca. Esto deja la decision de
      normalizar (filtrar USD-only, convertir via FX, aceptar) al consumidor.

Cadencia de carga (separado, este script solo crea las tablas):
    - Cron Oracle: Domingo 12:00 UTC -> escribe a Railway (~5-10 min)
    - sync_railway_to_local.py: baja incrementalmente cada vez que corre

Restatements (caveat conocido):
    yahooquery puede revisar cifras de un Q ya reportado. El refresh
    semanal escribe con ON CONFLICT (ticker, fiscal_period_end) DO UPDATE,
    asi que Railway siempre tiene la version fresca. El sync incremental
    por fiscal_period_end NO trae correcciones a Q viejos. Si en algun
    momento detectamos discrepancias importantes, agregar un flag
    --full-resync al sync.

Uso:
    python scripts/oneshot/create_fundamentales_tables.py --target both
    python scripts/oneshot/create_fundamentales_tables.py --target railway
    python scripts/oneshot/create_fundamentales_tables.py --target local
    python scripts/oneshot/create_fundamentales_tables.py --target both --dry-run
"""

import sys
import os
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, text

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -- Entorno ------------------------------------------------------------------

def _parse_env_file(path: str) -> dict:
    """Lee un .env y retorna dict clave->valor (sin tocar os.environ)."""
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
    """SQLAlchemy engine -> Railway (lee .env.local directamente)."""
    env = _parse_env_file(os.path.join(ROOT, ".env.local"))
    db_url = env.get("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError("DATABASE_URL no encontrado en .env.local")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, connect_args={"sslmode": "require"})


def get_local_engine():
    """SQLAlchemy engine -> DB local (lee .env directamente)."""
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    name = env.get("DB_NAME", "activos_ml")
    user = env.get("DB_USER", "postgres")
    pwd  = env.get("DB_PASSWORD", "")
    return create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}")


# -- DDL ----------------------------------------------------------------------

DDL_INCOME = """
CREATE TABLE IF NOT EXISTS fundamentales_income_q (
    id                    SERIAL        PRIMARY KEY,
    ticker                VARCHAR(20)   NOT NULL,
    fiscal_period_end     DATE          NOT NULL,
    period_type           VARCHAR(5)    NOT NULL DEFAULT '3M',
    reporting_currency    VARCHAR(5),
    total_revenue         NUMERIC(20,2),
    cost_of_revenue       NUMERIC(20,2),
    gross_profit          NUMERIC(20,2),
    operating_expense     NUMERIC(20,2),
    operating_income      NUMERIC(20,2),
    ebit                  NUMERIC(20,2),
    ebitda                NUMERIC(20,2),
    interest_expense      NUMERIC(20,2),
    pretax_income         NUMERIC(20,2),
    tax_provision         NUMERIC(20,2),
    net_income            NUMERIC(20,2),
    diluted_eps           NUMERIC(14,6),
    basic_eps             NUMERIC(14,6),
    diluted_avg_shares    BIGINT,
    raw_json              JSONB,
    fetched_at            TIMESTAMP     NOT NULL DEFAULT NOW(),
    CONSTRAINT fundamentales_income_q_uniq UNIQUE (ticker, fiscal_period_end)
)
"""

DDL_BALANCE = """
CREATE TABLE IF NOT EXISTS fundamentales_balance_q (
    id                    SERIAL        PRIMARY KEY,
    ticker                VARCHAR(20)   NOT NULL,
    fiscal_period_end     DATE          NOT NULL,
    period_type           VARCHAR(5)    NOT NULL DEFAULT '3M',
    reporting_currency    VARCHAR(5),
    total_assets          NUMERIC(20,2),
    current_assets        NUMERIC(20,2),
    cash_and_equivalents  NUMERIC(20,2),
    inventory             NUMERIC(20,2),
    accounts_receivable   NUMERIC(20,2),
    total_liabilities     NUMERIC(20,2),
    current_liabilities   NUMERIC(20,2),
    accounts_payable      NUMERIC(20,2),
    current_debt          NUMERIC(20,2),
    long_term_debt        NUMERIC(20,2),
    total_debt            NUMERIC(20,2),
    stockholders_equity   NUMERIC(20,2),
    retained_earnings     NUMERIC(20,2),
    share_issued          BIGINT,
    raw_json              JSONB,
    fetched_at            TIMESTAMP     NOT NULL DEFAULT NOW(),
    CONSTRAINT fundamentales_balance_q_uniq UNIQUE (ticker, fiscal_period_end)
)
"""

DDL_CASHFLOW = """
CREATE TABLE IF NOT EXISTS fundamentales_cashflow_q (
    id                          SERIAL        PRIMARY KEY,
    ticker                      VARCHAR(20)   NOT NULL,
    fiscal_period_end           DATE          NOT NULL,
    period_type                 VARCHAR(5)    NOT NULL DEFAULT '3M',
    reporting_currency          VARCHAR(5),
    operating_cash_flow         NUMERIC(20,2),
    investing_cash_flow         NUMERIC(20,2),
    financing_cash_flow         NUMERIC(20,2),
    capital_expenditure         NUMERIC(20,2),
    free_cash_flow              NUMERIC(20,2),
    depreciation_amortization   NUMERIC(20,2),
    stock_based_compensation    NUMERIC(20,2),
    repurchase_of_capital_stock NUMERIC(20,2),
    cash_dividends_paid         NUMERIC(20,2),
    change_in_cash              NUMERIC(20,2),
    net_income                  NUMERIC(20,2),
    raw_json                    JSONB,
    fetched_at                  TIMESTAMP     NOT NULL DEFAULT NOW(),
    CONSTRAINT fundamentales_cashflow_q_uniq UNIQUE (ticker, fiscal_period_end)
)
"""

# valuation_q incluye period_type en la PK porque conviven '3M' (trimestre puro,
# uno por Q reportado) y 'TTM' (rolling, mas reciente). Filtrar por period_type
# en queries segun convenga.
DDL_VALUATION = """
CREATE TABLE IF NOT EXISTS fundamentales_valuation_q (
    id                         SERIAL        PRIMARY KEY,
    ticker                     VARCHAR(20)   NOT NULL,
    period_end                 DATE          NOT NULL,
    period_type                VARCHAR(5)    NOT NULL,
    pe_ratio                   NUMERIC(14,4),
    pb_ratio                   NUMERIC(14,4),
    ps_ratio                   NUMERIC(14,4),
    peg_ratio                  NUMERIC(14,4),
    enterprise_value_revenue   NUMERIC(14,4),
    enterprise_value_ebitda    NUMERIC(14,4),
    enterprise_value           NUMERIC(20,2),
    market_cap                 NUMERIC(20,2),
    forward_pe_ratio           NUMERIC(14,4),
    raw_json                   JSONB,
    fetched_at                 TIMESTAMP     NOT NULL DEFAULT NOW(),
    CONSTRAINT fundamentales_valuation_q_uniq UNIQUE (ticker, period_end, period_type)
)
"""

# Lista (tabla, ddl_create, lista_indices) -- mismo patron para las 4
TABLAS = [
    (
        "fundamentales_income_q", DDL_INCOME,
        [
            "CREATE INDEX IF NOT EXISTS idx_fund_income_ticker_fecha "
            "ON fundamentales_income_q (ticker, fiscal_period_end)",
            "CREATE INDEX IF NOT EXISTS idx_fund_income_fecha "
            "ON fundamentales_income_q (fiscal_period_end)",
        ],
    ),
    (
        "fundamentales_balance_q", DDL_BALANCE,
        [
            "CREATE INDEX IF NOT EXISTS idx_fund_balance_ticker_fecha "
            "ON fundamentales_balance_q (ticker, fiscal_period_end)",
            "CREATE INDEX IF NOT EXISTS idx_fund_balance_fecha "
            "ON fundamentales_balance_q (fiscal_period_end)",
        ],
    ),
    (
        "fundamentales_cashflow_q", DDL_CASHFLOW,
        [
            "CREATE INDEX IF NOT EXISTS idx_fund_cashflow_ticker_fecha "
            "ON fundamentales_cashflow_q (ticker, fiscal_period_end)",
            "CREATE INDEX IF NOT EXISTS idx_fund_cashflow_fecha "
            "ON fundamentales_cashflow_q (fiscal_period_end)",
        ],
    ),
    (
        "fundamentales_valuation_q", DDL_VALUATION,
        [
            "CREATE INDEX IF NOT EXISTS idx_fund_valuation_ticker_fecha "
            "ON fundamentales_valuation_q (ticker, period_end)",
            "CREATE INDEX IF NOT EXISTS idx_fund_valuation_fecha "
            "ON fundamentales_valuation_q (period_end)",
            "CREATE INDEX IF NOT EXISTS idx_fund_valuation_period_type "
            "ON fundamentales_valuation_q (period_type)",
        ],
    ),
]


def crear_tabla(engine, etiqueta: str, tabla: str, ddl: str,
                indices: list, dry_run: bool):
    """Crea una tabla + sus indices en el engine dado (idempotente)."""
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = :t
            )
        """), {"t": tabla}).scalar()

        if existe:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()
            log(f"[{etiqueta}] {tabla}: ya existe ({n} filas). Verificando indices...")
            if dry_run:
                log(f"[{etiqueta}]   DRY-RUN: verificaria {len(indices)} indices.")
                return
            # Los indices son IF NOT EXISTS, idempotentes
            for stmt in indices:
                conn.execute(text(stmt))
            conn.commit()
            log(f"[{etiqueta}]   indices OK.")
            return

        if dry_run:
            log(f"[{etiqueta}] {tabla}: DRY-RUN, crearia tabla + {len(indices)} indices.")
            return

        conn.execute(text(ddl.strip()))
        for stmt in indices:
            conn.execute(text(stmt))
        conn.commit()
        log(f"[{etiqueta}] {tabla}: creada (0 filas) + {len(indices)} indices.")


def crear_en(engine, etiqueta: str, dry_run: bool):
    """Crea las 4 tablas en el engine dado."""
    log(f"[{etiqueta}] === fundamentales: 4 tablas ===")
    for tabla, ddl, indices in TABLAS:
        crear_tabla(engine, etiqueta, tabla, ddl, indices, dry_run)


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crea las 4 tablas fundamentales_* en Railway y/o local"
    )
    parser.add_argument("--target", choices=["railway", "local", "both"],
                        default="both", help="Donde crear las tablas")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sin crear")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  CREATE fundamentales (4 tablas)  |  target={args.target}"
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
