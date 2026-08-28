"""
create_fundamentales_sec_tables.py
Crea las 3 tablas de la fuente SEC XBRL en local.

Contexto (docs/fuentes_fundamentales.md):
    Fuente PARALELA a yahooquery. Las dos conviven; mas adelante se decide
    cual sigue y cual se deprecia. NINGUN consumidor actual lee estas tablas:
    fundamentales_income_q y companiaa siguen intactas.

Por que UNA tabla de serie y no cuatro:
    yahooquery usa 4 (income/balance/cashflow/valuation) porque asi viene su
    API. SEC NO organiza por estado contable -- publica hechos sueltos, y el
    normalizador produce una fila por (ticker, periodo) con los 31 conceptos
    juntos. Volver a partirla seria re-imponerle a SEC la forma de otra fuente,
    y esa particion es justo donde aparece la ambiguedad de "a que estado
    pertenece este concepto".

Las 3 tablas:
    fundamentales_sec_q        la serie normalizada, 1 fila por ticker+periodo
    fundamentales_sec_avisos   los avisos del normalizador, CONSULTABLES
    fundamentales_sec_ingesta  control de descarga por ticker (incremental)

Decisiones de diseno:
    - `origen` JSONB en vez de 62 columnas: guarda {concepto: {tag, derivado}}.
      Es el rastro de auditoria -- que tag produjo cada numero y si se derivo
      por desacumulacion. Es lo que se mira cuando un valor no cuadra. Misma
      idea que el raw_json de las fundamentales_*.
    - Los avisos van en tabla propia, no en JSONB: la consulta util es "todos
      los tickers con cambio_de_tag en revenue", y eso contra JSONB es incomodo.
    - El POINT-IN-TIME no se almacena como filas multiples. La tabla guarda la
      vista vigente (ultimo `filed`); el point-in-time se RE-DERIVA corriendo
      normalizar(hasta_filed=...) sobre el cache en disco. Guardar cada version
      duplicaria las filas por una capacidad que todavia no se consume.

Las columnas de concepto se generan DESDE src/utils/sec_xbrl.py para que el
esquema no pueda desincronizarse del normalizador.

Uso:
    python scripts/oneshot/create_fundamentales_sec_tables.py
    python scripts/oneshot/create_fundamentales_sec_tables.py --dry-run
"""

import sys
import os
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import text

from scripts.oneshot.create_fundamentales_tables import get_local_engine
from src.utils.sec_xbrl import CONCEPTOS, FLUJO_PONDERADO

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _tipo(concepto: str) -> str:
    """
    Tipo SQL de cada concepto.
      - importes: NUMERIC(24,2). El activo de JPM ronda 5e12; 24 digitos sobra.
      - acciones: NUMERIC(20,2). Conteos, no importes.
      - por accion: NUMERIC(20,6). Necesita decimales.
    """
    if concepto.startswith("eps"):
        return "NUMERIC(20,6)"
    if concepto.startswith("shares"):
        return "NUMERIC(20,2)"
    return "NUMERIC(24,2)"


def _columnas_concepto() -> str:
    lineas = []
    for concepto in CONCEPTOS:
        clase = CONCEPTOS[concepto]
        nota = {"aditivo": "flujo", "ponderado": "ponderado", "instante": "balance"}[clase]
        lineas.append(f"    {concepto:<24s} {_tipo(concepto):<16s},   -- {nota}")
    return "\n".join(lineas)


def ddl_serie() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS fundamentales_sec_q (
    id                       SERIAL          PRIMARY KEY,
    ticker                   VARCHAR(20)     NOT NULL,
    cik                      INTEGER,
    period_end               DATE            NOT NULL,
    fiscal_year              SMALLINT,
    fiscal_quarter           SMALLINT,

    -- Conceptos normalizados (generados desde src/utils/sec_xbrl.py)
{_columnas_concepto()}

    -- Rastro de auditoria: {{concepto: {{tag, derivado}}}}
    origen                   JSONB,
    -- 'filed' mas reciente que respalda la fila (procedencia / point-in-time)
    filed_max                DATE,
    computed_at              TIMESTAMP       NOT NULL DEFAULT NOW(),

    -- UNIQUE FULL (sin WHERE): requisito de ON CONFLICT. Ver CLAUDE.md.
    CONSTRAINT fundamentales_sec_q_uniq UNIQUE (ticker, period_end)
)
"""


DDL_AVISOS = """
CREATE TABLE IF NOT EXISTS fundamentales_sec_avisos (
    id            SERIAL       PRIMARY KEY,
    ticker        VARCHAR(20)  NOT NULL,
    tipo          VARCHAR(40)  NOT NULL,   -- cambio_de_tag | ponderado_discordante
                                           -- | ponderado_implausible
    concepto      VARCHAR(40)  NOT NULL,
    period_end    DATE,                    -- NULL en avisos de serie completa
    detalle       TEXT,
    tags          JSONB,
    computed_at   TIMESTAMP    NOT NULL DEFAULT NOW()
)
"""

DDL_INGESTA = """
CREATE TABLE IF NOT EXISTS fundamentales_sec_ingesta (
    ticker            VARCHAR(20)  PRIMARY KEY,
    cik               INTEGER      NOT NULL,
    -- accession del ultimo 10-Q/10-K visto. Es el DISPARADOR del incremental:
    -- submissions pesa ~164 KB contra ~3.8 MB de companyfacts, asi que se
    -- consulta ese primero y solo se baja el pesado si el accession cambio.
    ultimo_accn       VARCHAR(30),
    ultimo_form       VARCHAR(10),
    ultimo_filed      DATE,
    periodos          SMALLINT,
    periodo_min       DATE,
    periodo_max       DATE,
    avisos            SMALLINT,
    bytes_descargados BIGINT,
    error             TEXT,
    fetched_at        TIMESTAMP,
    updated_at        TIMESTAMP    NOT NULL DEFAULT NOW()
)
"""

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_fund_sec_ticker_fecha "
    "ON fundamentales_sec_q (ticker, period_end)",
    "CREATE INDEX IF NOT EXISTS idx_fund_sec_fecha "
    "ON fundamentales_sec_q (period_end)",
    "CREATE INDEX IF NOT EXISTS idx_fund_sec_fy "
    "ON fundamentales_sec_q (fiscal_year, fiscal_quarter)",
    "CREATE INDEX IF NOT EXISTS idx_fund_sec_avisos_tipo "
    "ON fundamentales_sec_avisos (tipo, concepto)",
    "CREATE INDEX IF NOT EXISTS idx_fund_sec_avisos_ticker "
    "ON fundamentales_sec_avisos (ticker)",
]

TABLAS = [("fundamentales_sec_q", ddl_serie),
          ("fundamentales_sec_avisos", lambda: DDL_AVISOS),
          ("fundamentales_sec_ingesta", lambda: DDL_INGESTA)]


def crear(engine, dry_run: bool):
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
            conn.execute(text(ddl().strip()))
            log(f"{tabla}: creada.")
        if not dry_run:
            for stmt in INDICES:
                conn.execute(text(stmt))
            conn.commit()
            log(f"indices verificados ({len(INDICES)}).")


def main():
    p = argparse.ArgumentParser(
        description="Crea las 3 tablas de la fuente SEC XBRL (LOCAL-only)")
    p.add_argument("--dry-run", action="store_true", help="Simula sin crear")
    p.add_argument("--mostrar-ddl", action="store_true", help="Imprime el DDL y sale")
    args = p.parse_args()

    if args.mostrar_ddl:
        print(ddl_serie())
        print(DDL_AVISOS)
        print(DDL_INGESTA)
        return

    print()
    print(SEP)
    print(f"  CREATE tablas fundamentales_sec_*{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()
    log(f"conceptos desde src/utils/sec_xbrl.py: {len(CONCEPTOS)} "
        f"({sum(1 for v in CONCEPTOS.values() if v == 'aditivo')} flujo aditivo, "
        f"{len(FLUJO_PONDERADO)} ponderado, "
        f"{sum(1 for v in CONCEPTOS.values() if v == 'instante')} balance)")
    crear(get_local_engine(), args.dry_run)
    print()
    print(SEP)
    print("  OK")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
