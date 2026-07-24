"""
create_earnings_historico_table.py
Crea la tabla earnings_historico en la DB LOCAL (one-shot).

Contexto (Tarea earnings-reaccion, 2026-07):
    Para analizar el comportamiento del precio/volumen en las ruedas posteriores
    a cada balance necesitamos la FECHA DE ANUNCIO historica por trimestre. Las
    tablas que ya teniamos NO la dan:
      - earnings_calendar: solo la PROXIMA fecha (1 fila/ticker).
      - fundamentales_*_q: fiscal_period_end = CIERRE del trimestre, no el
        anuncio (hay ~2-6 semanas de diferencia).
    La fuente es Alpha Vantage (funcion EARNINGS): reportedDate (anuncio) +
    fiscalDateEnding (cierre) + reportTime (pre/post-market), historia larga.

Diseno:
    ticker             VARCHAR      -- ticker del universo
    fiscal_period_end  DATE         -- cierre del Q (= fundamentales_*_q.fiscal_period_end)
    announcement_date  DATE         -- fecha REAL del anuncio del balance (reportedDate)
    report_time        VARCHAR      -- 'pre-market' | 'post-market' | 'time-not-supplied'
    fetched_at         TIMESTAMP    -- cuando se trajo esta fila
    PK (ticker, fiscal_period_end)

    El JOIN natural con fundamentales_*_q es por (ticker, fiscal_period_end):
    misma clave, asi la reaccion del precio se cruza con el fundamental del Q.

    report_time define el DIA 0 de la ventana: si 'pre-market' la reaccion es
    el mismo announcement_date; si 'post-market' es la rueda habil siguiente
    (el mercado ya estaba cerrado cuando se publico). La logica del dia 0 vive
    en el consumidor (dashboard/util), no en la tabla.

LOCAL-only (Plan C: dato historico recuperable de una API, no necesita Railway).

Uso:
    python scripts/oneshot/create_earnings_historico_table.py
    python scripts/oneshot/create_earnings_historico_table.py --dry-run
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


def _parse_env_file(path: str) -> dict:
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


def get_local_engine():
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    name = env.get("DB_NAME", "activos_ml")
    user = env.get("DB_USER", "postgres")
    pwd  = env.get("DB_PASSWORD", "")
    return create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}")


DDL = [
    """
    CREATE TABLE IF NOT EXISTS earnings_historico (
        ticker             VARCHAR(20) NOT NULL,
        fiscal_period_end  DATE        NOT NULL,
        announcement_date  DATE        NOT NULL,
        report_time        VARCHAR(24),
        fetched_at         TIMESTAMP   NOT NULL DEFAULT NOW(),
        PRIMARY KEY (ticker, fiscal_period_end)
    )
    """,
    # Indice por (ticker, announcement_date): la vista arma la serie de eventos
    # de un ticker ordenada por fecha de anuncio.
    "CREATE INDEX IF NOT EXISTS idx_earnings_hist_ticker_anndate "
    "ON earnings_historico (ticker, announcement_date)",
]


def main():
    parser = argparse.ArgumentParser(
        description="Crea la tabla earnings_historico en la DB local")
    parser.add_argument("--dry-run", action="store_true", help="Simula sin crear")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  CREATE earnings_historico (LOCAL)"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    engine = get_local_engine()
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'earnings_historico'
            )
        """)).scalar()

        if existe:
            n = conn.execute(text("SELECT COUNT(*) FROM earnings_historico")).scalar()
            log(f"Ya existe ({n} filas). Nada que crear.")
            return

        if args.dry_run:
            log("DRY-RUN: crearia la tabla earnings_historico.")
            return

        for stmt in DDL:
            conn.execute(text(stmt.strip()))
        conn.commit()
        log("Tabla earnings_historico creada (0 filas).")

    print()
    log("Completado.")
    print()


if __name__ == "__main__":
    main()
