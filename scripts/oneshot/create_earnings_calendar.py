"""
create_earnings_calendar.py
Crea la tabla earnings_calendar en Railway y/o local.

Contexto:
    earnings_filter.py consultaba yfinance por cada ticker en cada corrida
    de cada bot (~1.500-1.800 llamadas/dia) -> rate limit. La solucion es
    cachear la proxima fecha de earnings en una tabla y refrescarla semanal.

Arquitectura (igual que opciones_snapshot):
    - Railway = fuente: el refresh semanal corre en Oracle cron y escribe aca.
    - Local   = copia: sync_railway_to_local.py baja la tabla (modo replace).
    - Los bots FT leen la copia local; los bots Alpaca leen la de Railway.
      earnings_filter usa get_engine() -> cada consumidor lee su propia DB.

Diseno de la tabla:
    ticker              VARCHAR PK   -- una fila por ticker
    earnings_date       DATE NULL    -- NULL = sin fecha conocida (fail-safe)
    fecha_actualizacion TIMESTAMP    -- cuando se refresco esta fila

    earnings_date es NULLABLE a proposito. Si yfinance no tiene la fecha de
    un ticker, la fila igual existe con earnings_date = NULL. NULL es inocuo
    en PostgreSQL y earnings_filter lo trata como "sin balance conocido":
    el ticker no entra al mapa de earnings y las estrategias siguen con sus
    otros criterios de salida sin truncarse.

Uso:
    python scripts/migrations/create_earnings_calendar.py --target both
    python scripts/migrations/create_earnings_calendar.py --target railway
    python scripts/migrations/create_earnings_calendar.py --target local
    python scripts/migrations/create_earnings_calendar.py --target both --dry-run
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


# ── Entorno ──────────────────────────────────────────────────────────────────

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


# ── DDL ──────────────────────────────────────────────────────────────────────

DDL = [
    """
    CREATE TABLE IF NOT EXISTS earnings_calendar (
        ticker              VARCHAR(20) PRIMARY KEY,
        earnings_date       DATE,
        fecha_actualizacion TIMESTAMP NOT NULL DEFAULT NOW()
    )
    """,
    # Indice por fecha: las consultas de salida filtran por proximidad de
    # earnings. Tabla chica (~199 filas) pero el indice es barato y prolijo.
    "CREATE INDEX IF NOT EXISTS idx_earnings_calendar_date "
    "ON earnings_calendar (earnings_date)",
]


def crear_en(engine, etiqueta: str, dry_run: bool):
    """Crea la tabla earnings_calendar en el engine dado (idempotente)."""
    log(f"[{etiqueta}] verificando earnings_calendar...")

    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'earnings_calendar'
            )
        """)).scalar()

        if existe:
            n = conn.execute(text("SELECT COUNT(*) FROM earnings_calendar")).scalar()
            log(f"[{etiqueta}] ya existe ({n} filas). Nada que crear.")
            return

        if dry_run:
            log(f"[{etiqueta}] DRY-RUN: crearia la tabla earnings_calendar.")
            return

        for stmt in DDL:
            conn.execute(text(stmt.strip()))
        conn.commit()
        log(f"[{etiqueta}] tabla earnings_calendar creada (0 filas).")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crea la tabla earnings_calendar en Railway y/o local"
    )
    parser.add_argument("--target", choices=["railway", "local", "both"],
                        default="both", help="Donde crear la tabla")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sin crear")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  CREATE earnings_calendar  |  target={args.target}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    if args.target in ("railway", "both"):
        crear_en(get_railway_engine(), "RAILWAY", args.dry_run)

    if args.target in ("local", "both"):
        crear_en(get_local_engine(), "LOCAL", args.dry_run)

    print()
    log("Completado.")
    print()


if __name__ == "__main__":
    main()
