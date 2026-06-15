"""
create_llm_uso_tokens_table.py
Crea la tabla llm_uso_tokens en la DB local.

Contexto:
    Registro de consumo de tokens del chat del dashboard (vista "Consultas (IA)",
    orquestador src/agent). Una fila por consulta: tokens de entrada/salida
    REALES que devuelve la API de Gemini (usage_metadata), agrupables por fecha.

    LOCAL-only (Plan C): es un log de uso del frontend local, no necesita Railway.
    Lo escribe el dashboard con el engine local normal (NO el rol mcp_reader, que
    es SELECT-only).

Diseno de la tabla:
    id              BIGSERIAL PK
    ts              TIMESTAMPTZ  -- momento exacto de la consulta
    fecha           DATE         -- para agrupar el consumo por dia
    usuario         TEXT         -- multiusuario a futuro (cuotas); default 'local'
    modelo          TEXT         -- ej. gemini-2.5-flash
    tokens_entrada  INTEGER      -- prompt_token_count (sumado por rondas)
    tokens_salida   INTEGER      -- candidates + thoughts (sumado por rondas)
    tokens_total    INTEGER
    n_rondas        INTEGER      -- nro de llamadas LLM de esa consulta
    tools           TEXT         -- tools usadas (auditoria)
    pregunta        TEXT         -- texto de la consulta (auditar cual gasto mas)

Uso:
    python scripts/oneshot/create_llm_uso_tokens_table.py
    python scripts/oneshot/create_llm_uso_tokens_table.py --dry-run
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


def get_local_engine():
    """SQLAlchemy engine -> DB local (lee .env directamente)."""
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    name = env.get("DB_NAME", "activos_ml")
    user = env.get("DB_USER", "postgres")
    pwd  = env.get("DB_PASSWORD", "")
    return create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}")


DDL = [
    """
    CREATE TABLE IF NOT EXISTS llm_uso_tokens (
        id              BIGSERIAL PRIMARY KEY,
        ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        fecha           DATE        NOT NULL DEFAULT CURRENT_DATE,
        usuario         TEXT        NOT NULL DEFAULT 'local',
        modelo          TEXT,
        tokens_entrada  INTEGER     NOT NULL DEFAULT 0,
        tokens_salida   INTEGER     NOT NULL DEFAULT 0,
        tokens_total    INTEGER     NOT NULL DEFAULT 0,
        n_rondas        INTEGER     NOT NULL DEFAULT 0,
        tools           TEXT,
        pregunta        TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_llm_uso_tokens_usuario_fecha "
    "ON llm_uso_tokens (usuario, fecha)",
]


def crear_en(engine, etiqueta: str, dry_run: bool):
    """Crea la tabla llm_uso_tokens en el engine dado (idempotente)."""
    log(f"[{etiqueta}] verificando llm_uso_tokens...")

    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'llm_uso_tokens'
            )
        """)).scalar()

        if existe:
            n = conn.execute(text("SELECT COUNT(*) FROM llm_uso_tokens")).scalar()
            log(f"[{etiqueta}] ya existe ({n} filas). Nada que crear.")
            return

        if dry_run:
            log(f"[{etiqueta}] DRY-RUN: crearia la tabla llm_uso_tokens.")
            return

        for stmt in DDL:
            conn.execute(text(stmt.strip()))
        conn.commit()
        log(f"[{etiqueta}] tabla llm_uso_tokens creada (0 filas).")


def main():
    parser = argparse.ArgumentParser(
        description="Crea la tabla llm_uso_tokens en la DB local"
    )
    parser.add_argument("--dry-run", action="store_true", help="Simula sin crear")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  CREATE llm_uso_tokens  |  target=local"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    crear_en(get_local_engine(), "LOCAL", args.dry_run)

    print()
    log("Completado.")
    print()


if __name__ == "__main__":
    main()
