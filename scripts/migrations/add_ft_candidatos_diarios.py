"""
add_ft_candidatos_diarios.py
Crea la tabla ft_candidatos_diarios.

Proposito:
    Registrar TODOS los candidatos que cumplen condiciones de entrada cada dia,
    tanto los que efectivamente entraron como los que quedaron afuera por limite
    de capital (80%) o de posiciones (max 5).

    Permite visualizar en Streamlit las "Oportunidades": activos con senal
    persistente que no pudimos capturar por restriccion de capital.

Uso:
    python scripts/migrations/add_ft_candidatos_diarios.py
    python scripts/migrations/add_ft_candidatos_diarios.py --status
    python scripts/migrations/add_ft_candidatos_diarios.py --dry-run
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from datetime import datetime
from sqlalchemy import text
from src.data.database import get_engine


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


DDL = """
CREATE TABLE IF NOT EXISTS ft_candidatos_diarios (
    id              SERIAL PRIMARY KEY,
    estrategia_id   INT          NOT NULL REFERENCES ft_estrategias(id),
    fecha           DATE         NOT NULL,
    ticker          VARCHAR(20)  NOT NULL,
    score           NUMERIC(6,3) NOT NULL,
    entro           BOOLEAN      NOT NULL DEFAULT FALSE,
    motivo_skip     VARCHAR(80),
    UNIQUE(estrategia_id, fecha, ticker)
);

CREATE INDEX IF NOT EXISTS idx_ft_cand_eid_fecha
    ON ft_candidatos_diarios(estrategia_id, fecha DESC);
"""


def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'ft_candidatos_diarios'
        """)).scalar()

        if not existe:
            log("Tabla ft_candidatos_diarios NO existe todavia.")
            return

        rows = conn.execute(text("""
            SELECT e.nombre, c.fecha, COUNT(*) AS total,
                   SUM(CASE WHEN c.entro THEN 1 ELSE 0 END) AS abiertos,
                   SUM(CASE WHEN NOT c.entro THEN 1 ELSE 0 END) AS oportunidades
            FROM ft_candidatos_diarios c
            JOIN ft_estrategias e ON e.id = c.estrategia_id
            GROUP BY e.nombre, c.fecha
            ORDER BY e.nombre, c.fecha DESC
        """)).fetchall()

        print()
        print("  === ft_candidatos_diarios ===")
        print(f"  {'ESTRATEGIA':<22} {'FECHA':<12} {'TOTAL':>6} {'ABIERTOS':>9} {'OPORT.':>7}")
        print("  " + "-" * 60)
        for r in rows:
            print(f"  {r[0]:<22} {str(r[1]):<12} {r[2]:>6} {r[3]:>9} {r[4]:>7}")
        print()


def cmd_run(dry_run: bool = False):
    engine = get_engine()

    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'ft_candidatos_diarios'
        """)).scalar()

        if existe:
            log("Tabla ft_candidatos_diarios ya existe.")
            return

        if dry_run:
            log("[DRY RUN] Se crearia ft_candidatos_diarios.")
            print(DDL)
            return

        for stmt in DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
        log("Tabla ft_candidatos_diarios creada correctamente.")
        log("Indice idx_ft_cand_eid_fecha creado.")


def main():
    parser = argparse.ArgumentParser(
        description="Crea tabla ft_candidatos_diarios"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status",  action="store_true")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    else:
        cmd_run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
