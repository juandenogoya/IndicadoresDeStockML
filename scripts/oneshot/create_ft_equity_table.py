"""
create_ft_equity_table.py
Crea la tabla ft_equity_diaria: equity curve MARCADA A MERCADO por estrategia.

Contexto (2026-07-21, ver docs/forward_testing/METRICAS.md):
    ft_metricas_diarias.capital_total esta valuado a COSTO de entrada
    (capital_inmovilizado = SUM(capital_entrada)), no a mercado. Solo se mueve
    cuando se cierra una operacion -> es una curva de PnL realizado, no una
    equity curve. Toda metrica de riesgo sobre ella subestima el riesgo y el
    max drawdown de posiciones abiertas es invisible.

    ft_equity_diaria es una capa DERIVADA: funcion pura de ft_operaciones +
    precios_diarios, recomputable hacia atras sin datos nuevos. La escribe
    scripts/forward_testing/ft_compute_equity.py. Ningun bot escribe aca.

    ft_metricas_diarias NO se toca: sigue siendo el log operativo que leen
    ft_reporte_html.py y el MCP.

LOCAL-only (Plan C: FT corre 100% en local).

Uso:
    python scripts/oneshot/create_ft_equity_table.py            # crea
    python scripts/oneshot/create_ft_equity_table.py --dry-run  # muestra SQL
    python scripts/oneshot/create_ft_equity_table.py --status   # estado
    python scripts/oneshot/create_ft_equity_table.py --drop     # elimina (CUIDADO)
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forward_testing")))

from ft_env import configurar_entorno_local  # noqa: E402
configurar_entorno_local()

from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


DDL = """
CREATE TABLE IF NOT EXISTS ft_equity_diaria (
    estrategia_id      INTEGER NOT NULL REFERENCES ft_estrategias(id),
    fecha              DATE    NOT NULL,

    equity             NUMERIC(14,2) NOT NULL,
    cash               NUMERIC(14,2) NOT NULL,
    valor_mercado      NUMERIC(14,2) NOT NULL,
    costo_posiciones   NUMERIC(14,2) NOT NULL,

    n_posiciones       SMALLINT NOT NULL DEFAULT 0,
    exposicion_pct     NUMERIC(7,4),

    pnl_realizado_dia  NUMERIC(14,2) NOT NULL DEFAULT 0,
    pnl_no_realizado   NUMERIC(14,2) NOT NULL DEFAULT 0,
    retorno_dia_pct    NUMERIC(9,6),
    retorno_acum_pct   NUMERIC(9,4),

    precios_stale      SMALLINT NOT NULL DEFAULT 0,
    calculado_en       TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (estrategia_id, fecha)
);

CREATE INDEX IF NOT EXISTS idx_ft_equity_fecha
    ON ft_equity_diaria(fecha);

COMMENT ON TABLE ft_equity_diaria IS
    'Equity curve diaria MARCADA A MERCADO por estrategia. Capa derivada: '
    'funcion pura de ft_operaciones + precios_diarios, recomputable. '
    'Solo dias habiles NYSE, sin huecos. Ver docs/forward_testing/METRICAS.md';
COMMENT ON COLUMN ft_equity_diaria.equity IS
    'cash + valor_mercado. El valor REAL de la estrategia ese dia.';
COMMENT ON COLUMN ft_equity_diaria.valor_mercado IS
    'SUM(close(ticker, fecha) * cantidad) de las posiciones abiertas al cierre.';
COMMENT ON COLUMN ft_equity_diaria.costo_posiciones IS
    'SUM(capital_entrada) de las abiertas = cost basis. Comparar contra '
    'valor_mercado da el PnL no realizado.';
COMMENT ON COLUMN ft_equity_diaria.precios_stale IS
    'Cuantas posiciones se marcaron con un close arrastrado (forward-fill) '
    'porque el ticker no tenia precio ese dia. 0 = todos frescos.';
COMMENT ON COLUMN ft_equity_diaria.retorno_dia_pct IS
    'vs el dia habil ANTERIOR DE LA SERIE. NULL en la primera fila.';
"""


def crear(engine):
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    log("Tabla ft_equity_diaria creada (o ya existente).")


def estado(engine):
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'ft_equity_diaria'
            )
        """)).scalar()

        if not existe:
            log("ft_equity_diaria NO existe.")
            return

        row = conn.execute(text("""
            SELECT COUNT(*) AS n,
                   COUNT(DISTINCT estrategia_id) AS n_est,
                   MIN(fecha) AS desde, MAX(fecha) AS hasta
            FROM ft_equity_diaria
        """)).fetchone()

        log(f"ft_equity_diaria: {row.n} filas | {row.n_est} estrategias "
            f"| {row.desde} -> {row.hasta}")

        if row.n:
            det = conn.execute(text("""
                SELECT e.id, e.nombre, COUNT(*) AS dias,
                       MIN(q.fecha) AS desde, MAX(q.fecha) AS hasta,
                       MAX(q.retorno_acum_pct) AS mejor_acum
                FROM ft_equity_diaria q
                JOIN ft_estrategias e ON e.id = q.estrategia_id
                GROUP BY e.id, e.nombre ORDER BY e.id
            """)).fetchall()
            for d in det:
                print(f"   {d.id:>2} {d.nombre:<28} {d.dias:>3} dias  "
                      f"{d.desde} -> {d.hasta}")


def borrar(engine):
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ft_equity_diaria"))
        conn.commit()
    log("Tabla ft_equity_diaria eliminada.")


def main():
    ap = argparse.ArgumentParser(description="Crea la tabla ft_equity_diaria (LOCAL).")
    ap.add_argument("--dry-run", action="store_true", help="muestra el SQL sin ejecutar")
    ap.add_argument("--status", action="store_true", help="estado actual de la tabla")
    ap.add_argument("--drop", action="store_true", help="elimina la tabla (CUIDADO)")
    args = ap.parse_args()

    if args.dry_run:
        print(DDL)
        return

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")

    if args.status:
        estado(engine)
    elif args.drop:
        resp = input("Eliminar ft_equity_diaria? Se puede recomputar entera. [si/NO]: ")
        if resp.strip().lower() == "si":
            borrar(engine)
        else:
            log("Cancelado.")
    else:
        crear(engine)
        estado(engine)


if __name__ == "__main__":
    main()
