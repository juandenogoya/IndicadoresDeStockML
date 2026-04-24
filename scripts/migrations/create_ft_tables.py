"""
create_ft_tables.py
Crea las tablas del motor de forward-testing (prefijo ft_).

Tablas:
    ft_estrategias      -- registro de instancias de estrategia
    ft_operaciones      -- trades virtuales (entrada + salida)
    ft_metricas_diarias -- equity curve diaria por estrategia

Uso:
    python scripts/migrations/create_ft_tables.py           # crea tablas
    python scripts/migrations/create_ft_tables.py --dry-run # muestra SQL sin ejecutar
    python scripts/migrations/create_ft_tables.py --status  # estado actual de las tablas
    python scripts/migrations/create_ft_tables.py --drop    # elimina tablas (CUIDADO)
"""

import sys
import os
import argparse
from datetime import datetime

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

from sqlalchemy import text
from src.data.database import get_engine


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── DDL ──────────────────────────────────────────────────────────────────────

DDL_ESTRATEGIAS = """
CREATE TABLE IF NOT EXISTS ft_estrategias (
    id                   SERIAL PRIMARY KEY,
    nombre               TEXT NOT NULL UNIQUE,
    descripcion          TEXT,
    logica               TEXT NOT NULL,
    parametros           JSONB,
    capital_inicial      NUMERIC(12,2) NOT NULL DEFAULT 100000,
    capital_actual       NUMERIC(12,2) NOT NULL DEFAULT 100000,
    cash_disponible      NUMERIC(12,2) NOT NULL DEFAULT 100000,
    capital_inmovilizado NUMERIC(12,2) NOT NULL DEFAULT 0,
    activa               BOOLEAN NOT NULL DEFAULT TRUE,
    fecha_inicio         DATE,
    fecha_fin            DATE,
    creado_en            TIMESTAMP DEFAULT NOW()
);
COMMENT ON TABLE ft_estrategias IS
    'Registro de instancias de estrategia para forward-testing. Una fila por experimento.';
COMMENT ON COLUMN ft_estrategias.capital_actual IS
    'cash_disponible + capital_inmovilizado. Se actualiza al cerrar cada operacion.';
COMMENT ON COLUMN ft_estrategias.capital_inmovilizado IS
    'Suma de (precio_entrada * cantidad) de posiciones abiertas.';
COMMENT ON COLUMN ft_estrategias.parametros IS
    'JSON con parametros especificos de la instancia. Ej: {score_min: 65, rsi_max: 68}';
"""

DDL_OPERACIONES = """
CREATE TABLE IF NOT EXISTS ft_operaciones (
    id               SERIAL PRIMARY KEY,
    estrategia_id    INT NOT NULL REFERENCES ft_estrategias(id),
    ticker           TEXT NOT NULL,
    lado             TEXT NOT NULL DEFAULT 'long',

    -- Entrada
    fecha_entrada    DATE NOT NULL,
    precio_entrada   NUMERIC(12,4) NOT NULL,
    cantidad         INT NOT NULL,
    capital_entrada  NUMERIC(12,2) NOT NULL,
    stop_loss        NUMERIC(12,4),
    take_profit      NUMERIC(12,4),
    score_entrada    NUMERIC(6,2),
    detalle_entrada  JSONB,

    -- Salida (NULL mientras la posicion esta abierta)
    fecha_salida     DATE,
    precio_salida    NUMERIC(12,4),
    pnl              NUMERIC(12,2),
    pnl_pct          NUMERIC(8,4),
    motivo_salida    TEXT,

    creado_en        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ft_op_estrategia
    ON ft_operaciones(estrategia_id);
CREATE INDEX IF NOT EXISTS idx_ft_op_ticker
    ON ft_operaciones(ticker);
CREATE INDEX IF NOT EXISTS idx_ft_op_abiertas
    ON ft_operaciones(estrategia_id) WHERE fecha_salida IS NULL;

COMMENT ON TABLE ft_operaciones IS
    'Trades virtuales del forward-testing. Posicion abierta = fecha_salida IS NULL.';
COMMENT ON COLUMN ft_operaciones.motivo_salida IS
    'Valores: EARNINGS_MANANA, TRAILING_SL, CHOCH_BEAR, ESTRUCTURA_ROTA,
     TIME_STOP_Nd, SCORE_DEGRADADO_X, STOP_LOSS_EMERGENCIA, TAKE_PROFIT';
COMMENT ON COLUMN ft_operaciones.detalle_entrada IS
    'Snapshot de condiciones al entrar. Ej: {cond_sma50: true, rsi: 58.3, ...}';
"""

DDL_METRICAS = """
CREATE TABLE IF NOT EXISTS ft_metricas_diarias (
    estrategia_id         INT NOT NULL REFERENCES ft_estrategias(id),
    fecha                 DATE NOT NULL,
    capital_total         NUMERIC(12,2) NOT NULL,
    cash_disponible       NUMERIC(12,2) NOT NULL,
    capital_inmovilizado  NUMERIC(12,2) NOT NULL,
    posiciones_abiertas   INT NOT NULL DEFAULT 0,
    operaciones_cerradas_dia INT NOT NULL DEFAULT 0,
    pnl_dia               NUMERIC(12,2) NOT NULL DEFAULT 0,
    retorno_acumulado_pct NUMERIC(8,4),

    PRIMARY KEY (estrategia_id, fecha)
);

CREATE INDEX IF NOT EXISTS idx_ft_metricas_fecha
    ON ft_metricas_diarias(fecha);

COMMENT ON TABLE ft_metricas_diarias IS
    'Equity curve diaria por estrategia. Una fila por estrategia por dia.';
COMMENT ON COLUMN ft_metricas_diarias.retorno_acumulado_pct IS
    '(capital_total - capital_inicial) / capital_inicial * 100';
"""

TABLAS = [
    ("ft_estrategias",      DDL_ESTRATEGIAS),
    ("ft_operaciones",      DDL_OPERACIONES),
    ("ft_metricas_diarias", DDL_METRICAS),
]


# ── Comandos ─────────────────────────────────────────────────────────────────

def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        log("Estado de tablas ft_* en la DB:")
        print()
        for nombre, _ in TABLAS:
            existe = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = :nombre
            """), {"nombre": nombre}).scalar()

            if existe:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {nombre}")).scalar()
                print(f"  [OK]  {nombre:<30s}  {n:>6d} filas")
            else:
                print(f"  [--]  {nombre:<30s}  (no existe)")
        print()


def cmd_create(dry_run: bool = False):
    engine = get_engine()
    log(f"Creando tablas ft_* {'[DRY RUN] ' if dry_run else ''}...")
    print()

    for nombre, ddl in TABLAS:
        with engine.connect() as conn:
            existe = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = :nombre
            """), {"nombre": nombre}).scalar()

        if existe:
            log(f"  [SKIP]  {nombre} ya existe.")
            continue

        if dry_run:
            log(f"  [DRY RUN] CREATE TABLE {nombre}")
            print(ddl)
        else:
            with engine.connect() as conn:
                conn.execute(text(ddl))
                conn.commit()
            log(f"  [OK]    {nombre} creada.")

    print()
    if not dry_run:
        log("Listo. Verificando resultado:")
        cmd_status()


def cmd_drop():
    """Elimina las tablas en orden inverso (respeta FK). Solo para dev."""
    engine = get_engine()
    orden_drop = [
        "ft_metricas_diarias",
        "ft_operaciones",
        "ft_estrategias",
    ]
    log("ATENCION: eliminando tablas ft_* (orden FK-safe)...")
    print()
    with engine.connect() as conn:
        for nombre in orden_drop:
            conn.execute(text(f"DROP TABLE IF EXISTS {nombre} CASCADE"))
            log(f"  [DROP]  {nombre}")
        conn.commit()
    print()
    log("Tablas eliminadas.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crea tablas ft_* para forward-testing"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra DDL sin ejecutar")
    parser.add_argument("--status", action="store_true",
                        help="Estado actual de las tablas")
    parser.add_argument("--drop", action="store_true",
                        help="Elimina las tablas ft_* (IRREVERSIBLE en prod)")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.drop:
        confirmacion = input("Escribi CONFIRMAR para eliminar las tablas ft_*: ")
        if confirmacion.strip() == "CONFIRMAR":
            cmd_drop()
        else:
            log("Cancelado.")
    else:
        cmd_create(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
