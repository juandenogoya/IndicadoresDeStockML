"""
bt_create_schema.py
Crea las tablas del motor de backtesting historico (prefijo bt_hist_).

Tablas:
    bt_hist_estrategias      -- instancias de BT con parametros y metricas resumen
    bt_hist_operaciones      -- libro de trades historicos simulados
    bt_hist_metricas_diarias -- equity curve dia a dia
    bt_hist_candidatos       -- candidatos evaluados por dia (entraron o no)

Relacion con ft_*:
    Las tablas bt_hist_* son independientes de ft_* (forward-testing en vivo).
    Mismo diseno conceptual, prefijo distinto para evitar mezcla de datos
    simulados historicamente con los del FT corriendo desde hoy.

Uso:
    python scripts/backtesting_historico/bt_create_schema.py            # crea tablas
    python scripts/backtesting_historico/bt_create_schema.py --dry-run  # muestra DDL
    python scripts/backtesting_historico/bt_create_schema.py --status   # estado actual
    python scripts/backtesting_historico/bt_create_schema.py --drop     # elimina (dev)
    python scripts/backtesting_historico/bt_create_schema.py --reset    # drop + create
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
CREATE TABLE IF NOT EXISTS bt_hist_estrategias (
    id                   SERIAL PRIMARY KEY,

    -- Identidad de la instancia
    nombre               TEXT NOT NULL,
    descripcion          TEXT,
    logica               TEXT NOT NULL,
    parametros           JSONB,

    -- Capital
    capital_inicial      NUMERIC(12,2) NOT NULL DEFAULT 100000,
    capital_final        NUMERIC(12,2),

    -- Periodo de backtesting
    fecha_bt_inicio      DATE NOT NULL,
    fecha_bt_fin         DATE NOT NULL,
    dias_habiles         SMALLINT,

    -- Configuracion y limitaciones
    sin_earnings_filter  BOOLEAN NOT NULL DEFAULT TRUE,
    universo_tickers     SMALLINT,
    precio_ejecucion     TEXT NOT NULL DEFAULT 'close_dia_senal',

    -- Metricas resumen (calculadas al finalizar el BT)
    retorno_total_pct    NUMERIC(8,4),
    drawdown_max_pct     NUMERIC(8,4),
    win_rate_pct         NUMERIC(6,2),
    profit_factor        NUMERIC(8,4),
    total_operaciones    INT,
    dias_promedio_pos    NUMERIC(6,2),
    sharpe_simplificado  NUMERIC(8,4),

    -- Estado de ejecucion
    estado               TEXT NOT NULL DEFAULT 'pendiente',
    ejecutado_en         TIMESTAMP,
    creado_en            TIMESTAMP DEFAULT NOW(),

    UNIQUE (nombre, fecha_bt_inicio, fecha_bt_fin)
);

COMMENT ON TABLE bt_hist_estrategias IS
    'Instancias de backtesting historico. Una fila por corrida (estrategia + periodo).';
COMMENT ON COLUMN bt_hist_estrategias.estado IS
    'Valores: pendiente | ejecutando | completado | error';
COMMENT ON COLUMN bt_hist_estrategias.sin_earnings_filter IS
    'TRUE siempre en esta fase. yfinance retorna fechas futuras para periodos pasados.';
COMMENT ON COLUMN bt_hist_estrategias.precio_ejecucion IS
    'close_dia_senal = precio de cierre del dia de la senal (supuesto aceptado).';
"""

DDL_OPERACIONES = """
CREATE TABLE IF NOT EXISTS bt_hist_operaciones (
    id               SERIAL PRIMARY KEY,
    bt_id            INT NOT NULL REFERENCES bt_hist_estrategias(id),
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

    -- Salida (NULL si posicion abierta al final del periodo)
    fecha_salida     DATE,
    precio_salida    NUMERIC(12,4),
    pnl              NUMERIC(12,2),
    pnl_pct          NUMERIC(8,4),
    motivo_salida    TEXT,
    dias_abierta     SMALLINT,

    creado_en        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bt_op_bt_id
    ON bt_hist_operaciones(bt_id);
CREATE INDEX IF NOT EXISTS idx_bt_op_ticker
    ON bt_hist_operaciones(ticker);
CREATE INDEX IF NOT EXISTS idx_bt_op_abiertas
    ON bt_hist_operaciones(bt_id) WHERE fecha_salida IS NULL;

COMMENT ON TABLE bt_hist_operaciones IS
    'Trades simulados del backtesting historico. fecha_salida IS NULL = abierta al fin del periodo.';
COMMENT ON COLUMN bt_hist_operaciones.motivo_salida IS
    'Valores: TRAILING_SL | CHOCH_BEAR | ESTRUCTURA_ROTA | TIME_STOP_20D |
     SCORE_DEGRADADO | STOP_LOSS_ATR | TAKE_PROFIT_ATR | FIN_PERIODO';
COMMENT ON COLUMN bt_hist_operaciones.dias_abierta IS
    'Dias habiles entre fecha_entrada y fecha_salida (o fin del periodo).';
"""

DDL_METRICAS = """
CREATE TABLE IF NOT EXISTS bt_hist_metricas_diarias (
    bt_id                    INT NOT NULL REFERENCES bt_hist_estrategias(id),
    fecha                    DATE NOT NULL,
    capital_total            NUMERIC(12,2) NOT NULL,
    cash_disponible          NUMERIC(12,2) NOT NULL,
    capital_inmovilizado     NUMERIC(12,2) NOT NULL,
    posiciones_abiertas      INT NOT NULL DEFAULT 0,
    operaciones_cerradas_dia INT NOT NULL DEFAULT 0,
    pnl_dia                  NUMERIC(12,2) NOT NULL DEFAULT 0,
    retorno_acumulado_pct    NUMERIC(8,4),
    drawdown_pct             NUMERIC(8,4),

    PRIMARY KEY (bt_id, fecha)
);

CREATE INDEX IF NOT EXISTS idx_bt_metricas_fecha
    ON bt_hist_metricas_diarias(fecha);

COMMENT ON TABLE bt_hist_metricas_diarias IS
    'Equity curve diaria del backtesting. Una fila por instancia BT por dia habil.';
COMMENT ON COLUMN bt_hist_metricas_diarias.drawdown_pct IS
    '(capital_total - max_capital_hasta_hoy) / max_capital_hasta_hoy * 100. Siempre <= 0.';
"""

DDL_CANDIDATOS = """
CREATE TABLE IF NOT EXISTS bt_hist_candidatos (
    id              SERIAL PRIMARY KEY,
    bt_id           INT NOT NULL REFERENCES bt_hist_estrategias(id),
    fecha           DATE NOT NULL,
    ticker          TEXT NOT NULL,
    score           NUMERIC(6,3) NOT NULL,
    entro           BOOLEAN NOT NULL DEFAULT FALSE,
    motivo_skip     TEXT,
    precio_apertura NUMERIC(12,4),
    retorno_5d      NUMERIC(8,4),
    retorno_10d     NUMERIC(8,4),
    retorno_20d     NUMERIC(8,4),

    UNIQUE (bt_id, fecha, ticker)
);

CREATE INDEX IF NOT EXISTS idx_bt_cand_bt_id_fecha
    ON bt_hist_candidatos(bt_id, fecha DESC);

COMMENT ON TABLE bt_hist_candidatos IS
    'Candidatos evaluados por dia en el BT. Incluye los que no entraron (para counterfactual).';
COMMENT ON COLUMN bt_hist_candidatos.retorno_5d IS
    'Retorno % calculado 5 dias habiles despues de precio_apertura. NULL si aun no disponible.';
"""

TABLAS = [
    ("bt_hist_estrategias",      DDL_ESTRATEGIAS),
    ("bt_hist_operaciones",      DDL_OPERACIONES),
    ("bt_hist_metricas_diarias", DDL_METRICAS),
    ("bt_hist_candidatos",       DDL_CANDIDATOS),
]

# Orden seguro para DROP (respetar FK)
TABLAS_DROP_ORDER = [
    "bt_hist_candidatos",
    "bt_hist_metricas_diarias",
    "bt_hist_operaciones",
    "bt_hist_estrategias",
]


# ── Comandos ─────────────────────────────────────────────────────────────────

def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        log("Estado de tablas bt_hist_* en la DB:")
        print()

        for nombre, _ in TABLAS:
            existe = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = :nombre AND table_schema = 'public'
            """), {"nombre": nombre}).scalar()

            if existe:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {nombre}")).scalar()
                print(f"  [OK]  {nombre:<35s}  {n:>6d} filas")
            else:
                print(f"  [--]  {nombre:<35s}  (no existe)")

        print()

        # Si hay instancias, mostrar resumen
        tiene_tabla = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'bt_hist_estrategias' AND table_schema = 'public'
        """)).scalar()

        if tiene_tabla:
            filas = conn.execute(text("""
                SELECT e.id, e.nombre, e.logica, e.estado,
                       e.fecha_bt_inicio, e.fecha_bt_fin,
                       e.capital_inicial, e.capital_final,
                       e.retorno_total_pct, e.total_operaciones,
                       COUNT(o.id) AS ops_cargadas
                FROM bt_hist_estrategias e
                LEFT JOIN bt_hist_operaciones o ON o.bt_id = e.id
                GROUP BY e.id
                ORDER BY e.id
            """)).fetchall()

            if filas:
                print(f"  {'ID':<4} {'NOMBRE':<22} {'LOGICA':<18} "
                      f"{'DESDE':<12} {'HASTA':<12} "
                      f"{'RETORNO':>8} {'OPS':>5} {'ESTADO'}")
                print("  " + "-" * 105)
                for r in filas:
                    ret = f"{float(r.retorno_total_pct):+.2f}%" if r.retorno_total_pct else "  N/A  "
                    print(f"  {r.id:<4} {r.nombre:<22} {r.logica:<18} "
                          f"{str(r.fecha_bt_inicio):<12} {str(r.fecha_bt_fin):<12} "
                          f"{ret:>8} {(r.ops_cargadas or 0):>5} {r.estado}")
                print()


def cmd_create(dry_run: bool = False):
    engine = get_engine()
    log(f"Creando tablas bt_hist_* {'[DRY RUN] ' if dry_run else ''}...")
    print()

    alguna_creada = False
    for nombre, ddl in TABLAS:
        with engine.connect() as conn:
            existe = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = :nombre AND table_schema = 'public'
            """), {"nombre": nombre}).scalar()

        if existe:
            log(f"  [SKIP]  {nombre} ya existe.")
            continue

        if dry_run:
            log(f"  [DRY RUN] CREATE TABLE {nombre}")
            print(ddl)
            print()
        else:
            with engine.connect() as conn:
                # Ejecutar cada statement por separado
                for stmt in ddl.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))
                conn.commit()
            log(f"  [OK]    {nombre} creada.")
            alguna_creada = True

    print()
    if not dry_run:
        if alguna_creada:
            log("Listo. Verificando resultado:")
        else:
            log("Todas las tablas ya existian.")
        cmd_status()


def cmd_drop(confirmar: bool = False):
    if not confirmar:
        respuesta = input(
            "ATENCION: esto elimina TODOS los datos de BT historico.\n"
            "Escribi CONFIRMAR para continuar: "
        )
        if respuesta.strip() != "CONFIRMAR":
            log("Cancelado.")
            return

    engine = get_engine()
    log("Eliminando tablas bt_hist_* (orden FK-safe)...")
    print()
    with engine.connect() as conn:
        for nombre in TABLAS_DROP_ORDER:
            conn.execute(text(f"DROP TABLE IF EXISTS {nombre} CASCADE"))
            log(f"  [DROP]  {nombre}")
        conn.commit()
    print()
    log("Tablas eliminadas.")


def cmd_reset():
    log("RESET: drop + create de tablas bt_hist_*")
    cmd_drop(confirmar=False)
    print()
    cmd_create(dry_run=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Administra el schema bt_hist_* para backtesting historico"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra DDL sin ejecutar nada")
    parser.add_argument("--status",  action="store_true",
                        help="Estado actual de las tablas bt_hist_*")
    parser.add_argument("--drop",    action="store_true",
                        help="Elimina todas las tablas bt_hist_* (pide confirmacion)")
    parser.add_argument("--reset",   action="store_true",
                        help="Drop + create (util durante desarrollo)")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.drop:
        cmd_drop()
    elif args.reset:
        cmd_reset()
    else:
        cmd_create(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
