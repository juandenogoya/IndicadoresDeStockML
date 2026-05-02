"""
init_tablas_ar.py
Crea las tablas para el modulo Argentina y siembra activos_ar.

Tablas creadas:
  activos_ar            -- master list de tickers BCBA
  alertas_scanner_ar    -- senales diarias del scanner AR
  operaciones_bt_ar     -- trades simulados del backtest AR
  resultados_bt_ar      -- resumen de metricas por estrategia/ticker
  opciones_ar_gregas    -- snapshot diario de Greeks (solo strikes liquidos)

Uso:
  python scripts/migrations/init_tablas_ar.py           # crea tablas + siembra
  python scripts/migrations/init_tablas_ar.py --dry-run # muestra SQL sin ejecutar
  python scripts/migrations/init_tablas_ar.py --status  # muestra estado actual
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


# ── Definicion de tickers AR ─────────────────────────────────────────────────

ACTIVOS_AR = [
    # Bancos
    # BMA / LOMA / VALO: activo=False — IOL no devuelve historico via serieHistorica
    {"ticker": "BMA",   "nombre": "Banco Macro S.A.",              "sector": "Bancos",          "tiene_opciones": True,  "activo": False},
    {"ticker": "BBAR",  "nombre": "BBVA Argentina S.A.",           "sector": "Bancos",          "tiene_opciones": False, "activo": True},
    {"ticker": "SUPV",  "nombre": "Grupo Supervielle S.A.",        "sector": "Bancos",          "tiene_opciones": False, "activo": True},
    {"ticker": "GGAL",  "nombre": "Grupo Financiero Galicia S.A.", "sector": "Bancos",          "tiene_opciones": True,  "activo": True},
    {"ticker": "BHIP",  "nombre": "Banco Hipotecario S.A.",        "sector": "Bancos",          "tiene_opciones": False, "activo": True},
    {"ticker": "VALO",  "nombre": "Banco de Valores S.A.",         "sector": "Bancos",          "tiene_opciones": True,  "activo": False},
    # Finanzas / Mercados
    {"ticker": "BYMA",  "nombre": "Bolsas y Mercados Argentinos",  "sector": "Finanzas",        "tiene_opciones": True,  "activo": True},
    {"ticker": "IRSA",  "nombre": "IRSA Inversiones y Repr.",      "sector": "Inmobiliario",    "tiene_opciones": False, "activo": True},
    # Telecom / Medios
    {"ticker": "TECO2", "nombre": "Telecom Argentina S.A.",        "sector": "Telecom",         "tiene_opciones": False, "activo": True},
    {"ticker": "CVH",   "nombre": "Cablevision Holding S.A.",      "sector": "Telecom",         "tiene_opciones": False, "activo": True},
    {"ticker": "GCLA",  "nombre": "Grupo Clarin S.A.",             "sector": "Medios",          "tiene_opciones": False, "activo": True},
    # Tecnologia / Industrial
    {"ticker": "MIRG",  "nombre": "Mirgor S.A.C.I.F.",             "sector": "Tecnologia",      "tiene_opciones": True,  "activo": True},
    # A3 = Matba Rofex (no Aeropuertos); sin historico IOL -> activo=False
    {"ticker": "A3",    "nombre": "Matba Rofex S.A.",              "sector": "Finanzas",        "tiene_opciones": False, "activo": False},
    # Agro / Alimentos
    {"ticker": "CRES",  "nombre": "Cresud S.A.C.I.F. y A.",       "sector": "Agro",            "tiene_opciones": False, "activo": True},
    {"ticker": "MORI",  "nombre": "Morixe Hermanos S.A.",          "sector": "Alimentos",       "tiene_opciones": False, "activo": True},
    {"ticker": "LEDE",  "nombre": "Ledesma S.A.A.I.",              "sector": "Agro",            "tiene_opciones": False, "activo": True},
    {"ticker": "MOLI",  "nombre": "Molinos Agro S.A.",             "sector": "Agro",            "tiene_opciones": False, "activo": True},
    {"ticker": "MOLA",  "nombre": "Molinos de la Pampa",           "sector": "Agro",            "tiene_opciones": False, "activo": True},
    {"ticker": "HAVA",  "nombre": "Havanna Holding S.A.",          "sector": "Alimentos",       "tiene_opciones": False, "activo": True},
    {"ticker": "SAMI",  "nombre": "San Miguel S.A.",               "sector": "Agro",            "tiene_opciones": False, "activo": True},
    {"ticker": "SEMI",  "nombre": "Semillas Selectas S.A.",        "sector": "Agro",            "tiene_opciones": False, "activo": True},
    {"ticker": "AGRO",  "nombre": "Agrometal S.A.I.C.",            "sector": "Maquinaria",      "tiene_opciones": False, "activo": True},
    # Energia
    {"ticker": "YPFD",  "nombre": "YPF S.A.",                      "sector": "Energia",         "tiene_opciones": True,  "activo": True},
    {"ticker": "EDN",   "nombre": "Edenor S.A.",                   "sector": "Energia",         "tiene_opciones": True,  "activo": True},
    {"ticker": "PAMP",  "nombre": "Pampa Energia S.A.",            "sector": "Energia",         "tiene_opciones": True,  "activo": True},
    {"ticker": "TGSU2", "nombre": "Transportadora de Gas del Sur", "sector": "Gas",             "tiene_opciones": True,  "activo": True},
    {"ticker": "CEPU",  "nombre": "Central Puerto S.A.",           "sector": "Energia",         "tiene_opciones": True,  "activo": True},
    {"ticker": "TGNO4", "nombre": "Transportadora de Gas del Norte","sector": "Gas",            "tiene_opciones": False, "activo": True},
    {"ticker": "TRAN",  "nombre": "Transener S.A.",                "sector": "Energia",         "tiene_opciones": False, "activo": True},
    {"ticker": "METR",  "nombre": "Metrogas S.A.",                 "sector": "Gas",             "tiene_opciones": False, "activo": True},
    # Materiales
    {"ticker": "LOMA",  "nombre": "Loma Negra C.I.A.S.A.",        "sector": "Materiales",      "tiene_opciones": False, "activo": False},
    {"ticker": "TXAR",  "nombre": "Ternium Argentina S.A.",        "sector": "Materiales",      "tiene_opciones": True,  "activo": True},
    {"ticker": "ALUA",  "nombre": "Aluar Aluminio Arg. S.A.",      "sector": "Materiales",      "tiene_opciones": True,  "activo": True},
]


# ── DDL ──────────────────────────────────────────────────────────────────────

DDL_ACTIVOS_AR = """
CREATE TABLE IF NOT EXISTS activos_ar (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL UNIQUE,
    nombre          VARCHAR(100),
    sector          VARCHAR(50),
    tiene_opciones  BOOLEAN      DEFAULT FALSE,
    activo          BOOLEAN      DEFAULT TRUE,
    created_at      TIMESTAMP    DEFAULT NOW()
);
"""

DDL_ALERTAS_SCANNER_AR = """
CREATE TABLE IF NOT EXISTS alertas_scanner_ar (
    id              SERIAL PRIMARY KEY,
    scan_fecha      TIMESTAMP    DEFAULT NOW(),
    ticker          VARCHAR(10)  NOT NULL,
    sector          VARCHAR(50),
    precio_cierre   NUMERIC(14,4),
    precio_fecha    DATE,
    atr14           NUMERIC(14,4),
    -- Indicadores clave
    rsi14           NUMERIC(8,4),
    macd_hist       NUMERIC(14,6),
    dist_sma20      NUMERIC(10,4),
    dist_sma50      NUMERIC(10,4),
    dist_sma200     NUMERIC(10,4),
    adx             NUMERIC(8,4),
    -- Scoring
    score_ponderado NUMERIC(5,4),
    condiciones_ok  SMALLINT,
    -- Estructura de mercado
    estructura_10   SMALLINT,
    bos_bull_10     SMALLINT,
    bos_bear_10     SMALLINT,
    choch_bull_10   SMALLINT,
    choch_bear_10   SMALLINT,
    -- Output de alerta
    alert_score     NUMERIC(5,1),
    alert_nivel     VARCHAR(20),
    alert_detalle   TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS alertas_scanner_ar_ticker_dia_uniq
    ON alertas_scanner_ar (ticker, date(scan_fecha));
CREATE INDEX IF NOT EXISTS idx_alertas_ar_ticker_fecha
    ON alertas_scanner_ar (ticker, scan_fecha DESC);
CREATE INDEX IF NOT EXISTS idx_alertas_ar_nivel
    ON alertas_scanner_ar (alert_nivel, scan_fecha DESC);
"""

DDL_OPERACIONES_BT_AR = """
CREATE TABLE IF NOT EXISTS operaciones_bt_ar (
    id                  SERIAL PRIMARY KEY,
    estrategia_entrada  VARCHAR(10) NOT NULL,
    estrategia_salida   VARCHAR(10) NOT NULL,
    ticker              VARCHAR(10) NOT NULL,
    segmento            VARCHAR(15) NOT NULL,
    fecha_entrada       DATE,
    precio_entrada      NUMERIC(14,4),
    score_entrada       NUMERIC(5,4),
    stop_loss           NUMERIC(14,4),
    take_profit         NUMERIC(14,4),
    fecha_salida        DATE,
    precio_salida       NUMERIC(14,4),
    motivo_salida       VARCHAR(30),
    dias_posicion       INTEGER,
    retorno_pct         NUMERIC(10,4),
    resultado           VARCHAR(10)
);
CREATE INDEX IF NOT EXISTS idx_bt_ar_estrategia
    ON operaciones_bt_ar (estrategia_entrada, estrategia_salida, ticker);
"""

DDL_RESULTADOS_BT_AR = """
CREATE TABLE IF NOT EXISTS resultados_bt_ar (
    id                      SERIAL PRIMARY KEY,
    estrategia_entrada      VARCHAR(10) NOT NULL,
    estrategia_salida       VARCHAR(10) NOT NULL,
    ticker                  VARCHAR(10) NOT NULL,
    segmento                VARCHAR(15) NOT NULL,
    total_operaciones       INTEGER,
    ganancias               INTEGER,
    perdidas                INTEGER,
    win_rate                NUMERIC(6,4),
    retorno_promedio_pct    NUMERIC(10,4),
    retorno_total_pct       NUMERIC(10,4),
    max_drawdown_pct        NUMERIC(10,4),
    profit_factor           NUMERIC(10,4),
    dias_promedio_posicion  NUMERIC(8,2),
    UNIQUE (estrategia_entrada, estrategia_salida, ticker, segmento)
);
"""

DDL_OPCIONES_AR_GREGAS = """
CREATE TABLE IF NOT EXISTS opciones_ar_gregas (
    id                  SERIAL PRIMARY KEY,
    fecha               DATE         NOT NULL,
    ticker_subyacente   VARCHAR(10)  NOT NULL,
    symbol              VARCHAR(20)  NOT NULL,
    option_type         VARCHAR(1)   NOT NULL,
    strike_price        NUMERIC(14,2),
    expiration          DATE,
    bid_price           NUMERIC(14,4),
    ask_price           NUMERIC(14,4),
    theoretical_price   NUMERIC(14,4),
    implied_volatility  NUMERIC(8,4),
    delta               NUMERIC(8,4),
    gamma               NUMERIC(10,6),
    theta               NUMERIC(10,4),
    vega                NUMERIC(10,4),
    rho                 NUMERIC(10,4),
    volume              INTEGER,
    spot_price          NUMERIC(14,4),
    risk_free_rate      NUMERIC(8,4),
    created_at          TIMESTAMP    DEFAULT NOW(),
    UNIQUE (fecha, symbol)
);
CREATE INDEX IF NOT EXISTS idx_opciones_ar_ticker_fecha
    ON opciones_ar_gregas (ticker_subyacente, fecha DESC);
CREATE INDEX IF NOT EXISTS idx_opciones_ar_expiracion
    ON opciones_ar_gregas (expiration, ticker_subyacente);
"""

DDL_PCR_AR_DIARIO = """
CREATE TABLE IF NOT EXISTS pcr_ar_diario (
    id                  SERIAL PRIMARY KEY,
    fecha               DATE         NOT NULL,
    ticker              VARCHAR(10)  NOT NULL,
    expiration          DATE         NOT NULL,
    -- Volumenes del dia (snapshot cierre ~16:45 ART)
    vol_calls           INTEGER      DEFAULT 0,
    vol_puts            INTEGER      DEFAULT 0,
    pcr                 NUMERIC(8,4),
    -- Strikes activos
    n_strikes_calls     SMALLINT,
    n_strikes_puts      SMALLINT,
    -- IV ATM promedio (3 strikes mas cercanos al spot)
    iv_atm_call         NUMERIC(8,4),
    iv_atm_put          NUMERIC(8,4),
    -- Skew: IV put OTM / IV ATM (medida de tail risk)
    iv_skew             NUMERIC(8,4),
    -- Spot y tasa al momento del snapshot
    spot_price          NUMERIC(14,4),
    risk_free_rate      NUMERIC(8,4),
    -- Dias hasta vencimiento al momento del snapshot
    dias_vto            SMALLINT,
    created_at          TIMESTAMP    DEFAULT NOW(),
    UNIQUE (fecha, ticker, expiration)
);
CREATE INDEX IF NOT EXISTS idx_pcr_ar_ticker_fecha
    ON pcr_ar_diario (ticker, fecha DESC);
CREATE INDEX IF NOT EXISTS idx_pcr_ar_expiracion
    ON pcr_ar_diario (expiration, ticker);
"""

ALL_DDL = [
    ("activos_ar",          DDL_ACTIVOS_AR),
    ("alertas_scanner_ar",  DDL_ALERTAS_SCANNER_AR),
    ("operaciones_bt_ar",   DDL_OPERACIONES_BT_AR),
    ("resultados_bt_ar",    DDL_RESULTADOS_BT_AR),
    ("opciones_ar_gregas",  DDL_OPCIONES_AR_GREGAS),
    ("pcr_ar_diario",       DDL_PCR_AR_DIARIO),
]


# ── Comandos ──────────────────────────────────────────────────────────────────

def cmd_status(engine):
    log("Estado actual de tablas AR:")
    with engine.connect() as conn:
        for nombre, _ in ALL_DDL:
            existe = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = :t)"
            ), {"t": nombre}).scalar()
            if existe:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {nombre}")).scalar()
                log(f"  {nombre:<25} OK  ({n} filas)")
            else:
                log(f"  {nombre:<25} FALTA")

        # activos_ar detalle
        existe_ar = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='activos_ar')"
        )).scalar()
        if existe_ar:
            rows = conn.execute(text(
                "SELECT sector, COUNT(*) AS n FROM activos_ar "
                "WHERE activo = true GROUP BY sector ORDER BY n DESC"
            )).fetchall()
            log("\n  Sectores en activos_ar:")
            for r in rows:
                log(f"    {r[0]:<20} {r[1]} tickers")


def cmd_crear(engine, dry_run: bool):
    if dry_run:
        log("DRY RUN: SQL que se ejecutaria:")
        for nombre, ddl in ALL_DDL:
            print(f"\n-- {nombre} --")
            print(ddl.strip())
        return

    with engine.connect() as conn:
        for nombre, ddl in ALL_DDL:
            conn.execute(text(ddl))
            conn.commit()
            log(f"  Tabla creada (o ya existia): {nombre}")

    log("\nSembrando activos_ar...")
    _seed_activos_ar(engine, dry_run)


def _seed_activos_ar(engine, dry_run: bool):
    with engine.connect() as conn:
        existentes = {
            r[0] for r in conn.execute(text("SELECT ticker FROM activos_ar")).fetchall()
        }

    nuevos = [a for a in ACTIVOS_AR if a["ticker"] not in existentes]
    if not nuevos:
        log("  activos_ar ya tiene todos los tickers. Nada que insertar.")
        return

    if dry_run:
        log(f"  Insertaria {len(nuevos)} tickers:")
        for a in nuevos:
            log(f"    {a['ticker']:<8} {a['sector']:<20} opciones={a['tiene_opciones']}")
        return

    sql = """
        INSERT INTO activos_ar (ticker, nombre, sector, tiene_opciones, activo)
        VALUES (:ticker, :nombre, :sector, :tiene_opciones, :activo)
        ON CONFLICT (ticker) DO NOTHING
    """
    with engine.connect() as conn:
        for a in nuevos:
            conn.execute(text(sql), a)
        conn.commit()
    log(f"  Insertados {len(nuevos)} tickers en activos_ar.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inicializa tablas del modulo Argentina.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra SQL sin ejecutar")
    parser.add_argument("--status",  action="store_true", help="Solo muestra estado actual")
    args = parser.parse_args()

    engine = get_engine()

    if args.status:
        cmd_status(engine)
    else:
        cmd_crear(engine, dry_run=args.dry_run)
        if not args.dry_run:
            log("\nVerificacion final:")
            cmd_status(engine)


if __name__ == "__main__":
    main()
