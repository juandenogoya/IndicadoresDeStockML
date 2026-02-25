"""
01_init_db.py
Crea el schema completo de la base de datos activos_ml.
Ejecutar UNA sola vez al iniciar el proyecto.

Uso:
    python scripts/01_init_db.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import get_connection, test_conexion

# ─────────────────────────────────────────────────────────────
# DDL: Definición de tablas
# ─────────────────────────────────────────────────────────────

DDL = """

-- ── 1. Activos (master) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS activos (
    id         SERIAL PRIMARY KEY,
    ticker     VARCHAR(10)  NOT NULL UNIQUE,
    nombre     VARCHAR(100),
    sector     VARCHAR(50),
    activo     BOOLEAN      DEFAULT TRUE,
    created_at TIMESTAMP    DEFAULT NOW()
);

-- ── 2. Precios Diarios OHLCV ──────────────────────────────────
CREATE TABLE IF NOT EXISTS precios_diarios (
    id         SERIAL PRIMARY KEY,
    ticker     VARCHAR(10)  NOT NULL,
    fecha      DATE         NOT NULL,
    open       NUMERIC(14,4),
    high       NUMERIC(14,4),
    low        NUMERIC(14,4),
    close      NUMERIC(14,4),
    volume     BIGINT,
    adj_close  NUMERIC(14,4),
    created_at TIMESTAMP    DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_precios_ticker_fecha
    ON precios_diarios (ticker, fecha DESC);

-- ── 3. Indicadores Técnicos ───────────────────────────────────
CREATE TABLE IF NOT EXISTS indicadores_tecnicos (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(10)  NOT NULL,
    fecha       DATE         NOT NULL,

    -- Medias Móviles
    sma21       NUMERIC(14,4),
    sma50       NUMERIC(14,4),
    sma200      NUMERIC(14,4),

    -- Distancia % del precio a cada SMA
    dist_sma21  NUMERIC(10,4),
    dist_sma50  NUMERIC(10,4),
    dist_sma200 NUMERIC(10,4),

    -- Momentum / RSI
    rsi14       NUMERIC(8,4),
    momentum    NUMERIC(14,4),

    -- MACD
    macd        NUMERIC(14,6),
    macd_signal NUMERIC(14,6),
    macd_hist   NUMERIC(14,6),

    -- Volatilidad
    atr14       NUMERIC(14,4),
    bb_upper    NUMERIC(14,4),
    bb_middle   NUMERIC(14,4),
    bb_lower    NUMERIC(14,4),

    -- Volumen
    obv         NUMERIC(22,2),
    vol_relativo NUMERIC(10,4),

    -- Fuerza de tendencia
    adx         NUMERIC(8,4),

    created_at  TIMESTAMP    DEFAULT NOW(),
    updated_at  TIMESTAMP    DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_indicadores_ticker_fecha
    ON indicadores_tecnicos (ticker, fecha DESC);

-- ── 4. Scoring Técnico Rule-Based ─────────────────────────────
CREATE TABLE IF NOT EXISTS scoring_tecnico (
    id               SERIAL PRIMARY KEY,
    ticker           VARCHAR(10) NOT NULL,
    fecha            DATE        NOT NULL,

    -- Condiciones binarias (TRUE = señal alcista)
    cond_rsi         BOOLEAN,
    cond_macd        BOOLEAN,
    cond_sma21       BOOLEAN,
    cond_sma50       BOOLEAN,
    cond_sma200      BOOLEAN,
    cond_momentum    BOOLEAN,

    -- Score calculado
    score_ponderado  NUMERIC(5,4),    -- 0.00 a 1.00
    condiciones_ok   SMALLINT,        -- cuantas condiciones se cumplen (0-6)

    -- Señal resultante
    senal            VARCHAR(10),     -- LONG | NEUTRAL

    created_at       TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_scoring_ticker_fecha
    ON scoring_tecnico (ticker, fecha DESC);

-- ── 5. Operaciones del Backtesting ───────────────────────────
CREATE TABLE IF NOT EXISTS operaciones_backtest (
    id                  SERIAL PRIMARY KEY,
    estrategia_entrada  VARCHAR(5)   NOT NULL,  -- E1,E2,E3,E4
    estrategia_salida   VARCHAR(5)   NOT NULL,  -- S1,S2,S3,S4
    ticker              VARCHAR(10)  NOT NULL,
    segmento            VARCHAR(15)  NOT NULL,  -- TRAIN|TEST|BACKTEST

    fecha_entrada       DATE         NOT NULL,
    precio_entrada      NUMERIC(14,4),
    score_entrada       NUMERIC(5,4),

    fecha_salida        DATE,
    precio_salida       NUMERIC(14,4),
    motivo_salida       VARCHAR(20),            -- STOP_LOSS|TAKE_PROFIT|SENAL|TIMEOUT

    dias_posicion       INTEGER,
    retorno_pct         NUMERIC(10,4),
    resultado           VARCHAR(10),            -- GANANCIA|PERDIDA|NEUTRO

    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ops_estrategia
    ON operaciones_backtest (estrategia_entrada, estrategia_salida, ticker);

-- ── 6. Resumen de Resultados por Estrategia ───────────────────
CREATE TABLE IF NOT EXISTS resultados_backtest (
    id                     SERIAL PRIMARY KEY,
    estrategia_entrada     VARCHAR(5)  NOT NULL,
    estrategia_salida      VARCHAR(5)  NOT NULL,
    ticker                 VARCHAR(10),         -- NULL = resultado global
    segmento               VARCHAR(15)  NOT NULL,

    total_operaciones      INTEGER,
    ganancias              INTEGER,
    perdidas               INTEGER,
    neutros                INTEGER,

    win_rate               NUMERIC(6,4),        -- 0.00 a 1.00
    retorno_promedio_pct   NUMERIC(10,4),
    retorno_total_pct      NUMERIC(10,4),
    max_drawdown_pct       NUMERIC(10,4),
    profit_factor          NUMERIC(10,4),
    dias_promedio_posicion NUMERIC(8,2),

    created_at             TIMESTAMP DEFAULT NOW(),
    UNIQUE (estrategia_entrada, estrategia_salida, ticker, segmento)
);

-- ── 7. Log de ejecuciones ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS log_ejecuciones (
    id          SERIAL PRIMARY KEY,
    script      VARCHAR(100),
    accion      VARCHAR(100),
    detalle     TEXT,
    estado      VARCHAR(20),    -- OK | ERROR
    created_at  TIMESTAMP DEFAULT NOW()
);

-- ── 8. Features Sectoriales (Z-Scores) ────────────────────────
CREATE TABLE IF NOT EXISTS features_sector (
    id                     SERIAL PRIMARY KEY,
    ticker                 VARCHAR(10)  NOT NULL,
    fecha                  DATE         NOT NULL,
    sector                 VARCHAR(50),

    -- Z-Scores relativos al sector
    z_rsi_sector           NUMERIC(8,4),
    z_retorno_1d_sector    NUMERIC(8,4),
    z_retorno_5d_sector    NUMERIC(8,4),
    z_vol_sector           NUMERIC(8,4),
    z_dist_sma50_sector    NUMERIC(8,4),
    z_adx_sector           NUMERIC(8,4),

    -- Breadth y ranking
    pct_long_sector        NUMERIC(6,4),   -- % tickers LONG en el sector
    rank_retorno_sector    INTEGER,         -- ranking por retorno diario (1=mejor)

    -- Promedios sectoriales (contexto)
    rsi_sector_avg         NUMERIC(8,4),
    adx_sector_avg         NUMERIC(8,4),
    retorno_1d_sector_avg  NUMERIC(8,4),

    created_at             TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_features_sector_ticker_fecha
    ON features_sector (ticker, fecha DESC);

-- ── 9. Feature Store ML (tabla principal de entrenamiento) ─────
CREATE TABLE IF NOT EXISTS features_ml (
    id                     SERIAL PRIMARY KEY,
    ticker                 VARCHAR(10)  NOT NULL,
    nombre                 VARCHAR(100),
    sector                 VARCHAR(50),
    fecha                  DATE         NOT NULL,
    segmento               VARCHAR(15),            -- TRAIN | TEST | BACKTEST

    -- Precio y volumen
    close                  NUMERIC(14,4),
    vol_relativo           NUMERIC(10,4),

    -- Indicadores técnicos
    rsi14                  NUMERIC(8,4),
    macd_hist              NUMERIC(14,6),
    dist_sma21             NUMERIC(10,4),
    dist_sma50             NUMERIC(10,4),
    dist_sma200            NUMERIC(10,4),
    adx                    NUMERIC(8,4),
    atr14                  NUMERIC(14,4),
    momentum               NUMERIC(14,4),
    bb_upper               NUMERIC(14,4),
    bb_middle              NUMERIC(14,4),
    bb_lower               NUMERIC(14,4),

    -- Scoring rule-based
    score_ponderado        NUMERIC(5,4),
    condiciones_ok         SMALLINT,
    cond_rsi               BOOLEAN,
    cond_macd              BOOLEAN,
    cond_sma21             BOOLEAN,
    cond_sma50             BOOLEAN,
    cond_sma200            BOOLEAN,
    cond_momentum          BOOLEAN,

    -- Features sectoriales (Z-Scores)
    z_rsi_sector           NUMERIC(8,4),
    z_retorno_1d_sector    NUMERIC(8,4),
    z_retorno_5d_sector    NUMERIC(8,4),
    z_vol_sector           NUMERIC(8,4),
    z_dist_sma50_sector    NUMERIC(8,4),
    z_adx_sector           NUMERIC(8,4),
    pct_long_sector        NUMERIC(6,4),
    rank_retorno_sector    INTEGER,
    rsi_sector_avg         NUMERIC(8,4),
    adx_sector_avg         NUMERIC(8,4),
    retorno_1d_sector_avg  NUMERIC(8,4),

    -- Variables objetivo (retornos futuros)
    retorno_1d             NUMERIC(8,4),
    retorno_5d             NUMERIC(8,4),
    retorno_10d            NUMERIC(8,4),
    retorno_20d            NUMERIC(8,4),

    -- Labels para clasificación
    label                  VARCHAR(10),    -- GANANCIA | PERDIDA | NEUTRO
    label_binario          SMALLINT,       -- 1 = GANANCIA, 0 = NEUTRO/PERDIDA

    created_at             TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_features_ml_segmento
    ON features_ml (segmento, ticker, fecha DESC);

CREATE INDEX IF NOT EXISTS idx_features_ml_label
    ON features_ml (label, segmento);

-- ── 10. Resultados de modelos ML ──────────────────────────────
CREATE TABLE IF NOT EXISTS resultados_modelos_ml (
    id              SERIAL PRIMARY KEY,
    scope           VARCHAR(50),       -- 'global' | sector name
    algoritmo       VARCHAR(10),       -- 'rf' | 'xgb' | 'lgbm'
    segmento        VARCHAR(15),       -- 'TRAIN' | 'TEST' | 'BACKTEST'
    n_filas         INTEGER,
    n_features      INTEGER,
    accuracy        NUMERIC(7,4),
    precision_w     NUMERIC(7,4),
    recall_w        NUMERIC(7,4),
    f1_w            NUMERIC(7,4),
    precision_1     NUMERIC(7,4),      -- clase GANANCIA
    recall_1        NUMERIC(7,4),
    f1_1            NUMERIC(7,4),      -- metrica principal
    roc_auc         NUMERIC(7,4),
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (scope, algoritmo, segmento)
);

-- ── 11. Plan de despliegue — modelo ganador por sector ─────────
CREATE TABLE IF NOT EXISTS modelos_produccion (
    id              SERIAL PRIMARY KEY,
    scope           VARCHAR(50) NOT NULL,  -- 'global' | sector name
    tipo            VARCHAR(20),           -- 'global' | 'sectorial'
    algoritmo       VARCHAR(20),           -- 'rf' | 'xgb' | 'lgbm'
    modelo_path     VARCHAR(300),
    n_features      INTEGER,
    f1_test         NUMERIC(7,4),
    f1_backtest     NUMERIC(7,4),
    roc_auc_test    NUMERIC(7,4),
    activo          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (scope)
);

-- ── 13. Features Precio/Accion (estructura OHLCV para modelo V2) ─
CREATE TABLE IF NOT EXISTS features_precio_accion (
    ticker VARCHAR(10) NOT NULL,
    fecha  DATE        NOT NULL,

    -- Grupo 1: Anatomia de vela (9)
    body_pct              NUMERIC(10,4),
    body_ratio            NUMERIC(6,4),
    upper_shadow_pct      NUMERIC(6,4),
    lower_shadow_pct      NUMERIC(6,4),
    es_alcista            SMALLINT,
    gap_apertura_pct      NUMERIC(10,4),
    rango_diario_pct      NUMERIC(10,4),
    rango_rel_atr         NUMERIC(10,4),
    clv                   NUMERIC(7,4),

    -- Grupo 2: Patrones clasicos (8)
    patron_doji              SMALLINT,
    patron_hammer            SMALLINT,
    patron_shooting_star     SMALLINT,
    patron_marubozu          SMALLINT,
    patron_engulfing_bull    SMALLINT,
    patron_engulfing_bear    SMALLINT,
    inside_bar               SMALLINT,
    outside_bar              SMALLINT,

    -- Grupo 3: Estructura rolling (8)
    body_pct_ma5          NUMERIC(10,4),
    velas_alcistas_5d     SMALLINT,
    velas_alcistas_10d    SMALLINT,
    rango_expansion       SMALLINT,
    dist_max_20d          NUMERIC(10,4),
    dist_min_20d          NUMERIC(10,4),
    pos_rango_20d         NUMERIC(6,4),
    tendencia_velas       SMALLINT,

    -- Grupo 4: Volumen direccional (7)
    vol_ratio_5d          NUMERIC(10,4),
    vol_spike             SMALLINT,
    up_vol_5d             NUMERIC(6,4),
    ad_flow               NUMERIC(20,4),
    chaikin_mf_20         NUMERIC(7,4),
    vol_price_confirm     SMALLINT,
    vol_price_diverge     SMALLINT,

    PRIMARY KEY (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_fpa_ticker_fecha
    ON features_precio_accion (ticker, fecha);

-- ── 14. Features Market Structure (swings, HH/HL, BOS/CHoCH) ───
CREATE TABLE IF NOT EXISTS features_market_structure (
    ticker VARCHAR(10) NOT NULL,
    fecha  DATE        NOT NULL,

    -- Ventana N=5 (tactico, ~11 barras)
    is_sh_5        SMALLINT,
    is_sl_5        SMALLINT,
    estructura_5   SMALLINT,
    dist_sh_5_pct  NUMERIC(10,4),
    dist_sl_5_pct  NUMERIC(10,4),
    dias_sh_5      SMALLINT,
    dias_sl_5      SMALLINT,
    impulso_5_pct  NUMERIC(10,4),
    bos_bull_5     SMALLINT,
    bos_bear_5     SMALLINT,
    choch_bull_5   SMALLINT,
    choch_bear_5   SMALLINT,

    -- Ventana N=10 (estrategico, ~21 barras)
    is_sh_10       SMALLINT,
    is_sl_10       SMALLINT,
    estructura_10  SMALLINT,
    dist_sh_10_pct NUMERIC(10,4),
    dist_sl_10_pct NUMERIC(10,4),
    dias_sh_10     SMALLINT,
    dias_sl_10     SMALLINT,
    impulso_10_pct NUMERIC(10,4),
    bos_bull_10    SMALLINT,
    bos_bear_10    SMALLINT,
    choch_bull_10  SMALLINT,
    choch_bear_10  SMALLINT,

    PRIMARY KEY (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_fms_ticker_fecha
    ON features_market_structure (ticker, fecha);

-- ── 12. Comparacion ML-Filter vs Baseline (backtesting) ────────
CREATE TABLE IF NOT EXISTS resultados_ml_filter (
    id                       SERIAL PRIMARY KEY,
    estrategia_entrada       VARCHAR(5)   NOT NULL,
    estrategia_salida        VARCHAR(5)   NOT NULL,
    scope                    VARCHAR(50)  NOT NULL,   -- 'GLOBAL' | sector
    segmento                 VARCHAR(15)  NOT NULL,   -- TRAIN | TEST | BACKTEST
    umbral_ml                NUMERIC(4,2) NOT NULL,   -- threshold (e.g. 0.50)

    -- Baseline (sin filtro ML)
    ops_original             INTEGER,
    win_rate_orig            NUMERIC(7,4),
    ret_promedio_orig        NUMERIC(10,4),
    ret_total_orig           NUMERIC(10,4),
    profit_factor_orig       NUMERIC(10,4),
    max_dd_orig              NUMERIC(10,4),

    -- Con filtro ML
    ops_ml                   INTEGER,
    ops_rechazadas           INTEGER,
    pct_rechazo              NUMERIC(6,4),
    win_rate_ml              NUMERIC(7,4),
    ret_promedio_ml          NUMERIC(10,4),
    ret_total_ml             NUMERIC(10,4),
    profit_factor_ml         NUMERIC(10,4),
    max_dd_ml                NUMERIC(10,4),

    -- Rechazadas (verificacion: deben ser peores que las aprobadas)
    win_rate_rechazadas      NUMERIC(7,4),
    ret_promedio_rechazadas  NUMERIC(10,4),
    profit_factor_rechazadas NUMERIC(10,4),

    -- Deltas (ML - Original)
    delta_win_rate           NUMERIC(7,4),
    delta_ret_promedio       NUMERIC(10,4),
    delta_profit_factor      NUMERIC(10,4),

    created_at               TIMESTAMP DEFAULT NOW(),
    UNIQUE (estrategia_entrada, estrategia_salida, scope, segmento, umbral_ml)
);

CREATE INDEX IF NOT EXISTS idx_mlfilter_estrategia
    ON resultados_ml_filter (estrategia_entrada, estrategia_salida, scope, segmento);

"""

# ─────────────────────────────────────────────────────────────
# Datos iniciales: poblar tabla activos
# ─────────────────────────────────────────────────────────────

ACTIVOS_SEED = [
    # Financials
    ("JPM",  "JPMorgan Chase",         "Financials"),
    ("BAC",  "Bank of America",        "Financials"),
    ("MS",   "Morgan Stanley",         "Financials"),
    ("GS",   "Goldman Sachs",          "Financials"),
    ("WFC",  "Wells Fargo",            "Financials"),
    # Consumer Staples
    ("KO",   "Coca-Cola",              "Consumer Staples"),
    ("PEP",  "PepsiCo",               "Consumer Staples"),
    ("PG",   "Procter & Gamble",       "Consumer Staples"),
    ("PM",   "Philip Morris",          "Consumer Staples"),
    ("CL",   "Colgate-Palmolive",      "Consumer Staples"),
    ("MO",   "Altria Group",           "Consumer Staples"),
    # Consumer Discretionary
    ("TGT",  "Target",                 "Consumer Discretionary"),
    ("WMT",  "Walmart",                "Consumer Discretionary"),
    ("COST", "Costco",                 "Consumer Discretionary"),
    ("HD",   "Home Depot",             "Consumer Discretionary"),
    ("LOW",  "Lowe's",                 "Consumer Discretionary"),
    # Automotive
    ("F",    "Ford Motor",             "Automotive"),
    ("GM",   "General Motors",         "Automotive"),
    ("STLA", "Stellantis",             "Automotive"),
]

# ─────────────────────────────────────────────────────────────
# Migración V2: agrega modelo_version a tablas ML
# Idempotente: puede ejecutarse varias veces sin error.
# ─────────────────────────────────────────────────────────────

DDL_MIGRATION_V2 = """
ALTER TABLE resultados_modelos_ml
    ADD COLUMN IF NOT EXISTS modelo_version VARCHAR(10) DEFAULT 'v1';

ALTER TABLE modelos_produccion
    ADD COLUMN IF NOT EXISTS modelo_version VARCHAR(10) DEFAULT 'v1';

DO $mig$
BEGIN
    -- Reemplazar constraint (scope, algoritmo, segmento)
    -- por (scope, algoritmo, segmento, modelo_version)
    BEGIN
        ALTER TABLE resultados_modelos_ml
            DROP CONSTRAINT resultados_modelos_ml_scope_algoritmo_segmento_key;
    EXCEPTION WHEN undefined_object THEN NULL;
    END;

    BEGIN
        ALTER TABLE resultados_modelos_ml
            ADD CONSTRAINT rmml_scope_alg_seg_ver_key
            UNIQUE (scope, algoritmo, segmento, modelo_version);
    EXCEPTION WHEN duplicate_table OR duplicate_object THEN NULL;
    END;

    -- Reemplazar constraint (scope) por (scope, modelo_version)
    BEGIN
        ALTER TABLE modelos_produccion
            DROP CONSTRAINT modelos_produccion_scope_key;
    EXCEPTION WHEN undefined_object THEN NULL;
    END;

    BEGIN
        ALTER TABLE modelos_produccion
            ADD CONSTRAINT mp_scope_ver_key
            UNIQUE (scope, modelo_version);
    EXCEPTION WHEN duplicate_table OR duplicate_object THEN NULL;
    END;
END $mig$;
"""


# ─────────────────────────────────────────────────────────────
# Migración V3: tabla features_market_structure
# Idempotente: puede ejecutarse varias veces sin error.
# ─────────────────────────────────────────────────────────────

DDL_MIGRATION_PA = """
CREATE TABLE IF NOT EXISTS operaciones_bt_pa (
    id                  SERIAL PRIMARY KEY,
    estrategia_entrada  VARCHAR(5)   NOT NULL,  -- EV1,EV2,EV3,EV4
    estrategia_salida   VARCHAR(5)   NOT NULL,  -- SV1,SV2,SV3,SV4
    ticker              VARCHAR(10)  NOT NULL,
    segmento            VARCHAR(15)  NOT NULL,  -- TRAIN|TEST|BACKTEST

    fecha_entrada       DATE         NOT NULL,
    precio_entrada      NUMERIC(14,4),
    score_entrada       NUMERIC(5,4),           -- score_ponderado al entrar (puede ser NULL)

    fecha_salida        DATE,
    precio_salida       NUMERIC(14,4),
    motivo_salida       VARCHAR(20),            -- STOP_LOSS|TAKE_PROFIT|TARGET|ESTRUCTURA|SENAL|TIMEOUT|FIN_SEGMENTO

    dias_posicion       INTEGER,
    retorno_pct         NUMERIC(10,4),
    resultado           VARCHAR(10),            -- GANANCIA|PERDIDA|NEUTRO

    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ops_pa_estrategia
    ON operaciones_bt_pa (estrategia_entrada, estrategia_salida, ticker);

CREATE TABLE IF NOT EXISTS resultados_bt_pa (
    id                     SERIAL PRIMARY KEY,
    estrategia_entrada     VARCHAR(5)   NOT NULL,
    estrategia_salida      VARCHAR(5)   NOT NULL,
    ticker                 VARCHAR(10),         -- NULL = resultado global
    segmento               VARCHAR(15)  NOT NULL,

    total_operaciones      INTEGER,
    ganancias              INTEGER,
    perdidas               INTEGER,
    neutros                INTEGER,

    win_rate               NUMERIC(6,4),
    retorno_promedio_pct   NUMERIC(10,4),
    retorno_total_pct      NUMERIC(10,4),
    max_drawdown_pct       NUMERIC(10,4),
    profit_factor          NUMERIC(10,4),
    dias_promedio_posicion NUMERIC(8,2),

    created_at             TIMESTAMP DEFAULT NOW(),
    UNIQUE (estrategia_entrada, estrategia_salida, ticker, segmento)
);
"""


DDL_MIGRATION_SCANNER = """
CREATE TABLE IF NOT EXISTS alertas_scanner (
    id              SERIAL PRIMARY KEY,
    scan_fecha      TIMESTAMP    NOT NULL DEFAULT NOW(),
    ticker          VARCHAR(10)  NOT NULL,
    sector          VARCHAR(50),               -- NULL si ticker nuevo y sector desconocido
    persistido_en_db BOOLEAN     DEFAULT FALSE, -- TRUE si se grabo el ticker en activos/precios

    -- Precio de referencia
    precio_cierre   NUMERIC(14,4),
    precio_fecha    DATE,
    atr14           NUMERIC(14,4),

    -- ML V3
    ml_prob_ganancia  NUMERIC(6,4),             -- P(ganancia) del modelo V3
    ml_modelo_usado   VARCHAR(50),              -- 'global' | sector

    -- Condiciones de entrada PA (EV1-EV4)
    pa_ev1          SMALLINT DEFAULT 0,
    pa_ev2          SMALLINT DEFAULT 0,
    pa_ev3          SMALLINT DEFAULT 0,
    pa_ev4          SMALLINT DEFAULT 0,

    -- Senales bajistas
    bear_bos10      SMALLINT DEFAULT 0,
    bear_choch10    SMALLINT DEFAULT 0,
    bear_estructura SMALLINT DEFAULT 0,         -- 1 si estructura_10 == -1

    -- Scoring rule-based
    score_ponderado   NUMERIC(5,4),
    condiciones_ok    SMALLINT,

    -- Market structure (contexto)
    estructura_10     SMALLINT,
    dist_sl_10_pct    NUMERIC(10,4),
    dist_sh_10_pct    NUMERIC(10,4),
    dias_sl_10        SMALLINT,
    dias_sh_10        SMALLINT,

    -- Resultado del scanner
    alert_score     NUMERIC(5,1),              -- 0-100
    alert_nivel     VARCHAR(20),               -- COMPRA_FUERTE|COMPRA|NEUTRAL|VENTA|VENTA_FUERTE
    alert_detalle   TEXT,                      -- resumen legible

    -- Post-facto: precios reales para validar la alerta (se llenan despues)
    precio_1d_real  NUMERIC(14,4),
    precio_5d_real  NUMERIC(14,4),
    precio_20d_real NUMERIC(14,4),
    retorno_1d_real NUMERIC(8,4),
    retorno_5d_real NUMERIC(8,4),
    retorno_20d_real NUMERIC(8,4),
    verificado      BOOLEAN DEFAULT FALSE,

    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alertas_ticker_fecha
    ON alertas_scanner (ticker, scan_fecha DESC);

CREATE INDEX IF NOT EXISTS idx_alertas_nivel
    ON alertas_scanner (alert_nivel, scan_fecha DESC);
"""


DDL_MIGRATION_MODELO_ASIGNADO = """
ALTER TABLE activos
    ADD COLUMN IF NOT EXISTS modelo_asignado VARCHAR(50);
"""


DDL_MIGRATION_V3 = """
CREATE TABLE IF NOT EXISTS features_market_structure (
    ticker VARCHAR(10) NOT NULL,
    fecha  DATE        NOT NULL,
    is_sh_5        SMALLINT,
    is_sl_5        SMALLINT,
    estructura_5   SMALLINT,
    dist_sh_5_pct  NUMERIC(10,4),
    dist_sl_5_pct  NUMERIC(10,4),
    dias_sh_5      SMALLINT,
    dias_sl_5      SMALLINT,
    impulso_5_pct  NUMERIC(10,4),
    bos_bull_5     SMALLINT,
    bos_bear_5     SMALLINT,
    choch_bull_5   SMALLINT,
    choch_bear_5   SMALLINT,
    is_sh_10       SMALLINT,
    is_sl_10       SMALLINT,
    estructura_10  SMALLINT,
    dist_sh_10_pct NUMERIC(10,4),
    dist_sl_10_pct NUMERIC(10,4),
    dias_sh_10     SMALLINT,
    dias_sl_10     SMALLINT,
    impulso_10_pct NUMERIC(10,4),
    bos_bull_10    SMALLINT,
    bos_bear_10    SMALLINT,
    choch_bull_10  SMALLINT,
    choch_bear_10  SMALLINT,
    PRIMARY KEY (ticker, fecha)
);
CREATE INDEX IF NOT EXISTS idx_fms_ticker_fecha
    ON features_market_structure (ticker, fecha);
"""


INSERT_ACTIVO = """
    INSERT INTO activos (ticker, nombre, sector)
    VALUES (%s, %s, %s)
    ON CONFLICT (ticker) DO NOTHING
"""


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  INIT DB — activos_ml")
    print("=" * 60)

    print("\n[1/3] Verificando conexion...")
    test_conexion()

    print("\n[2/3] Creando tablas...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
    print("  Tablas creadas correctamente.")

    print("\n[3/3] Insertando activos iniciales...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            for row in ACTIVOS_SEED:
                cur.execute(INSERT_ACTIVO, row)
    print(f"  {len(ACTIVOS_SEED)} activos insertados.")

    print("\n[4/6] Ejecutando migracion V2 (modelo_version)...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_MIGRATION_V2)
    print("  Migracion V2 completada.")

    print("\n[5/6] Ejecutando migracion V3 (features_market_structure)...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_MIGRATION_V3)
    print("  Migracion V3 completada.")

    print("\n[6/7] Ejecutando migracion PA (operaciones_bt_pa, resultados_bt_pa)...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_MIGRATION_PA)
    print("  Migracion PA completada.")

    print("\n[7/8] Ejecutando migracion Scanner (alertas_scanner)...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_MIGRATION_SCANNER)
    print("  Migracion Scanner completada.")

    print("\n[8/8] Ejecutando migracion modelo_asignado (activos)...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_MIGRATION_MODELO_ASIGNADO)
    print("  Migracion modelo_asignado completada.")

    print("\n" + "=" * 60)
    print("  SCHEMA INICIALIZADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()
