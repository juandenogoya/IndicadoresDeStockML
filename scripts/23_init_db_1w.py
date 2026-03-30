"""
23_init_db_1w.py
Crea el schema de tablas Weekly (1W) en Railway.
Son espejo exacto de las tablas 1D, con sufijo _1w.

Tablas creadas:
    precios_semanales          <- resample de precios_diarios (O=first, H=max, L=min, C=last, V=sum)
    indicadores_tecnicos_1w    <- espejo de indicadores_tecnicos
    features_precio_accion_1w  <- espejo de features_precio_accion (32 features)
    features_market_structure_1w <- espejo de features_market_structure (24 features)
    operaciones_bt_pa_1w       <- espejo de operaciones_bt_pa (incluye stop_loss/take_profit)
    resultados_bt_pa_1w        <- espejo de resultados_bt_pa

Idempotente: CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS.
Puede ejecutarse varias veces sin error.

Uso:
    python scripts/23_init_db_1w.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import get_connection, test_conexion


# ─────────────────────────────────────────────────────────────
# DDL: Tablas 1W
# ─────────────────────────────────────────────────────────────

DDL_1W = """

-- ── 1. Precios Semanales OHLCV ────────────────────────────────
--    Derivado de precios_diarios via resample.
--    fecha_semana = ultimo dia habil de la semana (viernes o jueves si feriado).
CREATE TABLE IF NOT EXISTS precios_semanales (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(10)  NOT NULL,
    fecha_semana DATE         NOT NULL,   -- ultimo dia habil de la semana
    open        NUMERIC(14,4),
    high        NUMERIC(14,4),
    low         NUMERIC(14,4),
    close       NUMERIC(14,4),
    volume      BIGINT,
    adj_close   NUMERIC(14,4),
    n_dias      SMALLINT,                -- dias habiles en la semana (1-5)
    created_at  TIMESTAMP    DEFAULT NOW(),
    updated_at  TIMESTAMP    DEFAULT NOW(),
    UNIQUE (ticker, fecha_semana)
);

CREATE INDEX IF NOT EXISTS idx_precios_sem_ticker_fecha
    ON precios_semanales (ticker, fecha_semana DESC);


-- ── 2. Indicadores Tecnicos 1W ────────────────────────────────
--    Mismos indicadores que indicadores_tecnicos, calculados sobre barras semanales.
--    sma21 = 21 semanas (~5 meses), sma50 = 50 semanas (~1 ano), sma200 = 200 semanas (~4 anos).
CREATE TABLE IF NOT EXISTS indicadores_tecnicos_1w (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(10)  NOT NULL,
    fecha       DATE         NOT NULL,   -- fecha_semana (ultimo dia habil)

    -- Medias Moviles
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

CREATE INDEX IF NOT EXISTS idx_ind_tec_1w_ticker_fecha
    ON indicadores_tecnicos_1w (ticker, fecha DESC);


-- ── 3. Features Precio/Accion 1W ─────────────────────────────
--    Espejo de features_precio_accion (32 features), calculado sobre velas semanales.
CREATE TABLE IF NOT EXISTS features_precio_accion_1w (
    ticker VARCHAR(10) NOT NULL,
    fecha  DATE        NOT NULL,         -- fecha_semana (ultimo dia habil)

    -- Grupo 1: Anatomia de vela (9)
    body_pct              NUMERIC(10,4),
    body_ratio            NUMERIC(6,4),
    upper_shadow_pct      NUMERIC(6,4),
    lower_shadow_pct      NUMERIC(6,4),
    es_alcista            SMALLINT,
    gap_apertura_pct      NUMERIC(10,4),
    rango_diario_pct      NUMERIC(10,4),  -- en 1W = rango semanal pct
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
    -- En 1W: ventanas en semanas (5 semanas ~ 1 mes, 10 semanas ~ 2 meses, 20 semanas ~ 5 meses)
    body_pct_ma5          NUMERIC(10,4),
    velas_alcistas_5d     SMALLINT,       -- en 1W = velas_alcistas_5w
    velas_alcistas_10d    SMALLINT,       -- en 1W = velas_alcistas_10w
    rango_expansion       SMALLINT,
    dist_max_20d          NUMERIC(10,4),  -- en 1W = dist_max_20w
    dist_min_20d          NUMERIC(10,4),  -- en 1W = dist_min_20w
    pos_rango_20d         NUMERIC(6,4),   -- en 1W = pos_rango_20w
    tendencia_velas       SMALLINT,

    -- Grupo 4: Volumen direccional (7)
    vol_ratio_5d          NUMERIC(10,4),  -- en 1W = vol_ratio_5w
    vol_spike             SMALLINT,
    up_vol_5d             NUMERIC(6,4),   -- en 1W = up_vol_5w
    ad_flow               NUMERIC(20,4),
    chaikin_mf_20         NUMERIC(7,4),
    vol_price_confirm     SMALLINT,
    vol_price_diverge     SMALLINT,

    PRIMARY KEY (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_fpa_1w_ticker_fecha
    ON features_precio_accion_1w (ticker, fecha);


-- ── 4. Features Market Structure 1W ──────────────────────────
--    Espejo de features_market_structure (24 features), sobre barras semanales.
--    N=5 = 5 semanas (~1 mes), N=10 = 10 semanas (~2.5 meses).
CREATE TABLE IF NOT EXISTS features_market_structure_1w (
    ticker VARCHAR(10) NOT NULL,
    fecha  DATE        NOT NULL,         -- fecha_semana (ultimo dia habil)

    -- Ventana N=5 semanas (tactico, ~11 barras = 11 semanas)
    is_sh_5        SMALLINT,
    is_sl_5        SMALLINT,
    estructura_5   SMALLINT,
    dist_sh_5_pct  NUMERIC(10,4),
    dist_sl_5_pct  NUMERIC(10,4),
    dias_sh_5      SMALLINT,            -- en 1W = semanas_sh_5
    dias_sl_5      SMALLINT,            -- en 1W = semanas_sl_5
    impulso_5_pct  NUMERIC(10,4),
    bos_bull_5     SMALLINT,
    bos_bear_5     SMALLINT,
    choch_bull_5   SMALLINT,
    choch_bear_5   SMALLINT,

    -- Ventana N=10 semanas (estrategico, ~21 barras = 21 semanas)
    is_sh_10       SMALLINT,
    is_sl_10       SMALLINT,
    estructura_10  SMALLINT,
    dist_sh_10_pct NUMERIC(10,4),
    dist_sl_10_pct NUMERIC(10,4),
    dias_sh_10     SMALLINT,            -- en 1W = semanas_sh_10
    dias_sl_10     SMALLINT,            -- en 1W = semanas_sl_10
    impulso_10_pct NUMERIC(10,4),
    bos_bull_10    SMALLINT,
    bos_bear_10    SMALLINT,
    choch_bull_10  SMALLINT,
    choch_bear_10  SMALLINT,

    PRIMARY KEY (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_fms_1w_ticker_fecha
    ON features_market_structure_1w (ticker, fecha);


-- ── 5. Operaciones BT PA 1W ───────────────────────────────────
--    Espejo de operaciones_bt_pa, incluyendo stop_loss/take_profit desde el inicio.
--    dias_posicion = semanas_posicion en este contexto.
CREATE TABLE IF NOT EXISTS operaciones_bt_pa_1w (
    id                  SERIAL PRIMARY KEY,
    estrategia_entrada  VARCHAR(5)   NOT NULL,  -- EV1,EV2,EV3,EV4
    estrategia_salida   VARCHAR(5)   NOT NULL,  -- SV1,SV2,SV3,SV4
    ticker              VARCHAR(10)  NOT NULL,
    segmento            VARCHAR(15)  NOT NULL,  -- TRAIN|TEST|BACKTEST

    fecha_entrada       DATE         NOT NULL,  -- fecha_semana de entrada
    precio_entrada      NUMERIC(14,4),
    score_entrada       NUMERIC(5,4),

    fecha_salida        DATE,                   -- fecha_semana de salida
    precio_salida       NUMERIC(14,4),
    motivo_salida       VARCHAR(20),            -- STOP_LOSS|TAKE_PROFIT|TARGET|ESTRUCTURA|SENAL|TIMEOUT|FIN_SEGMENTO

    dias_posicion       INTEGER,                -- cantidad de semanas en posicion
    retorno_pct         NUMERIC(10,4),
    resultado           VARCHAR(10),            -- GANANCIA|PERDIDA|NEUTRO

    stop_loss           FLOAT,                  -- precio de stop loss
    take_profit         FLOAT,                  -- precio de take profit

    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ops_pa_1w_estrategia
    ON operaciones_bt_pa_1w (estrategia_entrada, estrategia_salida, ticker);


-- ── 6. Resultados BT PA 1W ────────────────────────────────────
--    Espejo de resultados_bt_pa.
CREATE TABLE IF NOT EXISTS resultados_bt_pa_1w (
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
    dias_promedio_posicion NUMERIC(8,2),        -- promedio de semanas en posicion

    created_at             TIMESTAMP DEFAULT NOW(),
    UNIQUE (estrategia_entrada, estrategia_salida, ticker, segmento)
);

"""


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

TABLAS = [
    "precios_semanales",
    "indicadores_tecnicos_1w",
    "features_precio_accion_1w",
    "features_market_structure_1w",
    "operaciones_bt_pa_1w",
    "resultados_bt_pa_1w",
]


def main():
    print("=" * 60)
    print("  INIT DB 1W — Tablas Weekly")
    print("=" * 60)
    print(f"  Tablas a crear: {len(TABLAS)}")
    for t in TABLAS:
        print(f"    - {t}")
    print()

    print("[1/3] Verificando conexion...")
    test_conexion()

    print("\n[2/3] Creando tablas 1W...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_1W)
    print("  Tablas creadas correctamente (idempotente).")

    print("\n[3/3] Verificando tablas en DB...")
    from src.data.database import query_df
    df = query_df("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = ANY(ARRAY[
              'precios_semanales',
              'indicadores_tecnicos_1w',
              'features_precio_accion_1w',
              'features_market_structure_1w',
              'operaciones_bt_pa_1w',
              'resultados_bt_pa_1w'
          ])
        ORDER BY table_name
    """)

    encontradas = df["table_name"].tolist() if not df.empty else []
    for t in TABLAS:
        estado = "OK" if t in encontradas else "FALTA"
        print(f"  {estado:6} | {t}")

    print()
    if len(encontradas) == len(TABLAS):
        print("=" * 60)
        print("  SCHEMA 1W INICIALIZADO EXITOSAMENTE")
        print(f"  {len(TABLAS)} tablas disponibles en Railway")
        print("=" * 60)
        print()
        print("  Proximos pasos:")
        print("    Etapa 2: python scripts/24_poblar_precios_semanales.py")
        print("    Etapa 3: src/indicators/technical_1w.py")
    else:
        faltantes = set(TABLAS) - set(encontradas)
        print(f"  ADVERTENCIA: {len(faltantes)} tabla(s) no encontradas: {faltantes}")


if __name__ == "__main__":
    main()
