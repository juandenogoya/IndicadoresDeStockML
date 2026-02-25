CREATE OR REPLACE VIEW v_senales_diarias AS
WITH macd_lag AS (
    SELECT
        i.ticker,
        i.fecha,
        i.macd_hist,
        LAG(i.macd_hist) OVER (PARTITION BY i.ticker ORDER BY i.fecha) AS prev_macd_hist
    FROM indicadores_tecnicos i
)
SELECT
    p.ticker,
    a.nombre,
    a.sector,
    p.fecha,

    -- Precios
    ROUND(p.close::NUMERIC, 2)          AS precio_cierre,
    ROUND(p.open::NUMERIC, 2)           AS precio_apertura,
    ROUND(p.high::NUMERIC, 2)           AS precio_max,
    ROUND(p.low::NUMERIC, 2)            AS precio_min,
    p.volume                            AS volumen,

    -- Valores numericos de indicadores
    ROUND(i.rsi14::NUMERIC, 2)          AS rsi14,
    ROUND(i.macd_hist::NUMERIC, 4)      AS macd_hist,
    ROUND(i.sma21::NUMERIC, 2)          AS sma21,
    ROUND(i.sma50::NUMERIC, 2)          AS sma50,
    ROUND(i.sma200::NUMERIC, 2)         AS sma200,
    ROUND(i.adx::NUMERIC, 2)            AS adx,
    ROUND(i.vol_relativo::NUMERIC, 2)   AS vol_relativo,
    ROUND(i.atr14::NUMERIC, 2)          AS atr14,

    -- Señal RSI
    CASE
        WHEN i.rsi14 < 30  THEN 'FUERTE OVERSOLD'
        WHEN i.rsi14 < 40  THEN 'OVERSOLD'
        WHEN i.rsi14 < 60  THEN 'NEUTRAL'
        WHEN i.rsi14 < 70  THEN 'OVERBOUGHT'
        ELSE                    'FUERTE OVERBOUGHT'
    END AS senal_rsi,

    -- Señal MACD (detecta cruces con LAG del dia anterior)
    CASE
        WHEN ml.prev_macd_hist <= 0 AND i.macd_hist > 0  THEN 'CRUCE ALCISTA'
        WHEN i.macd_hist > 0                              THEN 'POSITIVO'
        WHEN ml.prev_macd_hist >= 0 AND i.macd_hist < 0  THEN 'CRUCE BAJISTA'
        ELSE                                                   'NEGATIVO'
    END AS senal_macd,

    -- Señal SMA21
    CASE
        WHEN p.close > i.sma21  THEN 'POR SOBRE'
        ELSE                         'POR DEBAJO'
    END AS senal_sma21,

    -- Señal SMA50
    CASE
        WHEN p.close > i.sma50  THEN 'POR SOBRE'
        ELSE                         'POR DEBAJO'
    END AS senal_sma50,

    -- Señal SMA200
    CASE
        WHEN p.close > i.sma200 THEN 'POR SOBRE'
        ELSE                         'POR DEBAJO'
    END AS senal_sma200,

    -- Señal Momentum (ADX como amplificador de fuerza)
    CASE
        WHEN i.momentum > 0 AND i.adx > 25  THEN 'COMPRA FUERTE'
        WHEN i.momentum > 0                  THEN 'COMPRA'
        WHEN i.momentum < 0 AND i.adx > 25  THEN 'VENTA FUERTE'
        WHEN i.momentum < 0                  THEN 'VENTA'
        ELSE                                      'NEUTRAL'
    END AS senal_momentum,

    -- Señal Volumen relativo
    CASE
        WHEN i.vol_relativo > 1.5   THEN 'MUY ALTO'
        WHEN i.vol_relativo > 1.2   THEN 'ALTO'
        WHEN i.vol_relativo >= 0.8  THEN 'NORMAL'
        ELSE                             'BAJO'
    END AS senal_volumen,

    -- Score ponderado y señal general
    ROUND(s.score_ponderado::NUMERIC, 2)  AS score_ponderado,
    s.condiciones_ok,
    s.senal                               AS senal_general,

    -- Condiciones individuales (detalle)
    s.cond_rsi,
    s.cond_macd,
    s.cond_sma21,
    s.cond_sma50,
    s.cond_sma200,
    s.cond_momentum

FROM precios_diarios      p
JOIN activos              a  ON p.ticker = a.ticker
JOIN indicadores_tecnicos i  ON p.ticker = i.ticker  AND p.fecha = i.fecha
JOIN scoring_tecnico      s  ON p.ticker = s.ticker  AND p.fecha = s.fecha
JOIN macd_lag             ml ON p.ticker = ml.ticker AND p.fecha = ml.fecha
ORDER BY p.fecha DESC, a.sector, p.ticker;
