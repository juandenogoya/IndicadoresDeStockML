-- =============================================================================
-- bt_analisis.sql
-- Queries de analisis sobre resultados del backtesting historico.
-- DB: local PostgreSQL (localhost:5432/activos_ml)
-- Tablas: bt_hist_estrategias, bt_hist_operaciones,
--         bt_hist_metricas_diarias, bt_hist_candidatos
-- =============================================================================
-- INDICE DE SECCIONES
--   1. Resumen ejecutivo comparativo
--   2. Equity curves (retorno acumulado dia a dia)
--   3. Drawdown curves
--   4. Analisis de operaciones — ganadores / perdedores
--   5. Distribucion de motivos de salida
--   6. Win rate y retorno por sector (TECH_SECTOR y COMBO)
--   7. Analisis de duracion de posiciones
--   8. Mejores y peores tickers por estrategia
--   9. Actividad diaria — entradas y salidas por dia
--  10. Analisis de candidatos (conversion rate y score distribution)
--  11. Comparacion C1 vs C2 por estrategia
--  12. Correlacion score de entrada vs retorno
-- =============================================================================


-- =============================================================================
-- 1. RESUMEN EJECUTIVO COMPARATIVO
-- Una sola tabla con las 6 instancias ordenadas por periodo y retorno.
-- =============================================================================

SELECT
    e.id                                          AS bt_id,
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                                           AS estrategia,
    e.fecha_bt_inicio                             AS desde,
    e.fecha_bt_fin                                AS hasta,
    e.dias_habiles,
    e.capital_inicial,
    e.capital_final,
    ROUND(e.retorno_total_pct, 2)                 AS retorno_pct,
    ROUND(e.drawdown_max_pct,  2)                 AS dd_max_pct,
    ROUND(e.win_rate_pct,      2)                 AS win_rate_pct,
    ROUND(e.profit_factor,     4)                 AS profit_factor,
    e.total_operaciones                           AS trades,
    ROUND(e.dias_promedio_pos, 1)                 AS dias_prom_pos,
    ROUND(e.sharpe_simplificado, 4)               AS sharpe
FROM bt_hist_estrategias e
WHERE e.estado = 'completado'
ORDER BY e.fecha_bt_inicio, e.retorno_total_pct DESC;


-- =============================================================================
-- 2. EQUITY CURVES — retorno acumulado diario por estrategia
-- Usar para graficar las 3 curvas superpuestas por periodo.
-- =============================================================================

-- C1: Jun-Dic 2025
SELECT
    m.fecha,
    MAX(CASE WHEN e.logica = 'tecnico_sectorial' THEN ROUND(m.retorno_acumulado_pct, 4) END) AS tech_sector,
    MAX(CASE WHEN e.logica = 'combo_tech_candle' THEN ROUND(m.retorno_acumulado_pct, 4) END) AS combo,
    MAX(CASE WHEN e.logica = 'smc_estructura'    THEN ROUND(m.retorno_acumulado_pct, 4) END) AS smc
FROM bt_hist_metricas_diarias m
JOIN bt_hist_estrategias e ON e.id = m.bt_id
WHERE e.fecha_bt_inicio = '2025-06-01'
  AND e.estado = 'completado'
GROUP BY m.fecha
ORDER BY m.fecha;

-- C2: 2024-2025 completo
SELECT
    m.fecha,
    MAX(CASE WHEN e.logica = 'tecnico_sectorial' THEN ROUND(m.retorno_acumulado_pct, 4) END) AS tech_sector,
    MAX(CASE WHEN e.logica = 'combo_tech_candle' THEN ROUND(m.retorno_acumulado_pct, 4) END) AS combo,
    MAX(CASE WHEN e.logica = 'smc_estructura'    THEN ROUND(m.retorno_acumulado_pct, 4) END) AS smc
FROM bt_hist_metricas_diarias m
JOIN bt_hist_estrategias e ON e.id = m.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.estado = 'completado'
GROUP BY m.fecha
ORDER BY m.fecha;


-- =============================================================================
-- 3. DRAWDOWN CURVES — profundidad del drawdown dia a dia
-- =============================================================================

-- C2: 2024-2025 (periodo mas representativo)
SELECT
    m.fecha,
    MAX(CASE WHEN e.logica = 'tecnico_sectorial' THEN ROUND(m.drawdown_pct, 4) END) AS tech_sector_dd,
    MAX(CASE WHEN e.logica = 'combo_tech_candle' THEN ROUND(m.drawdown_pct, 4) END) AS combo_dd,
    MAX(CASE WHEN e.logica = 'smc_estructura'    THEN ROUND(m.drawdown_pct, 4) END) AS smc_dd
FROM bt_hist_metricas_diarias m
JOIN bt_hist_estrategias e ON e.id = m.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.estado = 'completado'
GROUP BY m.fecha
ORDER BY m.fecha;

-- Periodos de maximo drawdown por estrategia (C2)
SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                                AS estrategia,
    m.fecha                            AS fecha_peor_dd,
    ROUND(m.drawdown_pct, 4)           AS drawdown_pct,
    ROUND(m.capital_total, 2)          AS capital_en_ese_dia
FROM bt_hist_metricas_diarias m
JOIN bt_hist_estrategias e ON e.id = m.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.estado = 'completado'
  AND m.drawdown_pct = (
      SELECT MIN(m2.drawdown_pct)
      FROM bt_hist_metricas_diarias m2
      WHERE m2.bt_id = m.bt_id
  )
ORDER BY m.drawdown_pct;


-- =============================================================================
-- 4. ANALISIS DE OPERACIONES — ganadores / perdedores
-- =============================================================================

-- Distribucion de PnL por estrategia y periodo
SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                                            AS estrategia,
    e.fecha_bt_inicio                              AS desde,
    COUNT(*)                                       AS total_trades,
    SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END)    AS ganadores,
    SUM(CASE WHEN o.pnl <= 0 THEN 1 ELSE 0 END)   AS perdedores,
    ROUND(AVG(o.pnl), 2)                           AS pnl_promedio,
    ROUND(AVG(CASE WHEN o.pnl > 0 THEN o.pnl END), 2) AS ganancia_prom,
    ROUND(AVG(CASE WHEN o.pnl <= 0 THEN o.pnl END), 2) AS perdida_prom,
    ROUND(MAX(o.pnl), 2)                           AS mejor_trade,
    ROUND(MIN(o.pnl), 2)                           AS peor_trade,
    ROUND(SUM(CASE WHEN o.pnl > 0 THEN o.pnl ELSE 0 END), 2) AS suma_ganancias,
    ROUND(SUM(CASE WHEN o.pnl <= 0 THEN o.pnl ELSE 0 END), 2) AS suma_perdidas
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
WHERE e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'  -- excluir posiciones abiertas al cierre
GROUP BY e.logica, e.fecha_bt_inicio
ORDER BY e.fecha_bt_inicio, e.logica;

-- Top 10 mejores trades (C2)
SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END           AS estrategia,
    o.ticker,
    o.fecha_entrada,
    o.fecha_salida,
    o.dias_abierta,
    ROUND(o.precio_entrada, 2) AS p_entrada,
    ROUND(o.precio_salida,  2) AS p_salida,
    ROUND(o.pnl, 2)            AS pnl,
    ROUND(o.pnl_pct, 2)        AS pnl_pct,
    o.motivo_salida
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'
ORDER BY o.pnl DESC
LIMIT 10;

-- Top 10 peores trades (C2)
SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END           AS estrategia,
    o.ticker,
    o.fecha_entrada,
    o.fecha_salida,
    o.dias_abierta,
    ROUND(o.precio_entrada, 2) AS p_entrada,
    ROUND(o.precio_salida,  2) AS p_salida,
    ROUND(o.pnl, 2)            AS pnl,
    ROUND(o.pnl_pct, 2)        AS pnl_pct,
    o.motivo_salida
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'
ORDER BY o.pnl ASC
LIMIT 10;


-- =============================================================================
-- 5. DISTRIBUCION DE MOTIVOS DE SALIDA
-- =============================================================================

SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                          AS estrategia,
    e.fecha_bt_inicio            AS desde,
    o.motivo_salida,
    COUNT(*)                     AS cantidad,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (
        PARTITION BY o.bt_id
    ), 1)                        AS pct_del_total,
    ROUND(AVG(o.pnl), 2)         AS pnl_prom,
    ROUND(SUM(o.pnl), 2)         AS pnl_total,
    SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) AS ganadores,
    SUM(CASE WHEN o.pnl <= 0 THEN 1 ELSE 0 END) AS perdedores
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
WHERE e.estado = 'completado'
GROUP BY o.bt_id, e.logica, e.fecha_bt_inicio, o.motivo_salida
ORDER BY e.fecha_bt_inicio, e.logica, cantidad DESC;


-- =============================================================================
-- 6. WIN RATE Y RETORNO POR SECTOR (TECH_SECTOR_v1 y COMBO_v1, C2)
-- Sector via JOIN con activos (no en detalle_entrada JSONB).
-- =============================================================================

SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
    END                                           AS estrategia,
    a.sector,
    COUNT(*)                                      AS trades,
    ROUND(SUM(o.pnl), 2)                          AS pnl_total,
    ROUND(AVG(o.pnl_pct), 2)                      AS retorno_prom_pct,
    ROUND(
        SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    1)                                            AS win_rate_pct,
    SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END)   AS gana,
    SUM(CASE WHEN o.pnl <= 0 THEN 1 ELSE 0 END)  AS pierde
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
JOIN activos a ON a.ticker = o.ticker
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.logica IN ('tecnico_sectorial', 'combo_tech_candle')
  AND e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'
  AND a.sector IS NOT NULL
GROUP BY e.logica, a.sector
ORDER BY e.logica, pnl_total DESC;

-- Resumen: sectores con edge positivo vs negativo (TECH_SECTOR_v1 C2)
SELECT
    a.sector,
    COUNT(*)                        AS trades,
    ROUND(SUM(o.pnl), 2)           AS pnl_total,
    ROUND(AVG(o.pnl_pct), 2)       AS retorno_prom_pct,
    ROUND(
        SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    1)                              AS win_rate_pct,
    CASE WHEN SUM(o.pnl) > 0 THEN 'POSITIVO' ELSE 'NEGATIVO' END AS edge
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
JOIN activos a ON a.ticker = o.ticker
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.logica = 'tecnico_sectorial'
  AND e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'
  AND a.sector IS NOT NULL
GROUP BY a.sector
ORDER BY pnl_total DESC;


-- =============================================================================
-- 7. ANALISIS DE DURACION DE POSICIONES
-- =============================================================================

SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                                    AS estrategia,
    e.fecha_bt_inicio                      AS desde,
    ROUND(AVG(o.dias_abierta), 1)          AS dias_prom,
    MIN(o.dias_abierta)                    AS dias_min,
    MAX(o.dias_abierta)                    AS dias_max,
    PERCENTILE_CONT(0.5) WITHIN GROUP
        (ORDER BY o.dias_abierta)          AS dias_mediana,
    -- Retorno promedio segun duracion
    ROUND(AVG(CASE WHEN o.dias_abierta <= 5  THEN o.pnl_pct END), 2) AS retorno_pct_1_5d,
    ROUND(AVG(CASE WHEN o.dias_abierta BETWEEN 6 AND 10 THEN o.pnl_pct END), 2) AS retorno_pct_6_10d,
    ROUND(AVG(CASE WHEN o.dias_abierta BETWEEN 11 AND 15 THEN o.pnl_pct END), 2) AS retorno_pct_11_15d,
    ROUND(AVG(CASE WHEN o.dias_abierta > 15 THEN o.pnl_pct END), 2) AS retorno_pct_15d_mas
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
WHERE e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'
  AND o.dias_abierta IS NOT NULL
GROUP BY e.logica, e.fecha_bt_inicio
ORDER BY e.fecha_bt_inicio, e.logica;


-- =============================================================================
-- 8. MEJORES Y PEORES TICKERS POR ESTRATEGIA (C2)
-- =============================================================================

-- Tickers con mayor PnL acumulado (minimo 3 trades)
SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                                AS estrategia,
    o.ticker,
    COUNT(*)                           AS veces_operado,
    ROUND(SUM(o.pnl), 2)              AS pnl_total,
    ROUND(AVG(o.pnl_pct), 2)          AS retorno_prom_pct,
    SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) AS ganadores
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'
GROUP BY e.logica, o.ticker
HAVING COUNT(*) >= 3
ORDER BY e.logica, pnl_total DESC;

-- Top 5 mejores y 5 peores por estrategia (C2, todos los trades)
WITH ranked AS (
    SELECT
        e.logica,
        o.ticker,
        COUNT(*)                  AS trades,
        ROUND(SUM(o.pnl), 2)      AS pnl_total,
        ROUND(AVG(o.pnl_pct), 2)  AS retorno_prom_pct,
        ROW_NUMBER() OVER (PARTITION BY e.logica ORDER BY SUM(o.pnl) DESC) AS rk_top,
        ROW_NUMBER() OVER (PARTITION BY e.logica ORDER BY SUM(o.pnl) ASC)  AS rk_bot
    FROM bt_hist_operaciones o
    JOIN bt_hist_estrategias e ON e.id = o.bt_id
    WHERE e.fecha_bt_inicio = '2024-01-01'
      AND e.estado = 'completado'
      AND o.motivo_salida != 'FIN_PERIODO'
    GROUP BY e.logica, o.ticker
)
SELECT
    CASE logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                     AS estrategia,
    CASE WHEN rk_top <= 5 THEN 'TOP_5' ELSE 'WORST_5' END AS categoria,
    ticker,
    trades,
    pnl_total,
    retorno_prom_pct
FROM ranked
WHERE rk_top <= 5 OR rk_bot <= 5
ORDER BY logica, pnl_total DESC;


-- =============================================================================
-- 9. ACTIVIDAD DIARIA — entradas y salidas por dia (C2, TECH_SECTOR)
-- Util para ver si hay concentracion de actividad en ciertos periodos.
-- =============================================================================

SELECT
    fecha,
    SUM(entradas)  AS entradas,
    SUM(salidas)   AS salidas,
    SUM(pnl_dia)   AS pnl_dia
FROM (
    -- Entradas
    SELECT o.fecha_entrada AS fecha, COUNT(*) AS entradas, 0 AS salidas, 0.0 AS pnl_dia
    FROM bt_hist_operaciones o
    JOIN bt_hist_estrategias e ON e.id = o.bt_id
    WHERE e.fecha_bt_inicio = '2024-01-01' AND e.logica = 'tecnico_sectorial'
    GROUP BY o.fecha_entrada
    UNION ALL
    -- Salidas
    SELECT o.fecha_salida, 0, COUNT(*), SUM(o.pnl)
    FROM bt_hist_operaciones o
    JOIN bt_hist_estrategias e ON e.id = o.bt_id
    WHERE e.fecha_bt_inicio = '2024-01-01' AND e.logica = 'tecnico_sectorial'
      AND o.fecha_salida IS NOT NULL
    GROUP BY o.fecha_salida
) t
GROUP BY fecha
ORDER BY fecha;


-- =============================================================================
-- 10. ANALISIS DE CANDIDATOS — conversion rate y score distribution
-- =============================================================================

-- Tasa de conversion por estrategia y periodo
SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                                               AS estrategia,
    e.fecha_bt_inicio                                 AS desde,
    COUNT(*)                                          AS candidatos_evaluados,
    SUM(CASE WHEN c.entro THEN 1 ELSE 0 END)          AS entraron,
    ROUND(
        SUM(CASE WHEN c.entro THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    1)                                                AS conversion_pct,
    ROUND(AVG(c.score), 3)                            AS score_prom_todos,
    ROUND(AVG(CASE WHEN c.entro THEN c.score END), 3) AS score_prom_entraron,
    ROUND(AVG(CASE WHEN NOT c.entro THEN c.score END), 3) AS score_prom_rechazados
FROM bt_hist_candidatos c
JOIN bt_hist_estrategias e ON e.id = c.bt_id
WHERE e.estado = 'completado'
GROUP BY e.logica, e.fecha_bt_inicio
ORDER BY e.fecha_bt_inicio, e.logica;

-- Motivos de rechazo mas frecuentes (por estrategia, C2)
SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                         AS estrategia,
    COALESCE(c.motivo_skip, 'entro')  AS motivo,
    COUNT(*)                    AS cantidad,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY c.bt_id), 1) AS pct
FROM bt_hist_candidatos c
JOIN bt_hist_estrategias e ON e.id = c.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.estado = 'completado'
GROUP BY c.bt_id, e.logica, c.motivo_skip
ORDER BY e.logica, cantidad DESC;


-- =============================================================================
-- 11. COMPARACION C1 vs C2 — consistencia de las estrategias
-- =============================================================================

SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                               AS estrategia,
    CASE
        WHEN e.fecha_bt_inicio = '2025-06-01' THEN 'C1_jun_dic_2025'
        WHEN e.fecha_bt_inicio = '2024-01-01' THEN 'C2_2024_2025'
    END                               AS periodo,
    ROUND(e.retorno_total_pct, 2)     AS retorno_pct,
    ROUND(e.drawdown_max_pct,  2)     AS dd_max_pct,
    ROUND(e.win_rate_pct,      2)     AS win_rate_pct,
    ROUND(e.profit_factor,     4)     AS profit_factor,
    e.total_operaciones               AS trades,
    ROUND(e.sharpe_simplificado, 4)   AS sharpe,
    -- Retorno anualizado aproximado
    ROUND(
        e.retorno_total_pct / e.dias_habiles * 252,
    2)                                AS retorno_anualizado_aprox
FROM bt_hist_estrategias e
WHERE e.estado = 'completado'
ORDER BY e.logica, e.fecha_bt_inicio;

-- Delta C1 vs C2: cuanto se mantuvo el retorno al ampliar el periodo
SELECT
    CASE c1.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
        WHEN 'smc_estructura'    THEN 'SMC_v1'
    END                                                       AS estrategia,
    ROUND(c1.retorno_total_pct, 2)                            AS retorno_c1,
    ROUND(c2.retorno_total_pct, 2)                            AS retorno_c2,
    ROUND(c2.retorno_total_pct - c1.retorno_total_pct, 2)    AS delta_retorno,
    ROUND(c1.win_rate_pct, 1)                                 AS win_rate_c1,
    ROUND(c2.win_rate_pct, 1)                                 AS win_rate_c2,
    ROUND(c1.profit_factor, 4)                                AS pf_c1,
    ROUND(c2.profit_factor, 4)                                AS pf_c2
FROM bt_hist_estrategias c1
JOIN bt_hist_estrategias c2
    ON c1.logica = c2.logica
   AND c1.fecha_bt_inicio = '2025-06-01'
   AND c2.fecha_bt_inicio = '2024-01-01'
WHERE c1.estado = 'completado' AND c2.estado = 'completado'
ORDER BY c2.retorno_total_pct DESC;


-- =============================================================================
-- 12. CORRELACION SCORE DE ENTRADA VS RETORNO (TECH_SECTOR y COMBO, C2)
-- Verifica si un score mas alto en el momento de entrada predice mejor PnL.
-- =============================================================================

SELECT
    CASE e.logica
        WHEN 'tecnico_sectorial' THEN 'TECH_SECTOR_v1'
        WHEN 'combo_tech_candle' THEN 'COMBO_v1'
    END                                        AS estrategia,
    ROUND(o.score_entrada, 1)                  AS score_entrada,
    COUNT(*)                                   AS trades,
    SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) AS ganadores,
    ROUND(
        SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    1)                                         AS win_rate_pct,
    ROUND(AVG(o.pnl), 2)                       AS pnl_prom,
    ROUND(AVG(o.pnl_pct), 2)                   AS retorno_prom_pct,
    ROUND(SUM(o.pnl), 2)                       AS pnl_total
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.logica IN ('tecnico_sectorial', 'combo_tech_candle')
  AND e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'
GROUP BY e.logica, ROUND(o.score_entrada, 1)
ORDER BY e.logica, score_entrada;

-- Para SMC: score_entrada (0,1,2,3) vs retorno
SELECT
    o.score_entrada,
    COUNT(*)                                    AS trades,
    SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) AS ganadores,
    ROUND(
        SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    1)                                          AS win_rate_pct,
    ROUND(AVG(o.pnl_pct), 2)                   AS retorno_prom_pct,
    ROUND(SUM(o.pnl), 2)                        AS pnl_total
FROM bt_hist_operaciones o
JOIN bt_hist_estrategias e ON e.id = o.bt_id
WHERE e.fecha_bt_inicio = '2024-01-01'
  AND e.logica = 'smc_estructura'
  AND e.estado = 'completado'
  AND o.motivo_salida != 'FIN_PERIODO'
GROUP BY o.score_entrada
ORDER BY o.score_entrada;
