"""
SQL templates para las tools del MCP server.

Todas son SELECT-only. Parametros via placeholders asyncpg ($1, $2, ...)
para prevenir inyeccion SQL. Nunca f-strings con valores del usuario.

Convencion de nombres:
  SQL_<DOMINIO>_<OPERACION>

Tablas fuente principales:
  precios_diarios, features_precio_accion, features_market_structure,
  indicadores_tecnicos, opciones_resumen_diario, opciones_zscore_diario,
  alertas_scanner, activos, ticker_zscore_diario, features_regimen_macro,
  futuros_diarios, bt_hist_estrategias, bt_hist_operaciones,
  bt_hist_metricas_diarias, ft_estrategias, ft_posiciones_diarias,
  ft_operaciones, ft_metricas_diarias.
"""

# ── Exploration: list_tables ───────────────────────────────────────────────────

SQL_LIST_TABLES = """
    SELECT
        t.table_name,
        NULLIF(GREATEST(c.reltuples::bigint, -1), -1) AS row_count_estimate
    FROM information_schema.tables t
    LEFT JOIN pg_class c
        ON  c.relname       = t.table_name
        AND c.relnamespace  = (
            SELECT oid FROM pg_namespace WHERE nspname = 'public'
        )
    WHERE t.table_schema = 'public'
      AND t.table_type   = 'BASE TABLE'
      AND ($1::text IS NULL OR t.table_name ILIKE $1)
    ORDER BY t.table_name
"""
# $1: patron ILIKE o NULL para todas las tablas
# row_count_estimate: NULL = tabla no analizada (ANALYZE pendiente)
#                     0..N = estimacion de pg_class.reltuples


# ── Exploration: describe_table (3 queries separadas) ─────────────────────────

SQL_DESCRIBE_TABLE_CHECK = """
    SELECT reltuples::bigint AS row_count_estimate
    FROM   pg_class
    WHERE  relname        = $1
      AND  relnamespace   = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
"""
# $1: table_name. Retorna 1 fila si existe, 0 filas si no existe.

SQL_DESCRIBE_TABLE_COLUMNS = """
    SELECT
        column_name AS name,
        CASE
            WHEN character_maximum_length IS NOT NULL
                THEN data_type || '(' || character_maximum_length || ')'
            ELSE data_type
        END                     AS type,
        (is_nullable = 'YES')   AS nullable,
        column_default          AS "default"
    FROM   information_schema.columns
    WHERE  table_schema = 'public'
      AND  table_name   = $1
    ORDER  BY ordinal_position
"""
# $1: table_name
# column_default se devuelve tal cual (puede ser "nextval('...')" para serial)

SQL_DESCRIBE_TABLE_INDEXES = """
    SELECT
        i.relname                       AS name,
        pg_get_indexdef(ix.indexrelid)  AS definition
    FROM   pg_index      ix
    JOIN   pg_class      t  ON t.oid = ix.indrelid
    JOIN   pg_class      i  ON i.oid = ix.indexrelid
    JOIN   pg_namespace  n  ON n.oid = t.relnamespace
    WHERE  n.nspname = 'public'
      AND  t.relname  = $1
    ORDER  BY i.relname
"""
# $1: table_name


# ── Exploration: list_tickers ─────────────────────────────────────────────────

SQL_LIST_TICKERS = """
    SELECT
        ticker,
        sector,
        industry,
        modelo_asignado,
        activo
    FROM   activos
    WHERE  ($1::text IS NULL OR LOWER(sector) = LOWER($1))
      AND  (NOT $2        OR modelo_asignado IS NOT NULL)
    ORDER  BY ticker
"""
# $1: sector (NULL = sin filtro de sector)
# $2: only_ml_active boolean (FALSE = sin filtro, TRUE = solo con modelo)


# ── Stocks: get_price_history ─────────────────────────────────────────────────

SQL_PRICE_HISTORY = """
    SELECT
        ticker,
        fecha,
        open,
        high,
        low,
        close,
        volume,
        adj_close
    FROM   precios_diarios
    WHERE  ticker = $1
      AND  fecha  >= $2
      AND  fecha  <= $3
    ORDER  BY fecha DESC
"""
# $1: ticker  $2: desde (date)  $3: hasta (date)
# Excluye: id, created_at (ruido para el LLM)


# ── Stocks: get_technical_indicators ─────────────────────────────────────────

SQL_TECHNICAL_INDICATORS = """
    SELECT
        ticker,
        fecha,
        sma21, sma50, sma200,
        dist_sma21, dist_sma50, dist_sma200,
        rsi14,
        momentum,
        macd, macd_signal, macd_hist,
        atr14,
        bb_upper, bb_middle, bb_lower,
        obv,
        vol_relativo,
        adx
    FROM   indicadores_tecnicos
    WHERE  ticker = $1
      AND  fecha  >= $2
      AND  fecha  <= $3
    ORDER  BY fecha DESC
"""
# $1: ticker  $2: desde (date)  $3: hasta (date)
# Excluye: id, created_at, updated_at


# ── Stocks: get_price_action ──────────────────────────────────────────────────

SQL_PRICE_ACTION = """
    SELECT
        ticker,
        fecha,
        body_pct, body_ratio,
        upper_shadow_pct, lower_shadow_pct,
        es_alcista,
        gap_apertura_pct, rango_diario_pct, rango_rel_atr,
        clv,
        patron_doji, patron_hammer, patron_shooting_star,
        patron_marubozu, patron_engulfing_bull, patron_engulfing_bear,
        inside_bar, outside_bar,
        body_pct_ma5,
        velas_alcistas_5d, velas_alcistas_10d,
        rango_expansion,
        dist_max_20d, dist_min_20d, pos_rango_20d,
        tendencia_velas,
        vol_ratio_5d, vol_spike, up_vol_5d,
        ad_flow, chaikin_mf_20,
        vol_price_confirm, vol_price_diverge
    FROM   features_precio_accion
    WHERE  ticker = $1
      AND  fecha  >= $2
      AND  fecha  <= $3
    ORDER  BY fecha DESC
"""
# $1: ticker  $2: desde (date)  $3: hasta (date)


# ── Stocks: get_market_structure ──────────────────────────────────────────────

SQL_MARKET_STRUCTURE = """
    SELECT
        ticker,
        fecha,
        is_sh_5, is_sl_5, estructura_5,
        dist_sh_5_pct, dist_sl_5_pct,
        dias_sh_5, dias_sl_5,
        impulso_5_pct,
        bos_bull_5, bos_bear_5, choch_bull_5, choch_bear_5,
        is_sh_10, is_sl_10, estructura_10,
        dist_sh_10_pct, dist_sl_10_pct,
        dias_sh_10, dias_sl_10,
        impulso_10_pct,
        bos_bull_10, bos_bear_10, choch_bull_10, choch_bear_10
    FROM   features_market_structure
    WHERE  ticker = $1
      AND  fecha  >= $2
      AND  fecha  <= $3
    ORDER  BY fecha DESC
"""
# $1: ticker  $2: desde (date)  $3: hasta (date)


# ── Opciones: rango de fechas disponibles ─────────────────────────────────────

SQL_OPTIONS_DATE_RANGE = """
    WITH recent AS (
        SELECT DISTINCT fecha_snapshot
        FROM   opciones_snapshot
        WHERE  ticker = $1
        ORDER  BY fecha_snapshot DESC
        LIMIT  $2
    )
    SELECT
        MAX(fecha_snapshot) AS max_fecha,
        MIN(fecha_snapshot) AS cutoff_fecha
    FROM recent
"""
# $1: ticker  $2: dias_historia (int)
# Retorna max_fecha (snapshot mas reciente) y cutoff_fecha (el mas antiguo
# dentro de la ventana de N snapshots). Si el ticker no existe, ambos NULL.


# ── Opciones: tendencia diaria macro ─────────────────────────────────────────

SQL_OPTIONS_TENDENCIA = """
    SELECT
        r.fecha,
        r.call_vol,
        r.put_vol,
        r.pcr_vol,
        r.call_oi,
        r.put_oi,
        r.pcr_oi,
        r.iv_call_avg,
        r.iv_put_avg,
        r.precio_sub,
        r.n_contratos,
        z.vol_total_zscore,
        z.pcr_vol_zscore,
        z.iv_zscore,
        z.vol_relativo,
        z.percentil_vol
    FROM   opciones_resumen_diario r
    LEFT   JOIN opciones_zscore_diario z
           ON  z.ticker = r.ticker
           AND z.fecha  = r.fecha
    WHERE  r.ticker = $1
      AND  r.fecha  >= $2
    ORDER  BY r.fecha DESC
"""
# $1: ticker  $2: cutoff_fecha
# JOIN LEFT porque zscore puede no existir para todos los dias
# iv_skew (iv_put - iv_call) se calcula en Python


# ── Opciones: PCR por vencimiento, serie diaria ───────────────────────────────

SQL_OPTIONS_PCR_POR_VENCIMIENTO = """
    SELECT
        vencimiento,
        fecha_snapshot,
        (vencimiento - $3::date)::int                                            AS dias_a_venc,
        SUM(CASE WHEN tipo = 'call' THEN COALESCE(volumen, 0) ELSE 0 END)       AS call_vol,
        SUM(CASE WHEN tipo = 'put'  THEN COALESCE(volumen, 0) ELSE 0 END)       AS put_vol,
        SUM(CASE WHEN tipo = 'call' THEN COALESCE(open_interest, 0) ELSE 0 END) AS call_oi,
        SUM(CASE WHEN tipo = 'put'  THEN COALESCE(open_interest, 0) ELSE 0 END) AS put_oi
    FROM   opciones_snapshot
    WHERE  ticker          = $1
      AND  fecha_snapshot >= $2
      AND  vencimiento     > $3
    GROUP  BY vencimiento, fecha_snapshot
    ORDER  BY vencimiento ASC, fecha_snapshot DESC
"""
# $1: ticker  $2: cutoff_fecha  $3: max_fecha
# vencimiento > max_fecha = solo contratos vivos
# dias_a_venc calculado desde max_fecha (constante por vencimiento)
# PCR se calcula en Python para manejar division por cero limpiamente


# ── Opciones: acumulacion de OI — top 10 calls + top 10 puts ─────────────────

SQL_OPTIONS_ACUMULACION_OI = """
    WITH date_bounds AS (
        SELECT
            MIN(fecha_snapshot) AS fecha_ini,
            MAX(fecha_snapshot) AS fecha_fin
        FROM   opciones_snapshot
        WHERE  ticker          = $1
          AND  fecha_snapshot >= $2
    ),
    snap_fin AS (
        SELECT
            s.vencimiento,
            s.tipo,
            s.strike,
            s.open_interest                                                     AS oi_fin,
            s.iv                                                                AS iv_actual,
            s.precio_subyacente
        FROM   opciones_snapshot s
        CROSS  JOIN date_bounds d
        WHERE  s.ticker         = $1
          AND  s.fecha_snapshot = d.fecha_fin
          AND  s.vencimiento    > d.fecha_fin
          AND  s.open_interest  > 0
    ),
    snap_ini AS (
        SELECT
            s.vencimiento,
            s.tipo,
            s.strike,
            s.open_interest AS oi_ini
        FROM   opciones_snapshot s
        CROSS  JOIN date_bounds d
        WHERE  s.ticker         = $1
          AND  s.fecha_snapshot = d.fecha_ini
    ),
    changes AS (
        SELECT
            f.vencimiento,
            f.tipo,
            f.strike,
            COALESCE(i.oi_ini, 0)                                              AS oi_inicio,
            f.oi_fin,
            (f.oi_fin - COALESCE(i.oi_ini, 0))                                AS delta_oi,
            f.iv_actual,
            f.precio_subyacente,
            ROUND(
                ((f.strike / NULLIF(f.precio_subyacente, 0)) - 1) * 100, 2
            )                                                                  AS moneyness_pct
        FROM   snap_fin  f
        LEFT   JOIN snap_ini i USING (vencimiento, tipo, strike)
    )
    SELECT * FROM (
        (SELECT * FROM changes WHERE tipo = 'call' ORDER BY delta_oi DESC LIMIT 10)
        UNION ALL
        (SELECT * FROM changes WHERE tipo = 'put'  ORDER BY delta_oi DESC LIMIT 10)
    ) ranked
    ORDER  BY tipo, delta_oi DESC
"""
# $1: ticker  $2: cutoff_fecha
# snap_fin: posiciones al ultimo dia del periodo (contratos vivos)
# snap_ini: LEFT JOIN = contratos nuevos en el periodo tienen oi_inicio=0
# delta_oi positivo = acumulacion de posiciones, negativo = cierre
# moneyness_pct = (strike/precio_sub - 1)*100  (positivo = strike sobre precio)


# ══════════════════════════════════════════════════════════════════════════════
# Overview: get_ticker_overview
# ══════════════════════════════════════════════════════════════════════════════

# ── Overview: perfil del activo ───────────────────────────────────────────────

SQL_OVERVIEW_PERFIL = """
    SELECT
        sector,
        industry,
        modelo_asignado,
        activo
    FROM   activos
    WHERE  ticker = $1
"""
# $1: ticker. Retorna 1 fila si existe, 0 filas si el ticker no esta en activos.


# ── Overview: precio OHLCV (ultimos 21 dias) ──────────────────────────────────

SQL_OVERVIEW_PRECIO = """
    SELECT
        fecha,
        open,
        high,
        low,
        close,
        volume,
        adj_close
    FROM   precios_diarios
    WHERE  ticker = $1
    ORDER  BY fecha DESC
    LIMIT  21
"""
# $1: ticker. 21 dias = suficiente para variaciones 1d, 5d y 20d.
# Filas en orden DESC: [0]=hoy, [1]=ayer, [5]=5d atras, [20]=20d atras.


# ── Overview: indicadores tecnicos (ultimo dia) ───────────────────────────────

SQL_OVERVIEW_TECNICOS = """
    SELECT
        fecha,
        sma21, sma50, sma200,
        dist_sma21, dist_sma50, dist_sma200,
        rsi14, momentum,
        macd, macd_signal, macd_hist,
        atr14,
        bb_upper, bb_middle, bb_lower,
        obv, vol_relativo, adx
    FROM   indicadores_tecnicos
    WHERE  ticker = $1
    ORDER  BY fecha DESC
    LIMIT  1
"""
# $1: ticker. Ultimo dia disponible.


# ── Overview: price action (ultimo dia) ───────────────────────────────────────

SQL_OVERVIEW_PRICE_ACTION = """
    SELECT
        fecha,
        body_pct, body_ratio,
        upper_shadow_pct, lower_shadow_pct,
        es_alcista,
        gap_apertura_pct, rango_diario_pct, rango_rel_atr, clv,
        patron_doji, patron_hammer, patron_shooting_star,
        patron_marubozu, patron_engulfing_bull, patron_engulfing_bear,
        inside_bar, outside_bar,
        tendencia_velas,
        vol_spike, vol_price_confirm, vol_price_diverge
    FROM   features_precio_accion
    WHERE  ticker = $1
    ORDER  BY fecha DESC
    LIMIT  1
"""
# $1: ticker. Subset de columnas relevantes para el overview.


# ── Overview: market structure (ultimo dia) ───────────────────────────────────

SQL_OVERVIEW_MARKET_STRUCTURE = """
    SELECT
        fecha,
        estructura_5,
        bos_bull_5, bos_bear_5, choch_bull_5, choch_bear_5,
        dist_sh_5_pct, dist_sl_5_pct, impulso_5_pct,
        estructura_10,
        bos_bull_10, bos_bear_10, choch_bull_10, choch_bear_10,
        dist_sh_10_pct, dist_sl_10_pct, impulso_10_pct
    FROM   features_market_structure
    WHERE  ticker = $1
    ORDER  BY fecha DESC
    LIMIT  1
"""
# $1: ticker. Solo señales de estructura (BOS/CHoCH) y distancias clave.


# ── Overview: z-score de volumen (ultimo dia) ─────────────────────────────────

SQL_OVERVIEW_VOLUMEN_ZSCORE = """
    SELECT *
    FROM   ticker_zscore_diario
    WHERE  ticker = $1
    ORDER  BY fecha DESC
    LIMIT  1
"""
# $1: ticker. SELECT * porque las columnas exactas dependen de la version del schema.


# ── Overview: resumen de opciones (ultimo dia disponible) ─────────────────────

SQL_OVERVIEW_OPCIONES_RESUMEN = """
    SELECT
        fecha,
        pcr_vol, pcr_oi,
        iv_call_avg, iv_put_avg,
        call_vol, put_vol,
        call_oi, put_oi,
        precio_sub,
        n_contratos
    FROM   opciones_resumen_diario
    WHERE  ticker = $1
    ORDER  BY fecha DESC
    LIMIT  1
"""
# $1: ticker. Resumen macro del ultimo snapshot de opciones.


# ── Overview: top OI por tipo en el ultimo snapshot ──────────────────────────

SQL_OVERVIEW_OPCIONES_TOP_OI = """
    WITH latest AS (
        SELECT MAX(fecha_snapshot) AS max_fecha
        FROM   opciones_snapshot
        WHERE  ticker = $1
    )
    SELECT
        s.tipo,
        s.strike,
        s.vencimiento,
        s.open_interest,
        s.iv,
        ROUND(
            ((s.strike / NULLIF(s.precio_subyacente, 0)) - 1) * 100, 2
        )                   AS moneyness_pct
    FROM   opciones_snapshot s
    CROSS  JOIN latest l
    WHERE  s.ticker         = $1
      AND  s.fecha_snapshot = l.max_fecha
      AND  s.vencimiento    > l.max_fecha
      AND  s.open_interest  > 0
    ORDER  BY s.tipo, s.open_interest DESC
"""
# $1: ticker. Contratos vivos con mayor OI en el ultimo snapshot.
# Top 3 calls + top 3 puts se filtran en Python.


# ══════════════════════════════════════════════════════════════════════════════
# Screener: screen_tickers
# ══════════════════════════════════════════════════════════════════════════════

SQL_SCREEN_TICKERS = """
    WITH
    -- Ultimo dato disponible por ticker en cada tabla fuente
    latest_tec AS (
        SELECT ticker, MAX(fecha) AS fecha
        FROM   indicadores_tecnicos
        GROUP  BY ticker
    ),
    latest_ms AS (
        SELECT ticker, MAX(fecha) AS fecha
        FROM   features_market_structure
        GROUP  BY ticker
    ),
    latest_pa AS (
        SELECT ticker, MAX(fecha) AS fecha
        FROM   features_precio_accion
        GROUP  BY ticker
    ),
    latest_alerta AS (
        SELECT ticker, MAX(scan_fecha) AS scan_fecha
        FROM   alertas_scanner
        GROUP  BY ticker
    ),
    latest_precio AS (
        SELECT ticker, MAX(fecha) AS fecha
        FROM   precios_diarios
        GROUP  BY ticker
    ),
    -- Vista unificada: estado actual de cada ticker
    base AS (
        SELECT
            a.ticker,
            a.sector,
            a.industry,
            a.modelo_asignado,
            -- Tecnicos
            t.rsi14,
            t.macd_hist,
            t.adx,
            t.sma200,
            t.vol_relativo,
            -- Market Structure (señales y estructura)
            ms.estructura_5,
            ms.choch_bull_5,
            ms.choch_bear_5,
            ms.bos_bull_5,
            ms.bos_bear_5,
            -- Price Action (patrones y features clave)
            pa.es_alcista,
            pa.vol_spike,
            pa.patron_engulfing_bull,
            pa.patron_engulfing_bear,
            pa.patron_hammer,
            pa.patron_shooting_star,
            pa.patron_marubozu,
            pa.patron_doji,
            -- Alertas ML
            al.alert_score,
            al.alert_nivel,
            -- Precio
            p.close,
            p.fecha        AS fecha_precio
        FROM   activos a
        LEFT   JOIN latest_tec    lte ON lte.ticker = a.ticker
        LEFT   JOIN indicadores_tecnicos t
               ON  t.ticker = lte.ticker AND t.fecha  = lte.fecha
        LEFT   JOIN latest_ms     lms ON lms.ticker = a.ticker
        LEFT   JOIN features_market_structure ms
               ON  ms.ticker = lms.ticker AND ms.fecha = lms.fecha
        LEFT   JOIN latest_pa     lpa ON lpa.ticker = a.ticker
        LEFT   JOIN features_precio_accion pa
               ON  pa.ticker = lpa.ticker AND pa.fecha  = lpa.fecha
        LEFT   JOIN latest_alerta lal ON lal.ticker = a.ticker
        LEFT   JOIN alertas_scanner al
               ON  al.ticker = lal.ticker AND al.scan_fecha = lal.scan_fecha
        LEFT   JOIN latest_precio lp  ON lp.ticker  = a.ticker
        LEFT   JOIN precios_diarios p
               ON  p.ticker  = lp.ticker  AND p.fecha  = lp.fecha
    )
    SELECT *
    FROM   base
    WHERE
        -- $1  solo_ml_activos (bool): TRUE = solo tickers con modelo RF
        (NOT $1 OR modelo_asignado IS NOT NULL)
        -- $2  sector (text | NULL)
        AND ($2::text    IS NULL OR LOWER(sector)    = LOWER($2))
        -- $3  rsi_min (float | NULL)
        AND ($3::float8  IS NULL OR rsi14            >= $3)
        -- $4  rsi_max (float | NULL)
        AND ($4::float8  IS NULL OR rsi14            <= $4)
        -- $5  adx_min (float | NULL)
        AND ($5::float8  IS NULL OR adx              >= $5)
        -- $6  vol_relativo_min (float | NULL)
        AND ($6::float8  IS NULL OR vol_relativo     >= $6)
        -- $7  sobre_sma200 (bool | NULL): TRUE=close>sma200, FALSE=close<=sma200
        AND ($7::boolean IS NULL OR
            CASE WHEN $7 THEN close > sma200 ELSE close <= sma200 END)
        -- $8  macd_direccion (text | NULL): 'alcista'|'bajista'
        AND ($8::text    IS NULL OR
            CASE $8::text
                WHEN 'alcista' THEN macd_hist > 0
                WHEN 'bajista' THEN macd_hist <= 0
                ELSE TRUE
            END)
        -- $9  estructura_5 (int | NULL): 1=alcista | 0=neutral | -1=bajista
        AND ($9::int     IS NULL OR estructura_5     = $9)
        -- $10 señal_smc (text | NULL): 'choch_bull'|'choch_bear'|'bos_bull'|'bos_bear'
        -- Nota: columnas choch_*/bos_* pueden ser smallint(0/1) o boolean segun version
        --       de la DB; ::int != 0 normaliza ambos tipos a boolean.
        AND ($10::text   IS NULL OR
            CASE $10::text
                WHEN 'choch_bull' THEN choch_bull_5::int != 0
                WHEN 'choch_bear' THEN choch_bear_5::int != 0
                WHEN 'bos_bull'   THEN bos_bull_5::int   != 0
                WHEN 'bos_bear'   THEN bos_bear_5::int   != 0
                ELSE TRUE
            END)
        -- $11 es_alcista (bool | NULL)
        AND ($11::boolean IS NULL OR (es_alcista::int != 0) = $11)
        -- $12 vol_spike (bool | NULL): TRUE = requiere spike de volumen
        AND ($12::boolean IS NULL OR COALESCE(vol_spike::int != 0, FALSE) = $12)
        -- $13 patron (text | NULL): 'engulfing_bull'|'hammer'|'doji'|etc.
        AND ($13::text   IS NULL OR
            CASE $13::text
                WHEN 'engulfing_bull' THEN patron_engulfing_bull::int != 0
                WHEN 'engulfing_bear' THEN patron_engulfing_bear::int != 0
                WHEN 'hammer'         THEN patron_hammer::int         != 0
                WHEN 'shooting_star'  THEN patron_shooting_star::int  != 0
                WHEN 'marubozu'       THEN patron_marubozu::int       != 0
                WHEN 'doji'           THEN patron_doji::int           != 0
                ELSE TRUE
            END)
        -- $14 alert_nivel (text | NULL): 'COMPRA_FUERTE'|'COMPRA'|'NEUTRO'|'VENTA'|'VENTA_FUERTE'
        AND ($14::text   IS NULL OR alert_nivel      = $14)
        -- $15 alert_score_min (float | NULL)
        AND ($15::float8 IS NULL OR alert_score      >= $15)
    ORDER BY
        CASE WHEN $16::text = 'rsi14'        THEN rsi14        END DESC NULLS LAST,
        CASE WHEN $16::text = 'adx'          THEN adx          END DESC NULLS LAST,
        CASE WHEN $16::text = 'vol_relativo' THEN vol_relativo  END DESC NULLS LAST,
        CASE WHEN $16::text = 'alert_score'
              OR  $16        IS NULL          THEN alert_score   END DESC NULLS LAST,
        ticker ASC
    LIMIT $17
"""
# Parametros (17 en total, opcion B: cada filtro es nullable -> IS NULL = sin filtro):
# $1  solo_ml_activos   $2  sector            $3  rsi_min
# $4  rsi_max           $5  adx_min           $6  vol_relativo_min
# $7  sobre_sma200      $8  macd_direccion    $9  estructura_5
# $10 señal_smc         $11 es_alcista        $12 vol_spike
# $13 patron            $14 alert_nivel       $15 alert_score_min
# $16 ordenar_por       $17 limit
#
# Columnas bool de detalle (choch_*, bos_*, patron_*) se usan solo para
# filtrar en SQL; en Python se sintetizan en señal_smc y patron_activo.


# ══════════════════════════════════════════════════════════════════════════════
# ML Alert History: get_ml_alert_history
# ══════════════════════════════════════════════════════════════════════════════

SQL_ML_ALERT_HISTORY = """
    SELECT
        ticker,
        scan_fecha::date        AS fecha_scan,
        precio_fecha,
        sector,
        precio_cierre,
        ml_prob_ganancia,
        ml_modelo_usado,
        -- Condiciones PA (sintetizadas como conteo)
        (pa_ev1 + pa_ev2 + pa_ev3 + pa_ev4)  AS condiciones_pa,
        condiciones_ok,
        -- Contexto market structure
        estructura_10,
        -- Score y nivel ML
        alert_score,
        alert_nivel,
        alert_detalle,
        -- Post-facto: retornos reales (null hasta que se verifique)
        retorno_1d_real,
        retorno_5d_real,
        retorno_20d_real,
        verificado
    FROM   alertas_scanner
    WHERE
        ($1::text    IS NULL OR ticker      = UPPER($1))
        AND ($2::date    IS NULL OR scan_fecha::date >= $2)
        AND ($3::date    IS NULL OR scan_fecha::date <= $3)
        AND ($4::text    IS NULL OR alert_nivel = $4)
        AND ($5::float8  IS NULL OR alert_score >= $5)
        AND (NOT $6 OR verificado = TRUE)
    ORDER  BY scan_fecha DESC, alert_score DESC NULLS LAST
    LIMIT  $7
"""
# $1  ticker           (text | NULL)
# $2  desde            (date | NULL) -- filtra por scan_fecha
# $3  hasta            (date | NULL) -- filtra por scan_fecha
# $4  alert_nivel      (text | NULL) -- COMPRA_FUERTE|COMPRA|NEUTRO|VENTA|VENTA_FUERTE
# $5  alert_score_min  (float8 | NULL)
# $6  solo_verificados (bool)  -- TRUE = solo alertas con retornos reales cargados
# $7  limit            (int)
