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
