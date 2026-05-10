"""
Tools de exploracion generica del universo y la DB.

Primera fase que accede a PostgreSQL. El pool se inicializa en el primer
uso (lazy). Requiere DATABASE_URL en el entorno via settings.json del
cliente MCP.

Tools exportadas:
  list_tables(pattern)                         -> list[dict]
  describe_table(table_name)                   -> dict
  list_tickers(sector, only_ml_active)         -> list[dict]

Pendiente (fase 1F):
  run_select(sql, limit) -- requiere safety.py con validacion sqlglot
"""

from mcp_server.db.pool import get_pool
from mcp_server.db.queries import (
    SQL_DESCRIBE_TABLE_CHECK,
    SQL_DESCRIBE_TABLE_COLUMNS,
    SQL_DESCRIBE_TABLE_INDEXES,
    SQL_LIST_TABLES,
    SQL_LIST_TICKERS,
)

EXPLORATION_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


async def list_tables(pattern: str | None = None) -> list[dict]:
    """
    Lista las tablas accesibles en el schema public de la DB.

    Usar como primer paso para explorar la DB o antes de llamar
    describe_table(). Las tablas de sistema (pg_catalog,
    information_schema) estan excluidas.

    Args:
        pattern: Patron de filtro case-insensitive (SQL ILIKE).
                 Ejemplos: "precios%", "%opciones%", "bt_hist_%".
                 Si es None, devuelve todas las tablas.

    Returns:
        Lista de dicts ordenados alfabeticamente por tabla:
          [{"table_name": "activos",
            "row_count_estimate": 199},
           {"table_name": "alertas_scanner",
            "row_count_estimate": 1412},
           ...]

        row_count_estimate: estimacion de pg_class.reltuples (O(1),
          no hace COUNT(*)). None si la tabla nunca fue analizada
          con ANALYZE. 0 puede significar tabla vacia o estadisticas
          desactualizadas.

    Ejemplo de uso:
        list_tables()              -- todas las tablas
        list_tables("precios%")    -- tablas que empiezan con "precios"
        list_tables("%opciones%")  -- tablas que contienen "opciones"
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(SQL_LIST_TABLES, pattern)
        return [dict(row) for row in rows]
    except Exception as exc:
        return [{"error": f"No se pudo listar tablas: {exc}"}]


async def describe_table(table_name: str) -> dict:
    """
    Retorna el schema completo de una tabla: columnas, tipos e indices.

    Usar para entender la estructura de una tabla antes de consultar
    datos. Complementar con list_tables() si no se conoce el nombre exacto.

    Args:
        table_name: Nombre exacto de la tabla (case-sensitive en PostgreSQL).
                    Ejemplos: "precios_diarios", "alertas_scanner".

    Returns:
        Dict con el schema completo:
          {
            "table_name": "precios_diarios",
            "row_count_estimate": 206035,
            "columns": [
              {"name": "id",     "type": "integer",      "nullable": false,
               "default": "nextval('precios_diarios_id_seq')"},
              {"name": "fecha",  "type": "date",         "nullable": false,
               "default": null},
              {"name": "ticker", "type": "character varying(20)",
               "nullable": false, "default": null},
              {"name": "close",  "type": "double precision",
               "nullable": true,  "default": null},
              ...
            ],
            "indexes": [
              {"name": "precios_diarios_pkey",
               "definition": "CREATE UNIQUE INDEX ..."},
              ...
            ]
          }

        Si la tabla no existe, retorna:
          {"error": "Tabla 'X' no existe. Usar list_tables() para ver las disponibles."}

        column_default se retorna tal cual (puede contener expresiones
        SQL como "nextval('...')" para columnas serial/identity).

    Raises (capturado, retorna como dict con "error"):
        ConnectionError: si la DB no responde.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            # 1. Verificar que la tabla existe y obtener row count
            existence_row = await conn.fetchrow(SQL_DESCRIBE_TABLE_CHECK, table_name)
            if existence_row is None:
                return {
                    "error": (
                        f"Tabla '{table_name}' no existe en schema public. "
                        "Usar list_tables() para ver las tablas disponibles."
                    )
                }

            # 2. Columnas
            col_rows = await conn.fetch(SQL_DESCRIBE_TABLE_COLUMNS, table_name)

            # 3. Indices
            idx_rows = await conn.fetch(SQL_DESCRIBE_TABLE_INDEXES, table_name)

        return {
            "table_name": table_name,
            "row_count_estimate": dict(existence_row)["row_count_estimate"],
            "columns": [dict(row) for row in col_rows],
            "indexes": [dict(row) for row in idx_rows],
        }
    except Exception as exc:
        return {"error": f"Error al describir tabla '{table_name}': {exc}"}


async def list_tickers(
    sector: str | None = None,
    only_ml_active: bool = False,
) -> list[dict]:
    """
    Lista los tickers del universo del proyecto (199 totales).

    Usar antes de consultar datos por ticker para verificar que el ticker
    esta en el universo. Si no esta, los datos no existiran en las tablas.

    Args:
        sector: Filtro por sector (case-insensitive, match exacto).
                Ejemplos: "Technology", "Financials", "Energy".
                Si es None, devuelve todos los sectores.
        only_ml_active: Si True, filtra solo los tickers con modelo ML
                        asignado (modelo_asignado IS NOT NULL).
                        Hay 22 tickers ML activos en el universo.
                        Si False (default), devuelve los 199.

    Returns:
        Lista de dicts ordenados por ticker:
          [{"ticker": "AAPL",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "modelo_asignado": "rf_v3",
            "activo": true},
           ...]

        modelo_asignado: nombre del modelo ML asignado, o None si el
          ticker no tiene modelo activo (solo backtesting).
        activo: siempre True para los 199 tickers del universo activo.

    Ejemplos de uso:
        list_tickers()                         -- todos los 199
        list_tickers(sector="Financials")      -- solo sector Financials
        list_tickers(only_ml_active=True)      -- solo los 22 con modelo ML
        list_tickers("Technology", True)       -- tech con modelo ML
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(SQL_LIST_TICKERS, sector, only_ml_active)
        return [dict(row) for row in rows]
    except Exception as exc:
        return [{"error": f"No se pudo listar tickers: {exc}"}]
