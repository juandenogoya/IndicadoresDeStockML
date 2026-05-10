"""
Configuracion del MCP server via pydantic-settings.
Lee variables de entorno y .env existente del proyecto.

Variables esperadas:
  DATABASE_URL          DSN del rol mcp_reader (postgres://mcp_reader:...@host/db)
  QUERIES_CATALOG_PATH  Ruta al repo ~/queries-catalog/
  LOG_LEVEL             DEBUG | INFO | WARNING (default: INFO)
  MCP_SERVER_NAME       Nombre del server (default: db-consultor)
  MCP_SERVER_VERSION    Version del server (default: 0.1.0)
"""
