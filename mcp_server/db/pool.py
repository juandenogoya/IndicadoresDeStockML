"""
Pool de conexiones asyncpg -- lifecycle del MCP server.

Crea un pool global al arrancar el server (lifespan) y lo cierra al detenerlo.
Expone get_pool() para que las tools obtengan conexiones.

Funciones publicas:
  init_pool(dsn: str) -> asyncpg.Pool
  close_pool() -> None
  get_pool() -> asyncpg.Pool
"""
