"""
Tools de exploracion generica del universo y la DB.

Tools:
  list_tickers(sector, only_ml_active) -> list[dict]
      Devuelve tickers del universo (199 totales). Filtros opcionales.
      Source: tabla activos.

  list_tables(pattern) -> list[str]
      Devuelve tablas accesibles. Filtra por patron LIKE si se provee.

  describe_table(table_name) -> dict
      Schema de la tabla: columnas, tipos, indices.

  run_select(sql, limit) -> dict
      Ejecuta SELECT validado via safety.py. Inyecta LIMIT automaticamente.
"""
