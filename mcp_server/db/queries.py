"""
SQL templates parametrizados para las tools del MCP server.

Todas las queries son SELECT-only. Los parametros usan placeholders
numerados de asyncpg ($1, $2, ...) para prevenir inyeccion SQL.

Tablas fuente principales:
  precios_diarios, features_precio_accion, features_market_structure,
  indicadores_tecnicos, opciones_resumen_diario, opciones_zscore_diario,
  alertas_scanner, activos, ticker_zscore_diario, features_regimen_macro,
  futuros_diarios, bt_hist_estrategias, bt_hist_operaciones,
  bt_hist_metricas_diarias, ft_estrategias, ft_posiciones_diarias,
  ft_operaciones, ft_metricas_diarias.
"""
