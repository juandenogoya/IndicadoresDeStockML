"""
Tools de precios y features de precio-accion.

Tools:
  get_price_history(ticker, days_back) -> list[dict]
      OHLCV diario de los ultimos N dias habiles.
      Source: precios_diarios.

  get_price_action(ticker, days_back, group) -> list[dict]
      32 features de precio-accion. group filtra por categoria:
      "anatomia" (9) | "patrones" (8) | "rolling" (8) | "volumen" (7) | None (todas).
      Source: features_precio_accion.
"""
