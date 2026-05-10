"""
Tool de vista cruzada (orquestadora).

Tools:
  get_ticker_overview(ticker, days_back) -> dict
      Orquesta: get_price_history + get_price_action (grupo "patrones") +
      get_market_structure + get_options_summary + get_ml_alert_history.
      Devuelve dict con shape unificado por fecha.
      Es la tool de facto para preguntas tipo "como esta BAC".
"""
