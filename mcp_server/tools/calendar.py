"""
Tools de calendario bursatil NYSE.

Importa funciones puras de src.utils.trading_calendar (sin side effects).

Tools:
  check_trading_day(date) -> dict
      Verifica si una fecha es habil NYSE.
      Retorna: {"date", "is_trading_day", "reason", "next_trading_day"}

  get_last_trading_day() -> dict
      Retorna el ultimo dia habil hasta hoy.
      Retorna: {"date", "weekday", "days_ago"}
"""
