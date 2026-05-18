"""
Tools de backtesting historico (Fase 1.5).

Tools (pendientes de implementar):
  list_strategies() -> list[dict]
      Estrategias del backtesting historico.
      Source: bt_hist_estrategias.

  get_strategy_metrics(strategy, period) -> dict
      Metricas resumen: retorno total, drawdown, win rate, profit factor.
      Source: bt_hist_metricas_diarias.

  get_strategy_trades(strategy, sort_by, top_n) -> list[dict]
      Top trades por PnL, fecha o duracion.
      Source: bt_hist_operaciones.

  get_equity_curve(strategy) -> list[dict]
      Equity curve diaria.
      Source: bt_hist_metricas_diarias.
"""
