"""
Tools de regimen macro y futuros (Fase 1.5).

Tools (pendientes de implementar):
  get_market_regime(date) -> dict
      es_mercado_alcista, es_nq_liderando, divergencia_russell,
      vol_spike_macro, dist_sma50_es.
      Source: features_regimen_macro.

  get_futures_snapshot(tickers, days_back) -> list[dict]
      OHLCV + indicadores tecnicos de futuros.
      Source: futuros_diarios JOIN indicadores_tecnicos_futuros.

  get_unusual_activity(date, sector, z_threshold) -> list[dict]
      Cross ticker_zscore_diario + opciones_zscore_diario donde ambos > threshold.

  get_ticker_zscore(ticker, days_back) -> list[dict]
      Z-scores de volumen y retorno. Source: ticker_zscore_diario.
"""
