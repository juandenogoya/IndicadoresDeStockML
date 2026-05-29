# scripts/legacy_1w/

Pipeline **semanal (1W)** historico: poblaba las tablas precios_semanales,
indicadores_tecnicos_1w, features_precio_accion_1w, features_market_structure_1w,
operaciones/resultados_bt_pa_1w (scripts 23-30, merge 31/3/2026).

**Deprecado el 28/5/2026 (Plan C).** Ya nadie lo corre (ningun cron/wrapper) y sus
tablas quedaron congeladas en 2026-04-02. Los consumidores migraron a calcular el
timeframe semanal AL VUELO desde precios_diarios:
- Dashboard: dashboard/sintesis_data.py (_semanal_bundle).
- Scanner (contexto MTF del Telegram): src/indicators/mtf_context.py (on-the-fly).

NO se borran: sirven de receta si se reactiva el 1W; git conserva el historial.

La LOGICA PURA reutilizable sigue viva en src/ (NO se movio):
- src/data/resample_weekly.py            (resample W-FRI)
- src/indicators/market_structure_1w.py  (SMC semanal; la usa mtf_context y el dashboard)
- src/indicators/technical_1w.py, precio_accion_1w.py
- src/backtesting/simulator_pa_1w.py

Pendiente (follow-up): el MCP get_ticker_sintesis aun lee indicadores_tecnicos_1w
(congelada) para el RSI/MACD semanal -> migrar a on-the-fly o marcar como stale.
