"""
Tools de alertas del scanner ML.

Convencion de columnas (critica):
  - Fecha: scan_fecha (NO "fecha_alerta", ese nombre no existe).
  - Nivel: alert_nivel (NO "senal"; valores: COMPRA_FUERTE, COMPRA, NEUTRO,
    VENTA, VENTA_FUERTE).
  - Score ML: alert_score (0-100). Distincion vs scoring_tecnico.score (rule-based).

Tools:
  get_ml_alert_history(ticker, days_back) -> list[dict]
      Historico de alertas: alert_nivel, alert_score, alert_detalle.
      Source: alertas_scanner. Columna fecha: scan_fecha.
"""
