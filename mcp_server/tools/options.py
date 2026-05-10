"""
Tools de opciones.

Convencion de datos (heredada del proyecto):
  - OI = T-1 en yfinance. Siempre notificarlo en la respuesta.
  - IV con cobertura < 30%: marcar con advertencia.
  - Gaps conocidos sin data recuperable: 2026-04-23, 2026-04-25.
  - Labels PCR: < 0.7 ALCISTA | 0.7-1.0 NEUTRO | > 1.0 BAJISTA.

Tools:
  get_options_summary(ticker, days_back) -> list[dict]
      PCR vol, PCR OI, IV calls/puts, n contratos, max OI strike, precio
      subyacente, label PCR calculada.
      Source: opciones_resumen_diario.

  get_options_zscore(ticker, days_back) -> list[dict]
      Z-scores de vol calls/puts/total, PCR, IV vs ventana 60d.
      Source: opciones_zscore_diario.
"""
