"""
Acceso a la API de Polygon.io.

MISMA REGLA QUE src/data/sec/: nada de este paquete importa del lado de
trading (src/trading, src/pipeline, scripts/). Solo stdlib + requests. El
estado que necesita para decidir que pedir se lo pasa el llamador; estos
modulos no consultan la base.

Para que se agrego Polygon, y para que NO:

  SIRVE  splits    -- historial exacto y autoritativo. Verificado contra 8
                      splits reales, incluidos KLAC 10:1 y CRWD 4:1 del
                      incidente de julio de 2026. Es el ingrediente que
                      faltaba para rebasar las series de acciones que hoy
                      nuestras guardas rechazan por no poder distinguir un
                      split de un error de dato.
  SIRVE  acciones  -- `weighted_shares_outstanding` es el TOTAL de todas las
                      clases, que es justo lo que companyfacts descarta en
                      los filers multi-clase. Medido contra nuestros puntos
                      confiables: MSFT 0,9955..1,0024 y HSY 0,9995.
  NO SIRVE  deuda  -- el balance de `financials` trae 12 conceptos de alto
                      nivel, sin apertura de deuda corta/larga.
"""
