# MCP Server db-consultor -- Instrucciones de uso
# Version: 0.1.0 | Ultima actualizacion: 2026-05-10

Este archivo es la fuente unica de las reglas de uso del servidor MCP
que expone la base de datos del proyecto IndicadoresDeStockML.
Cada cliente (Gemini CLI, Claude Desktop, Claude Code, Cursor) lo consume
via symlink o copia. No referenciar rutas externas a este repo.

---

## Que es este server

Servidor MCP consultivo que expone la base de datos activos_ml como
herramientas consumibles por clientes LLM. Permite hacer preguntas en
lenguaje natural sobre el sistema (precios, indicadores, opciones,
alertas, backtests) y recibir respuestas con datos extraidos directamente
de las tablas existentes.

Universo: 199 tickers (22 activos en ML + 177 solo backtesting).
DB local: PostgreSQL, sincronizada con Railway via sync_local.bat.

---

## Principio rector: SOLO CONSULTIVO

Este server NO modifica datos, scripts, parametros ni infraestructura.
No llames tools que no sean de lectura. Si el usuario pide modificar algo,
aclarar que el server es exclusivamente de consulta.

---

## Reglas criticas (no negociables)

### REGLA 1 -- FECHAS (la mas importante)

NUNCA asumir si una fecha es habil bursatil.
SIEMPRE llamar `check_trading_day` o `get_last_trading_day` antes de
razonar sobre ventanas temporales, "el ultimo dia", "ayer", etc.

Ejemplos de errores que esta regla previene:
- Asumir que lunes = dia habil (puede ser feriado NYSE).
- Asumir que "ayer" tiene datos (puede ser fin de semana o feriado).
- Calcular "ultimos 5 dias" sin verificar el punto de partida.

### REGLA 2 -- Universo cerrado

El universo es de 199 tickers conocidos. Si el usuario pregunta por un
ticker fuera del universo, responder que no esta en el universo en lugar
de intentar buscar datos o inventarlos.
Usar `list_tickers()` para verificar si un ticker esta en el universo.

### REGLA 3 -- Distincion de scores (dos cosas distintas)

Existen DOS scores diferentes en el sistema:

| Score | Tabla/columna | Tipo | Estado en DB local |
|---|---|---|---|
| Rule-based | scoring_tecnico.score | RSI/MACD/SMA200/SMA50/Momentum/SMA21 | VACIA (37k filas solo en Railway) |
| ML (Random Forest V3) | alertas_scanner.alert_score | Random Forest V3 | POBLADA |

Si el usuario pide "score" sin aclarar, usar el ML (alertas_scanner) y
mencionar la distincion. Si piden el rule-based y la tabla esta vacia,
advertir y sugerir regeneracion o consulta a Railway.

### REGLA 4 -- Columnas con nombres no obvios

Estas columnas tienen nombres que NO son los que uno esperaria:

| Tabla | Columna de fecha correcta | Nombre incorrecto (no existe) |
|---|---|---|
| precios_semanales | fecha_semana | "fecha" |
| alertas_scanner | scan_fecha | "fecha_alerta" |

Usar SIEMPRE los nombres correctos. La columna "senal" no existe en
alertas_scanner; usar alert_nivel.

### REGLA 5 -- OI = T-1 en yfinance

Cuando se reporte open_interest de opciones, agregar siempre la nota:
"OI corresponde al cierre del dia anterior (yfinance reporta OI con 1
dia de retraso)."

### REGLA 6 -- IV con cobertura baja

Si la cobertura de IV es menor al 30%, marcar la respuesta con:
"IV cov: X%, baja cobertura -- tomar con reserva."

### REGLA 7 -- Gaps de datos conocidos en opciones

Las fechas 2026-04-23 y 2026-04-25 no tienen datos de opciones
recuperables. Si caen en la ventana consultada, marcarlo explicitamente:
"[SIN DATA - gap conocido]". No inventar ni omitir en silencio.

### REGLA 8 -- scoring_tecnico vacia en local

Si se llama la tool get_rule_based_score con DSN local, advertir:
"La tabla scoring_tecnico esta vacia en DB local. Regenerar con el
script correspondiente o consultar Railway."

---

## Reglas de granularidad y formato de respuesta

- Por defecto: tabla markdown densa + parrafo descriptivo breve.
- NO interpretar ni dar recomendaciones de inversion. Describir que
  muestran los datos. Las conclusiones las saca el usuario.
- Si el usuario pide profundizar ("explicame mas", "por que", "detallame"),
  ampliar usando los thresholds y convenciones documentadas del proyecto.
- Cuando una senal no este disponible (gap, cobertura baja, tabla vacia),
  marcarlo explicitamente. No inventar ni omitir en silencio.

---

## Reglas de composicion de sentimiento

Cuando el usuario pide "sentimiento" sin aclarar cual, devolver TODAS las
senales disponibles para la ventana consultada. Dejar que el usuario
interprete; no sintetizar en un unico juicio.

| Senal | Tabla / columna | Interpretacion |
|---|---|---|
| Estructura SMC corto plazo | features_market_structure.estructura_5 | -1 bajista / 0 neutral / +1 alcista |
| Estructura SMC medio plazo | features_market_structure.estructura_10 | idem |
| Tendencia semanal | features_market_structure_1w.estructura_10 | idem -- OJO: tabla CONGELADA (pipeline 1w deprecado 28/5/2026). get_ticker_sintesis marca el semanal como desactualizado si la fecha es vieja. |
| Score ML | alertas_scanner.alert_score | 0-100 |
| Nivel ML | alertas_scanner.alert_nivel | COMPRA_FUERTE / COMPRA / NEUTRO / VENTA / VENTA_FUERTE |
| Sesgo PCR volumen | opciones_resumen_diario.pcr_vol | <0.7 ALCISTA / 0.7-1.0 NEUTRO / >1.0 BAJISTA |
| Z-score PCR | opciones_zscore_diario.pcr_vol_zscore | abs > 2 = inusual |
| Patrones de vela | features_precio_accion.patron_* | bool segun patron |
| Regimen macro | features_regimen_macro.es_mercado_alcista | 1 alcista / 0 bajista (ES vs SMA50w) |

Thresholds de alert_nivel (ML):
  COMPRA_FUERTE: alert_score >= 70
  COMPRA:        alert_score >= 60
  NEUTRO:        alert_score 50-59
  VENTA:         alert_score < 50
  VENTA_FUERTE:  alert_score < 30

Thresholds PCR volumen:
  ALCISTA: pcr_vol < 0.7
  NEUTRO:  pcr_vol 0.7-1.0
  BAJISTA: pcr_vol > 1.0

---

## Tablas principales y sus particularidades

| Tabla | Descripcion | Notas |
|---|---|---|
| precios_diarios | OHLCV diario de 199 tickers | 2 anos de historia |
| precios_semanales | OHLCV semanal | col fecha = fecha_semana |
| features_precio_accion | 32 features de anatomia/patrones/rolling/volumen | |
| features_market_structure | Features SMC (BOS, CHoCH, swings) | ventana 5 y 10 |
| indicadores_tecnicos | 18 indicadores tecnicos | SMA/RSI/MACD/ATR/BB/OBV/ADX |
| alertas_scanner | Alertas ML diarias | col fecha = scan_fecha |
| opciones_resumen_diario | PCR, IV, OI resumen por ticker/fecha | OI = T-1 |
| opciones_zscore_diario | Z-scores opciones vs ventana 60d | desde 18/4/2026 |
| ticker_zscore_diario | Z-scores volumen/retorno acciones | desde 2021-04-12 |
| activos | 199 tickers, sector, industry, modelo_asignado | |
| features_regimen_macro | Regimen macro diario | futuros indices |
| bt_hist_estrategias | Estrategias backtesting historico | |
| ft_estrategias | Estrategias forward testing | |

---

## Convenciones de naming de las tools

- snake_case en ingles.
- Verbo de accion al inicio: get_*, list_*, describe_*, check_*, save_*.
- Outputs estructurados (dicts/lists), no strings libres.

---

## Limitaciones conocidas

- DB local puede estar desactualizada si no se corrio sync_local.bat.
  El pipeline Oracle actualiza Railway cada dia habil (L-V 22:00 UTC).
  Correr sync_local.bat despues de cada pipeline para refrescar local.
- scoring_tecnico vacia en local (ver Regla 8).
- opciones: gaps 2026-04-23 y 2026-04-25 (ver Regla 7).
- El server NO tiene acceso a datos en tiempo real ni a APIs externas.
  Solo lee de la DB local sincronizada.
