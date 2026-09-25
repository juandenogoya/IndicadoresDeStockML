# Foto al cierre por Telegram (25/9/2026)

Mensaje diario con el ESTADO de los tickers que marcaron las estrategias FT,
para decidir si vale la pena abrir el chart y revisar a mano.

- Motor puro: `src/utils/foto_ticker.py` (stdlib, tests en `tests/test_foto_ticker.py`)
- Script: `scripts/manual/telegram_foto.py` (LOCAL-only)
- Estado: funcionando a mano. **Todavia NO esta en la rutina diaria** (se agrega
  cuando se haya visto andar unos dias).

## Que NO es

No es un pronostico ni una senal de compra/venta: no lleva score, probabilidad
ni direccion esperada. Lo medido en el proyecto no respalda leerlo asi (FT sin
diferencia distinguible contra el control, ML v3 AUC 0,51, alertas de
comportamiento inusual sin direccion a 5 ruedas: dashboard/README.md). Por eso
el mensaje dice "Estado, no pronostico" y no usa la palabra "oportunidad".

## Que tickers

Los candidatos de la ULTIMA corrida de las estrategias FT activas
(`ft_candidatos_diarios` JOIN `ft_estrategias` WHERE activa):

1. **En 2+ familias** (hasta 6): el ticker aparece en estrategias de familias
   distintas. Se cuenta por FAMILIA y no por instancia:

   | Familia | `logica` | Instancias activas al 25/9 |
   |---|---|---|
   | ML | `ml_scanner` | ML_SCANNER v1, v2 |
   | TECH | `tecnico` | TECH_v1 |
   | TECH_SECTOR | `tecnico_sectorial*` | TECH_SECTOR v1, v2, OPTIONS v1, v2, OIEXIT |
   | SMC | `smc*` | SMC_v1, SMC_v3_N5, SMC_v3_N3 |

   Las 5 de TECH_SECTOR comparten la regla de entrada: un ticker en las 5 es
   UNA coincidencia, no cinco.
2. **Top por familia** (default 3, `--top N`): mayor score dentro de la
   familia (el maximo entre sus instancias). El score solo se compara dentro de
   una familia; entre familias las escalas no tienen nada que ver.

Cada ticker lleva una sola foto aunque este en las dos listas. Tope duro: 15.

Nota: las estrategias de seguimiento de tendencia exigen precio > SMA200 y RSI
45-68, asi que sus candidatos rara vez van a estar en sobreventa.

## Que muestra cada ticker

| Dato | Fuente | Regla |
|---|---|---|
| Cierre y var % | `precios_diarios` | contra la rueda anterior |
| RSI 14 | `indicadores_tecnicos` | <35 Sobreventa / 35-65 Neutro / >65 Sobrecompra |
| MACD | `indicadores_tecnicos` | linea > senal = Compra, < = Venta (sin neutro) |
| Volumen | `precios_diarios` | volumen del dia / MEDIANA de las 252 ruedas ANTERIORES (sin la propia) |
| SMA50 / SMA200 | `indicadores_tecnicos` | distancia % y sobre/bajo |
| 5 ruedas | `precios_diarios` | cierre de hoy contra el de 5 ruedas atras |
| Earnings | `earnings_calendar` | ruedas habiles NYSE hasta la proxima fecha |
| Tabla | las anteriores | las 5 ruedas previas + la de datos, con los cambios de estado |

Decisiones:

- **Umbrales de RSI y MACD = los de `clasificacion_tecnica.py`** (los usan el
  dashboard y el MCP). Si se cambian ahi, cambian en todos lados a la vez; no
  hay una definicion propia del mensaje.
- **MACD sin neutro**: con la regla del proyecto el neutro exige linea = senal
  exacto, casi nunca pasa. En su lugar se marca el CRUCE en la tabla (`MACD->C`).
- **Mediana y no promedio** en el volumen: los dias de balance inflan el
  promedio. Con menos de 126 ruedas previas no se informa.
- **Earnings pasados = "sin fecha proxima"**: `earnings_calendar` guarda solo la
  PROXIMA fecha y se refresca semanal; una fecha anterior a la rueda es dato
  viejo, no un balance proximo.
- Cambios de estado marcados en la tabla: `MACD->C|V`, `RSI->SV|N|SC`,
  `SMA50->sobre|bajo`, `SMA200->sobre|bajo`.

## Fechas

`ft_candidatos_diarios.fecha` es la fecha de CORRIDA del bot (`date.today()`),
no la de los datos. La foto se arma sobre la ultima rueda de `precios_diarios`
de cada ticker, y el encabezado informa esa rueda. Si un ticker quedo en una
rueda distinta que el resto, su bloque lo avisa (`OJO: ultimo dato del ...`).

## Uso

```
python scripts/manual/telegram_foto.py --dry-run            imprime, no envia
python scripts/manual/telegram_foto.py                      envia
python scripts/manual/telegram_foto.py --tickers NVDA,AMD   foto a demanda
python scripts/manual/telegram_foto.py --top 2 --fecha-candidatos 2026-09-24
```

Salida: 0 enviado / 2 nada para mandar / 1 error de DB o de envio.

Formato: HTML de Telegram. Los bloques se empaquetan en mensajes de hasta 4000
caracteres sin partir ninguno (partir un `<pre>` deja HTML invalido y Telegram
rechaza el mensaje entero), por eso NO se usa `_send_long`.

## Pendiente

- Incorporarlo a `rutina_diaria` como paso final despues de `ft` (solo
  informa, nunca frena), una vez validado a mano.
