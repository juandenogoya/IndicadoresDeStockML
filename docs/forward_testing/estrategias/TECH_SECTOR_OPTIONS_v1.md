# TECH_SECTOR_OPTIONS_v1 — Documentacion de Estrategia

**Estado**: EN DESARROLLO
**ID en DB**: 8
**Version anterior**: [TECH_SECTOR_v1.md](TECH_SECTOR_v1.md)
**Script**: `scripts/forward_testing/ft_bot_tech_sectorial_options_v1.py`
**Logica base**: `tecnico_sectorial_options_v1`

---

## Concepto

Combina la logica de TECH_SECTOR_v1 (score tecnico + particion sectorial) con
una capa de confirmacion basada en el posicionamiento real del mercado de opciones
(PCR_OI por ventana de vencimiento).

La tesis de fondo: los operadores institucionales posicionan capital en opciones
antes de mover el subyacente. Si el score tecnico dice "comprar" y el mercado de
opciones tambien esta posicionado alcista en los contratos vivos, tenemos dos
fuentes independientes de informacion alineadas. Esa convergencia deberia reducir
entradas en falsos breakouts.

**Pregunta que responde**:
?El posicionamiento en opciones (PCR_OI por ventana de vencimiento) agrega valor
predictivo sobre la senal tecnica sola? ?Filtrar entradas cuando el mercado de
opciones no confirma reduce perdedores sin eliminar ganadores?

**Diferencia respecto a TECH_SECTOR_v1**:
- ENTRADA: agrega un segundo gate obligatorio basado en PCR_OI de opciones
- SALIDA: la retencion cuando el score se degrada ahora depende de si las opciones
  confirman o contradicen la degradacion tecnica
- Todo lo demas (capital, sectores, sizing, SL/TP) es identico a v1

**Diferencia respecto a TECH_SECTOR_v2**:
- v2 usa candle_score_5d y up_vol_5d (datos de precio/volumen del subyacente)
- esta estrategia usa PCR_OI de opciones (mercado derivado, informacion independiente)
- Son complementarias, no redundantes

---

## Parametros Globales (sin cambios respecto a v1)

| Parametro | Valor | Descripcion |
|---|---|---|
| capital_total | $100,000.00 | Capital asignado |
| n_sectores | 9 | Sectores activos |
| capital_por_sector | $11,111.11 | capital_total / n_sectores |
| capital_por_posicion | $2,222.22 | capital_por_sector / max_pos_sector |
| max_posiciones_sector | 5 | Maximo de posiciones abiertas por sector |
| SL_atr_mult | 2.0 | Stop loss: 2x ATR14 desde entrada |
| TP_atr_mult | 4.0 | Take profit: 4x ATR14 desde entrada |

---

## Fuente de datos de opciones

Tabla: `opciones_snapshot`
Columnas utilizadas: `vencimiento`, `tipo` (call/put), `open_interest`, `strike`
Fecha: ultimo snapshot disponible (`MAX(fecha_snapshot)`)
Cobertura actual: 199 tickers, 11 vencimientos, rango 3 a 63 dias al vencimiento.

### Ventanas de vencimiento

Los contratos vivos se agrupan en tres ventanas segun dias al vencimiento
calculados como `(vencimiento - fecha_snapshot)`:

| Ventana | Rango dias | Que captura |
|---|---|---|
| Corto | 1 – 14 dias | Weeklies. Posicionamiento tactico inmediato. |
| Medio | 15 – 45 dias | Vencimientos mensuales. Mayor OI, mas confiable. |
| Largo | 46 – 90 dias | Posicionamiento institucional de medio plazo. |

### PCR_OI por ventana

```
PCR_OI_ventana = SUM(open_interest WHERE tipo='put') /
                 SUM(open_interest WHERE tipo='call')
                 para todos los contratos en esa ventana
```

### Veredicto por ventana

```
PCR_OI_ventana < 1.0   →  Alcista (A)   mas OI en calls -> mercado apuesta suba
PCR_OI_ventana >= 1.0  →  Bajista  (B)  mas OI en puts  -> mercado apuesta baja
```

### Score de opciones (pcr_score)

```
pcr_score = cantidad de ventanas con veredicto Alcista
Rango: 0 (todas bajistas) a 3 (todas alcistas)
```

### Minimo de liquidez por ventana

Para que el veredicto de una ventana sea valido, debe cumplir:
```
SUM(call_OI + put_OI) en esa ventana >= 500 contratos de OI
```

Si una ventana no alcanza este minimo o no tiene contratos en ese rango:
el ticker es DESCARTADO para esta estrategia (no se evalua con ventanas
parciales). Motivo registrado en ft_candidatos_diarios: `SIN_LIQUIDEZ_OPCIONES`.

**Fundamento**: un PCR_OI calculado sobre OI muy bajo (ej: 30 contratos) es
estadisticamente poco confiable. Es mejor no operar que operar con senal ruidosa.

---

## Logica de Entrada

### Gate 1 — Score tecnico (identico a v1)

Condiciones obligatorias:
1. `tech_score >= 4.0`
2. `precio > SMA200` (implicito: si no, tech_score = 0)
3. ticker sin posicion abierta en esta estrategia
4. ticker no cerrado en la misma corrida del dia
5. ticker sin earnings proximos
6. sector con slots y capital disponible

### Gate 2 — Confirmacion de opciones (nuevo)

```
pcr_score >= 2   →  al menos 2 de 3 ventanas Alcistas → PASAR
pcr_score < 2    →  no entra → motivo_skip = 'PCR_CONSENSO_INSUFICIENTE'
```

Ejemplos de combinaciones:

| Corto | Medio | Largo | pcr_score | Decision |
|---|---|---|---|---|
| A | A | A | 3 | ENTRA — confluencia total |
| A | A | B | 2 | ENTRA — mayoria alcista |
| A | B | A | 2 | ENTRA — mayoria alcista |
| B | A | A | 2 | ENTRA — mayoria alcista |
| A | B | B | 1 | NO ENTRA — mayoria bajista |
| B | A | B | 1 | NO ENTRA — mayoria bajista |
| B | B | A | 1 | NO ENTRA — mayoria bajista |
| B | B | B | 0 | NO ENTRA — confluencia bajista |

**Ambos gates deben pasar.** Un tech_score de 5.5 con pcr_score = 1 no entra.
Un pcr_score = 3 con tech_score = 3.5 tampoco entra.

### Ranking dentro del sector

```
Primario:    tech_score DESC        (los indicadores mandan)
Secundario:  pcr_score DESC         (tiebreaker: mas ventanas alcistas gana)
```

Si hay empate en ambos (raro): se abre en orden alfabetico de ticker.

### Sizing y SL/TP (identico a v1)

```
qty           = floor($2,222.22 / precio_entrada)
capital_real  = qty * precio_entrada
SL            = precio_entrada - (2.0 * ATR14)
TP            = precio_entrada + (4.0 * ATR14)
```

---

## Logica de Salida

### Prioridades

| Prioridad | Condicion | Motivo | Cambio vs v1 |
|---|---|---|---|
| P0 | Earnings manana | `EARNINGS_MANANA` | Sin cambio |
| P1 | precio_actual <= stop_loss | `STOP_LOSS_ATR` | Sin cambio |
| P2 | precio_actual >= take_profit | `TAKE_PROFIT_ATR` | Sin cambio |
| P3 | tech_score <= 3.5 AND pcr_score = 0 | `SCORE_DEGRADADO_OPCIONES` | Nuevo — AND |
| P3b | tech_score <= 3.5 AND pcr_score >= 1 | → **retener** | Nuevo — retencion |

### Detalle P3 y P3b: degradacion con confirmacion de opciones

**En v1**: cerrar si `tech_score = 0` (criterio unico)

**En esta estrategia**:

```
tech_score <= 3.5  AND  pcr_score = 0  →  CERRAR
  El tecnico se degrado Y las tres ventanas de opciones son bajistas.
  Doble confirmacion: el subyacente y los derivados dicen lo mismo.
  Motivo: SCORE_DEGRADADO_OPCIONES

tech_score <= 3.5  AND  pcr_score >= 1  →  RETENER
  El tecnico se degrado PERO al menos una ventana de opciones sigue alcista.
  El mercado de derivados no confirma la degradacion tecnica.
  Puede ser un pullback temporal antes de continuar la tendencia.
  Motivo de log: RETENCION_OPCIONES (registrar en ft_posiciones_diarias)
```

**Interpretacion de la retencion**:
El mercado de opciones refleja el posicionamiento real de capital, no solo el
precio del dia. Si alguna ventana sigue con mas call OI que put OI mientras el
precio temporalmente cruza la SMA50/SMA21, el dinero institucional no esta
huyendo todavia. Salimos cuando ambos mercados (subyacente + derivados) coinciden.

**Por que NO usamos 0/3 como exit autonomo**:
El PCR_OI es un indicador lento — el OI se acumula durante dias y semanas. Usar
3/3 Bajista para cerrar independientemente del score tecnico puede generar salidas
prematuras en posiciones que recien entran o que estan en fase de consolidacion
antes de moverse. Esta decision se revisara en v2 con datos reales de cuantas
retenciones terminaron en ganancia vs perdida.

---

## Tabla comparativa v1 vs TECH_SECTOR_OPTIONS_v1

| Aspecto | TECH_SECTOR_v1 | TECH_SECTOR_OPTIONS_v1 | Fuente de datos |
|---|---|---|---|
| Gate de entrada | tech_score >= 4.0 | tech_score >= 4.0 AND pcr_score >= 2 | tech + opciones_snapshot |
| Ranking en sector | tech_score DESC | tech_score DESC, pcr_score DESC | tech + opciones |
| Cierre por score | score = 0 | score <= 3.5 AND pcr_score = 0 | tech + opciones |
| Retencion | No existe | score <= 3.5 AND pcr_score >= 1 | opciones |
| SL/TP | 2x/4x ATR | 2x/4x ATR | Sin cambio |
| Rotacion | No existe | No existe | — |

---

## Parametros de la estrategia — Resumen completo

| Parametro | Valor | Tipo |
|---|---|---|
| tech_score_entrada_min | 4.0 | Entrada — Gate 1 |
| pcr_score_entrada_min | 2 | Entrada — Gate 2 |
| ventana_corto_dias | 1 – 14 | Opciones |
| ventana_medio_dias | 15 – 45 | Opciones |
| ventana_largo_dias | 46 – 90 | Opciones |
| pcr_oi_umbral_alcista | 1.0 | Opciones (< umbral = Alcista) |
| min_oi_por_ventana | 500 | Liquidez minima para validar ventana |
| tech_score_salida_max | 3.5 | Salida — Gate tecnico |
| pcr_score_salida_max | 0 | Salida — Gate opciones (AND) |
| pcr_score_retencion_min | 1 | Retencion (al menos 1 ventana Alcista) |
| SL_atr_mult | 2.0 | Emergencia |
| TP_atr_mult | 4.0 | Emergencia |

---

## Metricas de validacion especificas para esta estrategia

Ademas de las metricas estandar (retorno, drawdown, win rate):

**1. Efectividad del Gate 2 (filtro de opciones en entrada)**:
- Cuantos tickers pasaron Gate 1 pero fueron rechazados por Gate 2 (pcr_score < 2)
- De esos rechazados, cuantos habrian sido ganadores vs perdedores N dias despues
- Esto valida si el filtro discrimina correctamente o descarta candidatos buenos

**2. Efectividad de la retencion RETENCION_OPCIONES**:
- Cuantas veces se retuvo por pcr_score >= 1 cuando tech_score <= 3.5
- De esas retenciones, cuantas terminaron en ganancia vs perdida al cierre final
- Valida si mantener cuando las opciones discrepan del tecnico agrega valor

**3. Tasa de descarte por liquidez**:
- Cuantos tickers se descartan diariamente por SIN_LIQUIDEZ_OPCIONES
- Si es muy alta (> 30% del universo) indica que el filtro de liquidez es demasiado
  restrictivo o que muchos tickers tienen opciones iliquidas

---

## Metricas (pendiente — estrategia en desarrollo)

| Metrica | Valor |
|---|---|
| Fecha inicio | pendiente |
| Retorno total | — |
| Max drawdown | — |
| Operaciones totales | — |
| Win rate | — |
| Avg dias abierta | — |
| Tickers filtrados por Gate 2 (diario prom) | — |
| Retenciones por RETENCION_OPCIONES | — |
| Descartes por SIN_LIQUIDEZ_OPCIONES | — |

---

## Hipotesis para v2 (TECH_SECTOR_OPTIONS_v2)

A definir tras observar resultados de v1. Posibles direcciones:

- **Reemplazar PCR_OI con PCR_VOL**: el volumen diario refleja la actividad del dia,
  puede ser mas reactivo (v2 usara PCR_VOL en lugar de OI)
- **Exit autonomo por opciones**: si pcr_score = 0/3 por N dias consecutivos, cerrar
  aunque el tecnico no se haya degradado (requiere datos de v1 para calibrar N)
- **Ponderacion por ventana**: dar mas peso a la ventana medio (15-45d) donde
  concentra el mayor OI y es la mas informativa
- **Subir pcr_score_entrada_min a 3**: requerir confluencia total (3/3 Alcista) si
  los datos muestran que 2/3 genera demasiados falsos positivos
- **Combinar con TECH_SECTOR_v2**: sumar el filtro de opciones al filtro de velas
  (candle_score_5d) — tres capas de confirmacion independientes
