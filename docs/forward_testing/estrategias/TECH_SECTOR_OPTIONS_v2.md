# TECH_SECTOR_OPTIONS_v2 — Documentacion de Estrategia

**Estado**: EN DESARROLLO
**ID en DB**: 9
**Version anterior**: [TECH_SECTOR_OPTIONS_v1.md](TECH_SECTOR_OPTIONS_v1.md)
**Script**: `scripts/forward_testing/ft_bot_tech_sectorial_options_v2.py`
**Logica base**: `tecnico_sectorial_options_v2`

---

## Concepto

Identica a TECH_SECTOR_OPTIONS_v1 en estructura, parametros y logica de gates.
La unica diferencia: el Put/Call Ratio se calcula sobre el **volumen diario**
de contratos (`PCR_VOL`) en lugar del **open interest acumulado** (`PCR_OI`).

La tesis: el open interest refleja el posicionamiento *acumulado* (lento, se
construye en dias o semanas); el volumen refleja la actividad *del dia* (rapido,
se resetea cada jornada). Si el dinero institucional empieza a rotar su
posicionamiento, el volumen lo muestra antes que el OI — el OI recien cambia
cuando esos contratos se mantienen abiertos al cierre.

**Pregunta que responde**:
?El volumen de opciones (mas reactivo) es mejor confirmador de la senal tecnica
que el open interest (mas estable)? ?El PCR_VOL anticipa giros que el PCR_OI ve
tarde, o simplemente agrega ruido?

**Diferencia respecto a TECH_SECTOR_OPTIONS_v1**:
- v1 calcula PCR usando `SUM(open_interest)` por ventana
- v2 calcula PCR usando `SUM(volumen)` por ventana
- Todo lo demas (gates, umbrales, ventanas, liquidez, salida, sizing, SL/TP)
  es IDENTICO. Esto es deliberado: aislar la variable "OI vs VOL" para que la
  comparacion v1 vs v2 sea limpia.

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
Columnas utilizadas: `vencimiento`, `tipo` (call/put), **`volumen`**, `strike`
Fecha: ultimo snapshot disponible (`MAX(fecha_snapshot)`)

### Ventanas de vencimiento (identicas a v1)

| Ventana | Rango dias | Que captura |
|---|---|---|
| Corto | 1 – 14 dias | Weeklies. Actividad tactica inmediata. |
| Medio | 15 – 45 dias | Vencimientos mensuales. |
| Largo | 46 – 90 dias | Posicionamiento de medio plazo. |

### PCR_VOL por ventana

```
PCR_VOL_ventana = SUM(volumen WHERE tipo='put') /
                  SUM(volumen WHERE tipo='call')
                  para todos los contratos en esa ventana
```

### Veredicto por ventana (identico a v1)

```
PCR_VOL_ventana < 1.0   →  Alcista (A)   mas volumen en calls
PCR_VOL_ventana >= 1.0  →  Bajista  (B)  mas volumen en puts
```

### Score de opciones (pcr_score)

```
pcr_score = cantidad de ventanas con veredicto Alcista
Rango: 0 (todas bajistas) a 3 (todas alcistas)
```

### Minimo de liquidez por ventana

```
SUM(call_VOL + put_VOL) en esa ventana >= 500 contratos de volumen
```

El umbral 500 se mantiene igual que v1. Verificado contra el ultimo snapshot:
el volumen diario es menor que el OI acumulado en las ventanas medio/largo,
pero la cobertura efectiva (tickers que pasan las 3 ventanas) queda en el
orden de los ~123 validos que produce v1 con OI. La ventana largo es el cuello
de botella (~130/195 tickers pasan vol>=500).

Si una ventana no alcanza el minimo: el ticker es DESCARTADO para esta
estrategia. Motivo en ft_candidatos_diarios: `SIN_LIQUIDEZ_OPCIONES`.

**Nota sobre el ruido del volumen**: el volumen diario es mas volatil que el
OI — un dia tranquilo puede dar lecturas bajas o cero en un contrato. Por eso
el filtro de liquidez de 500 es importante: descarta ventanas donde el PCR_VOL
seria estadisticamente poco confiable.

---

## Logica de Entrada (identica a v1)

### Gate 1 — Score tecnico

Condiciones obligatorias:
1. `tech_score >= 4.0`
2. `precio > SMA200` (implicito)
3. ticker sin posicion abierta en esta estrategia
4. ticker no cerrado en la misma corrida del dia
5. ticker sin earnings proximos
6. sector con slots y capital disponible

### Gate 2 — Confirmacion de opciones

```
pcr_score >= 2   →  PASAR
pcr_score < 2    →  no entra → motivo_skip = 'PCR_CONSENSO_INSUFICIENTE'
```

**Ambos gates deben pasar.**

### Ranking dentro del sector

```
Primario:    tech_score DESC
Secundario:  pcr_score DESC
```

### Sizing y SL/TP (identico a v1)

```
qty           = floor($2,222.22 / precio_entrada)
capital_real  = qty * precio_entrada
SL            = precio_entrada - (2.0 * ATR14)
TP            = precio_entrada + (4.0 * ATR14)
```

---

## Logica de Salida (identica a v1)

| Prioridad | Condicion | Motivo |
|---|---|---|
| P0 | Earnings manana | `EARNINGS_MANANA` |
| P1 | precio_actual <= stop_loss | `STOP_LOSS_ATR` |
| P2 | precio_actual >= take_profit | `TAKE_PROFIT_ATR` |
| P3 | tech_score <= 3.5 AND pcr_score = 0 | `SCORE_DEGRADADO_OPCIONES` |
| P3b | tech_score <= 3.5 AND pcr_score >= 1 | → **retener** (`RETENCION_OPCIONES`) |

La logica de salida se mantiene **reforzadora** (las opciones confirman, no
reemplazan al tecnico). Aunque el PCR_VOL es mas reactivo que el PCR_OI y
permitiria una salida autonoma por opciones, esa decision se posterga: no se
agrega aqui para mantener la comparacion v1 vs v2 limpia (una sola variable
cambiada). El exit autonomo por opciones queda como hipotesis para v3.

---

## Tabla comparativa v1 vs v2

| Aspecto | TECH_SECTOR_OPTIONS_v1 | TECH_SECTOR_OPTIONS_v2 |
|---|---|---|
| Metrica de opciones | PCR_OI (open interest) | PCR_VOL (volumen diario) |
| Naturaleza del dato | Acumulado, lento, estable | Diario, reactivo, ruidoso |
| Columna usada | `open_interest` | `volumen` |
| Ventanas | corto/medio/largo | Identicas |
| Umbral alcista | < 1.0 | < 1.0 |
| Liquidez minima | 500 OI/ventana | 500 VOL/ventana |
| Gates entrada | tech>=4.0 AND pcr_score>=2 | Identico |
| Logica salida | Reforzadora | Identica |
| SL/TP | 2x/4x ATR | Identico |

---

## Parametros de la estrategia — Resumen completo

| Parametro | Valor | Tipo |
|---|---|---|
| tech_score_entrada_min | 4.0 | Entrada — Gate 1 |
| pcr_score_entrada_min | 2 | Entrada — Gate 2 |
| ventana_corto_dias | 1 – 14 | Opciones |
| ventana_medio_dias | 15 – 45 | Opciones |
| ventana_largo_dias | 46 – 90 | Opciones |
| pcr_vol_umbral_alcista | 1.0 | Opciones (< umbral = Alcista) |
| min_vol_por_ventana | 500 | Liquidez minima para validar ventana |
| tech_score_salida_max | 3.5 | Salida — Gate tecnico |
| pcr_score_salida_max | 0 | Salida — Gate opciones (AND) |
| pcr_score_retencion_min | 1 | Retencion (al menos 1 ventana Alcista) |
| SL_atr_mult | 2.0 | Emergencia |
| TP_atr_mult | 4.0 | Emergencia |

---

## Metricas de validacion especificas

El objetivo central de esta estrategia es **comparar PCR_VOL contra PCR_OI**.
Corre en paralelo con v1 sobre el mismo universo y los mismos gates tecnicos.

**1. Divergencia de veredicto v1 vs v2**:
- Para los tickers validos en ambas: cuantas veces el pcr_score difiere
- Cuando difieren, ?cual veredicto acerto mas (entrada que termino ganadora)?

**2. Reactividad**:
- ?El PCR_VOL cambia de veredicto antes que el PCR_OI ante un giro de precio?
- Medir el lag promedio en dias entre cambio de veredicto VOL y cambio OI

**3. Efectividad del Gate 2 (igual que v1)**:
- Tickers que pasaron Gate 1 pero rechazados por Gate 2
- De esos, cuantos habrian sido ganadores vs perdedores

**4. Tasa de descarte por liquidez**:
- Comparar descartes SIN_LIQUIDEZ_OPCIONES de v2 (VOL) vs v1 (OI)

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
| Divergencia pcr_score vs v1 (diario prom) | — |
| Retenciones por RETENCION_OPCIONES | — |
| Descartes por SIN_LIQUIDEZ_OPCIONES | — |

---

## Hipotesis para v3 (TECH_SECTOR_OPTIONS_v3)

A definir tras observar resultados de v1 vs v2. Posibles direcciones:

- **Exit autonomo por opciones**: si pcr_score = 0/3 por N dias consecutivos,
  cerrar aunque el tecnico no se haya degradado. El PCR_VOL reactivo hace esto
  mas viable que con OI.
- **PCR hibrido**: combinar OI (estructura) + VOL (reactividad) en un solo score.
- **Ponderacion por ventana**: dar mas peso a la ventana corto (1-14d) donde el
  volumen es mas alto y representativo.
- **Combinar con TECH_SECTOR_v2**: sumar el filtro de opciones al filtro de velas.
