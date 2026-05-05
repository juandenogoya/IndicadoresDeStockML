# COMBO_v1 — Documentacion de Estrategia

**Estado**: ACTIVA
**ID en DB**: 5
**Inicio**: 2026-04-28
**Script**: `scripts/forward_testing/ft_bot_combo_v1.py`
**Logica base**: `combo_tech_candle`

---

## Concepto

Extension de TECH_SECTOR_v1 incorporando `candle_score_5d` como criterio de desempate
en el ranking de candidatos. Misma estructura sectorial, mismo scoring tecnico,
pero cuando dos candidatos tienen el mismo tech_score, gana el de mejor estructura de velas.

**Pregunta que responde**: ?agregar la estructura de velas de los ultimos 5 dias como
desempate mejora la seleccion de activos dentro de cada sector?
?Seleccionamos activos con mejor momentum de corto plazo?

---

## Diferencias respecto a TECH_SECTOR_v1

| Aspecto | TECH_SECTOR_v1 | COMBO_v1 |
|---|---|---|
| Ranking candidatos | `tech_score DESC` | `tech_score DESC, candle_score_5d DESC` |
| Filtro candle score | Sin filtro | `candle_score_5d >= -3.0` (excluye bajistas extremos) |
| Resto de logica | identico | identico |

---

## Parametros Globales

| Parametro | Valor | Descripcion |
|---|---|---|
| capital_total | $100,000.00 | Capital asignado a la estrategia |
| n_sectores | 9 | Sectores activos |
| capital_por_sector | $11,111.11 | capital_total / n_sectores |
| capital_por_posicion | $2,222.22 | capital_por_sector / max_pos_sector |
| max_posiciones_sector | 5 | Maximo de posiciones abiertas por sector |
| candle_score_min | -3.0 | Filtro minimo de candle score para entrar |

---

## Logica de Entrada

### Filtro de candidatos (por sector)
Condiciones que TODAS deben cumplirse:

1. `tech_score >= 4.0` (igual que TECH_SECTOR_v1)
2. `candle_score_5d >= -3.0` (filtro adicional: excluye estructuras de velas muy bajistas)
3. `precio > SMA200` (implicito en tech_score)
4. ticker NO tiene posicion abierta en esta estrategia
5. ticker NO fue cerrado en la misma corrida
6. ticker NO bloqueado por earnings
7. sector tiene slots y capital disponible

### Scoring y ranking
```
ranking = ORDER BY tech_score DESC, candle_score_5d DESC
```
El desempate por candle_score_5d actua cuando dos o mas candidatos tienen
el mismo tech_score (frecuente al ser un score discreto con pocos valores posibles).

### Sizing y SL/TP
Identicos a TECH_SECTOR_v1:
```
qty = floor($2,222.22 / precio_entrada)
SL = precio_entrada - (2.0 * ATR14)
TP = precio_entrada + (4.0 * ATR14)
```

---

## Logica de Salida

Identica a TECH_SECTOR_v1.

### Exit primario
`tech_score_actual = 0` → `SCORE_DEGRADADO_0.0`

### Exit emergencia
- `precio_actual <= stop_loss` → `SL`
- `precio_actual >= take_profit` → `TP`

### Exit condicional
- Earnings al dia siguiente → `EARNINGS_MANANA`

**Mismo problema de v1**: exit demasiado binario.
Sin diferenciacion de contexto (acumulacion vs agotamiento).

---

## Metricas al 2026-05-05

| Metrica | Valor |
|---|---|
| Dias activa | 7 |
| Capital actual | ~$100,407 |
| Retorno total | +0.41% |
| Posiciones abiertas | 38 |
| Operaciones cerradas | 11 |
| PnL no realizado | +$596.57 |
| PnL realizado | -$189.09 |
| Cash disponible | $18,412.95 |

**Distribucion sectorial al 02/05/2026**:

| Sector | Pos | Capital invertido | Slots libres |
|---|---|---|---|
| Technology | 5/5 | ~$10,522 | 0 |
| Consumer Cyclical | 5/5 | ~$10,905 | 0 |
| Financial Services | 5/5 | ~$10,348 | 0 |
| Industrials | 5/5 | ~$10,592 | 0 |
| Energy | 5/5 | ~$10,829 | 0 |
| Consumer Defensive | 5/5 | ~$10,904 | 0 |
| Healthcare | 3/5 | ~$6,509 | 2 |
| Communication Services | 3/5 | ~$6,422 | 2 |
| Basic Materials | 2/5 | ~$4,363 | 3 |

**Capital ocioso**: ~$18,413 en sectores con slots libres (Healthcare, Comm., Basic Mat.)
Razon: esos sectores no tienen 5 candidatos que cumplan el score minimo.
Esto es correcto por diseno: no se fuerzan entradas de baja calidad para llenar cupos.

---

## Observaciones de v1

1. **Mejor resultado inicial entre todas las estrategias**: +0.41% vs -2.65% de TECH_v1
2. **Capital ocioso estructural**: ~18% del capital sin desplegar en sectores "delgados"
3. **Mismo problema de exit que TECH_SECTOR_v1**: exit binario, sin logica de retencion
4. **Sin rotacion**: si un sector esta lleno con posiciones de score 4.0 y aparece uno de 5.5, no se rota

---

## Hipotesis para v2

Ver [COMBO_v2.md](COMBO_v2.md) cuando este disponible.

Lineas exploradas para la siguiente version:
- Usar `candle_score_5d` tambien como filtro de salida (no solo de entrada)
- Logica de retension: no cerrar si `candle_score_5d > 0` Y `lateral_ratio < 0.8`
- Rotacion intrasectorial: cerrar la posicion de menor score para abrir la de mayor score
- Subir el filtro de entrada de `candle_score_5d >= -3.0` a `>= 0.0` (solo momentum neutro o positivo)
