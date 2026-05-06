# TECH_SECTOR_v2 — Documentacion de Estrategia

**Estado**: EN DESARROLLO
**ID en DB**: 6
**Version anterior**: [TECH_SECTOR_v1.md](TECH_SECTOR_v1.md)
**Script**: `scripts/forward_testing/ft_bot_tech_sectorial_v2.py`
**Logica base**: `tecnico_sectorial_v2`

---

## Concepto

Extension de TECH_SECTOR_v1 con dos cambios en la logica de SALIDA:

1. **Retencion condicional por señal de mercado**: no cierra una posicion con
   score = 0 si el mercado muestra señales de acumulacion activa (candle_score
   positivo o volumen comprador presente). Elimina el exit binario de v1.

2. **Rotacion intrasectorial**: si el sector esta lleno y aparece un candidato
   con score significativamente mayor al peor de los abiertos, rota capital.

La entrada es identica a v1. Cambiamos UNA cosa a la vez para poder atribuir
correctamente cualquier diferencia de resultado.

**Pregunta que responde**:
?Retener posiciones en acumulacion y rotar capital intrasectorialmente
mejora el retorno respecto a la version con exit binario?

**Hipotesis principal**:
El exit binario de v1 genera cierres innecesarios de posiciones que estan
consolidando antes de moverse. Retenerlas mientras el mercado lo justifique
deberia reducir el giro de capital y mejorar el retorno ajustado.

---

## Parametros Globales (sin cambios respecto a v1)

| Parametro | Valor | Descripcion |
|---|---|---|
| capital_total | $100,000.00 | Capital asignado |
| n_sectores | 9 | Sectores activos |
| capital_por_sector | $11,111.11 | capital_total / n_sectores |
| capital_por_posicion | $2,222.22 | capital_por_sector / max_pos_sector |
| max_posiciones_sector | 5 | Maximo de posiciones abiertas por sector |

---

## Logica de Entrada (identica a v1)

### Condiciones obligatorias
1. `tech_score >= 4.0`
2. `precio > SMA200` (implicito: si no, tech_score = 0)
3. ticker sin posicion abierta en esta estrategia
4. ticker no cerrado en la misma corrida del dia
5. ticker sin earnings proximos
6. sector con slots y capital disponible

### Ranking dentro del sector
```
ORDER BY tech_score DESC
```

### Sizing y SL/TP
```
qty  = floor($2,222.22 / precio_entrada)
SL   = precio_entrada - (2.0 * ATR14)
TP   = precio_entrada + (4.0 * ATR14)
```

---

## Logica de Salida — CAMBIOS RESPECTO A v1

### Exit condicional: earnings (sin cambio, prioridad maxima)
Motivo: `EARNINGS_MANANA`

### Exit emergencia: SL / TP hit (sin cambio)
Motivo: `SL` / `TP`

### Exit primario rediseñado: señal de mercado combinada

**En v1**: cerrar si `tech_score = 0` (unico criterio)

**En v2**: cerrar si se cumplen TODAS las siguientes condiciones simultaneamente:
```
tech_score      = 0      → score nulo (precio bajo SMA200)
candle_score_5d < 0      → momentum de velas negativo en los ultimos 5 dias
up_vol_5d       < 2      → menos de 2 dias alcistas con volumen en ultimos 5
```
Motivo registrado: `SCORE_DEGRADADO_SIN_MOMENTUM`

**Retener posicion** (NO cerrar) si score = 0 pero AL MENOS UNA se cumple:
```
candle_score_5d >= 0     → estructura de velas neutra o positiva
up_vol_5d       >= 2     → presencia compradora con volumen confirmado
```
Logica OR: basta con que UNA condicion sea verdadera para retener.

**Interpretacion**:
El precio bajo SMA200 puede ser un pullback transitorio. Si las velas de los
ultimos 5 dias no deterioraron y/o el volumen comprador sigue presente, el
mercado no esta diciendo "salir". Cerramos solo cuando el mercado lo dice en
forma convergente: score nulo + velas bajistas + volumen comprador ausente.

---

### Exit nuevo: rotacion intrasectorial

Condicion de evaluacion (evaluar cada vez que hay un nuevo candidato con sector lleno):
```
sector con 5/5 posiciones
Y score_candidato_nuevo >= (tech_score_actual_peor_posicion + 1.0)
```

Accion:
```
→ cerrar la posicion con menor tech_score_actual en el sector
→ abrir el candidato nuevo
```
Motivo cierre: `ROTACION_SECTORIAL`

**Parametro delta = 1.0**: evita rotaciones por diferencias marginales.
Es un parametro de diseño relativo (no arbitrario absoluto) — requiere una
diferencia de al menos 1 punto en el score discreto (0-5.5) para justificar
el movimiento de capital. Revisable en v3 segun resultados.

**Orden de evaluacion**: siempre despues de los exits de emergencia y primario.
No se rota si la posicion que se cerraria ya tiene un exit de SL, TP o earnings pendiente.

---

## Tabla comparativa v1 vs v2

| Aspecto | v1 | v2 | Señal de mercado usada |
|---|---|---|---|
| Condicion de cierre primario | score = 0 | score = 0 AND candle < 0 AND up_vol < 2 | tech_score + candle_score_5d + up_vol_5d |
| Retencion | No existe | OR(candle >= 0, up_vol >= 2) | candle_score_5d, up_vol_5d |
| Rotacion intrasectorial | No existe | Si delta_score >= 1.0 | tech_score relativo |
| Time stop | No existe | No existe | — |
| SL/TP | 2x/4x ATR | 2x/4x ATR | Sin cambio |

---

## Parametros de v2 — Resumen

| Parametro | Valor | Tipo |
|---|---|---|
| tech_score_entrada_min | 4.0 | Entrada |
| SL_atr_mult | 2.0 | Salida emergencia |
| TP_atr_mult | 4.0 | Salida emergencia |
| retencion_candle_score_min | 0 | Salida primaria |
| retencion_up_vol_5d_min | 2 | Salida primaria |
| retencion_logica | OR | Salida primaria |
| rotacion_delta_score | 1.0 | Rotacion |

---

## Metricas (pendiente — estrategia en desarrollo)

Completar una vez activa.

| Metrica | Valor |
|---|---|
| Fecha inicio | pendiente |
| Retorno total | — |
| Max drawdown | — |
| Operaciones totales | — |
| Win rate | — |
| Avg dias abierta | — |
| Rotaciones ejecutadas | — |
| Retenciones ejecutadas | — |

**Metrica de validacion especifica para v2**:
Registrar cuantas veces se ejecuto la retencion y si la posicion cerro
posteriormente en ganancia o perdida. Esto valida si el criterio de retencion
esta funcionando o retrasa cierres que igualmente terminan en perdida.

---

## Hipotesis para v3

A definir tras observar resultados de v2. Posibles direcciones:
- Subir delta de rotacion a 1.5 si hay demasiadas rotaciones innecesarias
- Cambiar logica OR a AND en retencion si se retienen demasiadas posiciones perdedoras
- Agregar candle_score_5d como filtro de ENTRADA (hoy solo se usa en salida)
