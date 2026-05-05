# {NOMBRE_ESTRATEGIA} — Documentacion

**Estado**: [ACTIVA | EN DESARROLLO | PAUSADA | DESCONTINUADA]
**ID en DB**: {id}
**Inicio**: {YYYY-MM-DD}
**Fin**: {YYYY-MM-DD o "activa"}
**Script**: `scripts/forward_testing/{nombre_script}.py`
**Logica base**: `{logica}`
**Version anterior**: [{NOMBRE_v_anterior}.md]({NOMBRE_v_anterior}.md)

---

## Concepto

[Descripcion en 2-4 lineas de QUE hace esta estrategia y por que existe.]

**Pregunta que responde**:
[?Que hipotesis estamos probando con esta version?]

**Diferencia respecto a version anterior** (si aplica):
[Que cambio especificamente: parametros, logica, filtros, etc.]

---

## Parametros Globales

| Parametro | Valor | Descripcion |
|---|---|---|
| capital_total | $100,000.00 | Capital asignado |
| ... | ... | ... |

---

## Logica de Entrada

### Filtro de candidatos
Condiciones que TODAS deben cumplirse:

1. `condicion_1`
2. `condicion_2`
3. ...

### Scoring y ranking
[Como se puntuan y ordenan los candidatos]

### Sizing
```
[formula de calculo de qty y capital por trade]
```

### SL / TP
```
SL = ...
TP = ... (o "ninguno" si es salida estructural)
```

---

## Logica de Salida

| Prioridad | Condicion | Motivo registrado |
|---|---|---|
| P0 | Earnings manana | EARNINGS_MANANA |
| P1 | ... | ... |

[Descripcion adicional de la logica de salida si es compleja]

---

## Metricas

**Periodo**: {fecha_inicio} a {fecha_fin o "presente"}

| Metrica | Valor |
|---|---|
| Dias activa | |
| Retorno total | |
| Max drawdown | |
| Posiciones abiertas (prom) | |
| Operaciones totales | |
| Win rate | |
| Avg retorno ganadora | |
| Avg retorno perdedora | |
| Avg dias abierta | |
| Capital ocioso promedio | |

---

## Observaciones del Periodo

[Lo que observamos mientras corria la estrategia. Comportamientos inesperados,
patrones notados, sectores que funcionaron mejor/peor, etc.]

---

## Problemas Identificados

1. **Problema 1**: descripcion
2. **Problema 2**: descripcion

---

## Hipotesis para la siguiente version

Ver [{NOMBRE_v_siguiente}.md]({NOMBRE_v_siguiente}.md) cuando este disponible.

- Hipotesis 1: ...
- Hipotesis 2: ...
