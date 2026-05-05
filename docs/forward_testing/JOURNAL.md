# Journal de Decisiones — Forward Testing

**Formato**: fecha | categoria | descripcion | hipotesis | resultado esperado
**Regla**: una entrada por decision relevante. No documentar cambios de formato
o correcciones de bugs obvios — solo decisiones de diseno, parametros y estrategia.

---

## 2026-04

### 2026-04-23 — DISENO
**Creacion del sistema Forward Testing**
Decidimos construir un motor propio de forward testing sobre la DB existente,
en lugar de usar un backtester externo (Backtrader, Zipline, etc.).
**Razon**: necesitamos evaluar estrategias en tiempo real con la misma data que
usa el scanner, no con data historica limpia. El FT es la extension natural del pipeline.
**Estructura definida**: ft_estrategias, ft_operaciones, ft_candidatos_diarios, ft_metricas_diarias.
**Precio de ejecucion**: cierre del dia (consciente y aceptado — sin acceso intraday).
**Ref**: docs/estrategias_ft.md

---

### 2026-04-28 — LANZAMIENTO
**Primera corrida de los 5 bots en produccion**
Estrategias activas desde hoy:
- ML_SCANNER_v1 (id=1): benchmark Bot1 Alpaca
- TECH_v1 (id=2): benchmark Bot2 Alpaca
- SMC_v1 (id=3): benchmark Bot3 Alpaca
- COMBO_v1 (id=5): primera estrategia nueva — sectorial + candle score

Capital inicial: $100,000 cada una.
TECH_SECTOR_v1 (id=4) arranca unos dias despues por ajustes de bugs.

---

## 2026-05

### 2026-05-02 — OBSERVACION
**TECH_SECTOR_v1: cierra y reabre las mismas posiciones**
Observamos que el bot cierra todas las posiciones con motivo SCORE_DEGRADADO_0.0
y las reabre inmediatamente. Esto indica que el scoring es demasiado binario:
cuando precio < SMA200 un solo dia, score = 0 y el bot cierra todo.
**Efecto**: capital girado innecesariamente, costos simulados sin beneficio real.
**Hipotesis**: una posicion con candle_score_5d positivo y lateral_ratio < 0.5
esta en acumulacion — no deberia cerrarse solo por un dia con score = 0.
**Pendiente**: disenar logica de retencion para TECH_SECTOR_v2.

---

### 2026-05-02 — BUG FIX
**Bug: dias_abierta enviado como lista en lugar de entero**
`trading_days_between()` retorna una lista de fechas, no un entero.
Se estaba pasando la lista directamente a la columna smallint de la DB.
**Fix**: usar `len(trading_days_between(entrada, fecha))`.
**Commit**: bc4a6b3

---

### 2026-05-02 — BUG FIX
**Bug: columnas es_alcista/vol_price_confirm en tabla incorrecta**
La query buscaba estas columnas en `features_market_structure`, pero
estan en `features_precio_accion`. Son features de precio-accion, no de estructura.
**Fix**: cambiar tabla en la query batch de registrar_estado_posiciones().
**Commit**: fda1328

---

### 2026-05-05 — DISENO
**Creacion de la capa de observacion diaria**
Agregamos `ft_posiciones_diarias` para trackear el estado de cada posicion
dia a dia: tech_score, candle_score, lateral_ratio, rango_5d, up_vol_5d.
**Objetivo**: tener datos suficientes para analizar patrones antes de modificar
las estrategias. No queremos cambiar parametros sin entender primero como se
comportan las posiciones actuales.
**Ref**: scripts/migrations/add_ft_posiciones_diarias.py

---

### 2026-05-05 — DISENO
**Creacion de retornos contrafactuales en ft_candidatos_diarios**
Para las oportunidades que NO se abrieron, calculamos el retorno que hubieran
tenido a 5, 10 y 20 dias habiles.
**Objetivo**: responder si estamos dejando dinero sobre la mesa al no abrir
ciertas posiciones (por slots llenos o capital insuficiente).

---

### 2026-05-05 — DECISION
**Denominador de retorno: capital total, no capital invertido**
El capital no desplegado en una estrategia esta inmovilizado — no esta disponible
para otras estrategias. Por lo tanto el denominador del retorno debe ser
el capital total asignado ($100,000), no el capital invertido.
Esto penaliza correctamente las estrategias que no despliegan bien su capital.

---

### 2026-05-05 — EXPLORACION INICIADA
**Hipotesis: incorporar Momentum/Lateral/Retorno como features de salida/rotacion**
Las features de la capa de observacion (candle_score_5d, lateral_ratio, up_vol_5d)
podrian mejorar las decisiones de salida y rotacion en:
- TECH_SECTOR_v1: evitar cierres innecesarios de posiciones en acumulacion
- SMC_v1: filtrar entradas en contextos laterales, anticipar salidas estancadas
**Estado**: en diseno. Ver TECH_SECTOR_v2.md y SMC_v2.md cuando esten disponibles.
**Proximos pasos**: definir parametros especificos antes de implementar.

---

## Template de entrada

```
### YYYY-MM-DD — [DISENO | LANZAMIENTO | OBSERVACION | BUG FIX | DECISION | RESULTADO | EXPLORACION]
**Titulo descriptivo**
Descripcion de que ocurrio o que decidimos.
**Razon / Hipotesis**: por que tomamos esta decision.
**Efecto esperado**: que esperamos que cambie.
**Resultado real**: (completar cuando tengamos datos)
**Ref**: archivo o commit relacionado
```
