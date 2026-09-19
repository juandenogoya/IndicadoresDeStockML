# TECH_SECTOR_v1 — Documentacion de Estrategia

**Estado**: ACTIVA
**ID en DB**: 4
**Inicio**: 2026-05-02
**Script**: `scripts/forward_testing/ft_bot_tech_sectorial.py`
**Logica base**: `tecnico_sectorial`
**Analisis de salidas**: [../ANALISIS_SALIDAS.md](../ANALISIS_SALIDAS.md) (19/9/2026). La
seccion "Logica de Salida" de abajo se corrigio ese dia: la version anterior describia
el bug del score 0,0, no la regla.
**Paso 1 (19/9/2026, pre-registrado)**: sacar la SMA21, el MACD o el RSI de la salida NO
mejora (5 anios de backtest, 4.902 operaciones): mismo exceso promedio y peor cola. La
salida queda como esta. Detalle: ANALISIS_SALIDAS.md sec. 7.
**Paso 2 (19/9/2026, pre-registrado)**: grilla de pesos (589 reglas de salida distintas),
midiendo que hace el precio DESPUES de salir: ninguna combinacion saca en mejores
momentos (0 de 588 candidatas). Despues de la salida actual la accion hace lo mismo que
el universo; los pesos solo cambian cuanto se queda la posicion y, con eso, la cola.
Detalle: ANALISIS_SALIDAS.md sec. 8.

---

## Concepto

Extension de TECH_v1 incorporando diversificacion sectorial forzada.
El universo de 199 tickers se divide en 9 sectores (clasificacion Yahoo Finance).
Cada sector opera de forma independiente con su propio pool de capital y slots.
No hay competencia entre sectores — un ticker de Technology no compite con uno de Energy.

**Pregunta que responde**: ?es mejor el mismo scoring tecnico con diversificacion
sectorial que sin ella? ?La limitacion por sector protege o limita los retornos?

---

## Parametros Globales

| Parametro | Valor | Descripcion |
|---|---|---|
| capital_total | $100,000.00 | Capital asignado a la estrategia |
| n_sectores | 9 | Sectores activos |
| capital_por_sector | $11,111.11 | capital_total / n_sectores |
| capital_por_posicion | $2,222.22 | capital_por_sector / max_pos_sector |
| max_posiciones_sector | 5 | Maximo de posiciones abiertas por sector |

---

## Logica de Entrada

### Filtro de candidatos (por sector)
Condiciones que TODAS deben cumplirse:

1. `tech_score >= 4.0` (score minimo de entrada)
2. `precio > SMA200` (implicito en tech_score — si no, score = 0)
3. ticker NO tiene posicion abierta en esta estrategia
4. ticker NO fue cerrado en la misma corrida del dia
5. ticker NO bloqueado por earnings
6. sector tiene slots disponibles (posiciones_abiertas_sector < 5)
7. sector tiene capital disponible (capital_invertido_sector < $11,111.11)

### Scoring para ranking
```
tech_score = f(SMA200, SMA50, SMA21, MACD, RSI)
```
Rango: 0 a 5.5 puntos.
Candidatos dentro de cada sector se ordenan por `tech_score DESC`.
Se abre posicion en los N primeros que quepan en slots y capital disponible.

### Sizing
```
qty = floor(capital_por_posicion / precio_entrada)
capital_real = qty * precio_entrada
```
Capital por posicion: ~$2,222.22 (puede variar segun precio y qty disponible).

### SL / TP (ATR-based)
```
SL = precio_entrada - (2.0 * ATR14)
TP = precio_entrada + (4.0 * ATR14)
```
Ratio riesgo/beneficio implicito: 1:2

---

## Logica de Salida

Orden de prioridad (`src/strategies/sectorial.evaluar_cierres_sectorial`, variante v1):
balance manana -> score -> stop -> take profit.

### Exit primario: degradacion de score
Condicion: `tech_score_actual <= 3.5` (SCORE_SALIDA).
Motivo registrado: `SCORE_DEGRADADO_<score>` (ej. `SCORE_DEGRADADO_3.0`).

Con los pesos del score (SMA50 2,0 / SMA21 1,0 / MACD 1,5 / RSI 1,0, filtro SMA200
obligatorio) ningun score cae entre 3,5 y 4,0, asi que **salir con <= 3,5 es exactamente
dejar de cumplir la entrada** (SMA200 y SMA50 y 2 de 3 entre SMA21 / MACD / RSI en
45-68). No hay distancia real entre umbral de entrada y de salida. Detalle y tabla de
combinaciones: ANALISIS_SALIDAS.md sec. 4.3.

**CORRECCION (19/9/2026).** Esta seccion decia que el exit primario era "score = 0
(precio < SMA200)" y anotaba como problema de la regla que "el bot cerraba las 5
posiciones de Technology y las reabria en la misma corrida" (2/5/2026). Eso era un BUG:
a la consulta de salida le faltaba el JOIN con precios_diarios, el close llegaba en 0 y
el score daba 0 todos los dias. Arreglado el 29/5/2026 (`ft_cambios`
fix_score_cero_salida). De las 412 salidas "SCORE_DEGRADADO_0.0" previas, 357 tenian
score real >= 4,0: la estrategia vendia al dia siguiente de comprar. **La historia de
salidas de esta estrategia solo mide la regla desde la rueda 2026-05-29.**

### Exit emergencia: SL hit
Condicion: `precio_actual <= stop_loss` (entrada - 2 x ATR14, fijo desde la entrada).
Motivo registrado: `STOP_LOSS_ATR`. Como va despues del score, casi nunca llega a
disparar (4 de 372 salidas validas).

### Exit emergencia: TP hit
Condicion: `precio_actual >= take_profit` (entrada + 4 x ATR14).
Motivo registrado: `TAKE_PROFIT_ATR`.

### Exit condicional: earnings
Condicion: earnings del ticker al dia siguiente
Motivo registrado: `EARNINGS_MANANA`
Prioridad: maxima (se ejecuta antes que cualquier otro criterio)

---

## Metricas al 2026-05-05

| Metrica | Valor |
|---|---|
| Dias activa | 3 |
| Capital actual | ~$100,036 |
| Retorno total | +0.04% |
| Posiciones abiertas | En seguimiento |
| Operaciones cerradas | En seguimiento |

**Nota**: los datos del primer dia de corrida (2026-05-02) tienen ruido por multiples
ejecuciones de debug. Los datos limpios comienzan desde 2026-05-05.

---

## Problemas Identificados en v1

1. **Exit binario**: score = 0 por un dia cierra todo. Sin graduacion.
   **(19/9/2026: diagnostico EQUIVOCADO -- era el bug del score 0,0, ver "Logica de
   Salida". La salida real es score <= 3,5, y si es "binaria" en otro sentido: sale en
   cuanto se da vuelta una sola condicion. Ver ANALISIS_SALIDAS.md.)**
2. **Sin diferenciacion de contexto**: no distingue entre acumulacion (lateral_ratio < 0.5)
   y tendencia agotada. Ambos reciben el mismo tratamiento de salida.
3. **Sin rotacion inteligente**: si el sector esta lleno (5/5) y aparece un candidato
   con score significativamente mayor que el peor de los 5, no se produce rotacion.
4. **Sin filtro de entrada por contexto de velas**: un activo con tech_score alto pero
   candle_score_5d negativo podria ser una trampa (precio tecnicamente bien pero sin momentum).

---

## Hipotesis para v2

Ver [TECH_SECTOR_v2.md](TECH_SECTOR_v2.md) cuando este disponible.

Lineas de exploracion:
- Incorporar `candle_score_5d` como filtro de entrada (confirmacion de momentum de velas)
- Incorporar `lateral_ratio` como filtro de salida (no cerrar si en acumulacion)
- Logica de rotacion intrasectorial (cerrar el peor para abrir el mejor)
- Umbral de salida gradual en lugar de binario

---

## Comparacion con TECH_v1

| Aspecto | TECH_v1 | TECH_SECTOR_v1 |
|---|---|---|
| Diversificacion | Sin restriccion sectorial | 9 sectores independientes |
| Max posiciones | 5 total | 5 por sector (45 potencial) |
| Capital por trade | 15% del capital actual | $2,222 fijo |
| Candidatos competencia | Global (los 199 tickers) | Por sector |
| Efecto concentracion | Posible | Limitado por sector |
