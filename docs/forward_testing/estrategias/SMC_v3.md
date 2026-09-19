# SMC_v3 — Documentacion de Estrategia (FT_SMC_v3_N5 y FT_SMC_v3_N3)

**Estado**: ACTIVAS (desde la rueda 2026-09-16)
**ID en DB**: 12 (`FT_SMC_v3_N5`) y 13 (`FT_SMC_v3_N3`)
**Control**: [SMC_v1.md](SMC_v1.md) (sigue corriendo sin cambios)
**Version anterior descartada**: [SMC_v2.md](SMC_v2.md)
**Script**: `scripts/forward_testing/ft_bot_smc_v3.py --ventana {5|3}`
**Logica base**: `smc_estructura_confirmada`
**Insumos**: `scripts/forward_testing/ft_scoring_estructura.py` sobre
`features_estructura` + `features_velas`
**Registro del cambio**: `ft_cambios.clave = smc_v3_estructura_confirmada` (id 16)
**Origen**: Tarea 23, Fase 2b. Medicion y pre-registro en
[docs/estructura_velas.md](../../estructura_velas.md) sec. 9.3

---

## Concepto

La MISMA regla de FT_SMC_v1, leyendo una estructura que se puede reproducir en vivo.

FT_SMC_v1 lee `features_market_structure`, que detecta los swings con una ventana
CENTRADA: el swing de la barra p se anota en p y solo se conoce en p+N. En la tabla
guardada eso es informacion del futuro, y en la ultima fila (la que usa el bot) es un
swing PROVISIONAL, que puede desaparecer al dia siguiente. La v3 usa swings
CONFIRMADOS: el swing de la barra p aparece en p+N y no se mueve nunca mas
(invariancia por test, `tests/test_estructura.py`).

**Pregunta que responde**: con la misma regla de entrada, salida y tamano de posicion,
leer la estructura confirmada -- y confirmarla rapido -- elige mejores operaciones que
la lectura provisional de la v1?

**Por que dos instancias**: el backtest mostro que la conclusion depende de la
velocidad de confirmacion (N), pero no distingue entre N=5 y N=3: dan casi el mismo
total y difieren por anio. Elegir el mejor de los dos mirando el backtest seria
sobreajuste hecho a mano, asi que corren las dos y decide la operacion real. Estaba
pre-registrado antes de correr el backtest.

---

## Que cambia y que no

| | FT_SMC_v1 (control) | FT_SMC_v3_N5 / N3 |
|---|---|---|
| Fuente de estructura | `features_market_structure` | `features_estructura` |
| Swings | ventana centrada; provisionales en la ultima barra | CONFIRMADOS en p+N, inmutables |
| Ventana N | 10 | 5 / 3 |
| Fuente de patrones de vela | `features_precio_accion` (envolvente que no envuelve, martillo sin contexto; medido: 71% de las envolventes no envuelven y 52% de los martillos son hanging man) | `features_velas` (definicion clasica con contexto; auditada al 100% contra el OHLCV) |
| `es_alcista` | columna de `features_precio_accion` | derivada del precio (`close > open`) |
| Ancla del lookback de 12 dias | `CURRENT_DATE` (el reloj) | ultima rueda de DATOS de la tabla |
| Score de entrada | `ft_scoring.calcular_score_estructura` | **el mismo, sin tocar** |
| Filtros obligatorios, trailing SL, salidas, sizing | -- | identicos |

El score se importa del modulo de la v1 a proposito: cualquier diferencia de resultado
es atribuible a la FUENTE y a N, no a una reimplementacion.

Auditoria de las dos fuentes contra el OHLCV (docs/estructura_velas.md sec. 12,
`scripts/manual/auditar_features_tablas.py`): `features_estructura` y `features_velas`
pasan el 100% de los chequeos (swings reales, invariancia, patrones que cumplen su
definicion). Lo que lee la v1 de `features_precio_accion` no: de ahi que la confirmacion
por vela del score (+1 por envolvente o martillo) sea, en la v1, mayormente ruido.

---

## Parametros (identicos a SMC_v1 salvo la ventana)

| Parametro | Valor | Descripcion |
|---|---|---|
| capital_inicial | $100.000 | por instancia |
| ventana_confirmacion | 5 (N5) / 3 (N3) | barras de confirmacion del swing |
| score_entrada_min | 1 | sobre un maximo de 3 |
| lookback_dias | 12 | dias calendario para buscar CHoCH/BOS bull (~10 habiles) |
| min_sl_dist_pct | 1,0% | distancia minima al swing low |
| max_sl_dist_pct | 8,0% | distancia maxima al swing low (+2% de margen al validar con el cierre real) |
| dias_max_pos | 20 | time stop (se mantiene: quitarlo fue el error de la v2) |
| max_posiciones | 5 | simultaneas |
| riesgo_por_trade | 15% | del capital actual |
| max_deploy_pct | 80% | techo de despliegue |

### Entrada (todas obligatorias)

1. CHoCH bull **o** BOS bull CONFIRMADO en el lookback (12 dias, anclado al dato);
2. `estructura_N >= 0` (no rota a la baja);
3. `choch_bear_N = 0` (sin cambio bajista hoy);
4. vela alcista hoy (`close > open`);
5. `dist_sl_N_pct` entre 1% y 8%.

Score de calidad para rankear (0-3): +1 si hubo CHoCH bull, +1 si hay confirmacion
(`vol_spike` **o** envolvente alcista **o** martillo, ahora con la definicion clasica),
+1 si `estructura_N = +1`.

### Salida (por prioridad)

P0 earnings manana -> P1 trailing SL roto -> P2 CHoCH bajista confirmado ->
P3 estructura rota (`estructura_N = -1`) -> P4 time stop 20 dias.
Sin take profit: la filosofia es salir por estructura.
Trailing SL = ultimo swing low CONFIRMADO, solo sube.

---

## Evidencia previa al despliegue (backtest pre-registrado)

Periodo 2021-09-01 -> 2026-09-16, motor `scripts/backtesting_historico`
(`--historia vieja|nueva --ventana N`), verificado contra el backtest guardado de
SMC_v1. Universo equal-weight del periodo: +86,21%. Regla fijada ANTES de correr: pasa
si gana plata y le gana al universo ajustado por exposicion en 4 de los 6 tramos
anuales.

| Variante | Retorno | Max DD | Ops | Ret/op | Expo | Anios | Veredicto |
|---|---|---|---|---|---|---|---|
| SMC_v1 historia vieja N=10 | +61,13% | -7,47% | 328 | +1,40% | 41% | 4/6 | pasa, con historia que mira al futuro |
| SMC_v1 historia nueva N=10 | +13,51% | -6,66% | 341 | +0,39% | 40% | 2/6 | NO pasa |
| **historia nueva N=5** | **+59,76%** | -10,29% | 446 | +1,17% | 46% | 4/6 | **pasa** |
| **historia nueva N=3** | **+58,22%** | -9,42% | 521 | +0,94% | 48% | 4/6 | **pasa** |

Sin costos. Lectura: la regla de SMC tal como se diseno dependia de ver los swings
antes de tiempo; con estructura confirmada sirve solo si la confirmacion es rapida.

**Esto NO es una promesa de rendimiento.** Los parametros de la regla (stop 1-8%,
lookback 12 dias, time stop 20 dias) vienen de la v1, que se armo mirando resultados:
el sesgo de "parametros elegidos a ojo" sigue vivo aunque la regla sea determinista.
El examen es el forward testing.

---

## Como se va a evaluar

1. **Contra FT_SMC_v1**, que corre en paralelo con la misma regla y la fuente vieja:
   mide el efecto de la FUENTE + N. Misma logica base, mismos dias.
2. **Entre N5 y N3**: mide la velocidad de confirmacion. No se elige uno hasta tener
   muestra (minimos de `ft_comparar` / `ft_tramos`: 20 ruedas y 10 operaciones por
   lado; con ~70-100 operaciones por anio en el backtest, eso es varios meses).
3. Metricas de `ft_equity_diaria` (equity a mercado) y `src/utils/ft_metricas`, con
   IC95 obligatorio. Contra el universo, ajustado por exposicion.

Nada de esto corta tramos de otras estrategias: son instancias nuevas
(`cambia_decisiones = FALSE`).

---

## Riesgos conocidos

- **N chico = mas señales y mas ruido**: N=3 hizo 521 operaciones en el backtest
  contra 328 de la v1. Mas operaciones = mas costo si algun dia se cuentan costos.
- **Los dos comparten el mismo dia de datos**: si `features_estructura` no se
  actualiza (Paso 2c fallado), los dos bots leen la fila vieja. El Paso 2c no frena la
  rutina si falla, asi que el guard de coherencia y `estado_pipeline` son los que
  avisan.
- **Solapamiento con SMC_v1**: las tres estrategias pueden abrir el mismo ticker. Son
  carteras virtuales independientes, no hay conflicto de capital, pero al comparar hay
  que mirar las operaciones compartidas y las exclusivas.
