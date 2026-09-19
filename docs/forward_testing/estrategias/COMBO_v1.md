# COMBO_v1 — Documentacion de Estrategia

**Estado**: **DISCONTINUADA el 17/9/2026** (ver [Cierre](#cierre-17092026)).
Corrio del 2026-04-28 al 2026-09-16.
**ID en DB**: 5 (`activa = FALSE`)
**Inicio**: 2026-04-28
**Script**: `scripts/forward_testing/ft_bot_combo_v1.py` (se conserva, fuera de la rutina)
**Logica base**: `combo_tech_candle`
**Registro del cambio**: `ft_cambios.clave = combo_v1_discontinuada` (id 17)

> Motivo en una linea: el scoring de velas de 5 dias, que es lo UNICO que esta
> estrategia agrega sobre TECH_SECTOR_v1, no aporta -- ni en 5 anios de backtest ni
> en los 98 dias de forward testing.

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

---

## Cierre (17/09/2026)

### Que se estaba probando

Una sola cosa: si el `candle_score_5d` (patrones de vela + volumen de los ultimos 5
dias, mas las senales `bos_*_5` / `choch_*_5`) mejora la seleccion dentro de cada
sector. Todo el resto es identico a TECH_SECTOR_v1, que por eso funciona como CONTROL
exacto: mismo capital, mismos 9 sectores, mismo score tecnico de entrada y salida,
mismos SL/TP por ATR, mismo sizing.

El candle score entraba de dos formas:
- **desempate** del ranking dentro del sector (`tech_score DESC, candle_score_5d DESC`);
- **filtro** de entrada: se excluye el ticker con `candle_score_5d < -3,0`.

### Periodo y parametros evaluados

| | |
|---|---|
| Periodo de datos | 2026-04-28 -> 2026-09-16 (98 ruedas de equity) |
| Capital | $100.000, 9 sectores x $11.111, 5 posiciones por sector al 20% |
| Entrada | `tech_score >= 4,0` y `candle_score_5d >= -3,0` |
| Salida | `tech_score = 0` (degradado), SL 2x ATR14, TP 4x ATR14, earnings manana |
| Ranking | `tech_score DESC, candle_score_5d DESC` |
| Fuente del candle score | `features_precio_accion` + `features_market_structure` (ventana 5) |
| Backtest | 2021-09-01 -> 2026-09-16, motor `scripts/backtesting_historico` |

### Resultados en forward testing

| Metrica | COMBO_v1 | TECH_SECTOR_v1 (control, mismas 98 ruedas) |
|---|---|---|
| Retorno (equity a mercado) | **-0,09%** | **+2,29%** |
| Max drawdown | 5,06% | 5,42% |
| Volatilidad anualizada | 11,29% | -- |
| Sortino | -0,43 | -- |
| Operaciones cerradas | 477 | 750 |
| Aciertos | 33,3% | -- |
| PnL realizado | -98,40 USD | -- |
| Expectancy por operacion | -0,21 USD (-0,007%) | -- |
| Profit factor | 0,996 | -- |
| Payoff ratio | 1,96 | -- |
| Retorno medio por operacion | -0,108% (n=452) | +0,015% (n=750) |

**Diferencia por operacion contra el control: -0,123%, IC95 [-0,919%; +0,674%] ->
NO distinguible de cero.** Con 4-5 meses y ~450 operaciones el forward testing no
alcanza para probar que el candle score hace dano; lo que muestra es que no hay
ninguna senal de que ayude, y va en la misma direccion que el backtest.

Salidas (operaciones cerradas por motivo):

| Motivo | Ops | PnL USD | Medio |
|---|---|---|---|
| SCORE_DEGRADADO_3.0 | 185 | -4.849 | -1,25% |
| SCORE_DEGRADADO_0.0 | 81 | -8.133 | -4,70% |
| EARNINGS_MANANA | 58 | +2.804 | +2,21% |
| TAKE_PROFIT_ATR | 39 | +14.867 | +17,98% |
| SCORE_DEGRADADO_3.5 | 34 | -1.774 | -2,44% |
| ESTRATEGIA_DISCONTINUADA | 25 | +950 | +1,81% |
| resto (SCORE_DEGRADADO 1.0-2.5, STOP_LOSS_ATR, SPLIT_FIX) | 55 | -2.964 | -- |

### Resultados en backtest (5 anios, con entradas y salidas)

Pre-registrado en `docs/estructura_velas.md` sec. 9.3. Universo equal-weight del
periodo: +86,21%. Una variante "pasa" si gana plata y le gana al universo ajustado
por exposicion en 4 de los 6 tramos anuales.

| Variante | Retorno | Max DD | Ops | Ret/op | Anios que le gana al universo | Veredicto |
|---|---|---|---|---|---|---|
| COMBO_v1 con la historia vieja | +22,28% | -8,29% | 4.808 | +0,33% | 2/6 | NO pasa |
| COMBO_v1 con la historia sin futuro | +19,87% | -8,15% | 4.856 | +0,30% | 2/6 | NO pasa |
| **TECH_SECTOR_v1 (sin velas)** | **+24,43%** | -8,35% | 4.945 | +0,33% | 2/6 | NO pasa |

Sin costos. Las tres hacen ~960 operaciones por anio: con costos quedan peor.

### Motivos de la baja

1. **El aporte que la estrategia venia a medir no existe.** Con el mismo motor
   sectorial, agregar el candle score da MENOS retorno que no agregarlo: +19,9% vs
   +24,4% en 5 anios, y -0,09% vs +2,29% en los 98 dias de FT. El signo es el mismo
   en las dos mediciones, que son independientes.
2. **La corriente de abajo tampoco esta.** Sobre 148.000 velas desde 2023-06, ningun
   patron -- ni los del codigo viejo ni los clasicos con contexto -- tiene retorno en
   exceso distinguible de cero a 5 ni a 20 ruedas (`docs/estructura_velas.md` sec.
   5.4). No hay nada que un desempate por velas pueda explotar.
3. **Uno de sus dos insumos era invalido.** El `struct_score` del candle score sale de
   `features_market_structure` (`bos_*_5`, `choch_*_5`), cuya historia mira 10 ruedas
   al futuro. Eso no afecta lo que decidio en vivo (lee la ultima fila), pero si
   invalida cualquier medicion historica del componente. Arreglarlo implicaba
   reconstruir el score entero, y el punto 1 dice que no vale la pena.
4. **No es un problema de calibracion de umbrales.** El filtro `>= -3,0` excluye muy
   poco y el desempate solo ordena dentro del mismo `tech_score`: con un aporte
   medido en cero, mover el umbral cambia el ruido, no el resultado.

### Que NO dice este cierre

- No dice que el analisis de velas sea inutil en general: dice que **un score agregado
  de velas de 5 dias no mejora la seleccion dentro de un sector** en este universo y
  en diario, medido de dos formas.
- No dice que TECH_SECTOR_v1 sea buena: tampoco pasa la regla del backtest (2/6
  anios). Es el control, no el ganador.
- No mide el candle score como criterio de SALIDA: nunca se uso asi (esa idea quedo
  en las "hipotesis para v2", que no se implementaron).

### Cierre operativo

- Las **25 posiciones abiertas** se liquidaron al cierre del 2026-09-16 con
  `motivo_salida = ESTRATEGIA_DISCONTINUADA` (+950,04 USD). Es una salida ARTIFICIAL
  y esta etiquetada para poder excluirla: la serie "limpia" de la estrategia termina
  con las 452 operaciones anteriores (PnL -1.048,44 USD).
- `ft_estrategias.activa = FALSE` (id 5): `cargar_estrategia()` devuelve None, asi que
  el bot no puede operar aunque alguien lo corra suelto.
- Bot retirado de `scripts/manual/ft_run_diario.bat` (el bloque del BOT 5 quedo como
  comentario con el motivo).
- La entrada de `ft_setup_estrategias.ESTRATEGIAS` queda con la marca
  `discontinuada` y sus parametros: no se re-inserta, pero el registro no se pierde.
- Equity final: **99.901,60 USD (-0,10%)** al 2026-09-16, con 0 posiciones.
- Script de baja: `scripts/oneshot/discontinuar_estrategias_ft.py`.

### Como reabrirlo

Si alguna vez se quiere volver a probar velas dentro del motor sectorial, el
experimento tiene que ser distinto, no el mismo con otro umbral: usar los patrones
del modulo nuevo (`src/indicators/velas.py`, definicion clasica con contexto) y la
estructura confirmada (`features_estructura`), y medirlo primero en backtest con la
regla de lectura pre-registrada. Con lo medido hasta hoy, la hipotesis nace en contra.
