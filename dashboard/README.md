# Dashboard — Informe descriptivo por ticker

> Spec de diseno. Acordada en sesion de 26-27/5/2026. El desarrollo arranca
> en una sesion dedicada. Este documento es la fuente de verdad del diseno.

## Proposito y filosofia

Herramienta **DESCRIPTIVA** (no predictiva): describe la situacion actual de
un ticker cruzando tecnico + opciones + sector, con reglas de confluencia /
divergencia y una conclusion legible. NO recomienda comprar/vender ni pretende
predecir retornos.

- **Complemento de TradingView/Investing, no reemplazo.** El usuario grafica en
  TV; aca obtiene la lectura de opciones+sector que TV no da masticada.
- **Diferencial real**: capa de opciones contextualizada (PCR por plazo, muros
  de OI como S/R) + **sentimiento sectorial via opciones** (casi inexistente en
  retail) + sintesis con reglas trazables.
- **NO competir** en lo commodity (graficos, indicadores sueltos): TV gana.
- El ML queda fuera por ahora (no mostro edge concluyente; ver memoria ml_modelos).

## Layout del informe

```
XYZ · Technology / Semiconductors · $123.45 · cierre 22/05
VEREDICTO: <estado> — <frase de 1 linea>
───────────────────────────────────────────────
TECNICO              DIARIO        SEMANAL (al cierre <viernes>)
  RSI                Sobrecompra   Neutral
  MACD               Compra        Venta
  Tendencia (SMA)    Alcista       —        (solo diario)
  Fuerza (ADX)       Fuerte        —        (solo diario)
  Estructura (SMC)   CHoCH alcista / BOS / lateral

OPCIONES — Sesgo (ticker)          NIVELES — Muros de OI (ticker)
  Plazo PCR_vol PCR_oi Sesgo         Plazo  Soporte      Resistencia
  Corto 0.57    0.61   Alcista       Corto  115 (-6%)    130 (+5%)
  Medio 0.72    0.72   Alcista       Medio  118 (-4%)    128 (+3%)
  Largo 0.51    0.52   Alcista       Largo  —            —

SENTIMIENTO — Sector (Technology)
  Plazo PCR_vol_sec Sesgo    ¿Inusual?
  Corto 0.95        Neutral  z +1.8 (cobertura alta)
  Medio 0.88        Alcista  z +1.0
  Largo 0.70        Alcista  normal
  [pie] z = desvios del PCR del sector vs su media historica.
        z>+1 cobertura inusual (bajista atipico); z<-1 optimismo inusual; |z|<1 normal.

LECTURA DE REGLAS
  ✓ <regla activada 1>
  ✓ <regla activada 2>
  Veredicto del conjunto: <estado>

CONCLUSION
  RAPIDA: 3-4 bullets accionables
  DETALLADA: parrafo con matices y que vigilar
```

- Encabezado: ticker, sector/industria (industria solo dato), precio, fecha.
  **Veredicto de 1 linea arriba** (escaneo rapido) + conclusion al final.
- **Sector** para el analisis (10 grupos); industria solo como dato de encabezado.
- **Semanal**: W-FRI, semana CERRADA, con fecha visible (se actualiza los viernes).

## Estado (veredicto)

**3 valores**: `ALCISTA` / `NEUTRAL` / `BAJISTA` + frase corta. (No usar gradiente
de 5 ni un 4to estado: "neutral" absorbe "mixto" y "sin datos". No usar score
numerico: da falsa precision.)

## Arbol de decision (3 capas)

**Capa 1 — cada dimension del TICKER vota su sesgo (igual peso):**
| Dimension | Vota segun | Resultado |
|---|---|---|
| Tecnico (A) | MACD/RSI diario+semanal | Alcista / Bajista / Mixto |
| Opciones (B) | PCR_oi por plazo | Alcista / Bajista / Mixto |
| Estructura (F) | BOS/CHoCH/estructura SMC | Alcista / Bajista / Neutral |

**Capa 2 — combinar votos -> ESTADO:**
- Consenso 2/3 (de las disponibles) en una direccion -> ALCISTA / BAJISTA.
- Repartido / mixto -> NEUTRAL.
- Dimension sin datos (ej. opciones sin liquidez) se OMITE del voto; la frase
  lo aclara. Si solo hay 2 dimensiones, exigir 2/2.

**Capa 3 — complementos (modulan la FRASE, NO cambian el estado):**
- **Sector (D)**: acompana o no acompana. NO modifica el estado (decision del
  26/5: el estado describe al ticker; el sector es contexto del entorno; es
  asimetrico — si el sector diverge al alza "acompana", no "sube" la perspectiva
  del ticker). Va en la frase: "...con el sector acompanando" / "...el sector se
  esta cubriendo".
- **Muros (C)**: ej. estado Alcista + resistencia cercana -> sigue Alcista, frase
  "alcista, pero con techo de opciones cerca".
- **Price action**: NO familia propia; modifica reglas de muro (confirmacion):
  patron de reversion + en muro + volumen -> refuerza la lectura de techo/piso.

## Familias de reglas

Sesgo (votan el estado): **A** (tecnico), **B** (opciones), **F** (SMC).
Contexto (modulan la frase): **C** (muros + price action), **D** (sector).
Familia **E** (actividad inusual: vol/IV z-score de opciones_zscore_diario) ->
**fase 2** (es mas radar/screening que analisis de un ticker).

| ID | Cuando aplica | Senala | Rol |
|----|----|----|----|
| A1 | MACD Compra D y W | momentum alcista alineado | sesgo alcista |
| A2 | MACD Venta D y W | momentum bajista alineado | sesgo bajista |
| A3 | MACD difiere D vs W | senal poco confiable | mixto |
| A4 | RSI diario Sobrecompra | riesgo correccion | matiz (frase) |
| A5 | RSI diario Sobreventa | posible rebote | matiz (frase) |
| A6 | SMA alcista + ADX fuerte | tendencia con fuerza | refuerza |
| B1 | PCR_oi alcista 3 ventanas | posicionamiento alcista | sesgo alcista |
| B2 | PCR_oi bajista 3 ventanas | posicionamiento bajista | sesgo bajista |
| B3 | sesgo opciones divergente por plazo | cambio de expectativa | matiz |
| B4 | MACD venta + PCR_oi medio bajista | presion bajista confirmada | sesgo bajista |
| C1 | RSI sobrecompra + call wall cercano arriba (<3%) | techo potencial | matiz |
| C2 | RSI sobreventa + put wall cercano abajo (<3%) | rebote potencial | matiz |
| C3 | precio entre soporte y resistencia cercanos | rango acotado | matiz |
| F1 | CHoCH alcista reciente | giro al alza (anticipa) | sesgo alcista |
| F2 | CHoCH bajista reciente | giro a la baja | sesgo bajista |
| F3 | BOS alcista + estructura alcista | continuacion alcista | refuerza |
| F4 | estructura bajista (estructura_10=-1) | contexto bajista | refuerza bajista |
| D1 | ticker alcista + sector alcista | el sector acompana | contexto (frase) |
| D2 | ticker alcista + sector cobertura inusual (z>+1.5) | sector no acompana | contexto (frase) |
| D3 | sector con \|z\| alto | posible rotacion sectorial | contexto |

(Price action: refuerza C1/C2 cuando hay patron de reversion + volumen.)

## Formato y entrega

- **v1: dashboard Streamlit LOCAL** (corre en la PC, lee DB local). No Streamlit
  Cloud: bajo Plan C los datos viven en local. Selector de ticker -> informe.
- **Boton exportar a JPG**: `YYYYMMDD_ticker.jpg`. Reusar la maquinaria de
  infografias existente (scripts/reports/). 
- Escalar (acceso remoto) = fase posterior, a definir.

## Papel de trabajo (transparencia) — FASE 2

Capa de drill-down: por cada dato, ver dato crudo + feature + como se calculo +
tabla/fuente. Base = un "diccionario de metricas" (definicion unica por metrica:
formula, ventana, tabla origen). En Streamlit se resuelve con expandible/tooltip.
NO bloquea el v1.

## Que existe y que falta construir

Ya existe (reutilizable):
- Datos: precios_diarios, indicadores_tecnicos (+_1w), features_precio_accion,
  features_market_structure (SMC), opciones_pcr_plazo_diario,
  opciones_sector_pcr_plazo_diario, precios_semanales (resample W-FRI ok).
- Logica: src/utils/clasificacion_tecnica.py (RSI/MACD), src/utils/opciones_plazo.py.
- Tool MCP get_ticker_sintesis (parcial: tecnico D/W + opciones plazo + sector +
  reglas) — buen punto de partida.
- Infografias: scripts/reports/ (PNG para X).

Falta (el grueso del desarrollo):
1. **Logica del arbol** (3 dimensiones -> estado de 3 + frase) — NO esta implementada.
2. **Sumar SMC y price action** a la sintesis (hoy no estan en la tool).
3. **Dashboard Streamlit** (selector + render de cuadros).
4. **Export JPG**.

## Plan de construccion (orden incremental)

1. **Logica de sintesis** (el cerebro): arbol + SMC + price action. Testeable sin
   UI. Permite validacion de COHERENCIA (cualitativa) sobre tickers reales —
   NO backtesting de retorno (es descriptivo; ver decision en memoria).
2. **Dashboard Streamlit local**.
3. **Export JPG**.
4. (Fase 2) Papel de trabajo + diccionario de metricas + familia E (radar).

## Decisiones de diseno tomadas (resumen)

- Descriptivo, no predictivo. Complemento de TV.
- Estado: 3 valores + frase. Sin score numerico. Sin 4to estado.
- 3 dimensiones igual peso, consenso 2/3, sin datos se omite.
- Sector / muros / price action = complemento en la frase, NO modifican estado.
- Sector (no industria) para el analisis.
- Semanal W-FRI cerrado, con fecha.
- Conclusion por plantilla de frases predefinidas por regla.
- Validacion = coherencia cualitativa, NO backtesting de retorno (no incluye
  opciones por falta de historia, y mediria la vara equivocada en algo descriptivo).
