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
- **Contexto de TF superior (28/5/2026)**: la fila Estructura (SMC) muestra tambien
  la columna Semanal (tendencia_1w = SMC sobre barras semanales) y se suma fila
  "Tendencia mensual" (retorno 4 semanas). Todo calculado AL VUELO desde
  precios_diarios (resample W-FRI + market_structure_1w), sin tablas _1w. Es
  CONTEXTO: no modifica el veredicto.

## Estado (veredicto)

**3 valores**: `ALCISTA` / `NEUTRAL` / `BAJISTA` + frase corta. (No usar gradiente
de 5 ni un 4to estado: "neutral" absorbe "mixto" y "sin datos". No usar score
numerico: da falsa precision.)

## Arbol de decision (3 capas)

**Capa 1 — cada dimension del TICKER vota su sesgo (igual peso):**
| Dimension | Vota segun | Resultado |
|---|---|---|
| Tecnico (A) | MACD/RSI diario+semanal | Alcista / Bajista / Mixto |
| Opciones (B) | PCR_oi por plazo, SOLO por concordancia | Alcista / Bajista / Mixto |
| Estructura (F) | BOS/CHoCH/estructura SMC ventana 10 | Alcista / Bajista / Neutral |

> **B vota solo por concordancia (decision 27/5/2026).** Los 3 plazos
> (corto/medio/largo) son sentimientos distintos y SIEMPRE se muestran por
> separado en la tabla de opciones. B aporta al veredicto unicamente cuando los
> 3 plazos coinciden en direccion (3 alcistas -> voto alcista; 3 bajistas -> voto
> bajista). Si divergen entre plazos, B NO vota una direccion (cuenta como Mixto
> y se omite del consenso) y la divergencia se reporta tal cual como matiz en la
> frase (regla B3). Nunca se promedia/agrega un PCR unico: agregar desvirtua el
> mensaje descriptivo.
>
> **SMC (F) usa la ventana estrategica (10)** (estructura_10 / bos_10 / choch_10),
> coherente con el horizonte diario+semanal del informe.

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
es mas radar/screening que analisis de un ticker. **IMPLEMENTADO v1 (28/5/2026)**:
modo "Radar del dia" (dashboard/radar.py + cargar_radar). Senales vol/IV/PCR z,
tags legibles, guarda por percentil_vol, contexto sectorial, click -> informe.

**Screener por veredicto (1/6/2026)**: 2da seccion del modo "Radar del dia".
multiselect de veredictos (ALCISTA/NEUTRAL/BAJISTA) + multiselect de sectores
(vacio = todos) + boton Buscar -> tabla de tickers con ese veredicto sintetico.
cargar_veredictos_universo cacheada por fecha de datos con @st.cache_data: ~2 min
el primer Buscar del dia, luego instantaneo (los filtros veredicto/sector se
aplican sobre el universo cacheado -> cambiar filtros NO recalcula). Seleccion de
fila -> "Abrir informe" (reusa el nav del radar).

| ID | Cuando aplica | Senala | Rol |
|----|----|----|----|
| A1 | MACD Compra D y W | momentum alcista alineado | sesgo alcista |
| A2 | MACD Venta D y W | momentum bajista alineado | sesgo bajista |
| A3 | MACD difiere D vs W | senal poco confiable | mixto |
| A4 | RSI diario Sobrecompra | riesgo correccion | matiz (frase) |
| A5 | RSI diario Sobreventa | posible rebote | matiz (frase) |
| A6 | SMA alcista + ADX fuerte | tendencia con fuerza | refuerza |
| B1 | PCR_oi alcista en las 3 ventanas (concordancia) | posicionamiento alcista | sesgo alcista |
| B2 | PCR_oi bajista en las 3 ventanas (concordancia) | posicionamiento bajista | sesgo bajista |
| B3 | sesgo opciones divergente por plazo | cambio de expectativa; B no vota | matiz |
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

## Papel de trabajo (transparencia) — IMPLEMENTADO v1 (28/5/2026)

Capa de transparencia: por cada metrica del informe, ver valor + dato crudo +
como se calculo + ventana + tabla/fuente + umbral. Base = un "diccionario de
metricas" (definicion unica por metrica: dashboard/metricas.py:DICCIONARIO).
Implementacion: pestana "Papel de trabajo" (st.tabs) con tabla por seccion
(Tecnico / Opciones por plazo / Sector por plazo) + crudos en vivo. Export a PDF
A4 horizontal (dashboard/export_jpg.py:generar_papel_pdf + templates/papel.html).
Profundidad elegida: definicion + crudo donde este a mano (no se traen series
pesadas como el close de RSI). Pendiente (futuro): drill-down por celda
(popover), que hoy st.dataframe no permite sin renderizar filas custom.

## Que existe y que falta construir

Ya existe (reutilizable):
- Datos: precios_diarios, indicadores_tecnicos (+_1w), features_precio_accion,
  features_market_structure (SMC), opciones_pcr_plazo_diario,
  opciones_sector_pcr_plazo_diario, precios_semanales (resample W-FRI ok).
- Logica: src/utils/clasificacion_tecnica.py (RSI/MACD), src/utils/opciones_plazo.py.
- Tool MCP get_ticker_sintesis (parcial: tecnico D/W + opciones plazo + sector +
  reglas) — buen punto de partida.
- Infografias: scripts/reports/ (PNG para X).

Estado de construccion (28/5/2026): TODO el v1 + Fase 2 v1 HECHO. Ver detalle de
archivos en memory/dashboard.md.
1. [HECHO] **Logica del arbol** -> src/utils/dashboard_sintesis.py
2. [HECHO] **SMC y price action** en la sintesis (clasificacion_tecnica.py)
3. [HECHO] **Dashboard Streamlit** -> dashboard/app.py (+ view.py, sintesis_data.py)
4. [HECHO] **Export JPG** del informe -> dashboard/export_jpg.py
5. [HECHO] **Papel de trabajo** + diccionario de metricas + export PDF -> dashboard/metricas.py
6. [HECHO] **Radar (familia E)** v1 -> dashboard/radar.py + modo Radar en app.py
7. [PENDIENTE] Radar: earnings overlay + cruce accion+opciones (requiere refrescar
   ticker_zscore_diario, hoy congelada al 15/05). Drill-down por celda en el papel.

## Plan de construccion (orden incremental) -- COMPLETADO v1+Fase2 (28/5/2026)

1. [HECHO] **Logica de sintesis** (el cerebro): arbol + SMC + price action.
   Validado por COHERENCIA cualitativa sobre tickers reales (no backtesting).
2. [HECHO] **Dashboard Streamlit local**.
3. [HECHO] **Export JPG**.
4. [HECHO] (Fase 2) Papel de trabajo + diccionario de metricas + familia E (radar).

## Decisiones de diseno tomadas (resumen)

- Descriptivo, no predictivo. Complemento de TV.
- Estado: 3 valores + frase. Sin score numerico. Sin 4to estado.
- 3 dimensiones igual peso, consenso 2/3, sin datos se omite.
- Opciones (B) vota solo por concordancia de los 3 plazos; nunca se agrega/promedia
  un PCR unico. Divergencia por plazo = B no vota (Mixto) + matiz B3 en la frase.
- SMC (F) usa la ventana estrategica (10).
- Sector / muros / price action = complemento en la frase, NO modifican estado.
- Sector (no industria) para el analisis.
- Semanal W-FRI cerrado, con fecha.
- Conclusion por plantilla de frases predefinidas por regla.
- Validacion = coherencia cualitativa, NO backtesting de retorno (no incluye
  opciones por falta de historia, y mediria la vara equivocada en algo descriptivo).

## Vista "Consultas (IA)" — chat en lenguaje natural (15/6/2026)

Cuarto modo del sidebar. Caja de chat que responde preguntas en lenguaje natural
con datos REALES de la DB local, usando el LLM (Gemini) + las 16 tools del MCP
server (loop agentico: pregunta -> LLM -> tool_calls -> MCP -> LLM -> respuesta).

- Orquestador REUTILIZABLE en `src/agent/` (no acoplado a Streamlit; mismo motor
  servira al bot de Telegram, Fase 3 del MCP). Conecta al server por stdio con el
  rol `mcp_reader` (SELECT-only) -> el chat SOLO lee, cero riesgo de escritura.
- Eficiencia de tokens (key Gemini pospago): system prompt condensado, descripciones
  de tools truncadas, thinking off, salida acotada y poda del historial multi-turno.
- Registro de consumo: cada consulta se guarda en `llm_uso_tokens` (tokens reales
  entrada/salida, por fecha y usuario). La UI muestra el consumo por respuesta y un
  acumulado por dia. Detalle de implementacion: memory/dashboard.md.
