# Infografia fundamental -- spec de diseno
# v1 -- 2026-06-01

> Spec del producto visual de Analisis Fundamental por ticker. Formaliza el
> diseno (el prototipo existia funcionando pero sin diseno deliberado) y define
> los puntos de pulido. Sigue la regla del proyecto: documentar antes de tocar.
>
> Hermano de docs/reportes.md (infografia TECNICA) y docs/fundamentales_calculo.md
> (de donde salen los numeros). Esta es la infografia FUNDAMENTAL.

## 1. Proposito y audiencia (decidido 2026-06-01)

- **Destino: publicar en X / redes.** Imagen unica, autocontenida, que se
  entienda sin contexto previo.
- **Audiencia: terceros** que no conocen la empresa ni el sistema. Se consume
  mayormente en **movil** (feed de X) -> legibilidad y contraste mandan.
- **Gancho**: la banda de KPIs grandes arriba (lo primero que se ve al hacer
  scroll). El diferencial vs infografias comerciales: la columna **vs sector**
  (comparacion con pares de la misma region) -- nadie en retail lo da masticado.
- **Filosofia** (igual que el resto del proyecto): DESCRIPTIVA, foto de lo que
  paso. NO predictiva. Por eso NO lleva: DCF, fair value, ratings subjetivos
  (estrellas), ni recomendacion de compra/venta. Solo datos reales + comparacion.

## 2. Que NO hace (limites deliberados)

- No proyecta (sin DCF, sin precio objetivo).
- No opina (sin ratings de management/moat tipo InvestingVisuals).
- No mezcla con lo tecnico/opciones (eso vive en la infografia tecnica y el
  dashboard). Esta es solo el fundamental.
- No usa LLM: 100% datos + reglas deterministas (cero tokens, reproducible).

## 3. Fuente de datos

Lee DIRECTO de la DB local (los fundamentales viven en local, Plan C):
- fundamentales_ratios_q  : ratios del ultimo Q + perfil (banco/no-banco).
- fundamentales_ticker_vs_sector : comparacion vs mediana de pares.
- ticker_pais             : region.
- precios_diarios         : ultimo cierre USD.
- fundamentales_income_q + _cashflow_q : serie para el sparkline.

Motor: scripts/reports/make_infografia_fundamental.py (HTML+CSS -> WeasyPrint ->
PNG via PyMuPDF, misma maquinaria que la infografia tecnica). Templates:
templates/infografia_fundamental.{html,css}.

## 4. Layout (5 bloques -- estructura CONGELADA, solo se pule estetica)

```
+--------------------------------------------------------------+
| TICKER (grande) | Sector . Region                            |  ENCABEZADO
|                 | Cierre USD XX (fecha)                       |
|                 | Ultimo Q . moneda de reporte               |
+--------------------------------------------------------------+
| [ KPI ]  [ KPI ]  [ KPI ]  [ KPI ]            (banda arena)  |  GANCHO
+--------------------------------------------------------------+
| Ingresos (+FCF)          | Crecimiento interanual            |
| [sparkline hasta 8 Q]    | tabla YoY                         |
+--------------------------------------------------------------+
| Valuacion (vs sector)    | Calidad/Rentabilidad (vs sector)  |
| tabla 3 col              | tabla 3 col                       |
+--------------------------------------------------------------+
| leyenda peer-basis (con quien se comparo)                    |
| SOLVENCIA (cajas)                                            |
| footer: handle . "foto fundamental, no recomendacion"        |
+--------------------------------------------------------------+
```

## 5. Indicadores por perfil (CONGELADO -- set actual, no cambia en v1)

El set depende del perfil (ver docs/fundamentales_calculo.md). Coloreo verde/
rojo en la columna "vs sector" segun direccion de la metrica (PER bajo=bueno,
ROE alto=bueno; absolutos=neutro sin color).

### NO-FINANCIERO
| Bloque | Metricas |
|--------|----------|
| KPIs (banda) | Margen bruto, Margen operativo, Margen neto, Margen FCF (TTM) |
| Crecimiento | Ingresos / Ut.neta / BPA / FCF -- YoY |
| Valuacion (vs sector) | PER, P/B, P/S, EV/EBITDA |
| Calidad (vs sector) | ROE, ROA, ROIC, Margen neto |
| Solvencia | Liquidez corriente, Deuda/Patrimonio, Caja/Deuda neta |
| Sparkline | Ingresos + FCF (2 lineas) |

### FINANCIERO (banca/seguros/brokers)
| Bloque | Metricas |
|--------|----------|
| KPIs (banda) | Margen neto, ROE, ROTCE, Ratio eficiencia (TTM) |
| Crecimiento | Ingresos / Ut.neta / BPA -- YoY (sin FCF) |
| Valuacion (vs sector) | PER, P/B, P/S (sin EV/EBITDA) |
| Calidad (vs sector) | ROE, ROTCE, ROA, Margen neto, Ratio eficiencia |
| Solvencia | Deuda/Patrimonio (sin liquidez corriente NI deuda neta) |
| Sparkline | Solo Ingresos (FCF no aplica a banca) |

## 6. Decisiones de diseno tomadas

- **Ticker como texto grande** (no logo). Conseguir/cachear logos es trabajo
  aparte; el texto en tipografia fuerte da identidad suficiente. (Reevaluable.)
- **Paleta crema/alto contraste** (consistente con la infografia tecnica):
  fondo #faf8f3, banda arena #ece6d8, acento negro, verde/rojo solo en vs-sector.
- **Cierre en USD explicito** en el encabezado (el ADR cotiza USD aunque reporte
  en otra moneda; los absolutos van en moneda de reporte, rotulados).
- **Leyenda peer-basis visible**: "vs Sector/Region (n=X)" o "sin pares
  regionales -> vs USA" o "pocas empresas". Honestidad sobre con quien se compara.
- **Sparkline minimalista** (forma de la serie, sin ejes): muestra trayectoria,
  las magnitudes van en KPIs/crecimiento.

## 7. Puntos de PULIDO para X (lo que se implementa en v1)

El prototipo funciona pero nacio en formato 820x1180. Para redes:

1. **Formato 4:5 (1080x1350 px)** [DECIDIDO]: ratio que mas ocupa el feed de
   X/IG en movil sin recorte. Hoy 820x1180 (~0.69) queda angosto. Ajustar @page
   y --page-w a 1080, alto 1350, reescalar tipografias proporcionalmente.
2. **Tipografia mas grande en KPIs y ticker**: en movil los KPIs son el gancho;
   subir tamano para que se lean en miniatura del feed.
3. **Resolucion de salida 2x** (ya esta: matrix 2,2) -> nitido en pantallas retina.
4. **Footer con fecha de generacion + handle**: trazabilidad y branding.
5. **Caveat de moneda visible** cuando reporting_currency != USD (ADRs): que el
   lector sepa que los absolutos no estan en USD.
6. **Deuda neta en bancos: OCULTAR** [DECIDIDO]. Para banca es poco informativa
   (incluye depositos/funding). Igual que ya ocultamos liquidez corriente, el
   bloque solvencia de financieros queda solo con Deuda/Patrimonio.

NO se toca en v1: el set de indicadores (congelado), la estructura de 5 bloques,
la logica de perfiles, la fuente de datos.

## 7b. Diseno v2 -- bloques nuevos (decidido 1/6/2026)

Tras comparar con TradingView (capturas de JPM): TV gana en datos de banca
regulatoria (prestamos/depositos/CET1) que yahooquery NO expone (0/18). NO
competimos ahi. Nuestro plus sigue siendo el comparativo sectorial. Rol elegido:
**foto fundamental completa y autonoma** (se entiende sola, comparativo = plus).

Se agregan 2 bloques (manteniendo formato 4:5, opcion A -- apretar si hace falta):

### Bloque "Dividendos" (ambos perfiles)
- **Dividend yield TTM** = CashDividendsPaid_ttm / MarketCap.
- **Payout ratio TTM** = CashDividendsPaid_ttm / NetIncome_ttm (NULL si NI<=0).
- Si no paga dividendos -> "No distribuye dividendos" (honesto).
- Validado vs TV: JPM yield 2.18%/payout 29% (TV 1.97%/28%); AAPL 0.42%/13%;
  KO 3.35%/80%; XP/BABA no pagan -> None. Cobertura CashDividendsPaid 15/18 fin.
- Caveat: usamos caja real (CashDividendsPaid) -> leve dif. vs dividendo
  declarado anualizado de TV. Ambos validos.

### Bloque "Tendencia de margenes" (mini-sparkline multi-Q)
- Evolucion en los ultimos Q de: margen neto (no-banco) o ROE (banco).
- Refuerza la lectura de TRAYECTORIA (acelera/desacelera), que un TTM no da.
- Equivalente compacto al panel "Rendimiento" de TV.

Lo que NO se hace (coherente con tener-o-no-tener el dato): prestamos/depositos/
CET1 (no estan en yahooquery), waterfall ingresos->beneficio (alto esfuerzo SVG),
descriptivos CEO/empleados/beta (no en las tablas fundamentales). Set de ratios
existente NO se toca.

## 8. Estado / proximos pasos

1. [HECHO] Prototipo funcional perfil-aware (make_infografia_fundamental.py).
   Validado vs balances oficiales MU/JPM.
2. [HECHO] Spec de diseno (este doc): proposito X, layout congelado, indicadores,
   pulido.
3. [HECHO 1/6/2026] Pulido implementado:
   - Formato 4:5 -> 1620x2025 px @2x (ratio 0.800 exacto). CSS reescalado.
   - Tipografias mas grandes para movil (ticker 84px, KPIs 46px).
   - Deuda neta OCULTA en bancos (JPM: solvencia solo Deuda/Patrimonio).
   - Caveat de moneda en ADRs (BABA: "absolutos en CNY, no USD" en rojo).
   - Sparkline reescalado 560x210.
   Validado: MU/JPM/BABA generados OK, los 3 a 4:5 exacto.
4. [HECHO 1/6/2026] Boton en el dashboard: vista "Analisis Financiero" modo
   "Por ticker" -> seccion "Compartir en redes" con boton "Generar infografia
   (PNG)" + preview + download_button. make_infografia_fundamental.py expone
   generar_infografia(ticker)->Path (API reutilizable); el dashboard la importa
   de forma diferida (al click) para no cargar weasyprint en el arranque.
5. [HECHO 1/6/2026] Bloques v2 (seccion 7b) implementados:
   - Bloque Dividendos: yield TTM + payout TTM, o "No distribuye dividendos".
   - Bloque Tendencia de margenes: mini-sparkline 1 serie (margen neto no-banco /
     ROE banco), color verde/rojo segun direccion, etiqueta primer y ultimo valor.
   - Layout 4:5 mantenido (opcion A: compactados paddings). Salida 1620x2025,
     una sola pagina. Validado MU (paga, margen 28->41%), JPM (banco), BABA
     (no paga -> leyenda, margen 9->10%).
6. [HECHO 2/6/2026] Ajustes de presentacion:
   - Valuacion y Calidad: columna "Sector" (mediana de pares, peer_median) entre
     Valor y vs-sector. La diferencia % deja de ser abstracta (ej. PER 28.9 vs
     sector 25.1 = +15%). _fila_cmp recibe fmt_fn (_ratio valuacion / _pct calidad)
     para formatear la mediana en la misma escala. Maneja NULL (metricas sin par,
     ej. ROTCE/Ratio eficiencia en banca -> "—"). Tablas pasan a 4 columnas; .sec
     en gris (secundaria) para que el valor del ticker siga destacando.
   - Dividendos: 3er dato "Div./accion (TTM)" = DPS anualizado = yield_TTM x cierre
     (USD, mismo plano que cierre/yield; util incluso en ADRs). div-val 33px para
     entrar 3 items. Validado VALE (USD 1.06), JPM (USD 6.46 = 2.18% x 296.58).
   - NO se toco: el set de indicadores, perfiles, fuente de datos, layout 4:5.
7. [HECHO 2/6/2026] Valuacion AL CIERRE del dia: PER/P-B/P-S/EV-EBITDA usan las
   columnas *_px (recalculadas con el precio actual, no el congelado de Yahoo). El
   bloque Valuacion lee r["pe_ratio_px"]/etc. y la columna Sector usa la mediana
   recalculada (compute_sector_valuacion_px). NULL -> "-" (sin fallback a Yahoo).
   Ver docs/fundamentales_calculo.md ("Multiplos al cierre"). Ej CVS: PER 51.7->39.9.
8. [FUTURO] Logo de empresa; version horizontal 16:9; tambien sumarla a la app
   Streamlit de reportes (scripts/reports/app.py). Dots por trimestre en los
   sparklines (evaluado 2/6, no implementado: el grafico ya es multi-Q). EV/EBITDA:
   revisar el EBITDA_ttm (denominador) que difiere de TV en algunos tickers.
