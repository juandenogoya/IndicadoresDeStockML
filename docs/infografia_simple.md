# Infografia SIMPLE (tecnico + fundamental) -- spec de diseno
# v1 -- 2026-06-21 (PREVIO a codificar: documentar antes de tocar)

> Spec de una TERCERA infografia, mas simple y visual, pensada para redes
> sociales. NO reemplaza a las dos existentes: convive con ellas.
>
> Familia de infografias del proyecto:
> - TECNICA (docs/reportes.md)            : datos del MCP, densa, por ticker.
> - FUNDAMENTAL (docs/infografia_fundamental.md) : foto fundamental completa 4:5.
> - SIMPLE (este doc)                     : 1 tarjeta combinada, minima, "de un
>                                           vistazo", con 2 GRAFICOS.
>
> Diferencia clave con las otras dos: estas son tablas/KPIs densos; la SIMPLE
> prioriza 2 graficos (precio+muros y PER+ROE) y muy poco texto. Es "menos datos,
> mejor contados".

## 1. Proposito y audiencia

- **Destino: publicar en X / redes.** Imagen unica, autocontenida, escaneable en
  el feed movil en 2 segundos.
- **Audiencia: terceros** que no conocen el ticker. Legibilidad y contraste mandan.
- **Gancho**: el grafico de precio con los muros arriba + la cinta de veredicto.
- **Filosofia** (igual que el resto): DESCRIPTIVA, foto de lo que paso, NO
  predictiva. Sin DCF, sin precio objetivo, sin recomendacion. Sin LLM (100%
  datos + reglas deterministas, cero tokens, reproducible).

## 2. Que NO hace (limites deliberados)

- No proyecta (sin fair value / precio objetivo).
- No es exhaustiva: es el "resumen visual". Para el detalle estan las otras dos
  infografias y el dashboard.
- No usa LLM.
- El veredicto que muestra es DESCRIPTIVO (sintesis de 3 dimensiones), NO una
  recomendacion de compra/venta (mismo criterio que el dashboard).

## 3. Fuente de datos (todo LOCAL, Plan C)

| Bloque | Tabla / fuente | Columnas clave |
|--------|----------------|----------------|
| Precio (linea) | precios_diarios | fecha, close (~4 meses) |
| Muros S/R | opciones_pcr_plazo_diario | soporte_strike/soporte_fuerza, resistencia_strike/resistencia_fuerza, ventana |
| Opciones por plazo | opciones_pcr_plazo_diario | pcr_vol, veredicto_oi (por ventana corto/medio/largo) |
| PER ticker (linea) | fundamentales_ratios_q | pe_ratio (fiscal), por trimestre calendario |
| PER sector (barras) | fundamentales_ratios_q JOIN activos | mediana de pe_ratio del mismo sector, por trimestre calendario |
| Veredicto (cinta) | src/utils/dashboard_sintesis.sintetizar via dashboard/sintesis_data | estado (ALCISTA/NEUTRAL/BAJISTA) |

Motor: scripts/reports/make_infografia_simple.py (HTML+CSS -> WeasyPrint -> PNG
via PyMuPDF, misma maquinaria que las otras dos). Graficos como SVG INLINE en
Python (mismo patron que los sparklines de make_infografia_fundamental: se arma
el `<svg>` con `<polyline>`/`<rect>` a mano y WeasyPrint lo dibuja -- NO se usa
matplotlib ni PNG embebido). Templates: templates/infografia_simple.{html,css}.

## 4. Layout (CONGELADO -- vertical 4:5, 1080x1350 @1x / 2x salida)

```
+------------------------------------------------------+
| TICKER (grande)        Sector                        |  ENCABEZADO
|                        $precio (fecha del dato)      |
| [ CINTA VEREDICTO: ALCISTA / NEUTRAL / BAJISTA ]     |  (color segun estado)
+------------------------------------------------------+
| TECNICO -- Precio y muros de opciones                |  BLOQUE 1 (grande)
|                                                      |
|  [ grafico: linea de precio (~4 meses)               |
|    --- resistencia mas fuerte (linea roja + etiqueta)|
|    --- soporte mas fuerte     (linea verde + etiqueta)]
|                                                      |
+------------------------------------------------------+
| OPCIONES por plazo            PCR_vol      Sesgo OI  |  BLOQUE 2 (compacto)
|   Corto                       [chip+num]   [chip]    |
|   Medio                       [chip+num]   [chip]    |
|   Largo                       [chip+num]   [chip]    |
+------------------------------------------------------+
| VALUACION -- PER: TICKER vs su sector (~5 trimestres)|  BLOQUE 3
|                                                      |
|  [ linea PER del ticker sobre barras PER mediano del |
|    sector, MISMO eje, por trimestre calendario ]     |
+------------------------------------------------------+
| footer: handle . fecha de generacion .               |
|         "descriptivo, no recomendacion"              |
+------------------------------------------------------+
```

## 5. Detalle por bloque

### Bloque 1 -- Precio + Muros (el gancho visual)
- Linea de precio de ~4 meses (~85 ruedas): suficiente para ver estructura sin
  ruido. Configurable.
- Se marcan SOLO 2 niveles (para que quede limpio):
  - **Soporte mas fuerte**: la fila con mayor `soporte_fuerza` entre las 3
    ventanas (strike por DEBAJO del precio). Linea verde horizontal.
  - **Resistencia mas fuerte**: mayor `resistencia_fuerza` entre las 3 ventanas
    (strike por ARRIBA). Linea roja horizontal.
  - Son INDEPENDIENTES: el soporte y la resistencia pueden venir de ventanas
    distintas. La etiqueta lo aclara: `Sop 200 (f64%, corto)`.
- Si el ticker no tiene opciones liquidas (sin muro valido) -> se dibuja solo el
  precio y se omite la linea faltante (no se inventa un nivel).

### Bloque 2 -- Opciones por plazo (solo direccion)
- 3 filas (Corto / Medio / Largo) x 2 senales: **PCR_vol** y **Sesgo OI**.
- Coloreo por direccion, 3 ESTADOS (no binario -- el PCR_vol tiene zona neutra
  real y forzarlo a alcista/bajista mentiria):
  - PCR_vol: <0.7 ALCISTA (verde) / 0.7-1.0 NEUTRAL (gris) / >1.0 BAJISTA (rojo).
  - Sesgo OI: veredicto_oi A=ALCISTA (verde) / B=BAJISTA (rojo) / sin dato (gris).
- Bajo el chip de PCR_vol se muestra el numero chico (ej. `0.65`): da sustancia
  sin recargar. El sesgo OI va solo como chip (es una lectura de posicionamiento).
- Si no hay opciones -> "Sin datos de opciones" y el bloque se colapsa.

### Bloque 3 -- Valuacion: PER del TICKER vs PER del SECTOR
CAMBIO respecto del plan original (era PER linea + ROE barras). Motivo: `roe_ttm`
es TTM y en TODO el universo tiene solo ~2 trimestres de dato (promedio 2.0; 182
de 196 tickers exactamente 2) -> una "trayectoria de ROE" es imposible (siempre 2
barras). Se pivoteo a **PER del ticker (linea) vs PER mediano de su sector
(barras)**, ambos de `pe_ratio` fiscal, que tiene ~4-5 trimestres de profundidad.

- **UN SOLO EJE**: los dos son PER (misma escala, directamente comparables). NO es
  doble eje (el doble eje es el error #1 de dataviz; aca no aplica porque es la
  misma metrica). Barras = benchmark del sector; linea = el ticker encima.
- **Linea = PER del ticker** (`pe_ratio` fiscal por trimestre). Etiqueta solo el
  ultimo punto (PER actual); evita colision con la etiqueta de la barra.
- **Barras = PER mediano del sector** (`percentile_cont(0.5)` de `pe_ratio` sobre
  los tickers ACTIVOS del mismo `activos.sector`). Etiqueta cada barra.
- **Trimestre CALENDARIO, no fiscal**: los `fiscal_period_end` NO estan alineados
  entre tickers (NVDA cierra ene/abr/jul/oct; la mayoria mar/jun/sep/dic). Se
  bucketea por `date_trunc('quarter', fiscal_period_end)` para que el sector en
  cada bucket tenga a todos los pares (no solo los que cierran esa fecha exacta).
  Eje X = "'YY Qn" (ultimos ~5-6 buckets con dato del ticker).
- Historia: "cotiza con PREMIO o DESCUENTO vs sus pares, a lo largo del tiempo"
  -- el diferencial del proyecto (comparativo sectorial). Ej: NVDA premium sobre
  Technology que se comprime; JPM en linea con Financial Services; VALE re-rateando.
- Base consistente: ticker y sector con el MISMO `pe_ratio` fiscal (no se mezcla
  con *_px). Los ratios son inmunes a la moneda de reporte (sirve para ADRs).
- Perfil: aplica a bancos y no-bancos (ambos tienen PER). Peer-set SIMPLE por
  `sector` (no region-aware como fundamentales_ticker_vs_sector, que es snapshot
  del ultimo Q; aca hace falta la serie temporal). Simplificacion aceptada para v1.
- Sin datos: si el ticker no tiene serie de PER utilizable -> "serie de valuacion
  insuficiente".

## 6. Decisiones de diseno tomadas (21/6/2026)

- **Una sola tarjeta combinada** (tecnico + fundamental), 3 bloques. [DECIDIDO]
  Se prioriza el "de un vistazo" por sobre la separacion en 2 imagenes.
- **Veredicto en el encabezado como cinta de color**. [DECIDIDO -- REVERSIBLE]
  Es el gancho social mas fuerte y el "cerebro" ya existe (sintetizar). Rotulado
  como descriptivo, no recomendacion. Si se prefiere puro-datos, se saca sin
  afectar el resto.
- **Muros: solo 1 soporte + 1 resistencia (los de mayor fuerza)**, independientes,
  con etiqueta de ventana. [DECIDIDO] Mostrar los 6 recargaria el grafico.
- **Opciones en 3 estados** (verde/gris/rojo), no binario. Numero de PCR_vol chico
  bajo el chip. [DECIDIDO]
- **Bloque fundamental = PER del ticker (linea) vs PER mediano del sector (barras)**,
  mismo eje. [DECIDIDO 21/6 -- pivoteo desde PER/ROE por falta de profundidad de
  roe_ttm; ver Bloque 3.] Bucketeado por trimestre calendario.
- **Graficos como SVG inline** (mismo patron que los sparklines), sin matplotlib.
- **Formato 4:5** (consistente con la fundamental) y **paleta crema/alto
  contraste** (consistente con las 3 infografias). Salida @2x.
- **Sin LLM**.

## 7. Motor y archivos (a crear en la etapa de implementacion)

| Archivo | Rol |
|---------|-----|
| scripts/reports/make_infografia_simple.py | Data-load LOCAL + builders SVG (precio+muros, combo PER/ROE, chips opciones) + `generar_infografia_simple(ticker)->Path` (API reutilizable, import diferido desde el dashboard) |
| scripts/reports/templates/infografia_simple.html | Estructura (encabezado + 3 bloques + footer), placeholders Jinja2 |
| scripts/reports/templates/infografia_simple.css | Estilo 4:5, paleta, tipografias |
| dashboard/app.py | Boton "Generar infografia simple (PNG)" -- ubicacion a definir (vista "Informe por ticker", que ya es tecnico; import diferido para no cargar weasyprint en el arranque) |

Reutiliza: el pipeline HTML->WeasyPrint->PyMuPDF de las otras dos infografias; el
cerebro dashboard_sintesis.sintetizar para el veredicto; opciones_plazo/muros v2
para los muros y el sesgo.

## 8. Casos borde (manejo explicito, sin inventar)

- Ticker sin opciones liquidas -> Bloque 1 sin lineas de muro; Bloque 2 colapsado.
- Ticker sin fundamentales / semestral -> Bloque 3 con lo disponible o leyenda.
- Precio con < 4 meses de historia (alta reciente) -> grafica lo que haya.
- Veredicto no computable (sin datos suficientes) -> cinta gris "sin veredicto".

## 9. Estado / proximos pasos

1. [HECHO 21/6/2026] Spec de diseno (este doc). Aprobado el layout combinado 4:5
   + muro mas fuerte. Bloque 3 pivoteado en implementacion a PER-vs-sector.
2. [HECHO 21/6/2026] Implementacion: scripts/reports/make_infografia_simple.py +
   templates/infografia_simple.{html,css} + boton "Infografia simple (PNG)" en el
   sidebar de la vista "Informe por ticker" (import diferido). SVG inline. Validado
   render con NVDA (tech), JPM (banco), VALE (ADR), BAC (banco con Q2 fresco). Todos
   1 pagina, coherentes. El graficado se ajusto mirando el PNG real (altura de
   graficos 262 para que entre en 4:5, labels sin colision).
3. [FUTURO] Logo; version horizontal 16:9; sumarla a scripts/reports/app.py;
   peer-set region-aware para el PER del sector (hoy es por sector, simple).
