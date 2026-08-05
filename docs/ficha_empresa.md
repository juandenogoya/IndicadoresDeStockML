# Ficha de empresa -- tarjeta "presentacion" (fondo oscuro)

> CUARTA infografia del proyecto (5/8/2026). A diferencia de las otras 3
> (tecnica, fundamental, simple), esta es una **presentacion de la empresa contra
> si misma**: los datos PROPIOS del ultimo trimestre reportado y su variacion
> INTERANUAL (vs el mismo trimestre del ano anterior). SIN pares / sin benchmark
> sectorial (decision del usuario). Fondo OSCURO (navy + dorado + crema) para
> diferenciarla de la familia crema.

Motor: `scripts/reports/make_ficha_empresa.py` (+ `templates/ficha_empresa.{html,css}`).
Reusa el pipeline HTML/CSS -> WeasyPrint -> PNG (PyMuPDF @2x) de las otras
infografias. Sin LLM, sin estimaciones. API reutilizable:
`generar_ficha_empresa(ticker) -> Path` (import diferido desde el dashboard).

## 1. Filosofia

DESCRIPTIVA, no predictiva. Retrato del desempeno del ultimo Q reportado y como
viene evolucionando contra su propio ano anterior. NO compara con pares (eso lo
hace el dashboard financiero via fundamentales_ticker_vs_sector); aca el foco es
la empresa sola, para no depender de buckets de pares (que en regiones chicas --
ej. MELI en "Resto" -- quedan sin muestra para los multiplos).

## 2. Fuentes (LOCAL, Plan C)

| Bloque | Tabla | Campos |
|--------|-------|--------|
| Absoluto trimestral (ingresos, resultado neto) | `fundamentales_income_q` | total_revenue, net_income |
| YoY %, margenes, ROE/ROIC (o ROTCE/efficiency), FCF, solvencia, valuacion _px, perfil, moneda | `fundamentales_ratios_q` | revenue_yoy_pct, net_income_yoy_pct, eps_yoy_pct, gross/operating/net_margin_ttm, roe/roa/roic_ttm, rotce_ttm, efficiency_ratio_ttm, fcf_ttm, net_debt, current_ratio, debt_to_equity, pe/ps/ev_ebitda/pb *_px, profile, reporting_currency |
| Nombre de la empresa | `activos` | nombre |

## 3. Trimestre ANCLA (regla clave, corrige un bug general)

El trimestre MAS reciente suele ser un **"stub"**: yahooquery trae el EPS del Q
recien reportado antes que el income statement completo (ej. JPM Q2'26 con EPS
pero `total_revenue`/`net_income` NULL). Tomar "el ultimo Q a secas" mostraria una
ficha vacia. **Se ancla al ultimo Q con `total_revenue` NO NULL** (el ultimo
efectivamente reportado). La VALUACION en cambio usa la fila MAS reciente (su
`_px` trae el precio al cierre de hoy, que puede diferir del ancla). Afecta a
CUALQUIER ticker recien reportado, no solo bancos.

## 4. Secciones (adaptadas por PERFIL)

El perfil (`profile`: financiero / no_financiero) viene de la capa v2 -- ver
`docs/fundamentales_calculo.md`. La ficha adapta 2 secciones:

1. **Resultado del trimestre (variacion interanual)** -- igual en ambos perfiles:
   Ingresos, Resultado neto, BPA (Q), cada uno con su delta YoY (verde/rojo).
   (Se usa "interanual", no "vs ano anterior", para esquivar la n-tilde en ASCII.)
2. **Margenes y retornos (TTM)** -- POR PERFIL:
   - no-financiero: Margen bruto / operativo / neto (con delta pp YoY) + ROE + ROIC.
   - financiero (banco): ROE + **ROTCE** + **Efficiency ratio** (los margenes
     industriales/ROIC vienen NULL por perfil; se omiten).
3. **Caja y solvencia (TTM)** -- solo tiles con dato:
   - FCF (+ margen y YoY) -> NULL en bancos, se colapsa.
   - **Deuda neta -> SOLO no-financieros** (TotalDebt-Cash no es lectura de
     solvencia valida en un banco; su "deuda" incluye fondeo/depositos).
   - Current ratio (NULL en bancos), Deuda/Patrimonio (aplica a ambos).
4. **Valuacion (al cierre)** -- tira de contexto (PER, P/Ventas, EV/EBITDA, P/VL).
   Usa `_px` (al cierre de hoy); si el ticker no es USD, cae al multiplo fiscal.
   EV/EBITDA queda "-" en bancos (no tienen EBITDA). PROVISIONAL: a revisar si se
   sostiene o se saca.

## 5. Formato y moneda

- 4:5 (1080x1350 @2x). Fondo navy, ticker/acentos dorados, numeros crema, deltas
  verde (sube) / rojo (baja). Grilla de tiles (cols-3/4/5).
- Absolutos en la MONEDA DE REPORTE de la empresa (MELI USD; ADRs en su moneda,
  rotulado "Cifras en {moneda}"). Al ser 1 empresa, los absolutos son validos
  (no se comparan cross-ticker).
- Nombre: se oculta si `activos.nombre` == ticker (evita "MELI / MELI").

## 6. Casos borde

- Ticker sin fundamentales -> error claro (sugiere refresh).
- Ultimo Q stub -> se ancla al Q previo poblado (seccion 3).
- Bank sin FCF/net_debt/current -> esas tiles se colapsan; queda Deuda/Patrimonio.
- ADR no-USD -> valuacion _px NULL -> multiplo fiscal; absolutos en moneda local.

## 7. Estado / pendientes

- [HECHO 5/8/2026] Motor + templates + validado MELI (no-banco) y JPM (banco).
- [PENDIENTE] Boton en el dashboard (vista "Informe por ticker", import diferido).
- [A DEFINIR] Sostener o no la tira de Valuacion.
- Depende de la clasificacion de perfil v2: los 18 financieros estan FIJADOS en
  `FINANCIERO_OVERRIDE` (la regla auto es fragil con retencion de 8Q). Ver
  `docs/fundamentales_calculo.md`.
