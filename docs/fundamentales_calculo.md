# Fundamentales: datos crudos, perfiles y calculo de ratios
# Borrador v1 -- 2026-06-01 (PENDIENTE de curaduria del usuario, ver seccion 4)

> ## Multiplos al CIERRE del dia (*_px) -- 2/6/2026
> Los multiplos PER/P-B/P-S/EV-EBITDA que trae yahooquery (fundamentales_valuation_q)
> congelan el PRECIO en la fecha del balance -> quedan desactualizados todo el
> trimestre (ej. CVS: PER Yahoo 51.7 vs 39.9 al precio de hoy). El denominador TTM
> si se mantiene el trimestre (correcto), pero el numerador (precio) cambia a diario.
>
> SOLUCION: columnas `pe_ratio_px/pb_ratio_px/ps_ratio_px/ev_ebitda_px` (+ `precio_px`,
> `fecha_px`, `shares_out`) en fundamentales_ratios_q, recalculadas a DIARIO con el
> cierre de precios_diarios y los denominadores TTM del ultimo Q:
>   PER=P/eps_ttm | P/B=P/bvps | P/S=P*shares/revenue_ttm | EV/EBITDA=(P*shares+net_debt)/ebitda_ttm
> Logica pura: src/utils/multiplos_px.py. Recompute: scripts/compute_multiplos_px.py
> (DB->local, sin Yahoo), enganchado al final de recovery_incremental (target local),
> junto con el comparativo sectorial de valuacion (compute_sector_valuacion_px, pisa
> las 4 filas de valuacion en fundamentales_ticker_vs_sector con value+mediana al cierre).
> Validado vs TradingView (CVS PER 39.9 vs 39.79; P/B 1.5 vs 1.49; P/S 0.28 vs 0.28;
> EV/EBITDA 11.2 vs 10.21). Los pe_ratio/etc. de Yahoo NO se pisan (foto del Q).
> Consumidores (infografia fundamental + dashboard financiero) leen *_px; si NULL
> (eps<=0/sin dato, ~30 tickers en PER) muestran "-" (sin fallback al multiplo desfasado
> de Yahoo). El MCP no usa multiplos.
>
> EBITDA: ebitda_ttm usa **NormalizedEBITDA** (excluye cargos extraordinarios one-off),
> con fallback al EBITDA reportado. Motivo: el reportado puede ser negativo en un Q con
> impairment (CVS Q3'25 EBITDA -1.56B) y deformar el TTM -> EV/EBITDA inflado. El
> normalizado coincide con TV y refleja capacidad operativa recurrente. Afecta tambien
> net_debt_to_ebitda_ttm. (compute_fundamentales_ratios.py, clave NormalizedEBITDA del
> raw_json income; recomputable sin re-fetch.)

> Documento de diseno PREVIO a codificar la v2 del calculo de ratios. Sigue la
> regla del proyecto: documentar antes de implementar. Captura conocimiento que
> NO se deriva leyendo el codigo: que claves crudas expone yahooquery, como se
> comportan distinto bancos vs no-bancos, y que formula usa cada ratio segun el
> perfil de la empresa.

## 1. Proposito y alcance

La capa `fundamentales_ratios_q` (v1, ya en produccion) calcula ratios con UNA
sola formula para todos los tickers. La validacion contra balances oficiales
(seccion 2) mostro que esa formula unica falla en empresas FINANCIERAS porque
yahooquery arma su income statement con otra estructura. Esta v2:

1. Calcula los ratios desde los datos MAS CRUDOS posibles (no delegar a Yahoo
   salvo donde no agregue error -- ver seccion 5).
2. Aplica DOS perfiles de calculo (no-financiero / financiero) porque tienen
   dinamicas contables distintas (Opcion 2, decidida 2026-06-01).
3. Mantiene NULL honesto donde un ratio no aplica, en vez de inventar un numero.

Universo: 195 tickers con fundamentales (4 sin Q: HMY/RIO/UL/VOD semestrales).

## 2. Validacion realizada (evidencia: los crudos son correctos)

Se cruzaron los datos crudos de yahooquery contra el balance/income OFICIAL
(press releases / 10-Q) de 3 empresas:

| Ticker | Tipo | Moneda | Resultado crudos | Resultado ratios v1 |
|--------|------|--------|------------------|---------------------|
| MU  | no-banco | USD | EXACTO al millon (revenue, NI, EPS, assets, equity) | OK |
| XP  | broker/banco | BRL | EXACTO (NI R$1,318M) | margenes basura, ROE NULL |
| JPM | banco | USD | EXACTO (NI 16,494M, assets 4,900,475M, equity 364,038M) | margenes NULL, ROE/BVPS bajos |

Conclusion: **la captura de datos crudos es fidedigna** (incluso multi-moneda).
El problema esta 100% en la CAPA DE CALCULO de ratios para financieras.

Bugs detectados en v1 (a corregir en v2):
- **B1 -- Margenes de bancos sin sentido**: Yahoo no da GrossProfit/OperatingIncome
  para bancos; cuando los da, son negativos/incoherentes. Los margenes que
  dividen por estas lineas dan basura (XP op margin -55%, oficial +30%).
- **B2 -- ROE/ROA NULL por desfase de fecha**: el ratio se computa sobre el
  ultimo Q de income, pero si el balance de ese Q aun no salio (caso XP: income
  marzo, balance diciembre) -> equity NULL -> ROE/ROA/ROIC NULL.
- **B3 -- BVPS/ROE de bancos subvaluados por acciones preferentes**: usamos
  StockholdersEquity (TOTAL, incluye preferentes) / acciones totales. JPM
  oficial usa COMMON equity / common shares. JPM BVPS: nuestro 88.68 vs
  oficial 128.38 (-31%). El ROE bajo (16.2% vs ~19%) tiene la misma raiz.

## 3. Inventario de datos crudos (yahooquery)

Cada fila de las 4 tablas raw guarda ~50-80 campos en `raw_json` (JSONB). Solo
~15 estan en columnas dedicadas; el resto vive en el JSON y es accesible sin
re-fetch. Lo relevante para el calculo:

### 3.1 Income statement -- claves COMUNES (todos los perfiles)
```
TotalRevenue, OperatingRevenue, NetIncome, NetIncomeCommonStockholders,
PretaxIncome, TaxProvision, TaxRateForCalcs, DilutedEPS, BasicEPS,
DilutedAverageShares, BasicAverageShares, InterestExpense, InterestIncome,
NetInterestIncome, ReconciledDepreciation, NormalizedIncome
```

### 3.2 Income -- claves SOLO no-financiero (estructura industrial)
```
CostOfRevenue, GrossProfit, OperatingExpense, OperatingIncome, EBIT, EBITDA,
NormalizedEBITDA, ResearchAndDevelopment, SellingGeneralAndAdministration,
TotalExpenses, TotalOperatingIncomeAsReported
```
> Su AUSENCIA es la huella de un banco. Un banco no tiene "costo de la
> mercaderia vendida" ni "resultado operativo" en el sentido industrial.

### 3.3 Income -- claves SOLO financiero (estructura bancaria/seguros)
```
PreferredStockDividends, OtherunderPreferredStockDividend, SalariesAndWages,
InsuranceAndClaims (seguros), GeneralAndAdministrativeExpense,
SellingAndMarketingExpense
```

### 3.4 Balance sheet -- claves clave para ratios
COMUNES:
```
TotalAssets, StockholdersEquity, CommonStockEquity, TangibleBookValue,
TotalEquityGrossMinorityInterest, TotalLiabilitiesNetMinorityInterest,
TotalDebt, LongTermDebt, CashAndCashEquivalents, RetainedEarnings,
ShareIssued, OrdinarySharesNumber, InvestedCapital, NetTangibleAssets
```
SOLO no-financiero (capital de trabajo):
```
CurrentAssets, CurrentLiabilities, Inventory, AccountsPayable, WorkingCapital,
NetPPE, GrossPPE
```
SOLO financiero:
```
PreferredStockEquity, PreferredSharesNumber, PreferredStock,
HeldToMaturitySecurities, TradingSecurities
```
> CLAVE PARA B3: `CommonStockEquity` (equity SIN preferentes) existe en bancos
> y no-bancos. Usar ESTE, no `StockholdersEquity`, para BVPS y ROE common.

### 3.5 Cash flow -- claves clave
```
OperatingCashFlow, FreeCashFlow, CapitalExpenditure, FinancingCashFlow,
InvestingCashFlow, CashDividendsPaid, RepurchaseOfCapitalStock, NetIncome,
DepreciationAndAmortization, ChangesInCash
```
> Bancos suelen NO traer CapitalExpenditure/FreeCashFlow (FCF no es metrica
> central de un banco) -> NULL honesto.

## 4. Perfiles de empresa (clasificacion + curaduria)

### 4.0 Principio (decision del usuario, 2026-06-01)
**La verdad del modelo de negocio esta en las CUENTAS CONTABLES que la empresa
presenta, NO en la etiqueta de sector.** El sector nominal (GICS/Yahoo) es una
pista, no la fuente de verdad: una empresa puede pertenecer a "Financial
Services" pero llevar contabilidad industrial (V, MA), y la actividad real de
una empresa puede mutar/adaptarse con el tiempo. Por eso clasificamos por la
ESTRUCTURA CONTABLE SOSTENIDA (a lo largo de todos los Q), no por el sector ni
por un solo trimestre.

### 4.1 Regla automatica robusta (multi-Q, NO sector nominal, NO un solo Q)
Mirar UN solo Q falla por huecos de datos de Yahoo (un industrial puede no
traer GrossProfit en un Q puntual -> falso positivo). La regla mira la fraccion
de Q en que aparece cada marcador sobre TODA la historia del ticker:

```
gross_ratio = #Q con GrossProfit / #Q totales
opinc_ratio = #Q con OperatingIncome / #Q totales
nii_ratio   = #Q con NetInterestIncome / #Q totales

es_financiero_auto = (gross_ratio < 0.3) AND (opinc_ratio < 0.3) AND (nii_ratio > 0.7)
```
Interpretacion: "casi nunca tuvo estructura industrial Y casi siempre tuvo
ingreso neto por intereses" = banco/aseguradora estructural. Validado: los 17
financieros dan gross 0/N y opinc 0/N en TODOS sus Q (estructura inequivoca);
los falsos positivos (WMT/TGT/PATH/SNOW/AAP) desaparecen al mirar multi-Q.

### 4.2 Override manual (para hibridos -- la regla auto NO alcanza)
Algunas empresas son hibridas: presentan estructura industrial (gross/opinc) Y
ademas marcadores financieros (salaries, NII) en todos los Q. La regla auto NO
las captura. Se resuelven con una LISTA DE OVERRIDE curada por el usuario.

Caso confirmado: **XP** (broker brasileno). gross 35/49, opinc 35/49, NII 35/49,
salaries 35/49 -> la regla auto lo daria no-financiero, PERO su operating income
de Yahoo es incoherente (Q reciente -R$1,314M con NI +R$1,318M; op margin -55%
vs +30% oficial). Decision del usuario (opcion A): tratarlo como FINANCIERO
pleno -> sus margenes industriales se ignoran (NULL), se le aplican las reglas
de banco. Razon: el dato industrial que Yahoo le arma es basura; mejor NULL
honesto que un margen roto.

> Mecanismo: la clasificacion final = regla auto (4.1) UNION override manual.
> Persistir el perfil resultante en una columna `profile` (auditable, editable).
> El override vive como lista explicita en el codigo del compute (no hardcode
> disperso), documentada aca.

> ACTUALIZACION 5/8/2026 -- los 18 financieros se FIJAN en `FINANCIERO_OVERRIDE`:
> la regla auto (4.1) fue calibrada con ~49 Q de historia (NII 35/49 = 0.71). Con
> la retencion actual de **8 trimestres**, el `nii_ratio` de varios bancos cae bajo
> el umbral 0.70 (JPM/C/GS/CB dan 5/8 = 0.62, porque los Q recientes -- incluido el
> "stub" recien reportado -- aun no traen NetInterestIncome poblado) -> la regla
> los tiraba a `no_financiero`. Como la curaduria de 4.3 YA es la fuente de verdad,
> se pinean los 18 explicitamente en el override; asi el perfil no depende de la
> ventana de datos. La regla auto queda como backstop para tickers nuevos. Decision
> del usuario: identificar los casos particulares, no ajustar un umbral por default.

> PURGA de huerfanas (5/8/2026): el compute ahora, tras el UPSERT, BORRA las filas
> de `fundamentales_ratios_q` cuyo (ticker, fiscal_period_end) ya no tiene respaldo
> en `fundamentales_income_q` (trimestres que envejecieron fuera de la ventana
> lookback_q y quedaban huerfanos -- p.ej. 145 filas NULL-profile remanentes de la
> migracion v1->v2, que nunca se reescribian). La tabla es DERIVADA/recomputable y
> un Q sin income crudo no se puede recalcular ni lo consume nadie (todo usa el
> ultimo Q). Mantiene ratios_q en sync con la ventana y evita reacumulacion.
> `_purgar_huerfanas` esta SCOPEADA a los tickers procesados -> corridas con
> `--tickers X` no tocan el resto del universo.

### 4.3 Curaduria CONFIRMADA (usuario, 2026-06-01)
Evidencia multi-Q (fraccion de Q con cada marcador). Perfil final = decision.

**FINANCIERO (17, por regla auto -- gross 0/N y opinc 0/N en todos los Q):**

| Ticker | Sector nominal | gross | opinc | NII | Tipo |
|--------|---------------|-------|-------|-----|------|
| JPM  | Financial Services | 0/49 | 0/49 | 35/49 | banco |
| BAC  | Financial Services | 0/42 | 0/42 | 35/42 | banco |
| C    | Financial Services | 0/49 | 0/49 | 35/49 | banco |
| WFC  | Financial Services | 0/42 | 0/42 | 35/42 | banco |
| GS   | Financial Services | 0/49 | 0/49 | 35/49 | banco inversion |
| MS   | Financial Services | 0/42 | 0/42 | 35/42 | banco inversion |
| AXP  | Financial Services | 0/49 | 0/49 | 35/49 | tarjetas/credito |
| SCHW | Financial Services | 0/35 | 0/35 | 35/35 | broker |
| AIG  | Financial Services | 0/42 | 0/42 | 35/42 | seguros |
| CB   | Financial Services | 0/49 | 0/49 | 35/49 | seguros |
| PGR  | Financial Services | 0/49 | 0/49 | 35/49 | seguros |
| LNC  | Financial Services | 0/49 | 0/49 | 35/49 | seguros |
| NU   | Financial Services | 0/42 | 0/42 | 35/42 | banco digital (BRL) |
| UPST | Financial Services | 0/49 | 0/49 | 35/49 | fintech credito |
| ITUB | Financial Services | 0/42 | 0/42 | 35/42 | banco (BRL) |
| BBD  | Financial Services | 0/49 | 0/49 | 35/49 | banco (BRL) |
| BSBR | Financial Services | 0/36 | 0/36 | 30/36 | banco (BRL) |

**FINANCIERO por OVERRIDE manual (1):**

| Ticker | Sector nominal | gross | opinc | NII | Razon |
|--------|---------------|-------|-------|-----|-------|
| XP   | Financial Services | 35/49 | 35/49 | 35/49 | hibrido; op income Yahoo basura -> opcion A (financiero pleno) |

**NO-FINANCIERO aunque el sector diga "Financial Services" (7):**
Tienen gross+opinc en 35/35 Q -> contabilidad industrial real. El usuario
confirma: pertenecen al sector financiero por etiqueta, pero su actividad
(procesar pagos / rating / gestion / infra fintech) lleva indicadores propios
de la industria, no del sector financiero.

| Ticker | gross | opinc | Actividad real |
|--------|-------|-------|----------------|
| V    | 35/35 | 35/35 | procesadora de pagos |
| MA   | 42/42 | 42/42 | procesadora de pagos |
| MCO  | 49/49 | 49/49 | rating / data |
| SPGI | 35/35 | 35/35 | rating / data |
| BLK  | 30/30 | 30/30 | asset manager |
| PYPL | 35/35 | 35/35 | pagos |
| FISV | 35/49 | 35/49 | infraestructura fintech |

**Falsos positivos descartados (5):** AAP, TGT, WMT, PATH, SNOW. Al mirar
multi-Q tienen estructura industrial clara; la ausencia de gross/opinc era un
hueco de datos en un Q puntual. -> NO-FINANCIERO.

Resumen final: **18 FINANCIERO** (17 auto + XP override) **/ 177 NO-FINANCIERO**.

## 5. Ratios por perfil (formulas)

Politica: TTM = suma de 4 Q para flujos (income, cashflow); el ultimo valor
para stocks (balance). Crecimiento QoQ vs Q-1, YoY vs Q-4. Todo division-safe
(denominador 0 o NULL -> NULL).

### 5.1 Que se calcula NOSOTROS vs que viene de Yahoo
- **Calculamos nosotros** (desde crudos): margenes, ROE, ROA, ROIC, BVPS, EPS
  TTM, crecimientos, FCF, solvencia, deuda neta.
- **De Yahoo (`valuation_measures`)**: PER, P/B, P/S, EV/EBITDA. Validados OK en
  MU/XP/JPM. Calcularlos a mano exige el precio del dia EXACTO de cada Q
  (mas friccion, mas riesgo de discrepar). Se MANTIENEN de Yahoo en v2; migrar
  a calculo propio queda como mejora futura (requiere serie de precios por Q).

### 5.2 Perfil NO-FINANCIERO (formulas)
| Ratio | Formula | Crudos |
|-------|---------|--------|
| Margen bruto TTM | GrossProfit_ttm / Revenue_ttm | income |
| Margen operativo TTM | OperatingIncome_ttm / Revenue_ttm | income |
| Margen neto TTM | NetIncome_ttm / Revenue_ttm | income |
| Margen FCF TTM | FreeCashFlow_ttm / Revenue_ttm | cashflow |
| ROE TTM | NetIncomeCommonStockholders_ttm / CommonStockEquity | income+balance |
| ROA TTM | NetIncome_ttm / TotalAssets | income+balance |
| ROIC TTM | NOPAT_ttm / (TotalDebt + CommonStockEquity - Cash) | income+balance |
| | NOPAT = EBIT_ttm * (1 - tasa); tasa = clip(Tax_ttm/Pretax_ttm,0,1) | |
| BVPS | CommonStockEquity / OrdinarySharesNumber | balance |
| EPS TTM | DilutedEPS sumado 4 Q | income |
| Liquidez corriente | CurrentAssets / CurrentLiabilities | balance |
| Working capital | CurrentAssets - CurrentLiabilities | balance |
| Deuda/Patrimonio | TotalDebt / CommonStockEquity | balance |
| Deuda neta | TotalDebt - Cash | balance |
| Crecimiento rev/NI/EPS/FCF | QoQ y YoY sobre cada serie | varios |

> CAMBIO clave v2 vs v1: usar `CommonStockEquity` (no `StockholdersEquity`) y
> `NetIncomeCommonStockholders` (no `NetIncome`) en ROE/BVPS -> corrige B3.

### 5.3 Perfil FINANCIERO (banca / seguros / brokers)
NO aplican (se dejan NULL con leyenda):
```
margen bruto, margen operativo, ROIC, liquidez corriente, working capital,
margen FCF (FCF no es metrica central de un banco)
```
SI aplican:
| Ratio | Formula | Nota |
|-------|---------|------|
| ROE TTM | (NetIncome_ttm - PreferredDividends_ttm) / CommonStockEquity | corrige B3 |
| ROA TTM | NetIncome_ttm / TotalAssets | bancos: ROA tipico ~1% (normal) |
| Margen neto TTM | NetIncome_ttm / TotalRevenue | con caveat (Yahoo "revenue" de banco no = revenue oficial) |
| BVPS | CommonStockEquity / OrdinarySharesNumber | |
| EPS TTM | DilutedEPS 4 Q | |
| Crecimiento NI/EPS | QoQ, YoY | el crecimiento de revenue de banco es ruidoso |
| Deuda/Patrimonio | TotalDebt / CommonStockEquity | en bancos es alto por naturaleza |
| PER, P/B, P/S | de Yahoo | validado OK en JPM (PER 14.7, P/B 2.3) |

Set bancario propio (INCLUIDO en v2, decision usuario 2026-06-01):
Disponibilidad verificada sobre los 18 financieros (raw_json):
- NetInterestIncome 18/18, TangibleBookValue 18/18, NetIncomeCommonStockholders
  18/18, TotalRevenue 18/18, PretaxIncome 18/18, TotalAssets 18/18 -> OK.
- NetLoansReceivable / GrossLoan / TotalDeposits = 0/18 -> Yahoo NO expone la
  cartera de prestamos ni depositos. Por eso el NIM "de libro" no es calculable.

| Ratio bancario | Formula | Estado |
|----------------|---------|--------|
| **ROTCE TTM** | NetIncomeCommonStockholders_ttm / TangibleBookValue | OK (solido). Preview JPM 20.6% (oficial ROTCE 23%, dif. por TTM vs Q anualizado) |
| **Efficiency ratio TTM** | (TotalRevenue_ttm - PretaxIncome_ttm) / TotalRevenue_ttm | APROXIMADO (proxy de gastos no-interes / ingresos). Preview JPM 60%, BAC 65% (rango bancario real); aseguradoras altas (AIG 85%, PGR 84%) por estructura distinta. Leyenda: "aproximado". |

DESCARTADO -- NIM (net interest margin):
NetInterestIncome / activos productivos. Sin loans+securities (0/18) el unico
denominador posible es TotalAssets, que da basura (preview: XP -0.1%, AIG -0.3%,
PGR -0.2%). Meter un NIM mal calculado viola la filosofia (mejor NO tenerlo que
tenerlo falso). Si en el futuro se consigue la cartera (otra fuente), se agrega.

Pendiente v3: NIM real (requiere cartera de prestamos de otra fuente).

### 5.4 Fix B2 (desfase de fecha) -- aplica a AMBOS perfiles
Cuando el balance del Q exacto del income no existe, hacer **as-of join**: usar
el balance mas reciente con fecha <= fecha del income. El equity cambia poco
Q a Q, asi que ROE/ROA/BVPS se pueden computar en el Q mas nuevo en vez de NULL.

## 6. Resumen de decisiones (para la etapa de codigo)

1. Dos perfiles: NO-FINANCIERO / FINANCIERO. Clasificacion por regla auto
   multi-Q (4.1) UNION override manual curado (4.2). Curaduria cerrada (4.3):
   18 financieros / 177 no-financieros.
2. Usar `CommonStockEquity` y `NetIncomeCommonStockholders` (corrige B3).
3. As-of join de balance: balance mas reciente con fecha <= income (corrige B2).
4. Financieras: margenes industriales y ROIC -> NULL con leyenda (corrige B1).
   XP incluido (opcion A): sus margenes industriales se ignoran.
5. PER/PB/PS/EV-EBITDA: se mantienen de Yahoo (validados); migrar a propio = futuro.
6. Persistir el perfil en columna `profile` (auditable). Override como lista
   explicita en el compute, documentada en 4.2/4.3.
7. Set bancario propio INCLUIDO en v2: ROTCE (solido) + efficiency ratio
   (aproximado, con leyenda). NIM DESCARTADO (sin cartera de prestamos en
   yahooquery, 0/18; daria un numero falso). NIM real -> v3 (otra fuente).

## 7. Proximos pasos (orden acordado)
1. [HECHO] Este documento (inventario + formulas + regla de perfiles).
2. [HECHO] Curaduria: usuario confirmo la lista (4.3) -- 18 financieros, XP por
   override (opcion A), 7 no-financieros del sector financiero, 5 falsos
   positivos descartados (2026-06-01).
3. [HECHO] Revision de formulas (seccion 5) con el usuario (2026-06-01):
   ROE/ROIC con CommonStockEquity OK; financieras NULL en margenes ind./ROIC/
   liquidez/WC/FCF OK; set bancario INCLUIDO en v2 (ROTCE + efficiency; NIM
   descartado por falta de cartera). Disponibilidad de claves verificada en DB.
4. [PENDIENTE] Codificar v2 del compute: agregar columna `profile`, regla auto +
   override, formulas por perfil (5.2/5.3), as-of join (5.4), CommonStockEquity.
   Recompute + re-validar contra MU/XP/JPM (los 3 balances oficiales ya cruzados).
