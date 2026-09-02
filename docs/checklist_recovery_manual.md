# Checklist de Recovery Manual

Guia paso a paso para reaccionar cuando el cron automatico (Oracle / GH Actions)
falla y faltan datos. Actualizado 28/5/2026 al flujo **Plan C**.

## Plan C en una linea

- **LOCAL PostgreSQL = fuente de verdad** para OHLCV, indicadores, features, scanner,
  z-scores, ML y backtesting. Se mantiene desde Windows (pasos 1-3 abajo).
- **Railway = SOLO opciones_snapshot** (lo escribe el cron Oracle/GH; es la unica data
  irrecuperable post-mercado siguiente). De ahi se baja a local con un sync.
- El timeframe **semanal (1W) ya no se puebla** (pipeline deprecado): se calcula al
  vuelo. No hay nada que recuperar ahi.

---

## Conceptos basicos

| DB | Rol bajo Plan C | Como apunta el script |
|----|-----------------|----------------------|
| **Local** | Fuente de verdad (todo menos opciones) | `.env` con `DB_HOST/PORT/NAME/USER/PASSWORD` (sin `DATABASE_URL`) |
| **Railway** | Solo opciones_snapshot | scripts que cargan `.env.local` -> `DATABASE_URL`=Railway |

| Bat (scripts/manual/) | Target | Para que sirve |
|-----|--------|-----------------|
| `status_local.bat` | Local | Estado de la DB local (la que importa) |
| `status.bat` | Railway | Estado de Railway (sobre todo opciones) |
| `cron_paso1_precios_yq.bat` | Local | **Paso 1**: precios + futuros + indicadores + z-scores (yahooquery, incremental) |
| `cron_paso2_features.bat` | Local | **Paso 2**: features PA + Market Structure |
| `cron_paso3_scanner.bat` | Local | **Paso 3**: scanner ML + alertas + Telegram |
| `recovery_incremental.bat` | Local | Motor del Paso 1 directo (detecta pendientes via MAX(fecha) y baja solo lo faltante) |
| `sync_opciones_railway_to_local.bat` | Railway -> Local | Baja las 3 tablas de opciones a local (incremental) |
| `poblar_opciones_yq.bat` | Railway | Carga manual del snapshot de opciones US (yahooquery) |
| `recover_opciones_tickers.py` | Railway | Recovery quirurgico de opciones de tickers puntuales |
| `sync_local.bat` | Railway -> Local | Sync completo Railway -> local (todas las tablas) |
| `ft_run_diario.bat` | Local | **Paso 5**: deriva opciones ([0b]) + 10 bots FT + equity + reporte HTML + push senales + veredictos |
| `chequeo_rutina.py` | Local | Diagnostico: que tablas quedaron atras y que .bat las arregla. Sale != 0 si hay mezcla de ruedas |
| `sync_to_railway.bat` | Local -> Railway | Subir local -> Railway (raro bajo Plan C) |

---

## Flujo diario normal (no es recovery)

Post-cierre NYSE (despues de 21:00 UTC), desde Windows, en orden:

```
1. sync_opciones_railway_to_local.bat  (baja el CRUDO de opciones desde Railway)
2. cron_paso1_precios_yq.bat   (precios + futuros + indicadores + z-scores, LOCAL)
3. cron_paso2_features.bat     (features PA + Market Structure, LOCAL)
4. cron_paso3_scanner.bat      (scanner ML + alertas + Telegram, LOCAL)
5. ft_run_diario.bat           (deriva opciones + 10 bots FT + equity + reportes)
```

**El paso 5 no es opcional y no es "solo los bots"**: su paso [0b] corre
`compute_opciones_derivadas.py`, que es lo UNICO que computa en local las 5
tablas derivadas de opciones (resumen, zscore, sector_zscore, pcr_plazo,
sector_pcr_plazo). Sin el, el crudo baja pero nadie lo procesa: el 2/9/2026 el
sistema estuvo cruzando tecnico del 1/9 con opciones del 31/8 durante todo el
dia, sin una sola queja, y 9 tickers tenian el veredicto equivocado.

El paso 1 va primero por costumbre, pero da igual: las opciones NO alimentan
precios, features ni el scanner. Son dos ramas independientes que recien se
juntan en el paso 5. Lo que si conviene es correr el .bat de sync **por
separado** aunque `ft_run_diario` sincronice por dentro: el .bat encadena ademas
`retencion_opciones_railway.py`, que purga Railway dejando 10 dias y es lo que
evita que se llene (incidente del 20/7/2026). El sync interno de ft_run no lo hace.

### Guard de coherencia (2/9/2026)

`ft_run_diario.bat` corre `scripts/manual/chequeo_rutina.py` DESPUES del paso
[0b] y ANTES del primer bot. Si las tablas no estan alineadas entre si, ABORTA
sin operar y dice que .bat falta correr.

- NO frena por antiguedad: operar con el cierre anterior es la convencion del
  proyecto (73% de las operaciones FT, ver CLAUDE.md "FT asincronico").
- SI frena por MEZCLA: tablas con fechas distintas entre si.
- Forzar igual: `set FT_IGNORAR_FRESCURA=1` antes de correr el .bat.
- Standalone, para ver el estado sin correr nada:
  `python scripts/manual/chequeo_rutina.py`

La misma logica alimenta la banda de estado del dashboard (motor puro
compartido: `src/utils/estado_pipeline.py`).

Con esto los datos crudos quedan completos y frescos en local. El z-score de acciones
corre solo al final del Paso 1 (ya no es manual). El semanal (RSI/MACD/SMC 1W) se
calcula al vuelo en el dashboard y en mtf_context; no requiere paso.

---

## CASO A: faltan precios / features / scanner en LOCAL

**Sintoma**: `status_local.bat` muestra `Fecha max < ultimo dia habil`, o no llegaron
los mensajes de Telegram del scanner.

### Paso 1 -- Diagnostico
1. `status_local.bat` -> ver hasta que fecha tiene cada tabla.
2. Confirmar dia habil NYSE: `python scripts/manual/check_fecha.py` (si el dia faltante
   es feriado/weekend, no hay nada que hacer).

### Paso 2 -- Recovery de precios (Paso 1)
**Antes**: que no haya otro script yfinance/yahooquery corriendo (el rate limit es por
IP; el `yfinance_lock` aborta si detecta concurrencia).

```
cron_paso1_precios_yq.bat
```
Detecta los tickers con MAX(fecha) atrasada y baja SOLO los pendientes (no los 199
ciego). Reporta los que NO logro completar. Incluye z-scores de acciones al final.

### Paso 3 -- Verificar que quedo COMPLETO
```
status_local.bat
```
Buscar: `Fecha max = <esperada> [OK]` y la lista de "Tickers desactualizados" VACIA.
Si quedan pendientes: esperar ~30 min (rate limit) y re-correr el Paso 1 (es idempotente).

### Paso 4 -- Features y scanner
```
cron_paso2_features.bat     (features sobre los precios nuevos)
cron_paso3_scanner.bat      (alertas ML)
```

Bajo Plan C **no se sube a Railway** (Railway no es la verdad para esto).

---

## CASO B: falta el snapshot de Opciones US

**Sintoma**: `status.bat` (Railway) muestra `OPCIONES SNAPSHOT` con `!! FALTA` en
algun dia, o `sync_opciones_railway_to_local.bat` no trae la fecha esperada.

### Paso 1 -- Diagnostico
1. `status.bat` -> seccion `OPCIONES SNAPSHOT`, identificar fechas faltantes.
2. Verificar dia habil NYSE.

### Paso 2 -- Timing (critico)
El snapshot solo es valido **post-cierre NYSE** (20:00 UTC en adelante) y **antes de
que abra el mercado del dia siguiente** (Yahoo solo sirve la chain vigente; al abrir,
las strikes del cierre anterior dejan de estar disponibles -> irrecuperable).

### Paso 3 -- Recovery a Railway
```
poblar_opciones_yq.bat          (carga el snapshot a Railway via yahooquery; idempotente)
```
Para tickers puntuales que fallaron: `python scripts/manual/recover_opciones_tickers.py`.

### Paso 4 -- Bajar a local
```
sync_opciones_railway_to_local.bat
```

### Paso 5 -- Verificar
`status.bat` (la fecha aparece con OK) y `status_local.bat` (opciones bajadas).

---

## CASO C: falta el scan diario (alertas_scanner) en LOCAL

**Sintoma**: huecos en `ALERTAS SCANNER` de `status_local.bat`.

Las alertas dependen de: `precios_diarios` + `features_precio_accion` +
`features_market_structure` al dia.

1. `status_local.bat` -> verificar los 3 prerequisitos.
2. Si faltan precios/features: hacer CASO A primero (pasos 1-2 y features).
3. Con prerequisitos OK:
   ```
   cron_paso3_scanner.bat
   ```

---

## CASO D: dia perdido completo (precios + opciones + scanner)

```
1. status_local.bat                    -> huecos en local
2. cron_paso1_precios_yq.bat           -> precios + futuros + indicadores + z-scores
3. status_local.bat                    -> confirmar Paso 1 completo
4. cron_paso2_features.bat             -> features
5. cron_paso3_scanner.bat              -> scanner + alertas
6. poblar_opciones_yq.bat              -> opciones US a Railway (si falto)
7. sync_opciones_railway_to_local.bat  -> bajar opciones a local
8. status_local.bat + status.bat       -> verificacion final
```

---

## Reglas de oro

### A. NUNCA correr 2 cosas a la vez contra Yahoo
yahooquery y yfinance comparten provider; Yahoo rate-limita la IP completa, no por
proceso. El `src/utils/yfinance_lock.py` aborta si detecta otro script activo.

### B. NUNCA dar por valido un "OK" sin verificar
`status_local.bat` y su lista de "Tickers desactualizados" son la verdad. Si tiene
nombres, el recovery quedo incompleto aunque el script haya dicho OK.

### C. NUNCA correr scripts pre-mercado
NYSE abre 13:30 UTC. Bajar precios antes de la apertura devuelve el cierre anterior
pero algunos scripts lo etiquetan con la fecha de hoy -> datos corruptos. Correr
siempre **post-cierre (>=21:00 UTC)** o temprano al dia siguiente antes de la apertura.

### D. OHLCV es prerequisito de todo lo demas
features_* dependen de precios_diarios; scanner depende de features_*; z-scores
dependen de precios_diarios. Orden siempre:
**precios -> indicadores -> features -> scanner**. Opciones corre en paralelo.

### E. Opciones es lo unico irrecuperable
Si el snapshot del dia no se tomo antes de la apertura siguiente, se perdio. Por eso
hay 4 intentos de cron. El resto (precios/features/scanner) se recalcula cuando sea.

---

## Comandos de verificacion rapida

```
status_local.bat     (estado del local = la fuente de verdad)
status.bat           (estado de Railway = opciones)
python scripts/manual/check_fecha.py    (es dia habil NYSE?)
```

### Query ad-hoc al local (Python, .env cargado, sin DATABASE_URL)
```python
import os
from sqlalchemy import create_engine, text
url = (f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
       f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}")
e = create_engine(url)
print(e.connect().execute(text("SELECT MAX(fecha) FROM precios_diarios")).scalar())
```

### Procesos Python en Windows
```
taskkill /F /IM python.exe                 (matar)
Get-Process python | Select Id, StartTime  (listar, PowerShell)
```
