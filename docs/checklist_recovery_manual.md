# Checklist de Recovery Manual

> ⚠️ **PARCIALMENTE DESACTUALIZADO (pre Plan C / pre yahooquery).** Algunos scripts
> citados ya NO existen. Reemplazos (28/5/2026):
> - `cron_paso1_precios.bat` -> **`cron_paso1_precios_yq.bat`** (yahooquery, target local)
> - `recovery_pipeline_local.bat` -> 3 pasos sueltos: `cron_paso1_precios_yq.bat` +
>   `cron_paso2_features.bat` + `cron_paso3_scanner.bat`
> - `poblar_opciones.bat` -> **`poblar_opciones_yq.bat`**
> Bajo Plan C local es la fuente de verdad (no Railway). Reescritura completa = pendiente.

Guia paso a paso para reaccionar cuando algun cron automatico (Oracle o GH Actions)
falla y los datos no llegan a Railway / local.

---

## Conceptos basicos

| DB | Cuando se usa | Como apunta el script |
|----|--------------|----------------------|
| **Railway** | Fuente de verdad de produccion (cron Oracle escribe aqui) | `.env.local` con `DATABASE_URL` |
| **Local** | Backup / espejo para desarrollo y emergencias | `.env` con `DB_HOST/PORT/NAME/USER/PASSWORD` |

| Bat | Target | Para que sirve |
|-----|--------|-----------------|
| `status.bat` | Railway | Ver estado de Railway |
| `status_local.bat` | Local | Ver estado del local |
| `cron_paso1_precios.bat` | Local | Solo paso 1 manual |
| `cron_paso2_features.bat` | Local | Solo paso 2 manual |
| `cron_paso3_scanner.bat` | Local | Solo paso 3 manual |
| `recovery_pipeline_local.bat` | Local | Paso 1+2+3 secuencial en una corrida |
| `poblar_opciones.bat` | Railway (¡cuidado!) | Cargar opciones US manual |
| `sync_local.bat` | Railway -> Local | Bajar datos de Railway a local |
| `sync_to_railway.bat` | Local -> Railway | Subir datos de local a Railway |

---

## CASO A: El cron de Oracle del pipeline diario fallo

**Sintoma**: no recibis los mensajes de Telegram con "Paso 1/2/3 OK", o `status.bat`
muestra `Fecha max < ayer`.

### Paso 1: Diagnostico
1. Ejecutar `status.bat` (Railway): ver hasta que fecha tiene cada tabla
2. Confirmar dia habil: si el ultimo dia faltante es feriado o weekend, no hay nada que hacer

### Paso 2: Verificar Oracle
Si te animas a SSH:
```bash
ssh -i "C:/Users/juand/.ssh/ssh-key-2026-05-05.key" ubuntu@141.148.57.58
tail -100 /tmp/cron_pipeline_diario.log
crontab -l   # verificar que la entrada de 22:00 UTC sigue ahi
```

### Paso 3: Recovery local
**Antes de empezar**: asegurarse que `pipeline_automatico.bat` (Windows Task Scheduler)
NO este corriendo. Verificar en `taskschd.msc` y/o `taskkill /F /IM python.exe` si hace falta.

1. Ejecutar `scripts\manual\recovery_pipeline_local.bat`
2. Esperar ~95 min
3. **No correrlo durante market hours** (NYSE 13:30-20:00 UTC) si la IP esta calentada;
   esperar post-cierre

### Paso 4: Verificar que recovery local quedo COMPLETO
**Critico**: el recovery puede ser parcial sin que el script lo diga claro.

```bash
# verificacion manual
status_local.bat
```

Buscar en el output:
- `Total tickers : 199` (o 200 si esta el SE duplicado)
- `Fecha max : <fecha esperada> [OK]`
- **Lista de "Tickers desactualizados" debe estar VACIA** o solo tener nombres conocidos

Si hay tickers con fecha vieja:
1. Esperar 30 min para que se libere el rate limit
2. Volver a ejecutar `recovery_pipeline_local.bat`
3. yfinance hace upsert idempotente; los tickers ya OK no se tocan

Repetir hasta que la lista de desactualizados quede limpia.

### Paso 5: Subir a Railway
```bash
scripts\sync_to_railway.bat
```

Hace 4 sub-pasos: precios -> indicadores -> features -> scanner. Confirma cada uno con `s`.

### Paso 6: Verificacion final
```bash
status.bat            # Railway
status_local.bat      # Local
```

Ambas deberian mostrar las fechas al dia y los tickers desactualizados vacios.

---

## CASO B: El cron del snapshot de Opciones US fallo

**Sintoma**: `status.bat` muestra `OPCIONES SNAPSHOT` con `!! FALTA` en algun dia.

### Paso 1: Diagnostico
1. Ejecutar `status.bat` y mirar la seccion `OPCIONES SNAPSHOT`
2. Identificar las fechas faltantes
3. Verificar que NO sea feriado NYSE

### Paso 2: Verificar timing
**El snapshot solo funciona post-cierre NYSE** (20:00 UTC en adelante).
Si es antes del cierre, Yahoo Finance no tiene datos completos -> esperar.

### Paso 3: Recovery manual
**OJO con el target**: `poblar_opciones.bat` apunta al destino definido en `.env.local`,
que es Railway. Si quieres traerlo a local primero hay que cambiar de approach.

#### Opcion B1: cargar directo en Railway (recomendado)
```bash
scripts\manual\poblar_opciones.bat
```
1. Pide la fecha (YYYY-MM-DD)
2. Hace dry-run primero
3. Pide confirmacion antes de escribir

#### Opcion B2: cargar primero en local, despues subir
1. Modificar manualmente el target (o cambiar .env.local a apuntar local)
2. Correr `poblar_opciones.bat`
3. Subir con `sync_to_railway.bat`

### Paso 4: Verificar
```bash
status.bat
```
La fecha que cargaste deberia aparecer con `OK` y filas >= 10,000.

---

## CASO C: Falta el scan diario (alertas_scanner)

**Sintoma**: `status.bat` muestra solo algunas fechas en `ALERTAS SCANNER` (huecos).

### Requisito previo
Las alertas se calculan a partir de:
- `precios_diarios` actualizado
- `features_precio_accion` actualizado
- `features_market_structure` actualizado

### Recovery
1. Verificar que los 3 prerequisitos esten al dia (`status.bat`)
2. Si faltan: ir al CASO A primero
3. Si estan OK, correr:
   ```bash
   scripts\manual\cron_paso3_scanner.bat
   ```
4. Despues `sync_to_railway.bat` (solo paso 4: scanner)

---

## CASO D: Recovery completo de un dia perdido

Si te perdiste un dia entero (precios + opciones + scanner):

```
1. status.bat              -> identificar todos los huecos
2. recovery_pipeline_local.bat  -> precios+features+scanner local
3. status_local.bat        -> verificar que recovery quedo completo
4. poblar_opciones.bat     -> cargar el snapshot de opciones a Railway
5. sync_to_railway.bat     -> subir TODO a Railway
6. status.bat              -> verificacion final
```

Duracion total estimada: 2-3 horas.

---

## Reglas de oro

### A. NUNCA correr 2 cosas a la vez contra yfinance
- Si Windows Task Scheduler corre `pipeline_automatico.bat`, NO corras nada manual al mismo tiempo
- Yahoo rate-limita la IP completa, no por proceso

### B. NUNCA dar por valido un OK del script
Cuando `cron_diario.py` dice "199/199 tickers con datos", verificar igual con:
```bash
status_local.bat
```
La lista de "Tickers desactualizados" es la verdad. Si tiene nombres = recovery incompleto.

### C. NUNCA correr scripts pre-mercado
NYSE abre 13:30 UTC (10:30 ART). `fast_info` y `yf.download()` antes de la apertura
devuelven el cierre del dia anterior, pero los scripts lo etiquetan con la fecha de hoy
-> **datos corruptos**. Correr siempre **post-cierre (20:00 UTC en adelante)** o el dia
siguiente temprano antes de la pre-apertura.

### D. SIEMPRE verificar status_local DESPUES de un recovery
Antes de subir a Railway, asegurate que el local este bien. Si subes datos parciales,
mezclas verdad con basura en Railway y se complica.

### E. OHLCV es prerequisito de todo lo demas
- features_* dependen de precios_diarios
- scanner depende de features_*
- HV (en opciones) depende de precios_diarios
- Z-scores dependen de precios_diarios

Orden de recovery siempre: **precios -> indicadores -> features -> scanner -> opciones**
(opciones puede correr independiente pero los calculos derivados de IV vs HV requieren
precios al dia).

---

## Comandos de verificacion rapida

### Estado de Railway en un vistazo
```bash
status.bat
```

### Estado de local en un vistazo
```bash
status_local.bat
```

### Query ad-hoc a Railway
```python
# desde Python (.env.local cargado)
from sqlalchemy import create_engine, text
import os
e = create_engine(os.environ["DATABASE_URL"])
print(e.connect().execute(text("SELECT MAX(fecha) FROM precios_diarios")).scalar())
```

### Query ad-hoc al local
```python
# desde Python (.env cargado)
from sqlalchemy import create_engine, text
import os
url = f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
e = create_engine(url)
print(e.connect().execute(text("SELECT MAX(fecha) FROM precios_diarios")).scalar())
```

### Matar procesos Python en Windows
```bash
taskkill /F /IM python.exe
```

### Ver procesos Python activos
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select Id, StartTime
```
