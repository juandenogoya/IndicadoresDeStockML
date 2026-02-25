# Deploy a Railway — Guia paso a paso

## Arquitectura final
```
Railway
  ├── PostgreSQL Service  (activos_ml en la nube)
  └── Python Cron Service (scanner L-V 17:30 ET)
        ├── src/            (modulos del proyecto)
        ├── scripts/        (cron_diario.py, etc.)
        └── models_v3/      (champion.joblib x4, ~11MB)
```

---

## FASE 1 — Migrar la base de datos

### 1.1 Crear el PostgreSQL en Railway
1. Ir a https://railway.app/dashboard
2. **New Project** → **Database** → **PostgreSQL**
3. Una vez creado, ir a la DB → tab **Variables**
4. Copiar el valor de `DATABASE_URL` (formato: `postgresql://user:pass@host:port/dbname`)

### 1.2 Exportar la DB local
Abrir una terminal y ejecutar:
```bash
# Reemplazar con tus credenciales locales
pg_dump -h localhost -U postgres -d activos_ml -F c -f activos_ml_backup.dump
```
El archivo `activos_ml_backup.dump` queda en la carpeta actual.

### 1.3 Importar a Railway
```bash
# Reemplazar DATABASE_URL con el valor copiado en paso 1.1
pg_restore --no-owner --no-privileges -d "DATABASE_URL" activos_ml_backup.dump
```

Si pg_restore no esta disponible, instalar PostgreSQL client:
- Windows: descargar desde https://www.postgresql.org/download/windows/
- O usar: `winget install PostgreSQL.PostgreSQL`

### 1.4 Verificar la migracion
```bash
psql "DATABASE_URL" -c "SELECT COUNT(*) FROM precios_diarios;"
```
Debe mostrar el mismo numero que la DB local.

---

## FASE 2 — Subir el codigo a GitHub

El proyecto necesita estar en un repositorio de GitHub para que Railway
lo lea automaticamente en cada push.

### 2.1 Crear repo en GitHub
1. Ir a https://github.com/new
2. Nombre sugerido: `indicadores-ml`
3. Privado (recomendado, contiene credenciales de Telegram)

### 2.2 Crear .gitignore
Asegurarse de que `.gitignore` excluye:
```
.env
*.dump
__pycache__/
models/
models_v2/
```
(Los archivos grandes models_v1/v2 no se necesitan en Railway)

### 2.3 Primer push
```bash
cd "C:\Users\juand\OneDrive\Escritorio\Indicadores y Machine Learning"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/indicadores-ml.git
git push -u origin main
```

---

## FASE 3 — Crear el servicio cron en Railway

### 3.1 Nuevo servicio desde GitHub
1. En el proyecto Railway → **New Service** → **GitHub Repo**
2. Seleccionar el repo `indicadores-ml`
3. Railway va a detectar el `railway.toml` automaticamente

### 3.2 Configurar variables de entorno
En el servicio → tab **Variables**, agregar:

| Variable              | Valor                          |
|-----------------------|--------------------------------|
| `DATABASE_URL`        | (link automatico si es el mismo proyecto Railway) |
| `TELEGRAM_BOT_TOKEN`  | (copiar de tu .env local)      |
| `TELEGRAM_CHAT_ID`    | (copiar de tu .env local)      |

**Importante:** Si la DB de Railway esta en el mismo proyecto, Railway
puede inyectar `DATABASE_URL` automaticamente via **Service Variables**.
Para eso: en el servicio Python → Variables → **Add Reference** → seleccionar
la DB → variable `DATABASE_URL`.

### 3.3 Verificar el cron schedule
El `railway.toml` ya tiene configurado:
```toml
cronSchedule = "30 20 * * 1-5"
```
Esto es las 20:30 UTC = 17:30 ET (hora de cierre NYSE + 30 min).

Para verificar en tu zona horaria (Argentina UTC-3): 17:30 ET = 19:30 ART.

Si prefieres otro horario, cambiar en railway.toml:
- 17:00 ET (apertura +30min) → `"30 14 * * 1-5"`
- 16:30 ET (cierre exacto)  → `"30 21 * * 1-5"`
- 20:00 ET (noche)          → `"0 1 * * 2-6"` (1am UTC = dia siguiente)

### 3.4 Primer deploy manual (test)
1. En Railway → el servicio Python → **Deploy** → **Trigger Deployment**
2. Ver logs en tiempo real: Dashboard → Deployments → el ultimo → **View Logs**
3. Verificar que llega el mensaje de Telegram

---

## FASE 4 — Uso normal post-deploy

### Desde la PC local (con la DB en Railway)
Agregar al `.env` local:
```
DATABASE_URL=postgresql://user:pass@host:port/dbname
```
Y comentar las variables DB_HOST / DB_USER / etc.
Todos los scripts del proyecto seguiran funcionando igual, pero
ahora leen/escriben contra la DB de Railway.

### Monitoreo
- **Logs del cron**: Railway Dashboard → tu proyecto → servicio Python → Deployments
- **Ver alertas guardadas**: cualquier cliente PostgreSQL (DBeaver, pgAdmin, TablePlus)
  conectando con el DATABASE_URL de Railway
- **Telegram**: el bot envia el resumen diario a las 19:30 ART automaticamente

---

## Resumen de archivos creados para Railway

| Archivo              | Descripcion                                      |
|----------------------|--------------------------------------------------|
| `railway.toml`       | Config del cron (horario + comando)              |
| `.railwayignore`     | Excluye models_v1/v2 y archivos de desarrollo    |
| `scripts/cron_diario.py` | Orquestador: scanner + verificacion + Telegram |
| `src/utils/config.py`| Actualizado para parsear DATABASE_URL            |
