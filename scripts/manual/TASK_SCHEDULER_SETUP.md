# Windows Task Scheduler — Configuracion del pipeline automatico

## Tarea: Pipeline diario (L-V 10:00 ART)

### Paso 1 — Abrir Task Scheduler
1. Presionar `Win + R` → escribir `taskschd.msc` → Enter
2. En el panel derecho: **Create Basic Task...**

### Paso 2 — General
- **Name:** `IndicadoresML - Pipeline Diario`
- **Description:** Pipeline diario: precios fast_info + features + scanner ML

### Paso 3 — Trigger
- When: **Daily**
- Start: fecha de hoy, hora **10:00:00 AM**
- Recur every: **1 days**
- Click **Next**

### Paso 4 — Action
- Action: **Start a program**
- Program/script:
  ```
  C:\Users\juand\OneDrive\Escritorio\Indicadores y Machine Learning\scripts\pipeline_automatico.bat
  ```
- Start in (opcional pero recomendado):
  ```
  C:\Users\juand\OneDrive\Escritorio\Indicadores y Machine Learning
  ```

### Paso 5 — Finalizar y ajustar configuracion avanzada
Antes de cerrar, tildar **"Open the Properties dialog..."** y hacer clic en **Finish**.

En Properties:
- Tab **General**:
  - Tildar: `Run whether user is logged on or not`
  - Tildar: `Run with highest privileges`
- Tab **Conditions**:
  - Destildar: `Start the task only if the computer is on AC power` (para laptops)
- Tab **Settings**:
  - Tildar: `Run task as soon as possible after a scheduled start is missed`
    (esto garantiza que si la PC estaba apagada a las 10:00, corre cuando se enciende)
  - If the task is already running: `Do not start a new instance`

### Paso 6 — Limitar a dias habiles
La tarea se configura como "diaria" pero el script `pipeline_automatico.bat` verifica
internamente si es dia habil NYSE. Si no lo es, termina en segundos sin hacer nada.

---

## Verificar que funciona

### Test manual (primera vez):
1. Abrir Task Scheduler
2. Buscar `IndicadoresML - Pipeline Diario`
3. Click derecho → **Run**
4. Verificar log en: `logs\pipeline_auto\pipeline_YYYYMMDD_HHMM.log`

### Ver logs:
```
logs\pipeline_auto\
    pipeline_20260505_1000.log
    pipeline_20260506_1000.log
    ...
```

---

## Resumen de tiempos

| Hora ART | Hora UTC | Evento |
|---|---|---|
| 10:00 | 13:00 | Task Scheduler inicia pipeline |
| 10:45 | 13:45 | Paso 1 fast_info termina |
| 10:50 | 13:50 | Paso 2 features termina |
| 11:50 | 14:50 | Paso 3 scanner termina |
| 12:15 | 15:15 | Bots Alpaca (GH Actions) leen datos frescos de Railway |

---

## Notas importantes

- La PC debe estar encendida a las 10:00 ART L-V.
- Si la PC estaba apagada, la tarea corre automaticamente al encenderse
  (opcion "Run task as soon as possible after a missed start" activada).
- El script limpia las cookies de yfinance automaticamente antes de cada corrida.
- Los logs quedan en `logs/pipeline_auto/` con fecha y hora en el nombre.
