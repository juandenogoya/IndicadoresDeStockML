#!/bin/bash
# =============================================================
#  oracle_pipeline_diario.sh
#  Pipeline diario secuencial: precios -> features -> scanner
#  Disenado para cron en Oracle Cloud (Ubuntu 22.04).
#
#  Horario cron: L-V 22:00 UTC = 19:00 ART (2hs post-cierre NYSE)
#  Duracion estimada: ~95 minutos
#  Log: /tmp/cron_pipeline_diario.log
# =============================================================

REPO="/home/ubuntu/IndicadoresDeStockML"
PYTHON="$REPO/venv/bin/python3"
LOG="/tmp/cron_pipeline_diario.log"
T_START=$(date +%s)

# ── Cargar credenciales Telegram desde .env ───────────────────
if [ -f "$REPO/.env" ]; then
    TELEGRAM_BOT_TOKEN=$(grep '^TELEGRAM_BOT_TOKEN=' "$REPO/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
    TELEGRAM_CHAT_ID=$(grep '^TELEGRAM_CHAT_ID='   "$REPO/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
fi

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $1" | tee -a "$LOG"
}

# Envia mensaje Telegram (exit silencioso si no hay credenciales)
telegram() {
    [ -z "$TELEGRAM_BOT_TOKEN" ] && return 0
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d  "chat_id=${TELEGRAM_CHAT_ID}" \
        -d  "parse_mode=HTML" \
        --data-urlencode "text=${msg}" > /dev/null 2>&1
}

# Minutos transcurridos desde timestamp Unix $1
dur_min() { echo $(( ($(date +%s) - $1) / 60 )); }

# Cuenta lineas con rate limit en el log desde la linea $1+1
contar_rl() {
    tail -n +$(( $1 + 1 )) "$LOG" 2>/dev/null \
        | grep -iE "rate.?limit|YFRateLimit|Too Many Requests|429" \
        | wc -l
}

# ── Inicio ────────────────────────────────────────────────────
echo "" >> "$LOG"
log "============================================================"
log "  PIPELINE DIARIO ORACLE  (inicio)"
log "============================================================"

cd "$REPO" || { log "ERROR: no se puede acceder a $REPO"; exit 1; }

FECHA=$(date -u '+%Y-%m-%d')

# ── Verificar dia habil ───────────────────────────────────────
log "Verificando dia habil NYSE..."
"$PYTHON" scripts/manual/check_fecha.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    log "No es dia habil. Pipeline cancelado."
    exit 0
fi
log "Dia habil confirmado."

# ── Paso 1: Precios ───────────────────────────────────────────
log ""
log "Paso 1 -- Precios + indicadores (199 tickers)..."
T1=$(date +%s)
L1=$(wc -l < "$LOG")

"$PYTHON" scripts/cron_diario.py --step precios >> "$LOG" 2>&1
EXIT1=$?

DUR1=$(dur_min $T1)
RL1=$(contar_rl $L1)

if [ $EXIT1 -ne 0 ]; then
    log "ERROR en Paso 1. Abortando pipeline."
    telegram "<b>Oracle Pipeline -- ERROR Paso 1</b>
Fecha: ${FECHA}
Precios + indicadores: FALLO
Duracion: ${DUR1} min
Pipeline abortado. Revisar: /tmp/cron_pipeline_diario.log"
    exit 1
fi

log "Paso 1 OK."
if [ "$RL1" -gt 0 ]; then
    telegram "<b>Oracle Pipeline -- Paso 1 OK (con warnings)</b>
Fecha: ${FECHA}
Precios + indicadores: OK
Duracion: ${DUR1} min
Rate limits detectados: ${RL1} (puede haber tickers sin actualizar)"
else
    telegram "<b>Oracle Pipeline -- Paso 1 OK</b>
Fecha: ${FECHA}
Precios + indicadores: OK
Duracion: ${DUR1} min"
fi

# ── Paso 2: Features ─────────────────────────────────────────
log ""
log "Paso 2 -- Features PA + Market Structure..."
T2=$(date +%s)

"$PYTHON" scripts/cron_diario.py --step features >> "$LOG" 2>&1
EXIT2=$?

DUR2=$(dur_min $T2)

if [ $EXIT2 -ne 0 ]; then
    log "ERROR en Paso 2. Abortando pipeline."
    telegram "<b>Oracle Pipeline -- ERROR Paso 2</b>
Fecha: ${FECHA}
Features PA + Market Structure: FALLO
Duracion: ${DUR2} min
Pipeline abortado. Revisar: /tmp/cron_pipeline_diario.log"
    exit 1
fi

log "Paso 2 OK."
telegram "<b>Oracle Pipeline -- Paso 2 OK</b>
Fecha: ${FECHA}
Features PA + Market Structure: OK
Duracion: ${DUR2} min"

# ── Paso 3: Scanner ML ───────────────────────────────────────
log ""
log "Paso 3 -- Scanner ML + Telegram..."
T3=$(date +%s)

"$PYTHON" scripts/cron_diario.py --step scanner >> "$LOG" 2>&1
EXIT3=$?

DUR3=$(dur_min $T3)
DUR_TOTAL=$(dur_min $T_START)

if [ $EXIT3 -ne 0 ]; then
    log "ERROR en Paso 3."
    telegram "<b>Oracle Pipeline -- ERROR Paso 3</b>
Fecha: ${FECHA}
Scanner ML: FALLO
Duracion: ${DUR3} min
Revisar: /tmp/cron_pipeline_diario.log"
    exit 1
fi

log "Paso 3 OK."
telegram "<b>Oracle Pipeline -- Paso 3 OK</b>
Fecha: ${FECHA}
Scanner ML + alertas: OK
Duracion paso: ${DUR3} min | Total pipeline: ${DUR_TOTAL} min"

# ── Fin ───────────────────────────────────────────────────────
log ""
log "Pipeline completado exitosamente."
log "============================================================"
exit 0
