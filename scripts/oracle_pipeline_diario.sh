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

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $1" | tee -a "$LOG"
}

# ── Inicio ────────────────────────────────────────────────────
echo "" >> "$LOG"
log "============================================================"
log "  PIPELINE DIARIO ORACLE  (inicio)"
log "============================================================"

cd "$REPO" || { log "ERROR: no se puede acceder a $REPO"; exit 1; }

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
"$PYTHON" scripts/cron_diario.py --step precios >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    log "ERROR en Paso 1. Abortando pipeline."
    exit 1
fi
log "Paso 1 OK."

# ── Paso 2: Features ─────────────────────────────────────────
log ""
log "Paso 2 -- Features PA + Market Structure..."
"$PYTHON" scripts/cron_diario.py --step features >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    log "ERROR en Paso 2. Abortando pipeline."
    exit 1
fi
log "Paso 2 OK."

# ── Paso 3: Scanner ML ───────────────────────────────────────
log ""
log "Paso 3 -- Scanner ML + Telegram..."
"$PYTHON" scripts/cron_diario.py --step scanner >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    log "ERROR en Paso 3."
    exit 1
fi
log "Paso 3 OK."

# ── Fin ───────────────────────────────────────────────────────
log ""
log "Pipeline completado exitosamente."
log "============================================================"
exit 0
