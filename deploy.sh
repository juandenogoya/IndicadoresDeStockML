#!/usr/bin/env bash
# deploy.sh — Commit y push a GitHub en un solo comando.
# Streamlit Cloud redespliega automaticamente con cada push.
#
# Uso:
#   bash deploy.sh "mensaje del commit"
#   bash deploy.sh          (usa mensaje generico con timestamp)

set -e

MSG="${1:-"chore: actualizacion $(date '+%Y-%m-%d %H:%M')"}"

echo "=== GIT STATUS ==="
git status --short

echo ""
echo "=== COMMIT: $MSG ==="
git add -A
git commit -m "$MSG

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

echo ""
echo "=== PUSH a origin/main ==="
git push origin main

echo ""
echo "OK — Streamlit Cloud redesplegara en ~1-2 minutos."
echo "    https://indicadoresat.streamlit.app"
