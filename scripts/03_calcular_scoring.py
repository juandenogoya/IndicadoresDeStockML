"""
03_calcular_scoring.py
Calcula el scoring rule-based para todos los activos y persiste
los resultados en la tabla scoring_tecnico de PostgreSQL.

Uso:
    python scripts/03_calcular_scoring.py
    python scripts/03_calcular_scoring.py --tickers JPM GS MS   (tickers específicos)
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import ALL_TICKERS
from src.scoring.rule_based import (
    procesar_scoring_todos,
    resumen_scoring,
    señal_actual,
)
from src.data.database import ejecutar_sql


def log_ejecucion(accion: str, detalle: str, estado: str):
    sql = """
        INSERT INTO log_ejecuciones (script, accion, detalle, estado)
        VALUES (%s, %s, %s, %s)
    """
    try:
        ejecutar_sql(sql, ("03_calcular_scoring", accion, detalle, estado))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Fase 2 - Scoring Rule-Based")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers específicos. Default: todos"
    )
    args = parser.parse_args()

    inicio = datetime.now()
    tickers = args.tickers or ALL_TICKERS

    print("\n" + "=" * 65)
    print("  FASE 2 — SCORING RULE-BASED")
    print("=" * 65)
    print(f"  Activos a procesar : {len(tickers)}")
    print(f"  Inicio             : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Umbral LONG        : score >= 0.60")
    print(f"  Pesos              : RSI=20% MACD=20% SMA200=20%")
    print(f"                       SMA50=15% Momentum=15% SMA21=10%")
    print("=" * 65)

    # ── Calcular y persistir scoring ──────────────────────────
    try:
        resultados = procesar_scoring_todos(tickers=tickers, guardar_db=True)
        log_ejecucion("SCORING", f"{len(resultados)} activos OK", "OK")
    except Exception as e:
        log_ejecucion("SCORING", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    fin = datetime.now()

    # ── Resumen por activo ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMEN POR ACTIVO")
    print("=" * 65)
    df_resumen = resumen_scoring()
    if not df_resumen.empty:
        print(df_resumen.to_string(index=False))

    # ── Señal actual (última sesión disponible) ───────────────
    print("\n" + "=" * 65)
    print("  SEÑAL ACTUAL — ULTIMA SESION DISPONIBLE")
    print("=" * 65)
    df_señal = señal_actual(tickers)
    if not df_señal.empty:
        cols = ["ticker", "fecha", "score_ponderado", "condiciones_ok", "senal"]
        print(df_señal[cols].to_string(index=False))

    # ── Footer ────────────────────────────────────────────────
    duracion = (fin - inicio).seconds
    total_registros = sum(len(df) for df in resultados.values())

    print("\n" + "=" * 65)
    print("  SCORING COMPLETADO")
    print("=" * 65)
    print(f"  Activos procesados : {len(resultados)}/{len(tickers)}")
    print(f"  Total registros    : {total_registros:,}")
    print(f"  Tiempo total       : {duracion}s")
    print(f"  Fin                : {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
