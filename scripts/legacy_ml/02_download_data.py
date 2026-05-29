"""
02_download_data.py
Pipeline completo de la Fase 1:
  1. Descarga OHLCV de todos los activos desde yfinance
  2. Calcula todos los indicadores técnicos
  3. Persiste todo en PostgreSQL

Uso:
    python scripts/02_download_data.py
    python scripts/02_download_data.py --start 2024-01-01   (actualización parcial)
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import START_DATE, ALL_TICKERS
from src.data.download import descargar_todos
from src.indicators.technical import procesar_indicadores_todos
from src.data.database import ejecutar_sql


def log_ejecucion(script: str, accion: str, detalle: str, estado: str):
    """Registra la ejecución en la tabla de logs."""
    sql = """
        INSERT INTO log_ejecuciones (script, accion, detalle, estado)
        VALUES (%s, %s, %s, %s)
    """
    try:
        ejecutar_sql(sql, (script, accion, detalle, estado))
    except Exception:
        pass  # No interrumpir el flujo principal por un error de log


def main():
    parser = argparse.ArgumentParser(description="Pipeline Fase 1 - Descarga y Calculo")
    parser.add_argument(
        "--start", type=str, default=START_DATE,
        help=f"Fecha de inicio (YYYY-MM-DD). Default: {START_DATE}"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers específicos a procesar. Default: todos"
    )
    args = parser.parse_args()

    inicio = datetime.now()
    tickers = args.tickers or ALL_TICKERS

    print("\n" + "=" * 60)
    print("  FASE 1 — PIPELINE DESCARGA Y CALCULO DE INDICADORES")
    print("=" * 60)
    print(f"  Fecha inicio datos : {args.start}")
    print(f"  Activos a procesar : {len(tickers)}")
    print(f"  Inicio ejecucion   : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── PASO 1: Descarga OHLCV ────────────────────────────────
    print("\n[PASO 1/2] DESCARGA OHLCV")
    try:
        datos = descargar_todos(
            tickers=tickers,
            start=args.start,
            guardar_db=True,
        )
        log_ejecucion("02_download_data", "DESCARGA_OHLCV",
                      f"{len(datos)} activos OK", "OK")
    except Exception as e:
        log_ejecucion("02_download_data", "DESCARGA_OHLCV", str(e), "ERROR")
        print(f"\n[ERROR] Descarga fallida: {e}")
        sys.exit(1)

    if not datos:
        print("[ERROR] No se descargaron datos. Abortando.")
        sys.exit(1)

    # ── PASO 2: Calculo de Indicadores ────────────────────────
    print("\n[PASO 2/2] CALCULO DE INDICADORES TECNICOS")
    try:
        indicadores = procesar_indicadores_todos(datos, guardar_db=True)
        log_ejecucion("02_download_data", "CALCULO_INDICADORES",
                      f"{len(indicadores)} activos OK", "OK")
    except Exception as e:
        log_ejecucion("02_download_data", "CALCULO_INDICADORES", str(e), "ERROR")
        print(f"\n[ERROR] Calculo de indicadores fallido: {e}")
        sys.exit(1)

    # ── Resumen Final ─────────────────────────────────────────
    fin = datetime.now()
    duracion = (fin - inicio).seconds

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"  Activos con OHLCV       : {len(datos)}")
    print(f"  Activos con indicadores : {len(indicadores)}")
    print(f"  Tiempo total            : {duracion}s")
    print(f"  Fin ejecucion           : {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
