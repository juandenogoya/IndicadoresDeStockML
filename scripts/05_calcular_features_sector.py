"""
05_calcular_features_sector.py
Runner de la Fase 4a — Cálculo de Z-Scores y métricas sectoriales.

Calcula la posición relativa de cada ticker dentro de su sector
y persiste los resultados en la tabla features_sector.

Uso:
    python scripts/05_calcular_features_sector.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.indicators.sector_features import procesar_features_sector
from src.data.database import ejecutar_sql


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) VALUES (%s,%s,%s,%s)",
            ("05_features_sector", accion, detalle, estado)
        )
    except Exception:
        pass


def main():
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  FASE 4a — FEATURES SECTORIALES (Z-Scores)")
    print("=" * 65)
    print(f"  Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("\n[1/1] Calculando features sectoriales...")

    try:
        df = procesar_features_sector(guardar_db=True)
        log_ejecucion("SECTOR_FEATURES", f"{len(df):,} registros", "OK")

    except Exception as e:
        log_ejecucion("SECTOR_FEATURES", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    fin      = datetime.now()
    duracion = (fin - inicio).seconds

    print("\n" + "=" * 65)
    print("  FEATURES SECTORIALES COMPLETADAS")
    print("=" * 65)
    print(f"  Tickers procesados : {df['ticker'].nunique()}")
    print(f"  Sectores           : {df['sector'].nunique()} ({', '.join(df['sector'].unique())})")
    print(f"  Total registros    : {len(df):,}")
    print(f"  Rango de fechas    : {df['fecha'].min()} / {df['fecha'].max()}")
    print(f"  Tiempo total       : {duracion}s")
    print(f"  Datos en DB        : tabla features_sector")
    print("=" * 65)


if __name__ == "__main__":
    main()
