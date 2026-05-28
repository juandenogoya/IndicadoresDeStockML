"""
12_calcular_market_structure.py
Runner de la Fase 9 -- Calculo de features de estructura de mercado.

Detecta swing highs/lows (ventanas N=5 y N=10), clasifica la estructura
de precio (HH/HL vs LH/LL) y detecta eventos BOS/CHoCH.
Output: tabla features_market_structure (24 features por barra).

Uso:
    python scripts/12_calcular_market_structure.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.indicators.market_structure import (
    procesar_features_market_structure,
    FEATURE_COLS_MS,
)
from src.data.database import query_df, ejecutar_sql


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) "
            "VALUES (%s,%s,%s,%s)",
            ("12_calc_ms", accion, detalle, estado)
        )
    except Exception:
        pass


def verificar_prerequisitos():
    """Verifica que precios_diarios tiene datos OHLCV suficientes."""
    r = query_df("""
        SELECT
            COUNT(*)              AS n_filas,
            COUNT(DISTINCT ticker) AS n_tickers,
            MIN(fecha)            AS fecha_inicio,
            MAX(fecha)            AS fecha_fin
        FROM precios_diarios
        WHERE close > 0 AND high > 0 AND low > 0
    """)
    row = r.iloc[0]
    n     = int(row["n_filas"])
    ticks = int(row["n_tickers"])
    print(f"  precios_diarios: {n:,} filas ({ticks} tickers)")
    print(f"  Rango: {row['fecha_inicio']} -> {row['fecha_fin']}")
    if n < 1000:
        raise RuntimeError(
            f"Solo {n} filas en precios_diarios. Ejecutar scripts 02-03 primero."
        )
    return n


def main():
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  FASE 9 -- FEATURES MARKET STRUCTURE")
    print("=" * 65)
    print(f"  Ventanas        : N=5 (tactico), N=10 (estrategico)")
    print(f"  Features total  : {len(FEATURE_COLS_MS)} (12 per ventana)")
    print(f"  Output tabla    : features_market_structure")
    print(f"  Inicio          : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("\n  Verificando prerequisitos...")
    try:
        n_filas = verificar_prerequisitos()
    except RuntimeError as e:
        print(f"\n[ERROR prerequisito] {e}")
        sys.exit(1)

    print(f"\n  Prerequisitos OK ({n_filas:,} barras disponibles)")

    try:
        print("\n  Procesando features de market structure...")
        df_feat = procesar_features_market_structure()
        log_ejecucion(
            "CALC_MS",
            f"{len(df_feat):,} filas - {len(FEATURE_COLS_MS)} features",
            "OK"
        )
    except Exception as e:
        log_ejecucion("CALC_MS", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Resumen de cobertura
    print("\n  Resumen de cobertura por ticker:")
    r = query_df("""
        SELECT
            ticker,
            COUNT(*)                          AS total_barras,
            SUM(CASE WHEN estructura_5  != 0 THEN 1 ELSE 0 END) AS est5_definida,
            SUM(CASE WHEN estructura_10 != 0 THEN 1 ELSE 0 END) AS est10_definida,
            SUM(COALESCE(bos_bull_5, 0) + COALESCE(bos_bear_5, 0)) AS bos5_events,
            SUM(COALESCE(choch_bull_5, 0) + COALESCE(choch_bear_5, 0)) AS choch5_events
        FROM features_market_structure
        GROUP BY ticker
        ORDER BY ticker
    """)
    print(f"  {'Ticker':<8} {'Barras':>8} {'Est5%':>7} {'Est10%':>7} {'BOS5':>6} {'CHoCH5':>7}")
    print("  " + "-" * 46)
    for _, row in r.iterrows():
        pct5  = row["est5_definida"]  / row["total_barras"] * 100
        pct10 = row["est10_definida"] / row["total_barras"] * 100
        print(
            f"  {row['ticker']:<8} {int(row['total_barras']):>8,} "
            f"{pct5:>6.1f}% {pct10:>6.1f}% "
            f"{int(row['bos5_events']):>6} {int(row['choch5_events']):>7}"
        )

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n{'='*65}")
    print(f"  Completado en {duracion}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print("\n  Siguiente paso: python scripts/13_train_models_v3.py")


if __name__ == "__main__":
    main()
