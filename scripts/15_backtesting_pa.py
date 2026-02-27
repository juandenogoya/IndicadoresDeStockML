"""
15_backtesting_pa.py
Runner de la Fase 11 -- Backtesting PA Challenger (EV1-EV4 / SV1-SV4).

Usa features de precio/accion y estructura de mercado como senales de
entrada y salida, separado del backtesting rule-based original.

Prerequisitos:
    features_precio_accion    (script 07a)
    features_market_structure (script 12)
    scoring_tecnico           (script 04)

Output:
    operaciones_bt_pa   -- detalle de cada operacion simulada
    resultados_bt_pa    -- metricas por EV x SV x ticker x segmento

Uso:
    python scripts/15_backtesting_pa.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtesting.simulator_pa import ejecutar_backtesting_pa, ESTRATEGIAS_ENTRADA_PA, ESTRATEGIAS_SALIDA_PA, FECHA_INICIO_BT
from src.backtesting.metrics_pa import calcular_y_guardar_resultados_pa, ranking_estrategias_pa
from src.data.database import query_df, ejecutar_sql


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) "
            "VALUES (%s,%s,%s,%s)",
            ("15_bt_pa", accion, detalle, estado)
        )
    except Exception:
        pass


def verificar_prerequisitos():
    """Verifica que las tablas fuente tienen datos suficientes."""
    r = query_df("""
        SELECT
            (SELECT COUNT(*) FROM precios_diarios)            AS n_pd,
            (SELECT COUNT(*) FROM features_precio_accion)     AS n_pa,
            (SELECT COUNT(*) FROM features_market_structure)  AS n_ms,
            (SELECT COUNT(*) FROM scoring_tecnico)            AS n_sc
    """)
    row = r.iloc[0]
    n_pd = int(row["n_pd"])
    n_pa = int(row["n_pa"])
    n_ms = int(row["n_ms"])
    n_sc = int(row["n_sc"])

    print(f"  precios_diarios          : {n_pd:,} filas")
    print(f"  features_precio_accion   : {n_pa:,} filas")
    print(f"  features_market_structure: {n_ms:,} filas")
    print(f"  scoring_tecnico          : {n_sc:,} filas")

    if n_pd < 1000:
        raise RuntimeError("precios_diarios vacia. Ejecutar scripts 02-03.")
    if n_pa == 0:
        raise RuntimeError("features_precio_accion vacia. Ejecutar script 07a.")
    if n_ms == 0:
        raise RuntimeError("features_market_structure vacia. Ejecutar script 12.")
    if n_sc == 0:
        raise RuntimeError("scoring_tecnico vacio. Ejecutar script 04.")

    return n_pd


def imprimir_ranking(segmento: str, n: int = 8):
    """Muestra el top N de combinaciones PA para un segmento."""
    df = ranking_estrategias_pa(segmento=segmento, min_ops=5)
    if df.empty:
        print(f"  Sin resultados PA para segmento {segmento} (min 5 ops).")
        return

    top = min(n, len(df))
    print(f"\n  TOP {top} PA - Segmento {segmento}:")
    print(f"  {'EV':<5} {'SV':<5} {'Ops':>5} {'WR%':>7} {'RetProm':>9} {'PF':>8} {'MaxDD':>8} {'DiasProm':>9}")
    print("  " + "-" * 60)
    for _, row in df.head(top).iterrows():
        print(
            f"  {row['estrategia_entrada']:<5} {row['estrategia_salida']:<5} "
            f"{int(row['total_operaciones']):>5} "
            f"{float(row['win_rate_pct']):>7.1f} "
            f"{float(row['ret_promedio']):>9.2f} "
            f"{float(row['profit_factor']):>8.2f} "
            f"{float(row['max_dd']):>8.2f} "
            f"{float(row['dias_prom']):>9.1f}"
        )


def imprimir_distribucion_motivos(df_ops):
    """Muestra como se distribuyen los motivos de salida."""
    if df_ops.empty:
        return
    print(f"\n  Distribucion de motivos de salida (FULL):")
    dist = df_ops["motivo_salida"].value_counts()
    total = len(df_ops)
    for motivo, cnt in dist.items():
        print(f"    {motivo:<20}: {cnt:>5}  ({cnt/total*100:.1f}%)")


def main():
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  FASE 11 -- BACKTESTING PA CHALLENGER")
    print("=" * 65)
    print(f"  Estrategias entrada : {', '.join(ESTRATEGIAS_ENTRADA_PA)}")
    print(f"  Estrategias salida  : {', '.join(ESTRATEGIAS_SALIDA_PA)}")
    print(f"  Combinaciones       : {len(ESTRATEGIAS_ENTRADA_PA) * len(ESTRATEGIAS_SALIDA_PA)} (4x4)")
    print(f"  Periodo             : {FECHA_INICIO_BT} -> hoy  (segmento: FULL)")
    print(f"  Tablas output       : operaciones_bt_pa, resultados_bt_pa")
    print(f"  Inicio              : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("\n  Verificando prerequisitos...")
    try:
        verificar_prerequisitos()
    except RuntimeError as e:
        print(f"\n[ERROR prerequisito] {e}")
        sys.exit(1)
    print("  Prerequisitos OK.")

    # Simulacion
    try:
        df_ops = ejecutar_backtesting_pa()
        log_ejecucion(
            "BACKTEST_PA",
            f"{len(df_ops):,} operaciones simuladas",
            "OK"
        )
    except Exception as e:
        log_ejecucion("BACKTEST_PA", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if df_ops.empty:
        print("\n[WARN] Sin operaciones generadas. Revisa las condiciones de entrada.")
        sys.exit(0)

    # Metricas
    print("\n  Calculando y guardando metricas PA...")
    try:
        calcular_y_guardar_resultados_pa(df_ops)
        log_ejecucion("METRICAS_PA", "resultados_bt_pa actualizado", "OK")
    except Exception as e:
        log_ejecucion("METRICAS_PA", str(e), "ERROR")
        print(f"\n[ERROR metricas] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Ranking segmento FULL
    imprimir_ranking("FULL")

    # Distribucion de motivos en BACKTEST
    imprimir_distribucion_motivos(df_ops)

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n{'='*65}")
    print(f"  PA completado en {duracion}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print("\n  Siguiente paso: python scripts/16_comparar_backtest.py")


if __name__ == "__main__":
    main()
