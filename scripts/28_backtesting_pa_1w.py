"""
28_backtesting_pa_1w.py
Backtesting PA Challenger sobre barras semanales (1W).

Ejecuta la matriz 4x4 (EV1-4 x SV1-4) para todos los tickers
usando features semanales de precio/accion y market structure.

Diferencias clave vs 1D:
    - Datos:    precios_semanales + indicadores_tecnicos_1w
                + features_precio_accion_1w + features_market_structure_1w
    - Timeout:  20 semanas (~5 meses, no dias)
    - SV3:      solo estructura_10==-1 (sin score_ponderado semanal)
    - Tablas:   operaciones_bt_pa_1w + resultados_bt_pa_1w

Uso:
    python scripts/28_backtesting_pa_1w.py
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    from src.data.database import query_df
    from src.backtesting.simulator_pa_1w import ejecutar_backtesting_pa_1w
    from src.backtesting.metrics_pa_1w import (
        calcular_y_guardar_resultados_pa_1w,
        ranking_estrategias_pa_1w,
    )

    log("=" * 65)
    log("  BACKTESTING PA 1W -- MATRIZ 4x4 (EV1-4 x SV1-4)")
    log("=" * 65)
    log("  Datos  : precios_semanales + indicadores_tecnicos_1w")
    log("         + features_precio_accion_1w + features_market_structure_1w")
    log("  Destino: operaciones_bt_pa_1w + resultados_bt_pa_1w")
    log("  SV3    : solo estructura_10==-1 (sin score_ponderado semanal)")
    log("  Timeout: 20 semanas (~5 meses)")
    print()

    # Cargar todos los tickers activos
    df_activos = query_df("SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker")
    tickers = df_activos["ticker"].tolist() if not df_activos.empty else []
    log(f"  Tickers: {len(tickers)}")
    print()

    t0 = time.time()

    # ── Simulacion completa ───────────────────────────────────
    df_ops = ejecutar_backtesting_pa_1w(
        tickers=tickers,
        guardar_db=True,
        truncate=True,
    )

    elapsed_sim = time.time() - t0
    print()
    log(f"  Simulacion completada: {len(df_ops):,} ops | {elapsed_sim:.0f}s")

    # ── Calculo de metricas ───────────────────────────────────
    log("=" * 65)
    log("  CALCULO DE METRICAS")
    log("=" * 65)
    if not df_ops.empty:
        calcular_y_guardar_resultados_pa_1w(df_ops)
    else:
        log("  [WARN] Sin operaciones para calcular metricas.")

    elapsed_total = time.time() - t0

    # ── Verificacion en DB ────────────────────────────────────
    print()
    log("=" * 65)
    log("  RESUMEN")
    log("=" * 65)

    df_check = query_df("""
        SELECT COUNT(*)                AS total_ops,
               COUNT(DISTINCT ticker) AS tickers,
               MIN(fecha_entrada)     AS primera,
               MAX(fecha_salida)      AS ultima
        FROM operaciones_bt_pa_1w
    """)
    if not df_check.empty:
        r = df_check.iloc[0]
        log(f"  Operaciones totales : {int(r['total_ops']):,}")
        log(f"  Tickers con ops     : {int(r['tickers'])}")
        log(f"  Rango               : {r['primera']} -> {r['ultima']}")

    df_motivos = query_df("""
        SELECT motivo_salida, COUNT(*) AS n
        FROM   operaciones_bt_pa_1w
        GROUP  BY motivo_salida
        ORDER  BY n DESC
    """)
    if not df_motivos.empty:
        log("\n  Motivos de salida:")
        for _, row in df_motivos.iterrows():
            log(f"    {row['motivo_salida']:15} : {int(row['n']):,}")

    log(f"\n  Tiempo total        : {elapsed_total:.0f}s")

    # ── Ranking de estrategias ────────────────────────────────
    print()
    log("  RANKING PA 1W (min 3 ops, ordenado por PF):")
    df_rank = ranking_estrategias_pa_1w(segmento="FULL", min_ops=3)
    if not df_rank.empty:
        log(f"  {'EE':4} {'ES':4} {'Ops':>5} {'WR%':>6} {'RetProm':>8} {'MaxDD':>7} {'PF':>7} {'Sem':>5}")
        log("  " + "-" * 52)
        for _, row in df_rank.iterrows():
            log(
                f"  {row['estrategia_entrada']:4} {row['estrategia_salida']:4} "
                f"{int(row['total_operaciones']):>5} "
                f"{float(row['win_rate_pct']):>6.1f} "
                f"{float(row['ret_promedio']):>8.2f} "
                f"{float(row['max_dd']):>7.2f} "
                f"{float(row['profit_factor']):>7.2f} "
                f"{float(row['semanas_prom']):>5.1f}"
            )
    else:
        log("  Sin resultados con min 3 ops.")

    print()
    log("  Proximo paso:")
    log("    Etapa 7: python scripts/29_comparar_1d_vs_1w.py")


if __name__ == "__main__":
    main()
