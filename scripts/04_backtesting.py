"""
04_backtesting.py
Runner de la Fase 3 — Matriz Challenger 4×4 (entradas × salidas).

Ejecuta el backtesting completo para todos los activos en los 3 segmentos
(TRAIN / TEST / BACKTEST), calcula métricas y muestra el ranking.

Uso:
    python scripts/04_backtesting.py
    python scripts/04_backtesting.py --segmento TRAIN         (solo un segmento)
    python scripts/04_backtesting.py --tickers JPM GS         (solo algunos tickers)
    python scripts/04_backtesting.py --segmento BACKTEST      (evaluación final)
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import ALL_TICKERS
from src.backtesting.simulator import ejecutar_backtesting, SEGMENTOS
from src.backtesting.metrics import calcular_y_guardar_resultados, ranking_estrategias
from src.data.database import ejecutar_sql


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) VALUES (%s,%s,%s,%s)",
            ("04_backtesting", accion, detalle, estado)
        )
    except Exception:
        pass


def imprimir_ranking(segmento: str):
    print(f"\n{'='*75}")
    print(f"  RANKING ESTRATEGIAS — Segmento: {segmento}")
    print(f"{'='*75}")
    df = ranking_estrategias(segmento=segmento, min_ops=5)
    if df.empty:
        print("  Sin datos suficientes.")
        return
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Fase 3 - Backtesting Challenger 4x4")
    parser.add_argument("--tickers",   nargs="+", default=None)
    parser.add_argument("--segmento",  type=str,  default=None,
                        choices=["TRAIN", "TEST", "BACKTEST"],
                        help="Ejecutar solo un segmento específico")
    args = parser.parse_args()

    inicio   = datetime.now()
    tickers  = args.tickers or ALL_TICKERS
    segmentos = [args.segmento] if args.segmento else SEGMENTOS

    print("\n" + "=" * 75)
    print("  FASE 3 — BACKTESTING CHALLENGER 4×4")
    print("=" * 75)
    print(f"  Activos          : {len(tickers)}")
    print(f"  Segmentos        : {segmentos}")
    print(f"  Combinaciones    : 4 entradas × 4 salidas = 16 por activo/segmento")
    print(f"  Total combis     : {len(tickers) * 16 * len(segmentos)}")
    print(f"  Inicio           : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    # ── PASO 1: Simulación ────────────────────────────────────
    print("\n[PASO 1/2] SIMULACION DE OPERACIONES")
    try:
        df_ops = ejecutar_backtesting(
            tickers=tickers,
            segmentos=segmentos,
            guardar_db=True,
        )
        log_ejecucion("SIMULACION", f"{len(df_ops):,} operaciones", "OK")
    except Exception as e:
        log_ejecucion("SIMULACION", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    if df_ops.empty:
        print("[WARN] No se generaron operaciones.")
        sys.exit(0)

    # ── PASO 2: Métricas y resultados ─────────────────────────
    print("\n[PASO 2/2] CALCULO DE METRICAS")
    try:
        calcular_y_guardar_resultados(df_ops)
        log_ejecucion("METRICAS", "OK", "OK")
    except Exception as e:
        log_ejecucion("METRICAS", str(e), "ERROR")
        print(f"\n[ERROR Metricas] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Rankings ──────────────────────────────────────────────
    for seg in segmentos:
        imprimir_ranking(seg)

    # ── Resumen de operaciones por segmento y estrategia ──────
    print(f"\n{'='*75}")
    print("  RESUMEN DE OPERACIONES")
    print(f"{'='*75}")
    resumen = (
        df_ops.groupby(["segmento", "estrategia_entrada", "estrategia_salida"])
        .agg(
            ops=("retorno_pct", "count"),
            win_rate=("resultado", lambda x: (x == "GANANCIA").mean()),
            ret_total=("retorno_pct", "sum"),
            ret_prom=("retorno_pct", "mean"),
        )
        .round(2)
        .reset_index()
    )
    resumen["win_rate"] = (resumen["win_rate"] * 100).round(1)
    print(resumen.to_string(index=False))

    # ── Distribución de motivos de salida ─────────────────────
    print(f"\n{'='*75}")
    print("  MOTIVOS DE SALIDA (global)")
    print(f"{'='*75}")
    motivos = df_ops["motivo_salida"].value_counts()
    motivos_pct = (motivos / len(df_ops) * 100).round(1)
    for m, cnt in motivos.items():
        print(f"  {m:<20} {cnt:>6,}  ({motivos_pct[m]}%)")

    # ── Footer ────────────────────────────────────────────────
    fin      = datetime.now()
    duracion = (fin - inicio).seconds

    print(f"\n{'='*75}")
    print("  BACKTESTING COMPLETADO")
    print(f"{'='*75}")
    print(f"  Total operaciones simuladas : {len(df_ops):,}")
    print(f"  Tiempo total               : {duracion}s")
    print(f"  Fin                        : {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Resultados en DB           : tabla resultados_backtest")
    print("=" * 75)


if __name__ == "__main__":
    main()
