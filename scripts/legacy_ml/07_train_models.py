"""
07_train_models.py
Runner de la Fase 5 — Entrenamiento ML Challenger 3 niveles.

Nivel 1: Modelo Global       (RF vs XGB vs LGBM, todos los sectores)
Nivel 2: Modelos Sectoriales (RF vs XGB vs LGBM por sector)
Nivel 3: Challenger Final    (Global Champion vs Sector Champion por sector)

Automotive queda fuera del challenger sectorial (3 tickers insuficientes).
Siempre usará el Global Champion.

Uso:
    python scripts/07_train_models.py
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.trainer import (
    ejecutar_pipeline_ml,
    ALGORITMOS_DISPONIBLES,
    FEATURE_COLS,
    HAS_XGB,
    HAS_LGBM,
)
from src.data.database import ejecutar_sql, query_df
from src.utils.config import SECTORES_ML, MODELS_DIR


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) VALUES (%s,%s,%s,%s)",
            ("07_train_models", accion, detalle, estado)
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Tablas de resultados formateadas
# ─────────────────────────────────────────────────────────────

def imprimir_tabla_metricas():
    """Lee resultados de DB y muestra tabla comparativa por scope."""
    sql = """
        SELECT scope, algoritmo, segmento,
               ROUND(f1_1 * 100, 1)        AS f1_g,
               ROUND(roc_auc * 100, 1)     AS roc,
               ROUND(precision_1 * 100, 1) AS prec,
               ROUND(recall_1 * 100, 1)    AS rec,
               n_filas
        FROM resultados_modelos_ml
        ORDER BY scope, segmento,
                 CASE segmento WHEN 'TRAIN' THEN 1 WHEN 'TEST' THEN 2 ELSE 3 END,
                 f1_1 DESC
    """
    df = query_df(sql)

    print(f"\n{'='*75}")
    print("  RESULTADOS CHALLENGER — F1(GANANCIA) por Scope x Algoritmo")
    print(f"{'='*75}")

    for scope in df["scope"].unique():
        sub = df[df["scope"] == scope]
        print(f"\n  Scope: {scope}  ({sub['n_filas'].iloc[0]:,} filas TEST aprox.)")
        print(f"  {'Algoritmo':<8} {'Segmento':<10} {'F1_G%':<8} {'ROC%':<8} "
              f"{'Prec%':<8} {'Rec%':<6} {'Filas'}")
        print(f"  {'-'*8} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
        for _, row in sub.iterrows():
            print(
                f"  {row['algoritmo']:<8} {row['segmento']:<10} "
                f"{row['f1_g']:<8} {row['roc']:<8} "
                f"{row['prec']:<8} {row['rec']:<6} {int(row['n_filas'])}"
            )


def imprimir_challenger_final(deployment: list):
    """Imprime la tabla del challenger final Global vs Sectorial."""
    print(f"\n{'='*75}")
    print("  CHALLENGER FINAL — Global vs Sectorial")
    print(f"{'='*75}")
    print(
        f"  {'Sector':<26} {'Global_F1%':<12} {'Sector_F1%':<12} "
        f"{'Winner':<12} {'Win_F1_TEST%':<14} {'Win_F1_BT%'}"
    )
    print(f"  {'-'*26} {'-'*11} {'-'*11} {'-'*11} {'-'*13} {'-'*10}")

    for d in deployment:
        delta = d["sector_f1_te"] - d["global_f1_te"]
        delta_str = f"(+{delta*100:.1f}%)" if delta > 0 else f"({delta*100:.1f}%)"
        print(
            f"  {d['sector']:<26} "
            f"{d['global_f1_te']*100:<12.1f}"
            f"{d['sector_f1_te']*100:<12.1f}"
            f"{d['winner_tipo'].upper():<12} "
            f"{d['winner_f1_te']*100:<14.1f}"
            f"{d['winner_f1_bt']*100:.1f}  {delta_str}"
        )


def imprimir_plan_despliegue():
    """Lee modelos_produccion y muestra el plan de despliegue final."""
    sql = """
        SELECT scope, tipo, algoritmo,
               ROUND(f1_test * 100, 1)    AS f1_test,
               ROUND(f1_backtest * 100, 1) AS f1_bt,
               ROUND(roc_auc_test * 100, 1) AS roc
        FROM modelos_produccion
        WHERE activo = TRUE
        ORDER BY scope
    """
    df = query_df(sql)

    print(f"\n{'='*75}")
    print("  PLAN DE DESPLIEGUE — Modelo por sector")
    print(f"{'='*75}")
    print(
        f"  {'Sector/Scope':<26} {'Tipo':<12} {'Algoritmo':<14} "
        f"{'F1_TEST%':<10} {'F1_BT%':<10} {'ROC%'}"
    )
    print(f"  {'-'*26} {'-'*12} {'-'*14} {'-'*10} {'-'*10} {'-'*6}")

    for _, row in df.iterrows():
        f1_te = f"{row['f1_test']:.1f}%"   if row['f1_test']    else "N/A"
        f1_bt = f"{row['f1_bt']:.1f}%"     if row['f1_bt']      else "N/A"
        roc   = f"{row['roc']:.1f}%"       if row['roc']        else "N/A"
        print(
            f"  {row['scope']:<26} {row['tipo']:<12} {row['algoritmo']:<14} "
            f"{f1_te:<10} {f1_bt:<10} {roc}"
        )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fase 5 - Entrenamiento ML Challenger")
    args = parser.parse_args()

    inicio = datetime.now()

    print("\n" + "=" * 75)
    print("  FASE 5 — ENTRENAMIENTO ML (RF x XGB x LGBM)")
    print("=" * 75)
    print(f"  Modelo global       : todos los sectores (19 tickers)")
    print(f"  Sectores propios    : {', '.join(SECTORES_ML)}")
    print(f"  Automotive          : Global Champion (sin modelo propio)")
    print(f"  Algoritmos activos  : {', '.join(ALGORITMOS_DISPONIBLES)}")
    if not HAS_XGB:
        print("  [WARN] XGBoost no disponible. Instalar: pip install xgboost")
    if not HAS_LGBM:
        print("  [WARN] LightGBM no disponible. Instalar: pip install lightgbm")
    print(f"  Features totales    : {len(FEATURE_COLS)}")
    print(f"  Modelos en          : {MODELS_DIR}")
    print(f"  Inicio              : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    # ── Pipeline principal ────────────────────────────────────
    try:
        resultados = ejecutar_pipeline_ml(verbose=True)
        log_ejecucion("TRAIN", "Pipeline ML completado OK", "OK")
    except Exception as e:
        log_ejecucion("TRAIN", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Tablas de resultados ──────────────────────────────────
    imprimir_tabla_metricas()
    imprimir_challenger_final(resultados["deployment"])
    imprimir_plan_despliegue()

    # ── Footer ────────────────────────────────────────────────
    fin      = datetime.now()
    duracion = (fin - inicio).seconds

    print(f"\n{'='*75}")
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"{'='*75}")
    print(f"  Modelos guardados   : {MODELS_DIR}")
    print(f"  Metricas en DB      : resultados_modelos_ml")
    print(f"  Despliegue en DB    : modelos_produccion")
    print(f"  Tiempo total        : {duracion}s")
    print(f"  Fin                 : {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
