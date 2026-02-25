"""
13_train_models_v3.py
Runner de la Fase 10 -- Entrenamiento del modelo V3 con 53 features.

Integra features_ml (29 features V1) + features_market_structure (24 features MS)
para entrenar un nuevo challenger con estructura de precio multi-barra.

Los modelos V1 en models/ y V2 en models_v2/ NO se modifican.
Los nuevos modelos V3 se guardan en models_v3/.

Uso:
    python scripts/13_train_models_v3.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.trainer_v3 import (
    ejecutar_pipeline_ml_v3,
    FEATURE_COLS_V3,
)
from src.ml.trainer import FEATURE_COLS, ALGORITMOS_DISPONIBLES
from src.indicators.market_structure import FEATURE_COLS_MS
from src.data.database import query_df, ejecutar_sql
from src.utils.config import MODELS_V3_DIR


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) "
            "VALUES (%s,%s,%s,%s)",
            ("13_train_v3", accion, detalle, estado)
        )
    except Exception:
        pass


def verificar_prerequisitos():
    """Verifica que features_ml y features_market_structure tienen datos."""
    r = query_df("""
        SELECT
            (SELECT COUNT(*) FROM features_ml)                    AS n_fm,
            (SELECT COUNT(DISTINCT ticker) FROM features_ml)       AS tickers_fm,
            (SELECT COUNT(*) FROM features_market_structure)       AS n_ms,
            (SELECT COUNT(DISTINCT ticker) FROM features_market_structure) AS tickers_ms
    """)
    row = r.iloc[0]
    n_fm = int(row["n_fm"])
    n_ms = int(row["n_ms"])

    print(f"  features_ml              : {n_fm:,} filas ({int(row['tickers_fm'])} tickers)")
    print(f"  features_market_structure: {n_ms:,} filas ({int(row['tickers_ms'])} tickers)")

    if n_fm == 0:
        raise RuntimeError("features_ml esta vacia. Ejecutar scripts 05-07 primero.")
    if n_ms == 0:
        raise RuntimeError(
            "features_market_structure esta vacia. "
            "Ejecutar 'python scripts/12_calcular_market_structure.py' primero."
        )

    # Verificar JOIN disponible
    join_count = query_df("""
        SELECT COUNT(*) AS n
        FROM features_ml fm
        JOIN features_market_structure fms
            ON fm.ticker = fms.ticker AND fm.fecha = fms.fecha
        WHERE fm.label_binario IS NOT NULL
    """)
    n_join = int(join_count.iloc[0]["n"])
    print(f"  Filas disponibles (JOIN) : {n_join:,}")

    if n_join < 1000:
        raise RuntimeError(
            f"Solo {n_join} filas en el JOIN - datos insuficientes para training."
        )

    return n_join


def imprimir_resumen_deployment(resultados: dict):
    """Muestra el resumen del deployment V3 por sector."""
    deployment = resultados.get("deployment", [])
    if not deployment:
        return

    print(f"\n  DEPLOYMENT V3 - Decision por sector:")
    print(f"  {'Sector':<30} {'Ganador':<12} {'Alg':<20} {'F1_TEST_V3':>10} {'F1_BT_V3':>8}")
    print("  " + "-" * 82)

    for d in deployment:
        sector = d["sector"]
        tipo   = d["winner_tipo"].upper()
        alg    = d["winner_alg"]
        f1_te  = d["winner_f1_te"]
        f1_bt  = d["winner_f1_bt"]
        print(f"  {sector:<30} {tipo:<12} {alg:<20} {f1_te:>10.4f} {f1_bt:>8.4f}")

    # Global
    gc = resultados["global"]["champion"]
    print(f"\n  Global V3:")
    print(f"    Algoritmo campeon  : {gc['algoritmo'].upper()}")
    print(f"    F1_GANANCIA_TEST   : {gc['TEST']['f1_1']:.4f}")
    print(f"    F1_GANANCIA_BT     : {gc['BACKTEST']['f1_1']:.4f}")
    print(f"    ROC-AUC_TEST       : {gc['TEST']['roc_auc']:.4f}")


def main():
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  FASE 10 -- TRAINING MODELO V3 (53 FEATURES)")
    print("=" * 65)
    print(f"  Features V1         : {len(FEATURE_COLS)} (indicadores + scoring + z-scores)")
    print(f"  Features MS         : {len(FEATURE_COLS_MS)} (swings, estructura, BOS/CHoCH)")
    print(f"  Features V3 total   : {len(FEATURE_COLS_V3)}")
    print(f"  Algoritmos          : {', '.join(a.upper() for a in ALGORITMOS_DISPONIBLES)}")
    print(f"  Destino modelos     : {MODELS_V3_DIR}")
    print(f"  Inicio              : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("\n  Verificando prerequisitos...")
    try:
        n_join = verificar_prerequisitos()
    except RuntimeError as e:
        print(f"\n[ERROR prerequisito] {e}")
        sys.exit(1)

    print(f"\n  Prerequisitos OK ({n_join:,} filas disponibles para training)")

    try:
        resultados = ejecutar_pipeline_ml_v3(verbose=True)
        log_ejecucion(
            "TRAIN_V3",
            f"Pipeline V3 completo - {len(FEATURE_COLS_V3)} features",
            "OK"
        )
    except Exception as e:
        log_ejecucion("TRAIN_V3", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    imprimir_resumen_deployment(resultados)

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n{'='*65}")
    print(f"  V3 completado en {duracion}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print("\n  Siguiente paso: python scripts/14_comparar_v1_v3.py")


if __name__ == "__main__":
    main()
