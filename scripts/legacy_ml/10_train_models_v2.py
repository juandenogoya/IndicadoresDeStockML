"""
10_train_models_v2.py
Runner de la Fase 8 -- Entrenamiento del modelo V2 con 59 features.

Integra features_ml (29 features V1) + features_precio_accion (30 features PA)
para entrenar un nuevo challenger con mayor capacidad descriptiva.

Los modelos V1 en models/ NO se modifican.
Los nuevos modelos V2 se guardan en models_v2/.

Uso:
    python scripts/10_train_models_v2.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.trainer_v2 import (
    ejecutar_pipeline_ml_v2,
    FEATURE_COLS_V2,
)
from src.ml.trainer import FEATURE_COLS, ALGORITMOS_DISPONIBLES
from src.indicators.precio_accion import FEATURE_COLS_PA
from src.data.database import query_df, ejecutar_sql
from src.utils.config import MODELS_V2_DIR


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) "
            "VALUES (%s,%s,%s,%s)",
            ("10_train_v2", accion, detalle, estado)
        )
    except Exception:
        pass


def verificar_prerequisitos():
    """Verifica que features_ml y features_precio_accion tienen datos."""
    r = query_df("""
        SELECT
            (SELECT COUNT(*) FROM features_ml)             AS n_fm,
            (SELECT COUNT(DISTINCT ticker) FROM features_ml) AS tickers_fm,
            (SELECT COUNT(*) FROM features_precio_accion)  AS n_pa,
            (SELECT COUNT(DISTINCT ticker) FROM features_precio_accion) AS tickers_pa
    """)
    row = r.iloc[0]
    n_fm = int(row["n_fm"])
    n_pa = int(row["n_pa"])

    print(f"  features_ml          : {n_fm:,} filas ({int(row['tickers_fm'])} tickers)")
    print(f"  features_precio_accion: {n_pa:,} filas ({int(row['tickers_pa'])} tickers)")

    if n_fm == 0:
        raise RuntimeError("features_ml esta vacia. Ejecutar scripts 05-07 primero.")
    if n_pa == 0:
        raise RuntimeError(
            "features_precio_accion esta vacia. "
            "Ejecutar 'python scripts/09_calcular_precio_accion.py' primero."
        )

    # Verificar JOIN disponible
    join_count = query_df("""
        SELECT COUNT(*) AS n
        FROM features_ml fm
        JOIN features_precio_accion fpa
            ON fm.ticker = fpa.ticker AND fm.fecha = fpa.fecha
        WHERE fm.label_binario IS NOT NULL
    """)
    n_join = int(join_count.iloc[0]["n"])
    print(f"  Filas disponibles (JOIN): {n_join:,}")

    if n_join < 1000:
        raise RuntimeError(
            f"Solo {n_join} filas en el JOIN — datos insuficientes para training."
        )

    return n_join


def imprimir_resumen_deployment(resultados: dict):
    """Muestra el resumen del deployment V2 por sector."""
    deployment = resultados.get("deployment", [])
    if not deployment:
        return

    print(f"\n  DEPLOYMENT V2 — Decision por sector:")
    print(f"  {'Sector':<30} {'Ganador':<12} {'Alg':<20} {'F1_TEST_V2':>10} {'F1_BT_V2':>8}")
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
    print(f"\n  Global V2:")
    print(f"    Algoritmo campeón  : {gc['algoritmo'].upper()}")
    print(f"    F1_GANANCIA_TEST   : {gc['TEST']['f1_1']:.4f}")
    print(f"    F1_GANANCIA_BT     : {gc['BACKTEST']['f1_1']:.4f}")
    print(f"    ROC-AUC_TEST       : {gc['TEST']['roc_auc']:.4f}")


def main():
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  FASE 8 -- TRAINING MODELO V2 (59 FEATURES)")
    print("=" * 65)
    print(f"  Features V1         : {len(FEATURE_COLS)} (indicadores + scoring + z-scores)")
    print(f"  Features PA         : {len(FEATURE_COLS_PA)} (vela, patrones, rolling, volumen)")
    print(f"  Features V2 total   : {len(FEATURE_COLS_V2)}")
    print(f"  Algoritmos          : {', '.join(a.upper() for a in ALGORITMOS_DISPONIBLES)}")
    print(f"  Destino modelos     : {MODELS_V2_DIR}")
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
        resultados = ejecutar_pipeline_ml_v2(verbose=True)
        log_ejecucion(
            "TRAIN_V2",
            f"Pipeline V2 completo - {len(FEATURE_COLS_V2)} features",
            "OK"
        )
    except Exception as e:
        log_ejecucion("TRAIN_V2", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    imprimir_resumen_deployment(resultados)

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n{'='*65}")
    print(f"  V2 completado en {duracion}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print("\n  Siguiente paso: python scripts/11_comparar_versiones.py")


if __name__ == "__main__":
    main()
