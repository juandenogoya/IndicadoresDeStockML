"""
06_build_feature_store.py
Runner de la Fase 4b — Construcción del Feature Store para ML.

Compila todos los features (indicadores + scoring + sector Z-scores)
con retornos futuros y etiquetas, y los persiste en features_ml.

Uso:
    python scripts/06_build_feature_store.py
    python scripts/06_build_feature_store.py --no-db   (solo calcular, sin guardar)
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.feature_store import construir_feature_store, cargar_features_entrenamiento
from src.data.database import ejecutar_sql


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) VALUES (%s,%s,%s,%s)",
            ("06_feature_store", accion, detalle, estado)
        )
    except Exception:
        pass


def imprimir_resumen_features(df):
    """Imprime un resumen estadístico del feature store construido."""
    print(f"\n{'='*65}")
    print("  RESUMEN DEL FEATURE STORE")
    print(f"{'='*65}")

    # Por sector
    print("\n  Distribución por sector y segmento:")
    resumen = (
        df[df["label"].notna()]
        .groupby(["sector", "segmento"])
        .agg(
            filas=("ticker", "count"),
            ganancia=("label", lambda x: (x == "GANANCIA").mean()),
        )
        .round(3)
    )
    print(resumen.to_string())

    # Features con nulos
    feature_cols = [
        "rsi14", "macd_hist", "adx", "score_ponderado",
        "z_rsi_sector", "z_retorno_1d_sector", "pct_long_sector"
    ]
    print("\n  Nulos en features principales:")
    for col in feature_cols:
        if col in df.columns:
            n_nulos = df[col].isna().sum()
            if n_nulos > 0:
                print(f"    {col:<30}: {n_nulos:>6,} nulos")

    # Rango de fechas global
    print(f"\n  Rango de fechas : {df['fecha'].min().date()} / {df['fecha'].max().date()}")
    print(f"  Total features  : {df.shape[1]} columnas")
    print(f"  Total filas     : {len(df):,}")


def main():
    parser = argparse.ArgumentParser(description="Fase 4b - Feature Store ML")
    parser.add_argument("--no-db", action="store_true",
                        help="Calcular sin persistir en la base de datos")
    args = parser.parse_args()

    guardar_db = not args.no_db
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  FASE 4b — FEATURE STORE ML")
    print("=" * 65)
    print(f"  Guardar en DB    : {'Sí' if guardar_db else 'No (modo dry-run)'}")
    print(f"  Inicio           : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Pipeline principal ────────────────────────────────────
    print("\n[1/1] Construyendo feature store...")
    try:
        df = construir_feature_store(guardar_db=guardar_db)
        log_ejecucion("BUILD", f"{len(df):,} registros", "OK")

    except Exception as e:
        log_ejecucion("BUILD", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Resumen ───────────────────────────────────────────────
    imprimir_resumen_features(df)

    # ── Verificación final: cargar desde DB ───────────────────
    if guardar_db:
        print(f"\n{'='*65}")
        print("  VERIFICACIÓN DESDE DB")
        print(f"{'='*65}")
        try:
            for seg in ["TRAIN", "TEST", "BACKTEST"]:
                df_seg = cargar_features_entrenamiento(segmento=seg, solo_con_label=True)
                ganancia_rate = (df_seg["label"] == "GANANCIA").mean()
                print(
                    f"  {seg:<10}: {len(df_seg):>6,} filas con label  "
                    f"| GANANCIA rate: {ganancia_rate*100:.1f}%"
                )
        except Exception as e:
            print(f"  [WARN] No se pudo verificar desde DB: {e}")

    fin      = datetime.now()
    duracion = (fin - inicio).seconds

    print(f"\n{'='*65}")
    print("  FEATURE STORE COMPLETADO")
    print(f"{'='*65}")
    print(f"  Tiempo total  : {duracion}s")
    print(f"  Fin           : {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    if guardar_db:
        print(f"  Datos en DB   : tabla features_ml")
    print("=" * 65)

    print("\n  Próximo paso → Fase 5: Entrenamiento de modelos ML")
    print("    python scripts/07_train_models.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
