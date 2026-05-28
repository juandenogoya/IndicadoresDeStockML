"""
14_comparar_v1_v3.py
Compara metricas de modelos V1 (29 features) vs V3 (53 features = V1 + MS).

Lee resultados de la tabla resultados_modelos_ml y genera:
    1. Tabla de comparacion V1 vs V3 (F1, ROC-AUC, Accuracy por scope)
    2. Tabla de champions por scope
    3. Resumen de deployment V1 vs V3
    4. Informacion de las features de market structure
    5. Dictamen final: V3 mejora a V1?

Uso:
    python scripts/14_comparar_v1_v3.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import query_df
from src.ml.trainer_v3 import FEATURE_COLS_V3
from src.ml.trainer import FEATURE_COLS
from src.indicators.market_structure import FEATURE_COLS_MS


def main():
    inicio = datetime.now()

    print("\n" + "=" * 70)
    print("  COMPARACION V1 vs V3 - Market Structure Features")
    print("=" * 70)
    print(f"  V1 : {len(FEATURE_COLS)} features (indicadores tecnicos + scoring + z-scores)")
    print(f"  V3 : {len(FEATURE_COLS_V3)} features (V1 + {len(FEATURE_COLS_MS)} market structure)")
    print(f"  Ejecutado: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Metricas comparadas (TEST + BACKTEST) ─────────────────────
    df = query_df("""
        SELECT
            scope, algoritmo, segmento,
            modelo_version,
            n_features,
            f1_1        AS f1_ganancia,
            roc_auc,
            accuracy,
            precision_1,
            recall_1
        FROM resultados_modelos_ml
        WHERE modelo_version IN ('v1', 'v3')
        ORDER BY scope, algoritmo, segmento, modelo_version
    """)

    if df.empty:
        print("\n[ERROR] No hay datos de V1 o V3 en resultados_modelos_ml.")
        print("  Ejecutar scripts 05-07 (V1) y 13 (V3) primero.")
        sys.exit(1)

    v1 = df[df["modelo_version"] == "v1"].copy()
    v3 = df[df["modelo_version"] == "v3"].copy()

    print(f"\n  Registros encontrados: V1={len(v1)}  V3={len(v3)}")

    # ── 2. Comparacion TEST por scope x algoritmo ─────────────────────
    v1_te = v1[v1["segmento"] == "TEST"].set_index(["scope", "algoritmo"])
    v3_te = v3[v3["segmento"] == "TEST"].set_index(["scope", "algoritmo"])

    claves_comunes = v1_te.index.intersection(v3_te.index)

    print(f"\n  COMPARACION TEST - V1 vs V3 (F1_GANANCIA)")
    print(f"  {'Scope':<30} {'Alg':<6} {'V1':>8} {'V3':>8} {'Delta':>8} {'Mejor':>6}")
    print("  " + "-" * 66)

    deltas = []
    for (scope, alg) in sorted(claves_comunes):
        f1_v1 = float(v1_te.loc[(scope, alg), "f1_ganancia"])
        f1_v3 = float(v3_te.loc[(scope, alg), "f1_ganancia"])
        delta = f1_v3 - f1_v1
        deltas.append(delta)
        mejor = "V3 +" if delta > 0.005 else ("V1" if delta < -0.005 else "~=")
        print(f"  {scope:<30} {alg:<6} {f1_v1:>8.4f} {f1_v3:>8.4f} {delta:>+8.4f} {mejor:>6}")

    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        print(f"\n  Media delta F1 TEST : {avg_delta:+.4f}")
        print(f"  V3 > V1 en {sum(d > 0.005 for d in deltas)}/{len(deltas)} combinaciones")

    # ── 3. Comparacion por scope (champion = mejor algoritmo en TEST) ──
    print(f"\n  CHAMPIONS POR SCOPE (mejor algoritmo en TEST)")
    print(f"  {'Scope':<30} {'Vers':>4} {'Alg':<6} {'F1_TEST':>8} {'F1_BT':>8} {'ROC':>8}")
    print("  " + "-" * 70)

    for scope in sorted(df["scope"].unique()):
        for ver in ["v1", "v3"]:
            sub = df[(df["scope"] == scope) & (df["modelo_version"] == ver) & (df["segmento"] == "TEST")]
            if sub.empty:
                continue
            best = sub.loc[sub["f1_ganancia"].idxmax()]
            bt_row = df[
                (df["scope"] == scope) &
                (df["modelo_version"] == ver) &
                (df["algoritmo"] == best["algoritmo"]) &
                (df["segmento"] == "BACKTEST")
            ]
            f1_bt = float(bt_row["f1_ganancia"].values[0]) if len(bt_row) else float("nan")
            print(
                f"  {scope:<30} {ver:>4} {best['algoritmo']:<6} "
                f"{float(best['f1_ganancia']):>8.4f} "
                f"{f1_bt:>8.4f} "
                f"{float(best['roc_auc']):>8.4f}"
            )

    # ── 4. Deployment V1 vs V3 ────────────────────────────────────────
    desp = query_df("""
        SELECT scope, modelo_version, tipo, algoritmo, f1_test, f1_backtest, roc_auc_test
        FROM modelos_produccion
        WHERE modelo_version IN ('v1', 'v3')
        ORDER BY scope, modelo_version
    """)

    if not desp.empty:
        print(f"\n  DEPLOYMENT - modelos_produccion")
        print(f"  {'Scope':<30} {'Ver':>4} {'Tipo':<10} {'Alg':<20} {'F1_TEST':>8} {'F1_BT':>8}")
        print("  " + "-" * 82)
        for _, row in desp.iterrows():
            f1t = float(row["f1_test"])  if row["f1_test"]  is not None else float("nan")
            f1b = float(row["f1_backtest"]) if row["f1_backtest"] is not None else float("nan")
            print(
                f"  {str(row['scope']):<30} {str(row['modelo_version']):>4} "
                f"{str(row['tipo']):<10} {str(row['algoritmo']):<20} "
                f"{f1t:>8.4f} {f1b:>8.4f}"
            )

    # ── 5. Info features Market Structure ─────────────────────────────
    print(f"\n  FEATURES MARKET STRUCTURE ({len(FEATURE_COLS_MS)} features)")
    print(f"  Ventana N=5  (tactico, ~11 barras):")
    ms5  = [f for f in FEATURE_COLS_MS if "_5" in f]
    ms10 = [f for f in FEATURE_COLS_MS if "_10" in f]
    print(f"    {', '.join(ms5)}")
    print(f"  Ventana N=10 (estrategico, ~21 barras, alineado con retorno_20d):")
    print(f"    {', '.join(ms10)}")

    # ── 6. Cobertura de la tabla market structure ──────────────────────
    cov = query_df("""
        SELECT
            COUNT(*)                          AS total_barras,
            SUM(CASE WHEN estructura_5  != 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_est5,
            SUM(CASE WHEN estructura_10 != 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_est10,
            SUM(COALESCE(bos_bull_5, 0) + COALESCE(bos_bear_5, 0))   AS total_bos5,
            SUM(COALESCE(choch_bull_5, 0) + COALESCE(choch_bear_5, 0)) AS total_choch5,
            SUM(COALESCE(bos_bull_10, 0) + COALESCE(bos_bear_10, 0))  AS total_bos10,
            SUM(COALESCE(choch_bull_10, 0) + COALESCE(choch_bear_10, 0)) AS total_choch10
        FROM features_market_structure
    """)
    r = cov.iloc[0]
    print(f"\n  COBERTURA features_market_structure:")
    print(f"    Total barras      : {int(r['total_barras']):,}")
    print(f"    Estructura N=5    : {float(r['pct_est5']):.1f}% definida")
    print(f"    Estructura N=10   : {float(r['pct_est10']):.1f}% definida")
    print(f"    Eventos BOS N=5   : {int(r['total_bos5']):,}")
    print(f"    Eventos CHoCH N=5 : {int(r['total_choch5']):,}")
    print(f"    Eventos BOS N=10  : {int(r['total_bos10']):,}")
    print(f"    Eventos CHoCH N=10: {int(r['total_choch10']):,}")

    # ── 7. Dictamen final ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  DICTAMEN FINAL")
    print("=" * 70)

    if not deltas:
        print("  [SIN DATOS] No hay suficientes datos para comparar V1 vs V3.")
    else:
        avg_delta_test = sum(deltas) / len(deltas)

        # Backtest deltas
        v1_bt = v1[v1["segmento"] == "BACKTEST"].set_index(["scope", "algoritmo"])
        v3_bt = v3[v3["segmento"] == "BACKTEST"].set_index(["scope", "algoritmo"])
        bt_keys = v1_bt.index.intersection(v3_bt.index)
        bt_deltas = []
        for (scope, alg) in bt_keys:
            f1_v1b = float(v1_bt.loc[(scope, alg), "f1_ganancia"])
            f1_v3b = float(v3_bt.loc[(scope, alg), "f1_ganancia"])
            bt_deltas.append(f1_v3b - f1_v1b)
        avg_delta_bt = sum(bt_deltas) / len(bt_deltas) if bt_deltas else 0.0

        print(f"\n  V1  : {len(FEATURE_COLS)} features (baseline)")
        print(f"  V3  : {len(FEATURE_COLS_V3)} features (+{len(FEATURE_COLS_MS)} market structure)")
        print(f"\n  Media delta F1_GANANCIA TEST     : {avg_delta_test:+.4f}")
        print(f"  Media delta F1_GANANCIA BACKTEST : {avg_delta_bt:+.4f}")
        mejoras_test = sum(d > 0.005 for d in deltas)
        print(f"  Combinaciones V3 > V1 en TEST    : {mejoras_test}/{len(deltas)}")

        if avg_delta_test > 0.005:
            print(f"\n  >> DICTAMEN: POSITIVO <<")
            print(f"     V3 mejora sobre V1 en TEST (dF1 = {avg_delta_test:+.4f})")
            print(f"     Las features de market structure aportan valor predictivo.")
        elif avg_delta_test > 0:
            print(f"\n  >> DICTAMEN: NEUTRAL <<")
            print(f"     V3 mejora marginalmente sobre V1 en TEST (dF1 = {avg_delta_test:+.4f})")
            print(f"     La mejora es estadisticamente poco significativa.")
        else:
            print(f"\n  >> DICTAMEN: NEGATIVO <<")
            print(f"     V3 NO mejora sobre V1 en TEST (dF1 = {avg_delta_test:+.4f})")
            print(f"     Las features de market structure no aportan valor adicional.")

    print("=" * 70)

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n  Comparacion completada en {duracion}s")


if __name__ == "__main__":
    main()
