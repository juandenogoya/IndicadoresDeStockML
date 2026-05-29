"""
11_comparar_versiones.py
Compara resultados del modelo V1 (29 features) vs V2 (59 features).

Tablas consultadas:
    resultados_modelos_ml  -- metricas por scope/algoritmo/segmento/version
    modelos_produccion     -- deployment decision por scope/version

Uso:
    python scripts/11_comparar_versiones.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import query_df
from src.ml.trainer import FEATURE_COLS
from src.ml.trainer_v2 import FEATURE_COLS_V2
from src.indicators.precio_accion import FEATURE_COLS_PA


# ─────────────────────────────────────────────────────────────
# Tablas de comparacion
# ─────────────────────────────────────────────────────────────

def imprimir_comparacion_metricas():
    """Tabla V1 vs V2 por scope / algoritmo / segmento."""
    df = query_df("""
        WITH v1 AS (
            SELECT scope, algoritmo, segmento,
                   n_features, accuracy, f1_w, f1_1, roc_auc
            FROM resultados_modelos_ml
            WHERE modelo_version = 'v1'
        ),
        v2 AS (
            SELECT scope, algoritmo, segmento,
                   n_features, accuracy, f1_w, f1_1, roc_auc
            FROM resultados_modelos_ml
            WHERE modelo_version = 'v2'
        )
        SELECT
            v1.scope,
            v1.algoritmo,
            v1.segmento,
            v1.n_features   AS nf_v1,
            v2.n_features   AS nf_v2,
            ROUND(v1.f1_1::NUMERIC, 4)           AS f1_v1,
            ROUND(v2.f1_1::NUMERIC, 4)           AS f1_v2,
            ROUND((v2.f1_1 - v1.f1_1)::NUMERIC, 4) AS delta_f1,
            ROUND(v1.roc_auc::NUMERIC, 4)        AS roc_v1,
            ROUND(v2.roc_auc::NUMERIC, 4)        AS roc_v2,
            ROUND((v2.roc_auc - v1.roc_auc)::NUMERIC, 4) AS delta_roc
        FROM v1
        JOIN v2
            ON v1.scope     = v2.scope
           AND v1.algoritmo = v2.algoritmo
           AND v1.segmento  = v2.segmento
        ORDER BY v1.scope, v1.algoritmo,
                 CASE v1.segmento
                     WHEN 'TRAIN'    THEN 1
                     WHEN 'TEST'     THEN 2
                     WHEN 'BACKTEST' THEN 3
                 END
    """)

    if df.empty:
        print("  [!] Sin datos comparables (ejecutar 10 primero)")
        return

    print(f"\n  {'Scope':<32} {'Alg':<5} {'Seg':<10} "
          f"{'F1_v1':>6} {'F1_v2':>6} {'dF1':>7}  "
          f"{'ROC_v1':>6} {'ROC_v2':>6} {'dROC':>7}")
    print("  " + "-" * 92)

    current_scope = None
    for _, r in df.iterrows():
        if r["scope"] != current_scope:
            if current_scope is not None:
                print()
            current_scope = r["scope"]

        d_f1  = float(r["delta_f1"])
        d_roc = float(r["delta_roc"])
        marker_f1  = "+" if d_f1  > 0.005 else ("-" if d_f1  < -0.005 else " ")
        marker_roc = "+" if d_roc > 0.005 else ("-" if d_roc < -0.005 else " ")

        print(f"  {str(r['scope']):<32} {r['algoritmo']:<5} {r['segmento']:<10} "
              f"{float(r['f1_v1']):>6.4f} {float(r['f1_v2']):>6.4f} "
              f"{marker_f1}{d_f1:>6.4f}  "
              f"{float(r['roc_v1']):>6.4f} {float(r['roc_v2']):>6.4f} "
              f"{marker_roc}{d_roc:>6.4f}")


def imprimir_resumen_champion():
    """Resumen del campeon por scope (TEST y BACKTEST) V1 vs V2."""
    df = query_df("""
        WITH best AS (
            SELECT
                m.scope,
                m.modelo_version,
                m.algoritmo,
                m.segmento,
                m.f1_1,
                m.roc_auc,
                ROW_NUMBER() OVER (
                    PARTITION BY m.scope, m.modelo_version, m.segmento
                    ORDER BY m.f1_1 DESC
                ) AS rnk
            FROM resultados_modelos_ml m
            WHERE m.segmento IN ('TEST', 'BACKTEST')
        )
        SELECT
            v1.scope,
            v1.segmento,
            v1.algoritmo     AS alg_v1,
            v2.algoritmo     AS alg_v2,
            ROUND(v1.f1_1::NUMERIC, 4)              AS f1_v1,
            ROUND(v2.f1_1::NUMERIC, 4)              AS f1_v2,
            ROUND((v2.f1_1 - v1.f1_1)::NUMERIC, 4) AS delta_f1,
            ROUND(v1.roc_auc::NUMERIC, 4)           AS roc_v1,
            ROUND(v2.roc_auc::NUMERIC, 4)           AS roc_v2
        FROM best v1
        JOIN best v2
            ON v1.scope   = v2.scope
           AND v1.segmento = v2.segmento
        WHERE v1.modelo_version = 'v1'
          AND v2.modelo_version = 'v2'
          AND v1.rnk = 1
          AND v2.rnk = 1
        ORDER BY v1.scope,
                 CASE v1.segmento WHEN 'TEST' THEN 1 ELSE 2 END
    """)

    if df.empty:
        print("  [!] Sin datos comparables")
        return

    print(f"\n  {'Scope':<32} {'Seg':<10} {'Alg_v1':<8} {'Alg_v2':<8} "
          f"{'F1_v1':>7} {'F1_v2':>7} {'Delta':>8}  {'ROC_v1':>7} {'ROC_v2':>7}")
    print("  " + "-" * 100)

    mejoras = 0
    for _, r in df.iterrows():
        d_f1 = float(r["delta_f1"])
        tag   = "[+]" if d_f1 > 0.005 else ("[-]" if d_f1 < -0.005 else "[ ]")
        if d_f1 > 0.005 and r["segmento"] in ("TEST", "BACKTEST"):
            mejoras += 1
        print(f"  {str(r['scope']):<32} {r['segmento']:<10} {r['alg_v1']:<8} {r['alg_v2']:<8} "
              f"{float(r['f1_v1']):>7.4f} {float(r['f1_v2']):>7.4f} "
              f"{tag}{d_f1:>6.4f}  "
              f"{float(r['roc_v1']):>7.4f} {float(r['roc_v2']):>7.4f}")

    print(f"\n  Scopes con mejora significativa en TEST/BT: {mejoras}")


def imprimir_comparacion_deployment():
    """Compara el plan de deployment V1 vs V2."""
    df = query_df("""
        SELECT
            v1.scope,
            v1.tipo          AS tipo_v1,
            v2.tipo          AS tipo_v2,
            v1.algoritmo     AS alg_v1,
            v2.algoritmo     AS alg_v2,
            v1.n_features    AS nf_v1,
            v2.n_features    AS nf_v2,
            ROUND(v1.f1_test::NUMERIC, 4)              AS f1_test_v1,
            ROUND(v2.f1_test::NUMERIC, 4)              AS f1_test_v2,
            ROUND((v2.f1_test - v1.f1_test)::NUMERIC, 4) AS delta_f1_test,
            ROUND(v1.f1_backtest::NUMERIC, 4)          AS f1_bt_v1,
            ROUND(v2.f1_backtest::NUMERIC, 4)          AS f1_bt_v2
        FROM modelos_produccion v1
        JOIN modelos_produccion v2
            ON v1.scope = v2.scope
        WHERE v1.modelo_version = 'v1'
          AND v2.modelo_version = 'v2'
        ORDER BY v1.scope
    """)

    if df.empty:
        print("  [!] Sin datos de deployment para comparar")
        return

    print(f"\n  {'Scope':<32} {'Tipo_v1':<10} {'Tipo_v2':<10} "
          f"{'nF_v1':>5} {'nF_v2':>5}  "
          f"{'F1_TEST_v1':>10} {'F1_TEST_v2':>10} {'Delta':>7}  "
          f"{'F1_BT_v1':>8} {'F1_BT_v2':>8}")
    print("  " + "-" * 112)

    for _, r in df.iterrows():
        d = r["delta_f1_test"]
        try:
            d_str = f"{float(d):>+7.4f}" if d is not None and str(d) != "nan" else "    N/A"
        except Exception:
            d_str = "    N/A"

        f1_te_v1 = f"{float(r['f1_test_v1']):.4f}" if r["f1_test_v1"] is not None and str(r["f1_test_v1"]) != "nan" else "  N/A"
        f1_te_v2 = f"{float(r['f1_test_v2']):.4f}" if r["f1_test_v2"] is not None and str(r["f1_test_v2"]) != "nan" else "  N/A"
        f1_bt_v1 = f"{float(r['f1_bt_v1']):.4f}"   if r["f1_bt_v1"]   is not None and str(r["f1_bt_v1"])   != "nan" else "  N/A"
        f1_bt_v2 = f"{float(r['f1_bt_v2']):.4f}"   if r["f1_bt_v2"]   is not None and str(r["f1_bt_v2"])   != "nan" else "  N/A"

        print(f"  {str(r['scope']):<32} {str(r['tipo_v1']):<10} {str(r['tipo_v2']):<10} "
              f"{int(r['nf_v1']):>5} {int(r['nf_v2']):>5}  "
              f"{f1_te_v1:>10} {f1_te_v2:>10} {d_str}  "
              f"{f1_bt_v1:>8} {f1_bt_v2:>8}")


def imprimir_top_nuevas_features():
    """Muestra la importancia media de features PA vs V1 en el scope global."""
    # Verificar si hay datos para v2
    check = query_df(
        "SELECT COUNT(*) AS n FROM resultados_modelos_ml WHERE modelo_version = 'v2'"
    )
    if int(check.iloc[0]["n"]) == 0:
        return

    print(f"\n  Features PA (30 nuevas) en contexto del modelo V2:")
    print(f"  V1 features: {len(FEATURE_COLS)} | PA features: {len(FEATURE_COLS_PA)} | Total V2: {len(FEATURE_COLS_V2)}")
    print(f"\n  Grupos de features PA:")
    grupos = {
        "Anatomia de vela  (9)": [
            "body_pct", "body_ratio", "upper_shadow_pct", "lower_shadow_pct",
            "es_alcista", "gap_apertura_pct", "rango_diario_pct", "rango_rel_atr", "clv"
        ],
        "Patrones clasicos (8)": [
            "patron_doji", "patron_hammer", "patron_shooting_star", "patron_marubozu",
            "patron_engulfing_bull", "patron_engulfing_bear", "inside_bar", "outside_bar"
        ],
        "Estructura rolling(7)": [
            "body_pct_ma5", "velas_alcistas_5d", "velas_alcistas_10d",
            "rango_expansion", "dist_max_20d", "dist_min_20d", "pos_rango_20d"
        ],
        "Volumen direccional(6)": [
            "vol_ratio_5d", "vol_spike", "up_vol_5d",
            "chaikin_mf_20", "vol_price_confirm", "vol_price_diverge"
        ],
    }
    for grupo, feats in grupos.items():
        print(f"    {grupo}: {', '.join(feats)}")


def imprimir_conclusion(df_champion):
    """Genera un dictamen automatico basado en los resultados."""
    if df_champion.empty:
        return

    test_rows = df_champion[df_champion["segmento"] == "TEST"]
    bt_rows   = df_champion[df_champion["segmento"] == "BACKTEST"]

    def safe_mean(series):
        try:
            return series.astype(float).mean()
        except Exception:
            return 0.0

    avg_delta_test = safe_mean(test_rows["delta_f1"])
    avg_delta_bt   = safe_mean(bt_rows["delta_f1"])

    n_mejora_test = (test_rows["delta_f1"].astype(float) > 0.005).sum()
    n_mejora_bt   = (bt_rows["delta_f1"].astype(float) > 0.005).sum()
    n_total       = len(test_rows)

    print(f"\n  DICTAMEN AUTOMATICO V1 vs V2:")
    print(f"  {'='*60}")
    print(f"  dF1 promedio TEST     : {avg_delta_test:+.4f}  "
          f"({n_mejora_test}/{n_total} scopes mejoran)")
    print(f"  dF1 promedio BACKTEST : {avg_delta_bt:+.4f}  "
          f"({n_mejora_bt}/{n_total} scopes mejoran)")

    if avg_delta_test > 0.01 and avg_delta_bt >= 0:
        verdict = "RECOMENDADO -- V2 mejora en TEST y no regresa en BACKTEST"
    elif avg_delta_test > 0 and avg_delta_bt > 0:
        verdict = "POSITIVO    -- V2 mejora en ambos segmentos (magnitud moderada)"
    elif avg_delta_test > 0 and avg_delta_bt < -0.01:
        verdict = "MIXTO       -- V2 mejora TEST pero retrocede en BACKTEST (overfitting?)"
    elif avg_delta_test <= 0:
        verdict = "NEGATIVO    -- V2 no mejora sobre V1 en TEST"
    else:
        verdict = "NEUTRAL     -- Diferencia no significativa"

    print(f"  Verdict               : {verdict}")
    print(f"  {'='*60}")


def main():
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  COMPARACION V1 (29 features) vs V2 (59 features)")
    print("=" * 65)
    print(f"  Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")

    # Verificar que existan datos de ambas versiones
    check = query_df("""
        SELECT modelo_version, COUNT(*) AS n
        FROM resultados_modelos_ml
        WHERE modelo_version IN ('v1', 'v2')
        GROUP BY modelo_version
    """)

    if check.empty:
        print("\n[!] No hay datos en resultados_modelos_ml.")
        print("    Ejecutar scripts/07_train_models.py y scripts/10_train_models_v2.py")
        sys.exit(0)

    versiones = set(check["modelo_version"].tolist())
    for ver in ["v1", "v2"]:
        n = check.loc[check["modelo_version"] == ver, "n"].values
        n_val = int(n[0]) if len(n) > 0 else 0
        status = "OK" if n_val > 0 else "FALTA"
        print(f"  {ver}: {n_val:>4} registros [{status}]")

    if "v1" not in versiones:
        print("\n[!] Faltan datos V1. Ejecutar scripts/07_train_models.py primero.")
        sys.exit(1)
    if "v2" not in versiones:
        print("\n[!] Faltan datos V2. Ejecutar scripts/10_train_models_v2.py primero.")
        sys.exit(1)

    # Tabla 1: Todas las metricas por scope/algoritmo/segmento
    print(f"\n{'='*65}")
    print(f"  1. METRICAS COMPLETAS (F1_GANANCIA + ROC-AUC)")
    print(f"{'='*65}")
    imprimir_comparacion_metricas()

    # Tabla 2: Solo campeon por scope (mejor algoritmo)
    print(f"\n{'='*65}")
    print(f"  2. CAMPEON POR SCOPE (mejor algoritmo de cada version)")
    print(f"{'='*65}")
    df_champ = query_df("""
        WITH best AS (
            SELECT scope, modelo_version, algoritmo, segmento, f1_1, roc_auc,
                   ROW_NUMBER() OVER (
                       PARTITION BY scope, modelo_version, segmento
                       ORDER BY f1_1 DESC
                   ) AS rnk
            FROM resultados_modelos_ml
            WHERE segmento IN ('TEST', 'BACKTEST')
        )
        SELECT
            v1.scope, v1.segmento,
            v1.algoritmo AS alg_v1, v2.algoritmo AS alg_v2,
            ROUND(v1.f1_1::NUMERIC, 4)              AS f1_v1,
            ROUND(v2.f1_1::NUMERIC, 4)              AS f1_v2,
            ROUND((v2.f1_1 - v1.f1_1)::NUMERIC, 4) AS delta_f1,
            ROUND(v1.roc_auc::NUMERIC, 4)           AS roc_v1,
            ROUND(v2.roc_auc::NUMERIC, 4)           AS roc_v2
        FROM best v1
        JOIN best v2 ON v1.scope = v2.scope AND v1.segmento = v2.segmento
        WHERE v1.modelo_version = 'v1'
          AND v2.modelo_version = 'v2'
          AND v1.rnk = 1 AND v2.rnk = 1
        ORDER BY v1.scope, CASE v1.segmento WHEN 'TEST' THEN 1 ELSE 2 END
    """)
    imprimir_resumen_champion()
    imprimir_conclusion(df_champ)

    # Tabla 3: Deployment comparison
    print(f"\n{'='*65}")
    print(f"  3. PLAN DE DEPLOYMENT V1 vs V2")
    print(f"{'='*65}")
    imprimir_comparacion_deployment()

    # Tabla 4: Info sobre nuevas features
    print(f"\n{'='*65}")
    print(f"  4. FEATURES PRECIO/ACCION (PA) INCORPORADAS")
    print(f"{'='*65}")
    imprimir_top_nuevas_features()

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n{'='*65}")
    print(f"  Comparacion completada en {duracion}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
