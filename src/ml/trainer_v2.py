"""
trainer_v2.py
Pipeline V2 de entrenamiento ML — integra features OHLCV de precio/accion.

FEATURE_COLS_V2 = FEATURE_COLS (29) + FEATURE_COLS_PA (30) = 59 features.

Misma arquitectura de 3 niveles que V1:
    Nivel 1: Modelo Global V2
    Nivel 2: Modelos Sectoriales V2
    Nivel 3: Challenger Final V2

Modelos guardados en models_v2/ (V1 en models/ queda intacto).
Metricas y deployment almacenados con modelo_version='v2'.
"""

import os
import joblib
import psycopg2.extras
import pandas as pd
from typing import Dict, List, Tuple

from src.data.database import query_df, get_connection
from src.utils.config import SECTORES_ML, MODELS_V2_DIR
from src.indicators.precio_accion import FEATURE_COLS_PA
from src.ml.trainer import (
    FEATURE_COLS,
    TARGET_COL,
    ALGORITMOS_DISPONIBLES,
    _BOOL_COLS,
    _COLS_CARGA,
    feature_engineering,
    construir_modelo,
    calcular_metricas,
    guardar_metricas_db,
)


# ─────────────────────────────────────────────────────────────
# Feature set V2: 29 V1 + 30 PA = 59 features
# ─────────────────────────────────────────────────────────────

FEATURE_COLS_V2: List[str] = FEATURE_COLS + FEATURE_COLS_PA  # 59 columnas

# Columnas PA que se cargan de features_precio_accion
_PA_COLS_LOAD: List[str] = FEATURE_COLS_PA


# ─────────────────────────────────────────────────────────────
# Utilidades de ruta (V2)
# ─────────────────────────────────────────────────────────────

def _scope_dir_v2(scope: str) -> str:
    """Directorio de modelos V2 para un scope."""
    return os.path.join(MODELS_V2_DIR, scope.replace(" ", "_").lower())


def cargar_champion_v2(scope: str):
    """Carga el modelo campeón V2 serializado de un scope."""
    path = os.path.join(_scope_dir_v2(scope), "champion.joblib")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# Carga de datos V2: JOIN features_ml + features_precio_accion
# ─────────────────────────────────────────────────────────────

def cargar_datos_scope_v2(scope: str) -> Dict[str, pd.DataFrame]:
    """
    Carga TRAIN / TEST / BACKTEST para un scope, con JOIN a
    features_precio_accion. Solo devuelve filas con features completas
    en ambas tablas (INNER JOIN).
    """
    sector_filter = (
        "" if scope == "global"
        else f"AND fm.sector = '{scope}'"
    )

    fm_cols = ", ".join(f"fm.{c}" for c in _COLS_CARGA)
    pa_cols = ", ".join(f"fpa.{c}" for c in _PA_COLS_LOAD)

    segmentos = {}
    for seg in ["TRAIN", "TEST", "BACKTEST"]:
        sql = f"""
            SELECT {fm_cols}, {pa_cols}
            FROM features_ml fm
            JOIN features_precio_accion fpa
                ON fm.ticker = fpa.ticker AND fm.fecha = fpa.fecha
            WHERE fm.segmento = '{seg}'
              AND fm.label_binario IS NOT NULL
              {sector_filter}
            ORDER BY fm.ticker, fm.fecha
        """
        df = query_df(sql)
        # Booleanos PostgreSQL -> float para sklearn
        for col in _BOOL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(float)
        segmentos[seg] = df

    return segmentos


# ─────────────────────────────────────────────────────────────
# Preparacion X/y con features V2
# ─────────────────────────────────────────────────────────────

def preparar_xy_v2(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica ingenieria de features (bb_posicion, atr14_pct, momentum_pct)
    y separa X con FEATURE_COLS_V2 / y con label_binario.
    """
    df = feature_engineering(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    X = df[FEATURE_COLS_V2].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────
# Metricas V2 (sobreescribe n_features = 59)
# ─────────────────────────────────────────────────────────────

def calcular_metricas_v2(
    model, X: pd.DataFrame, y: pd.Series,
    scope: str, algoritmo: str, segmento: str,
) -> dict:
    """Evalua un modelo V2 y retorna metricas con n_features=59."""
    m = calcular_metricas(model, X, y, scope, algoritmo, segmento)
    m["n_features"] = len(FEATURE_COLS_V2)
    return m


# ─────────────────────────────────────────────────────────────
# Feature importance V2
# ─────────────────────────────────────────────────────────────

def mostrar_importancias_v2(model, scope: str, top_n: int = 12):
    """Imprime las top N features mas importantes del campeon V2."""
    try:
        importancias = model.named_steps["clf"].feature_importances_
        fi = pd.DataFrame({"feature": FEATURE_COLS_V2, "importance": importancias})
        fi = fi.sort_values("importance", ascending=False).head(top_n)
        print(f"\n  Top {top_n} features V2 [{scope}]:")
        for _, row in fi.iterrows():
            bar = "#" * max(1, int(row["importance"] * 250))
            print(f"    {row['feature']:<32} {row['importance']:.4f}  {bar}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Entrenamiento de un scope V2
# ─────────────────────────────────────────────────────────────

def entrenar_scope_v2(scope: str, verbose: bool = True) -> dict:
    """
    Entrena el challenger RF x XGB x LGBM para un scope con V2 features.

    Returns:
        dict con scope, modelos, champion
    """
    if verbose:
        label = "GLOBAL" if scope == "global" else scope
        print(f"\n  [{label}]")

    segs = cargar_datos_scope_v2(scope)
    X_tr, y_tr = preparar_xy_v2(segs["TRAIN"])
    X_te, y_te = preparar_xy_v2(segs["TEST"])
    X_bt, y_bt = preparar_xy_v2(segs["BACKTEST"])

    if verbose:
        print(f"    Filas  | TRAIN: {len(y_tr):,}  TEST: {len(y_te):,}  BT: {len(y_bt):,}")
        print(f"    GANANCIA TRAIN: {y_tr.mean()*100:.1f}%  "
              f"TEST: {y_te.mean()*100:.1f}%  BT: {y_bt.mean()*100:.1f}%")
        print(f"    Features: {len(FEATURE_COLS_V2)} (V1:{len(FEATURE_COLS)} + PA:{len(FEATURE_COLS_PA)})")

    resultados = {}

    for alg in ALGORITMOS_DISPONIBLES:
        if verbose:
            print(f"    [{alg.upper():4s}] entrenando...", end=" ", flush=True)
        try:
            model = construir_modelo(alg, len(y_tr))
            model.fit(X_tr, y_tr)

            m_tr = calcular_metricas_v2(model, X_tr, y_tr, scope, alg, "TRAIN")
            m_te = calcular_metricas_v2(model, X_te, y_te, scope, alg, "TEST")
            m_bt = calcular_metricas_v2(model, X_bt, y_bt, scope, alg, "BACKTEST")

            resultados[alg] = {"model": model, "TRAIN": m_tr, "TEST": m_te, "BACKTEST": m_bt}

            if verbose:
                print(
                    f"F1_G: TRAIN={m_tr['f1_1']:.3f}  "
                    f"TEST={m_te['f1_1']:.3f}  "
                    f"BT={m_bt['f1_1']:.3f}  "
                    f"ROC={m_te['roc_auc']:.3f}"
                )
        except Exception as e:
            print(f"\n    [ERROR {alg}] {e}")

    if not resultados:
        raise RuntimeError(f"Ningun modelo pudo entrenarse para scope '{scope}'")

    champion_alg = max(resultados, key=lambda a: resultados[a]["TEST"]["f1_1"])
    if verbose:
        f1_ch = resultados[champion_alg]["TEST"]["f1_1"]
        print(f"    --> CHAMPION V2: {champion_alg.upper()}  "
              f"(F1_GANANCIA_TEST = {f1_ch:.4f})")

    return {
        "scope":    scope,
        "modelos":  resultados,
        "champion": {
            "algoritmo": champion_alg,
            "model":     resultados[champion_alg]["model"],
            "TEST":      resultados[champion_alg]["TEST"],
            "BACKTEST":  resultados[champion_alg]["BACKTEST"],
        },
    }


# ─────────────────────────────────────────────────────────────
# Persistencia de modelos en disco (V2)
# ─────────────────────────────────────────────────────────────

def guardar_modelos_v2(resultados_todos: List[dict]):
    """Guarda todos los modelos V2 (joblib) en models_v2/."""
    for sr in resultados_todos:
        scope = sr["scope"]
        d = _scope_dir_v2(scope)
        os.makedirs(d, exist_ok=True)

        for alg, ar in sr["modelos"].items():
            joblib.dump(ar["model"], os.path.join(d, f"{alg}.joblib"))

        joblib.dump(sr["champion"]["model"], os.path.join(d, "champion.joblib"))

    print(f"  Modelos V2 guardados en: {MODELS_V2_DIR}")


# ─────────────────────────────────────────────────────────────
# Persistencia de metricas en DB (V2)
# ─────────────────────────────────────────────────────────────

def guardar_metricas_db_v2(resultados_todos: List[dict]):
    """Persiste metricas V2 con modelo_version='v2'."""
    guardar_metricas_db(resultados_todos, modelo_version="v2")


# ─────────────────────────────────────────────────────────────
# Challenger Final V2: Nivel 3
# ─────────────────────────────────────────────────────────────

def ejecutar_challenger_final_v2(
    resultado_global: dict,
    resultados_sector: List[dict],
) -> List[dict]:
    """
    Para cada sector en SECTORES_ML, compara Global V2 vs Sector V2.
    Serializa el ganador como 'deployed.joblib' en models_v2/sector/.
    """
    global_model = resultado_global["champion"]["model"]
    global_alg   = resultado_global["champion"]["algoritmo"]

    deployment = []

    for sr in resultados_sector:
        sector       = sr["scope"]
        sector_model = sr["champion"]["model"]
        sector_alg   = sr["champion"]["algoritmo"]

        segs = cargar_datos_scope_v2(sector)
        X_te, y_te = preparar_xy_v2(segs["TEST"])
        X_bt, y_bt = preparar_xy_v2(segs["BACKTEST"])

        m_glob_te = calcular_metricas_v2(global_model,  X_te, y_te, sector, f"global_{global_alg}",  "TEST")
        m_sect_te = calcular_metricas_v2(sector_model,  X_te, y_te, sector, f"sector_{sector_alg}",  "TEST")
        m_glob_bt = calcular_metricas_v2(global_model,  X_bt, y_bt, sector, f"global_{global_alg}",  "BACKTEST")
        m_sect_bt = calcular_metricas_v2(sector_model,  X_bt, y_bt, sector, f"sector_{sector_alg}",  "BACKTEST")

        if m_sect_te["f1_1"] > m_glob_te["f1_1"]:
            winner_tipo  = "sectorial"
            winner_alg   = sector_alg
            winner_model = sector_model
            winner_te    = m_sect_te
            winner_bt    = m_sect_bt
        else:
            winner_tipo  = "global"
            winner_alg   = f"global_{global_alg}"
            winner_model = global_model
            winner_te    = m_glob_te
            winner_bt    = m_glob_bt

        deployed_path = os.path.join(_scope_dir_v2(sector), "deployed.joblib")
        joblib.dump(winner_model, deployed_path)

        deployment.append({
            "sector":        sector,
            "winner_tipo":   winner_tipo,
            "winner_alg":    winner_alg,
            "deployed_path": deployed_path,
            "global_f1_te":  m_glob_te["f1_1"],
            "sector_f1_te":  m_sect_te["f1_1"],
            "winner_f1_te":  winner_te["f1_1"],
            "winner_f1_bt":  winner_bt["f1_1"],
            "winner_roc_te": winner_te["roc_auc"],
            "global_alg":    global_alg,
            "sector_alg":    sector_alg,
            "m_glob_te":     m_glob_te,
            "m_sect_te":     m_sect_te,
            "m_glob_bt":     m_glob_bt,
            "m_sect_bt":     m_sect_bt,
        })

    return deployment


# ─────────────────────────────────────────────────────────────
# Persistencia del despliegue V2
# ─────────────────────────────────────────────────────────────

def guardar_despliegue_db_v2(deployment: List[dict], resultado_global: dict):
    """Persiste el plan de despliegue V2 en modelos_produccion."""
    records = []

    gc     = resultado_global["champion"]
    gc_dir = _scope_dir_v2("global")

    records.append({
        "scope":          "global",
        "tipo":           "global",
        "algoritmo":      gc["algoritmo"],
        "modelo_path":    os.path.join(gc_dir, "champion.joblib"),
        "n_features":     len(FEATURE_COLS_V2),
        "f1_test":        gc["TEST"]["f1_1"],
        "f1_backtest":    gc["BACKTEST"]["f1_1"],
        "roc_auc_test":   gc["TEST"]["roc_auc"],
        "modelo_version": "v2",
    })

    # Automotive usa el Global Champion V2
    records.append({
        "scope":          "Automotive",
        "tipo":           "global",
        "algoritmo":      gc["algoritmo"],
        "modelo_path":    os.path.join(gc_dir, "champion.joblib"),
        "n_features":     len(FEATURE_COLS_V2),
        "f1_test":        None,
        "f1_backtest":    None,
        "roc_auc_test":   None,
        "modelo_version": "v2",
    })

    for d in deployment:
        records.append({
            "scope":          d["sector"],
            "tipo":           d["winner_tipo"],
            "algoritmo":      d["winner_alg"],
            "modelo_path":    d["deployed_path"],
            "n_features":     len(FEATURE_COLS_V2),
            "f1_test":        d["winner_f1_te"],
            "f1_backtest":    d["winner_f1_bt"],
            "roc_auc_test":   d["winner_roc_te"],
            "modelo_version": "v2",
        })

    sql = """
        INSERT INTO modelos_produccion
            (scope, tipo, algoritmo, modelo_path, n_features,
             f1_test, f1_backtest, roc_auc_test, modelo_version)
        VALUES
            (%(scope)s, %(tipo)s, %(algoritmo)s, %(modelo_path)s, %(n_features)s,
             %(f1_test)s, %(f1_backtest)s, %(roc_auc_test)s, %(modelo_version)s)
        ON CONFLICT (scope, modelo_version) DO UPDATE SET
            tipo         = EXCLUDED.tipo,
            algoritmo    = EXCLUDED.algoritmo,
            modelo_path  = EXCLUDED.modelo_path,
            n_features   = EXCLUDED.n_features,
            f1_test      = EXCLUDED.f1_test,
            f1_backtest  = EXCLUDED.f1_backtest,
            roc_auc_test = EXCLUDED.roc_auc_test,
            activo       = TRUE,
            created_at   = NOW()
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=20)
    print(f"  Despliegue V2 persistido: {len(records)} scopes en modelos_produccion.")


# ─────────────────────────────────────────────────────────────
# Pipeline principal V2 completo (3 niveles)
# ─────────────────────────────────────────────────────────────

def ejecutar_pipeline_ml_v2(verbose: bool = True) -> dict:
    """
    Pipeline completo V2:
        Nivel 1 -> Global Champion V2
        Nivel 2 -> Sector Champions V2 (3 sectores)
        Nivel 3 -> Challenger Final V2 -> Deployment Decision

    Returns:
        dict con global, sectoriales, deployment
    """
    os.makedirs(MODELS_V2_DIR, exist_ok=True)

    # Nivel 1: Global
    print("\n  NIVEL 1 — Modelo Global V2 (todos los sectores, 59 features)")
    print("  " + "-" * 58)
    resultado_global = entrenar_scope_v2("global", verbose=verbose)
    mostrar_importancias_v2(resultado_global["champion"]["model"], "global")

    # Nivel 2: Sectoriales
    print(f"\n  NIVEL 2 — Modelos Sectoriales V2 ({', '.join(SECTORES_ML)})")
    print("  " + "-" * 58)
    resultados_sector = [
        entrenar_scope_v2(sector, verbose=verbose)
        for sector in SECTORES_ML
    ]

    # Guardar modelos en disco
    todos = [resultado_global] + resultados_sector
    guardar_modelos_v2(todos)

    # Persistir metricas con modelo_version='v2'
    guardar_metricas_db_v2(todos)

    # Nivel 3: Challenger Final
    print("\n  NIVEL 3 — Challenger Final V2 (Global vs Sector por sector)")
    print("  " + "-" * 58)
    deployment = ejecutar_challenger_final_v2(resultado_global, resultados_sector)

    # Persistir despliegue
    guardar_despliegue_db_v2(deployment, resultado_global)

    return {
        "global":      resultado_global,
        "sectoriales": resultados_sector,
        "deployment":  deployment,
    }
