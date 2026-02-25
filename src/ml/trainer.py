"""
trainer.py
Motor de entrenamiento ML — Challenger RF × XGBoost × LightGBM.

Arquitectura de 3 niveles:
    Nivel 1: Modelo Global       — todos los sectores (19 tickers)
    Nivel 2: Modelos Sectoriales — Financials / Consumer Staples /
                                   Consumer Discretionary
    Nivel 3: Challenger Final    — Global Champion vs Sector Champion
                                   por cada uno de los 3 sectores.
             Automotive siempre usa el Global Champion (solo 3 tickers).

Métrica principal de evaluación: F1 para clase GANANCIA (label_binario=1)
    → maximiza el balance precision/recall para identificar trades ganadores.

Feature engineering aplicado antes del entrenamiento:
    bb_posicion  = (close - bb_lower) / (bb_upper - bb_lower)   [0-1]
    atr14_pct    = atr14 / close * 100                           [% volatilidad]
    momentum_pct = momentum / (close - momentum) * 100           [% retorno ~10d]
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import psycopg2.extras
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from src.data.database import query_df, get_connection, ejecutar_sql
from src.utils.config import SECTORES_ML, MODELS_DIR


# ─────────────────────────────────────────────────────────────
# Especificación de features
# ─────────────────────────────────────────────────────────────

# Columnas que se cargan de features_ml (incluye las necesarias
# para la ingeniería de features, aunque no vayan al modelo directamente)
_COLS_CARGA = [
    "ticker", "sector", "fecha", "segmento",
    # Para ingeniería (no entran al modelo en bruto)
    "close", "bb_upper", "bb_lower", "atr14", "momentum",
    # Indicadores directos
    "rsi14", "macd_hist", "adx", "vol_relativo",
    "dist_sma21", "dist_sma50", "dist_sma200",
    # Scoring rule-based
    "score_ponderado", "condiciones_ok",
    "cond_rsi", "cond_macd", "cond_sma21",
    "cond_sma50", "cond_sma200", "cond_momentum",
    # Z-Scores sectoriales
    "z_rsi_sector", "z_retorno_1d_sector", "z_retorno_5d_sector",
    "z_vol_sector", "z_dist_sma50_sector", "z_adx_sector",
    "pct_long_sector", "rank_retorno_sector",
    "rsi_sector_avg", "adx_sector_avg", "retorno_1d_sector_avg",
    # Target
    "label_binario",
]

# Features finales que entran al modelo (29 columnas)
FEATURE_COLS = [
    # Osciladores e indicadores (scale-independent o normalizado)
    "rsi14",           # 0-100
    "macd_hist",       # price-scale pero signo es el dato relevante
    "adx",             # 0-100 (fuerza de tendencia)
    "vol_relativo",    # ratio vs promedio

    # Distancias relativas a SMAs (en %)
    "dist_sma21",
    "dist_sma50",
    "dist_sma200",

    # Engineered: reemplazan close/atr14/momentum/bb_*
    "bb_posicion",     # 0 = banda inferior, 1 = banda superior
    "atr14_pct",       # ATR como % del precio (volatilidad)
    "momentum_pct",    # retorno ~10d en % (dirección + fuerza)

    # Scoring rule-based
    "score_ponderado",  # 0-1
    "condiciones_ok",   # 0-6
    "cond_rsi",
    "cond_macd",
    "cond_sma21",
    "cond_sma50",
    "cond_sma200",
    "cond_momentum",

    # Z-Scores sectoriales (scale-independent por construcción)
    "z_rsi_sector",
    "z_retorno_1d_sector",
    "z_retorno_5d_sector",
    "z_vol_sector",
    "z_dist_sma50_sector",
    "z_adx_sector",

    # Breadth y promedios sectoriales
    "pct_long_sector",
    "rank_retorno_sector",
    "rsi_sector_avg",
    "adx_sector_avg",
    "retorno_1d_sector_avg",
]

TARGET_COL = "label_binario"

# Algoritmos disponibles según instalación
ALGORITMOS_DISPONIBLES = (
    ["rf"] +
    (["xgb"]  if HAS_XGB  else []) +
    (["lgbm"] if HAS_LGBM else [])
)

_BOOL_COLS = [
    "cond_rsi", "cond_macd", "cond_sma21",
    "cond_sma50", "cond_sma200", "cond_momentum",
]


# ─────────────────────────────────────────────────────────────
# Setup DB: crear tablas si no existen
# ─────────────────────────────────────────────────────────────

_DDL_TABLAS_ML = """
CREATE TABLE IF NOT EXISTS resultados_modelos_ml (
    id              SERIAL PRIMARY KEY,
    scope           VARCHAR(50),       -- 'global' | sector name
    algoritmo       VARCHAR(10),       -- 'rf' | 'xgb' | 'lgbm'
    segmento        VARCHAR(15),       -- 'TRAIN' | 'TEST' | 'BACKTEST'
    n_filas         INTEGER,
    n_features      INTEGER,
    accuracy        NUMERIC(7,4),
    precision_w     NUMERIC(7,4),
    recall_w        NUMERIC(7,4),
    f1_w            NUMERIC(7,4),
    precision_1     NUMERIC(7,4),      -- clase GANANCIA
    recall_1        NUMERIC(7,4),
    f1_1            NUMERIC(7,4),      -- metrica principal
    roc_auc         NUMERIC(7,4),
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (scope, algoritmo, segmento)
);

CREATE TABLE IF NOT EXISTS modelos_produccion (
    id              SERIAL PRIMARY KEY,
    scope           VARCHAR(50) NOT NULL,  -- 'global' | sector name
    tipo            VARCHAR(20),           -- 'global' | 'sectorial'
    algoritmo       VARCHAR(20),           -- 'rf' | 'xgb' | 'lgbm'
    modelo_path     VARCHAR(300),
    n_features      INTEGER,
    f1_test         NUMERIC(7,4),
    f1_backtest     NUMERIC(7,4),
    roc_auc_test    NUMERIC(7,4),
    activo          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (scope)
);
"""


def crear_tablas_ml():
    """Crea las tablas de resultados ML si no existen."""
    ejecutar_sql(_DDL_TABLAS_ML)


# ─────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────

def cargar_datos_scope(scope: str) -> Dict[str, pd.DataFrame]:
    """
    Carga TRAIN / TEST / BACKTEST para un scope.

    scope = 'global'      → todos los sectores
    scope = nombre_sector → filtrado por sector
    """
    sector_filter = (
        "" if scope == "global"
        else f"AND sector = '{scope}'"
    )
    cols_sql = ", ".join(_COLS_CARGA)
    segmentos = {}

    for seg in ["TRAIN", "TEST", "BACKTEST"]:
        sql = f"""
            SELECT {cols_sql}
            FROM features_ml
            WHERE segmento = '{seg}'
              AND label_binario IS NOT NULL
              {sector_filter}
            ORDER BY ticker, fecha
        """
        df = query_df(sql)
        # Booleanos PostgreSQL → float para sklearn
        for col in _BOOL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(float)
        segmentos[seg] = df

    return segmentos


# ─────────────────────────────────────────────────────────────
# Ingeniería de features
# ─────────────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features derivadas que reemplazan valores en escala absoluta.

    bb_posicion  : posición dentro de las Bandas de Bollinger (0–1)
    atr14_pct    : ATR normalizado por precio (% de volatilidad)
    momentum_pct : momentum de ~10d normalizado por precio base
    """
    df = df.copy()

    # Posición en Bollinger Bands
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_posicion"] = np.where(
        bb_range > 0,
        ((df["close"] - df["bb_lower"]) / bb_range).clip(0, 1),
        0.5,
    )

    # ATR como % del precio actual
    df["atr14_pct"] = np.where(
        df["close"] > 0,
        df["atr14"] / df["close"] * 100,
        np.nan,
    )

    # Momentum como % del precio base (close[t-10] ≈ close - momentum)
    precio_base = df["close"] - df["momentum"]
    df["momentum_pct"] = np.where(
        precio_base > 0,
        df["momentum"] / precio_base * 100,
        np.nan,
    )

    return df


def preparar_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica ingeniería de features y separa X / y.
    Descarta filas donde label_binario es NaN.
    """
    df = feature_engineering(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────
# Construcción de modelos (Pipeline imputer + clasificador)
# ─────────────────────────────────────────────────────────────

def construir_modelo(algoritmo: str, n_train: int) -> Pipeline:
    """
    Construye un Pipeline sklearn con:
        1. SimpleImputer(strategy='median')  — maneja NaN residuales
        2. Clasificador con hyperparámetros adaptativos al tamaño del train

    min_leaf escala con n_train para evitar overfitting en sectores pequeños.
    """
    # Regularización adaptativa
    min_leaf = max(15, n_train // 300)

    imputer = SimpleImputer(strategy="median")

    if algoritmo == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=min_leaf,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    elif algoritmo == "xgb":
        if not HAS_XGB:
            raise ImportError("XGBoost no instalado: pip install xgboost")
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=min_leaf,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )

    elif algoritmo == "lgbm":
        if not HAS_LGBM:
            raise ImportError("LightGBM no instalado: pip install lightgbm")
        clf = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=min_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )

    else:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")

    return Pipeline([("imputer", imputer), ("clf", clf)])


# ─────────────────────────────────────────────────────────────
# Evaluación
# ─────────────────────────────────────────────────────────────

def calcular_metricas(
    model, X: pd.DataFrame, y: pd.Series,
    scope: str, algoritmo: str, segmento: str,
) -> dict:
    """
    Evalúa un modelo y retorna todas las métricas de clasificación.
    Métrica principal: f1_1 (F1 para clase GANANCIA).
    """
    y_pred = model.predict(X)

    try:
        y_proba = model.predict_proba(X)[:, 1]
        roc = round(float(roc_auc_score(y, y_proba)), 4)
    except Exception:
        roc = 0.0

    return {
        "scope":       scope,
        "algoritmo":   algoritmo,
        "segmento":    segmento,
        "n_filas":     int(len(y)),
        "n_features":  len(FEATURE_COLS),
        "accuracy":    round(float(accuracy_score(y, y_pred)), 4),
        "precision_w": round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4),
        "recall_w":    round(float(recall_score(y, y_pred,    average="weighted", zero_division=0)), 4),
        "f1_w":        round(float(f1_score(y, y_pred,        average="weighted", zero_division=0)), 4),
        "precision_1": round(float(precision_score(y, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_1":    round(float(recall_score(y, y_pred,    pos_label=1, zero_division=0)), 4),
        "f1_1":        round(float(f1_score(y, y_pred,        pos_label=1, zero_division=0)), 4),
        "roc_auc":     roc,
    }


# ─────────────────────────────────────────────────────────────
# Entrenamiento de un scope (Nivel 1 y 2)
# ─────────────────────────────────────────────────────────────

def entrenar_scope(scope: str, verbose: bool = True) -> dict:
    """
    Entrena el challenger RF × XGB × LGBM para un scope dado.
    Evalúa en TRAIN, TEST y BACKTEST.
    El campeón es el de mayor f1_1 en TEST.

    Returns:
        dict con:
            scope    : nombre del scope
            modelos  : {algoritmo: {model, TRAIN, TEST, BACKTEST metrics}}
            champion : {algoritmo, model, TEST metrics, BACKTEST metrics}
    """
    if verbose:
        label = "GLOBAL" if scope == "global" else scope
        print(f"\n  [{label}]")

    # Cargar y preparar datos
    segs = cargar_datos_scope(scope)
    X_tr, y_tr = preparar_xy(segs["TRAIN"])
    X_te, y_te = preparar_xy(segs["TEST"])
    X_bt, y_bt = preparar_xy(segs["BACKTEST"])

    if verbose:
        print(f"    Filas  | TRAIN: {len(y_tr):,}  TEST: {len(y_te):,}  BT: {len(y_bt):,}")
        print(f"    GANANCIA TRAIN: {y_tr.mean()*100:.1f}%  "
              f"TEST: {y_te.mean()*100:.1f}%  BT: {y_bt.mean()*100:.1f}%")

    resultados = {}

    for alg in ALGORITMOS_DISPONIBLES:
        if verbose:
            print(f"    [{alg.upper():4s}] entrenando...", end=" ", flush=True)
        try:
            model = construir_modelo(alg, len(y_tr))
            model.fit(X_tr, y_tr)

            m_tr = calcular_metricas(model, X_tr, y_tr, scope, alg, "TRAIN")
            m_te = calcular_metricas(model, X_te, y_te, scope, alg, "TEST")
            m_bt = calcular_metricas(model, X_bt, y_bt, scope, alg, "BACKTEST")

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
        raise RuntimeError(f"Ningún modelo pudo entrenarse para scope '{scope}'")

    # Campeón: mayor f1_1 en TEST
    champion_alg = max(resultados, key=lambda a: resultados[a]["TEST"]["f1_1"])
    if verbose:
        f1_ch = resultados[champion_alg]["TEST"]["f1_1"]
        print(f"    --> CHAMPION: {champion_alg.upper()}  "
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
# Persistencia de modelos en disco
# ─────────────────────────────────────────────────────────────

def _scope_dir(scope: str) -> str:
    """Directorio de modelos para un scope."""
    return os.path.join(MODELS_DIR, scope.replace(" ", "_").lower())


def guardar_modelos(resultados_todos: List[dict]):
    """Guarda todos los modelos (joblib) organizados por scope."""
    for sr in resultados_todos:
        scope = sr["scope"]
        d = _scope_dir(scope)
        os.makedirs(d, exist_ok=True)

        for alg, ar in sr["modelos"].items():
            joblib.dump(ar["model"], os.path.join(d, f"{alg}.joblib"))

        # Champion
        champ_path = os.path.join(d, "champion.joblib")
        joblib.dump(sr["champion"]["model"], champ_path)

    print(f"  Modelos guardados en: {MODELS_DIR}")


def cargar_champion(scope: str):
    """Carga el modelo campeón serializado de un scope."""
    path = os.path.join(_scope_dir(scope), "champion.joblib")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# Persistencia de métricas en DB
# ─────────────────────────────────────────────────────────────

def guardar_metricas_db(resultados_todos: List[dict], modelo_version: str = "v1"):
    """Persiste las métricas de todos los modelos/segmentos en resultados_modelos_ml."""
    records = []
    for sr in resultados_todos:
        for alg, ar in sr["modelos"].items():
            for seg in ["TRAIN", "TEST", "BACKTEST"]:
                m = ar[seg]
                rec = {k: m[k] for k in [
                    "scope", "algoritmo", "segmento", "n_filas", "n_features",
                    "accuracy", "precision_w", "recall_w", "f1_w",
                    "precision_1", "recall_1", "f1_1", "roc_auc",
                ]}
                rec["modelo_version"] = modelo_version
                records.append(rec)

    sql = """
        INSERT INTO resultados_modelos_ml
            (scope, algoritmo, segmento, modelo_version, n_filas, n_features,
             accuracy, precision_w, recall_w, f1_w,
             precision_1, recall_1, f1_1, roc_auc)
        VALUES
            (%(scope)s, %(algoritmo)s, %(segmento)s, %(modelo_version)s,
             %(n_filas)s, %(n_features)s,
             %(accuracy)s, %(precision_w)s, %(recall_w)s, %(f1_w)s,
             %(precision_1)s, %(recall_1)s, %(f1_1)s, %(roc_auc)s)
        ON CONFLICT (scope, algoritmo, segmento, modelo_version) DO UPDATE SET
            n_filas      = EXCLUDED.n_filas,
            accuracy     = EXCLUDED.accuracy,
            precision_w  = EXCLUDED.precision_w,
            recall_w     = EXCLUDED.recall_w,
            f1_w         = EXCLUDED.f1_w,
            precision_1  = EXCLUDED.precision_1,
            recall_1     = EXCLUDED.recall_1,
            f1_1         = EXCLUDED.f1_1,
            roc_auc      = EXCLUDED.roc_auc,
            created_at   = NOW()
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=100)
    print(f"  Metricas persistidas: {len(records)} registros en resultados_modelos_ml.")


# ─────────────────────────────────────────────────────────────
# Challenger Final: Nivel 3 — Global vs Sector
# ─────────────────────────────────────────────────────────────

def ejecutar_challenger_final(
    resultado_global: dict,
    resultados_sector: List[dict],
) -> List[dict]:
    """
    Para cada sector en SECTORES_ML, compara:
        - Global Champion  evaluado en TEST/BT filtrado por sector
        - Sector Champion  evaluado en TEST/BT filtrado por sector

    El ganador (mayor f1_1 en TEST sector-filtrado) se serializa
    como 'deployed.joblib' en el directorio del sector.

    Returns:
        Lista de dicts con la decisión de despliegue por sector.
    """
    global_model = resultado_global["champion"]["model"]
    global_alg   = resultado_global["champion"]["algoritmo"]

    deployment = []

    for sr in resultados_sector:
        sector       = sr["scope"]
        sector_model = sr["champion"]["model"]
        sector_alg   = sr["champion"]["algoritmo"]

        # Cargar datos del sector para TEST y BACKTEST
        segs_sector = cargar_datos_scope(sector)
        X_te, y_te = preparar_xy(segs_sector["TEST"])
        X_bt, y_bt = preparar_xy(segs_sector["BACKTEST"])

        # Evaluar ambos modelos en datos SECTOR-filtrados
        m_glob_te  = calcular_metricas(global_model,  X_te, y_te, sector, f"global_{global_alg}",  "TEST")
        m_sect_te  = calcular_metricas(sector_model,  X_te, y_te, sector, f"sector_{sector_alg}",  "TEST")
        m_glob_bt  = calcular_metricas(global_model,  X_bt, y_bt, sector, f"global_{global_alg}",  "BACKTEST")
        m_sect_bt  = calcular_metricas(sector_model,  X_bt, y_bt, sector, f"sector_{sector_alg}",  "BACKTEST")

        # Ganador: mayor f1_1 en TEST
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

        # Serializar modelo ganador como 'deployed'
        deployed_path = os.path.join(_scope_dir(sector), "deployed.joblib")
        joblib.dump(winner_model, deployed_path)

        deployment.append({
            "sector":         sector,
            "winner_tipo":    winner_tipo,
            "winner_alg":     winner_alg,
            "deployed_path":  deployed_path,
            "global_f1_te":   m_glob_te["f1_1"],
            "sector_f1_te":   m_sect_te["f1_1"],
            "winner_f1_te":   winner_te["f1_1"],
            "winner_f1_bt":   winner_bt["f1_1"],
            "winner_roc_te":  winner_te["roc_auc"],
            "global_alg":     global_alg,
            "sector_alg":     sector_alg,
            "m_glob_te":      m_glob_te,
            "m_sect_te":      m_sect_te,
            "m_glob_bt":      m_glob_bt,
            "m_sect_bt":      m_sect_bt,
        })

    return deployment


# ─────────────────────────────────────────────────────────────
# Persistencia del plan de despliegue
# ─────────────────────────────────────────────────────────────

def guardar_despliegue_db(deployment: List[dict], resultado_global: dict,
                          modelo_version: str = "v1"):
    """Persiste el modelo campeón por scope en modelos_produccion."""
    records = []

    # 1. Global champion (aplica a Automotive y como referencia)
    gc     = resultado_global["champion"]
    gc_dir = _scope_dir("global")
    records.append({
        "scope":          "global",
        "tipo":           "global",
        "algoritmo":      gc["algoritmo"],
        "modelo_path":    os.path.join(gc_dir, "champion.joblib"),
        "n_features":     len(FEATURE_COLS),
        "f1_test":        gc["TEST"]["f1_1"],
        "f1_backtest":    gc["BACKTEST"]["f1_1"],
        "roc_auc_test":   gc["TEST"]["roc_auc"],
        "modelo_version": modelo_version,
    })

    # 2. Automotive — usa Global Champion sin modelo propio
    records.append({
        "scope":          "Automotive",
        "tipo":           "global",
        "algoritmo":      gc["algoritmo"],
        "modelo_path":    os.path.join(gc_dir, "champion.joblib"),
        "n_features":     len(FEATURE_COLS),
        "f1_test":        None,
        "f1_backtest":    None,
        "roc_auc_test":   None,
        "modelo_version": modelo_version,
    })

    # 3. Sectores con challenger
    for d in deployment:
        records.append({
            "scope":          d["sector"],
            "tipo":           d["winner_tipo"],
            "algoritmo":      d["winner_alg"],
            "modelo_path":    d["deployed_path"],
            "n_features":     len(FEATURE_COLS),
            "f1_test":        d["winner_f1_te"],
            "f1_backtest":    d["winner_f1_bt"],
            "roc_auc_test":   d["winner_roc_te"],
            "modelo_version": modelo_version,
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
    print(f"  Despliegue persistido: {len(records)} scopes en modelos_produccion.")


# ─────────────────────────────────────────────────────────────
# Feature importance del modelo campeón
# ─────────────────────────────────────────────────────────────

def mostrar_importancias(model, scope: str, top_n: int = 10):
    """Imprime las top N features más importantes del campeón."""
    try:
        importancias = model.named_steps["clf"].feature_importances_
        fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": importancias})
        fi = fi.sort_values("importance", ascending=False).head(top_n)
        print(f"\n  Top {top_n} features [{scope}]:")
        for _, row in fi.iterrows():
            bar = "#" * max(1, int(row["importance"] * 250))
            print(f"    {row['feature']:<30} {row['importance']:.4f}  {bar}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Pipeline principal completo (3 niveles)
# ─────────────────────────────────────────────────────────────

def ejecutar_pipeline_ml(verbose: bool = True) -> dict:
    """
    Pipeline completo de entrenamiento:
        Nivel 1 → Global Champion
        Nivel 2 → Sector Champions (3 sectores)
        Nivel 3 → Challenger Final → Deployment Decision

    Returns:
        dict con todos los resultados y el deployment plan
    """
    crear_tablas_ml()
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Nivel 1: Global ──────────────────────────────────────
    print("\n  NIVEL 1 — Modelo Global (todos los sectores)")
    print("  " + "-" * 58)
    resultado_global = entrenar_scope("global", verbose=verbose)
    mostrar_importancias(resultado_global["champion"]["model"], "global")

    # ── Nivel 2: Sectoriales ─────────────────────────────────
    print(f"\n  NIVEL 2 — Modelos Sectoriales ({', '.join(SECTORES_ML)})")
    print("  " + "-" * 58)
    resultados_sector = [
        entrenar_scope(sector, verbose=verbose)
        for sector in SECTORES_ML
    ]

    # ── Guardar modelos en disco ──────────────────────────────
    todos = [resultado_global] + resultados_sector
    guardar_modelos(todos)

    # ── Persistir métricas ────────────────────────────────────
    guardar_metricas_db(todos)

    # ── Nivel 3: Challenger Final ─────────────────────────────
    print("\n  NIVEL 3 — Challenger Final (Global vs Sector por sector)")
    print("  " + "-" * 58)
    deployment = ejecutar_challenger_final(resultado_global, resultados_sector)

    # ── Persistir despliegue ──────────────────────────────────
    guardar_despliegue_db(deployment, resultado_global)

    return {
        "global":      resultado_global,
        "sectoriales": resultados_sector,
        "deployment":  deployment,
    }
