"""
signal_engine.py
Carga modelos V3 y genera senales para un ticker dado.

Responsabilidades:
    1. Cargar modelos champion V3 (global + sectoriales)
    2. Seleccionar el modelo correcto segun el sector del ticker
    3. Correr predict_proba con las 53 features
    4. Evaluar condiciones EV1-EV4 (alcistas) y senales bajistas
    5. Retornar dict con todos los resultados
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.utils.config import MODELS_V3_DIR, SECTORES_ML
from src.backtesting.strategies_pa import check_entrada_pa
from src.ml.trainer_v3 import FEATURE_COLS_V3


# ─────────────────────────────────────────────────────────────
# Carga de modelos
# ─────────────────────────────────────────────────────────────

_MODELOS_CACHE: Dict = {}   # cache para no recargar en cada llamada


def _scope_dir(scope: str) -> str:
    return os.path.join(MODELS_V3_DIR, scope.replace(" ", "_").lower())


def cargar_modelos_v3() -> Dict:
    """
    Carga los modelos champion V3 disponibles:
        'global' + un modelo por cada sector en SECTORES_ML.

    Usa cache en memoria para evitar IO repetido.

    Returns:
        dict { scope_name: sklearn Pipeline }
    """
    global _MODELOS_CACHE
    if _MODELOS_CACHE:
        return _MODELOS_CACHE

    import joblib

    scopes = ["global"] + SECTORES_ML
    modelos = {}

    for scope in scopes:
        path = os.path.join(_scope_dir(scope), "champion.joblib")
        if os.path.exists(path):
            try:
                modelos[scope] = joblib.load(path)
            except Exception as e:
                print(f"  [WARN] No se pudo cargar modelo {scope}: {e}")
        else:
            print(f"  [WARN] Modelo no encontrado: {path}")

    if "global" not in modelos:
        raise FileNotFoundError(
            f"Modelo global V3 no encontrado en {_scope_dir('global')}/champion.joblib"
        )

    _MODELOS_CACHE = modelos
    print(f"  [Modelos V3] Cargados: {list(modelos.keys())}")
    return modelos


def _obtener_modelo_asignado_db(ticker: str) -> Optional[str]:
    """
    Consulta activos.modelo_asignado para el ticker.
    Retorna el scope asignado (ej: 'global', 'Financials') o None si no existe.
    """
    from src.data.database import query_df
    sql = "SELECT modelo_asignado FROM activos WHERE ticker = :ticker"
    try:
        df = query_df(sql, params={"ticker": ticker})
        if df.empty:
            return None
        val = df.iloc[0]["modelo_asignado"]
        return str(val) if val is not None and str(val) != "None" else None
    except Exception:
        return None


def seleccionar_modelo(sector: Optional[str], modelos: Dict,
                       ticker: Optional[str] = None) -> tuple:
    """
    Selecciona el modelo correcto para el ticker/sector.

    Prioridad:
        1. activos.modelo_asignado (evaluado empiricamente con script 19)
        2. Champion sectorial (si el sector tiene modelo)
        3. Champion global (fallback)

    Returns:
        (modelo, scope_usado)
    """
    # 1. Asignacion especifica en DB (set por script 19)
    if ticker:
        asignado = _obtener_modelo_asignado_db(ticker)
        if asignado and asignado in modelos:
            return modelos[asignado], asignado

    # 2. Champion sectorial
    if sector and sector in modelos:
        return modelos[sector], sector

    # 3. Global fallback
    return modelos["global"], "global"


# ─────────────────────────────────────────────────────────────
# Evaluacion de condiciones PA con fila simulada
# ─────────────────────────────────────────────────────────────

def _evaluar_ev(features_pa: Dict, estrategia: str) -> bool:
    """
    Evalua una condicion de entrada EV usando check_entrada_pa.
    Construye una pd.Series simulando una fila de DataFrame.
    """
    row = pd.Series(features_pa)
    try:
        return check_entrada_pa(row, estrategia)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Pipeline principal de senales
# ─────────────────────────────────────────────────────────────

def evaluar_ticker(features_v3: Dict, features_pa: Dict,
                   sector: Optional[str], modelos: Dict,
                   ticker: Optional[str] = None) -> Dict:
    """
    Genera todas las senales para un ticker.

    Args:
        features_v3: dict con las 53 features del modelo V3
        features_pa: dict con features PA y market structure adicionales
        sector:      sector del ticker (None = usar global)
        modelos:     dict de modelos cargados con cargar_modelos_v3()
        ticker:      codigo del activo (para consultar modelo_asignado en DB)

    Returns:
        dict con:
            ml_prob_ganancia  : float [0,1]
            ml_modelo_usado   : str
            pa_ev1..ev4       : int (0 o 1)
            bear_bos10        : int
            bear_choch10      : int
            bear_estructura   : int
    """
    # ── 1. ML V3: predict_proba ────────────────────────────────
    modelo, scope_usado = seleccionar_modelo(sector, modelos, ticker=ticker)

    # Construir vector de features en el orden correcto
    X_vals = []
    for col in FEATURE_COLS_V3:
        val = features_v3.get(col, np.nan)
        X_vals.append(val if not (isinstance(val, float) and np.isnan(val)) else np.nan)

    X = np.array(X_vals, dtype=float).reshape(1, -1)

    try:
        prob = modelo.predict_proba(X)[0]
        # classes_=[0,1] => prob[1] = P(ganancia)
        ml_prob_ganancia = float(prob[1])
    except Exception as e:
        print(f"  [WARN] predict_proba fallo: {e}")
        ml_prob_ganancia = 0.0

    # ── 2. Condiciones de entrada PA (EV1-EV4) ────────────────
    ev1 = int(_evaluar_ev(features_pa, "EV1"))
    ev2 = int(_evaluar_ev(features_pa, "EV2"))
    ev3 = int(_evaluar_ev(features_pa, "EV3"))
    ev4 = int(_evaluar_ev(features_pa, "EV4"))

    # ── 3. Senales bajistas ───────────────────────────────────
    def _safe_int(val):
        if val is None:
            return 0
        try:
            return int(val)
        except Exception:
            return 0

    bear_bos10 = _safe_int(features_pa.get("bos_bear_10"))
    bear_choch10 = _safe_int(features_pa.get("choch_bear_10"))

    est10 = features_pa.get("estructura_10")
    bear_estructura = 1 if (est10 is not None and _safe_int(est10) == -1) else 0

    return {
        "ml_prob_ganancia":  round(ml_prob_ganancia, 4),
        "ml_modelo_usado":   scope_usado,
        "pa_ev1":            ev1,
        "pa_ev2":            ev2,
        "pa_ev3":            ev3,
        "pa_ev4":            ev4,
        "bear_bos10":        bear_bos10,
        "bear_choch10":      bear_choch10,
        "bear_estructura":   bear_estructura,
    }
