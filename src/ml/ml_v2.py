"""
src/ml/ml_v2.py
Modelo ML v2 del scanner (Etapa 3c, 13/9/2026): carga validada, preparacion de
features y cortes de probabilidad equivalentes a los de la v1.

Sin DB. joblib/sklearn se importan recien al cargar el modelo, asi que las
funciones de cortes y de armado de X se pueden probar sin artefacto.

QUE ES LA v2
    La configuracion congelada en la Tarea 20 (docs/ml_reentrenamiento.md sec. 8):
    RandomForest global (construir_modelo('rf'), 53 features V3) envuelto en
    CalibratedClassifierCV isotonica (cv=3), label absoluto (retorno_20d > +1%),
    sin ponderador sectorial. La entrena scripts/ml/entrenar_ml_v2.py sobre
    features_ml. Corre EN PARALELO a la v1 (estrategia FT_ML_SCANNER_v2); la v1
    sigue con sus modelos V3 como control.

POR QUE CORTES EQUIVALENTES Y NO LOS DE LA v1
    El score compuesto del scanner (alert_classifier) suma puntos ML por umbrales
    fijos de probabilidad (0,75 / 0,65 / 0,55 / 0,45 / 0,35) pensados para la v1.
    Calibrada, la probabilidad se comprime: en el walk-forward P(prob >= 0,65) fue
    7,7% contra 13,8% de la v1 en vivo. Con los mismos umbrales la v2 daria la
    mitad de senales fuertes y comparar v1 contra v2 mediria selectividad, no
    modelo. Por eso cada corte de la v2 deja arriba la MISMA fraccion de filas que
    el corte correspondiente de la v1 sobre las MISMAS filas.

CONSISTENCIA ENTRENAMIENTO / SERVICIO
    Al entrenar, las features de market structure sin pivot confirmado se rellenan
    con 0 (walkforward_ml.cargar_dataset, trainer_v3.preparar_xy_v3). La v1 en vivo
    las pasa como NaN y su imputer pone la mediana. La v2 sirve igual que entrena:
    `matriz_x` lleva a 0 las columnas de `meta["columnas_cero"]`. Los demas NaN
    (p.ej. las 11 sectoriales de los 4 tickers sin contexto) quedan para el imputer.

EL ARTEFACTO
    models_ml_v2/rf_cal_global.joblib queda FUERA de git; models_ml_v2/metadata.json
    va EN git (rango de datos, metricas del holdout, compuerta, cortes, versiones y
    sha256). La carga verifica el sha256 y el orden de las features: un modelo que no
    es el de la metadata, o que espera otras columnas, no se usa.
"""

import hashlib
import json
import os
import warnings
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIR_MODELO_V2 = os.path.join(RAIZ, "models_ml_v2")
ARCHIVO_MODELO = "rf_cal_global.joblib"
ARCHIVO_META = "metadata.json"

# Umbrales de probabilidad de la v1 en alert_classifier._puntos_ml, de mayor a menor.
CORTES_V1: Tuple[float, ...] = (0.75, 0.65, 0.55, 0.45, 0.35)


class ModeloV2Invalido(Exception):
    """El artefacto no coincide con su metadata: no se debe usar."""


# -- Cortes equivalentes ------------------------------------------------------

def fracciones_sobre(probs: Sequence[float], cortes: Sequence[float]) -> List[float]:
    """Fraccion de filas con probabilidad >= cada corte (NaN excluidos)."""
    p = np.asarray(probs, dtype=float)
    p = p[~np.isnan(p)]
    if p.size == 0:
        raise ValueError("sin probabilidades validas")
    return [float((p >= c).mean()) for c in cortes]


def cortes_equivalentes(probs_ref: Sequence[float], probs_nuevo: Sequence[float],
                        cortes_ref: Sequence[float] = CORTES_V1) -> Tuple[float, ...]:
    """
    Para cada corte de referencia, el corte del modelo nuevo que deja arriba la
    misma cantidad de filas. `probs_ref` y `probs_nuevo` son las probabilidades de
    los dos modelos sobre LAS MISMAS filas, en el mismo orden.

    El corte es la k-esima probabilidad nueva mas alta, con k = filas arriba del
    corte de referencia. Con empates (la isotonica produce escalones) pueden quedar
    algunas filas mas arriba: por eso el entrenamiento guarda tambien las
    fracciones realizadas. Levanta ValueError si un corte no deja filas arriba o si
    los cortes no quedan estrictamente decrecientes (dos niveles colapsados).
    """
    ref = np.asarray(probs_ref, dtype=float)
    nuevo = np.asarray(probs_nuevo, dtype=float)
    if ref.shape != nuevo.shape:
        raise ValueError("probs_ref y probs_nuevo tienen que ser las mismas filas")
    validas = ~(np.isnan(ref) | np.isnan(nuevo))
    ref, nuevo = ref[validas], nuevo[validas]
    if ref.size == 0:
        raise ValueError("sin filas validas")

    orden = np.sort(nuevo)[::-1]
    cortes = []
    for c in cortes_ref:
        k = int((ref >= c).sum())
        if k == 0:
            raise ValueError(f"el corte de referencia {c} no deja filas arriba")
        cortes.append(float(orden[k - 1]))

    if any(b >= a for a, b in zip(cortes, cortes[1:])):
        raise ValueError(f"cortes no estrictamente decrecientes: {cortes}")
    return tuple(cortes)


# -- Features -----------------------------------------------------------------

def matriz_x(filas: Union[Mapping[str, float], Sequence[Mapping[str, float]]],
             features: Sequence[str], columnas_cero: Sequence[str] = ()) -> np.ndarray:
    """
    Arma X en el orden de `features` desde dicts (el `features_v3` del scanner).
    Falta o None -> NaN; NaN en `columnas_cero` -> 0, como al entrenar.
    """
    if isinstance(filas, Mapping):
        filas = [filas]
    cero = set(columnas_cero)
    X = np.empty((len(filas), len(features)), dtype=float)
    for i, fila in enumerate(filas):
        for j, col in enumerate(features):
            v = fila.get(col)
            v = np.nan if v is None else float(v)
            if np.isnan(v) and col in cero:
                v = 0.0
            X[i, j] = v
    return X


# -- Artefacto ----------------------------------------------------------------

def sha256_archivo(ruta: str) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_modelo_v2(directorio: str = DIR_MODELO_V2,
                     features_esperadas: Optional[Sequence[str]] = None) -> Tuple[object, Dict]:
    """
    Carga el modelo v2 y su metadata, validando antes de usarlo:
      - el sha256 del artefacto coincide con el de la metadata;
      - si se pasan `features_esperadas`, son las mismas y en el mismo orden;
      - la version de sklearn distinta solo avisa (joblib puede cargar igual,
        pero las probabilidades podrian no ser identicas).
    FileNotFoundError si falta algo; ModeloV2Invalido si no valida.
    """
    ruta_meta = os.path.join(directorio, ARCHIVO_META)
    ruta_modelo = os.path.join(directorio, ARCHIVO_MODELO)
    if not os.path.exists(ruta_meta):
        raise FileNotFoundError(f"falta {ruta_meta}")
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"falta {ruta_modelo} (el artefacto no va en git: "
                                f"regenerarlo con scripts/ml/entrenar_ml_v2.py)")

    with open(ruta_meta, encoding="utf-8") as fh:
        meta = json.load(fh)

    sha = sha256_archivo(ruta_modelo)
    if sha != meta.get("sha256"):
        raise ModeloV2Invalido(f"sha256 del artefacto ({sha[:12]}) no coincide con "
                               f"la metadata ({str(meta.get('sha256'))[:12]})")
    if features_esperadas is not None and list(meta.get("features", [])) != list(features_esperadas):
        raise ModeloV2Invalido("las features de la metadata no son las esperadas "
                               "(distinto conjunto u orden)")

    import joblib
    import sklearn
    version_meta = meta.get("versiones", {}).get("sklearn")
    if version_meta and version_meta != sklearn.__version__:
        warnings.warn(f"modelo v2 entrenado con sklearn {version_meta}, "
                      f"instalado {sklearn.__version__}")

    return joblib.load(ruta_modelo), meta


def cortes_de(meta: Mapping) -> Tuple[float, ...]:
    """Cortes de la v2 guardados en la metadata, de mayor a menor."""
    return tuple(float(c) for c in meta["cortes"]["v2"])


def prob_v2(modelo, meta: Mapping, features_v3: Mapping[str, float]) -> float:
    """P(label=1) de la v2 para un ticker, con las features del scanner."""
    X = matriz_x(features_v3, meta["features"], meta.get("columnas_cero", ()))
    clases = list(modelo.classes_)
    return float(modelo.predict_proba(X)[0][clases.index(1)])
