"""
Tests de src/ml/ml_v2.py: cortes equivalentes, armado de X y carga validada.

Lo que importa fijar:
  - los cortes de la v2 dejan arriba la misma cantidad de filas que los de la v1
    sobre las mismas filas (si no, v1 vs v2 compara selectividad y no modelo);
  - al servir, las features de market structure vacias van a 0 como al entrenar;
  - un artefacto que no coincide con su metadata no se carga.
"""

import json
import math

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.ml import ml_v2


# -- cortes ---------------------------------------------------------------------

def test_fracciones_sobre():
    assert ml_v2.fracciones_sobre([0.1, 0.5, 0.7, 0.9, np.nan], [0.8, 0.5]) == [0.25, 0.75]


def test_cortes_equivalentes_dejan_la_misma_cantidad_arriba():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0.2, 0.9, 5000)
    nuevo = 0.3 + 0.4 * ref ** 2 + rng.normal(0, 0.01, 5000)   # otra escala, casi mismo orden
    cortes = ml_v2.cortes_equivalentes(ref, nuevo)

    for c_ref, c_nuevo in zip(ml_v2.CORTES_V1, cortes):
        assert (nuevo >= c_nuevo).sum() == (ref >= c_ref).sum()
    assert all(b < a for a, b in zip(cortes, cortes[1:]))


def test_cortes_equivalentes_con_transformacion_monotona_exacta():
    ref = np.linspace(0.0, 1.0, 1001)
    nuevo = np.sqrt(ref)
    cortes = ml_v2.cortes_equivalentes(ref, nuevo)
    for c_ref, c_nuevo in zip(ml_v2.CORTES_V1, cortes):
        assert c_nuevo == pytest.approx(math.sqrt(c_ref), abs=2e-3)


def test_cortes_equivalentes_exige_las_mismas_filas():
    with pytest.raises(ValueError):
        ml_v2.cortes_equivalentes([0.5, 0.6], [0.5, 0.6, 0.7])


def test_corte_sin_filas_arriba_falla():
    with pytest.raises(ValueError, match="no deja filas"):
        ml_v2.cortes_equivalentes([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])


def test_niveles_colapsados_por_empates_fallan():
    ref = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    nuevo = np.full(6, 0.5)                                    # sin orden: todo empata
    with pytest.raises(ValueError, match="decrecientes"):
        ml_v2.cortes_equivalentes(ref, nuevo)


# -- matriz_x ---------------------------------------------------------------------

def test_matriz_x_orden_cero_y_nan():
    fila = {"b": 2.0, "ms1": np.nan, "a": 1.0, "sec": np.nan}
    X = ml_v2.matriz_x(fila, ["a", "b", "ms1", "ms2", "sec"], columnas_cero=["ms1", "ms2"])
    assert X.shape == (1, 5)
    assert X[0, 0] == 1.0 and X[0, 1] == 2.0
    assert X[0, 2] == 0.0 and X[0, 3] == 0.0          # MS vacia o ausente -> 0
    assert np.isnan(X[0, 4])                          # el resto queda para el imputer


def test_matriz_x_varias_filas_y_none():
    X = ml_v2.matriz_x([{"a": 1}, {"a": None}], ["a"])
    assert X[0, 0] == 1.0 and np.isnan(X[1, 0])


# -- carga ----------------------------------------------------------------------

def _artefacto(tmp_path, features=("f1", "f2")):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + rng.normal(0, 0.5, 200) > 0).astype(int)
    modelo = LogisticRegression().fit(X, y)
    ruta = tmp_path / ml_v2.ARCHIVO_MODELO
    joblib.dump(modelo, ruta)
    import sklearn
    meta = {"version": "ml_v2_test", "features": list(features), "columnas_cero": [],
            "cortes": {"v2": [0.8, 0.7, 0.6, 0.5, 0.4]},
            "versiones": {"sklearn": sklearn.__version__},
            "sha256": ml_v2.sha256_archivo(str(ruta))}
    (tmp_path / ml_v2.ARCHIVO_META).write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_carga_valida_y_predice(tmp_path):
    _artefacto(tmp_path)
    modelo, meta = ml_v2.cargar_modelo_v2(str(tmp_path), features_esperadas=["f1", "f2"])
    p = ml_v2.prob_v2(modelo, meta, {"f1": 2.0, "f2": 0.0})
    assert 0.5 < p <= 1.0
    assert ml_v2.cortes_de(meta) == (0.8, 0.7, 0.6, 0.5, 0.4)


def test_artefacto_alterado_no_carga(tmp_path):
    _artefacto(tmp_path)
    with open(tmp_path / ml_v2.ARCHIVO_MODELO, "ab") as fh:
        fh.write(b"x")
    with pytest.raises(ml_v2.ModeloV2Invalido, match="sha256"):
        ml_v2.cargar_modelo_v2(str(tmp_path))


def test_features_en_otro_orden_no_carga(tmp_path):
    _artefacto(tmp_path)
    with pytest.raises(ml_v2.ModeloV2Invalido, match="features"):
        ml_v2.cargar_modelo_v2(str(tmp_path), features_esperadas=["f2", "f1"])


def test_sin_artefacto_avisa_como_regenerarlo(tmp_path):
    _artefacto(tmp_path)
    (tmp_path / ml_v2.ARCHIVO_MODELO).unlink()
    with pytest.raises(FileNotFoundError, match="entrenar_ml_v2"):
        ml_v2.cargar_modelo_v2(str(tmp_path))
