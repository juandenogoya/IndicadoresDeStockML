"""
Tests de src/pipeline/alert_classifier.py con cortes de probabilidad parametrizables
(Etapa 3d) y de la suma de la v2 en el scanner (scripts/cron_diario.py).

Lo que se fija:
  - sin `cortes_ml` el clasificador es EXACTAMENTE la v1 (la estrategia de control
    no se puede mover por agregar la v2);
  - con los cortes de la v2 una misma probabilidad suma los puntos de su escala;
  - una falla de la v2 deja sus columnas en None y nunca rompe el resultado de la v1.
"""

import importlib.util
import os

import numpy as np
import pytest

from src.pipeline import alert_classifier as ac

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SIG = dict(ml_prob_ganancia=0.5, pa_ev1=1, pa_ev2=0, pa_ev3=0, pa_ev4=0,
           bear_bos10=0, bear_choch10=0, bear_estructura=0)
META = {"score_ponderado": 0.65}
CORTES_V2 = (0.768849, 0.606518, 0.557172, 0.471122, 0.364417)


@pytest.mark.parametrize("prob,pts", [
    (0.80, 30), (0.75, 30), (0.7499, 22), (0.65, 22), (0.60, 14), (0.55, 14),
    (0.50, 5), (0.45, 5), (0.40, -5), (0.35, -5), (0.10, -15),
])
def test_puntos_ml_de_la_v1_sin_cambios(prob, pts):
    assert ac._puntos_ml(prob)[0] == pts


def test_sin_cortes_es_la_v1_en_toda_la_escala():
    for p in np.linspace(0, 1, 201):
        s = dict(SIG, ml_prob_ganancia=float(p))
        assert ac.clasificar_alerta(s, META) == ac.clasificar_alerta(s, META, cortes_ml=ac.CORTES_ML_V1)


def test_descripcion_de_la_v1_sin_cambios():
    assert ac._puntos_ml(0.80) == (30, "ML fuerte alcista (80%)")
    assert ac._puntos_ml(0.20) == (-15, "ML bajista (20%)")


def test_cortes_v2_suman_puntos_con_su_escala():
    assert ac._puntos_ml(0.61)[0] == 14
    assert ac._puntos_ml(0.61, CORTES_V2)[0] == 22
    s = dict(SIG, ml_prob_ganancia=0.61)
    score_v1, _, _ = ac.clasificar_alerta(s, META)
    score_v2, _, _ = ac.clasificar_alerta(s, META, cortes_ml=CORTES_V2)
    assert score_v2 - score_v1 == 8


@pytest.mark.parametrize("cortes", [
    (0.7, 0.6),                    # cantidad distinta de niveles
    (0.5, 0.6, 0.4, 0.3, 0.2),     # no decreciente
    (0.7, 0.6, 0.6, 0.4, 0.3),     # dos niveles colapsados
])
def test_cortes_invalidos(cortes):
    with pytest.raises(ValueError):
        ac.clasificar_alerta(SIG, META, cortes_ml=cortes)


def test_ml_v2_usa_los_cortes_del_clasificador():
    from src.ml import ml_v2
    assert ml_v2.CORTES_V1 == ac.CORTES_ML_V1


# -- la v2 en el scanner ----------------------------------------------------------

def _cron_diario():
    spec = importlib.util.spec_from_file_location(
        "cron_diario_test", os.path.join(RAIZ, "scripts", "cron_diario.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Modelo:
    classes_ = np.array([0, 1])

    def __init__(self, p=None, falla=False):
        self.p, self.falla = p, falla

    def predict_proba(self, X):
        if self.falla:
            raise RuntimeError("modelo roto")
        return np.array([[1 - self.p, self.p]])


def _v2(modelo):
    meta = {"version": "ml_v2_test", "features": ["f1"], "columnas_cero": []}
    return {"modelo": modelo, "meta": meta, "cortes": CORTES_V2}


def test_agregar_v2_suma_las_cuatro_columnas():
    cd = _cron_diario()
    r = cd._agregar_v2({"ticker": "AAA"}, dict(SIG), META, {"f1": 1.0}, _v2(_Modelo(0.61234)))
    assert r["ml_prob_v2"] == 0.6123
    assert r["ml_modelo_v2"] == "ml_v2_test"
    score, nivel, _ = ac.clasificar_alerta(dict(SIG, ml_prob_ganancia=0.6123), META, cortes_ml=CORTES_V2)
    assert (r["alert_score_v2"], r["alert_nivel_v2"]) == (score, nivel)


def test_agregar_v2_sin_modelo_deja_none():
    cd = _cron_diario()
    r = cd._agregar_v2({"ticker": "AAA"}, dict(SIG), META, {"f1": 1.0}, None)
    assert all(r[c] is None for c in ("ml_prob_v2", "ml_modelo_v2", "alert_score_v2", "alert_nivel_v2"))


def test_agregar_v2_con_falla_no_toca_la_v1():
    cd = _cron_diario()
    base = {"ticker": "AAA", "alert_nivel": "COMPRA", "alert_score": 65.0}
    r = cd._agregar_v2(dict(base), dict(SIG), META, {"f1": 1.0}, _v2(_Modelo(falla=True)))
    assert r["alert_nivel"] == "COMPRA" and r["alert_score"] == 65.0
    assert r["alert_nivel_v2"] is None and r["ml_prob_v2"] is None


def test_persistencia_incluye_las_columnas_v2():
    cd = _cron_diario()
    import inspect
    fuente = inspect.getsource(cd._persistir_alertas)
    for col in ("ml_prob_v2", "ml_modelo_v2", "alert_score_v2", "alert_nivel_v2"):
        assert f'"{col}"' in fuente
