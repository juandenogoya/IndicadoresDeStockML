"""
Tests de la logica de eleccion de fuente en refresh_acciones_circulacion.

Se prueba `construir_mejor`, que es donde vive la decision y donde no habia
cobertura: que nivel se usa, si se rebasa o no, y cuando entra Polygon. La
funcion es pura respecto de sus argumentos (la DB queda del lado de
series_sec/series_polygon/splits_por_ticker), asi que se puede importar el
script y llamarla sin base de datos.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "refresh_acciones_circulacion",
    os.path.join(ROOT, "scripts", "refresh_acciones_circulacion.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _serie(pares):
    return [{"fecha": f, "shares": float(s)} for f, s in pares]


# ------------------------------------------------------------------ etiqueta

def test_polygon_no_lleva_prefijo_sec():
    """La etiqueta persistida tiene que decir la verdad sobre el origen."""
    assert R._etiqueta("portada", False) == "sec_portada"
    assert R._etiqueta("portada", True) == "sec_portada_rb"
    assert R._etiqueta("polygon", False) == "polygon"
    assert R._etiqueta("polygon", True) == "polygon_rb"


def test_la_etiqueta_mas_larga_entra_en_la_columna():
    """
    varchar(40) tras la migracion acciones_fuente_mas_ancha. El refresh moria
    con StringDataRightTruncation DESPUES de bajar los 200 tickers, o sea que
    pasarse costaba la corrida entera.
    """
    largo = max(len(R._etiqueta(f, rb))
                for f in R.PREFERENCIA_FUENTE for rb in (False, True))
    assert largo <= 40


# -------------------------------------------------------------- eleccion

def test_polygon_es_el_ultimo_recurso():
    """Si la portada ya valida y extiende, Polygon no se usa aunque este."""
    yah = _serie([("2022-12-31", 1000), ("2023-12-31", 990)])
    niveles = {
        "portada": _serie([("2021-03-31", 1030), ("2022-12-31", 1000)]),
        "polygon": _serie([("2021-03-31", 1031), ("2022-12-31", 1001)]),
    }
    serie, diag = construir(yah, niveles)
    assert diag["extendido"] is True
    assert diag["fuente_sec"] == "portada"


def test_polygon_entra_cuando_sec_no_tiene_nada():
    """
    El caso V: los tres niveles vienen VACIOS porque companyfacts descarta los
    hechos dimensionados y Visa taguea cada conteo por clase. No es que el
    nivel elegido sea malo -- no hay ninguno.
    """
    yah = _serie([("2022-12-31", 1000), ("2023-12-31", 990)])
    niveles = {"polygon": _serie([("2021-03-31", 1030), ("2022-12-31", 1002)])}
    serie, diag = construir(yah, niveles)
    assert diag["extendido"] is True
    assert diag["fuente_sec"] == "polygon"
    assert serie[0]["fecha"] == "2021-03-31"


def test_sin_ninguna_fuente_no_revienta():
    yah = _serie([("2022-12-31", 1000)])
    serie, diag = construir(yah, {})
    assert diag["extendido"] is False
    # La serie vuelve enriquecida (periodo, fuente), pero sin puntos agregados.
    assert [(p["fecha"], p["shares"]) for p in serie] ==            [(p["fecha"], p["shares"]) for p in yah]


# ------------------------------------------------------------------ rebase

def test_se_prueba_primero_sin_rebase():
    """
    El caso HON. Polygon publica 1,061 para la escision de Solstice, que es un
    ajuste de PRECIO y no mueve el conteo. La serie sin rebase ya valida, asi
    que un ratio espurio no tiene que poder desplazarla.
    """
    yah = _serie([("2022-12-31", 1000), ("2023-12-31", 990)])
    niveles = {"portada": _serie([("2021-03-31", 1030), ("2022-12-31", 1000)])}
    serie, diag = construir(yah, niveles, splits=[{"fecha": "2026-01-01",
                                                   "ratio": 1.061}])
    assert diag["extendido"] is True
    assert diag["rebase"] == 0, "la sana gana: no se rebaso"


def test_el_rebase_rescata_lo_que_sin_el_se_pierde():
    """
    El caso WMT/NVDA al reves: en la base vieja la serie NO aparea contra
    yahoo, y solo entra despues de llevarla a la base de hoy.
    """
    # El punto SEC POSTERIOR al split ya viene en la base nueva; el anterior no.
    # Esa es la forma real de la serie, y es la que hace fallar la variante
    # sin rebase: 1030 contra 3000 es un salto de x2,9.
    yah = _serie([("2022-12-31", 3000), ("2023-12-31", 2970)])
    niveles = {"portada": _serie([("2021-03-31", 1030), ("2022-12-31", 3000)])}

    sin_rebase, d0 = construir(yah, niveles)
    assert d0["extendido"] is False, "sin rebase la serie no aparea: ese es el punto"

    serie, diag = construir(yah, niveles, splits=[{"fecha": "2022-06-01",
                                                   "ratio": 3.0}])
    assert diag["extendido"] is True
    assert diag["rebase"] > 0
    assert serie[0]["shares"] == pytest.approx(3090.0)


def construir(yah, niveles, splits=None):
    return R.construir_mejor(yah, niveles, "2021-01-01", splits)
