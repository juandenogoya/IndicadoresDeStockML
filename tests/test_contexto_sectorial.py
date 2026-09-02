"""
Tests de src/utils/contexto_sectorial.py -- la marca "Sin contexto sectorial".

Lo que se prueba no es la funcion (es un `in` sobre una tupla) sino el
CONTRATO que otros modulos asumen:

  - que la marca coincida con lo que el productor de features realmente
    excluye (si divergen, se marca al ticker equivocado);
  - que un sector ausente se trate como sin contexto, porque asi lo trata el
    SQL del productor (NULL NOT IN (...) es NULL -> fila excluida);
  - que la marca sea vacia y concatenable en el caso normal, que son 196 de
    los 200 tickers.
"""

import re

import pytest

from src.utils.contexto_sectorial import (
    SECTORES_SIN_FEATURES, FEATURES_SECTORIALES, ETIQUETA, ETIQUETA_CORTA,
    sin_contexto_sectorial, marca,
)


# -- el contrato con el productor --------------------------------------------

def test_los_sectores_excluidos_son_los_que_el_productor_deja_afuera():
    """sector_features.py arma su WHERE desde esta misma constante. Si alguien
    vuelve a escribir la lista a mano en el SQL, este test no lo detecta, pero
    el import si: la idea es que no haya dos listas."""
    from src.indicators import sector_features
    assert sector_features.SECTORES_SIN_FEATURES is SECTORES_SIN_FEATURES


def test_las_11_features_son_las_que_lee_el_pipeline():
    from src.pipeline.feature_calculator import _SECTOR_ZCOLS
    assert list(_SECTOR_ZCOLS) == list(FEATURES_SECTORIALES)


def test_son_once_features():
    """El numero aparece en la etiqueta y en la documentacion. Si cambia la
    lista, hay que actualizar el texto que se le muestra al usuario."""
    assert len(FEATURES_SECTORIALES) == 11
    assert len(set(FEATURES_SECTORIALES)) == 11


# -- la clasificacion --------------------------------------------------------

@pytest.mark.parametrize("sector", list(SECTORES_SIN_FEATURES))
def test_sectores_excluidos_no_tienen_contexto(sector):
    assert sin_contexto_sectorial(sector) is True


@pytest.mark.parametrize("sector", [
    "Technology", "Financial Services", "Industrials",
    "Consumer Cyclical", "Consumer Defensive", "Healthcare",
    "Energy", "Basic Materials", "Communication Services",
])
def test_sectores_con_pares_tienen_contexto(sector):
    assert sin_contexto_sectorial(sector) is False


def test_sector_ausente_cuenta_como_sin_contexto():
    """No es defensa generica: el productor tampoco le calcula features a una
    fila con sector NULL, asi que marcarla es lo honesto."""
    assert sin_contexto_sectorial(None) is True
    assert sin_contexto_sectorial("") is True


def test_la_comparacion_es_exacta():
    """'Real Estate Investment Trusts' no es 'Real Estate': un sector nuevo con
    nombre parecido tiene sus propios pares y NO debe heredar la marca."""
    assert sin_contexto_sectorial("Real Estate Investment Trusts") is False
    assert sin_contexto_sectorial("real estate") is False


# -- la etiqueta -------------------------------------------------------------

def test_marca_vacia_en_el_caso_normal():
    """Concatenable sin condicionales: es lo que hacen los llamadores."""
    assert marca("Technology") == ""
    assert f"COMPRA_FUERTE{marca('Technology')}" == "COMPRA_FUERTE"


def test_marca_presente_y_delimitada():
    m = marca("Utilities")
    assert ETIQUETA in m
    assert m.startswith(" [") and m.endswith("]")


def test_marca_corta_para_tablas_de_ancho_fijo():
    corta = marca("Real Estate", corta=True)
    assert ETIQUETA_CORTA in corta
    assert len(corta) < len(marca("Real Estate"))


def test_las_etiquetas_son_ascii():
    """Windows cp1252: un caracter fuera de ASCII revienta el print del scanner
    y el log del cron (regla #2 del proyecto)."""
    for texto in (ETIQUETA, ETIQUETA_CORTA, marca("Utilities")):
        texto.encode("ascii")
        assert re.fullmatch(r"[ -~]*", texto)
