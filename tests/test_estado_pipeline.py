"""
Tests de src/utils/estado_pipeline.py -- el diagnostico de la rutina diaria.

Lo que se prueba es la DISTINCION que motiva el modulo: "todo viejo pero
coherente" (esta bien, es la convencion del proyecto) tiene que dar un
resultado distinto de "mezcla de ruedas" (esta mal). Un modulo que confunda
esos dos casos es peor que no tenerlo, porque frena la rutina normal o deja
pasar la corrupta.
"""

from datetime import date, datetime

import pytest

from src.utils.estado_pipeline import (
    TABLAS, ANCLA, CRUDO_OPCIONES, diagnosticar, enumerar, resumen, _a_fecha,
)

D1 = date(2026, 9, 1)   # ultimo cierre
D0 = date(2026, 8, 31)  # rueda anterior


def _todo_en(f):
    """Todas las tablas conocidas en la misma fecha."""
    return {t.nombre: f for t in TABLAS}


# -- coercion de fechas ------------------------------------------------------

def test_a_fecha_acepta_date_datetime_y_texto():
    assert _a_fecha(D1) == D1
    assert _a_fecha("2026-09-01") == D1
    # alertas_scanner es TIMESTAMP: el datetime no puede volver entero, porque
    # despues se le resta un date. datetime es subclase de date -> el orden de
    # los isinstance importa.
    assert _a_fecha(datetime(2026, 9, 1, 21, 0, 0)) == D1
    assert type(_a_fecha(datetime(2026, 9, 1, 21, 0))) is date


def test_a_fecha_devuelve_none_ante_basura():
    assert _a_fecha(None) is None
    assert _a_fecha("no es fecha") is None


# -- el eje 1: antiguedad NO es error ----------------------------------------

def test_todo_viejo_pero_coherente_no_es_mezcla():
    """La convencion del proyecto: los bots operan con el cierre anterior el
    73% de las veces. Eso NO debe frenar nada."""
    diag = diagnosticar(_todo_en(D0), esperado=D1)
    assert diag["mezcla"] is False
    assert diag["desalineadas"] == []
    assert diag["ancla_al_dia"] is False      # si se informa que esta atrasado
    assert diag["atraso_ancla"] == 1


def test_todo_al_dia():
    diag = diagnosticar(_todo_en(D1), esperado=D1)
    assert diag["mezcla"] is False
    assert diag["ancla_al_dia"] is True
    assert diag["atraso_ancla"] == 0
    assert diag["arreglos"] == []
    assert "al dia" in resumen(diag)


# -- el eje 2: mezcla SI es error --------------------------------------------

def test_opciones_derivadas_atras_es_mezcla():
    """El caso real del 2/9/2026: precios al 1/9, derivadas de opciones al 31/8
    porque no se corrio ft_run_diario."""
    fechas = _todo_en(D1)
    fechas["opciones_pcr_plazo_diario"] = D0
    fechas["opciones_sector_pcr_plazo_diario"] = D0
    diag = diagnosticar(fechas, esperado=D1)

    assert diag["mezcla"] is True
    assert "Opciones (PCR plazo)" in diag["desalineadas"]
    # El crudo esta al dia -> NO es el caso irrecuperable.
    assert diag["snapshot_ausente"] is False
    assert any("ft_run_diario" in a for a in diag["arreglos"])


def test_mezcla_nombra_el_bat_que_corresponde():
    fechas = _todo_en(D1)
    fechas["features_precio_accion"] = D0
    diag = diagnosticar(fechas, esperado=D1)
    assert diag["arreglos"] == ["cron_paso2_features.bat"]


# -- el caso grave: el crudo no llego ----------------------------------------

def test_crudo_de_opciones_atras_se_reporta_aparte():
    """Si falta el CRUDO, el problema no es procesar: es que la chain no se
    capturo, y es irrecuperable apenas abre el mercado. Otro comando, y con
    reloj."""
    fechas = _todo_en(D1)
    fechas[CRUDO_OPCIONES] = D0
    diag = diagnosticar(fechas, esperado=D1)

    assert diag["snapshot_ausente"] is True
    assert diag["mezcla"] is True
    assert "sync_opciones_railway_to_local.bat" in diag["arreglos"]
    assert "IRRECUPERABLE" in resumen(diag)


def test_el_resumen_prioriza_el_caso_irrecuperable():
    """Con crudo ausente Y otras tablas atras, manda el irrecuperable: es el
    unico que tiene plazo."""
    fechas = _todo_en(D1)
    fechas[CRUDO_OPCIONES] = D0
    fechas["features_precio_accion"] = D0
    diag = diagnosticar(fechas, esperado=D1)
    assert "IRRECUPERABLE" in resumen(diag)


# -- salidas vs insumos ------------------------------------------------------

def test_las_salidas_no_cuentan_como_mezcla():
    """veredictos y equity son SALIDAS de la rutina: que esten atrasadas es un
    sintoma de que falto correr algo, no una causa de decisiones corruptas.
    No deben frenar los bots."""
    fechas = _todo_en(D1)
    fechas["veredictos_universo_diario"] = D0
    fechas["ft_equity_diaria"] = D0
    diag = diagnosticar(fechas, esperado=D1)
    assert diag["mezcla"] is False
    assert diag["desalineadas"] == []


# -- bordes ------------------------------------------------------------------

def test_tabla_mas_fresca_que_el_ancla_no_es_mezcla():
    """Una tabla ADELANTE de los precios es un paso ya corrido, no un peligro."""
    fechas = _todo_en(D0)
    fechas["opciones_snapshot"] = D1
    diag = diagnosticar(fechas, esperado=D1)
    assert diag["mezcla"] is False
    assert diag["snapshot_ausente"] is False


def test_tabla_ausente_se_omite():
    """Una instalacion que todavia no creo veredictos_universo_diario no debe
    romper el diagnostico."""
    fechas = _todo_en(D1)
    del fechas["veredictos_universo_diario"]
    diag = diagnosticar(fechas, esperado=D1)
    nombres = [f["tabla"] for f in diag["tablas"]]
    assert "veredictos_universo_diario" not in nombres
    assert diag["mezcla"] is False


def test_sin_ancla_no_revienta():
    diag = diagnosticar({}, esperado=D1)
    assert diag["ancla"] is None
    assert diag["mezcla"] is False
    assert "vacia" in resumen(diag)


def test_ancla_atrasada_sugiere_el_paso_1():
    diag = diagnosticar(_todo_en(D0), esperado=D1)
    assert diag["arreglos"] == ["cron_paso1_precios_yq.bat"]


@pytest.mark.parametrize("items,esperado", [
    ([], ""),
    (["A"], "A"),
    (["A", "B"], "A y B"),
    (["A", "B", "C"], "A, B y C"),
])
def test_enumerar(items, esperado):
    assert enumerar(items) == esperado
