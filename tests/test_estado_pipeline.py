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
    # Si alguna tabla trae TIMESTAMP, el datetime no puede volver entero porque
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


# -- fecha de DATOS vs fecha de REGISTRO (regresion del 2/9/2026) ------------

def test_alertas_scanner_se_mide_por_la_fecha_de_datos():
    """Se media por `scan_fecha` (cuando corrio el scanner) en vez de
    `precio_fecha` (a que cierre corresponde lo que calculo). Es la unica tabla
    del registro cuya columna de fecha "natural" no es la de datos."""
    t = next(t for t in TABLAS if t.nombre == "alertas_scanner")
    assert t.columna == "precio_fecha"
    assert t.columna_registro == "created_at"


def test_ninguna_tabla_se_mide_por_una_columna_de_registro():
    """Guard de nombres para la proxima tabla que se agregue: si la columna de
    diagnostico se llama como un reloj de escritura, es la fecha equivocada."""
    relojes = {"created_at", "updated_at", "computed_at", "calculado_en",
               "scan_fecha", "fetched_at"}
    culpables = [t.nombre for t in TABLAS if t.columna in relojes]
    assert culpables == []


def test_corrida_reciente_con_datos_viejos_es_mezcla():
    """EL CASO REAL DEL 2/9/2026. El scanner habia corrido el dia anterior
    (registro fresco) pero sobre el cierre de anteayer (datos atrasados), y el
    diagnostico decia "todo al dia y alineado" mientras los bots cruzaban
    senal ML de una rueda con tecnico de otra."""
    fechas = _todo_en(D1)
    fechas["alertas_scanner"] = D0            # datos: rueda anterior
    registros = {"alertas_scanner": datetime(2026, 9, 1, 16, 41)}  # corrio ayer

    diag = diagnosticar(fechas, esperado=D1, registros=registros)

    assert diag["mezcla"] is True
    assert "Scanner ML" in diag["desalineadas"]
    assert diag["arreglos"] == ["cron_paso3_scanner.bat"]


def test_el_registro_no_participa_del_diagnostico():
    """Un registro fresco no puede tapar datos viejos: si entrara al calculo,
    volveriamos a tener el falso 'todo al dia' del 2/9."""
    fechas = _todo_en(D1)
    fechas["alertas_scanner"] = D0

    sin_reg = diagnosticar(fechas, esperado=D1)
    con_reg = diagnosticar(fechas, esperado=D1,
                           registros={"alertas_scanner": datetime(2026, 9, 2, 23, 59)})

    assert sin_reg["mezcla"] == con_reg["mezcla"] is True
    assert sin_reg["desalineadas"] == con_reg["desalineadas"]


def test_el_registro_se_expone_para_informar():
    reg = datetime(2026, 9, 1, 16, 41)
    diag = diagnosticar(_todo_en(D1), esperado=D1,
                        registros={"alertas_scanner": reg})
    fila = next(f for f in diag["tablas"] if f["tabla"] == "alertas_scanner")
    assert fila["registro"] == reg


def test_registro_ausente_no_rompe():
    """Las tablas de features no guardan reloj de escritura."""
    diag = diagnosticar(_todo_en(D1), esperado=D1, registros=None)
    assert all(f["registro"] is None for f in diag["tablas"])


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
