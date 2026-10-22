"""
Tests de src/utils/ft_fill.py -- realismo del fill de FT.

Lo que importa fijar (docs/forward_testing/ANALISIS_FILL.md sec. 2):
  - las dos patas se mueven JUNTAS: si falta una, la operacion no entra;
  - nunca se asume un numero donde no hay dato (None, NaN, precio <= 0);
  - la unidad de independencia es la RUEDA, no la operacion;
  - la regla de lectura exige las TRES condiciones.
"""
import math

import pytest

from src.utils import ft_fill as F


def _fills(e_act, s_act, e_ap=None, s_ap=None, e_ci=None, s_ci=None):
    return {
        "entrada": {"actual": e_act, "apertura_sig": e_ap, "cierre_sig": e_ci},
        "salida": {"actual": s_act, "apertura_sig": s_ap, "cierre_sig": s_ci},
    }


# --- retorno -----------------------------------------------------------------

def test_retorno_basico():
    assert F.retorno(100.0, 110.0) == pytest.approx(10.0)
    assert F.retorno(100.0, 90.0) == pytest.approx(-10.0)


@pytest.mark.parametrize("pe, ps", [
    (None, 110.0), (100.0, None), (0.0, 110.0), (-5.0, 110.0),
    (float("nan"), 110.0), (100.0, float("nan")),
])
def test_retorno_sin_dato_es_none(pe, ps):
    assert F.retorno(pe, ps) is None


# --- las dos patas juntas ----------------------------------------------------

def test_revaluar_usa_la_misma_variante_en_las_dos_patas():
    f = _fills(100.0, 110.0, e_ap=101.0, s_ap=112.0)
    assert F.revaluar(f, "actual") == pytest.approx(10.0)
    # (112-101)/101 = 10,891%
    assert F.revaluar(f, "apertura_sig") == pytest.approx(10.8910891, abs=1e-6)


def test_una_sola_pata_no_alcanza():
    """El caso que motiva el diseno: con la salida re-valuada y la entrada no,
    saldria un artefacto que no existe. Tiene que devolver None."""
    solo_salida = _fills(100.0, 110.0, e_ap=None, s_ap=112.0)
    assert F.revaluar(solo_salida, "apertura_sig") is None
    solo_entrada = _fills(100.0, 110.0, e_ap=101.0, s_ap=None)
    assert F.revaluar(solo_entrada, "apertura_sig") is None


def test_comparar_operacion_devuelve_la_diferencia_pareada():
    f = _fills(100.0, 110.0, e_ap=100.0, s_ap=105.0)
    r = F.comparar_operacion(f, "apertura_sig")
    assert r["actual"] == pytest.approx(10.0)
    assert r["variante"] == pytest.approx(5.0)
    assert r["dif"] == pytest.approx(-5.0)


def test_comparar_operacion_sin_variante_no_entra():
    assert F.comparar_operacion(_fills(100.0, 110.0), "apertura_sig") is None


def test_variante_desconocida_falla_fuerte():
    with pytest.raises(ValueError):
        F.precio_de(_fills(100.0, 110.0), "entrada", "apertura_de_ayer")


def test_el_efecto_se_cancela_cuando_las_dos_patas_se_mueven_igual():
    """Comprar 1% peor y vender 1% peor deja el retorno casi igual: es
    exactamente el fenomeno que el analisis tiene que poder ver."""
    f = _fills(100.0, 110.0, e_ap=101.0, s_ap=111.1)
    assert F.revaluar(f, "apertura_sig") == pytest.approx(10.0, abs=1e-9)
    assert F.comparar_operacion(f, "apertura_sig")["dif"] == pytest.approx(0.0, abs=1e-9)


# --- la rueda como unidad ----------------------------------------------------

def test_la_unidad_es_la_rueda_no_la_operacion():
    """Tres operaciones el mismo dia pesan como UNA. Con la rueda como unidad
    hay 2 observaciones, no 4."""
    difs = [3.0, 3.0, 3.0, -3.0]
    ruedas = ["2026-05-04"] * 3 + ["2026-05-05"]
    res = F.resumen_variante(difs, ruedas, [0.0] * 4, difs)
    assert res["ops"] == 4
    assert res["ruedas"] == 2
    assert res["dif_media"] == pytest.approx(0.0)       # (3 + -3) / 2
    assert res["dif_media_ops"] == pytest.approx(1.5)   # (3+3+3-3) / 4


def test_percentil_y_tasa_de_ganadoras():
    rets = [-10.0, -1.0, 0.0, 1.0, 10.0]
    res = F.resumen_variante([0.0] * 5, list("abcde"), rets, rets)
    assert res["ganadoras_actual"] == pytest.approx(40.0)
    assert res["p5_actual"] == pytest.approx(-8.2, abs=1e-9)


# --- consistencia por mes ----------------------------------------------------

def test_signo_por_mes_promedia_dentro_del_mes():
    por_mes = F.signo_por_mes([1.0, 3.0, -2.0], ["2026-04", "2026-04", "2026-05"])
    assert por_mes == {"2026-04": pytest.approx(2.0), "2026-05": pytest.approx(-2.0)}


def test_meses_al_mismo_lado():
    por_mes = {"2026-04": -1.0, "2026-05": -2.0, "2026-06": 0.5, "2026-07": -0.1}
    assert F.meses_al_mismo_lado(por_mes, -1.0) == 3
    assert F.meses_al_mismo_lado(por_mes, 1.0) == 1
    assert F.meses_al_mismo_lado(por_mes, 0.0) == 0


def test_un_mes_en_cero_no_cuenta_para_ninguno():
    por_mes = {"2026-04": 0.0, "2026-05": 1.0}
    assert F.meses_al_mismo_lado(por_mes, 1.0) == 1
    assert F.meses_al_mismo_lado(por_mes, -1.0) == 0


# --- signo del total por estrategia ------------------------------------------

def test_da_vuelta_el_signo_detecta_solo_los_que_cambian():
    act = {"A": 12.0, "B": -3.0, "C": 5.0}
    var = {"A": -1.0, "B": -8.0, "C": 2.0}
    assert F.da_vuelta_el_signo(act, var) == ["A"]


def test_da_vuelta_el_signo_ignora_lo_que_falta():
    assert F.da_vuelta_el_signo({"A": 1.0}, {}) == []


# --- la regla de lectura: las TRES o nada ------------------------------------

def _res(dif, lo, hi, base):
    return {"dif_media": dif, "ic95_lo": lo, "ic95_hi": hi, "ret_actual": base}


MESES_OK = {"2026-04": -1.0, "2026-05": -1.0, "2026-06": -1.0,
            "2026-07": -1.0, "2026-08": 0.2, "2026-09": 0.1}


def test_pasa_cuando_se_cumplen_las_tres():
    # dif -0,5 sobre un control de 1,0 => 50% >= 25%
    v = F.leer(_res(-0.5, -0.8, -0.2, 1.0), MESES_OK, [])
    assert v["ic95_excluye_cero"] and v["material"] and v["consistente"]
    assert v["pasa"] is True


def test_no_pasa_si_el_ic_incluye_el_cero():
    v = F.leer(_res(-0.5, -0.9, 0.1, 1.0), MESES_OK, [])
    assert v["ic95_excluye_cero"] is False
    assert v["pasa"] is False


def test_no_pasa_si_la_magnitud_es_chica_aunque_el_ic_excluya_el_cero():
    """El caso que motiva el umbral: con 3.129 operaciones casi cualquier
    diferencia excluye el cero. Chica y significativa no alcanza."""
    v = F.leer(_res(-0.10, -0.15, -0.05, 1.0), MESES_OK, [])
    assert v["ic95_excluye_cero"] is True
    assert v["material_por_magnitud"] is False
    assert v["pasa"] is False


def test_una_magnitud_chica_pasa_si_da_vuelta_un_signo():
    v = F.leer(_res(-0.10, -0.15, -0.05, 1.0), MESES_OK, ["FT_SMC_v1"])
    assert v["material_por_signo"] is True
    assert v["pasa"] is True


def test_no_pasa_sin_consistencia_mensual():
    alternado = {"2026-04": -1.0, "2026-05": 1.0, "2026-06": -1.0,
                 "2026-07": 1.0, "2026-08": -1.0, "2026-09": 1.0}
    v = F.leer(_res(-0.5, -0.8, -0.2, 1.0), alternado, [])
    assert v["meses_al_mismo_lado"] == 3
    assert v["consistente"] is False
    assert v["pasa"] is False


def test_el_umbral_se_calcula_sobre_el_control():
    v = F.leer(_res(-0.5, -0.8, -0.2, 4.0), MESES_OK, [])
    assert v["umbral_material"] == pytest.approx(1.0)   # 25% de 4,0
    assert v["material_por_magnitud"] is False          # 0,5 < 1,0


def test_control_negativo_usa_el_valor_absoluto():
    v = F.leer(_res(-0.5, -0.8, -0.2, -1.0), MESES_OK, [])
    assert v["umbral_material"] == pytest.approx(0.25)
    assert v["material_por_magnitud"] is True


# --- constantes del pre-registro, que no se muevan sin querer ----------------

def test_las_constantes_son_las_pre_registradas():
    assert F.FRACCION_MATERIAL == 0.25
    assert F.MESES_MINIMOS == 4
    assert F.MESES_TOTALES == 6
    assert F.PRINCIPAL == "apertura_sig"
    assert F.CONTROL == "actual"
    assert F.VARIANTES == ("actual", "apertura_sig", "cierre_sig")
    assert "ESTRATEGIA_DISCONTINUADA" in F.MOTIVOS_EXCLUIDOS
    assert "DELISTING" in F.MOTIVOS_EXCLUIDOS
