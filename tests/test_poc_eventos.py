"""
Tests de src/utils/poc_eventos.py.

Los dos que importan de verdad:
  - la version vectorizada da EXACTAMENTE lo mismo que la de referencia lenta;
  - un evento en t no cambia cuando llegan barras nuevas (invariancia), que es la
    familia de bugs de este repo (CLAUDE.md: features_market_structure).
"""

import numpy as np
import pytest

from src.utils import perfil_precio as pp
from src.utils import poc_eventos as pe


# -- bandas --------------------------------------------------------------------

def test_el_ancho_de_un_tramo_no_depende_de_donde_este():
    """La grilla es geometrica: n franjas miden lo mismo en 10 USD que en 600."""
    n = 19
    anchos = []
    for k0 in (-200, 0, 150, 1100):
        val, vah = pe.banda_de_tramo(k0, 0, n)
        anchos.append((vah / val - 1.0) * 100.0)
    assert max(anchos) - min(anchos) < 1e-9
    assert abs(anchos[0] - pe.ancho_pct_de_tramo(n)) < 1e-9


def test_la_banda_de_control_mide_igual_que_la_real():
    rng = np.random.default_rng(0)
    p = rng.random(60) + 0.01
    a, b = 20, 27
    real = pe.banda_de_tramo(0, a, b - a + 1)
    ancho_real = (real[1] / real[0] - 1.0) * 100.0
    for modo in ("visitado", "denso"):
        for s in range(40):
            banda = pe.banda_control(p, 0, a, b, modo, np.random.default_rng(s))
            if banda is None:
                continue
            ancho = (banda[1] / banda[0] - 1.0) * 100.0
            assert abs(ancho - ancho_real) < 1e-9


def test_la_banda_de_control_nunca_se_solapa_con_el_area_de_valor():
    p = np.ones(50)
    a, b = 20, 25
    n = b - a + 1
    for s in pe.posiciones_control(p, a, b, "visitado"):
        assert (s + n - 1 < a) or (s > b)


def test_denso_es_un_subconjunto_de_visitado():
    rng = np.random.default_rng(3)
    p = rng.random(80)
    p[p < 0.3] = 0.0                      # franjas no visitadas
    a, b = 30, 34
    vis = set(pe.posiciones_control(p, a, b, "visitado").tolist())
    den = set(pe.posiciones_control(p, a, b, "denso").tolist())
    assert den <= vis
    assert len(den) < len(vis)


def test_sin_posicion_valida_devuelve_none():
    p = np.ones(6)
    assert pe.banda_control(p, 0, 0, 5, "visitado", np.random.default_rng(0)) is None


def test_modo_desconocido_falla():
    with pytest.raises(ValueError):
        pe.posiciones_control(np.ones(20), 5, 8, "cualquiera")


# -- eventos -------------------------------------------------------------------

def _serie(seed, n=400):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    val = close * (1.0 - rng.uniform(0.02, 0.06, n))
    vah = close * (1.0 + rng.uniform(0.02, 0.06, n))
    # desalinear la banda para que haya entradas y salidas de verdad
    val = np.roll(val, 7)
    vah = np.roll(vah, 7)
    return close, val, vah


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vectorizado_igual_a_la_referencia(seed):
    close, val, vah = _serie(seed)
    assert pe.detectar_eventos(close, val, vah) == pe._detectar_eventos_ref(close, val, vah)


@pytest.mark.parametrize("fuera", [1, 3, 5, 10])
def test_igual_a_la_referencia_con_otros_fuera(fuera):
    close, val, vah = _serie(11)
    assert (pe.detectar_eventos(close, val, vah, fuera)
            == pe._detectar_eventos_ref(close, val, vah, fuera))


def test_invariancia_las_barras_nuevas_no_cambian_los_eventos_viejos():
    """calcular(datos[:t+1]) tiene que coincidir con calcular(datos) hasta t."""
    close, val, vah = _serie(7)
    completo = pe.detectar_eventos(close, val, vah)
    for corte in (120, 250, 399):
        parcial = pe.detectar_eventos(close[:corte], val[:corte], vah[:corte])
        assert parcial == [e for e in completo if e.i < corte]


def test_usa_la_banda_de_t_menos_1_y_no_la_de_t():
    # banda de la rueda 4 = [90, 110]; el cierre de la 5 entra ahi.
    close = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 100.0])
    val = np.array([np.nan] * 4 + [90.0, 999.0])
    vah = np.array([np.nan] * 4 + [110.0, 999.0])
    # con la banda de t-1 (indice 4) hay evento en 5
    ev = pe.detectar_eventos(close, val, vah, fuera=4)
    assert len(ev) == 1 and ev[0].i == 5
    assert ev[0].val == 90.0 and ev[0].vah == 110.0


def test_direccion_desde_arriba_y_desde_abajo():
    val = np.full(8, 90.0)
    vah = np.full(8, 110.0)
    arriba = np.array([200.0] * 6 + [100.0, 100.0])
    abajo = np.array([10.0] * 6 + [100.0, 100.0])
    assert pe.detectar_eventos(arriba, val, vah, fuera=5)[0].direccion == pe.DESDE_ARRIBA
    assert pe.detectar_eventos(abajo, val, vah, fuera=5)[0].direccion == pe.DESDE_ABAJO


def test_solo_la_rueda_de_ENTRADA_cuenta_no_las_de_adentro():
    val = np.full(10, 90.0)
    vah = np.full(10, 110.0)
    close = np.array([200.0, 200.0, 200.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    # entra en el indice 3 (las 3 previas estuvieron afuera) y se queda adentro:
    # las ruedas 4..9 NO son eventos, porque en t-1 ya estaba adentro.
    assert [e.i for e in pe.detectar_eventos(close, val, vah, fuera=3)] == [3]


def test_no_hay_evento_si_falta_historia_para_el_fuera():
    val = np.full(10, 90.0)
    vah = np.full(10, 110.0)
    close = np.array([200.0, 200.0, 200.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    # con fuera=5 la entrada del indice 3 no califica: no hay 5 ruedas previas
    assert pe.detectar_eventos(close, val, vah, fuera=5) == []


def test_una_sola_entrada_por_visita():
    val = np.full(20, 90.0)
    vah = np.full(20, 110.0)
    close = np.array([200.0] * 6 + [100.0] * 8 + [200.0] * 6)
    ev = pe.detectar_eventos(close, val, vah, fuera=5)
    assert len(ev) == 1 and ev[0].i == 6


def test_banda_no_finita_no_genera_evento():
    close = np.full(12, 100.0)
    val = np.full(12, np.nan)
    vah = np.full(12, np.nan)
    assert pe.detectar_eventos(close, val, vah) == []


def test_serie_vacia_o_corta():
    assert pe.detectar_eventos([], [], []) == []
    assert pe.detectar_eventos([1.0], [1.0], [1.0]) == []


# -- exceso --------------------------------------------------------------------

def test_retorno_forward():
    c = np.array([100.0, 110.0, 121.0, 133.1])
    r = pe.retorno_forward(c, 1)
    assert np.allclose(r[:3], 0.1)
    assert np.isnan(r[3])
    r2 = pe.retorno_forward(c, 2)
    assert np.allclose(r2[:2], 0.21)
    assert np.isnan(r2[2]) and np.isnan(r2[3])


def test_retorno_forward_horizonte_mayor_que_la_serie():
    assert np.all(np.isnan(pe.retorno_forward(np.array([1.0, 2.0]), 5)))


def test_exceso_resta_el_universo():
    ev = [pe.Evento(i=1, direccion=pe.DESDE_ARRIBA, val=1.0, vah=2.0)]
    fwd_t = np.array([np.nan, 0.05, np.nan])
    fwd_u = np.array([np.nan, 0.02, np.nan])
    out = pe.exceso_eventos(ev, fwd_t, fwd_u)
    assert len(out) == 1
    assert abs(out[0]["exceso_pp"] - 3.0) < 1e-9
    assert abs(out[0]["ret_pp"] - 5.0) < 1e-9


def test_exceso_descarta_el_evento_sin_dato():
    ev = [pe.Evento(i=0, direccion=pe.DESDE_ABAJO, val=1.0, vah=2.0)]
    assert pe.exceso_eventos(ev, np.array([np.nan]), np.array([0.01])) == []


# -- lectura contra el control -------------------------------------------------

def test_fuera_de_banda():
    muestras = list(range(0, 100))
    assert pe.fuera_de_banda(200.0, muestras) == "arriba"
    assert pe.fuera_de_banda(-5.0, muestras) == "abajo"
    assert pe.fuera_de_banda(50.0, muestras) == "adentro"
    assert pe.fuera_de_banda(1.0, []) is None


def test_fuera_de_banda_ignora_nan():
    assert pe.fuera_de_banda(50.0, [np.nan, 0.0, 100.0]) == "adentro"


# -- integracion con perfil_precio ---------------------------------------------

def test_la_banda_del_area_de_valor_se_reconstruye_con_banda_de_tramo():
    """El area de valor que devuelve perfil_precio es un tramo de franjas."""
    rng = np.random.default_rng(5)
    n = 200
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high, low = close * 1.01, close * 0.99
    vol = rng.uniform(1e6, 5e6, n)
    k0, p = pp.perfil_bins(high, low, vol)
    k, a, b = pp._area_valor(p, pp.COBERTURA)
    perfil = pp.calcular(high, low, vol)
    val, vah = pe.banda_de_tramo(k0, a, b - a + 1)
    assert abs(val - perfil.val) < 1e-9
    assert abs(vah - perfil.vah) < 1e-9
