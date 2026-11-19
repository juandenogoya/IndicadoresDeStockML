"""
Tests de src/utils/niveles_precio.py.

Los que importan:
  - el precio del giro sale de LA BARRA DEL GIRO, no de la de confirmacion (es el
    error que tiene leer `features_estructura.is_sh_N` y tomar el high de esa fila);
  - invariancia: las barras nuevas no cambian los giros ya confirmados;
  - un nivel sintetico con toques conocidos se encuentra, y en el precio correcto.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.estructura import swings_en_barra
from src.utils import niveles_precio as npx


def _fechas(n, inicio="2021-01-04"):
    return pd.bdate_range(inicio, periods=n).to_list()


def _zigzag(piso, techo, tramo, ciclos):
    """Serie que baja al piso y sube al techo, `ciclos` veces."""
    out = []
    for _ in range(ciclos):
        out += list(np.linspace(techo, piso, tramo))
        out += list(np.linspace(piso, techo, tramo))[1:]
    return np.array(out)


# -- giros ---------------------------------------------------------------------

def test_el_precio_del_giro_es_el_de_su_barra_no_el_de_la_confirmacion():
    c = _zigzag(100.0, 120.0, 12, 3)
    g = npx.giros_confirmados(c, c, n=5)
    assert g, "deberia haber giros"
    for x in g:
        assert x.precio == pytest.approx(c[x.i]), "el precio no es el de la barra del giro"
    # y los minimos detectados valen 100, no un valor 5 barras despues
    lows = [x.precio for x in g if x.tipo == npx.L]
    assert lows and all(abs(p - 100.0) < 1e-9 for p in lows)


def test_un_swing_sin_confirmar_no_se_devuelve():
    c = _zigzag(100.0, 120.0, 12, 3)
    n = 5
    sh, sl = swings_en_barra(c, c, n)
    ultimo = max(np.flatnonzero(np.nan_to_num(sl, nan=0.0).astype(bool)))
    # pidiendo hasta justo antes de su confirmacion, ese swing no puede estar
    g = npx.giros_confirmados(c, c, n, hasta=ultimo + n - 1)
    assert all(x.i != ultimo for x in g)
    g2 = npx.giros_confirmados(c, c, n, hasta=ultimo + n)
    assert any(x.i == ultimo for x in g2)


def test_invariancia_las_barras_nuevas_no_cambian_los_giros_viejos():
    c = _zigzag(100.0, 120.0, 10, 6)
    n = 5
    completo = npx.giros_confirmados(c, c, n)
    for corte in (40, 80, len(c) - 1):
        parcial = npx.giros_confirmados(c[:corte + 1], c[:corte + 1], n)
        esperado = [g for g in completo if g.i + n <= corte]
        assert parcial == esperado, f"corte {corte}"


def test_serie_vacia_o_corta():
    assert npx.giros_confirmados([], [], 5) == []
    assert npx.detectar([], [], [], n=5) == []


# -- toques --------------------------------------------------------------------

def test_la_separacion_colapsa_una_lateralizacion():
    g = [npx.Giro(i, 100.0, npx.L) for i in (10, 12, 14, 16, 60)]
    assert len(npx.toques_en_banda(g, 99.0, 101.0, sep=15)) == 2
    assert len(npx.toques_en_banda(g, 99.0, 101.0, sep=1)) == 5


def test_toques_respeta_la_banda():
    g = [npx.Giro(0, 100.0, npx.L), npx.Giro(40, 110.0, npx.H)]
    assert len(npx.toques_en_banda(g, 99.0, 101.0)) == 1


# -- deteccion -----------------------------------------------------------------

def test_encuentra_un_nivel_sintetico_en_el_precio_correcto():
    c = _zigzag(100.0, 130.0, 14, 5)
    niveles = npx.detectar(c, c, _fechas(len(c)), n=5, sep=10, min_toques=3)
    assert niveles
    precios = [x.precio for x in niveles]
    assert any(abs(p - 100.0) / 100.0 < 0.01 for p in precios), precios
    assert any(abs(p - 130.0) / 130.0 < 0.01 for p in precios), precios


def test_la_polaridad_separa_soporte_de_resistencia():
    c = _zigzag(100.0, 130.0, 14, 5)
    niveles = npx.detectar(c, c, _fechas(len(c)), n=5, sep=10, min_toques=3)
    piso = min(niveles, key=lambda x: x.precio)
    techo = max(niveles, key=lambda x: x.precio)
    assert piso.lows > 0 and piso.highs == 0 and piso.polaridad == "soporte"
    assert techo.highs > 0 and techo.lows == 0 and techo.polaridad == "resistencia"


def test_ningun_par_de_niveles_queda_mas_cerca_que_dos_anchos():
    rng = np.random.default_rng(3)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 900)))
    niveles = npx.detectar(c * 1.005, c * 0.995, _fechas(900), tol=0.01, min_toques=3)
    minimo = npx.separacion_minima(0.01)
    for i, a in enumerate(niveles):
        for b in niveles[i + 1:]:
            # en log: abs(a/b - 1) no es simetrico y dejaba pasar pares cercanos
            assert abs(np.log(a.precio / b.precio)) >= minimo - 1e-12


def test_el_nivel_queda_centrado_en_sus_toques():
    rng = np.random.default_rng(11)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 900)))
    for nv in npx.detectar(c * 1.005, c * 0.995, _fechas(900), min_toques=3):
        assert nv.val < nv.precio < nv.vah


def test_orden_persistencia_primero():
    rng = np.random.default_rng(5)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, 1200)))
    niveles = npx.detectar(c * 1.004, c * 0.996, _fechas(1200), min_toques=3)
    claves = [(-x.anios, -x.toques, x.precio) for x in niveles]
    assert claves == sorted(claves)


def test_cuenta_anios_distintos():
    c = _zigzag(100.0, 130.0, 14, 5)
    f = _fechas(len(c))
    uno = npx.detectar(c, c, f, n=5, sep=10, min_toques=3)
    # las mismas barras repartidas en varios anios tienen que dar mas anios
    f2 = [pd.Timestamp("2021-01-04") + pd.Timedelta(days=30 * k) for k in range(len(c))]
    varios = npx.detectar(c, c, f2, n=5, sep=10, min_toques=3)
    assert max(x.anios for x in varios) > max(x.anios for x in uno)


def test_min_toques_filtra():
    c = _zigzag(100.0, 130.0, 14, 5)
    f = _fechas(len(c))
    assert npx.detectar(c, c, f, n=5, sep=10, min_toques=99) == []


# -- nulo ----------------------------------------------------------------------

def test_permutar_conserva_inicio_largo_y_distribucion():
    rng = np.random.default_rng(1)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    h, lo = c * 1.01, c * 0.99
    c2, h2, l2 = npx.permutar(c, h, lo, np.random.default_rng(7))
    assert len(c2) == len(c) and c2[0] == pytest.approx(c[0])
    r1 = np.sort(c[1:] / c[:-1])
    r2 = np.sort(c2[1:] / c2[:-1])
    assert np.allclose(r1, r2)
    assert np.all(h2 >= c2 * 0.999) and np.all(l2 <= c2 * 1.001)


def test_permutar_cambia_el_orden():
    rng = np.random.default_rng(2)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    c2, _, _ = npx.permutar(c, c * 1.01, c * 0.99, np.random.default_rng(9))
    assert not np.allclose(c, c2)
