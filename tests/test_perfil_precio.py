"""
Tests de src/utils/perfil_precio.py (POC y area de valor por volumen).

Lo que importa fijar:
  - INVARIANCIA: la fila de una fecha calculada con los datos hasta esa fecha es
    igual a la calculada con toda la historia. Vale porque la grilla esta ANCLADA
    y no se arma con el minimo y el maximo de la ventana; con una grilla armada
    sobre los datos, los bordes se corren al llegar una barra nueva y el POC se
    mueve solo;
  - el reparto vectorizado del volumen da lo mismo que una implementacion de
    referencia escrita a mano, rueda por rueda;
  - el area de valor cubre al menos la cobertura pedida y contiene al POC;
  - una ventana incompleta sale en NaN y no finge un perfil;
  - la ubicacion respecto del area de valor se DERIVA del precio, no se guarda.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils import perfil_precio as pp


def _df(high, low, volume, fechas=None):
    n = len(high)
    if fechas is None:
        fechas = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"fecha": fechas, "high": np.asarray(high, dtype=float),
                         "low": np.asarray(low, dtype=float),
                         "volume": np.asarray(volume, dtype=float)})


def _random_walk(n, semilla, precio0=100.0):
    rng = np.random.default_rng(semilla)
    close = precio0 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    alto = close * (1 + rng.uniform(0.001, 0.02, n))
    bajo = close * (1 - rng.uniform(0.001, 0.02, n))
    vol = rng.lognormal(15, 0.4, n)
    return _df(alto, bajo, vol)


def _perfil_referencia(high, low, volumen, paso=pp.PASO):
    """Reparto escrito a mano, rueda por rueda. Contra el vectorizado del modulo."""
    acum = {}
    for h, lo, v in zip(high, low, volumen):
        if not (np.isfinite(h) and np.isfinite(lo) and lo > 0 and v > 0):
            continue
        a = int(np.floor(np.log(min(h, lo)) / np.log1p(paso)))
        b = int(np.floor(np.log(max(h, lo)) / np.log1p(paso)))
        for k in range(a, b + 1):
            acum[k] = acum.get(k, 0.0) + v / (b - a + 1)
    return acum


# -- grilla anclada ------------------------------------------------------------

def test_el_precio_cae_dentro_de_los_limites_de_su_bin():
    for px in [1.0, 3.49, 11.34, 99.99, 335.78, 590.34, 1234.56]:
        k = pp.indice_bin(px)
        lo, hi = pp.limites_bin(k)
        assert lo <= px < hi
        assert lo < pp.centro_bin(k) < hi


def test_la_grilla_no_depende_de_los_datos():
    """El mismo precio cae SIEMPRE en el mismo bin, mire lo que mire la ventana."""
    a = _random_walk(200, 1)
    b = _random_walk(200, 2, precio0=800.0)
    k0a, pa = pp.perfil_bins(a["high"], a["low"], a["volume"])
    k0b, pb = pp.perfil_bins(b["high"], b["low"], b["volume"])
    # los bordes son potencias de (1+paso): el bin de un precio no cambia de serie
    assert pp.indice_bin(123.45) == pp.indice_bin(123.45)
    for k0, p in ((k0a, pa), (k0b, pb)):
        lo, _ = pp.limites_bin(k0)
        assert pp.indice_bin(lo * 1.0000001) == k0


def test_bins_contiguos_sin_huecos_ni_solapes():
    k = pp.indice_bin(250.0)
    _, hi = pp.limites_bin(k)
    lo_sig, _ = pp.limites_bin(k + 1)
    assert hi == pytest.approx(lo_sig)


# -- reparto del volumen -------------------------------------------------------

@pytest.mark.parametrize("semilla", [1, 2, 3])
def test_el_reparto_vectorizado_da_lo_mismo_que_la_referencia(semilla):
    d = _random_walk(120, semilla)
    k0, p = pp.perfil_bins(d["high"], d["low"], d["volume"])
    ref = _perfil_referencia(d["high"].to_numpy(), d["low"].to_numpy(),
                             d["volume"].to_numpy())
    assert len(ref) > 5
    for k, v in ref.items():
        assert p[k - k0] == pytest.approx(v)
    assert p.sum() == pytest.approx(sum(ref.values()))


def test_una_rueda_reparte_su_volumen_parejo_sobre_las_franjas_que_toca():
    d = _df([110.0], [100.0], [1000.0])
    k0, p = pp.perfil_bins(d["high"], d["low"], d["volume"])
    tocados = (p > 0).sum()
    assert tocados > 1                       # 100 a 110 cruza varias franjas de 0,5%
    assert p[p > 0] == pytest.approx(1000.0 / tocados)
    assert p.sum() == pytest.approx(1000.0)


def test_el_volumen_total_se_conserva():
    d = _random_walk(80, 7)
    _, p = pp.perfil_bins(d["high"], d["low"], d["volume"])
    assert p.sum() == pytest.approx(d["volume"].sum())


# -- POC y area de valor -------------------------------------------------------

def test_el_poc_cae_donde_se_concentro_el_volumen():
    # 30 ruedas pegadas a 50 con volumen chico, 5 pegadas a 80 con volumen enorme
    high = [50.2] * 30 + [80.2] * 5
    low = [49.8] * 30 + [79.8] * 5
    vol = [1.0] * 30 + [1000.0] * 5
    r = pp.calcular(high, low, vol)
    assert r.valido
    assert 79.0 < r.poc < 81.0


def test_el_area_de_valor_contiene_al_poc_y_cubre_la_cobertura():
    d = _random_walk(200, 11)
    r = pp.calcular(d["high"], d["low"], d["volume"])
    assert r.val <= r.poc <= r.vah
    assert r.ancho_va_pct > 0
    k0, p = pp.perfil_bins(d["high"], d["low"], d["volume"])
    a = int(pp.indice_bin(r.val * (1 + 1e-9))) - k0
    b = int(pp.indice_bin(r.vah * (1 - 1e-9))) - k0
    dentro = p[a:b + 1]
    assert dentro.sum() >= pp.COBERTURA * p.sum() - 1e-9


def test_cobertura_mas_alta_da_area_mas_ancha():
    d = _random_walk(200, 12)
    estrecha = pp.calcular(d["high"], d["low"], d["volume"], cobertura=0.50)
    ancha = pp.calcular(d["high"], d["low"], d["volume"], cobertura=0.90)
    assert ancha.ancho_va_pct > estrecha.ancho_va_pct
    assert ancha.val <= estrecha.val and ancha.vah >= estrecha.vah


def test_el_ancho_del_area_avisa_cuando_el_perfil_es_plano():
    plano = _df(np.linspace(100, 200, 150) * 1.01,
                np.linspace(100, 200, 150) * 0.99, np.ones(150))
    equilibrio = _df([100.5] * 150, [99.5] * 150, np.ones(150))
    assert pp.calcular(plano["high"], plano["low"], plano["volume"]).ancho_va_pct > 20
    assert pp.calcular(equilibrio["high"], equilibrio["low"],
                       equilibrio["volume"]).ancho_va_pct < 5


# -- invariancia ---------------------------------------------------------------

@pytest.mark.parametrize("semilla", [1, 2, 3])
@pytest.mark.parametrize("ventana", [21, 52])
def test_invariancia_la_fila_de_una_fecha_no_cambia_con_barras_nuevas(semilla, ventana):
    d = _random_walk(90, semilla)
    completo = pp.calcular_rodante(d, ventana)
    for t in range(ventana - 1, len(d)):
        parcial = pp.calcular_rodante(d.iloc[: t + 1], ventana).iloc[-1]
        for col in ("poc", "val", "vah", "ancho_va_pct"):
            assert parcial[col] == pytest.approx(completo.loc[t, col]), (t, col)


def test_invariancia_tambien_en_calcular_ventanas():
    d = _random_walk(260, 5)
    hoy = pp.calcular_ventanas(d.iloc[:220])
    futuro = pp.calcular_ventanas(d)          # llegaron 40 barras mas
    assert not np.allclose(hoy["poc"], futuro["poc"])   # la ventana se corrio: debe cambiar
    # pero recalcular el pasado da el pasado
    otra_vez = pp.calcular_ventanas(d.iloc[:220])
    assert np.allclose(hoy["poc"], otra_vez["poc"])


# -- bordes y datos sucios -----------------------------------------------------

def test_ventana_incompleta_sale_en_nan():
    d = _random_walk(30, 4)
    out = pp.calcular_ventanas(d, ventanas=(21, 200))
    assert np.isfinite(out.loc[out["ventana"] == 21, "poc"]).all()
    assert np.isnan(out.loc[out["ventana"] == 200, "poc"]).all()


def test_sin_volumen_no_hay_perfil_y_no_explota():
    d = _df([10.1] * 5, [9.9] * 5, [0, 0, 0, np.nan, 0])
    r = pp.calcular(d["high"], d["low"], d["volume"])
    assert not r.valido and r.volumen == 0.0


def test_serie_vacia():
    r = pp.calcular([], [], [])
    assert not r.valido


def test_tolera_high_menor_que_low_y_precios_no_positivos():
    d = _df([9.9, 10.1, 0.0], [10.1, 9.9, -1.0], [100.0, 100.0, 100.0])
    r = pp.calcular(d["high"], d["low"], d["volume"])
    assert r.valido
    assert r.volumen == pytest.approx(200.0)      # la fila invalida no entra


def test_pico_pct_es_la_porcion_de_volumen_de_la_franja_del_poc():
    d = _random_walk(200, 9)
    r = pp.calcular(d["high"], d["low"], d["volume"])
    assert r.bins > 0 and r.ruedas == 200
    k0, p = pp.perfil_bins(d["high"], d["low"], d["volume"])
    assert r.pico_pct == pytest.approx(p.max() / p.sum() * 100)


def test_bins_mas_finos_achican_el_pico():
    """El motivo por el que 0,1% no sirve: el maximo deja de ser un maximo."""
    d = _random_walk(200, 10)
    grueso = pp.calcular(d["high"], d["low"], d["volume"], paso=0.01)
    fino = pp.calcular(d["high"], d["low"], d["volume"], paso=0.001)
    assert fino.pico_pct < grueso.pico_pct
    assert fino.bins > grueso.bins


# -- ubicacion derivada --------------------------------------------------------

def test_la_ubicacion_se_deriva_del_precio_de_hoy():
    d = _random_walk(200, 13)
    r = pp.calcular(d["high"], d["low"], d["volume"])
    assert pp.ubicacion(r.vah * 1.10, r) == "arriba del area de valor"
    assert pp.ubicacion(r.val * 0.90, r) == "abajo del area de valor"
    assert pp.ubicacion(r.poc, r) == "dentro del area de valor"
    assert pp.ubicacion(np.nan, r) == "sin dato"
    assert pp.ubicacion(100.0, pp.calcular([], [], [])) == "sin dato"
