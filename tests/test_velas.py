"""
Tests de src/indicators/velas.py (patrones con definicion clasica y contexto).

Lo que importa fijar (docs/estructura_velas.md sec. 5):
  - la envolvente ENVUELVE el cuerpo previo (el modulo viejo solo comparaba tamano);
  - martillo / hanging man y estrella / inverted hammer son la misma forma leida con
    la tendencia previa, de cualquier color, con la sombra opuesta chica;
  - una sola etiqueta por forma (un doji libelula tras caida es martillo);
  - invariancia: cada fila usa solo barras hasta su fecha.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import velas

CAIDA = [20, 19, 18, 17, 16, 15]
SUBA = [10, 11, 12, 13, 14, 15]


def _df(filas):
    """filas: lista de (open, high, low, close)."""
    a = np.asarray(filas, dtype=float)
    return pd.DataFrame({
        "fecha": pd.bdate_range("2024-01-01", periods=len(a)),
        "open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3],
    })


def _con_contexto(previas, vela):
    filas = [(c, c + 0.1, c - 0.1, c) for c in previas]
    return _df(filas + [vela])


def _ultima(df):
    return velas.calcular_velas(df).iloc[-1]


def _activos(fila):
    """Patrones de forma activos (inside/outside dependen de la vela previa)."""
    return {c for c in velas.COLUMNAS if c.startswith("patron_") and fila[c] == 1}


# -- martillo / hanging man ------------------------------------------------------------

MARTILLO_VERDE = (14.8, 15.05, 14.0, 15.0)
MARTILLO_ROJO = (15.0, 15.05, 14.0, 14.8)


def test_martillo_tras_caida():
    assert _activos(_ultima(_con_contexto(CAIDA, MARTILLO_VERDE))) == {"patron_hammer"}


def test_martillo_rojo_tambien_es_martillo():
    assert _activos(_ultima(_con_contexto(CAIDA, MARTILLO_ROJO))) == {"patron_hammer"}


def test_misma_forma_tras_suba_es_hanging_man():
    assert _activos(_ultima(_con_contexto(SUBA, MARTILLO_VERDE))) == {"patron_hanging_man"}


def test_sombra_superior_larga_no_es_martillo():
    fila = _ultima(_con_contexto(CAIDA, (14.8, 15.4, 14.0, 15.0)))
    assert fila["patron_hammer"] == 0


# -- estrella fugaz / inverted hammer ---------------------------------------------------

ESTRELLA = (15.2, 16.0, 14.95, 15.0)


def test_estrella_fugaz_tras_suba():
    assert _activos(_ultima(_con_contexto(SUBA, ESTRELLA))) == {"patron_shooting_star"}


def test_misma_forma_tras_caida_es_inverted_hammer():
    assert _activos(_ultima(_con_contexto(CAIDA, ESTRELLA))) == {"patron_inverted_hammer"}


# -- doji ----------------------------------------------------------------------------------

def test_doji_con_sombras_parejas():
    assert _activos(_ultima(_con_contexto(CAIDA, (15.0, 15.5, 14.5, 15.02)))) == {"patron_doji"}


def test_doji_libelula_tras_caida_es_martillo_no_doji():
    assert _activos(_ultima(_con_contexto(CAIDA, (15.0, 15.02, 14.0, 15.0)))) == {"patron_hammer"}


def test_doji_libelula_sin_contexto_es_doji():
    df = _df([(15.0, 15.1, 14.9, 15.0), (15.0, 15.02, 14.0, 15.0)])
    assert _activos(_ultima(df)) == {"patron_doji"}


# -- envolventes -----------------------------------------------------------------------------

PREVIA_ROJA = (15.5, 15.6, 14.9, 15.0)
PREVIA_VERDE = (15.0, 15.6, 14.9, 15.5)


def test_envolvente_alcista_envuelve_el_cuerpo_previo():
    fila = _ultima(_df([PREVIA_ROJA, (14.9, 15.8, 14.8, 15.7)]))
    assert fila["patron_engulfing_bull"] == 1


def test_vela_mas_grande_que_abre_arriba_del_cierre_previo_no_es_envolvente():
    # El modulo viejo la marcaba: cuerpo mas grande, color opuesto.
    fila = _ultima(_df([PREVIA_ROJA, (15.2, 16.6, 15.1, 16.5)]))
    assert fila["patron_engulfing_bull"] == 0


def test_previa_plana_no_cuenta_como_roja():
    fila = _ultima(_df([(15.0, 15.2, 14.8, 15.0), (14.9, 15.8, 14.8, 15.7)]))
    assert fila["patron_engulfing_bull"] == 0


def test_cuerpo_identico_invertido_no_es_envolvente():
    fila = _ultima(_df([PREVIA_ROJA, (15.0, 15.6, 14.9, 15.5)]))
    assert fila["patron_engulfing_bull"] == 0


def test_envolvente_bajista():
    fila = _ultima(_df([PREVIA_VERDE, (15.6, 15.7, 14.8, 14.9)]))
    assert fila["patron_engulfing_bear"] == 1
    assert fila["patron_engulfing_bull"] == 0


# -- marubozu, inside/outside, bordes --------------------------------------------------------

def test_marubozu_con_direccion():
    alcista = _ultima(_df([(15.0, 16.05, 14.95, 16.0)]))
    bajista = _ultima(_df([(16.0, 16.05, 14.95, 15.0)]))
    assert alcista["patron_marubozu_bull"] == 1 and alcista["patron_marubozu_bear"] == 0
    assert bajista["patron_marubozu_bear"] == 1 and bajista["patron_marubozu_bull"] == 0


def test_inside_y_outside_bar():
    base = (15.0, 16.0, 14.0, 15.5)
    assert _ultima(_df([base, (15.2, 15.8, 14.5, 15.4)]))["inside_bar"] == 1
    assert _ultima(_df([base, (15.2, 16.2, 13.8, 15.4)]))["outside_bar"] == 1


def test_vela_sin_rango_no_marca_forma():
    fila = _ultima(_con_contexto(CAIDA, (15.0, 15.0, 15.0, 15.0)))
    assert _activos(fila) == set()


def test_no_modifica_la_entrada_y_conserva_ticker():
    df = _con_contexto(CAIDA, MARTILLO_VERDE)
    df["ticker"] = "AAA"
    copia = df.copy()
    out = velas.calcular_velas(df.iloc[::-1])
    pd.testing.assert_frame_equal(df, copia)
    assert list(out.columns[:2]) == ["ticker", "fecha"]
    assert out["fecha"].is_monotonic_increasing


# -- invariancia -------------------------------------------------------------------------------

def test_invariancia():
    rng = np.random.default_rng(5)
    n = 150
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    df = pd.DataFrame({"fecha": pd.bdate_range("2023-01-02", periods=n),
                       "open": open_, "high": high, "low": low, "close": close})
    completo = velas.calcular_velas(df)
    assert completo[velas.COLUMNAS].to_numpy().sum() > 0
    for t in range(n):
        parcial = velas.calcular_velas(df.iloc[: t + 1]).iloc[-1]
        assert (parcial[velas.COLUMNAS] == completo.loc[t, velas.COLUMNAS]).all(), t
