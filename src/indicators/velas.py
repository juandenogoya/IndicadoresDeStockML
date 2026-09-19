"""
velas.py
Patrones de vela con definicion clasica y contexto de tendencia previa.
Modulo PURO (numpy + pandas): sin DB, sin config, sin side effects. Lo pueden
importar el pipeline, el dashboard y el MCP.

POR QUE EXISTE (docs/estructura_velas.md sec. 5)
    precio_accion.py marca "envolvente" comparando solo tamano de cuerpo y color
    (el 73% de las marcadas no envuelve), y martillo / estrella fugaz exigen color,
    no limitan la sombra opuesta y no miran la tendencia previa (el 46,5% de los
    "martillos" esta arriba del rango: son hanging man). Medido: ningun patron,
    con la definicion vieja ni con la clasica, anticipa retorno. Este modulo existe
    para DESCRIBIR bien, no porque las velas sean una senal.

LA REGLA (test de invariancia en tests/test_velas.py)
    Cada fila usa solo barras hasta su fecha:
    calcular_velas(df[:t+1]).iloc[-1] == calcular_velas(df).iloc[t]

DEFINICIONES (fracciones del rango high - low de la vela)
    Forma de martillo:  cuerpo <= 30%, sombra inferior >= 2 x cuerpo, sombra
                        superior <= 10%. Cualquier color.
    Forma de estrella:  espejo (sombra superior larga, inferior <= 10%).
    Contexto:           retorno de las RUEDAS_CONTEXTO ruedas previas a la vela,
                        close[t-1] / close[t-1-RUEDAS_CONTEXTO] - 1: < 0 caida,
                        > 0 suba. Sin historia suficiente o = 0: sin contexto.

    patron_hammer           forma de martillo tras caida
    patron_hanging_man      forma de martillo tras suba
    patron_shooting_star    forma de estrella tras suba
    patron_inverted_hammer  forma de estrella tras caida
    patron_doji             cuerpo <= 5% y ninguno de los cuatro anteriores (un doji
                            libelula tras una caida es un martillo: la sombra larga
                            es la informacion). Una sola etiqueta por forma.
    patron_engulfing_bull   vela previa bajista (close < open), actual alcista, y el
                            cuerpo actual envuelve al previo: open <= close previo y
                            close >= open previo, con al menos una desigualdad
                            estricta. Sin requisito de contexto.
    patron_engulfing_bear   espejo.
    patron_marubozu_bull    cuerpo >= 85%, vela alcista.
    patron_marubozu_bear    cuerpo >= 85%, vela bajista.
    inside_bar              high < high previo y low > low previo.
    outside_bar             high > high previo y low < low previo.

    Vela sin rango (high == low) o con datos faltantes: ningun patron de forma.
    close == open no es alcista ni bajista.

Los umbrales son la propuesta inicial de la Fase 1; si cambian, se cambian aca y en
docs/estructura_velas.md, no en los consumidores.
"""

from typing import List

import numpy as np
import pandas as pd

DOJI_CUERPO_MAX = 0.05
MARTILLO_CUERPO_MAX = 0.30
SOMBRA_LARGA_X_CUERPO = 2.0
SOMBRA_OPUESTA_MAX = 0.10
MARUBOZU_CUERPO_MIN = 0.85
RUEDAS_CONTEXTO = 5

# Tolerancia para comparar fracciones en los bordes (0,30 * rango no siempre da
# exactamente 0,30 en punto flotante).
_EPS = 1e-9

COLUMNAS: List[str] = [
    "patron_doji",
    "patron_hammer", "patron_hanging_man",
    "patron_shooting_star", "patron_inverted_hammer",
    "patron_engulfing_bull", "patron_engulfing_bear",
    "patron_marubozu_bull", "patron_marubozu_bear",
    "inside_bar", "outside_bar",
]


def _anterior(x: np.ndarray, k: int = 1) -> np.ndarray:
    """out[t] = x[t-k]; NaN al inicio."""
    out = np.full(len(x), np.nan)
    if 0 < k < len(x):
        out[k:] = x[:-k]
    return out


def calcular_velas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Patrones de vela para UNA serie (un ticker).

    Args:
        df: columnas fecha, open, high, low, close (ticker opcional). Cualquier
            orden: se ordena por fecha. No se modifica.

    Returns:
        DataFrame ordenado por fecha: [ticker], fecha + COLUMNAS (enteros 0/1).
    """
    d = df.sort_values("fecha").reset_index(drop=True)
    o = d["open"].to_numpy(dtype=float)
    h = d["high"].to_numpy(dtype=float)
    lo = d["low"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        rango = h - lo
        rango_ok = rango > 0
        r = np.where(rango_ok, rango, np.nan)
        cuerpo = np.abs(c - o) / r
        sombra_sup = (h - np.maximum(o, c)) / r
        sombra_inf = (np.minimum(o, c) - lo) / r

        alcista = c > o
        bajista = c < o

        forma_martillo = ((cuerpo <= MARTILLO_CUERPO_MAX + _EPS)
                          & (sombra_inf + _EPS >= SOMBRA_LARGA_X_CUERPO * cuerpo)
                          & (sombra_sup <= SOMBRA_OPUESTA_MAX + _EPS))
        forma_estrella = ((cuerpo <= MARTILLO_CUERPO_MAX + _EPS)
                          & (sombra_sup + _EPS >= SOMBRA_LARGA_X_CUERPO * cuerpo)
                          & (sombra_inf <= SOMBRA_OPUESTA_MAX + _EPS))

        c1 = _anterior(c, 1)
        ret_previo = c1 / _anterior(c, 1 + RUEDAS_CONTEXTO) - 1.0
        caida = ret_previo < 0
        suba = ret_previo > 0

        hammer = forma_martillo & caida
        hanging = forma_martillo & suba
        star = forma_estrella & suba
        inverted = forma_estrella & caida
        doji = (cuerpo <= DOJI_CUERPO_MAX + _EPS) & ~(hammer | hanging | star | inverted)

        o1 = _anterior(o, 1)
        h1 = _anterior(h, 1)
        l1 = _anterior(lo, 1)
        prev_bajista = c1 < o1
        prev_alcista = c1 > o1
        eng_bull = (alcista & prev_bajista & (o <= c1) & (c >= o1)
                    & ((o < c1) | (c > o1)))
        eng_bear = (bajista & prev_alcista & (o >= c1) & (c <= o1)
                    & ((o > c1) | (c < o1)))

        marubozu = cuerpo >= MARUBOZU_CUERPO_MIN - _EPS
        inside = (h < h1) & (lo > l1)
        outside = (h > h1) & (lo < l1)

    valores = {
        "patron_doji": doji,
        "patron_hammer": hammer,
        "patron_hanging_man": hanging,
        "patron_shooting_star": star,
        "patron_inverted_hammer": inverted,
        "patron_engulfing_bull": eng_bull,
        "patron_engulfing_bear": eng_bear,
        "patron_marubozu_bull": marubozu & alcista,
        "patron_marubozu_bear": marubozu & bajista,
        "inside_bar": inside,
        "outside_bar": outside,
    }
    out = pd.DataFrame({"fecha": d["fecha"]})
    if "ticker" in d.columns:
        out.insert(0, "ticker", d["ticker"])
    for col in COLUMNAS:
        out[col] = np.asarray(valores[col], dtype=bool).astype(int)
    return out
