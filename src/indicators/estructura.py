"""
estructura.py
Estructura de mercado (SMC) con swings CONFIRMADOS: cada fila usa solo las barras
hasta su fecha. Modulo PURO (numpy + pandas): sin DB, sin config, sin side effects.
Lo pueden importar el pipeline, el dashboard y el MCP.

POR QUE EXISTE (docs/estructura_velas.md sec. 4)
    market_structure.py detecta swings con rolling(2N+1, center=True) y los anota
    en la barra del swing, cuando recien se conocen N barras despues. Su historia
    guardada mira N ruedas al futuro: un swing high "guardado" da -4,7% de exceso a
    5 ruedas y los modelos ML entrenados ahi dan AUC 0,65 contra 0,52 con lo que se
    sabia cada dia. Ademas marca swings provisionales en las ultimas N barras, que
    desaparecen al dia siguiente.

LA REGLA DE ESTE MODULO (test de invariancia en tests/test_estructura.py)
    calcular_estructura(df[:t+1]).iloc[-1] == calcular_estructura(df).iloc[t]
    La fila de una fecha no cambia cuando llegan barras nuevas.

DEFINICIONES (ventana N)
    Swing high en la barra p: high[p] > max(high[p-N .. p-1]) y
    high[p] >= max(high[p+1 .. p+N]). Las dos ventanas completas. Con maximos
    iguales dentro de la ventana gana el primero (un solo swing). Swing low: espejo
    sobre low.
    Se CONFIRMA en c = p + N, cuando existen las N barras de la derecha. Desde c
    en adelante el swing existe; antes, no.

    is_sh_N / is_sl_N   1 si en esta barra se confirmo un swing (el de la barra t-N).
                        NO significa "esta barra es un swing": eso no se sabe hoy.
    dist_sh_N_pct       (close - ultimo swing high confirmado) / ese nivel * 100.
    dist_sl_N_pct       idem con el ultimo swing low confirmado.
    dias_sh_N/dias_sl_N barras desde la barra del ultimo swing confirmado (>= N),
                        con tope `tope_dias`.
    impulso_N_pct       |ultimo SH - ultimo SL| / ultimo SL * 100.
    estructura_N        +1 si los dos ultimos swings confirmados de cada tipo son
                        HH y HL; -1 si LH y LL; 0 en otro caso o sin 2 de cada tipo.
    bos_bull_N          el close cruza hacia arriba el ultimo swing high conocido en
                        la barra anterior (close[t] > nivel y close[t-1] <= nivel)
                        y la estructura de la barra anterior es >= 0.
    choch_bull_N        el mismo cruce con estructura anterior < 0.
    bos_bear_N          cruce hacia abajo del ultimo swing low, estructura anterior <= 0.
    choch_bear_N        el mismo cruce con estructura anterior > 0.

    La clasificacion BOS/CHoCH y la de estructura son las de market_structure.py
    (incluido que con estructura 0 una ruptura cuenta como BOS en las dos
    direcciones). Lo unico que cambia es QUE swings existen en cada fecha: asi la
    diferencia entre las dos versiones se atribuye solo al leakage.

    Mismos nombres de columna que market_structure.FEATURE_COLS_MS.

    Sirve igual para barras semanales: `tope_dias` = TOPE_DIAS_SEMANAL y las
    "barras" son semanas.
"""

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

VENTANAS: Tuple[int, ...] = (5, 10)

# Ventanas que se PERSISTEN en features_estructura. N=3 se agrego el 17/9/2026
# para las estrategias FT_SMC_v3_N3/N5: el backtest de la seccion 9.3 del doc
# mostro que la lectura de estructura solo gana si el swing se confirma rapido
# (N=10 +13,5% vs N=5 +59,8% y N=3 +58,2%), y cual de las dos es mejor no se
# decide con el backtest. VENTANAS queda en (5, 10) porque es el set que tiene
# paridad de nombres con market_structure.FEATURE_COLS_MS.
VENTANAS_TABLA: Tuple[int, ...] = (3, 5, 10)

TOPE_DIAS_DIARIO = 252
TOPE_DIAS_SEMANAL = 260

_BASES = ("is_sh", "is_sl", "estructura", "dist_sh", "dist_sl",
          "dias_sh", "dias_sl", "impulso", "bos_bull", "bos_bear",
          "choch_bull", "choch_bear")


def columnas(ventanas: Iterable[int] = VENTANAS) -> List[str]:
    """Nombres de columna en el orden de market_structure.FEATURE_COLS_MS."""
    cols: List[str] = []
    for n in ventanas:
        cols.extend([
            f"is_sh_{n}", f"is_sl_{n}", f"estructura_{n}",
            f"dist_sh_{n}_pct", f"dist_sl_{n}_pct",
            f"dias_sh_{n}", f"dias_sl_{n}", f"impulso_{n}_pct",
            f"bos_bull_{n}", f"bos_bear_{n}",
            f"choch_bull_{n}", f"choch_bear_{n}",
        ])
    return cols


COLUMNAS: List[str] = columnas()


def columnas_enteras(ventanas: Iterable[int] = VENTANAS) -> List[str]:
    """Flags y estructura. dias_* NO entran: van como float (NaN sin swing)."""
    return [c for c in columnas(ventanas)
            if c.startswith(("is_", "estructura_", "bos_", "choch_"))]


COLUMNAS_ENTERAS: List[str] = columnas_enteras()


# ── Swings ────────────────────────────────────────────────────────────────────

def _extremo_previo(x: np.ndarray, n: int, cual: str) -> np.ndarray:
    """max/min de x[i-n .. i-1]; NaN si la ventana no esta completa."""
    r = pd.Series(x, dtype="float64").shift(1).rolling(n, min_periods=n)
    return (r.max() if cual == "max" else r.min()).to_numpy()


def _extremo_siguiente(x: np.ndarray, n: int, cual: str) -> np.ndarray:
    """max/min de x[i+1 .. i+n]; NaN si la ventana no esta completa."""
    return _extremo_previo(x[::-1], n, cual)[::-1]


def swings_en_barra(high, low, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    (swing_high, swing_low) marcados EN LA BARRA DEL SWING.

    Usan las N barras siguientes: no son informacion de esa fecha. Solo sirven para
    confirmar en p + N (lo hace calcular_estructura). No usar directo como feature.
    """
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    with np.errstate(invalid="ignore"):
        sh = (h > _extremo_previo(h, n, "max")) & (h >= _extremo_siguiente(h, n, "max"))
        sl = (lo < _extremo_previo(lo, n, "min")) & (lo <= _extremo_siguiente(lo, n, "min"))
    return sh, sl


def _desplazar(x: np.ndarray, k: int, relleno) -> np.ndarray:
    """x corrido k > 0 posiciones hacia adelante: out[t] = x[t-k]; relleno al inicio."""
    x = np.asarray(x)
    es_nan = isinstance(relleno, float) and np.isnan(relleno)
    out = np.full(len(x), relleno, dtype=float if es_nan else x.dtype)
    if 0 < k < len(x):
        out[k:] = x[:-k]
    return out


def _ultimos(evento: np.ndarray, precio: np.ndarray, barra: np.ndarray):
    """Para cada t: precio y barra del ultimo evento hasta t, y precio del penultimo."""
    n = len(evento)
    ult = np.full(n, np.nan)
    pen = np.full(n, np.nan)
    bar = np.full(n, np.nan)
    idx = np.flatnonzero(evento)
    if idx.size == 0:
        return ult, pen, bar
    cum = np.cumsum(evento.astype(int))
    m1 = cum >= 1
    ult[m1] = precio[idx][cum[m1] - 1]
    bar[m1] = barra[idx][cum[m1] - 1]
    m2 = cum >= 2
    pen[m2] = precio[idx][cum[m2] - 2]
    return ult, pen, bar


def _pct(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den_ok = np.where((den == 0) | np.isnan(den), np.nan, den)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den_ok * 100.0


def _estructura_n(h: np.ndarray, lo: np.ndarray, c: np.ndarray,
                  n: int, tope_dias: int) -> dict:
    total = len(h)
    pos = np.arange(total, dtype=float)

    sh_barra, sl_barra = swings_en_barra(h, lo, n)
    conf_sh = _desplazar(sh_barra, n, False).astype(bool)
    conf_sl = _desplazar(sl_barra, n, False).astype(bool)
    precio_sh = _desplazar(h, n, np.nan)
    precio_sl = _desplazar(lo, n, np.nan)
    barra_swing = pos - n

    last_sh, prev_sh, bar_sh = _ultimos(conf_sh, precio_sh, barra_swing)
    last_sl, prev_sl, bar_sl = _ultimos(conf_sl, precio_sl, barra_swing)

    with np.errstate(invalid="ignore"):
        hh = last_sh > prev_sh
        lh = last_sh < prev_sh
        hl = last_sl > prev_sl
        ll = last_sl < prev_sl
    dos_de_cada = (np.cumsum(conf_sh) >= 2) & (np.cumsum(conf_sl) >= 2)
    estructura = np.where(hh & hl, 1, np.where(lh & ll, -1, 0))
    estructura = np.where(dos_de_cada, estructura, 0).astype(int)

    dias_sh = np.where(np.isnan(bar_sh), np.nan, np.clip(pos - bar_sh, 0, tope_dias))
    dias_sl = np.where(np.isnan(bar_sl), np.nan, np.clip(pos - bar_sl, 0, tope_dias))

    with np.errstate(invalid="ignore"):
        impulso = np.where(~np.isnan(last_sh) & ~np.isnan(last_sl),
                           _pct(np.abs(last_sh - last_sl), last_sl), np.nan)

    # BOS / CHoCH: nivel y estructura conocidos en la barra anterior.
    nivel_sh = _desplazar(last_sh, 1, np.nan)
    nivel_sl = _desplazar(last_sl, 1, np.nan)
    c_prev = _desplazar(c, 1, np.nan)
    est_prev = _desplazar(estructura, 1, 0)
    with np.errstate(invalid="ignore"):
        arriba = (c > nivel_sh) & (c_prev <= nivel_sh)
        abajo = (c < nivel_sl) & (c_prev >= nivel_sl)

    return {
        f"is_sh_{n}": conf_sh.astype(int),
        f"is_sl_{n}": conf_sl.astype(int),
        f"estructura_{n}": estructura,
        f"dist_sh_{n}_pct": _pct(c - last_sh, last_sh),
        f"dist_sl_{n}_pct": _pct(c - last_sl, last_sl),
        f"dias_sh_{n}": dias_sh,
        f"dias_sl_{n}": dias_sl,
        f"impulso_{n}_pct": impulso,
        f"bos_bull_{n}": (arriba & (est_prev >= 0)).astype(int),
        f"bos_bear_{n}": (abajo & (est_prev <= 0)).astype(int),
        f"choch_bull_{n}": (arriba & (est_prev < 0)).astype(int),
        f"choch_bear_{n}": (abajo & (est_prev > 0)).astype(int),
    }


def calcular_estructura(df: pd.DataFrame, ventanas: Iterable[int] = VENTANAS,
                        tope_dias: int = TOPE_DIAS_DIARIO) -> pd.DataFrame:
    """
    Estructura de mercado con swings confirmados para UNA serie (un ticker).

    Args:
        df:        columnas fecha, high, low, close (open/volume/ticker opcionales).
                   Cualquier orden: se ordena por fecha. No se modifica.
        ventanas:  valores de N (default 5 y 10).
        tope_dias: tope de dias_sh/dias_sl (252 diario, 260 semanal).

    Returns:
        DataFrame ordenado por fecha: [ticker], fecha + columnas(ventanas).
    """
    d = df.sort_values("fecha").reset_index(drop=True)
    h = d["high"].to_numpy(dtype=float)
    lo = d["low"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)

    out = pd.DataFrame({"fecha": d["fecha"]})
    if "ticker" in d.columns:
        out.insert(0, "ticker", d["ticker"])
    for n in ventanas:
        for col, valores in _estructura_n(h, lo, c, int(n), tope_dias).items():
            out[col] = valores
    return out
