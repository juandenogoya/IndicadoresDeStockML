"""
poc_eventos.py
Eventos de entrada a una banda de precio y su exceso contra el universo. Modulo
PURO (numpy + pandas): sin DB, sin config, sin side effects.

PARA QUE
    Contestar la pregunta de docs/perfil_precio.md sec. 4: cuando el precio entra
    en el area de valor, hace algo distinguible de lo que hace al entrar en una
    banda CUALQUIERA del mismo ticker? Este modulo no sabe nada del POC: recibe
    bandas y devuelve eventos y excesos, asi que la banda real y las de control
    pasan exactamente por el mismo codigo. Esa simetria es el punto.

LA BANDA DEL EVENTO ES LA DE t-1, NO LA DE t
    `perfil_precio.calcular_rodante` incluye la rueda t en su ventana. Usar la
    banda de t para un evento en t meteria el dia del evento adentro del nivel que
    se supone que lo predijo. Aca el evento en t se juzga SIEMPRE con la banda de
    t-1, que solo vio ruedas <= t-1. Es la misma familia de error que
    features_market_structure (CLAUDE.md): una lectura "la ultima que haya" nunca
    falla y devuelve un numero plausible.

Y ES UNA SOLA BANDA PARA TODO EL EVENTO
    La condicion "estuvo afuera las N ruedas previas" se evalua contra esa MISMA
    banda de t-1, no contra las bandas moviles de cada rueda. Asi el evento queda
    definido como "el precio se acerco a ESTA banda desde afuera", que es lo que
    se quiere medir, y no como el cruce de una frontera que se mueve sola. Las
    ruedas t-N..t-1 son todas anteriores a t-1, asi que no se mira el futuro.

ANCHO IDENTICO ENTRE LA BANDA REAL Y LAS DE CONTROL
    Las franjas de perfil_precio son geometricas y ancladas, asi que un tramo de n
    franjas mide (1+paso)^n - 1 este donde este. Las bandas de control se
    construyen como un tramo de la MISMA cantidad de franjas -> mismo ancho
    porcentual exacto, y ninguna diferencia medida puede venir del ancho.

LO QUE ESTE MODULO NO HACE
    No decide si una banda es soporte o resistencia (eso lo da la direccion de
    llegada, que se deriva del precio), no mide rentabilidad (no hay costos ni
    reglas de salida) y no sabe que es el POC.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.perfil_precio import PASO, limites_bin

DESDE_ARRIBA = "desde_arriba"
DESDE_ABAJO = "desde_abajo"
FUERA = 5          # ruedas que el precio tiene que haber estado afuera
CUARTIL_DENSO = 0.75


@dataclass(frozen=True)
class Evento:
    """Una entrada a la banda. `i` es el indice de la rueda del evento."""
    i: int
    direccion: str
    val: float
    vah: float


# -- bandas --------------------------------------------------------------------

def banda_de_tramo(k0: int, inicio: int, n_bins: int,
                   paso: float = PASO) -> Tuple[float, float]:
    """
    (val, vah) de un tramo de `n_bins` franjas que arranca en la franja local
    `inicio` de un perfil cuya primera franja es `k0`.
    """
    val = float(limites_bin(k0 + inicio, paso)[0])
    vah = float(limites_bin(k0 + inicio + n_bins - 1, paso)[1])
    return val, vah


def ancho_pct_de_tramo(n_bins: int, paso: float = PASO) -> float:
    """Ancho porcentual de un tramo de n franjas. No depende de DONDE este."""
    return ((1.0 + paso) ** n_bins - 1.0) * 100.0


def posiciones_control(p: np.ndarray, a: int, b: int, modo: str) -> np.ndarray:
    """
    Arranques validos para una banda de control del mismo ancho que [a, b].

    modo 'visitado': la franja de arranque tiene volumen > 0.
    modo 'denso'   : ademas esta en el cuartil superior de volumen DE LAS
                     ELEGIBLES, o sea "el lugar mas denso que NO es el area de
                     valor".

    En los dos casos el tramo tiene que entrar entero en el perfil y NO solaparse
    con el area de valor: si se solapara, el control estaria midiendo en parte la
    misma zona que la banda real.

    OJO con el umbral del modo denso (corregido el 19/11/2026, ANTES de correr la
    medicion): calcularlo sobre TODAS las franjas lo deja vacio por construccion.
    El area de valor es, por definicion, donde esta el 70% del volumen, asi que el
    cuartil superior del perfil completo cae casi entero adentro del area -- que es
    justo lo que este control excluye. Medido en la prueba corta: 0 eventos. El
    umbral se calcula sobre las CANDIDATAS, y asi el control responde lo que tiene
    que responder: el area de valor le gana al lugar mas denso que no es ella?
    """
    n = int(b - a + 1)
    total = len(p)
    if n <= 0 or total < n:
        return np.zeros(0, dtype=np.int64)

    arranques = np.arange(total - n + 1, dtype=np.int64)
    # el tramo [s, s+n-1] no puede intersecar [a, b]
    sin_solape = (arranques + n - 1 < a) | (arranques > b)
    cand = arranques[sin_solape]
    if cand.size == 0:
        return cand

    visitado = p[cand] > 0
    cand = cand[visitado]
    if modo == "visitado" or cand.size == 0:
        return cand
    if modo != "denso":
        raise ValueError(f"modo desconocido: {modo}")

    umbral = float(np.quantile(p[cand], CUARTIL_DENSO))
    return cand[p[cand] >= umbral]


def banda_control(p: np.ndarray, k0: int, a: int, b: int, modo: str,
                  rng: np.random.Generator,
                  paso: float = PASO) -> Optional[Tuple[float, float]]:
    """Una banda de control al azar, o None si no hay posicion valida."""
    cand = posiciones_control(p, a, b, modo)
    if cand.size == 0:
        return None
    return banda_de_tramo(k0, int(rng.choice(cand)), int(b - a + 1), paso)


def banda_simetrica(poc: float, pct: float) -> Tuple[float, float]:
    """Banda fija de +-pct alrededor del POC (secundario de la sec. 4.8)."""
    return poc * (1.0 - pct), poc * (1.0 + pct)


# -- eventos -------------------------------------------------------------------

def detectar_eventos(close: Sequence[float], val: Sequence[float],
                     vah: Sequence[float], fuera: int = FUERA) -> List[Evento]:
    """
    Entradas a la banda. Para un evento en t se usa la banda de t-1 y se exige que
    el cierre haya estado FUERA de esa misma banda en t-fuera .. t-1.

    Vectorizado; `_detectar_eventos_ref` es la implementacion de referencia lenta
    con la que los tests lo comparan.
    """
    c = np.asarray(close, dtype=float)
    lo = np.asarray(val, dtype=float)
    hi = np.asarray(vah, dtype=float)
    n = c.size
    if n == 0 or lo.size != n or hi.size != n:
        return []

    t = np.arange(1, n, dtype=np.int64)
    t = t[t - fuera >= 0]
    if t.size == 0:
        return []

    blo, bhi = lo[t - 1], hi[t - 1]
    ok = np.isfinite(blo) & np.isfinite(bhi) & np.isfinite(c[t])
    dentro = ok & (c[t] >= blo) & (c[t] <= bhi)

    # las `fuera` ruedas previas, todas afuera de ESA banda
    afuera = dentro.copy()
    for d in range(1, fuera + 1):
        prev = c[t - d]
        afuera &= np.isfinite(prev) & ((prev < blo) | (prev > bhi))

    idx = np.nonzero(afuera)[0]
    out = []
    for j in idx:
        tt = int(t[j])
        arriba = c[tt - 1] > bhi[j]
        out.append(Evento(i=tt,
                          direccion=DESDE_ARRIBA if arriba else DESDE_ABAJO,
                          val=float(blo[j]), vah=float(bhi[j])))
    return out


def _detectar_eventos_ref(close, val, vah, fuera: int = FUERA) -> List[Evento]:
    """Implementacion de REFERENCIA, lenta y literal. Solo para los tests."""
    c = list(map(float, close))
    lo = list(map(float, val))
    hi = list(map(float, vah))
    out = []
    for t in range(1, len(c)):
        if t - fuera < 0:
            continue
        blo, bhi = lo[t - 1], hi[t - 1]
        if not (np.isfinite(blo) and np.isfinite(bhi) and np.isfinite(c[t])):
            continue
        if not (blo <= c[t] <= bhi):
            continue
        previas = [c[t - d] for d in range(1, fuera + 1)]
        if not all(np.isfinite(x) and (x < blo or x > bhi) for x in previas):
            continue
        out.append(Evento(i=t,
                          direccion=DESDE_ARRIBA if c[t - 1] > bhi else DESDE_ABAJO,
                          val=blo, vah=bhi))
    return out


# -- exceso contra el universo -------------------------------------------------

def retorno_forward(close: Sequence[float], h: int) -> np.ndarray:
    """close[t+h]/close[t] - 1, NaN donde no hay t+h. En fraccion, no en %."""
    c = np.asarray(close, dtype=float)
    out = np.full(c.size, np.nan)
    if h <= 0 or c.size <= h:
        return out
    base = c[:-h]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:-h] = np.where(base > 0, c[h:] / base - 1.0, np.nan)
    return out


def exceso_eventos(eventos: Iterable[Evento], fwd_ticker: np.ndarray,
                   fwd_universo: np.ndarray) -> List[dict]:
    """
    Por evento: exceso = retorno del ticker - retorno equal-weight del universo,
    los dos a h ruedas desde la MISMA rueda. En puntos porcentuales.

    `fwd_universo` viene indexado igual que el ticker (una entrada por rueda del
    ticker): alinearlo por FECHA es responsabilidad de quien llama.
    """
    out = []
    for e in eventos:
        rt, ru = fwd_ticker[e.i], fwd_universo[e.i]
        if not (np.isfinite(rt) and np.isfinite(ru)):
            continue
        out.append({"i": e.i, "direccion": e.direccion,
                    "exceso_pp": float((rt - ru) * 100.0),
                    "ret_pp": float(rt * 100.0)})
    return out


def fuera_de_banda(valor: float, muestras: Sequence[float],
                   lo_pct: float = 5.0, hi_pct: float = 95.0) -> Optional[str]:
    """
    Donde cae `valor` respecto de la banda p5-p95 de los controles.
    Devuelve 'arriba', 'abajo', 'adentro', o None si no hay muestras.
    """
    m = np.asarray([x for x in muestras if np.isfinite(x)], dtype=float)
    if m.size == 0:
        return None
    lo, hi = np.percentile(m, [lo_pct, hi_pct])
    if valor > hi:
        return "arriba"
    if valor < lo:
        return "abajo"
    return "adentro"
