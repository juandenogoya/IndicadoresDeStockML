"""
niveles_precio.py
Niveles de soporte y resistencia por acumulacion de GIROS. Modulo PURO
(numpy + pandas + src.indicators.estructura): sin DB, sin config, sin side effects.

QUE ES, Y QUE NO ES
    Un nivel es una franja de precio donde el precio DIO VUELTA varias veces: ahi
    se acumulan swings confirmados, de arriba (resistencia) o de abajo (soporte).
    Es DESCRIPTIVO: dice donde reacciono, no que vaya a reaccionar. La tasa de
    rebote se midio el 6/9/2026 y no predice, y el POC se midio el 19/11/2026 y no
    se distingue del azar (docs/perfil_precio.md). Esto es otro objeto: no mide
    permanencia (donde el precio se QUEDO, que es el POC) sino giros (donde dio
    vuelta, que es lo que marca el ojo en un grafico).

GIROS, NO CIERRES NI VOLUMEN
    El ojo marca GIROS, que por definicion son lugares donde el precio estuvo
    POCO; contar cierres o volumen mide PERMANENCIA. Son objetos opuestos y
    ningun contador de permanencia reproduce el grafico (medido el 18/11/2026).

EL PRECIO DEL GIRO ES EL DE SU PROPIA BARRA
    `estructura.swings_en_barra` marca el swing EN LA BARRA DEL SWING. La tabla
    `features_estructura` guarda otra cosa: su `is_sh_N` marca la barra de
    CONFIRMACION (`_desplazar(sh_barra, n)`), y el precio del swing esta n barras
    antes. Leer `high` en la barra de la tabla da un precio corrido hasta n
    ruedas. Aca se usa `swings_en_barra` directo y el precio sale de la barra
    correcta.

SIN INFORMACION FUTURA
    Un swing en la barra i recien se sabe en i+n. `giros_confirmados(..., hasta=D)`
    devuelve solo los swings ya confirmados al cierre de D. Para la vista "los
    niveles de hoy" eso deja afuera las ultimas n ruedas, que es lo correcto.

LA SERIE TIENE QUE SER NOMINAL
    Un nivel psicologico es un precio nominal: la gente recuerda "208 dolares".
    `precios_diarios` NO sirve para esto: su historia esta en precios ajustados por
    dividendos y los ultimos meses en nominales (ver AGENDA, 19/11/2026). Medido en
    CVX: dos clusters de 13 y 19 giros que se ven como niveles de 1,7% de ancho son
    escaleras nominales de 15%. El ajuste no solo desparrama niveles reales:
    FABRICA niveles, porque un precio nominal que baja al ritmo del dividendo se
    vuelve una linea horizontal perfecta en la serie ajustada.
"""

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.indicators.estructura import swings_en_barra

H = "H"          # swing high: el precio llego desde abajo y fue rechazado
L = "L"          # swing low: el precio llego desde arriba y reboto

VENTANA = 5      # N de los swings
TOL = 0.010      # media banda: +-1% -> banda de ~2%
SEP = 15         # ruedas minimas entre dos toques del MISMO nivel
PASO_GRILLA = 0.0025
MIN_TOQUES = 3


@dataclass(frozen=True)
class Giro:
    i: int
    precio: float
    tipo: str


@dataclass(frozen=True)
class Nivel:
    """Una franja donde se acumularon giros. Precios en la moneda del ticker."""
    precio: float        # centro geometrico de la banda
    val: float
    vah: float
    toques: int          # giros, ya separados por SEP ruedas
    highs: int
    lows: int
    anios: int           # anios calendario distintos con al menos un toque
    primer_i: int
    ultimo_i: int

    @property
    def polaridad(self) -> str:
        if self.highs and self.lows:
            return "ambos"
        return "resistencia" if self.highs else "soporte"


# -- giros ---------------------------------------------------------------------

def giros_confirmados(high, low, n: int = VENTANA,
                      hasta: Optional[int] = None) -> List[Giro]:
    """
    Swings marcados en SU barra, devueltos solo si ya estan confirmados.

    `hasta` es el indice de la ultima rueda conocida (por defecto, la ultima de la
    serie). Un swing en i se confirma en i+n, asi que se exige i + n <= hasta.
    """
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    if h.size == 0:
        return []
    tope = (len(h) - 1) if hasta is None else int(hasta)
    sh, sl = swings_en_barra(h, lo, n)
    out = []
    for i in np.flatnonzero(np.nan_to_num(sh, nan=0.0).astype(bool)):
        if i + n <= tope:
            out.append(Giro(int(i), float(h[i]), H))
    for i in np.flatnonzero(np.nan_to_num(sl, nan=0.0).astype(bool)):
        if i + n <= tope:
            out.append(Giro(int(i), float(lo[i]), L))
    return sorted(out, key=lambda g: (g.i, g.tipo))


# -- toques --------------------------------------------------------------------

def toques_en_banda(giros: Sequence[Giro], val: float, vah: float,
                    sep: int = SEP) -> List[Giro]:
    """
    Giros dentro de [val, vah], exigiendo `sep` ruedas entre uno y el siguiente.

    La separacion existe para que una lateralizacion corta no cuente como cinco
    toques: son un solo episodio visto de cerca.
    """
    dentro = [g for g in giros if val <= g.precio <= vah]
    out: List[Giro] = []
    ult = None
    for g in sorted(dentro, key=lambda x: x.i):
        if ult is None or g.i - ult >= sep:
            out.append(g)
            ult = g.i
    return out


def _banda(centro: float, tol: float) -> Tuple[float, float]:
    return centro * (1.0 - tol), centro * (1.0 + tol)


def separacion_minima(tol: float) -> float:
    """Distancia LOGARITMICA minima entre dos niveles: dos anchos de banda."""
    return float(np.log1p(4.0 * tol))


def _muy_cerca(a: float, b: float, tol: float) -> bool:
    """
    En log, porque `abs(a/b - 1)` NO es simetrico: con 156,09 y 162,45 da 3,92%
    en un sentido y 4,08% en el otro, asi que la de-duplicacion dependia de cual
    nivel se aceptaba primero. Sobre una grilla geometrica la distancia es el log.
    """
    return abs(float(np.log(a / b))) < separacion_minima(tol)


def _anios(giros: Sequence[Giro], anio_de: np.ndarray) -> int:
    return len({int(anio_de[g.i]) for g in giros})


def _nivel(centro: float, tq: Sequence[Giro], tol: float,
           anio_de: np.ndarray) -> Nivel:
    val, vah = _banda(centro, tol)
    return Nivel(precio=float(centro), val=float(val), vah=float(vah),
                 toques=len(tq),
                 highs=sum(1 for g in tq if g.tipo == H),
                 lows=sum(1 for g in tq if g.tipo == L),
                 anios=_anios(tq, anio_de),
                 primer_i=min(g.i for g in tq), ultimo_i=max(g.i for g in tq))


# -- deteccion -----------------------------------------------------------------

def detectar(high, low, fechas: Sequence, n: int = VENTANA, tol: float = TOL,
             sep: int = SEP, paso_grilla: float = PASO_GRILLA,
             min_toques: int = MIN_TOQUES, max_niveles: int = 12,
             hasta: Optional[int] = None) -> List[Nivel]:
    """
    Niveles ordenados por PERSISTENCIA (anios distintos) y despues por toques.

    El ranking lo encabeza la persistencia y no el conteo: cuatro toques en un
    trimestre son un solo episodio, y cinco repartidos en dos anios atravesaron
    varios regimenes. Es lo que distingue las bandas que el ojo acepta.

    La seleccion es GOLOSA sobre una grilla geometrica fina: se toma el centro con
    mas toques, se lo RE-CENTRA en la media geometrica de sus propios toques (para
    que el resultado no dependa de donde cayo la grilla) y se excluye su entorno
    antes de buscar el siguiente. Dos niveles no pueden quedar a menos de dos
    anchos de banda: mas cerca que eso, el ojo ve una zona y no dos.
    """
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    g = giros_confirmados(h, lo, n, hasta)
    if len(g) < min_toques:
        return []

    anio_de = np.asarray([pd.Timestamp(f).year for f in fechas], dtype=np.int64)
    precios = np.array([x.precio for x in g], dtype=float)

    grilla = np.exp(np.arange(np.log(precios.min()), np.log(precios.max()),
                              np.log1p(paso_grilla)))
    if grilla.size == 0:
        return []
    cuentas = np.array([len(toques_en_banda(g, *_banda(c, tol), sep=sep))
                        for c in grilla])

    elegidos: List[Nivel] = []
    usados: List[float] = []
    orden = np.argsort(-cuentas, kind="stable")
    for k in orden:
        if cuentas[k] < min_toques:
            break
        c = float(grilla[k])
        if any(_muy_cerca(c, u, tol) for u in usados):
            continue
        tq = toques_en_banda(g, *_banda(c, tol), sep=sep)
        if len(tq) < min_toques:
            continue
        # re-centrar: el nivel lo definen sus toques, no la grilla
        centro = float(np.exp(np.mean(np.log([x.precio for x in tq]))))
        tq = toques_en_banda(g, *_banda(centro, tol), sep=sep)
        if len(tq) < min_toques:
            continue
        if any(_muy_cerca(centro, u, tol) for u in usados):
            continue
        elegidos.append(_nivel(centro, tq, tol, anio_de))
        usados.append(centro)
        if len(elegidos) >= max_niveles:
            break

    return sorted(elegidos, key=lambda x: (-x.anios, -x.toques, x.precio))


# -- nulo ----------------------------------------------------------------------

def permutar(close, high, low, rng: np.random.Generator):
    """
    Serie con los MISMOS retornos de cierre en otro orden, y el rango H-L de cada
    rueda conservado en proporcion. Mismo inicio, misma distribucion, mismo n:
    lo unico que cambia es el ORDEN. Es el nulo de "aca no hay estructura".
    """
    c = np.asarray(close, dtype=float)
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    r = c[1:] / c[:-1]
    perm = rng.permutation(r)
    c2 = np.empty_like(c)
    c2[0] = c[0]
    c2[1:] = c[0] * np.cumprod(perm)
    # el rango de cada rueda viaja con su barra, no con su posicion
    idx = np.argsort(rng.permutation(len(c)))
    fh = np.where(c > 0, h / c, 1.0)[idx]
    fl = np.where(c > 0, lo / c, 1.0)[idx]
    return c2, c2 * fh, c2 * fl
