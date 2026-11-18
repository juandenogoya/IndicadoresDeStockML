"""
perfil_precio.py
Perfil de precio por volumen: POC y area de valor. Modulo PURO (numpy + pandas):
sin DB, sin config, sin side effects. Lo pueden importar el pipeline, el dashboard
y el MCP.

QUE ES
    El precio no se repite nunca exacto, asi que no se puede contar "cuantas veces
    estuvo en 335,78". Se corta el eje de precios en franjas (bins) y se reparte el
    volumen de cada rueda sobre las franjas que toco su rango H-L. El POC es la
    franja con mas volumen de la ventana; el area de valor son las franjas que
    acumulan el 70% del volumen alrededor del POC.

LAS TRES DECISIONES, Y POR QUE NINGUNA PIDE TUNEO (medido el 18/11/2026)

    PASO = 0,5%. El ancho del bin casi no mueve el resultado: en JPM, ventana 200,
    el POC va de 336,29 (bins de 0,1%) a 335,78 (bins de 2%), medio punto con el
    ancho del bin cambiando 20 veces. Recien a 5% se degrada, y ahi la franja mide
    28 dolares. Lo que si importa es que el pico siga siendo un pico: con bins de
    0,1% la franja del POC se queda con el 0,9% del volumen de la ventana, que ya
    no es un maximo sino ruido. Eso lo informa `pico_pct`.

    VOLUMEN y no tiempo. En las ventanas de 21/52/200 las dos versiones dan el
    MISMO POC en 11 de 15 casos medidos (mediana de la diferencia 0,00%, maxima
    2,96%): dentro de una ventana el volumen diario es parejo y no mueve el
    maximo. Se usa volumen igual, porque es lo que corresponde conceptualmente y
    no cuesta nada. La diferencia aparece en ventanas largas donde cambia el
    regimen de volumen (AVGO a 5 anios: 344,68 por volumen contra 44,15 por
    tiempo), pero ahi el perfil ya es plano y el POC no sirve por otro motivo.

    GRILLA ANCLADA en potencias de (1+PASO) desde 1,0, igual para todos los
    tickers y para siempre. No se construye con el minimo y el maximo de la
    ventana: con una grilla asi los bordes se corren cada dia y parte del
    movimiento del POC lo fabrica la grilla. Anclada, la franja 335,78-337,46 es
    literalmente la misma hoy que hace dos anios, y dos POC de fechas distintas se
    comparan por su indice de bin.

    Geometrica y no en dolares: un bin de 1 dolar es enorme para Ford a 11 y
    ridiculo para LMT a 590. En porcentaje significa lo mismo en los 199.

EL ANCHO DEL AREA DE VALOR ES EL ERROR BAR DEL POC, no un adorno
    Un POC sin su area de valor engana. CAT al 17/11/2026 da POC 811,50 en la
    ventana de 52, que parece un nivel; su area de valor mide 40,5% de ancho, o
    sea que el perfil es plano y el 811,50 es el maximo del ruido. JPM en la
    ventana de 21 da un area de 3,0%: ahi si hay equilibrio y el POC significa
    algo. Quien consuma esto decide con `ancho_va_pct`, y el umbral se fija de
    antemano, no mirando resultados.

INVARIANCIA (test en tests/test_perfil_precio.py)
    El perfil de una ventana que termina en t usa solo barras hasta t, y la grilla
    no depende de los datos. La fila de una fecha no cambia cuando llegan barras
    nuevas.

LO QUE ESTE MODULO NO HACE, A PROPOSITO
    No dice si el POC es soporte o resistencia: eso depende del precio de HOY y se
    DERIVA en cada lectura con `ubicacion()` (si se guardara como columna, el dia
    que el precio lo cruce la etiqueta queda al reves). No marca nodos de alto
    volumen, vacios ni el dia de mayor volumen. Y no afirma nada sobre que hace el
    precio cuando vuelve a estas zonas: la tasa de rebote se midio el 6/9/2026 y
    no predice (85% dentro de muestra, 56% fuera), ver
    memory/zonas_precio_descartado.md.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

PASO = 0.005          # ancho del bin: 0,5%
COBERTURA = 0.70      # fraccion del volumen que define el area de valor
VENTANAS: Tuple[int, ...] = (21, 52, 200)


@dataclass(frozen=True)
class Perfil:
    """Resultado de UNA ventana. Precios en la moneda del ticker."""
    poc: float            # centro de la franja con mas volumen
    val: float            # borde inferior del area de valor
    vah: float            # borde superior del area de valor
    ancho_va_pct: float   # (vah/val - 1) * 100; el error bar del POC
    ruedas: int
    bins: int             # franjas que toco el precio en la ventana
    pico_pct: float       # % del volumen que se lleva la franja del POC
    volumen: float

    @property
    def valido(self) -> bool:
        return bool(np.isfinite(self.poc))


_VACIO = Perfil(np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan, 0.0)

CAMPOS = ("poc", "val", "vah", "ancho_va_pct", "ruedas", "bins",
          "pico_pct", "volumen")


# -- grilla anclada ------------------------------------------------------------

def indice_bin(precio, paso: float = PASO):
    """Indice de franja de un precio: floor(ln(p) / ln(1+paso)). Ancla universal 1,0."""
    p = np.asarray(precio, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.floor(np.log(p) / np.log1p(paso))
    return k if k.ndim else float(k)


def limites_bin(k, paso: float = PASO):
    """(borde_inferior, borde_superior) de la franja k."""
    k = np.asarray(k, dtype=float)
    return (1.0 + paso) ** k, (1.0 + paso) ** (k + 1.0)


def centro_bin(k, paso: float = PASO):
    """Centro geometrico de la franja k."""
    lo, hi = limites_bin(k, paso)
    return np.sqrt(lo * hi)


# -- perfil --------------------------------------------------------------------

def perfil_bins(high, low, volumen, paso: float = PASO
                ) -> Tuple[Optional[int], np.ndarray]:
    """
    Volumen por franja. Devuelve (indice de la primera franja, array de volumenes).

    El volumen de cada rueda se reparte UNIFORME sobre las franjas que toca su
    rango H-L. Es una aproximacion: la barra diaria no dice donde dentro del dia
    se opero. Medida contra otros tres supuestos (todo al cierre, todo al precio
    tipico, triangular), el POC se mueve 1,0-3,6% en las ventanas de 52 y 200 y
    hasta 13,3% en la de 21, que es la razon por la que la ventana corta se lee
    con cuidado. La validacion contra las barras de 30 minutos quedo PENDIENTE:
    `precios_intradia_30m` captura 11 de las 13 barras y solo el 57-70% del
    volumen diario, y lo que le falta es el tramo final de la rueda.
    """
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    v = np.asarray(volumen, dtype=float)
    if h.size == 0:
        return None, np.zeros(0)

    v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    alto = np.maximum(h, lo)            # tolera H<L por dato sucio
    bajo = np.minimum(h, lo)
    ok = np.isfinite(alto) & np.isfinite(bajo) & (bajo > 0) & (v > 0)
    if not ok.any():
        return None, np.zeros(0)

    k_lo = np.asarray(indice_bin(bajo[ok], paso), dtype=np.int64)
    k_hi = np.asarray(indice_bin(alto[ok], paso), dtype=np.int64)
    k0, k1 = int(k_lo.min()), int(k_hi.max())
    p = np.zeros(k1 - k0 + 1)

    n = (k_hi - k_lo + 1).astype(np.int64)
    peso = np.repeat(v[ok] / n, n)
    # indices k_lo_i .. k_hi_i concatenados, sin loop por rueda
    arranques = np.repeat(np.cumsum(n) - n, n)
    idx = np.repeat(k_lo, n) + (np.arange(int(n.sum())) - arranques)
    np.add.at(p, idx - k0, peso)
    return k0, p


def _area_valor(p: np.ndarray, cobertura: float) -> Tuple[int, int, int]:
    """(bin del POC, bin inferior, bin superior) del area de valor, indices locales."""
    k = int(np.argmax(p))
    total = p.sum()
    if total <= 0:
        return k, k, k
    acum, a, b = p[k], k, k
    n = len(p)
    while acum < cobertura * total and (a > 0 or b < n - 1):
        izq = p[a - 1] if a > 0 else -np.inf
        der = p[b + 1] if b < n - 1 else -np.inf
        if der >= izq:
            b += 1
            acum += p[b]
        else:
            a -= 1
            acum += p[a]
    return k, a, b


def calcular(high, low, volumen, paso: float = PASO,
             cobertura: float = COBERTURA) -> Perfil:
    """POC y area de valor de UNA ventana ya recortada."""
    k0, p = perfil_bins(high, low, volumen, paso)
    if k0 is None or p.sum() <= 0:
        return _VACIO
    k, a, b = _area_valor(p, cobertura)
    val = float(limites_bin(k0 + a, paso)[0])
    vah = float(limites_bin(k0 + b, paso)[1])
    return Perfil(
        poc=float(centro_bin(k0 + k, paso)),
        val=val, vah=vah,
        ancho_va_pct=(vah / val - 1.0) * 100.0,
        ruedas=int(np.asarray(high).size),
        bins=int((p > 0).sum()),
        pico_pct=float(p[k] / p.sum() * 100.0),
        volumen=float(p.sum()),
    )


def _columnas(df: pd.DataFrame):
    d = df.sort_values("fecha") if "fecha" in df.columns else df
    return (d["high"].to_numpy(dtype=float),
            d["low"].to_numpy(dtype=float),
            d["volume"].to_numpy(dtype=float), d)


def calcular_ventanas(df: pd.DataFrame, ventanas: Iterable[int] = VENTANAS,
                      paso: float = PASO, cobertura: float = COBERTURA) -> pd.DataFrame:
    """
    Una fila por ventana, tomando las ultimas N barras de df.

    df: columnas high, low, volume (fecha opcional, solo para ordenar).
    Una ventana con menos barras que N sale en NaN: el perfil de 200 ruedas de un
    ticker que tiene 80 no es el perfil de 200 ruedas.
    """
    h, lo, v, _ = _columnas(df)
    filas = []
    for n in ventanas:
        n = int(n)
        r = calcular(h[-n:], lo[-n:], v[-n:], paso, cobertura) if len(h) >= n else _VACIO
        fila = {"ventana": n}
        fila.update({c: getattr(r, c) for c in CAMPOS})
        filas.append(fila)
    return pd.DataFrame(filas)


def calcular_rodante(df: pd.DataFrame, ventana: int, paso: float = PASO,
                     cobertura: float = COBERTURA) -> pd.DataFrame:
    """
    POC y area de valor rueda por rueda (ventana movil). Para analisis, no para el
    pipeline: es O(n x ventana) y recalcula el perfil entero en cada fecha.
    """
    h, lo, v, d = _columnas(df)
    n = int(ventana)
    filas = []
    for t in range(len(h)):
        r = (_VACIO if t + 1 < n else
             calcular(h[t + 1 - n:t + 1], lo[t + 1 - n:t + 1], v[t + 1 - n:t + 1],
                      paso, cobertura))
        filas.append({"poc": r.poc, "val": r.val, "vah": r.vah,
                      "ancho_va_pct": r.ancho_va_pct})
    out = pd.DataFrame(filas)
    if "fecha" in d.columns:
        out.insert(0, "fecha", d["fecha"].to_numpy())
    return out


def ubicacion(precio: float, p: Perfil) -> str:
    """
    Donde esta el precio respecto del area de valor. Se DERIVA en cada lectura: el
    POC no es soporte ni resistencia por si mismo, lo es respecto del precio de hoy.
    """
    if not p.valido or not np.isfinite(precio):
        return "sin dato"
    if precio > p.vah:
        return "arriba del area de valor"
    if precio < p.val:
        return "abajo del area de valor"
    return "dentro del area de valor"
