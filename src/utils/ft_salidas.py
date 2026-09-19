"""
ft_salidas.py -- Modulo PURO para analizar las SALIDAS de las estrategias FT.

Sin DB, sin config, sin side effects. Lo usa scripts/forward_testing/ft_analisis_salidas.py.
Metodo, definiciones y resultados: docs/forward_testing/ANALISIS_SALIDAS.md.

QUE RESPONDE
    Con la entrada como supuesto (no se toca): salir ese dia, fue mejor que seguir
    adentro? Tres piezas:

    1. Que hizo el precio DESPUES de la salida, contra el universo del mismo tramo y en
       unidades de la volatilidad del propio ticker. Medido en crudo, en un tramo alcista
       casi toda salida parece temprana: es la trampa del label absoluto del ML
       (docs/features_ml.md sec. 9).
    2. La regla de salida de TECH_SECTOR_v1 como funcion de sus 5 condiciones, y las
       variantes pre-registradas. Con los pesos de scoring.py el score es una regla de
       si/no: los unicos valores posibles son 0/1/1,5/2/2,5/3/3,5/4/4,5/5,5, asi que
       "entrar con >= 4,0" y "salir con <= 3,5" son la misma condicion negada.
    3. Que condicion disparo cada salida (contra la rueda anterior).

LA UNIDAD DE INDEPENDENCIA ES EL DIA
    Muchas salidas caen el mismo dia y comparten mercado. Los IC95 se calculan sobre la
    media POR DIA de salida (cada dia pesa uno), no sobre operaciones.
"""

import math
from collections import OrderedDict
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence

from scipy.stats import t as _t

from src.strategies.scoring import (PTS_MACD, PTS_RSI, PTS_SMA21, PTS_SMA50,
                                    RSI_MAX, calcular_score_tecnico)

# Umbrales de TECH_SECTOR_v1 (scripts/forward_testing/ft_bot_tech_sectorial.py).
# Un test compara estos valores contra el bot para que no se desincronicen.
SCORE_ENTRADA_V1 = 4.0
SCORE_SALIDA_V1 = 3.5

# Clasificacion del precio despues de la salida, en desvios del propio ticker.
UMBRAL_Z = 0.5

CONDICIONES = ("sma200", "sma50", "sma21", "macd", "rsi")
_PUNTOS = OrderedDict([("sma50", PTS_SMA50), ("sma21", PTS_SMA21),
                       ("macd", PTS_MACD), ("rsi", PTS_RSI)])

# Variantes de la regla de salida (pregunta 1, docs/forward_testing/ANALISIS_SALIDAS.md).
# "sin_X" = la condicion X NO puede disparar una salida: se la trata como cumplida al
# evaluar la salida. NO es quitarle los puntos: con eso la salida seria MAS frecuente,
# lo contrario de lo que se quiere probar. La entrada no cambia en ninguna variante.
VARIANTES = OrderedDict([
    ("actual", ()),
    ("sin_sma21", ("sma21",)),
    ("sin_macd", ("macd",)),
    ("sin_rsi", ("rsi",)),
])


# --- la regla -----------------------------------------------------------------

def condiciones_desde_fila(row: dict) -> dict:
    """Las 5 condiciones tal como las evalua scoring.calcular_score_tecnico, + el RSI."""
    _, d = calcular_score_tecnico(row)
    return {"sma200": bool(d["filtro_sma200"]), "sma50": bool(d["cond_sma50"]),
            "sma21": bool(d["cond_sma21"]), "macd": bool(d["cond_macd"]),
            "rsi": bool(d["cond_rsi"]), "rsi_valor": float(d["rsi"])}


def score(cond: dict, neutralizar: Sequence[str] = ()) -> float:
    """Score con los pesos de scoring.py. `neutralizar`: condiciones que se tratan como
    cumplidas. Con el filtro SMA200 fallido (y no neutralizado) el score es 0."""
    c = {k: (True if k in neutralizar else bool(cond[k])) for k in CONDICIONES}
    if not c["sma200"]:
        return 0.0
    return round(sum(p for k, p in _PUNTOS.items() if c[k]), 2)


def cumple_entrada(cond: dict, umbral: float = SCORE_ENTRADA_V1) -> bool:
    return score(cond) >= umbral


def sale(cond: dict, variante: str = "actual", umbral: float = SCORE_SALIDA_V1) -> bool:
    """La salida POR SCORE de TECH_SECTOR_v1 (sin earnings, stop ni take profit)."""
    if variante not in VARIANTES:
        raise ValueError(f"variante {variante!r}; esperaba una de {list(VARIANTES)}")
    return score(cond, neutralizar=VARIANTES[variante]) <= umbral


def combinaciones() -> List[dict]:
    """Las 32 combinaciones de las 5 condiciones."""
    return [dict(zip(CONDICIONES, bits)) for bits in product((False, True), repeat=5)]


def valores_posibles() -> List[float]:
    """Todos los valores que puede tomar el score con los pesos actuales."""
    return sorted({score(c) for c in combinaciones()})


def etiqueta_estado(cond: dict) -> str:
    """'SMA200 SMA50 . MACD RSI' -- las condiciones cumplidas, '.' las que no."""
    return " ".join(k.upper() if cond[k] else "." for k in CONDICIONES)


# --- que disparo la salida ----------------------------------------------------

def gatillos(prev: dict, hoy: dict) -> List[str]:
    """Condiciones que pasaron de cumplidas a no cumplidas entre dos ruedas."""
    out = []
    for k, nombre in (("sma200", "pierde SMA200"), ("sma50", "pierde SMA50"),
                      ("sma21", "pierde SMA21"), ("macd", "pierde MACD")):
        if prev[k] and not hoy[k]:
            out.append(nombre)
    if prev["rsi"] and not hoy["rsi"]:
        out.append("RSI sale por arriba (>68)" if hoy["rsi_valor"] > RSI_MAX
                   else "RSI sale por abajo (<45)")
    return out or ["sin cambio en la ultima rueda"]


# --- que hizo el precio despues -----------------------------------------------

def familia_salida(motivo: str) -> str:
    """Agrupa los motivos de ft_operaciones en familias comparables."""
    m = str(motivo)
    if "SPLIT_FIX" in m:
        return "SPLIT_FIX"
    if m.startswith("SCORE_DEGRADADO"):
        return "SCORE"
    if m in ("STOP_LOSS_ATR", "STOP_LOSS", "TRAILING_SL", "SL_PROTECCION", "BACKSTOP_CHANDELIER"):
        return "STOP"
    if m.startswith("TAKE_PROFIT"):
        return "TAKE_PROFIT"
    if m.startswith("TIME_STOP"):
        return "TIME_STOP"
    if m == "EARNINGS_MANANA":
        return "BALANCE"
    if m == "ROTACION_SECTORIAL":
        return "ROTACION"
    return m


def exceso_z(ret_ticker: float, ret_universo: float, sigma_diaria: float, h: int):
    """(exceso, z): exceso = retorno del ticker menos el del universo en la misma
    ventana; z = exceso / (sigma_diaria * sqrt(h)). z es None si sigma no sirve."""
    if ret_ticker is None or ret_universo is None:
        return None, None
    if any(isinstance(x, float) and math.isnan(x) for x in (ret_ticker, ret_universo)):
        return None, None
    ex = float(ret_ticker) - float(ret_universo)
    if sigma_diaria is None or not (sigma_diaria > 0) or h <= 0:
        return ex, None
    return ex, ex / (float(sigma_diaria) * math.sqrt(h))


def clasificar(z: Optional[float], umbral: float = UMBRAL_Z) -> Optional[str]:
    """'a_tiempo' (siguio peor que el universo), 'temprano' (siguio mejor) o 'indiferente'."""
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return None
    if z < -umbral:
        return "a_tiempo"
    if z > umbral:
        return "temprano"
    return "indiferente"


def ic95_por_dia(valores: Iterable[float], dias: Iterable) -> Dict[str, float]:
    """Media e IC95 con el DIA como unidad: se promedia cada dia y el intervalo se
    calcula sobre esas medias (t de Student con dias-1 grados). Devuelve tambien la
    media simple por operacion, de referencia."""
    por_dia: Dict = {}
    ops = []
    for v, d in zip(valores, dias):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        por_dia.setdefault(d, []).append(float(v))
        ops.append(float(v))
    medias = [sum(x) / len(x) for x in por_dia.values()]
    n = len(medias)
    res = {"ops": len(ops), "dias": n,
           "media_ops": (sum(ops) / len(ops)) if ops else float("nan"),
           "media": float("nan"), "lo": float("nan"), "hi": float("nan")}
    if n == 0:
        return res
    m = sum(medias) / n
    res["media"] = m
    if n >= 3:
        var = sum((x - m) ** 2 for x in medias) / (n - 1)
        se = math.sqrt(var / n)
        q = float(_t.ppf(0.975, n - 1))
        res["lo"], res["hi"] = m - q * se, m + q * se
    return res


# --- re-simulacion de la salida con la entrada fija (paso 1) -------------------

def score_por_rueda(conds: Sequence[Optional[dict]], variante: str) -> List[bool]:
    """Para cada rueda, si la salida por score de la variante manda salir. Una rueda
    sin indicadores (None) no hace salir por score, igual que el bot."""
    return [False if c is None else sale(c, variante) for c in conds]


def ruedas_de_balance(fechas: Sequence, anuncios: Iterable) -> List[bool]:
    """True en la ultima rueda ANTES de cada anuncio de balance: es la rueda en que
    earnings_filter cierra (dia habil anterior al anuncio). La ultima rueda de la serie
    nunca se marca: no se sabe si el anuncio cae en la rueda siguiente."""
    from bisect import bisect_left
    out = [False] * len(fechas)
    for a in set(anuncios):
        j = bisect_left(list(fechas), a) - 1
        if 0 <= j < len(fechas) - 1:
            out[j] = True
    return out


def primera_salida(close: Sequence[float], sale_score: Sequence[bool],
                   balance: Sequence[bool], i_entrada: int,
                   stop: Optional[float], take: Optional[float]):
    """(indice, motivo) de la primera rueda posterior a la entrada en que la posicion
    sale, con la prioridad de TECH_SECTOR_v1: balance -> score -> stop -> take profit.
    Stop y take profit fijos desde la entrada y evaluados contra el close, como en FT.
    (None, None) si no sale dentro de la serie (censura)."""
    for j in range(i_entrada + 1, len(close)):
        c = close[j]
        if c is None or c != c:
            continue
        if balance[j]:
            return j, "BALANCE"
        if sale_score[j]:
            return j, "SCORE"
        if stop is not None and c <= stop:
            return j, "STOP"
        if take is not None and c >= take:
            return j, "TAKE_PROFIT"
    return None, None


def _percentil(xs: Sequence[float], q: float) -> float:
    v = sorted(x for x in xs if x == x)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# Regla de lectura del paso 1 (docs/forward_testing/ANALISIS_SALIDAS.md sec. 6).
ANIOS_MIN_POSITIVOS = 4
CAIDA_COLA_MAX_PP = 1.0


def evaluar_variante(difs: Sequence[float], dias: Sequence, anios: Sequence[int],
                     ret_variante: Sequence[float], ret_actual: Sequence[float]) -> dict:
    """Aplica la regla pre-registrada a la diferencia PAREADA de exceso por operacion
    (variante menos actual, mismas entradas). Las tres condiciones:
      1. IC95 de la diferencia media (dia de entrada como unidad) por encima de cero;
      2. media positiva en >= ANIOS_MIN_POSITIVOS tramos anuales;
      3. el percentil 5 del retorno por operacion no cae mas de CAIDA_COLA_MAX_PP."""
    ic = ic95_por_dia(difs, dias)
    por_anio = OrderedDict()
    for a in sorted(set(anios)):
        sel = [(d, f) for d, f, y in zip(difs, dias, anios) if y == a]
        por_anio[a] = ic95_por_dia([d for d, _ in sel], [f for _, f in sel])["media"]
    positivos = sum(1 for m in por_anio.values() if m == m and m > 0)
    p5v, p5a = _percentil(ret_variante, 5), _percentil(ret_actual, 5)
    c1 = ic["lo"] == ic["lo"] and ic["lo"] > 0
    c2 = positivos >= ANIOS_MIN_POSITIVOS
    c3 = (p5a - p5v) <= CAIDA_COLA_MAX_PP
    return {"ic": ic, "por_anio": por_anio, "anios_positivos": positivos,
            "anios_total": len(por_anio), "p5_variante": p5v, "p5_actual": p5a,
            "c1_ic_sobre_cero": c1, "c2_anios": c2, "c3_cola": c3,
            "pasa": bool(c1 and c2 and c3)}


# --- paso 2: la regla de salida como conjunto de combinaciones (grilla de pesos) ---
# Una regla de salida con umbral fijo queda definida por QUE combinaciones de las 5
# condiciones hacen salir. Se representa como una mascara de 32 bits: el bit k corresponde
# a la combinacion cuyo indice es k (bit 0 = sma200, 1 = sma50, 2 = sma21, 3 = macd, 4 = rsi).

VALORES_GRILLA = (0, 1, 1.5, 2, 3)
PESOS_ACTUALES = OrderedDict([("sma50", PTS_SMA50), ("sma21", PTS_SMA21),
                              ("macd", PTS_MACD), ("rsi", PTS_RSI)])
N_ESTADOS = 32
ESTADO_SIN_DATOS = 32          # rueda sin indicadores: nunca hace salir por score


def indice_estado(cond: dict) -> int:
    return sum(1 << k for k, c in enumerate(CONDICIONES) if cond[c])


def estado_desde_indice(i: int) -> dict:
    return {c: bool((i >> k) & 1) for k, c in enumerate(CONDICIONES)}


def score_pesos(cond: dict, pesos: dict, sma200="obligatoria") -> float:
    """Score con pesos arbitrarios. sma200 = 'obligatoria' (filtro, como hoy) o un peso."""
    if sma200 == "obligatoria":
        if not cond["sma200"]:
            return 0.0
        base = 0.0
    else:
        base = float(sma200) * bool(cond["sma200"])
    return round(base + sum(float(p) * bool(cond[k]) for k, p in pesos.items()), 4)


def mascara_pesos(pesos: dict, sma200="obligatoria", umbral: float = SCORE_SALIDA_V1) -> int:
    m = 0
    for i in range(N_ESTADOS):
        if score_pesos(estado_desde_indice(i), pesos, sma200) <= umbral + 1e-9:
            m |= 1 << i
    return m


def mascara_variante(variante: str) -> int:
    """Mascara de las variantes del paso 1 ('actual', 'sin_sma21', ...)."""
    return sum(1 << i for i in range(N_ESTADOS) if sale(estado_desde_indice(i), variante))


def reglas_grilla(valores: Sequence[float] = VALORES_GRILLA) -> "OrderedDict[int, list]":
    """{mascara: [(sma200, pesos), ...]} -- las reglas DISTINTAS de la grilla y los
    juegos de pesos que producen cada una."""
    out: "OrderedDict[int, list]" = OrderedDict()
    for w200 in ("obligatoria",) + tuple(valores):
        for vals in product(valores, repeat=4):
            pesos = OrderedDict(zip(("sma50", "sma21", "macd", "rsi"), vals))
            out.setdefault(mascara_pesos(pesos, w200), []).append((w200, pesos))
    return out


def etiqueta_pesos(sma200, pesos: dict) -> str:
    s200 = "SMA200 oblig" if sma200 == "obligatoria" else f"SMA200 {sma200:g}"
    return s200 + " | " + " ".join(f"{k.upper()} {v:g}" for k, v in pesos.items())


def bits_reglas(mascaras: Sequence[int]):
    """Matriz booleana (reglas x 33): columna i = la combinacion i hace salir; la columna
    ESTADO_SIN_DATOS es siempre False."""
    import numpy as np
    b = np.zeros((len(mascaras), N_ESTADOS + 1), dtype=bool)
    for r, m in enumerate(mascaras):
        for i in range(N_ESTADOS):
            b[r, i] = bool((m >> i) & 1)
    return b


def diferencias_de_regla(mascara: int, referencia: int) -> dict:
    """Combinaciones que la regla deja de hacer salir y las que agrega, contra la referencia."""
    menos = [etiqueta_estado(estado_desde_indice(i)) for i in range(N_ESTADOS)
             if (referencia >> i) & 1 and not (mascara >> i) & 1]
    mas = [etiqueta_estado(estado_desde_indice(i)) for i in range(N_ESTADOS)
           if (mascara >> i) & 1 and not (referencia >> i) & 1]
    return {"deja_de_salir": menos, "empieza_a_salir": mas}


def es_candidata(media_post: float, media_tramo: float, p5_regla: float, p5_actual: float,
                 caida_cola_max: float = CAIDA_COLA_MAX_PP) -> bool:
    """Seleccion del paso 2 (sec. 8.1): sale en mejores momentos (post < 0), no pierde en el
    tramo (tramo >= 0) y no empeora la cola mas de caida_cola_max puntos."""
    ok = all(x == x for x in (media_post, media_tramo, p5_regla, p5_actual))
    return bool(ok and media_post < 0 and media_tramo >= 0
                and (p5_actual - p5_regla) <= caida_cola_max)
