"""
ft_entradas.py -- Modulo PURO para analizar las ENTRADAS de las estrategias FT.

Sin DB, sin config, sin side effects. Lo usa
scripts/forward_testing/ft_analisis_entradas.py.
Metodo, definiciones y resultados: docs/forward_testing/ANALISIS_ENTRADAS.md.

QUE RESPONDE
    Con el reparto (5 por sector) y la salida de TECH_SECTOR_v1 fijos, la entrada es la
    unica variable. Antes de armar ninguna grilla hay que saber en que ESTADOS cae el
    universo y con que frecuencia: un peso sobre un estado que ocurre 50 veces en cinco
    anios no es una palanca, es ruido con nombre.

LA REGLA ACTUAL, ENUMERADA
    Con los pesos de src/strategies/scoring.py (SMA50 2,0 | SMA21 1,0 | MACD 1,5 |
    RSI 1,0) y umbral 4,0, solo 4 de los 16 estados binarios califican:
        {SMA50, SMA21, RSI} = 4,0   {SMA50, SMA21, MACD} = 4,5
        {SMA50, RSI, MACD}  = 4,5   las cuatro          = 5,5
    Es decir, la entrada de TECH_SECTOR_v1 es exactamente:
        close > SMA200  Y  close > SMA50  Y  (al menos 2 de {SMA21, MACD, RSI})
    SMA50 no "pesa mas": es obligatoria de hecho (sin ella el maximo es 3,5 < 4,0).
    `regla_v1_booleana` escribe eso sin pesos y un test lo compara, fila por fila,
    contra `calcular_score_tecnico(...) >= 4,0`.

ZONAS DE DISTANCIA (paso 0, cortes en % fijo)
    SMA50 y SMA21 dejan de ser un bit y pasan a zonas de `dist_sma* = (close-sma)/sma*100`.
    Los cortes de SMA21 salen de escalar los de SMA50 por raiz(50/21) = 1,543 -- criterio
    declarado ANTES de mirar la distribucion, no ajustado a ella.

    CONVENCION DE BORDES: intervalos (a, b], abiertos por izquierda. Asi la frontera en 0
    separa exactamente igual que la condicion binaria `close > sma`: dist > 0 cae en las
    zonas positivas y dist <= 0 en las negativas. Con intervalos [a, b) una fila con
    close == sma quedaria del lado positivo y la condicion binaria diria que no.

    OJO CON EL REDONDEO: `indicadores_tecnicos.dist_sma*` esta redondeada a 4 decimales.
    Una fila con close apenas por encima de la media puede tener dist guardada 0.0000 y
    caer en la zona negativa mientras la condicion binaria dice True. Por eso la zona se
    calcula desde `distancia_pct(close, sma)` (sin redondear); `cambia_de_zona` cuenta
    cuantas filas clasificarian distinto con la columna guardada y `discrepancias_borde`
    busca diferencias que el redondeo NO explica (columna que no corresponde al close).

UNIDADES
    Los cortes de este paso son PORCENTAJE FIJO, igual para los 200 tickers. Eso mezcla
    volatilidades: 5% sobre la SMA50 son mas de cuatro dias de rango en un ticker con
    ATR% 1,2 y un dia cualquiera en uno con ATR% 5,0. Es deliberado y es la primera de
    las tres calibraciones previstas (universal -> por volatilidad -> por sector). La
    consecuencia a vigilar esta en el doc: con tope de 5 por sector, si las zonas
    extremas puntuan alto la cartera se llena de los tickers mas volatiles de cada sector
    sin que nadie lo haya decidido.
"""

import math
from collections import OrderedDict
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.strategies.scoring import (PTS_MACD, PTS_RSI, PTS_SMA21, PTS_SMA50,
                                    RSI_MAX, RSI_MIN, calcular_score_tecnico)

# Umbral de entrada de TECH_SECTOR_v1 (scripts/forward_testing/ft_bot_tech_sectorial.py).
# Un test lo compara contra el bot para que no se desincronicen.
SCORE_ENTRADA_V1 = 4.0

# Condiciones binarias, en el orden en que se enumeran.
CONDICIONES = ("sma200", "sma50", "sma21", "macd", "rsi")
_PUNTOS = OrderedDict([("sma50", PTS_SMA50), ("sma21", PTS_SMA21),
                       ("macd", PTS_MACD), ("rsi", PTS_RSI)])

# --- zonas --------------------------------------------------------------------

# Factor de escala de los cortes de SMA21 respecto de los de SMA50.
FACTOR_S21 = math.sqrt(50.0 / 21.0)          # 1,5430...

CORTES_S50 = (-10.0, -5.0, 0.0, 5.0, 10.0)
CORTES_S21 = (-6.5, -3.5, 0.0, 3.5, 6.5)     # ~ CORTES_S50 / FACTOR_S21, redondeado

ZONAS_S50 = ("S50_BAJA", "S50_MEDIA_NEG", "S50_CERCA_NEG",
             "S50_CERCA", "S50_MEDIA", "S50_ALTA")
ZONAS_S21 = ("S21_BAJA", "S21_MEDIA_NEG", "S21_CERCA_NEG",
             "S21_CERCA", "S21_MEDIA", "S21_ALTA")
ZONAS_MACD = ("MACD_DOWN", "MACD_UP")
ZONAS_RSI = ("RSI_LOW", "RSI_IN", "RSI_HIGH")   # < 45 | [45, 68] | > 68

# Cortes del RSI para `zona`/`clasificar_serie`, que usan intervalos (a, b]. El primer
# corte se baja un epsilon para que RSI == RSI_MIN caiga en RSI_IN, como en scoring.py
# (RSI_MIN <= rsi <= RSI_MAX, los dos bordes inclusive).
CORTES_RSI = (RSI_MIN - 1e-9, RSI_MAX)

# Zonas que la condicion binaria correspondiente da por cumplidas.
ZONAS_S50_OK = ("S50_CERCA", "S50_MEDIA", "S50_ALTA")
ZONAS_S21_OK = ("S21_CERCA", "S21_MEDIA", "S21_ALTA")

EJES = ("s50", "s21", "macd", "rsi")
_ZONAS_POR_EJE = OrderedDict([("s50", ZONAS_S50), ("s21", ZONAS_S21),
                              ("macd", ZONAS_MACD), ("rsi", ZONAS_RSI)])


def _es_nan(valor) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def distancia_pct(close, sma) -> Optional[float]:
    """(close - sma) / sma * 100, sin redondear. None si falta un dato o sma <= 0."""
    if _es_nan(close) or _es_nan(sma):
        return None
    close, sma = float(close), float(sma)
    if sma <= 0:
        return None
    return (close - sma) / sma * 100.0


def zona(valor, cortes: Sequence[float], etiquetas: Sequence[str]) -> Optional[str]:
    """Etiqueta de `valor` con intervalos (a, b]. len(etiquetas) == len(cortes) + 1."""
    if len(etiquetas) != len(cortes) + 1:
        raise ValueError("etiquetas debe tener un elemento mas que cortes")
    if _es_nan(valor):
        return None
    valor = float(valor)
    for i, corte in enumerate(cortes):
        if valor <= corte:
            return etiquetas[i]
    return etiquetas[-1]


def zona_s50(dist) -> Optional[str]:
    return zona(dist, CORTES_S50, ZONAS_S50)


def zona_s21(dist) -> Optional[str]:
    return zona(dist, CORTES_S21, ZONAS_S21)


def zona_macd(macd, signal) -> Optional[str]:
    """MACD_UP si macd > signal. El `and hist > 0` de scoring.py es la misma condicion
    escrita dos veces (hist = macd - signal), no agrega ningun estado."""
    if _es_nan(macd) or _es_nan(signal):
        return None
    return ZONAS_MACD[1] if float(macd) > float(signal) else ZONAS_MACD[0]


def zona_rsi(rsi) -> Optional[str]:
    """Tres zonas para MEDIR. La regla sigue decidiendo con el binario (ver `rsi_ok`):
    fallar por debilidad (<45) y por fuerza (>68) son lecturas opuestas y separarlas
    no cuesta nada."""
    if _es_nan(rsi):
        return None
    rsi = float(rsi)
    if rsi < RSI_MIN:
        return ZONAS_RSI[0]
    if rsi <= RSI_MAX:
        return ZONAS_RSI[1]
    return ZONAS_RSI[2]


def rsi_ok(rsi) -> bool:
    """La condicion binaria de scoring.py: RSI_MIN <= rsi <= RSI_MAX."""
    if _es_nan(rsi):
        return False
    return RSI_MIN <= float(rsi) <= RSI_MAX


def clasificar_serie(valores, cortes: Sequence[float], etiquetas: Sequence[str]):
    """Version vectorizada de `zona` para una Series de pandas. Misma convencion de
    bordes (a, b]. Un test la compara contra la escalar fila por fila."""
    import pandas as pd

    if len(etiquetas) != len(cortes) + 1:
        raise ValueError("etiquetas debe tener un elemento mas que cortes")
    bins = [-math.inf] + list(cortes) + [math.inf]
    out = pd.cut(valores, bins=bins, labels=list(etiquetas), right=True)
    return out.astype(object).where(out.notna(), None)


# --- estado de una fila -------------------------------------------------------

def condiciones_desde_fila(row: dict) -> Dict[str, bool]:
    """Las 5 condiciones binarias tal como las evalua scoring.calcular_score_tecnico."""
    _, d = calcular_score_tecnico(row)
    return {"sma200": bool(d["filtro_sma200"]), "sma50": bool(d["cond_sma50"]),
            "sma21": bool(d["cond_sma21"]), "macd": bool(d["cond_macd"]),
            "rsi": bool(d["cond_rsi"])}


def estado_desde_fila(row: dict) -> dict:
    """Estado por ZONAS de una fila de indicadores.

    Claves: filtro_sma200 (bool), dist_sma200/50/21 (float, sin redondear),
    s50/s21/macd/rsi (etiqueta de zona o None), rsi_ok (bool).
    """
    close = row.get("close")
    d200 = distancia_pct(close, row.get("sma200"))
    d50 = distancia_pct(close, row.get("sma50"))
    d21 = distancia_pct(close, row.get("sma21"))
    return {
        "filtro_sma200": bool(d200 is not None and d200 > 0),
        "dist_sma200": d200, "dist_sma50": d50, "dist_sma21": d21,
        "s50": zona_s50(d50), "s21": zona_s21(d21),
        "macd": zona_macd(row.get("macd"), row.get("macd_signal")),
        "rsi": zona_rsi(row.get("rsi14")), "rsi_ok": rsi_ok(row.get("rsi14")),
    }


def clave_estado(estado: dict) -> Optional[str]:
    """'S50_CERCA|S21_CERCA|MACD_UP|RSI_IN'. None si algun eje quedo sin clasificar."""
    partes = [estado.get(eje) for eje in EJES]
    if any(p is None for p in partes):
        return None
    return "|".join(partes)


def estados_posibles() -> List[str]:
    """Las 144 claves de estado (6 x 6 x 2 x 2) con el RSI binario para decidir, o las
    216 con el RSI desglosado en tres para medir -- esta funcion devuelve las 216."""
    return ["|".join(combo) for combo in product(*_ZONAS_POR_EJE.values())]


def zonas_del_eje(eje: str) -> Tuple[str, ...]:
    if eje not in _ZONAS_POR_EJE:
        raise ValueError(f"eje {eje!r}; esperaba uno de {list(_ZONAS_POR_EJE)}")
    return _ZONAS_POR_EJE[eje]


def condiciones_desde_estado(estado: dict) -> Optional[Dict[str, bool]]:
    """Las 5 condiciones binarias derivadas de las zonas. Sirve para comprobar que las
    zonas no perdieron informacion respecto de la regla actual."""
    if estado.get("s50") is None or estado.get("s21") is None or estado.get("macd") is None:
        return None
    return {"sma200": bool(estado["filtro_sma200"]),
            "sma50": estado["s50"] in ZONAS_S50_OK,
            "sma21": estado["s21"] in ZONAS_S21_OK,
            "macd": estado["macd"] == ZONAS_MACD[1],
            "rsi": bool(estado["rsi_ok"])}


# --- la regla actual ----------------------------------------------------------

def score(cond: Dict[str, bool]) -> float:
    """Score con los pesos de scoring.py. Con el filtro SMA200 fallido, 0."""
    if not cond["sma200"]:
        return 0.0
    return round(sum(p for k, p in _PUNTOS.items() if cond[k]), 2)


def cumple_entrada_por_score(cond: Dict[str, bool],
                             umbral: float = SCORE_ENTRADA_V1) -> bool:
    """La entrada tal como la decide el bot: score >= umbral."""
    return score(cond) >= umbral


def regla_v1_booleana(cond: Dict[str, bool]) -> bool:
    """La MISMA regla escrita sin pesos: SMA200 y SMA50 y al menos 2 de las otras tres.

    Implementacion INDEPENDIENTE de `cumple_entrada_por_score`, a proposito: un test
    las compara sobre las 32 combinaciones y sobre la historia completa. Si difieren,
    la enumeracion del doc esta mal y cualquier grilla construida encima mide otra cosa.
    """
    if not cond["sma200"] or not cond["sma50"]:
        return False
    return (int(bool(cond["sma21"])) + int(bool(cond["macd"]))
            + int(bool(cond["rsi"]))) >= 2


def combinaciones_binarias() -> List[Dict[str, bool]]:
    """Las 32 combinaciones de las 5 condiciones."""
    return [dict(zip(CONDICIONES, bits)) for bits in product((False, True), repeat=5)]


def valores_posibles_score() -> List[float]:
    """Todos los valores que puede tomar el score con los pesos actuales."""
    return sorted({score(c) for c in combinaciones_binarias()})


def reglas_por_umbral() -> "OrderedDict[float, Tuple[str, ...]]":
    """Umbral -> estados binarios que califican. Muestra que el umbral no es un
    parametro continuo: todo el rango produce unas pocas reglas distintas."""
    out = OrderedDict()
    for umbral in valores_posibles_score():
        if umbral <= 0:
            continue
        estados = tuple(sorted(etiqueta_estado(c) for c in combinaciones_binarias()
                               if score(c) >= umbral))
        out[umbral] = estados
    return out


def etiqueta_estado(cond: Dict[str, bool]) -> str:
    """'SMA200 SMA50 . MACD RSI' -- las condiciones cumplidas, '.' las que no."""
    return " ".join(k.upper() if cond[k] else "." for k in CONDICIONES)


# --- juegos de pesos ----------------------------------------------------------

# Pesos como tupla (sma50, sma21, macd, rsi). SMA200 no pondera: es filtro.
PESOS_V1 = (PTS_SMA50, PTS_SMA21, PTS_MACD, PTS_RSI)

# Lista CERRADA de juegos para la seccion `seleccion`. Criterio de construccion,
# declarado: (a) la v1; (b) los dos umbrales vecinos con los pesos de la v1 -- los
# unicos dos que producen una regla distinta; (c) subir el peso de cada condicion NO
# obligatoria hasta un valor que cambie la regla booleana, una por vez. No se eligieron
# mirando resultados de rendimiento: de ellos solo se conoce cuanto cambia la SELECCION
# (docs/forward_testing/ANALISIS_ENTRADAS.md sec. 7).
JUEGOS_PESOS = OrderedDict([
    ("v1", (PESOS_V1, SCORE_ENTRADA_V1)),
    ("umbral_3_5", (PESOS_V1, 3.5)),
    ("umbral_4_5", (PESOS_V1, 4.5)),
    ("macd_fuerte", ((2.0, 1.0, 3.0, 1.0), 4.0)),
    ("rsi_fuerte", ((2.0, 1.0, 1.5, 3.0), 4.0)),
    ("sma21_fuerte", ((2.0, 2.5, 1.5, 1.0), 4.0)),
])


# --- el espacio de reglas booleanas -------------------------------------------
#
# Una regla es una funcion booleana de las 4 condiciones puntuadas (SMA200 queda como
# filtro previo). Se representa como MASCARA de 16 bits: el bit i vale 1 si el estado i
# califica, con  i = 8*sma50 + 4*sma21 + 2*macd + 1*rsi.
#
# Solo se consideran las MONOTONAS: cumplir una condicion mas nunca puede sacar a un
# candidato. Es el unico supuesto, y es el mismo que hace el score de hoy al usar pesos
# no negativos. Son 168 (numero de Dedekind M(4)), 166 sin las dos triviales.
#
# Por que mascaras y no una grilla de pesos: 148 de las 166 son expresables como score
# ponderado + umbral y 18 NO lo son con ningun juego de pesos. Enumerar reglas cubre
# todo el espacio de ponderaciones, agrega esas 18, y no repite decisiones -- una grilla
# de pesos produce cientos de filas que son la misma regla.
#
# EL RANKING NO ES PARTE DE LA MASCARA. Una regla booleana dice quien CALIFICA, no en
# que orden entra, y con el tope de 5 por sector el orden decide en el 42,8% de los
# sector-ruedas. Si se variaran las dos cosas a la vez, ninguna diferencia seria
# atribuible. Por eso en el paso 1 el ranking queda FIJO en el score ponderado de
# scoring.py (que ordena cualquier subconjunto de estados) con desempate alfabetico,
# igual que el bot, y el ranking se trata como una palanca aparte -- su piso de ruido
# lo da el control de sorteo. Ver docs/forward_testing/ANALISIS_ENTRADAS.md sec. 9.

CONDICIONES_PUNTUADAS = ("sma50", "sma21", "macd", "rsi")
_PESO_BIT = (8, 4, 2, 1)
N_ESTADOS = 16


def indice_estado(sma50: bool, sma21: bool, macd: bool, rsi: bool) -> int:
    """Estado binario -> indice 0..15."""
    return (8 * bool(sma50) + 4 * bool(sma21) + 2 * bool(macd) + bool(rsi))


def estado_de_indice(i: int) -> Dict[str, bool]:
    return {k: bool(i & p) for k, p in zip(CONDICIONES_PUNTUADAS, _PESO_BIT)}


def es_monotona(mascara: int) -> bool:
    """True si agregar una condicion cumplida nunca saca al candidato."""
    for i in range(N_ESTADOS):
        if not (mascara >> i) & 1:
            continue
        for p in _PESO_BIT:
            if not i & p and not (mascara >> (i | p)) & 1:
                return False
    return True


def reglas_monotonas(incluir_triviales: bool = False) -> List[int]:
    """Las 166 mascaras monotonas no triviales (168 con 'siempre' y 'nunca')."""
    todas = [m for m in range(1 << N_ESTADOS) if es_monotona(m)]
    if incluir_triviales:
        return todas
    return [m for m in todas if 0 < bin(m).count("1") < N_ESTADOS]


def mascara_desde_pesos(pesos: Sequence[float] = PESOS_V1,
                        umbral: float = SCORE_ENTRADA_V1) -> int:
    """La mascara que produce un score ponderado con su umbral."""
    m = 0
    for i in range(N_ESTADOS):
        cond = dict(estado_de_indice(i), sma200=True)
        s = score_ponderado(cond, pesos)
        if s >= umbral and s > 0:
            m |= 1 << i
    return m


def mascara_v1() -> int:
    """La regla de TECH_SECTOR_v1: SMA50 y al menos 2 de {SMA21, MACD, RSI}."""
    return mascara_desde_pesos()


def es_de_umbral(mascara: int, pesos_posibles: Iterable[float] = (0, 0.5, 1, 1.5, 2,
                                                                  2.5, 3, 4, 5)) -> bool:
    """True si algun score ponderado con umbral produce esta regla. Las que dan False
    son las que ninguna grilla de pesos puede alcanzar."""
    valores = tuple(pesos_posibles)
    for w in product(valores, repeat=4):
        scores = [sum(wi * b for wi, b in zip(w, (
            bool(i & 8), bool(i & 4), bool(i & 2), bool(i & 1))))
            for i in range(N_ESTADOS)]
        for umbral in sorted(set(scores)):
            if umbral <= 0:
                continue
            m = 0
            for i, s in enumerate(scores):
                if s >= umbral:
                    m |= 1 << i
            if m == mascara:
                return True
    return False


def implicantes_primos(mascara: int) -> List[Tuple[str, ...]]:
    """Los conjuntos MINIMOS de condiciones que hacen entrar. Describen la regla sin
    perdida: entra si se cumple alguno de ellos entero."""
    minimos = []
    for i in range(N_ESTADOS):
        if not (mascara >> i) & 1:
            continue
        # minimo = sacar cualquier condicion lo deja afuera
        if all(not (mascara >> (i & ~p)) & 1 for p in _PESO_BIT if i & p):
            minimos.append(tuple(k for k, p in zip(CONDICIONES_PUNTUADAS, _PESO_BIT)
                                 if i & p))
    return minimos


def texto_regla(mascara: int) -> str:
    """Descripcion legible: OR de los implicantes primos. '-' si no entra nunca."""
    primos = implicantes_primos(mascara)
    if not primos:
        return "-"
    if primos == [()]:
        return "siempre"
    return " | ".join("&".join(p.upper() for p in grupo) if grupo else "siempre"
                      for grupo in primos)


def evaluar_mascara(mascara: int, c_sma50, c_sma21, c_macd, c_rsi):
    """Vectorizado: arrays booleanos de condiciones -> array booleano de 'califica'.
    No incluye el filtro SMA200, que se aplica aparte."""
    import numpy as np

    idx = (8 * np.asarray(c_sma50, dtype=np.int8)
           + 4 * np.asarray(c_sma21, dtype=np.int8)
           + 2 * np.asarray(c_macd, dtype=np.int8)
           + np.asarray(c_rsi, dtype=np.int8))
    tabla = np.array([(mascara >> i) & 1 for i in range(N_ESTADOS)], dtype=bool)
    return tabla[idx]


def score_ponderado(cond: Dict[str, bool], pesos: Sequence[float] = PESOS_V1) -> float:
    """Score con pesos arbitrarios. Con el filtro SMA200 fallido, 0."""
    if not cond["sma200"]:
        return 0.0
    w50, w21, wmacd, wrsi = pesos
    total = (w50 * bool(cond["sma50"]) + w21 * bool(cond["sma21"])
             + wmacd * bool(cond["macd"]) + wrsi * bool(cond["rsi"]))
    return round(float(total), 4)


def estados_que_califican(pesos: Sequence[float] = PESOS_V1,
                          umbral: float = SCORE_ENTRADA_V1) -> frozenset:
    """Etiquetas de los estados binarios que entran con (pesos, umbral).

    Es la identidad de la REGLA: dos juegos de pesos distintos con el mismo conjunto
    son la misma estrategia, dia por dia. Sirve para no llenar una grilla de duplicados.
    """
    return frozenset(etiqueta_estado(c) for c in combinaciones_binarias()
                     if c["sma200"] and score_ponderado(c, pesos) >= umbral
                     and score_ponderado(c, pesos) > 0)


def reglas_distintas(juegos: "OrderedDict" = None) -> "OrderedDict[frozenset, List[str]]":
    """Regla -> nombres de los juegos que la producen. Agrupa los duplicados."""
    juegos = JUEGOS_PESOS if juegos is None else juegos
    out = OrderedDict()
    for nombre, (pesos, umbral) in juegos.items():
        out.setdefault(estados_que_califican(pesos, umbral), []).append(nombre)
    return out


# --- control de calidad de los insumos ----------------------------------------

def discrepancias_borde(dist_calculada, dist_guardada, tol: float = 1e-4) -> bool:
    """True si la distancia recomputada y la columna guardada difieren MATERIALMENTE.

    El redondeo a 4 decimales de `dist_sma*` produce diferencias de hasta 5e-5, por
    debajo de `tol`: esta funcion NO las marca, a proposito. Lo que marca es una
    diferencia que el redondeo no explica -- la columna no corresponde a ese close
    (por ejemplo un split corregido en precios_diarios y no propagado).

    El caso de borde que afecta a las zonas es otro y se mide con `cambia_de_zona`.
    """
    if _es_nan(dist_calculada) or _es_nan(dist_guardada):
        return not (_es_nan(dist_calculada) and _es_nan(dist_guardada))
    return abs(float(dist_calculada) - float(dist_guardada)) > tol


def cambia_de_zona(dist_calculada, dist_guardada, cortes: Sequence[float],
                   etiquetas: Sequence[str]) -> bool:
    """True si usar la columna guardada en vez de la distancia recomputada cambia la
    zona. Son las filas con el close pegado a un corte -- sobre todo el corte 0, donde
    la zona y la condicion binaria `close > sma` pueden discrepar.
    """
    return zona(dist_calculada, cortes, etiquetas) != zona(dist_guardada, cortes,
                                                           etiquetas)


def cortes_escalados(cortes: Sequence[float], factor: float) -> Tuple[float, ...]:
    """Cortes de una media escalados por `factor` (criterio a priori de SMA21)."""
    return tuple(round(c / factor, 2) for c in cortes)
