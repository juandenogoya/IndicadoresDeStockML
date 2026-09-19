"""
ft_salidas_smc.py
Regla de SALIDA de FT_SMC_v1 como funcion pura, y la grilla del analisis de salidas
(docs/forward_testing/ANALISIS_SALIDAS.md sec. 10). Sin DB ni pandas.

LA REGLA (scripts/forward_testing/ft_bot_smc.py + ft_scoring.py)
    En cada rueda el bot primero SUBE el stop (nunca lo baja) y despues evalua, en orden:
        P0 balance en la rueda siguiente
        P1 stop: close <= stop. Stop inicial = ultimo swing low de 10 barras a la entrada
           (close / (1 + dist_sl_10_pct/100)); cada rueda sube a ese nivel si es mayor
        P2 CHoCH bajista (choch_bear_10 = 1)
        P3 estructura rota (estructura_10 = -1)
        P4 time stop: 20 dias CORRIDOS desde la entrada
    Sin fila de estructura ese dia el bot solo mira el balance. Sin close, no evalua nada.
    Sale al close.

LA GRILLA (pre-registrada, doc sec. 10.4)
    stop      trail10 (actual) | trail5 (sigue el swing low de 5 barras) | fijo | sin
    choch     si | no
    estructura si | no
    tiempo    10 | 15 | 20 (actual) | 30 | 45 dias corridos | sin
    El balance queda siempre.

Las fechas entran como ENTEROS de dia (ordinal): el time stop cuenta dias corridos.
"""

from collections import namedtuple
from typing import Callable, Dict, Hashable, List, Optional, Sequence

import numpy as np

DIAS_TIME_STOP_V1 = 20
STOPS = ("trail10", "trail5", "fijo", "sin")
TIEMPOS = (10, 15, 20, 30, 45, None)
MOTIVOS = ("BALANCE", "STOP", "CHOCH", "ESTRUCTURA", "TIEMPO")
MAX_POSICIONES_V1 = 5

Regla = namedtuple("Regla", "stop choch estructura tiempo")
REGLA_ACTUAL = Regla("trail10", True, True, DIAS_TIME_STOP_V1)


def swing_low(close: float, dist_pct: float) -> float:
    """Precio del ultimo swing low desde el close y la distancia en %: la formula de
    ft_scoring._swing_low_precio, sin el redondeo a 4 decimales. 0 si no hay dato."""
    if not (dist_pct > 0) or not (close > 0):   # tambien descarta NaN
        return 0.0
    return close / (1.0 + dist_pct / 100.0)


def grilla() -> List[Regla]:
    """Las 96 combinaciones, la regla actual primero."""
    todas = [Regla(s, c, e, t) for s in STOPS for c in (True, False)
             for e in (True, False) for t in TIEMPOS]
    todas.remove(REGLA_ACTUAL)
    return [REGLA_ACTUAL] + todas


def etiqueta(r: Regla) -> str:
    return (f"stop {r.stop:<7} choch {'si' if r.choch else 'no'} estr {'si' if r.estructura else 'no'} "
            f"tiempo {r.tiempo if r.tiempo is not None else 'sin'}")


def stop_inicial(regla: Regla, close_e: float, dist10_e: float, dist5_e: float) -> Optional[float]:
    """Stop al entrar. trail5 arranca en el swing low de 5 barras (si no hay, en el de 10)."""
    if regla.stop == "sin":
        return None
    if regla.stop == "trail5":
        sl = swing_low(close_e, dist5_e)
        if sl > 0:
            return sl
    sl = swing_low(close_e, dist10_e)
    return sl if sl > 0 else None


def primera_salida(dia: Sequence[int], close: Sequence[float], dist10: Sequence[float],
                   dist5: Sequence[float], choch10: Sequence[float], estr10: Sequence[float],
                   datos: Sequence[bool], balance: Sequence[bool], i: int, regla: Regla):
    """(indice, motivo, stop) de la primera rueda posterior a la entrada i en que la
    posicion sale con `regla`, siguiendo el orden del bot. (None, None, stop) si no sale
    dentro de la serie (censura). Version de REFERENCIA, rueda por rueda."""
    sl = stop_inicial(regla, close[i], dist10[i], dist5[i])
    dist = dist5 if regla.stop == "trail5" else dist10
    trailing = regla.stop in ("trail10", "trail5")
    for j in range(i + 1, len(close)):
        c = close[j]
        if not (c == c):
            continue
        if datos[j] and trailing and sl is not None:
            nuevo = swing_low(c, dist[j])
            if nuevo > sl:
                sl = nuevo
        if balance[j]:
            return j, "BALANCE", sl
        if not datos[j]:
            continue
        if sl is not None and c <= sl:
            return j, "STOP", sl
        if regla.choch and choch10[j] == 1:
            return j, "CHOCH", sl
        if regla.estructura and estr10[j] == -1:
            return j, "ESTRUCTURA", sl
        if regla.tiempo is not None and dia[j] - dia[i] >= regla.tiempo:
            return j, "TIEMPO", sl
    return None, None, sl


def salidas_reglas(dia, close, dist10, dist5, choch10, estr10, datos, balance, i: int,
                   reglas: Sequence[Regla]):
    """Lo mismo que primera_salida para muchas reglas a la vez (vectorizado sobre las
    ruedas). Devuelve (indices, motivos): arrays de len(reglas); indice -1 y motivo -1 si
    no sale. motivo = posicion en MOTIVOS. La primera rueda de balance cierra en todas las
    reglas: no hace falta mirar mas alla."""
    close = np.asarray(close, dtype=float)
    datos = np.asarray(datos, dtype=bool)
    balance = np.asarray(balance, dtype=bool)
    j0 = i + 1
    ok_all = ~np.isnan(close[j0:])
    bal_all = balance[j0:] & ok_all
    fin = len(close) if not bal_all.any() else j0 + int(np.argmax(bal_all)) + 1
    c = close[j0:fin]
    ok = ~np.isnan(c)
    dat = datos[j0:fin] & ok
    bal = balance[j0:fin] & ok
    dia_ = np.asarray(dia[j0:fin], dtype=np.int64) - int(dia[i])
    ch = dat & (np.asarray(choch10[j0:fin], dtype=float) == 1)
    es = dat & (np.asarray(estr10[j0:fin], dtype=float) == -1)

    golpe = {}
    for s in STOPS:
        sl0 = stop_inicial(Regla(s, True, True, None), close[i], dist10[i], dist5[i])
        if sl0 is None:
            golpe[s] = np.zeros(len(c), dtype=bool)
            continue
        if s in ("trail10", "trail5"):
            d = np.asarray((dist5 if s == "trail5" else dist10)[j0:fin], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                usa = dat & (d > 0) & (c > 0)
                nuevo = np.where(usa, c / (1.0 + d / 100.0), 0.0)
            camino = np.maximum.accumulate(np.maximum(nuevo, sl0))
        else:
            camino = np.full(len(c), sl0)
        with np.errstate(invalid="ignore"):
            golpe[s] = dat & (c <= camino)
    tiempo = {t: (dat & (dia_ >= t)) if t is not None else np.zeros(len(c), dtype=bool)
              for t in TIEMPOS}

    idx = np.full(len(reglas), -1, dtype=np.int64)
    mot = np.full(len(reglas), -1, dtype=np.int64)
    for k, r in enumerate(reglas):
        t_mask = tiempo[r.tiempo] if r.tiempo in tiempo else (dat & (dia_ >= r.tiempo))
        partes = (bal, golpe[r.stop], ch if r.choch else None, es if r.estructura else None, t_mask)
        cond = bal | golpe[r.stop] | t_mask
        if r.choch:
            cond = cond | ch
        if r.estructura:
            cond = cond | es
        if not cond.any():
            continue
        q = int(np.argmax(cond))
        idx[k] = j0 + q
        for m, p in enumerate(partes):
            if p is not None and p[q]:
                mot[k] = m
                break
    return idx, mot


def entradas_por_ticker(senal: Sequence[bool], desde: int,
                        salida: Callable[[int], Optional[int]]) -> List[int]:
    """Indices de entrada de UN ticker con una posicion a la vez: la primera senal desde
    `desde`; al salir, la primera senal desde la rueda de salida (el bot cierra antes de
    evaluar entradas: puede volver a entrar la misma rueda). salida(i) -> indice de salida o
    None (sin salida: no hay mas entradas)."""
    out, k, n = [], desde, len(senal)
    while k < n:
        if senal[k]:
            out.append(k)
            j = salida(k)
            if j is None:
                break
            k = j
        else:
            k += 1
    return out


def cartera_con_tope(dias: Sequence[int], candidatos: Dict[int, list],
                     salida: Callable[[Hashable], Optional[int]],
                     max_pos: int = MAX_POSICIONES_V1) -> List[Hashable]:
    """Entradas de una cartera con tope de posiciones. candidatos: {dia: [(score, ticker,
    clave), ...]}; entra por score descendente y ticker (desempate). salida(clave) -> dia de
    salida (None = no sale). El dia de salida libera el cupo ese mismo dia (el bot cierra y
    despues evalua entradas). Un ticker con posicion abierta no vuelve a entrar."""
    abiertas: Dict[str, Optional[int]] = {}
    entradas = []
    for d in dias:
        for tk in [t for t, s in abiertas.items() if s is not None and s <= d]:
            del abiertas[tk]
        libres = max_pos - len(abiertas)
        if libres <= 0:
            continue
        for score, tk, clave in sorted(candidatos.get(d, []), key=lambda x: (-x[0], x[1])):
            if libres <= 0:
                break
            if tk in abiertas:
                continue
            abiertas[tk] = salida(clave)
            entradas.append(clave)
            libres -= 1
    return entradas


def familia_ft(motivo: str) -> str:
    """Motivo registrado por el bot -> motivo de la re-simulacion."""
    m = str(motivo or "")
    if m.startswith("EARNINGS"):
        return "BALANCE"
    if m == "TRAILING_SL":
        return "STOP"
    if m == "CHOCH_BEAR":
        return "CHOCH"
    if m == "ESTRUCTURA_ROTA":
        return "ESTRUCTURA"
    if m.startswith("TIME_STOP"):
        return "TIEMPO"
    return m
