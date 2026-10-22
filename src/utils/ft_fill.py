"""
ft_fill.py -- Modulo PURO: eran ALCANZABLES los precios con los que FT registro
sus operaciones?

Sin DB, sin config, sin side effects. Lo usa
scripts/forward_testing/ft_analisis_fill.py.
Metodo, pre-registro y resultados: docs/forward_testing/ANALISIS_FILL.md.

QUE RESPONDE
    FT registra entrada y salida al close de `fecha_datos`, la ultima rueda que
    precios_diarios tenia cuando corrio el bot. Ese close YA HABIA OCURRIDO al
    decidir. Este modulo re-valua cada operacion al primer precio que se podia
    conseguir de verdad y mide la diferencia PAREADA contra el fill registrado.

    No evalua ninguna estrategia. Mide si lo ya medido se puede leer.

LAS DOS PATAS SE MUEVEN JUNTAS
    Re-valuar solo la salida mide un artefacto que no existe: si se compra peor y
    se vende peor el efecto se cancela en parte, y cuanto se cancela ES el
    resultado. `revaluar` exige las dos patas o devuelve None.

SE TRABAJA EN RETORNO, NO EN DOLARES
    Si cambia el precio de entrada, la cantidad de acciones comprables con el
    mismo capital tambien cambia; mezclar eso seria medir el sizing. El efecto en
    dolares queda FUERA DE ALCANCE a proposito.

LA UNIDAD DE INDEPENDENCIA ES LA RUEDA
    Las salidas del mismo dia comparten mercado. Los IC95 salen de
    ft_salidas.ic95_por_dia, el mismo helper que los analisis anteriores: una
    sola definicion de "IC con el dia como unidad" en el proyecto.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence

from src.utils.ft_salidas import ic95_por_dia

# Variantes de precio de fill. "actual" es el control (lo que FT registro).
VARIANTES = ("actual", "apertura_sig", "cierre_sig")
PRINCIPAL = "apertura_sig"
CONTROL = "actual"

# Salidas que NO las decidio una regla: se excluyen del analisis.
MOTIVOS_EXCLUIDOS = ("ESTRATEGIA_DISCONTINUADA", "DELISTING")

# Regla de lectura PRE-REGISTRADA (doc sec. 2).
FRACCION_MATERIAL = 0.25      # la diferencia se come >= 25% del retorno del control
MESES_MINIMOS = 4             # el signo se repite en >= 4 de los 6 meses
MESES_TOTALES = 6


def retorno(precio_entrada: Optional[float],
            precio_salida: Optional[float]) -> Optional[float]:
    """Retorno de una operacion long, en porcentaje. None si falta un precio o la
    entrada no es positiva (nunca se asume un numero donde no hay dato)."""
    if precio_entrada is None or precio_salida is None:
        return None
    pe, ps = float(precio_entrada), float(precio_salida)
    if not (pe > 0) or math.isnan(pe) or math.isnan(ps):
        return None
    return (ps - pe) / pe * 100.0


def precio_de(fills: dict, pata: str, variante: str) -> Optional[float]:
    """Precio de una pata ('entrada'|'salida') bajo una variante.

    `fills` es lo que arma el script por operacion:
        {"entrada": {"actual": x, "apertura_sig": y, "cierre_sig": z},
         "salida":  {...}}
    """
    if variante not in VARIANTES:
        raise ValueError("variante desconocida: %r" % (variante,))
    por_pata = fills.get(pata) or {}
    v = por_pata.get(variante)
    if v is None:
        return None
    v = float(v)
    return None if (math.isnan(v) or v <= 0) else v


def revaluar(fills: dict, variante: str) -> Optional[float]:
    """Retorno de la operacion con AMBAS patas en la misma variante.
    None si falta cualquiera de las dos: no se mezclan variantes entre patas."""
    pe = precio_de(fills, "entrada", variante)
    ps = precio_de(fills, "salida", variante)
    if pe is None or ps is None:
        return None
    return retorno(pe, ps)


def comparar_operacion(fills: dict, variante: str) -> Optional[dict]:
    """Retornos del control y de la variante, y su diferencia PAREADA.
    None si alguno de los dos no se puede calcular (la operacion no entra)."""
    base = revaluar(fills, CONTROL)
    alt = revaluar(fills, variante)
    if base is None or alt is None:
        return None
    return {"actual": base, "variante": alt, "dif": alt - base}


def _media(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else float("nan")


def _percentil(xs: Sequence[float], q: float) -> float:
    """Percentil por interpolacion lineal (mismo criterio que ft_salidas)."""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def resumen_variante(difs: Sequence[float], ruedas: Sequence,
                     ret_actual: Sequence[float],
                     ret_variante: Sequence[float]) -> dict:
    """Metricas de una variante contra el control, con la RUEDA como unidad."""
    ic = ic95_por_dia(difs, ruedas)
    return {
        "ops": ic["ops"],
        "ruedas": ic["dias"],
        "dif_media": ic["media"],            # media de medias por rueda
        "dif_media_ops": ic["media_ops"],
        "ic95_lo": ic["lo"],
        "ic95_hi": ic["hi"],
        "ret_actual": _media(ret_actual),
        "ret_variante": _media(ret_variante),
        "ganadoras_actual": _tasa_positiva(ret_actual),
        "ganadoras_variante": _tasa_positiva(ret_variante),
        "p5_actual": _percentil(ret_actual, 0.05),
        "p5_variante": _percentil(ret_variante, 0.05),
    }


def _tasa_positiva(xs: Sequence[float]) -> float:
    return (sum(1 for x in xs if x > 0) / len(xs) * 100.0) if xs else float("nan")


def signo_por_mes(difs: Iterable[float], meses: Iterable[str]) -> Dict[str, float]:
    """Diferencia media por mes (clave 'AAAA-MM'), para la consistencia."""
    acum: Dict[str, List[float]] = {}
    for d, m in zip(difs, meses):
        if d is None or (isinstance(d, float) and math.isnan(d)):
            continue
        acum.setdefault(str(m), []).append(float(d))
    return {m: _media(v) for m, v in sorted(acum.items())}


def meses_al_mismo_lado(por_mes: Dict[str, float], signo: float) -> int:
    """Cuantos meses van al mismo lado que `signo` (0 no cuenta para ninguno)."""
    if signo == 0 or math.isnan(signo):
        return 0
    return sum(1 for v in por_mes.values()
               if not math.isnan(v) and v != 0 and (v > 0) == (signo > 0))


def da_vuelta_el_signo(totales_actual: Dict[str, float],
                       totales_variante: Dict[str, float]) -> List[str]:
    """Estrategias cuyo retorno TOTAL cambia de signo con la variante."""
    vuelta = []
    for k, a in totales_actual.items():
        b = totales_variante.get(k)
        if b is None or math.isnan(a) or math.isnan(b):
            continue
        if a != 0 and b != 0 and (a > 0) != (b > 0):
            vuelta.append(k)
    return sorted(vuelta)


def leer(res: dict, por_mes: Dict[str, float], signos_dados_vuelta: Sequence[str],
         fraccion: float = FRACCION_MATERIAL,
         meses_minimos: int = MESES_MINIMOS) -> dict:
    """Aplica la REGLA DE LECTURA pre-registrada (doc sec. 2). Las tres o nada.

    1. IC95 de la diferencia pareada excluye el cero;
    2. |dif| >= `fraccion` del retorno medio del control, O da vuelta el signo
       del total de alguna estrategia;
    3. el signo se repite en >= `meses_minimos` meses.
    """
    lo, hi = res.get("ic95_lo"), res.get("ic95_hi")
    excluye_cero = bool(
        lo is not None and hi is not None
        and not math.isnan(lo) and not math.isnan(hi)
        and ((lo > 0 and hi > 0) or (lo < 0 and hi < 0))
    )

    dif = res.get("dif_media", float("nan"))
    base = res.get("ret_actual", float("nan"))
    umbral = abs(base) * fraccion if not math.isnan(base) else float("nan")
    por_magnitud = bool(not math.isnan(dif) and not math.isnan(umbral)
                        and abs(dif) >= umbral)
    por_signo = len(signos_dados_vuelta) > 0
    material = por_magnitud or por_signo

    n_meses = meses_al_mismo_lado(por_mes, dif)
    consistente = n_meses >= meses_minimos

    return {
        "ic95_excluye_cero": excluye_cero,
        "umbral_material": umbral,
        "material_por_magnitud": por_magnitud,
        "material_por_signo": por_signo,
        "material": material,
        "meses_al_mismo_lado": n_meses,
        "meses_total": len(por_mes),
        "consistente": consistente,
        "pasa": bool(excluye_cero and material and consistente),
    }
