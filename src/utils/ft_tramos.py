"""
ft_tramos.py
Medir un cambio en Forward Testing: corta la historia de cada estrategia en
TRAMOS segun los cambios registrados en `ft_cambios` y compara el tramo de
antes contra el de despues. Modulo PURO.

Diseno: docs/forward_testing/METRICAS.md, seccion 12.

PURO (mismo contrato que ft_metricas.py):
    - Sin DB, config, dotenv, archivos ni red. Entran listas y dicts, sale un
      dict. Testeable sin DB.
    - Determinista: el bootstrap usa semilla fija.

CONVENCION DE FECHAS (la regla que hace coherente todo el modulo):
    `fecha_efectiva` de un cambio = la primera rueda de DATOS con la que la
    estrategia decidio ya con el cambio (el `fecha_datos` de la primera corrida
    que lo tuvo, NO el dia del commit ni el de la corrida). Un tramo que va de
    `desde` a `hasta` contiene:
        - retornos:     desde <  fecha      <= hasta
          La decision tomada con el dato de `desde` se ejecuta al cierre de ese
          dia: el primer retorno que produce es el del dia siguiente.
        - operaciones:  desde <= fecha_datos <  hasta, la entrada Y la salida.
          Una operacion que abre antes del corte y cierra despues vivio bajo
          las dos reglas: no se asigna a ningun lado (se informa aparte).
    `None` = sin limite. Dos tramos contiguos comparten el punto de corte.

POR QUE UN GRUPO DE CONTROL Y NO SOLO EL UNIVERSO:
    Las estrategias operan con 55-80% de exposicion y con stops. Contra un
    indice invertido al 100%, cualquier cambio de regimen de mercado parece un
    efecto del cambio: en una caida, tener caja ya "le gana" al indice
    (JOURNAL 2026-07-21 RESULTADO). Las estrategias NO afectadas, en los mismos
    dias, comparten esa estructura y ese mercado. Comparar la diferencia diaria
    contra ellas antes y despues del corte es una diferencia-en-diferencias.

REGLA (METRICAS.md seccion 8): ningun resultado sin su IC95. Si el intervalo
incluye cero es NO CONCLUYENTE; si la muestra no llega al minimo es
INSUFICIENTE y no se publica el numero.
"""

import math
import random

from src.utils.ft_metricas import metricas_trade

# Muestra minima por lado para publicar una comparacion. Con menos, el IC95 de
# una diferencia de medias diarias mide decenas de puntos por mes: el numero
# existe pero no dice nada, y se leeria como si dijera.
MIN_RUEDAS_TRAMO = 20      # ~1 mes de mercado
MIN_OPS_TRAMO = 10

# La diferencia de retorno medio diario se expresa en puntos por MES (x21):
# anualizar tramos de semanas multiplica el ruido (METRICAS.md seccion 9).
RUEDAS_MES = 21

# Operaciones cuya salida es contrafactual (cerradas por un derrumbe que no
# existio, ver JOURNAL 2026-07-21): fuera de las metricas de operacion.
SUFIJOS_CONTRAFACTUALES = ("_SPLIT_FIX",)

# Tipos validos de ft_cambios.tipo (la CHECK de la tabla se arma desde aca).
TIPOS_CAMBIO = ("BUG_FIX", "PARAMETRO", "MODELO", "DATOS", "MEDICION",
                "REFACTOR", "INFRA")

INSUFICIENTE = "INSUFICIENTE"
NO_CONCLUYENTE = "NO CONCLUYENTE"
MEJORA = "MEJORA"
EMPEORA = "EMPEORA"

# t de Student al 97,5% para grados de libertad chicos, donde la expansion de
# Cornish-Fisher de t_critico_95 pierde precision.
_T_CHICOS = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
_Z_975 = 1.959964


# ── Helpers ───────────────────────────────────────────────────────────────────

def _media(xs):
    return sum(xs) / len(xs)


def _varianza(xs):
    """Varianza muestral (ddof=1). Requiere len(xs) >= 2."""
    m = _media(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _en(fecha, desde, hasta, incluye_desde):
    """
    Pertenencia a un tramo. `incluye_desde=False` es la regla de los retornos
    (desde < f <= hasta); `True` la de las decisiones (desde <= f < hasta).
    """
    if desde is not None and (fecha < desde if incluye_desde else fecha <= desde):
        return False
    if hasta is not None and (fecha >= hasta if incluye_desde else fecha > hasta):
        return False
    return True


def t_critico_95(gl):
    """
    Valor critico bilateral al 95% de la t de Student con `gl` grados de
    libertad (Welch los da no enteros). Cornish-Fisher a partir de 5 gl: error
    < 0,005 en gl=10 y exacto a efectos practicos desde gl=20.
    """
    if gl is None or gl <= 0:
        return None
    if gl < 5:
        return _T_CHICOS[max(1, min(4, int(math.floor(gl))))]
    z = _Z_975
    return (z + (z ** 3 + z) / (4 * gl)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * gl ** 2))


# ── Cortes y tramos ───────────────────────────────────────────────────────────

def cortes_de(estrategia_id, cambios):
    """
    Fechas efectivas de los cambios que CORTAN tramo para una estrategia:
    solo los que cambian decisiones y la incluyen. Ordenadas, sin repetir.
    """
    return sorted({c["fecha_efectiva"] for c in cambios
                   if c.get("cambia_decisiones")
                   and estrategia_id in (c.get("estrategias") or ())})


def tramos(fechas, cortes):
    """
    Parte una serie de fechas en tramos por los cortes.

    Un corte en la primera fecha de la serie (o antes) se ignora: la estrategia
    nacio con el cambio ya aplicado. Un corte en la ultima fecha o despues
    abre un tramo con 0 ruedas: el cambio existe pero todavia no produjo
    retornos.

    Devuelve [{desde, hasta, n_ruedas}], con n_ruedas = retornos del tramo.
    """
    if not fechas:
        return []
    fechas = sorted(fechas)
    primera = fechas[0]
    bordes = [c for c in sorted(set(cortes)) if c > primera]
    limites = [None] + bordes + [None]
    out = []
    for desde, hasta in zip(limites, limites[1:]):
        n = sum(1 for f in fechas[1:] if _en(f, desde, hasta, False))
        out.append({"desde": desde, "hasta": hasta, "n_ruedas": n})
    return out


# ── Series de retornos ────────────────────────────────────────────────────────

def retornos_por_fecha(serie):
    """
    [(fecha, valor)] -> {fecha: retorno del dia}. El retorno de la fecha t es
    valor_t / valor_{t-1} - 1; la primera fecha no tiene retorno. Un valor
    None corta la cadena: nunca se calcula un retorno que salte un hueco.
    """
    out = {}
    previo = None
    for fecha, valor in sorted(serie, key=lambda x: x[0]):
        if valor is None:
            previo = None
            continue
        v = float(valor)
        if previo:
            out[fecha] = v / previo - 1.0
        previo = v
    return out


def recortar(retornos, desde, hasta):
    """Los retornos {fecha: r} que caen en el tramo (desde < fecha <= hasta)."""
    return {f: r for f, r in retornos.items() if _en(f, desde, hasta, False)}


def promedio_diario(retornos_por_estrategia):
    """
    {id: {fecha: r}} -> {fecha: promedio de las estrategias con retorno ese dia}.
    Es la cartera de CONTROL: equiponderada entre estrategias, rebalanceada a
    diario.
    """
    acum = {}
    for rets in retornos_por_estrategia.values():
        for f, r in rets.items():
            s, n = acum.get(f, (0.0, 0))
            acum[f] = (s + r, n + 1)
    return {f: s / n for f, (s, n) in acum.items()}


def diferencia_diaria(a, b):
    """{fecha: a - b} en las fechas presentes en las dos series."""
    return {f: a[f] - b[f] for f in a if f in b}


def acumulado_pct(retornos):
    """Retorno compuesto de una secuencia de retornos diarios, en %."""
    eq = 1.0
    for r in retornos:
        eq *= 1.0 + r
    return (eq - 1.0) * 100


# ── Inferencia ────────────────────────────────────────────────────────────────

def comparar_medias(antes, despues, minimo, escala=1.0):
    """
    Diferencia de medias (despues - antes) con IC95 de Welch (varianzas
    distintas, t con los grados de libertad de Welch-Satterthwaite).

    `escala` multiplica medias, diferencia, SE e intervalo (ej. x21x100 para
    pasar un retorno diario a puntos por mes).

    Si alguno de los dos lados tiene menos de `minimo` observaciones el
    veredicto es INSUFICIENTE y los numeros quedan en None: con esa muestra el
    punto no significa nada y publicarlo invita a leerlo.
    """
    n_a, n_d = len(antes), len(despues)
    out = {"n_antes": n_a, "n_despues": n_d, "minimo": minimo,
           "media_antes": None, "media_despues": None, "diferencia": None,
           "se": None, "gl": None, "ic95_lo": None, "ic95_hi": None,
           "veredicto": INSUFICIENTE}
    if n_a < max(minimo, 2) or n_d < max(minimo, 2):
        return out

    ma, md = _media(antes), _media(despues)
    ea, ed = _varianza(antes) / n_a, _varianza(despues) / n_d
    se = math.sqrt(ea + ed)
    den = (ea ** 2 / (n_a - 1)) + (ed ** 2 / (n_d - 1))
    gl = (ea + ed) ** 2 / den if den > 0 else float(n_a + n_d - 2)
    t = t_critico_95(gl)

    dif = md - ma
    lo, hi = dif - t * se, dif + t * se
    if lo > 0:
        veredicto = MEJORA
    elif hi < 0:
        veredicto = EMPEORA
    else:
        veredicto = NO_CONCLUYENTE

    out.update({
        "media_antes": ma * escala, "media_despues": md * escala,
        "diferencia": dif * escala, "se": se * escala, "gl": gl,
        "ic95_lo": lo * escala, "ic95_hi": hi * escala,
        "veredicto": veredicto,
    })
    return out


def comparar_diferencias(antes, despues, antes_ctrl, despues_ctrl, minimo, escala=1.0):
    """
    Diferencia-en-diferencias de medias: (despues - antes) de la estrategia
    menos (despues - antes) del control, con IC95. Los cuatro grupos se toman
    independientes: SE = raiz de la suma de las cuatro varianzas de la media, y
    grados de libertad de Welch-Satterthwaite generalizado.

    Existe para las OPERACIONES. El resultado medio por operacion depende del
    mercado del tramo tanto como de la regla: sin restar lo que le paso al
    control en los mismos tramos, un tramo alcista antes del corte hace
    "empeorar" cualquier cambio. Caso real que lo motivo: en el fix del 29/5,
    TECH_SECTOR_v1 daba EMPEORA por operacion con el universo +6,7% en las 24
    ruedas previas y +0,7% en las 72 posteriores.

    INSUFICIENTE (sin numeros) si cualquiera de los cuatro grupos no llega a
    `minimo`.
    """
    grupos = [antes, despues, antes_ctrl, despues_ctrl]
    out = {"n_antes": len(antes), "n_despues": len(despues),
           "n_antes_control": len(antes_ctrl), "n_despues_control": len(despues_ctrl),
           "minimo": minimo, "diferencia": None, "se": None, "gl": None,
           "ic95_lo": None, "ic95_hi": None, "veredicto": INSUFICIENTE}
    if any(len(g) < max(minimo, 2) for g in grupos):
        return out

    e = [_varianza(g) / len(g) for g in grupos]
    se = math.sqrt(sum(e))
    den = sum(ei ** 2 / (len(g) - 1) for ei, g in zip(e, grupos))
    gl = sum(e) ** 2 / den if den > 0 else float(sum(len(g) for g in grupos) - 4)
    t = t_critico_95(gl)

    dif = ((_media(despues) - _media(antes))
           - (_media(despues_ctrl) - _media(antes_ctrl)))
    lo, hi = dif - t * se, dif + t * se
    if lo > 0:
        veredicto = MEJORA
    elif hi < 0:
        veredicto = EMPEORA
    else:
        veredicto = NO_CONCLUYENTE

    out.update({"diferencia": dif * escala, "se": se * escala, "gl": gl,
                "ic95_lo": lo * escala, "ic95_hi": hi * escala,
                "veredicto": veredicto})
    return out


def ic95_bootstrap(valores, estadistico, n_boot=2000, semilla=20260913):
    """
    IC95 por bootstrap (percentiles, remuestreo iid) de un estadistico sin
    formula cerrada de error estandar, como el Sortino.

    El remuestreo iid ignora la autocorrelacion; en retornos diarios de
    cartera es chica y el sesgo va hacia un intervalo algo mas angosto.

    Devuelve (lo, hi) o None si hay menos de 2 valores o si el estadistico no
    se pudo calcular (devolvio None) en mas del 10% de las remuestras.
    """
    vals = list(valores)
    n = len(vals)
    if n < 2:
        return None
    rng = random.Random(semilla)
    est = []
    for _ in range(n_boot):
        v = estadistico([vals[rng.randrange(n)] for _ in range(n)])
        if v is not None:
            est.append(v)
    if len(est) < 0.9 * n_boot:
        return None
    est.sort()
    k = len(est) - 1
    return est[int(math.floor(0.025 * k))], est[int(math.ceil(0.975 * k))]


# ── Operaciones ───────────────────────────────────────────────────────────────

def es_contrafactual(op):
    motivo = op.get("motivo_salida") or ""
    return any(motivo.endswith(s) for s in SUFIJOS_CONTRAFACTUALES)


def operaciones_de_tramo(operaciones, desde, hasta):
    """
    Operaciones CERRADAS que vivieron enteras en el tramo: entrada y salida con
    fecha de dato en [desde, hasta). Las abiertas no tienen resultado; las que
    cruzan un borde se cuentan con operaciones_que_cruzan().

    Cada operacion es un dict con f_entrada / f_salida (fechas de DATO, no de
    registro), pnl, pnl_pct y motivo_salida.

    Devuelve (operaciones_dentro, n_contrafactuales_excluidas).
    """
    dentro, contraf = [], 0
    for op in operaciones:
        fe, fs = op.get("f_entrada"), op.get("f_salida")
        if fe is None or fs is None:
            continue
        if _en(fe, desde, hasta, True) and _en(fs, desde, hasta, True):
            if es_contrafactual(op):
                contraf += 1
            else:
                dentro.append(op)
    return dentro, contraf


def operaciones_que_cruzan(operaciones, corte):
    """Cerradas que entraron con la regla vieja y salieron con la nueva."""
    return sum(1 for op in operaciones
               if op.get("f_entrada") is not None and op.get("f_salida") is not None
               and op["f_entrada"] < corte <= op["f_salida"])


# ── Evaluacion de un cambio ───────────────────────────────────────────────────

def grupo_control(cambio, estrategias_ids, cambios, desde=None, hasta=None):
    """
    Estrategias de control de un cambio: las NO afectadas y sin un corte PROPIO
    estrictamente dentro de (desde, hasta). Una que cambio sus propias reglas
    en la ventana ya no mide solo el mercado.

    Devuelve (control, excluidas_por_corte_propio), ambas ordenadas.
    """
    afectadas = set(cambio.get("estrategias") or ())
    control, excluidas = [], []
    for eid in sorted(set(estrategias_ids) - afectadas):
        propio = any((desde is None or c > desde) and (hasta is None or c < hasta)
                     for c in cortes_de(eid, cambios))
        (excluidas if propio else control).append(eid)
    return control, excluidas


def _lado(r_s, r_bench, ops, desde, hasta):
    rs = recortar(r_s, desde, hasta)
    comunes = [f for f in rs if f in r_bench]
    ops_in, n_cf = operaciones_de_tramo(ops, desde, hasta)
    return {
        "desde": desde,
        "hasta": hasta,
        "n_ruedas": len(rs),
        "retorno_pct": acumulado_pct(rs[f] for f in sorted(rs)) if rs else None,
        "benchmark_pct": (acumulado_pct(r_bench[f] for f in sorted(comunes))
                          if comunes else None),
        "trade": metricas_trade(ops_in),
        "n_contrafactuales": n_cf,
    }, ops_in


def evaluar_cambio(cambio, cambios, series, benchmark=None, operaciones=None):
    """
    Antes contra despues de un cambio, por cada estrategia afectada.

    Args:
        cambio:      dict de ft_cambios (fecha_efectiva, estrategias,
                     cambia_decisiones, clave, titulo)
        cambios:     todos los cambios registrados (ubican los cortes vecinos)
        series:      {estrategia_id: [(fecha, equity a mercado)]}
        benchmark:   [(fecha, indice)] del universo equiponderado, opcional
        operaciones: [{estrategia_id, f_entrada, f_salida, pnl, pnl_pct,
                       motivo_salida}], opcional

    Para cada afectada, el tramo de ANTES va desde su corte previo (o el inicio)
    hasta el cambio, y el de DESPUES desde el cambio hasta su corte siguiente
    (o hoy). Comparaciones, cada una con su IC95 y veredicto:
        vs_control             diferencia diaria contra la cartera de control,
                               en puntos por mes. La PRINCIPAL.
        vs_universo            idem contra el universo equiponderado.
                               Referencia: incluye el efecto de tener caja.
        expectancy_vs_control  pnl_pct medio por operacion cerrada dentro de
                               cada tramo, en diferencia-en-diferencias contra
                               las operaciones del control en los mismos tramos.
        expectancy             la misma sin control. Descriptiva: mezcla la
                               regla con el mercado del tramo.
    """
    c = cambio["fecha_efectiva"]
    rets = {eid: retornos_por_fecha(s) for eid, s in series.items() if s}
    r_bench = retornos_por_fecha(benchmark) if benchmark else {}
    ops_por = {}
    for op in operaciones or ():
        ops_por.setdefault(op["estrategia_id"], []).append(op)

    ventanas, nacidas = [], []
    for eid in sorted(set(cambio.get("estrategias") or ())):
        if eid not in rets:
            continue
        if c <= min(f for f, _ in series[eid]):
            nacidas.append(eid)
            continue
        cortes = cortes_de(eid, cambios)
        previo = max((x for x in cortes if x < c), default=None)
        siguiente = min((x for x in cortes if x > c), default=None)
        ventanas.append((eid, previo, siguiente))

    out = {"clave": cambio.get("clave"), "titulo": cambio.get("titulo"),
           "fecha_efectiva": c, "estrategias": sorted(cambio.get("estrategias") or ()),
           "nacidas_con_el_cambio": nacidas, "control": [],
           "control_excluidas": [], "filas": []}
    if not ventanas:
        return out

    desde_ctrl = (None if any(v[1] is None for v in ventanas)
                  else min(v[1] for v in ventanas))
    hasta_ctrl = (None if any(v[2] is None for v in ventanas)
                  else max(v[2] for v in ventanas))
    control, excluidas = grupo_control(cambio, rets.keys(), cambios,
                                       desde_ctrl, hasta_ctrl)
    r_ctrl = promedio_diario({k: rets[k] for k in control})
    out["control"], out["control_excluidas"] = control, excluidas

    escala_mes = RUEDAS_MES * 100
    for eid, previo, siguiente in ventanas:
        r_s = rets[eid]
        ops = ops_por.get(eid, [])
        antes, ops_a = _lado(r_s, r_bench, ops, previo, c)
        despues, ops_d = _lado(r_s, r_bench, ops, c, siguiente)

        vs_control = None
        if control:
            act = diferencia_diaria(r_s, r_ctrl)
            vs_control = comparar_medias(list(recortar(act, previo, c).values()),
                                         list(recortar(act, c, siguiente).values()),
                                         MIN_RUEDAS_TRAMO, escala_mes)
        vs_universo = None
        if r_bench:
            act = diferencia_diaria(r_s, r_bench)
            vs_universo = comparar_medias(list(recortar(act, previo, c).values()),
                                          list(recortar(act, c, siguiente).values()),
                                          MIN_RUEDAS_TRAMO, escala_mes)

        pct = lambda xs: [float(o["pnl_pct"]) for o in xs if o.get("pnl_pct") is not None]  # noqa: E731
        expectancy_ctrl = None
        if control:
            ops_ctrl = [o for k in control for o in ops_por.get(k, [])]
            ctrl_a, _ = operaciones_de_tramo(ops_ctrl, previo, c)
            ctrl_d, _ = operaciones_de_tramo(ops_ctrl, c, siguiente)
            expectancy_ctrl = comparar_diferencias(pct(ops_a), pct(ops_d),
                                                   pct(ctrl_a), pct(ctrl_d),
                                                   MIN_OPS_TRAMO)
        out["filas"].append({
            "estrategia_id": eid,
            "antes": antes,
            "despues": despues,
            "vs_control": vs_control,
            "vs_universo": vs_universo,
            "expectancy_vs_control": expectancy_ctrl,
            "expectancy": comparar_medias(pct(ops_a), pct(ops_d), MIN_OPS_TRAMO),
            "n_cruzan": operaciones_que_cruzan(ops, c),
        })
    return out


def tramo_vigente(estrategia_id, cambios, serie, operaciones=None):
    """
    El tramo en curso de una estrategia: desde su ultimo corte (o el inicio),
    cuantas ruedas y operaciones cerradas lleva y cuanto le falta para poder
    compararse. Responde "cuando se va a poder medir el ultimo cambio".
    """
    if not serie:
        return None
    fechas = sorted(f for f, _ in serie)
    corte = max((c for c in cortes_de(estrategia_id, cambios) if c > fechas[0]),
                default=None)
    rs = recortar(retornos_por_fecha(serie), corte, None)
    propias = [o for o in operaciones or () if o.get("estrategia_id") == estrategia_id]
    ops_in, _ = operaciones_de_tramo(propias, corte, None)
    return {
        "estrategia_id": estrategia_id,
        "corte": corte,
        "desde": corte if corte is not None else fechas[0],
        "n_ruedas": len(rs),
        "n_ops": len(ops_in),
        "faltan_ruedas": max(0, MIN_RUEDAS_TRAMO - len(rs)),
        "faltan_ops": max(0, MIN_OPS_TRAMO - len(ops_in)),
    }
