"""
carteras_perfil.py
Motor PURO del backtest de las carteras por perfil de riesgo (pre-registro:
docs/forward_testing/estrategias/CARTERAS_PERFIL.md).

QUE HACE
    1. seleccionar(): top 3 por FAMILIA (Tecnica / SMC / ML) sobre los elegibles
       de la rueda, desempate por SORTEO (nunca por alfabeto), como maximo 1 por
       sector en la familia Tecnica.
    2. por_cartera(): reparte los tickers elegidos en las 4 carteras segun el
       perfil del ticker y los ordena por prioridad (mas familias > mejor puesto >
       sorteo).
    3. simular(): una cartera rueda por rueda -- 15% del capital ACTUAL por compra,
       6 lugares, no entra otro hasta vender, SL/TP sobre el CLOSE, balance
       (transversal) y reentrada desde la rueda siguiente.
    4. tramos() / resumen() / leer(): metricas por tramo anual y la regla de lectura
       pre-registrada (mediana de la seleccion contra mediana del azar).

Sin DB, sin red. La carga de datos y la reconstruccion de los elegibles las hace
scripts/backtesting_historico/bt_carteras_perfil.py. ASCII en los textos.
"""

import random
from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Optional

from src.utils import ft_metricas

# ── Parametros pre-registrados ────────────────────────────────────────────────

CAPITAL_INICIAL = 100_000.0
PCT_COMPRA = 0.15
MAX_POSICIONES = 6
TOP_POR_FAMILIA = 3
MAX_POR_SECTOR = {"Tecnica": 1}       # solo la familia Tecnica tiene tope por sector

CARTERAS = ("Conservadora", "Moderada", "Arriesgada", "Especulativa")
SALIDAS = {                            # (stop loss, take profit) sobre el precio de entrada
    "Conservadora": (0.07, 0.12),
    "Moderada":     (0.07, 0.15),
    "Arriesgada":   (0.15, 0.30),
    "Especulativa": (0.20, 0.50),
}
FAMILIAS = ("Tecnica", "SMC", "ML")

MOTIVO_SL = "STOP_LOSS"
MOTIVO_TP = "TAKE_PROFIT"
MOTIVO_EARN = "EARNINGS_MANANA"
MOTIVO_FIN = "ABIERTA_AL_FINAL"
MOTIVO_TIME = "TIME_STOP"

TRAMOS_MIN_GANADOS = 4                 # de 6 tramos anuales
_EPS = 1e-9


# ── 1. Seleccion ──────────────────────────────────────────────────────────────

def top_familia(elegibles: list, rng: random.Random, n: int = TOP_POR_FAMILIA,
                max_por_sector: Optional[int] = None) -> list:
    """
    elegibles: [(ticker, score, sector)] que cumplen la regla de entrada.
    Devuelve los n mejores por score. Empates: SORTEO -- se baraja con `rng` y
    despues se ordena de forma estable por score, asi el orden entre empatados lo
    decide el sorteo y nunca el orden de entrada (que suele ser alfabetico).
    """
    lista = sorted(elegibles, key=lambda x: x[0])     # base determinista antes de barajar
    rng.shuffle(lista)
    lista.sort(key=lambda x: -x[1])
    out, por_sector = [], {}
    for tk, score, sector in lista:
        if max_por_sector is not None and por_sector.get(sector, 0) >= max_por_sector:
            continue
        out.append(tk)
        por_sector[sector] = por_sector.get(sector, 0) + 1
        if len(out) >= n:
            break
    return out


def seleccionar(elegibles_por_familia: dict, rng: random.Random,
                n: int = TOP_POR_FAMILIA) -> dict:
    """
    elegibles_por_familia: {familia: [(ticker, score, sector)]}.
    Devuelve {ticker: {"familias": set, "puesto": mejor puesto (1..n)}}.
    Las familias se recorren en orden fijo (FAMILIAS) para que la semilla
    reproduzca siempre el mismo resultado.
    """
    out = {}
    for fam in FAMILIAS:
        el = elegibles_por_familia.get(fam)
        if not el:
            continue
        tops = top_familia(el, rng, n, MAX_POR_SECTOR.get(fam))
        for puesto, tk in enumerate(tops, 1):
            d = out.setdefault(tk, {"familias": set(), "puesto": puesto})
            d["familias"].add(fam)
            d["puesto"] = min(d["puesto"], puesto)
    return out


def por_cartera(elegidos: dict, perfil: dict, rng: random.Random) -> dict:
    """
    Reparte los elegidos en las carteras segun `perfil` {ticker: cartera} y los
    ordena por prioridad: mas familias > mejor puesto > sorteo. Un ticker sin
    perfil queda afuera.
    """
    out = {c: [] for c in CARTERAS}
    for tk in sorted(elegidos):
        c = perfil.get(tk)
        if c in out:
            out[c].append(tk)
    for c, lista in out.items():
        clave = {tk: rng.random() for tk in lista}
        lista.sort(key=lambda tk: (-len(elegidos[tk]["familias"]),
                                   elegidos[tk]["puesto"], clave[tk]))
    return out


# ── 2. Simulacion de una cartera ──────────────────────────────────────────────

@dataclass
class Rueda:
    """Lo que la cartera necesita saber de una rueda."""
    fecha: date
    close: dict                 # {ticker: close}
    balance: set                # tickers con balance en esta rueda o la siguiente
    candidatos: list            # tickers ordenados por prioridad para ESTA cartera


@dataclass(frozen=True)
class Salida:
    """Regla de salida. None en sl/tp = no se usa. Por defecto (sl/tp None y
    balance True) simular() toma los SL/TP pre-registrados de la cartera."""
    balance: bool = True              # sale la rueda anterior al balance
    sl: Optional[float] = None
    tp: Optional[float] = None
    time_stop: Optional[int] = None   # ruedas en posicion
    usar_default: bool = True         # True: sl/tp = SALIDAS[cartera]


def simular(ruedas: list, cartera: str, capital: float = CAPITAL_INICIAL,
            pct_compra: float = PCT_COMPRA, max_pos: int = MAX_POSICIONES,
            salida: Optional[Salida] = None) -> dict:
    """
    Recorre las ruedas en orden. En cada una:
      1. SALIDAS de las posiciones abiertas, sobre el close de la rueda:
         balance (prioridad maxima) -> stop loss -> take profit.
      2. EQUITY = cash + posiciones marcadas al close (si un ticker no tiene close
         ese dia se marca con el ultimo conocido).
      3. ENTRADAS en orden de prioridad mientras haya lugar: no si ya esta en
         cartera, no si salio en esta misma rueda, no si tiene balance cerca, no
         sin close. qty = int(pct_compra * equity / close); si no alcanza el cash
         para esa compra completa, no entra.
    `salida` (default: la pre-registrada) permite variar SOLO la regla de cierre;
    la entrada (incluido el bloqueo por balance) no cambia.
    Devuelve {"equity": [(fecha, equity, invertido)], "operaciones": [...]}.
    """
    salida = salida or Salida()
    if salida.usar_default:
        sl_pct, tp_pct = SALIDAS[cartera]
    else:
        sl_pct, tp_pct = salida.sl, salida.tp
    cash = float(capital)
    abiertas = {}            # ticker -> {fecha, precio, qty, ultimo, i}
    ops, curva = [], []

    for i_rueda, r in enumerate(ruedas):
        salieron = set()
        for tk in list(abiertas):
            pos = abiertas[tk]
            c = r.close.get(tk)
            if c is None:
                continue
            pos["ultimo"] = c
            motivo = None
            # variacion contra la entrada, con tolerancia: 100 * 1,12 da
            # 112,00000000000001 y un cierre exacto en +12% no disparaba el TP
            var = c / pos["precio"] - 1
            if salida.balance and tk in r.balance:
                motivo = MOTIVO_EARN
            elif sl_pct is not None and var <= -sl_pct + _EPS:
                motivo = MOTIVO_SL
            elif tp_pct is not None and var >= tp_pct - _EPS:
                motivo = MOTIVO_TP
            elif salida.time_stop is not None and i_rueda - pos["i"] >= salida.time_stop:
                motivo = MOTIVO_TIME
            if motivo:
                cash += pos["qty"] * c
                ops.append(_operacion(tk, pos, r.fecha, c, motivo))
                del abiertas[tk]
                salieron.add(tk)

        invertido = sum(p["qty"] * p["ultimo"] for p in abiertas.values())
        equity = cash + invertido

        for tk in r.candidatos:
            if len(abiertas) >= max_pos:
                break
            if tk in abiertas or tk in salieron or tk in r.balance:
                continue
            c = r.close.get(tk)
            if not c or c <= 0:
                continue
            qty = int(pct_compra * equity / c)
            if qty <= 0 or qty * c > cash:
                continue
            cash -= qty * c
            abiertas[tk] = {"fecha": r.fecha, "precio": c, "qty": qty, "ultimo": c,
                            "i": i_rueda}

        invertido = sum(p["qty"] * p["ultimo"] for p in abiertas.values())
        curva.append((r.fecha, cash + invertido, invertido))

    if ruedas:
        fin = ruedas[-1].fecha
        for tk, pos in abiertas.items():
            ops.append(_operacion(tk, pos, fin, pos["ultimo"], MOTIVO_FIN))
    return {"equity": curva, "operaciones": ops}


def _operacion(tk, pos, fecha_salida, precio_salida, motivo) -> dict:
    pnl = (precio_salida - pos["precio"]) * pos["qty"]
    return {
        "ticker": tk, "fecha_entrada": pos["fecha"], "fecha_salida": fecha_salida,
        "precio_entrada": pos["precio"], "precio_salida": precio_salida,
        "qty": pos["qty"], "pnl": pnl,
        "pnl_pct": (precio_salida / pos["precio"] - 1) * 100,
        "dias": (fecha_salida - pos["fecha"]).days,
        "motivo_salida": motivo,
    }


# ── 3. Control al azar ────────────────────────────────────────────────────────

def candidatos_azar(universo_perfil: list, k: int, rng: random.Random) -> list:
    """k tickers al azar del perfil (sin reposicion). Base ordenada para que la
    semilla reproduzca siempre lo mismo."""
    base = sorted(universo_perfil)
    if k <= 0 or not base:
        return []
    return rng.sample(base, min(k, len(base)))


# ── 4. Metricas y lectura ─────────────────────────────────────────────────────

def tramos(curva: list) -> dict:
    """Retorno por ANIO calendario (el primero y el ultimo parciales): equity del
    cierre del anio contra el del cierre del anio anterior (o el capital inicial)."""
    out, por_anio = {}, {}
    for f, eq, _ in curva:
        por_anio[f.year] = eq
    base = CAPITAL_INICIAL
    for anio in sorted(por_anio):
        fin = por_anio[anio]
        out[anio] = (fin / base - 1) * 100 if base else None
        base = fin
    return out


def resumen(sim: dict) -> dict:
    """Metricas de una corrida."""
    curva = sim["equity"]
    if not curva:
        return {}
    eq = [e for _, e, _ in curva]
    expo = [inv / e for _, e, inv in curva if e]
    rets = ft_metricas.retornos_desde_equity(eq)
    dd = ft_metricas.max_drawdown(eq) or {}
    so = ft_metricas.sortino(rets) or {}
    ops = sim["operaciones"]
    cerradas = [o for o in ops if o["motivo_salida"] != MOTIVO_FIN]
    mt = ft_metricas.metricas_trade(cerradas)
    motivos = {}
    for o in ops:
        motivos[o["motivo_salida"]] = motivos.get(o["motivo_salida"], 0) + 1
    dias = sorted(o["dias"] for o in cerradas)
    pcts = sorted(o["pnl_pct"] for o in cerradas)
    ret_total = (eq[-1] / CAPITAL_INICIAL - 1) * 100
    expo_media = sum(expo) / len(expo) if expo else 0.0
    llenas = sum(1 for _, e, inv in curva if e and inv / e >= PCT_COMPRA * (MAX_POSICIONES - 1))
    return {
        "retorno_total_pct": ret_total,
        "exposicion_media_pct": expo_media * 100,
        "retorno_por_exposicion_pct": ret_total / expo_media if expo_media else None,
        "max_dd_pct": dd.get("max_dd_pct"),
        "sortino": so.get("sortino"),
        "operaciones": len(cerradas),
        "abiertas_al_final": motivos.get(MOTIVO_FIN, 0),
        "win_rate": mt.get("win_rate"),
        "expectancy": mt.get("expectancy"),
        "profit_factor": mt.get("profit_factor"),
        "motivos": motivos,
        "dias_mediana": median(dias) if dias else None,
        "dias_p90": dias[int(0.9 * (len(dias) - 1))] if dias else None,
        "peor_op_pct": pcts[0] if pcts else None,
        "p5_op_pct": pcts[int(0.05 * (len(pcts) - 1))] if pcts else None,
        "ruedas_llena_pct": 100.0 * llenas / len(curva),
        "tramos": tramos(curva),
    }


def _banda(xs: list) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {}
    q = lambda p: xs[int(round(p * (len(xs) - 1)))]
    return {"min": xs[0], "p25": q(0.25), "mediana": median(xs), "p75": q(0.75), "max": xs[-1]}


def leer(res_sel: list, res_azar: list) -> dict:
    """
    Regla de lectura pre-registrada para UNA cartera.
    res_sel / res_azar: listas de resumen() (una por semilla).
    PASA si la mediana de la seleccion gana plata en el total Y le gana a la
    mediana del azar en >= TRAMOS_MIN_GANADOS tramos anuales.
    """
    tot_sel = [r["retorno_total_pct"] for r in res_sel]
    tot_azar = [r["retorno_total_pct"] for r in res_azar]
    anios = sorted({a for r in res_sel for a in r["tramos"]})
    por_tramo, ganados = {}, 0
    for a in anios:
        ms = median([r["tramos"][a] for r in res_sel if r["tramos"].get(a) is not None])
        ma = median([r["tramos"][a] for r in res_azar if r["tramos"].get(a) is not None])
        gana = ms > ma
        ganados += int(gana)
        por_tramo[a] = {"seleccion": ms, "azar": ma, "gana": gana}
    med_sel = median(tot_sel)
    cond1 = med_sel > 0
    cond2 = ganados >= TRAMOS_MIN_GANADOS
    return {
        "pasa": cond1 and cond2,
        "gana_plata": cond1,
        "tramos_ganados": ganados,
        "n_tramos": len(anios),
        "por_tramo": por_tramo,
        "banda_seleccion": _banda(tot_sel),
        "banda_azar": _banda(tot_azar),
        "solape": _solape(tot_sel, tot_azar),
    }


def _solape(a: list, b: list) -> float:
    """Fraccion de pares (seleccion, azar) en que el azar le gana a la seleccion."""
    if not a or not b:
        return None
    return sum(1 for x in a for y in b if y >= x) / (len(a) * len(b))
