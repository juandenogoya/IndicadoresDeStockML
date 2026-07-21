"""
ft_metricas.py
Metricas de riesgo y rendimiento sobre una serie de equity. Modulo PURO.

Diseno: docs/forward_testing/METRICAS.md

PURO significa:
    - No importa src.data.database, config, ni dotenv.
    - No lee ni escribe DB, archivos ni red.
    - Entra una lista de numeros, sale un dict. Determinista.
    - Por eso es testeable sin DB e importable por el MCP server
      (patron 1 de "Patrones decididos para el MCP server" en CLAUDE.md).

Convenciones:
    - `equity` es la serie MARCADA A MERCADO (ft_equity_diaria.equity),
      en dias habiles consecutivos y sin huecos. Calcular estas metricas
      sobre una serie a costo o de muestreo irregular da resultados
      invalidos: ver METRICAS.md seccion 1.
    - Los retornos son fracciones (0.01 = 1%), no porcentajes.
    - Las funciones devuelven None cuando la muestra no alcanza, nunca
      un numero inventado ni NaN.

REGLA NO NEGOCIABLE (METRICAS.md seccion 8):
    Todo Sharpe/Sortino se reporta con `n` y su IC95%. Si el intervalo
    incluye cero el ratio es NO CONCLUYENTE y no alcanza para justificar
    un cambio de estrategia. Por eso `sharpe()` devuelve el intervalo
    junto al punto, y no solo el punto.
"""

import math

# Dias habiles por anio (NYSE, aproximacion estandar)
DIAS_HABILES_ANIO = 252

# Tasa libre de riesgo anual por defecto (tasa cash 2026).
# No es despreciable: 4% anual son ~0.0155%/dia contra medias diarias
# del orden de 0.14% -> ignorarla infla el ratio ~11%.
RF_ANUAL_DEFAULT = 0.04


# ── Helpers ───────────────────────────────────────────────────────────────────

def _media(xs):
    return sum(xs) / len(xs)


def _desvio(xs, ddof=1):
    """Desvio estandar muestral. None si no hay grados de libertad."""
    n = len(xs)
    if n - ddof < 1:
        return None
    m = _media(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def rf_diaria(rf_anual=RF_ANUAL_DEFAULT):
    """Convierte una tasa anual a su equivalente diaria compuesta."""
    return (1.0 + rf_anual) ** (1.0 / DIAS_HABILES_ANIO) - 1.0


# ── Serie de retornos ─────────────────────────────────────────────────────────

def retornos_desde_equity(equity):
    """
    Retornos diarios simples a partir de la serie de equity.

    La serie DEBE venir en dias habiles consecutivos: cada retorno se asume
    de un (1) dia para poder anualizar con raiz(252). Si la serie tiene
    huecos, ese supuesto se rompe.

    Devuelve una lista de largo len(equity) - 1.
    """
    out = []
    for prev, cur in zip(equity, equity[1:]):
        if prev is None or cur is None or prev == 0:
            continue
        out.append(cur / prev - 1.0)
    return out


# ── Metricas de riesgo ────────────────────────────────────────────────────────

def volatilidad_anual(retornos):
    """Desvio estandar de los retornos diarios, anualizado."""
    sd = _desvio(retornos)
    if sd is None:
        return None
    return sd * math.sqrt(DIAS_HABILES_ANIO)


def max_drawdown(equity):
    """
    Peor caida desde un maximo previo.

    Solo tiene sentido sobre equity a mercado: sobre una serie a costo el
    drawdown de las posiciones abiertas es invisible (METRICAS.md #1).

    Devuelve dict con:
        max_dd_pct        caida maxima en % (positivo, 0 = nunca cayo)
        idx_peak          indice del maximo previo a la peor caida
        idx_valle         indice del valle
        idx_recuperacion  indice donde se recupero el peak (None si sigue abajo)
        duracion_dias     dias entre peak y recuperacion (None si no recupero)
    """
    if not equity or len(equity) < 2:
        return None

    peak = equity[0]
    idx_peak_run = 0
    peor = 0.0
    idx_peak = idx_valle = 0

    for i, v in enumerate(equity):
        if v > peak:
            peak = v
            idx_peak_run = i
        if peak > 0:
            dd = (peak - v) / peak
            if dd > peor:
                peor = dd
                idx_peak = idx_peak_run
                idx_valle = i

    # Recuperacion: primer dia despues del valle que vuelve al peak
    idx_rec = None
    if peor > 0:
        nivel = equity[idx_peak]
        for i in range(idx_valle + 1, len(equity)):
            if equity[i] >= nivel:
                idx_rec = i
                break

    return {
        "max_dd_pct":       round(peor * 100, 4),
        "idx_peak":         idx_peak,
        "idx_valle":        idx_valle,
        "idx_recuperacion": idx_rec,
        "duracion_dias":    (idx_rec - idx_peak) if idx_rec is not None else None,
    }


def sharpe_se(sharpe_anual, n):
    """
    Error estandar del Sharpe anualizado (Lo, 2002).

        SE(SR_anual) = sqrt((1 + SR_diario^2 / 2) / n) * sqrt(252)

    Con n<40 el resultado es de varios puntos de Sharpe: por eso el punto
    solo no sirve para decidir nada.
    """
    if sharpe_anual is None or n < 2:
        return None
    sr_d = sharpe_anual / math.sqrt(DIAS_HABILES_ANIO)
    return math.sqrt((1.0 + 0.5 * sr_d ** 2) / n) * math.sqrt(DIAS_HABILES_ANIO)


def sharpe(retornos, rf_anual=RF_ANUAL_DEFAULT):
    """
    Sharpe anualizado CON su intervalo de confianza.

        (mean(r) - rf_d) / std(r) * sqrt(252)

    Devuelve dict con: sharpe, n, se, ic95_lo, ic95_hi, concluyente.
    `concluyente` es False cuando el IC95% incluye cero -> el ratio no
    distingue la estrategia del azar.
    """
    n = len(retornos)
    sd = _desvio(retornos)
    if sd is None or sd == 0:
        return None

    exceso = _media(retornos) - rf_diaria(rf_anual)
    sr = exceso / sd * math.sqrt(DIAS_HABILES_ANIO)
    se = sharpe_se(sr, n)

    lo = hi = None
    concluyente = False
    if se is not None:
        lo, hi = sr - 1.96 * se, sr + 1.96 * se
        concluyente = (lo * hi) > 0   # ambos del mismo signo => no incluye 0

    return {
        "sharpe":      round(sr, 4),
        "n":           n,
        "se":          round(se, 4) if se is not None else None,
        "ic95_lo":     round(lo, 4) if lo is not None else None,
        "ic95_hi":     round(hi, 4) if hi is not None else None,
        "concluyente": concluyente,
        "rf_anual":    rf_anual,
    }


def sortino(retornos, rf_anual=RF_ANUAL_DEFAULT):
    """
    Sortino anualizado: como Sharpe pero castigando solo la volatilidad
    a la baja.

        downside_dev = sqrt(mean(min(r - rf_d, 0)^2))

    Es el ratio PRIMARIO del FT: las estrategias son long-only con stop,
    o sea asimetricas por diseno. Sharpe penalizaria las subidas fuertes
    que las estrategias de corrida buscan justamente capturar.

    Devuelve None si no hubo ningun dia por debajo de rf (sin downside no
    hay ratio definido).
    """
    n = len(retornos)
    if n < 2:
        return None

    rf_d = rf_diaria(rf_anual)
    downside = [min(r - rf_d, 0.0) ** 2 for r in retornos]
    dd = math.sqrt(sum(downside) / n)
    if dd == 0:
        return None

    sr = (_media(retornos) - rf_d) / dd * math.sqrt(DIAS_HABILES_ANIO)
    return {"sortino": round(sr, 4), "n": n, "rf_anual": rf_anual}


def information_ratio(retornos, retornos_bench):
    """
    Retorno activo por unidad de tracking error.

        mean(r - r_bench) / std(r - r_bench) * sqrt(252)

    Responde la pregunta que importa en long-only: la SELECCION de tickers
    aporta algo por encima de comprar el benchmark?

    Ambas series deben estar alineadas dia a dia y tener el mismo largo.
    """
    if len(retornos) != len(retornos_bench) or len(retornos) < 2:
        return None
    activo = [a - b for a, b in zip(retornos, retornos_bench)]
    sd = _desvio(activo)
    if sd is None or sd == 0:
        return None
    return round(_media(activo) / sd * math.sqrt(DIAS_HABILES_ANIO), 4)


def beta(retornos, retornos_bench):
    """Sensibilidad al benchmark: cov(r, rb) / var(rb). Cuanto es solo mercado."""
    n = len(retornos)
    if n != len(retornos_bench) or n < 2:
        return None
    mr, mb = _media(retornos), _media(retornos_bench)
    var_b = sum((b - mb) ** 2 for b in retornos_bench) / (n - 1)
    if var_b == 0:
        return None
    cov = sum((a - mr) * (b - mb) for a, b in zip(retornos, retornos_bench)) / (n - 1)
    return round(cov / var_b, 4)


# ── Resumen de serie ──────────────────────────────────────────────────────────

def resumen_riesgo(equity, retornos_bench=None, rf_anual=RF_ANUAL_DEFAULT,
                   exposiciones=None):
    """
    Resumen completo sobre una serie de equity a mercado.

    Args:
        equity:         serie de equity en dias habiles consecutivos
        retornos_bench: retornos diarios del benchmark, alineados con los
                        retornos de `equity` (largo len(equity)-1). Opcional.
        rf_anual:       tasa libre de riesgo anual
        exposiciones:   serie de exposicion_pct (fraccion), opcional

    Devuelve dict listo para render. Las claves que no se pueden calcular
    quedan en None (nunca se inventa un valor).
    """
    if not equity or len(equity) < 2:
        return {"n_dias": len(equity) if equity else 0, "insuficiente": True}

    r = retornos_desde_equity(equity)
    dd = max_drawdown(equity)
    sh = sharpe(r, rf_anual)
    so = sortino(r, rf_anual)

    out = {
        "n_dias":            len(equity),
        "n_retornos":        len(r),
        "insuficiente":      False,
        "retorno_total_pct": round((equity[-1] / equity[0] - 1) * 100, 4),
        "volatilidad_anual": (round(volatilidad_anual(r) * 100, 4)
                              if volatilidad_anual(r) is not None else None),
        "max_dd_pct":        dd["max_dd_pct"] if dd else None,
        "dd_duracion_dias":  dd["duracion_dias"] if dd else None,
        "dd_recuperado":     (dd["idx_recuperacion"] is not None) if dd else None,
        "sharpe":            sh,
        "sortino":           so["sortino"] if so else None,
        "rf_anual":          rf_anual,
        "exposicion_media":  (round(_media(exposiciones) * 100, 2)
                              if exposiciones else None),
        "information_ratio": None,
        "beta":              None,
    }

    if retornos_bench is not None and len(retornos_bench) == len(r):
        out["information_ratio"] = information_ratio(r, retornos_bench)
        out["beta"] = beta(r, retornos_bench)
        eq_b = 1.0
        for rb in retornos_bench:
            eq_b *= (1.0 + rb)
        out["benchmark_retorno_pct"] = round((eq_b - 1) * 100, 4)

    return out


# ── Metricas de trade ─────────────────────────────────────────────────────────

def metricas_trade(operaciones):
    """
    Metricas a nivel operacion. Aca esta el n grande: entre 13 y 618
    operaciones por estrategia, contra 28-39 dias de serie. Es donde hay
    senal estadistica real a este horizonte.

    Args:
        operaciones: lista de dicts con al menos `pnl` y `pnl_pct`.
                     Opcionales: `dias`, `r_multiple`, `motivo_salida`.

    Devuelve dict con n, win_rate, expectancy, profit_factor, payoff,
    duracion media y expectancy en R (si viene `r_multiple`).
    """
    ops = [o for o in operaciones if o.get("pnl") is not None]
    n = len(ops)
    if n == 0:
        return {"n": 0}

    pnls = [float(o["pnl"]) for o in ops]
    pcts = [float(o["pnl_pct"]) for o in ops if o.get("pnl_pct") is not None]
    gan = [p for p in pnls if p > 0]
    per = [p for p in pnls if p < 0]

    pct_gan = [float(o["pnl_pct"]) for o in ops
               if o.get("pnl_pct") is not None and float(o["pnl"]) > 0]
    pct_per = [float(o["pnl_pct"]) for o in ops
               if o.get("pnl_pct") is not None and float(o["pnl"]) < 0]

    suma_per = abs(sum(per))
    dias = [o["dias"] for o in ops if o.get("dias") is not None]
    rs = [float(o["r_multiple"]) for o in ops if o.get("r_multiple") is not None]

    return {
        "n":              n,
        "n_ganadoras":    len(gan),
        "n_perdedoras":   len(per),
        "win_rate":       round(len(gan) / n * 100, 2),
        "pnl_total":      round(sum(pnls), 2),
        "expectancy_pct": round(_media(pcts), 4) if pcts else None,
        "expectancy_usd": round(_media(pnls), 2),
        # profit_factor None (no 0 ni inf) cuando no hubo perdidas: el
        # cociente no esta definido y un 0 se leeria al reves.
        "profit_factor":  round(sum(gan) / suma_per, 3) if suma_per > 0 else None,
        "payoff_ratio":   (round(_media(pct_gan) / abs(_media(pct_per)), 3)
                           if pct_gan and pct_per and _media(pct_per) != 0 else None),
        "ganancia_media_pct": round(_media(pct_gan), 4) if pct_gan else None,
        "perdida_media_pct":  round(_media(pct_per), 4) if pct_per else None,
        "duracion_media_dias": round(_media(dias), 1) if dias else None,
        "expectancy_r":   round(_media(rs), 3) if rs else None,
        "n_con_r":        len(rs),
    }


def metricas_por_motivo(operaciones):
    """
    metricas_trade() agrupado por motivo_salida.

    Es el corte que conecta con el analisis de TIME_STOP en SMC_v1
    (JOURNAL 2026-07-18): permite comparar la calidad de cada regla de
    salida entre versiones de una estrategia.
    """
    grupos = {}
    for o in operaciones:
        grupos.setdefault(o.get("motivo_salida") or "SIN_MOTIVO", []).append(o)
    return {k: metricas_trade(v) for k, v in
            sorted(grupos.items(), key=lambda kv: -len(kv[1]))}
