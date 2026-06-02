"""
multiplos_px.py
Recalcula los multiplos de PRECIO (PER, P/B, P/S, EV/EBITDA) con el cierre del
dia, manteniendo los denominadores TTM del ultimo balance.

Motivo (2/6/2026): los multiplos que trae yahooquery (fundamentales_valuation_q)
son una FOTO del Q -- el precio quedo congelado en la fecha del balance. El
denominador (TTM) si se mantiene todo el trimestre (es correcto), pero el
numerador (precio) cambia todos los dias. Resultado: un PER de Yahoo puede decir
51.7 cuando al precio de hoy es 39.8. Estos multiplos "_px" usan el cierre actual
para que reflejen la valuacion AL MOMENTO del analisis.

Funcion PURA: sin DB, sin side effects. ASCII en strings (regla del proyecto).

Formulas (P = cierre del dia; denominadores TTM del ultimo Q):
    PER        = P / eps_ttm
    P/B        = P / book_value_per_share
    P/S        = (P * shares) / revenue_ttm      [market_cap = P * shares]
    EV/EBITDA  = (P * shares + net_debt) / ebitda_ttm
"""

from typing import Optional


def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # NaN != NaN
    return None if f != f else f


def _safe_div(num, den) -> Optional[float]:
    n, d = _f(num), _f(den)
    if n is None or d is None or d == 0:
        return None
    return n / d


def recalcular_multiplos(cierre, eps_ttm, book_value_per_share, shares,
                         revenue_ttm, ebitda_ttm, net_debt) -> dict:
    """
    Devuelve {pe_ratio_px, pb_ratio_px, ps_ratio_px, ev_ebitda_px} con el cierre
    actual. Cada uno None si falta su denominador o es <= 0 (no se fuerza un valor).

    Convenciones (homogeneas con compute_fundamentales_ratios / Yahoo):
      - PER y P/S NULL si el denominador (eps_ttm / revenue_ttm) <= 0
        (multiplo sin sentido con base negativa).
      - P/B NULL si bvps <= 0 (patrimonio negativo).
      - EV/EBITDA NULL si ebitda_ttm <= 0.
      - net_debt puede ser negativo (caja neta): resta del EV, valido.
    """
    P = _f(cierre)
    if P is None or P <= 0:
        return {"pe_ratio_px": None, "pb_ratio_px": None,
                "ps_ratio_px": None, "ev_ebitda_px": None}

    eps = _f(eps_ttm)
    bvps = _f(book_value_per_share)
    sh = _f(shares)
    rev = _f(revenue_ttm)
    ebitda = _f(ebitda_ttm)
    nd = _f(net_debt)

    pe = _safe_div(P, eps) if (eps is not None and eps > 0) else None
    pb = _safe_div(P, bvps) if (bvps is not None and bvps > 0) else None

    market_cap = (P * sh) if sh is not None else None
    ps = _safe_div(market_cap, rev) if (rev is not None and rev > 0 and market_cap is not None) else None

    ev_ebitda = None
    if market_cap is not None and ebitda is not None and ebitda > 0 and nd is not None:
        ev_ebitda = (market_cap + nd) / ebitda

    return {
        "pe_ratio_px":  round(pe, 4) if pe is not None else None,
        "pb_ratio_px":  round(pb, 4) if pb is not None else None,
        "ps_ratio_px":  round(ps, 4) if ps is not None else None,
        "ev_ebitda_px": round(ev_ebitda, 4) if ev_ebitda is not None else None,
    }
