"""
compute_multiplos_px.py
Recalcula los multiplos de PRECIO (PER/P-B/P-S/EV-EBITDA) de todos los tickers
con el CIERRE del dia y los persiste en las columnas *_px de fundamentales_ratios_q
(ultimo Q de cada ticker). Los pe_ratio/etc. originales de Yahoo NO se tocan.

Motivo: el multiplo de Yahoo congela el precio en la fecha del balance; el precio
cambia a diario. Ver src/utils/multiplos_px.py. Este recompute es DB->local (sin
Yahoo) y corre a diario, enganchado al final de recovery_incremental (target local).

Numerador = cierre de hoy; denominadores TTM = ultimo Q (se mantienen el trimestre).
shares = OrdinarySharesNumber del ultimo balance (fallback: net_income_ttm/eps_ttm).

Uso:
    from scripts.compute_multiplos_px import compute_multiplos_px
    n = compute_multiplos_px(engine)            # actualiza *_px
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import psycopg2.extras
from sqlalchemy import text

from src.data.database import get_engine, get_connection, query_df
from src.utils.multiplos_px import recalcular_multiplos, _f


def _shares_de_balance(raw_json) -> float | None:
    """OrdinarySharesNumber (fallback ShareIssued) del raw_json del balance."""
    rj = raw_json
    if isinstance(rj, str):
        try:
            rj = json.loads(rj)
        except Exception:
            return None
    if not isinstance(rj, dict):
        return None
    return _f(rj.get("OrdinarySharesNumber")) or _f(rj.get("ShareIssued"))


_SQL_UPDATE = """
    UPDATE fundamentales_ratios_q SET
        pe_ratio_px  = %(pe)s,
        pb_ratio_px  = %(pb)s,
        ps_ratio_px  = %(ps)s,
        ev_ebitda_px = %(ev)s,
        precio_px    = %(px)s,
        fecha_px     = %(fpx)s,
        shares_out   = %(sh)s
    WHERE ticker = %(t)s AND fiscal_period_end = %(fpe)s
"""


def compute_multiplos_px(engine=None) -> int:
    """
    Recalcula y persiste *_px para el ultimo Q de cada ticker. Retorna numero de
    filas actualizadas. Funcion idempotente (re-correr pisa con el cierre del dia).
    """
    eng = engine or get_engine()

    # 1. Ultimo Q de cada ticker con sus denominadores TTM.
    ratios = query_df("""
        SELECT DISTINCT ON (ticker)
               ticker, fiscal_period_end, eps_ttm, book_value_per_share,
               revenue_ttm, ebitda_ttm, net_debt, net_income_ttm,
               reporting_currency
        FROM   fundamentales_ratios_q
        ORDER  BY ticker, fiscal_period_end DESC
    """)
    if ratios.empty:
        return 0

    # 2. Cierre mas reciente por ticker (precios_diarios LOCAL).
    precios = query_df("""
        SELECT DISTINCT ON (ticker) ticker, fecha AS fecha_px, close
        FROM   precios_diarios
        ORDER  BY ticker, fecha DESC
    """)
    px_map = {r["ticker"]: (r["close"], r["fecha_px"]) for _, r in precios.iterrows()}

    # 3. Shares del ultimo balance por ticker (para P/S y EV/EBITDA).
    bal = query_df("""
        SELECT DISTINCT ON (ticker) ticker, raw_json
        FROM   fundamentales_balance_q
        ORDER  BY ticker, fiscal_period_end DESC
    """)
    sh_map = {r["ticker"]: _shares_de_balance(r["raw_json"]) for _, r in bal.iterrows()}

    registros = []
    for _, r in ratios.iterrows():
        tk = r["ticker"]
        close, fecha_px = px_map.get(tk, (None, None))
        if close is None:
            continue
        shares = sh_map.get(tk)
        if shares is None:  # fallback: derivar de NI_ttm / eps_ttm
            shares = _f(r["net_income_ttm"]) / _f(r["eps_ttm"]) if (
                _f(r["eps_ttm"]) not in (None, 0) and _f(r["net_income_ttm"]) is not None) else None

        m = recalcular_multiplos(
            close, r["eps_ttm"], r["book_value_per_share"], shares,
            r["revenue_ttm"], r["ebitda_ttm"], r["net_debt"],
            reporting_currency=(r["reporting_currency"] or "USD"),
        )
        registros.append({
            "pe": m["pe_ratio_px"], "pb": m["pb_ratio_px"],
            "ps": m["ps_ratio_px"], "ev": m["ev_ebitda_px"],
            "px": _f(close), "fpx": fecha_px,
            "sh": round(shares, 2) if shares is not None else None,
            "t": tk, "fpe": r["fiscal_period_end"],
        })

    if not registros:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, _SQL_UPDATE, registros, page_size=200)

    return len(registros)


if __name__ == "__main__":
    os.environ.pop("DATABASE_URL", None)  # forzar local
    n = compute_multiplos_px()
    print(f"OK: multiplos *_px recalculados para {n} tickers (ultimo Q, cierre del dia).")
