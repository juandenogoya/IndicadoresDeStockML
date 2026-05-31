"""
compute_fundamentales_ratios.py
Computa fundamentales_ratios_q a partir de las 4 tablas raw (LOCAL).

Funcion PURA: lee income/balance/cashflow/valuation + sector (de activos),
deriva ratios y crecimiento, y hace UPSERT en fundamentales_ratios_q. NO toca
yahooquery -> se recomputa cuando se quiera (iterar formulas sin re-fetch).

Bases (acordado 31/5/2026):
    - Crecimiento: TRIMESTRAL. QoQ = (Q - Q_{-1})/|Q_{-1}|; YoY = (Q - Q_{-4})/|Q_{-4}|.
    - Rentabilidad / retornos / margenes: TTM (rolling 4Q).

ROIC (formula standard):
    NOPAT_ttm = EBIT_ttm * (1 - tasa_imp),  tasa_imp = clip(tax_ttm/pretax_ttm, 0, 1)
    Capital Invertido = total_debt + stockholders_equity - cash
    roic_ttm = NOPAT_ttm / Capital Invertido
    -> NULL si pretax_ttm <= 0 (no rentable pre-tax: ROIC no significativo) o
       si falta EBIT (bancos / ~22 tickers).

Validacion de escala (defensa ante "valores en miles" / moneda):
    Los ratios son inmunes a escala/moneda. Para detectar un absoluto mal
    escalado se corren 2 cross-checks (solo WARN, no bloquean):
      1. equity:  (market_cap / equity)  ~=  pb_ratio de Yahoo   [solo USD]
      2. shares:  (net_income / share_issued)  ~=  diluted_eps    [cualquiera]
    Si la razon cae fuera de [0.2, 5] (>5x o <1/5) -> WARNING con el ticker.

Uso:
    python scripts/compute_fundamentales_ratios.py
    python scripts/compute_fundamentales_ratios.py --tickers AAPL,BABA,JPM
    python scripts/compute_fundamentales_ratios.py --dry-run
"""

import sys
import os
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from scripts.oneshot.create_fundamentales_tables import (
    get_local_engine, _parse_env_file,
)

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -- Helpers de calculo (sign-safe, division-safe) ----------------------------

def _safe_div(num, den):
    """num/den elemento a elemento; den==0/NaN -> NaN. Acepta Series o None."""
    if num is None or den is None:
        return pd.Series(np.nan, index=getattr(den, "index", None))
    den = pd.Series(den).replace(0, np.nan)
    return pd.Series(num).astype(float) / den.astype(float)


def _growth(s):
    """Devuelve (qoq, yoy) de una serie ordenada asc. (cur-prev)/|prev|."""
    if s is None:
        return None, None
    s = pd.Series(s).astype(float)
    qoq = (s - s.shift(1)) / s.shift(1).abs().replace(0, np.nan)
    yoy = (s - s.shift(4)) / s.shift(4).abs().replace(0, np.nan)
    return qoq, yoy


def _ttm(s, n=len):
    """Rolling 4Q sum, min_periods=4 (NaN si <4 trimestres)."""
    if s is None:
        return None
    return pd.Series(s).astype(float).rolling(window=4, min_periods=4).sum()


# Columnas persistidas (orden = INSERT)
OUT_COLS = [
    "ticker", "fiscal_period_end", "period_type", "reporting_currency",
    "sector", "industry",
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "pe_yoy_pct", "pb_yoy_pct",
    "book_value_per_share", "eps_q", "eps_ttm",
    "roe_ttm", "roa_ttm", "roic_ttm",
    "gross_margin_ttm", "operating_margin_ttm", "net_margin_ttm",
    "net_margin_yoy_delta", "operating_margin_yoy_delta", "opex_to_revenue_ttm",
    "revenue_qoq_pct", "revenue_yoy_pct",
    "net_income_qoq_pct", "net_income_yoy_pct",
    "eps_qoq_pct", "eps_yoy_pct",
    "fcf_ttm", "fcf_margin_ttm", "fcf_qoq_pct", "fcf_yoy_pct",
    "current_ratio", "working_capital", "debt_to_equity",
    "net_debt", "net_debt_to_ebitda_ttm",
    "revenue_ttm", "net_income_ttm", "ebitda_ttm",
    "n_quarters_available",
]
PK_COLS = ["ticker", "fiscal_period_end"]


def _read_raw(engine, tickers):
    where, params = "", None
    if tickers:
        where = "WHERE ticker = ANY(%(tks)s)"
        params = {"tks": tickers}

    def rd(tabla, fecha_col):
        df = pd.read_sql(f"SELECT * FROM {tabla} {where}", engine, params=params)
        if not df.empty:
            df[fecha_col] = pd.to_datetime(df[fecha_col])
        return df

    inc = rd("fundamentales_income_q",    "fiscal_period_end")
    bal = rd("fundamentales_balance_q",   "fiscal_period_end")
    cf  = rd("fundamentales_cashflow_q",  "fiscal_period_end")
    val = rd("fundamentales_valuation_q", "period_end")
    # sector/industry desde activos
    try:
        act = pd.read_sql("SELECT ticker, sector, industry FROM activos", engine)
    except Exception:
        act = pd.DataFrame(columns=["ticker", "sector", "industry"])
    return inc, bal, cf, val, act


def _build_frame(tk, inc, bal, cf, val):
    di = inc[inc["ticker"] == tk].copy()
    if di.empty:
        return None
    db = bal[bal["ticker"] == tk].copy()
    dc = cf[cf["ticker"] == tk].copy()
    dv = val[(val["ticker"] == tk) & (val["period_type"] == "3M")].copy()

    ci = [c for c in ["fiscal_period_end", "reporting_currency", "total_revenue",
                      "gross_profit", "operating_income", "operating_expense",
                      "ebit", "ebitda", "net_income", "pretax_income",
                      "tax_provision", "diluted_eps"] if c in di.columns]
    di = di[ci].rename(columns={"net_income": "ni"})

    cb = [c for c in ["fiscal_period_end", "total_assets", "current_assets",
                      "current_liabilities", "cash_and_equivalents", "total_debt",
                      "stockholders_equity", "share_issued"] if c in db.columns]
    db = db[cb] if cb else pd.DataFrame(columns=["fiscal_period_end"])

    cc = [c for c in ["fiscal_period_end", "free_cash_flow"] if c in dc.columns]
    dc = dc[cc] if cc else pd.DataFrame(columns=["fiscal_period_end"])

    m = di.merge(db, on="fiscal_period_end", how="outer")
    m = m.merge(dc, on="fiscal_period_end", how="outer")

    if not dv.empty:
        cv = [c for c in ["period_end", "pe_ratio", "pb_ratio", "ps_ratio",
                          "enterprise_value_ebitda", "market_cap"] if c in dv.columns]
        dv = dv[cv].rename(columns={"period_end": "fiscal_period_end",
                                    "enterprise_value_ebitda": "ev_ebitda"})
        dv = dv.drop_duplicates(subset=["fiscal_period_end"], keep="last")
        m = m.merge(dv, on="fiscal_period_end", how="left")

    m["ticker"] = tk
    return m.sort_values("fiscal_period_end").reset_index(drop=True)


def _compute(m, warnings):
    o = pd.DataFrame()
    o["ticker"] = m["ticker"]
    o["fiscal_period_end"] = m["fiscal_period_end"].dt.date
    o["period_type"] = "3M"
    o["reporting_currency"] = m.get("reporting_currency")

    g = m.get  # alias
    rev, ni, eps, fcf = g("total_revenue"), g("ni"), g("diluted_eps"), g("free_cash_flow")
    equity, assets = g("stockholders_equity"), g("total_assets")
    debt, cash = g("total_debt"), g("cash_and_equivalents")
    shares = g("share_issued")

    # TTM aggregates
    rev_ttm, ni_ttm, eb_ttm = _ttm(rev), _ttm(ni), _ttm(g("ebitda"))
    gp_ttm, oi_ttm, opex_ttm = _ttm(g("gross_profit")), _ttm(g("operating_income")), _ttm(g("operating_expense"))
    ebit_ttm, tax_ttm, pretax_ttm = _ttm(g("ebit")), _ttm(g("tax_provision")), _ttm(g("pretax_income"))
    fcf_ttm, eps_ttm = _ttm(fcf), _ttm(eps)

    # P1) Mercado vs valor real
    o["pe_ratio"], o["pb_ratio"] = g("pe_ratio"), g("pb_ratio")
    o["ps_ratio"], o["ev_ebitda"] = g("ps_ratio"), g("ev_ebitda")
    _, pe_yoy = _growth(g("pe_ratio")); _, pb_yoy = _growth(g("pb_ratio"))
    o["pe_yoy_pct"], o["pb_yoy_pct"] = pe_yoy, pb_yoy
    o["book_value_per_share"] = _safe_div(equity, shares)
    o["eps_q"], o["eps_ttm"] = eps, eps_ttm

    # P2) Rentabilidad/calidad (TTM) + crecimiento (Q)
    o["roe_ttm"] = _safe_div(ni_ttm, equity)
    o["roa_ttm"] = _safe_div(ni_ttm, assets)
    # ROIC
    tax_rate = _safe_div(tax_ttm, pretax_ttm).clip(0, 1)
    nopat = (pd.Series(ebit_ttm).astype(float) * (1 - tax_rate)) if ebit_ttm is not None else None
    inv_cap = None
    if debt is not None and equity is not None and cash is not None:
        inv_cap = pd.Series(debt).astype(float) + pd.Series(equity).astype(float) - pd.Series(cash).astype(float)
    roic = _safe_div(nopat, inv_cap)
    if pretax_ttm is not None:
        roic = roic.where(pd.Series(pretax_ttm).astype(float) > 0)  # NULL si pretax<=0
    o["roic_ttm"] = roic

    o["gross_margin_ttm"] = _safe_div(gp_ttm, rev_ttm)
    o["operating_margin_ttm"] = _safe_div(oi_ttm, rev_ttm)
    o["net_margin_ttm"] = _safe_div(ni_ttm, rev_ttm)
    o["net_margin_yoy_delta"] = o["net_margin_ttm"] - o["net_margin_ttm"].shift(4)
    o["operating_margin_yoy_delta"] = o["operating_margin_ttm"] - o["operating_margin_ttm"].shift(4)
    o["opex_to_revenue_ttm"] = _safe_div(opex_ttm, rev_ttm)

    rev_qoq, rev_yoy = _growth(rev); ni_qoq, ni_yoy = _growth(ni); eps_qoq, eps_yoy = _growth(eps)
    o["revenue_qoq_pct"], o["revenue_yoy_pct"] = rev_qoq, rev_yoy
    o["net_income_qoq_pct"], o["net_income_yoy_pct"] = ni_qoq, ni_yoy
    o["eps_qoq_pct"], o["eps_yoy_pct"] = eps_qoq, eps_yoy

    # P3) Cash (FCF)
    o["fcf_ttm"] = fcf_ttm
    o["fcf_margin_ttm"] = _safe_div(fcf_ttm, rev_ttm)
    fcf_qoq, fcf_yoy = _growth(fcf)
    o["fcf_qoq_pct"], o["fcf_yoy_pct"] = fcf_qoq, fcf_yoy

    # Estructura / solvencia
    o["current_ratio"] = _safe_div(g("current_assets"), g("current_liabilities"))
    ca, cl = g("current_assets"), g("current_liabilities")
    o["working_capital"] = (pd.Series(ca).astype(float) - pd.Series(cl).astype(float)) if (ca is not None and cl is not None) else np.nan
    o["debt_to_equity"] = _safe_div(debt, equity)
    net_debt = (pd.Series(debt).astype(float) - pd.Series(cash).astype(float)) if (debt is not None and cash is not None) else None
    o["net_debt"] = net_debt if net_debt is not None else np.nan
    o["net_debt_to_ebitda_ttm"] = _safe_div(net_debt, eb_ttm)

    # Agregados TTM crudos
    o["revenue_ttm"], o["net_income_ttm"], o["ebitda_ttm"] = rev_ttm, ni_ttm, eb_ttm
    o["n_quarters_available"] = len(m)

    # -- Validacion de escala (solo ultima fila, WARN) --
    _validate_scale(m, warnings)
    return o


def _validate_scale(m, warnings):
    """Cross-checks de escala sobre la ultima fila del ticker."""
    last = m.iloc[-1]
    tk = last["ticker"]
    cur = last.get("reporting_currency")
    # 1) equity scale (solo USD): market_cap/equity ~= pb_ratio
    mc, eq, pb = last.get("market_cap"), last.get("stockholders_equity"), last.get("pb_ratio")
    if cur == "USD" and mc and eq and pb and eq != 0 and pb != 0:
        ratio = (mc / eq) / pb
        if ratio < 0.2 or ratio > 5:
            warnings.append(f"{tk}: escala EQUITY sospechosa (mc/eq vs pb = {ratio:.2f}x)")
    # 2) shares scale: net_income/shares ~= diluted_eps
    ni, sh, eps = last.get("ni"), last.get("share_issued"), last.get("diluted_eps")
    if ni and sh and eps and sh != 0 and eps != 0:
        ratio = (ni / sh) / eps
        if ratio < 0.2 or ratio > 5:
            warnings.append(f"{tk}: escala SHARES sospechosa (ni/sh vs eps = {ratio:.2f}x)")


def _attach_sector(rows_df, act):
    if act.empty:
        rows_df["sector"] = None
        rows_df["industry"] = None
        return rows_df
    amap = act.set_index("ticker")
    rows_df["sector"] = rows_df["ticker"].map(amap["sector"])
    rows_df["industry"] = rows_df["ticker"].map(amap["industry"])
    return rows_df


def _to_records(df):
    df = df[OUT_COLS].copy().replace({np.nan: None})
    clean = []
    for r in df.to_dict("records"):
        cr = {}
        for k, v in r.items():
            if v is None:
                cr[k] = None
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                cr[k] = None
            elif hasattr(v, "item"):
                cr[k] = v.item()
            else:
                cr[k] = v
        clean.append(cr)
    return clean


def _upsert(env, rows):
    if not rows:
        return 0
    cols = OUT_COLS
    ph = ", ".join([f"%({c})s" for c in cols])
    upd = [c for c in cols if c not in PK_COLS]
    set_clause = ", ".join([f"{c}=EXCLUDED.{c}" for c in upd]) + ", computed_at=NOW()"
    sql = (f"INSERT INTO fundamentales_ratios_q ({', '.join(cols)}) VALUES ({ph}) "
           f"ON CONFLICT (ticker, fiscal_period_end) DO UPDATE SET {set_clause}")
    conn = psycopg2.connect(
        host=env.get("DB_HOST", "localhost"), port=int(env.get("DB_PORT", 5432)),
        dbname=env.get("DB_NAME", "activos_ml"), user=env.get("DB_USER", "postgres"),
        password=env.get("DB_PASSWORD", ""),
    )
    try:
        cur = conn.cursor()
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Computa fundamentales_ratios_q (local)")
    parser.add_argument("--tickers", default=None, help="CSV de tickers (default: todos)")
    parser.add_argument("--dry-run", action="store_true", help="Calcula sin escribir")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    print()
    print(SEP)
    print(f"  COMPUTE fundamentales_ratios_q"
          f"{'  [tickers='+','.join(tickers)+']' if tickers else '  [todos]'}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    eng = get_local_engine()
    log("Leyendo tablas raw + activos...")
    inc, bal, cf, val, act = _read_raw(eng, tickers)
    universo = sorted(inc["ticker"].unique().tolist()) if not inc.empty else []
    log(f"  income={len(inc)} balance={len(bal)} cashflow={len(cf)} "
        f"valuation={len(val)} | activos={len(act)} | tickers={len(universo)}")

    if not universo:
        log("Sin datos. Abortando.")
        sys.exit(1)

    all_rows, sin_data, warnings = [], [], []
    for tk in universo:
        frame = _build_frame(tk, inc, bal, cf, val)
        if frame is None or frame.empty:
            sin_data.append(tk)
            continue
        ratios = _compute(frame, warnings)
        all_rows.extend(_to_records(_attach_sector(ratios, act)))

    log(f"Filas calculadas: {len(all_rows)} ({len(universo)-len(sin_data)} tickers)")
    if warnings:
        log(f"AVISOS de escala ({len(warnings)}):")
        for w in warnings:
            log(f"  WARN {w}")
    else:
        log("Validacion de escala: OK (sin avisos).")

    if args.dry_run:
        if all_rows:
            last = all_rows[-1]
            prev = {k: last[k] for k in ["ticker", "fiscal_period_end", "sector",
                    "revenue_yoy_pct", "net_margin_ttm", "roe_ttm", "roic_ttm",
                    "pe_ratio", "pb_ratio", "book_value_per_share", "eps_ttm",
                    "fcf_ttm", "current_ratio", "n_quarters_available"]}
            log(f"DRY-RUN muestra ultima fila: {prev}")
    else:
        env = _parse_env_file(os.path.join(ROOT, ".env"))
        n = _upsert(env, all_rows)
        log(f"UPSERT local: {n} filas.")

    print()
    print(SEP)
    print(f"  OK  |  tickers: {len(universo)-len(sin_data)}  |  filas: {len(all_rows)}"
          f"  |  sin data: {len(sin_data)}  |  avisos escala: {len(warnings)}")
    if sin_data:
        print(f"  sin data: {sin_data}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
