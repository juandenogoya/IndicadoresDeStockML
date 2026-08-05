"""
compute_fundamentales_ratios.py
Computa fundamentales_ratios_q a partir de las 4 tablas raw (LOCAL). v2.

Funcion PURA: lee income/balance/cashflow/valuation + sector (de activos),
clasifica el PERFIL de cada ticker (financiero / no_financiero), deriva ratios
segun el perfil, y hace UPSERT. NO toca yahooquery -> recomputable sin re-fetch.

Diseno v2 (ver docs/fundamentales_calculo.md):
    Validacion contra balances oficiales (MU/XP/JPM) mostro que la formula
    unica de v1 fallaba en FINANCIERAS. v2 aplica DOS perfiles + 3 fixes:
      B1: financieras -> margenes industriales/ROIC/liquidez/WC/FCF = NULL.
      B2: as-of join de balance (balance mas reciente con fecha <= income) ->
          ROE/ROA/BVPS se computan aunque el balance del Q exacto no exista.
      B3: usar CommonStockEquity (no StockholdersEquity) y
          NetIncomeCommonStockholders en ROE/BVPS/ROIC -> corrige preferentes.

Perfiles (estructura contable SOSTENIDA, no sector nominal):
    - Regla auto multi-Q: gross_ratio<0.3 AND opinc_ratio<0.3 AND nii_ratio>0.7
      -> financiero. (los 18 bancos/aseguradoras dan gross 0/N y opinc 0/N.)
    - Override manual (FINANCIERO_OVERRIDE): hibridos que la regla no captura
      (XP: tiene gross/opinc pero su op income es basura -> financiero pleno).

Bases:
    - Crecimiento: TRIMESTRAL. QoQ vs Q-1, YoY vs Q-4 ((cur-prev)/|prev|).
    - Rentabilidad/retornos/margenes: TTM (rolling 4Q).

Set bancario (perfil financiero):
    - rotce_ttm = NetIncomeCommonStockholders_ttm / TangibleBookValue (solido).
    - efficiency_ratio_ttm = (TotalRevenue_ttm - PretaxIncome_ttm)/TotalRevenue_ttm
      (aproximado). NIM descartado (yahooquery no expone cartera de prestamos).

Uso:
    python scripts/compute_fundamentales_ratios.py
    python scripts/compute_fundamentales_ratios.py --tickers AAPL,XP,JPM
    python scripts/compute_fundamentales_ratios.py --dry-run
"""

import sys
import os
import argparse
import json
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

# -- Clasificacion de perfil (ver docs/fundamentales_calculo.md seccion 4) -----
# Override manual = LISTA CURADA de financieros (la fuente de verdad, doc 4.3).
# Se FIJAN los 18 explicitamente (17 bancos/aseguradoras + XP) en vez de confiar
# en la regla auto (4.1): con la retencion actual de 8 trimestres, el nii_ratio
# de bancos como JPM/C/GS/CB cae a 0.62 (5/8 Q con NetInterestIncome) y no llega
# al umbral 0.70 -> quedarian mal clasificados. La regla auto fue calibrada con
# ~49 Q de historia; con 8 Q es fragil para el NII. La curaduria explicita no
# depende de la ventana de datos (decision usuario 5/8/2026). La regla auto queda
# como BACKSTOP para tickers nuevos/desconocidos fuera de esta lista.
FINANCIERO_OVERRIDE = {
    "JPM", "BAC", "C", "WFC", "GS", "MS", "AXP", "SCHW", "AIG", "CB",
    "PGR", "LNC", "NU", "UPST", "ITUB", "BBD", "BSBR",  # 17 por estructura bancaria
    "XP",                                               # broker; op income Yahoo basura
}
# (opcional) forzar no-financiero pese a la regla, si hiciera falta a futuro:
NO_FINANCIERO_OVERRIDE = set()

GROSS_RATIO_MAX = 0.30   # casi nunca tuvo estructura industrial
OPINC_RATIO_MAX = 0.30
NII_RATIO_MIN   = 0.70   # casi siempre tuvo net interest income


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -- Helpers de calculo (sign-safe, division-safe) ----------------------------

def _safe_div(num, den):
    if num is None or den is None:
        return pd.Series(np.nan, index=getattr(den, "index", None))
    den = pd.Series(den).replace(0, np.nan)
    return pd.Series(num).astype(float) / den.astype(float)


def _growth(s):
    if s is None:
        return None, None
    s = pd.Series(s).astype(float)
    qoq = (s - s.shift(1)) / s.shift(1).abs().replace(0, np.nan)
    yoy = (s - s.shift(4)) / s.shift(4).abs().replace(0, np.nan)
    return qoq, yoy


def _ttm(s):
    if s is None:
        return None
    return pd.Series(s).astype(float).rolling(window=4, min_periods=4).sum()


def _jget(rj, key):
    """Lee una clave de raw_json (dict o str). None si falta/NaN."""
    if rj is None:
        return None
    if isinstance(rj, str):
        try:
            rj = json.loads(rj)
        except Exception:
            return None
    v = rj.get(key) if isinstance(rj, dict) else None
    if v is None:
        return None
    try:
        fv = float(v)
        return None if fv != fv else fv
    except (TypeError, ValueError):
        return v


# Columnas persistidas (orden = INSERT). +profile, +rotce, +efficiency (v2).
OUT_COLS = [
    "ticker", "fiscal_period_end", "period_type", "reporting_currency",
    "sector", "industry", "profile",
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
    "rotce_ttm", "efficiency_ratio_ttm",
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
    try:
        act = pd.read_sql("SELECT ticker, sector, industry FROM activos", engine)
    except Exception:
        act = pd.DataFrame(columns=["ticker", "sector", "industry"])
    return inc, bal, cf, val, act


# -- Clasificacion de perfil --------------------------------------------------

def clasificar_perfil(tk, di):
    """Devuelve 'financiero' | 'no_financiero' segun estructura multi-Q + override."""
    if tk in FINANCIERO_OVERRIDE:
        return "financiero"
    if tk in NO_FINANCIERO_OVERRIDE:
        return "no_financiero"
    n = len(di)
    if n == 0:
        return "no_financiero"
    def frac(col):
        if col not in di.columns:
            return 0.0
        return di[col].notna().sum() / n
    gross_r = frac("gross_profit")
    opinc_r = frac("operating_income")
    # NetInterestIncome vive en raw_json
    nii_cnt = sum(1 for rj in di["raw_json"] if _jget(rj, "NetInterestIncome") is not None) if "raw_json" in di.columns else 0
    nii_r = nii_cnt / n
    if gross_r < GROSS_RATIO_MAX and opinc_r < OPINC_RATIO_MAX and nii_r > NII_RATIO_MIN:
        return "financiero"
    return "no_financiero"


def _build_frame(tk, inc, bal, cf, val):
    """Merge as-of: balance/cashflow se alinean al ULTIMO registro con fecha
    <= fecha del income (corrige B2: balance del Q exacto puede no existir)."""
    di = inc[inc["ticker"] == tk].copy().sort_values("fiscal_period_end")
    if di.empty:
        return None, None
    perfil = clasificar_perfil(tk, di)

    db = bal[bal["ticker"] == tk].copy().sort_values("fiscal_period_end")
    dc = cf[cf["ticker"] == tk].copy().sort_values("fiscal_period_end")
    dv = val[(val["ticker"] == tk) & (val["period_type"] == "3M")].copy().sort_values("period_end")

    # Income: columnas dedicadas + claves crudas de raw_json (common equity etc)
    ci = [c for c in ["fiscal_period_end", "reporting_currency", "total_revenue",
                      "gross_profit", "operating_income", "operating_expense",
                      "ebit", "ebitda", "net_income", "pretax_income",
                      "tax_provision", "diluted_eps", "raw_json"] if c in di.columns]
    base = di[ci].rename(columns={"net_income": "ni", "raw_json": "inc_json"})

    # Extraer de inc_json: NetIncomeCommonStockholders, PreferredStockDividends
    base["ni_common"] = base["inc_json"].apply(
        lambda rj: _jget(rj, "NetIncomeCommonStockholders"))
    base["pref_div"] = base["inc_json"].apply(
        lambda rj: _jget(rj, "PreferredStockDividends"))
    # EBITDA normalizado (excluye cargos extraordinarios one-off). Para EV/EBITDA
    # usamos este (estandar de la industria, coincide con TV): el EBITDA reportado
    # puede ser negativo en un Q con impairment y deformar el TTM (ej. CVS Q3'25).
    base["ebitda_norm"] = base["inc_json"].apply(
        lambda rj: _jget(rj, "NormalizedEBITDA"))
    base = base.drop(columns=["inc_json"])

    # -- as-of merge de balance --
    if not db.empty:
        cb = [c for c in ["fiscal_period_end", "total_assets", "current_assets",
                          "current_liabilities", "cash_and_equivalents", "total_debt",
                          "stockholders_equity", "share_issued", "raw_json"]
              if c in db.columns]
        dbb = db[cb].rename(columns={"raw_json": "bal_json",
                                     "fiscal_period_end": "bal_fpe"})
        dbb["common_equity"] = dbb["bal_json"].apply(lambda rj: _jget(rj, "CommonStockEquity"))
        dbb["tangible_bv"]   = dbb["bal_json"].apply(lambda rj: _jget(rj, "TangibleBookValue"))
        dbb["ordinary_shares"] = dbb["bal_json"].apply(lambda rj: _jget(rj, "OrdinarySharesNumber"))
        dbb = dbb.drop(columns=["bal_json"])
        base = pd.merge_asof(base, dbb, left_on="fiscal_period_end",
                             right_on="bal_fpe", direction="backward")

    # -- as-of merge de cashflow (FCF) --
    if not dc.empty:
        cc = [c for c in ["fiscal_period_end", "free_cash_flow"] if c in dc.columns]
        dcc = dc[cc].rename(columns={"fiscal_period_end": "cf_fpe"})
        base = pd.merge_asof(base, dcc, left_on="fiscal_period_end",
                             right_on="cf_fpe", direction="backward")

    # -- valuation: join exacto por fecha (no as-of: es snapshot de mercado) --
    if not dv.empty:
        cv = [c for c in ["period_end", "pe_ratio", "pb_ratio", "ps_ratio",
                          "enterprise_value_ebitda", "market_cap"] if c in dv.columns]
        dvv = dv[cv].rename(columns={"period_end": "fiscal_period_end",
                                     "enterprise_value_ebitda": "ev_ebitda"})
        dvv = dvv.drop_duplicates(subset=["fiscal_period_end"], keep="last")
        base = base.merge(dvv, on="fiscal_period_end", how="left")

    base["ticker"] = tk
    return base.sort_values("fiscal_period_end").reset_index(drop=True), perfil


def _compute(m, perfil, warnings):
    es_fin = (perfil == "financiero")
    o = pd.DataFrame()
    o["ticker"] = m["ticker"]
    o["fiscal_period_end"] = m["fiscal_period_end"].dt.date
    o["period_type"] = "3M"
    o["reporting_currency"] = m.get("reporting_currency")
    o["profile"] = perfil

    g = m.get
    rev, ni, eps, fcf = g("total_revenue"), g("ni"), g("diluted_eps"), g("free_cash_flow")
    ni_common = g("ni_common")
    pref_div = g("pref_div")
    # Equity: preferir common equity (B3); fallback a stockholders_equity
    common_eq = g("common_equity")
    stock_eq = g("stockholders_equity")
    equity = common_eq if common_eq is not None else stock_eq
    if common_eq is not None and stock_eq is not None:
        equity = pd.Series(common_eq).fillna(pd.Series(stock_eq))
    assets = g("total_assets")
    debt, cash = g("total_debt"), g("cash_and_equivalents")
    shares = g("ordinary_shares")
    if shares is None:
        shares = g("share_issued")
    tangible_bv = g("tangible_bv")

    # NI para ROE: usar common si existe (resta preferentes implicito), si no NI
    ni_for_roe = ni_common if ni_common is not None else ni

    # EBITDA para TTM: preferir NormalizedEBITDA (excluye one-offs), fallback al
    # reportado donde el normalizado falte. Afecta ev_ebitda y net_debt/ebitda.
    _eb_norm, _eb_rep = g("ebitda_norm"), g("ebitda")
    if _eb_norm is not None and _eb_rep is not None:
        _eb_use = _eb_norm.fillna(_eb_rep)
    else:
        _eb_use = _eb_norm if _eb_norm is not None else _eb_rep

    # TTM
    rev_ttm, ni_ttm, eb_ttm = _ttm(rev), _ttm(ni), _ttm(_eb_use)
    ni_common_ttm = _ttm(ni_for_roe)
    gp_ttm, oi_ttm, opex_ttm = _ttm(g("gross_profit")), _ttm(g("operating_income")), _ttm(g("operating_expense"))
    ebit_ttm, tax_ttm, pretax_ttm = _ttm(g("ebit")), _ttm(g("tax_provision")), _ttm(g("pretax_income"))
    fcf_ttm, eps_ttm = _ttm(fcf), _ttm(eps)

    # P1) Mercado vs valor real (de Yahoo, ambos perfiles)
    o["pe_ratio"], o["pb_ratio"] = g("pe_ratio"), g("pb_ratio")
    o["ps_ratio"], o["ev_ebitda"] = g("ps_ratio"), g("ev_ebitda")
    _, pe_yoy = _growth(g("pe_ratio")); _, pb_yoy = _growth(g("pb_ratio"))
    o["pe_yoy_pct"], o["pb_yoy_pct"] = pe_yoy, pb_yoy
    o["book_value_per_share"] = _safe_div(equity, shares)   # B3: common equity
    o["eps_q"], o["eps_ttm"] = eps, eps_ttm

    # P2) Rentabilidad / calidad
    o["roe_ttm"] = _safe_div(ni_common_ttm, equity)         # B3
    o["roa_ttm"] = _safe_div(ni_ttm, assets)

    if es_fin:
        # Financieras: margenes industriales / ROIC / opex -> NULL
        nan_s = pd.Series(np.nan, index=o.index)
        o["roic_ttm"] = nan_s
        o["gross_margin_ttm"] = nan_s
        o["operating_margin_ttm"] = nan_s
        o["net_margin_ttm"] = _safe_div(ni_ttm, rev_ttm)    # neto si aplica (con caveat)
        o["net_margin_yoy_delta"] = o["net_margin_ttm"] - o["net_margin_ttm"].shift(4)
        o["operating_margin_yoy_delta"] = nan_s
        o["opex_to_revenue_ttm"] = nan_s
        # Set bancario
        o["rotce_ttm"] = _safe_div(ni_common_ttm, tangible_bv)
        o["efficiency_ratio_ttm"] = _safe_div(
            (pd.Series(rev_ttm) - pd.Series(pretax_ttm)) if (rev_ttm is not None and pretax_ttm is not None) else None,
            rev_ttm)
    else:
        # No-financieras: ROIC standard con common equity (B3)
        tax_rate = _safe_div(tax_ttm, pretax_ttm).clip(0, 1)
        nopat = (pd.Series(ebit_ttm).astype(float) * (1 - tax_rate)) if ebit_ttm is not None else None
        inv_cap = None
        if debt is not None and equity is not None and cash is not None:
            inv_cap = pd.Series(debt).astype(float) + pd.Series(equity).astype(float) - pd.Series(cash).astype(float)
        roic = _safe_div(nopat, inv_cap)
        if pretax_ttm is not None:
            roic = roic.where(pd.Series(pretax_ttm).astype(float) > 0)
        o["roic_ttm"] = roic
        o["gross_margin_ttm"] = _safe_div(gp_ttm, rev_ttm)
        o["operating_margin_ttm"] = _safe_div(oi_ttm, rev_ttm)
        o["net_margin_ttm"] = _safe_div(ni_ttm, rev_ttm)
        o["net_margin_yoy_delta"] = o["net_margin_ttm"] - o["net_margin_ttm"].shift(4)
        o["operating_margin_yoy_delta"] = o["operating_margin_ttm"] - o["operating_margin_ttm"].shift(4)
        o["opex_to_revenue_ttm"] = _safe_div(opex_ttm, rev_ttm)
        o["rotce_ttm"] = pd.Series(np.nan, index=o.index)
        o["efficiency_ratio_ttm"] = pd.Series(np.nan, index=o.index)

    # Crecimiento (ambos perfiles)
    rev_qoq, rev_yoy = _growth(rev); ni_qoq, ni_yoy = _growth(ni); eps_qoq, eps_yoy = _growth(eps)
    o["revenue_qoq_pct"], o["revenue_yoy_pct"] = rev_qoq, rev_yoy
    o["net_income_qoq_pct"], o["net_income_yoy_pct"] = ni_qoq, ni_yoy
    o["eps_qoq_pct"], o["eps_yoy_pct"] = eps_qoq, eps_yoy

    # P3) Cash -- FCF (financieras: NULL, FCF no es metrica de banco)
    if es_fin:
        nan_s = pd.Series(np.nan, index=o.index)
        o["fcf_ttm"] = nan_s
        o["fcf_margin_ttm"] = nan_s
        o["fcf_qoq_pct"] = nan_s
        o["fcf_yoy_pct"] = nan_s
    else:
        o["fcf_ttm"] = fcf_ttm
        o["fcf_margin_ttm"] = _safe_div(fcf_ttm, rev_ttm)
        fcf_qoq, fcf_yoy = _growth(fcf)
        o["fcf_qoq_pct"], o["fcf_yoy_pct"] = fcf_qoq, fcf_yoy

    # Estructura / solvencia
    if es_fin:
        # liquidez corriente / working capital no aplican a bancos
        o["current_ratio"] = pd.Series(np.nan, index=o.index)
        o["working_capital"] = pd.Series(np.nan, index=o.index)
    else:
        o["current_ratio"] = _safe_div(g("current_assets"), g("current_liabilities"))
        ca, cl = g("current_assets"), g("current_liabilities")
        o["working_capital"] = (pd.Series(ca).astype(float) - pd.Series(cl).astype(float)) if (ca is not None and cl is not None) else np.nan
    o["debt_to_equity"] = _safe_div(debt, equity)
    net_debt = (pd.Series(debt).astype(float) - pd.Series(cash).astype(float)) if (debt is not None and cash is not None) else None
    o["net_debt"] = net_debt if net_debt is not None else np.nan
    o["net_debt_to_ebitda_ttm"] = _safe_div(net_debt, eb_ttm) if not es_fin else pd.Series(np.nan, index=o.index)

    # Agregados TTM crudos
    o["revenue_ttm"], o["net_income_ttm"], o["ebitda_ttm"] = rev_ttm, ni_ttm, eb_ttm
    o["n_quarters_available"] = len(m)

    _validate_scale(m, warnings)
    return o


def _validate_scale(m, warnings):
    last = m.iloc[-1]
    tk = last["ticker"]
    cur = last.get("reporting_currency")
    mc = last.get("market_cap")
    eq = last.get("common_equity") if last.get("common_equity") is not None else last.get("stockholders_equity")
    pb = last.get("pb_ratio")
    if cur == "USD" and mc and eq and pb and eq != 0 and pb != 0:
        ratio = (mc / eq) / pb
        if ratio < 0.2 or ratio > 5:
            warnings.append(f"{tk}: escala EQUITY sospechosa (mc/eq vs pb = {ratio:.2f}x)")
    ni, sh, eps = last.get("ni"), last.get("ordinary_shares"), last.get("diluted_eps")
    if sh is None:
        sh = last.get("share_issued")
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
    parser = argparse.ArgumentParser(description="Computa fundamentales_ratios_q v2 (local)")
    parser.add_argument("--tickers", default=None, help="CSV de tickers (default: todos)")
    parser.add_argument("--dry-run", action="store_true", help="Calcula sin escribir")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    print()
    print(SEP)
    print(f"  COMPUTE fundamentales_ratios_q v2"
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

    all_rows, sin_data, warnings, perfiles = [], [], [], {}
    for tk in universo:
        frame, perfil = _build_frame(tk, inc, bal, cf, val)
        if frame is None or frame.empty:
            sin_data.append(tk)
            continue
        perfiles[perfil] = perfiles.get(perfil, 0) + 1
        ratios = _compute(frame, perfil, warnings)
        all_rows.extend(_to_records(_attach_sector(ratios, act)))

    log(f"Filas calculadas: {len(all_rows)} ({len(universo)-len(sin_data)} tickers)")
    log(f"Perfiles: {perfiles}")
    if warnings:
        log(f"AVISOS de escala ({len(warnings)}):")
        for w in warnings:
            log(f"  WARN {w}")
    else:
        log("Validacion de escala: OK (sin avisos).")

    if args.dry_run:
        if all_rows:
            last = all_rows[-1]
            prev = {k: last[k] for k in ["ticker", "fiscal_period_end", "profile",
                    "net_margin_ttm", "roe_ttm", "roic_ttm", "rotce_ttm",
                    "efficiency_ratio_ttm", "book_value_per_share", "n_quarters_available"]}
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
