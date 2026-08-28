"""
sec_xbrl_prototipo.py -- PROTOTIPO DE INVESTIGACION (27/8/2026)

NO es un script operativo. NO escribe en la DB. NO se encadena a nada.
Es el material que respalda docs/fuentes_fundamentales.md: sirve para
reproducir la medicion de si SEC XBRL es normalizable y a que costo.

Que hace:
  descargar()  -> baja companyfacts de data.sec.gov para una lista de tickers
  extraer()    -> normaliza un companyfacts a serie TRIMESTRAL (29 conceptos)
  comparar()   -> cruza lo normalizado contra fundamentales_income_q (yahooquery)

Los 4 problemas de SEC XBRL que resuelve (detalle en el doc):
  1. sinonimos     -> lista ORDENADA de tags candidatos por concepto
  2. cambio de tag -> se resuelve POR PERIODO, no por ticker (NVDA cambio en
                      2022, UNH en 2015, WMT en 2012)
  3. Q4 ausente    -> derivado como FY - 9M (el 10-K no publica el Q suelto)
  4. restatements  -> se toma el 'filed' mas reciente

REGLA APRENDIDA A LOS GOLPES: la lista de candidatos debe contener SOLO
sinonimos verdaderos. Se introdujeron 3 errores silenciosos (numeros
plausibles de OTRA cosa) y ninguno se habria detectado sin el cruce:
  - ProfitLoss como sinonimo de NetIncomeLoss  -> incluye minoritarios (FCX +41%)
  - ...BeforeIncomeTaxesDomestic como pretax   -> es el SEGMENTO domestico
  - EPS/acciones del Q4 por resta              -> son promedios PONDERADOS

Uso:
    python scripts/oneshot/sec_xbrl_prototipo.py descargar   # ~522 MB, 147 tickers
    python scripts/oneshot/sec_xbrl_prototipo.py cobertura
    python scripts/oneshot/sec_xbrl_prototipo.py comparar
"""
import os
import sys
import json
import glob
import time
import datetime as dt
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

CACHE = os.path.join(os.environ.get("TEMP", "."), "sec")
UA = "contacto@ejemplo.com research script"   # SEC exige User-Agent identificable

# ---- concepto -> tags candidatos, en orden de preferencia -------------------
FLOW = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenuesNetOfInterestExpense", "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet", "InterestAndDividendIncomeOperating"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold",
                        "CostOfGoodsSold", "CostOfServices"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "operating_expense": ["OperatingExpenses", "CostsAndExpenses"],
    "sga": ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    # OJO: '...BeforeIncomeTaxesDomestic' NO va aca -- es el segmento domestico.
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax_provision": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss"],          # ProfitLoss NO: incluye minoritarios
    "net_income_common": ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestIncomeExpenseNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization", "Depreciation"],
}
INSTANT = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "inventory": ["InventoryNet"],
    "debt_short": ["ShortTermBorrowings", "DebtCurrent", "LongTermDebtCurrent",
                   "OtherShortTermBorrowings"],
    "debt_long": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
    "shares_out": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
}


# ---------------------------------------------------------------- descarga --
def descargar(pares, destino=CACHE, pausa=0.55):
    """
    pares: [(ticker, cik_int), ...].  Reanudable: saltea lo ya bajado.

    OJO: SEC corta las descargas hechas con curl en loop, porque abre una
    conexion TLS nueva por pedido. Con requests.Session (keep-alive) y ~0.55s
    de pausa: 147 de 147, cero fallos.
    """
    import requests
    os.makedirs(destino, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    ok = fail = skip = 0
    for tk, cik in pares:
        p = os.path.join(destino, f"{tk}.json")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            skip += 1
            continue
        for intento in range(3):
            try:
                r = s.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json",
                          timeout=60)
                if r.status_code == 200:
                    open(p, "wb").write(r.content)
                    ok += 1
                    break
                time.sleep(2 * (intento + 1))
            except Exception:
                time.sleep(2 * (intento + 1))
        else:
            fail += 1
            print(f"  FALLO {tk} (CIK {cik})")
        time.sleep(pausa)
    print(f"descargados={ok} ya_estaban={skip} fallidos={fail}")
    return ok, skip, fail


def universo_usa():
    """[(ticker, cik)] de los tickers USA del universo, via ticker_pais."""
    import requests
    import pandas as pd
    os.environ.pop("DATABASE_URL", None)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    os.environ.pop("DATABASE_URL", None)
    from src.data.database import get_engine
    tp = pd.read_sql("SELECT ticker FROM ticker_pais WHERE region='USA'", get_engine())
    cm = requests.get("https://www.sec.gov/files/company_tickers.json",
                      headers={"User-Agent": UA}, timeout=60).json()
    cik = {v["ticker"]: int(v["cik_str"]) for v in cm.values()}
    return [(t, cik[t]) for t in sorted(tp.ticker) if t in cik]


# ------------------------------------------------------------ normalizacion --
def _d(a, b):
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def _facts(fx, tags):
    out = []
    for pri, t in enumerate(tags):
        for tax in ("us-gaap", "dei"):
            g = fx.get(tax, {})
            if t not in g:
                continue
            for unit, arr in g[t]["units"].items():
                for f in arr:
                    out.append(dict(f, _tag=t, _pri=pri, _unit=unit))
    return out


def _best(c):
    """Menor prioridad de tag gana; a igual tag, el 'filed' mas reciente."""
    return sorted(c, key=lambda f: (f["_pri"], f.get("filed", "")))[-1] if c else None


def _q_series(fx, tags):
    """Serie trimestral (3m) de un concepto de FLUJO, con el Q4 derivado."""
    buckets = defaultdict(list)
    for f in _facts(fx, tags):
        if not f.get("start"):
            continue
        n = _d(f["start"], f["end"])
        b = "Q" if n <= 100 else "H" if n <= 196 else "9M" if n <= 290 else "FY" if n <= 380 else None
        if b:
            buckets[(b, f["start"], f["end"])].append(f)
    q = {}
    for (b, s, e), c in buckets.items():
        if b != "Q":
            continue
        p = _best(c)
        if e not in q or p["_pri"] < q[e]["_pri"]:
            q[e] = p
    fy = {(s, e): _best(c) for (b, s, e), c in buckets.items() if b == "FY"}
    nm = {(s, e): _best(c) for (b, s, e), c in buckets.items() if b == "9M"}
    der = 0
    for (fs, fe), ff in fy.items():
        if fe in q:
            continue
        cand = [v for (ns, ne), v in nm.items() if ns == fs and ne < fe]
        if not cand:
            continue
        n9 = sorted(cand, key=lambda x: x["end"])[-1]
        q[fe] = dict(val=ff["val"] - n9["val"], end=fe, filed=ff.get("filed"),
                     _tag=ff["_tag"], _pri=ff["_pri"], _der=True)
        der += 1
    return q, der


def extraer(path):
    """companyfacts -> (nombre, {fecha_fin: {concepto: valor}}, meta)."""
    j = json.load(open(path, encoding="utf-8"))
    fx = j.get("facts", {})
    filas = defaultdict(dict)
    meta = {}
    for con, tags in FLOW.items():
        q, der = _q_series(fx, tags)
        meta[con] = dict(n=len(q), derivados=der,
                         tags=sorted({v["_tag"] for v in q.values()}))
        for e, f in q.items():
            filas[e][con] = f["val"]
            filas[e][con + "__tag"] = f["_tag"]
            if f.get("_der"):
                filas[e][con + "__der"] = True

    # El EPS y las acciones del Q4 NO son derivables por resta: son promedios
    # PONDERADOS (la formula correcta es 4*FY - 3*9M). Mientras no este
    # implementada se anulan -- es preferible un hueco a un numero equivocado.
    for e, v in filas.items():
        if v.get("eps_diluted__der"):
            v.pop("eps_diluted", None)
            v.pop("eps_diluted__tag", None)
        if v.get("shares_diluted__der"):
            v.pop("shares_diluted", None)
            v.pop("shares_diluted__tag", None)

    for con, tags in INSTANT.items():
        inst = defaultdict(list)
        for f in _facts(fx, tags):
            if f.get("start"):
                continue
            inst[f["end"]].append(f)
        vals = {e: _best(c) for e, c in inst.items()}
        meta[con] = dict(n=len(vals), derivados=0,
                         tags=sorted({v["_tag"] for v in vals.values() if v}))
        for e, f in vals.items():
            if f:
                filas[e][con] = f["val"]
                filas[e][con + "__tag"] = f["_tag"]
    return j.get("entityName", ""), filas, meta


def cargar_todo(origen=CACHE):
    """Corre extraer() sobre todo el cache. Devuelve (df_largo, df_meta)."""
    import pandas as pd
    filas, metas = [], []
    for p in sorted(glob.glob(os.path.join(origen, "*.json"))):
        tk = os.path.basename(p)[:-5]
        try:
            _, f, m = extraer(p)
        except Exception as e:
            metas.append(dict(ticker=tk, concepto="__ERROR__", n=0, derivados=0,
                              tags=str(e)[:60]))
            continue
        for con, mm in m.items():
            metas.append(dict(ticker=tk, concepto=con, n=mm["n"],
                              derivados=mm["derivados"], tags="|".join(mm["tags"][:3])))
        for end, vals in f.items():
            r = {"ticker": tk, "period_end": end}
            r.update({k: v for k, v in vals.items()
                      if not k.endswith("__tag") and not k.endswith("__der")})
            filas.append(r)
    return pd.DataFrame(filas), pd.DataFrame(metas)


# ------------------------------------------------------------- mediciones --
def cobertura():
    import pandas as pd
    sec, meta = cargar_todo()
    n = sec.ticker.nunique()
    rec = sec[sec.period_end >= "2025-01-01"]
    out = []
    for c in list(FLOW) + list(INSTANT):
        if c not in sec.columns:
            out.append(dict(concepto=c, tickers=0, pct=0, hist=0))
            continue
        tk = rec.groupby("ticker")[c].apply(lambda s: s.notna().any()).sum()
        out.append(dict(concepto=c, tipo="flujo" if c in FLOW else "instante",
                        tickers=int(tk), pct=round(100 * tk / n),
                        hist=int(meta[meta.concepto == c].n.median())))
    d = pd.DataFrame(out).sort_values("pct", ascending=False)
    print(f"== COBERTURA sobre {n} tickers (dato desde 2025) ==")
    print(d.to_string(index=False))
    return d


def comparar():
    """Cruza lo normalizado contra fundamentales_income_q (yahooquery)."""
    import pandas as pd
    os.environ.pop("DATABASE_URL", None)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    os.environ.pop("DATABASE_URL", None)
    from src.data.database import get_engine

    sec, _ = cargar_todo()
    db = pd.read_sql("""
        SELECT ticker, fiscal_period_end AS period_end,
               (raw_json->>'TotalRevenue')::numeric     revenue,
               (raw_json->>'NetIncome')::numeric        net_income,
               (raw_json->>'OperatingIncome')::numeric  operating_income,
               (raw_json->>'GrossProfit')::numeric      gross_profit,
               (raw_json->>'PretaxIncome')::numeric     pretax_income,
               (raw_json->>'DilutedEPS')::numeric       eps_diluted
        FROM fundamentales_income_q
        WHERE (raw_json->>'TotalRevenue') IS NOT NULL
          AND raw_json->>'TotalRevenue' <> ''
    """, get_engine())

    cols = ["revenue", "net_income", "eps_diluted", "operating_income",
            "gross_profit", "pretax_income"]
    a = sec[["ticker", "period_end"] + cols].copy()
    b = db[["ticker", "period_end"] + cols].copy()
    # SEC usa la fecha fiscal EXACTA (AAPL cierra 2026-06-27); la convencion
    # del proyecto es fin de mes -> matcheo con tolerancia de +-7 dias.
    a["period_end"] = pd.to_datetime(a.period_end)
    b["period_end"] = pd.to_datetime(b.period_end)
    m = pd.merge_asof(a.sort_values("period_end"), b.sort_values("period_end"),
                      on="period_end", by="ticker", suffixes=("_sec", "_db"),
                      direction="nearest", tolerance=pd.Timedelta(days=7))
    m = m[m.period_end >= "2024-01-01"]

    res = []
    for c in cols:
        x = pd.to_numeric(m[c + "_sec"], errors="coerce")
        y = pd.to_numeric(m[c + "_db"], errors="coerce")
        ok = x.notna() & y.notna() & (y.abs() > 1e-6)
        if not ok.any():
            continue
        dif = (x[ok] / y[ok] - 1).abs() * 100
        res.append(dict(concepto=c, comparables=int(ok.sum()),
                        exacto=f"{100*(dif<0.1).mean():.0f}%",
                        menor_1pct=f"{100*(dif<1).mean():.0f}%",
                        desvios_5pct=int((dif > 5).sum()),
                        tickers=int(m[ok][dif > 5].ticker.nunique())))
    print("== SEC normalizado vs yahooquery (trimestres desde 2024) ==")
    print(pd.DataFrame(res).to_string(index=False))
    return m


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "cobertura"
    if accion == "descargar":
        descargar(universo_usa())
    elif accion == "cobertura":
        cobertura()
    elif accion == "comparar":
        comparar()
    else:
        print(__doc__)
