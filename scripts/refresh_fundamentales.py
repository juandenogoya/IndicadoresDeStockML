"""
refresh_fundamentales.py
Carga / refresca las 4 tablas fundamentales_* desde yahooquery.

Trae los ultimos N trimestres (default 8) por ticker de las 3 statements +
valuation_measures y hace upsert con ON CONFLICT DO UPDATE.

Arquitectura (Plan C):
    Los fundamentales son RECUPERABLES (yahooquery los sirve historicos en
    cualquier momento), por lo que viven SOLO en local -- a diferencia de
    opciones_snapshot que requiere Railway por ser irrecuperable. Esto
    elimina la necesidad de cron Oracle, sync Railway->local y mantenimiento
    de schema en dos lados.

    Uso normal: --target local (default). El modo --target railway/both
    queda como opcion residual si en algun momento se necesita exponer
    fundamentales a un consumidor que viva en Railway.

Restatements (manejo nativo):
    Cada refresh re-lee los ultimos N Q y hace UPSERT. Si Yahoo revisa
    cifras de un Q viejo (restatement), la correccion entra automaticamente
    en la proxima corrida. No hace falta logica especial ni --full-resync.

Estrategia anti-rate-limit:
    - chunks de 20 tickers (yahooquery async dentro del chunk)
    - pausa 2s entre chunks
    - deteccion de rate limit: si los 3 DataFrames de statements vienen
      vacios para todo el chunk -> backoff exponencial (30s, 60s, 120s)
      hasta 3 reintentos antes de marcar el chunk como FAIL
    - yfinance_lock.acquire() al inicio (yahooquery y yfinance comparten
      provider e IP rate-limit)
    - Tiempo tipico full 199 tickers: ~3.5-4 min

Cobertura de columnas:
    - 12-15 columnas dedicadas por tabla (las mas usadas para ratios)
    - raw_json (JSONB) con la fila completa de yahooquery -> permite
      derivar nuevas metricas sin re-fetch

Multi-moneda:
    Los ADRs reportan en su moneda local (BABA en CNY, PBR en BRL, etc.).
    Se guarda reporting_currency por fila (campo currencyCode que viene en
    cada statement). De los 199 tickers: 170 USD + 29 monedas locales.
    Para analisis cross-ticker hay que filtrar por reporting_currency='USD'
    o normalizar via FX. La decision queda al consumidor.

Cadencia sugerida:
    Manual cuando se quiera. Las earnings llegan distribuidas, no hay
    urgencia. Wrapper conveniente: scripts/manual/refresh_fundamentales.bat

Uso:
    # Full universo a local (modo canonico)
    python scripts/refresh_fundamentales.py

    # Test con tickers especificos
    python scripts/refresh_fundamentales.py --tickers AAPL,BABA,JPM

    # Dry-run (no escribe, solo loguea)
    python scripts/refresh_fundamentales.py --dry-run

    # Profundidad distinta (default 8 Q)
    python scripts/refresh_fundamentales.py --lookback-q 12
"""

import sys
import os
import json
import time
import math
import argparse
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import psycopg2
import psycopg2.extras
from sqlalchemy import text

from src.utils.config import ALL_TICKERS
from src.utils.yfinance_lock import acquire as acquire_yf_lock
from scripts.oneshot.create_fundamentales_tables import (
    get_railway_engine, get_local_engine, _parse_env_file,
)

# -- Config --------------------------------------------------------------------

CHUNK_SIZE_DEFAULT = 20
LOOKBACK_Q_DEFAULT = 8
PAUSE_INTER_CHUNK  = 2.0
PAUSE_INTER_CALL   = 0.3
BACKOFF_BASE_SEC   = 30
MAX_RETRIES        = 3

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -- Column mapping yahooquery -> DB ------------------------------------------

INCOME_MAP = {
    "TotalRevenue":          "total_revenue",
    "CostOfRevenue":         "cost_of_revenue",
    "GrossProfit":           "gross_profit",
    "OperatingExpense":      "operating_expense",
    "OperatingIncome":       "operating_income",
    "EBIT":                  "ebit",
    "EBITDA":                "ebitda",
    "InterestExpense":       "interest_expense",
    "PretaxIncome":          "pretax_income",
    "TaxProvision":          "tax_provision",
    "NetIncome":             "net_income",
    "DilutedEPS":            "diluted_eps",
    "BasicEPS":              "basic_eps",
    "DilutedAverageShares":  "diluted_avg_shares",
}

BALANCE_MAP = {
    "TotalAssets":                          "total_assets",
    "CurrentAssets":                        "current_assets",
    "CashAndCashEquivalents":               "cash_and_equivalents",
    "Inventory":                            "inventory",
    "AccountsReceivable":                   "accounts_receivable",
    "TotalLiabilitiesNetMinorityInterest":  "total_liabilities",
    "CurrentLiabilities":                   "current_liabilities",
    "AccountsPayable":                      "accounts_payable",
    "CurrentDebt":                          "current_debt",
    "LongTermDebt":                         "long_term_debt",
    "TotalDebt":                            "total_debt",
    "StockholdersEquity":                   "stockholders_equity",
    "RetainedEarnings":                     "retained_earnings",
    "ShareIssued":                          "share_issued",
}

CASHFLOW_MAP = {
    "OperatingCashFlow":          "operating_cash_flow",
    "InvestingCashFlow":          "investing_cash_flow",
    "FinancingCashFlow":          "financing_cash_flow",
    "CapitalExpenditure":         "capital_expenditure",
    "FreeCashFlow":               "free_cash_flow",
    "DepreciationAndAmortization":"depreciation_amortization",
    "StockBasedCompensation":     "stock_based_compensation",
    "RepurchaseOfCapitalStock":   "repurchase_of_capital_stock",
    "CashDividendsPaid":          "cash_dividends_paid",
    "ChangesInCash":              "change_in_cash",
    "NetIncome":                  "net_income",
}

VALUATION_MAP = {
    "PeRatio":                       "pe_ratio",
    "PbRatio":                       "pb_ratio",
    "PsRatio":                       "ps_ratio",
    "PegRatio":                      "peg_ratio",
    "EnterprisesValueRevenueRatio":  "enterprise_value_revenue",
    "EnterprisesValueEBITDARatio":   "enterprise_value_ebitda",
    "EnterpriseValue":               "enterprise_value",
    "MarketCap":                     "market_cap",
    "ForwardPeRatio":                "forward_pe_ratio",
}


# -- Helpers -------------------------------------------------------------------

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _clean_val(v):
    """Convierte NaN/inf/pd.NaT/Timestamp -> None/str para JSON y SQL."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat() if hasattr(v, "isoformat") else str(v)
    if pd.isna(v):
        return None
    return v


def _row_to_raw_json(row: pd.Series) -> str:
    """Serializa una fila yahooquery como JSON string (para columna JSONB)."""
    clean = {k: _clean_val(v) for k, v in row.to_dict().items()}
    return json.dumps(clean, default=str)


# -- Fetch ---------------------------------------------------------------------

def fetch_chunk(tickers: list) -> dict:
    """Para un chunk de tickers, devuelve dict con DataFrames de los 4 endpoints."""
    from yahooquery import Ticker

    t = Ticker(tickers, asynchronous=True, max_workers=8)
    out = {"income": None, "balance": None, "cashflow": None, "valuation": None}

    out["income"] = t.income_statement(frequency="q")
    time.sleep(PAUSE_INTER_CALL)
    out["balance"] = t.balance_sheet(frequency="q")
    time.sleep(PAUSE_INTER_CALL)
    out["cashflow"] = t.cash_flow(frequency="q")
    time.sleep(PAUSE_INTER_CALL)
    out["valuation"] = t.valuation_measures
    return out


def _chunk_looks_rate_limited(chunk_data: dict) -> bool:
    """Heuristica: si las 3 statements vienen vacias/no-DF -> probable rate limit."""
    dfs = [chunk_data.get("income"), chunk_data.get("balance"), chunk_data.get("cashflow")]
    ok = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty]
    return len(ok) == 0


def fetch_chunk_with_retry(tickers: list, chunk_idx: int) -> dict:
    """Wrap fetch_chunk con backoff exponencial ante rate limit detectado."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = fetch_chunk(tickers)
        except Exception as e:
            log(f"  chunk {chunk_idx} attempt {attempt}: EXCEPTION {type(e).__name__}: {e}")
            data = {"income": None, "balance": None, "cashflow": None, "valuation": None}

        if not _chunk_looks_rate_limited(data):
            return data

        if attempt < MAX_RETRIES:
            backoff = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            log(f"  chunk {chunk_idx}: posible rate limit, backoff {backoff}s "
                f"(intento {attempt}/{MAX_RETRIES})")
            time.sleep(backoff)
        else:
            log(f"  chunk {chunk_idx}: rate limit persistente tras {MAX_RETRIES} intentos -- chunk FAIL")

    return data  # ultimo intento, devuelve lo que sea


# -- Extract -------------------------------------------------------------------

def _extract_statement_rows(df: pd.DataFrame, ticker: str, col_map: dict,
                            lookback_q: int, fecha_col: str = "asOfDate") -> list:
    """Para un statement (income/balance/cashflow), extrae las ultimas N filas 3M
    del ticker como list of dicts listos para upsert."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if ticker not in df.index.get_level_values(0):
        return []

    sub = df.loc[df.index == ticker].copy()
    if "periodType" in sub.columns:
        sub = sub[sub["periodType"] == "3M"]
    if sub.empty:
        return []
    sub = sub.sort_values(fecha_col).tail(lookback_q)

    rows = []
    for _, r in sub.iterrows():
        currency = r.get("currencyCode")
        row = {
            "ticker":              ticker,
            "fiscal_period_end":   _clean_val(r[fecha_col]),
            "period_type":         "3M",
            "reporting_currency":  _clean_val(currency),
            "raw_json":            _row_to_raw_json(r),
        }
        for yq_key, db_col in col_map.items():
            row[db_col] = _clean_val(r.get(yq_key))
        rows.append(row)
    return rows


def _extract_valuation_rows(df: pd.DataFrame, ticker: str,
                            lookback_q: int) -> list:
    """Valuation tiene period_type ('3M' + 'TTM') como parte de la PK."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if ticker not in df.index.get_level_values(0):
        return []

    sub = df.loc[df.index == ticker].copy()
    # Lookback mas amplio: conviven 3M y TTM, queremos al menos lookback_q de cada
    sub = sub.sort_values("asOfDate").tail(lookback_q * 2)

    rows = []
    for _, r in sub.iterrows():
        row = {
            "ticker":      ticker,
            "period_end":  _clean_val(r["asOfDate"]),
            "period_type": _clean_val(r.get("periodType")) or "3M",
            "raw_json":    _row_to_raw_json(r),
        }
        for yq_key, db_col in VALUATION_MAP.items():
            row[db_col] = _clean_val(r.get(yq_key))
        rows.append(row)
    return rows


# -- Upsert --------------------------------------------------------------------

def _upsert(conn, tabla: str, rows: list, pk_cols: list) -> int:
    """INSERT ... ON CONFLICT (pk) DO UPDATE para una lista de dicts."""
    if not rows:
        return 0

    # Todas las filas comparten claves (las definimos arriba)
    cols = list(rows[0].keys())
    cols_str = ", ".join(cols)
    placeholders = ", ".join([f"%({c})s" for c in cols])
    # SET col = EXCLUDED.col para todas las columnas NO-PK
    update_cols = [c for c in cols if c not in pk_cols]
    set_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
    # fetched_at se actualiza siempre (NOW())
    set_clause += ", fetched_at = NOW()"
    pk_str = ", ".join(pk_cols)

    sql = (
        f"INSERT INTO {tabla} ({cols_str}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk_str}) DO UPDATE SET {set_clause}"
    )

    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
    return cur.rowcount


def _pg_conn(env: dict):
    return psycopg2.connect(
        host=env.get("DB_HOST", "localhost"),
        port=int(env.get("DB_PORT", 5432)),
        dbname=env.get("DB_NAME", "activos_ml"),
        user=env.get("DB_USER", "postgres"),
        password=env.get("DB_PASSWORD", ""),
    )


def _railway_pg_conn():
    env = _parse_env_file(os.path.join(ROOT, ".env.local"))
    dsn = env.get("DATABASE_URL", "").replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(dsn, sslmode="require")


# -- Main loop -----------------------------------------------------------------

def procesar(tickers: list, lookback_q: int, chunk_size: int,
             targets: list, dry_run: bool) -> dict:
    """
    Procesa la lista de tickers en chunks, fetch + extract + upsert por target.
    Retorna stats globales.
    """
    # Preparar conexiones (psycopg2 para upsert eficiente con execute_batch)
    conns = {}
    if not dry_run:
        if "local" in targets:
            conns["local"] = _pg_conn(_parse_env_file(os.path.join(ROOT, ".env")))
        if "railway" in targets:
            conns["railway"] = _railway_pg_conn()

    stats = {
        "tickers_total":   len(tickers),
        "tickers_con_data":0,
        "tickers_sin_data":0,
        "chunks_ok":       0,
        "chunks_fail":     0,
        "filas_income":    0,
        "filas_balance":   0,
        "filas_cashflow":  0,
        "filas_valuation": 0,
    }

    chunk_list = list(_chunks(tickers, chunk_size))
    log(f"Procesando {len(tickers)} tickers en {len(chunk_list)} chunks de hasta {chunk_size}")

    for i, chunk in enumerate(chunk_list, 1):
        t0 = time.time()
        log(f"\n--- Chunk {i}/{len(chunk_list)}: {chunk[0]}..{chunk[-1]} ({len(chunk)} tickers) ---")
        data = fetch_chunk_with_retry(chunk, i)

        if _chunk_looks_rate_limited(data):
            stats["chunks_fail"] += 1
            log(f"  chunk {i} sin datos -- skip upsert")
            time.sleep(PAUSE_INTER_CHUNK)
            continue
        stats["chunks_ok"] += 1

        # Extraer filas por ticker
        rows_by_kind = {"income": [], "balance": [], "cashflow": [], "valuation": []}
        tickers_con_data_chunk = 0

        for tk in chunk:
            inc_rows  = _extract_statement_rows(data["income"],   tk, INCOME_MAP,   lookback_q)
            bal_rows  = _extract_statement_rows(data["balance"],  tk, BALANCE_MAP,  lookback_q)
            cf_rows   = _extract_statement_rows(data["cashflow"], tk, CASHFLOW_MAP, lookback_q)
            val_rows  = _extract_valuation_rows(data["valuation"], tk, lookback_q)

            if inc_rows or bal_rows or cf_rows or val_rows:
                tickers_con_data_chunk += 1
                rows_by_kind["income"].extend(inc_rows)
                rows_by_kind["balance"].extend(bal_rows)
                rows_by_kind["cashflow"].extend(cf_rows)
                rows_by_kind["valuation"].extend(val_rows)

        stats["tickers_con_data"] += tickers_con_data_chunk
        stats["tickers_sin_data"] += (len(chunk) - tickers_con_data_chunk)
        stats["filas_income"]    += len(rows_by_kind["income"])
        stats["filas_balance"]   += len(rows_by_kind["balance"])
        stats["filas_cashflow"]  += len(rows_by_kind["cashflow"])
        stats["filas_valuation"] += len(rows_by_kind["valuation"])

        log(f"  extraidos: income={len(rows_by_kind['income'])} "
            f"balance={len(rows_by_kind['balance'])} "
            f"cashflow={len(rows_by_kind['cashflow'])} "
            f"valuation={len(rows_by_kind['valuation'])} "
            f"({tickers_con_data_chunk}/{len(chunk)} tickers con data)")

        if dry_run:
            log(f"  DRY-RUN: skip upsert")
        else:
            for tgt, conn in conns.items():
                try:
                    _upsert(conn, "fundamentales_income_q",    rows_by_kind["income"],
                            ["ticker", "fiscal_period_end"])
                    _upsert(conn, "fundamentales_balance_q",   rows_by_kind["balance"],
                            ["ticker", "fiscal_period_end"])
                    _upsert(conn, "fundamentales_cashflow_q",  rows_by_kind["cashflow"],
                            ["ticker", "fiscal_period_end"])
                    _upsert(conn, "fundamentales_valuation_q", rows_by_kind["valuation"],
                            ["ticker", "period_end", "period_type"])
                    conn.commit()
                    log(f"  upsert {tgt}: OK")
                except Exception as e:
                    conn.rollback()
                    log(f"  upsert {tgt}: ERROR {type(e).__name__}: {e}")
                    raise

        log(f"  chunk {i} done en {time.time()-t0:.1f}s")
        if i < len(chunk_list):
            time.sleep(PAUSE_INTER_CHUNK)

    # Cerrar conexiones
    for conn in conns.values():
        try: conn.close()
        except Exception: pass

    return stats


def main():
    parser = argparse.ArgumentParser(description="Refresh fundamentales desde yahooquery")
    parser.add_argument("--target", choices=["railway", "local", "both"],
                        default="local", help="Donde escribir (default: local)")
    parser.add_argument("--tickers", default=None,
                        help="Lista CSV de tickers a procesar (default: ALL_TICKERS)")
    parser.add_argument("--lookback-q", type=int, default=LOOKBACK_Q_DEFAULT,
                        help=f"N trimestres recientes a guardar (default {LOOKBACK_Q_DEFAULT})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT,
                        help=f"Tickers por chunk async (default {CHUNK_SIZE_DEFAULT})")
    parser.add_argument("--dry-run", action="store_true", help="Simula sin escribir")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = sorted(set(ALL_TICKERS))

    targets = ["local", "railway"] if args.target == "both" else [args.target]

    print()
    print(SEP)
    print(f"  REFRESH fundamentales  |  target={args.target}  "
          f"tickers={len(tickers)}  lookback_q={args.lookback_q}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    if not args.dry_run:
        acquire_yf_lock(f"refresh_fundamentales (n={len(tickers)})")

    t0 = time.time()
    stats = procesar(
        tickers=tickers,
        lookback_q=args.lookback_q,
        chunk_size=args.chunk_size,
        targets=targets,
        dry_run=args.dry_run,
    )
    elapsed = time.time() - t0

    print()
    print(SEP)
    print("  RESUMEN:")
    print(f"    tickers           : {stats['tickers_total']}")
    print(f"    con data          : {stats['tickers_con_data']}")
    print(f"    sin data          : {stats['tickers_sin_data']}")
    print(f"    chunks OK / FAIL  : {stats['chunks_ok']} / {stats['chunks_fail']}")
    print(f"    filas income      : {stats['filas_income']}")
    print(f"    filas balance     : {stats['filas_balance']}")
    print(f"    filas cashflow    : {stats['filas_cashflow']}")
    print(f"    filas valuation   : {stats['filas_valuation']}")
    print(f"    tiempo total      : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(SEP)
    print()

    sys.exit(1 if stats["chunks_fail"] > 0 else 0)


if __name__ == "__main__":
    main()
