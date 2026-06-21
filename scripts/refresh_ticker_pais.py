"""
refresh_ticker_pais.py
Trae el pais de cada ticker (yahooquery assetProfile.country) y lo guarda en
la tabla ticker_pais (LOCAL), con la region derivada (USA/Europa/China/Resto).

Motivacion:
    El analisis sectorial fundamental debe comparar pares de la MISMA region
    para no mezclar prima de riesgo pais (un ADR chino cotiza a PER menor por
    riesgo, no por estar "barato"). reporting_currency es un proxy imperfecto
    (empresas extranjeras que reportan en USD caerian mal clasificadas), asi
    que tomamos el country real de assetProfile.

Diseno:
    - Tabla ticker_pais (ticker PK, country, region, fetched_at). Aislada,
      local-only (Plan C: dato recuperable). No toca la tabla activos.
    - region: mapeo coarse USA / Europa / China / Resto. El mapeo vive en
      codigo (REGION_MAP) -> se re-deriva re-corriendo el script si cambia.
    - La granularidad fina (curaduria por pais) se hace downstream filtrando
      por country; aca solo damos el bucket grueso por defecto.

Uso:
    python scripts/refresh_ticker_pais.py
    python scripts/refresh_ticker_pais.py --dry-run
    python scripts/refresh_ticker_pais.py --tickers AAPL,BABA,SAP
"""

import sys
import os
import argparse
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import psycopg2
import psycopg2.extras
from sqlalchemy import text

from src.data.universo import get_universo
from src.utils.yfinance_lock import acquire as acquire_yf_lock
from scripts.oneshot.create_fundamentales_tables import (
    get_local_engine, _parse_env_file,
)

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -- Mapeo pais -> region (coarse: USA / Europa / China / Resto) ---------------

EUROPA = {
    "United Kingdom", "Germany", "France", "Netherlands", "Switzerland",
    "Sweden", "Denmark", "Norway", "Finland", "Ireland", "Spain", "Italy",
    "Belgium", "Austria", "Portugal", "Luxembourg", "Poland",
}
CHINA = {"China", "Hong Kong"}
USA = {"United States"}


def map_region(country):
    if not country:
        return None
    if country in USA:
        return "USA"
    if country in CHINA:
        return "China"
    if country in EUROPA:
        return "Europa"
    return "Resto"   # Brazil, Mexico, Argentina, Japan, Taiwan, Canada, Israel...


DDL = """
CREATE TABLE IF NOT EXISTS ticker_pais (
    ticker      VARCHAR(20) PRIMARY KEY,
    country     VARCHAR(60),
    region      VARCHAR(20),
    fetched_at  TIMESTAMP NOT NULL DEFAULT NOW()
)
"""
DDL_IDX = "CREATE INDEX IF NOT EXISTS idx_ticker_pais_region ON ticker_pais (region)"


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_countries(tickers):
    """Devuelve dict ticker -> country usando assetProfile (chunks async)."""
    from yahooquery import Ticker
    out = {}
    for ch in _chunks(tickers, 40):
        t = Ticker(ch, asynchronous=True, max_workers=8)
        prof = t.asset_profile
        for tk in ch:
            d = prof.get(tk)
            out[tk] = d.get("country") if isinstance(d, dict) else None
        time.sleep(1.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Refresh ticker_pais (country+region) local")
    parser.add_argument("--tickers", default=None, help="CSV (default: universo de tabla activos)")
    parser.add_argument("--dry-run", action="store_true", help="No escribe")
    args = parser.parse_args()

    tickers = ([t.strip().upper() for t in args.tickers.split(",")]
               if args.tickers else get_universo())

    print()
    print(SEP)
    print(f"  REFRESH ticker_pais  |  tickers={len(tickers)}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    if not args.dry_run:
        acquire_yf_lock(f"refresh_ticker_pais (n={len(tickers)})")

    eng = get_local_engine()
    if not args.dry_run:
        with eng.connect() as c:
            c.execute(text(DDL.strip()))
            c.execute(text(DDL_IDX))
            c.commit()

    log("Trayendo assetProfile.country...")
    countries = fetch_countries(tickers)

    rows = []
    sin_pais = []
    for tk in tickers:
        country = countries.get(tk)
        region = map_region(country)
        if not country:
            sin_pais.append(tk)
        rows.append({"ticker": tk, "country": country, "region": region})

    # Resumen por region/pais
    df = pd.DataFrame(rows)
    log(f"Paises distintos: {df['country'].nunique(dropna=True)} | sin pais: {len(sin_pais)}")
    by_region = df.groupby("region", dropna=False).size().to_dict()
    log(f"Por region: {by_region}")
    if sin_pais:
        log(f"Sin pais (region NULL): {sin_pais}")

    if args.dry_run:
        log("DRY-RUN: no se escribe.")
    else:
        env = _parse_env_file(os.path.join(ROOT, ".env"))
        conn = psycopg2.connect(
            host=env.get("DB_HOST", "localhost"), port=int(env.get("DB_PORT", 5432)),
            dbname=env.get("DB_NAME", "activos_ml"), user=env.get("DB_USER", "postgres"),
            password=env.get("DB_PASSWORD", ""),
        )
        try:
            sql = ("INSERT INTO ticker_pais (ticker, country, region) "
                   "VALUES (%(ticker)s, %(country)s, %(region)s) "
                   "ON CONFLICT (ticker) DO UPDATE SET "
                   "country=EXCLUDED.country, region=EXCLUDED.region, fetched_at=NOW()")
            cur = conn.cursor()
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
            conn.commit()
            log(f"UPSERT local: {len(rows)} filas.")
        finally:
            conn.close()

    print()
    print(SEP)
    print(f"  OK  |  tickers: {len(tickers)}  |  sin pais: {len(sin_pais)}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
