"""
compute_fundamentales_sector.py
Computa fundamentales_ticker_vs_sector (comparativa ticker vs sector/region).

Funcion PURA: lee fundamentales_ratios_q (ultimo Q por ticker) + ticker_pais,
arma peer-sets por (sector, region), y deriva para cada ticker x metrica:
mediana/p25/p75 del sector, distancia a la mediana, percentil y flags. UPSERT
en fundamentales_ticker_vs_sector.

Politica de peer-set (umbral N=5, ver create_fundamentales_sector_table.py):
    1. (sector, region_ticker) con n>=5  -> basis='region'
    2. no-USA en bucket chico + sector USA n>=5 -> basis='usa_fallback'
    3. resto -> basis='none' (leyenda, sin benchmark)

Parametrizable (curaduria del usuario):
    --regions USA,Europa  -> restringe los PARES a esas regiones (el ticker se
    compara solo contra pares de las regiones elegidas). Default: politica de
    3 niveles de arriba. Esto permite "ver el PER de SAP vs solo-Europa" o
    "vs USA+Europa", etc.

Snapshot: ultimo Q de cada ticker (earnings escalonados -> fechas pueden
diferir unos dias). fiscal_period_end = ultimo Q del ticker.

Uso:
    python scripts/compute_fundamentales_sector.py
    python scripts/compute_fundamentales_sector.py --dry-run
    python scripts/compute_fundamentales_sector.py --regions USA,Europa
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
MIN_PEERS = 5          # umbral para bucket "robusto"
MIN_PEERS_METRIC = 3   # minimo de valores no-null para emitir estadisticos

METRICS = [
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe_ttm", "roa_ttm", "roic_ttm",
    "net_margin_ttm", "operating_margin_ttm", "revenue_yoy_pct",
]

OUT_COLS = [
    "ticker", "fiscal_period_end", "sector", "ticker_region", "peer_region",
    "peer_basis", "metric", "value", "peer_median", "peer_p25", "peer_p75",
    "vs_median_pct", "percentile", "peer_n", "low_sample",
]
PK_COLS = ["ticker", "fiscal_period_end", "metric"]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load(engine):
    """Ultimo Q por ticker de ratios + region (ticker_pais)."""
    sql = """
        WITH ult AS (
          SELECT DISTINCT ON (ticker) *
          FROM fundamentales_ratios_q
          ORDER BY ticker, fiscal_period_end DESC
        )
        SELECT u.*, p.region
        FROM ult u
        LEFT JOIN ticker_pais p ON p.ticker = u.ticker
    """
    df = pd.read_sql(sql, engine)
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"]).dt.date
    return df


def _stats(value, peers: pd.Series):
    """(median, p25, p75, vs_median_pct, percentile, n) sobre peers no-null."""
    s = pd.to_numeric(peers, errors="coerce").dropna()
    n = len(s)
    if n < MIN_PEERS_METRIC:
        return None, None, None, None, None, n
    med = float(s.median())
    p25 = float(s.quantile(0.25))
    p75 = float(s.quantile(0.75))
    vs = None
    if value is not None and not pd.isna(value) and med != 0:
        vs = (float(value) - med) / abs(med)
    pct = None
    if value is not None and not pd.isna(value):
        pct = float((s <= float(value)).sum()) / n
    return med, p25, p75, vs, pct, n


def _peer_values(df, sector, region, metric):
    """Serie de valores de la metrica para (sector, region)."""
    sub = df[(df["sector"] == sector) & (df["region"] == region)]
    return sub[metric]


def _bucket_size(df, sector, region):
    return int(((df["sector"] == sector) & (df["region"] == region)).sum())


def compute(df, regions_filter=None):
    """Devuelve list of dicts para UPSERT."""
    rows = []
    # Precalcular tamanos de bucket por (sector, region)
    for _, r in df.iterrows():
        tk = r["ticker"]
        sector = r["sector"]
        treg = r["region"]
        fpe = r["fiscal_period_end"]

        # -- Determinar peer-set y basis --
        if regions_filter:
            # Modo curaduria: pares = mismo sector, regiones elegidas
            peer_mask = (df["sector"] == sector) & (df["region"].isin(regions_filter))
            peer_df = df[peer_mask]
            basis = "custom"
            preg = "+".join(regions_filter)
            bucket_ok = len(peer_df) >= MIN_PEERS
        else:
            # Politica 3 niveles
            if sector and treg and _bucket_size(df, sector, treg) >= MIN_PEERS:
                peer_df = df[(df["sector"] == sector) & (df["region"] == treg)]
                basis, preg, bucket_ok = "region", treg, True
            elif (treg and treg != "USA" and sector
                  and _bucket_size(df, sector, "USA") >= MIN_PEERS):
                peer_df = df[(df["sector"] == sector) & (df["region"] == "USA")]
                basis, preg, bucket_ok = "usa_fallback", "USA", True
            else:
                peer_df, basis, preg, bucket_ok = None, "none", None, False

        for metric in METRICS:
            value = r.get(metric)
            value = None if pd.isna(value) else float(value)
            if not bucket_ok or peer_df is None:
                rows.append({
                    "ticker": tk, "fiscal_period_end": fpe, "sector": sector,
                    "ticker_region": treg, "peer_region": preg, "peer_basis": basis,
                    "metric": metric, "value": value, "peer_median": None,
                    "peer_p25": None, "peer_p75": None, "vs_median_pct": None,
                    "percentile": None, "peer_n": 0, "low_sample": True,
                })
                continue
            med, p25, p75, vs, pct, n = _stats(value, peer_df[metric])
            rows.append({
                "ticker": tk, "fiscal_period_end": fpe, "sector": sector,
                "ticker_region": treg, "peer_region": preg, "peer_basis": basis,
                "metric": metric, "value": value, "peer_median": med,
                "peer_p25": p25, "peer_p75": p75, "vs_median_pct": vs,
                "percentile": pct, "peer_n": n,
                "low_sample": (n < MIN_PEERS_METRIC),
            })
    return rows


def _clean(rows):
    out = []
    for r in rows:
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
        out.append(cr)
    return out


def _upsert(env, rows):
    if not rows:
        return 0
    ph = ", ".join([f"%({c})s" for c in OUT_COLS])
    upd = [c for c in OUT_COLS if c not in PK_COLS]
    setc = ", ".join([f"{c}=EXCLUDED.{c}" for c in upd]) + ", computed_at=NOW()"
    sql = (f"INSERT INTO fundamentales_ticker_vs_sector ({', '.join(OUT_COLS)}) "
           f"VALUES ({ph}) ON CONFLICT (ticker, fiscal_period_end, metric) "
           f"DO UPDATE SET {setc}")
    conn = psycopg2.connect(
        host=env.get("DB_HOST", "localhost"), port=int(env.get("DB_PORT", 5432)),
        dbname=env.get("DB_NAME", "activos_ml"), user=env.get("DB_USER", "postgres"),
        password=env.get("DB_PASSWORD", ""))
    try:
        cur = conn.cursor()
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


# Metricas de valuacion que dependen del precio -> tienen variante *_px (cierre
# del dia). Las demas (ROE/ROA/margenes/...) no dependen del precio.
_VALUACION_PX = {
    "pe_ratio": "pe_ratio_px", "pb_ratio": "pb_ratio_px",
    "ps_ratio": "ps_ratio_px", "ev_ebitda": "ev_ebitda_px",
}


def compute_sector_valuacion_px(engine) -> int:
    """
    Recalcula SOLO el comparativo sectorial de las 4 metricas de valuacion usando
    los multiplos al CIERRE del dia (*_px de fundamentales_ratios_q). Pisa esas 4
    filas (metric pe_ratio/pb_ratio/ps_ratio/ev_ebitda) en fundamentales_ticker_vs_sector
    con value+mediana al precio actual; el resto de metricas (ROE/margenes/...) no se toca.

    Pensada para correr a diario tras compute_multiplos_px (recovery_incremental).
    Reusa la logica de peer-set/mediana de compute(): se pisan las columnas base
    con sus *_px y se filtran las filas de valuacion. Retorna filas UPSERTeadas.
    """
    df = _load(engine)  # trae tambien las *_px (SELECT u.*)
    for base, px in _VALUACION_PX.items():
        if px in df.columns:
            df[base] = df[px]   # usar el multiplo al cierre como la metrica
    rows = [r for r in compute(df, regions_filter=None) if r["metric"] in _VALUACION_PX]
    rows = _clean(rows)
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    return _upsert(env, rows)


def main():
    parser = argparse.ArgumentParser(description="Computa fundamentales_ticker_vs_sector")
    parser.add_argument("--regions", default=None,
                        help="CSV de regiones para peer-set (curaduria). Ej: USA,Europa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--valuacion-px", action="store_true",
                        help="Solo pisa las 4 metricas de valuacion con los *_px "
                             "(cierre del dia). Correr DESPUES de compute_multiplos_px.")
    args = parser.parse_args()

    # Modo acotado: recalcular SOLO las 4 metricas de valuacion con los *_px.
    # Es el paso que cierra la cadena del refresh (y el que corre a diario dentro
    # de recovery_incremental). Sin esto, vs_sector queda con los multiplos
    # FISCALES (precio congelado en la fecha del balance).
    if args.valuacion_px:
        eng = get_local_engine()
        log("Pisando metricas de valuacion con los *_px (cierre del dia)...")
        n = compute_sector_valuacion_px(eng)
        print()
        print(SEP)
        print(f"  OK  |  valuacion *_px  |  filas: {n}")
        print(SEP)
        print()
        return

    regions_filter = None
    if args.regions:
        regions_filter = [x.strip() for x in args.regions.split(",") if x.strip()]

    print()
    print(SEP)
    print(f"  COMPUTE fundamentales_ticker_vs_sector"
          f"{'  [regions='+','.join(regions_filter)+']' if regions_filter else '  [politica 3-niveles]'}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    eng = get_local_engine()
    log("Leyendo ratios (ultimo Q) + ticker_pais...")
    df = _load(eng)
    log(f"  tickers: {len(df)} | con region: {df['region'].notna().sum()}")

    rows = compute(df, regions_filter)
    rows = _clean(rows)

    # Resumen por basis
    basis_count = {}
    for r in rows:
        b = r["peer_basis"]
        basis_count[b] = basis_count.get(b, 0) + 1
    n_tickers = df["ticker"].nunique()
    log(f"Filas: {len(rows)} ({n_tickers} tickers x {len(METRICS)} metricas)")
    log(f"Por basis (filas): {basis_count}")

    if args.dry_run:
        # Muestra: AAPL, SAP, BABA, ITUB con su PER vs sector
        sample_tk = ["AAPL", "SAP", "BABA", "ITUB", "NVDA"]
        log("DRY-RUN muestra (pe_ratio vs sector):")
        for r in rows:
            if r["ticker"] in sample_tk and r["metric"] == "pe_ratio":
                vm = r["vs_median_pct"]
                vm_s = f"{vm*100:+.0f}%" if vm is not None else "n/a"
                log(f"  {r['ticker']:5s} [{r['sector']}/{r['peer_region']} "
                    f"basis={r['peer_basis']} n={r['peer_n']}] "
                    f"PER={r['value']} vs med={r['peer_median']} ({vm_s} vs mediana)")
    else:
        env = _parse_env_file(os.path.join(ROOT, ".env"))
        n = _upsert(env, rows)
        log(f"UPSERT local: {n} filas.")

    print()
    print(SEP)
    print(f"  OK  |  tickers: {n_tickers}  |  filas: {len(rows)}  |  basis: {basis_count}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
