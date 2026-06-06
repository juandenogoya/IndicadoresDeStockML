"""
compute_opciones_derivadas.py
Computa las tablas DERIVADAS de opciones en LOCAL desde el crudo (opciones_snapshot).

Parte de la migracion "snapshot nube solo-crudo" (AGENDA Tarea 17): la nube captura
solo el crudo (opciones_snapshot); este script computa en LOCAL todo lo derivado para
una fecha, desde el crudo ya synced:
  1. HV (hv_20d)              desde precios_diarios LOCAL -> UPDATE opciones_snapshot
  2. opciones_resumen_diario  (recompute desde el crudo, SQL agregado)
  3. opciones_zscore_diario + opciones_sector_zscore_diario   (src.utils.zscore_pipeline)
  4. opciones_pcr_plazo_diario + opciones_sector_pcr_plazo_diario (src.utils.opciones_plazo)

Target SIEMPRE local: NO carga .env.local, usa get_local_engine() (lee .env directo).
Todas las funciones reciben el engine local explicito. Idempotente (UPSERT / UPDATE).

Uso:
    python scripts/compute_opciones_derivadas.py                 # ultima fecha en snapshot local
    python scripts/compute_opciones_derivadas.py --fecha 2026-06-05
"""

import sys
import os
import math
import argparse
from collections import defaultdict
from datetime import date, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import text
# get_local_engine arma el engine LOCAL leyendo .env directo, sin tocar os.environ
# (no carga .env.local -> DATABASE_URL queda sin setear -> todo apunta a local).
from scripts.migrations.sync_railway_to_local import get_local_engine

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── 1. HV ──────────────────────────────────────────────────────────────────────

def computar_hv(engine, fecha: date) -> tuple[int, int]:
    """
    Computa hv_20d (vol historica anualizada 20d) por ticker desde precios_diarios
    LOCAL, anclada en la fecha del snapshot (no CURRENT_DATE -> backfilleable).
    Hace UPDATE opciones_snapshot SET hv_20d para todas las filas de esa fecha/ticker.
    Retorna (tickers_con_hv, filas_actualizadas).
    """
    with engine.connect() as conn:
        tickers = [r[0] for r in conn.execute(text(
            "SELECT DISTINCT ticker FROM opciones_snapshot WHERE fecha_snapshot = :f"
        ), {"f": fecha}).fetchall()]
        if not tickers:
            return 0, 0
        rows = conn.execute(text("""
            SELECT ticker, close
            FROM   precios_diarios
            WHERE  ticker = ANY(:tickers)
              AND  close > 0
              AND  fecha <= CAST(:f AS date)
              AND  fecha >= CAST(:f AS date) - INTERVAL '40 days'
            ORDER  BY ticker, fecha ASC
        """), {"tickers": tickers, "f": fecha}).fetchall()

    closes = defaultdict(list)
    for t, c in rows:
        closes[t].append(float(c))

    hv_map: dict[str, float] = {}
    for t, cl in closes.items():
        if len(cl) < 5:
            continue
        c = cl[-21:]
        lr = [math.log(c[i] / c[i - 1]) for i in range(1, len(c))]
        n = len(lr)
        mean = sum(lr) / n
        var = sum((r - mean) ** 2 for r in lr) / (n - 1) if n > 1 else 0.0
        hv_map[t] = round(math.sqrt(var) * math.sqrt(252), 6)

    filas = 0
    with engine.begin() as conn:
        for t, hv in hv_map.items():
            r = conn.execute(text(
                "UPDATE opciones_snapshot SET hv_20d = :hv "
                "WHERE ticker = :t AND fecha_snapshot = :f"
            ), {"hv": hv, "t": t, "f": fecha})
            filas += r.rowcount or 0
    return len(hv_map), filas


# ── 2. resumen_diario (recompute desde el crudo) ───────────────────────────────

# Mismo agregado que cmd_backfill_resumen() de 33_opciones_snapshot.py, por :fecha.
SQL_RESUMEN = """
    WITH oi_per_key AS (
        SELECT fecha_snapshot, ticker, strike, vencimiento,
               SUM(COALESCE(open_interest, 0)) AS total_oi
        FROM   opciones_snapshot
        WHERE  fecha_snapshot = :fecha
        GROUP  BY fecha_snapshot, ticker, strike, vencimiento
    ),
    top_strike AS (
        SELECT DISTINCT ON (ticker)
               ticker, strike AS max_oi_strike, vencimiento AS max_oi_venc
        FROM   oi_per_key
        ORDER  BY ticker, total_oi DESC
    ),
    agg AS (
        SELECT
            fecha_snapshot AS fecha, ticker,
            SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END)       AS call_vol,
            SUM(CASE WHEN tipo='put'  THEN COALESCE(volumen,0) ELSE 0 END)       AS put_vol,
            SUM(CASE WHEN tipo='call' THEN COALESCE(open_interest,0) ELSE 0 END) AS call_oi,
            SUM(CASE WHEN tipo='put'  THEN COALESCE(open_interest,0) ELSE 0 END) AS put_oi,
            CASE WHEN SUM(CASE WHEN tipo='call' AND iv IS NOT NULL THEN COALESCE(open_interest,0) ELSE 0 END) > 0
                 THEN ROUND(SUM(CASE WHEN tipo='call' AND iv IS NOT NULL THEN iv*COALESCE(open_interest,0) ELSE 0 END)
                       / SUM(CASE WHEN tipo='call' AND iv IS NOT NULL THEN COALESCE(open_interest,0) ELSE 0 END)::NUMERIC, 6)
                 ELSE NULL END AS iv_call_avg,
            CASE WHEN SUM(CASE WHEN tipo='put' AND iv IS NOT NULL THEN COALESCE(open_interest,0) ELSE 0 END) > 0
                 THEN ROUND(SUM(CASE WHEN tipo='put' AND iv IS NOT NULL THEN iv*COALESCE(open_interest,0) ELSE 0 END)
                       / SUM(CASE WHEN tipo='put' AND iv IS NOT NULL THEN COALESCE(open_interest,0) ELSE 0 END)::NUMERIC, 6)
                 ELSE NULL END AS iv_put_avg,
            COUNT(*)               AS n_contratos,
            MAX(precio_subyacente) AS precio_sub
        FROM opciones_snapshot
        WHERE fecha_snapshot = :fecha
        GROUP BY fecha_snapshot, ticker
    )
    INSERT INTO opciones_resumen_diario (
        fecha, ticker, call_vol, put_vol, pcr_vol, call_oi, put_oi, pcr_oi,
        iv_call_avg, iv_put_avg, n_contratos, max_oi_strike, max_oi_venc, precio_sub
    )
    SELECT a.fecha, a.ticker, a.call_vol, a.put_vol,
           CASE WHEN a.call_vol > 0 THEN ROUND(a.put_vol::NUMERIC/a.call_vol, 4) ELSE NULL END,
           a.call_oi, a.put_oi,
           CASE WHEN a.call_oi > 0 THEN ROUND(a.put_oi::NUMERIC/a.call_oi, 4) ELSE NULL END,
           a.iv_call_avg, a.iv_put_avg, a.n_contratos, t.max_oi_strike, t.max_oi_venc, a.precio_sub
    FROM agg a LEFT JOIN top_strike t USING (ticker)
    ON CONFLICT (fecha, ticker) DO UPDATE SET
        call_vol=EXCLUDED.call_vol, put_vol=EXCLUDED.put_vol, pcr_vol=EXCLUDED.pcr_vol,
        call_oi=EXCLUDED.call_oi, put_oi=EXCLUDED.put_oi, pcr_oi=EXCLUDED.pcr_oi,
        iv_call_avg=EXCLUDED.iv_call_avg, iv_put_avg=EXCLUDED.iv_put_avg,
        n_contratos=EXCLUDED.n_contratos, max_oi_strike=EXCLUDED.max_oi_strike,
        max_oi_venc=EXCLUDED.max_oi_venc, precio_sub=EXCLUDED.precio_sub
"""


def computar_resumen(engine, fecha: date) -> int:
    with engine.begin() as conn:
        r = conn.execute(text(SQL_RESUMEN), {"fecha": fecha})
        return r.rowcount or 0


# ── Main ────────────────────────────────────────────────────────────────────────

def run(fecha: date = None):
    engine = get_local_engine()

    if fecha is None:
        with engine.connect() as conn:
            fecha = conn.execute(text("SELECT MAX(fecha_snapshot) FROM opciones_snapshot")).scalar()
    if fecha is None:
        log("No hay datos en opciones_snapshot local. Nada que computar.")
        return

    print()
    print(SEP)
    print(f"  COMPUTE OPCIONES DERIVADAS (LOCAL)  |  fecha = {fecha}")
    print(SEP)

    # 1. HV
    try:
        n_hv, n_filas = computar_hv(engine, fecha)
        log(f"  HV_20d            : {n_hv} tickers -> UPDATE {n_filas:,} filas snapshot")
    except Exception as e:
        log(f"  [ERROR] HV: {e}")

    # 2. resumen
    try:
        n_res = computar_resumen(engine, fecha)
        log(f"  resumen_diario    : {n_res} tickers")
    except Exception as e:
        log(f"  [ERROR] resumen: {e}")

    # 3. zscore + sector
    try:
        from src.utils.zscore_pipeline import (
            calcular_zscore_opciones, calcular_zscore_opciones_sector, init_tablas
        )
        init_tablas(engine)
        n_z = calcular_zscore_opciones(fecha, engine)
        log(f"  zscore opciones   : {n_z} tickers")
        n_zs = calcular_zscore_opciones_sector(fecha, engine)
        log(f"  zscore sector     : {n_zs} sectores")
    except Exception as e:
        log(f"  [ERROR] zscore: {e}")

    # 4. pcr_plazo + sector
    try:
        from src.utils.opciones_plazo import (
            calcular_pcr_plazo, calcular_pcr_sector_plazo, init_tabla, init_tabla_sector
        )
        init_tabla(engine)
        init_tabla_sector(engine)
        n_p = calcular_pcr_plazo(fecha, engine)
        log(f"  pcr_plazo         : {n_p} filas")
        n_ps = calcular_pcr_sector_plazo(fecha, engine)
        log(f"  pcr_plazo sector  : {n_ps} filas")
    except Exception as e:
        log(f"  [ERROR] pcr_plazo: {e}")

    print(SEP)
    log("  Completado.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Computa derivadas de opciones en LOCAL desde el crudo")
    parser.add_argument("--fecha", help="YYYY-MM-DD (default: ultima fecha en opciones_snapshot local)")
    args = parser.parse_args()
    fecha = date.fromisoformat(args.fecha) if args.fecha else None
    run(fecha)


if __name__ == "__main__":
    main()
