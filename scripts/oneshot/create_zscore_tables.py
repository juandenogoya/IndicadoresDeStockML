"""
create_zscore_tables.py
Crea las tablas de Z-scores y opcionalmente hace backfill historico.

Tablas creadas:
    opciones_zscore_diario  -- Z-scores de volumen y sentimiento de opciones
    ticker_zscore_diario    -- Z-scores de volumen y retorno de acciones

Uso:
    python scripts/migrations/create_zscore_tables.py             # solo crea tablas
    python scripts/migrations/create_zscore_tables.py --backfill  # crea + backfill historico
    python scripts/migrations/create_zscore_tables.py --status    # estado actual de ambas tablas
    python scripts/migrations/create_zscore_tables.py --dry-run   # muestra DDL sin ejecutar

Tiempos estimados para backfill:
    ticker_zscore_diario   : ~10-30 seg (SQL window functions, 2 anos de historia)
    opciones_zscore_diario : <5 seg    (solo dias habiles disponibles, <10 fechas al inicio)
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from sqlalchemy import text
from src.data.database import get_engine
from src.utils.zscore_pipeline import (
    DDL_OPCIONES_ZSCORE, DDL_TICKER_ZSCORE, init_tablas,
    backfill_zscore_tickers, backfill_zscore_opciones,
)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cmd_status():
    engine = get_engine()

    log("=" * 60)
    log("  ESTADO TABLAS Z-SCORE")
    log("=" * 60)

    with engine.connect() as conn:
        for tabla in ("ticker_zscore_diario", "opciones_zscore_diario"):
            # Verificar si existe
            existe = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = :t
            """), {"t": tabla}).scalar()

            if not existe:
                log(f"  {tabla}: NO EXISTE")
                continue

            # Estadisticas basicas
            row = conn.execute(text(f"""
                SELECT
                    COUNT(*)                         AS total_filas,
                    COUNT(DISTINCT ticker)           AS tickers,
                    COUNT(DISTINCT fecha)            AS fechas,
                    MIN(fecha)                       AS desde,
                    MAX(fecha)                       AS hasta,
                    AVG(ventana_dias)                AS ventana_avg
                FROM {tabla}
            """)).fetchone()

            log(f"")
            log(f"  {tabla}:")
            log(f"    Total filas   : {row[0]:,}")
            log(f"    Tickers       : {row[1]}")
            log(f"    Fechas        : {row[2]}")
            log(f"    Rango         : {row[3]} -> {row[4]}")
            log(f"    Ventana avg   : {float(row[5]):.1f} dias" if row[5] else "    Ventana avg   : N/A")

            # Top 5 Z-scores del ultimo dia
            col_z = "vol_total_zscore" if tabla == "opciones_zscore_diario" else "vol_zscore"
            top = conn.execute(text(f"""
                SELECT ticker, sector, {col_z}, vol_relativo, ventana_dias
                FROM   {tabla}
                WHERE  fecha = (SELECT MAX(fecha) FROM {tabla})
                  AND  {col_z} IS NOT NULL
                ORDER  BY {col_z} DESC
                LIMIT  5
            """)).fetchall()

            if top:
                fecha_top = conn.execute(text(
                    f"SELECT MAX(fecha) FROM {tabla}"
                )).scalar()
                log(f"    Top 5 por Z-score ({fecha_top}):")
                for r in top:
                    log(f"      {r[0]:<8s}  sector={r[1] or 'N/A':<30s}  "
                        f"Z={float(r[2]):+.2f}  RelVol={float(r[3]):.2f}x  "
                        f"ventana={r[4]}d")

    log("")


def cmd_create(dry_run: bool = False):
    """Crea las dos tablas si no existen."""
    if dry_run:
        log("  [DRY RUN] DDL que se ejecutaria:")
        print()
        print("-- ticker_zscore_diario:")
        print(DDL_TICKER_ZSCORE)
        print("-- opciones_zscore_diario:")
        print(DDL_OPCIONES_ZSCORE)
        return

    engine = get_engine()
    init_tablas(engine)
    log("  Tablas ticker_zscore_diario y opciones_zscore_diario listas.")


def cmd_backfill(dry_run: bool = False):
    """Backfill historico de ambas tablas usando SQL window functions."""
    engine = get_engine()

    if dry_run:
        # Contar cuantas filas se insertarian
        with engine.connect() as conn:
            n_precios = conn.execute(text(
                "SELECT COUNT(DISTINCT fecha) FROM precios_diarios WHERE volume > 0"
            )).scalar()
            n_opciones = conn.execute(text(
                "SELECT COUNT(DISTINCT fecha) FROM opciones_resumen_diario"
            )).scalar()
        log(f"  [DRY RUN] ticker_zscore_diario   : {n_precios} fechas en precios_diarios")
        log(f"  [DRY RUN] opciones_zscore_diario : {n_opciones} fechas en opciones_resumen_diario")
        log("  Ejecutar sin --dry-run para proceder.")
        return

    # Primero asegurarse que las tablas existen
    init_tablas(engine)

    log("  Backfill ticker_zscore_diario ...")
    t0 = datetime.now()
    n_tickers = backfill_zscore_tickers(engine)
    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()
    log(f"    -> {n_tickers:,} filas en {elapsed:.1f}s")

    log("  Backfill opciones_zscore_diario ...")
    t0 = datetime.now()
    n_opciones = backfill_zscore_opciones(engine)
    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()
    log(f"    -> {n_opciones:,} filas en {elapsed:.1f}s")

    log("")
    log(f"  Backfill completado:")
    log(f"    ticker_zscore_diario   : {n_tickers:,} filas")
    log(f"    opciones_zscore_diario : {n_opciones:,} filas")


def main():
    parser = argparse.ArgumentParser(
        description="Crea tablas Z-score y hace backfill historico"
    )
    parser.add_argument("--backfill", action="store_true",
                        help="Poblar con datos historicos via SQL window functions")
    parser.add_argument("--status",   action="store_true",
                        help="Mostrar estado actual de ambas tablas")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Mostrar que se haria sin ejecutar")
    args = parser.parse_args()

    log("=" * 60)
    log("  CREATE ZSCORE TABLES")
    log("=" * 60)

    if args.status:
        cmd_status()
        return

    if args.backfill:
        cmd_backfill(dry_run=args.dry_run)
        return

    # Default: solo crear tablas
    cmd_create(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
