"""
add_industry_activos.py
Agrega la columna 'industry' a la tabla activos y la puebla desde yfinance.

Uso:
    python scripts/migrations/add_industry_activos.py           # todos los tickers
    python scripts/migrations/add_industry_activos.py --dry-run # muestra sin guardar
    python scripts/migrations/add_industry_activos.py --status  # muestra estado actual
"""

import sys
import os
import argparse
import time
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

import yfinance as yf
from sqlalchemy import text
from src.data.database import get_engine


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        # Verificar si la columna existe
        col_exists = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'activos' AND column_name = 'industry'
        """)).scalar()

        if not col_exists:
            log("Columna 'industry' NO existe en activos.")
            return

        rows = conn.execute(text("""
            SELECT industry, COUNT(*) AS n,
                   STRING_AGG(ticker, ', ' ORDER BY ticker) AS tickers
            FROM activos
            WHERE activo = true
            GROUP BY industry
            ORDER BY n DESC
        """)).fetchall()

        null_count = conn.execute(text(
            "SELECT COUNT(*) FROM activos WHERE industry IS NULL AND activo = true"
        )).scalar()

    print()
    print("  === INDUSTRIAS en activos ===")
    print(f"  {'INDUSTRY':<40s}  {'N':>3s}  TICKERS")
    print("  " + "-" * 80)
    for r in rows:
        ind = str(r[0]) if r[0] else "NULL"
        print(f"  {ind:<40s}  {r[1]:>3d}  {r[2]}")
    print()
    print(f"  Sin industry (NULL): {null_count}")
    print()


def cmd_run(dry_run: bool = False):
    engine = get_engine()

    # 1. Agregar columna si no existe
    with engine.connect() as conn:
        col_exists = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'activos' AND column_name = 'industry'
        """)).scalar()

        if not col_exists:
            if dry_run:
                log("  [DRY RUN] ALTER TABLE activos ADD COLUMN industry VARCHAR(100)")
            else:
                conn.execute(text(
                    "ALTER TABLE activos ADD COLUMN industry VARCHAR(100)"
                ))
                conn.commit()
                log("  Columna 'industry' agregada a activos.")
        else:
            log("  Columna 'industry' ya existe.")

    # 2. Obtener tickers activos
    with engine.connect() as conn:
        col_exists_now = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'activos' AND column_name = 'industry'
        """)).scalar()

        if col_exists_now:
            rows = conn.execute(text(
                "SELECT ticker, sector, industry FROM activos WHERE activo = true ORDER BY ticker"
            )).fetchall()
        else:
            rows = conn.execute(text(
                "SELECT ticker, sector, NULL FROM activos WHERE activo = true ORDER BY ticker"
            )).fetchall()

    log(f"  Tickers a procesar: {len(rows)}")
    log("=" * 55)

    resultados = []
    errores = []

    for i, (ticker, sector, industry_actual) in enumerate(rows, 1):
        try:
            info = yf.Ticker(ticker).info
            industry_nueva = info.get("industry") or info.get("industryKey") or None

            status = ""
            if industry_nueva:
                if industry_actual != industry_nueva:
                    status = f"{industry_actual} -> {industry_nueva}" if industry_actual else f"(nuevo) {industry_nueva}"
                else:
                    status = "sin cambio"
            else:
                status = "no encontrado en yfinance"
                industry_nueva = industry_actual  # mantener valor existente

            log(f"  [{i:>3d}/{len(rows)}] {ticker:<8s}  {status}")
            resultados.append((ticker, industry_nueva))
            time.sleep(0.3)

        except Exception as e:
            log(f"  [{i:>3d}/{len(rows)}] {ticker:<8s}  ERROR: {e}")
            errores.append(ticker)
            resultados.append((ticker, None))

    # 3. Actualizar en bulk
    if not dry_run:
        actualizados = 0
        with engine.connect() as conn:
            for ticker, industry in resultados:
                if industry is not None:
                    conn.execute(text(
                        "UPDATE activos SET industry = :industry WHERE ticker = :ticker"
                    ), {"industry": industry, "ticker": ticker})
                    actualizados += 1
            conn.commit()
        log("")
        log(f"  Actualizados: {actualizados} tickers")
    else:
        log("")
        log(f"  [DRY RUN] Se actualizarian {sum(1 for _, v in resultados if v)} tickers")

    log(f"  Errores      : {len(errores)}")
    if errores:
        log(f"  Tickers con error: {errores}")
    log("  Completado.")


def main():
    parser = argparse.ArgumentParser(description="Agrega columna industry a activos")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra cambios sin escribir en DB")
    parser.add_argument("--status", action="store_true",
                        help="Muestra estado actual de la columna industry")
    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    cmd_run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
