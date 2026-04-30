"""
add_tickers_tv.py
Migracion: incorpora 75 tickers nuevos (TV expansion 2026-04-21).

Pasos:
  1. Detecta cuales tickers de ALL_TICKERS faltan en activos (DB).
  2. Fetch yfinance info (nombre, sector, industry) para cada uno.
  3. INSERT en activos (activo=TRUE, solo_bt=TRUE).
  4. Backfill de precios historicos en precios_diarios (2 anos).

Flags:
  --dry-run    Muestra lo que haria sin escribir nada.
  --status     Solo muestra cuantos tickers hay en activos vs config.
  --skip-prices  Inserta en activos pero NO baja precios historicos.

Uso:
  python scripts/migrations/add_tickers_tv.py --dry-run
  python scripts/migrations/add_tickers_tv.py
  python scripts/migrations/add_tickers_tv.py --skip-prices
"""

import sys
import os
import argparse
import time
import math
from datetime import datetime, date

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
import pandas as pd
from sqlalchemy import text
from src.data.database import get_engine
from src.utils.config import ALL_TICKERS


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_tickers_en_db(engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker FROM activos")).fetchall()
    return {r[0].strip() for r in rows}


def get_columnas_activos(engine) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'activos' ORDER BY ordinal_position"
        )).fetchall()
    return [r[0] for r in rows]


def fetch_yf_info(ticker: str) -> dict:
    """Retorna nombre, sector, industry desde yfinance. Tolerante a errores."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "nombre":   info.get("longName") or info.get("shortName") or ticker,
            "sector":   info.get("sector") or None,
            "industry": info.get("industry") or None,
        }
    except Exception:
        return {"nombre": ticker, "sector": None, "industry": None}


def insertar_activos(engine, tickers_nuevos: list[str], columnas: list[str], dry_run: bool):
    """INSERT en activos para cada ticker nuevo."""
    tiene_solo_bt  = "solo_bt"  in columnas
    tiene_industry = "industry" in columnas

    log(f"  Columnas activos: {columnas}")
    log(f"  tiene_solo_bt={tiene_solo_bt}  tiene_industry={tiene_industry}")
    log(f"  Fetching yfinance info para {len(tickers_nuevos)} tickers...")

    rows = []
    for i, t in enumerate(tickers_nuevos, 1):
        info = fetch_yf_info(t)
        rows.append({
            "ticker":   t,
            "nombre":   info["nombre"],
            "sector":   info["sector"],
            "industry": info["industry"],
        })
        log(f"    [{i:2d}/{len(tickers_nuevos)}] {t:<10} {info['sector'] or 'N/A':<30} {info['industry'] or 'N/A'}")
        time.sleep(0.25)

    if dry_run:
        log("  [DRY-RUN] No se escribe en activos.")
        return

    # Construir INSERT dinamico segun columnas disponibles
    for row in rows:
        cols  = ["ticker", "nombre", "sector", "activo"]
        vals  = [":ticker", ":nombre", ":sector", "TRUE"]
        params = {"ticker": row["ticker"], "nombre": row["nombre"], "sector": row["sector"]}

        if tiene_solo_bt:
            cols.append("solo_bt")
            vals.append("TRUE")
        if tiene_industry:
            cols.append("industry")
            vals.append(":industry")
            params["industry"] = row["industry"]

        sql = (
            f"INSERT INTO activos ({', '.join(cols)}) "
            f"VALUES ({', '.join(vals)}) "
            f"ON CONFLICT (ticker) DO UPDATE SET "
            f"nombre = EXCLUDED.nombre, "
            f"sector = EXCLUDED.sector"
            + (", industry = EXCLUDED.industry" if tiene_industry else "")
            + ", activo = TRUE"
        )

        with engine.connect() as conn:
            conn.execute(text(sql), params)
            conn.commit()

    log(f"  INSERT completado: {len(rows)} tickers en activos.")


def backfill_precios(tickers: list[str], dry_run: bool):
    """
    Descarga 2 anos de precios historicos para los tickers nuevos
    y los inserta en precios_diarios via upsert.

    upsert_precios(df) espera un DataFrame con columnas:
        ticker, fecha, open, high, low, close, volume, adj_close
    """
    from src.data.database import upsert_precios

    log(f"  Descargando precios historicos (2y) para {len(tickers)} tickers...")

    if dry_run:
        log("  [DRY-RUN] No se descarga ni guarda precios.")
        return

    # Lotes de 20 para no saturar yfinance
    LOTE = 20
    n_ok  = 0
    n_err = 0

    for i in range(0, len(tickers), LOTE):
        lote = tickers[i: i + LOTE]
        log(f"  Lote {i//LOTE + 1}/{math.ceil(len(tickers)/LOTE)}: {lote}")
        try:
            raw = yf.download(
                tickers=lote,
                period="2y",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=False,
            )
        except Exception as ex:
            log(f"  [ERROR] Descarga lote fallo: {ex}")
            n_err += len(lote)
            time.sleep(2)
            continue

        if raw is None or raw.empty:
            log(f"  [WARN] Lote sin datos.")
            n_err += len(lote)
            continue

        for ticker in lote:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl0 = raw.columns.get_level_values(0).unique().tolist()
                    lvl1 = raw.columns.get_level_values(1).unique().tolist()
                    if ticker in lvl0:
                        df_t = raw[ticker].copy()
                    elif ticker in lvl1:
                        df_t = raw.xs(ticker, axis=1, level=1).copy()
                    else:
                        raise ValueError(f"{ticker} no en batch")
                else:
                    df_t = raw.copy()

                df_t = df_t.reset_index()
                # Normalizar nombres de columnas
                df_t.columns = [
                    (c[0].lower() if isinstance(c, tuple) else str(c).lower())
                    for c in df_t.columns
                ]
                df_t = df_t.rename(columns={"date": "fecha"})
                df_t = df_t.dropna(subset=["close"])
                df_t = df_t[df_t["close"] > 0]

                if df_t.empty:
                    log(f"    {ticker}: sin datos validos, skip.")
                    n_err += 1
                    continue

                # Asegurar columnas requeridas por upsert_precios
                df_t["ticker"]    = ticker
                df_t["adj_close"] = df_t.get("adj close", df_t["close"])
                if "volume" not in df_t.columns:
                    df_t["volume"] = 0

                cols_req = ["ticker", "fecha", "open", "high", "low", "close", "volume", "adj_close"]
                df_final = df_t[cols_req].copy()

                upsert_precios(df_final)
                log(f"    {ticker}: {len(df_final)} filas OK.")
                n_ok += 1

            except Exception as ex:
                log(f"    {ticker}: ERROR — {ex}")
                n_err += 1

        time.sleep(1.5)  # pausa entre lotes

    log(f"  Backfill precios: {n_ok} ok / {n_err} errores.")


def cmd_status(engine):
    en_db = get_tickers_en_db(engine)
    en_config = set(ALL_TICKERS)
    solo_db     = en_db - en_config
    solo_config = en_config - en_db

    log(f"  Tickers en activos (DB): {len(en_db)}")
    log(f"  Tickers en ALL_TICKERS (config): {len(en_config)}")
    if solo_config:
        log(f"  En config pero NO en DB ({len(solo_config)}): {sorted(solo_config)}")
    else:
        log("  Todos los tickers de config estan en DB.")
    if solo_db:
        log(f"  En DB pero NO en config ({len(solo_db)}): {sorted(solo_db)}")


def main():
    parser = argparse.ArgumentParser(description="Incorpora tickers TV al pipeline.")
    parser.add_argument("--dry-run",     action="store_true", help="Muestra sin escribir.")
    parser.add_argument("--status",      action="store_true", help="Solo muestra estado.")
    parser.add_argument("--skip-prices", action="store_true", help="Inserta activos pero no baja precios.")
    args = parser.parse_args()

    engine = get_engine()

    log("=" * 60)
    log("  MIGRACION: add_tickers_tv")
    log(f"  Modo: {'DRY-RUN' if args.dry_run else 'REAL'}")
    log("=" * 60)

    if args.status:
        cmd_status(engine)
        return

    # Detectar tickers faltantes en DB
    en_db      = get_tickers_en_db(engine)
    en_config  = set(ALL_TICKERS)
    faltantes  = sorted(en_config - en_db)
    columnas   = get_columnas_activos(engine)

    log(f"  Tickers en config : {len(en_config)}")
    log(f"  Tickers en DB     : {len(en_db)}")
    log(f"  A insertar        : {len(faltantes)}")

    if not faltantes:
        log("  Nada que hacer. Todos los tickers ya estan en activos.")
        return

    # 1. INSERT en activos
    log("")
    log("  [1/2] Insertando en activos...")
    insertar_activos(engine, faltantes, columnas, args.dry_run)

    # 2. Backfill de precios
    if not args.skip_prices:
        log("")
        log("  [2/2] Backfill de precios historicos (2 anos)...")
        backfill_precios(faltantes, args.dry_run)
    else:
        log("  [2/2] Backfill de precios omitido (--skip-prices).")

    log("")
    log("  Migracion completada.")


if __name__ == "__main__":
    main()
