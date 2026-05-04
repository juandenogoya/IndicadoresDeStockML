"""
actualizar_precios_fast_info.py
Actualiza el ultimo cierre EOD para los 199 tickers via yf.Ticker().fast_info.

Endpoint usado: /v8/finance/quote/ (cotizaciones) -- distinto al OHLCV historico
(/v8/finance/chart/) que Yahoo Finance rate-limita agresivamente en IPs cloud.

Ventaja vs yf.download():
  - Funciona en GH Actions / Railway (mismo tipo de endpoint que opciones, que no esta bloqueado)
  - Sin lotes grandes: cada ticker = 1 request pequeno
  - Mucho menos propenso a rate limit

Limitacion: solo captura la SESION MAS RECIENTE cerrada (no sirve para backfill
de multiples dias). Para backfill usar: cron_diario.py --step precios

Uso:
    python scripts/actualizar_precios_fast_info.py
    python scripts/actualizar_precios_fast_info.py --dry-run   (muestra datos sin guardar)
"""

import sys
import os
import time
import argparse
from datetime import date, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

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

from src.data.database import query_df, upsert_precios
from src.pipeline.data_manager import cargar_precios_db
from src.indicators.technical import procesar_indicadores_ticker
from src.utils.trading_calendar import is_trading_day, prev_trading_day

SEP = "=" * 62
TICKER_DELAY = 1.0   # segundos entre tickers (anti rate-limit por volumen)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def determinar_fecha_sesion() -> date:
    """
    Retorna la fecha de la ultima sesion cerrada.
    - Si hoy es dia habil: retorna hoy (script corre post-cierre NYSE, 21:00+ UTC).
    - Si hoy es fin de semana o feriado: retorna el ultimo dia habil anterior.
    """
    hoy = date.today()
    return hoy if is_trading_day(hoy) else prev_trading_day(hoy)


def obtener_tickers() -> list[str]:
    df = query_df("SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker")
    return df["ticker"].tolist() if not df.empty else []


def descargar_fast_info(ticker: str) -> dict | None:
    """
    Obtiene OHLCV de la ultima sesion via fast_info.
    Retorna dict con campos o None si el ticker no tiene datos validos.
    """
    try:
        fi = yf.Ticker(ticker).fast_info

        close = fi.last_price
        if close is None or close <= 0:
            return None

        open_  = fi.open
        high   = fi.day_high
        low    = fi.day_low
        vol    = fi.last_volume

        # Validaciones basicas: si open/high/low son None los aceptamos como NULL
        # (la unica columna NOT NULL en precios_diarios es close).
        return {
            "open":   float(open_)  if open_ is not None else None,
            "high":   float(high)   if high  is not None else None,
            "low":    float(low)    if low   is not None else None,
            "close":  float(close),
            "volume": int(float(vol)) if vol is not None and vol > 0 else None,
        }

    except Exception as e:
        log(f"    fast_info error: {str(e)[:80]}")
        return None


def procesar_ticker(ticker: str, fecha: date, data: dict, dry_run: bool) -> bool:
    """
    Upsert precio + recalculo de indicadores para un ticker.
    Retorna True si OK.
    """
    df_row = pd.DataFrame([{
        "fecha":     pd.Timestamp(fecha),
        "open":      data["open"],
        "high":      data["high"],
        "low":       data["low"],
        "close":     data["close"],
        "volume":    data["volume"],
        "ticker":    ticker,
        "adj_close": data["close"],
    }])

    if dry_run:
        return True  # sin escritura a DB

    upsert_precios(df_row)

    df_full = cargar_precios_db(ticker, ultimas_n=500)
    if len(df_full) < 250:
        raise ValueError(f"historico insuficiente ({len(df_full)} barras)")

    procesar_indicadores_ticker(ticker, df_full, guardar_db=True)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Actualiza precios EOD via fast_info (endpoint cotizaciones)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra datos descargados sin guardar en DB"
    )
    args = parser.parse_args()
    dry_run = args.dry_run

    hoy           = date.today()
    fecha_sesion  = determinar_fecha_sesion()

    print()
    print(SEP)
    print(f"  ACTUALIZAR PRECIOS (fast_info)  |  {hoy}")
    print(f"  Sesion objetivo : {fecha_sesion}")
    print(f"  Endpoint        : /v8/finance/quote/ (cotizaciones)")
    if dry_run:
        print(f"  MODO             : DRY-RUN (sin escritura a DB)")
    print(SEP)
    print()

    tickers = obtener_tickers()
    if not tickers:
        print("  [ERROR] Sin tickers activos en DB.")
        sys.exit(1)

    log(f"  {len(tickers)} tickers | delay entre tickers: {TICKER_DELAY}s")
    print()

    n_ok   = 0
    n_warn = 0
    n_err  = 0

    for i, ticker in enumerate(tickers, 1):
        data = descargar_fast_info(ticker)

        if data is None:
            log(f"  [{i:03d}/{len(tickers)}] {ticker:<8s} WARN  sin datos fast_info")
            n_warn += 1
        else:
            try:
                procesar_ticker(ticker, fecha_sesion, data, dry_run)
                suffix = "DRY" if dry_run else "OK"
                log(f"  [{i:03d}/{len(tickers)}] {ticker:<8s} {suffix}    "
                    f"close={data['close']:>9.2f}  fecha={fecha_sesion}")
                n_ok += 1
            except Exception as e:
                log(f"  [{i:03d}/{len(tickers)}] {ticker:<8s} ERROR {str(e)[:60]}")
                n_err += 1

        if i < len(tickers):
            time.sleep(TICKER_DELAY)

    print()
    print(SEP)
    print(f"  {'DRY-RUN' if dry_run else 'RESULTADO'}:")
    print(f"  OK    : {n_ok:3d} / {len(tickers)}")
    print(f"  WARN  : {n_warn:3d}")
    print(f"  ERROR : {n_err:3d}")
    print(SEP)
    print()

    sys.exit(0 if n_ok > 0 else 1)


if __name__ == "__main__":
    main()
