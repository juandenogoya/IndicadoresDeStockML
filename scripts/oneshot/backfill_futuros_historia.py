"""
backfill_futuros_historia.py  (ONE-SHOT)
Extiende la historia de futuros_diarios hacia atras para los 4 futuros de
INDICE (ES/YM/NQ/RTY), que hoy arrancan recien 2025-05-09 (~15 meses) mientras
precios_diarios arranca 2020-01-02. Sin esto, cualquier comparacion contra el
benchmark (ES) queda limitada a ~15 meses.

Motor: yahooquery (probado: sirve ES desde 2000; empalma EXACTO con las filas
que ya venian de yfinance -> re-descargar el rango completo deja la serie de una
sola fuente, sin escalon en el empalme).

Alcance / decisiones:
    - Solo los 4 futuros de INDICE. Los 8 macro/commodity se dejan como estan
      (no aportan al benchmark y estan atados al feature set macro).
    - target=local unicamente (Plan C: local es la fuente de verdad de OHLCV).
    - UPSERT sobre futuros_diarios (ON CONFLICT ya existe): re-descarga el rango
      completo, sobrescribe idempotente. Aditivo: NO perturba el ML (script 36
      solo recompute los ultimos 15 dias; la beta del perfil usa ventana 252).
    - Descarta la barra EN CURSO (fecha >= hoy): los futuros cotizan ~24h y
      yahooquery devuelve la sesion parcial del dia (regla #10 del proyecto).

Uso:
    python scripts/oneshot/backfill_futuros_historia.py --status
    python scripts/oneshot/backfill_futuros_historia.py --dry-run
    python scripts/oneshot/backfill_futuros_historia.py
    python scripts/oneshot/backfill_futuros_historia.py --desde 2020-01-02
"""

import sys
import os
import argparse
import datetime as dt
from datetime import date, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Forzar target LOCAL antes de importar src.data.database (config.py lee
# DATABASE_URL al momento del import).
try:
    from dotenv import load_dotenv
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass
os.environ.pop("DATABASE_URL", None)

import pandas as pd
import psycopg2.extras
from sqlalchemy import text
from yahooquery import Ticker

from src.data.database import get_engine, get_connection
from src.utils.yfinance_lock import acquire


FUTUROS_INDICE = ["ES=F", "YM=F", "NQ=F", "RTY=F"]
DESDE_DEFAULT = "2020-01-02"   # empata con MIN(fecha) de precios_diarios
_OHLC = ["open", "high", "low", "close"]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _to_date(v):
    """Normaliza el valor de fecha de yahooquery a datetime.date."""
    if isinstance(v, dt.datetime):     # Timestamp/datetime (posible barra en curso tz-aware)
        return v.date()
    if isinstance(v, dt.date):         # ya es date (barras diarias completas)
        return v
    return pd.to_datetime(v).date()


def _f(v):
    """Valor -> float redondeado o None."""
    try:
        return round(float(v), 4) if v is not None and not pd.isna(v) else None
    except (TypeError, ValueError):
        return None


def descargar(sym: str, desde: str) -> list[dict]:
    """OHLCV diario de un futuro desde `desde`, descartando la barra en curso."""
    try:
        raw = Ticker(sym).history(start=desde)
    except Exception as e:
        log(f"  [ERROR] {sym}: {e}")
        return []

    # yahooquery devuelve str/dict ante error, DataFrame vacio si no hay data.
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        log(f"  [WARN] {sym}: sin DataFrame ({type(raw).__name__})")
        return []

    df = raw.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        log(f"  [WARN] {sym}: columnas inesperadas {list(df.columns)}")
        return []

    hoy = date.today()
    filas = []
    for _, r in df.iterrows():
        try:
            fecha = _to_date(r["date"])
        except Exception:
            continue
        if fecha >= hoy:                 # descarta la sesion parcial del dia
            continue
        close = _f(r.get("close"))
        if close is None or close <= 0:
            continue
        filas.append({
            "ticker": sym, "fecha": fecha,
            "open": _f(r.get("open")), "high": _f(r.get("high")),
            "low": _f(r.get("low")), "close": close,
            "volume": int(float(r["volume"])) if _f(r.get("volume")) else None,
        })
    return filas


def persistir(filas: list[dict]) -> int:
    """UPSERT bulk en futuros_diarios (ON CONFLICT ya definido en la tabla)."""
    if not filas:
        return 0
    SQL = """
        INSERT INTO futuros_diarios (ticker, fecha, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            open   = EXCLUDED.open,
            high   = EXCLUDED.high,
            low    = EXCLUDED.low,
            close  = EXCLUDED.close,
            volume = EXCLUDED.volume
    """
    valores = [(f["ticker"], f["fecha"], f["open"], f["high"], f["low"],
                f["close"], f["volume"]) for f in filas]
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, SQL, valores, page_size=500)
    return len(filas)


def cmd_status():
    eng = get_engine()
    with eng.connect() as c:
        rows = c.execute(text("""
            SELECT ticker, COUNT(*) n, MIN(fecha) desde, MAX(fecha) hasta
            FROM futuros_diarios WHERE ticker = ANY(:tks)
            GROUP BY ticker ORDER BY ticker
        """), {"tks": FUTUROS_INDICE}).fetchall()
    print(f"\n  {'TICKER':<8} {'FILAS':>6}  {'DESDE':<12} {'HASTA':<12}")
    print("  " + "-" * 42)
    for r in rows:
        print(f"  {r[0]:<8} {r[1]:>6}  {str(r[2]):<12} {str(r[3]):<12}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Backfill historico de futuros de indice (yahooquery, LOCAL)")
    ap.add_argument("--desde", default=DESDE_DEFAULT, help=f"fecha inicio (default {DESDE_DEFAULT})")
    ap.add_argument("--dry-run", action="store_true", help="descarga y muestra, no escribe")
    ap.add_argument("--status", action="store_true", help="rango actual en DB y sale")
    args = ap.parse_args()

    if args.status:
        cmd_status()
        return

    acquire("backfill_futuros_historia")

    log("=" * 56)
    log(f"Backfill futuros de indice | desde {args.desde} | LOCAL"
        + (" | DRY RUN" if args.dry_run else ""))
    log("=" * 56)

    total = 0
    for sym in FUTUROS_INDICE:
        filas = descargar(sym, args.desde)
        if not filas:
            log(f"  {sym:<7} SIN DATOS")
            continue
        fmin = min(f["fecha"] for f in filas)
        fmax = max(f["fecha"] for f in filas)
        if args.dry_run:
            log(f"  {sym:<7} {len(filas):>5} filas  ({fmin} -> {fmax})  [dry-run]")
        else:
            n = persistir(filas)
            log(f"  {sym:<7} {n:>5} filas  ({fmin} -> {fmax})  UPSERT OK")
            total += n

    log("=" * 56)
    log(f"[DRY RUN] nada escrito." if args.dry_run else f"Total UPSERT: {total:,} filas")
    if not args.dry_run:
        cmd_status()


if __name__ == "__main__":
    main()
