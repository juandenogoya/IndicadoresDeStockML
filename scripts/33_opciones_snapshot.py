"""
33_opciones_snapshot.py
Recolector diario de opciones sobre acciones (yfinance).

Graba un snapshot de la chain de opciones para los 124 tickers activos:
    - Vencimientos dentro de los proximos MAX_DTE dias (default: 90)
    - Strikes con volumen > 0 o open_interest > 0  (sin actividad = ignorados)
    - Calls y puts por separado

Filtro de vencimientos (MAX_DTE):
    Contratos con vencimiento > 90 dias tienen escasa actividad diaria y son
    menos relevantes para el analisis de sentimiento de corto/medio plazo.
    Limitar el horizonte reduce el numero de HTTP calls a yfinance y el
    volumen de datos almacenados sin perder informacion util.
    Configurable via .env: OPCIONES_MAX_DTE=90

La fecha del snapshot es HOY; los datos reflejan el mercado del dia anterior
(mismo principio que precios_diarios y el resto del pipeline).

Precio subyacente y HV_20d se obtienen directamente de precios_diarios.

Uso:
    python scripts/33_opciones_snapshot.py            # todos los tickers
    python scripts/33_opciones_snapshot.py --init     # crea la tabla en DB
    python scripts/33_opciones_snapshot.py --ticker AAPL MSFT
    python scripts/33_opciones_snapshot.py --dry-run  # muestra sin guardar
"""

import sys
import os
import argparse
import math
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

import yfinance as yf
import psycopg2.extras
from sqlalchemy import text
from src.data.database import get_engine, get_connection
from src.utils.config import ALL_TICKERS

# ── Parametros configurables ──────────────────────────────────────────────────

# Solo vencimientos dentro de los proximos N dias calendario.
# 90 dias captura weeklies + monthlies del trimestre — suficiente para sentimiento.
MAX_DTE = int(os.getenv("OPCIONES_MAX_DTE", "90"))


# ── DDL ──────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS opciones_snapshot (
    id                SERIAL        PRIMARY KEY,
    fecha_snapshot    DATE          NOT NULL,
    ticker            VARCHAR(20)   NOT NULL,
    vencimiento       DATE          NOT NULL,
    tipo              VARCHAR(4)    NOT NULL,     -- 'call' | 'put'
    strike            NUMERIC(12,4) NOT NULL,
    volumen           INTEGER,
    open_interest     INTEGER,
    iv                NUMERIC(10,6),              -- volatilidad implicita (decimal: 0.25 = 25%)
    bid               NUMERIC(12,4),
    ask               NUMERIC(12,4),
    precio_subyacente NUMERIC(12,4),              -- ultimo close en precios_diarios
    hv_20d            NUMERIC(10,6),              -- volatilidad historica 20d anualizada (decimal)
    created_at        TIMESTAMP     DEFAULT NOW(),
    UNIQUE (fecha_snapshot, ticker, vencimiento, tipo, strike)
);

CREATE INDEX IF NOT EXISTS idx_opciones_fecha       ON opciones_snapshot (fecha_snapshot);
CREATE INDEX IF NOT EXISTS idx_opciones_ticker      ON opciones_snapshot (ticker);
CREATE INDEX IF NOT EXISTS idx_opciones_vencimiento ON opciones_snapshot (vencimiento);
CREATE INDEX IF NOT EXISTS idx_opciones_tipo        ON opciones_snapshot (tipo);
"""


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_tabla():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    log("Tabla opciones_snapshot lista.")


# ── Precios y HV desde DB ─────────────────────────────────────────────────────

def _get_precios_subyacentes(tickers: list[str]) -> dict[str, float]:
    """Ultimo close disponible en precios_diarios por ticker."""
    engine = get_engine()
    placeholders = ", ".join(f"'{t}'" for t in tickers)
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT ON (ticker) ticker, close
            FROM   precios_diarios
            WHERE  ticker IN ({placeholders})
              AND  close > 0
            ORDER  BY ticker, fecha DESC
        """)).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def _get_hv_20d(tickers: list[str]) -> dict[str, float]:
    """
    Volatilidad historica anualizada a 20 dias para cada ticker.
    HV = std(log_returns_20d) * sqrt(252).  Retorna decimal (0.25 = 25%).
    """
    engine = get_engine()
    placeholders = ", ".join(f"'{t}'" for t in tickers)
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT ticker, close
            FROM   precios_diarios
            WHERE  ticker IN ({placeholders})
              AND  close  > 0
              AND  fecha  >= CURRENT_DATE - INTERVAL '35 days'
            ORDER  BY ticker, fecha ASC
        """)).fetchall()

    closes_by_ticker: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        closes_by_ticker[r[0]].append(float(r[1]))

    hv_map: dict[str, float] = {}
    for ticker, closes in closes_by_ticker.items():
        if len(closes) < 5:
            continue
        c = closes[-21:]                       # hasta 20 retornos
        log_ret = [math.log(c[i] / c[i - 1]) for i in range(1, len(c))]
        n    = len(log_ret)
        mean = sum(log_ret) / n
        var  = sum((r - mean) ** 2 for r in log_ret) / (n - 1) if n > 1 else 0.0
        hv_map[ticker] = round(math.sqrt(var) * math.sqrt(252), 6)

    return hv_map


# ── Colector por ticker ───────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Convierte un valor pandas a float o None (maneja NaN / None)."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    """Convierte un valor pandas a int o None."""
    f = _safe_float(val)
    return None if f is None else int(f)


def recolectar_ticker(
    ticker: str,
    fecha_snapshot: date,
    precio_subyacente: Optional[float],
    hv_20d: Optional[float],
    max_dte: int = None,
) -> list[dict]:
    """
    Descarga la chain de opciones para un ticker, limitada a vencimientos
    dentro de los proximos max_dte dias (default: MAX_DTE del env).
    Solo incluye strikes con volumen > 0 o open_interest > 0.
    """
    from datetime import timedelta
    limite_vencimiento = fecha_snapshot + timedelta(days=(max_dte or MAX_DTE))

    try:
        yft  = yf.Ticker(ticker)
        exps = yft.options          # tuple de strings "YYYY-MM-DD"
        if not exps:
            return []
    except Exception:
        return []

    # Filtrar expirations fuera del horizonte — evita HTTP calls innecesarios
    exps = [e for e in exps if date.fromisoformat(e) <= limite_vencimiento]
    if not exps:
        return []

    filas = []

    for exp_str in exps:
        try:
            exp_date = date.fromisoformat(exp_str)
            chain    = yft.option_chain(exp_str)
        except Exception:
            continue

        # Pausa entre expirations para no saturar la API de Yahoo Finance
        time.sleep(0.5)

        for tipo, df in [("call", chain.calls), ("put", chain.puts)]:
            if df is None or df.empty:
                continue

            for _, opt in df.iterrows():
                vol = _safe_int(opt.get("volume"))
                oi  = _safe_int(opt.get("openInterest"))

                # Sin actividad -> ignorar
                if not vol and not oi:
                    continue

                filas.append({
                    "fecha_snapshot":    fecha_snapshot,
                    "ticker":            ticker,
                    "vencimiento":       exp_date,
                    "tipo":              tipo,
                    "strike":            float(opt["strike"]),
                    "volumen":           vol,
                    "open_interest":     oi,
                    "iv":                _safe_float(opt.get("impliedVolatility")),
                    "bid":               _safe_float(opt.get("bid")),
                    "ask":               _safe_float(opt.get("ask")),
                    "precio_subyacente": precio_subyacente,
                    "hv_20d":            hv_20d,
                })

    return filas


# ── Persistencia ──────────────────────────────────────────────────────────────

def persistir_filas(filas: list[dict]) -> int:
    """
    Inserta todas las filas en una sola query usando execute_values.
    Dramaticamente mas rapido que executemany sobre conexiones remotas (Railway).
    ON CONFLICT DO NOTHING -> idempotente.
    """
    if not filas:
        return 0

    SQL = """
        INSERT INTO opciones_snapshot (
            fecha_snapshot, ticker, vencimiento, tipo, strike,
            volumen, open_interest, iv, bid, ask,
            precio_subyacente, hv_20d
        ) VALUES %s
        ON CONFLICT (fecha_snapshot, ticker, vencimiento, tipo, strike)
        DO NOTHING
    """
    valores = [
        (
            f["fecha_snapshot"], f["ticker"],  f["vencimiento"], f["tipo"],  f["strike"],
            f["volumen"],        f["open_interest"], f["iv"],   f["bid"],    f["ask"],
            f["precio_subyacente"], f["hv_20d"],
        )
        for f in filas
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, SQL, valores, page_size=500)

    return len(filas)


# ── Runner ────────────────────────────────────────────────────────────────────

def cmd_run(tickers: list[str], dry_run: bool = False):
    fecha_hoy = date.today()

    log("=" * 60)
    log(f"  OPCIONES SNAPSHOT  |  {fecha_hoy}")
    log(f"  Tickers   : {len(tickers)}")
    log(f"  Modo      : {'DRY RUN' if dry_run else 'REAL'}")
    log("=" * 60)

    log("  Cargando precios subyacentes y HV_20d desde DB...")
    precios = _get_precios_subyacentes(tickers)
    hvs     = _get_hv_20d(tickers)
    log(f"  Precios  : {len(precios)} tickers")
    log(f"  HV_20d   : {len(hvs)} tickers")

    total_filas = 0
    sin_opciones = 0
    errores      = 0

    for i, ticker in enumerate(tickers, 1):
        precio = precios.get(ticker)
        hv     = hvs.get(ticker)

        try:
            filas = recolectar_ticker(ticker, fecha_hoy, precio, hv)

            if not filas:
                log(f"  [{i:3d}/{len(tickers)}] {ticker:<8s}  sin opciones activas")
                sin_opciones += 1
                time.sleep(0.2)
                continue

            if dry_run:
                log(f"  [{i:3d}/{len(tickers)}] {ticker:<8s}  {len(filas):5d} filas  [DRY RUN]")
            else:
                n = persistir_filas(filas)
                log(f"  [{i:3d}/{len(tickers)}] {ticker:<8s}  {n:5d} filas insertadas")
                total_filas += n

            # Pausa para no saturar la API de yfinance
            time.sleep(0.3)

        except Exception as e:
            log(f"  [{i:3d}/{len(tickers)}] {ticker:<8s}  ERROR: {e}")
            errores += 1

    log("")
    log(f"  Filas insertadas : {total_filas:,}")
    log(f"  Sin opciones     : {sin_opciones}")
    log(f"  Errores          : {errores}")
    log("  Completado.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Snapshot diario de opciones (yfinance)")
    parser.add_argument("--init",    action="store_true",
                        help="Crea la tabla opciones_snapshot en DB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Descarga datos pero no escribe en DB")
    parser.add_argument("--ticker",  nargs="+",
                        help="Tickers especificos (default: los 124 del pipeline)")
    args = parser.parse_args()

    if args.init:
        init_tabla()
        return

    tickers = args.ticker if args.ticker else list(ALL_TICKERS)
    cmd_run(tickers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
