"""
34_futuros_indices.py
Historico diario de futuros de indices de referencia (yfinance).

Instrumentos:
    YM=F   Dow Jones Mini (US30)
    ES=F   S&P 500 E-mini
    NQ=F   Nasdaq-100 E-mini
    RTY=F  Russell 2000 E-mini

    GC=F   Gold futures (CMX)
    SI=F   Silver futures (CMX)
    UB=F   Ultra T-Bond 30Y (CBOT)
    TN=F   Ultra 10-Year Note (CBOT)
    ZL=F   Soybean Oil (CBOT)
    ZM=F   Soybean Meal (CBOT)
    XAE=F  E-mini Energy Select Sector (CME)
    XAU=F  E-mini Utilities Select Sector (CME)

Criterio de historial:
    Solo se cargan 3 anos hacia atras (inicio backfill).
    Datos mas antiguos aportan regimenes de mercado irrelevantes para
    modelos actuales y no justifican el costo de compute.

Actualizacion diaria:
    Descarga los ultimos 5 dias (idempotente por UNIQUE constraint).
    Corre en el cron principal ANTES de features para que el contexto
    macro este disponible el mismo dia.

Uso:
    python scripts/34_futuros_indices.py           # actualiza ultimos dias
    python scripts/34_futuros_indices.py --init    # crea tabla + backfill 3 anos
    python scripts/34_futuros_indices.py --status  # muestra rango de datos en DB
"""

import sys
import os
import argparse
from datetime import date, datetime, timedelta

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


# ── Instrumentos ──────────────────────────────────────────────────────────────

FUTUROS = {
    # Indices de renta variable
    "YM=F":  "Dow Jones Mini (US30)",
    "ES=F":  "S&P 500 E-mini",
    "NQ=F":  "Nasdaq-100 E-mini",
    "RTY=F": "Russell 2000 E-mini",
    # Metales preciosos
    "GC=F":  "Gold futures",
    "SI=F":  "Silver futures",
    # Renta fija (tasas largas)
    "UB=F":  "Ultra T-Bond 30Y",
    "TN=F":  "Ultra 10-Year Note",
    # Commodities agricolas
    "ZL=F":  "Soybean Oil",
    "ZM=F":  "Soybean Meal",
    # Futuros sectoriales
    "XAE=F": "E-mini Energy Select Sector",
    "XAU=F": "E-mini Utilities Select Sector",
}

ANOS_BACKFILL = 3   # historial inicial al correr --init


# ── DDL ──────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS futuros_diarios (
    id         SERIAL        PRIMARY KEY,
    ticker     VARCHAR(10)   NOT NULL,
    fecha      DATE          NOT NULL,
    open       NUMERIC(14,4),
    high       NUMERIC(14,4),
    low        NUMERIC(14,4),
    close      NUMERIC(14,4) NOT NULL,
    volume     BIGINT,
    created_at TIMESTAMP     DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_futuros_ticker ON futuros_diarios (ticker);
CREATE INDEX IF NOT EXISTS idx_futuros_fecha  ON futuros_diarios (fecha);
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
    log("Tabla futuros_diarios lista.")


# ── Descarga ──────────────────────────────────────────────────────────────────

def descargar_futuro(ticker: str, start: str, end: str = None) -> list[dict]:
    """
    Descarga OHLCV diario para un futuro entre start y end.
    Retorna lista de dicts listos para insertar.
    """
    kwargs = dict(start=start, auto_adjust=True, progress=False)
    if end:
        kwargs["end"] = end

    try:
        raw = yf.download(ticker, **kwargs)
    except Exception as e:
        log(f"  [ERROR] {ticker}: {e}")
        return []

    if raw is None or raw.empty:
        return []

    # Normalizar columnas (MultiIndex si se pasa lista, flat si es uno solo)
    if hasattr(raw.columns, "levels"):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.reset_index()
    raw = raw.rename(columns={"date": "fecha", "Date": "fecha"})
    raw["fecha"] = raw["fecha"].dt.date

    filas = []
    for _, r in raw.iterrows():
        close_val = r.get("close") or r.get("adj close")
        if close_val is None:
            continue
        try:
            close_val = float(close_val)
        except (TypeError, ValueError):
            continue
        if close_val <= 0:
            continue

        def _f(col):
            v = r.get(col)
            try:
                return round(float(v), 4) if v is not None else None
            except (TypeError, ValueError):
                return None

        filas.append({
            "ticker": ticker,
            "fecha":  r["fecha"],
            "open":   _f("open"),
            "high":   _f("high"),
            "low":    _f("low"),
            "close":  round(close_val, 4),
            "volume": int(float(r["volume"])) if r.get("volume") else None,
        })

    return filas


# ── Persistencia ──────────────────────────────────────────────────────────────

def persistir(filas: list[dict]) -> int:
    """Upsert bulk. ON CONFLICT actualiza OHLCV (por si yfinance corrige datos)."""
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
    valores = [
        (f["ticker"], f["fecha"], f["open"], f["high"], f["low"], f["close"], f["volume"])
        for f in filas
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, SQL, valores, page_size=500)

    return len(filas)


# ── Runners ───────────────────────────────────────────────────────────────────

def cmd_init():
    """Crea la tabla y carga 3 anos de historial para todos los futuros."""
    init_tabla()

    start = (date.today() - timedelta(days=ANOS_BACKFILL * 365)).isoformat()
    log(f"Backfill desde {start} para {len(FUTUROS)} futuros...")
    log("=" * 50)

    total = 0
    for ticker, nombre in FUTUROS.items():
        filas = descargar_futuro(ticker, start=start)
        if not filas:
            log(f"  {ticker:<8s}  {nombre:<30s}  SIN DATOS")
            continue
        n = persistir(filas)
        fecha_min = min(f["fecha"] for f in filas)
        fecha_max = max(f["fecha"] for f in filas)
        log(f"  {ticker:<8s}  {nombre:<30s}  {n:4d} filas  ({fecha_min} -> {fecha_max})")
        total += n

    log("=" * 50)
    log(f"Total insertado: {total:,} filas")


def cmd_update():
    """Descarga los ultimos 5 dias para mantener la tabla al dia (cron diario)."""
    start = (date.today() - timedelta(days=7)).isoformat()   # 7 dias calendario = ~5 habiles
    hoy   = date.today()
    total = 0
    errores = 0

    for ticker in FUTUROS:
        filas = descargar_futuro(ticker, start=start)
        # Excluir sesion parcial del dia actual (futuros cotizan 24hs y yfinance
        # retorna la sesion en curso aunque no haya cerrado — igual que precios_diarios).
        filas = [f for f in filas if f["fecha"] < hoy]
        if not filas:
            errores += 1
            continue
        total += persistir(filas)

    log(f"  Futuros actualizados: {total} filas ({len(FUTUROS) - errores}/{len(FUTUROS)} instrumentos OK)")
    return total


def cmd_status():
    """Muestra el rango de fechas disponible por ticker."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker,
                   COUNT(*)        AS n,
                   MIN(fecha)      AS desde,
                   MAX(fecha)      AS hasta,
                   MAX(close)      AS max_close,
                   MIN(close)      AS min_close
            FROM futuros_diarios
            GROUP BY ticker
            ORDER BY ticker
        """)).fetchall()

    if not rows:
        print("  Sin datos en futuros_diarios.")
        return

    print()
    print(f"  {'TICKER':<8s}  {'FILAS':>5s}  {'DESDE':<12s}  {'HASTA':<12s}  "
          f"{'MIN':>10s}  {'MAX':>10s}  NOMBRE")
    print("  " + "-" * 70)
    for r in rows:
        nombre = FUTUROS.get(r[0], "")
        print(f"  {r[0]:<8s}  {r[1]:>5d}  {str(r[2]):<12s}  {str(r[3]):<12s}  "
              f"{float(r[5]):>10,.1f}  {float(r[4]):>10,.1f}  {nombre}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Historico futuros de indices (yfinance)")
    parser.add_argument("--init",   action="store_true",
                        help="Crea tabla y carga backfill de 3 anos")
    parser.add_argument("--status", action="store_true",
                        help="Muestra rango de datos disponibles en DB")
    args = parser.parse_args()

    if args.init:
        cmd_init()
    elif args.status:
        cmd_status()
    else:
        cmd_update()


if __name__ == "__main__":
    main()
