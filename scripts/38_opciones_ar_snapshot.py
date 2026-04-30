"""
38_opciones_ar_snapshot.py
Snapshot diario de Greeks para opciones argentinas (BCBA).

IMPORTANTE: debe correr durante horario BCBA (10:30-17:00 ART = 13:30-20:00 UTC).
Fuera de ese horario la mayoria de los Greeks son null y el snapshot no tiene valor.

Flujo:
  - Consulta activos_ar WHERE tiene_opciones = TRUE (~12 tickers)
  - Por cada ticker: get_options_chain via IOL REST API
  - Filtra solo strikes liquidos (IV no nula)
  - Guarda en opciones_ar_gregas con UNIQUE (fecha, symbol)

Uso:
  python scripts/38_opciones_ar_snapshot.py           # todos los tickers con opciones
  python scripts/38_opciones_ar_snapshot.py --ticker GGAL YPFD
  python scripts/38_opciones_ar_snapshot.py --dry-run
  python scripts/38_opciones_ar_snapshot.py --status
  python scripts/38_opciones_ar_snapshot.py --force   # corre aunque sea fuera de horario
"""

import sys
import os
import argparse
import time
from datetime import datetime, date

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

import psycopg2.extras
from sqlalchemy import text
from src.data.database import get_engine, get_connection
from src.data.iol_client import IOLClient


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Horario de mercado BCBA ───────────────────────────────────────────────────
# L-V 10:30-17:00 ART (UTC-3) = 13:30-20:00 UTC

def en_horario_bcba() -> bool:
    """Verifica si el momento actual esta dentro del horario de mercado BCBA (UTC)."""
    ahora_utc = datetime.utcnow()
    if ahora_utc.weekday() >= 5:
        return False
    hora_utc = ahora_utc.hour + ahora_utc.minute / 60
    return 13.5 <= hora_utc <= 20.0


# ── Persistencia ──────────────────────────────────────────────────────────────

SQL_UPSERT = """
    INSERT INTO opciones_ar_gregas (
        fecha, ticker_subyacente, symbol, option_type, strike_price, expiration,
        bid_price, ask_price, theoretical_price, implied_volatility,
        delta, gamma, theta, vega, rho, volume, spot_price, risk_free_rate
    ) VALUES (
        %(fecha)s, %(ticker_subyacente)s, %(symbol)s, %(option_type)s,
        %(strike_price)s, %(expiration)s,
        %(bid_price)s, %(ask_price)s, %(theoretical_price)s, %(implied_volatility)s,
        %(delta)s, %(gamma)s, %(theta)s, %(vega)s, %(rho)s,
        %(volume)s, %(spot_price)s, %(risk_free_rate)s
    )
    ON CONFLICT (fecha, symbol) DO UPDATE SET
        bid_price           = EXCLUDED.bid_price,
        ask_price           = EXCLUDED.ask_price,
        theoretical_price   = EXCLUDED.theoretical_price,
        implied_volatility  = EXCLUDED.implied_volatility,
        delta               = EXCLUDED.delta,
        gamma               = EXCLUDED.gamma,
        theta               = EXCLUDED.theta,
        vega                = EXCLUDED.vega,
        rho                 = EXCLUDED.rho,
        volume              = EXCLUDED.volume,
        spot_price          = EXCLUDED.spot_price,
        risk_free_rate      = EXCLUDED.risk_free_rate
"""


def persistir_gregas(records: list[dict]):
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, SQL_UPSERT, records, page_size=200)
    log(f"  Guardadas {len(records)} filas en opciones_ar_gregas.")


# ── Procesamiento de un ticker ────────────────────────────────────────────────

def procesar_ticker(client: IOLClient, ticker: str, hoy: date, dry_run: bool) -> int:
    """Retorna cantidad de strikes guardados."""
    try:
        df = client.get_options_chain_df(ticker)
        if df is None or df.empty:
            log(f"  {ticker}: sin datos liquidos en el chain.")
            return 0

        df["fecha"] = hoy
        records = df.to_dict(orient="records")

        # Normalizar tipos para psycopg2
        for r in records:
            for k, v in r.items():
                if hasattr(v, "item"):
                    r[k] = v.item()

        log(f"  {ticker}: {len(records)} strikes liquidos  "
            f"spot={records[0].get('spot_price', 'N/A')}")

        if not dry_run:
            persistir_gregas(records)

        return len(records)

    except Exception as e:
        log(f"  {ticker}: ERROR — {e}")
        return 0


# ── Comandos ──────────────────────────────────────────────────────────────────

def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker_subyacente,
                   fecha,
                   COUNT(*)                                       AS n_strikes,
                   ROUND(AVG(implied_volatility)::numeric, 4)    AS iv_promedio,
                   ROUND(SUM(CASE WHEN option_type='V' THEN COALESCE(volume,0) ELSE 0 END)::numeric
                       / NULLIF(SUM(CASE WHEN option_type='C' THEN COALESCE(volume,0) ELSE 0 END), 0), 3) AS pcr
            FROM opciones_ar_gregas
            WHERE fecha >= CURRENT_DATE - 7
            GROUP BY ticker_subyacente, fecha
            ORDER BY fecha DESC, ticker_subyacente
        """)).fetchall()
    if not rows:
        log("Sin datos de opciones AR en los ultimos 7 dias.")
        return
    log(f"Snapshots recientes ({len(rows)}):")
    for r in rows:
        log(f"  {r[0]:<8} {r[1]}  strikes={r[2]:>3}  IV_prom={r[3]}  PCR={r[4]}")


def cmd_run(tickers_filter: list[str], dry_run: bool, force: bool):
    if not force and not en_horario_bcba():
        log("BCBA cerrado (fuera de L-V 10:30-17:00 ART). Greeks serian null.")
        log("Usa --force para ejecutar de todas formas.")
        return

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT ticker FROM activos_ar "
            "WHERE activo = true AND tiene_opciones = true ORDER BY ticker"
        )).fetchall()

    tickers = [r[0] for r in rows]
    if tickers_filter:
        tickers = [t for t in tickers if t in tickers_filter]

    if not tickers:
        log("No hay tickers con opciones activos. Verifica activos_ar.tiene_opciones.")
        return

    hoy = date.today()
    log(f"Snapshot opciones AR: {len(tickers)} tickers — {hoy}")
    if dry_run:
        log("DRY RUN activado: no se guardara nada.")

    client = IOLClient()
    total_strikes = 0

    for i, ticker in enumerate(tickers, 1):
        log(f"  [{i:>2}/{len(tickers)}] {ticker}...")
        n = procesar_ticker(client, ticker, hoy, dry_run)
        total_strikes += n
        time.sleep(0.5)  # cortesia con la API

    modo = "DRY RUN" if dry_run else "GUARDADO"
    log(f"\n{modo}: {total_strikes} strikes procesados en {len(tickers)} tickers.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Snapshot diario de Greeks opciones BCBA."
    )
    parser.add_argument("--ticker",  nargs="+", help="Filtrar tickers especificos")
    parser.add_argument("--dry-run", action="store_true", help="Calcula pero no guarda")
    parser.add_argument("--status",  action="store_true", help="Muestra ultimos snapshots en DB")
    parser.add_argument("--force",   action="store_true", help="Ignora check de horario BCBA")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    else:
        cmd_run(
            tickers_filter=args.ticker or [],
            dry_run=args.dry_run,
            force=args.force,
        )


if __name__ == "__main__":
    main()
