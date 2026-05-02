"""
38_opciones_ar_snapshot.py
Snapshot diario de Greeks + PCR para opciones argentinas (BCBA).

IMPORTANTE: debe correr durante horario BCBA (10:30-17:00 ART = 13:30-20:00 UTC).
Fuera de ese horario la mayoria de los Greeks son null y el snapshot no tiene valor.
El snapshot de cierre (16:45 ART = 19:45 UTC) es el clave para capturar volumen acumulado.

Flujo:
  - Consulta activos_ar WHERE tiene_opciones = TRUE (~12 tickers)
  - Por cada ticker: get_options_chain via IOL REST API
  - Filtra solo strikes liquidos (IV no nula)
  - Guarda en opciones_ar_gregas con UNIQUE (fecha, symbol)   <- detalle por strike
  - Agrega por vencimiento y guarda en pcr_ar_diario           <- PCR diario limpio

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
from collections import defaultdict
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


# ── PCR por vencimiento ───────────────────────────────────────────────────────

SQL_UPSERT_PCR = """
    INSERT INTO pcr_ar_diario (
        fecha, ticker, expiration,
        vol_calls, vol_puts, pcr,
        n_strikes_calls, n_strikes_puts,
        iv_atm_call, iv_atm_put, iv_skew,
        spot_price, risk_free_rate, dias_vto
    ) VALUES (
        %(fecha)s, %(ticker)s, %(expiration)s,
        %(vol_calls)s, %(vol_puts)s, %(pcr)s,
        %(n_strikes_calls)s, %(n_strikes_puts)s,
        %(iv_atm_call)s, %(iv_atm_put)s, %(iv_skew)s,
        %(spot_price)s, %(risk_free_rate)s, %(dias_vto)s
    )
    ON CONFLICT (fecha, ticker, expiration) DO UPDATE SET
        vol_calls        = EXCLUDED.vol_calls,
        vol_puts         = EXCLUDED.vol_puts,
        pcr              = EXCLUDED.pcr,
        n_strikes_calls  = EXCLUDED.n_strikes_calls,
        n_strikes_puts   = EXCLUDED.n_strikes_puts,
        iv_atm_call      = EXCLUDED.iv_atm_call,
        iv_atm_put       = EXCLUDED.iv_atm_put,
        iv_skew          = EXCLUDED.iv_skew,
        spot_price       = EXCLUDED.spot_price,
        risk_free_rate   = EXCLUDED.risk_free_rate,
        dias_vto         = EXCLUDED.dias_vto
"""


def _iv_atm(strikes_data: list[dict], spot: float) -> float | None:
    """Devuelve la IV del strike mas cercano al spot. Retorna None si no hay datos."""
    candidates = [r for r in strikes_data if r.get("implied_volatility") is not None]
    if not candidates:
        return None
    closest = min(candidates, key=lambda r: abs(r.get("strike_price", 0) - spot))
    return closest.get("implied_volatility")


def calcular_pcr_por_vencimiento(records: list[dict], ticker: str, hoy: date) -> list[dict]:
    """
    Agrega los registros por vencimiento y calcula PCR + IV ATM + skew.

    records: lista de dicts tal como se persisten en opciones_ar_gregas
    Retorna: lista de dicts listos para INSERT en pcr_ar_diario
    """
    # Agrupar por expiration
    por_vto: dict[str, dict] = defaultdict(lambda: {
        "calls": [], "puts": [], "spot": None, "rfr": None, "dias_vto": None
    })

    for r in records:
        exp = r.get("expiration")
        if exp is None:
            continue
        exp_key = str(exp)
        ot = r.get("option_type", "")
        if ot == "C":
            por_vto[exp_key]["calls"].append(r)
        elif ot in ("V", "P"):
            por_vto[exp_key]["puts"].append(r)
        # Spot y RFR son iguales para todos los strikes del mismo ticker
        if por_vto[exp_key]["spot"] is None:
            por_vto[exp_key]["spot"] = r.get("spot_price")
            por_vto[exp_key]["rfr"] = r.get("risk_free_rate")
            # dias_vto: puede venir directo del record o calcularlo desde expiration
            dv = r.get("dias_vto")
            if dv is None and exp is not None:
                try:
                    exp_date = exp if isinstance(exp, date) else date.fromisoformat(str(exp))
                    dv = (exp_date - hoy).days
                except Exception:
                    dv = None
            por_vto[exp_key]["dias_vto"] = dv

    pcr_records = []
    for exp_key, data in por_vto.items():
        calls = data["calls"]
        puts  = data["puts"]
        spot  = data["spot"] or 0.0

        vol_calls = sum(int(r.get("volume") or 0) for r in calls)
        vol_puts  = sum(int(r.get("volume") or 0) for r in puts)

        pcr = None
        if vol_calls > 0:
            pcr = round(vol_puts / vol_calls, 4)

        iv_c = _iv_atm(calls, spot)
        iv_p = _iv_atm(puts,  spot)
        iv_skew = None
        if iv_c is not None and iv_p is not None:
            iv_skew = round(iv_p - iv_c, 6)

        # Parsear expiration como date
        try:
            exp_date = date.fromisoformat(str(exp_key))
        except Exception:
            continue

        pcr_records.append({
            "fecha":           hoy,
            "ticker":          ticker,
            "expiration":      exp_date,
            "vol_calls":       vol_calls,
            "vol_puts":        vol_puts,
            "pcr":             pcr,
            "n_strikes_calls": len(calls),
            "n_strikes_puts":  len(puts),
            "iv_atm_call":     round(iv_c, 6) if iv_c is not None else None,
            "iv_atm_put":      round(iv_p, 6) if iv_p is not None else None,
            "iv_skew":         iv_skew,
            "spot_price":      data["spot"],
            "risk_free_rate":  data["rfr"],
            "dias_vto":        data["dias_vto"],
        })

    return pcr_records


def persistir_pcr(pcr_records: list[dict]):
    if not pcr_records:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, SQL_UPSERT_PCR, pcr_records, page_size=100)
    log(f"  Guardados {len(pcr_records)} vencimientos en pcr_ar_diario.")


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

        # Resumen rapido de volumen total para el log
        vol_total = sum(int(r.get("volume") or 0) for r in records)
        log(f"  {ticker}: {len(records)} strikes liquidos  "
            f"spot={records[0].get('spot_price', 'N/A')}  "
            f"vol_total={vol_total}")

        if not dry_run:
            # 1) Detalle por strike
            persistir_gregas(records)
            # 2) PCR agregado por vencimiento
            pcr_records = calcular_pcr_por_vencimiento(records, ticker, hoy)
            persistir_pcr(pcr_records)
            if pcr_records:
                for p in sorted(pcr_records, key=lambda x: x["expiration"]):
                    pcr_str = f"{p['pcr']:.3f}" if p["pcr"] is not None else "N/A (sin vol)"
                    log(f"    PCR {p['expiration']} | "
                        f"calls={p['vol_calls']:>5}  puts={p['vol_puts']:>5}  "
                        f"PCR={pcr_str}  dias={p['dias_vto']}")
        else:
            # En dry-run: solo calcular y loggear
            pcr_records = calcular_pcr_por_vencimiento(records, ticker, hoy)
            if pcr_records:
                for p in sorted(pcr_records, key=lambda x: x["expiration"]):
                    pcr_str = f"{p['pcr']:.3f}" if p["pcr"] is not None else "N/A"
                    log(f"    [DRY] PCR {p['expiration']} | "
                        f"calls={p['vol_calls']:>5}  puts={p['vol_puts']:>5}  "
                        f"PCR={pcr_str}")

        return len(records)

    except Exception as e:
        log(f"  {ticker}: ERROR -- {e}")
        return 0


# ── Comandos ──────────────────────────────────────────────────────────────────

def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        # Detalle strikes
        rows = conn.execute(text("""
            SELECT ticker_subyacente,
                   fecha,
                   COUNT(*)                                       AS n_strikes,
                   ROUND(AVG(implied_volatility)::numeric, 4)    AS iv_promedio,
                   SUM(CASE WHEN option_type='C' THEN COALESCE(volume,0) ELSE 0 END) AS vol_calls,
                   SUM(CASE WHEN option_type IN ('V','P') THEN COALESCE(volume,0) ELSE 0 END) AS vol_puts
            FROM opciones_ar_gregas
            WHERE fecha >= CURRENT_DATE - 7
            GROUP BY ticker_subyacente, fecha
            ORDER BY fecha DESC, ticker_subyacente
        """)).fetchall()

        # PCR por vencimiento
        pcr_rows = conn.execute(text("""
            SELECT ticker, fecha, expiration, vol_calls, vol_puts, pcr, dias_vto
            FROM pcr_ar_diario
            WHERE fecha >= CURRENT_DATE - 7
            ORDER BY fecha DESC, ticker, expiration
        """)).fetchall()

    if not rows:
        log("Sin datos de opciones AR en los ultimos 7 dias.")
    else:
        log(f"opciones_ar_gregas — ultimos 7 dias ({len(rows)} grupos ticker/fecha):")
        for r in rows:
            pcr_calc = None
            if r[5] and r[4]:  # vol_puts / vol_calls
                try:
                    pcr_calc = round(r[5] / r[4], 3)
                except Exception:
                    pass
            log(f"  {r[0]:<8} {r[1]}  strikes={r[2]:>3}  IV_prom={r[3]}  "
                f"vol_C={r[4]:>6}  vol_P={r[5]:>6}  PCR={pcr_calc}")

    if pcr_rows:
        log(f"\npcr_ar_diario — ultimos 7 dias ({len(pcr_rows)} filas):")
        for r in pcr_rows:
            pcr_str = f"{r[5]:.3f}" if r[5] is not None else "N/A"
            log(f"  {r[0]:<8} {r[1]}  vto={r[2]}  C={r[3]:>5}  P={r[4]:>5}  "
                f"PCR={pcr_str}  dias={r[6]}")
    else:
        log("\npcr_ar_diario: sin datos en los ultimos 7 dias.")


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
