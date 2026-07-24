"""
refresh_earnings_historico.py
Puebla earnings_historico con la fecha de anuncio de cada balance por trimestre,
desde Alpha Vantage (funcion EARNINGS). Fuente de la variable que faltaba para
analizar la reaccion del precio a los balances.

POR QUE ALPHA VANTAGE:
    Una sola llamada por ticker devuelve TODA la historia trimestral con:
      - fiscalDateEnding  -> fiscal_period_end (empata con fundamentales_*_q)
      - reportedDate      -> announcement_date (la fecha REAL del anuncio)
      - reportTime        -> report_time (pre-market / post-market)
    (Trae ademas EPS/estimados/sorpresa, que NO usamos: solo queremos el hecho
    duro de cuando reporto, para cruzar con el comportamiento del precio.)

RESTRICCION DE CUOTA (key free): 25 llamadas/dia, 5/min. El backfill de ~200
    tickers no entra en una corrida -> este script es REANUDABLE y CUOTA-AWARE:
      - Trae hasta --max-calls tickers por corrida (default 20, margen bajo 25).
      - Pausa PAUSA_SEG entre llamadas (>= 12s => <= 5/min).
      - Se corta limpio; la proxima corrida sigue por los que faltan.
    Backfill inicial: correr el .bat 1 vez/dia ~10 dias y queda completo.

TRES CASOS, UN MECANISMO (todos "conseguir 1 llamada por ticker que lo necesita"):
    1. backfill inicial  -> tickers SIN filas en earnings_historico.
    2. ticker nuevo      -> idem: al darlo de alta no tiene filas, entra solo.
    3. incremental       -> tickers cuya proxima fecha (earnings_calendar) ya
                            paso desde el ultimo fetch -> re-consulta para
                            apendicear el Q recien reportado.

    Prioridad: primero los SIN historia (1 y 2), luego los desactualizados (3).

LOCAL-only. Como AV devuelve historia completa por llamada, cada fetch REEMPLAZA
la historia del ticker (upsert por (ticker, fiscal_period_end)); idempotente.

Uso:
    python scripts/refresh_earnings_historico.py --backfill            # llena faltantes (<=20)
    python scripts/refresh_earnings_historico.py --backfill --max-calls 5
    python scripts/refresh_earnings_historico.py                       # incremental (default)
    python scripts/refresh_earnings_historico.py --ticker HOOD         # uno puntual (alta)
    python scripts/refresh_earnings_historico.py --status              # que falta / que hay
    python scripts/refresh_earnings_historico.py --backfill --dry-run
"""

import sys
import os
import time
import argparse
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import requests
from sqlalchemy import create_engine, text

SEP = "=" * 64

AV_URL          = "https://www.alphavantage.co/query"
DESDE_ANIO      = 2020    # historia minima que queremos (announcement_date >= 2020-01-01)
MAX_CALLS_DEF   = 20      # margen bajo el tope free de 25/dia
PAUSA_SEG       = 13      # >= 12s entre llamadas => <= 5/min (limite free)
HTTP_TIMEOUT    = 30


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Entorno (LOCAL-only) ──────────────────────────────────────────────────────

def _parse_env_file(path: str) -> dict:
    result = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                result[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return result


_ENV = _parse_env_file(os.path.join(ROOT, ".env"))


def get_local_engine():
    host = _ENV.get("DB_HOST", "localhost")
    port = _ENV.get("DB_PORT", "5432")
    name = _ENV.get("DB_NAME", "activos_ml")
    user = _ENV.get("DB_USER", "postgres")
    pwd  = _ENV.get("DB_PASSWORD", "")
    return create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}")


def _api_key() -> str:
    k = _ENV.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if not k:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY no esta en .env")
    return k


# ── Seleccion de tickers a traer ──────────────────────────────────────────────

def universo_local(engine) -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text(
            "SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker"
        )).fetchall()]


def tickers_sin_historia(engine) -> list[str]:
    """Universo vivo que NO tiene ninguna fila en earnings_historico."""
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("""
            SELECT a.ticker FROM activos a
            LEFT JOIN earnings_historico e ON e.ticker = a.ticker
            WHERE a.activo = TRUE AND e.ticker IS NULL
            ORDER BY a.ticker
        """)).fetchall()]


def tickers_desactualizados(engine) -> list[str]:
    """
    Tickers CON historia cuya proxima fecha conocida (earnings_calendar) ya paso
    respecto de nuestra ultima announcement_date -> hay un Q nuevo por traer.
    """
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("""
            SELECT a.ticker
            FROM activos a
            JOIN earnings_calendar c ON c.ticker = a.ticker
            JOIN (SELECT ticker, MAX(announcement_date) md
                  FROM earnings_historico GROUP BY ticker) e ON e.ticker = a.ticker
            WHERE a.activo = TRUE
              AND c.earnings_date IS NOT NULL
              AND c.earnings_date <= CURRENT_DATE       -- la fecha ya paso
              AND c.earnings_date  > e.md               -- y es posterior a lo que tenemos
            ORDER BY a.ticker
        """)).fetchall()]


# ── Alpha Vantage ─────────────────────────────────────────────────────────────

def fetch_av_earnings(ticker: str, key: str) -> list[dict] | str:
    """
    Trae quarterlyEarnings de Alpha Vantage. Devuelve lista de filas listas para
    upsert, o un string con el motivo si no hubo datos (rate limit / vacio).
    """
    try:
        r = requests.get(AV_URL, params={
            "function": "EARNINGS", "symbol": ticker, "apikey": key,
        }, timeout=HTTP_TIMEOUT)
        d = r.json()
    except Exception as e:
        return f"ERROR HTTP: {str(e)[:80]}"

    # AV senaliza rate limit / problemas con 'Note' o 'Information'
    if "Note" in d or "Information" in d:
        return "RATE_LIMIT: " + str(d.get("Note") or d.get("Information"))[:100]

    q = d.get("quarterlyEarnings")
    if not q:
        return "SIN quarterlyEarnings"

    filas = []
    for x in q:
        fpe = x.get("fiscalDateEnding")
        rep = x.get("reportedDate")
        if not fpe or not rep or rep in ("None", "null"):
            continue
        if rep < f"{DESDE_ANIO}-01-01":     # recorte a la historia que queremos
            continue
        filas.append({
            "ticker": ticker,
            "fiscal_period_end": fpe,
            "announcement_date": rep,
            "report_time": (x.get("reportTime") or "").strip() or None,
        })
    return filas


def upsert(engine, filas: list[dict]) -> int:
    if not filas:
        return 0
    sql = text("""
        INSERT INTO earnings_historico
            (ticker, fiscal_period_end, announcement_date, report_time, fetched_at)
        VALUES (:ticker, :fiscal_period_end, :announcement_date, :report_time, NOW())
        ON CONFLICT (ticker, fiscal_period_end) DO UPDATE
        SET announcement_date = EXCLUDED.announcement_date,
            report_time       = EXCLUDED.report_time,
            fetched_at        = NOW()
    """)
    with engine.connect() as conn:
        conn.execute(sql, filas)
        conn.commit()
    return len(filas)


# ── Runner ────────────────────────────────────────────────────────────────────

def procesar(engine, tickers: list[str], key: str, max_calls: int,
             dry_run: bool) -> tuple[int, int, bool]:
    """
    Procesa hasta max_calls tickers. Devuelve (ok, filas_total, corte_rate_limit).
    Se detiene apenas AV senaliza rate limit (no tiene sentido seguir gastando).
    """
    ok = filas_tot = 0
    for i, t in enumerate(tickers[:max_calls], 1):
        res = fetch_av_earnings(t, key)
        if isinstance(res, str):
            if res.startswith("RATE_LIMIT"):
                log(f"  [{i}/{min(len(tickers),max_calls)}] {t:<8s} {res}")
                log("  -> cuota diaria agotada. Corte limpio; reanudar manana.")
                return ok, filas_tot, True
            log(f"  [{i}/{min(len(tickers),max_calls)}] {t:<8s} {res}")
        else:
            n = len(res) if dry_run else upsert(engine, res)
            filas_tot += n
            ok += 1
            rango = f"{res[-1]['announcement_date']}..{res[0]['announcement_date']}" if res else "-"
            log(f"  [{i}/{min(len(tickers),max_calls)}] {t:<8s} {n:2d} Q  ({rango})"
                f"{'  [DRY]' if dry_run else ''}")
        if i < min(len(tickers), max_calls):
            time.sleep(PAUSA_SEG)
    return ok, filas_tot, False


def cmd_status(engine):
    sin = tickers_sin_historia(engine)
    desa = tickers_desactualizados(engine)
    with engine.connect() as conn:
        con = conn.execute(text(
            "SELECT COUNT(DISTINCT ticker) FROM earnings_historico")).scalar()
        tot = conn.execute(text("SELECT COUNT(*) FROM earnings_historico")).scalar()
        rango = conn.execute(text(
            "SELECT MIN(announcement_date), MAX(announcement_date) "
            "FROM earnings_historico")).fetchone()
    log(f"Con historia    : {con} tickers ({tot} filas, "
        f"{rango[0]} -> {rango[1]})")
    log(f"Sin historia    : {len(sin)}  {sin[:12]}{' ...' if len(sin) > 12 else ''}")
    log(f"Desactualizados : {len(desa)}  {desa[:12]}{' ...' if len(desa) > 12 else ''}")
    dias = -(-len(sin) // MAX_CALLS_DEF) if sin else 0
    if sin:
        log(f"Backfill restante: ~{dias} corridas de {MAX_CALLS_DEF} (key free 25/dia).")


def main():
    ap = argparse.ArgumentParser(
        description="Puebla earnings_historico desde Alpha Vantage (cuota-aware)")
    ap.add_argument("--backfill", action="store_true",
                    help="Llena tickers SIN historia (prioridad) + desactualizados")
    ap.add_argument("--ticker", help="Un ticker puntual (uso en alta de universo)")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS_DEF,
                    help=f"Maximo de llamadas por corrida (default {MAX_CALLS_DEF})")
    ap.add_argument("--status", action="store_true", help="Muestra que falta y sale")
    ap.add_argument("--dry-run", action="store_true", help="No escribe en DB")
    args = ap.parse_args()

    engine = get_local_engine()

    print()
    print(SEP)
    print(f"  REFRESH earnings_historico (Alpha Vantage, LOCAL)"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)

    if args.status:
        cmd_status(engine)
        return

    key = _api_key()

    if args.ticker:
        cola = [args.ticker.upper()]
        modo = f"ticker puntual {cola[0]}"
    elif args.backfill:
        # Prioridad: primero los que no tienen NADA, luego los desactualizados.
        cola = tickers_sin_historia(engine) + tickers_desactualizados(engine)
        modo = "backfill (sin historia + desactualizados)"
    else:
        cola = tickers_desactualizados(engine)
        modo = "incremental (desactualizados)"

    log(f"Modo: {modo}")
    if not cola:
        log("Nada que traer. earnings_historico al dia.")
        return
    log(f"En cola: {len(cola)} | procesare hasta {args.max_calls} esta corrida.")
    log(f"Pausa entre llamadas: {PAUSA_SEG}s (limite free 5/min).")
    print()

    ini = time.time()
    ok, filas, corte = procesar(engine, cola, key, args.max_calls, args.dry_run)
    print()
    log(f"Procesados OK   : {ok} tickers | {filas} filas {'(simuladas)' if args.dry_run else 'upserted'}")
    restan = len(cola) - ok
    if restan > 0 and not corte:
        log(f"Quedan {restan} en cola -> correr de nuevo (otra tanda).")
    elif restan > 0 and corte:
        log(f"Quedan {restan} en cola -> reanudar en la proxima ventana de cuota.")
    log(f"Tiempo: {time.time()-ini:.0f}s")
    print()


if __name__ == "__main__":
    main()
