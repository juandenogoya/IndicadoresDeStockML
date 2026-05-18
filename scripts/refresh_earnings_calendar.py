"""
refresh_earnings_calendar.py
Refresca la tabla earnings_calendar con la proxima fecha de earnings de
cada ticker, consultada al earnings calendar de Nasdaq.

Contexto:
    earnings_filter.py consultaba yfinance por cada ticker en cada corrida
    de cada bot (~1.500-1.800 llamadas/dia) -> rate limit por IP. Este script
    hace UNA pasada semanal y cachea el resultado en earnings_calendar.

Fuentes evaluadas (mayo 2026):
    - yfinance (Yahoo quoteSummary): throttle por IP a mitad de corrida.
    - FMP (financialmodelingprep): el plan free solo cubre 54/199 tickers.
    - Nasdaq earnings calendar: cobertura completa, gratis, indexado por
      FECHA -> se elige esta.

Endpoint usado (keyless, requiere headers de browser):
    GET https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD
    Devuelve TODAS las empresas que reportan ese dia. Es la API del sitio
    web de Nasdaq -- NO usa API key. (NASDAQ_API_KEY del .env es de Nasdaq
    Data Link, otro producto, y no aplica al earnings calendar.)

Como funciona:
    Recorre los proximos DIAS_HORIZONTE dias habiles. Por cada dia consulta
    que empresas reportan. La proxima fecha de earnings de un ticker es el
    primer dia (mas temprano) en que aparece. Un trimestre entra en ~63
    dias habiles, asi que el horizonte cubre un ciclo completo.

Arquitectura:
    - Corre en Oracle cron (semanal) -> escribe en Railway.
    - sync_railway_to_local.py baja la tabla a local (modo replace).
    - Para uso/test local: --target local o --target both.

Manejo de tickers sin fecha:
    Si un ticker no aparece en todo el horizonte, se guarda igual la fila
    con earnings_date = NULL. NULL es "sin balance conocido": earnings_filter
    no aplica restriccion y las estrategias siguen con sus otros criterios.

Uso:
    python scripts/refresh_earnings_calendar.py --target railway
    python scripts/refresh_earnings_calendar.py --target both
    python scripts/refresh_earnings_calendar.py --target local --dry-run
"""

import sys
import os
import json
import time
import argparse
import urllib.request
import urllib.error
from datetime import datetime, date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, text

SEP = "=" * 64

# Parametros
NASDAQ_URL        = "https://api.nasdaq.com/api/calendar/earnings"
DIAS_HORIZONTE    = 100   # dias calendario hacia adelante a escanear (~68 habiles)
SLEEP_ENTRE_DIAS  = 0.5   # segundos entre consultas (cortesia)
HTTP_TIMEOUT      = 30
MAX_REINTENTOS    = 2     # reintentos por dia ante error transitorio

# Headers de browser: api.nasdaq.com rechaza requests sin User-Agent valido
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Entorno ──────────────────────────────────────────────────────────────────

def _parse_env_file(path: str) -> dict:
    """Lee un .env y retorna dict clave->valor (sin tocar os.environ)."""
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


def get_railway_engine():
    """
    SQLAlchemy engine -> Railway.

    Busca DATABASE_URL en .env.local primero (caso Windows) y como fallback
    en .env (caso Oracle, que escribe a Railway y no tiene .env.local).
    """
    db_url = ""
    for fname in (".env.local", ".env"):
        env = _parse_env_file(os.path.join(ROOT, fname))
        db_url = env.get("DATABASE_URL", "").strip()
        if db_url:
            break
    if not db_url:
        raise ValueError("DATABASE_URL no encontrado en .env.local ni .env")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, connect_args={"sslmode": "require"})


def get_local_engine():
    """SQLAlchemy engine -> DB local (lee .env directamente)."""
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    name = env.get("DB_NAME", "activos_ml")
    user = env.get("DB_USER", "postgres")
    pwd  = env.get("DB_PASSWORD", "")
    return create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}")


# ── Datos ────────────────────────────────────────────────────────────────────

def obtener_tickers(engine) -> list[str]:
    """Lista de tickers desde la tabla activos."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT ticker FROM activos ORDER BY ticker"
        )).fetchall()
    return [r[0] for r in rows]


def consultar_dia(fecha_iso: str) -> set[str] | None:
    """
    Consulta el earnings calendar de Nasdaq para un dia.
    Retorna el set de simbolos que reportan ese dia, o None ante error.
    """
    url = f"{NASDAQ_URL}?date={fecha_iso}"
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            rows = ((data or {}).get("data") or {}).get("rows") or []
            return {
                r["symbol"].strip().upper()
                for r in rows
                if r.get("symbol")
            }
        except Exception:
            if intento < MAX_REINTENTOS:
                time.sleep(2.0)
            else:
                return None
    return None


def escanear_horizonte(tickers: list[str]) -> tuple[dict, int, int]:
    """
    Escanea los proximos DIAS_HORIZONTE dias habiles en el earnings calendar
    de Nasdaq. Retorna (datos, dias_consultados, dias_con_error).

    datos = {ticker: date | None}. La fecha es el primer dia (mas temprano)
    en que el ticker aparece reportando.
    """
    universo    = set(tickers)
    encontrados = {}   # ticker -> date (primera aparicion)
    hoy         = date.today()

    dias_consultados = 0
    dias_con_error   = 0

    for offset in range(DIAS_HORIZONTE + 1):
        d = hoy + timedelta(days=offset)
        if d.weekday() >= 5:           # sabado/domingo: Nasdaq no reporta
            continue

        simbolos = consultar_dia(d.isoformat())
        dias_consultados += 1

        if simbolos is None:
            dias_con_error += 1
        else:
            # iteramos en orden ascendente -> la primera aparicion es la mas temprana
            for s in (simbolos & universo):
                if s not in encontrados:
                    encontrados[s] = d

        if dias_consultados % 10 == 0:
            log(f"  progreso: {dias_consultados} dias habiles | "
                f"tickers con fecha: {len(encontrados)}")
        time.sleep(SLEEP_ENTRE_DIAS)

    datos = {t: encontrados.get(t) for t in tickers}
    return datos, dias_consultados, dias_con_error


# ── Upsert ───────────────────────────────────────────────────────────────────

def upsert_earnings(engine, etiqueta: str, datos: dict[str, date]):
    """Upsert de la tabla earnings_calendar (ON CONFLICT por ticker)."""
    sql = text("""
        INSERT INTO earnings_calendar (ticker, earnings_date, fecha_actualizacion)
        VALUES (:ticker, :earnings_date, NOW())
        ON CONFLICT (ticker) DO UPDATE
        SET earnings_date       = EXCLUDED.earnings_date,
            fecha_actualizacion = NOW()
    """)
    registros = [
        {"ticker": t, "earnings_date": f} for t, f in datos.items()
    ]
    with engine.connect() as conn:
        conn.execute(sql, registros)
        conn.commit()
    con_fecha = sum(1 for v in datos.values() if v is not None)
    log(f"[{etiqueta}] upsert OK: {len(datos)} tickers "
        f"({con_fecha} con fecha, {len(datos) - con_fecha} sin fecha)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Refresca earnings_calendar desde el earnings calendar de Nasdaq"
    )
    parser.add_argument("--target", choices=["railway", "local", "both"],
                        default="railway", help="Donde escribir (default railway)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Consulta Nasdaq pero no escribe en DB")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  REFRESH earnings_calendar (Nasdaq)  |  target={args.target}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    rail_eng  = get_railway_engine()
    local_eng = get_local_engine()

    src_eng = local_eng if args.target == "local" else rail_eng
    tickers = obtener_tickers(src_eng)
    log(f"Tickers a cubrir: {len(tickers)}")
    log(f"Horizonte: {DIAS_HORIZONTE} dias calendario hacia adelante")
    print()

    log("Escaneando el earnings calendar de Nasdaq (un request por dia habil)...")
    datos, dias_consultados, dias_con_error = escanear_horizonte(tickers)
    print()

    con_fecha = sum(1 for v in datos.values() if v is not None)
    sin_fecha = len(tickers) - con_fecha
    log(f"Dias habiles consultados: {dias_consultados} | con error: {dias_con_error}")
    log(f"Resultado: {con_fecha} tickers con fecha | {sin_fecha} sin fecha (NULL).")

    if dias_con_error > dias_consultados * 0.20:
        log(f"[ALERTA] {dias_con_error} dias con error "
            f"({dias_con_error / max(dias_consultados,1):.0%}). "
            f"Cobertura parcial: revisar conectividad con api.nasdaq.com.")
    print()

    if args.dry_run:
        log("DRY-RUN: no se escribe en DB.")
        ejemplos = [(t, str(f)) for t, f in datos.items() if f is not None][:8]
        log(f"  Ejemplos con fecha: {ejemplos}")
        print()
        log("Completado.")
        return

    if args.target in ("railway", "both"):
        upsert_earnings(rail_eng, "RAILWAY", datos)
    if args.target in ("local", "both"):
        upsert_earnings(local_eng, "LOCAL", datos)

    print()
    log("Completado.")
    print()


if __name__ == "__main__":
    main()
