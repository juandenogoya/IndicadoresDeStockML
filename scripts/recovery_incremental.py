"""
recovery_incremental.py
Recovery incremental de precios_diarios + futuros_diarios + indicadores tecnicos.

Resuelve el problema del script ciego de cron_diario --step precios, que reporta
"199/199 OK" aunque yfinance devuelva datos parciales. Este script:

  1. Detecta qué tickers tienen MAX(fecha) < ultimo dia habil
  2. Agrupa por rango de fechas faltantes
  3. Descarga SOLO los pendientes, en batches chicos (default 3)
  4. Verifica explicitamente que cada upsert quedó persistido
  5. Repite hasta completar o agotar reintentos
  6. Reporta tickers que NO se pudieron completar (no oculta fallos)

Uso:
    python scripts/recovery_incremental.py                       # local, default
    python scripts/recovery_incremental.py --target local        # idem
    python scripts/recovery_incremental.py --target railway      # apunta a Railway
    python scripts/recovery_incremental.py --dry-run             # solo diagnostico
    python scripts/recovery_incremental.py --batch-size 5        # mas tickers por download
    python scripts/recovery_incremental.py --max-cycles 8        # mas reintentos
    python scripts/recovery_incremental.py --skip-futuros        # solo precios
    python scripts/recovery_incremental.py --skip-indicadores    # solo OHLCV, no recalcula

Codigo de salida:
    0 = todo OK, lista de pendientes vacia al final
    1 = quedaron tickers sin completar (ver log)
"""

import os
import sys
import time
import argparse
from datetime import date, datetime, timedelta
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ── Constantes ────────────────────────────────────────────────────────────────
BATCH_SIZE_DEFAULT       = 3
PAUSE_BETWEEN_BATCHES_S  = 3
PAUSE_AFTER_RATELIMIT_S  = 300   # 5 min
MAX_CYCLES_DEFAULT       = 5
INDICADOR_HIST_MIN_BARS  = 250   # mínimo historico para recalcular indicadores

FUTUROS_SYMBOLS = [
    "ES=F", "YM=F", "NQ=F", "RTY=F",  # indices US (los criticos)
    "GC=F", "SI=F",                    # metales
    "UB=F", "TN=F",                    # bonos
    "ZL=F", "ZM=F",                    # agricolas
    "XAE=F", "XAU=F",                  # otros
]

SEP = "=" * 70


# ── Env loader (sin tocar os.environ globalmente hasta saber target) ──────────

def _parse_env_file(path: str) -> dict:
    """Lee un archivo .env y retorna dict clave -> valor."""
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


def setup_target_env(target: str):
    """
    Configura os.environ para que get_engine() del proyecto apunte al target deseado.

    target='local'   -> elimina DATABASE_URL para que get_engine() use DB_CONFIG (local)
    target='railway' -> setea DATABASE_URL al DSN de Railway desde .env.local
    """
    env_general = _parse_env_file(os.path.join(ROOT, ".env"))
    env_local   = _parse_env_file(os.path.join(ROOT, ".env.local"))

    # Telegram / IOL / etc. desde .env, sin pisar lo que ya esté
    for k, v in env_general.items():
        os.environ.setdefault(k, v)

    if target == "local":
        # DB_CONFIG (host, port, name, user, password) ya esta en env_general.
        # get_engine() solo usa DB_CONFIG si DATABASE_URL NO esta seteada.
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
    else:
        rurl = env_local.get("DATABASE_URL", "").strip()
        if not rurl:
            raise ValueError("DATABASE_URL no encontrado en .env.local")
        os.environ["DATABASE_URL"] = rurl


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Excepcion para rate limit ─────────────────────────────────────────────────

class RateLimitDetected(Exception):
    pass


def _is_ratelimit_msg(err: str) -> bool:
    return ("Too Many Requests" in err
            or "Rate limited" in err
            or "YFRateLimit" in err
            or "429" in err)


# ── Lectura de estado de la DB ────────────────────────────────────────────────

def get_pendientes_precios(engine, target_date: date) -> dict:
    """
    Retorna dict {ticker: max_fecha} para tickers cuyo MAX(fecha) < target_date.
    Si ticker no existe en precios_diarios, max_fecha es None.
    """
    import pandas as pd
    from sqlalchemy import text

    with engine.connect() as c:
        tickers_all = pd.read_sql(
            text("SELECT DISTINCT ticker FROM activos WHERE activo = TRUE ORDER BY ticker"),
            c
        )["ticker"].tolist()
        df_max = pd.read_sql(
            text("SELECT ticker, MAX(fecha) AS max_fecha FROM precios_diarios GROUP BY ticker"),
            c
        )
        max_map = {r["ticker"]: r["max_fecha"] for _, r in df_max.iterrows()}

    pendientes = {}
    for t in tickers_all:
        mf = max_map.get(t)
        if mf is None or mf < target_date:
            pendientes[t] = mf
    return pendientes


def get_pendientes_futuros(engine, target_date: date) -> dict:
    """
    Idem para futuros: {symbol: max_fecha} para los que MAX(fecha) < target_date.
    """
    import pandas as pd
    from sqlalchemy import text

    with engine.connect() as c:
        df_max = pd.read_sql(
            text("SELECT ticker, MAX(fecha) AS max_fecha FROM futuros_diarios GROUP BY ticker"),
            c
        )
        max_map = {r["ticker"]: r["max_fecha"] for _, r in df_max.iterrows()}

    pendientes = {}
    for s in FUTUROS_SYMBOLS:
        mf = max_map.get(s)
        if mf is None or mf < target_date:
            pendientes[s] = mf
    return pendientes


def agrupar_por_rango(pendientes: dict, target_date: date) -> dict:
    """
    Agrupa tickers por rango (start, end) de fechas a descargar.
    Retorna {(start, end): [tickers]}.
    """
    grupos = {}
    for ticker, max_fecha in pendientes.items():
        if max_fecha is None:
            start = target_date - timedelta(days=400)   # 1 anio approx
        else:
            start = max_fecha + timedelta(days=1)
        end = target_date
        if start > end:
            continue
        key = (start, end)
        grupos.setdefault(key, []).append(ticker)
    return grupos


# ── Descarga via yfinance ─────────────────────────────────────────────────────

def download_batch(tickers: list, start: date, end: date) -> dict:
    """
    Descarga OHLCV via yf.download() en batch. Retorna {ticker: DataFrame}.
    Levanta RateLimitDetected si hay rate limit (yfinance lo imprime a
    stdout/stderr en lugar de lanzar excepcion).
    """
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    import yfinance as yf
    import pandas as pd

    out_buf = io.StringIO()
    err_buf = io.StringIO()

    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            df = yf.download(
                tickers,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                group_by="ticker",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
    except Exception as e:
        if _is_ratelimit_msg(str(e)):
            raise RateLimitDetected(str(e))
        log(f"  yf.download EXCEPCION: {str(e)[:120]}")
        return {}

    # yfinance escribe los rate limit a stdout/stderr en lugar de lanzar
    captured = (out_buf.getvalue() + " " + err_buf.getvalue()).strip()
    if captured and _is_ratelimit_msg(captured):
        raise RateLimitDetected(captured[:200])

    if df is None or df.empty:
        return {}

    result = {}
    if len(tickers) == 1:
        # yf con 1 ticker: columnas planas
        t = tickers[0]
        clean = df.dropna(how="all")
        if not clean.empty:
            result[t] = clean
    else:
        for t in tickers:
            try:
                tdf = df[t].dropna(how="all")
                if not tdf.empty:
                    result[t] = tdf
            except (KeyError, AttributeError):
                pass
    return result


# ── Upsert + recalculo de indicadores ─────────────────────────────────────────

def df_yf_a_df_db(df_yf, ticker: str):
    """Convierte el DataFrame que devuelve yfinance al formato de precios_diarios."""
    import pandas as pd
    if df_yf is None or df_yf.empty:
        return pd.DataFrame()

    # yfinance ya viene con index=date y columnas Open/High/Low/Close/Adj Close/Volume
    out = pd.DataFrame({
        "fecha":     pd.to_datetime(df_yf.index).normalize(),
        "open":      df_yf["Open"].astype("float64"),
        "high":      df_yf["High"].astype("float64"),
        "low":       df_yf["Low"].astype("float64"),
        "close":     df_yf["Close"].astype("float64"),
        "adj_close": (df_yf["Adj Close"] if "Adj Close" in df_yf.columns else df_yf["Close"]).astype("float64"),
        "volume":    df_yf["Volume"].fillna(0).astype("int64"),
        "ticker":    ticker,
    })
    return out.dropna(subset=["close"])


def procesar_ticker(ticker: str, df_yf, engine, recalc_indicadores: bool) -> bool:
    """
    Upsert OHLCV + (opcional) recalculo de indicadores tecnicos.
    Retorna True si quedo persistido OK.
    """
    df_db = df_yf_a_df_db(df_yf, ticker)
    if df_db.empty:
        return False

    from src.data.database import upsert_precios
    upsert_precios(df_db)

    if recalc_indicadores:
        try:
            from src.pipeline.data_manager import cargar_precios_db
            from src.indicators.technical import procesar_indicadores_ticker
            df_full = cargar_precios_db(ticker, ultimas_n=500)
            if len(df_full) >= INDICADOR_HIST_MIN_BARS:
                procesar_indicadores_ticker(ticker, df_full, guardar_db=True)
        except Exception as e:
            log(f"    [WARN] indicadores {ticker}: {str(e)[:80]}")

    return True


def procesar_futuro(symbol: str, df_yf, engine) -> bool:
    """
    Upsert OHLCV en futuros_diarios. No recalculamos indicadores futuros
    (lo hace 35_calcular_indicadores_futuros.py despues si hace falta).
    """
    import pandas as pd
    from sqlalchemy import text

    df_db = df_yf_a_df_db(df_yf, symbol)
    if df_db.empty:
        return False

    # Upsert directo a futuros_diarios. La tabla espera columnas: fecha, ticker,
    # open, high, low, close, adj_close, volume. ON CONFLICT (ticker, fecha).
    rows = df_db.to_dict("records")
    with engine.connect() as c:
        for r in rows:
            c.execute(text("""
                INSERT INTO futuros_diarios
                  (fecha, ticker, open, high, low, close, adj_close, volume)
                VALUES
                  (:fecha, :ticker, :open, :high, :low, :close, :adj_close, :volume)
                ON CONFLICT (ticker, fecha) DO UPDATE SET
                  open = EXCLUDED.open,
                  high = EXCLUDED.high,
                  low  = EXCLUDED.low,
                  close= EXCLUDED.close,
                  adj_close = EXCLUDED.adj_close,
                  volume    = EXCLUDED.volume
            """), r)
        c.commit()
    return True


# ── Orchestracion por seccion ────────────────────────────────────────────────

def recover_precios(engine, target_date: date, args) -> dict:
    """
    Loop incremental para precios_diarios. Retorna {ticker: max_fecha} de los que
    no se pudieron completar.
    """
    log(SEP)
    log(f"PRECIOS DIARIOS  --  target_date = {target_date}")
    log(SEP)

    for cycle in range(1, args.max_cycles + 1):
        pendientes = get_pendientes_precios(engine, target_date)
        log("")
        log(f"--- Ciclo {cycle}/{args.max_cycles} ---  Pendientes: {len(pendientes)}")

        if not pendientes:
            log("OK -- no quedan pendientes en precios.")
            return {}

        if args.dry_run:
            log("DRY-RUN -- listado de pendientes:")
            for t, mf in sorted(pendientes.items())[:50]:
                log(f"  {t:<8s}  max={mf}")
            if len(pendientes) > 50:
                log(f"  ... y {len(pendientes) - 50} mas")
            return pendientes

        grupos = agrupar_por_rango(pendientes, target_date)
        ok_cycle, fail_cycle, vacios_cycle = 0, 0, 0
        rate_limited = False

        for (start, end), tickers in grupos.items():
            log(f"\nRango [{start} -> {end}]  ({len(tickers)} tickers)")

            for i in range(0, len(tickers), args.batch_size):
                batch = tickers[i:i + args.batch_size]
                try:
                    res = download_batch(batch, start, end)
                except RateLimitDetected:
                    log(f"  [RATE-LIMIT] pausa {PAUSE_AFTER_RATELIMIT_S}s y reintentamos en proximo ciclo")
                    rate_limited = True
                    break

                for t in batch:
                    if t in res and not res[t].empty:
                        try:
                            if procesar_ticker(t, res[t], engine, not args.skip_indicadores):
                                ok_cycle += 1
                                log(f"  [OK]   {t}  (filas={len(res[t])})")
                            else:
                                vacios_cycle += 1
                                log(f"  [VACIO] {t} (df_yf no upsertable)")
                        except Exception as e:
                            fail_cycle += 1
                            log(f"  [FAIL] {t}: {str(e)[:80]}")
                    else:
                        vacios_cycle += 1
                        log(f"  [---]  {t} (yfinance sin datos)")

                time.sleep(PAUSE_BETWEEN_BATCHES_S)

            if rate_limited:
                break

        log(f"\nResumen ciclo {cycle}: OK={ok_cycle}  FAIL={fail_cycle}  VACIOS={vacios_cycle}")

        if rate_limited:
            log(f"Durmiendo {PAUSE_AFTER_RATELIMIT_S}s antes de proximo ciclo...")
            time.sleep(PAUSE_AFTER_RATELIMIT_S)
        elif ok_cycle == 0 and len(pendientes) > 0:
            # Salvaguardia: ningun ticker se actualizo. Sospechoso de rate limit silencioso.
            log(f"Ciclo sin exitos. Pausa {PAUSE_AFTER_RATELIMIT_S}s antes de reintentar...")
            time.sleep(PAUSE_AFTER_RATELIMIT_S)

    pendientes_final = get_pendientes_precios(engine, target_date)
    return pendientes_final


def recover_futuros(engine, target_date: date, args) -> dict:
    """Idem para futuros_diarios."""
    log("\n" + SEP)
    log(f"FUTUROS DIARIOS  --  target_date = {target_date}")
    log(SEP)

    for cycle in range(1, args.max_cycles + 1):
        pendientes = get_pendientes_futuros(engine, target_date)
        log("")
        log(f"--- Ciclo {cycle}/{args.max_cycles} ---  Pendientes: {len(pendientes)}")

        if not pendientes:
            log("OK -- no quedan pendientes en futuros.")
            return {}

        if args.dry_run:
            log("DRY-RUN -- listado de pendientes:")
            for s, mf in sorted(pendientes.items()):
                log(f"  {s:<8s}  max={mf}")
            return pendientes

        grupos = agrupar_por_rango(pendientes, target_date)
        ok_cycle, fail_cycle = 0, 0
        rate_limited = False

        for (start, end), syms in grupos.items():
            log(f"\nRango [{start} -> {end}]  ({len(syms)} simbolos)")
            for i in range(0, len(syms), args.batch_size):
                batch = syms[i:i + args.batch_size]
                try:
                    res = download_batch(batch, start, end)
                except RateLimitDetected:
                    log(f"  [RATE-LIMIT] pausa {PAUSE_AFTER_RATELIMIT_S}s y reintentamos en proximo ciclo")
                    rate_limited = True
                    break

                for s in batch:
                    if s in res and not res[s].empty:
                        try:
                            if procesar_futuro(s, res[s], engine):
                                ok_cycle += 1
                                log(f"  [OK]   {s}  (filas={len(res[s])})")
                            else:
                                log(f"  [VACIO] {s}")
                        except Exception as e:
                            fail_cycle += 1
                            log(f"  [FAIL] {s}: {str(e)[:80]}")
                    else:
                        log(f"  [---]  {s} (yfinance sin datos)")
                time.sleep(PAUSE_BETWEEN_BATCHES_S)

            if rate_limited:
                break

        log(f"\nResumen ciclo {cycle}: OK={ok_cycle}  FAIL={fail_cycle}")

        if rate_limited:
            time.sleep(PAUSE_AFTER_RATELIMIT_S)
        elif ok_cycle == 0 and len(pendientes) > 0:
            log(f"Ciclo sin exitos. Pausa {PAUSE_AFTER_RATELIMIT_S}s antes de reintentar...")
            time.sleep(PAUSE_AFTER_RATELIMIT_S)

    pendientes_final = get_pendientes_futuros(engine, target_date)
    return pendientes_final


# ── Determinar fecha target ───────────────────────────────────────────────────

def calcular_target_date() -> date:
    """
    Ultimo dia habil cuyos datos deberian estar disponibles.
    Regla: si UTC < 21:00, el cierre de hoy NO esta cerrado todavia,
    asi que el target es el dia habil anterior.
    """
    from src.utils.trading_calendar import is_trading_day, prev_trading_day

    hoy = date.today()
    target = hoy if is_trading_day(hoy) else prev_trading_day(hoy)

    utc_now = datetime.utcnow()
    if target == hoy and utc_now.hour < 21:
        target = prev_trading_day(target)
    return target


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Recovery incremental de precios_diarios + futuros_diarios"
    )
    parser.add_argument("--target", choices=["local", "railway"], default="local",
                        help="DB destino (default: local)")
    parser.add_argument("--max-cycles", type=int, default=MAX_CYCLES_DEFAULT,
                        help=f"Maximo de ciclos de reintento (default: {MAX_CYCLES_DEFAULT})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
                        help=f"Tickers por lote en yf.download (default: {BATCH_SIZE_DEFAULT})")
    parser.add_argument("--skip-futuros", action="store_true",
                        help="Saltea la seccion de futuros")
    parser.add_argument("--skip-indicadores", action="store_true",
                        help="No recalcula indicadores tecnicos (solo OHLCV)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Diagnostico: muestra pendientes sin descargar")
    args = parser.parse_args()

    setup_target_env(args.target)

    # Imports DESPUES de setup_target_env (modulos pueden leer DATABASE_URL al importar)
    from src.data.database import get_engine

    engine = get_engine()
    target_date = calcular_target_date()

    print()
    print(SEP)
    print(f"  RECOVERY INCREMENTAL  |  target_DB={args.target.upper()}  |  fecha={target_date}")
    print(f"  batch_size={args.batch_size}  max_cycles={args.max_cycles}")
    if args.dry_run:
        print(f"  MODO DRY-RUN (sin escritura)")
    if args.skip_futuros:
        print(f"  SKIP futuros")
    if args.skip_indicadores:
        print(f"  SKIP indicadores")
    print(SEP)
    print()

    # PRECIOS
    pend_precios = recover_precios(engine, target_date, args)

    # FUTUROS
    pend_futuros = {}
    if not args.skip_futuros:
        pend_futuros = recover_futuros(engine, target_date, args)

    # ── REPORTE FINAL ─────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"  RESUMEN FINAL  ({args.target.upper()})")
    print(SEP)

    if pend_precios:
        print(f"  PRECIOS: {len(pend_precios)} tickers NO COMPLETADOS")
        for t, mf in sorted(pend_precios.items()):
            print(f"    !! {t:<8s}  max_fecha={mf}")
    else:
        print(f"  PRECIOS: OK -- todos los tickers al dia ({target_date})")

    if not args.skip_futuros:
        if pend_futuros:
            print(f"\n  FUTUROS: {len(pend_futuros)} simbolos NO COMPLETADOS")
            for s, mf in sorted(pend_futuros.items()):
                print(f"    !! {s:<8s}  max_fecha={mf}")
        else:
            print(f"\n  FUTUROS: OK -- todos los simbolos al dia ({target_date})")
    print(SEP)
    print()

    sys.exit(0 if not pend_precios and not pend_futuros else 1)


if __name__ == "__main__":
    main()
