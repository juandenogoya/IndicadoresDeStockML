"""
universo.py -- Alta / baja de tickers del universo (Tarea 14).

El universo se sirve de la tabla `activos` (fuente unica desde el 18/6, ver
src/data/universo.py). Este script da de ALTA o BAJA un ticker de forma prolija,
considerando todas las aristas:

ALTA (add):
  1. Descarga 2 anios de OHLCV (yahooquery) -> precios_diarios (LOCAL).
  2. UPSERT en `activos` (sector/industry, activo=TRUE, modelo_asignado=global_rf)
     -> DUAL-WRITE local + Railway (el snapshot corre Oracle->Railway y debe verlo).
  3. Indicadores tecnicos + features (market structure / precio accion).
  4. Z-scores de acciones (ticker_zscore_diario).
  5. Analisis fundamental (balances -> ratios -> comparativa sectorial).
  6. Log en universo_cambios.
  El ticker nace CALIENTE: como todo deriva de la historia de precios, los
  indicadores/z-scores quedan completos dia 1 (sin esperas de warmup). El unico
  residuo es el z-score de OPCIONES (~60d), que no es backfilleable.

BAJA (remove):
  - Guard: si hay posiciones FT abiertas (ft_operaciones, LOCAL) -> ABORTA salvo
    --force (las posiciones Alpaca paper se reportan, no bloquean).
  - SOFT delete: activos.activo=FALSE -> DUAL-WRITE local + Railway. Conserva
    TODA la historia (point-in-time).
  - Log en universo_cambios.

Solo ACCIONES (los ETF quedan para una 2da pasada).

Uso:
    python scripts/manual/universo.py add NVDA --sector Technology --industry Semiconductors
    python scripts/manual/universo.py add KO  --sector "Consumer Defensive" --dry-run
    python scripts/manual/universo.py remove XYZ --motivo "delisting"
    python scripts/manual/universo.py list
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Entorno LOCAL para todo el trabajo de datos (precios/indicadores/features/z).
from scripts.forward_testing.ft_env import configurar_entorno_local
configurar_entorno_local()

from sqlalchemy import text
from scripts.push_senales_bot import _engine_local, _engine_railway


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ── Helpers DB ────────────────────────────────────────────────────────────────

def _ya_existe(eng, ticker) -> dict:
    """Devuelve {existe, activo} del ticker en `activos`."""
    with eng.connect() as c:
        r = c.execute(text("SELECT activo FROM activos WHERE ticker=:t"),
                      {"t": ticker}).fetchone()
    if r is None:
        return {"existe": False, "activo": None}
    return {"existe": True, "activo": bool(r[0])}


def _upsert_activos(eng, ticker, nombre, sector, industry):
    """UPSERT del ticker en `activos` (sirve para local y Railway)."""
    with eng.begin() as c:
        c.execute(text("""
            INSERT INTO activos (ticker, nombre, sector, industry, activo, modelo_asignado)
            VALUES (:t, :n, :s, :i, TRUE, 'global_rf')
            ON CONFLICT (ticker) DO UPDATE SET
                nombre = EXCLUDED.nombre,
                sector = EXCLUDED.sector,
                industry = COALESCE(EXCLUDED.industry, activos.industry),
                activo = TRUE,
                modelo_asignado = COALESCE(activos.modelo_asignado, EXCLUDED.modelo_asignado)
        """), {"t": ticker, "n": nombre, "s": sector, "i": industry})


def _set_activo(eng, ticker, valor: bool):
    with eng.begin() as c:
        c.execute(text("UPDATE activos SET activo=:v WHERE ticker=:t"),
                  {"v": valor, "t": ticker})


def _log_cambio(eng, ticker, accion, sector, motivo, detalle: dict):
    with eng.begin() as c:
        c.execute(text("""
            INSERT INTO universo_cambios (ticker, accion, fecha, sector, motivo, detalle)
            VALUES (:t, :a, :f, :s, :m, CAST(:d AS JSONB))
        """), {"t": ticker, "a": accion, "f": date.today(), "s": sector,
               "m": motivo, "d": json.dumps(detalle)})


# ── Descarga OHLCV ────────────────────────────────────────────────────────────

def _descargar_ohlcv(ticker, anios):
    """2 anios de OHLCV via yahooquery -> df shape precios_diarios. None si falla."""
    import pandas as pd
    from src.utils import yahooquery_loader as yq
    start = date.today() - timedelta(days=365 * anios + 10)
    res = yq.download_batch([ticker], start=start, end=date.today())
    df_yf = res.get(ticker)
    if df_yf is None or df_yf.empty:
        return None
    df = pd.DataFrame({
        "ticker": ticker,
        "fecha":  pd.to_datetime(df_yf.index).normalize(),
        "open":   df_yf["Open"].astype(float),
        "high":   df_yf["High"].astype(float),
        "low":    df_yf["Low"].astype(float),
        "close":  df_yf["Close"].astype(float),
        "adj_close": (df_yf["Adj Close"] if "Adj Close" in df_yf.columns
                      else df_yf["Close"]).astype(float),
        "volume": df_yf["Volume"].fillna(0).astype("int64"),
    })
    return df.dropna(subset=["close"]).reset_index(drop=True)


def _nombre_yahoo(ticker) -> str:
    """longName de yahooquery, fallback al ticker."""
    try:
        from yahooquery import Ticker
        p = Ticker(ticker).price
        d = p.get(ticker) if isinstance(p, dict) else None
        if isinstance(d, dict):
            return d.get("longName") or d.get("shortName") or ticker
    except Exception:
        pass
    return ticker


def _run(script_rel, extra_args):
    """Subprocess a un script del repo con el python del venv. No fatal."""
    cmd = [sys.executable, os.path.join(ROOT, script_rel)] + extra_args
    log(f"    -> {os.path.basename(script_rel)} {' '.join(extra_args)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"    [WARN] {os.path.basename(script_rel)} fallo (rc={r.returncode}); "
            f"se puede re-correr manual. Cola: {r.stdout.strip()[-200:]}")
    return r.returncode == 0


# ── Comando ADD ───────────────────────────────────────────────────────────────

def cmd_add(args):
    ticker = args.ticker.strip().upper()
    sep = "=" * 64
    log(sep)
    log(f"ALTA de {ticker}  {'[DRY RUN]' if args.dry_run else ''}")
    log(sep)

    eng_l = _engine_local()
    estado = _ya_existe(eng_l, ticker)
    if estado["existe"] and estado["activo"]:
        log(f"[ABORT] {ticker} ya esta ACTIVO en el universo.")
        return
    if estado["existe"] and not estado["activo"]:
        log(f"[INFO] {ticker} existe pero estaba dado de baja -> se reactiva.")

    # 1. OHLCV
    log(f"1) Descargando {args.anios}a de OHLCV (yahooquery)...")
    df = _descargar_ohlcv(ticker, args.anios)
    if df is None or len(df) < 60:
        log(f"[ABORT] OHLCV insuficiente para {ticker} "
            f"({0 if df is None else len(df)} barras, min 60). Reintentar.")
        return
    n_barras = len(df)
    f_ini, f_fin = df["fecha"].min().date(), df["fecha"].max().date()
    log(f"   {n_barras} barras | {f_ini} -> {f_fin}")
    if n_barras < 250:
        log(f"   [WARN] {n_barras} barras (<250): SMA200 y features quedaran parciales "
            f"hasta acumular mas historia.")

    nombre = args.nombre or _nombre_yahoo(ticker)
    detalle = {"anios": args.anios, "n_barras": n_barras,
               "rango": [str(f_ini), str(f_fin)], "industry": args.industry,
               "nombre": nombre}

    if args.dry_run:
        log(f"[DRY RUN] Haria: activos UPSERT (local+Railway) sector={args.sector} "
            f"industry={args.industry} nombre='{nombre}'; precios {n_barras} barras; "
            f"indicadores; features; z-scores; fundamentales; log universo_cambios.")
        return

    # 2. activos DUAL-WRITE (local + Railway)
    log("2) UPSERT en activos (local + Railway)...")
    _upsert_activos(eng_l, ticker, nombre, args.sector, args.industry)
    try:
        _upsert_activos(_engine_railway(), ticker, nombre, args.sector, args.industry)
        log("   activos OK en local y Railway.")
    except Exception as e:
        log(f"   [WARN] no se pudo escribir activos en Railway ({e}). "
            f"El snapshot no vera el ticker hasta sincronizar. Re-correr.")

    # 3. precios + indicadores (LOCAL)
    log("3) Persistiendo precios + indicadores...")
    from src.data.database import upsert_precios
    from src.indicators.technical import procesar_indicadores_ticker
    upsert_precios(df)
    procesar_indicadores_ticker(ticker, df, guardar_db=True)

    # 4. features (bulk; recomputa todos -- idempotente)
    if not args.no_features:
        log("4) Calculando features (precio_accion + market_structure, bulk)...")
        from src.indicators.precio_accion import procesar_features_precio_accion
        from src.indicators.market_structure import procesar_features_market_structure
        procesar_features_precio_accion()
        procesar_features_market_structure()
    else:
        log("4) [SKIP features]")

    # 5. z-scores de acciones (desde el inicio de la historia del ticker)
    log("5) Backfill z-scores acciones...")
    from src.utils.zscore_pipeline import backfill_zscore_tickers
    n_z = backfill_zscore_tickers(desde=f_ini)
    log(f"   ticker_zscore_diario: {n_z} filas (desde {f_ini})")

    # 6. fundamentales (balances -> ratios -> pais -> comparativa sectorial)
    if not args.no_fundamentales:
        log("6) Analisis fundamental...")
        _run("scripts/refresh_fundamentales.py", ["--tickers", ticker])
        _run("scripts/compute_fundamentales_ratios.py", ["--tickers", ticker])
        _run("scripts/refresh_ticker_pais.py", ["--tickers", ticker])
        _run("scripts/compute_fundamentales_sector.py", [])
        # Multiplos al cierre (*_px) + comparativo de valuacion (DB->local, sin Yahoo)
        try:
            from scripts.compute_multiplos_px import compute_multiplos_px
            from scripts.compute_fundamentales_sector import compute_sector_valuacion_px
            n_px = compute_multiplos_px(eng_l)
            n_sx = compute_sector_valuacion_px(eng_l)
            log(f"   multiplos *_px: {n_px} | comparativo valuacion *_px: {n_sx}")
        except Exception as e:
            log(f"   [WARN] multiplos_px fallo ({e}); se llena en el proximo recovery.")
    else:
        log("6) [SKIP fundamentales]")

    # 6b. historia de balances (fecha de anuncio por Q, para la vista de
    # reaccion a balances). 1 llamada a Alpha Vantage. Si la cuota free ya se
    # agoto hoy, el batch reanudable (refresh_earnings_historico --backfill) lo
    # levanta despues: aca no bloquea el alta.
    log("6b) Historia de balances (Alpha Vantage)...")
    _run("scripts/refresh_earnings_historico.py", ["--ticker", ticker])

    # 7. log
    _log_cambio(eng_l, ticker, "ALTA", args.sector, args.motivo, detalle)
    try:
        _log_cambio(_engine_railway(), ticker, "ALTA", args.sector, args.motivo, detalle)
    except Exception:
        pass

    log(sep)
    log(f"ALTA COMPLETA: {ticker} ({args.sector}) incorporado al universo.")
    log("   Ya es parte del analisis sectorial y fundamental. Opciones: el PCR/muros")
    log("   salen dia 1 del proximo snapshot; el z-score de opciones acumula ~60d.")
    log(sep)


# ── Comando REMOVE ────────────────────────────────────────────────────────────

def cmd_remove(args):
    ticker = args.ticker.strip().upper()
    sep = "=" * 64
    log(sep)
    log(f"BAJA de {ticker}  {'[DRY RUN]' if args.dry_run else ''}")
    log(sep)

    eng_l = _engine_local()
    estado = _ya_existe(eng_l, ticker)
    if not estado["existe"]:
        log(f"[ABORT] {ticker} no existe en activos.")
        return
    if not estado["activo"]:
        log(f"[INFO] {ticker} ya estaba dado de baja (activo=FALSE).")

    # Guard CRITICO: posiciones FT abiertas (LOCAL) -- lo que importa
    with eng_l.connect() as c:
        ft_abiertas = c.execute(text("""
            SELECT estrategia_id, fecha_entrada, cantidad
            FROM ft_operaciones WHERE ticker=:t AND fecha_salida IS NULL
        """), {"t": ticker}).fetchall()
    if ft_abiertas:
        log(f"[GUARD] {ticker} tiene {len(ft_abiertas)} posicion(es) FT ABIERTAS "
            f"(ft_operaciones). Estrategias: {sorted(set(r[0] for r in ft_abiertas))}.")
        if not args.force:
            log("        ABORTA. Cerralas primero (o usa --force para quitar igual; "
                "quedaran huerfanas sin senales de salida).")
            return
        log("        --force: se continua; las posiciones FT quedaran huerfanas.")

    # Info Alpaca paper (no bloquea)
    try:
        er = _engine_railway()
        with er.connect() as c:
            for t in ["posiciones_bot", "posiciones_bot_tech", "posiciones_bot_candle"]:
                n = c.execute(text(f"SELECT COUNT(*) FROM {t} WHERE ticker=:t AND estado='ABIERTA'"),
                              {"t": ticker}).scalar()
                if n:
                    log(f"   [INFO] {n} posicion(es) paper ABIERTA en {t} (Alpaca, no critico).")
    except Exception:
        pass

    if args.dry_run:
        log(f"[DRY RUN] Haria: activos.activo=FALSE (local+Railway) + log BAJA. "
            f"Historia conservada.")
        return

    # SOFT delete dual-write
    _set_activo(eng_l, ticker, False)
    try:
        _set_activo(_engine_railway(), ticker, False)
        log("activos.activo=FALSE OK en local y Railway.")
    except Exception as e:
        log(f"[WARN] no se pudo actualizar activos en Railway ({e}). Re-correr.")

    detalle = {"ft_abiertas": len(ft_abiertas), "forzado": bool(args.force)}
    _log_cambio(eng_l, ticker, "BAJA", estado.get("sector"), args.motivo, detalle)
    try:
        _log_cambio(_engine_railway(), ticker, "BAJA", None, args.motivo, detalle)
    except Exception:
        pass

    log(sep)
    log(f"BAJA COMPLETA (soft): {ticker} deja de procesarse; historia conservada.")
    log(sep)


# ── Comando LIST ──────────────────────────────────────────────────────────────

def cmd_list(args):
    eng = _engine_local()
    with eng.connect() as c:
        n_act = c.execute(text("SELECT COUNT(*) FROM activos WHERE activo=TRUE")).scalar()
        n_baja = c.execute(text("SELECT COUNT(*) FROM activos WHERE activo=FALSE")).scalar()
        ultimos = c.execute(text("""
            SELECT ticker, accion, fecha, motivo FROM universo_cambios
            ORDER BY id DESC LIMIT 10
        """)).fetchall()
    print(f"\nUniverso: {n_act} activos | {n_baja} dados de baja\n")
    print("Ultimos cambios:")
    if not ultimos:
        print("  (sin registros en universo_cambios)")
    for r in ultimos:
        print(f"  {r[2]}  {r[1]:5}  {r[0]:7}  {r[3] or ''}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Alta/baja de tickers del universo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Incorporar un ticker (accion)")
    a.add_argument("ticker")
    a.add_argument("--sector", required=True, help="Sector (ej: Technology)")
    a.add_argument("--industry", default=None, help="Industria (opcional)")
    a.add_argument("--nombre", default=None, help="Nombre empresa (default: yahooquery)")
    a.add_argument("--anios", type=int, default=2, help="Anios de OHLCV a backfillear (default 2)")
    a.add_argument("--motivo", default=None)
    a.add_argument("--no-features", action="store_true", help="Saltar features (bulk)")
    a.add_argument("--no-fundamentales", action="store_true", help="Saltar analisis fundamental")
    a.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("remove", help="Dar de baja un ticker (soft delete)")
    r.add_argument("ticker")
    r.add_argument("--motivo", default=None)
    r.add_argument("--force", action="store_true", help="Quitar aun con posiciones FT abiertas")
    r.add_argument("--dry-run", action="store_true")

    sub.add_parser("list", help="Mostrar universo + ultimos cambios")

    sub.add_parser("cache", help="Refrescar el cache en disco del universo "
                                 "(red de seguridad si la DB se cae)")

    args = ap.parse_args()
    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "remove":
        cmd_remove(args)
    elif args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "cache":
        cmd_cache(args)


def cmd_cache(args):
    """
    Reescribe data/cache/universo.json desde `activos`.

    El cache se refresca solo en cada lectura exitosa, asi que normalmente no
    hace falta. Sirve para SEMBRARLO en una maquina que todavia no lo tiene
    (ej. Oracle recien clonado, o Oracle con su DB caida: ahi no puede
    construirlo solo y sin cache caeria a la lista hardcodeada).
    """
    from src.data.universo import refrescar_cache, CACHE_PATH
    data = refrescar_cache()
    print(f"  Cache refrescado: {data['n']} tickers")
    print(f"  Archivo         : {CACHE_PATH}")
    print(f"  Actualizado     : {data['actualizado']}")


if __name__ == "__main__":
    main()
