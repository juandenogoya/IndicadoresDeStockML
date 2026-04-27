"""
cron_diario.py
Script orquestador para GitHub Actions (cron L-V a las 13:00 UTC).

Soporta ejecucion por pasos individuales via --step para el workflow
particionado en jobs GH Actions encadenados con `needs`.

Modos de uso:
    python scripts/cron_diario.py --step precios   # Paso 1: precios + futuros
    python scripts/cron_diario.py --step features  # Paso 2: features PA + SMC
    python scripts/cron_diario.py --step scanner   # Paso 3: scanner ML + Telegram
    python scripts/cron_diario.py --step verificar # Paso 4: verificacion post-facto
    python scripts/cron_diario.py                  # Legacy: todos los pasos en uno

Pasos y tiempos estimados (199 tickers):
    Paso 1 precios  : ~35 min  (yfinance batch + indicadores secuencial)
    Paso 2 features : ~4 min   (features PA + market structure bulk)
    Paso 3 scanner  : ~60 min  (ML inference secuencial 199 tickers)
    Paso 4 verificar: ~2 min   (retornos post-facto)

Notas:
    Backtesting PA removido del cron (23/4/2026) — ver scripts/34_bt_pa_manual.py
    Push a main cada <60 dias para mantener GH Actions scheduled activo.
"""

import sys
import os
import argparse
import traceback
import psycopg2.extras
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def log(msg: str):
    """Print con timestamp para Railway logs."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _native(v):
    import numpy as np
    if isinstance(v, np.integer):  return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.bool_):    return bool(v)
    return v


def _persistir_alertas(resultados: list):
    """
    Upsert de resultados en alertas_scanner.
    ON CONFLICT (ticker, date(scan_fecha)) -> actualiza todos los campos.
    Idempotente: correr varias veces el mismo dia sobreescribe el scan anterior.
    No depende del SERIAL id para detectar duplicados.
    """
    from src.data.database import get_connection

    campos = (
        "scan_fecha", "ticker", "sector", "persistido_en_db",
        "precio_cierre", "precio_fecha", "atr14",
        "ml_prob_ganancia", "ml_modelo_usado",
        "pa_ev1", "pa_ev2", "pa_ev3", "pa_ev4",
        "bear_bos10", "bear_choch10", "bear_estructura",
        "score_ponderado", "condiciones_ok",
        "estructura_10", "dist_sl_10_pct", "dist_sh_10_pct",
        "dias_sl_10", "dias_sh_10",
        "alert_score", "alert_nivel", "alert_detalle",
    )

    records = [
        {k: _native(r.get(k)) for k in campos}
        for r in resultados if not r.get("error")
    ]
    if not records:
        return

    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in campos
        if c not in ("scan_fecha", "ticker")
    )

    sql = f"""
        INSERT INTO alertas_scanner ({', '.join(campos)})
        VALUES ({', '.join(f'%({c})s' for c in campos)})
        ON CONFLICT (ticker, date(scan_fecha))
        DO UPDATE SET {update_set}
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=100)

    log(f"  DB: {len(records)} alertas guardadas (upsert por ticker/dia).")


def paso_actualizar_datos() -> dict:
    """
    Paso 0: Actualiza precios e indicadores tecnicos para TODOS los tickers en DB.

    Descarga en una sola llamada batch (yf.download con lista de tickers) para
    evitar rate-limiting en GitHub Actions. Luego procesa cada ticker con los
    datos recibidos y recalcula indicadores desde el historico completo en DB.
    """
    import yfinance as yf
    import pandas as pd
    from src.pipeline.data_manager import cargar_precios_db
    from src.data.database import upsert_precios, query_df
    from src.indicators.technical import procesar_indicadores_ticker

    # Obtener TODOS los tickers activos en DB
    try:
        df_activos = query_df(
            "SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker"
        )
        tickers_db = df_activos["ticker"].tolist() if not df_activos.empty else []
    except Exception as e:
        log(f"  [WARN] No se pudo leer activos de DB: {e}")
        from src.utils.config import ALL_TICKERS
        tickers_db = ALL_TICKERS

    if not tickers_db:
        log("  [WARN] Lista de tickers vacia, saltando actualizacion.")
        return {"ok": 0, "error": 0}

    # Descarga en lotes para evitar YFRateLimitError con 200 tickers en una sola llamada.
    # 50 tickers/lote, 8s entre lotes, reintento de 90s si hay rate limit.
    import time

    BATCH_SIZE   = 50
    BATCH_DELAY  = 8    # segundos entre lotes (evita rate limit proactivo)
    RETRY_DELAY  = 90   # segundos de espera si el lote falla con rate limit

    batches = [tickers_db[i:i + BATCH_SIZE]
               for i in range(0, len(tickers_db), BATCH_SIZE)]
    log(f"  {len(tickers_db)} tickers | {len(batches)} lotes de hasta {BATCH_SIZE}...")

    # ticker -> DataFrame raw (sin procesar) del lote
    ticker_raw = {}

    for b_idx, batch in enumerate(batches, 1):
        log(f"  Lote {b_idx}/{len(batches)} ({len(batch)} tickers)...")
        raw = None

        for attempt in (1, 2):
            try:
                raw = yf.download(
                    tickers=batch,
                    period="1mo",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
                break   # descarga OK
            except Exception as e:
                err_s = str(e)
                if attempt == 1 and "RateLimit" in err_s:
                    log(f"    [WARN] Rate limit en lote {b_idx}, reintentando en {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    log(f"    [ERROR] Lote {b_idx} fallo: {err_s[:80]}")
                    raw = None
                    break

        if raw is None or raw.empty:
            log(f"    [WARN] Lote {b_idx}: sin datos.")
        else:
            # Extraer DataFrame por ticker del lote
            for ticker in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        lvl0 = raw.columns.get_level_values(0).unique().tolist()
                        lvl1 = raw.columns.get_level_values(1).unique().tolist()
                        if ticker in lvl0:
                            ticker_raw[ticker] = raw[ticker].copy()
                        elif ticker in lvl1:
                            ticker_raw[ticker] = raw.xs(ticker, axis=1, level=1).copy()
                    else:
                        # Lote de un solo ticker (sin MultiIndex)
                        ticker_raw[ticker] = raw.copy()
                except Exception:
                    pass
            log(f"    Lote {b_idx} OK: {raw.shape[0]} filas.")

        if b_idx < len(batches):
            time.sleep(BATCH_DELAY)

    log(f"  Descarga completada: {len(ticker_raw)}/{len(tickers_db)} tickers con datos.")

    # Procesar cada ticker: upsert precios + recalculo indicadores
    n_ok = 0
    n_err = 0

    for i, ticker in enumerate(tickers_db, 1):
        try:
            if ticker not in ticker_raw:
                raise ValueError("sin datos en batch")

            df_t = ticker_raw[ticker].reset_index()
            df_t.columns = [
                c[0].lower() if isinstance(c, tuple) else str(c).lower()
                for c in df_t.columns
            ]
            df_t = df_t.rename(columns={"date": "fecha", "price": "fecha"})
            if "fecha" not in df_t.columns:
                raise ValueError("columna fecha no encontrada en batch")

            df_t["fecha"] = pd.to_datetime(df_t["fecha"])
            cols_needed = [c for c in ["fecha", "open", "high", "low", "close", "volume"]
                           if c in df_t.columns]
            df_t = df_t[cols_needed].dropna(subset=["close"])
            df_t = df_t[df_t["close"] > 0]

            # Excluir datos del dia actual: el mercado puede no haber cerrado aun.
            # Solo se almacenan cierres confirmados (dias anteriores a hoy).
            from datetime import date as _date
            df_t = df_t[df_t["fecha"].dt.date < _date.today()]

            df_t["ticker"] = ticker
            df_t["adj_close"] = df_t["close"]
            df_t = df_t.sort_values("fecha").reset_index(drop=True)

            if df_t.empty:
                raise ValueError("sin datos validos en resultado batch")

            upsert_precios(df_t)

            # Cargar historico completo desde DB para SMA200
            df_full = cargar_precios_db(ticker, ultimas_n=500)
            if len(df_full) < 250:
                raise ValueError(f"historico insuficiente ({len(df_full)} barras)")

            # Recalcular indicadores
            procesar_indicadores_ticker(ticker, df_full, guardar_db=True)
            ultima = df_t["fecha"].max()
            log(f"    [{i:03d}/{len(tickers_db)}] {ticker}: OK (ultima: {ultima.date()})")
            n_ok += 1

        except Exception as e:
            log(f"    [{i:03d}/{len(tickers_db)}] {ticker}: ERROR - {str(e)[:80]}")
            n_err += 1

    return {"ok": n_ok, "error": n_err}


def paso_actualizar_features_db() -> dict:
    """
    Paso 1: Upsert de features_precio_accion y features_market_structure.

    Lee todos los tickers de precios_diarios (no filtrado por ALL_TICKERS)
    y hace upsert de las features calculadas. Necesario para que
    paso_backtesting_pa() use datos del dia en FIN_SEGMENTO.

    Prerequisito: paso_actualizar_datos() debe haber corrido primero.
    """
    from src.indicators.precio_accion import procesar_features_precio_accion
    from src.indicators.market_structure import procesar_features_market_structure

    log("  Calculando features_precio_accion (upsert)...")
    df_pa = procesar_features_precio_accion()
    log(f"  features_precio_accion OK: {len(df_pa):,} filas")

    log("  Calculando features_market_structure (upsert)...")
    df_ms = procesar_features_market_structure()
    log(f"  features_market_structure OK: {len(df_ms):,} filas")

    return {"pa": len(df_pa), "ms": len(df_ms)}


def _paso_backtesting_pa_REMOVIDO() -> int:
    """
    REMOVIDO DEL CRON DIARIO el 23/4/2026.
    Ejecutar manualmente con: python scripts/34_bt_pa_manual.py

    Paso 3: Actualiza el backtesting PA de forma incremental (solo el dia actual).

    Procesa la ultima barra disponible para cada ticker:
    - Cierra posiciones FIN_SEGMENTO que cumplen condicion de salida hoy.
    - Actualiza precio/dias/retorno de posiciones que siguen abiertas.
    - Abre nuevas posiciones para combos con senal de entrada hoy.

    Mucho mas rapido que el full rerun: O(posiciones_abiertas) en lugar de
    O(tickers x barras_historicas x 16).

    Prerequisito: paso_actualizar_features_db() debe haber corrido primero.
    Prerequisito: scripts/21_migrar_bt_incremental.py ejecutado (columnas stop_loss/take_profit).
    """
    from src.backtesting.simulator_pa import ejecutar_backtesting_pa_incremental
    from src.backtesting.metrics_pa import calcular_y_guardar_resultados_pa
    from src.data.database import query_df

    stats = ejecutar_backtesting_pa_incremental()
    log(f"  BT Incremental: {stats['cerradas']} cerradas, "
        f"{stats['actualizadas']} actualizadas, "
        f"{stats['nuevas']} nuevas.")

    # Recalcular metricas con todas las ops (incluye FIN_SEGMENTO con precio actual)
    df_ops = query_df("SELECT * FROM operaciones_bt_pa ORDER BY fecha_entrada")
    if not df_ops.empty:
        calcular_y_guardar_resultados_pa(df_ops)
        log("  Resultados PA actualizados.")

    return stats["cerradas"] + stats["nuevas"]


def paso_scanner() -> list:
    """
    Corre el scanner para todos los tickers del universo.
    Retorna la lista de resultados.
    """
    from src.utils.config import ALL_TICKERS
    from src.pipeline.data_manager import preparar_ticker
    from src.pipeline.feature_calculator import calcular_features_completas
    from src.pipeline.signal_engine import cargar_modelos_v3, evaluar_ticker
    from src.pipeline.alert_classifier import clasificar_alerta
    from src.pipeline.telegram_notifier import enviar_resumen

    modelos    = cargar_modelos_v3()
    scan_fecha = datetime.now()
    resultados = []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        log(f"  [{i:02d}/{len(ALL_TICKERS)}] {ticker}...", )
        resultado_base = {
            "scan_fecha":        scan_fecha,
            "ticker":            ticker,
            "sector":            None,
            "persistido_en_db":  False,
        }
        try:
            df_ohlcv, sector, es_nuevo = preparar_ticker(ticker, persistir=False)
            resultado_base["sector"] = sector

            calc = calcular_features_completas(df_ohlcv, ticker, sector)
            if not calc["ok"]:
                raise ValueError(calc["error"])

            signals = evaluar_ticker(calc["features_v3"], calc["features_pa"],
                                     sector, modelos, ticker=ticker)
            alert_score, alert_nivel, alert_detalle = clasificar_alerta(
                signals, calc["meta"]
            )
            meta = calc["meta"]
            fp   = calc["features_pa"]

            r = {
                **resultado_base,
                "precio_cierre":     meta.get("precio_cierre"),
                "precio_fecha":      meta.get("precio_fecha"),
                "atr14":             meta.get("atr14"),
                "ml_prob_ganancia":  signals["ml_prob_ganancia"],
                "ml_modelo_usado":   signals["ml_modelo_usado"],
                "pa_ev1":            signals["pa_ev1"],
                "pa_ev2":            signals["pa_ev2"],
                "pa_ev3":            signals["pa_ev3"],
                "pa_ev4":            signals["pa_ev4"],
                "bear_bos10":        signals["bear_bos10"],
                "bear_choch10":      signals["bear_choch10"],
                "bear_estructura":   signals["bear_estructura"],
                "score_ponderado":   meta.get("score_ponderado"),
                "condiciones_ok":    meta.get("condiciones_ok"),
                "estructura_10":     fp.get("estructura_10"),
                "dist_sl_10_pct":    fp.get("dist_sl_10_pct"),
                "dist_sh_10_pct":    fp.get("dist_sh_10_pct"),
                "dias_sl_10":        fp.get("dias_sl_10"),
                "dias_sh_10":        fp.get("dias_sh_10"),
                # Patrones de vela (para Mensaje 2 Telegram)
                "patron_hammer":         fp.get("patron_hammer"),
                "patron_shooting_star":  fp.get("patron_shooting_star"),
                "patron_engulfing_bull": fp.get("patron_engulfing_bull"),
                "patron_engulfing_bear": fp.get("patron_engulfing_bear"),
                "es_alcista":            fp.get("es_alcista"),
                "vol_spike":             fp.get("vol_spike"),
                "alert_score":       alert_score,
                "alert_nivel":       alert_nivel,
                "alert_detalle":     alert_detalle,
            }
            log(f"    -> {alert_nivel} (score={alert_score:.0f} ml={signals['ml_prob_ganancia']:.0%})")

        except Exception as e:
            log(f"    ERROR: {str(e)[:80]}")
            r = {**resultado_base, "error": str(e)}

        resultados.append(r)

    # Contexto MTF (batch, una sola query)
    from src.indicators.mtf_context import get_contexto_mtf_batch
    tickers_ok = [r["ticker"] for r in resultados if not r.get("error")]
    if tickers_ok:
        ctx_mtf = get_contexto_mtf_batch(tickers_ok)
        for r in resultados:
            ctx = ctx_mtf.get(r["ticker"], {})
            r["tendencia_1w"] = ctx.get("tendencia_1w", "neutral")
            r["tendencia_1m"] = ctx.get("tendencia_1m", "neutral")

    # Persistir + Telegram
    _persistir_alertas(resultados)
    enviar_resumen(resultados)
    return resultados


def paso_futuros() -> int:
    """
    Paso 0b: Actualiza los ultimos dias de futuros (12 instrumentos: indices,
    metales, bonos, commodities agricolas, sectoriales).
    No critico: si falla el pipeline continua sin contexto macro.
    """
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "futuros_indices",
        pathlib.Path(__file__).parent / "34_futuros_indices.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cmd_update()


def paso_indicadores_futuros() -> int:
    """
    Paso 0c: Recalcula indicadores tecnicos sobre futuros (ultimos 10 dias).
    Depende de futuros actualizados (paso 0b). No critico.
    """
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "indicadores_futuros",
        pathlib.Path(__file__).parent / "35_calcular_indicadores_futuros.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cmd_update()


def paso_regimen_macro() -> int:
    """
    Paso 0d: Recalcula features de regimen macro (ultimos 15 dias).
    Depende de indicadores futuros actualizados (paso 0c). No critico.
    """
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "features_regimen_macro",
        pathlib.Path(__file__).parent / "36_features_regimen_macro.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cmd_update()


def paso_verificacion() -> int:
    """
    Rellena retornos post-facto para alertas pendientes.
    Retorna el numero de registros actualizados.
    """
    # Importar la funcion core de verificacion directamente
    import importlib, importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "verificar_alertas",
        pathlib.Path(__file__).parent / "18_verificar_alertas.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verificar_alertas()


def watchdog_datos():
    """
    Verifica que los precios en DB no tengan mas de 3 dias calendario de atraso.
    Si estan desactualizados envia alerta Telegram ANTES de correr el resto.

    Esto detecta el caso donde GitHub Actions estuvo pausado o fallando
    en silencio: el primer dia que vuelve a correr, avisa inmediatamente.
    """
    from src.data.database import query_df
    try:
        df = query_df("SELECT MAX(fecha) AS ultima FROM precios_diarios")
        if df.empty or df.iloc[0]["ultima"] is None:
            return
        import pandas as pd
        from datetime import date
        ultima = pd.to_datetime(df.iloc[0]["ultima"]).date()
        hoy    = date.today()
        dias   = (hoy - ultima).days
        if dias > 3:
            msg = (
                f"WATCHDOG: datos desactualizados\n"
                f"Ultima fecha en precios_diarios: {ultima}\n"
                f"Hoy: {hoy} ({dias} dias de atraso)\n"
                f"Posible causa: cron pausado o GitHub Actions inactivo."
            )
            log(f"  [WARN] {msg.replace(chr(10), ' | ')}")
            try:
                from src.pipeline.telegram_notifier import _send
                _send(f"<b>ALERTA WATCHDOG</b>\n<code>{msg}</code>")
            except Exception as e:
                log(f"  [WARN] Telegram watchdog no enviado: {e}")
        else:
            log(f"  Datos OK: ultima fecha {ultima} ({dias} dias)")
    except Exception as e:
        log(f"  [WARN] Watchdog fallido (no critico): {e}")


def _header(titulo: str):
    log("=" * 55)
    log(f"  {titulo}")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log("=" * 55)


def cmd_precios():
    """
    Paso 1: Precios + Futuros + Indicadores macro.
    Critico: exit 1 si paso_actualizar_datos() falla (bloqueante para features y scanner).
    Futuros e indicadores adicionales son no criticos.
    """
    _header("CRON  Paso 1 -- Precios + Futuros")

    log("\n[PRE] Watchdog de datos frescos...")
    watchdog_datos()

    log("\n[1a] Actualizacion diaria de precios e indicadores tecnicos...")
    try:
        stats = paso_actualizar_datos()
        log(f"  Precios OK: {stats['ok']} tickers, {stats['error']} errores.")
    except Exception:
        log(f"  ERROR CRITICO en precios:\n{traceback.format_exc()}")
        sys.exit(1)

    log("\n[1b] Futuros de indices...")
    try:
        paso_futuros()
    except Exception:
        log(f"  ERROR en futuros (no critico): {traceback.format_exc()[:200]}")

    log("\n[1c] Indicadores tecnicos sobre futuros...")
    try:
        paso_indicadores_futuros()
    except Exception:
        log(f"  ERROR en indicadores futuros (no critico): {traceback.format_exc()[:200]}")

    log("\n[1d] Features de regimen macro...")
    try:
        paso_regimen_macro()
    except Exception:
        log(f"  ERROR en regimen macro (no critico): {traceback.format_exc()[:200]}")

    log("\n  Paso 1 finalizado.")
    log("=" * 55)


def cmd_features():
    """
    Paso 2: Features precio accion + Market Structure.
    Critico: exit 1 si falla (scanner depende de estos datos).
    """
    _header("CRON  Paso 2 -- Features PA + Market Structure")

    log("\n[2] Upsert features_precio_accion y features_market_structure...")
    try:
        stats = paso_actualizar_features_db()
        log(f"  Features OK: PA={stats['pa']:,} filas, MS={stats['ms']:,} filas.")
    except Exception:
        log(f"  ERROR CRITICO en features:\n{traceback.format_exc()}")
        sys.exit(1)

    log("\n  Paso 2 finalizado.")
    log("=" * 55)


def cmd_scanner():
    """
    Paso 3: Scanner ML + Telegram resumen.
    Critico: exit 1 si falla.
    """
    _header("CRON  Paso 3 -- Scanner ML")

    log("\n[3] Scanner de alertas ML (199 tickers)...")
    try:
        resultados = paso_scanner()
        n_ok  = sum(1 for r in resultados if not r.get("error"))
        n_err = len(resultados) - n_ok
        log(f"  Scanner OK: {n_ok} tickers, {n_err} errores.")
    except Exception:
        log(f"  ERROR CRITICO en scanner:\n{traceback.format_exc()}")
        sys.exit(1)

    log("\n  Paso 3 finalizado.")
    log("=" * 55)


def cmd_verificar():
    """
    Paso 4: Verificacion post-facto de alertas (retorno_1d/5d/20d_real).
    No critico: siempre exit 0.
    """
    _header("CRON  Paso 4 -- Verificacion post-facto")

    log("\n[4] Verificacion post-facto...")
    try:
        n = paso_verificacion()
        log(f"  Verificacion OK: {n} alertas actualizadas.")
    except Exception:
        log(f"  ERROR en verificacion (no critico): {traceback.format_exc()[:300]}")

    log("\n  Paso 4 finalizado.")
    log("=" * 55)


def cmd_all():
    """
    Modo legacy: todos los pasos en un solo proceso.
    Util para pruebas locales. En produccion se usa --step con jobs separados.
    """
    _header("CRON DIARIO  |  Todos los pasos (modo legacy)")

    log("\n[PRE] Watchdog de datos frescos...")
    watchdog_datos()

    errores = []

    log("\n[0/4] Actualizacion diaria de precios e indicadores...")
    try:
        stats = paso_actualizar_datos()
        log(f"  Actualizar OK: {stats['ok']} tickers, {stats['error']} errores.")
    except Exception:
        log(f"  ERROR en actualizacion (continua):\n{traceback.format_exc()[:300]}")

    log("\n[0b/4] Futuros de indices...")
    try:
        paso_futuros()
    except Exception:
        log(f"  ERROR en futuros (no critico):\n{traceback.format_exc()[:200]}")

    log("\n[1/4] Upsert features_precio_accion y features_market_structure...")
    try:
        stats_feat = paso_actualizar_features_db()
        log(f"  Features OK: PA={stats_feat['pa']:,}, MS={stats_feat['ms']:,} filas.")
    except Exception:
        log(f"  ERROR en features (continua):\n{traceback.format_exc()[:300]}")

    log("\n[2/4] Scanner de alertas...")
    try:
        resultados = paso_scanner()
        n_ok  = sum(1 for r in resultados if not r.get("error"))
        n_err = len(resultados) - n_ok
        log(f"  Scanner OK: {n_ok} tickers, {n_err} errores.")
    except Exception:
        msg = traceback.format_exc()
        log(f"  ERROR CRITICO en scanner:\n{msg}")
        errores.append(f"Scanner:\n{msg}")

    log("\n[3/4] Verificacion post-facto...")
    try:
        n = paso_verificacion()
        log(f"  Verificacion OK: {n} alertas actualizadas.")
    except Exception:
        msg = traceback.format_exc()
        log(f"  ERROR en verificacion:\n{msg}")
        errores.append(f"Verificacion:\n{msg}")

    log("\n" + "=" * 55)
    if errores:
        log(f"  FINALIZADO CON {len(errores)} ERROR(ES)")
        try:
            from src.pipeline.telegram_notifier import _send
            _send("<b>ERROR cron diario</b>\n<code>"
                  + "\n---\n".join(e[:300] for e in errores)
                  + "</code>")
        except Exception:
            pass
        sys.exit(1)

    log("  FINALIZADO SIN ERRORES")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Cron diario — pipeline de datos, features y scanner ML.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=["precios", "features", "scanner", "verificar"],
        default=None,
        metavar="PASO",
        help=(
            "Ejecutar un paso especifico del pipeline:\n"
            "  precios   -- Descarga precios + futuros (Paso 1, ~35 min)\n"
            "  features  -- Calcula features PA + SMC  (Paso 2, ~4 min)\n"
            "  scanner   -- Corre modelos ML + Telegram (Paso 3, ~60 min)\n"
            "  verificar -- Verifica retornos post-facto (Paso 4, ~2 min)\n"
            "Sin --step ejecuta todos los pasos en secuencia (modo legacy)."
        ),
    )
    args = parser.parse_args()

    dispatch = {
        "precios":  cmd_precios,
        "features": cmd_features,
        "scanner":  cmd_scanner,
        "verificar": cmd_verificar,
    }

    if args.step:
        dispatch[args.step]()
    else:
        cmd_all()


if __name__ == "__main__":
    main()
