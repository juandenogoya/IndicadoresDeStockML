"""
cron_diario.py
Script orquestador para Railway (cron L-V a las 17:30 ET / 20:30 UTC).

Pasos:
    0. Actualizar precios e indicadores tecnicos (delta ultimos 10 dias)
    1. Upsert features_precio_accion y features_market_structure
    2. Scanner de alertas para todos los tickers del universo
    3. Backtesting PA (TRUNCATE + re-run) -- mantiene FIN_SEGMENTO al dia
    4. Verificacion post-facto de alertas pendientes
    5. Notificacion Telegram con resumen + errores criticos

Diseñado para correr como proceso desechable (start + exit).
Logs disponibles en Railway Dashboard > Deployments > Logs.
"""

import sys
import os
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
    Inserta resultados en alertas_scanner.
    Antes del INSERT elimina los registros de hoy para los mismos tickers,
    garantizando que solo quede el ultimo scan del dia (idempotente).
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

    tickers_ok = [r["ticker"] for r in records]

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Eliminar registros del dia para evitar duplicados
            cur.execute(
                "DELETE FROM alertas_scanner "
                "WHERE date(scan_fecha) = CURRENT_DATE AND ticker = ANY(%s)",
                (tickers_ok,)
            )
            n_del = cur.rowcount

            # Insertar el scan actual (unico del dia)
            sql = f"""
                INSERT INTO alertas_scanner ({', '.join(campos)})
                VALUES ({', '.join(f'%({c})s' for c in campos)})
            """
            psycopg2.extras.execute_batch(cur, sql, records, page_size=100)

    log(f"  DB: {len(records)} alertas guardadas "
        f"({n_del} registros previos del dia eliminados).")


def paso_actualizar_datos() -> dict:
    """
    Paso 0: Actualiza precios e indicadores tecnicos para TODOS los tickers en DB.

    Flujo por ticker:
      1. Descarga ultimos 10 dias desde yfinance -> upsert en precios_diarios
         (yfinance en lugar de Stooq: volumenes correctos para datos recientes;
          Stooq devuelve volumenes identicos/placeholder para las ultimas fechas)
      2. Carga historico completo (500 barras) desde DB
      3. Recalcula indicadores sobre el historico completo -> upsert en indicadores_tecnicos

    Nota: necesitamos el historico completo porque calcular_indicadores() requiere
    al menos 200 barras para SMA200. No se puede usar solo el delta de 10 dias.
    """
    from datetime import date, timedelta
    from src.pipeline.data_manager import descargar_yfinance, cargar_precios_db
    from src.data.database import upsert_precios, query_df
    from src.indicators.technical import procesar_indicadores_ticker

    # Obtener TODOS los tickers activos en DB (incluye tickers externos como VZ)
    try:
        df_activos = query_df(
            "SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker"
        )
        tickers_db = df_activos["ticker"].tolist() if not df_activos.empty else []
    except Exception as e:
        log(f"  [WARN] No se pudo leer activos de DB: {e}")
        # Fallback a ALL_TICKERS de config
        from src.utils.config import ALL_TICKERS
        tickers_db = ALL_TICKERS

    if not tickers_db:
        log("  [WARN] Lista de tickers vacia, saltando actualizacion.")
        return {"ok": 0, "error": 0}

    log(f"  {len(tickers_db)} tickers (via yfinance, periodo 1mo)...")

    n_ok = 0
    n_err = 0

    for i, ticker in enumerate(tickers_db, 1):
        try:
            # 1. Descargar precios recientes con yfinance (volumenes correctos) y upsert
            df_new = descargar_yfinance(ticker, periodo="1mo")
            if df_new.empty:
                log(f"    [{i:02d}/{len(tickers_db)}] {ticker}: yfinance sin datos.")
                n_err += 1
                continue
            # upsert_precios requiere columna adj_close
            if "adj_close" not in df_new.columns:
                df_new["adj_close"] = df_new["close"]
            upsert_precios(df_new)

            # 2. Cargar historico completo desde DB para calculo valido de SMA200
            df_full = cargar_precios_db(ticker, ultimas_n=500)
            if len(df_full) < 250:
                log(f"    [{i:02d}/{len(tickers_db)}] {ticker}: historico insuficiente "
                    f"({len(df_full)} barras).")
                n_err += 1
                continue

            # 3. Calcular y guardar indicadores sobre el historico completo
            procesar_indicadores_ticker(ticker, df_full, guardar_db=True)
            ultima = df_new["fecha"].max()
            log(f"    [{i:02d}/{len(tickers_db)}] {ticker}: OK (ultima barra: {ultima})")
            n_ok += 1

        except Exception as e:
            log(f"    [{i:02d}/{len(tickers_db)}] {ticker}: ERROR - {str(e)[:80]}")
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


def paso_backtesting_pa() -> int:
    """
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
                "alert_score":       alert_score,
                "alert_nivel":       alert_nivel,
                "alert_detalle":     alert_detalle,
            }
            log(f"    -> {alert_nivel} (score={alert_score:.0f} ml={signals['ml_prob_ganancia']:.0%})")

        except Exception as e:
            log(f"    ERROR: {str(e)[:80]}")
            r = {**resultado_base, "error": str(e)}

        resultados.append(r)

    # Persistir + Telegram
    _persistir_alertas(resultados)
    enviar_resumen(resultados)
    return resultados


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


def main():
    log("=" * 55)
    log("  CRON DIARIO  |  Actualizar + Scanner + BT-PA + Verificacion")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log("=" * 55)

    errores = []

    # ── Paso 0: Actualizar precios e indicadores ───────────────
    log("\n[0/4] Actualizacion diaria de precios e indicadores...")
    try:
        stats = paso_actualizar_datos()
        log(f"  Actualizar OK: {stats['ok']} tickers actualizados, "
            f"{stats['error']} con error.")
    except Exception:
        msg = traceback.format_exc()
        log(f"  ERROR en actualizacion (no critico, continua):\n{msg[:300]}")
        # No se agrega a errores criticos — el scanner puede seguir con datos existentes

    # ── Paso 1: Features PA + Market Structure ─────────────────
    log("\n[1/4] Upsert features_precio_accion y features_market_structure...")
    try:
        stats_feat = paso_actualizar_features_db()
        log(f"  Features OK: PA={stats_feat['pa']:,} filas, MS={stats_feat['ms']:,} filas.")
    except Exception:
        msg = traceback.format_exc()
        log(f"  ERROR en features (no critico, continua):\n{msg[:300]}")
        # No critico: el scanner calcula features en memoria de todas formas

    # ── Paso 2: Scanner ───────────────────────────────────────
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

    # ── Paso 3: Backtesting PA ────────────────────────────────
    log("\n[3/4] Backtesting PA (TRUNCATE + re-run para estado_pa)...")
    try:
        n_ops = paso_backtesting_pa()
        log(f"  Backtesting PA OK: {n_ops:,} operaciones.")
    except Exception:
        msg = traceback.format_exc()
        log(f"  ERROR en backtesting PA (no critico):\n{msg[:300]}")
        # No critico: estado_pa quedara con datos del dia anterior

    # ── Paso 4: Verificacion post-facto ───────────────────────
    log("\n[4/4] Verificacion post-facto...")
    try:
        n = paso_verificacion()
        log(f"  Verificacion OK: {n} alertas actualizadas.")
    except Exception:
        msg = traceback.format_exc()
        log(f"  ERROR en verificacion:\n{msg}")
        errores.append(f"Verificacion:\n{msg}")

    # ── Resultado final ──────────────────────────────────────
    log("\n" + "=" * 55)
    if errores:
        log(f"  FINALIZADO CON {len(errores)} ERROR(ES)")
        try:
            from src.pipeline.telegram_notifier import _send
            _send("<b>ERROR cron Railway</b>\n<code>"
                  + "\n---\n".join(e[:300] for e in errores)
                  + "</code>")
        except Exception:
            pass
        sys.exit(1)

    log("  FINALIZADO SIN ERRORES")
    sys.exit(0)


if __name__ == "__main__":
    main()
