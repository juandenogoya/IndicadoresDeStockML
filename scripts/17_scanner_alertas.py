"""
17_scanner_alertas.py
Scanner de alertas ML para cualquier lista de tickers.

Uso:
    # Tickers de la DB con los modelos existentes
    python scripts/17_scanner_alertas.py JPM BAC MS GS

    # Con un ticker nuevo (sin persistir en DB)
    python scripts/17_scanner_alertas.py NVDA AAPL MSFT

    # Con un ticker nuevo y persistirlo en DB para futuras corridas
    python scripts/17_scanner_alertas.py NVDA --persistir

    # Escanear todos los tickers del universo original
    python scripts/17_scanner_alertas.py --todos

    # Sin enviar Telegram
    python scripts/17_scanner_alertas.py JPM BAC --no-telegram

Funcionamiento:
    1. Para cada ticker: obtiene datos (DB o yfinance)
    2. Calcula 53 features V3 + features PA en tiempo real
    3. Corre el modelo ML V3 apropiado (sectorial o global)
    4. Evalua condiciones EV1-EV4 alcistas y senales bajistas
    5. Calcula score compuesto 0-100 y nivel de alerta
    6. Guarda en alertas_scanner (DB)
    7. Envia resumen por Telegram

Output:
    Tabla resumen en consola + registro en alertas_scanner + mensaje Telegram
"""

import sys
import os
import argparse
import psycopg2.extras
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import ALL_TICKERS
from src.pipeline.data_manager import preparar_ticker
from src.pipeline.feature_calculator import calcular_features_completas
from src.pipeline.signal_engine import cargar_modelos_v3, evaluar_ticker
from src.pipeline.alert_classifier import clasificar_alerta
from src.pipeline.telegram_notifier import enviar_resumen
from src.data.database import get_connection


# ─────────────────────────────────────────────────────────────
# Persistencia en alertas_scanner
# ─────────────────────────────────────────────────────────────

def _native(v):
    """Convierte numpy types a Python nativos para psycopg2."""
    import numpy as np
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def persistir_alertas(resultados: list):
    """
    Inserta los resultados en la tabla alertas_scanner.
    """
    if not resultados:
        return

    sql = """
        INSERT INTO alertas_scanner (
            scan_fecha, ticker, sector, persistido_en_db,
            precio_cierre, precio_fecha, atr14,
            ml_prob_ganancia, ml_modelo_usado,
            pa_ev1, pa_ev2, pa_ev3, pa_ev4,
            bear_bos10, bear_choch10, bear_estructura,
            score_ponderado, condiciones_ok,
            estructura_10, dist_sl_10_pct, dist_sh_10_pct,
            dias_sl_10, dias_sh_10,
            alert_score, alert_nivel, alert_detalle
        ) VALUES (
            %(scan_fecha)s, %(ticker)s, %(sector)s, %(persistido_en_db)s,
            %(precio_cierre)s, %(precio_fecha)s, %(atr14)s,
            %(ml_prob_ganancia)s, %(ml_modelo_usado)s,
            %(pa_ev1)s, %(pa_ev2)s, %(pa_ev3)s, %(pa_ev4)s,
            %(bear_bos10)s, %(bear_choch10)s, %(bear_estructura)s,
            %(score_ponderado)s, %(condiciones_ok)s,
            %(estructura_10)s, %(dist_sl_10_pct)s, %(dist_sh_10_pct)s,
            %(dias_sl_10)s, %(dias_sh_10)s,
            %(alert_score)s, %(alert_nivel)s, %(alert_detalle)s
        )
    """
    # Filtrar solo los resultados sin error
    records = []
    for r in resultados:
        if r.get("error"):
            continue
        rec = {k: _native(v) for k, v in r.items()
               if k in (
                   "scan_fecha", "ticker", "sector", "persistido_en_db",
                   "precio_cierre", "precio_fecha", "atr14",
                   "ml_prob_ganancia", "ml_modelo_usado",
                   "pa_ev1", "pa_ev2", "pa_ev3", "pa_ev4",
                   "bear_bos10", "bear_choch10", "bear_estructura",
                   "score_ponderado", "condiciones_ok",
                   "estructura_10", "dist_sl_10_pct", "dist_sh_10_pct",
                   "dias_sl_10", "dias_sh_10",
                   "alert_score", "alert_nivel", "alert_detalle",
               )}
        records.append(rec)

    if not records:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=100)

    print(f"  [DB] Alertas persistidas: {len(records)} registros en alertas_scanner.")


# ─────────────────────────────────────────────────────────────
# Procesamiento de un ticker
# ─────────────────────────────────────────────────────────────

def procesar_ticker(ticker: str, modelos: dict,
                    persistir: bool = False,
                    scan_fecha: datetime = None) -> dict:
    """
    Ejecuta el pipeline completo para un ticker.

    Returns:
        dict con todos los resultados (o 'error' si fallo)
    """
    scan_fecha = scan_fecha or datetime.now()
    resultado_base = {
        "scan_fecha":    scan_fecha,
        "ticker":        ticker,
        "sector":        None,
        "persistido_en_db": False,
    }

    # ── 1. Obtener datos OHLCV ────────────────────────────────
    try:
        df_ohlcv, sector, es_nuevo = preparar_ticker(ticker, persistir=persistir)
        resultado_base["sector"]           = sector
        resultado_base["persistido_en_db"] = (es_nuevo and persistir)
    except ValueError as e:
        return {**resultado_base, "error": str(e)}
    except Exception as e:
        return {**resultado_base, "error": f"Error obteniendo datos: {e}"}

    # ── 2. Calcular features ──────────────────────────────────
    calc = calcular_features_completas(df_ohlcv, ticker, sector)
    if not calc["ok"]:
        return {**resultado_base, "error": calc["error"]}

    features_v3 = calc["features_v3"]
    features_pa = calc["features_pa"]
    meta        = calc["meta"]

    # ── 3. Generar senales ────────────────────────────────────
    signals = evaluar_ticker(features_v3, features_pa, sector, modelos, ticker=ticker)

    # ── 4. Clasificar alerta ──────────────────────────────────
    alert_score, alert_nivel, alert_detalle = clasificar_alerta(signals, meta)

    # ── 5. Armar resultado completo ───────────────────────────
    return {
        **resultado_base,
        # Precio
        "precio_cierre": meta.get("precio_cierre"),
        "precio_fecha":  meta.get("precio_fecha"),
        "atr14":         meta.get("atr14"),
        # ML
        "ml_prob_ganancia": signals["ml_prob_ganancia"],
        "ml_modelo_usado":  signals["ml_modelo_usado"],
        # PA
        "pa_ev1": signals["pa_ev1"],
        "pa_ev2": signals["pa_ev2"],
        "pa_ev3": signals["pa_ev3"],
        "pa_ev4": signals["pa_ev4"],
        # Bajistas
        "bear_bos10":      signals["bear_bos10"],
        "bear_choch10":    signals["bear_choch10"],
        "bear_estructura": signals["bear_estructura"],
        # Scoring
        "score_ponderado": meta.get("score_ponderado"),
        "condiciones_ok":  meta.get("condiciones_ok"),
        # Market structure
        "estructura_10":   features_pa.get("estructura_10"),
        "dist_sl_10_pct":  features_pa.get("dist_sl_10_pct"),
        "dist_sh_10_pct":  features_pa.get("dist_sh_10_pct"),
        "dias_sl_10":      features_pa.get("dias_sl_10"),
        "dias_sh_10":      features_pa.get("dias_sh_10"),
        # Alerta
        "alert_score":   alert_score,
        "alert_nivel":   alert_nivel,
        "alert_detalle": alert_detalle,
    }


# ─────────────────────────────────────────────────────────────
# Tabla resumen en consola
# ─────────────────────────────────────────────────────────────

_NIVEL_ORDER = {
    "COMPRA_FUERTE": 0,
    "COMPRA":        1,
    "NEUTRAL":       2,
    "VENTA":         3,
    "VENTA_FUERTE":  4,
    None:            5,
}


def imprimir_resumen(resultados: list):
    """Imprime tabla de resultados ordenada por nivel y score."""
    print("\n" + "=" * 75)
    print(f"  SCANNER DE ALERTAS ML  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 75)
    print(f"  {'TICKER':<6}  {'SECTOR':<22}  {'NIVEL':<14}  {'SCORE':>5}  "
          f"{'ML%':>5}  {'PRECIO':>8}  {'EV':>4}  {'BEAR':>4}")
    print("-" * 75)

    ok_results  = sorted(
        [r for r in resultados if not r.get("error")],
        key=lambda x: (_NIVEL_ORDER.get(x.get("alert_nivel"), 5), -x.get("alert_score", 0))
    )
    err_results = [r for r in resultados if r.get("error")]

    for r in ok_results:
        ticker = r["ticker"]
        sector = (r.get("sector") or "?")[:22]
        nivel  = r.get("alert_nivel", "?")[:14]
        score  = f"{r.get('alert_score', 0):.0f}"
        ml_pct = f"{r.get('ml_prob_ganancia', 0):.0%}"
        precio = f"${r.get('precio_cierre', 0):.2f}"
        ev_flags = "".join([
            str(r.get("pa_ev1", 0)),
            str(r.get("pa_ev2", 0)),
            str(r.get("pa_ev3", 0)),
            str(r.get("pa_ev4", 0)),
        ])
        bear_flags = "".join([
            "B" if r.get("bear_bos10")      else ".",
            "C" if r.get("bear_choch10")    else ".",
            "E" if r.get("bear_estructura") else ".",
        ])
        print(f"  {ticker:<6}  {sector:<22}  {nivel:<14}  {score:>5}  "
              f"{ml_pct:>5}  {precio:>8}  {ev_flags:>4}  {bear_flags:>4}")

    if err_results:
        print("-" * 75)
        print("  ERRORES:")
        for r in err_results:
            print(f"  {r['ticker']}: {str(r.get('error',''))[:60]}")

    print("=" * 75)
    n_compra  = sum(1 for r in ok_results if "COMPRA" in r.get("alert_nivel", ""))
    n_venta   = sum(1 for r in ok_results if "VENTA"  in r.get("alert_nivel", ""))
    n_neutral = sum(1 for r in ok_results if r.get("alert_nivel") == "NEUTRAL")
    print(f"  Resumen: {n_compra} COMPRA  |  {n_neutral} NEUTRAL  |  {n_venta} VENTA  "
          f"|  {len(err_results)} ERRORES")
    print("  EV: ev1/ev2/ev3/ev4  BEAR: B=bos10 C=choch10 E=estructura")
    print("=" * 75)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scanner de alertas ML para una lista de tickers."
    )
    parser.add_argument(
        "tickers", nargs="*",
        help="Lista de tickers a escanear (p.ej. JPM BAC AAPL)"
    )
    parser.add_argument(
        "--todos", action="store_true",
        help="Escanear todos los tickers del universo (ALL_TICKERS)"
    )
    parser.add_argument(
        "--persistir", action="store_true",
        help="Persistir tickers nuevos en la DB (activos + precios_diarios)"
    )
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="No enviar resumen por Telegram"
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="No guardar resultados en alertas_scanner"
    )

    args = parser.parse_args()

    if args.todos:
        tickers = ALL_TICKERS
    elif args.tickers:
        tickers = [t.upper().strip() for t in args.tickers]
    else:
        parser.print_help()
        print("\n[ERROR] Debes especificar tickers o usar --todos")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  SCANNER ALERTAS ML  |  {len(tickers)} tickers")
    print("=" * 60)
    print(f"  Persistir nuevos: {args.persistir}")
    print(f"  Guardar DB:       {not args.no_db}")
    print(f"  Telegram:         {not args.no_telegram}")
    print("-" * 60)

    # ── Cargar modelos una sola vez ───────────────────────────
    print("\n[1/3] Cargando modelos V3...")
    try:
        modelos = cargar_modelos_v3()
    except FileNotFoundError as e:
        print(f"[ERROR CRITICO] {e}")
        sys.exit(1)

    # ── Procesar cada ticker ──────────────────────────────────
    print(f"\n[2/3] Procesando {len(tickers)} tickers...\n")
    scan_fecha = datetime.now()
    resultados = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        r = procesar_ticker(ticker, modelos, persistir=args.persistir, scan_fecha=scan_fecha)
        resultados.append(r)

        if r.get("error"):
            print(f"ERROR: {str(r['error'])[:60]}")
        else:
            nivel = r.get("alert_nivel", "?")
            score = r.get("alert_score", 0)
            ml    = r.get("ml_prob_ganancia", 0)
            print(f"OK -> {nivel} (score={score:.0f}, ml={ml:.0%})")

    # ── Imprimir tabla ────────────────────────────────────────
    imprimir_resumen(resultados)

    # ── Guardar en DB ─────────────────────────────────────────
    if not args.no_db:
        print("\n[3/3] Guardando en DB...")
        try:
            persistir_alertas(resultados)
        except Exception as e:
            print(f"  [ERROR] No se pudo guardar en DB: {e}")

    # ── Telegram ──────────────────────────────────────────────
    if not args.no_telegram:
        print("  Enviando resumen por Telegram...")
        ok = enviar_resumen(resultados)
        if ok:
            print("  Telegram: mensaje enviado correctamente.")
        else:
            print("  Telegram: ERROR al enviar mensaje.")

    print("\nScanner completado.")


if __name__ == "__main__":
    main()
