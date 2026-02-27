"""
22_batch_incorporar_tickers.py
Incorpora en batch los 102 nuevos tickers al sistema de backtesting PA.

Fases:
    1. Descarga historial (yfinance 5y) + guarda precios e indicadores en DB
    2. Calcula features_precio_accion y features_market_structure (bulk)
    3. Ejecuta BT historico PA (EV1-4 x SV1-4) solo para los nuevos tickers
       (sin TRUNCATE — los 22 tickers existentes quedan intactos)
    4. Recalcula metricas resultados_bt_pa
    5. Resumen final

Duracion estimada: 45-75 minutos (datos + features + BT).

Uso:
    python scripts/22_batch_incorporar_tickers.py
    python scripts/22_batch_incorporar_tickers.py --solo-datos     # skip BT
    python scripts/22_batch_incorporar_tickers.py --desde BABA     # continuar desde ticker
"""

import sys
import os
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── 102 nuevos tickers (no estaban en DB al 2026-02-27) ──────
NUEVOS_TICKERS = [
    # Technology
    "NVDA", "AAPL", "GOOG", "MSFT", "AMZN", "META", "TSM", "AVGO",
    "ASML", "MU", "AMD", "INTC", "IBM", "QCOM", "CRM", "DELL", "MSI",
    "SNOW", "ACN", "AI", "GLOB", "ERIC",
    # Automotive / EV
    "TSLA", "HMC", "XPEV", "NIO", "NIU",
    # Healthcare
    "LLY", "JNJ", "UNH", "PFE", "MRNA", "GSK", "CVS",
    # Energy
    "XOM", "CVX", "BP", "SHEL", "TTE", "OXY", "HAL", "FSLR", "VIST",
    # Financials (additional)
    "V", "MA", "C", "AIG", "PYPL", "UPST",
    # Consumer Discretionary (additional)
    "MCD", "NKE", "MELI", "ABNB", "EBAY", "ETSY", "TRIP", "SNAP",
    "LYFT", "UBER", "NFLX", "DIS", "AAP",
    # Consumer Staples (additional)
    "UL", "HSY",
    # Industrials
    "CAT", "RTX", "HON", "LMT", "DE", "UPS", "MMM", "BA", "RKLB",
    # Materials / Mining
    "NEM", "PAAS", "CDE", "HL", "HMY", "AU", "MP", "LAC", "B",
    # Real Estate
    "PLD",
    # Airlines
    "DAL", "UAL", "AAL",
    # Telecom (additional)
    "T", "VOD",
    # Brazil
    "PBR", "ITUB", "VALE", "NU", "BBD", "BSBR", "XP", "STNE", "PAGS", "SID",
    # China / Southeast Asia
    "BABA", "BIDU", "JD", "SE",
]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# Fase 1: Descarga + precios + indicadores (per ticker)
# ─────────────────────────────────────────────────────────────

def fase_datos(tickers: list, desde: str = None) -> tuple:
    """
    Descarga y persiste precios + indicadores tecnicos para cada ticker.
    Retorna (ok: list, errores: dict).
    """
    from src.pipeline.data_manager import descargar_yfinance, persistir_ticker_nuevo
    from src.indicators.technical import procesar_indicadores_ticker
    import yfinance as yf

    if desde:
        idx = tickers.index(desde) if desde in tickers else 0
        tickers = tickers[idx:]
        log(f"  Reanudando desde ticker #{idx+1}: {desde}")

    ok      = []
    errores = {}

    for i, ticker in enumerate(tickers, 1):
        log(f"  [{i:3}/{len(tickers)}] {ticker} — descargando...")
        t0 = time.time()

        try:
            # 1. Descargar OHLCV (5 anos)
            df_ohlcv = descargar_yfinance(ticker, "5y")
            n_barras = len(df_ohlcv)

            if n_barras < 100:
                raise ValueError(f"solo {n_barras} barras (minimo 100)")

            # 2. Auto-detectar sector desde yfinance
            sector = None
            try:
                info   = yf.Ticker(ticker).info
                sector = info.get("sector") or info.get("sectorDisp") or None
            except Exception:
                pass

            # 3. Guardar precios en DB y registrar en activos
            persistir_ticker_nuevo(df_ohlcv, ticker, sector=sector)

            # 4. Calcular indicadores tecnicos y guardar
            procesar_indicadores_ticker(ticker, df_ohlcv, guardar_db=True)

            elapsed = time.time() - t0
            sector_str = sector or "N/D"
            log(f"         OK | {n_barras} barras | sector: {sector_str} | {elapsed:.1f}s")
            ok.append(ticker)

        except Exception as e:
            log(f"         ERROR: {e}")
            errores[ticker] = str(e)

    return ok, errores


# ─────────────────────────────────────────────────────────────
# Fase 2: Features bulk (PA + MS para todos los tickers en DB)
# ─────────────────────────────────────────────────────────────

def fase_features():
    """
    Calcula features_precio_accion y features_market_structure para todos
    los tickers en precios_diarios (bulk, con upsert).
    """
    from src.indicators.precio_accion import procesar_features_precio_accion
    from src.indicators.market_structure import procesar_features_market_structure

    log("  Calculando features_precio_accion (todos los tickers)...")
    t0 = time.time()
    df_pa = procesar_features_precio_accion()
    log(f"  features_precio_accion OK: {len(df_pa):,} filas | {time.time()-t0:.0f}s")

    log("  Calculando features_market_structure (todos los tickers)...")
    t0 = time.time()
    df_ms = procesar_features_market_structure()
    log(f"  features_market_structure OK: {len(df_ms):,} filas | {time.time()-t0:.0f}s")

    return len(df_pa), len(df_ms)


# ─────────────────────────────────────────────────────────────
# Fase 3: BT historico para los nuevos tickers (sin TRUNCATE)
# ─────────────────────────────────────────────────────────────

def fase_backtesting(tickers_ok: list):
    """
    Ejecuta la simulacion PA historica (EV1-4 x SV1-4) solo para tickers_ok.
    Borra registros existentes de esos tickers antes de insertar (idempotente).
    NO afecta los 22 tickers originales.
    """
    from src.backtesting.simulator_pa import ejecutar_backtesting_pa

    n_combos = len(tickers_ok) * 16
    log(f"  Simulando {len(tickers_ok)} tickers x 16 combos = {n_combos} simulaciones...")

    t0    = time.time()
    df_ops = ejecutar_backtesting_pa(
        tickers=tickers_ok,
        guardar_db=True,
        truncate=False,   # DELETE solo estos tickers, no TRUNCATE general
    )

    log(f"  BT OK: {len(df_ops):,} operaciones | {time.time()-t0:.0f}s")
    return df_ops


# ─────────────────────────────────────────────────────────────
# Fase 4: Recalcular metricas globales
# ─────────────────────────────────────────────────────────────

def fase_metricas():
    """Recalcula resultados_bt_pa con todas las operaciones (22 + nuevos)."""
    from src.backtesting.metrics_pa import calcular_y_guardar_resultados_pa
    from src.data.database import query_df

    log("  Cargando todas las operaciones para recalcular metricas...")
    df_all = query_df("SELECT * FROM operaciones_bt_pa ORDER BY fecha_entrada")
    log(f"  Total operaciones: {len(df_all):,}")

    calcular_y_guardar_resultados_pa(df_all)
    log("  Metricas guardadas en resultados_bt_pa.")
    return len(df_all)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Incorpora en batch 102 nuevos tickers al sistema BT PA."
    )
    parser.add_argument(
        "--solo-datos", action="store_true",
        help="Ejecutar solo fase 1 (descarga) y fase 2 (features), sin BT"
    )
    parser.add_argument(
        "--desde", type=str, default=None,
        help="Reanudar desde un ticker especifico en la fase 1 (ej: --desde BABA)"
    )
    parser.add_argument(
        "--solo-bt", action="store_true",
        help="Ejecutar solo fases 3 y 4 (BT + metricas), asumiendo datos ya en DB"
    )
    args = parser.parse_args()

    t_inicio = time.time()
    log("=" * 60)
    log("  BATCH INCORPORACION DE TICKERS (102 nuevos)")
    log("=" * 60)
    log(f"  Total a procesar: {len(NUEVOS_TICKERS)} tickers")
    if args.desde:
        log(f"  Reanudar desde: {args.desde}")
    print()

    tickers_ok = NUEVOS_TICKERS[:]
    errores    = {}

    # ── Fase 1+2: Datos ──────────────────────────────────────────
    if not args.solo_bt:
        log("[FASE 1/4] Descargando precios e indicadores...")
        tickers_ok, errores = fase_datos(NUEVOS_TICKERS, desde=args.desde)
        log(f"  Fase 1 completada: {len(tickers_ok)} OK | {len(errores)} errores\n")

        log("[FASE 2/4] Calculando features (PA + MS)...")
        fase_features()
        log("  Fase 2 completada.\n")

    if args.solo_datos:
        log("  --solo-datos: omitiendo fases 3 y 4.")
    else:
        # ── Fase 3: BT ───────────────────────────────────────────
        log(f"[FASE 3/4] Backtesting PA para {len(tickers_ok)} tickers...")
        if tickers_ok:
            fase_backtesting(tickers_ok)
        else:
            log("  Sin tickers OK para BT.")
        log("  Fase 3 completada.\n")

        # ── Fase 4: Metricas ─────────────────────────────────────
        log("[FASE 4/4] Recalculando metricas globales...")
        n_total = fase_metricas()
        log(f"  Fase 4 completada: {n_total:,} operaciones totales.\n")

    # ── Resumen ──────────────────────────────────────────────────
    elapsed = time.time() - t_inicio
    log("=" * 60)
    log("  RESUMEN FINAL")
    log("=" * 60)
    log(f"  Tickers procesados OK : {len(tickers_ok)}")
    log(f"  Tickers con error     : {len(errores)}")
    log(f"  Tiempo total          : {elapsed/60:.1f} minutos")

    if errores:
        log("\n  Tickers con error:")
        for ticker, msg in errores.items():
            log(f"    {ticker:6}: {msg[:80]}")

    log("\n  Proximos pasos:")
    log("    1. git add + commit + push (config.py y scripts)")
    log("    2. El cron del dia siguiente procesa los 124 tickers automaticamente")
    log("    3. Verificar en Google Sheets que aparezcan los nuevos tickers")


if __name__ == "__main__":
    main()
