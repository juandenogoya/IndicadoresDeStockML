"""
24_poblar_precios_semanales.py
Resamplea precios_diarios -> precios_semanales para los 124 tickers.

Lee precios diarios de Railway, resamplea a semanal (OHLCV W-FRI)
y persiste en la tabla precios_semanales.

- fecha_semana = ultimo dia habil real de la semana
- Semana en curso excluida (datos parciales)
- Idempotente: ON CONFLICT DO UPDATE

Uso:
    python scripts/24_poblar_precios_semanales.py
    python scripts/24_poblar_precios_semanales.py --ticker JPM   # solo un ticker
    python scripts/24_poblar_precios_semanales.py --desde MSFT   # reanudar desde ticker
"""

import sys
import os
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Resamplea precios_diarios -> precios_semanales (124 tickers)."
    )
    parser.add_argument("--ticker", type=str, default=None,
                        help="Procesar solo este ticker.")
    parser.add_argument("--desde", type=str, default=None,
                        help="Reanudar desde este ticker (orden alfabetico).")
    args = parser.parse_args()

    from src.data.database import query_df
    from src.data.resample_weekly import procesar_ticker_semanal

    # ── Obtener lista de tickers ──────────────────────────────────
    df_activos = query_df(
        "SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker"
    )
    tickers = df_activos["ticker"].tolist() if not df_activos.empty else []

    if not tickers:
        log("ERROR: no hay tickers en la tabla activos.")
        sys.exit(1)

    # Filtros CLI
    if args.ticker:
        tickers = [args.ticker]
    elif args.desde:
        if args.desde in tickers:
            idx = tickers.index(args.desde)
            tickers = tickers[idx:]
            log(f"  Reanudando desde ticker #{idx+1}: {args.desde}")
        else:
            log(f"  WARN: {args.desde} no encontrado, procesando todos.")

    log("=" * 60)
    log("  POBLAR PRECIOS SEMANALES")
    log("=" * 60)
    log(f"  Tickers a procesar: {len(tickers)}")
    log(f"  Logica: resample diario -> semanal (W-FRI, fecha_semana = ultimo dia habil)")
    log(f"  Semana en curso excluida (datos parciales)")
    print()

    t0_total = time.time()
    n_ok     = 0
    n_err    = 0
    errores  = {}

    for i, ticker in enumerate(tickers, 1):
        t0 = time.time()
        res = procesar_ticker_semanal(ticker)
        elapsed = time.time() - t0

        if res["ok"]:
            log(
                f"  [{i:03d}/{len(tickers)}] {ticker:6} | "
                f"{res['n_semanas']:>4} semanas | "
                f"{str(res['fecha_primera'])} -> {str(res['fecha_ultima'])} | "
                f"{elapsed:.1f}s"
            )
            n_ok += 1
        else:
            log(f"  [{i:03d}/{len(tickers)}] {ticker:6} | ERROR: {res['error']}")
            errores[ticker] = res["error"]
            n_err += 1

    # ── Resumen ───────────────────────────────────────────────────
    total = time.time() - t0_total
    print()
    log("=" * 60)
    log("  RESUMEN")
    log("=" * 60)
    log(f"  Tickers OK    : {n_ok}")
    log(f"  Tickers error : {n_err}")
    log(f"  Tiempo total  : {total:.0f}s")

    if errores:
        log("\n  Errores:")
        for t, msg in errores.items():
            log(f"    {t:6}: {msg[:70]}")

    # ── Verificacion final en DB ──────────────────────────────────
    print()
    log("  Verificacion en Railway:")
    from src.data.database import query_df as qdf
    df_check = qdf("""
        SELECT ticker,
               COUNT(*)          AS n_semanas,
               MIN(fecha_semana) AS primera,
               MAX(fecha_semana) AS ultima
        FROM   precios_semanales
        GROUP  BY ticker
        ORDER  BY ticker
    """)
    if not df_check.empty:
        log(f"  Total tickers en precios_semanales: {len(df_check)}")
        log(f"  Total semanas en DB               : {df_check['n_semanas'].sum():,}")
        log(f"  Rango de fechas global            : "
            f"{df_check['primera'].min()} -> {df_check['ultima'].max()}")
    else:
        log("  WARN: precios_semanales vacia.")

    print()
    log("  Proximo paso:")
    log("    Etapa 3: python scripts/25_calcular_indicadores_1w.py")


if __name__ == "__main__":
    main()
