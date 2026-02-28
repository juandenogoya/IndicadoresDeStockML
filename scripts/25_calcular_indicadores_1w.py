"""
25_calcular_indicadores_1w.py
Calcula indicadores tecnicos semanales para los 124 tickers.

Lee precios semanales de Railway (precios_semanales),
calcula RSI, MACD, SMA, ATR, BB, ADX, OBV, Momentum
y persiste en indicadores_tecnicos_1w.

Mismos periodos que 1D (en barras semanales):
    SMA21=21w, SMA50=50w, SMA200=200w, RSI14=14w,
    MACD 12/26/9w, ATR14=14w, BB20=20w, ADX14=14w

Nota: tickers con menos de 200 semanas no tendran SMA200.
      LAC tiene 125 semanas -> SMA200 ausente.

Uso:
    python scripts/25_calcular_indicadores_1w.py
    python scripts/25_calcular_indicadores_1w.py --ticker JPM
    python scripts/25_calcular_indicadores_1w.py --desde MSFT
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
        description="Calcula indicadores tecnicos 1W para 124 tickers."
    )
    parser.add_argument("--ticker", type=str, default=None,
                        help="Procesar solo este ticker.")
    parser.add_argument("--desde", type=str, default=None,
                        help="Reanudar desde este ticker.")
    args = parser.parse_args()

    from src.data.database import query_df
    from src.indicators.technical_1w import procesar_indicadores_ticker_1w

    # ── Lista de tickers ──────────────────────────────────────
    df_activos = query_df(
        "SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker"
    )
    tickers = df_activos["ticker"].tolist() if not df_activos.empty else []

    if args.ticker:
        tickers = [args.ticker]
    elif args.desde and args.desde in tickers:
        idx = tickers.index(args.desde)
        tickers = tickers[idx:]
        log(f"  Reanudando desde #{idx+1}: {args.desde}")

    log("=" * 60)
    log("  INDICADORES TECNICOS 1W")
    log("=" * 60)
    log(f"  Tickers: {len(tickers)}")
    log(f"  Periodos: SMA21/50/200 | RSI14 | MACD 12/26/9 | ATR14 | BB20 | ADX14")
    print()

    t0_total = time.time()
    n_ok     = 0
    n_err    = 0
    n_warn   = 0
    errores  = {}

    for i, ticker in enumerate(tickers, 1):
        t0 = time.time()
        try:
            df_ind = procesar_indicadores_ticker_1w(ticker, guardar_db=True)
            elapsed = time.time() - t0

            if df_ind.empty:
                log(f"  [{i:03d}/{len(tickers)}] {ticker:6} | WARN: sin datos o insuficientes semanas")
                n_warn += 1
            else:
                ultima = df_ind["fecha"].max()
                log(
                    f"  [{i:03d}/{len(tickers)}] {ticker:6} | "
                    f"{len(df_ind):>4} registros | "
                    f"ultimo: {ultima} | "
                    f"{elapsed:.1f}s"
                )
                n_ok += 1

        except Exception as e:
            elapsed = time.time() - t0
            log(f"  [{i:03d}/{len(tickers)}] {ticker:6} | ERROR: {str(e)[:70]}")
            errores[ticker] = str(e)
            n_err += 1

    # ── Resumen ───────────────────────────────────────────────
    total = time.time() - t0_total
    print()
    log("=" * 60)
    log("  RESUMEN")
    log("=" * 60)
    log(f"  Tickers OK    : {n_ok}")
    log(f"  Tickers WARN  : {n_warn}  (sin SMA200 por historico corto)")
    log(f"  Tickers ERROR : {n_err}")
    log(f"  Tiempo total  : {total:.0f}s")

    if errores:
        log("\n  Errores:")
        for t, msg in errores.items():
            log(f"    {t:6}: {msg[:70]}")

    # ── Verificacion en DB ────────────────────────────────────
    print()
    log("  Verificacion en Railway:")
    df_check = query_df("""
        SELECT ticker,
               COUNT(*)    AS n_registros,
               MIN(fecha)  AS primera,
               MAX(fecha)  AS ultima
        FROM   indicadores_tecnicos_1w
        GROUP  BY ticker
        ORDER  BY ticker
    """)
    if not df_check.empty:
        log(f"  Total tickers en indicadores_tecnicos_1w: {len(df_check)}")
        log(f"  Total registros                          : {df_check['n_registros'].sum():,}")
        log(f"  Rango global                             : "
            f"{df_check['primera'].min()} -> {df_check['ultima'].max()}")

        # Muestra tickers con pocos registros (historico corto)
        pocos = df_check[df_check["n_registros"] < 50]
        if not pocos.empty:
            log(f"\n  Tickers con menos de 50 registros (historico corto):")
            for _, row in pocos.iterrows():
                log(f"    {row['ticker']:6}: {row['n_registros']} registros")
    else:
        log("  WARN: indicadores_tecnicos_1w vacia.")

    print()
    log("  Proximo paso:")
    log("    Etapa 4: python scripts/26_calcular_features_pa_1w.py")


if __name__ == "__main__":
    main()
