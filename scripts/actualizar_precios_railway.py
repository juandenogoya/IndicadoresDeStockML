"""
actualizar_precios_railway.py
Script local para actualizar precios diarios en Railway via yfinance.

Debe correr a las 19:00 Argentina (22:00 UTC), post-cierre NYSE.
GitHub Actions corre 1 hora despues (20:00 Argentina / 23:00 UTC)
y encontrara los datos frescos ya disponibles en Railway.

Uso:
    python scripts/actualizar_precios_railway.py
    python scripts/actualizar_precios_railway.py --periodo 5d   # solo ultimos 5 dias
    python scripts/actualizar_precios_railway.py --ticker JPM   # solo un ticker

Automatizacion (Windows Task Scheduler):
    Programa -> python
    Argumentos -> scripts/actualizar_precios_railway.py
    Directorio -> C:/Users/juand/OneDrive/Escritorio/Indicadores y Machine Learning
    Hora -> 19:00 L-V
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
        description="Actualiza precios en Railway via yfinance (script local)."
    )
    parser.add_argument("--periodo", default="5d",
                        help="Periodo de descarga: 1d, 5d, 1mo (default: 5d)")
    parser.add_argument("--ticker", default=None,
                        help="Actualizar solo este ticker (default: todos)")
    args = parser.parse_args()

    import yfinance as yf
    import pandas as pd
    from src.data.database import upsert_precios, query_df
    from src.pipeline.data_manager import cargar_precios_db
    from src.indicators.technical import procesar_indicadores_ticker

    # Obtener lista de tickers desde Railway
    try:
        df_activos = query_df(
            "SELECT ticker FROM activos WHERE activo = TRUE ORDER BY ticker"
        )
        tickers = df_activos["ticker"].tolist() if not df_activos.empty else []
    except Exception as e:
        log(f"ERROR leyendo activos de Railway: {e}")
        sys.exit(1)

    if args.ticker:
        if args.ticker not in tickers:
            log(f"WARN: {args.ticker} no esta en activos de Railway. Igual se intenta.")
        tickers = [args.ticker]

    log("=" * 55)
    log(f"  ACTUALIZACION DIARIA DE PRECIOS — Railway")
    log(f"  {len(tickers)} tickers | periodo: {args.periodo}")
    log("=" * 55)

    # Descarga batch: una sola llamada para todos los tickers
    log(f"\n  Descargando via yfinance (batch)...")
    t0 = time.time()
    try:
        raw_all = yf.download(
            tickers=tickers,
            period=args.periodo,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as e:
        log(f"ERROR en descarga batch: {e}")
        sys.exit(1)

    if raw_all is None or raw_all.empty:
        log("ERROR: yfinance no retorno datos.")
        sys.exit(1)

    elapsed = time.time() - t0
    log(f"  Batch descargado: {raw_all.shape[0]} filas | {elapsed:.1f}s\n")

    n_ok = 0
    n_err = 0
    errores = {}

    for i, ticker in enumerate(tickers, 1):
        try:
            # Extraer datos del ticker del DataFrame batch
            if isinstance(raw_all.columns, pd.MultiIndex):
                lvl0 = raw_all.columns.get_level_values(0).unique().tolist()
                lvl1 = raw_all.columns.get_level_values(1).unique().tolist()
                if ticker in lvl0:
                    df_t = raw_all[ticker].copy()
                elif ticker in lvl1:
                    df_t = raw_all.xs(ticker, axis=1, level=1).copy()
                else:
                    raise ValueError(f"{ticker} no encontrado en resultado batch")
            else:
                df_t = raw_all.copy()

            df_t = df_t.reset_index()
            df_t.columns = [
                c[0].lower() if isinstance(c, tuple) else str(c).lower()
                for c in df_t.columns
            ]
            df_t = df_t.rename(columns={"date": "fecha", "price": "fecha"})
            df_t["fecha"] = pd.to_datetime(df_t["fecha"])

            cols = [c for c in ["fecha", "open", "high", "low", "close", "volume"]
                    if c in df_t.columns]
            df_t = df_t[cols].dropna(subset=["close"])
            df_t = df_t[df_t["close"] > 0]
            df_t["ticker"] = ticker
            df_t["adj_close"] = df_t["close"]
            df_t = df_t.sort_values("fecha").reset_index(drop=True)

            if df_t.empty:
                raise ValueError("sin datos validos")

            # Upsert en Railway
            upsert_precios(df_t)
            ultima = df_t["fecha"].max().date()

            # Recalcular indicadores con historico completo de DB
            df_full = cargar_precios_db(ticker, ultimas_n=500)
            if len(df_full) >= 250:
                procesar_indicadores_ticker(ticker, df_full, guardar_db=True)
                log(f"  [{i:03d}/{len(tickers)}] {ticker}: OK | {len(df_t)} barras | cierre: {ultima}")
            else:
                log(f"  [{i:03d}/{len(tickers)}] {ticker}: precios OK | historico insuf. para indicadores")

            n_ok += 1

        except Exception as e:
            log(f"  [{i:03d}/{len(tickers)}] {ticker}: ERROR - {str(e)[:80]}")
            errores[ticker] = str(e)
            n_err += 1

    # Resumen
    total = time.time() - t0
    print()
    log("=" * 55)
    log(f"  RESUMEN")
    log("=" * 55)
    log(f"  Tickers OK    : {n_ok}")
    log(f"  Tickers error : {n_err}")
    log(f"  Tiempo total  : {total:.0f}s")

    if errores:
        log("\n  Errores:")
        for t, msg in errores.items():
            log(f"    {t}: {msg[:60]}")

    log("\n  Railway actualizado. GH Actions puede correr a las 20:00 Argentina.")


if __name__ == "__main__":
    main()
