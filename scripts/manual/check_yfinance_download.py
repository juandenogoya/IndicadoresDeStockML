"""
check_yfinance_download.py
Verifica si yf.download() esta disponible y que fechas retorna.
No escribe nada en DB — solo lectura.

Uso:
    python scripts/manual/check_yfinance_download.py
"""

import sys
import os
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import yfinance as yf

# Muestra de tickers representativos (1 lote chico)
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

SEP = "=" * 58

def main():
    hoy   = date.today()
    start = (hoy - timedelta(days=7)).isoformat()

    print()
    print(SEP)
    print(f"  CHECK YFINANCE DOWNLOAD  |  {hoy}")
    print(SEP)
    print(f"  Tickers : {', '.join(TICKERS)}")
    print(f"  Periodo : {start}  ->  hoy")
    print(f"  Metodo  : yf.download() (mismo que usa el pipeline)")
    print()

    try:
        raw = yf.download(
            tickers=TICKERS,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as e:
        print(f"  [ERROR] yf.download() lanzo excepcion:")
        print(f"  {e}")
        print()
        print(SEP)
        sys.exit(1)

    if raw is None or raw.empty:
        print("  [FALLO] yf.download() retorno DataFrame vacio.")
        print("  Posible causa: rate limit activo en esta IP.")
        print("  Recomendacion: esperar 15-30 minutos y reintentar.")
        print()
        print(SEP)
        sys.exit(1)

    # Mostrar fechas disponibles por ticker
    import pandas as pd

    print(f"  {'Ticker':<8}  {'Ultima fecha':>14}  {'Filas':>6}  {'Ultimo cierre':>14}")
    print("  " + "-" * 50)

    ok = 0
    for ticker in TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0).unique().tolist()
                lvl1 = raw.columns.get_level_values(1).unique().tolist()
                if ticker in lvl0:
                    df = raw[ticker].dropna(how="all")
                elif ticker in lvl1:
                    df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                else:
                    print(f"  {ticker:<8}  {'sin datos':>14}")
                    continue
            else:
                df = raw.dropna(how="all")

            if df.empty:
                print(f"  {ticker:<8}  {'sin datos':>14}")
                continue

            ultima_fecha = df.index[-1]
            if hasattr(ultima_fecha, "date"):
                ultima_fecha = ultima_fecha.date()

            ultimo_close = df["Close"].iloc[-1] if "Close" in df.columns else df.iloc[-1, 0]
            n_filas = len(df)

            print(f"  {ticker:<8}  {str(ultima_fecha):>14}  {n_filas:>6}  ${float(ultimo_close):>12.2f}")
            ok += 1

        except Exception as e:
            print(f"  {ticker:<8}  [ERROR] {str(e)[:40]}")

    print()
    if ok == len(TICKERS):
        print(f"  [OK] {ok}/{len(TICKERS)} tickers con datos.")
    elif ok > 0:
        print(f"  [PARCIAL] {ok}/{len(TICKERS)} tickers con datos.")
    else:
        print(f"  [FALLO] 0/{len(TICKERS)} tickers con datos.")

    print(SEP)
    print()
    sys.exit(0 if ok > 0 else 1)


if __name__ == "__main__":
    main()
