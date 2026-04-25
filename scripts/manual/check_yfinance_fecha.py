"""
check_yfinance_fecha.py
Detecta a que fecha de cierre corresponden los datos actuales de yfinance,
comparando precios en tiempo real contra los cierres historicos en DB.

Uso interno desde poblar_opciones.bat — no ejecutar directamente.
Retorna exit code 0 siempre (solo informativo).
"""

import os
import sys
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

import yfinance as yf
from sqlalchemy import create_engine, text

REFERENCIAS = ["AAPL", "NVDA", "MSFT", "TSLA"]


def get_engine():
    return create_engine(os.environ["DATABASE_URL"])


def main():
    print()
    print("  Verificando fecha real de datos yfinance...")
    print("  " + "-" * 50)

    # Obtener precios actuales de yfinance
    precios_yf = {}
    for ticker in REFERENCIAS:
        try:
            fi = yf.Ticker(ticker).fast_info
            precio = getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None)
            if precio:
                precios_yf[ticker] = float(precio)
        except Exception:
            pass

    if not precios_yf:
        print("  No se pudo obtener precios de yfinance.")
        return

    # Comparar contra cierres historicos en DB (ultimos 10 dias habiles)
    engine = get_engine()
    with engine.connect() as conn:
        tickers_str = "', '".join(precios_yf.keys())
        rows = conn.execute(text(f"""
            SELECT ticker, fecha, close
            FROM precios_diarios
            WHERE ticker IN ('{tickers_str}')
              AND fecha >= CURRENT_DATE - INTERVAL '15 days'
            ORDER BY ticker, fecha DESC
        """)).fetchall()

    # Agrupar por fecha y calcular diferencia promedio
    from collections import defaultdict
    diffs_por_fecha = defaultdict(list)
    for ticker, fecha, close in rows:
        if ticker in precios_yf and close and float(close) > 0:
            diff_pct = abs(precios_yf[ticker] - float(close)) / float(close) * 100
            diffs_por_fecha[fecha].append(diff_pct)

    if not diffs_por_fecha:
        print("  Sin historial en DB para comparar.")
        return

    # La fecha con menor diferencia promedio = fecha de los datos yfinance
    fecha_match = min(diffs_por_fecha, key=lambda f: sum(diffs_por_fecha[f]) / len(diffs_por_fecha[f]))
    diff_promedio = sum(diffs_por_fecha[fecha_match]) / len(diffs_por_fecha[fecha_match])

    print(f"  {'Ticker':<8} {'Precio yfinance':>16}  {'Cierre ' + str(fecha_match):>18}  {'Diff':>6}")
    print("  " + "-" * 55)
    for ticker, fecha, close in rows:
        if ticker in precios_yf and fecha == fecha_match:
            diff = (precios_yf[ticker] - float(close)) / float(close) * 100
            print(f"  {ticker:<8} ${precios_yf[ticker]:>14.2f}  ${float(close):>16.2f}  {diff:>+5.1f}%")

    print()
    if diff_promedio < 1.0:
        confianza = "ALTA"
    elif diff_promedio < 3.0:
        confianza = "MEDIA"
    else:
        confianza = "BAJA"

    print(f"  >>> yfinance esta mostrando datos del cierre: {fecha_match} <<<")
    print(f"  >>> Diferencia promedio: {diff_promedio:.2f}%  (confianza: {confianza}) <<<")
    print()


if __name__ == "__main__":
    main()
    sys.exit(0)
