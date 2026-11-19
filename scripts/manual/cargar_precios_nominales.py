"""
cargar_precios_nominales.py -- OHLCV NOMINAL en una tabla aparte. LOCAL-only.

POR QUE EXISTE UNA SEGUNDA TABLA DE PRECIOS
    `precios_diarios` es un PATCHWORK: su historia esta en precios AJUSTADOS POR
    DIVIDENDOS y los ultimos meses en nominales, con la costura en la fecha del
    ultimo backfill de cada ticker (medido el 19/11/2026: CVX coincide con el
    nominal desde 2026-02-17 y HON desde 2026-06-29; el descuento llega al 18% en
    2021). Ver el bloque "PENDIENTE GRANDE" de la AGENDA.

    Un nivel psicologico es un precio NOMINAL: la gente recuerda "208 dolares".
    Sobre la serie ajustada el ejercicio es imposible, y peor que imposible:
    FABRICA niveles, porque un precio nominal que baja al ritmo del dividendo se
    vuelve una linea horizontal perfecta al ajustar.

    Esta tabla NO toca el pipeline y NADIE del pipeline la lee. Es el insumo del
    detector de niveles y nada mas. Si algun dia se arregla `precios_diarios`, el
    detector cambia de fuente y esta tabla se borra.

DE DONDE SALE EL NOMINAL
    De la MISMA llamada de yahooquery que ya hace el proyecto. Con
    `adj_ohlc=False` las columnas open/high/low/close son NOMINALES y `adjclose`
    viene aparte (verificado el 19/11/2026: CVX 2022-12-30, close 179,49 = crudo
    de Alpha Vantage; adjclose 152,67 = ajustado de Alpha Vantage). No hace falta
    ninguna fuente nueva ni gastar cuota.
"""

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
load_dotenv(".env", override=True)
os.environ.pop("DATABASE_URL", None)          # LOCAL manda (CLAUDE.md)

from sqlalchemy import text

from src.data.database import get_engine
from src.data.universo import get_universo
from src.utils.yfinance_lock import acquire

TABLA = "precios_nominales"
LOTE = 40

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    ticker      TEXT        NOT NULL,
    fecha       DATE        NOT NULL,
    open        NUMERIC(14,4),
    high        NUMERIC(14,4),
    low         NUMERIC(14,4),
    close       NUMERIC(14,4),
    volume      BIGINT,
    bajado_en   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, fecha)
)
"""


def log(m):
    print(m, flush=True)


def estado(eng):
    with eng.connect() as c:
        existe = c.execute(text(
            "SELECT to_regclass(:t)"), {"t": TABLA}).scalar()
        if not existe:
            log(f"{TABLA}: no existe todavia.")
            return
        r = c.execute(text(f"""SELECT COUNT(*) n, COUNT(DISTINCT ticker) tk,
                               MIN(fecha) a, MAX(fecha) b FROM {TABLA}""")).one()
        log(f"{TABLA}: {r.n:,} filas | {r.tk} tickers | {r.a} -> {r.b}")


def bajar(tickers, desde, hasta):
    """OHLCV NOMINAL de yahooquery. adj_ohlc=False -> open/high/low/close crudos."""
    from yahooquery import Ticker
    t = Ticker(list(tickers), asynchronous=True)
    df = t.history(start=desde.strftime("%Y-%m-%d"),
                   end=(hasta + timedelta(days=1)).strftime("%Y-%m-%d"),
                   interval="1d", adj_ohlc=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if "symbol" not in df.columns:
        df["symbol"] = list(tickers)[0]
    df["fecha"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[df["fecha"].dt.date <= hasta]
    cols = ["symbol", "fecha", "open", "high", "low", "close", "volume"]
    falta = [c for c in cols if c not in df.columns]
    if falta:
        raise RuntimeError(f"yahooquery no trajo {falta}")
    df = df[cols].rename(columns={"symbol": "ticker"}).dropna(subset=["close"])
    return df


def guardar(eng, df):
    filas = [{"t": r.ticker, "f": r.fecha.date(), "o": float(r.open),
              "h": float(r.high), "l": float(r.low), "c": float(r.close),
              "v": int(r.volume) if pd.notna(r.volume) else None}
             for r in df.itertuples()]
    if not filas:
        return 0
    q = text(f"""INSERT INTO {TABLA} (ticker,fecha,open,high,low,close,volume)
                 VALUES (:t,:f,:o,:h,:l,:c,:v)
                 ON CONFLICT (ticker,fecha) DO UPDATE SET
                   open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                   close=EXCLUDED.close, volume=EXCLUDED.volume,
                   bajado_en=now()""")
    with eng.begin() as c:
        c.execute(q, filas)
    return len(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="coma-separados; por defecto, el universo")
    ap.add_argument("--desde", default="2021-01-01")
    ap.add_argument("--hasta", help="por defecto, la ultima rueda de precios_diarios")
    ap.add_argument("--crear", action="store_true", help="crea la tabla si falta")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="baja y no escribe")
    a = ap.parse_args()

    eng = get_engine()
    if a.status:
        estado(eng)
        return 0
    if a.crear:
        with eng.begin() as c:
            c.execute(text(DDL))
        log(f"{TABLA} creada (o ya existia).")

    tickers = ([x.strip().upper() for x in a.tickers.split(",")]
               if a.tickers else sorted(get_universo()))
    desde = pd.Timestamp(a.desde).date()
    if a.hasta:
        hasta = pd.Timestamp(a.hasta).date()
    else:
        with eng.connect() as c:
            hasta = c.execute(text("SELECT MAX(fecha) FROM precios_diarios")).scalar()
    # La ultima rueda de precios_diarios ya cerro: evita la barra EN CURSO de hoy
    # (regla 10 de CLAUDE.md).
    log(f"{len(tickers)} tickers | {desde} -> {hasta}")

    # rate limit de yahoo es por IP: el lock es global del proyecto (CLAUDE.md #9)
    acquire("cargar_precios_nominales")

    total = 0
    for k in range(0, len(tickers), LOTE):
        lote = tickers[k:k + LOTE]
        df = bajar(lote, desde, hasta)
        if df.empty:
            log(f"  [{k // LOTE + 1}] sin datos para {len(lote)} tickers")
            continue
        n = 0 if a.dry_run else guardar(eng, df)
        total += n
        log(f"  [{k // LOTE + 1}] {len(lote)} tickers -> {len(df):,} filas"
            f"{' (dry-run)' if a.dry_run else f', {n:,} escritas'}")

    log(f"\nTotal escrito: {total:,}")
    estado(eng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
