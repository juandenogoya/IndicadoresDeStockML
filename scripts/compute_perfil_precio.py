"""
compute_perfil_precio.py -- POC y area de valor por rueda y por ventana. LOCAL.

Escribe `perfil_precio_diario`: una fila por (ticker, fecha, ventana) con el POC,
el area de valor y su ancho. Ventanas 21 / 52 / 200 = corto / medio / largo.

POR QUE UNA TABLA SI LA VENTANA ES MOVIL
    Porque la ventana se desliza pero el resultado NO se pisa: el POC del 19/11
    calculado sobre las 50 ruedas previas es un hecho del 19/11 que no cambia, y
    el del 20/11 es OTRA fila. Es lo mismo que hace `indicadores_tecnicos` con la
    SMA50. La tabla existe para poder BACKTESTEAR: sin la serie historica no hay
    backtest, hay una foto de hoy.

CAPA DERIVADA Y RECOMPUTABLE
    Es un CACHE, no una fuente de verdad: funcion pura de `precios_nominales` via
    `src/utils/perfil_precio.py`. Si cambia el insumo (una correccion de split, o
    el dia que se arregle el patchwork de precios_diarios), se reconstruye entera
    con `--completo`.

POINT-IN-TIME POR CONSTRUCCION
    La fila de la rueda t usa `[t+1-ventana : t+1]`: solo el pasado y la propia
    rueda. No hay informacion futura y la fila de una fecha no cambia cuando
    llegan barras nuevas. (El detector de NIVELES no tiene esta propiedad: mira
    toda la historia y para backtestear hay que pedirle `hasta=D`.)

SOBRE PRECIOS NOMINALES, NO SOBRE precios_diarios
    `precios_diarios` es un patchwork de dos bases (ver patrones criticos de
    CLAUDE.md). Medido el 19/11/2026 sobre 10 tickers, comparando el POC de hoy en
    las dos series: en la ventana 21 dan identicos los 10; en la de 200, HON
    difiere 6,70% (229,05 contra 244,39, 13 bins) porque su ventana cruza la
    costura del 2026-06-29. Y esa medicion SUBESTIMA el problema: miro la ventana
    de hoy, mientras que en las fechas historicas -- las del backtest -- la
    ventana entera cae en la zona ajustada.
    COROLARIO: lo que compare el precio contra este POC tiene que usar cierres
    NOMINALES. Mezclar bases es el bug.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(".env", override=True)
os.environ.pop("DATABASE_URL", None)

from sqlalchemy import text

from src.data.database import get_engine
from src.utils import perfil_precio as pp

TABLA = "perfil_precio_diario"
FUENTE = "precios_nominales"
VENTANAS = pp.VENTANAS                     # (21, 52, 200)

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    ticker        TEXT        NOT NULL,
    fecha         DATE        NOT NULL,
    ventana       SMALLINT    NOT NULL,
    poc           NUMERIC(14,4),
    val           NUMERIC(14,4),
    vah           NUMERIC(14,4),
    ancho_va_pct  NUMERIC(8,3),
    bins          INTEGER,
    pico_pct      NUMERIC(8,3),
    volumen       DOUBLE PRECISION,
    calculado_en  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, fecha, ventana)
)
"""

UPSERT = f"""
INSERT INTO {TABLA} (ticker,fecha,ventana,poc,val,vah,ancho_va_pct,bins,pico_pct,volumen)
VALUES (:t,:f,:v,:poc,:val,:vah,:an,:bi,:pi,:vol)
ON CONFLICT (ticker,fecha,ventana) DO UPDATE SET
  poc=EXCLUDED.poc, val=EXCLUDED.val, vah=EXCLUDED.vah,
  ancho_va_pct=EXCLUDED.ancho_va_pct, bins=EXCLUDED.bins,
  pico_pct=EXCLUDED.pico_pct, volumen=EXCLUDED.volumen, calculado_en=now()
"""


def log(m):
    print(m, flush=True)


def estado(eng):
    with eng.connect() as c:
        if not c.execute(text("SELECT to_regclass(:t)"), {"t": TABLA}).scalar():
            log(f"{TABLA}: no existe todavia.")
            return
        r = c.execute(text(f"""SELECT COUNT(*) n, COUNT(DISTINCT ticker) tk,
                               MIN(fecha) a, MAX(fecha) b FROM {TABLA}""")).one()
        log(f"{TABLA}: {r.n:,} filas | {r.tk} tickers | {r.a} -> {r.b}")
        for v in VENTANAS:
            q = text(f"""SELECT COUNT(*) n, COUNT(*) FILTER (WHERE poc IS NULL) nulos
                         FROM {TABLA} WHERE ventana = :v""")
            x = c.execute(q, {"v": v}).one()
            log(f"  ventana {v:3d}: {x.n:>8,} filas ({x.nulos:,} sin POC)")


def tickers_y_ultimas(eng, tickers):
    with eng.connect() as c:
        disp = [r[0] for r in c.execute(text(
            f"SELECT DISTINCT ticker FROM {FUENTE} ORDER BY ticker"))]
        ya = dict(c.execute(text(
            f"SELECT ticker, MAX(fecha) FROM {TABLA} GROUP BY ticker")).fetchall()) \
            if c.execute(text("SELECT to_regclass(:t)"), {"t": TABLA}).scalar() else {}
    if tickers:
        pedidos = [t.strip().upper() for t in tickers.split(",")]
        falta = [t for t in pedidos if t not in disp]
        if falta:
            log(f"  OJO: sin datos en {FUENTE}: {', '.join(falta)}")
        disp = [t for t in disp if t in pedidos]
    return disp, ya


def filas_de_ticker(eng, tk, desde_idx_min=None):
    with eng.connect() as c:
        d = pd.DataFrame(c.execute(text(
            f"SELECT fecha,high,low,volume FROM {FUENTE} WHERE ticker=:t ORDER BY fecha"),
            {"t": tk}).fetchall(), columns=["fecha", "high", "low", "volume"])
    for k in ("high", "low", "volume"):
        d[k] = pd.to_numeric(d[k])
    # A datetime64: el driver devuelve `datetime.date` y `np.searchsorted` del
    # camino incremental no puede compararlo contra un np.datetime64.
    d["fecha"] = pd.to_datetime(d["fecha"])
    return d


def computar(d: pd.DataFrame, tk: str, desde=None):
    """Una fila por (fecha, ventana) desde `desde` (inclusive) hasta el final."""
    h = d.high.to_numpy(float)
    lo = d.low.to_numpy(float)
    vol = d.volume.to_numpy(float)
    fechas = d.fecha.to_numpy()
    arranque = 0
    if desde is not None:
        pos = np.searchsorted(fechas, np.datetime64(pd.Timestamp(desde)))
        arranque = int(pos)
    out = []
    for v in VENTANAS:
        ini = max(arranque, v - 1)
        for t in range(ini, len(h)):
            r = pp.calcular(h[t + 1 - v:t + 1], lo[t + 1 - v:t + 1], vol[t + 1 - v:t + 1])
            if not r.valido:
                continue
            out.append({"t": tk, "f": pd.Timestamp(fechas[t]).date(), "v": int(v),
                        "poc": r.poc, "val": r.val, "vah": r.vah,
                        "an": round(r.ancho_va_pct, 3), "bi": r.bins,
                        "pi": round(r.pico_pct, 3), "vol": r.volumen})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crear", action="store_true")
    ap.add_argument("--completo", action="store_true",
                    help="recalcula toda la historia (la tabla es un cache)")
    ap.add_argument("--tickers")
    ap.add_argument("--desde", help="AAAA-MM-DD; por defecto, incremental")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    eng = get_engine()
    if a.status:
        estado(eng)
        return 0
    if a.crear:
        with eng.begin() as c:
            c.execute(text(DDL))
        log(f"{TABLA} creada (o ya existia).")

    tks, ya = tickers_y_ultimas(eng, a.tickers)
    if not tks:
        log(f"Sin tickers en {FUENTE}. Corre cargar_precios_nominales.py")
        return 1
    log(f"{len(tks)} tickers | ventanas {VENTANAS} | "
        f"{'COMPLETO' if a.completo else ('desde ' + a.desde if a.desde else 'incremental')}")

    t0 = datetime.now()
    total = 0
    for i, tk in enumerate(tks, 1):
        d = filas_de_ticker(eng, tk)
        if d.empty:
            continue
        if a.completo:
            desde = None
        elif a.desde:
            desde = pd.Timestamp(a.desde)
        else:
            u = ya.get(tk)
            desde = (pd.Timestamp(u) + pd.Timedelta(days=1)) if u else None
        filas = computar(d, tk, desde)
        if filas and not a.dry_run:
            with eng.begin() as c:
                c.execute(text(UPSERT), filas)
        total += len(filas)
        if i % 25 == 0 or i == len(tks):
            log(f"  {i}/{len(tks)} | {total:,} filas | "
                f"{(datetime.now() - t0).total_seconds():.0f}s")

    log(f"\n{'(dry-run) ' if a.dry_run else ''}Filas: {total:,} en "
        f"{(datetime.now() - t0).total_seconds():.0f}s")
    estado(eng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
