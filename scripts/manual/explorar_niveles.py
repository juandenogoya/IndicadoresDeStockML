"""
explorar_niveles.py -- niveles de soporte y resistencia de un ticker. SOLO LEE.

Lee `precios_nominales` (NO `precios_diarios`: su historia esta en precios
ajustados por dividendos y sobre esa serie los niveles son artefactos; ver el
encabezado de cargar_precios_nominales.py y la AGENDA del 19/11/2026).

Un nivel es una franja donde se acumularon GIROS confirmados: swing lows (el
precio llego desde arriba y reboto -> soporte) y swing highs (llego desde abajo y
fue rechazado -> resistencia). Una franja con las dos cosas es un cambio de
polaridad. Es DESCRIPTIVO: dice donde reacciono, no que vaya a reaccionar.

El orden lo encabeza la PERSISTENCIA (anios distintos con toque) y no el conteo:
cuatro toques en un trimestre son un solo episodio; cinco repartidos en dos anios
atravesaron varios regimenes.

Con `--nulo N` se corre la misma deteccion sobre N series PERMUTADAS del mismo
ticker (mismos retornos en otro orden). Eso no mide si el nivel sirve: mide si
"reacciono k veces" es un hecho o la aritmetica del azar, que es lo que hace falta
para poder llamarlo FUERTE.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
load_dotenv(".env", override=True)
os.environ.pop("DATABASE_URL", None)

from sqlalchemy import text

from src.data.database import get_engine
from src.utils import niveles_precio as npx


def cargar(eng, ticker, desde=None):
    q = """SELECT fecha, open, high, low, close FROM precios_nominales
           WHERE ticker = :t {f} ORDER BY fecha"""
    p = {"t": ticker}
    if desde:
        q = q.format(f="AND fecha >= :d")
        p["d"] = desde
    else:
        q = q.format(f="")
    with eng.connect() as c:
        df = pd.DataFrame(c.execute(text(q), p).fetchall(),
                          columns=["fecha", "open", "high", "low", "close"])
    for k in ("open", "high", "low", "close"):
        df[k] = pd.to_numeric(df[k])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("--desde", help="AAAA-MM-DD; por defecto toda la historia")
    ap.add_argument("--ventana", type=int, default=npx.VENTANA, help="N del swing")
    ap.add_argument("--tol", type=float, default=npx.TOL * 100,
                    help="media banda en %%; 1.0 = banda de 2%%")
    ap.add_argument("--sep", type=int, default=npx.SEP)
    ap.add_argument("--min-toques", type=int, default=npx.MIN_TOQUES)
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--nulo", type=int, default=0, metavar="N",
                    help="N permutaciones para el piso de azar")
    a = ap.parse_args()

    tol = a.tol / 100.0
    eng = get_engine()
    d = cargar(eng, a.ticker.upper(), a.desde)
    if d.empty:
        print(f"Sin datos de {a.ticker} en precios_nominales. "
              f"Corre cargar_precios_nominales.py")
        return 1

    h, lo, c = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    print(f"{a.ticker.upper()}: {len(d)} ruedas {d.fecha.min().date()} -> "
          f"{d.fecha.max().date()} | precio {lo.min():.2f} - {h.max():.2f}")
    g = npx.giros_confirmados(h, lo, a.ventana)
    print(f"giros confirmados (N={a.ventana}): {len(g)} "
          f"({sum(1 for x in g if x.tipo == npx.H)} H / "
          f"{sum(1 for x in g if x.tipo == npx.L)} L)")
    print(f"banda +-{a.tol:.2f}% | separacion {a.sep} ruedas | "
          f"minimo {a.min_toques} toques\n")

    niveles = npx.detectar(h, lo, d.fecha.to_list(), n=a.ventana, tol=tol,
                           sep=a.sep, min_toques=a.min_toques, max_niveles=a.max)
    if not niveles:
        print("Sin niveles con esos parametros.")
        return 0

    piso = None
    if a.nulo:
        print(f"Piso de azar: {a.nulo} permutaciones...", flush=True)
        best_t, best_a = [], []
        for s in range(a.nulo):
            rng = np.random.default_rng([20261119, s])
            c2, h2, l2 = npx.permutar(c, h, lo, rng)
            nn = npx.detectar(h2, l2, d.fecha.to_list(), n=a.ventana, tol=tol,
                              sep=a.sep, min_toques=a.min_toques, max_niveles=a.max)
            best_t.append(max((x.toques for x in nn), default=0))
            best_a.append(max((x.anios for x in nn), default=0))
        piso = {"toques_p95": float(np.percentile(best_t, 95)),
                "toques_max": max(best_t),
                "anios_p95": float(np.percentile(best_a, 95)),
                "anios_max": max(best_a)}
        print(f"  el mejor nivel de una serie BARAJADA llega a "
              f"{piso['toques_p95']:.0f} toques (p95) / {piso['toques_max']} (max) "
              f"y {piso['anios_p95']:.0f} anios (p95) / {piso['anios_max']} (max)\n")

    ult = len(d) - 1
    print(f"{'nivel':>9s} {'banda':>19s} {'toq':>4s} {'H':>3s} {'L':>3s} "
          f"{'anios':>6s} {'polaridad':<12s} {'primero':>11s} {'ultimo':>11s} {'hace':>6s}")
    for nv in niveles:
        marca = ""
        if piso and (nv.toques > piso["toques_max"] or nv.anios > piso["anios_max"]):
            marca = "  <-- sobre el azar"
        print(f"{nv.precio:9.2f} {nv.val:8.2f}-{nv.vah:<10.2f} {nv.toques:4d} "
              f"{nv.highs:3d} {nv.lows:3d} {nv.anios:6d} {nv.polaridad:<12s} "
              f"{str(d.fecha.iloc[nv.primer_i].date()):>11s} "
              f"{str(d.fecha.iloc[nv.ultimo_i].date()):>11s} "
              f"{ult - nv.ultimo_i:5d}r{marca}")

    print(f"\nprecio al {d.fecha.iloc[-1].date()}: {c[-1]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
