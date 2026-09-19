"""
medir_valor_velas.py -- Tarea 23: que marcan los patrones de vela de
features_precio_accion, que marcarian con la definicion clasica de
src/indicators/velas.py, y si alguno anticipa retorno.

Reproduce docs/estructura_velas.md sec. 5. Solo lee la DB LOCAL (~1 min).

Secciones:
  1  Frecuencia de cada patron: tabla vieja vs modulo nuevo.
  2  Precision de la tabla vieja: envolventes que envuelven, martillos con forma
     clasica y contexto.
  3  Coincidencia: de lo que marca la tabla vieja, cuanto marca el modulo nuevo.
  4  Retorno forward en exceso sobre el universo del dia (media por dia, IC95),
     vieja vs nueva.

Resultado de referencia (17/9/2026, 148.132 velas desde 2023-06):
  envolvente alcista vieja 12,69% de las velas, 27,4% envuelve de verdad; ningun
  patron, viejo ni clasico, con exceso distinto de cero a 5 o 20 ruedas.

Uso (desde la raiz):
    python scripts/ml/medir_valor_velas.py
    python scripts/ml/medir_valor_velas.py --desde 2021-06-01
"""

import argparse
import os
import sys
import warnings

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, ROOT)

# LOCAL-only: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.database import query_df  # noqa: E402
from src.indicators import velas  # noqa: E402

VIEJAS = ["patron_doji", "patron_hammer", "patron_shooting_star", "patron_marubozu",
          "patron_engulfing_bull", "patron_engulfing_bear", "inside_bar", "outside_bar"]


def exceso(df, mask, h):
    por_dia = df.loc[mask].groupby("fecha")[h].mean().dropna()
    if len(por_dia) < 20:
        return "pocos dias"
    m = por_dia.mean()
    se = por_dia.std(ddof=1) / np.sqrt(len(por_dia))
    return f"{m*100:+.2f}% [{(m-1.96*se)*100:+.2f};{(m+1.96*se)*100:+.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--desde", default="2023-06-01")
    args = ap.parse_args()

    precios = query_df("""
        SELECT ticker, fecha, open, high, low, close FROM precios_diarios
        WHERE close > 0 AND high > 0 AND low > 0 AND open > 0 ORDER BY ticker, fecha
    """)
    precios["fecha"] = pd.to_datetime(precios["fecha"])
    nuevas = pd.concat([velas.calcular_velas(g) for _, g in precios.groupby("ticker", sort=True)],
                       ignore_index=True)
    viejas = query_df(f"""
        SELECT ticker, fecha, {", ".join(VIEJAS)}, upper_shadow_pct, lower_shadow_pct, pos_rango_20d
        FROM features_precio_accion
    """)
    viejas["fecha"] = pd.to_datetime(viejas["fecha"])

    d = precios.merge(nuevas, on=["ticker", "fecha"])
    d = d.merge(viejas, on=["ticker", "fecha"], how="inner", suffixes=("", "_vieja"))
    d = d.sort_values(["ticker", "fecha"])
    g = d.groupby("ticker")
    d["o1"], d["c1"] = g["open"].shift(1), g["close"].shift(1)
    for h in (5, 20):
        d[f"f{h}"] = g["close"].shift(-h) / d["close"] - 1
    d = d[d.fecha >= args.desde].copy()
    for h in (5, 20):
        d[f"f{h}_ex"] = d[f"f{h}"] - d.groupby("fecha")[f"f{h}"].transform("mean")
    print(f"velas con fila en features_precio_accion desde {args.desde}: {len(d):,} | "
          f"tickers {d.ticker.nunique()}")

    def vieja(col):
        return d[col + "_vieja"] if col + "_vieja" in d.columns else d[col]

    print("\n1) FRECUENCIA (% de velas)")
    print("   tabla vieja:")
    for c in VIEJAS:
        print(f"     {c:26s} {vieja(c).mean():6.2%}")
    print("   modulo nuevo (velas.py):")
    for c in velas.COLUMNAS:
        print(f"     {c:26s} {d[c].mean():6.2%}")

    print("\n2) PRECISION DE LA TABLA VIEJA")
    eb = vieja("patron_engulfing_bull") == 1
    env = ((d.close > d.open) & (d.c1 < d.o1) & (d.open <= d.c1) & (d.close >= d.o1)
           & ((d.open < d.c1) | (d.close > d.o1)))
    print(f"   envolvente alcista: {int(eb.sum()):,} marcadas, envuelve {(eb & env).sum() / eb.sum():.1%}, "
          f"abre por encima del cierre previo {(eb & (d.open > d.c1)).sum() / eb.sum():.1%}")
    hm = vieja("patron_hammer") == 1
    print(f"   martillo: {int(hm.sum()):,} marcados, sombra superior <= 10% "
          f"{(hm & (d.upper_shadow_pct <= 0.10)).sum() / hm.sum():.1%}, en el tercio superior del "
          f"rango 20d {(hm & (d.pos_rango_20d > 0.66)).sum() / hm.sum():.1%}")

    print("\n3) COINCIDENCIA: de lo que marca la vieja, % que marca la nueva")
    pares = [("patron_engulfing_bull", "patron_engulfing_bull"),
             ("patron_engulfing_bear", "patron_engulfing_bear"),
             ("patron_hammer", "patron_hammer"), ("patron_hammer", "patron_hanging_man"),
             ("patron_shooting_star", "patron_shooting_star"),
             ("patron_shooting_star", "patron_inverted_hammer"),
             ("patron_doji", "patron_doji")]
    for v, n in pares:
        m = vieja(v) == 1
        print(f"   vieja {v:24s} -> nueva {n:24s} {(m & (d[n] == 1)).sum() / max(m.sum(), 1):6.1%}")

    print("\n4) RETORNO FORWARD EN EXCESO vs universo del dia")
    for c in ("patron_engulfing_bull", "patron_engulfing_bear", "patron_hammer",
              "patron_shooting_star", "patron_doji"):
        m = vieja(c) == 1
        print(f"   VIEJA {c:26s} n={int(m.sum()):7,}  5r {exceso(d, m, 'f5_ex')}  20r {exceso(d, m, 'f20_ex')}")
    for c in velas.COLUMNAS:
        m = d[c] == 1
        print(f"   NUEVA {c:26s} n={int(m.sum()):7,}  5r {exceso(d, m, 'f5_ex')}  20r {exceso(d, m, 'f20_ex')}")


if __name__ == "__main__":
    main()
