"""
explorar_perfil_precio.py -- que sale del perfil de precio en el universo. SOLO LEE.

Corre src/utils/perfil_precio sobre los tickers activos en las ventanas de 21, 52 y
200 ruedas y muestra POC, area de valor y donde esta el precio. No escribe nada: no
hay tabla, no hay paso de rutina y no hay consumidor. Existe para mirar los numeros
ANTES de decidir si esto merece persistirse.

COMO SE LEE LA SALIDA

  El numero que manda es `ancho VA`, el ancho del area de valor. Es el error bar del
  POC: con un area angosta hubo equilibrio y el POC es un precio con significado; con
  un area ancha el perfil es plano y el POC es el maximo del ruido. Un POC suelto, sin
  su ancho al lado, engana -- por eso las dos columnas van juntas en todas las vistas.

  `pico` es la porcion del volumen de la ventana que se lleva la franja del POC. Dice
  si el maximo es un maximo o si el perfil esta repartido en partes iguales.

  `ubicacion` se DERIVA del cierre de hoy en cada lectura, no se guarda: el POC no es
  soporte ni resistencia por si mismo, lo es respecto del precio actual.

LO QUE ESTA SALIDA NO DICE
  Nada sobre que hace el precio cuando vuelve a estas zonas. La tasa de rebote en
  soportes y resistencias se midio el 6/9/2026 y no predice (85% dentro de muestra,
  56% fuera; memory/zonas_precio_descartado.md). Esto es descriptivo.

Uso:
    python scripts/manual/explorar_perfil_precio.py                  resumen del universo
    python scripts/manual/explorar_perfil_precio.py --detalle        tabla ticker por ticker
    python scripts/manual/explorar_perfil_precio.py --tickers JPM,F  solo esos
    python scripts/manual/explorar_perfil_precio.py --perfil JPM     el histograma de uno
    python scripts/manual/explorar_perfil_precio.py --csv            deja el detalle en reportes/

Salida: 0 = corrio | 1 = error.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.pop("DATABASE_URL", None)          # LOCAL manda (Plan C)

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine
from src.data.universo import get_universo
from src.utils import perfil_precio as pp

SQL = """
SELECT ticker, fecha, high, low, close, volume
FROM (
  SELECT ticker, fecha, high, low, close, volume,
         ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
  FROM precios_diarios
  WHERE ticker = ANY(:tk)
) q
WHERE rn <= :maxv
ORDER BY ticker, fecha
"""


def cargar(tickers, maxv):
    with get_engine().connect() as cx:
        df = pd.read_sql(text(SQL), cx, params={"tk": list(tickers), "maxv": int(maxv)})
    for c in ("high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def medir(df, ventanas):
    filas = []
    for tk, d in df.groupby("ticker", sort=True):
        d = d.sort_values("fecha")
        cierre = float(d["close"].iloc[-1])
        for _, r in pp.calcular_ventanas(d, ventanas).iterrows():
            perfil = pp.Perfil(**{c: r[c] for c in pp.CAMPOS})
            filas.append({
                "ticker": tk, "fecha": d["fecha"].iloc[-1], "cierre": cierre,
                "ventana": int(r["ventana"]), "poc": r["poc"], "val": r["val"],
                "vah": r["vah"], "ancho_va_pct": r["ancho_va_pct"],
                "pico_pct": r["pico_pct"],
                "dist_poc_pct": (cierre / r["poc"] - 1) * 100 if np.isfinite(r["poc"]) else np.nan,
                "ubicacion": pp.ubicacion(cierre, perfil),
            })
    return pd.DataFrame(filas)


def resumen(out, ventanas):
    print("\n=== RESUMEN DEL UNIVERSO ===\n")
    print(f"{'vent':>6}{'tickers':>9}{'ancho VA mediana':>19}{'p10':>8}{'p90':>8}"
          f"{'pico':>9}{'angostos':>11}")
    for v in ventanas:
        s = out[(out["ventana"] == v) & out["poc"].notna()]
        if s.empty:
            print(f"{v:>6}{0:>9}{'sin datos':>19}")
            continue
        a = s["ancho_va_pct"]
        print(f"{v:>6}{len(s):>9}{a.median():>18.1f}%{a.quantile(.10):>7.1f}%"
              f"{a.quantile(.90):>7.1f}%{s['pico_pct'].median():>8.1f}%"
              f"{(a < 10).mean() * 100:>10.0f}%")
    print("\n  'angostos' = area de valor de menos de 10% de ancho (el POC significa algo)")

    print("\n=== DONDE ESTA EL PRECIO RESPECTO DEL AREA DE VALOR ===\n")
    print(f"{'vent':>6}{'dentro':>9}{'arriba':>9}{'abajo':>8}{'|dist al POC| mediana':>24}")
    for v in ventanas:
        s = out[(out["ventana"] == v) & out["poc"].notna()]
        if s.empty:
            continue
        u = s["ubicacion"].value_counts()
        print(f"{v:>6}{u.get('dentro del area de valor', 0):>9}"
              f"{u.get('arriba del area de valor', 0):>9}"
              f"{u.get('abajo del area de valor', 0):>8}"
              f"{s['dist_poc_pct'].abs().median():>23.1f}%")

    print("\n=== LOS 10 CON EL AREA MAS ANGOSTA (ventana 52) ===\n")
    s = out[(out["ventana"] == 52) & out["poc"].notna()].nsmallest(10, "ancho_va_pct")
    print(f"{'tick':6}{'cierre':>10}{'VAL':>10}{'POC':>10}{'VAH':>10}{'ancho':>8}"
          f"{'dist POC':>10}  ubicacion")
    for _, r in s.iterrows():
        print(f"{r['ticker']:6}{r['cierre']:>10.2f}{r['val']:>10.2f}{r['poc']:>10.2f}"
              f"{r['vah']:>10.2f}{r['ancho_va_pct']:>7.1f}%{r['dist_poc_pct']:>9.1f}%"
              f"  {r['ubicacion']}")


def detalle(out):
    print("\n=== DETALLE POR TICKER ===\n")
    print(f"{'tick':6}{'cierre':>10}{'vent':>6}{'VAL':>10}{'POC':>10}{'VAH':>10}"
          f"{'ancho':>8}{'dist POC':>10}  ubicacion")
    for tk, d in out.groupby("ticker", sort=True):
        for _, r in d.iterrows():
            if not np.isfinite(r["poc"]):
                print(f"{tk:6}{r['cierre']:>10.2f}{r['ventana']:>6}"
                      f"{'sin ventana completa':>48}")
                continue
            print(f"{tk:6}{r['cierre']:>10.2f}{r['ventana']:>6}{r['val']:>10.2f}"
                  f"{r['poc']:>10.2f}{r['vah']:>10.2f}{r['ancho_va_pct']:>7.1f}%"
                  f"{r['dist_poc_pct']:>9.1f}%  {r['ubicacion']}")
        print()


def histograma(df, tk, ventana, paso=0.01):
    d = df[df["ticker"] == tk].sort_values("fecha")
    if len(d) < ventana:
        print(f"{tk}: solo {len(d)} ruedas, la ventana de {ventana} no esta completa")
        return
    d = d.iloc[-ventana:]
    k0, p = pp.perfil_bins(d["high"], d["low"], d["volume"], paso)
    if k0 is None:
        print(f"{tk}: sin volumen")
        return
    r = pp.calcular(d["high"], d["low"], d["volume"], paso)
    cierre = float(d["close"].iloc[-1])
    print(f"\n=== {tk}: perfil de las ultimas {ventana} ruedas, franjas de {paso*100:g}% ===")
    print(f"    cierre {cierre:.2f}   POC {r.poc:.2f}   area de valor {r.val:.2f} - "
          f"{r.vah:.2f} ({r.ancho_va_pct:.1f}% de ancho)   {pp.ubicacion(cierre, r)}\n")
    mx = p.max()
    k_poc = int(np.argmax(p))
    for i in range(len(p) - 1, -1, -1):
        lo, hi = pp.limites_bin(k0 + i, paso)
        marca = ""
        if i == k_poc:
            marca = " <-- POC"
        elif lo <= r.val < hi:
            marca = " <-- VAL"
        elif lo < r.vah <= hi:
            marca = " <-- VAH"
        if lo <= cierre < hi:
            marca += "   [cierre]"
        print(f"  {lo:9.2f} - {hi:9.2f} |{'#' * int(round(p[i] / mx * 42)):<42}|{marca}")


def main():
    ap = argparse.ArgumentParser(description="Perfil de precio: POC y area de valor. Solo lee.")
    ap.add_argument("--tickers", help="lista separada por comas (default: universo activo)")
    ap.add_argument("--ventanas", default="21,52,200")
    ap.add_argument("--detalle", action="store_true", help="tabla ticker por ticker")
    ap.add_argument("--perfil", metavar="TICKER", help="histograma de un ticker")
    ap.add_argument("--perfil-ventana", type=int, default=200)
    ap.add_argument("--csv", action="store_true", help="deja el detalle en reportes/")
    args = ap.parse_args()

    ventanas = [int(x) for x in args.ventanas.split(",") if x.strip()]
    tickers = ([t.strip().upper() for t in args.tickers.split(",")] if args.tickers
               else sorted(get_universo()))
    maxv = max(ventanas)

    print(f"perfil de precio -- {len(tickers)} tickers, ventanas {ventanas}, "
          f"bins de {pp.PASO*100:g}%, area de valor {pp.COBERTURA*100:g}%")
    df = cargar(tickers, maxv)
    if df.empty:
        print("sin datos en precios_diarios para esos tickers")
        return 1
    faltan = set(tickers) - set(df["ticker"])
    if faltan:
        print(f"  sin datos: {', '.join(sorted(faltan))}")

    if args.perfil:
        histograma(df, args.perfil.upper(), args.perfil_ventana)
        return 0

    out = medir(df, ventanas)
    resumen(out, ventanas)
    if args.detalle:
        detalle(out)
    if args.csv:
        dest = os.path.join(ROOT, "reportes", "perfil_precio.csv")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        out.to_csv(dest, index=False)
        print(f"\n  detalle en {dest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrumpido")
        sys.exit(1)
