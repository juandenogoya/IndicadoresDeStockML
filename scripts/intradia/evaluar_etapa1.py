"""
evaluar_etapa1.py -- compuerta de la etapa 1 del snapshot intradiario (LOCAL, solo lee).

Corre sobre lo que el sync bajo a local (precios_intradia_30m) y lo cruza con
precios_diarios. Responde si la CAPTURA es confiable antes de construir alertas
encima (etapa 2). Diseno: docs/intradia.md.

REGLA DE LECTURA (pre-registrada el 30/10/2026, antes de la primera corrida):
la etapa 1 PASA si se cumplen las cuatro:
    1. cobertura >= 90% del universo en TODOS los cortes capturados;
    2. cero cortes faltantes entre la primera y la ultima rueda del periodo
       (si falta alguno, se lista: la explicacion -- ej. la VM estuvo caida --
       se escribe, pero el criterio no se relaja);
    3. demora (capturado_en - hora del corte) mediana <= 10 min y maxima <= 30;
    4. high/low acumulado del corte fuera del rango diario de precios_diarios
       en <= 1% de los ticker-cortes (tolerancia 0,1%): un corte no puede tener
       un maximo mayor que el maximo del dia.

INFORMATIVO, sin umbral (no hay base para fijarlo antes de medir):
    - revisiones de Yahoo: la misma barra vista por apertura_90 y por cierre_60;
    - open de la primera barra vs open diario (subasta de apertura);
    - volumen acumulado al corte / volumen diario. A las 15:00 ET deberia ser
      < 1; un ratio > 1,02 es incoherente y se cuenta.

Uso:
    python scripts/intradia/evaluar_etapa1.py
    python scripts/intradia/evaluar_etapa1.py --desde 2026-11-02 --hasta 2026-11-20

Salida: 0 pasa / 2 no pasa / 1 error o sin datos.
"""

import argparse
import os
import sys
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))
os.environ.pop("DATABASE_URL", None)   # LOCAL siempre (CLAUDE.md, forzar target local)

import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine
from src.utils import sesion_nyse
from src.utils.intradia import TABLA
from src.utils.trading_calendar import trading_days_between

COBERTURA_MIN = 0.90
DEMORA_MEDIANA_MAX = 10.0
DEMORA_MAX = 30.0
TOL_RANGO = 0.001
VIOLACIONES_MAX = 0.01

SQL_AGG = f"""
WITH agg AS (
    SELECT ticker, fecha, corte,
           MAX(hora_corte)   AS hora_corte,
           MAX(capturado_en) AS capturado_en,
           MAX(high)         AS high,
           MIN(low)          AS low,
           SUM(volumen)      AS vol,
           (ARRAY_AGG(open ORDER BY inicio_barra))[1] AS open_1,
           MAX(inicio_barra) AS ultima,
           COUNT(*)          AS n
    FROM {TABLA}
    WHERE fecha BETWEEN :d AND :h
    GROUP BY ticker, fecha, corte
)
SELECT a.*, p.open AS p_open, p.high AS p_high, p.low AS p_low, p.volume AS p_vol
FROM agg a
LEFT JOIN precios_diarios p ON p.ticker = a.ticker AND p.fecha = a.fecha
"""

SQL_REVISIONES = f"""
SELECT a.close AS ca, b.close AS cb, a.volumen AS va, b.volumen AS vb
FROM {TABLA} a
JOIN {TABLA} b ON b.ticker = a.ticker AND b.inicio_barra = a.inicio_barra
WHERE a.corte = 'apertura_90' AND b.corte = 'cierre_60'
  AND a.fecha BETWEEN :d AND :h
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde")
    ap.add_argument("--hasta")
    args = ap.parse_args()

    eng = get_engine()
    with eng.connect() as c:
        lim = c.execute(text(f"SELECT MIN(fecha), MAX(fecha) FROM {TABLA}")).fetchone()
        n_universo = c.execute(text("SELECT COUNT(*) FROM activos WHERE activo")).scalar()
    if lim[0] is None:
        print(f"  {TABLA} vacia en local: correr el sync primero.")
        return 1
    desde = date.fromisoformat(args.desde) if args.desde else lim[0]
    hasta = date.fromisoformat(args.hasta) if args.hasta else lim[1]

    df = pd.read_sql(text(SQL_AGG), eng, params={"d": desde, "h": hasta})
    rev = pd.read_sql(text(SQL_REVISIONES), eng, params={"d": desde, "h": hasta})
    if df.empty:
        print(f"  Sin capturas entre {desde} y {hasta}.")
        return 1
    for col in ("high", "low", "vol", "open_1", "p_open", "p_high", "p_low", "p_vol"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["hora_corte"] = pd.to_datetime(df["hora_corte"], utc=True)
    df["ultima"] = pd.to_datetime(df["ultima"], utc=True)
    df["capturado_en"] = pd.to_datetime(df["capturado_en"], utc=True)
    df["completo"] = df["ultima"] == df["hora_corte"] - pd.Timedelta(minutes=sesion_nyse.BARRA_MIN)

    print("=" * 70)
    print(f"  EVALUACION ETAPA 1 -- snapshot intradiario  |  {desde} -> {hasta}")
    print(f"  Universo activo: {n_universo} tickers")
    print("=" * 70)
    criterios = {}

    # 1. Cobertura por corte
    por_corte = (df.groupby(["fecha", "corte"])
                   .agg(completos=("completo", "sum"),
                        demora=("capturado_en", "max"),
                        hora=("hora_corte", "max"))
                   .reset_index())
    por_corte["pct"] = por_corte["completos"] / n_universo
    por_corte["demora_min"] = (por_corte["demora"] - por_corte["hora"]).dt.total_seconds() / 60.0
    bajos = por_corte[por_corte["pct"] < COBERTURA_MIN]
    print(f"\n  1. Cobertura (>= {COBERTURA_MIN:.0%} en todos los cortes)")
    print(f"     cortes capturados: {len(por_corte)} | minima {por_corte['pct'].min():.1%} "
          f"| mediana {por_corte['pct'].median():.1%}")
    for _, r in bajos.iterrows():
        print(f"     [BAJA] {r.fecha} {r.corte}: {int(r.completos)}/{n_universo} ({r.pct:.1%})")
    criterios["cobertura"] = bajos.empty

    # 2. Cortes faltantes
    esperados = {(d, c) for d in trading_days_between(desde, hasta)
                 for c in sesion_nyse.cortes(d)}
    presentes = set(zip(por_corte["fecha"], por_corte["corte"]))
    faltantes = sorted(esperados - presentes)
    print(f"\n  2. Cortes faltantes (cero)")
    print(f"     esperados {len(esperados)} | presentes {len(presentes & esperados)} "
          f"| faltan {len(faltantes)}")
    for d, c in faltantes:
        print(f"     [FALTA] {d} {c}")
    criterios["faltantes"] = not faltantes

    # 3. Demora
    med, mx = por_corte["demora_min"].median(), por_corte["demora_min"].max()
    print(f"\n  3. Demora (mediana <= {DEMORA_MEDIANA_MAX:.0f} min, maxima <= {DEMORA_MAX:.0f})")
    print(f"     mediana {med:.1f} min | maxima {mx:.1f} min")
    criterios["demora"] = med <= DEMORA_MEDIANA_MAX and mx <= DEMORA_MAX

    # 4. Rango contra el diario
    con_diario = df.dropna(subset=["p_high", "p_low"])
    viol = con_diario[(con_diario["high"] > con_diario["p_high"] * (1 + TOL_RANGO)) |
                      (con_diario["low"] < con_diario["p_low"] * (1 - TOL_RANGO))]
    frac = len(viol) / len(con_diario) if len(con_diario) else float("nan")
    print(f"\n  4. High/low del corte dentro del rango diario (violaciones <= {VIOLACIONES_MAX:.0%})")
    print(f"     ticker-cortes con diario: {len(con_diario)} de {len(df)} | "
          f"violaciones {len(viol)} ({frac:.2%})")
    for _, r in viol.head(10).iterrows():
        print(f"     {r.ticker:<6} {r.fecha} {r.corte}: h={r.high} (dia {r.p_high}) "
              f"l={r.low} (dia {r.p_low})")
    criterios["rango"] = bool(len(con_diario)) and frac <= VIOLACIONES_MAX
    if not len(con_diario):
        print("     sin precios_diarios para estas ruedas: correr el recovery y repetir.")

    # Informativo
    print("\n  INFORMATIVO (sin umbral)")
    if not rev.empty:
        for col in ("ca", "cb", "va", "vb"):
            rev[col] = pd.to_numeric(rev[col], errors="coerce")
        cambio_c = (rev["ca"] - rev["cb"]).abs() > 1e-9
        cambio_v = (rev["va"] - rev["vb"]).abs() > 0
        print(f"     revisiones Yahoo (misma barra, apertura vs cierre): {len(rev)} barras | "
              f"close cambia {cambio_c.mean():.1%} | volumen cambia {cambio_v.mean():.1%}")
    else:
        print("     revisiones Yahoo: sin pares apertura/cierre todavia.")
    if len(con_diario):
        dif_open = (con_diario["open_1"] / con_diario["p_open"] - 1).abs() * 100
        print(f"     |open 1a barra / open diario - 1|: p50 {dif_open.median():.3f}% "
              f"| p95 {dif_open.quantile(.95):.3f}%")
        for corte in sesion_nyse.CORTES:
            sub = con_diario[con_diario["corte"] == corte]
            if sub.empty:
                continue
            ratio = sub["vol"] / sub["p_vol"]
            print(f"     volumen al corte / diario ({corte}): p5 {ratio.quantile(.05):.2f} "
                  f"| p50 {ratio.median():.2f} | p95 {ratio.quantile(.95):.2f} "
                  f"| > 1,02: {(ratio > 1.02).sum()}")

    pasa = all(criterios.values())
    print("\n" + "=" * 70)
    for k, v in criterios.items():
        print(f"  {k:<10} {'PASA' if v else 'NO PASA'}")
    print(f"  ETAPA 1: {'PASA' if pasa else 'NO PASA'}")
    print("=" * 70)
    return 0 if pasa else 2


if __name__ == "__main__":
    sys.exit(main())
