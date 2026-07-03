"""
pcr_predictive_validity.py -- Tarea 20 (addendum): valida si el PCR predice el precio.

ANTES de sumar features de opciones al ML, medimos si el PCR (put/call ratio)
tiene poder predictivo real sobre el retorno futuro. Test = Information Coefficient
(IC = Spearman cross-seccional por fecha) del PCR en t vs el retorno a h dias.

DOS medidas clave:
    IC (nivel)    : rankea tickers por el PCR crudo. Mezcla rasgos ESTATICOS
                    (nombres que estructuralmente tienen PCR alto).
    IC (dinamico) : rankea por PCR - baseline del propio ticker (la DESVIACION =
                    el verdadero "cambio de sentimiento"). Es la señal tradeable.

HALLAZGO (2026-07-02, 48 dias de historia, UN regimen alcista):
    IC(nivel) fuerte y consistente (+0.14 a +0.17, breadth ~90%), CONTRARIAN
    (PCR alto -> retorno alto). PERO IC(dinamico) se DERRUMBA a ~0 (incluso
    negativo). => la aparente señal es un ARTEFACTO ESTATICO de un regimen, NO
    el sentimiento institucional cumpliendose. El cambio de posicionamiento no
    predice. Es PREMATURO usar PCR como feature del ML (inyectaria un factor
    fragil de un solo regimen).

LIMITE DURO: opciones_snapshot arranca 2026-04-18 (el snapshot es nuevo, no hay
mas historia local). La historia CRECE dia a dia -> RE-CORRER este test en
~6-12 meses (2+ regimenes) para ver si el IC dinamico aparece de verdad.

Uso:
    ./venv/Scripts/python.exe scripts/ml/pcr_predictive_validity.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import create_engine

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HORIZONS = [5, 10, 20]          # dias habiles hacia adelante
VENTANAS = ["corto", "medio", "largo"]


def _engine():
    d = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip().strip('"').strip("'")
    return create_engine(
        f"postgresql+psycopg2://{d['DB_USER']}:{d['DB_PASSWORD']}@"
        f"{d.get('DB_HOST', 'localhost')}:{d.get('DB_PORT', 5432)}/{d.get('DB_NAME', 'activos_ml')}"
    )


def cargar(eng):
    pcr = pd.read_sql(
        "SELECT fecha,ticker,ventana,pcr_oi,pcr_vol FROM opciones_pcr_plazo_diario "
        "WHERE pcr_oi IS NOT NULL", eng)
    px = pd.read_sql("SELECT fecha,ticker,close FROM precios_diarios WHERE fecha >= '2026-04-01'", eng)
    cov = pd.read_sql("SELECT MIN(fecha_snapshot) d0, MAX(fecha_snapshot) d1, "
                      "COUNT(DISTINCT fecha_snapshot) n FROM opciones_snapshot", eng)
    pcr["fecha"] = pd.to_datetime(pcr["fecha"])
    px["fecha"] = pd.to_datetime(px["fecha"])
    px = px.sort_values(["ticker", "fecha"])
    for h in HORIZONS:
        px[f"fwd{h}"] = px.groupby("ticker")["close"].shift(-h) / px["close"] - 1
    m = pcr.merge(px[["ticker", "fecha"] + [f"fwd{h}" for h in HORIZONS]],
                  on=["ticker", "fecha"], how="left")
    # señal dinamica: desviacion del PCR respecto al baseline del ticker/ventana
    m["pcr_dev"] = m["pcr_oi"] - m.groupby(["ticker", "ventana"])["pcr_oi"].transform("mean")
    return m, cov.iloc[0]


def ic_por_fecha(sub, sig, ret):
    ics = []
    for _, g in sub.groupby("fecha"):
        gg = g[[sig, ret]].dropna()
        if len(gg) >= 20 and gg[sig].nunique() > 3:
            ic = spearmanr(gg[sig], gg[ret])[0]
            if np.isfinite(ic):
                ics.append(ic)
    return np.array(ics)


def quintil_spread(sub, sig, ret):
    sp = []
    for _, g in sub.groupby("fecha"):
        gg = g[[sig, ret]].dropna()
        if len(gg) < 20 or gg[sig].nunique() <= 4:
            continue
        q = gg[ret].groupby(pd.qcut(gg[sig], 5, duplicates="drop", labels=False)).mean()
        if 4 in q.index and 0 in q.index:
            sp.append(q[4] - q[0])
    return np.array(sp)


def main():
    eng = _engine()
    m, cov = cargar(eng)
    print("\n" + "=" * 72)
    print("  VALIDEZ PREDICTIVA DEL PCR (Tarea 20 addendum)")
    print("=" * 72)
    print(f"  opciones_snapshot crudo: {cov['d0']} -> {cov['d1']} ({cov['n']} dias). "
          f"UN regimen -> potencia limitada.")
    print("=" * 72)
    print("  IC(nivel) = rank PCR crudo | IC(dinamico) = rank desviacion vs baseline")
    print("  IC>0 = PCR alto -> retorno alto (CONTRARIAN)\n")
    print(f"  {'ventana':7s} {'señal':7s} {'horiz':>5s} {'IC_nivel':>9s} {'t':>6s} "
          f"{'breadth':>8s} {'IC_dinam':>9s} {'Q5-Q1%':>8s} {'n':>4s}")
    for vent in VENTANAS:
        for sig in ["pcr_oi", "pcr_vol"]:
            for h in HORIZONS:
                ret = f"fwd{h}"
                sub = m[m["ventana"] == vent]
                ic = ic_por_fecha(sub, sig, ret)
                if len(ic) < 3:
                    continue
                t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
                breadth = 100 * np.mean(ic > 0)
                icd = ic_por_fecha(sub, "pcr_dev", ret) if sig == "pcr_oi" else np.array([np.nan])
                sp = quintil_spread(sub, sig, ret)
                print(f"  {vent:7s} {sig:7s} {h:4d}d {ic.mean():+9.3f} {t:6.2f} "
                      f"{breadth:7.0f}% {np.nanmean(icd):+9.3f} {100*np.mean(sp):+8.2f} {len(ic):4d}")
    print("\n  VEREDICTO: IC(nivel) fuerte pero es rasgo ESTATICO de 1 regimen; IC(dinamico)")
    print("  ~0 -> el cambio de sentimiento no predice (aun). PREMATURO como feature ML.")
    print("  RE-CORRER con 2+ regimenes de historia (crece dia a dia).")
    print("=" * 72)


if __name__ == "__main__":
    main()
