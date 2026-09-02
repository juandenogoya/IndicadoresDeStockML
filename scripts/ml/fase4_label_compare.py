"""
fase4_label_compare.py -- Fase 4 de la Tarea 20.

Compara dos definiciones de LABEL en el MISMO walk-forward purgado:
    absoluto : label_binario actual (retorno_20d > +1%).  [regime-dependent]
    relativo : el ticker BATE la mediana de su sector esa fecha (cross-seccional).
               Base ~50% por construccion -> regime-neutral.

Ambos entrenan un RF-global identico (construir_modelo('rf')) sobre los mismos
folds y se evaluan con el MISMO yardstick economico en el holdout OOS:
    - top-decile mean retorno_20d (absoluto)
    - excess vs MERCADO   (top ret - media del fold)  <- PnL-relevante
    - excess vs SECTOR    (top (ret - mediana sector) medio)
    - beat-sector rate     (P(ret > mediana sector | decil alto), base 0.50)
    - consistencia across folds + t-stat (fold como unidad independiente)

La pregunta: entrenar RELATIVO da un decil alto con excess mas alto y/o mas
consistente que entrenar ABSOLUTO? Si no mueve la aguja -> 0.60 es el techo de
estas features y se congela el absoluto calibrado.

Uso:
    PYTHONPATH=. python scripts/ml/fase4_label_compare.py
"""

import os
import sys
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.ml.trainer import construir_modelo, TARGET_COL
from scripts.ml.walkforward_ml import cargar_dataset, FEATURE_COLS_V3, OUT_DIR


def tstat(x):
    x = np.asarray(x, dtype=float)
    s = x.std(ddof=1)
    return x.mean() / (s / np.sqrt(len(x))) if s > 0 else np.nan


def eval_topdecil(prob, sub):
    """Metricas economicas del decil alto de `prob` sobre el holdout `sub`."""
    n = len(sub); k = max(1, n // 10)
    top = sub.iloc[np.argsort(-prob)[:k]]
    ret_top = top["retorno_20d"].mean()
    excess_mkt = ret_top - sub["retorno_20d"].mean()
    excess_sec = (top["retorno_20d"] - top["ret_sector_med"]).mean()
    beat_sec = (top["retorno_20d"] > top["ret_sector_med"]).mean()
    return dict(ret_top=ret_top, excess_mkt=excess_mkt,
                excess_sec=excess_sec, beat_sec=beat_sec)


def correr(holdout_len=126, embargo=20, start_date="2021-06-01"):
    print("\n" + "=" * 72)
    print("  FASE 4 -- COMPARACION DE LABEL (absoluto vs relativo-al-sector)")
    print("=" * 72)
    df = cargar_dataset()
    # Mediana de retorno_20d por (sector, fecha) -> label relativo cross-seccional
    df["ret_sector_med"] = df.groupby(["sector", "fecha"])["retorno_20d"].transform("median")
    df["label_abs"] = df[TARGET_COL].astype(int)
    df["label_rel"] = (df["retorno_20d"] > df["ret_sector_med"]).astype(int)
    print(f"  Dataset: {len(df):,} filas | base abs={df['label_abs'].mean():.3f} "
          f"rel={df['label_rel'].mean():.3f}")

    fechas = np.sort(df["fecha"].unique())
    idx0 = int(np.searchsorted(fechas, pd.Timestamp(start_date)))
    folds = []
    h = idx0
    while h + holdout_len <= len(fechas):
        if h - embargo > 30:
            folds.append((h, min(h + holdout_len, len(fechas))))
        h += holdout_len
    print(f"  Folds: {len(folds)} | holdout={holdout_len} embargo={embargo} start={start_date}")
    print("=" * 72)

    rows = []
    for fi, (h, h_end) in enumerate(folds, 1):
        ho_ini, ho_fin = fechas[h], fechas[h_end - 1]
        train_end = fechas[h - embargo - 1]
        tr = df[df["fecha"] <= train_end]
        ho = df[(df["fecha"] >= ho_ini) & (df["fecha"] <= ho_fin)].reset_index(drop=True)
        if len(ho) == 0:
            continue
        Xtr, Xho = tr[FEATURE_COLS_V3], ho[FEATURE_COLS_V3]

        t0 = datetime.now()
        res = {}
        for etiqueta, ycol in [("ABS", "label_abs"), ("REL", "label_rel")]:
            y = tr[ycol].astype(int).values
            if len(np.unique(y)) < 2:
                continue
            m = construir_modelo("rf", len(y))
            m.fit(Xtr, y)
            prob = m.predict_proba(Xho)[:, 1]
            res[etiqueta] = eval_topdecil(prob, ho)
        secs = (datetime.now() - t0).seconds

        for etiqueta, r in res.items():
            rows.append(dict(fold=fi, entren=etiqueta,
                             ho_ini=str(ho_ini.astype("datetime64[D]")), **r))
        a, b = res.get("ABS", {}), res.get("REL", {})
        print(f"  Fold {fi:2d} [{str(ho_ini.astype('datetime64[D]'))}] ({secs}s)  "
              f"excess_mkt  ABS={a.get('excess_mkt', float('nan')):+.2f}  "
              f"REL={b.get('excess_mkt', float('nan')):+.2f}   "
              f"beat_sec  ABS={a.get('beat_sec', float('nan')):.2f}  "
              f"REL={b.get('beat_sec', float('nan')):.2f}")

    res = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("  RESUMEN (media across folds; t-stat con fold como unidad)")
    print("=" * 72)
    print(f"  {'entren':7s} {'ret_top':>8s} {'excess_mkt':>11s} {'t':>6s} {'pos':>5s} "
          f"{'excess_sec':>11s} {'beat_sec':>9s} {'t_beat':>7s}")
    for etq in ["ABS", "REL"]:
        s = res[res["entren"] == etq]
        if s.empty:
            continue
        t_mkt = tstat(s["excess_mkt"])
        t_beat = tstat(s["beat_sec"] - 0.5)
        print(f"  {etq:7s} {s['ret_top'].mean():8.2f} {s['excess_mkt'].mean():11.2f} "
              f"{t_mkt:6.2f} {int((s['excess_mkt']>0).sum()):3d}/{len(s):<2d} "
              f"{s['excess_sec'].mean():11.2f} {s['beat_sec'].mean():9.3f} {t_beat:7.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, "fase4_label_compare.csv"), index=False)
    print("\n  Guardado: reportes/ml_walkforward/fase4_label_compare.csv")
    print("=" * 72)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=126)
    ap.add_argument("--embargo", type=int, default=20)
    ap.add_argument("--start", default="2021-06-01")
    a = ap.parse_args()
    correr(a.holdout, a.embargo, a.start)


if __name__ == "__main__":
    main()
