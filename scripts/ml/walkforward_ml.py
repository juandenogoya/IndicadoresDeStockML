"""
walkforward_ml.py -- Fase 2 de la Tarea 20 (reentrenamiento ML).

Walk-forward PURGADO sobre features_ml (rebuild 196t/2020-2026 de la Fase 1).
Objetivo (COMPUERTA 1): decidir si el RF-global calibrado supera de forma
ESTABLE a un baseline lineal, y si la calibracion arregla el bucket alto de
probabilidad. Genera predicciones OOS por fold para la Fase 3 (pesos sectoriales).

Diseno:
    - Un modelo GLOBAL (todos los sectores pooled), 53 features V3.
    - Folds por FECHA GLOBAL: holdout rodante de HOLDOUT_LEN dias habiles,
      paso = HOLDOUT_LEN, train EXPANSIVO. Arranque en START_DATE (2021-06).
    - PURGA/EMBARGO: el label es retorno_20d -> se descartan del train las
      ultimas EMBARGO barras antes del holdout (su outcome cae dentro del holdout).
    - 3 modelos en los mismos folds:
        rf      : RandomForest de produccion (construir_modelo('rf')).
        rf_cal  : el mismo RF envuelto en CalibratedClassifierCV isotonica (cv=3).
        en      : baseline lineal elastic-net logistico (control de complejidad).
    - Metricas OOS por fold: base rate, ROC-AUC, precision/lift en el decil alto,
      precision@0.65 (proxy COMPRA_FUERTE), retorno_20d medio del decil, Brier
      (calibracion). Pooled + estabilidad across folds.

NO toca produccion ni Railway. Todo local, lectura de features_ml + JOIN
features_market_structure. Salidas en reportes/ml_walkforward/.

Uso:
    PYTHONPATH=. python scripts/ml/walkforward_ml.py
    PYTHONPATH=. python scripts/ml/walkforward_ml.py --holdout 126 --embargo 20 --start 2021-06-01
"""

import os
import sys
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.data.database import query_df
from src.ml.trainer import FEATURE_COLS, TARGET_COL, feature_engineering, construir_modelo, _BOOL_COLS
from src.indicators.market_structure import FEATURE_COLS_MS

FEATURE_COLS_V3 = FEATURE_COLS + FEATURE_COLS_MS
OUT_DIR = os.path.join(ROOT, "reportes", "ml_walkforward")

# Cap del train del baseline lineal (saga) para que no se vuelva lento.
EN_MAX_TRAIN = 60000


# ------------------------------------------------------------------
# Carga
# ------------------------------------------------------------------

def cargar_dataset() -> pd.DataFrame:
    """features_ml JOIN features_market_structure, solo filas con label."""
    ms_cols = ", ".join(f"fms.{c}" for c in FEATURE_COLS_MS)
    sql = f"""
        SELECT fm.ticker, fm.sector, fm.fecha, fm.close, fm.atr14, fm.momentum,
               fm.bb_upper, fm.bb_lower, fm.retorno_20d, fm.label_binario,
               fm.rsi14, fm.macd_hist, fm.adx, fm.vol_relativo,
               fm.dist_sma21, fm.dist_sma50, fm.dist_sma200,
               fm.score_ponderado, fm.condiciones_ok,
               fm.cond_rsi, fm.cond_macd, fm.cond_sma21, fm.cond_sma50,
               fm.cond_sma200, fm.cond_momentum,
               fm.z_rsi_sector, fm.z_retorno_1d_sector, fm.z_retorno_5d_sector,
               fm.z_vol_sector, fm.z_dist_sma50_sector, fm.z_adx_sector,
               fm.pct_long_sector, fm.rank_retorno_sector,
               fm.rsi_sector_avg, fm.adx_sector_avg, fm.retorno_1d_sector_avg,
               {ms_cols}
        FROM features_ml fm
        JOIN features_market_structure fms
          ON fm.ticker = fms.ticker AND fm.fecha = fms.fecha
        WHERE fm.label_binario IS NOT NULL
        ORDER BY fm.fecha, fm.ticker
    """
    df = query_df(sql)
    df["fecha"] = pd.to_datetime(df["fecha"])
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df = feature_engineering(df)
    # MS sin pivot confirmado -> 0 (igual que trainer_v3.preparar_xy_v3)
    ms_present = [c for c in FEATURE_COLS_MS if c in df.columns]
    df[ms_present] = df[ms_present].fillna(0)
    return df


# ------------------------------------------------------------------
# Modelos
# ------------------------------------------------------------------

def modelo_elasticnet() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            C=1.0, max_iter=200, tol=1e-3, n_jobs=-1, random_state=42,
        )),
    ])


# ------------------------------------------------------------------
# Metricas
# ------------------------------------------------------------------

def metricas_oos(prob, y, ret, base, decil=0.10, thr=0.65):
    n = len(y)
    k = max(1, int(n * decil))
    order = np.argsort(-prob)
    top = order[:k]
    prec_dec = float(y[top].mean())
    lift_dec = prec_dec / base if base > 0 else np.nan
    ret_dec = float(ret[top].mean())
    mask = prob >= thr
    prec_thr = float(y[mask].mean()) if mask.sum() > 0 else np.nan
    n_thr = int(mask.sum())
    try:
        auc = float(roc_auc_score(y, prob))
    except Exception:
        auc = np.nan
    brier = float(brier_score_loss(y, prob))
    return dict(auc=auc, prec_decil=prec_dec, lift_decil=lift_dec,
                ret_decil=ret_dec, prec_065=prec_thr, n_065=n_thr, brier=brier)


def tabla_calibracion(prob, y, bins=10):
    """Rate observado por bin de probabilidad (reliability)."""
    df = pd.DataFrame({"p": prob, "y": y})
    df["bin"] = np.clip((df["p"] * bins).astype(int), 0, bins - 1)
    g = df.groupby("bin").agg(p_med=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return g


# ------------------------------------------------------------------
# Walk-forward
# ------------------------------------------------------------------

def correr(holdout_len, embargo, start_date, verbose=True):
    print("\n" + "=" * 70)
    print("  FASE 2 -- WALK-FORWARD PURGADO (RF global + rf_cal + elastic-net)")
    print("=" * 70)
    df = cargar_dataset()
    print(f"  Dataset: {len(df):,} filas, {df['ticker'].nunique()} tickers, "
          f"{df['fecha'].min().date()} -> {df['fecha'].max().date()}")

    fechas = np.sort(df["fecha"].unique())
    start_ts = pd.Timestamp(start_date)
    idx0 = int(np.searchsorted(fechas, start_ts))

    # Ventanas de holdout
    folds = []
    h = idx0
    while h + holdout_len <= len(fechas):
        train_end_idx = h - embargo
        if train_end_idx > 30:  # minimo de train
            folds.append((h, min(h + holdout_len, len(fechas))))
        h += holdout_len
    print(f"  Folds: {len(folds)}  |  holdout={holdout_len}d  embargo={embargo}d  "
          f"start={start_date}")
    print("=" * 70)

    oos_all = []
    metric_rows = []

    for fi, (h, h_end) in enumerate(folds, 1):
        ho_ini, ho_fin = fechas[h], fechas[h_end - 1]
        train_end = fechas[h - embargo - 1]
        tr = df[df["fecha"] <= train_end]
        ho = df[(df["fecha"] >= ho_ini) & (df["fecha"] <= ho_fin)]
        if len(ho) == 0 or tr[TARGET_COL].nunique() < 2:
            continue

        Xtr, ytr = tr[FEATURE_COLS_V3], tr[TARGET_COL].astype(int).values
        Xho, yho = ho[FEATURE_COLS_V3], ho[TARGET_COL].astype(int).values
        ret_ho = ho["retorno_20d"].astype(float).values
        base = float(yho.mean())

        t0 = datetime.now()
        # rf de produccion
        rf = construir_modelo("rf", len(ytr))
        rf.fit(Xtr, ytr)
        p_rf = rf.predict_proba(Xho)[:, 1]

        # rf calibrado (isotonica, cv=3 dentro del train)
        rf_cal = CalibratedClassifierCV(construir_modelo("rf", len(ytr)),
                                        method="isotonic", cv=3)
        rf_cal.fit(Xtr, ytr)
        p_cal = rf_cal.predict_proba(Xho)[:, 1]

        # elastic-net (baseline lineal; cap de train por velocidad de saga)
        if len(ytr) > EN_MAX_TRAIN:
            samp = tr.sample(EN_MAX_TRAIN, random_state=42)
            Xen, yen = samp[FEATURE_COLS_V3], samp[TARGET_COL].astype(int).values
        else:
            Xen, yen = Xtr, ytr
        en = modelo_elasticnet()
        en.fit(Xen, yen)
        p_en = en.predict_proba(Xho)[:, 1]

        secs = (datetime.now() - t0).seconds

        for nombre, prob in [("rf", p_rf), ("rf_cal", p_cal), ("en", p_en)]:
            m = metricas_oos(prob, yho, ret_ho, base)
            m.update(fold=fi, modelo=nombre, ho_ini=str(ho_ini.astype("datetime64[D]")),
                     ho_fin=str(ho_fin.astype("datetime64[D]")), n_train=len(ytr),
                     n_holdout=len(yho), base_rate=round(base, 4))
            metric_rows.append(m)

        oos = ho[["fecha", "ticker", "sector", "retorno_20d", TARGET_COL]].copy()
        oos["fold"] = fi
        oos["prob_rf"] = p_rf
        oos["prob_rf_cal"] = p_cal
        oos["prob_en"] = p_en
        oos_all.append(oos)

        if verbose:
            r = {row["modelo"]: row for row in metric_rows if row["fold"] == fi}
            print(f"  Fold {fi:2d} [{str(ho_ini.astype('datetime64[D]'))} -> "
                  f"{str(ho_fin.astype('datetime64[D]'))}] base={base:.2f} n_ho={len(yho):5d} "
                  f"({secs}s)")
            print(f"      AUC   rf={r['rf']['auc']:.3f} cal={r['rf_cal']['auc']:.3f} "
                  f"en={r['en']['auc']:.3f}  |  LIFT@decil rf={r['rf']['lift_decil']:.2f} "
                  f"cal={r['rf_cal']['lift_decil']:.2f} en={r['en']['lift_decil']:.2f}")
            print(f"      Brier rf={r['rf']['brier']:.3f} cal={r['rf_cal']['brier']:.3f}  "
                  f"|  ret@decil rf={r['rf']['ret_decil']:+.2f} cal={r['rf_cal']['ret_decil']:+.2f} "
                  f"en={r['en']['ret_decil']:+.2f}")

    met = pd.DataFrame(metric_rows)
    oos_df = pd.concat(oos_all, ignore_index=True) if oos_all else pd.DataFrame()

    # ---- Resumen pooled + estabilidad ----
    print("\n" + "=" * 70)
    print("  RESUMEN POR MODELO (media +/- desvio across folds)")
    print("=" * 70)
    for modelo in ["rf", "rf_cal", "en"]:
        sub = met[met["modelo"] == modelo]
        print(f"  {modelo:7s}  AUC={sub['auc'].mean():.3f}+/-{sub['auc'].std():.3f}  "
              f"LIFT@decil={sub['lift_decil'].mean():.2f}+/-{sub['lift_decil'].std():.2f}  "
              f"ret@decil={sub['ret_decil'].mean():+.2f}  "
              f"Brier={sub['brier'].mean():.3f}  "
              f"folds_lift>1={int((sub['lift_decil'] > 1).sum())}/{len(sub)}")

    # Pooled OOS (concatenando todos los holdouts)
    if not oos_df.empty:
        print("\n  CALIBRACION POOLED (reliability por decil de prob):")
        y_pool = oos_df[TARGET_COL].astype(int).values
        for col, nombre in [("prob_rf", "rf"), ("prob_rf_cal", "rf_cal")]:
            tab = tabla_calibracion(oos_df[col].values, y_pool)
            top = tab.iloc[-1]
            print(f"    {nombre:7s} top-bin(p~{top['p_med']:.2f}) obs={top['obs']:.2f} "
                  f"n={int(top['n'])}  | bins obs: "
                  + " ".join(f"{v:.2f}" for v in tab['obs'].values))

    # ---- Persistencia ----
    os.makedirs(OUT_DIR, exist_ok=True)
    met.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False)
    if not oos_df.empty:
        oos_df.to_parquet(os.path.join(OUT_DIR, "oos_predictions.parquet"), index=False)
    print("\n  Guardado: reportes/ml_walkforward/{fold_metrics.csv, oos_predictions.parquet}")
    print("=" * 70)
    return met, oos_df


def main():
    ap = argparse.ArgumentParser(description="Fase 2 -- walk-forward purgado ML")
    ap.add_argument("--holdout", type=int, default=126, help="dias habiles por holdout (~6m)")
    ap.add_argument("--embargo", type=int, default=20, help="dias de purga (horizonte del label)")
    ap.add_argument("--start", default="2021-06-01", help="fecha de arranque del 1er holdout")
    args = ap.parse_args()
    correr(args.holdout, args.embargo, args.start)


if __name__ == "__main__":
    main()
