"""
screen_sectorial_v3.py
Tarea 23, Fase 3a, paso 1: hay heterogeneidad POR SECTOR en el modelo honesto?
Solo LEE las predicciones fuera de muestra que dejo el walk-forward. No entrena nada.

POR QUE (docs/estructura_velas.md sec. 9.7)
    El modelo v3 global no pasa la compuerta (AUC media 0,51 en 6 folds purgados). La
    pregunta siguiente es si entrenar POR SECTOR da otra cosa. Antes de entrenar 9
    modelos conviene medir la PREMISA de esa idea: la discriminacion del global varia
    por sector mas alla del ruido? Si en ningun sector hay senal, no hay heterogeneidad
    que un modelo sectorial pueda capturar y se frena ahi.

    Ademas el rechazo previo de los modelos sectoriales (ml_reentrenamiento.md sec. 2.5)
    se midio con las features que miran al futuro, y ese sesgo favorecia al global: con
    mas filas agrupadas explota mejor la fuga que un sectorial chico. Por eso se vuelve
    a preguntar.

LA UNIDAD DE INDEPENDENCIA ES EL FOLD, NO LA FILA
    Dentro de un fold, las ~15.000 filas comparten el mismo mercado de esos 6 meses: no
    son 15.000 observaciones independientes. El IC95 se calcula sobre las 6 mediciones
    por fold de cada sector. Un IC calculado sobre filas daria intervalos ~40 veces mas
    angostos y declararia significativo cualquier ruido.

REGLA DE LECTURA (fijada ANTES de mirar, doc sec. 9.7)
    Un sector califica para el paso 2 si, en cualquiera de las dos ramas de features,
    cumple las TRES:
      1. AUC media entre folds >= 0,54
      2. IC95 de esa media excluye 0,50
      3. AUC > 0,50 en al menos 5 de los 6 folds
    Con 9 sectores x 2 ramas = 18 comparaciones, la condicion 3 es el control de falsos
    positivos.

Uso:
    python scripts/ml/screen_sectorial_v3.py
    python scripts/ml/screen_sectorial_v3.py --oos reportes/ml_v3/wf_oos_vieja.parquet
"""

import argparse
import json
import math
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

DIR_REPORTES = os.path.join(ROOT, "reportes", "ml_v3")
OOS_DEFAULT = os.path.join(DIR_REPORTES, "wf_oos_nueva.parquet")

# Compuerta del paso 1, pre-registrada en docs/estructura_velas.md sec. 9.7.
COMPUERTA = {
    "auc_media_min": 0.54,
    "ic95_excluye_050": True,
    "folds_sobre_050_min": 5,
}
# t de Student al 97,5% con 5 grados de libertad (6 folds).
_T_5GL = 2.571
MIN_FILAS_FOLD = 300      # un sector-fold con menos filas no se mide


def log(msg=""):
    print(msg, flush=True)


def auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def ic95_folds(valores) -> tuple:
    """IC95 de la media de las mediciones POR FOLD (t con n-1 grados)."""
    v = np.asarray([x for x in valores if not np.isnan(x)], dtype=float)
    n = len(v)
    if n < 3:
        return float("nan"), float("nan")
    se = float(v.std(ddof=1) / math.sqrt(n))
    t = _T_5GL if n == 6 else 1.959964
    return float(v.mean()) - t * se, float(v.mean()) + t * se


def metricas_sector(sub: pd.DataFrame) -> dict:
    """Por fold: AUC, lift del decil alto y exceso a 20 ruedas del decil alto."""
    aucs, lifts, excesos, ns = [], [], [], []
    for fold, g in sub.groupby("fold"):
        if len(g) < MIN_FILAS_FOLD:
            continue
        y = g["label_binario"].astype(int).to_numpy()
        p = g["prob"].to_numpy()
        r = g["retorno_20d"].astype(float).to_numpy()
        base = float(y.mean())
        a = auc(y, p)
        k = max(1, int(len(p) * 0.10))
        top = np.argsort(-p)[:k]
        aucs.append(a)
        lifts.append(float(y[top].mean()) / base if base > 0 else np.nan)
        excesos.append(float(r[top].mean() - r.mean()))
        ns.append(len(g))
    lo, hi = ic95_folds(aucs)
    validos = [a for a in aucs if not np.isnan(a)]
    return {
        "folds": len(validos),
        "filas": int(sum(ns)),
        "auc_media": float(np.mean(validos)) if validos else np.nan,
        "auc_desvio": float(np.std(validos, ddof=1)) if len(validos) > 1 else np.nan,
        "ic95_lo": lo, "ic95_hi": hi,
        "folds_sobre_050": int(sum(1 for a in validos if a > 0.50)),
        "auc_por_fold": [round(a, 4) for a in aucs],
        "lift_medio": float(np.mean(lifts)) if lifts else np.nan,
        "exceso_medio": float(np.mean(excesos)) if excesos else np.nan,
    }


def califica(m: dict) -> dict:
    c1 = bool(m["auc_media"] >= COMPUERTA["auc_media_min"])
    c2 = bool(m["ic95_lo"] > 0.50 or m["ic95_hi"] < 0.50)
    c3 = bool(m["folds_sobre_050"] >= COMPUERTA["folds_sobre_050_min"])
    return {"auc_media": c1, "ic95_excluye_050": c2, "consistencia_folds": c3,
            "califica": bool(c1 and c2 and c3)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Screen de heterogeneidad sectorial (paso 1)")
    ap.add_argument("--oos", default=OOS_DEFAULT,
                    help="parquet de predicciones fuera de muestra del walk-forward")
    args = ap.parse_args()

    if not os.path.exists(args.oos):
        log(f"[ERROR] falta {args.oos}. Generarlo con:")
        log("  python scripts/ml/entrenar_ml_v3.py --solo-wf")
        return 1

    df = pd.read_parquet(args.oos)
    log("=" * 86)
    log("PASO 1 -- HAY HETEROGENEIDAD POR SECTOR? (pre-registro: doc sec. 9.7)")
    log("=" * 86)
    log(f"  Predicciones fuera de muestra: {args.oos}")
    log(f"  {len(df):,} filas | {df['fold'].nunique()} folds | "
        f"{df['sector'].nunique()} sectores | ramas: {sorted(df['brazo'].unique())}")
    log(f"  Compuerta: AUC media >= {COMPUERTA['auc_media_min']} + IC95 excluye 0,50 + "
        f"AUC > 0,50 en >= {COMPUERTA['folds_sobre_050_min']}/6 folds")

    resultado, calificados = {}, []
    for brazo, sub_b in df.groupby("brazo"):
        log("")
        log("-" * 86)
        log(f"RAMA: {brazo}")
        log("-" * 86)
        log(f"  {'sector':<24} {'filas':>7} {'AUC':>7} {'desvio':>7} "
            f"{'IC95':>18} {'>0,50':>6} {'lift':>6} {'exc20d':>7}  califica")

        glob = metricas_sector(sub_b)
        filas_sec = []
        for sector, sub in sub_b.groupby("sector"):
            m = metricas_sector(sub)
            if m["folds"] < 3:
                log(f"  {str(sector):<24} {m['filas']:>7,} -- solo {m['folds']} folds, se omite")
                continue
            c = califica(m)
            filas_sec.append((sector, m, c))
            if c["califica"]:
                calificados.append((brazo, sector, m))
            log(f"  {str(sector):<24} {m['filas']:>7,} {m['auc_media']:>7.4f} "
                f"{m['auc_desvio']:>7.4f} "
                f"[{m['ic95_lo']:>6.4f};{m['ic95_hi']:>6.4f}] "
                f"{m['folds_sobre_050']:>4}/6 {m['lift_medio']:>6.2f} "
                f"{m['exceso_medio']:>+7.2f}  {'SI' if c['califica'] else 'no'}")

        log(f"  {'GLOBAL (todas las filas)':<24} {glob['filas']:>7,} "
            f"{glob['auc_media']:>7.4f} {glob['auc_desvio']:>7.4f} "
            f"[{glob['ic95_lo']:>6.4f};{glob['ic95_hi']:>6.4f}] "
            f"{glob['folds_sobre_050']:>4}/6 {glob['lift_medio']:>6.2f} "
            f"{glob['exceso_medio']:>+7.2f}")

        # Heterogeneidad: dispersion ENTRE sectores vs dispersion ENTRE folds dentro
        # del sector. Si la de sectores no supera a la de folds, lo que se ve es ruido
        # temporal repartido, no una diferencia estable por sector.
        if filas_sec:
            entre_sectores = float(np.std([m["auc_media"] for _, m, _ in filas_sec], ddof=1))
            dentro = float(np.mean([m["auc_desvio"] for _, m, _ in filas_sec]))
            log("")
            log(f"  dispersion de AUC ENTRE sectores : {entre_sectores:.4f}")
            log(f"  dispersion media DENTRO de sector (entre folds): {dentro:.4f}")
            log(f"  -> {'los sectores se separan mas que el ruido temporal' if entre_sectores > dentro else 'el ruido temporal domina: los sectores no se separan'}")
            resultado[brazo] = {
                "global": {k: v for k, v in glob.items()},
                "sectores": {str(s): {"metricas": m, "criterios": c} for s, m, c in filas_sec},
                "dispersion_entre_sectores": entre_sectores,
                "dispersion_dentro_sector": dentro,
            }

    log("")
    log("=" * 86)
    os.makedirs(DIR_REPORTES, exist_ok=True)
    salida = os.path.join(DIR_REPORTES, "screen_sectorial.json")
    with open(salida, "w", encoding="utf-8") as fh:
        json.dump({"compuerta": COMPUERTA, "oos": os.path.basename(args.oos),
                   "resultado": resultado,
                   "califican": [{"brazo": b, "sector": str(s)} for b, s, _ in calificados]},
                  fh, indent=2, default=float)

    if calificados:
        log(f"RESULTADO: {len(calificados)} sector(es) califican para el paso 2:")
        for brazo, sector, m in calificados:
            log(f"  {sector} ({brazo}): AUC {m['auc_media']:.4f} "
                f"IC95 [{m['ic95_lo']:.4f}; {m['ic95_hi']:.4f}], "
                f"{m['folds_sobre_050']}/6 folds > 0,50")
        log("\nSigue el paso 2: modelos sectoriales sobre los mismos folds, head-to-head")
        log("contra el global en las mismas filas (doc sec. 9.7).")
        log(f"\nDetalle: {salida}")
        return 0

    log("RESULTADO: NINGUN sector califica. No hay heterogeneidad que un modelo")
    log("sectorial pueda capturar -> se FRENA el paso 2 y se documenta.")
    log(f"\nDetalle: {salida}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
