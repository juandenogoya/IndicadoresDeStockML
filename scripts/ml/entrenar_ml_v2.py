"""
entrenar_ml_v2.py -- Etapa 3c: entrena el modelo ML v2 del scanner (Fase 5, Tarea 20).

Config congelada en docs/ml_reentrenamiento.md sec. 8: RF global (construir_modelo
'rf', 53 features V3) + CalibratedClassifierCV isotonica (cv=3) + label absoluto
(retorno_20d > +1%) + sin ponderador sectorial. Todo LOCAL, sin Yahoo.

Pasos:
  1. Carga features_ml + features_market_structure con la MISMA funcion del
     walk-forward validado (walkforward_ml.cargar_dataset).
  2. HOLDOUT: entrena hasta HOLDOUT + EMBARGO ruedas antes del final y evalua las
     ultimas HOLDOUT ruedas con label (el embargo evita que el label de 20 dias de
     las ultimas filas de train mire dentro del holdout).
  3. COMPUERTA con umbrales fijados ANTES de ver el resultado (ver COMPUERTA). Si
     no pasa, termina sin guardar nada.
  4. CORTES: probabilidad de la v1 (modelos V3 desplegados, misma seleccion de
     modelo que el scanner) y de la v2 sobre las mismas filas del holdout
     posteriores al despliegue de la v1; cortes equivalentes (ml_v2).
  5. Reentrena con TODAS las filas con label y guarda models_ml_v2/:
     rf_cal_global.joblib (fuera de git) + metadata.json (en git).

Salidas de analisis en reportes/ml_v2/ (gitignoreado).

Uso (desde la raiz):
    python scripts/ml/entrenar_ml_v2.py --dry-run   # pasos 1-4, no reentrena ni guarda
    python scripts/ml/entrenar_ml_v2.py             # todo
    python scripts/ml/entrenar_ml_v2.py --forzar    # pisa un modelo v2 existente
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import date, datetime

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, AQUI)

# LOCAL-only: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV

from src.data.database import query_df
from src.indicators.market_structure import FEATURE_COLS_MS
from src.ml import ml_v2
from src.ml.trainer import TARGET_COL, construir_modelo
from src.ml.trainer_v3 import FEATURE_COLS_V3
from walkforward_ml import cargar_dataset, metricas_oos

os.environ.pop("DATABASE_URL", None)

HOLDOUT_RUEDAS = 126
EMBARGO_RUEDAS = 20

# La v1 desplegada (models_v3, 10/4/2026) se entreno con features_ml hasta el
# 12/3/2026. Para fijar cortes se usan solo filas posteriores a su despliegue,
# donde sus probabilidades tambien son fuera de muestra.
V1_DESPLEGADA = date(2026, 4, 10)

# COMPUERTA, fijada ANTES de ver el resultado. Referencia del walk-forward (9 folds,
# docs/ml_reentrenamiento.md sec. 8): AUC 0,599; lift@decil 1,34 (9/9 > 1); decil
# alto 64,5% contra 48,9% de base. Los pisos quedan bastante debajo porque el
# holdout es UNA ventana de 126 ruedas (un regimen): frena un modelo roto, no
# certifica el edge.
COMPUERTA = {
    "auc_min": 0.55,
    "lift_decil_min": 1.10,
    # retorno 20d medio del decil alto menos el del holdout, en puntos
    "exceso_decil_min": 0.0,
    # Brier menor que el de pronosticar siempre la tasa base
    "brier_mejor_que_constante": True,
    # |probabilidad media - tasa observada| en el decil alto
    "calibracion_decil_max_desvio": 0.10,
}

DIR_REPORTES = os.path.join(ROOT, "reportes", "ml_v2")


def log(msg=""):
    print(msg, flush=True)


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (date, datetime, pd.Timestamp)):
        return str(o)[:10]
    raise TypeError(f"no serializable: {type(o)}")


def modelo_config(n_train: int) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(construir_modelo("rf", n_train), method="isotonic", cv=3)


def matriz(df: pd.DataFrame) -> np.ndarray:
    # ndarray y no DataFrame: el scanner sirve con ndarray (ml_v2.matriz_x) y asi
    # sklearn no avisa por nombres de columnas al predecir.
    return df[FEATURE_COLS_V3].to_numpy(dtype=float)


def etiquetas(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COL].astype(int).to_numpy()


def partir_holdout(df: pd.DataFrame, ruedas: int, embargo: int):
    fechas = np.sort(df["fecha"].unique())
    if len(fechas) < ruedas + embargo + 250:
        raise ValueError(f"historia insuficiente: {len(fechas)} ruedas con label")
    ho_ini = pd.Timestamp(fechas[-ruedas])
    train_hasta = pd.Timestamp(fechas[-ruedas - embargo - 1])
    return df[df["fecha"] <= train_hasta], df[df["fecha"] >= ho_ini], train_hasta, ho_ini


def calibracion_deciles(prob: np.ndarray, y: np.ndarray) -> list:
    partes = np.array_split(np.argsort(-prob), 10)
    return [{"decil": i + 1, "prob_media": round(float(prob[p].mean()), 4),
             "tasa_obs": round(float(y[p].mean()), 4), "n": int(len(p))}
            for i, p in enumerate(partes)]


def evaluar_compuerta(m: dict, cal: list, base: float) -> dict:
    brier_constante = base * (1 - base)
    desvio = abs(cal[0]["prob_media"] - cal[0]["tasa_obs"])
    criterios = {
        "auc": (m["auc"], m["auc"] >= COMPUERTA["auc_min"]),
        "lift_decil": (m["lift_decil"], m["lift_decil"] >= COMPUERTA["lift_decil_min"]),
        "exceso_decil": (m["exceso_decil"], m["exceso_decil"] > COMPUERTA["exceso_decil_min"]),
        "brier_vs_constante": ((m["brier"], brier_constante), m["brier"] < brier_constante),
        "calibracion_decil_alto": (desvio, desvio <= COMPUERTA["calibracion_decil_max_desvio"]),
    }
    return {"criterios": {k: {"valor": v, "pasa": bool(ok)} for k, (v, ok) in criterios.items()},
            "pasa": all(ok for _, ok in criterios.values())}


def probs_v1(df: pd.DataFrame) -> np.ndarray:
    """Probabilidad de la v1 con la misma seleccion de modelo que el scanner
    (signal_engine.seleccionar_modelo): modelo_asignado -> sector -> global."""
    from src.pipeline.signal_engine import cargar_modelos_v3
    modelos = cargar_modelos_v3()
    activos = query_df("SELECT ticker, sector, modelo_asignado FROM activos")
    scope = {}
    for r in activos.itertuples():
        asignado = r.modelo_asignado if r.modelo_asignado and str(r.modelo_asignado) != "None" else None
        if asignado and asignado in modelos:
            scope[r.ticker] = asignado
        elif r.sector in modelos:
            scope[r.ticker] = r.sector
        else:
            scope[r.ticker] = "global"
    s = df["ticker"].map(scope).fillna("global").to_numpy()
    X = matriz(df)
    p = np.full(len(df), np.nan)
    for nombre in np.unique(s):
        idx = np.where(s == nombre)[0]
        p[idx] = modelos[nombre].predict_proba(X[idx])[:, 1]
    return p


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _linea_metricas(nombre, m):
    return (f"  {nombre:<10} AUC={m['auc']:.3f}  lift@decil={m['lift_decil']:.2f}  "
            f"acierto decil={m['prec_decil']:.3f}  ret decil={m['ret_decil']:+.2f}  "
            f"Brier={m['brier']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Entrena el modelo ML v2 del scanner (Etapa 3c)")
    ap.add_argument("--dry-run", action="store_true", help="holdout, compuerta y cortes; no guarda")
    ap.add_argument("--forzar", action="store_true", help="pisa un modelo v2 existente")
    ap.add_argument("--holdout", type=int, default=HOLDOUT_RUEDAS)
    ap.add_argument("--embargo", type=int, default=EMBARGO_RUEDAS)
    args = ap.parse_args()

    ruta_modelo = os.path.join(ml_v2.DIR_MODELO_V2, ml_v2.ARCHIVO_MODELO)
    if not args.dry_run and os.path.exists(ruta_modelo) and not args.forzar:
        log(f"[ERROR] ya existe {ruta_modelo}. Usar --forzar para pisarlo.")
        sys.exit(1)

    t0 = datetime.now()
    log("=" * 72)
    log(f"  ENTRENAR ML v2  |  {t0:%Y-%m-%d %H:%M}  {'[DRY RUN]' if args.dry_run else ''}")
    log("=" * 72)

    # 1. Datos
    df = cargar_dataset()
    log(f"  Dataset: {len(df):,} filas con label, {df['ticker'].nunique()} tickers, "
        f"{df['fecha'].min().date()} -> {df['fecha'].max().date()}, base {df[TARGET_COL].mean():.3f}")

    # 2. Holdout
    tr, ho, train_hasta, ho_ini = partir_holdout(df, args.holdout, args.embargo)
    log(f"  Holdout: train hasta {train_hasta.date()} ({len(tr):,} filas) | embargo {args.embargo} "
        f"| evalua {ho_ini.date()} -> {ho['fecha'].max().date()} ({len(ho):,} filas)")
    modelo_ho = modelo_config(len(tr))
    modelo_ho.fit(matriz(tr), etiquetas(tr))
    p_ho = modelo_ho.predict_proba(matriz(ho))[:, 1]
    y_ho = etiquetas(ho)
    ret_ho = ho["retorno_20d"].astype(float).to_numpy()
    base = float(y_ho.mean())
    m = metricas_oos(p_ho, y_ho, ret_ho, base)
    m["exceso_decil"] = m["ret_decil"] - float(ret_ho.mean())
    cal = calibracion_deciles(p_ho, y_ho)
    log(f"  Entrenado en {(datetime.now() - t0).seconds}s")

    # 3. Compuerta
    comp = evaluar_compuerta(m, cal, base)
    log("\n  HOLDOUT (base %.3f, retorno 20d medio %+.2f)" % (base, ret_ho.mean()))
    log(_linea_metricas("v2", m))
    log("  Calibracion por decil de probabilidad (prob media / tasa observada):")
    log("    " + "  ".join(f"d{c['decil']}:{c['prob_media']:.2f}/{c['tasa_obs']:.2f}" for c in cal))
    log("\n  COMPUERTA:")
    for k, v in comp["criterios"].items():
        log(f"    {k:<24} {str(v['valor']):<44} {'PASA' if v['pasa'] else 'NO PASA'}")

    # 4. Cortes contra la v1, mismas filas posteriores a su despliegue
    post = (ho["fecha"] > pd.Timestamp(V1_DESPLEGADA)).to_numpy()
    p_v1 = probs_v1(ho[post])
    p_v2_post = p_ho[post]
    cortes_v2 = ml_v2.cortes_equivalentes(p_v1, p_v2_post, ml_v2.CORTES_V1)
    frac_v1 = ml_v2.fracciones_sobre(p_v1, ml_v2.CORTES_V1)
    frac_v2 = ml_v2.fracciones_sobre(p_v2_post, cortes_v2)
    base_post = float(y_ho[post].mean())
    m_v1_post = metricas_oos(p_v1, y_ho[post], ret_ho[post], base_post)
    m_v2_post = metricas_oos(p_v2_post, y_ho[post], ret_ho[post], base_post)
    fecha_post = ho.loc[post, "fecha"]
    log(f"\n  CORTES (filas del holdout posteriores al {V1_DESPLEGADA}: {int(post.sum()):,}, "
        f"{fecha_post.min().date()} -> {fecha_post.max().date()}, base {base_post:.3f})")
    log("    corte v1  fraccion v1  ->  corte v2  fraccion v2  acierto v1  acierto v2")
    aciertos = []
    for c1, f1, c2, f2 in zip(ml_v2.CORTES_V1, frac_v1, cortes_v2, frac_v2):
        a1 = float(y_ho[post][p_v1 >= c1].mean())
        a2 = float(y_ho[post][p_v2_post >= c2].mean())
        aciertos.append({"corte_v1": c1, "acierto_v1": round(a1, 4), "corte_v2": round(c2, 6),
                         "acierto_v2": round(a2, 4)})
        log(f"    {c1:.2f}      {f1:.4f}       ->  {c2:.4f}    {f2:.4f}       {a1:.3f}       {a2:.3f}")
    log("  Mismo tramo, informativo (la v1 no es fuera de muestra si su scope sectorial "
        "se entreno con otra historia):")
    log(_linea_metricas("v1", m_v1_post))
    log(_linea_metricas("v2", m_v2_post))

    os.makedirs(DIR_REPORTES, exist_ok=True)
    salida = ho[["fecha", "ticker", "sector", "retorno_20d", TARGET_COL]].copy()
    salida["prob_v2"] = p_ho
    salida["prob_v1"] = np.nan
    salida.loc[post, "prob_v1"] = p_v1
    salida.to_parquet(os.path.join(DIR_REPORTES, "holdout_predicciones.parquet"), index=False)

    if not comp["pasa"]:
        log("\n  [FRENO] La compuerta NO pasa: no se reentrena ni se guarda nada.")
        sys.exit(1)
    if args.dry_run:
        log("\n  [DRY RUN] Compuerta OK. Sin reentrenar ni guardar.")
        return

    # 5. Modelo final con todas las filas con label
    t1 = datetime.now()
    modelo = modelo_config(len(df))
    modelo.fit(matriz(df), etiquetas(df))
    rf_params = modelo.estimator.named_steps
    clf = [s for s in rf_params.values() if hasattr(s, "n_estimators")][0]
    os.makedirs(ml_v2.DIR_MODELO_V2, exist_ok=True)
    joblib.dump(modelo, ruta_modelo, compress=3)
    sha = ml_v2.sha256_archivo(ruta_modelo)
    log(f"\n  Modelo final: {len(df):,} filas en {(datetime.now() - t1).seconds}s -> "
        f"{ruta_modelo} ({os.path.getsize(ruta_modelo) / 1e6:.1f} MB)")

    hoy = date.today()
    meta = {
        "version": f"ml_v2_{hoy:%Y%m%d}",
        "creado": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": {
            "modelo": "CalibratedClassifierCV(construir_modelo('rf', n), method='isotonic', cv=3)",
            "rf": {k: getattr(clf, k) for k in ("n_estimators", "max_depth", "min_samples_leaf",
                                                 "max_features", "class_weight", "random_state")},
            "label": "label_binario = retorno_20d > +1% (UMBRAL_NEUTRO)",
            "ponderador_sectorial": None,
            "referencia": "docs/ml_reentrenamiento.md sec. 8 (config congelada, Tarea 20)",
        },
        "features": list(FEATURE_COLS_V3),
        "columnas_cero": list(FEATURE_COLS_MS),
        "datos": {
            "origen": "features_ml JOIN features_market_structure (walkforward_ml.cargar_dataset)",
            "desde": df["fecha"].min(), "hasta": df["fecha"].max(),
            "filas": len(df), "tickers": int(df["ticker"].nunique()),
            "base_label": round(float(df[TARGET_COL].mean()), 4),
        },
        "holdout": {
            "ruedas": args.holdout, "embargo": args.embargo,
            "train_hasta": train_hasta, "desde": ho_ini, "hasta": ho["fecha"].max(),
            "filas_train": len(tr), "filas": len(ho), "base": round(base, 4),
            "metricas": {k: round(float(v), 4) for k, v in m.items()},
            "calibracion_deciles": cal,
        },
        "compuerta": {"umbrales": COMPUERTA, **comp},
        "cortes": {
            "v1": list(ml_v2.CORTES_V1),
            "v2": [round(c, 6) for c in cortes_v2],
            "fracciones_v1": [round(f, 4) for f in frac_v1],
            "fracciones_v2": [round(f, 4) for f in frac_v2],
            "aciertos": aciertos,
            "filas": int(post.sum()), "desde": fecha_post.min(), "hasta": fecha_post.max(),
            "nota": ("Cortes del modelo del HOLDOUT sobre filas posteriores al despliegue de la v1; "
                     "se aplican al modelo final. Comparar en vivo la cantidad de COMPRA_FUERTE v1 vs v2."),
        },
        "comparacion_mismo_tramo": {
            "v1": {k: round(float(v), 4) for k, v in m_v1_post.items()},
            "v2": {k: round(float(v), 4) for k, v in m_v2_post.items()},
        },
        "versiones": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__,
                      "joblib": joblib.__version__},
        "archivo": ml_v2.ARCHIVO_MODELO,
        "bytes": os.path.getsize(ruta_modelo),
        "sha256": sha,
    }
    with open(os.path.join(ml_v2.DIR_MODELO_V2, ml_v2.ARCHIVO_META), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=True, default=_json)

    # Verificacion: se carga como la cargaria el scanner y predice igual.
    cargado, meta_leida = ml_v2.cargar_modelo_v2(features_esperadas=FEATURE_COLS_V3)
    muestra = ho.iloc[:200]
    p_directo = modelo.predict_proba(matriz(muestra))[:, 1]
    p_servido = np.array([ml_v2.prob_v2(cargado, meta_leida, dict(zip(FEATURE_COLS_V3, fila)))
                          for fila in matriz(muestra)])
    log(f"  Verificacion de carga: sha OK, max |dif| directo vs servido = "
        f"{np.max(np.abs(p_directo - p_servido)):.2e}")
    log(f"  Metadata: {os.path.join(ml_v2.DIR_MODELO_V2, ml_v2.ARCHIVO_META)}")
    log(f"  Total {(datetime.now() - t0).seconds}s")


if __name__ == "__main__":
    main()
