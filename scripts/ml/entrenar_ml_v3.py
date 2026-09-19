"""
entrenar_ml_v3.py -- Tarea 23, Fase 3a: entrena y valida el modelo ML v3 con features
que NO miran al futuro. No toca produccion: el scanner, la v1 y la v2 siguen igual.

QUE CAMBIA CONTRA LA v2 (docs/estructura_velas.md sec. 4.4 y 9.5)
    Una sola cosa: las 24 features de market structure salen de `features_estructura`
    (swings CONFIRMADOS, invariantes) en vez de `features_market_structure`, cuya
    historia mira 10 ruedas al futuro. Misma config congelada de la Tarea 20 (RF global
    + calibracion isotonica, label absoluto), mismas 29 features restantes. Por eso el
    AUC de la v2 (0,641 de holdout) NO es comparable: se midio con informacion futura.

PRE-REGISTRO (escrito en docs/estructura_velas.md sec. 9.5 ANTES de correr esto)
    Particion: desarrollo = primer 80% de las ruedas con label; embargo de 20 ruedas;
    lockbox = ultimo 20%, que se mira UNA sola vez con la config ya congelada.
    Walk-forward purgado DENTRO del desarrollo: holdout 126, embargo 20, train
    expansivo desde la rueda 400 -> 6 folds.
    Dos brazos: A = 53 features (con estructura), B = 29 (ablacion sin estructura).
    Compuertas: ver COMPUERTA abajo. Si no pasan, no hay v3 y queda documentado.

    Las fechas de corte estan HARDCODEADAS en PARTICION_ESPERADA: si el dataset cambia
    (se recarga features_ml, entra un ticker), el script FRENA en vez de mover
    silenciosamente el lockbox, que es lo unico que no se puede volver a usar.

Pasos:
    1. Dataset (walkforward_ml.cargar_dataset("nueva")) y particion verificada.
    2. Walk-forward en el desarrollo, los dos brazos. Compuertas 1 y 2 -> brazo elegido.
    3. LOCKBOX, una sola evaluacion. Compuerta 3.
    4. Si paso todo: reentrena con desarrollo + lockbox y guarda models_ml_v3/
       (artefacto fuera de git + metadata.json en git con sha256, features, fechas).
    5. Cortes equivalentes a los de la v1 sobre las mismas filas, para la Fase 3b.

Salidas de analisis en reportes/ml_v3/.

Uso (desde la raiz):
    python scripts/ml/entrenar_ml_v3.py --dry-run     # pasos 1-3, no guarda nada
    python scripts/ml/entrenar_ml_v3.py               # todo
    python scripts/ml/entrenar_ml_v3.py --solo-wf     # solo el walk-forward (pasos 1-2)
    python scripts/ml/entrenar_ml_v3.py --forzar      # pisa un models_ml_v3 existente
"""

import argparse
import json
import math
import os
import platform
import sys
from datetime import date, datetime

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, AQUI)

# LOCAL-only: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from src.indicators.market_structure import FEATURE_COLS_MS  # noqa: E402
from src.ml import ml_v2  # noqa: E402
from src.ml.trainer import FEATURE_COLS, TARGET_COL, construir_modelo  # noqa: E402
from src.ml.trainer_v3 import FEATURE_COLS_V3  # noqa: E402
from walkforward_ml import cargar_dataset, metricas_oos  # noqa: E402
from entrenar_ml_v2 import _git_commit, probs_v1  # noqa: E402

DIR_MODELO_V3 = os.path.join(ROOT, "models_ml_v3")
DIR_REPORTES = os.path.join(ROOT, "reportes", "ml_v3")

# --- Pre-registro -------------------------------------------------------------
PCT_DESARROLLO = 0.80
EMBARGO_RUEDAS = 20
HOLDOUT_WF = 126
MIN_TRAIN_WF = 400          # ruedas antes del primer holdout (~1,6 anios)

# Fechas calculadas el 17/9/2026 sobre 1.463 ruedas con label. Sirven de guard:
# el lockbox no se puede mover despues de haberlo mirado.
PARTICION_ESPERADA = {
    "ruedas_total": 1463,
    "dev_desde": "2020-10-15", "dev_hasta": "2025-06-12", "dev_ruedas": 1170,
    "lock_desde": "2025-07-15", "lock_hasta": "2026-08-13", "lock_ruedas": 273,
}

BRAZOS = {"con_estructura": FEATURE_COLS_V3, "sin_estructura": FEATURE_COLS}

COMPUERTA = {
    # 1. walk-forward del brazo elegido
    "wf_auc_media_min": 0.54,
    "wf_auc_fold_min": 0.52,
    "wf_folds_min": 5,              # de 6
    # 2. ablacion: el brazo con estructura tiene que ganarle al de 29 features
    "ablacion_delta_auc_min": 0.005,
    "ablacion_folds_min": 4,        # de 6
    # 3. lockbox (una sola corrida)
    "lock_auc_min": 0.53,
    "lock_decil_sobre_base_pp_min": 5.0,
    "lock_exceso_ic95_excluye_cero": True,
}


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
    """La config congelada de la Tarea 20. Identica a la de la v2."""
    return CalibratedClassifierCV(construir_modelo("rf", n_train), method="isotonic", cv=3)


def matriz(df: pd.DataFrame, cols) -> np.ndarray:
    return df[list(cols)].to_numpy(dtype=float)


def etiquetas(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COL].astype(int).to_numpy()


def ic95_media(x: np.ndarray) -> tuple:
    """IC95 de la media por t de Student (normal para n grande)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    z = 1.959964
    m = float(x.mean())
    return m - z * se, m + z * se


# --- Paso 1: particion --------------------------------------------------------

def partir(df: pd.DataFrame) -> dict:
    fechas = np.sort(df["fecha"].unique())
    n = len(fechas)
    corte = int(n * PCT_DESARROLLO)
    dev_hasta = pd.Timestamp(fechas[corte - 1])
    lock_desde = pd.Timestamp(fechas[corte + EMBARGO_RUEDAS])
    p = {
        "ruedas_total": n,
        "dev_desde": pd.Timestamp(fechas[0]), "dev_hasta": dev_hasta,
        "dev_ruedas": corte,
        "embargo_desde": pd.Timestamp(fechas[corte]),
        "embargo_hasta": pd.Timestamp(fechas[corte + EMBARGO_RUEDAS - 1]),
        "lock_desde": lock_desde, "lock_hasta": pd.Timestamp(fechas[-1]),
        "lock_ruedas": n - corte - EMBARGO_RUEDAS,
        "fechas_dev": fechas[:corte],
    }
    return p


def verificar_particion(p: dict, forzar: bool) -> None:
    esperado = PARTICION_ESPERADA
    reales = {
        "ruedas_total": p["ruedas_total"],
        "dev_desde": str(p["dev_desde"])[:10], "dev_hasta": str(p["dev_hasta"])[:10],
        "dev_ruedas": p["dev_ruedas"],
        "lock_desde": str(p["lock_desde"])[:10], "lock_hasta": str(p["lock_hasta"])[:10],
        "lock_ruedas": p["lock_ruedas"],
    }
    difs = {k: (esperado[k], reales[k]) for k in esperado if esperado[k] != reales[k]}
    if not difs:
        log("  Particion IGUAL a la pre-registrada (sec. 9.5).")
        return
    log("  [ATENCION] la particion NO coincide con la pre-registrada:")
    for k, (esp, real) in difs.items():
        log(f"    {k}: pre-registrado {esp} | ahora {real}")
    if not forzar:
        raise SystemExit(
            "\n  El dataset cambio. Mover el lockbox despues de haberlo mirado invalida\n"
            "  la evaluacion. Actualizar PARTICION_ESPERADA y el doc a mano, o --forzar\n"
            "  si se acepta el cambio a conciencia.")
    log("  --forzar: se sigue con la particion nueva.")


# --- Paso 2: walk-forward -----------------------------------------------------

def folds_wf(fechas_dev: np.ndarray) -> list:
    folds, h = [], MIN_TRAIN_WF
    while h + HOLDOUT_WF <= len(fechas_dev):
        folds.append((h, h + HOLDOUT_WF))
        h += HOLDOUT_WF
    return folds


def walk_forward(df: pd.DataFrame, p: dict, guardar_oos: str = None) -> pd.DataFrame:
    """
    Walk-forward purgado sobre el desarrollo, los dos brazos.

    guardar_oos: ruta parquet donde dejar la prediccion FILA POR FILA de cada fold.
        Es lo que habilita analizar el resultado por sector sin volver a entrenar
        (scripts/ml/screen_sectorial_v3.py). Son predicciones fuera de muestra: cada
        fila la predijo un modelo entrenado solo con datos anteriores a su fold.
    """
    fechas = p["fechas_dev"]
    folds = folds_wf(fechas)
    log("")
    log("=" * 78)
    log(f"PASO 2 -- WALK-FORWARD PURGADO EN EL DESARROLLO ({len(folds)} folds, "
        f"holdout {HOLDOUT_WF}, embargo {EMBARGO_RUEDAS})")
    log("=" * 78)

    filas = []
    oos = []
    for i, (h, h_end) in enumerate(folds, 1):
        ho_ini, ho_fin = pd.Timestamp(fechas[h]), pd.Timestamp(fechas[h_end - 1])
        train_hasta = pd.Timestamp(fechas[h - EMBARGO_RUEDAS - 1])
        tr = df[df["fecha"] <= train_hasta]
        ho = df[(df["fecha"] >= ho_ini) & (df["fecha"] <= ho_fin)]
        y_tr, y_ho = etiquetas(tr), etiquetas(ho)
        ret_ho = ho["retorno_20d"].astype(float).to_numpy()
        base = float(y_ho.mean())

        log(f"  Fold {i}/{len(folds)}  train <= {train_hasta:%Y-%m-%d} ({len(tr):,}) | "
            f"holdout {ho_ini:%Y-%m-%d} -> {ho_fin:%Y-%m-%d} ({len(ho):,}, base {base:.3f})")

        for brazo, cols in BRAZOS.items():
            t0 = datetime.now()
            m = modelo_config(len(y_tr))
            m.fit(matriz(tr, cols), y_tr)
            prob = m.predict_proba(matriz(ho, cols))[:, 1]
            met = metricas_oos(prob, y_ho, ret_ho, base)
            met.update(fold=i, brazo=brazo, n_features=len(cols), base=round(base, 4),
                       ho_ini=ho_ini, ho_fin=ho_fin, n_train=len(tr), n_ho=len(ho),
                       exceso_decil=met["ret_decil"] - float(ret_ho.mean()),
                       segs=(datetime.now() - t0).seconds)
            filas.append(met)
            if guardar_oos:
                parte = ho[["fecha", "ticker", "sector", "retorno_20d", TARGET_COL]].copy()
                parte["fold"] = i
                parte["brazo"] = brazo
                parte["prob"] = prob
                oos.append(parte)
            log(f"      {brazo:<16} ({len(cols)} feats) AUC {met['auc']:.4f} | "
                f"decil {met['prec_decil']*100:.1f}% (base {base*100:.1f}%) | "
                f"lift {met['lift_decil']:.2f} | exceso 20d {met['exceso_decil']:+.2f} | "
                f"brier {met['brier']:.4f} | {met['segs']}s")

    if guardar_oos and oos:
        pd.concat(oos, ignore_index=True).to_parquet(guardar_oos, index=False)
        log("")
        log(f"  Predicciones fuera de muestra -> {guardar_oos}")
    return pd.DataFrame(filas)


def evaluar_wf(met: pd.DataFrame) -> dict:
    log("")
    log("-" * 78)
    log("RESUMEN DEL WALK-FORWARD")
    log("-" * 78)
    res = {}
    for brazo in BRAZOS:
        sub = met[met["brazo"] == brazo]
        res[brazo] = {
            "auc_media": float(sub["auc"].mean()),
            "auc_desvio": float(sub["auc"].std()),
            "folds_sobre_min": int((sub["auc"] > COMPUERTA["wf_auc_fold_min"]).sum()),
            "folds": len(sub),
            "lift_media": float(sub["lift_decil"].mean()),
            "exceso_medio": float(sub["exceso_decil"].mean()),
            "auc_por_fold": [round(float(v), 4) for v in sub["auc"]],
        }
        r = res[brazo]
        log(f"  {brazo:<16} AUC {r['auc_media']:.4f} +/- {r['auc_desvio']:.4f} | "
            f"folds AUC > {COMPUERTA['wf_auc_fold_min']}: {r['folds_sobre_min']}/{r['folds']} | "
            f"lift {r['lift_media']:.2f} | exceso 20d {r['exceso_medio']:+.2f}")

    a, b = res["con_estructura"], res["sin_estructura"]
    m_a = met[met["brazo"] == "con_estructura"].set_index("fold")["auc"]
    m_b = met[met["brazo"] == "sin_estructura"].set_index("fold")["auc"]
    gana = int((m_a > m_b).sum())
    delta = a["auc_media"] - b["auc_media"]

    log("")
    log("  ABLACION (compuerta 2): con estructura vs sin estructura")
    log(f"    delta AUC media {delta:+.4f} (minimo {COMPUERTA['ablacion_delta_auc_min']:+.4f}) | "
        f"le gana en {gana}/{len(m_a)} folds (minimo {COMPUERTA['ablacion_folds_min']})")
    pasa_ablacion = (delta >= COMPUERTA["ablacion_delta_auc_min"]
                     and gana >= COMPUERTA["ablacion_folds_min"])
    brazo = "con_estructura" if pasa_ablacion else "sin_estructura"
    log(f"    -> las 24 features de estructura {'ENTRAN' if pasa_ablacion else 'NO entran'}; "
        f"brazo elegido: {brazo}")

    r = res[brazo]
    pasa_wf = (r["auc_media"] >= COMPUERTA["wf_auc_media_min"]
               and r["folds_sobre_min"] >= COMPUERTA["wf_folds_min"])
    log("")
    log(f"  COMPUERTA 1 (walk-forward del brazo elegido): AUC media "
        f"{r['auc_media']:.4f} >= {COMPUERTA['wf_auc_media_min']}? "
        f"{'SI' if r['auc_media'] >= COMPUERTA['wf_auc_media_min'] else 'NO'} | "
        f"{r['folds_sobre_min']}/{r['folds']} folds > {COMPUERTA['wf_auc_fold_min']} "
        f"(min {COMPUERTA['wf_folds_min']})? "
        f"{'SI' if r['folds_sobre_min'] >= COMPUERTA['wf_folds_min'] else 'NO'}")
    log(f"  -> compuerta 1 {'PASA' if pasa_wf else 'NO PASA'}")

    return {"por_brazo": res, "ablacion": {"delta_auc": delta, "folds_gana": gana,
                                           "pasa": pasa_ablacion},
            "brazo": brazo, "pasa_wf": pasa_wf,
            "features": list(BRAZOS[brazo])}


# --- Paso 3: lockbox ----------------------------------------------------------

def evaluar_lockbox(df: pd.DataFrame, p: dict, cols: list) -> dict:
    log("")
    log("=" * 78)
    log("PASO 3 -- LOCKBOX (unica evaluacion, config ya congelada)")
    log("=" * 78)
    tr = df[df["fecha"] <= p["dev_hasta"]]
    lo = df[df["fecha"] >= p["lock_desde"]]
    y_tr, y_lo = etiquetas(tr), etiquetas(lo)
    ret = lo["retorno_20d"].astype(float).to_numpy()
    base = float(y_lo.mean())
    log(f"  train {len(tr):,} filas (<= {p['dev_hasta']:%Y-%m-%d}) | "
        f"lockbox {len(lo):,} filas ({p['lock_desde']:%Y-%m-%d} -> "
        f"{p['lock_hasta']:%Y-%m-%d}), base {base:.4f}")

    t0 = datetime.now()
    modelo = modelo_config(len(y_tr))
    modelo.fit(matriz(tr, cols), y_tr)
    prob = modelo.predict_proba(matriz(lo, cols))[:, 1]
    log(f"  entrenado en {(datetime.now() - t0).seconds}s")

    met = metricas_oos(prob, y_lo, ret, base)
    k = max(1, int(len(prob) * 0.10))
    top = np.argsort(-prob)[:k]
    exceso = ret[top] - float(ret.mean())
    ic_lo, ic_hi = ic95_media(exceso)
    decil_pp = (float(y_lo[top].mean()) - base) * 100

    log("")
    log(f"  AUC                      {met['auc']:.4f}   (minimo {COMPUERTA['lock_auc_min']})")
    log(f"  decil alto acierto       {float(y_lo[top].mean())*100:.2f}%  vs base "
        f"{base*100:.2f}%  -> {decil_pp:+.2f} pp "
        f"(minimo +{COMPUERTA['lock_decil_sobre_base_pp_min']} pp)")
    log(f"  retorno 20d decil alto   {met['ret_decil']:+.2f}%  vs universo del tramo "
        f"{float(ret.mean()):+.2f}%")
    log(f"  exceso medio             {float(exceso.mean()):+.3f}%  "
        f"IC95 [{ic_lo:+.3f}; {ic_hi:+.3f}]  (tiene que excluir el cero)")
    log(f"  brier                    {met['brier']:.4f}   (constante = {base*(1-base):.4f})")

    c1 = met["auc"] >= COMPUERTA["lock_auc_min"]
    c2 = decil_pp >= COMPUERTA["lock_decil_sobre_base_pp_min"]
    c3 = (ic_lo > 0) or (ic_hi < 0)
    pasa = bool(c1 and c2 and c3)
    log("")
    log(f"  COMPUERTA 3: AUC {'OK' if c1 else 'NO'} | decil {'OK' if c2 else 'NO'} | "
        f"IC95 del exceso excluye el cero: {'SI' if c3 else 'NO'}")
    log(f"  -> compuerta 3 {'PASA' if pasa else 'NO PASA'}")

    return {"auc": met["auc"], "base": base, "decil_acierto": float(y_lo[top].mean()),
            "decil_sobre_base_pp": decil_pp, "ret_decil": met["ret_decil"],
            "ret_universo": float(ret.mean()), "exceso_medio": float(exceso.mean()),
            "exceso_ic95": [ic_lo, ic_hi], "brier": met["brier"],
            "lift_decil": met["lift_decil"], "filas": len(lo),
            "criterios": {"auc": bool(c1), "decil": bool(c2), "exceso_ic95": bool(c3)},
            "pasa": pasa, "prob": prob, "filas_lock": lo}


# --- Paso 4-5: artefacto ------------------------------------------------------

def guardar(df, p, cols, wf, lock, forzar) -> int:
    ruta_modelo = os.path.join(DIR_MODELO_V3, ml_v2.ARCHIVO_MODELO)
    ruta_meta = os.path.join(DIR_MODELO_V3, ml_v2.ARCHIVO_META)
    if os.path.exists(ruta_meta) and not forzar:
        log(f"\n  [ERROR] ya existe {ruta_meta}. --forzar para pisarlo.")
        return 1

    log("")
    log("=" * 78)
    log("PASO 4 -- MODELO FINAL (desarrollo + lockbox)")
    log("=" * 78)
    y = etiquetas(df)
    t0 = datetime.now()
    modelo = modelo_config(len(y))
    modelo.fit(matriz(df, cols), y)
    log(f"  {len(df):,} filas en {(datetime.now() - t0).seconds}s")

    os.makedirs(DIR_MODELO_V3, exist_ok=True)
    joblib.dump(modelo, ruta_modelo, compress=3)
    sha = ml_v2.sha256_archivo(ruta_modelo)
    log(f"  -> {ruta_modelo} ({os.path.getsize(ruta_modelo) / 1e6:.1f} MB)")

    # Cortes equivalentes a los de la v1, sobre filas donde la v1 tambien es
    # fuera de muestra (posteriores a su despliegue). Igual que hizo la v2.
    cortes = {}
    post = lock["filas_lock"][lock["filas_lock"]["fecha"] > pd.Timestamp("2026-04-10")]
    if len(post) > 500:
        idx = lock["filas_lock"]["fecha"] > pd.Timestamp("2026-04-10")
        p_v1 = probs_v1(post)
        p_v3 = lock["prob"][idx.to_numpy()]
        equivalentes = ml_v2.cortes_equivalentes(p_v1, p_v3)
        cortes = {
            "v1": list(ml_v2.CORTES_V1),
            "v3": [round(float(c), 6) for c in equivalentes],
            "fracciones_v1": [round(f, 4) for f in ml_v2.fracciones_sobre(p_v1, ml_v2.CORTES_V1)],
            "fracciones_v3": [round(f, 4) for f in ml_v2.fracciones_sobre(p_v3, equivalentes)],
            "filas": int(len(post)),
            "desde": post["fecha"].min(), "hasta": post["fecha"].max(),
            "nota": ("Cortes del modelo del lockbox sobre filas posteriores al despliegue "
                     "de la v1, donde las dos son fuera de muestra. Se aplican al modelo final."),
        }
        log(f"  Cortes equivalentes a los de la v1 ({len(post):,} filas): "
            + ", ".join(f"{c:.4f}" for c in equivalentes))

    clf = construir_modelo("rf", len(y)).named_steps["clf"]
    meta = {
        "version": f"ml_v3_{date.today():%Y%m%d}",
        "creado": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": {
            "modelo": "CalibratedClassifierCV(construir_modelo('rf', n), method='isotonic', cv=3)",
            "rf": {k: getattr(clf, k) for k in ("n_estimators", "max_depth", "min_samples_leaf",
                                                "max_features", "class_weight", "random_state")},
            "label": "label_binario = retorno_20d > +1% (UMBRAL_NEUTRO)",
            "referencia": ("docs/estructura_velas.md sec. 9.5 (pre-registro) y "
                           "docs/ml_reentrenamiento.md sec. 8 (config congelada)"),
        },
        "features": list(cols),
        "columnas_cero": [c for c in FEATURE_COLS_MS if c in cols],
        "datos": {
            "origen": "features_ml JOIN features_estructura (walkforward_ml.cargar_dataset('nueva'))",
            "nota_estructura": ("features_estructura tiene swings CONFIRMADOS e invariantes; "
                                "features_market_structure (la que uso la v2) mira 10 ruedas "
                                "al futuro -> sus metricas no son comparables"),
            "desde": df["fecha"].min(), "hasta": df["fecha"].max(),
            "filas": len(df), "tickers": int(df["ticker"].nunique()),
            "base_label": round(float(df[TARGET_COL].mean()), 4),
        },
        "particion": {k: (str(v)[:10] if isinstance(v, pd.Timestamp) else v)
                      for k, v in p.items() if k != "fechas_dev"},
        "walk_forward": {"holdout": HOLDOUT_WF, "embargo": EMBARGO_RUEDAS,
                         "min_train": MIN_TRAIN_WF,
                         "por_brazo": wf["por_brazo"], "ablacion": wf["ablacion"],
                         "brazo_elegido": wf["brazo"]},
        "lockbox": {k: v for k, v in lock.items() if k not in ("prob", "filas_lock")},
        "compuerta": COMPUERTA,
        "cortes": cortes,
        "versiones": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__,
                      "joblib": joblib.__version__},
        "archivo": ml_v2.ARCHIVO_MODELO,
        "bytes": os.path.getsize(ruta_modelo),
        "sha256": sha,
    }
    with open(ruta_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=True, default=_json)
    log(f"  -> {ruta_meta}")

    # Verificacion: se carga y predice igual que en memoria (mismo guard que la v2).
    cargado, meta_leida = ml_v2.cargar_modelo_v2(directorio=DIR_MODELO_V3,
                                                 features_esperadas=list(cols))
    muestra = df.iloc[-200:]
    p1 = modelo.predict_proba(matriz(muestra, cols))[:, 1]
    p2 = cargado.predict_proba(matriz(muestra, cols))[:, 1]
    log(f"  Verificacion de carga: sha OK | max |dif| en memoria vs disco = "
        f"{np.abs(p1 - p2).max():.2e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrena y valida el modelo ML v3 (Fase 3a)")
    ap.add_argument("--dry-run", action="store_true", help="pasos 1-3, sin guardar")
    ap.add_argument("--solo-wf", action="store_true", help="solo particion y walk-forward")
    ap.add_argument("--forzar", action="store_true",
                    help="pisa models_ml_v3 y acepta una particion distinta a la pre-registrada")
    ap.add_argument("--estructura", default="nueva", choices=("nueva", "vieja"),
                    help=("CONTROL de diagnostico: 'vieja' corre los mismos folds sobre "
                          "features_market_structure (la que mira al futuro) para verificar "
                          "que el montaje del walk-forward detecta la diferencia. NO sirve "
                          "para entrenar nada: implica --solo-wf."))
    args = ap.parse_args()
    if args.estructura == "vieja":
        args.solo_wf = True

    os.makedirs(DIR_REPORTES, exist_ok=True)
    log("=" * 78)
    log("ML v3 -- Tarea 23 Fase 3a (pre-registro: docs/estructura_velas.md sec. 9.5)")
    log("=" * 78)

    if args.estructura == "vieja":
        log("  [CONTROL] estructura VIEJA (mira al futuro): solo diagnostico, no entrena.")
    df = cargar_dataset(args.estructura)
    log(f"  Dataset: {len(df):,} filas | {df['ticker'].nunique()} tickers | "
        f"{df['fecha'].min():%Y-%m-%d} -> {df['fecha'].max():%Y-%m-%d} | "
        f"base {df[TARGET_COL].mean():.4f}")

    p = partir(df)
    log(f"  Desarrollo : {p['dev_ruedas']} ruedas  {p['dev_desde']:%Y-%m-%d} -> "
        f"{p['dev_hasta']:%Y-%m-%d}")
    log(f"  Embargo    : {EMBARGO_RUEDAS} ruedas  {p['embargo_desde']:%Y-%m-%d} -> "
        f"{p['embargo_hasta']:%Y-%m-%d}")
    log(f"  Lockbox    : {p['lock_ruedas']} ruedas  {p['lock_desde']:%Y-%m-%d} -> "
        f"{p['lock_hasta']:%Y-%m-%d}")
    verificar_particion(p, args.forzar)

    met = walk_forward(df, p, guardar_oos=os.path.join(
        DIR_REPORTES, f"wf_oos_{args.estructura}.parquet"))
    met.drop(columns=["ho_ini", "ho_fin"]).to_csv(
        os.path.join(DIR_REPORTES, f"wf_folds_{args.estructura}.csv"), index=False)
    wf = evaluar_wf(met)

    if args.solo_wf:
        log("\n  --solo-wf: no se toca el lockbox.")
        return 0 if wf["pasa_wf"] else 1

    if not wf["pasa_wf"]:
        log("\n  La compuerta 1 no pasa: NO se abre el lockbox (se puede mirar una vez).")
        log("  Documentar el resultado en docs/estructura_velas.md sec. 9.5 y parar.")
        return 1

    lock = evaluar_lockbox(df, p, wf["features"])
    with open(os.path.join(DIR_REPORTES, "lockbox_v3.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in lock.items() if k not in ("prob", "filas_lock")},
                  fh, indent=2, default=_json)

    if not lock["pasa"]:
        log("\n  RESULTADO: la v3 NO pasa el lockbox. No hay modelo para desplegar.")
        log("  Queda documentado; no se reabren los parametros para hacerla pasar.")
        return 1

    log("\n  RESULTADO: la v3 PASA las tres compuertas.")
    if args.dry_run:
        log("  [DRY RUN] no se entrena el modelo final ni se guarda nada.")
        return 0
    return guardar(df, p, wf["features"], wf, lock, args.forzar)


if __name__ == "__main__":
    sys.exit(main())
