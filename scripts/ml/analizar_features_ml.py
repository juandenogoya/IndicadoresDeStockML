"""
analizar_features_ml.py
Tarea 23: las mediciones de docs/features_ml.md, reproducibles. Solo LEE (DB local +
las predicciones fuera de muestra del walk-forward). No entrena nada salvo en la
seccion `ablacion`, que re-corre los 6 folds y no guarda modelos.

POR QUE EXISTE
    El modelo v3 no pasa la compuerta (AUC 0,51). Antes de pensar modelos nuevos hacia
    falta saber que features usa, cuales aportan, que datos hay en la DB, cuanta
    historia tiene cada fuente y si el label mide lo que se cree. Estas mediciones lo
    responden y se vuelven a correr cuando cambian las features o el dataset.

SECCIONES (--seccion; por defecto todas MENOS ablacion)
    inventario   tablas de la DB: filas, columna de fecha, rango, ruedas, tickers  (sec. 5)
    historia     arranque de la serie de precio por ticker + dataset por anio     (sec. 6)
    cobertura    % de filas del dataset con dato en cada fuente candidata         (sec. 7)
    redundancia  relaciones funcionales exactas + pares |Spearman| >= 0,80        (sec. 4)
    folds        composicion de cada fold, base rate por trimestre (absoluto vs
                 relativo) y AUC del modelo juzgado con los dos labels            (sec. 8-9)
    valor        point-in-time de los multiplos, descomposicion del hueco del PER
                 y senal univariada de la familia valor, label absoluto y relativo (sec. 10)
    ablacion     quitar cada familia del set de 53 y re-correr los MISMOS 6 folds
                 purgados (sec. 3). ~20 min: solo si se pide explicitamente.

LA UNIDAD DE INDEPENDENCIA ES EL FOLD
    Los IC95 se calculan sobre las mediciones por fold (t de Student con n-1 grados),
    no sobre filas: dentro de un fold las ~15.000 filas comparten el mismo mercado.

Uso:
    python scripts/ml/analizar_features_ml.py
    python scripts/ml/analizar_features_ml.py --seccion valor
    python scripts/ml/analizar_features_ml.py --seccion ablacion
"""

import argparse
import math
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, AQUI)

# LOCAL es la fuente de verdad: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from scipy.stats import t as t_student  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.database import query_df  # noqa: E402
from src.indicators.market_structure import FEATURE_COLS_MS  # noqa: E402
from src.ml.trainer import FEATURE_COLS  # noqa: E402

OOS_DEFAULT = os.path.join(ROOT, "reportes", "ml_v3", "wf_oos_nueva.parquet")
MIN_FILAS_FOLD = 300

COLS_53 = list(FEATURE_COLS) + list(FEATURE_COLS_MS)

# Familias para la ablacion y la redundancia. Los nombres son los de docs/features_ml.md.
GRUPOS = {
    "indicadores": ["rsi14", "macd_hist", "adx", "vol_relativo"],
    "dist_smas": ["dist_sma21", "dist_sma50", "dist_sma200"],
    "engineered": ["bb_posicion", "atr14_pct", "momentum_pct"],
    "scoring": ["score_ponderado", "condiciones_ok", "cond_rsi", "cond_macd",
                "cond_sma21", "cond_sma50", "cond_sma200", "cond_momentum"],
    "sector_z": ["z_rsi_sector", "z_retorno_1d_sector", "z_retorno_5d_sector",
                 "z_vol_sector", "z_dist_sma50_sector", "z_adx_sector"],
    "sector_ctx": ["pct_long_sector", "rank_retorno_sector", "rsi_sector_avg",
                   "adx_sector_avg", "retorno_1d_sector_avg"],
    "estructura": list(FEATURE_COLS_MS),
    "flags_estruct": [c for c in FEATURE_COLS_MS if c.startswith(("is_", "bos_", "choch_"))],
}

# El dataset tal como lo arma walkforward_ml.cargar_dataset("nueva").
_SQL_DATASET = """
    FROM features_ml fm
    JOIN features_estructura fe ON fm.ticker = fe.ticker AND fm.fecha = fe.fecha
    WHERE fm.label_binario IS NOT NULL
"""


def log(msg=""):
    print(msg, flush=True)


def titulo(txt):
    log("")
    log("=" * 100)
    log(txt)
    log("=" * 100)


def ic95_folds(valores) -> tuple:
    """IC95 de la media de las mediciones POR FOLD, t de Student con n-1 grados."""
    v = np.asarray([x for x in valores if not np.isnan(x)], dtype=float)
    if len(v) < 3:
        return float("nan"), float("nan")
    se = float(v.std(ddof=1) / math.sqrt(len(v)))
    t = float(t_student.ppf(0.975, len(v) - 1))
    return float(v.mean()) - t * se, float(v.mean()) + t * se


def cargar_oos(ruta: str) -> pd.DataFrame:
    """Predicciones fuera de muestra del brazo de 53 features + label relativo."""
    if not os.path.exists(ruta):
        raise SystemExit(f"[ERROR] falta {ruta}. Generarlo con:\n"
                         "  python scripts/ml/entrenar_ml_v3.py --solo-wf")
    oos = pd.read_parquet(ruta)
    oos = oos[oos["brazo"] == "con_estructura"].copy()
    oos["fecha"] = pd.to_datetime(oos["fecha"])
    oos["retorno_20d"] = pd.to_numeric(oos["retorno_20d"], errors="coerce")
    oos["label_abs"] = oos["label_binario"].astype(int)
    # LABEL RELATIVO: el retorno del ticker contra la MEDIANA del universo de esa rueda.
    med = oos.groupby("fecha")["retorno_20d"].transform("median")
    oos["label_rel"] = (oos["retorno_20d"] > med).astype(int)
    return oos


# --- inventario ---------------------------------------------------------------

_CAND_FECHA = ["fecha", "fecha_datos", "precio_fecha", "scan_fecha", "fecha_snapshot",
               "fecha_semana", "period_end", "fiscal_period_end", "execution_date",
               "earnings_date", "fecha_entrada", "inicio"]


def seccion_inventario():
    titulo("INVENTARIO DE TABLAS -- DB LOCAL (docs/features_ml.md sec. 5)")
    tabs = query_df("""
        SELECT c.relname AS tabla, GREATEST(c.reltuples::bigint, 0) AS filas_est,
               pg_total_relation_size(c.oid) AS bytes
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r' ORDER BY c.relname""")
    cols = query_df("""SELECT table_name AS tabla, column_name AS col
                       FROM information_schema.columns WHERE table_schema = 'public'""")
    log(f"  {'tabla':<40} {'filas~':>10} {'MB':>7} {'col':>4} {'fecha':<18} "
        f"{'desde':<11} {'hasta':<11} {'ruedas':>7} {'tick':>5}")
    for _, r in tabs.iterrows():
        t = r["tabla"]
        nombres = set(cols.loc[cols["tabla"] == t, "col"])
        fcol = next((c for c in _CAND_FECHA if c in nombres), None)
        tiene_tk = "ticker" in nombres
        mn = mx = ruedas = tick = "-"
        if fcol:
            d = query_df(f"SELECT MIN({fcol})::text mn, MAX({fcol})::text mx FROM {t}").iloc[0]
            mn, mx = str(d["mn"])[:10], str(d["mx"])[:10]
            if int(r["filas_est"]) < 3_000_000:
                sel = f"COUNT(DISTINCT {fcol}) nf" + (", COUNT(DISTINCT ticker) nt" if tiene_tk else "")
                d2 = query_df(f"SELECT {sel} FROM {t}").iloc[0]
                ruedas = str(int(d2["nf"]))
                tick = str(int(d2["nt"])) if tiene_tk else "-"
        log(f"  {t:<40} {int(r['filas_est']):>10,} {r['bytes'] / 1048576:>7.1f} "
            f"{len(nombres):>4} {str(fcol or '-'):<18} {mn:<11} {mx:<11} {ruedas:>7} {tick:>5}")
    log(f"  TOTAL: {len(tabs)} tablas (filas~ = estimacion de pg_class)")


# --- historia -----------------------------------------------------------------

def seccion_historia():
    titulo("HISTORIA REAL POR TICKER -- el techo del dataset (sec. 6)")
    h = query_df("SELECT ticker, COUNT(*) n FROM precios_diarios GROUP BY ticker")
    log(f"  {len(h)} tickers | ruedas: min {h['n'].min()} / mediana {int(h['n'].median())} "
        f"/ max {h['n'].max()}")
    for u in (1600, 1200, 1000, 500):
        log(f"    tickers con >= {u:>4} ruedas: {int((h['n'] >= u).sum()):>3}")
    ini = query_df("""SELECT to_char(date_trunc('quarter', f), 'YYYY-MM') tri, COUNT(*) n
                      FROM (SELECT ticker, MIN(fecha) f FROM precios_diarios GROUP BY ticker) x
                      GROUP BY 1 ORDER BY 1""")
    log("  arranque de la serie de cada ticker:")
    acum = 0
    for _, x in ini.iterrows():
        acum += int(x["n"])
        log(f"    {x['tri']}  +{int(x['n']):>3}  (acumulado {acum:>3})")

    d = query_df(f"""SELECT COUNT(*) n, COUNT(DISTINCT fm.fecha) r, COUNT(DISTINCT fm.ticker) t,
                            MIN(fm.fecha)::text mn, MAX(fm.fecha)::text mx {_SQL_DATASET}""").iloc[0]
    log("")
    log(f"  DATASET: {int(d['n']):,} filas | {int(d['r']):,} ruedas | {int(d['t'])} tickers | "
        f"{d['mn']} -> {d['mx']}")
    a = query_df(f"""SELECT EXTRACT(YEAR FROM fm.fecha)::int anio, COUNT(*) n,
                            COUNT(DISTINCT fm.fecha) r {_SQL_DATASET} GROUP BY 1 ORDER BY 1""")
    log(f"  {'anio':<6} {'filas':>9} {'ruedas':>7} {'tickers/rueda':>14}")
    for _, x in a.iterrows():
        log(f"  {int(x['anio']):<6} {int(x['n']):>9,} {int(x['r']):>7} {x['n'] / max(x['r'], 1):>14.0f}")


# --- cobertura ----------------------------------------------------------------

# (tabla, granularidad, columna de fecha). "t" = por ticker y fecha, "f" = solo fecha.
FUENTES = [
    ("features_precio_accion", "t", "fecha"),
    ("features_velas", "t", "fecha"),
    ("ticker_zscore_diario", "t", "fecha"),
    ("fundamentales_sec_multiplos_d", "t", "fecha"),
    ("opciones_zscore_diario", "t", "fecha"),
    ("alertas_scanner", "t", "precio_fecha"),
    ("features_regimen_macro", "f", "fecha"),
]


def seccion_cobertura():
    titulo("COBERTURA DE CADA FUENTE SOBRE LAS FILAS DEL DATASET (sec. 7)")
    log(f"  {'fuente':<34} {'% filas':>9} {'desde':<12}")
    for tabla, gran, col in FUENTES:
        if gran == "t":
            # DISTINCT: tablas con varias filas por (ticker, fecha) inflarian el % (ej. plazos)
            join = f"""LEFT JOIN (SELECT DISTINCT ticker, {col} FROM {tabla}) s
                       ON s.ticker = fm.ticker AND s.{col} = fm.fecha"""
        else:
            join = f"""LEFT JOIN (SELECT DISTINCT {col} FROM {tabla}) s ON s.{col} = fm.fecha"""
        x = query_df(f"""SELECT COUNT(*) n, COUNT(s.{col}) m FROM features_ml fm
                         JOIN features_estructura fe ON fm.ticker = fe.ticker AND fm.fecha = fe.fecha
                         {join} WHERE fm.label_binario IS NOT NULL""").iloc[0]
        desde = query_df(f"SELECT MIN({col})::text d FROM {tabla}").iloc[0]["d"]
        log(f"  {tabla:<34} {100.0 * int(x['m']) / max(int(x['n']), 1):>8.1f}% {str(desde)[:10]:<12}")

    e = query_df(f"""
        WITH ds AS (SELECT fm.ticker, fm.fecha {_SQL_DATASET})
        SELECT COUNT(*) n,
          SUM(CASE WHEN EXISTS (SELECT 1 FROM earnings_historico eh WHERE eh.ticker = ds.ticker
               AND eh.announcement_date <= ds.fecha) THEN 1 ELSE 0 END) prev,
          SUM(CASE WHEN EXISTS (SELECT 1 FROM earnings_historico eh WHERE eh.ticker = ds.ticker
               AND eh.announcement_date > ds.fecha) THEN 1 ELSE 0 END) prox
        FROM ds""").iloc[0]
    log(f"  {'earnings_historico (balance anterior)':<34} {100.0 * int(e['prev']) / int(e['n']):>8.1f}%")
    log(f"  {'earnings_historico (balance proximo)':<34} {100.0 * int(e['prox']) / int(e['n']):>8.1f}%"
        "   <- look-ahead suave: usar ACOTADO")

    f = query_df("""SELECT ticker, COUNT(*) n, MIN(fecha)::text mn FROM futuros_diarios
                    GROUP BY ticker ORDER BY ticker""")
    log("")
    log("  futuros_diarios (materia prima de features_regimen_macro):")
    for _, x in f.iterrows():
        log(f"    {x['ticker']:<8} {int(x['n']):>5} ruedas desde {x['mn']}")


# --- redundancia --------------------------------------------------------------

def seccion_redundancia():
    from walkforward_ml import cargar_dataset
    titulo("REDUNDANCIA ENTRE LAS 53 (sec. 4)")
    df = cargar_dataset("nueva")
    pruebas = {
        "cond_sma21 == (dist_sma21 > 0)": df.cond_sma21 == (df.dist_sma21 > 0).astype(float),
        "cond_sma50 == (dist_sma50 > 0)": df.cond_sma50 == (df.dist_sma50 > 0).astype(float),
        "cond_sma200 == (dist_sma200 > 0)": df.cond_sma200 == (df.dist_sma200 > 0).astype(float),
        "cond_macd == (macd_hist > 0)": df.cond_macd == (df.macd_hist > 0).astype(float),
        "cond_momentum == (momentum_pct > 0)": df.cond_momentum == (df.momentum_pct > 0).astype(float),
        "condiciones_ok == suma de los 6 cond_*": df.condiciones_ok == df[
            ["cond_rsi", "cond_macd", "cond_sma21", "cond_sma50", "cond_sma200", "cond_momentum"]
        ].sum(axis=1),
    }
    log("  relaciones funcionales exactas (% de filas en que se cumplen):")
    for k, v in pruebas.items():
        log(f"    {100.0 * v.mean():6.2f}%  {k}")
    c = df[COLS_53].corr(method="spearman")
    pares = [(c.loc[a, b], a, b) for i, a in enumerate(COLS_53) for b in COLS_53[i + 1:]
             if abs(c.loc[a, b]) >= 0.80]
    log("")
    log(f"  pares con |Spearman| >= 0,80: {len(pares)}")
    for r, a, b in sorted(pares, key=lambda x: -abs(x[0]))[:25]:
        log(f"    {r:+.3f}  {a:<22} {b}")


# --- folds y label ------------------------------------------------------------

def seccion_folds(ruta_oos: str):
    oos = cargar_oos(ruta_oos)
    titulo("COMPOSICION DE LOS FOLDS (sec. 8)")
    log(f"  {'fold':<5} {'filas':>8} {'tickers':>8} {'desde':<12} {'hasta':<12} {'base':>6} {'AUC':>7}")
    for f, g in oos.groupby("fold"):
        log(f"  {f:<5} {len(g):>8,} {g['ticker'].nunique():>8} {str(g['fecha'].min())[:10]:<12} "
            f"{str(g['fecha'].max())[:10]:<12} {g['label_abs'].mean():>6.3f} "
            f"{roc_auc_score(g['label_abs'], g['prob']):>7.4f}")

    titulo("BASE RATE POR TRIMESTRE: label ABSOLUTO vs RELATIVO (sec. 9)")
    q = oos.assign(t=oos["fecha"].dt.to_period("Q").astype(str)).groupby("t").agg(
        n=("label_abs", "size"), a=("label_abs", "mean"), r=("label_rel", "mean"))
    log(f"  {'trimestre':<10} {'n':>8} {'absoluto':>9} {'relativo':>9}")
    for k, x in q.iterrows():
        log(f"  {k:<10} {int(x['n']):>8,} {x['a']:>9.3f} {x['r']:>9.3f}")
    log(f"  desvio entre trimestres: absoluto {q['a'].std():.4f} | relativo {q['r'].std():.4f}")

    titulo("LA MISMA PROBABILIDAD JUZGADA CON LOS DOS LABELS (sec. 9.1; diagnostico)")
    aa, ar = [], []
    for f, g in oos.groupby("fold"):
        aa.append(roc_auc_score(g["label_abs"], g["prob"]))
        ar.append(roc_auc_score(g["label_rel"], g["prob"]))
        log(f"  fold {f}  vs absoluto {aa[-1]:.4f}  vs relativo {ar[-1]:.4f}")
    for nom, v in (("absoluto", aa), ("relativo", ar)):
        lo, hi = ic95_folds(v)
        log(f"  media vs {nom:<9} {np.mean(v):.4f}  IC95 [{lo:.4f}; {hi:.4f}]  "
            f"{sum(1 for x in v if x > 0.5)}/6 folds > 0,50")
    log("  El modelo se ENTRENO con el label absoluto: esto es la misma prediccion")
    log("  medida con otra pregunta, no un modelo con label relativo.")


# --- familia valor ------------------------------------------------------------

def seccion_valor(ruta_oos: str):
    titulo("VALUACION: POINT-IN-TIME de fundamentales_sec_multiplos_d (sec. 10.1)")
    q = query_df("""SELECT COUNT(*) n,
        SUM(CASE WHEN filed_primero > fecha THEN 1 ELSE 0 END) futuro,
        SUM(CASE WHEN period_end > fecha THEN 1 ELSE 0 END) per_futuro,
        MIN(lag_dias) lmin, AVG(lag_dias) lavg, MAX(lag_dias) lmax
        FROM fundamentales_sec_multiplos_d""").iloc[0]
    log(f"  filas {int(q['n']):,} | filed_primero > fecha: {int(q['futuro'])} | "
        f"period_end > fecha: {int(q['per_futuro'])} | lag min {int(q['lmin'])} / medio "
        f"{float(q['lavg']):.0f} / max {int(q['lmax'])} dias")

    titulo("POR QUE FALTA EL PER -- descomposicion del hueco (sec. 10.2)")
    d = query_df(f"""
        WITH ds AS (SELECT fm.ticker, fm.fecha {_SQL_DATASET}),
             sec AS (SELECT DISTINCT ticker FROM fundamentales_sec_q)
        SELECT COUNT(*) n,
          SUM(CASE WHEN m.pe_ratio IS NOT NULL THEN 1 ELSE 0 END) con_per,
          SUM(CASE WHEN s.ticker IS NULL THEN 1 ELSE 0 END) sin_sec,
          SUM(CASE WHEN s.ticker IS NOT NULL AND m.ticker IS NULL THEN 1 ELSE 0 END) sin_fila,
          SUM(CASE WHEN m.ticker IS NOT NULL AND m.pe_ratio IS NULL
                   AND m.net_income_ttm <= 0 THEN 1 ELSE 0 END) perdida,
          SUM(CASE WHEN m.ticker IS NOT NULL AND m.pe_ratio IS NULL
                   AND (m.net_income_ttm > 0 OR m.net_income_ttm IS NULL) THEN 1 ELSE 0 END) otro
        FROM ds LEFT JOIN sec s ON s.ticker = ds.ticker
                LEFT JOIN fundamentales_sec_multiplos_d m ON m.ticker = ds.ticker AND m.fecha = ds.fecha
        """).iloc[0]
    n = int(d["n"])
    for k, t in (("con_per", "con PER"), ("sin_sec", "ticker sin fuente SEC (no-USA)"),
                 ("sin_fila", "tiene SEC pero no hay fila ese dia"),
                 ("perdida", "empresa con PERDIDA (PER indefinido)"),
                 ("otro", "gana plata y aun asi falta")):
        log(f"  {t:<40} {int(d[k]):>8,} ({100.0 * int(d[k]) / n:>5.1f}%)")
    s = query_df(f"""
        WITH ds AS (SELECT fm.ticker, fm.fecha, a.sector {_SQL_DATASET.replace(
            'WHERE', 'LEFT JOIN activos a ON a.ticker = fm.ticker WHERE')})
        SELECT ds.sector, 100.0 * COUNT(m.pe_ratio) / COUNT(*) cob
        FROM ds LEFT JOIN fundamentales_sec_multiplos_d m ON m.ticker = ds.ticker AND m.fecha = ds.fecha
        GROUP BY 1 ORDER BY 2""")
    log("  cobertura por sector: " + " | ".join(
        f"{x['sector']} {float(x['cob']):.1f}%" for _, x in s.iterrows()))

    titulo("SENAL UNIVARIADA DE LA FAMILIA VALOR, 6 folds (sec. 10.3-10.4)")
    oos = cargar_oos(ruta_oos)
    m = query_df("""SELECT ticker, fecha, pe_ratio, ps_ratio, ev_ebitda, fcf_yield, pe_pct,
                      net_income_ttm, market_cap, revenue_ttm
                    FROM fundamentales_sec_multiplos_d WHERE fecha >= :d""",
                 params={"d": str(oos["fecha"].min())[:10]})
    m["fecha"] = pd.to_datetime(m["fecha"])
    for c in m.columns[2:]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    df = oos.merge(m, on=["ticker", "fecha"], how="left")
    mc = df["market_cap"].replace(0, np.nan)
    df["per_inv"] = 1.0 / df["pe_ratio"].replace(0, np.nan)
    df["earnings_yield"] = df["net_income_ttm"] / mc      # definido tambien con perdida
    df["sales_yield"] = df["revenue_ttm"] / mc
    for b in ("earnings_yield", "ps_ratio", "fcf_yield", "ev_ebitda"):
        df[f"x_{b}"] = df.groupby("fecha")[b].rank(pct=True)   # percentil TRANSVERSAL

    variables = [("pe_ratio", "PER crudo"), ("per_inv", "1/PER"),
                 ("earnings_yield", "earnings yield"),
                 ("x_earnings_yield", "earnings yield percentil transversal"),
                 ("pe_pct", "percentil del PER vs su propia historia"),
                 ("ps_ratio", "P/S"), ("sales_yield", "sales yield"),
                 ("x_ps_ratio", "P/S percentil transversal"),
                 ("fcf_yield", "FCF yield"), ("x_fcf_yield", "FCF yield percentil transversal"),
                 ("ev_ebitda", "EV/EBITDA"), ("x_ev_ebitda", "EV/EBITDA percentil transversal")]
    comparaciones = 0
    for label in ("label_abs", "label_rel"):
        log("")
        log(f"  contra {label}:")
        log(f"  {'variable':<42} {'cob':>6} {'AUC':>7} {'IC95 por fold':>18} {'>0,50':>6} "
            f"{'IC Spearman':>12} {'modelo':>7}")
        for col, nombre in variables:
            aucs, ics, cobs, mod = [], [], [], []
            for _, g in df.groupby("fold"):
                s = g.dropna(subset=[col])
                cobs.append(len(s) / len(g))
                if len(s) < MIN_FILAS_FOLD or s[label].nunique() < 2:
                    continue
                v = s[col].astype(float).to_numpy()
                aucs.append(roc_auc_score(s[label], v))
                mod.append(roc_auc_score(s[label], s["prob"]))   # el modelo en las MISMAS filas
                ics.append(spearmanr(v, s["retorno_20d"].astype(float)).correlation)
            if not aucs:
                continue
            comparaciones += 1
            lo, hi = ic95_folds(aucs)
            log(f"  {nombre:<42} {100 * np.mean(cobs):>5.1f}% {np.mean(aucs):>7.4f} "
                f"[{lo:>7.4f};{hi:>7.4f}] {sum(1 for a in aucs if a > 0.5):>3}/{len(aucs)} "
                f"{np.mean(ics):>+12.4f} {np.mean(mod):>7.4f}")
    p_azar = 1 - (1 - 1 / 64) ** comparaciones
    log("")
    log(f"  {comparaciones} comparaciones: la probabilidad de que al menos una saque 6/6 folds")
    log(f"  por AZAR es {100 * p_azar:.0f}%. Lo que sobrevive a un screen es una hipotesis a")
    log("  pre-registrar, no un hallazgo.")


# --- ablacion -----------------------------------------------------------------

def seccion_ablacion():
    """Quitar cada familia y re-correr los 6 folds con la config congelada. ~20 min."""
    from entrenar_ml_v3 import (EMBARGO_RUEDAS, etiquetas, folds_wf, matriz,
                                modelo_config, partir)
    from walkforward_ml import cargar_dataset

    titulo("ABLACION POR FAMILIA -- quitar el grupo del set de 53 (sec. 3)")
    df = cargar_dataset("nueva")
    p = partir(df)
    fechas = p["fechas_dev"]
    folds = folds_wf(fechas)

    def correr(cols, etiqueta):
        aucs = []
        for h, h_end in folds:
            train_hasta = pd.Timestamp(fechas[h - EMBARGO_RUEDAS - 1])
            ini, fin = pd.Timestamp(fechas[h]), pd.Timestamp(fechas[h_end - 1])
            tr = df[df["fecha"] <= train_hasta]
            ho = df[(df["fecha"] >= ini) & (df["fecha"] <= fin)]
            m = modelo_config(len(tr))
            m.fit(matriz(tr, cols), etiquetas(tr))
            aucs.append(float(roc_auc_score(etiquetas(ho), m.predict_proba(matriz(ho, cols))[:, 1])))
        log(f"  {etiqueta:<22} {len(cols):>2} feats | AUC {np.mean(aucs):.4f} | "
            f"folds {' '.join(f'{a:.3f}' for a in aucs)}")
        return float(np.mean(aucs))

    base = correr(COLS_53, "TODAS (base)")
    efecto = {}
    for g, cs in GRUPOS.items():
        efecto[g] = base - correr([c for c in COLS_53 if c not in set(cs)], f"sin {g}")
    log("")
    log("  APORTE = AUC base - AUC sin la familia (positivo = aporta)")
    for g, e in sorted(efecto.items(), key=lambda x: -x[1]):
        log(f"    {g:<16} {e:+.4f}")


SECCIONES = ["inventario", "historia", "cobertura", "redundancia", "folds", "valor", "ablacion"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Mediciones de docs/features_ml.md (solo lee)")
    ap.add_argument("--seccion", choices=SECCIONES + ["todas"], default="todas",
                    help="'todas' corre todo MENOS ablacion (~20 min, pedirla aparte)")
    ap.add_argument("--oos", default=OOS_DEFAULT,
                    help="parquet de predicciones fuera de muestra del walk-forward")
    args = ap.parse_args()

    elegidas = [s for s in SECCIONES if s != "ablacion"] if args.seccion == "todas" else [args.seccion]
    for s in elegidas:
        if s in ("folds", "valor"):
            globals()[f"seccion_{s}"](args.oos)
        else:
            globals()[f"seccion_{s}"]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
