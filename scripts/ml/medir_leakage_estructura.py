"""
medir_leakage_estructura.py -- Tarea 23: cuanto mira al futuro la historia de
features_market_structure, cuanto le cambia eso a los modelos ML y que valor tienen
los eventos de estructura con swings CONFIRMADOS.

Reproduce docs/estructura_velas.md sec. 4 y 6, y es la medicion de la compuerta de
la Fase 2 (secciones F y G). Solo lee la DB LOCAL. Salidas en reportes/estructura_velas/
(gitignoreado).

Secciones:
  A-C  Panel "guardada vs lo que se sabia ese dia": para una muestra de tickers y
       ruedas, recalcula market_structure._calcular_ticker con las 500 barras hasta
       cada rueda (lo que ve el scanner) y lo compara con la fila guardada.
       Discrepancia por columna, distribucion de dias_sh/dias_sl y retorno forward
       en exceso. Es un loop por rueda: ~13 s por ticker con 780 ruedas.
  D    Modelos ML v1 (V3 global) y v2 sobre las mismas filas de features_ml,
       cambiando solo las 24 features de estructura. AUC y decil alto por tramo.
  E    Semanal: estructura_10 de la ultima semana cerrada vs la confirmada.
  F    Valor de los eventos de estructura con swings confirmados
       (src/indicators/estructura.py) sobre todo el universo. Con el modulo nuevo la
       historia es invariante: se calcula una sola vez, sin loop.
  G    Regla de entrada de FT_SMC (condiciones obligatorias de
       ft_scoring.calcular_score_estructura) con la historia vieja y con la nueva
       (features_estructura). Compuerta de la Fase 2.

Resultado de referencia (17/9/2026, 60 tickers x 780 ruedas):
  is_sh_10 = 1 guardada -4,72% de exceso a 5 ruedas, lo que se sabia ese dia -0,07%.
  AUC v1 0,655 guardada / 0,520 real (2023-08 -> 2025-01, fuera de su entrenamiento).

Uso (desde la raiz):
    python scripts/ml/medir_leakage_estructura.py                       # A-G (~16 min)
    python scripts/ml/medir_leakage_estructura.py --tickers 5 --ruedas 120
    python scripts/ml/medir_leakage_estructura.py --reusar-panel        # A-D sin recalcular
    python scripts/ml/medir_leakage_estructura.py --solo FG --desde 2021-06-01
"""

import argparse
import os
import pickle
import random
import sys
import time
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
from src.indicators import estructura  # noqa: E402
from src.indicators.market_structure import FEATURE_COLS_MS, _calcular_ticker  # noqa: E402

SALIDA = os.path.join(ROOT, "reportes", "estructura_velas")
PANEL = os.path.join(SALIDA, "panel_guardada_vs_pit.pkl")
VENTANA_SCANNER = 500          # data_manager.preparar_ticker carga 500 barras
MIN_RUEDAS_TICKER = 1390


# -- utilidades ----------------------------------------------------------------------

def exceso_por_dia(df: pd.DataFrame, mask, h: str):
    """Media por dia del exceso y su IC95 (los tickers del mismo dia no son independientes)."""
    por_dia = df.loc[mask].groupby("fecha")[h].mean().dropna()
    if len(por_dia) < 20:
        return None
    m = por_dia.mean()
    se = por_dia.std(ddof=1) / np.sqrt(len(por_dia))
    return m, m - 1.96 * se, m + 1.96 * se


def linea_exceso(df, mask, nombre, horizontes=("fwd_5_ex", "fwd_20_ex")):
    txt = f"   {nombre:36s} n={int(np.sum(mask)):7,}"
    for h in horizontes:
        r = exceso_por_dia(df, mask, h)
        if r is None:
            txt += f"  {h}: pocos dias"
        else:
            txt += f"  {h}={r[0]*100:+.2f}% [{r[1]*100:+.2f};{r[2]*100:+.2f}]"
    print(txt)


def agregar_forward(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "fecha"])
    for h in (5, 20):
        df[f"fwd_{h}"] = df.groupby("ticker")["close"].shift(-h) / df["close"] - 1
        df[f"fwd_{h}_ex"] = df[f"fwd_{h}"] - df.groupby("fecha")[f"fwd_{h}"].transform("mean")
    return df


# -- A-C: panel guardada vs point-in-time ------------------------------------------------

def construir_panel(n_tickers: int, n_ruedas: int, semilla: int) -> dict:
    tick = query_df(f"""
        SELECT ticker FROM precios_diarios GROUP BY ticker HAVING COUNT(*) >= {MIN_RUEDAS_TICKER}
    """)
    todos = sorted(tick["ticker"].tolist())
    random.seed(semilla)
    muestra = sorted(random.sample(todos, min(n_tickers, len(todos))))
    print(f"tickers con >= {MIN_RUEDAS_TICKER} ruedas: {len(todos)} | muestra: {len(muestra)}")

    ph = ", ".join(f"'{t}'" for t in muestra)
    precios = query_df(f"""
        SELECT ticker, fecha, open, high, low, close, volume FROM precios_diarios
        WHERE ticker IN ({ph}) AND close > 0 AND high > 0 AND low > 0 AND open > 0
        ORDER BY ticker, fecha
    """)
    precios["fecha"] = pd.to_datetime(precios["fecha"])
    guardada = query_df(f"""
        SELECT ticker, fecha, {", ".join(FEATURE_COLS_MS)} FROM features_market_structure
        WHERE ticker IN ({ph})
    """)
    guardada["fecha"] = pd.to_datetime(guardada["fecha"])

    filas, t0 = [], time.time()
    for k, (tk, g) in enumerate(precios.groupby("ticker", sort=True)):
        g = g.reset_index(drop=True)
        for i in range(max(VENTANA_SCANNER, len(g) - n_ruedas), len(g)):
            sub = g.iloc[i + 1 - VENTANA_SCANNER: i + 1][
                ["fecha", "open", "high", "low", "close", "volume"]]
            r = _calcular_ticker(sub).iloc[-1]
            d = {"ticker": tk, "fecha": r["fecha"]}
            d.update({c: r[c] for c in FEATURE_COLS_MS})
            filas.append(d)
        print(f"  {k + 1}/{len(muestra)} {tk} ({time.time() - t0:.0f}s)", flush=True)

    panel = pd.DataFrame(filas).merge(guardada, on=["ticker", "fecha"], how="left",
                                      suffixes=("_pit", "_sto"))
    panel = panel.merge(precios[["ticker", "fecha", "close"]], on=["ticker", "fecha"], how="left")
    obj = {"panel": panel, "muestra": muestra, "semilla": semilla}
    os.makedirs(SALIDA, exist_ok=True)
    with open(PANEL, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"panel {len(panel):,} filas -> {PANEL}")
    return obj


def seccion_a_c(panel: pd.DataFrame) -> None:
    precios = query_df(f"""
        SELECT ticker, fecha, close FROM precios_diarios
        WHERE ticker IN ({", ".join(f"'{t}'" for t in panel.ticker.unique())})
    """)
    precios["fecha"] = pd.to_datetime(precios["fecha"])
    fwd = precios.sort_values(["ticker", "fecha"]).copy()
    for h in (5, 20):
        fwd[f"fwd_{h}"] = fwd.groupby("ticker")["close"].shift(-h) / fwd["close"] - 1
    mad = panel.drop(columns=["close"]).merge(fwd, on=["ticker", "fecha"], how="left")
    mad = mad[mad["fwd_20"].notna()].copy()
    for h in (5, 20):
        mad[f"fwd_{h}_ex"] = mad[f"fwd_{h}"] - mad.groupby("fecha")[f"fwd_{h}"].transform("mean")
    print(f"\nfilas maduras (con retorno a 20 ruedas): {len(mad):,} | "
          f"{mad.fecha.min().date()} -> {mad.fecha.max().date()}")

    print("\nA) DISCREPANCIA guardada vs lo que se sabia ese dia (% de filas)")
    for n in (10, 5):
        for base in ("is_sh", "is_sl", "estructura", "dias_sh", "dias_sl",
                     "bos_bull", "bos_bear", "choch_bull", "choch_bear"):
            c = f"{base}_{n}"
            a, b = mad[c + "_sto"], mad[c + "_pit"]
            dif = ~((a == b) | (a.isna() & b.isna()))
            txt = f"   {c:14s} difiere {dif.mean():6.1%}"
            if base.startswith(("is_", "bos_", "choch_")):
                txt += (f"   | eventos guardada {int((a == 1).sum()):,} / en su dia "
                        f"{int((b == 1).sum()):,} / en ambas {int(((a == 1) & (b == 1)).sum()):,}")
            print(txt)
        d = (mad[f"dist_sl_{n}_pct_sto"] - mad[f"dist_sl_{n}_pct_pit"]).abs()
        print(f"   dist_sl_{n}_pct  difiere > 0,01 pp {(d > 0.01).mean():6.1%}")

    print("\nB) dias_sh_10 / dias_sl_10")
    for c in ("dias_sh_10", "dias_sl_10"):
        for v, lab in (("_sto", "guardada"), ("_pit", "en su dia")):
            s = mad[c + v]
            print(f"   {c} {lab:9s} ==0 {(s == 0).mean():6.1%}  <=2 {(s <= 2).mean():6.1%}  "
                  f"mediana {s.median():.0f}")

    print("\nC) RETORNO FORWARD EN EXCESO (vs muestra del dia), media por dia e IC95")
    for v, lab in (("_sto", "GUARDADA "), ("_pit", "EN SU DIA")):
        linea_exceso(mad, mad["is_sh_10" + v] == 1, f"{lab} is_sh_10=1")
        linea_exceso(mad, mad["is_sl_10" + v] == 1, f"{lab} is_sl_10=1")
        linea_exceso(mad, mad["dias_sh_10" + v] <= 2, f"{lab} dias_sh_10<=2")
        linea_exceso(mad, mad["dias_sl_10" + v] <= 2, f"{lab} dias_sl_10<=2")
        for ev in ("bos_bull_10", "choch_bull_10", "bos_bear_10", "choch_bear_10"):
            linea_exceso(mad, mad[ev + v] == 1, f"{lab} {ev}")
        linea_exceso(mad, mad["estructura_10" + v] == 1, f"{lab} estructura_10=+1")
        linea_exceso(mad, mad["estructura_10" + v] == -1, f"{lab} estructura_10=-1")


# -- D: modelos ML ------------------------------------------------------------------------

def seccion_d(panel: pd.DataFrame, muestra: list) -> None:
    import joblib
    from sklearn.metrics import roc_auc_score
    from src.ml.ml_v2 import cargar_modelo_v2
    from src.ml.trainer import _BOOL_COLS, feature_engineering
    from src.ml.trainer_v3 import FEATURE_COLS_V3

    print("\nD) MODELOS ML: mismas filas, estructura guardada vs la de su dia")
    ph = ", ".join(f"'{t}'" for t in muestra)
    fm = query_df(f"""
        SELECT ticker, fecha, close, atr14, momentum, bb_upper, bb_lower, retorno_20d,
               label_binario, rsi14, macd_hist, adx, vol_relativo,
               dist_sma21, dist_sma50, dist_sma200, score_ponderado, condiciones_ok,
               cond_rsi, cond_macd, cond_sma21, cond_sma50, cond_sma200, cond_momentum,
               z_rsi_sector, z_retorno_1d_sector, z_retorno_5d_sector, z_vol_sector,
               z_dist_sma50_sector, z_adx_sector, pct_long_sector, rank_retorno_sector,
               rsi_sector_avg, adx_sector_avg, retorno_1d_sector_avg
        FROM features_ml
        WHERE ticker IN ({ph}) AND label_binario IS NOT NULL
    """)
    fm["fecha"] = pd.to_datetime(fm["fecha"])
    for col in _BOOL_COLS:
        fm[col] = fm[col].astype(float)
    fm = feature_engineering(fm)

    def con_ms(sufijo):
        ms = panel[["ticker", "fecha"] + [c + sufijo for c in FEATURE_COLS_MS]].rename(
            columns={c + sufijo: c for c in FEATURE_COLS_MS})
        return fm.merge(ms, on=["ticker", "fecha"], how="inner").sort_values(
            ["ticker", "fecha"]).reset_index(drop=True)

    A, B = con_ms("_sto"), con_ms("_pit")
    print(f"   filas features_ml con label en el panel: {len(A):,}")

    v1 = joblib.load(os.path.join(ROOT, "models_v3", "global", "champion.joblib"))
    v2, _ = cargar_modelo_v2()

    def prob(model, df, ms_a_cero):
        X = df[FEATURE_COLS_V3].astype(float).copy()
        if ms_a_cero:
            X[FEATURE_COLS_MS] = X[FEATURE_COLS_MS].fillna(0)
        return model.predict_proba(X.values)[:, list(model.classes_).index(1)]

    # v1: entrenada con MS NaN -> 0, en vivo le llegan NaN (imputer). v2: 0 en ambos.
    p = {"v1_sto": prob(v1, A, True), "v1_pit": prob(v1, B, False),
         "v2_sto": prob(v2, A, True), "v2_pit": prob(v2, B, True)}
    y = A["label_binario"].astype(int).to_numpy()
    r20 = A["retorno_20d"].astype(float).to_numpy()   # en PORCENTAJE
    tramos = [
        ("hasta 2025-01-27 (v1 fuera de muestra)", A.fecha < "2025-01-28"),
        ("2025-01-28 -> 2026-04-10 (v1 en muestra)", (A.fecha >= "2025-01-28") & (A.fecha <= "2026-04-10")),
        ("desde 2026-04-13 (v1 fuera de muestra)", A.fecha > "2026-04-10"),
    ]
    for nombre, m in tramos:
        m = m.to_numpy()
        if m.sum() < 500:
            continue
        print(f"\n   {nombre}: {m.sum():,} filas | base {y[m].mean():.1%} | ret20 medio {r20[m].mean():+.2f}%")
        for key, lab in (("v1_sto", "v1 guardada "), ("v1_pit", "v1 en su dia"),
                         ("v2_sto", "v2 guardada "), ("v2_pit", "v2 en su dia")):
            pp = p[key][m]
            top = np.argsort(-pp)[: max(1, int(len(pp) * 0.10))]
            print(f"     {lab} AUC {roc_auc_score(y[m], pp):.3f} | decil alto acierto "
                  f"{y[m][top].mean():.1%} ret20 {np.nanmean(r20[m][top]):+.2f}% | "
                  f"prob>=0,65 {np.mean(pp >= 0.65):.1%}")
    d1 = np.abs(p["v1_sto"] - p["v1_pit"])
    cortes = sorted((0.75, 0.65, 0.55, 0.45, 0.35))
    cambia = np.mean(np.digitize(p["v1_sto"], cortes) != np.digitize(p["v1_pit"], cortes))
    print(f"\n   |prob v1 guardada - en su dia|: mediana {np.median(d1):.3f}  p90 {np.quantile(d1, .9):.3f}"
          f" | filas que cambian de tramo de puntos ML: {cambia:.1%}")


# -- E: semanal -----------------------------------------------------------------------------

def seccion_e() -> None:
    from src.data.resample_weekly import resample_a_semanal
    from src.indicators.market_structure_1w import _calcular_ticker_1w

    print("\nE) SEMANAL: ultima semana cerrada vs la misma semana confirmada")
    df = query_df("SELECT ticker, fecha, open, high, low, close, volume, adj_close "
                  "FROM precios_diarios ORDER BY ticker, fecha")
    filas = []
    for tk, g in df.groupby("ticker"):
        sem = resample_a_semanal(g)
        if len(sem) < 60:
            continue
        w = sem.rename(columns={"fecha_semana": "fecha"})[["fecha", "open", "high", "low", "close", "volume"]]
        full = _calcular_ticker_1w(w.copy())
        for i in range(max(30, len(w) - 110), len(w) - 10):
            vivo = _calcular_ticker_1w(w.iloc[: i + 1].copy()).iloc[-1]
            filas.append({"est_vivo": vivo["estructura_10"], "est_conf": full.iloc[i]["estructura_10"],
                          "sh_vivo": vivo["is_sh_10"], "sh_conf": full.iloc[i]["is_sh_10"]})
    r = pd.DataFrame(filas)
    print(f"   semanas: {len(r):,} | estructura_10 difiere {(r.est_vivo != r.est_conf).mean():.1%}"
          f" | is_sh_10 marcados en su semana {int((r.sh_vivo == 1).sum()):,} vs confirmados "
          f"{int((r.sh_conf == 1).sum()):,}")
    print((pd.crosstab(r.est_vivo, r.est_conf, normalize="index") * 100).round(1).to_string())


# -- F: valor de eventos con swings confirmados --------------------------------------------

def seccion_f(desde: str) -> None:
    print(f"\nF) VALOR DE EVENTOS CON SWINGS CONFIRMADOS (estructura.py), universo desde {desde}")
    df = query_df("""
        SELECT ticker, fecha, high, low, close FROM precios_diarios
        WHERE close > 0 AND high > 0 AND low > 0 ORDER BY ticker, fecha
    """)
    df["fecha"] = pd.to_datetime(df["fecha"])
    partes = [estructura.calcular_estructura(g) for _, g in df.groupby("ticker", sort=True)]
    ms = pd.concat(partes, ignore_index=True).merge(
        df[["ticker", "fecha", "close"]], on=["ticker", "fecha"])
    ms = agregar_forward(ms)
    ms = ms[(ms.fecha >= desde) & ms["fwd_20"].notna()].copy()
    print(f"   filas {len(ms):,} | tickers {ms.ticker.nunique()} | "
          f"{ms.fecha.min().date()} -> {ms.fecha.max().date()}")
    for n in (10, 5):
        for ev in (f"bos_bull_{n}", f"choch_bull_{n}", f"bos_bear_{n}", f"choch_bear_{n}",
                   f"is_sh_{n}", f"is_sl_{n}"):
            linea_exceso(ms, ms[ev] == 1, ev)
        linea_exceso(ms, ms[f"estructura_{n}"] == 1, f"estructura_{n}=+1")
        linea_exceso(ms, ms[f"estructura_{n}"] == -1, f"estructura_{n}=-1")


# -- G: regla de entrada de FT_SMC -------------------------------------------------------

RUEDAS_EVENTO_SMC = 9   # ft_scoring.LOOKBACK_DIAS = 12 dias de calendario ~ 8-9 ruedas


def _senal_smc(df: pd.DataFrame) -> pd.Series:
    """ft_scoring.calcular_score_estructura (condiciones obligatorias) sobre un panel."""
    evento = ((df["choch_bull_10"] == 1) | (df["bos_bull_10"] == 1)).astype(int)
    tuvo = evento.groupby(df["ticker"]).transform(
        lambda s: s.rolling(RUEDAS_EVENTO_SMC, min_periods=1).max())
    return ((tuvo == 1) & (df["estructura_10"] >= 0) & (df["choch_bear_10"] == 0)
            & (df["close"] > df["open"])
            & df["dist_sl_10_pct"].between(1.0, 8.0))


def seccion_g(desde: str) -> None:
    print(f"\nG) REGLA DE ENTRADA DE FT_SMC (condiciones obligatorias), universo desde {desde}")
    cols = "ticker, fecha, estructura_10, choch_bull_10, bos_bull_10, choch_bear_10, dist_sl_10_pct"
    precios = query_df("SELECT ticker, fecha, open, close FROM precios_diarios WHERE close > 0")
    precios["fecha"] = pd.to_datetime(precios["fecha"])
    for tabla, lab in (("features_market_structure", "GUARDADA (vieja)"),
                       ("features_estructura", "SIN FUTURO (nueva)")):
        ms = query_df(f"SELECT {cols} FROM {tabla}")
        ms["fecha"] = pd.to_datetime(ms["fecha"])
        df = precios.merge(ms, on=["ticker", "fecha"]).sort_values(["ticker", "fecha"])
        for c in ("estructura_10", "choch_bull_10", "bos_bull_10", "choch_bear_10", "dist_sl_10_pct"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = agregar_forward(df)
        df["senal"] = _senal_smc(df)
        d = df[(df.fecha >= desde) & df["fwd_20"].notna()]
        por_dia = d.groupby("fecha")["senal"].sum()
        print(f"   {lab}: senales {int(d['senal'].sum()):,} | por rueda media {por_dia.mean():.1f}")
        linea_exceso(d, d["senal"], f"{lab} senal SMC")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--ruedas", type=int, default=780)
    ap.add_argument("--semilla", type=int, default=20260917)
    ap.add_argument("--reusar-panel", action="store_true",
                    help="usa reportes/estructura_velas/panel_guardada_vs_pit.pkl")
    ap.add_argument("--solo", default="ABCDEFG",
                    help="secciones a correr, p.ej. ACF (A-C van juntas)")
    ap.add_argument("--desde", default="2021-06-01", help="inicio de la seccion F")
    args = ap.parse_args()
    solo = set(args.solo.upper())

    if solo & set("ABCD"):
        if args.reusar_panel and os.path.exists(PANEL):
            with open(PANEL, "rb") as fh:
                obj = pickle.load(fh)
            print(f"panel reusado: {len(obj['panel']):,} filas")
        else:
            obj = construir_panel(args.tickers, args.ruedas, args.semilla)
        if solo & set("ABC"):
            seccion_a_c(obj["panel"])
        if "D" in solo:
            seccion_d(obj["panel"], obj["muestra"])
    if "E" in solo:
        seccion_e()
    if "F" in solo:
        seccion_f(args.desde)
    if "G" in solo:
        seccion_g(args.desde)


if __name__ == "__main__":
    main()
