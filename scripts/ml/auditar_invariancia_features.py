"""
auditar_invariancia_features.py
Tarea 23, Fase 3a, paso 0: auditar las 29 features NO estructurales del modelo antes
de entrenar la v3. Solo LEE.

POR QUE (docs/estructura_velas.md sec. 11, regla 1)
    La revision encontro que las 24 features de market structure tenian informacion
    del futuro. Las otras 29 (indicadores tecnicos + scoring rule-based + features
    sectoriales) pasaron una auditoria ESTATICA -- el unico shift(-n) del pipeline es
    el label -- pero no una NUMERICA. Antes de entrenar la v3 conviene medirlo, no
    suponerlo: si otra feature mira al futuro, la v3 nace contaminada igual que la v1.

QUE MIDE (dos cosas distintas, no confundirlas)

    A. INVARIANCIA (mira al futuro?)
       calcular(datos[:t+1]).iloc[-1]  ==  calcular(datos).iloc[t]
       Misma serie, mismo origen, cortada en t. Si una feature usa barras posteriores
       a t, los dos numeros difieren. Para un indicador causal -- incluso recursivo
       como RSI/ATR/ADX, que arrastran su propio estado -- tienen que ser IDENTICOS,
       porque el valor en t solo depende de las barras hasta t.
       Diferencia distinta de cero aca = LEAKAGE.

    B. SKEW DE VENTANA (lo que el modelo ve en vivo es lo que vio al entrenar?)
       calcular(datos[t-499:t+1]).iloc[-1]  vs  calcular(datos).iloc[t]
       `features_ml` sale de `indicadores_tecnicos`, calculada sobre la historia
       COMPLETA del ticker. El scanner en vivo recalcula desde las ULTIMAS 500 barras
       (data_manager.preparar_ticker -> cargar_precios_db(ultimas_n=500) ->
       feature_calculator.calcular_features_completas). Para los indicadores
       recursivos, 500 barras de arranque no dan exactamente el mismo valor que 1.400.
       Diferencia aca NO es leakage: es train/serve skew, y hay que saber su tamano
       antes de decidir con las probabilidades del modelo.

Las features sectoriales (11 de las 29) son z-scores TRANSVERSALES: el ticker contra
sus pares EN LA MISMA FECHA, con ventanas hacia atras. No se recalculan aca porque
dependen del universo entero por fecha; su riesgo es otro y ya esta documentado
(features_sector: leer la fila de SU rueda, CLAUDE.md patrones criticos).

Uso:
    python scripts/ml/auditar_invariancia_features.py
    python scripts/ml/auditar_invariancia_features.py --tickers 20 --cortes 10
    python scripts/ml/auditar_invariancia_features.py --tickers AAPL,KLAC --semilla 7
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.indicators.technical import calcular_indicadores  # noqa: E402
from src.scoring.rule_based import calcular_scoring  # noqa: E402
from src.ml.trainer import FEATURE_COLS, feature_engineering  # noqa: E402
from src.utils.contexto_sectorial import FEATURES_SECTORIALES  # noqa: E402

VENTANA_SCANNER = 500      # data_manager.cargar_precios_db(ultimas_n=500)
TOL = 1e-9                 # las columnas vienen redondeadas a 4-6 decimales
MIN_RUEDAS = 600           # para que una ventana de 500 tenga sentido


def log(msg=""):
    print(msg, flush=True)


def _features_locales() -> list:
    """Las FEATURE_COLS que se pueden recalcular desde el OHLCV de un ticker."""
    return [c for c in FEATURE_COLS if c not in set(FEATURES_SECTORIALES)]


def calcular_fila(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """OHLCV -> las features locales del modelo, igual que feature_calculator."""
    ind = calcular_indicadores(df, ticker)
    ind["fecha"] = pd.to_datetime(ind["fecha"])
    precios = df[["fecha", "close"]].copy()
    precios["fecha"] = pd.to_datetime(precios["fecha"])
    sc = calcular_scoring(ind, precios, ticker)
    sc["fecha"] = pd.to_datetime(sc["fecha"])
    base = ind.merge(sc.drop(columns=[c for c in sc.columns
                                      if c in ind.columns and c != "fecha"]),
                     on="fecha", how="inner")
    base = base.merge(df[["fecha", "close"]].assign(
        fecha=pd.to_datetime(df["fecha"])), on="fecha", how="left",
        suffixes=("", "_px"))
    return feature_engineering(base)


def auditar_ticker(df: pd.DataFrame, ticker: str, cortes: list, cols: list) -> dict:
    """Devuelve {col: {'inv': max dif, 'skew': max dif}} para los cortes dados."""
    completo = calcular_fila(df, ticker).set_index("fecha")
    res = {c: {"inv": 0.0, "skew": 0.0, "n": 0} for c in cols}

    for t in cortes:
        fecha = pd.Timestamp(df["fecha"].iloc[t])
        if fecha not in completo.index:
            continue
        ref = completo.loc[fecha]

        parcial = calcular_fila(df.iloc[:t + 1], ticker)
        ventana = calcular_fila(df.iloc[max(0, t + 1 - VENTANA_SCANNER):t + 1], ticker)
        if parcial.empty or ventana.empty:
            continue
        f_par, f_ven = parcial.iloc[-1], ventana.iloc[-1]
        if pd.Timestamp(f_par["fecha"]) != fecha or pd.Timestamp(f_ven["fecha"]) != fecha:
            continue

        for c in cols:
            a, b, d = ref.get(c), f_par.get(c), f_ven.get(c)
            if a is None or pd.isna(a):
                continue
            res[c]["n"] += 1
            if b is not None and not pd.isna(b):
                res[c]["inv"] = max(res[c]["inv"], abs(float(a) - float(b)))
            if d is not None and not pd.isna(d):
                res[c]["skew"] = max(res[c]["skew"], abs(float(a) - float(d)))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Auditoria de invariancia de las features no estructurales")
    ap.add_argument("--tickers", default="12",
                    help="cantidad de tickers al azar, o lista separada por coma")
    ap.add_argument("--cortes", type=int, default=8, help="ruedas por ticker")
    ap.add_argument("--semilla", type=int, default=23)
    args = ap.parse_args()

    os.environ.pop("DATABASE_URL", None)   # LOCAL es la fuente de verdad
    from src.data.database import query_df

    cols = _features_locales()
    log("=" * 78)
    log("AUDITORIA DE INVARIANCIA -- features no estructurales del modelo")
    log("=" * 78)
    log(f"  features locales auditadas : {len(cols)} de {len(FEATURE_COLS)}")
    log(f"  sectoriales (no auditadas) : {len(FEATURES_SECTORIALES)} (z-scores "
        f"transversales por fecha)")
    log(f"  ventana en vivo            : {VENTANA_SCANNER} barras")

    universo = query_df("""
        SELECT ticker, COUNT(*) AS n FROM precios_diarios
        WHERE close > 0 AND high > 0 AND low > 0 AND open > 0
        GROUP BY ticker HAVING COUNT(*) >= :m ORDER BY ticker
    """, params={"m": MIN_RUEDAS})

    if args.tickers.replace(",", "").isdigit() and "," not in args.tickers:
        rng = np.random.default_rng(args.semilla)
        elegidos = list(rng.choice(universo["ticker"].to_numpy(),
                                   size=min(int(args.tickers), len(universo)),
                                   replace=False))
    else:
        elegidos = [t.strip().upper() for t in args.tickers.split(",")]
    log(f"  tickers                    : {len(elegidos)} -> {', '.join(sorted(elegidos))}")
    log("")

    total = {c: {"inv": 0.0, "skew": 0.0, "n": 0} for c in cols}
    rng = np.random.default_rng(args.semilla + 1)

    for tk in sorted(elegidos):
        df = query_df("""
            SELECT fecha, open, high, low, close, volume FROM precios_diarios
            WHERE ticker = :t AND close > 0 AND high > 0 AND low > 0 AND open > 0
            ORDER BY fecha
        """, params={"t": tk})
        if len(df) < MIN_RUEDAS:
            log(f"  {tk}: {len(df)} ruedas, se omite")
            continue
        df["fecha"] = pd.to_datetime(df["fecha"])

        bajo = max(250, VENTANA_SCANNER)          # ventana llena y SMA200 valida
        cortes = sorted(rng.choice(range(bajo, len(df)),
                                   size=min(args.cortes, len(df) - bajo),
                                   replace=False).tolist())
        res = auditar_ticker(df, tk, cortes, cols)
        peor_inv = max(res[c]["inv"] for c in cols)
        peor_skew = max((c, res[c]["skew"]) for c in cols)
        log(f"  {tk:<6} {len(df):>5} ruedas | cortes {len(cortes)} | "
            f"max dif invariancia {peor_inv:.2e} | "
            f"max dif ventana {peor_skew[1]:.4f} ({peor_skew[0]})")
        for c in cols:
            total[c]["inv"] = max(total[c]["inv"], res[c]["inv"])
            total[c]["skew"] = max(total[c]["skew"], res[c]["skew"])
            total[c]["n"] += res[c]["n"]

    log("")
    log("-" * 78)
    log("A. INVARIANCIA (dif != 0 = la feature mira al futuro)")
    log("-" * 78)
    log(f"  {'feature':<18} {'n':>5}  {'max |dif|':>12}  veredicto")
    fallan = []
    for c in cols:
        d = total[c]["inv"]
        ok = d <= TOL
        if not ok:
            fallan.append(c)
        log(f"  {c:<18} {total[c]['n']:>5}  {d:>12.2e}  {'OK' if ok else 'MIRA AL FUTURO'}")

    log("")
    log("-" * 78)
    log(f"B. SKEW DE VENTANA (entrenamiento con historia completa vs vivo con "
        f"{VENTANA_SCANNER} barras)")
    log("-" * 78)
    log(f"  {'feature':<18} {'n':>5}  {'max |dif|':>12}  nota")
    for c in sorted(cols, key=lambda x: -total[x]["skew"]):
        d = total[c]["skew"]
        nota = "identico" if d <= TOL else ("chico" if d < 0.01 else "MEDIR")
        log(f"  {c:<18} {total[c]['n']:>5}  {d:>12.4f}  {nota}")

    log("")
    log("=" * 78)
    if fallan:
        log(f"RESULTADO: {len(fallan)} features NO invariantes -> {fallan}")
        log("No entrenar la v3 con ellas hasta corregirlas (regla 1 del doc).")
        return 1
    log("RESULTADO: las features locales son INVARIANTES (no mira al futuro ninguna).")
    log("El skew de ventana no es leakage: es la diferencia entre como se entrena y")
    log("como se sirve. Queda documentado para la Fase 3a.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
