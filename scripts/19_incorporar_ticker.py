"""
19_incorporar_ticker.py
Evalua todos los modelos champion V3 contra el historial de un ticker
y asigna el mejor en la tabla activos.modelo_asignado.

El scanner (17_scanner_alertas.py / cron_diario.py) usa modelo_asignado
como primera prioridad al seleccionar el modelo para predict_proba.

Uso:
    python scripts/19_incorporar_ticker.py AAPL
    python scripts/19_incorporar_ticker.py MSFT NVDA TSLA
    python scripts/19_incorporar_ticker.py AAPL --umbral 0.05
    python scripts/19_incorporar_ticker.py AAPL --forzar   # re-evalua si ya tiene asignado

Proceso por ticker:
    1. Descarga 5 anos de historial via yfinance
    2. Calcula 53 features V3 (indicadores + market structure) + label_binario
    3. Split temporal 70/15/15 TRAIN/TEST/BACKTEST (igual que el pipeline original)
    4. Evalua los 4 champion.joblib en el TEST split (F1 clase GANANCIA)
    5. Muestra F1 en BACKTEST como validacion adicional
    6. Guarda modelo_asignado = scope ganador (por TEST F1) en activos tabla

Nota: los z-scores sectoriales quedan como NaN para tickers externos.
El SimpleImputer dentro del sklearn Pipeline los rellena con la media
de entrenamiento original, lo cual es el comportamiento correcto.
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import f1_score

from src.pipeline.data_manager import descargar_yfinance
from src.indicators.technical import calcular_indicadores
from src.scoring.rule_based import calcular_scoring
from src.indicators.market_structure import _calcular_ticker, FEATURE_COLS_MS
from src.ml.trainer import feature_engineering, _BOOL_COLS
from src.ml.trainer_v3 import FEATURE_COLS_V3, cargar_champion_v3
from src.utils.config import SECTORES_ML
from src.data.database import get_connection, query_df


UMBRAL_GANANCIA_DEFAULT = 0.03   # retorno_20d >= 3% -> label_binario = 1

# Split 70/15/15 identico al pipeline de entrenamiento original
RATIO_TRAIN     = 0.70
RATIO_TEST      = 0.15
# RATIO_BACKTEST = 0.15  (resto)


# ─────────────────────────────────────────────────────────────
# Construccion de dataset completo (SERIE completa, no solo ultima barra)
# ─────────────────────────────────────────────────────────────

_Z_COLS = [
    "z_rsi_sector", "z_retorno_1d_sector", "z_retorno_5d_sector",
    "z_vol_sector", "z_dist_sma50_sector", "z_adx_sector",
    "pct_long_sector", "rank_retorno_sector",
    "rsi_sector_avg", "adx_sector_avg", "retorno_1d_sector_avg",
]


def _construir_dataset(df_ohlcv: pd.DataFrame, ticker: str,
                        umbral: float) -> pd.DataFrame:
    """
    Calcula todas las features V3 (53 columnas) + label_binario para
    toda la serie historica. Retorna DataFrame listo para train/test.

    Los z-scores sectoriales quedan como NaN (SimpleImputer los maneja).
    """
    df = df_ohlcv.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # 1. Indicadores tecnicos
    df_ind = calcular_indicadores(df, ticker)
    df_ind["fecha"] = pd.to_datetime(df_ind["fecha"])

    # 2. Scoring rule-based
    df_precios_simple = df[["fecha", "close"]].copy()
    df_sc = calcular_scoring(df_ind, df_precios_simple, ticker)
    df_sc["fecha"] = pd.to_datetime(df_sc["fecha"])

    # 3. Merge base
    df_base = df.merge(df_ind, on="fecha", how="left", suffixes=("", "_ind"))
    dup = [c for c in df_base.columns if c.endswith("_ind")]
    df_base.drop(columns=dup, inplace=True, errors="ignore")

    df_base = df_base.merge(
        df_sc[["fecha", "score_ponderado", "condiciones_ok",
               "cond_rsi", "cond_macd", "cond_sma21",
               "cond_sma50", "cond_sma200", "cond_momentum"]],
        on="fecha", how="left"
    )

    # 4. Feature engineering (bb_posicion, atr14_pct, momentum_pct)
    df_base = feature_engineering(df_base)

    # 5. Market structure (24 features)
    df_ms = _calcular_ticker(
        df[["fecha", "open", "high", "low", "close", "volume"]].copy()
    )
    df_ms["fecha"] = pd.to_datetime(df_ms["fecha"])
    df_base = df_base.merge(
        df_ms[["fecha"] + FEATURE_COLS_MS], on="fecha", how="left",
        suffixes=("", "_ms")
    )

    # 6. Booleanos -> float para sklearn
    for col in _BOOL_COLS:
        if col in df_base.columns:
            df_base[col] = df_base[col].astype(float)

    # 7. NaN en features MS -> 0 (barras sin pivot confirmado)
    ms_present = [c for c in FEATURE_COLS_MS if c in df_base.columns]
    df_base[ms_present] = df_base[ms_present].fillna(0)

    # 8. Z-scores sectoriales: NaN (SimpleImputer los rellena con media de train)
    for col in _Z_COLS:
        if col not in df_base.columns:
            df_base[col] = np.nan

    # 9. Label: retorno forward de 20 barras
    df_base["retorno_20d"] = df_base["close"].pct_change(20).shift(-20)
    df_base["label_binario"] = np.where(
        df_base["retorno_20d"].isna(),
        np.nan,
        (df_base["retorno_20d"] >= umbral).astype(float)
    )

    # 10. Solo barras con indicadores calculados (rsi14 necesita ~14 barras)
    df_base = df_base[df_base["rsi14"].notna()].reset_index(drop=True)

    return df_base


# ─────────────────────────────────────────────────────────────
# Evaluacion de los 4 modelos champion en el TEST split
# ─────────────────────────────────────────────────────────────

def _evaluar_modelos(X_test: pd.DataFrame, y_test: pd.Series,
                     X_bt: pd.DataFrame = None, y_bt: pd.Series = None) -> dict:
    """
    Carga y evalua los 4 champion V3 en TEST (y opcionalmente en BACKTEST).
    Retorna dict {scope: {"test": f1, "backtest": f1_o_None}}.
    """
    scopes = ["global"] + SECTORES_ML
    resultados = {}
    for scope in scopes:
        try:
            model = cargar_champion_v3(scope)
            f1_test = f1_score(y_test, model.predict(X_test), zero_division=0)
            f1_bt   = None
            if X_bt is not None and len(X_bt) > 0:
                f1_bt = f1_score(y_bt, model.predict(X_bt), zero_division=0)
                f1_bt = round(float(f1_bt), 4)
            resultados[scope] = {"test": round(float(f1_test), 4), "backtest": f1_bt}
        except Exception as e:
            print(f"  [WARN] Error evaluando modelo '{scope}': {e}")
            resultados[scope] = {"test": 0.0, "backtest": None}
    return resultados


# ─────────────────────────────────────────────────────────────
# Persistencia en DB
# ─────────────────────────────────────────────────────────────

def _asignacion_actual(ticker: str):
    """Devuelve modelo_asignado actual del ticker, o None."""
    sql = "SELECT modelo_asignado FROM activos WHERE ticker = :ticker"
    try:
        df = query_df(sql, params={"ticker": ticker})
        if df.empty:
            return None
        val = df.iloc[0]["modelo_asignado"]
        return str(val) if (val is not None and str(val) not in ("None", "")) else None
    except Exception:
        return None


def _guardar_asignacion(ticker: str, scope: str):
    """
    Actualiza activos.modelo_asignado para el ticker.
    Si el ticker no existe en la tabla, lo inserta con sector=NULL.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM activos WHERE ticker = %s", (ticker,))
            n = cur.fetchone()[0]
            if n > 0:
                cur.execute(
                    "UPDATE activos SET modelo_asignado = %s WHERE ticker = %s",
                    (scope, ticker)
                )
            else:
                cur.execute(
                    """INSERT INTO activos (ticker, nombre, sector, modelo_asignado)
                       VALUES (%s, %s, NULL, %s)
                       ON CONFLICT (ticker) DO UPDATE
                       SET modelo_asignado = EXCLUDED.modelo_asignado""",
                    (ticker, ticker, scope)
                )


# ─────────────────────────────────────────────────────────────
# Pipeline principal para un ticker
# ─────────────────────────────────────────────────────────────

def incorporar_ticker(ticker: str,
                      umbral: float = UMBRAL_GANANCIA_DEFAULT,
                      forzar: bool = False,
                      sector: str = None) -> dict:
    """
    Descarga historial, evalua los 4 champion V3 y asigna el mejor.
    Usa split 70/15/15 TRAIN/TEST/BACKTEST, igual que el pipeline de entrenamiento.
    La seleccion del ganador se basa en F1 del segmento TEST.

    Returns:
        dict con resultado (modelo_asignado, f1_todos, n_train, n_test, n_backtest, error)
    """
    print(f"\n{'='*55}")
    print(f"  {ticker}")
    print(f"{'='*55}")

    # Verificar asignacion existente
    if not forzar:
        actual = _asignacion_actual(ticker)
        if actual:
            print(f"  Ya tiene modelo asignado: {actual}")
            print(f"  (usar --forzar para re-evaluar)")
            return {"ticker": ticker, "modelo_asignado": actual, "ya_existia": True}

    # 1. Descargar historial
    print("  [1/5] Descargando historial (5 anos yfinance)...")
    try:
        df_ohlcv = descargar_yfinance(ticker, "5y")
    except Exception as e:
        msg = f"Error descargando {ticker}: {e}"
        print(f"  ERROR: {msg}")
        return {"ticker": ticker, "error": msg}

    n_barras = len(df_ohlcv)
    print(f"         {n_barras} barras descargadas.")
    if n_barras < 300:
        msg = f"Solo {n_barras} barras disponibles (minimo 300)"
        print(f"  ERROR: {msg}")
        return {"ticker": ticker, "error": msg}

    # 1b. Auto-detectar sector desde yfinance si no fue provisto manualmente
    if not sector:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or info.get("sectorDisp") or None
            if sector:
                print(f"         Sector auto-detectado: '{sector}'")
            else:
                print("         Sector no disponible en yfinance.")
        except Exception as e:
            print(f"  [WARN] No se pudo obtener sector de yfinance: {e}")

    # 1c. Persistir OHLCV e indicadores en DB (para que aparezca en Analisis Tecnico)
    print("  [1c/5] Persistiendo precios e indicadores en DB...")
    try:
        from src.pipeline.data_manager import persistir_ticker_nuevo
        from src.indicators.technical import procesar_indicadores_ticker
        persistir_ticker_nuevo(df_ohlcv, ticker, sector=sector)
        procesar_indicadores_ticker(ticker, df_ohlcv, guardar_db=True)
        # Actualizar sector en activos (ON CONFLICT DO NOTHING no lo hace)
        if sector:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE activos SET sector = %s WHERE ticker = %s",
                        (sector, ticker)
                    )
            print(f"         Sector '{sector}' asignado a {ticker}.")
    except Exception as e:
        print(f"  [WARN] No se pudo persistir en DB: {e}")

    # 2. Construir dataset con 53 features + label
    print("  [2/5] Calculando 53 features V3 + label_binario...")
    df = _construir_dataset(df_ohlcv, ticker, umbral)
    df_valido = df.dropna(subset=["label_binario"]).reset_index(drop=True)

    n_valido = len(df_valido)
    n_pos    = int(df_valido["label_binario"].sum())
    print(f"         {n_valido} filas validas, {n_pos} positivas ({n_pos/n_valido:.1%})")

    if n_valido < 200:
        msg = f"Solo {n_valido} filas con label (minimo 200)"
        print(f"  ERROR: {msg}")
        return {"ticker": ticker, "error": msg}

    # 3. Split temporal 70/15/15 (igual que el pipeline de entrenamiento)
    n_train = int(n_valido * RATIO_TRAIN)
    n_test  = int(n_valido * RATIO_TEST)
    df_train    = df_valido.iloc[:n_train]
    df_test     = df_valido.iloc[n_train : n_train + n_test]
    df_backtest = df_valido.iloc[n_train + n_test:]

    n_test_pos = int(df_test["label_binario"].sum())
    n_bt_pos   = int(df_backtest["label_binario"].sum())
    print(f"  [3/5] Split: TRAIN={len(df_train)} | TEST={len(df_test)} "
          f"({n_test_pos} pos) | BACKTEST={len(df_backtest)} ({n_bt_pos} pos)")

    if len(df_test) < 15 or n_test_pos < 3:
        msg = f"TEST insuficiente: {len(df_test)} filas, {n_test_pos} positivas"
        print(f"  ERROR: {msg}")
        return {"ticker": ticker, "error": msg}

    X_test = df_test[FEATURE_COLS_V3].copy()
    y_test = df_test["label_binario"].astype(int)
    X_bt   = df_backtest[FEATURE_COLS_V3].copy()
    y_bt   = df_backtest["label_binario"].astype(int)

    # 4. Evaluar los 4 champion V3 en TEST + BACKTEST
    print("  [4/5] Evaluando modelos champion V3...")
    f1_todos = _evaluar_modelos(X_test, y_test, X_bt, y_bt)

    # Seleccionar ganador por F1 TEST (igual que el proceso original de training)
    mejor_scope = max(f1_todos, key=lambda s: f1_todos[s]["test"])
    mejor_f1    = f1_todos[mejor_scope]["test"]

    print(f"         {'Scope':<30} {'TEST':>8} {'BACKTEST':>10}")
    print(f"         {'-'*50}")
    for scope, vals in sorted(f1_todos.items(), key=lambda x: -x[1]["test"]):
        bt_str  = f"{vals['backtest']:.4f}" if vals["backtest"] is not None else "  N/A "
        ganador = " <-- GANADOR" if scope == mejor_scope else ""
        print(f"         {scope:<30} {vals['test']:>8.4f} {bt_str:>10}{ganador}")

    # 5. Guardar en DB
    print(f"  [5/5] Guardando modelo_asignado='{mejor_scope}' en activos...")
    _guardar_asignacion(ticker, mejor_scope)
    print(f"  OK -- {ticker} -> {mejor_scope} (F1 TEST={mejor_f1:.4f})")

    return {
        "ticker":          ticker,
        "modelo_asignado": mejor_scope,
        "f1_ganador":      mejor_f1,
        "f1_todos":        f1_todos,
        "n_train":         len(df_train),
        "n_test":          len(df_test),
        "n_backtest":      len(df_backtest),
        "ya_existia":      False,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evalua champion V3 y asigna el mejor modelo a un ticker."
    )
    parser.add_argument(
        "tickers", nargs="+",
        help="Tickers a incorporar (ej: AAPL MSFT NVDA)"
    )
    parser.add_argument(
        "--umbral", type=float, default=UMBRAL_GANANCIA_DEFAULT,
        help=f"Threshold retorno 20d para label=1 (default {UMBRAL_GANANCIA_DEFAULT:.0%})"
    )
    parser.add_argument(
        "--forzar", action="store_true",
        help="Re-evaluar aunque ya tenga modelo asignado"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  INCORPORAR TICKERS — Asignacion de Modelo Champion")
    print("=" * 55)
    print(f"  Tickers : {', '.join(t.upper() for t in args.tickers)}")
    print(f"  Umbral  : {args.umbral:.1%}  Split: 70/15/15 TRAIN/TEST/BACKTEST")
    print(f"  Forzar  : {args.forzar}")

    resultados = []
    for ticker in args.tickers:
        r = incorporar_ticker(
            ticker.upper(),
            umbral=args.umbral,
            forzar=args.forzar,
        )
        resultados.append(r)

    # Resumen final
    ok  = [r for r in resultados if "modelo_asignado" in r and not r.get("error")]
    err = [r for r in resultados if r.get("error")]

    print(f"\n{'='*55}")
    print("  RESUMEN FINAL")
    print(f"{'='*55}")
    for r in ok:
        ya     = " (ya existia)" if r.get("ya_existia") else ""
        f1_str = f" F1={r.get('f1_ganador', 0):.4f}" if not r.get("ya_existia") else ""
        print(f"  OK  {r['ticker']:<8} -> {r['modelo_asignado']}{f1_str}{ya}")
    for r in err:
        print(f"  ERR {r['ticker']:<8}  {r['error'][:70]}")
    print()

    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
