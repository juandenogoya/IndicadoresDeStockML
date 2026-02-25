"""
ml_backtest.py
Evaluación del impacto del filtro ML sobre las estrategias de backtesting.

Aplica los modelos ML desplegados (uno por sector) como filtro adicional
a las operaciones registradas en operaciones_backtest, y compara:

    ORIGINAL    : todas las operaciones que dispararon la señal de entrada
    ML-FILTRADO : solo las operaciones donde el modelo ML predijo GANANCIA
    RECHAZADAS  : operaciones rechazadas por ML (verificación de calidad)

Preguntas que responde:
    1. ¿Mejora el win_rate al aplicar el filtro ML?
    2. ¿Mejora el profit_factor?
    3. ¿Cuántas operaciones se pierden (rechazo)?
    4. ¿Las rechazadas son realmente malas trades? (su win_rate debería ser bajo)
    5. ¿Qué estrategia E×S se beneficia más del filtro?
    6. ¿El impacto es uniforme entre sectores?

Nota sobre sesgos temporales:
    El modelo fue entrenado en TRAIN. Las métricas de TEST y BACKTEST
    son completamente out-of-sample y son las más relevantes.
    Las métricas de TRAIN muestran el ajuste in-sample (referencia).
"""

import os
import joblib
import numpy as np
import pandas as pd
import psycopg2.extras
from typing import Dict, Optional, List

from src.data.database import query_df, get_connection, ejecutar_sql
from src.ml.trainer import feature_engineering, FEATURE_COLS, _scope_dir
from src.backtesting.metrics import calcular_metricas
from src.utils.config import MODELS_DIR


# ─────────────────────────────────────────────────────────────
# DDL — tabla de resultados de comparación
# ─────────────────────────────────────────────────────────────

_DDL_FILTER = """
CREATE TABLE IF NOT EXISTS resultados_ml_filter (
    id                     SERIAL PRIMARY KEY,
    estrategia_entrada     VARCHAR(5)   NOT NULL,
    estrategia_salida      VARCHAR(5)   NOT NULL,
    scope                  VARCHAR(50)  NOT NULL,   -- 'GLOBAL' | sector
    segmento               VARCHAR(15)  NOT NULL,   -- TRAIN | TEST | BACKTEST
    umbral_ml              NUMERIC(4,2) NOT NULL,   -- threshold (e.g. 0.50)

    -- ── Baseline (sin filtro ML) ──────────────────────────────
    ops_original           INTEGER,
    win_rate_orig          NUMERIC(7,4),
    ret_promedio_orig      NUMERIC(10,4),
    ret_total_orig         NUMERIC(10,4),
    profit_factor_orig     NUMERIC(10,4),
    max_dd_orig            NUMERIC(10,4),

    -- ── Con filtro ML ─────────────────────────────────────────
    ops_ml                 INTEGER,
    ops_rechazadas         INTEGER,
    pct_rechazo            NUMERIC(6,4),   -- % de ops rechazadas
    win_rate_ml            NUMERIC(7,4),
    ret_promedio_ml        NUMERIC(10,4),
    ret_total_ml           NUMERIC(10,4),
    profit_factor_ml       NUMERIC(10,4),
    max_dd_ml              NUMERIC(10,4),

    -- ── Rechazadas (verificación: deben ser peores) ───────────
    win_rate_rechazadas    NUMERIC(7,4),
    ret_promedio_rechazadas NUMERIC(10,4),
    profit_factor_rechazadas NUMERIC(10,4),

    -- ── Deltas (ML − Original) ────────────────────────────────
    delta_win_rate         NUMERIC(7,4),
    delta_ret_promedio     NUMERIC(10,4),
    delta_profit_factor    NUMERIC(10,4),

    created_at             TIMESTAMP DEFAULT NOW(),
    UNIQUE (estrategia_entrada, estrategia_salida, scope, segmento, umbral_ml)
);
"""


def crear_tabla_filter():
    """Crea la tabla resultados_ml_filter si no existe."""
    ejecutar_sql(_DDL_FILTER)


# ─────────────────────────────────────────────────────────────
# Carga de modelos desplegados
# ─────────────────────────────────────────────────────────────

def cargar_modelos_desplegados() -> Dict[str, object]:
    """
    Carga el modelo desplegado por sector (el ganador del Challenger Final).
    Automotive usa el Global Champion.

    Returns:
        dict {sector: modelo_sklearn}
    """
    global_path = os.path.join(_scope_dir("global"), "champion.joblib")
    if not os.path.exists(global_path):
        raise FileNotFoundError(
            f"Global champion no encontrado: {global_path}\n"
            "Ejecutar primero: python scripts/07_train_models.py"
        )
    global_model = joblib.load(global_path)

    modelos = {"global": global_model, "Automotive": global_model}

    for sector in ["Financials", "Consumer Staples", "Consumer Discretionary"]:
        path = os.path.join(_scope_dir(sector), "deployed.joblib")
        if os.path.exists(path):
            modelos[sector] = joblib.load(path)
            print(f"    [{sector}] / deployed.joblib")
        else:
            modelos[sector] = global_model
            print(f"    [{sector}] / global champion (deployed no encontrado)")

    return modelos


# ─────────────────────────────────────────────────────────────
# Carga de operaciones + features del día señal
# ─────────────────────────────────────────────────────────────

_FEATURE_LOAD_COLS = [
    "close", "bb_upper", "bb_lower", "atr14", "momentum",
    "rsi14", "macd_hist", "adx", "vol_relativo",
    "dist_sma21", "dist_sma50", "dist_sma200",
    "score_ponderado", "condiciones_ok",
    "cond_rsi", "cond_macd", "cond_sma21",
    "cond_sma50", "cond_sma200", "cond_momentum",
    "z_rsi_sector", "z_retorno_1d_sector", "z_retorno_5d_sector",
    "z_vol_sector", "z_dist_sma50_sector", "z_adx_sector",
    "pct_long_sector", "rank_retorno_sector",
    "rsi_sector_avg", "adx_sector_avg", "retorno_1d_sector_avg",
]
_BOOL_COLS = [
    "cond_rsi", "cond_macd", "cond_sma21",
    "cond_sma50", "cond_sma200", "cond_momentum",
]


def cargar_operaciones_con_features() -> pd.DataFrame:
    """
    Carga operaciones_backtest y une las features del DÍA SEÑAL.

    El día señal es la última fecha de features_ml disponible
    ANTES de fecha_entrada (trading day previo).

    Usa pd.merge_asof (lookup eficiente sin loop).

    Returns:
        DataFrame con todas las operaciones + features del día señal
    """
    # 1. Operaciones con sector
    ops = query_df("""
        SELECT
            ob.estrategia_entrada,
            ob.estrategia_salida,
            ob.ticker,
            ob.segmento,
            ob.fecha_entrada,
            ob.fecha_salida,
            ob.precio_entrada,
            ob.precio_salida,
            ob.motivo_salida,
            ob.dias_posicion,
            ob.retorno_pct,
            ob.resultado,
            a.sector
        FROM operaciones_backtest ob
        JOIN activos a ON ob.ticker = a.ticker
        ORDER BY ob.ticker, ob.fecha_entrada
    """)
    ops["fecha_entrada"] = pd.to_datetime(ops["fecha_entrada"])
    print(f"    Operaciones cargadas   : {len(ops):,} "
          f"({ops['ticker'].nunique()} tickers, "
          f"{ops['estrategia_entrada'].nunique()} × "
          f"{ops['estrategia_salida'].nunique()} combis)")

    # 2. Features de features_ml
    cols_sql = ", ".join(_FEATURE_LOAD_COLS)
    feats = query_df(f"""
        SELECT ticker, fecha, {cols_sql}
        FROM features_ml
        ORDER BY ticker, fecha
    """)
    feats["fecha"] = pd.to_datetime(feats["fecha"])

    for col in _BOOL_COLS:
        feats[col] = feats[col].astype(float)

    print(f"    Features cargadas      : {len(feats):,} filas")

    # 3. merge_asof: fecha_entrada − 1 día -> última feature antes de entrada
    #    Así encontramos el día señal (T) para la entrada (T+1)
    ops_sorted           = ops.sort_values("fecha_entrada").copy()
    ops_sorted["_key"]   = ops_sorted["fecha_entrada"] - pd.Timedelta(days=1)
    feats_sorted         = feats.sort_values("fecha").rename(columns={"fecha": "fecha_senal"})

    merged = pd.merge_asof(
        ops_sorted,
        feats_sorted,
        left_on="fecha_entrada",     # se busca en right_on via _key
        right_on="fecha_senal",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=5),  # máximo 5 días de gap (fines de semana/feriados)
    ).drop(columns=["_key"])

    con_feats = merged["fecha_senal"].notna().sum()
    sin_feats = merged["fecha_senal"].isna().sum()
    print(f"    Ops con features señal : {con_feats:,}  "
          f"(sin features: {sin_feats:,} -> prediccion neutra)")

    return merged


# ─────────────────────────────────────────────────────────────
# Aplicación del filtro ML
# ─────────────────────────────────────────────────────────────

def aplicar_filtro_ml(
    df: pd.DataFrame,
    modelos: Dict,
    umbral: float = 0.50,
) -> pd.DataFrame:
    """
    Aplica el modelo ML del sector correspondiente a cada operación.

    Para cada operación:
      - Si tiene features del día señal: predice con el modelo del sector
      - Si NO tiene features: predicción neutra (1 -> no filtrar)

    Añade columnas:
        ml_proba      : P(GANANCIA) según el modelo
        ml_prediccion : 1 si ml_proba >= umbral, 0 si no
    """
    df = df.copy()
    df["ml_proba"]      = np.nan
    df["ml_prediccion"] = np.nan

    mask_feat = df["fecha_senal"].notna()
    df_feat   = df[mask_feat].copy()

    for sector, grupo in df_feat.groupby("sector"):
        modelo = modelos.get(sector, modelos["global"])

        # Ingeniería de features (bb_posicion, atr14_pct, momentum_pct)
        grupo_eng = feature_engineering(grupo)
        X = grupo_eng[FEATURE_COLS].copy()

        try:
            probas = modelo.predict_proba(X)[:, 1]
            preds  = (probas >= umbral).astype(int)
        except Exception as e:
            print(f"    [WARN] Error prediciendo {sector}: {e} -> prediccion neutra")
            probas = np.full(len(X), 0.5)
            preds  = np.ones(len(X), dtype=int)

        df.loc[grupo.index, "ml_proba"]      = probas
        df.loc[grupo.index, "ml_prediccion"] = preds

    # Sin features -> predicción neutra (no filtrar, dejar pasar)
    df.loc[~mask_feat, "ml_proba"]      = 0.5
    df.loc[~mask_feat, "ml_prediccion"] = 1

    aprobadas = (df["ml_prediccion"] == 1).sum()
    rechazadas = (df["ml_prediccion"] == 0).sum()
    print(f"    ML aprueba: {aprobadas:,}  |  ML rechaza: {rechazadas:,}  "
          f"({rechazadas/len(df)*100:.1f}% rechazado)")

    return df


# ─────────────────────────────────────────────────────────────
# Cálculo de métricas comparativas
# ─────────────────────────────────────────────────────────────

def _metricas_grupo(df_ops: pd.DataFrame) -> dict:
    """
    Wrapper de calcular_metricas que retorna el dict resumido.
    Acepta DataFrame con columnas: resultado, retorno_pct, dias_posicion.
    """
    if df_ops.empty:
        return {
            "ops": 0, "win_rate": 0.0, "ret_promedio": 0.0,
            "ret_total": 0.0, "profit_factor": 0.0, "max_dd": 0.0,
        }
    m = calcular_metricas(df_ops)
    return {
        "ops":          m["total_operaciones"],
        "win_rate":     m["win_rate"],
        "ret_promedio": m["retorno_promedio_pct"],
        "ret_total":    m["retorno_total_pct"],
        "profit_factor": m["profit_factor"],
        "max_dd":       m["max_drawdown_pct"],
    }


def calcular_comparacion_completa(
    df: pd.DataFrame,
    umbral: float,
) -> pd.DataFrame:
    """
    Calcula la tabla comparativa Original vs ML-filtrado para:
        - Todas las combinaciones E×S (16)
        - Todos los segmentos (TRAIN, TEST, BACKTEST)
        - Todos los scopes (GLOBAL + 4 sectores)

    Returns:
        DataFrame con una fila por (EE, ES, scope, segmento)
    """
    scopes = ["GLOBAL"] + sorted(df["sector"].unique().tolist())
    registros = []

    for scope in scopes:
        df_scope = df if scope == "GLOBAL" else df[df["sector"] == scope]

        for ee in sorted(df_scope["estrategia_entrada"].unique()):
            for es in sorted(df_scope["estrategia_salida"].unique()):
                df_combo = df_scope[
                    (df_scope["estrategia_entrada"] == ee) &
                    (df_scope["estrategia_salida"] == es)
                ]
                for seg in ["TRAIN", "TEST", "BACKTEST"]:
                    grupo = df_combo[df_combo["segmento"] == seg]
                    if len(grupo) < 5:  # mínimo de ops para calcular métricas
                        continue

                    orig     = _metricas_grupo(grupo)
                    filtrado = _metricas_grupo(grupo[grupo["ml_prediccion"] == 1])
                    rechazad = _metricas_grupo(grupo[grupo["ml_prediccion"] == 0])

                    pct_rechazo = (rechazad["ops"] / orig["ops"]
                                   if orig["ops"] > 0 else 0.0)

                    registros.append({
                        "estrategia_entrada":       ee,
                        "estrategia_salida":        es,
                        "scope":                    scope,
                        "segmento":                 seg,
                        "umbral_ml":                round(umbral, 2),

                        # Baseline
                        "ops_original":             orig["ops"],
                        "win_rate_orig":            orig["win_rate"],
                        "ret_promedio_orig":        orig["ret_promedio"],
                        "ret_total_orig":           orig["ret_total"],
                        "profit_factor_orig":       orig["profit_factor"],
                        "max_dd_orig":              orig["max_dd"],

                        # ML-filtrado
                        "ops_ml":                   filtrado["ops"],
                        "ops_rechazadas":           rechazad["ops"],
                        "pct_rechazo":              round(pct_rechazo, 4),
                        "win_rate_ml":              filtrado["win_rate"],
                        "ret_promedio_ml":          filtrado["ret_promedio"],
                        "ret_total_ml":             filtrado["ret_total"],
                        "profit_factor_ml":         filtrado["profit_factor"],
                        "max_dd_ml":                filtrado["max_dd"],

                        # Rechazadas (verificación calidad del filtro)
                        "win_rate_rechazadas":      rechazad["win_rate"],
                        "ret_promedio_rechazadas":  rechazad["ret_promedio"],
                        "profit_factor_rechazadas": rechazad["profit_factor"],

                        # Deltas
                        "delta_win_rate":           round(filtrado["win_rate"] - orig["win_rate"], 4),
                        "delta_ret_promedio":       round(filtrado["ret_promedio"] - orig["ret_promedio"], 4),
                        "delta_profit_factor":      round(filtrado["profit_factor"] - orig["profit_factor"], 4),
                    })

    return pd.DataFrame(registros)


# ─────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────

def persistir_resultados(df: pd.DataFrame):
    """Persiste la tabla de comparación en resultados_ml_filter."""
    if df.empty:
        print("  [WARN] DataFrame vacío, nada que persistir.")
        return

    records = df.to_dict(orient="records")
    # Limpiar NaN
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None

    sql = """
        INSERT INTO resultados_ml_filter (
            estrategia_entrada, estrategia_salida, scope, segmento, umbral_ml,
            ops_original, win_rate_orig, ret_promedio_orig, ret_total_orig,
            profit_factor_orig, max_dd_orig,
            ops_ml, ops_rechazadas, pct_rechazo,
            win_rate_ml, ret_promedio_ml, ret_total_ml,
            profit_factor_ml, max_dd_ml,
            win_rate_rechazadas, ret_promedio_rechazadas, profit_factor_rechazadas,
            delta_win_rate, delta_ret_promedio, delta_profit_factor
        ) VALUES (
            %(estrategia_entrada)s, %(estrategia_salida)s, %(scope)s,
            %(segmento)s, %(umbral_ml)s,
            %(ops_original)s, %(win_rate_orig)s, %(ret_promedio_orig)s,
            %(ret_total_orig)s, %(profit_factor_orig)s, %(max_dd_orig)s,
            %(ops_ml)s, %(ops_rechazadas)s, %(pct_rechazo)s,
            %(win_rate_ml)s, %(ret_promedio_ml)s, %(ret_total_ml)s,
            %(profit_factor_ml)s, %(max_dd_ml)s,
            %(win_rate_rechazadas)s, %(ret_promedio_rechazadas)s,
            %(profit_factor_rechazadas)s,
            %(delta_win_rate)s, %(delta_ret_promedio)s, %(delta_profit_factor)s
        )
        ON CONFLICT (estrategia_entrada, estrategia_salida, scope, segmento, umbral_ml)
        DO UPDATE SET
            ops_original             = EXCLUDED.ops_original,
            win_rate_orig            = EXCLUDED.win_rate_orig,
            ret_promedio_orig        = EXCLUDED.ret_promedio_orig,
            ret_total_orig           = EXCLUDED.ret_total_orig,
            profit_factor_orig       = EXCLUDED.profit_factor_orig,
            max_dd_orig              = EXCLUDED.max_dd_orig,
            ops_ml                   = EXCLUDED.ops_ml,
            ops_rechazadas           = EXCLUDED.ops_rechazadas,
            pct_rechazo              = EXCLUDED.pct_rechazo,
            win_rate_ml              = EXCLUDED.win_rate_ml,
            ret_promedio_ml          = EXCLUDED.ret_promedio_ml,
            ret_total_ml             = EXCLUDED.ret_total_ml,
            profit_factor_ml         = EXCLUDED.profit_factor_ml,
            max_dd_ml                = EXCLUDED.max_dd_ml,
            win_rate_rechazadas      = EXCLUDED.win_rate_rechazadas,
            ret_promedio_rechazadas  = EXCLUDED.ret_promedio_rechazadas,
            profit_factor_rechazadas = EXCLUDED.profit_factor_rechazadas,
            delta_win_rate           = EXCLUDED.delta_win_rate,
            delta_ret_promedio       = EXCLUDED.delta_ret_promedio,
            delta_profit_factor      = EXCLUDED.delta_profit_factor,
            created_at               = NOW()
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=200)

    print(f"  Persistidos: {len(records):,} registros en resultados_ml_filter.")


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def ejecutar_evaluacion_ml(umbral: float = 0.50) -> pd.DataFrame:
    """
    Pipeline completo:
        1. Carga modelos desplegados (por sector)
        2. Carga operaciones + features del día señal
        3. Aplica filtro ML
        4. Calcula métricas comparativas (todas las E×S, sectores y segmentos)
        5. Persiste en resultados_ml_filter

    Returns:
        DataFrame completo con la comparación
    """
    crear_tabla_filter()

    print("\n  [1/4] Cargando modelos desplegados...")
    modelos = cargar_modelos_desplegados()

    print("\n  [2/4] Cargando operaciones + features señal...")
    df = cargar_operaciones_con_features()

    print(f"\n  [3/4] Aplicando filtro ML (umbral={umbral})...")
    df = aplicar_filtro_ml(df, modelos, umbral=umbral)

    print("\n  [4/4] Calculando métricas comparativas...")
    resultado = calcular_comparacion_completa(df, umbral=umbral)
    print(f"        {len(resultado):,} combinaciones calculadas "
          f"(E×S × scope × segmento)")

    persistir_resultados(resultado)

    return resultado
