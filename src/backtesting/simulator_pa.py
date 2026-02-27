"""
simulator_pa.py
Motor de simulacion de operaciones Long para el backtesting PA challenger.

Usa features de precio/accion y estructura de mercado como senales de
entrada y salida (EV1-EV4 x SV1-SV4).

Carga datos de 5 tablas:
    precios_diarios + indicadores_tecnicos + features_precio_accion
    + features_market_structure + scoring_tecnico

Escribe en operaciones_bt_pa (sin tocar operaciones_backtest).

Reglas de ejecucion:
    - Una posicion abierta a la vez por ticker
    - Senal en dia T -> entrada al OPEN del dia T+1
    - SL/TP evaluados contra HIGH/LOW intraday
    - Cierre por condicion estructural o score: al CLOSE del dia
    - Posicion abierta al final del historico: cierre forzado al CLOSE

Periodo fijo: desde FECHA_INICIO_BT (2023-01-01) hasta hoy.
Las estrategias son reglas fijas (no ML), no hay data leakage.
Un solo segmento "FULL" — sin split TRAIN/TEST/BACKTEST.
"""

import pandas as pd
import numpy as np
from typing import List
from src.utils.config import ALL_TICKERS
from src.backtesting.strategies_pa import (
    check_entrada_pa,
    calcular_stops_iniciales_pa,
    check_salida_pa,
    clasificar_resultado_pa,
)
from src.data.database import query_df, get_connection


# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────

FECHA_INICIO_BT        = "2023-01-01"   # Fecha fija de inicio del backtesting
ESTRATEGIAS_ENTRADA_PA = ["EV1", "EV2", "EV3", "EV4"]
ESTRATEGIAS_SALIDA_PA  = ["SV1", "SV2", "SV3", "SV4"]


# ─────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────

def cargar_datos_ticker_pa(ticker: str) -> pd.DataFrame:
    """
    Carga datos combinados de 5 tablas para un ticker desde FECHA_INICIO_BT.

    LEFT JOIN para features_precio_accion, features_market_structure y scoring_tecnico:
    las primeras N barras no tienen estas features (periodo de calentamiento).
    """
    sql = """
        SELECT
            p.fecha,
            p.open, p.high, p.low, p.close, p.volume,
            i.atr14,

            -- Features precio/accion (pueden ser NULL)
            pa.es_alcista,
            pa.patron_hammer,
            pa.patron_engulfing_bull,
            pa.vol_spike,
            pa.up_vol_5d,
            pa.vol_price_confirm,

            -- Features market structure (pueden ser NULL)
            ms.estructura_5,    ms.estructura_10,
            ms.bos_bull_5,      ms.bos_bear_5,
            ms.choch_bull_5,    ms.choch_bear_5,
            ms.bos_bull_10,     ms.bos_bear_10,
            ms.choch_bull_10,   ms.choch_bear_10,
            ms.dist_sh_5_pct,   ms.dist_sl_5_pct,
            ms.dias_sh_5,       ms.dias_sl_5,
            ms.dist_sh_10_pct,  ms.dist_sl_10_pct,
            ms.dias_sh_10,      ms.dias_sl_10,
            ms.impulso_5_pct,   ms.impulso_10_pct,

            -- Scoring rule-based (puede ser NULL, usado en SV3)
            s.score_ponderado

        FROM precios_diarios p
        JOIN  indicadores_tecnicos          i  ON p.ticker = i.ticker  AND p.fecha = i.fecha
        LEFT JOIN features_precio_accion   pa  ON p.ticker = pa.ticker AND p.fecha = pa.fecha
        LEFT JOIN features_market_structure ms  ON p.ticker = ms.ticker AND p.fecha = ms.fecha
        LEFT JOIN scoring_tecnico           s  ON p.ticker = s.ticker  AND p.fecha = s.fecha
        WHERE p.ticker = :ticker
          AND p.fecha  >= :fecha_inicio
        ORDER BY p.fecha ASC
    """
    df = query_df(sql, params={"ticker": ticker, "fecha_inicio": FECHA_INICIO_BT})
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Simulador de un segmento
# ─────────────────────────────────────────────────────────────

def simular_segmento_pa(df: pd.DataFrame, ticker: str,
                         est_entrada: str, est_salida: str,
                         segmento: str) -> List[dict]:
    """
    Simula operaciones Long PA para un ticker, estrategia y segmento.

    Reglas:
        - Una posicion abierta a la vez
        - Senal en dia T -> entrada al OPEN de T+1
        - SL evaluado intraday (low <= stop_loss)
        - TP fijo evaluado intraday (SV4): high >= take_profit
        - Salida por condicion estructural/score: al CLOSE del dia
        - Timeout: al CLOSE del ultimo dia permitido
        - Fin de segmento con posicion abierta: cierre forzado al ultimo CLOSE

    Returns:
        Lista de dicts con el detalle de cada operacion cerrada
    """
    operaciones    = []
    n              = len(df)

    en_posicion    = False
    precio_entrada = None
    fecha_entrada  = None
    score_entrada  = None
    stop_loss      = None
    take_profit    = None
    dias_posicion  = 0

    for i in range(n):
        row = df.iloc[i]

        # ── FUERA DE POSICION: evaluar entrada ────────────────
        if not en_posicion:
            if check_entrada_pa(row, est_entrada):
                if i + 1 >= n:
                    break   # no hay dia siguiente

                next_row = df.iloc[i + 1]
                if pd.isna(next_row["open"]) or next_row["open"] <= 0:
                    continue

                atr_hoy = float(row["atr14"]) if not pd.isna(row["atr14"]) \
                          else float(row["close"]) * 0.02

                precio_entrada = float(next_row["open"])
                fecha_entrada  = next_row["fecha"]

                # score_entrada: score_ponderado disponible o None
                sp = row.get("score_ponderado")
                score_entrada = round(float(sp), 4) if (sp is not None and not pd.isna(sp)) else None

                stop_loss, take_profit = calcular_stops_iniciales_pa(
                    precio_entrada, atr_hoy, est_salida
                )
                dias_posicion = 0
                en_posicion   = True

            continue   # avanzar al siguiente dia

        # ── EN POSICION: gestion diaria ───────────────────────
        dias_posicion += 1

        cerrar, motivo, precio_salida = check_salida_pa(
            row, est_salida, stop_loss, take_profit, dias_posicion
        )

        if cerrar:
            retorno_pct = (precio_salida / precio_entrada - 1) * 100
            resultado   = clasificar_resultado_pa(retorno_pct)

            operaciones.append({
                "estrategia_entrada": est_entrada,
                "estrategia_salida":  est_salida,
                "ticker":             ticker,
                "segmento":           segmento,
                "fecha_entrada":      fecha_entrada.date() if hasattr(fecha_entrada, "date") else fecha_entrada,
                "precio_entrada":     round(precio_entrada, 4),
                "score_entrada":      score_entrada,
                "fecha_salida":       row["fecha"].date() if hasattr(row["fecha"], "date") else row["fecha"],
                "precio_salida":      precio_salida,
                "motivo_salida":      motivo,
                "dias_posicion":      dias_posicion,
                "retorno_pct":        round(retorno_pct, 4),
                "resultado":          resultado,
                "stop_loss":          round(stop_loss, 4) if stop_loss is not None else None,
                "take_profit":        round(take_profit, 4) if take_profit is not None else None,
            })

            # Resetear estado
            en_posicion    = False
            precio_entrada = None
            stop_loss      = None
            take_profit    = None

    # ── Posicion abierta al final del segmento: cierre forzado
    if en_posicion and precio_entrada is not None:
        precio_salida = float(df.iloc[-1]["close"])
        retorno_pct   = (precio_salida / precio_entrada - 1) * 100
        operaciones.append({
            "estrategia_entrada": est_entrada,
            "estrategia_salida":  est_salida,
            "ticker":             ticker,
            "segmento":           segmento,
            "fecha_entrada":      fecha_entrada.date() if hasattr(fecha_entrada, "date") else fecha_entrada,
            "precio_entrada":     round(precio_entrada, 4),
            "score_entrada":      score_entrada,
            "fecha_salida":       df.iloc[-1]["fecha"].date(),
            "precio_salida":      round(precio_salida, 4),
            "motivo_salida":      "FIN_SEGMENTO",
            "dias_posicion":      dias_posicion,
            "retorno_pct":        round(retorno_pct, 4),
            "resultado":          clasificar_resultado_pa(retorno_pct),
            "stop_loss":          round(stop_loss, 4) if stop_loss is not None else None,
            "take_profit":        round(take_profit, 4) if take_profit is not None else None,
        })

    return operaciones


# ─────────────────────────────────────────────────────────────
# Backtesting completo: todos los tickers x todas las combis
# ─────────────────────────────────────────────────────────────

def ejecutar_backtesting_pa(tickers: list = None,
                             guardar_db: bool = True,
                             truncate: bool = True) -> pd.DataFrame:
    """
    Ejecuta la matriz 4x4 PA para todos los tickers.

    Usa el periodo completo desde FECHA_INICIO_BT hasta hoy,
    sin split TRAIN/TEST/BACKTEST (las estrategias son reglas fijas, sin entrenamiento).
    Todas las operaciones se etiquetan con segmento = "FULL".

    Args:
        tickers:    lista de tickers (None = ALL_TICKERS)
        guardar_db: persistir operaciones en PostgreSQL
        truncate:   si True (default), TRUNCATE antes de insertar;
                    si False, solo borra registros de los tickers especificados
                    (util para incorporar tickers nuevos sin perder historial)

    Returns:
        DataFrame con todas las operaciones simuladas
    """
    tickers = tickers or ALL_TICKERS

    todas_ops    = []
    total_combis = len(tickers) * len(ESTRATEGIAS_ENTRADA_PA) * len(ESTRATEGIAS_SALIDA_PA)

    print(f"\n  Periodo: {FECHA_INICIO_BT} -> hoy  |  Segmento: FULL")
    print(f"  {total_combis} combinaciones ({len(tickers)} tickers x 4 entradas x 4 salidas)\n")
    print("-" * 65)

    for ticker in tickers:
        print(f"  [{ticker}] Cargando datos...", end=" ")
        df_full = cargar_datos_ticker_pa(ticker)

        if len(df_full) < 50:
            print("insuficientes datos, omitido.")
            continue

        print(f"OK | {len(df_full)} barras ({df_full['fecha'].min().date()} a {df_full['fecha'].max().date()})")

        for ee in ESTRATEGIAS_ENTRADA_PA:
            for es in ESTRATEGIAS_SALIDA_PA:
                ops = simular_segmento_pa(df_full, ticker, ee, es, "FULL")
                todas_ops.extend(ops)

    print("-" * 65)
    print(f"  Simulacion PA completada: {len(todas_ops):,} operaciones totales.")

    df_ops = pd.DataFrame(todas_ops) if todas_ops else pd.DataFrame()

    if guardar_db and not df_ops.empty:
        _persistir_operaciones_pa(df_ops, truncate=truncate,
                                  tickers_filtro=tickers if not truncate else None)

    return df_ops


def _persistir_operaciones_pa(df_ops: pd.DataFrame,
                               truncate: bool = True,
                               tickers_filtro: list = None):
    """
    Persiste operaciones en operaciones_bt_pa.

    Si truncate=True (default): TRUNCATE ambas tablas antes de insertar.
    Si truncate=False: DELETE solo los registros de tickers_filtro,
    luego INSERT (util para agregar tickers nuevos sin borrar historial).
    """
    import psycopg2.extras
    from src.data.database import ejecutar_sql

    if truncate:
        ejecutar_sql("TRUNCATE TABLE operaciones_bt_pa RESTART IDENTITY")
        ejecutar_sql("TRUNCATE TABLE resultados_bt_pa  RESTART IDENTITY")
    elif tickers_filtro:
        # Borrar solo los tickers que vamos a re-insertar (idempotente)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM operaciones_bt_pa WHERE ticker = ANY(%s)",
                    (tickers_filtro,)
                )

    records = df_ops.to_dict(orient="records")

    sql = """
        INSERT INTO operaciones_bt_pa
            (estrategia_entrada, estrategia_salida, ticker, segmento,
             fecha_entrada, precio_entrada, score_entrada,
             fecha_salida, precio_salida, motivo_salida,
             dias_posicion, retorno_pct, resultado,
             stop_loss, take_profit)
        VALUES
            (%(estrategia_entrada)s, %(estrategia_salida)s, %(ticker)s, %(segmento)s,
             %(fecha_entrada)s, %(precio_entrada)s, %(score_entrada)s,
             %(fecha_salida)s, %(precio_salida)s, %(motivo_salida)s,
             %(dias_posicion)s, %(retorno_pct)s, %(resultado)s,
             %(stop_loss)s, %(take_profit)s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

    print(f"  Operaciones PA persistidas: {len(records):,} registros.")


# ─────────────────────────────────────────────────────────────
# Helper incremental: ultima barra de un ticker
# ─────────────────────────────────────────────────────────────

def _cargar_ultima_barra_ticker(ticker: str):
    """
    Carga la ultima barra disponible para un ticker con todas sus features.
    Retorna pd.Series o None si no hay datos o faltan indicadores.
    """
    sql = """
        SELECT
            p.fecha,
            p.open, p.high, p.low, p.close, p.volume,
            i.atr14,

            pa.es_alcista,
            pa.patron_hammer,
            pa.patron_engulfing_bull,
            pa.vol_spike,
            pa.up_vol_5d,
            pa.vol_price_confirm,

            ms.estructura_5,    ms.estructura_10,
            ms.bos_bull_5,      ms.bos_bear_5,
            ms.choch_bull_5,    ms.choch_bear_5,
            ms.bos_bull_10,     ms.bos_bear_10,
            ms.choch_bull_10,   ms.choch_bear_10,
            ms.dist_sh_5_pct,   ms.dist_sl_5_pct,
            ms.dias_sh_5,       ms.dias_sl_5,
            ms.dist_sh_10_pct,  ms.dist_sl_10_pct,
            ms.dias_sh_10,      ms.dias_sl_10,
            ms.impulso_5_pct,   ms.impulso_10_pct,

            s.score_ponderado

        FROM precios_diarios p
        JOIN  indicadores_tecnicos          i  ON p.ticker = i.ticker  AND p.fecha = i.fecha
        LEFT JOIN features_precio_accion   pa  ON p.ticker = pa.ticker AND p.fecha = pa.fecha
        LEFT JOIN features_market_structure ms  ON p.ticker = ms.ticker AND p.fecha = ms.fecha
        LEFT JOIN scoring_tecnico           s  ON p.ticker = s.ticker  AND p.fecha = s.fecha
        WHERE p.ticker = :ticker
        ORDER BY p.fecha DESC
        LIMIT 1
    """
    df = query_df(sql, params={"ticker": ticker})
    if df.empty:
        return None
    row = df.iloc[0].copy()
    row["fecha"] = pd.to_datetime(row["fecha"])
    return row


# ─────────────────────────────────────────────────────────────
# Backtesting incremental: solo el dia actual
# ─────────────────────────────────────────────────────────────

def ejecutar_backtesting_pa_incremental(tickers: list = None,
                                         guardar_db: bool = True) -> dict:
    """
    Actualiza el backtesting PA de forma incremental procesando solo el dia actual.

    Flujo:
        1. Lee posiciones abiertas (FIN_SEGMENTO) de operaciones_bt_pa.
        2. Para cada posicion: evalua si se cierra hoy con la ultima barra disponible.
           - Si cierra: UPDATE con motivo/precio/retorno reales.
           - Si sigue abierta: UPDATE precio_salida=close_actual, dias+1, retorno actual.
        3. Para combos (ticker x EV x SV) sin posicion: evalua entrada en la ultima barra.
           - Si hay senal: INSERT FIN_SEGMENTO (precio_entrada = close del dia de senal).

    Simplificacion vs full-rerun:
        - Entrada al close del dia de senal (no al open de T+1).
        - dias_posicion se incrementa en 1 por cada ejecucion del cron (aprox. trading days).

    Returns:
        dict con {"cerradas": n, "actualizadas": n, "nuevas": n}

    Prerequisito: columnas stop_loss y take_profit deben existir en operaciones_bt_pa.
    Ejecutar scripts/21_migrar_bt_incremental.py la primera vez.
    """
    import psycopg2.extras

    tickers = tickers or ALL_TICKERS

    # ── 1. Cargar posiciones abiertas ─────────────────────────────────────────
    df_abiertas = query_df("""
        SELECT id, estrategia_entrada, estrategia_salida, ticker, segmento,
               fecha_entrada, precio_entrada, score_entrada,
               stop_loss, take_profit, dias_posicion
        FROM operaciones_bt_pa
        WHERE motivo_salida = 'FIN_SEGMENTO'
    """)
    if not df_abiertas.empty:
        df_abiertas = df_abiertas[df_abiertas["ticker"].isin(tickers)].copy()

    # ── 2. Cargar ultima barra de cada ticker afectado ─────────────────────────
    tickers_abiertos = df_abiertas["ticker"].unique().tolist() if not df_abiertas.empty else []
    tickers_todos    = list(set(list(tickers) + tickers_abiertos))

    ultima_barra = {}
    for t in tickers_todos:
        row = _cargar_ultima_barra_ticker(t)
        if row is not None:
            ultima_barra[t] = row

    # ── 3. Procesar posiciones abiertas ───────────────────────────────────────
    ops_cerrar       = []
    ops_update       = []
    combos_ocupados  = set()   # (ticker, ee, es) con posicion abierta

    for _, pos in df_abiertas.iterrows():
        ticker = pos["ticker"]
        ee     = pos["estrategia_entrada"]
        es     = pos["estrategia_salida"]

        if ticker not in ultima_barra:
            combos_ocupados.add((ticker, ee, es))
            continue

        row = ultima_barra[ticker]
        pe  = float(pos["precio_entrada"])

        # Reconstruir stop_loss (fallback si NULL en DB)
        sl_raw = pos["stop_loss"]
        sl = pe * 0.95 if (sl_raw is None or pd.isna(sl_raw)) else float(sl_raw)

        tp_raw = pos["take_profit"]
        tp = None if (tp_raw is None or pd.isna(tp_raw)) else float(tp_raw)

        dias = int(pos["dias_posicion"]) + 1

        cerrar, motivo, precio_salida = check_salida_pa(row, es, sl, tp, dias)

        if cerrar:
            retorno_pct = (float(precio_salida) / pe - 1) * 100
            ops_cerrar.append({
                "id":            int(pos["id"]),
                "fecha_salida":  row["fecha"].date() if hasattr(row["fecha"], "date") else row["fecha"],
                "precio_salida": round(float(precio_salida), 4),
                "motivo_salida": motivo,
                "dias_posicion": dias,
                "retorno_pct":   round(retorno_pct, 4),
                "resultado":     clasificar_resultado_pa(retorno_pct),
            })
            # combo queda libre para nueva entrada

        else:
            # Sigue abierta: actualizar precio actual y dias
            combos_ocupados.add((ticker, ee, es))
            close_actual   = float(row["close"])
            retorno_actual = (close_actual / pe - 1) * 100
            ops_update.append({
                "id":            int(pos["id"]),
                "precio_salida": round(close_actual, 4),
                "dias_posicion": dias,
                "retorno_pct":   round(retorno_actual, 4),
                "resultado":     clasificar_resultado_pa(retorno_actual),
            })

    # ── 4. Evaluar nuevas entradas ────────────────────────────────────────────
    ops_nuevas = []

    for ticker in tickers:
        if ticker not in ultima_barra:
            continue

        row = ultima_barra[ticker]

        for ee in ESTRATEGIAS_ENTRADA_PA:
            if not check_entrada_pa(row, ee):
                continue

            # Hay senal de entrada para este ticker + estrategia de entrada
            atr_raw = row["atr14"] if "atr14" in row.index else None
            atr = float(atr_raw) if (atr_raw is not None and not pd.isna(atr_raw)) \
                  else float(row["close"]) * 0.02

            precio_entrada = float(row["close"])   # simplificacion: close del dia de senal

            sp = row["score_ponderado"] if "score_ponderado" in row.index else None
            score_entrada = round(float(sp), 4) \
                if (sp is not None and not pd.isna(sp)) else None

            fecha_entrada = row["fecha"].date() if hasattr(row["fecha"], "date") else row["fecha"]

            for es in ESTRATEGIAS_SALIDA_PA:
                if (ticker, ee, es) in combos_ocupados:
                    continue  # ya hay posicion abierta para este combo

                sl, tp = calcular_stops_iniciales_pa(precio_entrada, atr, es)

                ops_nuevas.append({
                    "estrategia_entrada": ee,
                    "estrategia_salida":  es,
                    "ticker":             ticker,
                    "segmento":           "FULL",
                    "fecha_entrada":      fecha_entrada,
                    "precio_entrada":     round(precio_entrada, 4),
                    "score_entrada":      score_entrada,
                    "fecha_salida":       fecha_entrada,   # mismo dia, se actualiza diariamente
                    "precio_salida":      round(precio_entrada, 4),
                    "motivo_salida":      "FIN_SEGMENTO",
                    "dias_posicion":      0,
                    "retorno_pct":        0.0,
                    "resultado":          "NEUTRO",
                    "stop_loss":          sl,
                    "take_profit":        tp,
                })

    # ── 5. Persistir en DB ────────────────────────────────────────────────────
    if guardar_db:
        with get_connection() as conn:
            with conn.cursor() as cur:

                if ops_cerrar:
                    psycopg2.extras.execute_batch(cur, """
                        UPDATE operaciones_bt_pa SET
                            fecha_salida  = %(fecha_salida)s,
                            precio_salida = %(precio_salida)s,
                            motivo_salida = %(motivo_salida)s,
                            dias_posicion = %(dias_posicion)s,
                            retorno_pct   = %(retorno_pct)s,
                            resultado     = %(resultado)s
                        WHERE id = %(id)s
                    """, ops_cerrar, page_size=100)

                if ops_update:
                    psycopg2.extras.execute_batch(cur, """
                        UPDATE operaciones_bt_pa SET
                            precio_salida = %(precio_salida)s,
                            dias_posicion = %(dias_posicion)s,
                            retorno_pct   = %(retorno_pct)s,
                            resultado     = %(resultado)s
                        WHERE id = %(id)s
                    """, ops_update, page_size=200)

                if ops_nuevas:
                    psycopg2.extras.execute_batch(cur, """
                        INSERT INTO operaciones_bt_pa
                            (estrategia_entrada, estrategia_salida, ticker, segmento,
                             fecha_entrada, precio_entrada, score_entrada,
                             fecha_salida, precio_salida, motivo_salida,
                             dias_posicion, retorno_pct, resultado,
                             stop_loss, take_profit)
                        VALUES
                            (%(estrategia_entrada)s, %(estrategia_salida)s, %(ticker)s, %(segmento)s,
                             %(fecha_entrada)s, %(precio_entrada)s, %(score_entrada)s,
                             %(fecha_salida)s, %(precio_salida)s, %(motivo_salida)s,
                             %(dias_posicion)s, %(retorno_pct)s, %(resultado)s,
                             %(stop_loss)s, %(take_profit)s)
                    """, ops_nuevas, page_size=200)

    print(f"  BT Incremental: {len(ops_cerrar)} cerradas, "
          f"{len(ops_update)} actualizadas, "
          f"{len(ops_nuevas)} nuevas.")

    return {
        "cerradas":     len(ops_cerrar),
        "actualizadas": len(ops_update),
        "nuevas":       len(ops_nuevas),
    }
