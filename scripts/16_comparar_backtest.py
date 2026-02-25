"""
16_comparar_backtest.py
Comparacion PA Challenger vs Backtesting Original (Rule-Based).

Compara las 16 combinaciones EV x SV (PA) contra las 16 combinaciones
E x S (rule-based) en los segmentos TEST y BACKTEST.

Metricas comparadas:
    - win_rate, retorno_promedio_pct, retorno_total_pct
    - profit_factor, max_drawdown_pct, dias_promedio_posicion

Verdict:
    POSITIVO : PA supera en profit_factor en BACKTEST (promedio de las 16 combis)
    NEUTRAL  : PA empata (+/- 0.05 de diferencia)
    NEGATIVO : PA no mejora

Uso:
    python scripts/16_comparar_backtest.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import query_df


# ─────────────────────────────────────────────────────────────
# Helpers de consulta
# ─────────────────────────────────────────────────────────────

def get_global_resultados(tabla: str, segmento: str, min_ops: int = 5):
    """
    Obtiene los resultados globales (ticker IS NULL) de una tabla de resultados
    para un segmento dado. Retorna DataFrame ordenado por profit_factor.
    """
    sql = f"""
        SELECT
            estrategia_entrada, estrategia_salida,
            total_operaciones,
            ROUND(win_rate * 100, 2)              AS win_rate_pct,
            ROUND(retorno_promedio_pct, 3)         AS ret_promedio,
            ROUND(retorno_total_pct, 2)            AS ret_total,
            ROUND(profit_factor::NUMERIC, 3)       AS profit_factor,
            ROUND(max_drawdown_pct, 2)             AS max_dd,
            ROUND(dias_promedio_posicion, 1)       AS dias_prom
        FROM {tabla}
        WHERE segmento           = :segmento
          AND ticker             IS NULL
          AND total_operaciones  >= :min_ops
        ORDER BY profit_factor DESC
    """
    return query_df(sql, params={"segmento": segmento, "min_ops": min_ops})


def get_promedios(tabla: str, segmento: str, min_ops: int = 5):
    """
    Calcula los promedios de todas las combinaciones globales
    para comparacion entre sistemas.
    """
    sql = f"""
        SELECT
            COUNT(*)                              AS n_combis,
            ROUND(AVG(win_rate * 100), 3)         AS avg_wr_pct,
            ROUND(AVG(retorno_promedio_pct), 3)   AS avg_ret_promedio,
            ROUND(AVG(retorno_total_pct), 2)      AS avg_ret_total,
            ROUND(AVG(profit_factor::NUMERIC), 3) AS avg_pf,
            ROUND(AVG(max_drawdown_pct), 2)       AS avg_max_dd,
            ROUND(AVG(dias_promedio_posicion), 1) AS avg_dias,
            ROUND(SUM(total_operaciones), 0)      AS total_ops
        FROM {tabla}
        WHERE segmento           = :segmento
          AND ticker             IS NULL
          AND total_operaciones  >= :min_ops
    """
    r = query_df(sql, params={"segmento": segmento, "min_ops": min_ops})
    return r.iloc[0] if not r.empty else None


def get_top1(tabla: str, segmento: str, min_ops: int = 5):
    """Retorna la mejor combinacion (por profit_factor) de la tabla."""
    df = get_global_resultados(tabla, segmento, min_ops)
    if df.empty:
        return None
    return df.iloc[0]


# ─────────────────────────────────────────────────────────────
# Impresion de tablas
# ─────────────────────────────────────────────────────────────

def _header_ranking():
    print(f"  {'E/EV':<6} {'S/SV':<6} {'Ops':>5} {'WR%':>7} {'RetProm':>9} {'PF':>8} {'MaxDD':>8} {'Dias':>6}")
    print("  " + "-" * 58)


def _fila_ranking(row):
    print(
        f"  {row['estrategia_entrada']:<6} {row['estrategia_salida']:<6} "
        f"{int(row['total_operaciones']):>5} "
        f"{float(row['win_rate_pct']):>7.1f} "
        f"{float(row['ret_promedio']):>9.3f} "
        f"{float(row['profit_factor']):>8.3f} "
        f"{float(row['max_dd']):>8.2f} "
        f"{float(row['dias_prom']):>6.1f}"
    )


def imprimir_ranking_completo(df, titulo: str, n: int = 8):
    """Imprime hasta n filas de un ranking."""
    print(f"\n  {titulo}")
    _header_ranking()
    for _, row in df.head(n).iterrows():
        _fila_ranking(row)


def imprimir_comparacion_promedios(prom_orig, prom_pa, segmento: str):
    """Imprime la tabla de promedios comparativos entre sistemas."""
    if prom_orig is None or prom_pa is None:
        print(f"\n  [WARN] Datos insuficientes para comparar promedios en {segmento}.")
        return

    dwr  = float(prom_pa["avg_wr_pct"])   - float(prom_orig["avg_wr_pct"])
    dret = float(prom_pa["avg_ret_promedio"]) - float(prom_orig["avg_ret_promedio"])
    dpf  = float(prom_pa["avg_pf"])        - float(prom_orig["avg_pf"])
    ddd  = float(prom_pa["avg_max_dd"])    - float(prom_orig["avg_max_dd"])

    print(f"\n  PROMEDIOS (todas las combis globales) -- Segmento {segmento}:")
    print(f"  {'Metrica':<25} {'ORIGINAL':>10} {'PA':>10} {'Delta':>10}")
    print("  " + "-" * 58)
    print(f"  {'Combinaciones':<25} {int(prom_orig['n_combis']):>10} {int(prom_pa['n_combis']):>10}")
    print(f"  {'Ops totales':<25} {int(prom_orig['total_ops']):>10} {int(prom_pa['total_ops']):>10}")
    print(f"  {'Win Rate %':<25} {float(prom_orig['avg_wr_pct']):>10.2f} {float(prom_pa['avg_wr_pct']):>10.2f} {dwr:>+10.2f}")
    print(f"  {'Retorno prom %':<25} {float(prom_orig['avg_ret_promedio']):>10.3f} {float(prom_pa['avg_ret_promedio']):>10.3f} {dret:>+10.3f}")
    print(f"  {'Profit Factor':<25} {float(prom_orig['avg_pf']):>10.3f} {float(prom_pa['avg_pf']):>10.3f} {dpf:>+10.3f}")
    print(f"  {'Max Drawdown %':<25} {float(prom_orig['avg_max_dd']):>10.2f} {float(prom_pa['avg_max_dd']):>10.2f} {ddd:>+10.2f}")
    print(f"  {'Dias prom posicion':<25} {float(prom_orig['avg_dias']):>10.1f} {float(prom_pa['avg_dias']):>10.1f}")


def imprimir_top1_comparacion(top_orig, top_pa, segmento: str):
    """Imprime la comparacion del top-1 entre sistemas."""
    if top_orig is None or top_pa is None:
        return

    print(f"\n  MEJOR COMBINACION -- Segmento {segmento}:")
    print(f"  {'Sistema':<12} {'Estrategia':<14} {'Ops':>5} {'WR%':>7} {'PF':>8} {'MaxDD':>8}")
    print("  " + "-" * 58)
    print(
        f"  {'ORIGINAL':<12} "
        f"{top_orig['estrategia_entrada']}/{top_orig['estrategia_salida']:<10} "
        f"{int(top_orig['total_operaciones']):>5} "
        f"{float(top_orig['win_rate_pct']):>7.1f} "
        f"{float(top_orig['profit_factor']):>8.3f} "
        f"{float(top_orig['max_dd']):>8.2f}"
    )
    print(
        f"  {'PA':<12} "
        f"{top_pa['estrategia_entrada']}/{top_pa['estrategia_salida']:<10} "
        f"{int(top_pa['total_operaciones']):>5} "
        f"{float(top_pa['win_rate_pct']):>7.1f} "
        f"{float(top_pa['profit_factor']):>8.3f} "
        f"{float(top_pa['max_dd']):>8.2f}"
    )


# ─────────────────────────────────────────────────────────────
# Verificacion de datos
# ─────────────────────────────────────────────────────────────

def verificar_datos():
    """Verifica que ambas tablas tienen datos."""
    r = query_df("""
        SELECT
            (SELECT COUNT(*) FROM resultados_backtest WHERE ticker IS NULL) AS n_orig,
            (SELECT COUNT(*) FROM resultados_bt_pa    WHERE ticker IS NULL) AS n_pa
    """)
    row = r.iloc[0]
    n_orig = int(row["n_orig"])
    n_pa   = int(row["n_pa"])
    print(f"  resultados_backtest (global): {n_orig:,} filas")
    print(f"  resultados_bt_pa    (global): {n_pa:,} filas")
    if n_orig == 0:
        raise RuntimeError("resultados_backtest vacio. Ejecutar script 03.")
    if n_pa == 0:
        raise RuntimeError("resultados_bt_pa vacio. Ejecutar script 15.")
    return n_orig, n_pa


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  COMPARACION: PA CHALLENGER vs BACKTESTING ORIGINAL")
    print("=" * 65)
    print(f"  Sistema original : E1-E4 x S1-S4 (scoring rule-based)")
    print(f"  Sistema PA       : EV1-EV4 x SV1-SV4 (price action + MS)")
    print(f"  Inicio           : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("\n  Verificando datos...")
    try:
        verificar_datos()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # ── Seccion 1: Rankings originales TEST ───────────────────
    print("\n" + "=" * 65)
    print("  1. RANKING ORIGINAL -- TEST")
    print("=" * 65)
    df_orig_test = get_global_resultados("resultados_backtest", "TEST")
    imprimir_ranking_completo(df_orig_test, "Top 8 E x S en TEST:", n=8)

    # ── Seccion 2: Rankings PA TEST ───────────────────────────
    print("\n" + "=" * 65)
    print("  2. RANKING PA -- TEST")
    print("=" * 65)
    df_pa_test = get_global_resultados("resultados_bt_pa", "TEST")
    imprimir_ranking_completo(df_pa_test, "Top 8 EV x SV en TEST:", n=8)

    # ── Seccion 3: Rankings originales BACKTEST ───────────────
    print("\n" + "=" * 65)
    print("  3. RANKING ORIGINAL -- BACKTEST")
    print("=" * 65)
    df_orig_bt = get_global_resultados("resultados_backtest", "BACKTEST")
    imprimir_ranking_completo(df_orig_bt, "Top 8 E x S en BACKTEST:", n=8)

    # ── Seccion 4: Rankings PA BACKTEST ───────────────────────
    print("\n" + "=" * 65)
    print("  4. RANKING PA -- BACKTEST")
    print("=" * 65)
    df_pa_bt = get_global_resultados("resultados_bt_pa", "BACKTEST")
    imprimir_ranking_completo(df_pa_bt, "Top 8 EV x SV en BACKTEST:", n=8)

    # ── Seccion 5: Comparacion de promedios ───────────────────
    print("\n" + "=" * 65)
    print("  5. COMPARACION DE PROMEDIOS")
    print("=" * 65)

    for seg in ["TEST", "BACKTEST"]:
        prom_orig = get_promedios("resultados_backtest", seg)
        prom_pa   = get_promedios("resultados_bt_pa", seg)
        imprimir_comparacion_promedios(prom_orig, prom_pa, seg)

    # ── Seccion 6: Comparacion top-1 ──────────────────────────
    print("\n" + "=" * 65)
    print("  6. COMPARACION TOP-1")
    print("=" * 65)

    for seg in ["TEST", "BACKTEST"]:
        top_orig = get_top1("resultados_backtest", seg)
        top_pa   = get_top1("resultados_bt_pa", seg)
        imprimir_top1_comparacion(top_orig, top_pa, seg)

    # ── Seccion 7: Verdict ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  7. VERDICT FINAL")
    print("=" * 65)

    prom_orig_bt = get_promedios("resultados_backtest", "BACKTEST")
    prom_pa_bt   = get_promedios("resultados_bt_pa",    "BACKTEST")

    if prom_orig_bt is not None and prom_pa_bt is not None:
        delta_pf  = float(prom_pa_bt["avg_pf"])  - float(prom_orig_bt["avg_pf"])
        delta_wr  = float(prom_pa_bt["avg_wr_pct"]) - float(prom_orig_bt["avg_wr_pct"])
        delta_ret = float(prom_pa_bt["avg_ret_promedio"]) - float(prom_orig_bt["avg_ret_promedio"])

        if delta_pf > 0.05:
            verdict = "POSITIVO"
            desc    = "El sistema PA supera al sistema original en Profit Factor (BACKTEST)."
        elif delta_pf > -0.05:
            verdict = "NEUTRAL"
            desc    = "El sistema PA empata con el sistema original en Profit Factor (BACKTEST)."
        else:
            verdict = "NEGATIVO"
            desc    = "El sistema PA NO supera al sistema original en Profit Factor (BACKTEST)."

        print(f"\n  Sistema evaluado  : PA Challenger (EV1-EV4 x SV1-SV4)")
        print(f"  Referencia        : Original (E1-E4 x S1-S4)")
        print(f"  Segmento          : BACKTEST (mas out-of-sample)")
        print(f"\n  delta_PF (prom 16 combis) : {delta_pf:+.4f}")
        print(f"  delta_WR%                 : {delta_wr:+.2f}%")
        print(f"  delta_RetProm             : {delta_ret:+.3f}%")
        print(f"\n  DICTAMEN: {verdict}")
        print(f"  {desc}")

        top_pa_bt = get_top1("resultados_bt_pa", "BACKTEST")
        if top_pa_bt is not None:
            print(f"\n  Mejor combinacion PA (BACKTEST):")
            print(f"    {top_pa_bt['estrategia_entrada']}/{top_pa_bt['estrategia_salida']} "
                  f"| PF={float(top_pa_bt['profit_factor']):.3f} "
                  f"| WR={float(top_pa_bt['win_rate_pct']):.1f}% "
                  f"| RetProm={float(top_pa_bt['ret_promedio']):.3f}%")
    else:
        print("\n  [WARN] Datos insuficientes para calcular verdict.")

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n{'='*65}")
    print(f"  Comparacion completada en {duracion}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
