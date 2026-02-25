"""
08_ml_backtest.py
Runner de la Fase 6 — Evaluacion del filtro ML sobre el backtesting.

Aplica los modelos ML desplegados como filtro adicional sobre las
operaciones registradas en operaciones_backtest y compara:

    ORIGINAL    : todas las operaciones con senal de entrada
    ML-FILTRADO : solo las operaciones donde el modelo predijo GANANCIA
    RECHAZADAS  : operaciones bloqueadas por ML (calidad del filtro)

Uso:
    python scripts/08_ml_backtest.py
    python scripts/08_ml_backtest.py --umbral 0.55
    python scripts/08_ml_backtest.py --estrategia E3 S2
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.ml_backtest import ejecutar_evaluacion_ml
from src.data.database import query_df, ejecutar_sql


# ─────────────────────────────────────────────────────────────
# Helpers de logging
# ─────────────────────────────────────────────────────────────

def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) "
            "VALUES (%s,%s,%s,%s)",
            ("08_ml_backtest", accion, detalle, estado)
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Tablas de resultados
# ─────────────────────────────────────────────────────────────

def _signo(val):
    """Formatea un delta con signo (+/-) y color textual."""
    if val is None:
        return "  N/A "
    return f"+{val:.2f}" if val >= 0 else f" {val:.2f}"


def imprimir_resumen_global(umbral: float):
    """
    Tabla 1: Resumen GLOBAL por estrategia E x S — segmentos TEST y BACKTEST.
    Muestra las 16 combinaciones con sus metricas Orig vs ML y deltas.
    """
    sql = """
        SELECT
            estrategia_entrada  AS ee,
            estrategia_salida   AS es,
            segmento            AS seg,
            ops_original        AS ops_o,
            ROUND(win_rate_orig      * 100, 1)  AS wr_o,
            ROUND(profit_factor_orig,        2) AS pf_o,
            ROUND(ret_promedio_orig,         2) AS ret_o,
            ops_ml,
            pct_rechazo,
            ROUND(win_rate_ml        * 100, 1)  AS wr_ml,
            ROUND(profit_factor_ml,          2) AS pf_ml,
            ROUND(ret_promedio_ml,           2) AS ret_ml,
            ROUND(delta_win_rate     * 100, 2)  AS d_wr,
            ROUND(delta_profit_factor,       2) AS d_pf,
            ROUND(delta_ret_promedio,        2) AS d_ret
        FROM resultados_ml_filter
        WHERE scope      = 'GLOBAL'
          AND segmento   IN ('TEST', 'BACKTEST')
          AND umbral_ml  = :umbral
        ORDER BY
            CASE segmento WHEN 'TEST' THEN 1 ELSE 2 END,
            pf_ml DESC
    """
    df = query_df(sql, params={"umbral": umbral})
    if df.empty:
        print("  [WARN] Sin datos en resultados_ml_filter.")
        return

    print(f"\n{'='*100}")
    print("  TABLA 1 — GLOBAL: Original vs ML-Filtrado  (TEST y BACKTEST)")
    print(f"  Umbral ML: {umbral:.2f}")
    print(f"{'='*100}")
    print(
        f"  {'E':<3} {'S':<3} {'Seg':<10} "
        f"{'Ops_O':<7} {'WR_O%':<7} {'PF_O':<6} {'Ret_O':<7} | "
        f"{'Ops_ML':<7} {'Rech%':<7} {'WR_ML%':<8} {'PF_ML':<7} {'Ret_ML':<7} | "
        f"{'dWR%':<7} {'dPF':<7} {'dRet'}"
    )
    print(f"  {'-'*3} {'-'*3} {'-'*10} "
          f"{'-'*6} {'-'*6} {'-'*5} {'-'*6}   "
          f"{'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6}   "
          f"{'-'*6} {'-'*6} {'-'*5}")

    prev_seg = None
    for _, r in df.iterrows():
        if r["seg"] != prev_seg:
            if prev_seg is not None:
                print()
            prev_seg = r["seg"]
        pct_r = f"{r['pct_rechazo']*100:.0f}%"
        print(
            f"  {r['ee']:<3} {r['es']:<3} {r['seg']:<10} "
            f"{int(r['ops_o']):<7} {r['wr_o']:<7} {r['pf_o']:<6} {r['ret_o']:<7} | "
            f"{int(r['ops_ml']):<7} {pct_r:<7} {r['wr_ml']:<8} {r['pf_ml']:<7} {r['ret_ml']:<7} | "
            f"{_signo(r['d_wr']):<7} {_signo(r['d_pf']):<7} {_signo(r['d_ret'])}"
        )


def imprimir_resumen_por_sector(umbral: float):
    """
    Tabla 2: Mejor estrategia (E3xS2) desglosada por sector.
    Permite ver que sectores se benefician mas del filtro ML.
    """
    sql = """
        SELECT
            scope               AS sector,
            segmento            AS seg,
            ops_original        AS ops_o,
            ROUND(win_rate_orig      * 100, 1)  AS wr_o,
            ROUND(profit_factor_orig,        2) AS pf_o,
            ops_ml,
            ROUND(pct_rechazo        * 100, 1)  AS pct_r,
            ROUND(win_rate_ml        * 100, 1)  AS wr_ml,
            ROUND(profit_factor_ml,          2) AS pf_ml,
            ROUND(win_rate_rechazadas* 100, 1)  AS wr_rech,
            ROUND(profit_factor_rechazadas,  2) AS pf_rech,
            ROUND(delta_win_rate     * 100, 2)  AS d_wr,
            ROUND(delta_profit_factor,       2) AS d_pf
        FROM resultados_ml_filter
        WHERE estrategia_entrada = 'E3'
          AND estrategia_salida  = 'S2'
          AND segmento IN ('TEST', 'BACKTEST')
          AND umbral_ml = :umbral
        ORDER BY
            scope,
            CASE segmento WHEN 'TEST' THEN 1 ELSE 2 END
    """
    df = query_df(sql, params={"umbral": umbral})
    if df.empty:
        return

    print(f"\n{'='*100}")
    print("  TABLA 2 — ESTRATEGIA E3xS2 por Sector  (mejor estrategia del backtesting)")
    print(f"  Umbral ML: {umbral:.2f}")
    print(f"{'='*100}")
    print(
        f"  {'Sector':<28} {'Seg':<10} "
        f"{'Ops_O':<7} {'WR_O%':<7} {'PF_O':<6} | "
        f"{'Ops_ML':<7} {'Rech%':<7} {'WR_ML%':<8} {'PF_ML':<7} | "
        f"{'WR_Rech%':<10} {'PF_Rech':<9} | "
        f"{'dWR%':<7} {'dPF'}"
    )
    print(f"  {'-'*28} {'-'*10} "
          f"{'-'*6} {'-'*6} {'-'*5}   "
          f"{'-'*6} {'-'*6} {'-'*7} {'-'*6}   "
          f"{'-'*9} {'-'*8}   "
          f"{'-'*6} {'-'*5}")

    prev_sector = None
    for _, r in df.iterrows():
        if r["sector"] != prev_sector:
            if prev_sector is not None:
                print()
            prev_sector = r["sector"]
        print(
            f"  {str(r['sector']):<28} {r['seg']:<10} "
            f"{int(r['ops_o']):<7} {r['wr_o']:<7} {r['pf_o']:<6} | "
            f"{int(r['ops_ml']):<7} {r['pct_r']:<7} {r['wr_ml']:<8} {r['pf_ml']:<7} | "
            f"{r['wr_rech']:<10} {r['pf_rech']:<9} | "
            f"{_signo(r['d_wr']):<7} {_signo(r['d_pf'])}"
        )


def imprimir_top_mejoras(umbral: float, top_n: int = 10):
    """
    Tabla 3: Top N combinaciones E x S que mas mejoran con el filtro ML.
    Solo segmento TEST (out-of-sample). Ordenado por delta_profit_factor.
    """
    sql = """
        SELECT
            estrategia_entrada  AS ee,
            estrategia_salida   AS es,
            scope,
            ops_original        AS ops_o,
            ROUND(profit_factor_orig,        2) AS pf_o,
            ROUND(win_rate_orig      * 100, 1)  AS wr_o,
            ROUND(profit_factor_ml,          2) AS pf_ml,
            ROUND(win_rate_ml        * 100, 1)  AS wr_ml,
            ROUND(pct_rechazo        * 100, 1)  AS pct_r,
            ROUND(delta_profit_factor,       2) AS d_pf,
            ROUND(delta_win_rate     * 100, 2)  AS d_wr
        FROM resultados_ml_filter
        WHERE segmento  = 'TEST'
          AND umbral_ml = :umbral
          AND ops_original >= 10
        ORDER BY delta_profit_factor DESC
        LIMIT :top_n
    """
    df = query_df(sql, params={"umbral": umbral, "top_n": top_n})
    if df.empty:
        return

    print(f"\n{'='*90}")
    print(f"  TABLA 3 — TOP {top_n} Combinaciones con Mayor Mejora ML  (segmento TEST)")
    print(f"  Umbral ML: {umbral:.2f}")
    print(f"{'='*90}")
    print(
        f"  {'#':<3} {'E':<3} {'S':<3} {'Scope':<28} "
        f"{'Ops_O':<7} {'PF_Orig':<9} {'WR_O%':<8} "
        f"{'PF_ML':<7} {'WR_ML%':<8} {'Rech%':<7} "
        f"{'dPF':<7} {'dWR%'}"
    )
    print(f"  {'-'*3} {'-'*3} {'-'*3} {'-'*28} "
          f"{'-'*6} {'-'*8} {'-'*7} "
          f"{'-'*6} {'-'*7} {'-'*6} "
          f"{'-'*6} {'-'*6}")
    for i, (_, r) in enumerate(df.iterrows(), 1):
        print(
            f"  {i:<3} {r['ee']:<3} {r['es']:<3} {str(r['scope']):<28} "
            f"{int(r['ops_o']):<7} {r['pf_o']:<9} {r['wr_o']:<8} "
            f"{r['pf_ml']:<7} {r['wr_ml']:<8} {r['pct_r']:<7} "
            f"{_signo(r['d_pf']):<7} {_signo(r['d_wr'])}"
        )


def imprimir_calidad_filtro(umbral: float):
    """
    Tabla 4: Calidad del filtro ML.
    Verifica que las ops rechazadas tienen peores metricas que las aprobadas.
    Solo estrategia E3xS2, segmentos TEST y BACKTEST.
    """
    sql = """
        SELECT
            scope,
            segmento,
            ops_original                         AS ops_total,
            ops_ml                               AS ops_aprobadas,
            ops_rechazadas,
            ROUND(win_rate_orig        * 100, 1) AS wr_orig,
            ROUND(win_rate_ml          * 100, 1) AS wr_aprobadas,
            ROUND(win_rate_rechazadas  * 100, 1) AS wr_rechazadas,
            ROUND(profit_factor_orig,          2) AS pf_orig,
            ROUND(profit_factor_ml,            2) AS pf_aprobadas,
            ROUND(profit_factor_rechazadas,    2) AS pf_rechazadas
        FROM resultados_ml_filter
        WHERE estrategia_entrada = 'E3'
          AND estrategia_salida  = 'S2'
          AND segmento IN ('TEST', 'BACKTEST')
          AND umbral_ml = :umbral
        ORDER BY scope, CASE segmento WHEN 'TEST' THEN 1 ELSE 2 END
    """
    df = query_df(sql, params={"umbral": umbral})
    if df.empty:
        return

    print(f"\n{'='*95}")
    print("  TABLA 4 — Calidad del Filtro ML: Aprobadas vs Rechazadas  (E3xS2)")
    print(f"  Umbral ML: {umbral:.2f}  |  Objetivo: WR_Rechazadas < WR_Aprobadas")
    print(f"{'='*95}")
    print(
        f"  {'Sector':<28} {'Seg':<10} "
        f"{'Total':<7} {'Aprobadas':<11} {'Rechazadas':<12} "
        f"{'WR_Orig%':<10} {'WR_Apro%':<10} {'WR_Rech%':<10} "
        f"{'PF_Orig':<9} {'PF_Apro':<9} {'PF_Rech'}"
    )
    print(f"  {'-'*28} {'-'*10} "
          f"{'-'*6} {'-'*10} {'-'*11} "
          f"{'-'*9} {'-'*9} {'-'*9} "
          f"{'-'*8} {'-'*8} {'-'*7}")

    prev_scope = None
    for _, r in df.iterrows():
        if r["scope"] != prev_scope:
            if prev_scope is not None:
                print()
            prev_scope = r["scope"]
        # Indicador visual de calidad del filtro
        filtro_ok = "OK" if (r["wr_rechazadas"] <= r["wr_aprobadas"] or
                              r["pf_rechazadas"] <= r["pf_aprobadas"]) else "??"
        print(
            f"  {str(r['scope']):<28} {r['segmento']:<10} "
            f"{int(r['ops_total']):<7} {int(r['ops_aprobadas']):<11} {int(r['ops_rechazadas']):<12} "
            f"{r['wr_orig']:<10} {r['wr_aprobadas']:<10} {r['wr_rechazadas']:<10} "
            f"{r['pf_orig']:<9} {r['pf_aprobadas']:<9} {r['pf_rechazadas']}  [{filtro_ok}]"
        )


def imprimir_comparacion_e3s2_completa(umbral: float):
    """
    Tabla 5: Vista completa de E3xS2 en GLOBAL — TRAIN + TEST + BACKTEST.
    Muestra la evolucion de las metricas a lo largo de los tres segmentos.
    """
    sql = """
        SELECT
            segmento,
            ops_original                         AS ops_o,
            ROUND(win_rate_orig        * 100, 1) AS wr_o,
            ROUND(profit_factor_orig,          2) AS pf_o,
            ROUND(ret_promedio_orig,           2) AS ret_o,
            ROUND(max_dd_orig,                 2) AS dd_o,
            ops_ml,
            ROUND(pct_rechazo          * 100, 1) AS pct_r,
            ROUND(win_rate_ml          * 100, 1) AS wr_ml,
            ROUND(profit_factor_ml,            2) AS pf_ml,
            ROUND(ret_promedio_ml,             2) AS ret_ml,
            ROUND(max_dd_ml,                   2) AS dd_ml,
            ROUND(delta_win_rate       * 100, 2) AS d_wr,
            ROUND(delta_profit_factor,         2) AS d_pf,
            ROUND(delta_ret_promedio,          2) AS d_ret
        FROM resultados_ml_filter
        WHERE estrategia_entrada = 'E3'
          AND estrategia_salida  = 'S2'
          AND scope      = 'GLOBAL'
          AND umbral_ml  = :umbral
        ORDER BY
            CASE segmento WHEN 'TRAIN' THEN 1 WHEN 'TEST' THEN 2 ELSE 3 END
    """
    df = query_df(sql, params={"umbral": umbral})
    if df.empty:
        return

    print(f"\n{'='*105}")
    print("  TABLA 5 — E3xS2 GLOBAL: Vista Completa (TRAIN + TEST + BACKTEST)")
    print(f"  Umbral ML: {umbral:.2f}  |  Metrica clave: Profit Factor")
    print(f"{'='*105}")
    print(
        f"  {'Segmento':<10} "
        f"{'Ops_O':<7} {'WR_O%':<7} {'PF_O':<6} {'Ret_O':<7} {'DD_O':<7} | "
        f"{'Ops_ML':<7} {'Rech%':<7} {'WR_ML%':<8} {'PF_ML':<7} {'Ret_ML':<7} {'DD_ML':<7} | "
        f"{'dWR%':<7} {'dPF':<7} {'dRet'}"
    )
    print(f"  {'-'*10} "
          f"{'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*6}   "
          f"{'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6}   "
          f"{'-'*6} {'-'*6} {'-'*5}")

    for _, r in df.iterrows():
        print(
            f"  {r['segmento']:<10} "
            f"{int(r['ops_o']):<7} {r['wr_o']:<7} {r['pf_o']:<6} "
            f"{r['ret_o']:<7} {r['dd_o']:<7} | "
            f"{int(r['ops_ml']):<7} {r['pct_r']:<7} {r['wr_ml']:<8} "
            f"{r['pf_ml']:<7} {r['ret_ml']:<7} {r['dd_ml']:<7} | "
            f"{_signo(r['d_wr']):<7} {_signo(r['d_pf']):<7} {_signo(r['d_ret'])}"
        )


def imprimir_conclusion(umbral: float):
    """
    Conclusion: tabla resumen de ganadores por segmento.
    """
    sql = """
        SELECT
            segmento,
            COUNT(*) FILTER (WHERE delta_profit_factor > 0) AS mejoran_pf,
            COUNT(*) FILTER (WHERE delta_profit_factor < 0) AS empeoran_pf,
            COUNT(*) FILTER (WHERE delta_win_rate > 0)      AS mejoran_wr,
            ROUND(AVG(delta_profit_factor), 3)               AS avg_d_pf,
            ROUND(AVG(delta_win_rate) * 100, 2)              AS avg_d_wr_pct,
            ROUND(AVG(pct_rechazo) * 100, 1)                 AS avg_rechazo_pct
        FROM resultados_ml_filter
        WHERE scope    = 'GLOBAL'
          AND umbral_ml = :umbral
        GROUP BY segmento
        ORDER BY CASE segmento WHEN 'TRAIN' THEN 1 WHEN 'TEST' THEN 2 ELSE 3 END
    """
    df = query_df(sql, params={"umbral": umbral})
    if df.empty:
        return

    total_combis = 16  # 4 entradas x 4 salidas
    print(f"\n{'='*75}")
    print("  CONCLUSION — Impacto Global del Filtro ML sobre las 16 Combis E x S")
    print(f"  Umbral ML: {umbral:.2f}  |  Total combis por segmento: {total_combis}")
    print(f"{'='*75}")
    print(
        f"  {'Segmento':<12} {'Mejoran PF':<12} {'Empeoran PF':<13} "
        f"{'Mejoran WR':<12} {'Avg dPF':<10} {'Avg dWR%':<10} {'Avg Rechazo%'}"
    )
    print(f"  {'-'*12} {'-'*11} {'-'*12} {'-'*11} {'-'*9} {'-'*9} {'-'*11}")
    for _, r in df.iterrows():
        print(
            f"  {r['segmento']:<12} {int(r['mejoran_pf']):<12} "
            f"{int(r['empeoran_pf']):<13} {int(r['mejoran_wr']):<12} "
            f"{_signo(r['avg_d_pf']):<10} {_signo(r['avg_d_wr_pct']):<10} "
            f"{r['avg_rechazo_pct']:.1f}%"
        )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fase 6 - Evaluacion filtro ML sobre backtesting"
    )
    parser.add_argument(
        "--umbral", type=float, default=0.50,
        help="Umbral de probabilidad ML para aprobar operacion (default: 0.50)"
    )
    parser.add_argument(
        "--estrategia", nargs=2, metavar=("ENTRADA", "SALIDA"),
        help="Mostrar detalle para una estrategia especifica (ej: E3 S2)"
    )
    args = parser.parse_args()

    inicio = datetime.now()
    umbral = args.umbral

    print("\n" + "=" * 75)
    print("  FASE 6 — FILTRO ML SOBRE BACKTESTING")
    print("=" * 75)
    print(f"  Umbral ML             : {umbral:.2f}")
    print(f"  Modelos desplegados   : uno por sector (Automotive = Global)")
    print(f"  Estrategias           : 4 entradas x 4 salidas = 16 combinaciones")
    print(f"  Segmentos             : TRAIN | TEST | BACKTEST")
    print(f"  Scopes                : GLOBAL + 4 sectores = 5")
    print(f"  Inicio                : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    # ── Pipeline principal ────────────────────────────────────
    try:
        resultado = ejecutar_evaluacion_ml(umbral=umbral)
        log_ejecucion(
            "ML_FILTER",
            f"umbral={umbral} | {len(resultado)} registros calculados",
            "OK"
        )
    except Exception as e:
        log_ejecucion("ML_FILTER", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Tablas de resultados ──────────────────────────────────
    imprimir_comparacion_e3s2_completa(umbral)
    imprimir_resumen_global(umbral)
    imprimir_resumen_por_sector(umbral)
    imprimir_calidad_filtro(umbral)
    imprimir_top_mejoras(umbral)
    imprimir_conclusion(umbral)

    # ── Footer ────────────────────────────────────────────────
    fin      = datetime.now()
    duracion = (fin - inicio).seconds

    print(f"\n{'='*75}")
    print("  EVALUACION COMPLETADA")
    print(f"{'='*75}")
    print(f"  Resultados en DB  : resultados_ml_filter")
    print(f"  Registros totales : {len(resultado):,}")
    print(f"  Tiempo total      : {duracion}s")
    print(f"  Fin               : {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
