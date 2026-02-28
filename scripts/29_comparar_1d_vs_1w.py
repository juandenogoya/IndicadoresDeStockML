"""
29_comparar_1d_vs_1w.py
Comparacion PA Challenger 1D vs PA Challenger 1W.

Compara las 16 combinaciones EV x SV sobre barras diarias (1D)
contra las mismas 16 sobre barras semanales (1W).

Tablas consultadas:
    resultados_bt_pa    (segmento FULL) -> sistema 1D
    resultados_bt_pa_1w (segmento FULL) -> sistema 1W

Secciones:
    1. Cobertura (tickers y operaciones por sistema)
    2. Ranking 1D -- FULL (top 16 combis)
    3. Ranking 1W -- FULL (top 16 combis)
    4. Comparacion directa por combinacion (delta PF por cada EV x SV)
    5. Promedios comparativos
    6. Mejor combi de cada sistema
    7. Verdict + Resumen ejecutivo

Nota sobre comparabilidad:
    - 1D: datos desde 2023-01-01 (barras diarias, ~750 dias)
    - 1W: datos limitados por warmup SMA200 (200 semanas):
        tickers ML (~22): ~122 semanas post-warmup
        tickers BT extra (~102): ~61 semanas post-warmup
    - Menor historico 1W = resultados mas inflados: usar con precaucion.

Uso:
    python scripts/29_comparar_1d_vs_1w.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import query_df


# ─────────────────────────────────────────────────────────────
# Helpers de consulta
# ─────────────────────────────────────────────────────────────

SEG = "FULL"


def get_global(tabla: str, min_ops: int = 1) -> "pd.DataFrame":
    """
    Obtiene resultados globales (ticker IS NULL) de una tabla.
    Usa DISTINCT ON para evitar duplicados cuando el mismo (ee, es) tiene
    multiples filas (caso de resultados_bt_pa con corridas acumuladas).
    """
    sql = f"""
        SELECT DISTINCT ON (estrategia_entrada, estrategia_salida)
            estrategia_entrada  AS ee,
            estrategia_salida   AS es,
            total_operaciones   AS ops,
            ROUND(win_rate * 100, 2)              AS wr_pct,
            ROUND(retorno_promedio_pct, 3)         AS ret_prom,
            ROUND(retorno_total_pct, 2)            AS ret_total,
            ROUND(profit_factor::NUMERIC, 3)       AS pf,
            ROUND(max_drawdown_pct, 2)             AS max_dd,
            ROUND(dias_promedio_posicion, 1)       AS dias_prom
        FROM {tabla}
        WHERE segmento          = :seg
          AND ticker             IS NULL
          AND total_operaciones >= :min_ops
        ORDER BY estrategia_entrada, estrategia_salida, profit_factor DESC
    """
    df = query_df(sql, params={"seg": SEG, "min_ops": min_ops})
    # Reordenar por PF desc para el ranking
    return df.sort_values("pf", ascending=False).reset_index(drop=True)


def get_promedios(tabla: str, min_ops: int = 1):
    """Calcula promedios de todas las combinaciones globales (una por EE x ES)."""
    sql = f"""
        WITH best AS (
            SELECT DISTINCT ON (estrategia_entrada, estrategia_salida)
                total_operaciones, win_rate, retorno_promedio_pct,
                profit_factor, max_drawdown_pct, dias_promedio_posicion
            FROM {tabla}
            WHERE segmento          = :seg
              AND ticker             IS NULL
              AND total_operaciones >= :min_ops
            ORDER BY estrategia_entrada, estrategia_salida, profit_factor DESC
        )
        SELECT
            COUNT(*)                              AS n_combis,
            ROUND(AVG(total_operaciones), 1)      AS avg_ops,
            ROUND(SUM(total_operaciones), 0)      AS total_ops,
            ROUND(AVG(win_rate * 100), 3)         AS avg_wr,
            ROUND(AVG(retorno_promedio_pct), 3)   AS avg_ret,
            ROUND(AVG(profit_factor::NUMERIC), 3) AS avg_pf,
            ROUND(AVG(max_drawdown_pct), 2)       AS avg_dd,
            ROUND(AVG(dias_promedio_posicion), 1) AS avg_dias
        FROM best
    """
    r = query_df(sql, params={"seg": SEG, "min_ops": min_ops})
    return r.iloc[0] if not r.empty else None


def verificar_datos():
    """Verifica que ambas tablas tienen datos globales FULL."""
    r = query_df("""
        SELECT
            (SELECT COUNT(*) FROM resultados_bt_pa
             WHERE segmento='FULL' AND ticker IS NULL) AS n_1d,
            (SELECT COUNT(*) FROM resultados_bt_pa_1w
             WHERE segmento='FULL' AND ticker IS NULL) AS n_1w
    """)
    row = r.iloc[0]
    n_1d = int(row["n_1d"])
    n_1w = int(row["n_1w"])
    print(f"  resultados_bt_pa    (FULL, global): {n_1d:,} filas")
    print(f"  resultados_bt_pa_1w (FULL, global): {n_1w:,} filas")
    if n_1d == 0:
        raise RuntimeError("resultados_bt_pa vacio. Ejecutar script 15.")
    if n_1w == 0:
        raise RuntimeError("resultados_bt_pa_1w vacio. Ejecutar script 28.")
    return n_1d, n_1w


def get_cobertura():
    """Retorna resumen de operaciones y tickers por sistema."""
    sql = """
        SELECT
            '1D' AS sistema,
            COUNT(*) AS total_ops,
            COUNT(DISTINCT ticker) AS tickers,
            MIN(fecha_entrada)     AS primera,
            MAX(fecha_salida)      AS ultima
        FROM operaciones_bt_pa
        UNION ALL
        SELECT
            '1W' AS sistema,
            COUNT(*) AS total_ops,
            COUNT(DISTINCT ticker) AS tickers,
            MIN(fecha_entrada)     AS primera,
            MAX(fecha_salida)      AS ultima
        FROM operaciones_bt_pa_1w
        ORDER BY sistema
    """
    return query_df(sql)


# ─────────────────────────────────────────────────────────────
# Helpers de impresion
# ─────────────────────────────────────────────────────────────

SEP  = "=" * 68
SEP2 = "-" * 68


def log(msg=""):
    print(msg, flush=True)


def _header():
    log(f"  {'EE':<5} {'ES':<5} {'Ops':>5} {'WR%':>7} {'RetProm':>9} {'PF':>8} {'MaxDD':>8} {'Dias':>6}")
    log("  " + SEP2[:54])


def _fila(row):
    log(
        f"  {row['ee']:<5} {row['es']:<5} "
        f"{int(row['ops']):>5} "
        f"{float(row['wr_pct']):>7.1f} "
        f"{float(row['ret_prom']):>9.3f} "
        f"{float(row['pf']):>8.3f} "
        f"{float(row['max_dd']):>8.2f} "
        f"{float(row['dias_prom']):>6.1f}"
    )


def imprimir_ranking(df, titulo: str, n: int = 16):
    log(f"\n  {titulo}")
    _header()
    for _, row in df.head(n).iterrows():
        _fila(row)


# ─────────────────────────────────────────────────────────────
# Comparacion directa por combinacion EV x SV
# ─────────────────────────────────────────────────────────────

def comparar_por_combinacion(df_1d, df_1w):
    """
    Para cada combinacion EV x SV, muestra PF 1D vs PF 1W y el delta.
    Solo combis presentes en ambos sistemas.
    """
    import pandas as pd

    df_1d = df_1d.set_index(["ee", "es"])
    df_1w = df_1w.set_index(["ee", "es"])

    combis = sorted(set(df_1d.index) & set(df_1w.index))
    if not combis:
        log("  [WARN] Sin combinaciones comunes.")
        return

    log(f"\n  {'EE':<5} {'ES':<5} {'Ops 1D':>8} {'PF 1D':>8} {'Ops 1W':>8} {'PF 1W':>8} {'dPF':>8}")
    log("  " + SEP2[:56])

    deltas = []
    for (ee, es) in combis:
        r1d = df_1d.loc[(ee, es)]
        r1w = df_1w.loc[(ee, es)]
        pf_1d = float(r1d["pf"])
        pf_1w = float(r1w["pf"])
        dpf   = pf_1w - pf_1d
        deltas.append(dpf)
        signo = "+" if dpf >= 0 else ""
        log(
            f"  {ee:<5} {es:<5} "
            f"{int(r1d['ops']):>8} "
            f"{pf_1d:>8.3f} "
            f"{int(r1w['ops']):>8} "
            f"{pf_1w:>8.3f} "
            f"{signo}{dpf:>7.3f}"
        )

    if deltas:
        import statistics
        avg_d  = statistics.mean(deltas)
        pos    = sum(1 for d in deltas if d > 0)
        neg    = sum(1 for d in deltas if d < 0)
        signo  = "+" if avg_d >= 0 else ""
        log("  " + SEP2[:56])
        log(f"  {'PROMEDIO delta PF':<24} {signo}{avg_d:+.3f}   |   1W > 1D en {pos}/{len(deltas)} combis")
    return deltas


# ─────────────────────────────────────────────────────────────
# Comparacion de promedios
# ─────────────────────────────────────────────────────────────

def imprimir_promedios(p1d, p1w):
    if p1d is None or p1w is None:
        log("  [WARN] Datos insuficientes para comparar promedios.")
        return

    d_wr  = float(p1w["avg_wr"])  - float(p1d["avg_wr"])
    d_ret = float(p1w["avg_ret"]) - float(p1d["avg_ret"])
    d_pf  = float(p1w["avg_pf"])  - float(p1d["avg_pf"])
    d_dd  = float(p1w["avg_dd"])  - float(p1d["avg_dd"])

    log(f"\n  {'Metrica':<26} {'1D (diario)':>12} {'1W (semanal)':>13} {'Delta':>10}")
    log("  " + SEP2[:64])
    log(f"  {'Combinaciones con datos':<26} {int(p1d['n_combis']):>12} {int(p1w['n_combis']):>13}")
    log(f"  {'Ops totales':<26} {int(p1d['total_ops']):>12,} {int(p1w['total_ops']):>13,}")
    log(f"  {'Ops promedio/combi':<26} {float(p1d['avg_ops']):>12.1f} {float(p1w['avg_ops']):>13.1f}")
    log(f"  {'Win Rate %':<26} {float(p1d['avg_wr']):>12.2f} {float(p1w['avg_wr']):>13.2f} {d_wr:>+10.2f}")
    log(f"  {'Retorno prom %':<26} {float(p1d['avg_ret']):>12.3f} {float(p1w['avg_ret']):>13.3f} {d_ret:>+10.3f}")
    log(f"  {'Profit Factor (avg)':<26} {float(p1d['avg_pf']):>12.3f} {float(p1w['avg_pf']):>13.3f} {d_pf:>+10.3f}")
    log(f"  {'Max Drawdown % (avg)':<26} {float(p1d['avg_dd']):>12.2f} {float(p1w['avg_dd']):>13.2f} {d_dd:>+10.2f}")
    log(f"  {'Duracion prom (barras)':<26} {float(p1d['avg_dias']):>12.1f} {float(p1w['avg_dias']):>13.1f}")
    log(f"  {'  (en 1W: dias = semanas)':<26}")


# ─────────────────────────────────────────────────────────────
# Cobertura por estrategia (operaciones por combi)
# ─────────────────────────────────────────────────────────────

def ops_por_estrategia_entrada():
    """Muestra cuantas operaciones genera cada estrategia de entrada en 1D vs 1W."""
    sql = """
        SELECT
            t.ee,
            t.ops_1d,
            t.ops_1w,
            ROUND(t.ops_1w::NUMERIC / NULLIF(t.ops_1d, 0) * 100, 1) AS ratio_pct
        FROM (
            SELECT
                d.estrategia_entrada AS ee,
                SUM(d.total_operaciones) FILTER (WHERE d.tabla = '1D') AS ops_1d,
                SUM(d.total_operaciones) FILTER (WHERE d.tabla = '1W') AS ops_1w
            FROM (
                SELECT estrategia_entrada, total_operaciones, '1D' AS tabla
                FROM resultados_bt_pa WHERE segmento='FULL' AND ticker IS NULL
                UNION ALL
                SELECT estrategia_entrada, total_operaciones, '1W' AS tabla
                FROM resultados_bt_pa_1w WHERE segmento='FULL' AND ticker IS NULL
            ) d
            GROUP BY d.estrategia_entrada
        ) t
        ORDER BY t.ee
    """
    return query_df(sql)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    inicio = datetime.now()

    log()
    log(SEP)
    log("  COMPARACION: PA CHALLENGER 1D vs 1W")
    log(SEP)
    log(f"  Sistema 1D : EV1-EV4 x SV1-SV4 | barras diarias | FULL")
    log(f"  Sistema 1W : EV1-EV4 x SV1-SV4 | barras semanales | FULL")
    log(f"  Inicio     : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    log(SEP)

    # ── Verificacion ──────────────────────────────────────────
    log("\n  Verificando datos...")
    try:
        verificar_datos()
    except RuntimeError as e:
        log(f"\n[ERROR] {e}")
        sys.exit(1)

    # ── Seccion 1: Cobertura ──────────────────────────────────
    log()
    log(SEP)
    log("  1. COBERTURA DE DATOS")
    log(SEP)
    df_cob = get_cobertura()
    if not df_cob.empty:
        log(f"\n  {'Sistema':<8} {'Ops':>6} {'Tickers':>8} {'Primera':>12} {'Ultima':>12}")
        log("  " + SEP2[:50])
        for _, row in df_cob.iterrows():
            log(
                f"  {row['sistema']:<8} {int(row['total_ops']):>6,} "
                f"{int(row['tickers']):>8} "
                f"{str(row['primera']):>12} "
                f"{str(row['ultima']):>12}"
            )

    df_ee = ops_por_estrategia_entrada()
    if not df_ee.empty:
        log(f"\n  Ops por estrategia de entrada:")
        log(f"  {'EE':<6} {'Ops 1D':>8} {'Ops 1W':>8} {'1W/1D%':>8}")
        log("  " + SEP2[:34])
        for _, row in df_ee.iterrows():
            ops_1d  = int(row['ops_1d'])  if row['ops_1d']  is not None else 0
            ops_1w  = int(row['ops_1w'])  if row['ops_1w']  is not None else 0
            ratio   = float(row['ratio_pct']) if row['ratio_pct'] is not None else 0.0
            log(f"  {row['ee']:<6} {ops_1d:>8,} {ops_1w:>8,} {ratio:>7.1f}%")

    log("\n  NOTA: 1W tiene menos ops por barra = mayor selectividad.")
    log("  La duracion en 1W se mide en SEMANAS (no dias).")
    log("  El warmup de SMA200 (200 semanas) limita el historico 1W:")
    log("    tickers ML (~22): ~122 semanas  |  tickers BT extra: ~61 semanas")

    # ── Seccion 2: Ranking 1D ─────────────────────────────────
    log()
    log(SEP)
    log("  2. RANKING PA 1D (FULL) -- Top 16 combis por PF")
    log(SEP)
    df_1d = get_global("resultados_bt_pa", min_ops=1)
    imprimir_ranking(df_1d, "EE x ES ordenado por Profit Factor (1D):", n=16)

    # ── Seccion 3: Ranking 1W ─────────────────────────────────
    log()
    log(SEP)
    log("  3. RANKING PA 1W (FULL) -- Top 16 combis por PF")
    log(SEP)
    df_1w = get_global("resultados_bt_pa_1w", min_ops=1)
    imprimir_ranking(df_1w, "EE x ES ordenado por Profit Factor (1W):", n=16)

    # ── Seccion 4: Delta PF por combinacion ───────────────────
    log()
    log(SEP)
    log("  4. COMPARACION DIRECTA POR COMBINACION (delta PF = 1W - 1D)")
    log(SEP)
    deltas = comparar_por_combinacion(df_1d.copy(), df_1w.copy())

    # ── Seccion 5: Promedios comparativos ─────────────────────
    log()
    log(SEP)
    log("  5. COMPARACION DE PROMEDIOS (todas las combis globales)")
    log(SEP)
    p1d = get_promedios("resultados_bt_pa")
    p1w = get_promedios("resultados_bt_pa_1w")
    imprimir_promedios(p1d, p1w)

    # ── Seccion 6: Mejor combi de cada sistema ────────────────
    log()
    log(SEP)
    log("  6. MEJOR COMBINACION POR SISTEMA")
    log(SEP)

    # Top 1 por PF (con minimo de ops para estadistica razonable)
    df_1d_min5 = get_global("resultados_bt_pa",    min_ops=5)
    df_1w_min5 = get_global("resultados_bt_pa_1w", min_ops=5)

    log(f"\n  {'Sistema':<10} {'EE':<5} {'ES':<5} {'Ops':>6} {'WR%':>7} {'RetProm':>9} {'PF':>9} {'MaxDD':>8}")
    log("  " + SEP2[:62])

    if not df_1d_min5.empty:
        top1d = df_1d_min5.iloc[0]
        log(
            f"  {'1D':<10} {top1d['ee']:<5} {top1d['es']:<5} "
            f"{int(top1d['ops']):>6} "
            f"{float(top1d['wr_pct']):>7.1f} "
            f"{float(top1d['ret_prom']):>9.3f} "
            f"{float(top1d['pf']):>9.3f} "
            f"{float(top1d['max_dd']):>8.2f}"
        )
    else:
        log("  1D         Sin datos con min 5 ops.")

    if not df_1w_min5.empty:
        top1w = df_1w_min5.iloc[0]
        log(
            f"  {'1W':<10} {top1w['ee']:<5} {top1w['es']:<5} "
            f"{int(top1w['ops']):>6} "
            f"{float(top1w['wr_pct']):>7.1f} "
            f"{float(top1w['ret_prom']):>9.3f} "
            f"{float(top1w['pf']):>9.3f} "
            f"{float(top1w['max_dd']):>8.2f}"
        )
    else:
        log("  1W         Sin datos con min 5 ops.")

    # ── Seccion 7: Verdict ────────────────────────────────────
    log()
    log(SEP)
    log("  7. VERDICT FINAL")
    log(SEP)

    if p1d is not None and p1w is not None and deltas:
        avg_delta_pf = sum(deltas) / len(deltas)
        combis_1w_mejor = sum(1 for d in deltas if d > 0)
        total_combis    = len(deltas)

        pf_1d_avg = float(p1d["avg_pf"])
        pf_1w_avg = float(p1w["avg_pf"])
        wr_delta  = float(p1w["avg_wr"]) - float(p1d["avg_wr"])
        ret_delta = float(p1w["avg_ret"]) - float(p1d["avg_ret"])
        dd_delta  = float(p1w["avg_dd"]) - float(p1d["avg_dd"])

        # Verdict basado en avg delta PF Y proporcion de combis ganadoras
        pct_mejor = combis_1w_mejor / total_combis if total_combis > 0 else 0

        if avg_delta_pf > 0.10 and pct_mejor >= 0.5:
            verdict = "POSITIVO"
            desc    = "1W supera a 1D en Profit Factor promedio. Senales semanales mas eficientes."
        elif avg_delta_pf > -0.10:
            verdict = "NEUTRAL"
            desc    = "1W empata con 1D en Profit Factor. Similar eficiencia a diferente escala temporal."
        else:
            verdict = "NEGATIVO"
            desc    = "1D supera a 1W en Profit Factor promedio con mayor muestra de datos."

        log(f"\n  METRICAS CLAVE:")
        log(f"    PF promedio 1D          : {pf_1d_avg:.3f}")
        log(f"    PF promedio 1W          : {pf_1w_avg:.3f}")
        log(f"    delta PF promedio       : {avg_delta_pf:+.3f}")
        log(f"    Combis donde 1W > 1D   : {combis_1w_mejor}/{total_combis} ({pct_mejor*100:.0f}%)")
        log(f"    delta WR%              : {wr_delta:+.2f}%")
        log(f"    delta RetProm%         : {ret_delta:+.3f}%")
        log(f"    delta MaxDD%           : {dd_delta:+.2f}%")
        log(f"    Ops 1D                 : {int(p1d['total_ops']):,}")
        log(f"    Ops 1W                 : {int(p1w['total_ops']):,}")

        log(f"\n  DICTAMEN: {verdict}")
        log(f"  {desc}")

        log(f"\n  ADVERTENCIAS SOBRE LA COMPARACION:")
        log(f"    - 1W tiene historico limitado por warmup SMA200 (200 semanas):")
        log(f"        tickers ML: ~122 sem post-warmup  (desde ~2023-10-27)")
        log(f"        BT extra:   ~61 sem post-warmup   (desde ~2024-12-27)")
        log(f"    - Historico corto infla metricas 1W (menos ops = mas volatilidad estadistica)")
        log(f"    - No hay out-of-sample real en 1W hasta acumular mas semanas")
        log(f"    - EV4/SV4 en 1W (PF=100, 0 perdidas en 20 ops) es OVER-OPTIMISTIC")
        log(f"    - La combinacion mas confiable en 1W es EV1/SV4 (115 ops, PF=3.48)")

        log(f"\n  RESUMEN EJECUTIVO:")
        log(f"    El timeframe 1W genera senales mucho mas selectivas ({int(p1w['total_ops']):,} ops).")
        log(f"    vs las {int(p1d['total_ops']):,} ops en 1D -- menor frecuencia, mayor impacto por trade.")
        log(f"    La duracion promedio de una operacion en 1W es {float(p1w['avg_dias']):.1f} SEMANAS")
        log(f"    (~{float(p1w['avg_dias'])*7:.0f} dias) vs {float(p1d['avg_dias']):.1f} dias en 1D.")
        log(f"    Para uso en produccion: 1W como FILTRO de contexto de largo plazo")
        log(f"    combinado con senales 1D (multi-timeframe) -- Etapa 8 futura.")

    else:
        log("\n  [WARN] Datos insuficientes para calcular verdict.")

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    log()
    log(SEP)
    log(f"  Comparacion completada en {duracion}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    log(SEP)
    log()
    log("  Proximos pasos:")
    log("    Etapa 8 (futura): ML V4 combinando features 1D + 1W")
    log("    Integracion MTF : usar estructura_10 de 1W como filtro en cron_diario")


if __name__ == "__main__":
    main()
