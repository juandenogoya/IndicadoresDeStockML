"""
09_calcular_precio_accion.py
Runner de la Fase 7a — Calculo de features de precio/accion y volumen.

Calcula 32 features desde precios_diarios + indicadores_tecnicos
y las persiste en features_precio_accion.

Uso:
    python scripts/09_calcular_precio_accion.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.indicators.precio_accion import procesar_features_precio_accion, FEATURE_COLS_PA
from src.data.database import query_df, ejecutar_sql


def log_ejecucion(accion: str, detalle: str, estado: str):
    try:
        ejecutar_sql(
            "INSERT INTO log_ejecuciones (script, accion, detalle, estado) "
            "VALUES (%s,%s,%s,%s)",
            ("09_precio_accion", accion, detalle, estado)
        )
    except Exception:
        pass


def imprimir_resumen():
    """Muestra estadisticas de la tabla generada."""
    stats = query_df("""
        SELECT
            COUNT(*)                      AS total_filas,
            COUNT(DISTINCT ticker)        AS tickers,
            MIN(fecha)                    AS fecha_min,
            MAX(fecha)                    AS fecha_max,
            ROUND(AVG(pos_rango_20d), 3)  AS pos_rango_avg,
            ROUND(AVG(chaikin_mf_20), 3)  AS chaikin_avg,
            SUM(patron_doji)              AS n_doji,
            SUM(patron_hammer)            AS n_hammer,
            SUM(patron_engulfing_bull)    AS n_eng_bull,
            SUM(vol_spike)                AS n_vol_spike
        FROM features_precio_accion
    """)
    r = stats.iloc[0]
    print(f"\n  Resumen features_precio_accion:")
    print(f"    Total filas          : {int(r['total_filas']):,}")
    print(f"    Tickers              : {int(r['tickers'])}")
    print(f"    Rango fechas         : {r['fecha_min']} / {r['fecha_max']}")
    print(f"    pos_rango_20d avg    : {r['pos_rango_avg']:.3f}  (esperado ~0.50)")
    print(f"    chaikin_mf_20 avg    : {r['chaikin_avg']:.3f}  (esperado ~0)")
    print(f"    Dojis                : {int(r['n_doji']):,}")
    print(f"    Hammers              : {int(r['n_hammer']):,}")
    print(f"    Engulfing Bull       : {int(r['n_eng_bull']):,}")
    print(f"    Vol spikes           : {int(r['n_vol_spike']):,}")

    # Distribucion de velas alcistas/bajistas
    dist = query_df("""
        SELECT
            sector,
            ROUND(AVG(es_alcista) * 100, 1)      AS pct_alcistas,
            ROUND(AVG(body_ratio) * 100, 1)       AS body_ratio_avg,
            ROUND(AVG(ABS(body_pct)), 2)          AS body_pct_abs_avg
        FROM features_precio_accion fpa
        JOIN activos a ON fpa.ticker = a.ticker
        GROUP BY sector
        ORDER BY sector
    """)
    print(f"\n  Por sector (% velas alcistas / body ratio / mov intradía):")
    for _, row in dist.iterrows():
        print(f"    {str(row['sector']):<28} {row['pct_alcistas']}% alcistas  "
              f"body_ratio={row['body_ratio_avg']}%  "
              f"mov_avg={row['body_pct_abs_avg']}%")


def main():
    inicio = datetime.now()

    print("\n" + "=" * 65)
    print("  FASE 7a — FEATURES PRECIO/ACCION Y VOLUMEN")
    print("=" * 65)
    print(f"  Features a calcular  : {len(FEATURE_COLS_PA)} (modelo) + 2 (tabla)")
    print(f"  Fuente               : precios_diarios JOIN indicadores_tecnicos")
    print(f"  Destino              : features_precio_accion")
    print(f"  Inicio               : {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    try:
        df = procesar_features_precio_accion()
        log_ejecucion("CALCULAR_PA", f"{len(df)} filas calculadas", "OK")
    except Exception as e:
        log_ejecucion("CALCULAR_PA", str(e), "ERROR")
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    imprimir_resumen()

    fin      = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n{'='*65}")
    print(f"  Completado en {duracion}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
