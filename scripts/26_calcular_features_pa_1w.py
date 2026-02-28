"""
26_calcular_features_pa_1w.py
Calcula 32 features de precio/accion sobre barras semanales (1W).

Lee precios_semanales + indicadores_tecnicos_1w de Railway,
calcula las mismas 32 features que el 1D y persiste en features_precio_accion_1w.

Uso:
    python scripts/26_calcular_features_pa_1w.py
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    from src.indicators.precio_accion_1w import procesar_features_precio_accion_1w
    from src.data.database import query_df

    log("=" * 60)
    log("  FEATURES PRECIO/ACCION 1W")
    log("=" * 60)
    log("  32 features | fuente: precios_semanales + indicadores_tecnicos_1w")
    log("  destino: features_precio_accion_1w")
    print()

    t0 = time.time()
    df_feat = procesar_features_precio_accion_1w()
    elapsed = time.time() - t0

    print()
    log("=" * 60)
    log("  RESUMEN")
    log("=" * 60)
    log(f"  Filas calculadas : {len(df_feat):,}")
    log(f"  Tickers          : {df_feat['ticker'].nunique()}")
    log(f"  Tiempo total     : {elapsed:.0f}s")

    # Verificacion en DB
    log("\n  Verificacion en Railway:")
    df_check = query_df("""
        SELECT ticker,
               COUNT(*)   AS n,
               MIN(fecha) AS primera,
               MAX(fecha) AS ultima
        FROM   features_precio_accion_1w
        GROUP  BY ticker
        ORDER  BY ticker
    """)
    if not df_check.empty:
        log(f"  Total tickers : {len(df_check)}")
        log(f"  Total filas   : {df_check['n'].sum():,}")
        log(f"  Rango global  : {df_check['primera'].min()} -> {df_check['ultima'].max()}")

        # Muestra sample de patrones detectados
        df_patrones = query_df("""
            SELECT
                SUM(patron_hammer)         AS hammers,
                SUM(patron_engulfing_bull) AS engulf_bull,
                SUM(patron_shooting_star)  AS shooting_star,
                SUM(patron_engulfing_bear) AS engulf_bear,
                SUM(vol_spike)             AS vol_spikes
            FROM features_precio_accion_1w
        """)
        if not df_patrones.empty:
            r = df_patrones.iloc[0]
            log(f"\n  Patrones detectados (total historico):")
            log(f"    Hammer          : {int(r['hammers']):,}")
            log(f"    Engulfing Bull  : {int(r['engulf_bull']):,}")
            log(f"    Shooting Star   : {int(r['shooting_star']):,}")
            log(f"    Engulfing Bear  : {int(r['engulf_bear']):,}")
            log(f"    Vol Spike       : {int(r['vol_spikes']):,}")
    else:
        log("  WARN: features_precio_accion_1w vacia.")

    print()
    log("  Proximo paso:")
    log("    Etapa 5: python scripts/27_calcular_features_ms_1w.py")


if __name__ == "__main__":
    main()
