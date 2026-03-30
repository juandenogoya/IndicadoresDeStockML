"""
27_calcular_features_ms_1w.py
Calcula 24 features de market structure sobre barras semanales (1W).

Lee precios_semanales de Railway, detecta swing highs/lows,
calcula BOS/CHoCH y persiste en features_market_structure_1w.

Ventanas:
    N=5  (tactico):    pivot en ventana 11 semanas (~2.5 meses)
    N=10 (estrategico): pivot en ventana 21 semanas (~5 meses)

Uso:
    python scripts/27_calcular_features_ms_1w.py
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
    from src.indicators.market_structure_1w import procesar_features_market_structure_1w
    from src.data.database import query_df

    log("=" * 60)
    log("  FEATURES MARKET STRUCTURE 1W")
    log("=" * 60)
    log("  24 features | N=5w (tactico) + N=10w (estrategico)")
    log("  fuente: precios_semanales")
    log("  destino: features_market_structure_1w")
    print()

    t0 = time.time()
    df_feat = procesar_features_market_structure_1w()
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
        FROM   features_market_structure_1w
        GROUP  BY ticker
        ORDER  BY ticker
    """)
    if not df_check.empty:
        log(f"  Total tickers : {len(df_check)}")
        log(f"  Total filas   : {df_check['n'].sum():,}")
        log(f"  Rango global  : {df_check['primera'].min()} -> {df_check['ultima'].max()}")

        # Muestra conteos de senales estructurales
        df_senales = query_df("""
            SELECT
                SUM(CASE WHEN estructura_10 =  1 THEN 1 ELSE 0 END) AS alcistas_10,
                SUM(CASE WHEN estructura_10 = -1 THEN 1 ELSE 0 END) AS bajistas_10,
                SUM(bos_bull_10)   AS bos_bull_10,
                SUM(bos_bear_10)   AS bos_bear_10,
                SUM(choch_bull_10) AS choch_bull_10,
                SUM(choch_bear_10) AS choch_bear_10
            FROM features_market_structure_1w
        """)
        if not df_senales.empty:
            r = df_senales.iloc[0]
            log(f"\n  Senales N=10 (estrategico):")
            log(f"    Estructura alcista (+1) : {int(r['alcistas_10']):,}")
            log(f"    Estructura bajista (-1) : {int(r['bajistas_10']):,}")
            log(f"    BOS Bull                : {int(r['bos_bull_10']):,}")
            log(f"    BOS Bear                : {int(r['bos_bear_10']):,}")
            log(f"    CHoCH Bull              : {int(r['choch_bull_10']):,}")
            log(f"    CHoCH Bear              : {int(r['choch_bear_10']):,}")
    else:
        log("  WARN: features_market_structure_1w vacia.")

    print()
    log("  Proximo paso:")
    log("    Etapa 6: python scripts/28_backtesting_pa_1w.py")


if __name__ == "__main__":
    main()
