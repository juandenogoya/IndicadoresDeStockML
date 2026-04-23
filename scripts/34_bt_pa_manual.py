"""
34_bt_pa_manual.py
Ejecucion manual del Backtesting PA (Price Action).

Removido del cron diario el 23/4/2026 para evitar timeout en GH Actions.
No es requerido por los bots ni por el scanner — es una herramienta de analisis.

Modos de uso:
    python scripts/34_bt_pa_manual.py              # incremental: solo procesa el dia actual
    python scripts/34_bt_pa_manual.py --full       # full: recorre todo el historico desde FECHA_INICIO_BT
    python scripts/34_bt_pa_manual.py --status     # muestra estado actual de operaciones_bt_pa

Que hace:
    Incremental: evalua posiciones abiertas (FIN_SEGMENTO) y abre nuevas basado en
                 senales del dia mas reciente disponible en DB.
    Full:        re-corre el backtesting completo desde 2023-01-01 (lento, ~10-20 min).

Tablas que actualiza:
    operaciones_bt_pa   — posiciones individuales con entrada/salida/PnL
    resultados_bt_pa    — metricas agregadas por ticker y globales
"""

import sys
import os
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cmd_incremental():
    """Procesa solo el dia actual — modo normal de actualizacion."""
    from src.backtesting.simulator_pa import ejecutar_backtesting_pa_incremental
    from src.backtesting.metrics_pa import calcular_y_guardar_resultados_pa
    from src.data.database import query_df

    log("=" * 55)
    log("  BT-PA MANUAL — Modo incremental (dia actual)")
    log("=" * 55)

    log("\n[1/2] Ejecutando backtesting incremental...")
    stats = ejecutar_backtesting_pa_incremental()
    log(f"  Cerradas  : {stats['cerradas']}")
    log(f"  Actualizadas: {stats['actualizadas']}")
    log(f"  Nuevas    : {stats['nuevas']}")

    log("\n[2/2] Recalculando metricas...")
    df_ops = query_df("SELECT * FROM operaciones_bt_pa ORDER BY fecha_entrada")
    if not df_ops.empty:
        calcular_y_guardar_resultados_pa(df_ops)
        log(f"  Resultados PA actualizados ({len(df_ops):,} operaciones).")
    else:
        log("  Sin operaciones en DB — nada que calcular.")

    log("\n  BT-PA incremental finalizado.")
    log("=" * 55)


def cmd_full():
    """Re-corre el backtesting completo desde FECHA_INICIO_BT. Lento (~10-20 min)."""
    from src.backtesting.simulator_pa import ejecutar_backtesting_pa
    from src.backtesting.metrics_pa import calcular_y_guardar_resultados_pa
    from src.data.database import query_df

    log("=" * 55)
    log("  BT-PA MANUAL — Modo FULL (recorre todo el historico)")
    log("  ADVERTENCIA: puede tardar 10-20 minutos.")
    log("=" * 55)

    log("\n[1/2] Ejecutando backtesting full...")
    n_ops = ejecutar_backtesting_pa()
    log(f"  Operaciones procesadas: {n_ops:,}")

    log("\n[2/2] Recalculando metricas...")
    df_ops = query_df("SELECT * FROM operaciones_bt_pa ORDER BY fecha_entrada")
    if not df_ops.empty:
        calcular_y_guardar_resultados_pa(df_ops)
        log(f"  Resultados PA actualizados ({len(df_ops):,} operaciones).")
    else:
        log("  Sin operaciones en DB — nada que calcular.")

    log("\n  BT-PA full finalizado.")
    log("=" * 55)


def cmd_status():
    """Muestra estado actual de operaciones_bt_pa."""
    from src.data.database import query_df

    log("=" * 55)
    log("  BT-PA — Estado actual en DB")
    log("=" * 55)

    df = query_df("""
        SELECT
            COUNT(*)                                        AS total_ops,
            COUNT(*) FILTER (WHERE estado_pa = 'FIN_SEGMENTO') AS abiertas,
            COUNT(*) FILTER (WHERE estado_pa != 'FIN_SEGMENTO') AS cerradas,
            MIN(fecha_entrada)                             AS primera_entrada,
            MAX(fecha_entrada)                             AS ultima_entrada,
            MAX(fecha_cierre)                              AS ultimo_cierre
        FROM operaciones_bt_pa
    """)

    if df.empty:
        log("  Sin datos en operaciones_bt_pa.")
        return

    r = df.iloc[0]
    print(f"""
  Total operaciones : {int(r['total_ops']):,}
  Abiertas (FIN_SEG): {int(r['abiertas']):,}
  Cerradas          : {int(r['cerradas']):,}
  Primera entrada   : {r['primera_entrada']}
  Ultima entrada    : {r['ultima_entrada']}
  Ultimo cierre     : {r['ultimo_cierre']}
""")

    df_res = query_df("""
        SELECT COUNT(*) AS n_resultados, MAX(updated_at) AS ultima_actualizacion
        FROM resultados_bt_pa
    """)
    if not df_res.empty:
        r2 = df_res.iloc[0]
        print(f"  Resultados calculados : {int(r2['n_resultados']):,}")
        print(f"  Ultima actualizacion  : {r2['ultima_actualizacion']}")


def main():
    parser = argparse.ArgumentParser(
        description="BT-PA manual — backtesting Price Action"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-corre el backtesting completo desde FECHA_INICIO_BT (lento)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Muestra estado actual de operaciones_bt_pa sin modificar nada"
    )
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.full:
        cmd_full()
    else:
        cmd_incremental()


if __name__ == "__main__":
    main()
