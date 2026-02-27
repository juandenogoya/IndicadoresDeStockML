"""
21_migrar_bt_incremental.py
Migracion one-time para habilitar el backtesting PA incremental.

Pasos:
    1. ALTER TABLE operaciones_bt_pa ADD COLUMN stop_loss FLOAT
    2. ALTER TABLE operaciones_bt_pa ADD COLUMN take_profit FLOAT
    3. Full rerun (TRUNCATE + re-simular desde 2023-01-01) para poblar los nuevos campos
    4. Verificacion: muestra conteo de registros y columnas nuevas

Ejecutar UNA SOLA VEZ, antes de activar el modo incremental en cron_diario.py.
Requiere DATABASE_URL apuntando a la base de datos correcta (Railway o local).

Uso:
    python scripts/21_migrar_bt_incremental.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def log(msg: str):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    from src.data.database import get_connection, query_df

    log("=" * 55)
    log("  MIGRACION BT INCREMENTAL")
    log("=" * 55)

    # ── 1. Agregar columnas stop_loss y take_profit ───────────────────────────
    log("\n[1/3] Agregando columnas stop_loss y take_profit a operaciones_bt_pa...")

    with get_connection() as conn:
        with conn.cursor() as cur:
            for col in ("stop_loss", "take_profit"):
                try:
                    cur.execute(f"ALTER TABLE operaciones_bt_pa ADD COLUMN {col} FLOAT")
                    conn.commit()
                    log(f"  {col}: columna agregada OK.")
                except Exception as e:
                    conn.rollback()
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        log(f"  {col}: ya existe, omitido.")
                    else:
                        log(f"  {col}: ERROR inesperado: {e}")
                        raise

    # ── 2. Full rerun baseline ─────────────────────────────────────────────────
    log("\n[2/3] Ejecutando full rerun para poblar stop_loss/take_profit...")
    log("  (TRUNCATE + re-simular desde 2023-01-01 — puede tardar varios minutos)")

    from src.backtesting.simulator_pa import ejecutar_backtesting_pa
    from src.backtesting.metrics_pa import calcular_y_guardar_resultados_pa

    df_ops = ejecutar_backtesting_pa()
    log(f"  Simulacion OK: {len(df_ops):,} operaciones generadas.")

    calcular_y_guardar_resultados_pa(df_ops)
    log("  Metricas guardadas en resultados_bt_pa.")

    # ── 3. Verificacion ────────────────────────────────────────────────────────
    log("\n[3/3] Verificacion final...")

    df_check = query_df("""
        SELECT
            COUNT(*)                                                    AS total,
            COUNT(stop_loss)                                            AS con_stop_loss,
            COUNT(take_profit)                                          AS con_take_profit,
            COUNT(CASE WHEN motivo_salida = 'FIN_SEGMENTO' THEN 1 END) AS abiertas,
            COUNT(CASE WHEN motivo_salida != 'FIN_SEGMENTO' THEN 1 END) AS cerradas
        FROM operaciones_bt_pa
    """)

    log("  Resumen operaciones_bt_pa:")
    for col, val in df_check.iloc[0].items():
        log(f"    {col}: {int(val)}")

    df_fs = query_df("""
        SELECT estrategia_entrada, estrategia_salida, COUNT(*) AS n_abiertas
        FROM operaciones_bt_pa
        WHERE motivo_salida = 'FIN_SEGMENTO'
        GROUP BY estrategia_entrada, estrategia_salida
        ORDER BY estrategia_entrada, estrategia_salida
    """)
    if not df_fs.empty:
        log("\n  Posiciones abiertas (FIN_SEGMENTO) por estrategia:")
        log(df_fs.to_string(index=False))
    else:
        log("  Sin posiciones abiertas (FIN_SEGMENTO).")

    log("\n" + "=" * 55)
    log("  MIGRACION COMPLETADA.")
    log("  Activar modo incremental en cron_diario.py ya esta listo.")
    log("=" * 55)


if __name__ == "__main__":
    main()
