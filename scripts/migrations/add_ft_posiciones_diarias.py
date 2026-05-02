"""
add_ft_posiciones_diarias.py
Migracion: capa de analisis post-operacion para Forward Testing.

Cambios:
  1. Crea ft_posiciones_diarias:
       Snapshot diario de cada posicion abierta + seguimiento post-cierre.
       Estados: abierta | cierre | post_5d | post_10d | post_20d

  2. Agrega 4 columnas a ft_candidatos_diarios:
       precio_apertura, retorno_5d, retorno_10d, retorno_20d

Uso:
    python scripts/migrations/add_ft_posiciones_diarias.py
    python scripts/migrations/add_ft_posiciones_diarias.py --check
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from sqlalchemy import text
from src.data.database import get_engine


def cmd_check(conn):
    """Muestra el estado actual de las tablas afectadas."""
    # ft_posiciones_diarias
    exists = conn.execute(text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'ft_posiciones_diarias'
        )
    """)).scalar()
    print(f"  ft_posiciones_diarias: {'OK (existe)' if exists else 'PENDIENTE (no existe)'}")

    # columnas en ft_candidatos_diarios
    cols = conn.execute(text("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'ft_candidatos_diarios'
        ORDER BY ordinal_position
    """)).fetchall()
    col_names = [r.column_name for r in cols]
    for c in ["precio_apertura", "retorno_5d", "retorno_10d", "retorno_20d"]:
        estado = "OK" if c in col_names else "PENDIENTE"
        print(f"  ft_candidatos_diarios.{c}: {estado}")


def cmd_migrate(conn):
    print("\n  [1/2] Creando ft_posiciones_diarias...")
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ft_posiciones_diarias (
            id                 SERIAL PRIMARY KEY,
            estrategia_id      INT          NOT NULL REFERENCES ft_estrategias(id),
            operacion_id       INT          NOT NULL REFERENCES ft_operaciones(id),
            ticker             VARCHAR(20)  NOT NULL,
            fecha              DATE         NOT NULL,

            -- Estado del ciclo de vida
            -- 'abierta'  : posicion abierta en esta fecha
            -- 'cierre'   : dia en que se cerro la posicion
            -- 'post_5d'  : 5 dias habiles despues del cierre (contrafactual)
            -- 'post_10d' : 10 dias habiles despues del cierre
            -- 'post_20d' : 20 dias habiles despues del cierre
            estado             VARCHAR(10)  NOT NULL,

            -- Precio y retorno
            precio_cierre      NUMERIC(12,4),
            precio_entrada_ref NUMERIC(12,4),
            retorno_pct        NUMERIC(8,4),

            -- Tiempo
            dias_abierta       SMALLINT,

            -- Momentum tecnico (recalculado del dia)
            tech_score         NUMERIC(5,2),
            candle_score_5d    NUMERIC(6,2),

            -- Deteccion de movimiento lateral / acumulacion
            rango_5d_pct       NUMERIC(8,4),  -- amplitud close 5d / precio_entrada * 100
            atr14              NUMERIC(12,4),
            lateral_ratio      NUMERIC(8,4),  -- rango_5d_pct / atr14  (< 0.5 = lateral)

            -- Calidad de volumen
            vol_price_confirm  SMALLINT,
            vol_price_diverge  SMALLINT,
            up_vol_5d          SMALLINT,

            -- Solo para estado='cierre'
            motivo_salida      VARCHAR(40),

            -- Una fila por operacion por fecha (idempotente por re-ejecucion del BAT)
            UNIQUE (operacion_id, fecha)
        )
    """))

    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_ft_pd_eid_fecha
            ON ft_posiciones_diarias(estrategia_id, fecha DESC)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_ft_pd_opid
            ON ft_posiciones_diarias(operacion_id)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_ft_pd_estado
            ON ft_posiciones_diarias(estado)
    """))
    print("     ft_posiciones_diarias creada con 3 indices.")

    print("\n  [2/2] Agregando columnas a ft_candidatos_diarios...")
    for col, tipo in [
        ("precio_apertura", "NUMERIC(12,4)"),
        ("retorno_5d",      "NUMERIC(8,4)"),
        ("retorno_10d",     "NUMERIC(8,4)"),
        ("retorno_20d",     "NUMERIC(8,4)"),
    ]:
        exists = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'ft_candidatos_diarios'
                  AND column_name = :col
            )
        """), {"col": col}).scalar()

        if not exists:
            conn.execute(text(
                f"ALTER TABLE ft_candidatos_diarios ADD COLUMN {col} {tipo}"
            ))
            print(f"     + {col} {tipo}")
        else:
            print(f"     = {col} ya existe, skip.")

    conn.commit()
    print("\n  Migracion completada.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Solo muestra el estado actual sin modificar")
    args = parser.parse_args()

    engine = get_engine()
    with engine.connect() as conn:
        print("\n=== Estado antes de migrar ===")
        cmd_check(conn)

        if not args.check:
            print("\n=== Ejecutando migracion ===")
            cmd_migrate(conn)
            print("\n=== Estado final ===")
            cmd_check(conn)

    print()


if __name__ == "__main__":
    main()
