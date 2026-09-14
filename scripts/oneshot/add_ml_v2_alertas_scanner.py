"""
add_ml_v2_alertas_scanner.py -- Etapa 3d (13/9/2026): columnas del modelo ML v2 en
alertas_scanner.

El scanner calcula la v2 EN PARALELO a la v1 (misma fila por ticker y rueda) y la
estrategia FT_ML_SCANNER_v2 lee estas columnas. Van en la misma tabla y no en una
aparte para que la verificacion post-facto (retorno_20d_real) cubra a las dos sin
codigo extra y la comparacion v1 vs v2 sea sobre las mismas filas.

    ml_prob_v2      probabilidad calibrada de la v2
    ml_modelo_v2    version del artefacto (metadata.json -> version)
    alert_score_v2  score compuesto con la probabilidad y los cortes de la v2
    alert_nivel_v2  nivel de alerta con ese score

NULLables: la historia previa queda en NULL (la v2 no existia) y una corrida sin el
artefacto tambien. Idempotente (IF NOT EXISTS). LOCAL: el scanner corre en local.

Uso (desde la raiz):
    python scripts/oneshot/add_ml_v2_alertas_scanner.py --dry-run
    python scripts/oneshot/add_ml_v2_alertas_scanner.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text

from src.data.database import get_engine

# LOCAL-only: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

DDL = """
ALTER TABLE alertas_scanner
    ADD COLUMN IF NOT EXISTS ml_prob_v2     NUMERIC,
    ADD COLUMN IF NOT EXISTS ml_modelo_v2   VARCHAR(40),
    ADD COLUMN IF NOT EXISTS alert_score_v2 NUMERIC,
    ADD COLUMN IF NOT EXISTS alert_nivel_v2 VARCHAR(20)
"""

COMENTARIOS = {
    "ml_prob_v2":     "Modelo ML v2 (RF calibrado, Etapa 3): probabilidad de ganancia a 20d",
    "ml_modelo_v2":   "Version del artefacto v2 (models_ml_v2/metadata.json)",
    "alert_score_v2": "Score compuesto con la probabilidad y los cortes de la v2",
    "alert_nivel_v2": "Nivel de alerta de la v2 (lo lee FT_ML_SCANNER_v2)",
}


def columnas(conn):
    return [r[0] for r in conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'alertas_scanner' AND column_name LIKE '%\\_v2' ORDER BY ordinal_position"
    ))]


def main():
    ap = argparse.ArgumentParser(description="Columnas de la v2 en alertas_scanner")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    engine = get_engine()
    with engine.connect() as conn:
        antes = columnas(conn)
    print(f"Columnas _v2 actuales: {antes or 'ninguna'}")
    if args.dry_run:
        print(DDL)
        return

    with engine.begin() as conn:
        conn.execute(text(DDL))
        for col, comentario in COMENTARIOS.items():
            conn.execute(text(f"COMMENT ON COLUMN alertas_scanner.{col} IS :c"), {"c": comentario})
    with engine.connect() as conn:
        print(f"Columnas _v2 despues: {columnas(conn)}")


if __name__ == "__main__":
    main()
