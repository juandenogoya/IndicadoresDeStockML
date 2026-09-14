"""
create_rutina_corridas.py
Crea la tabla rutina_corridas: una fila por cada paso de la rutina diaria que
se ejecuto, con cuando, cuanto tardo, como termino y que rueda de datos dejo.
LOCAL-only.

Contexto (Etapa 2, 13/9/2026): para fechar los cambios de FT hubo que
reconstruir el orden de las corridas cruzando timestamps de filas escritas,
archivos y commits. Esta tabla responde eso directo. La escribe
scripts/manual/rutina_diaria.py (la rutina completa y cada paso suelto) y
ft_run_diario.bat cuando se corre por su cuenta.

Uso:
    python scripts/oneshot/create_rutina_corridas.py            # dry-run: muestra el DDL
    python scripts/oneshot/create_rutina_corridas.py --apply    # crea (idempotente)
    python scripts/oneshot/create_rutina_corridas.py --status
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.pop("DATABASE_URL", None)   # LOCAL-only

from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402
from src.utils.rutina import RESULTADOS  # noqa: E402


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


_RESULTADOS_SQL = ", ".join(f"'{r}'" for r in RESULTADOS)

# Sin dos puntos en los COMMENT: text() de SQLAlchemy los toma como parametros.
DDL = f"""
CREATE TABLE IF NOT EXISTS rutina_corridas (
    id             SERIAL PRIMARY KEY,
    rutina_id      VARCHAR(20),
    paso           VARCHAR(40)  NOT NULL,
    origen         VARCHAR(10)  NOT NULL,
    inicio         TIMESTAMP    NOT NULL,
    fin            TIMESTAMP,
    duracion_s     INTEGER,
    exit_code      INTEGER,
    resultado      VARCHAR(12)  NOT NULL,
    rueda_antes    DATE,
    rueda_despues  DATE,
    detalle        JSONB,
    log_path       TEXT,
    git_commit     VARCHAR(12),
    registrado_en  TIMESTAMP    NOT NULL DEFAULT NOW(),
    CONSTRAINT rutina_corridas_resultado_chk CHECK (resultado IN ({_RESULTADOS_SQL})),
    CONSTRAINT rutina_corridas_origen_chk CHECK (origen IN ('rutina', 'suelto'))
);

CREATE INDEX IF NOT EXISTS idx_rutina_corridas_paso_inicio
    ON rutina_corridas (paso, inicio DESC);

COMMENT ON TABLE rutina_corridas IS
    'Una fila por paso ejecutado de la rutina diaria (sync, paso1, paso2, paso3, ft '
    'y recovery_incremental). La escribe scripts/manual/rutina_diaria.py';
COMMENT ON COLUMN rutina_corridas.rutina_id IS
    'AAAAMMDD_HHMM de la corrida completa que lo lanzo; NULL si se corrio suelto.';
COMMENT ON COLUMN rutina_corridas.resultado IS
    'EN_CURSO al arrancar. Si queda EN_CURSO, el proceso murio sin terminar '
    '(ventana cerrada, corte de luz).';
COMMENT ON COLUMN rutina_corridas.rueda_antes IS
    'Fecha de DATOS de la tabla del paso antes de correr (ver src/utils/rutina.py).';
COMMENT ON COLUMN rutina_corridas.detalle IS
    'notas, tickers pendientes con su ultimo dato y huecos en el medio de la serie.';
"""


def estado(engine):
    with engine.connect() as conn:
        if conn.execute(text("SELECT to_regclass('public.rutina_corridas')")).scalar() is None:
            log("rutina_corridas NO existe.")
            return
        rows = conn.execute(text("""
            SELECT paso, origen, inicio, resultado, duracion_s, rueda_despues
            FROM rutina_corridas ORDER BY inicio DESC LIMIT 15
        """)).fetchall()
    log(f"rutina_corridas: ultimas {len(rows)} filas")
    for r in rows:
        print(f"   {r.inicio:%Y-%m-%d %H:%M}  {r.paso:<22} {r.origen:<7} "
              f"{r.resultado:<12} {r.duracion_s or '-':>6}s  datos {r.rueda_despues}")


def main():
    ap = argparse.ArgumentParser(description="Crea rutina_corridas (LOCAL).")
    ap.add_argument("--apply", action="store_true", help="crea la tabla")
    ap.add_argument("--status", action="store_true", help="ultimas corridas")
    args = ap.parse_args()

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")
    if args.status:
        estado(engine)
        return 0
    if not args.apply:
        print(DDL)
        log("DRY-RUN: nada escrito. Correr con --apply.")
        return 0
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    log("rutina_corridas creada (o ya existia).")
    estado(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
