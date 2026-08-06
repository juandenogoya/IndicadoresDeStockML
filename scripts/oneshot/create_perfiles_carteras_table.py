"""
create_perfiles_carteras_table.py
Crea la tabla perfiles_ticker (Fase 3 del perfilado de carteras) en local.

Contexto (docs/perfiles_carteras.md):
    Snapshot del perfil de RIESGO de cada ticker del universo. Capa DERIVADA y
    recomputable: funcion de precios_diarios + futuros ES (benchmark) + activos,
    via el motor puro src/utils/perfil_metricas (Fase 1) + perfil_riesgo (Fase 2).
    La pobla scripts/compute_perfiles_carteras.py, cadencia MENSUAL (el perfil es
    una propiedad estable del instrumento, no una senal diaria).

Modelo (decidido 5/8/2026):
    - perfil = COMPORTAMIENTO cuantitativo puro (caja por cuartil del percentil
      composite del universo). El sector NO capa la etiqueta.
    - sector queda como CONTEXTO (caja_base) y fuente del flag `excepcion`
      (comportamiento se despega 2+ cajas de su sector).

PK (ticker, fecha): 1 fila por ticker por corrida mensual -> historia de perfiles
(habilita ver DRIFT: cuando un ticker migra de caja). LOCAL-only (Plan C).

Uso:
    python scripts/oneshot/create_perfiles_carteras_table.py
    python scripts/oneshot/create_perfiles_carteras_table.py --dry-run
"""

import sys
import os
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import text
from scripts.oneshot.create_fundamentales_tables import get_local_engine

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


DDL = """
CREATE TABLE IF NOT EXISTS perfiles_ticker (
    ticker            VARCHAR(20)   NOT NULL,
    fecha             DATE          NOT NULL,   -- snapshot (corrida mensual)
    sector            VARCHAR(64),
    industry          VARCHAR(96),
    -- clasificacion
    perfil            VARCHAR(20)   NOT NULL,   -- Conservadora/Moderada/Arriesgada/Especulativa
    perfil_ordinal    SMALLINT      NOT NULL,   -- 0..3 (= caja_cuant)
    caja_base         SMALLINT,                 -- prior sectorial 0..3 (contexto)
    caja_base_fuente  VARCHAR(16),              -- sector | fallback
    caja_cuant        SMALLINT,                 -- 0..3 por cuartil
    score_riesgo      NUMERIC(6,2),             -- composite percentil 0..100
    movio             SMALLINT,                 -- despegue vs sector (con signo)
    excepcion         BOOLEAN       NOT NULL DEFAULT FALSE,
    sin_cuant         BOOLEAN       NOT NULL DEFAULT FALSE,
    -- ranking intra-caja
    rank_en_caja      SMALLINT,
    n_en_caja         SMALLINT,
    pct_en_caja       NUMERIC(5,1),
    -- metricas de Fase 1 (crudas)
    atr_pct_d         NUMERIC(8,4),
    atr_pct_w         NUMERIC(8,4),
    atr_pct_m         NUMERIC(8,4),
    beta              NUMERIC(8,4),
    max_dd_1a         NUMERIC(6,2),
    max_dd_hist       NUMERIC(6,2),
    -- percentiles por eje (explican el score)
    pct_atr_pct_w     NUMERIC(5,2),
    pct_atr_pct_m     NUMERIC(5,2),
    pct_beta          NUMERIC(5,2),
    pct_max_dd_1a     NUMERIC(5,2),
    fecha_datos       DATE,                     -- ultima rueda de precios usada
    computed_at       TIMESTAMP     NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, fecha)
);
"""

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_perfiles_fecha ON perfiles_ticker (fecha)",
    "CREATE INDEX IF NOT EXISTS idx_perfiles_perfil ON perfiles_ticker (perfil)",
    "CREATE INDEX IF NOT EXISTS idx_perfiles_sector ON perfiles_ticker (sector)",
    "CREATE INDEX IF NOT EXISTS idx_perfiles_excepcion ON perfiles_ticker (excepcion)",
]


def main():
    ap = argparse.ArgumentParser(description="Crea perfiles_ticker en local")
    ap.add_argument("--dry-run", action="store_true", help="muestra el DDL sin ejecutar")
    args = ap.parse_args()

    log(SEP)
    log("Crear tabla perfiles_ticker (LOCAL)")
    log(SEP)

    if args.dry_run:
        log("[DRY RUN] DDL:")
        print(DDL)
        for ix in INDICES:
            print(ix + ";")
        return

    eng = get_local_engine()
    with eng.begin() as c:
        c.execute(text(DDL))
        log("Tabla perfiles_ticker creada (o ya existia).")
        for ix in INDICES:
            c.execute(text(ix))
        log(f"{len(INDICES)} indices verificados.")

    # verificacion
    with eng.connect() as c:
        cols = c.execute(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'perfiles_ticker'")).scalar()
    log(f"perfiles_ticker tiene {cols} columnas. Listo.")


if __name__ == "__main__":
    main()
