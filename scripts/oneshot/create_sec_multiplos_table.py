"""
create_sec_multiplos_table.py
Crea la tabla fundamentales_sec_multiplos_d (serie DIARIA de multiplos sobre la
fuente SEC XBRL) en local.

Contexto (docs/fuentes_fundamentales.md, Fase 3):
    Capa DERIVADA y recomputable: funcion pura de fundamentales_sec_q +
    precios_diarios via src/utils/fundamentales_ttm.py. Se puede borrar y
    reconstruir sin salir a la red. NINGUN consumidor la lee todavia.

Por que una fila por DIA y no por trimestre:
    El denominador (TTM) se mueve 4 veces al ano; el numerador (precio) todos
    los dias. Es lo mismo que ya hace multiplos_px con el ultimo Q, pero
    extendido a toda la historia -- y eso es lo que habilita la pregunta
    original: "este PER, contra su propia historia, esta caro o barato".
    Con una foto por trimestre no hay distribucion contra la cual comparar.

El ancla es `filed_primero`, no `period_end`:
    cada rueda toma el TTM que estaba PUBLICO ese dia. Un trimestre cerrado el
    30/9 no estuvo disponible el 30/9. `lag_dias` guarda la distancia, que es
    el control: si crece mucho, ese ticker dejo de reportar.

Lo que esta tabla NO resuelve (declarado, no corregido):
    - EBITDA = EBIT + D&A, sin excluir one-offs. Un Q con impairment lo deforma.
      yahooquery usa NormalizedEBITDA; SEC no publica un equivalente.
    - Financieras: no tagean OperatingIncomeLoss -> ebitda_ttm queda NULL ->
      ev_ebitda NULL solo. No hace falta una columna de perfil: la ausencia
      hace el trabajo. `net_debt` se calcula igual pero para un banco no
      significa lo que significa en una industrial -- no interpretarlo.
    - Empresas que se re-registraron con CIK nuevo (XOM, BLK, BG) arrancan su
      historia ahi. `n_periodos` lo deja a la vista.

Uso:
    python scripts/oneshot/create_sec_multiplos_table.py
    python scripts/oneshot/create_sec_multiplos_table.py --dry-run
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
TABLA = "fundamentales_sec_multiplos_d"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DDL = """
CREATE TABLE IF NOT EXISTS fundamentales_sec_multiplos_d (
    id                    SERIAL         PRIMARY KEY,
    ticker                VARCHAR(20)    NOT NULL,
    fecha                 DATE           NOT NULL,   -- rueda
    close                 NUMERIC(20,6),

    -- Ancla point-in-time: que trimestre estaba publico esta rueda
    period_end            DATE,
    filed_primero         DATE,
    lag_dias              SMALLINT,      -- fecha - filed_primero
    n_periodos            SMALLINT,      -- trimestres SEC disponibles al ancla

    -- Denominadores usados (se guardan para poder auditar el multiplo).
    -- TODOS son AGREGADOS. No hay ninguna magnitud por accion aca, y es a
    -- proposito: SEC re-expresa lo "por accion" ante un split y
    -- precios_diarios no, asi que eps_ttm o BVPS cruzados con un precio
    -- historico quedan partidos por el factor del split. Los agregados son
    -- invariantes. Ver src/utils/sec_acciones.py.
    revenue_ttm           NUMERIC(24,2),
    net_income_ttm        NUMERIC(24,2),
    ebitda_ttm            NUMERIC(24,2),
    fcf_ttm               NUMERIC(24,2),
    equity                NUMERIC(24,2),
    net_debt              NUMERIC(24,2),
    -- conteo POINT-IN-TIME (portada del filing vigente esa rueda)
    shares                NUMERIC(20,2),
    shares_fuente         VARCHAR(20),

    -- Multiplos al cierre de la rueda
    market_cap            NUMERIC(24,2),
    enterprise_value      NUMERIC(24,2),
    pe_ratio              NUMERIC(20,4),
    pb_ratio              NUMERIC(20,4),
    ps_ratio              NUMERIC(20,4),
    ev_ebitda             NUMERIC(20,4),
    fcf_yield             NUMERIC(20,6),  -- fcf_ttm / market_cap

    -- "Caro vs si misma": percentil del multiplo dentro de su propia historia
    -- en una ventana movil TRAILING (solo pasado, sin lookahead).
    pe_pct                NUMERIC(6,4),
    pb_pct                NUMERIC(6,4),
    ps_pct                NUMERIC(6,4),
    ev_ebitda_pct         NUMERIC(6,4),
    n_obs_pct             SMALLINT,       -- observaciones de la ventana

    computed_at           TIMESTAMP      NOT NULL DEFAULT NOW(),

    -- UNIQUE FULL (sin WHERE): requisito de ON CONFLICT. Ver CLAUDE.md.
    CONSTRAINT fundamentales_sec_multiplos_d_uniq UNIQUE (ticker, fecha)
)
"""

ALTERS = [
    f"ALTER TABLE {TABLA} ADD COLUMN IF NOT EXISTS equity NUMERIC(24,2)",
    f"ALTER TABLE {TABLA} ADD COLUMN IF NOT EXISTS shares_fuente VARCHAR(20)",
    # Se sacan: cruzadas con un precio historico quedan partidas por el split.
    f"ALTER TABLE {TABLA} DROP COLUMN IF EXISTS eps_ttm",
    f"ALTER TABLE {TABLA} DROP COLUMN IF EXISTS book_value_per_share",
]

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_sec_mult_ticker_fecha "
    "ON fundamentales_sec_multiplos_d (ticker, fecha)",
    "CREATE INDEX IF NOT EXISTS idx_sec_mult_fecha "
    "ON fundamentales_sec_multiplos_d (fecha)",
    # El corte tipico del screener: "que hay hoy debajo de su percentil 20".
    "CREATE INDEX IF NOT EXISTS idx_sec_mult_fecha_pe_pct "
    "ON fundamentales_sec_multiplos_d (fecha, pe_pct)",
]


def crear(engine, dry_run: bool):
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_schema='public' AND table_name=:t)
        """), {"t": TABLA}).scalar()
        if existe:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLA}")).scalar()
            log(f"{TABLA}: ya existe ({n} filas).")
        elif dry_run:
            log(f"{TABLA}: DRY-RUN, se crearia.")
            return
        else:
            conn.execute(text(DDL.strip()))
            log(f"{TABLA}: creada.")
        if not dry_run:
            for stmt in ALTERS:
                conn.execute(text(stmt))
            for stmt in INDICES:
                conn.execute(text(stmt))
            conn.commit()
            log(f"indices verificados ({len(INDICES)}).")


def main():
    p = argparse.ArgumentParser(
        description="Crea fundamentales_sec_multiplos_d (LOCAL-only)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mostrar-ddl", action="store_true")
    args = p.parse_args()

    if args.mostrar_ddl:
        print(DDL)
        return

    print()
    print(SEP)
    print(f"  CREATE {TABLA}{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()
    crear(get_local_engine(), args.dry_run)
    print()
    print(SEP)
    print("  OK")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
