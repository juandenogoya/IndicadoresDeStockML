"""
create_veredictos_universo_table.py
Crea la tabla veredictos_universo_diario (LOCAL).

Por que existe:
    El "Screener por veredicto" del dashboard calculaba el veredicto sintetico
    de los ~199 tickers EN VIVO, al apretar Buscar: cargar_datos_ticker +
    sintetizar por ticker, ~2 minutos la primera vez de cada dia. El resultado
    se cacheaba en memoria del proceso Streamlit, asi que se perdia al reiniciar
    la app y se pagaba de nuevo. Dos minutos de espera es la razon practica por
    la que la funcion se usa poco.

    El calculo NO depende de nada que pase durante el dia: es funcion del
    ultimo cierre. Corresponde precomputarlo una vez de noche, con el resto de
    la rutina, y que el dashboard SOLO LEA.

Por que la clave es (ticker, fecha) y no solo ticker:
    Guardar la historia sale gratis (1 fila por ticker por rueda, ~200/dia) y
    habilita algo que hoy no se puede: ver como se movio el reparto
    ALCISTA/NEUTRAL/BAJISTA del universo dia a dia. Esa serie es el insumo del
    bloque "clima del universo" de la vista Hoy.

OJO -- `fecha` es la fecha de DATOS, no la de corrida:
    Misma convencion que ft_operaciones.fecha_datos (ver CLAUDE.md, "FT
    asincronico"). La rutina nocturna es MANUAL: si una noche no se corre, la
    corrida siguiente escribe la fecha del cierre que efectivamente uso, no la
    del dia en que se ejecuto. Con la fecha de corrida, cruzar esta tabla con
    precios_diarios leeria el dia equivocado en silencio.

Capa DERIVADA y recomputable:
    Es funcion pura de precios_diarios + indicadores_tecnicos +
    opciones_pcr_plazo_diario + features_market_structure via
    dashboard.sintesis_data.cargar_veredictos_universo() y
    src.utils.dashboard_sintesis.sintetizar(). Se puede borrar y regenerar.

LOCAL-only (Plan C): el dashboard corre solo en Windows local y ningun bot ni
workflow lee esta tabla.

Uso:
    python scripts/oneshot/create_veredictos_universo_table.py
    python scripts/oneshot/create_veredictos_universo_table.py --dry-run
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
TABLA = "veredictos_universo_diario"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# PRIMARY KEY compuesta (no id SERIAL + UNIQUE): (ticker, fecha) ES la clave
# natural, y el indice unico que genera la PK es FULL -> sirve para ON CONFLICT
# (ver CLAUDE.md: un unique index parcial no alcanza).
DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    ticker       VARCHAR(20)  NOT NULL,
    fecha        DATE         NOT NULL,
    sector       VARCHAR(100),
    veredicto    VARCHAR(10)  NOT NULL,
    frase        TEXT,
    computed_at  TIMESTAMP    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, fecha)
)
"""

INDICES = [
    # El acceso dominante del dashboard es "todo el universo de una fecha,
    # filtrado por veredicto y sector".
    f"CREATE INDEX IF NOT EXISTS idx_vered_univ_fecha ON {TABLA} (fecha)",
    f"CREATE INDEX IF NOT EXISTS idx_vered_univ_fecha_ver ON {TABLA} (fecha, veredicto)",
]


def main():
    ap = argparse.ArgumentParser(description=f"Crea {TABLA} en la DB local")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print()
    print(SEP)
    print(f"  CREATE {TABLA}  |  target=local"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    engine = get_local_engine()
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_schema='public' AND table_name=:t)
        """), {"t": TABLA}).scalar()

        if existe:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLA}")).scalar()
            log(f"{TABLA}: ya existe ({n} filas). Verificando indices...")
            if args.dry_run:
                log(f"  DRY-RUN: verificaria {len(INDICES)} indices.")
                return
            for stmt in INDICES:
                conn.execute(text(stmt))
            conn.commit()
            log("  indices OK.")
            return

        if args.dry_run:
            log(f"{TABLA}: DRY-RUN, crearia la tabla + {len(INDICES)} indices.")
            return

        conn.execute(text(DDL.strip()))
        for stmt in INDICES:
            conn.execute(text(stmt))
        conn.commit()
        log(f"{TABLA}: creada (0 filas) + {len(INDICES)} indices.")

    print()
    print(SEP)
    print("  Siguiente paso: python scripts/compute_veredictos_universo.py")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
