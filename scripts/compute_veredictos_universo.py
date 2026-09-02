"""
compute_veredictos_universo.py
Precomputa el veredicto sintetico (ALCISTA / NEUTRAL / BAJISTA) de todo el
universo y lo persiste en veredictos_universo_diario (LOCAL).

Que resuelve:
    El screener del dashboard calculaba esto en vivo, al apretar Buscar: ~2
    minutos la primera vez de cada dia, cacheado solo en memoria del proceso
    Streamlit (se perdia al reiniciar la app). Aca se paga una vez de noche,
    con el resto de la rutina, y el dashboard pasa a SOLO LEER.

Donde va en la rutina nocturna:
    En ft_run_diario.bat, DESPUES del paso [0b] (compute_opciones_derivadas).
    Es obligatorio ese orden: el veredicto vota con opciones_pcr_plazo_diario,
    que ese paso es el que la calcula desde el crudo recien sincronizado. Antes
    de [0b], la dimension de opciones del veredicto leeria el dia anterior.

Idempotente:
    UPSERT por (ticker, fecha). Correrlo dos veces el mismo dia reescribe las
    mismas filas. Si una noche no se corre, la corrida siguiente NO rellena el
    hueco hacia atras -- escribe la fecha del cierre que uso. Para rellenar un
    dia viejo hace falta que precios_diarios este en ese estado, cosa que no
    pasa: por eso el hueco simplemente queda, y el dashboard lo muestra como
    dato viejo en vez de fingir que es de hoy.

Capa derivada y pura: se puede borrar la tabla entera y regenerarla.
LOCAL-only (Plan C).

Uso:
    python scripts/compute_veredictos_universo.py
    python scripts/compute_veredictos_universo.py --dry-run
    python scripts/compute_veredictos_universo.py --status
"""

import os
import sys
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# LOCAL-only: si DATABASE_URL estuviera seteada, get_engine caeria a Railway.
# Se saca ANTES de importar cualquier cosa que abra conexion.
os.environ.pop("DATABASE_URL", None)

from sqlalchemy import text

from dashboard.sintesis_data import cargar_veredictos_universo, fecha_datos
from scripts.oneshot.create_fundamentales_tables import get_local_engine

SEP = "=" * 64
TABLA = "veredictos_universo_diario"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


UPSERT = f"""
INSERT INTO {TABLA} (ticker, fecha, sector, veredicto, frase, computed_at)
VALUES (:ticker, :fecha, :sector, :veredicto, :frase, NOW())
ON CONFLICT (ticker, fecha) DO UPDATE SET
    sector      = EXCLUDED.sector,
    veredicto   = EXCLUDED.veredicto,
    frase       = EXCLUDED.frase,
    computed_at = NOW()
"""


def mostrar_status(engine) -> None:
    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT MAX(fecha) AS ultima, COUNT(DISTINCT fecha) AS dias, COUNT(*) AS filas
            FROM {TABLA}
        """)).mappings().first()
        if not row or row["ultima"] is None:
            log(f"{TABLA}: vacia.")
            return
        log(f"{TABLA}: {row['filas']} filas | {row['dias']} fechas | "
            f"ultima {row['ultima']}")
        reparto = conn.execute(text(f"""
            SELECT veredicto, COUNT(*) AS n FROM {TABLA}
            WHERE fecha = (SELECT MAX(fecha) FROM {TABLA})
            GROUP BY veredicto ORDER BY n DESC
        """)).mappings().all()
        detalle = " | ".join(f"{r['veredicto']} {r['n']}" for r in reparto)
        log(f"  reparto del {row['ultima']}: {detalle}")
    log(f"  fecha de datos actual (MAX precios_diarios): {fecha_datos()}")


def main():
    ap = argparse.ArgumentParser(
        description="Precomputa los veredictos del universo (LOCAL)")
    ap.add_argument("--dry-run", action="store_true",
                    help="calcula y reporta, sin escribir en la DB")
    ap.add_argument("--status", action="store_true",
                    help="solo muestra el estado de la tabla y sale")
    args = ap.parse_args()

    engine = get_local_engine()

    if args.status:
        mostrar_status(engine)
        return

    fecha = fecha_datos()
    if not fecha:
        log("ERROR: precios_diarios vacia, no hay fecha de datos. Aborto.")
        sys.exit(1)

    print()
    print(SEP)
    print(f"  VEREDICTOS DEL UNIVERSO  |  fecha de datos {fecha}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    t0 = datetime.now()
    log("Calculando (recorre el universo ticker por ticker, ~2 min)...")
    filas = cargar_veredictos_universo()
    seg = (datetime.now() - t0).total_seconds()

    if not filas:
        log("ERROR: el calculo no devolvio ningun ticker. Aborto sin escribir.")
        sys.exit(1)

    reparto = {}
    for f in filas:
        reparto[f["veredicto"]] = reparto.get(f["veredicto"], 0) + 1
    detalle = " | ".join(f"{k} {v}" for k, v in
                         sorted(reparto.items(), key=lambda kv: -kv[1]))
    log(f"{len(filas)} tickers evaluados en {seg:.0f}s -> {detalle}")

    if args.dry_run:
        log("DRY-RUN: no se escribio nada.")
        return

    params = [{"ticker": f["ticker"], "fecha": fecha, "sector": f["sector"],
               "veredicto": f["veredicto"], "frase": f["frase"]} for f in filas]
    with engine.connect() as conn:
        conn.execute(text(UPSERT), params)
        conn.commit()
    log(f"UPSERT OK en {TABLA} ({len(params)} filas, fecha {fecha}).")

    print()
    print(SEP)
    print(f"  Completado en {(datetime.now() - t0).total_seconds():.0f}s")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
