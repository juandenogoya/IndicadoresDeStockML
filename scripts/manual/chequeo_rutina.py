"""
chequeo_rutina.py
Verifica que la rutina diaria manual haya quedado COHERENTE antes de que algo
opere con esos datos.

QUE PROBLEMA RESUELVE
    La rutina es manual y son 4 pasos:

        1. sync_opciones_railway_to_local.bat   (crudo de opciones <- Railway)
        2. cron_paso1_precios_yq.bat            (precios + indicadores)
        3. cron_paso2_features.bat              (features)
        4. cron_paso3_scanner.bat               (scanner ML)
        5. ft_run_diario.bat                    (deriva opciones + BOTS + reportes)

    Saltarse uno no rompe nada de forma visible: cada tabla queda con su propia
    fecha y todo lo que las cruza sigue andando, mezclando ruedas en silencio.
    El 2/9/2026 no se corrio ft_run_diario y el sistema estuvo cruzando tecnico
    del 1/9 con opciones del 31/8 sin una sola queja.

DONDE VA
    Dentro de ft_run_diario.bat, DESPUES del paso [0b] y ANTES del primer bot.

    Despues de [0b] y no antes: [0b] es justamente el que deriva las opciones,
    asi que correrlo antes frenaria por algo que el propio script esta por
    arreglar.

    Antes del primer bot porque los 10 bots FT son lo unico de la cadena que
    ACTUA: abren y cierran posiciones. Todo lo demas (equity, reportes,
    veredictos) es descriptivo y se puede recomputar.

QUE FRENA Y QUE NO
    NO frena por antiguedad. Que los datos sean del cierre anterior es la
    convencion del proyecto: los bots FT operan asi el 73% de las veces
    (CLAUDE.md, "FT asincronico"). Un guard por antiguedad romperia el caso
    normal.

    SI frena por MEZCLA: tablas que no coinciden entre si. Ahi una decision se
    computa con tecnico de una rueda y opciones de otra.

    El razonamiento vive en src/utils/estado_pipeline.py (modulo PURO,
    compartido con la banda de frescura del dashboard) para que no haya dos
    definiciones de "estan alineados los datos".

Uso:
    python scripts/manual/chequeo_rutina.py
    python scripts/manual/chequeo_rutina.py --solo-avisar   (nunca falla)

Codigo de salida:
    0 = coherente (puede estar viejo, pero alineado)
    1 = MEZCLA de ruedas -> no conviene operar
    2 = no hay datos / error de conexion
"""

import os
import sys
import argparse
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# LOCAL-only: con DATABASE_URL seteada, get_engine cae a Railway (que bajo Plan C
# solo tiene opciones) y el diagnostico seria sobre la DB equivocada.
os.environ.pop("DATABASE_URL", None)

from src.data.database import query_df                       # noqa: E402
from src.utils.trading_calendar import prev_trading_day       # noqa: E402
from src.utils.estado_pipeline import (                       # noqa: E402
    TABLAS, diagnosticar, resumen,
)

SEP = "=" * 68


def _tablas_existentes(nombres):
    df = query_df(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name = ANY(:n)",
        {"n": list(nombres)},
    )
    return set(df["table_name"]) if not df.empty else set()


def leer_fechas() -> dict:
    """{tabla: fecha maxima} en UNA query (UNION ALL), solo de las que existen."""
    existentes = _tablas_existentes([t.nombre for t in TABLAS])
    partes = [f"SELECT '{t.nombre}' AS tabla, MAX({t.columna})::date AS fecha "
              f"FROM {t.nombre}"
              for t in TABLAS if t.nombre in existentes]
    if not partes:
        return {}
    df = query_df(" UNION ALL ".join(partes))
    return {r["tabla"]: r["fecha"] for _, r in df.iterrows()}


def imprimir(diag: dict) -> None:
    print()
    print(SEP)
    print("  CHEQUEO DE LA RUTINA DIARIA")
    print(SEP)
    print()
    print(f"  Ultimo dia habil NYSE cerrado : {diag['esperado']}")
    print(f"  Ancla (precios_diarios)       : {diag['ancla']}")
    print()
    print(f"  {'Tabla':<24} {'Fecha':<12} {'vs precios':<14} {'Insumo?'}")
    print(f"  {'-'*24} {'-'*12} {'-'*14} {'-'*7}")
    for f in diag["tablas"]:
        d = f["dias_vs_ancla"]
        if d is None:
            vs = "-"
        elif d <= 0:
            vs = "al dia"
        else:
            vs = f"{d} {'rueda' if d == 1 else 'ruedas'} atras"
        marca = " " if f["al_dia"] else "!"
        print(f" {marca}{f['etiqueta']:<24} {str(f['fecha']):<12} {vs:<14} "
              f"{'si' if f['critica'] else 'no'}")
    print()
    print(f"  {resumen(diag)}")
    if diag["arreglos"]:
        print()
        print("  Falta correr:")
        for a in diag["arreglos"]:
            print(f"    - {a}")
    print()
    print(SEP)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verifica la coherencia de la rutina diaria (LOCAL)")
    ap.add_argument("--solo-avisar", action="store_true",
                    help="imprime el diagnostico pero siempre sale con 0")
    args = ap.parse_args()

    fechas = leer_fechas()
    if not fechas:
        print("[ERROR] No se pudo leer ninguna tabla. Revisar la DB local.")
        return 2

    diag = diagnosticar(fechas, esperado=prev_trading_day(date.today()))
    imprimir(diag)

    if diag["ancla"] is None:
        return 2
    if diag["mezcla"] and not args.solo_avisar:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
