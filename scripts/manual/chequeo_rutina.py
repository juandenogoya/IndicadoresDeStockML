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

import pandas as pd

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


def leer_fechas():
    """({tabla: fecha de DATOS}, {tabla: ultima escritura}) en UNA query.

    Las dos fechas salen juntas porque contestan preguntas distintas y hubo que
    aprenderlo a la mala (2/9/2026): `MAX(columna)` dice a que rueda pertenece
    el contenido -- lo unico que sirve para detectar mezcla -- y
    `MAX(columna_registro)` dice cuando corrio el paso. El scanner tenia la
    segunda fresca y la primera atrasada.
    """
    existentes = _tablas_existentes([t.nombre for t in TABLAS])
    partes = []
    for t in TABLAS:
        if t.nombre not in existentes:
            continue
        reg = (f"MAX({t.columna_registro})::timestamp" if t.columna_registro
               else "NULL::timestamp")
        partes.append(f"SELECT '{t.nombre}' AS tabla, "
                      f"MAX({t.columna})::date AS fecha, "
                      f"{reg} AS registro FROM {t.nombre}")
    if not partes:
        return {}, {}
    df = query_df(" UNION ALL ".join(partes))

    # pandas convierte los NULL en NaT, que NO es None y ademas es instancia de
    # datetime -- pasa cualquier chequeo `is not None` y revienta al formatear.
    # Se normaliza aca, que es la frontera: estado_pipeline es puro y no tiene
    # por que conocer los tipos de pandas.
    def _limpio(v):
        return None if pd.isna(v) else v

    fechas = {r["tabla"]: _limpio(r["fecha"]) for _, r in df.iterrows()}
    registros = {r["tabla"]: _limpio(r["registro"]) for _, r in df.iterrows()}
    return fechas, registros


def imprimir(diag: dict) -> None:
    print()
    print(SEP)
    print("  CHEQUEO DE LA RUTINA DIARIA")
    print(SEP)
    print()
    print(f"  Ultimo dia habil NYSE cerrado : {diag['esperado']}")
    print(f"  Ancla (precios_diarios)       : {diag['ancla']}")
    print()
    # "Datos" y "Ultima corrida" son columnas distintas a proposito: el scanner
    # del 2/9/2026 tenia la corrida de ayer y los datos de anteayer, y con una
    # sola columna eso se ve como si estuviera al dia.
    print(f"  {'Tabla':<24} {'Datos':<12} {'vs precios':<14} "
          f"{'Ultima corrida':<17} {'Insumo?'}")
    print(f"  {'-'*24} {'-'*12} {'-'*14} {'-'*17} {'-'*7}")
    for f in diag["tablas"]:
        d = f["dias_vs_ancla"]
        if d is None:
            vs = "-"
        elif d <= 0:
            vs = "al dia"
        else:
            vs = f"{d} {'rueda' if d == 1 else 'ruedas'} atras"
        reg = f["registro"]
        reg_txt = reg.strftime("%Y-%m-%d %H:%M") if reg is not None else "-"
        marca = " " if f["al_dia"] else "!"
        print(f" {marca}{f['etiqueta']:<24} {str(f['fecha']):<12} {vs:<14} "
              f"{reg_txt:<17} {'si' if f['critica'] else 'no'}")
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

    fechas, registros = leer_fechas()
    if not fechas:
        print("[ERROR] No se pudo leer ninguna tabla. Revisar la DB local.")
        return 2

    diag = diagnosticar(fechas, esperado=prev_trading_day(date.today()),
                        registros=registros)
    imprimir(diag)

    if diag["ancla"] is None:
        return 2
    if diag["mezcla"] and not args.solo_avisar:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
