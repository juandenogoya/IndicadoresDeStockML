"""
ft_cambios.py
Registro de los cambios que afectan a las estrategias de Forward Testing
(tabla ft_cambios, LOCAL). Sin este registro un cambio no se puede medir: el
reporte corta la historia de cada estrategia en tramos por las fechas de aca.

Diseno y reglas: docs/forward_testing/METRICAS.md, seccion 12.
Tabla y carga inicial: scripts/oneshot/create_ft_cambios.py

CUANDO REGISTRAR: cada vez que se toca algo con lo que una estrategia decide
(logica, parametros, modelo, un bug de su query) o un dato o una pieza de infra
que mueve sus decisiones.

--cambia-decisiones: SOLO si cambia la LOGICA, los PARAMETROS o el MODELO de la
estrategia. Esos cortan tramos. Una correccion de datos puntual va sin el flag
aunque mueva alguna decision: queda registrada como marca, sin fragmentar la
historia en pedazos que nunca llegan a la muestra minima.

FECHA EFECTIVA: la primera rueda de DATOS con la que decide la estrategia ya con
el cambio. Los bots deciden con el ultimo cierre cargado: si el cambio entra
antes de la proxima corrida y esa corrida usa el cierre de hoy, es la rueda de
hoy. NO es la fecha del commit ni la de la corrida.

Uso:
    python scripts/forward_testing/ft_cambios.py list
    python scripts/forward_testing/ft_cambios.py add --clave ml_modelo_v4 \\
        --fecha-efectiva 2026-09-15 --tipo MODELO --estrategias 1 \\
        --cambia-decisiones --titulo "Modelo calibrado v4" \\
        [--detalle "..."] [--ref "commit abc1234"] [--dry-run]
"""

import sys
import os
import re
import argparse
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.forward_testing.ft_env import configurar_entorno_local  # noqa: E402
configurar_entorno_local()

from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402
from src.utils.ft_tramos import TIPOS_CAMBIO  # noqa: E402
from src.utils.trading_calendar import (  # noqa: E402
    is_trading_day, prev_trading_day, next_trading_day,
)

RE_CLAVE = re.compile(r"^[a-z0-9_]{3,60}$")


def _tabla_existe(conn):
    return conn.execute(text("SELECT to_regclass('public.ft_cambios')")).scalar() is not None


def _nombres(conn):
    rows = conn.execute(text("SELECT id, nombre, activa FROM ft_estrategias ORDER BY id")).fetchall()
    return {r.id: (r.nombre.replace("FT_", ""), r.activa) for r in rows}


def _lista(ids, nombres):
    activas = {i for i, (_, act) in nombres.items() if act}
    if activas and set(ids) >= activas:
        return "todas"
    return ", ".join(nombres.get(i, (str(i), None))[0] for i in ids)


def cmd_list(engine):
    with engine.connect() as conn:
        if not _tabla_existe(conn):
            print("ft_cambios no existe. Crearla con scripts/oneshot/create_ft_cambios.py --apply")
            return 1
        nombres = _nombres(conn)
        rows = conn.execute(text("""
            SELECT clave, fecha_efectiva, tipo, estrategias, cambia_decisiones, titulo
            FROM ft_cambios ORDER BY fecha_efectiva, id
        """)).fetchall()

    if not rows:
        print("ft_cambios esta vacia.")
        return 0
    print(f"{'fecha':<10}  {'corta':<5}  {'tipo':<9}  {'clave':<34}  estrategias / titulo")
    print("-" * 110)
    for r in rows:
        ests = _lista(r.estrategias, nombres)
        print(f"{r.fecha_efectiva}  {'SI' if r.cambia_decisiones else '-':<5}  "
              f"{r.tipo:<9}  {r.clave:<34}  {r.titulo}")
        print(f"{'':<66}[{ests}]")
    print(f"\n{len(rows)} cambios. 'corta' = cambia decisiones y parte la historia en tramos.")
    return 0


def cmd_add(engine, args):
    errores = []
    try:
        fecha = date.fromisoformat(args.fecha_efectiva)
    except ValueError:
        print(f"[ERROR] --fecha-efectiva invalida: {args.fecha_efectiva} (YYYY-MM-DD)")
        return 1
    if not is_trading_day(fecha):
        errores.append(f"{fecha} no es dia habil NYSE (anterior {prev_trading_day(fecha)}, "
                       f"siguiente {next_trading_day(fecha)}). La fecha efectiva es una rueda de datos.")
    if not RE_CLAVE.match(args.clave):
        errores.append("--clave: minusculas, digitos y guion bajo, 3 a 60 caracteres.")
    try:
        ids = sorted({int(x) for x in args.estrategias.split(",") if x.strip()})
    except ValueError:
        ids = []
    if not ids:
        errores.append("--estrategias: lista de ids separados por coma (ej. 1,4,9).")

    with engine.connect() as conn:
        if not _tabla_existe(conn):
            print("ft_cambios no existe. Crearla con scripts/oneshot/create_ft_cambios.py --apply")
            return 1
        nombres = _nombres(conn)
        faltan = [i for i in ids if i not in nombres]
        if faltan:
            errores.append(f"estrategias inexistentes en ft_estrategias: {faltan}")
        if conn.execute(text("SELECT 1 FROM ft_cambios WHERE clave = :c"),
                        {"c": args.clave}).scalar():
            errores.append(f"la clave '{args.clave}' ya existe (list para verla).")
        if errores:
            for e in errores:
                print(f"[ERROR] {e}")
            return 1

        # Aviso 1: ya hubo corridas con ese dato. Si no tenian el cambio, la
        # fecha efectiva esta mal y el primer dia del tramo nuevo mide la regla vieja.
        ultima = conn.execute(text("""
            SELECT MAX(COALESCE(fecha_datos, fecha)) FROM ft_posiciones_diarias
            WHERE estrategia_id = ANY(:ids)
        """), {"ids": ids}).scalar()
        if ultima is not None and ultima >= fecha:
            print(f"[AVISO] Ya hay corridas que decidieron con datos del {fecha} o posteriores "
                  f"(ultima rueda usada: {ultima}). Si esas corridas NO tenian el cambio, la fecha "
                  f"efectiva es la rueda siguiente a la ultima que corrio sin el.")

        # Aviso 2: sin estrategias afuera no hay contra que comparar.
        activas = {i for i, (_, act) in nombres.items() if act}
        if args.cambia_decisiones and activas <= set(ids):
            print("[AVISO] Afecta a todas las estrategias activas: el cambio no va a tener grupo "
                  "de control y solo se podra comparar contra el universo.")

        fila = {"clave": args.clave, "fecha": fecha, "tipo": args.tipo, "ids": ids,
                "cd": bool(args.cambia_decisiones), "titulo": args.titulo,
                "detalle": args.detalle, "ref": args.ref}
        ests = _lista(ids, nombres)
        print(f"{'[DRY-RUN] ' if args.dry_run else ''}{fecha}  {args.tipo}  "
              f"corta={'SI' if fila['cd'] else 'no'}  {args.clave}\n  {args.titulo}\n  [{ests}]")
        if args.dry_run:
            return 0

        nuevo = conn.execute(text("""
            INSERT INTO ft_cambios (clave, fecha_efectiva, tipo, estrategias,
                                    cambia_decisiones, titulo, detalle, ref)
            VALUES (:clave, :fecha, :tipo, :ids, :cd, :titulo, :detalle, :ref)
            ON CONFLICT (clave) DO NOTHING
            RETURNING id
        """), fila).scalar()
        if nuevo is None:
            print(f"[ERROR] La clave '{args.clave}' ya existe. No se modifico nada.")
            return 1
        conn.commit()
    print(f"Registrado (id {nuevo}).")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Registro de cambios de Forward Testing (ft_cambios).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="lista los cambios registrados")

    a = sub.add_parser("add", help="registra un cambio")
    a.add_argument("--clave", required=True, help="identificador unico (ej. ml_modelo_v4)")
    a.add_argument("--fecha-efectiva", required=True,
                   help="primera rueda de DATOS con la que decide la estrategia ya con el cambio")
    a.add_argument("--tipo", required=True, choices=TIPOS_CAMBIO)
    a.add_argument("--estrategias", required=True, help="ids afectados, separados por coma")
    a.add_argument("--cambia-decisiones", action="store_true",
                   help="cambia logica/parametros/modelo: corta tramos")
    a.add_argument("--titulo", required=True)
    a.add_argument("--detalle", default=None)
    a.add_argument("--ref", default=None, help="JOURNAL, commit o doc de referencia")
    a.add_argument("--dry-run", action="store_true", help="valida y muestra, sin escribir")
    args = ap.parse_args()

    engine = get_engine()
    if args.cmd == "list":
        return cmd_list(engine)
    return cmd_add(engine, args)


if __name__ == "__main__":
    sys.exit(main())
