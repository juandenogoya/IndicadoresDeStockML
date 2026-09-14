"""
chequeo_rutina.py
Verifica que la rutina diaria manual haya quedado COHERENTE antes de que algo
opere con esos datos.

QUE PROBLEMA RESUELVE
    La rutina es una secuencia de 5 pasos (desde el 13/9/2026 la corre entera
    scripts/manual/rutina_diaria.bat; los .bat de cada paso siguen existiendo):

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

    NO frena por HUECOS en el medio de la serie (Etapa 2, 13/9/2026): los
    informa. Un hueco no mueve ningun MAX(fecha), asi que ni la mezcla ni el
    recovery lo ven (2026-08-28 estuvo dos semanas sin precio para 157
    tickers). No hay herramienta que los rellene sola: frenar sin poder
    arreglar dejaria a los bots sin correr. Relleno manual:
    docs/checklist_recovery_manual.md, CASO E.

    El razonamiento vive en src/utils/estado_pipeline.py (modulo PURO,
    compartido con la banda de frescura del dashboard) para que no haya dos
    definiciones de "estan alineados los datos".

Uso:
    python scripts/manual/chequeo_rutina.py
    python scripts/manual/chequeo_rutina.py --solo-avisar   (nunca falla)
    python scripts/manual/chequeo_rutina.py --solo-huecos   (solo huecos; nunca falla)

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
from src.utils.trading_calendar import (                     # noqa: E402
    prev_trading_day, is_trading_day,
)
from src.utils.estado_pipeline import (                       # noqa: E402
    TABLAS, diagnosticar, resumen, _a_fecha,
    TABLAS_HUECOS, RUEDAS_VENTANA_HUECOS, huecos_intermedios, resumen_huecos,
)
from src.utils.rutina import PASOS, fmt_duracion             # noqa: E402

SEP = "=" * 68


def _tablas_existentes(nombres):
    df = query_df(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name = ANY(:n)",
        {"n": list(nombres)},
    )
    return set(df["table_name"]) if not df.empty else set()


def _limpio(v):
    # pandas convierte los NULL en NaT, que NO es None y ademas es instancia de
    # datetime -- pasa cualquier chequeo `is not None` y revienta al formatear.
    # Se normaliza aca, que es la frontera: estado_pipeline es puro y no tiene
    # por que conocer los tipos de pandas.
    return None if pd.isna(v) else v


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

    fechas = {r["tabla"]: _limpio(r["fecha"]) for _, r in df.iterrows()}
    registros = {r["tabla"]: _limpio(r["registro"]) for _, r in df.iterrows()}
    return fechas, registros


# ── Huecos en el medio de la serie ────────────────────────────────────────────

def ruedas_ventana(hasta, n=RUEDAS_VENTANA_HUECOS):
    """Las ultimas `n` ruedas NYSE hasta `hasta` inclusive, en orden, del
    calendario del proyecto (nunca deducidas)."""
    d = hasta if is_trading_day(hasta) else prev_trading_day(hasta)
    out = [d]
    while len(out) < n:
        d = prev_trading_day(d)
        out.append(d)
    return out[::-1]


def leer_huecos(tablas=TABLAS_HUECOS, hasta=None):
    """{tabla: resultado de huecos_intermedios} para los tickers activos."""
    ruedas = ruedas_ventana(hasta or date.today())
    existentes = _tablas_existentes(tablas)
    out = {}
    for tabla in tablas:
        if tabla not in existentes:
            continue
        df = query_df(
            f"SELECT t.ticker, t.fecha FROM {tabla} t "
            f"JOIN activos a ON a.ticker = t.ticker AND a.activo = TRUE "
            f"WHERE t.fecha >= :desde", {"desde": ruedas[0]})
        ini = query_df(f"SELECT ticker, MIN(fecha) AS inicio FROM {tabla} GROUP BY ticker")
        fechas = {}
        for tk, f in zip(df["ticker"], df["fecha"]):
            fechas.setdefault(tk, set()).add(_a_fecha(f))
        inicios = {tk: _a_fecha(f) for tk, f in zip(ini["ticker"], ini["inicio"])}
        out[tabla] = huecos_intermedios(fechas, ruedas, inicios=inicios)
    return out


def lineas_huecos(huecos):
    out = [f"  Huecos en el medio de la serie (ultimas {RUEDAS_VENTANA_HUECOS} ruedas; "
           f"se avisa, no frena):"]
    if not huecos:
        out.append("    no se pudo revisar ninguna tabla")
        return out
    conocidos = set()
    for tabla, h in huecos.items():
        if h["n"]:
            out.append(f"   !{tabla}: {h['n']} {'faltante' if h['n'] == 1 else 'faltantes'}")
            out += [f"      {x}" for x in resumen_huecos(h)]
        else:
            out.append(f"    {tabla}: sin huecos")
        conocidos.update(h["conocidos"])
    for tk, f, motivo in sorted(conocidos):
        out.append(f"    (conocido, no cuenta) {tk} {f}: {motivo}")
    if any(h["n"] for h in huecos.values()):
        out.append("    Relleno manual: docs/checklist_recovery_manual.md, CASO E.")
    return out


# ── Ultimas corridas (rutina_corridas) ────────────────────────────────────────

def leer_ultimas_corridas():
    """La ultima corrida de cada paso, o None si la tabla todavia no existe."""
    if "rutina_corridas" not in _tablas_existentes(["rutina_corridas"]):
        return None
    df = query_df(
        "SELECT DISTINCT ON (paso) paso, origen, inicio, resultado, duracion_s, rueda_despues "
        "FROM rutina_corridas ORDER BY paso, inicio DESC")
    return [{k: _limpio(v) for k, v in r.items()} for r in df.to_dict("records")]


def lineas_corridas(corridas):
    out = ["  Ultima corrida de cada paso (rutina_corridas):"]
    if corridas is None:
        out.append("    la tabla rutina_corridas no existe todavia")
        return out
    if not corridas:
        out.append("    sin corridas registradas")
        return out
    orden = {p.clave: i for i, p in enumerate(PASOS)}
    for c in sorted(corridas, key=lambda c: orden.get(c["paso"], 99)):
        inicio = pd.Timestamp(c["inicio"]).strftime("%Y-%m-%d %H:%M")
        marca = " " if c["resultado"] in ("OK", "PARCIAL") else "!"
        out.append(f"   {marca}{c['paso']:<21} {inicio}  {c['resultado']:<12} "
                   f"{fmt_duracion(c['duracion_s']):>8}  datos {c['rueda_despues'] or '-'}"
                   f"  ({c['origen']})")
    return out


# ── Salida ────────────────────────────────────────────────────────────────────

def imprimir(diag: dict, extras=None) -> None:
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
    for bloque in extras or []:
        print()
        for x in bloque:
            print(x)
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


def _extras():
    """Huecos y ultimas corridas. Informativos: si fallan, el guard sigue."""
    bloques = []
    for leer, formatear, nombre in ((leer_huecos, lineas_huecos, "huecos"),
                                    (leer_ultimas_corridas, lineas_corridas, "corridas")):
        try:
            bloques.append(formatear(leer()))
        except Exception as e:
            bloques.append([f"  [WARN] no se pudo leer {nombre}: {str(e).splitlines()[0][:120]}"])
    return bloques


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verifica la coherencia de la rutina diaria (LOCAL)")
    ap.add_argument("--solo-avisar", action="store_true",
                    help="imprime el diagnostico pero siempre sale con 0")
    ap.add_argument("--solo-huecos", action="store_true",
                    help="solo busca huecos en el medio de la serie; siempre sale con 0")
    args = ap.parse_args()

    if args.solo_huecos:
        for x in lineas_huecos(leer_huecos()):
            print(x)
        return 0

    fechas, registros = leer_fechas()
    if not fechas:
        print("[ERROR] No se pudo leer ninguna tabla. Revisar la DB local.")
        return 2

    diag = diagnosticar(fechas, esperado=prev_trading_day(date.today()),
                        registros=registros)
    imprimir(diag, extras=_extras())

    if diag["ancla"] is None:
        return 2
    if diag["mezcla"] and not args.solo_avisar:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
