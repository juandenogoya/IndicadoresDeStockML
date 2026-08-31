"""
sec_avisos.py -- lector de fundamentales_sec_avisos.

Por que existe
--------------
El normalizador de SEC viene emitiendo avisos desde la Fase 2 y la tabla
`fundamentales_sec_avisos` no tenia UN SOLO lector en todo el repo: aparecia
en el create_* que la define y en el refresh que la escribe, y nada mas.

El costo de eso fue concreto. El defecto de revenue -- dos tags XBRL que valen
cosas distintas mezclados dentro del mismo ejercicio, que dejo la
reconciliacion contra el anual publicado en 87,8% -- YA estaba senalado por el
aviso `cambio_de_tag` en los 20 peores tickers. Estuvo marcado todo el tiempo.
Nadie lo miro. Una red de seguridad que nadie consulta no es una red de
seguridad.

Que mirar primero
-----------------
Los avisos NO son todos iguales y ordenarlos por cantidad enganya. La
severidad la fija el modulo, no el volumen:

  mezcla_en_ejercicio      DEFECTO. Los trimestres de un ejercicio salieron de
                           tags distintos Y sus anuales difieren: la suma de
                           los 4 no es el anual de nada, y el TTM tampoco.
                           Se cura en src/data/sec/tags_curados.py.
  net_income_sin_...       HUECO DECLARADO. El modulo se abstuvo en vez de
  ponderado_implausible    inventar. No hay dato malo; hay dato faltante.
  mezcla_no_verificable    SOSPECHA. Mezcla de tags que no se puede comprobar
                           porque un solo tag publica anual.
  cambio_de_tag            INFORMATIVO. Casi siempre es una migracion de
  ponderado_*              taxonomia legitima. Ruidoso por diseno.

Uso:
    python scripts/manual/sec_avisos.py                 # resumen por tipo
    python scripts/manual/sec_avisos.py --defectos      # solo lo accionable
    python scripts/manual/sec_avisos.py --ticker JPM
    python scripts/manual/sec_avisos.py --tipo mezcla_en_ejercicio --detalle
    python scripts/manual/sec_avisos.py --defectos --alertar   # Telegram
"""

import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import psycopg2

from scripts.oneshot.create_fundamentales_tables import _parse_env_file

SEP = "=" * 74
TABLA = "fundamentales_sec_avisos"

# Severidad por tipo. Lo que no este aca se trata como informativo: un tipo
# nuevo no deberia hacerse pasar por defecto sin que alguien lo clasifique.
DEFECTO = "DEFECTO"
HUECO = "HUECO"
SOSPECHA = "SOSPECHA"
INFO = "info"

SEVERIDAD = {
    "mezcla_en_ejercicio": DEFECTO,
    "net_income_sin_minoritarios": HUECO,
    "net_income_derivado_sin_control": HUECO,
    "ponderado_implausible": HUECO,
    "mezcla_no_verificable": SOSPECHA,
    "ponderado_discordante": SOSPECHA,
    "ponderado_sin_control": INFO,
    "cambio_de_tag": INFO,
}
ORDEN = {DEFECTO: 0, HUECO: 1, SOSPECHA: 2, INFO: 3}


def _conn(env):
    return psycopg2.connect(host=env["DB_HOST"], port=env["DB_PORT"],
                            user=env["DB_USER"], password=env["DB_PASSWORD"],
                            dbname=env["DB_NAME"])


def leer(cur, ticker=None, tipo=None, solo_defectos=False):
    sql = (f"SELECT ticker, tipo, concepto, period_end, detalle FROM {TABLA} "
           f"WHERE 1=1")
    params = []
    if ticker:
        sql += " AND ticker = ANY(%s)"
        params.append(ticker)
    if tipo:
        sql += " AND tipo = %s"
        params.append(tipo)
    if solo_defectos:
        sql += " AND tipo = ANY(%s)"
        params.append([t for t, s in SEVERIDAD.items() if s == DEFECTO])
    sql += " ORDER BY ticker, tipo, concepto, period_end"
    cur.execute(sql, params)
    return cur.fetchall()


def resumen(filas):
    """(severidad, tipo, concepto) -> [avisos, tickers distintos]."""
    agg = defaultdict(lambda: [0, set()])
    for tk, tipo, concepto, _, _ in filas:
        clave = (SEVERIDAD.get(tipo, INFO), tipo, concepto or "")
        agg[clave][0] += 1
        agg[clave][1].add(tk)
    return agg


def imprimir_resumen(agg):
    print("%-9s %-32s %-18s %7s %8s"
          % ("severidad", "tipo", "concepto", "avisos", "tickers"))
    print("-" * 74)
    sev_previa = None
    for (sev, tipo, concepto), (n, tks) in sorted(
            agg.items(), key=lambda kv: (ORDEN[kv[0][0]], -kv[1][0])):
        if sev_previa is not None and sev != sev_previa:
            print()
        sev_previa = sev
        print("%-9s %-32s %-18s %7d %8d" % (sev, tipo, concepto, n, len(tks)))


def imprimir_detalle(filas, limite):
    por_ticker = defaultdict(list)
    for tk, tipo, concepto, period_end, detalle in filas:
        por_ticker[tk].append((tipo, concepto, period_end, detalle))
    for tk in sorted(por_ticker):
        print()
        print("  %s" % tk)
        for tipo, concepto, period_end, detalle in por_ticker[tk][:limite]:
            sev = SEVERIDAD.get(tipo, INFO)
            print("    [%s] %s / %s%s"
                  % (sev, tipo, concepto or "-",
                     "  (%s)" % period_end if period_end else ""))
            if detalle:
                print("        %s" % detalle[:150])
        sobran = len(por_ticker[tk]) - limite
        if sobran > 0:
            print("    ... y %d mas" % sobran)


def alertar(agg):
    """
    Avisa por Telegram SOLO de los defectos. Un canal que se usa para lo
    informativo deja de leerse, que es como esta tabla llego a no tener
    lectores.
    """
    defectos = {k: v for k, v in agg.items() if k[0] == DEFECTO}
    if not defectos:
        print("  sin defectos: no se envia alerta")
        return
    lineas = ["*SEC XBRL -- defectos en la normalizacion*", ""]
    for (_, tipo, concepto), (n, tks) in sorted(defectos.items()):
        lineas.append("- `%s` / %s: %d avisos en %d tickers"
                      % (tipo, concepto or "-", n, len(tks)))
        lineas.append("  " + ", ".join(sorted(tks)[:12]))
    lineas.append("")
    lineas.append("Curar en `src/data/sec/tags_curados.py`. Diagnostico: "
                  "`python scripts/oneshot/revenue_tags_reporte.py`")
    try:
        from src.pipeline.telegram_notifier import _send
        _send("\n".join(lineas))
        print("  alerta enviada")
    except Exception as e:
        print("  ERROR enviando alerta (no critico): %s" % str(e)[:120])


def main():
    p = argparse.ArgumentParser(description="Lector de " + TABLA)
    p.add_argument("--ticker", help="CSV de tickers")
    p.add_argument("--tipo", help="Filtra por un tipo de aviso")
    p.add_argument("--defectos", action="store_true",
                   help="Solo los avisos de severidad DEFECTO")
    p.add_argument("--detalle", action="store_true",
                   help="Lista los avisos uno por uno, agrupados por ticker")
    p.add_argument("--limite", type=int, default=6,
                   help="Avisos por ticker en --detalle (default 6)")
    p.add_argument("--alertar", action="store_true",
                   help="Manda los DEFECTOS por Telegram")
    args = p.parse_args()

    env = _parse_env_file(os.path.join(ROOT, ".env"))
    tickers = ([t.strip().upper() for t in args.ticker.split(",")]
               if args.ticker else None)

    with _conn(env) as cx:
        with cx.cursor() as cur:
            filas = leer(cur, tickers, args.tipo, args.defectos)

    print()
    print(SEP)
    print("  AVISOS DE LA FUENTE SEC XBRL")
    print(SEP)
    print()

    if not filas:
        print("  sin avisos para ese filtro")
        print()
        return

    agg = resumen(filas)
    imprimir_resumen(agg)

    if args.detalle or args.ticker or args.tipo:
        imprimir_detalle(filas, args.limite)

    print()
    print(SEP)
    n_def = sum(n for (sev, _, _), (n, _) in agg.items() if sev == DEFECTO)
    print("  total: %d avisos  |  DEFECTOS: %d" % (len(filas), n_def))
    if n_def and not args.defectos:
        print("  ver solo lo accionable:  "
              "python scripts/manual/sec_avisos.py --defectos --detalle")
    print(SEP)
    print()

    if args.alertar:
        alertar(agg)


if __name__ == "__main__":
    main()
