"""
refresh_polygon.py -- ingesta de Polygon.io (splits, acciones, cobertura).

LOCAL-only. REANUDABLE. A 4 pedidos por minuto una pasada completa son horas,
asi que todo el diseno gira alrededor de poder matarlo y volver a arrancarlo
sin perder lo hecho: cada (ticker, tarea) que termina queda anotado en
`polygon_ingesta` y la corrida siguiente lo saltea.

TAREAS
  --cobertura  existe el ticker en Polygon? Responde la pregunta de los 53 del
               universo que no estan en SEC (ADRs extranjeros: no presentan
               10-Q con XBRL, asi que companyfacts no los tiene y nunca los va
               a tener). 1 pedido por ticker.
  --splits     historial de splits desde --desde. 1 pedido por ticker.
  --acciones   conteo point-in-time en los cierres de trimestre. 1 pedido por
               (ticker, fecha), que es lo caro. Por eso NO pide toda la
               historia: pide desde --desde hasta un ano despues del primer
               punto que ya tenemos de yahooquery. Ese ano de mas no es
               desperdicio, es el SOLAPAMIENTO que permite validar la serie
               antes de usarla (regla 4 de docs/arquitectura_fuentes.md).
  --todo       las tres, en ese orden.

Nada de esto TOCA todavia acciones_circulacion ni los multiplos: son tablas
crudas de una fuente nueva, en paralelo, igual que se hizo con SEC. Mezclarlas
es un paso aparte y con validacion propia.

Uso:
    python scripts/refresh_polygon.py --todo
    python scripts/refresh_polygon.py --splits --forzar
    python scripts/refresh_polygon.py --status
"""

import argparse
import datetime as dt
import os
import sys
import time

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.polygon.client import Cliente, SinCredencial, detalles, splits

SEP = "=" * 64
DESDE_DEFECTO = "2021-01-01"
# Cuanto solapamiento pedir mas alla del primer punto que ya tenemos. Un ano
# son ~4 cierres de trimestre: suficiente para que validar_base tenga con que
# comparar, y barato en pedidos.
DIAS_SOLAPE = 400


def log(msg):
    print("[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def _conn():
    load_dotenv(os.path.join(ROOT, ".env"))
    os.environ.pop("DATABASE_URL", None)          # LOCAL, nunca Railway
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"))


def universo(cx, tickers=None):
    if tickers:
        return tickers
    with cx.cursor() as cur:
        cur.execute("SELECT ticker FROM activos WHERE activo ORDER BY ticker")
        return [r[0] for r in cur.fetchall()]


def hechas(cx, tarea):
    with cx.cursor() as cur:
        cur.execute("SELECT ticker FROM polygon_ingesta WHERE tarea=%s", (tarea,))
        return {r[0] for r in cur.fetchall()}


def anotar(cx, ticker, tarea, existe, n, detalle=""):
    with cx.cursor() as cur:
        cur.execute(
            "INSERT INTO polygon_ingesta (ticker,tarea,existe,n_filas,detalle,fetched_at) "
            "VALUES (%s,%s,%s,%s,%s,NOW()) "
            "ON CONFLICT (ticker,tarea) DO UPDATE SET "
            "  existe=EXCLUDED.existe, n_filas=EXCLUDED.n_filas, "
            "  detalle=EXCLUDED.detalle, fetched_at=NOW()",
            (ticker, tarea, existe, n, (detalle or "")[:400]))
    cx.commit()


def _guardar_acciones(cx, ticker, fecha, d):
    with cx.cursor() as cur:
        cur.execute(
            "INSERT INTO polygon_acciones "
            "  (ticker,fecha,share_class_shares,weighted_shares,market_cap,fetched_at) "
            "VALUES (%s,%s,%s,%s,%s,NOW()) "
            "ON CONFLICT (ticker,fecha) DO UPDATE SET "
            "  share_class_shares=EXCLUDED.share_class_shares, "
            "  weighted_shares=EXCLUDED.weighted_shares, "
            "  market_cap=EXCLUDED.market_cap, fetched_at=NOW()",
            (ticker, fecha, d.get("share_class_shares_outstanding"),
             d.get("weighted_shares_outstanding"), d.get("market_cap")))
    cx.commit()


# ------------------------------------------------------------ cobertura --
def tarea_cobertura(cx, cli, lista, forzar):
    ya = set() if forzar else hechas(cx, "cobertura")
    pend = [t for t in lista if t not in ya]
    log("cobertura: %d pendientes de %d" % (len(pend), len(lista)))
    hay = falta = 0
    for i, tk in enumerate(pend, 1):
        try:
            d = detalles(cli, tk)
        except Exception as e:
            log("  [!] %s: %s" % (tk, str(e)[:100]))
            continue
        if d is None:
            anotar(cx, tk, "cobertura", False, 0, "404")
            falta += 1
        else:
            hay += 1
            det = "%s | %s | cik=%s | %s" % ((d.get("name") or "")[:60],
                                             d.get("primary_exchange"),
                                             d.get("cik"), d.get("type"))
            anotar(cx, tk, "cobertura", True, 1, det)
            _guardar_acciones(cx, tk, dt.date.today().isoformat(), d)
        if i % 10 == 0 or i == len(pend):
            log("  cobertura %d/%d  (existen %d, faltan %d)" % (i, len(pend), hay, falta))
    return hay, falta


# --------------------------------------------------------------- splits --
def tarea_splits(cx, cli, lista, desde, forzar):
    ya = set() if forzar else hechas(cx, "splits")
    pend = [t for t in lista if t not in ya]
    log("splits: %d pendientes de %d" % (len(pend), len(lista)))
    total = 0
    for i, tk in enumerate(pend, 1):
        try:
            res = splits(cli, tk, desde=desde)
        except Exception as e:
            log("  [!] %s: %s" % (tk, str(e)[:100]))
            continue
        filas = []
        for s in res:
            desde_n = float(s.get("split_from") or 0)
            hasta_n = float(s.get("split_to") or 0)
            if not desde_n:
                continue
            filas.append({"ticker": tk, "execution_date": s["execution_date"],
                          "split_from": desde_n, "split_to": hasta_n,
                          "ratio": hasta_n / desde_n, "polygon_id": s.get("id")})
        if filas:
            with cx.cursor() as cur:
                psycopg2.extras.execute_batch(cur,
                    "INSERT INTO polygon_splits "
                    "  (ticker,execution_date,split_from,split_to,ratio,polygon_id,fetched_at) "
                    "VALUES (%(ticker)s,%(execution_date)s,%(split_from)s,%(split_to)s,"
                    "        %(ratio)s,%(polygon_id)s,NOW()) "
                    "ON CONFLICT (ticker,execution_date) DO UPDATE SET "
                    "  split_from=EXCLUDED.split_from, split_to=EXCLUDED.split_to, "
                    "  ratio=EXCLUDED.ratio, fetched_at=NOW()", filas)
            cx.commit()
        total += len(filas)
        anotar(cx, tk, "splits", True, len(filas))
        if i % 10 == 0 or i == len(pend):
            log("  splits %d/%d  (%d splits acumulados)" % (i, len(pend), total))
    return total


# ------------------------------------------------------------- acciones --
def cierres_trimestre(desde, hasta):
    """Cierres de trimestre calendario dentro de [desde, hasta]."""
    out = []
    for anio in range(int(desde[:4]), int(hasta[:4]) + 1):
        for mes, dia in ((3, 31), (6, 30), (9, 30), (12, 31)):
            f = "%04d-%02d-%02d" % (anio, mes, dia)
            if desde <= f <= hasta:
                out.append(f)
    return out


def fechas_pedidas(cx, ticker, desde):
    """
    Que fechas pedir para ESTE ticker.

    Hasta un ano despues del primer punto que ya tenemos de yahooquery: lo
    anterior es el hueco a llenar, y ese ano de mas es el solapamiento con el
    que se valida. Pedir la serie entera cuadruplicaria el costo en pedidos
    sin agregar nada que no se pueda validar igual.
    """
    with cx.cursor() as cur:
        cur.execute("SELECT MIN(fecha) FROM acciones_circulacion "
                    "WHERE ticker=%s AND fuente='yahooquery'", (ticker,))
        row = cur.fetchone()
    primero = row[0] if row and row[0] else None
    hoy = dt.date.today()
    tope = min((primero + dt.timedelta(days=DIAS_SOLAPE)) if primero else hoy, hoy)
    return cierres_trimestre(desde, tope.isoformat())


def tarea_acciones(cx, cli, lista, desde, forzar):
    ya = set() if forzar else hechas(cx, "acciones")
    pend = [t for t in lista if t not in ya]
    plan = {tk: fechas_pedidas(cx, tk, desde) for tk in pend}
    n_ped = sum(len(v) for v in plan.values())
    log("acciones: %d tickers pendientes | hasta %d pedidos (~%.0f min a 4/min)"
        % (len(pend), n_ped, n_ped / 4.0))
    hechos = 0
    for i, tk in enumerate(sorted(plan), 1):
        with cx.cursor() as cur:
            cur.execute("SELECT fecha FROM polygon_acciones WHERE ticker=%s", (tk,))
            tengo = {r[0].isoformat() for r in cur.fetchall()}
        con = 0
        for f in plan[tk]:
            if f in tengo:
                continue
            try:
                d = detalles(cli, tk, fecha=f)
            except Exception as e:
                log("  [!] %s %s: %s" % (tk, f, str(e)[:80]))
                continue
            hechos += 1
            if d:
                _guardar_acciones(cx, tk, f, d)
                con += 1
        anotar(cx, tk, "acciones", con > 0, con)
        log("  acciones %d/%d  %-6s %d puntos nuevos  (pedidos %d/%d)"
            % (i, len(plan), tk, con, hechos, n_ped))
    return hechos


# ----------------------------------------------------------------- main --
def estado(cx):
    with cx.cursor() as cur:
        for q, t in (("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM polygon_splits", "splits"),
                     ("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM polygon_acciones", "acciones")):
            cur.execute(q)
            n, tk = cur.fetchone()
            print("  %-10s %6d filas  %3d tickers" % (t, n, tk))
        cur.execute("SELECT tarea, COUNT(*), COUNT(*) FILTER (WHERE existe) "
                    "FROM polygon_ingesta GROUP BY tarea ORDER BY tarea")
        print()
        for tarea, n, ex in cur.fetchall():
            print("  ingesta %-10s %3d tickers hechos  (%s existen en Polygon)"
                  % (tarea, n, ex))
        cur.execute("SELECT ticker FROM polygon_ingesta "
                    "WHERE tarea='cobertura' AND NOT existe ORDER BY ticker")
        faltan = [r[0] for r in cur.fetchall()]
        if faltan:
            print()
            print("  NO estan en Polygon (%d): %s" % (len(faltan), " ".join(faltan)))


def main():
    p = argparse.ArgumentParser(description="Ingesta Polygon.io (LOCAL-only)")
    p.add_argument("--cobertura", action="store_true")
    p.add_argument("--splits", action="store_true")
    p.add_argument("--acciones", action="store_true")
    p.add_argument("--todo", action="store_true", help="las tres tareas")
    p.add_argument("--status", action="store_true")
    p.add_argument("--tickers", help="CSV; default: universo activo")
    p.add_argument("--desde", default=DESDE_DEFECTO)
    p.add_argument("--forzar", action="store_true", help="ignora el log de ingesta")
    args = p.parse_args()

    cx = _conn()
    if args.status:
        print(); print(SEP); print("  ESTADO POLYGON"); print(SEP); print()
        estado(cx); print()
        return 0

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    lista = universo(cx, tickers)
    try:
        cli = Cliente()
    except SinCredencial as e:
        print("  %s" % e)
        return 1

    t0 = time.time()
    print(); print(SEP)
    print("  REFRESH POLYGON  |  %d tickers" % len(lista))
    print(SEP); print()

    if args.todo or args.cobertura:
        hay, falta = tarea_cobertura(cx, cli, lista, args.forzar)
        log("cobertura lista: %d existen, %d no" % (hay, falta))
    if args.todo or args.splits:
        n = tarea_splits(cx, cli, lista, args.desde, args.forzar)
        log("splits listos: %d filas" % n)
    if args.todo or args.acciones:
        n = tarea_acciones(cx, cli, lista, args.desde, args.forzar)
        log("acciones listas: %d pedidos" % n)

    print(); print(SEP)
    print("  OK  |  %d pedidos  |  %d x 429  |  %.0f min (%.0f esperando cupo)"
          % (cli.n_pedidos, cli.n_429, (time.time() - t0) / 60,
             cli.caudal.espera_total / 60))
    print(SEP); print()
    estado(cx)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
