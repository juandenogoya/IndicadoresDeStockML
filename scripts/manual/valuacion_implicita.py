"""
valuacion_implicita.py -- escenarios de valuacion contra la historia propia.

Primer consumidor de la fuente SEC XBRL. Lee `fundamentales_sec_multiplos_d`
(LOCAL) y no escribe nada.

Responde las dos direcciones de la misma pregunta:

  "si el precio fuera X, que PER / EV-EBITDA / P-S implica, y donde cae eso
   dentro de la propia historia de la empresa?"
  "que precio hace falta para que su PER vuelva a la mediana de su historia?"

La logica vive en src/utils/valuacion_implicita.py, que es PURO. Aca solo se
lee la DB y se imprime.

COMO LEER LA SALIDA, que es donde se puede errar feo:

  El percentil dice DONDE cae el multiplo dentro de su propio rango. No dice
  si ese rango esta justificado. Parte de el viene del REGIMEN DE TASAS y no
  de la empresa -- un PER de 20 en 2021 con la tasa en cero no significa lo
  mismo que un PER de 20 hoy. "Barato contra si misma" es una observacion, no
  una tesis.

  La ventana es toda la historia disponible del ticker en la tabla, que
  arranca en 2021 y esta limitada por `precios_diarios`, no por SEC.

Uso:
    python scripts/manual/valuacion_implicita.py AAPL
    python scripts/manual/valuacion_implicita.py AAPL --precio 250
    python scripts/manual/valuacion_implicita.py AAPL --variacion 25
    python scripts/manual/valuacion_implicita.py AAPL --desde 2023-01-01
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import psycopg2

from scripts.oneshot.create_fundamentales_tables import _parse_env_file
from src.utils import valuacion_implicita as V

SEP = "=" * 70
TABLA = "fundamentales_sec_multiplos_d"

# Se muestran estos cuatro. fcf_yield queda afuera de la vista por defecto: se
# lee al reves que los demas (mas alto = mas barato) y mezclarlo en la misma
# columna de percentiles invita a leerlo mal.
VISTA = [("pe_ratio", "PER"), ("ps_ratio", "P/S"),
         ("pb_ratio", "P/B"), ("ev_ebitda", "EV/EBITDA")]

PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _conn(env):
    return psycopg2.connect(host=env["DB_HOST"], port=env["DB_PORT"],
                            user=env["DB_USER"], password=env["DB_PASSWORD"],
                            dbname=env["DB_NAME"])


CAMPOS = ["fecha", "close", "shares", "shares_dias", "net_debt", "revenue_ttm",
          "net_income_ttm", "ebitda_ttm", "fcf_ttm", "equity", "period_end",
          "filed_primero", "lag_dias"]


def leer(cur, ticker, desde):
    cur.execute("SELECT %s FROM %s WHERE ticker=%%s ORDER BY fecha DESC LIMIT 1"
                % (", ".join(CAMPOS), TABLA), (ticker,))
    fila = cur.fetchone()
    if not fila:
        return None, {}
    base = dict(zip(CAMPOS, fila))

    metricas = [m for m, _ in VISTA]
    sql = ("SELECT %s FROM %s WHERE ticker=%%s" % (", ".join(metricas), TABLA))
    params = [ticker]
    if desde:
        sql += " AND fecha >= %s"
        params.append(desde)
    sql += " ORDER BY fecha"
    cur.execute(sql, params)
    filas = cur.fetchall()
    historia = {m: [f[i] for f in filas] for i, m in enumerate(metricas)}
    return base, historia


def _n(v, dec=2):
    return ("%.*f" % (dec, v)) if v is not None else "n/d"


def _pct(p):
    return ("%3.0f%%" % (p * 100)) if p is not None else " n/d"


def main():
    p = argparse.ArgumentParser(
        description="Escenarios de valuacion implicita sobre la fuente SEC")
    p.add_argument("ticker")
    p.add_argument("--precio", type=float, help="Precio objetivo de la tesis")
    p.add_argument("--variacion", type=float,
                   help="Variacion en %% sobre el cierre (ej: 25 o -15)")
    p.add_argument("--desde", help="Recorta la historia (YYYY-MM-DD)")
    args = p.parse_args()

    ticker = args.ticker.upper()
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    with _conn(env) as cx:
        with cx.cursor() as cur:
            base, historia = leer(cur, ticker, args.desde)

    if not base:
        print("\n  %s no esta en %s.\n  La fuente SEC cubre ~147 tickers de "
              "region USA: los ADR extranjeros presentan 20-F anual y no "
              "tienen XBRL trimestral.\n" % (ticker, TABLA))
        return 1

    n_ruedas = len(historia.get("pe_ratio") or [])
    print()
    print(SEP)
    print("  %s  --  cierre %s del %s" % (ticker, _n(base["close"]), base["fecha"]))
    print(SEP)
    print("  TTM del trimestre cerrado el %s, publico desde el %s (%s dias "
          "antes de la rueda)" % (base["period_end"], base["filed_primero"],
                                  base["lag_dias"]))
    print("  Historia: %d ruedas%s  |  conteo de acciones con %s dias de "
          "antiguedad" % (n_ruedas, " desde " + args.desde if args.desde else "",
                          base["shares_dias"]))
    print()

    hoy = V.escenario(base, historia)
    print("  DONDE ESTA HOY")
    print("    %-11s %10s   %s" % ("", "valor", "percentil de su historia"))
    for clave, etiqueta in VISTA:
        v, pc = hoy["multiplos"][clave], hoy["percentiles"][clave]
        barra = ""
        if pc is not None:
            lleno = int(round(pc * 20))
            barra = "  [" + "#" * lleno + "." * (20 - lleno) + "]"
        print("    %-11s %10s   %s%s" % (etiqueta, _n(v), _pct(pc), barra))

    print()
    print("  QUE PRECIO IMPLICA CADA PERCENTIL DE SU PROPIA HISTORIA")
    ref = V.precios_de_referencia(base, historia, PERCENTILES)
    print("    %-11s %s" % ("", "  ".join("%8s" % ("p%02d" % (q * 100))
                                          for q in PERCENTILES)))
    for clave, etiqueta in VISTA:
        celdas = ["%8s" % _n(ref[clave][q], 1) for q in PERCENTILES]
        print("    %-11s %s" % (etiqueta, "  ".join(celdas)))

    if args.precio is not None or args.variacion is not None:
        esc = V.escenario(base, historia, precio_objetivo=args.precio,
                          variacion=(args.variacion / 100.0
                                     if args.variacion is not None else None))
        print()
        print("  TESIS: precio %s  (%+.1f%% desde el cierre)"
              % (_n(esc["precio"]), (esc["variacion"] or 0) * 100))
        for clave, etiqueta in VISTA:
            v, pc = esc["multiplos"][clave], esc["percentiles"][clave]
            marca = ("   <-- nunca visto en su historia"
                     if clave in esc["fuera_de_rango"] else "")
            print("    %-11s implicito %10s   percentil %s%s"
                  % (etiqueta, _n(v), _pct(pc), marca))
        if esc["fuera_de_rango"]:
            print()
            print("    Un multiplo fuera de rango no invalida la tesis: dice "
                  "que para sostenerla")
            print("    hay que creer algo que todavia no paso, y eso merece "
                  "argumento aparte.")

    print()
    print(SEP)
    print("  El percentil dice DONDE cae el multiplo en su propio rango, no si")
    print("  ese rango esta justificado: parte viene del regimen de tasas y no")
    print("  de la empresa. Es una observacion, no una tesis.")
    print(SEP)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
