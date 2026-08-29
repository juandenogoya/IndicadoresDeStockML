"""
valuacion_implicita.py -- escenarios de valuacion contra la historia propia.

Primer consumidor de la fuente SEC XBRL. Lee `fundamentales_sec_multiplos_d`
(LOCAL) y no escribe nada.

Es una CALCULADORA, no un predictor. No estima retornos ni dice si algo va a
subir: toma una tesis de precio y la traduce a las dos preguntas que si se
pueden contestar con datos.

  DIRECTA   "si el precio fuera X, que PER / P-S / EV-EBITDA implica, y donde
             cae eso dentro de la propia historia de la empresa?"
  INVERSA   "para que ese precio sea un multiplo NORMAL de esta empresa,
             cuanto tiene que crecer el negocio -- y alguna vez crecio asi?"

La segunda es la que convierte un numero en un juicio. Un +20% de precio mueve
todos los multiplos de precio exactamente +20%: eso es aritmetica, no
informacion. Lo que informa es contra que se compara ese multiplo, y que hace
falta que pase para llegar ahi.

La logica vive en src/utils/valuacion_implicita.py, que es PURO. Aca solo se
lee la DB y se imprime.

COMO LEER LA SALIDA, que es donde se puede errar feo:

  El percentil dice DONDE cae el multiplo dentro de su propio rango. No dice
  si ese rango esta justificado. Parte de el viene del REGIMEN DE TASAS y no
  de la empresa -- un PER de 20 en 2021 con la tasa en cero no significa lo
  mismo que un PER de 20 hoy. "Barato contra si misma" es una observacion, no
  una tesis.

  El crecimiento historico se mide sobre la serie TRIMESTRAL, no sobre la
  diaria: el TTM es una escalera que solo se mueve cuando sale un balance.

  La ventana es toda la historia disponible del ticker en la tabla, que
  arranca en 2021 y esta limitada por `precios_diarios`, no por SEC.

Uso:
    python scripts/manual/valuacion_implicita.py AAPL
    python scripts/manual/valuacion_implicita.py AAPL --precio 250
    python scripts/manual/valuacion_implicita.py AAPL --variacion 25
    python scripts/manual/valuacion_implicita.py AAPL --variacion 25 --referencia 75
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

# Que magnitud del negocio esta abajo de cada multiplo. Es lo que se nombra al
# decir "hace falta que las VENTAS crezcan tanto".
# La tercera forma va en genitivo ("del resultado neto") porque se incrusta
# en una frase; asi el verbo no tiene que concordar con magnitudes que son
# unas singulares y otras plurales.
DENOMINADOR = {
    "pe_ratio": ("net_income_ttm", "resultado neto", "del resultado neto"),
    "ps_ratio": ("revenue_ttm", "ventas", "de las ventas"),
    "pb_ratio": ("equity", "patrimonio", "del patrimonio"),
    "ev_ebitda": ("ebitda_ttm", "EBITDA", "del EBITDA"),
}

PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _conn(env):
    return psycopg2.connect(host=env["DB_HOST"], port=env["DB_PORT"],
                            user=env["DB_USER"], password=env["DB_PASSWORD"],
                            dbname=env["DB_NAME"])


CAMPOS = ["fecha", "close", "shares", "shares_dias", "net_debt", "revenue_ttm",
          "net_income_ttm", "ebitda_ttm", "fcf_ttm", "equity", "period_end",
          "filed_primero", "lag_dias"]

# Magnitudes cuya evolucion TRIMESTRAL hace falta para medir crecimiento.
TTM = ["revenue_ttm", "net_income_ttm", "ebitda_ttm", "equity"]


def leer(cur, ticker, desde):
    cur.execute("SELECT %s FROM %s WHERE ticker=%%s ORDER BY fecha DESC LIMIT 1"
                % (", ".join(CAMPOS), TABLA), (ticker,))
    fila = cur.fetchone()
    if not fila:
        return None, {}, {}
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

    # Serie TRIMESTRAL: un punto por balance, no por rueda. DISTINCT ON toma la
    # primera rueda en que cada period_end aparecio, que es cuando ese TTM se
    # hizo publico.
    sql = ("SELECT DISTINCT ON (period_end) period_end, %s FROM %s "
           "WHERE ticker=%%s AND period_end IS NOT NULL" % (", ".join(TTM), TABLA))
    params = [ticker]
    if desde:
        sql += " AND fecha >= %s"
        params.append(desde)
    sql += " ORDER BY period_end, fecha"
    cur.execute(sql, params)
    filas = cur.fetchall()
    trimestral = {c: [(f[0], f[i + 1]) for f in filas] for i, c in enumerate(TTM)}
    return base, historia, trimestral


def _n(v, dec=2):
    return ("%.*f" % (dec, v)) if v is not None else "n/d"


def _miles(v):
    """
    Magnitudes del negocio en MILLONES, con separador de miles.

    La fuente SEC guarda importes absolutos en unidades de moneda. Mostrarlos
    asi hace una columna de doce digitos que nadie lee; los millones son la
    escala en que estan escritos los balances de los que salieron.
    """
    if v is None:
        return "n/d"
    return ("{:,.0f}".format(float(v) / 1e6)).replace(",", ".")


def _pct(p):
    return ("%3.0f%%" % (p * 100)) if p is not None else " n/d"


def _var(x):
    """Variacion: lleva signo, porque lo que se lee es la direccion."""
    return ("%+.1f%%" % (x * 100)) if x is not None else "n/d"


def _niv(x):
    """Nivel (un ROE, un margen): SIN signo. Un '+119.9%' se lee como suba."""
    return ("%.1f%%" % (x * 100)) if x is not None else "n/d"


def _roes(trimestral):
    """Serie historica de ROE (resultado TTM / patrimonio) por trimestre."""
    salida = []
    for (_, ni), (_, eq) in zip(trimestral.get("net_income_ttm", []),
                                trimestral.get("equity", [])):
        if ni is None or eq is None or float(eq) <= 0:
            continue
        salida.append(float(ni) / float(eq))
    return sorted(salida)


def bloque_exigencia(base, historia, trimestral, precio, referencia):
    """La direccion inversa: que tiene que hacer el negocio."""
    ex = V.exigencia(base, historia, precio, referencia=referencia)

    print()
    print("  QUE TIENE QUE HACER EL NEGOCIO PARA QUE %s SEA UN PRECIO NORMAL"
          % _n(precio))
    print("    normal = ese multiplo de vuelta en el percentil %d de su propia"
          " historia" % round(referencia * 100))
    print()
    print("    %-11s %-15s %13s %13s %8s   %s"
          % ("", "tiene que", "hoy (MM)", "necesario", "crece",
             "su YoY historico"))

    imposibles = []
    for clave, etiqueta in VISTA:
        campo, nombre, articulado = DENOMINADOR[clave]
        d = ex[clave]
        crec = d["crecimiento"]

        hist = V.crecimiento_historico(trimestral.get(campo))
        if hist:
            med, mx = V.cuantil(hist, 0.50), hist[-1]
            resumen = "med %s  max %s  (%d obs)" % (_var(med), _var(mx), len(hist))
            if crec is not None and crec > mx:
                imposibles.append((etiqueta, articulado, crec, mx))
        else:
            resumen = "sin historia suficiente"

        print("    %-11s %-15s %13s %13s %8s   %s"
              % (etiqueta, nombre, _miles(d["actual"]), _miles(d["requerido"]),
                 _var(crec), resumen))

    # El resultado necesario, traducido a rentabilidad exigida.
    roe_req = V.roe_implicito(base, ex["pe_ratio"]["requerido"])
    roe_hoy = V.roe_implicito(base, base.get("net_income_ttm"))
    if roe_req is not None and roe_hoy is not None:
        roes = _roes(trimestral)
        extra = (" Su maximo historico fue %s." % _niv(roes[-1])) if roes else ""
        print()
        print("    Ese resultado exige un ROE de %s contra el patrimonio de "
              "hoy; ahora es %s.%s" % (_niv(roe_req), _niv(roe_hoy), extra))
        print("    Es un TECHO: si gana mas y no reparte todo el patrimonio "
              "crece, y el ROE necesario baja.")

    if imposibles:
        print()
        for etiqueta, articulado, crec, mx in imposibles:
            print("    %s: exige un crecimiento %s de %s, y su mejor anio "
                  "fue %s." % (etiqueta, articulado, _var(crec), _var(mx)))
        print("    No invalida la tesis. Dice que se apoya en algo que esta "
              "empresa todavia no hizo.")


def main():
    p = argparse.ArgumentParser(
        description="Escenarios de valuacion implicita sobre la fuente SEC")
    p.add_argument("ticker")
    p.add_argument("--precio", type=float, help="Precio objetivo de la tesis")
    p.add_argument("--variacion", type=float,
                   help="Variacion en %% sobre el cierre (ej: 25 o -15)")
    p.add_argument("--referencia", type=float, default=50.0,
                   help="Percentil que se toma como multiplo NORMAL (default 50)")
    p.add_argument("--desde", help="Recorta la historia (YYYY-MM-DD)")
    args = p.parse_args()

    ticker = args.ticker.upper()
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    with _conn(env) as cx:
        with cx.cursor() as cur:
            base, historia, trimestral = leer(cur, ticker, args.desde)

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

        bloque_exigencia(base, historia, trimestral, esc["precio"],
                         args.referencia / 100.0)

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
