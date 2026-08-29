"""
comparar_series_yahooquery.py -- yahooquery podria sostener esta capa?

La pregunta que cierra la eleccion de fuente. No "cual tiene mejores numeros"
sino: sobre que parte de la historia puede yahooquery siquiera OPINAR, y donde
opina, dice lo mismo que SEC?

Solo lee. No escribe.

Como se hace la comparacion JUSTA, que es todo el punto
------------------------------------------------------
yahooquery no publica fecha de presentacion: sus filas vienen con el CIERRE
fiscal y nada mas. Anclar por ahi meteria lookahead -- un trimestre cerrado el
30/9 no estaba disponible el 30/9 -- y le daria una ventaja artificial.

Por eso se le prestan a yahooquery las fechas `filed_primero` de SEC para los
MISMOS period_end. Asi las dos series se activan el mismo dia y la unica
diferencia que queda es EL DATO. Sin ese prestamo la comparacion mediria
timing, no calidad.

Por la misma razon se usan las acciones y la deuda neta de SEC para las dos:
lo que se compara son los denominadores TTM del estado de resultados.

Uso:
    python scripts/oneshot/comparar_series_yahooquery.py
"""

import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import psycopg2

from scripts.oneshot.create_fundamentales_tables import _parse_env_file

SEP = "=" * 76

# Ruedas que necesita el percentil "caro vs si misma". Es el numero que decide
# si una fuente puede sostener esta capa o no.
VENTANA_PCT = 756


def _conn(env):
    return psycopg2.connect(host=env["DB_HOST"], port=env["DB_PORT"],
                            user=env["DB_USER"], password=env["DB_PASSWORD"],
                            dbname=env["DB_NAME"])


def _f(v):
    return None if v is None else float(v)


def ttm_por_fecha(quarters, filed):
    """
    [(fecha_de_alta, {revenue,net_income,ebitda})] para una fuente.

    quarters: {period_end: {concepto: valor}}
    filed:    {period_end: fecha en que ese trimestre se hizo publico}

    El TTM de un period_end suma ESE trimestre y los 3 anteriores, y se activa
    en su fecha de publicacion. Si falta alguno de los 4, no hay TTM: sumar 3
    trimestres y llamarlo anual es el error que este proyecto viene evitando
    desde el principio.
    """
    ends = sorted(quarters)
    salida = []
    for i in range(3, len(ends)):
        ventana = ends[i - 3:i + 1]
        alta = filed.get(ends[i])
        if alta is None:
            continue
        acum = {}
        for concepto in ("revenue", "net_income", "ebitda"):
            vals = [quarters[e].get(concepto) for e in ventana]
            acum[concepto] = sum(vals) if all(v is not None for v in vals) else None
        salida.append((alta, acum))
    # Por la CLAVE: dos trimestres pueden compartir fecha de alta y los dicts
    # no se comparan entre si.
    return sorted(salida, key=lambda x: x[0])


def asof(serie, fecha):
    """Ultimo elemento con alta <= fecha. Serie chica: busqueda lineal."""
    elegido = None
    for alta, val in serie:
        if alta <= fecha:
            elegido = val
        else:
            break
    return elegido


def main():
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    with _conn(env) as cx:
        cur = cx.cursor()

        # SEC: trimestres + la fecha real de publicacion de cada uno.
        cur.execute("""SELECT ticker, period_end, filed_primero, revenue,
                              net_income, operating_income, d_and_a
                       FROM fundamentales_sec_q ORDER BY ticker, period_end""")
        sec_q = defaultdict(dict)
        filed = defaultdict(dict)
        for t, pe, fp, rev, ni, oi, da in cur.fetchall():
            ebitda = (_f(oi) + _f(da)) if (oi is not None and da is not None) else None
            sec_q[t][pe] = {"revenue": _f(rev), "net_income": _f(ni),
                            "ebitda": ebitda}
            if fp:
                filed[t][pe] = fp

        # yahooquery: los mismos conceptos, sin fecha de publicacion propia.
        cur.execute("""SELECT ticker, fiscal_period_end, total_revenue,
                              net_income, ebitda
                       FROM fundamentales_income_q
                       ORDER BY ticker, fiscal_period_end""")
        yq_q = defaultdict(dict)
        for t, pe, rev, ni, eb in cur.fetchall():
            yq_q[t][pe] = {"revenue": _f(rev), "net_income": _f(ni),
                           "ebitda": _f(eb)}

        cur.execute("""SELECT ticker, fecha, close, shares, net_debt
                       FROM fundamentales_sec_multiplos_d
                       ORDER BY ticker, fecha""")
        ruedas = defaultdict(list)
        for t, f, c, sh, nd in cur.fetchall():
            ruedas[t].append((f, _f(c), _f(sh), _f(nd)))

    print()
    print(SEP)
    print("  SEC vs yahooquery -- puede yahooquery sostener esta capa?")
    print(SEP)

    # ---------------------------------------------------------- cobertura --
    prof_sec, prof_yq = [], []
    for t in sec_q:
        prof_sec.append(len(sec_q[t]))
        prof_yq.append(len(yq_q.get(t, {})))
    prof_sec.sort(); prof_yq.sort()
    med = lambda xs: xs[len(xs) // 2] if xs else 0
    print()
    print("  PROFUNDIDAD (trimestres por ticker, mediana)")
    print("    SEC          %3d      yahooquery  %3d" % (med(prof_sec), med(prof_yq)))
    print("    Un TTM consume 4. El percentil 'caro vs si misma' necesita %d"
          % VENTANA_PCT)
    print("    ruedas de multiplos, o sea unos 16 trimestres de historia.")

    # ------------------------------------------------- acuerdo donde ambas --
    difs = defaultdict(list)
    ruedas_sec = ruedas_ambas = 0
    tickers_ambas = set()
    primera_yq = {}

    for t, filas in ruedas.items():
        s_ttm = ttm_por_fecha(sec_q.get(t, {}), filed.get(t, {}))
        y_ttm = ttm_por_fecha(yq_q.get(t, {}), filed.get(t, {}))
        if not s_ttm or not y_ttm:
            ruedas_sec += len(filas)
            continue
        for fecha, close, shares, nd in filas:
            ruedas_sec += 1
            s, y = asof(s_ttm, fecha), asof(y_ttm, fecha)
            if not s or not y or not close or not shares:
                continue
            mc = close * shares
            ev = (mc + nd) if nd is not None else None
            ruedas_ambas += 1
            tickers_ambas.add(t)
            primera_yq.setdefault(t, fecha)
            for etiqueta, campo, usa_ev in (("P/S", "revenue", False),
                                            ("PER", "net_income", False),
                                            ("EV/EBITDA", "ebitda", True)):
                num = ev if usa_ev else mc
                a, b = s.get(campo), y.get(campo)
                if not a or not b or a <= 0 or b <= 0 or num is None:
                    continue
                difs[etiqueta].append(abs((num / a) - (num / b)) / (num / b))

    print()
    print("  COBERTURA de la serie diaria")
    print("    ruedas con multiplo SEC:               %7d" % ruedas_sec)
    print("    ruedas donde yahooquery TAMBIEN puede: %7d   (%.1f%%)"
          % (ruedas_ambas, 100.0 * ruedas_ambas / max(ruedas_sec, 1)))
    print("    tickers donde yahooquery llega a un TTM: %d de %d"
          % (len(tickers_ambas), len(ruedas)))
    if primera_yq:
        print("    la serie de yahooquery arranca, en el mejor caso, el %s"
              % min(primera_yq.values()))

    print()
    print("  ACUERDO donde las DOS pueden opinar")
    print("    (mismo dia, mismas acciones y deuda: la unica diferencia es el TTM)")
    print("    %-12s %8s %10s %10s %12s" % ("", "n", "mediana", "p90", "dentro 5%"))
    for etiqueta in ("P/S", "PER", "EV/EBITDA"):
        xs = sorted(difs[etiqueta])
        if not xs:
            print("    %-12s %8s" % (etiqueta, "n/d"))
            continue
        q = lambda p: xs[min(int(len(xs) * p), len(xs) - 1)]
        dentro = sum(1 for x in xs if x <= 0.05) / len(xs) * 100
        print("    %-12s %8d %9.2f%% %9.2f%% %11.0f%%"
              % (etiqueta, len(xs), q(.50) * 100, q(.90) * 100, dentro))

    print()
    print(SEP)
    print("  Lectura: el acuerdo mide si dicen lo mismo DONDE LAS DOS HABLAN.")
    print("  La cobertura mide sobre que parte de la historia puede hablar")
    print("  yahooquery -- y es ahi, y no en el acuerdo, donde se decide si una")
    print("  fuente puede sostener el percentil 'caro vs si misma'.")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
