"""
revenue_tags_reporte.py
Cuadro de decision para CURAR el tag de revenue, ticker por ticker.

Por que existe
--------------
"Revenue" no es un renglon en XBRL. La SEC publica hechos sueltos y una misma
empresa puede tagear sus ventas con dos conceptos distintos que NO valen lo
mismo (bruto vs neto de intereses, con o sin impuestos internos, subtotal vs
total). El normalizador elige por prioridad de sinonimo, y eso puede resolver
distinto en los 10-Q que en el 10-K -- que es de donde sale el Q4. El
resultado es una MEZCLA DE TAGS dentro del mismo ejercicio.

Medido sobre los 147 tickers USA: la suma de los 4 trimestres reconcilia
contra el anual publicado en 99,6% de los casos para net_income y solo 87,8%
para revenue. No es un problema aritmetico y ningun algoritmo lo resuelve --
ver docs/fuentes_fundamentales.md seccion 15, donde estan las CINCO salidas
automaticas ya probadas y por que ninguna alcanza.

Lo que queda es una decision humana por ticker. Este script arma la evidencia
para tomarla: para cada ejercicio muestra cuanto da CADA tag candidato, que
tags usaron los 4 trimestres, y cuanto da yahooquery, que sirve de arbitro
independiente.

NO escribe nada. Solo lee el cache de SEC y la DB local.

Uso:
    python scripts/oneshot/revenue_tags_reporte.py
    python scripts/oneshot/revenue_tags_reporte.py --tickers JPM,PM,AMT
    python scripts/oneshot/revenue_tags_reporte.py --todos --csv salida.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import psycopg2

from scripts.oneshot.create_fundamentales_tables import _parse_env_file
from src.data.sec import client
from src.utils import sec_xbrl as X

SEP = "=" * 78

# Dos tags se consideran EL MISMO numero por debajo de esto. Es la linea que
# separa "la empresa tiene sinonimos redundantes" de "son dos renglones
# distintos y hay que elegir".
TOL = 0.01

# Desde donde mirar. Antes de 2019 casi todo el universo migro de
# SalesRevenueNet a RevenueFromContractWithCustomer* por ASC 606, y ese cambio
# de tag es LEGITIMO: ensuciaria el diagnostico sin aportar nada.
DESDE = "2019-01-01"

CACHE = os.path.join(ROOT, "data", "sec_cache")


def _conn(env):
    return psycopg2.connect(host=env["DB_HOST"], port=env["DB_PORT"],
                            user=env["DB_USER"], password=env["DB_PASSWORD"],
                            dbname=env["DB_NAME"])


def mm(v):
    """A millones, que es la unidad en la que se lee un balance."""
    return "            n/d" if v is None else "%15.1f" % (v / 1e6)


def anuales_por_tag(facts, tags):
    """
    {fin_de_ejercicio: {tag: {val, filed}}} con los hechos de duracion ANUAL.

    Se queda con la ultima reexpresion de cada tag (el 'filed' mas nuevo), que
    es la misma regla que usa el normalizador.
    """
    out = defaultdict(dict)
    for tag in tags:
        grupo = facts.get("us-gaap", {}).get(tag, {})
        for arr in grupo.get("units", {}).values():
            for h in arr:
                if h.get("val") is None or not h.get("start") or not h.get("end"):
                    continue
                if X._tramo(h["start"], h["end"]) != "FY":
                    continue
                prev = out[h["end"]].get(tag)
                if prev is None or h.get("filed", "") > prev["filed"]:
                    out[h["end"]][tag] = {"val": h["val"],
                                          "filed": h.get("filed", "")}
    return out


def discrepan(por_tag):
    """True si dos tags del mismo ejercicio difieren mas que TOL."""
    vals = [d["val"] for d in por_tag.values() if d["val"]]
    if len(vals) < 2:
        return False
    lo, hi = min(vals), max(vals)
    return abs(hi - lo) / max(abs(hi), 1.0) > TOL


def trimestres_por_ejercicio(periodos):
    """{fiscal_year: [(period_end, valor, tag), ...]} de la salida normalizada."""
    out = defaultdict(list)
    for p in periodos:
        if p.get("revenue") is None or p.get("fiscal_year") is None:
            continue
        out[p["fiscal_year"]].append((p["period_end"], p["revenue"],
                                      p.get("revenue__tag", "")))
    return out


def yq_anuales(cur, ticker):
    """
    {anio_de_cierre: suma de los 4 trimestres} segun yahooquery.

    Solo si estan los 4: una suma parcial no es comparable contra un anual y
    seria justamente el tipo de numero plausible-pero-falso que este reporte
    trata de evitar.
    """
    cur.execute("SELECT fiscal_period_end, total_revenue "
                "FROM fundamentales_income_q "
                "WHERE ticker=%s AND total_revenue IS NOT NULL "
                "ORDER BY fiscal_period_end", (ticker,))
    por_anio = defaultdict(list)
    for fin, val in cur.fetchall():
        por_anio[fin.year].append(float(val))
    return {a: sum(v) for a, v in por_anio.items() if len(v) == 4}


def analizar(datos, yq):
    """
    Devuelve (ambiguo, ejercicios). `ambiguo` = algun ejercicio donde dos tags
    anuales dan numeros distintos, que es cuando hay algo que elegir.
    """
    facts = (datos or {}).get("facts", {})
    anuales = anuales_por_tag(facts, X.FLUJO_ADITIVO["revenue"])
    por_fy = trimestres_por_ejercicio(X.normalizar(datos, desde=DESDE)["periodos"])

    ejercicios, ambiguo = [], False
    for fin in sorted(anuales):
        if fin < DESDE:
            continue
        por_tag = anuales[fin]
        disc = discrepan(por_tag)
        ambiguo = ambiguo or disc

        # El ejercicio se ubica por el anio del cierre, no por fiscal_year: hay
        # empresas cuyo ejercicio fiscal 2024 cierra en enero de 2025.
        qs = None
        for lista in por_fy.values():
            if lista and max(q[0] for q in lista)[:4] == fin[:4]:
                qs = sorted(lista)
                break
        tags_q = sorted({q[2] for q in (qs or [])})

        ejercicios.append({
            "fin": fin, "anuales": por_tag, "discrepan": disc,
            "suma_4q": sum(q[1] for q in qs) if qs and len(qs) == 4 else None,
            "n_q": len(qs or []), "tags_q": tags_q, "mezcla": len(tags_q) > 1,
            "yq": yq.get(int(fin[:4])),
        })
    return ambiguo, ejercicios


def sugerir(ejercicios):
    """
    Que tag elegiria el arbitraje contra yahooquery.

    NO decide: cuenta en cuantos ejercicios el anual de cada tag coincide con
    la suma de los 4 trimestres de yahooquery. Un tag que gana en TODOS los
    ejercicios verificables es un candidato fuerte; un empate, o cero votos,
    significa que la decision necesita mirar el balance de la empresa.
    """
    votos = defaultdict(int)
    verificables = 0
    for e in ejercicios:
        if e["yq"] is None:
            continue
        verificables += 1
        for tag, d in e["anuales"].items():
            if d["val"] and abs(d["val"] - e["yq"]) / max(abs(e["yq"]), 1.0) <= 0.02:
                votos[tag] += 1
    if not votos:
        return None, 0, verificables
    tag = max(votos, key=lambda t: votos[t])
    return tag, votos[tag], verificables


def imprimir(ticker, sector, ambiguo, ejercicios):
    sug, gano, verif = sugerir(ejercicios)
    print()
    print("-" * 78)
    print("  %-6s  %s%s" % (ticker, sector or "?",
                            "" if ambiguo else "   [sin ambiguedad]"))
    print("-" * 78)
    for e in ejercicios:
        print("  ejercicio %s   (%d Q normalizados)%s"
              % (e["fin"], e["n_q"], "   <-- MEZCLA DE TAGS" if e["mezcla"] else ""))
        if e["tags_q"]:
            print("      tags en los Q: %s" % ", ".join(e["tags_q"]))
        orden = sorted(e["anuales"], key=lambda t: -abs(e["anuales"][t]["val"] or 0))
        for tag in orden:
            d = e["anuales"][tag]
            ref = e["yq"]
            dif = ("" if not ref or not d["val"] else
                   "  vs yq %+8.2f%%" % ((d["val"] - ref) / abs(ref) * 100))
            print("      anual %s MM   %-50s%s" % (mm(d["val"]), tag, dif))
        print("      suma 4Q SEC      %s MM" % mm(e["suma_4q"]))
        print("      anual yahooquery %s MM" % mm(e["yq"]))
    if sug:
        print("  ARBITRAJE yahooquery -> %s   (coincide en %d de %d ejercicios)"
              % (sug, gano, verif))
    else:
        print("  ARBITRAJE yahooquery -> sin veredicto (%d ejercicios verificables)"
              % verif)


def main():
    p = argparse.ArgumentParser(description="Cuadro de decision del tag de revenue")
    p.add_argument("--tickers", help="CSV; default: todos los del cache")
    p.add_argument("--todos", action="store_true",
                   help="Imprime tambien los tickers SIN ambiguedad")
    p.add_argument("--csv", help="Vuelca el detalle a un CSV")
    args = p.parse_args()

    env = _parse_env_file(os.path.join(ROOT, ".env"))
    pedidos = ([t.strip().upper() for t in args.tickers.split(",")]
               if args.tickers else
               sorted(f[:-5] for f in os.listdir(CACHE) if f.endswith(".json")))

    print()
    print(SEP)
    print("  REVENUE -- CUADRO DE DECISION DEL TAG")
    print(SEP)

    ambiguos, limpios, sin_cache = [], [], []
    filas_csv = []

    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute("SELECT ticker, sector FROM activos")
            sectores = dict(cur.fetchall())

            for ticker in pedidos:
                datos = client.leer_cache(CACHE, ticker)
                if not datos:
                    sin_cache.append(ticker)
                    continue
                ambiguo, ejercicios = analizar(datos, yq_anuales(cur, ticker))
                (ambiguos if ambiguo else limpios).append(ticker)

                for e in ejercicios:
                    for tag, d in e["anuales"].items():
                        filas_csv.append({
                            "ticker": ticker, "sector": sectores.get(ticker, ""),
                            "ejercicio": e["fin"], "tag": tag,
                            "anual_sec": d["val"], "suma_4q_sec": e["suma_4q"],
                            "anual_yahooquery": e["yq"],
                            "tags_en_los_q": "|".join(e["tags_q"]),
                            "mezcla_de_tags": int(e["mezcla"]),
                            "tags_anuales_discrepan": int(e["discrepan"]),
                        })

                if ambiguo or args.todos:
                    imprimir(ticker, sectores.get(ticker), ambiguo, ejercicios)

    print()
    print(SEP)
    print("  ambiguos: %d   |   sin ambiguedad: %d   |   sin cache: %d"
          % (len(ambiguos), len(limpios), len(sin_cache)))
    if ambiguos:
        print()
        print("  A CURAR: " + ", ".join(ambiguos))
    print(SEP)
    print()

    if args.csv and filas_csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas_csv[0].keys()))
            w.writeheader()
            w.writerows(filas_csv)
        print("  CSV: %s  (%d filas)" % (args.csv, len(filas_csv)))
        print()


if __name__ == "__main__":
    main()
