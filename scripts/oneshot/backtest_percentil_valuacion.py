"""
backtest_percentil_valuacion.py -- el percentil "caro vs si misma", sirve?

Pregunta concreta: comprar un ticker cuando su multiplo esta BAJO dentro de su
propia historia, da mejor retorno futuro que comprarlo cuando esta ALTO?

Es un test de RAZONABILIDAD de la capa de valuacion, no una estrategia. Si el
percentil no ordena nada, la vista sirve para describir pero no para decidir, y
conviene saberlo antes de construir encima.

Solo lee. No escribe en la DB.

Las cuatro decisiones metodologicas, que son lo que separa esto de un numero
lindo y falso
-------------------------------------------------------------------------
1. RETORNO EN EXCESO, no retorno crudo. Todos los tickers se mueven con el
   mercado. Un bucket "barato" concentrado en 2022 mostraria lo que hizo el
   mercado despues de 2022 y no lo que aporta la senal. Se resta el promedio
   de TODOS los tickers con dato ESE MISMO DIA: el benchmark es el universo,
   sin traer una serie externa que habria que alinear aparte.

2. MUESTREO MENSUAL. Con observaciones diarias y ventanas de 12 meses, dos
   observaciones consecutivas comparten el 99,6% de su futuro. El n seria
   enorme y la independencia, cero. Se toma una rueda por mes por ticker.
   Aun asi las ventanas se pisan entre meses: los intervalos que se reportan
   son ORIENTATIVOS, no un test de hipotesis.

3. SIN LOOKAHEAD, por construccion de la fuente. El percentil es trailing y el
   TTM se ancla por `filed_primero` (cuando el balance se hizo publico), no
   por el cierre del trimestre. Esa es la razon por la que este backtest se
   puede hacer sobre SEC y no sobre yahooquery.

4. SESGO DE SUPERVIVENCIA declarado y NO corregido: el universo son los
   tickers vivos hoy. Los que salieron no estan. Empuja los retornos hacia
   arriba de forma pareja entre buckets, asi que deforma el nivel pero no
   tanto el ORDEN, que es lo que se esta midiendo.

Uso:
    python scripts/oneshot/backtest_percentil_valuacion.py
    python scripts/oneshot/backtest_percentil_valuacion.py --metrica pe
"""

import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import psycopg2

from scripts.oneshot.create_fundamentales_tables import _parse_env_file

SEP = "=" * 78
TABLA = "fundamentales_sec_multiplos_d"

METRICAS = [("pe_pct", "PER"), ("ps_pct", "P/S"),
            ("pb_pct", "P/B"), ("ev_ebitda_pct", "EV/EBITDA")]

# Horizontes en ruedas. 63 ~ 3 meses, 126 ~ 6, 252 ~ 12.
HORIZONTES = [(63, "3m"), (126, "6m"), (252, "12m")]

# Quintiles. Con deciles el bucket extremo queda con pocas observaciones por
# ticker y el resultado se vuelve ruido de unos pocos nombres.
CORTES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
ETIQUETAS = ["p00-20 (mas barato)", "p20-40", "p40-60", "p60-80",
             "p80-100 (mas caro)"]


def _conn(env):
    return psycopg2.connect(host=env["DB_HOST"], port=env["DB_PORT"],
                            user=env["DB_USER"], password=env["DB_PASSWORD"],
                            dbname=env["DB_NAME"])


def cargar(cur):
    """
    {ticker: [(fecha, close, {metrica: percentil}), ...]} ordenado por fecha.

    No hay flag para "aflojar" el percentil, y no es un olvido: el rigor se
    aplica AL COMPUTAR la tabla. Con el default estricto, la rueda que no tiene
    la ventana llena ya trae el percentil en NULL, asi que no hay nada que
    filtrar aca. Para medir sobre una ventana mas corta hay que recomputar
    fundamentales_sec_multiplos_d con --percentil-permisivo y volver a correr
    esto -- lo que ademas cambia el universo y por lo tanto el BENCHMARK, y eso
    es justamente lo que hay que tener presente al comparar las dos corridas.
    """
    cols = [m for m, _ in METRICAS]
    sql = ("SELECT ticker, fecha, close, %s FROM %s "
           "WHERE close IS NOT NULL ORDER BY ticker, fecha"
           % (", ".join(cols), TABLA))
    cur.execute(sql)
    series = defaultdict(list)
    for fila in cur.fetchall():
        ticker, fecha, close = fila[0], fila[1], float(fila[2])
        pcts = {m: (float(v) if v is not None else None)
                for (m, _), v in zip(METRICAS, fila[3:])}
        series[ticker].append((fecha, close, pcts))
    return series


def observaciones(series):
    """
    Una observacion por (ticker, mes): percentiles del dia y retorno futuro a
    cada horizonte. El retorno se mide sobre la MISMA serie de cierres, asi que
    esta en la misma base de split -- precios_diarios se corrige por divisor
    hacia atras y por eso un ratio de precios es siempre comparable.
    """
    out = []
    for ticker, filas in series.items():
        closes = [f[1] for f in filas]
        vistos = set()
        for i, (fecha, close, pcts) in enumerate(filas):
            clave = (fecha.year, fecha.month)
            if clave in vistos:
                continue
            vistos.add(clave)
            fut = {}
            for h, etiqueta in HORIZONTES:
                j = i + h
                if j < len(closes) and close > 0:
                    fut[etiqueta] = closes[j] / close - 1.0
            if fut:
                out.append({"ticker": ticker, "fecha": fecha,
                            "pcts": pcts, "fut": fut})
    return out


def a_exceso(obs):
    """
    Convierte el retorno de cada observacion en retorno EN EXCESO sobre el
    promedio del universo en esa misma fecha y horizonte. Las fechas con menos
    de 20 tickers se descartan: un promedio de 3 nombres no es un mercado.
    """
    por_fecha = defaultdict(lambda: defaultdict(list))
    for o in obs:
        for h, r in o["fut"].items():
            por_fecha[o["fecha"]][h].append(r)

    medias = {f: {h: (sum(v) / len(v) if len(v) >= 20 else None)
                  for h, v in hs.items()}
              for f, hs in por_fecha.items()}

    for o in obs:
        o["exc"] = {}
        for h, r in o["fut"].items():
            m = medias[o["fecha"]].get(h)
            if m is not None:
                o["exc"][h] = r - m
    return obs


def bucket(p):
    for i in range(len(CORTES) - 1):
        if CORTES[i] <= p < CORTES[i + 1]:
            return i
    return None


def mediana(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2.0


def tabla(obs, metrica, etiqueta):
    print()
    print("-" * 78)
    print("  %s  --  retorno EN EXCESO sobre el universo, por quintil del "
          "percentil" % etiqueta)
    print("-" * 78)
    print("  %-22s %7s %s" % ("quintil (barato -> caro)", "n",
                              "".join("%17s" % ("  " + h) for _, h in HORIZONTES)))

    filas = []
    for b in range(len(ETIQUETAS)):
        celdas, n_b = [], 0
        for _, h in HORIZONTES:
            xs = [o["exc"][h] for o in obs
                  if o["pcts"].get(metrica) is not None
                  and bucket(o["pcts"][metrica]) == b and h in o.get("exc", {})]
            n_b = max(n_b, len(xs))
            if not xs:
                celdas.append("%17s" % "n/d")
                continue
            med = mediana(xs) * 100
            positivos = sum(1 for x in xs if x > 0) / len(xs) * 100
            celdas.append("%10s %5s" % ("%+.1f%%" % med, "%2.0f%%" % positivos))
        print("  %-22s %7d %s" % (ETIQUETAS[b], n_b, "".join(celdas)))
        filas.append((b, n_b))

    print("  %-22s %7s %s" % ("", "", "".join("%17s" % "mediana / % > 0"
                                              for _ in HORIZONTES)))

    # Diferencial barato menos caro: es EL numero de este backtest.
    print()
    for _, h in HORIZONTES:
        baratos = [o["exc"][h] for o in obs
                   if o["pcts"].get(metrica) is not None
                   and bucket(o["pcts"][metrica]) == 0 and h in o.get("exc", {})]
        caros = [o["exc"][h] for o in obs
                 if o["pcts"].get(metrica) is not None
                 and bucket(o["pcts"][metrica]) == 4 and h in o.get("exc", {})]
        if baratos and caros:
            d = (mediana(baratos) - mediana(caros)) * 100
            print("    %-4s  barato menos caro: %+6.2f puntos  (n %d vs %d)"
                  % (h, d, len(baratos), len(caros)))


def main():
    p = argparse.ArgumentParser(description="Backtest del percentil de valuacion")
    p.add_argument("--metrica", help="pe | ps | pb | ev_ebitda (default: todas)")
    args = p.parse_args()

    env = _parse_env_file(os.path.join(ROOT, ".env"))
    with _conn(env) as cx:
        with cx.cursor() as cur:
            series = cargar(cur)

    obs = a_exceso(observaciones(series))
    con_exc = [o for o in obs if o.get("exc")]

    print()
    print(SEP)
    print("  BACKTEST DEL PERCENTIL DE VALUACION")
    print(SEP)
    fechas = [o["fecha"] for o in con_exc]
    print("  tickers: %d  |  observaciones (1 por ticker-mes): %d"
          % (len(series), len(con_exc)))
    if fechas:
        print("  ventana: %s a %s" % (min(fechas), max(fechas)))
    print("  benchmark: promedio del universo en la misma fecha y horizonte")

    elegidas = METRICAS
    if args.metrica:
        clave = args.metrica.lower().rstrip("_pct")
        elegidas = [(m, e) for m, e in METRICAS if m.startswith(clave)]

    for metrica, etiqueta in elegidas:
        tabla(con_exc, metrica, etiqueta)

    print()
    print(SEP)
    print("  Ventanas superpuestas entre meses: los n NO son independientes.")
    print("  Universo de supervivientes: el nivel esta inflado parejo, el ORDEN")
    print("  entre quintiles es lo que se esta midiendo.")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
