"""
medir_elegir_pri.py -- cuanto mueve arreglar el defecto de `_elegir`?

EL DEFECTO (documentado en src/utils/sec_xbrl.py): `_elegir` ordena ascendente
y toma el ultimo, con la clave (rango_forma, pri, filed). Como `pri` es la
POSICION del tag en la lista de sinonimos -- 0 = el mas preferido -- tomar el
mayor significa que **entre dos sinonimos gana el MENOS preferido**. Es lo
contrario de lo que hace la otra rama de seleccion del mismo modulo, el bloque
`directos` de _serie_aditiva, que compara `h["pri"] < prev["pri"]`.

Las dos ramas se contradicen. Arreglarlo mueve numeros en todo el universo, y
por eso hay que MEDIRLO antes y no cambiarlo de arrastre.

QUE HACE ESTE SCRIPT: normaliza cada ticker del cache DOS VECES -- una con el
`_elegir` actual y otra con el corregido -- y diffea periodo por periodo y
concepto por concepto.

No escribe en la DB. No sale a la red. Solo lee el cache en disco.

Por que monkeypatch y no editar el modulo: para que la medicion no dependa de
haber cambiado el codigo que se esta evaluando. Si el resultado dice que no
conviene, no hay nada que revertir.

Uso:
    python scripts/oneshot/medir_elegir_pri.py
    python scripts/oneshot/medir_elegir_pri.py --detalle revenue
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data.sec.tags_curados import para
from src.utils import sec_xbrl

SEP = "=" * 78
DESDE = "2018-01-01"          # la misma ventana de retencion que usa el refresh


def _elegir_corregido(cands, hasta_filed=None):
    """
    Identico al original salvo el signo de `pri`.

    Con `-h["pri"]`, tomar el ultimo del orden ascendente se queda con el pri
    MAS CHICO, o sea el sinonimo MAS preferido, que es lo que la lista de
    sinonimos declara y lo que hace la otra rama del modulo.
    """
    if hasta_filed:
        cands = [c for c in cands if c["filed"] and c["filed"] <= hasta_filed]
    if not cands:
        return None
    return sorted(cands,
                  key=lambda h: (sec_xbrl._rango_forma(h), -h["pri"],
                                 h["filed"]))[-1]


def normalizar_con(fn, facts, curados):
    """Corre normalizar() con la implementacion `fn` de _elegir."""
    original = sec_xbrl._elegir
    sec_xbrl._elegir = fn
    try:
        return sec_xbrl.normalizar(facts, desde=DESDE, tags_curados=curados)
    finally:
        sec_xbrl._elegir = original


def conceptos_de(fila):
    """Nombres de concepto de una fila (los que no son metadatos ni sufijos)."""
    return [k for k in fila
            if not k.endswith("__tag") and not k.endswith("__derivado")
            and k not in ("period_end", "fiscal_year", "fiscal_quarter",
                          "filed_primero", "filed_ultimo", "origen")]


def main():
    p = argparse.ArgumentParser(description="Mide el arreglo de _elegir")
    p.add_argument("--detalle", help="Muestra los casos de ESTE concepto")
    args = p.parse_args()

    archivos = sorted(glob.glob(os.path.join(ROOT, "data", "sec_cache", "*.json")))

    n_tickers = 0
    tickers_tocados = set()
    cambios_valor = Counter()          # concepto -> cuantos periodos cambian
    cambios_tag = Counter()
    tickers_por_concepto = defaultdict(set)
    magnitudes = defaultdict(list)     # concepto -> [cambio relativo]
    ejemplos = defaultdict(list)
    aparecen = Counter()               # el corregido llena un hueco
    desaparecen = Counter()

    for ruta in archivos:
        ticker = os.path.splitext(os.path.basename(ruta))[0]
        try:
            facts = json.load(open(ruta, encoding="utf-8"))
        except Exception:
            continue
        curados = para(ticker)
        try:
            a = normalizar_con(sec_xbrl._elegir, facts, curados)
            b = normalizar_con(_elegir_corregido, facts, curados)
        except Exception as e:
            print("  [!] %s: %s" % (ticker, e))
            continue
        n_tickers += 1

        pa = {f["period_end"]: f for f in a["periodos"]}
        pb = {f["period_end"]: f for f in b["periodos"]}
        for pe in sorted(set(pa) & set(pb)):
            fa, fb = pa[pe], pb[pe]
            for c in conceptos_de(fa):
                va, vb = fa.get(c), fb.get(c)
                ta, tb = fa.get(c + "__tag"), fb.get(c + "__tag")
                if ta != tb:
                    cambios_tag[c] += 1
                if va == vb:
                    continue
                if va is None and vb is not None:
                    aparecen[c] += 1
                elif va is not None and vb is None:
                    desaparecen[c] += 1
                cambios_valor[c] += 1
                tickers_por_concepto[c].add(ticker)
                tickers_tocados.add(ticker)
                if va and vb and va != 0:
                    rel = abs(float(vb) - float(va)) / abs(float(va))
                    magnitudes[c].append(rel)
                    if len(ejemplos[c]) < 6:
                        ejemplos[c].append((ticker, pe, ta, va, tb, vb, rel))

    print()
    print(SEP)
    print("  ARREGLAR _elegir: QUE SE MUEVE")
    print(SEP)
    print("  tickers normalizados: %d  |  tickers con algun cambio: %d (%.0f%%)"
          % (n_tickers, len(tickers_tocados),
             100.0 * len(tickers_tocados) / max(n_tickers, 1)))
    print("  ventana: periodos desde %s" % DESDE)

    if not cambios_valor:
        print()
        print("  NINGUN valor cambia. El arreglo es seguro y sin efecto"
              " observable.")
        print(SEP)
        return 0

    print()
    print("  %-22s %8s %8s %10s %10s %9s" % ("concepto", "periodos", "tickers",
                                             "mediana", "p90", "aparecen"))
    def med(xs):
        ys = sorted(xs)
        return ys[len(ys) // 2] if ys else 0.0
    for c, n in cambios_valor.most_common():
        ms = magnitudes[c]
        q90 = sorted(ms)[min(int(len(ms) * .9), len(ms) - 1)] if ms else 0.0
        print("  %-22s %8d %8d %9.1f%% %9.1f%% %9d"
              % (c, n, len(tickers_por_concepto[c]), med(ms) * 100, q90 * 100,
                 aparecen[c]))

    print()
    print("  huecos que el corregido LLENA:    %d" % sum(aparecen.values()))
    print("  valores que el corregido PIERDE:  %d" % sum(desaparecen.values()))

    if args.detalle:
        c = args.detalle
        print()
        print("-" * 78)
        print("  CASOS de %s (hasta 6)" % c)
        for tk, pe, ta, va, tb, vb, rel in ejemplos.get(c, []):
            print("    %-6s %s" % (tk, pe))
            print("       actual    %-52s %s" % (ta, "{:,.0f}".format(va / 1e6)))
            print("       corregido %-52s %s   (%+.1f%%)"
                  % (tb, "{:,.0f}".format(vb / 1e6), rel * 100))

    print()
    print(SEP)
    print("  Un cambio NO es una mejora por si mismo: hay que mirar si el tag")
    print("  que gana es el correcto. Los sinonimos estan ordenados a mano y")
    print("  ese orden es la hipotesis que este arreglo pasa a respetar.")
    print(SEP)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
