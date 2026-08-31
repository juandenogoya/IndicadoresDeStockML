"""
Agregar tags de deuda: cuanto LLENA y cuanto ROMPE?

Dos variantes, porque con el defecto de _elegir la POSICION importa al reves
de lo intuitivo (gana el pri MAS GRANDE):
  - al FINAL  -> el tag nuevo GANA a los actuales (pisa lo que hoy funciona)
  - al FRENTE -> el tag nuevo PIERDE (solo se usa si los otros no estan)

La segunda es el comportamiento de fallback que queremos, y no necesita tocar
_elegir. Esto lo verifica con datos.

Solo lee el cache. No escribe.
"""
import glob, json, os, sys
from collections import Counter, defaultdict

ROOT = r"C:\Users\juand\OneDrive\Escritorio\Indicadores y Machine Learning"
sys.path.insert(0, ROOT)
from src.data.sec.tags_curados import para
from src.utils import sec_xbrl

DESDE = "2018-01-01"
# Solo sinonimos LEGITIMOS de deuda de largo plazo.
# DebtLongtermAndShorttermCombinedAmount queda AFUERA a proposito: es la deuda
# TOTAL, no la de largo plazo. Sumarla a debt_long duplicaria el corto plazo
# en net_debt = debt_short + debt_long - cash. Necesita diseno propio.
NUEVOS = ["LongTermDebtAndCapitalLeaseObligations",
          "LongTermDebtAndFinanceLeaseObligations",
          "ConvertibleDebtNoncurrent"]

BASE = list(sec_xbrl.INSTANTE["debt_long"])


def correr(lista):
    sec_xbrl.INSTANTE["debt_long"] = lista
    out = {}
    for ruta in sorted(glob.glob(os.path.join(ROOT, "data", "sec_cache", "*.json"))):
        tk = os.path.splitext(os.path.basename(ruta))[0]
        try:
            facts = json.load(open(ruta, encoding="utf-8"))
            r = sec_xbrl.normalizar(facts, desde=DESDE, tags_curados=para(tk))
        except Exception:
            continue
        out[tk] = {f["period_end"]: (f.get("debt_long"), f.get("debt_long__tag"))
                   for f in r["periodos"]}
    sec_xbrl.INSTANTE["debt_long"] = BASE
    return out


def comparar(a, b, etiqueta):
    llena = pisa = igual = 0
    tk_llena, tk_pisa = set(), set()
    por_tag = Counter()
    for tk in a:
        for pe, (va, ta) in a[tk].items():
            vb, tb = b.get(tk, {}).get(pe, (None, None))
            if va is None and vb is not None:
                llena += 1; tk_llena.add(tk); por_tag[tb] += 1
            elif va is not None and vb is not None and va != vb:
                pisa += 1; tk_pisa.add(tk)
            else:
                igual += 1
    print()
    print("  --- %s ---" % etiqueta)
    print("    LLENA huecos:  %5d periodos  en %3d tickers" % (llena, len(tk_llena)))
    print("    PISA valores:  %5d periodos  en %3d tickers" % (pisa, len(tk_pisa)))
    print("    sin cambio:    %5d" % igual)
    if por_tag:
        print("    los huecos los llena:")
        for t, n in por_tag.most_common():
            print("       %-52s %4d" % (t, n))
    if tk_pisa:
        print("    tickers pisados: %s" % " ".join(sorted(tk_pisa))[:200])
    return tk_llena


print("=" * 78)
print("  AGREGAR TAGS DE DEUDA DE LARGO PLAZO")
print("=" * 78)
print("  base actual:", BASE)
print("  a agregar:  ", NUEVOS)

actual = correr(BASE)
al_frente = correr(NUEVOS + BASE)
al_final = correr(BASE + NUEVOS)

comparar(actual, al_final, "al FINAL de la lista (el nuevo GANA)")
recup = comparar(actual, al_frente, "al FRENTE de la lista (el nuevo es FALLBACK)")

# El unico numero que importa de verdad: cuantos tickers pasan a tener deuda
# en su ULTIMO periodo, que es el que decide el EV de la rueda de hoy.
antes = sum(1 for tk, d in actual.items()
            if d and actual[tk][max(d)][0] is not None)
despues = sum(1 for tk, d in al_frente.items()
              if d and al_frente[tk][max(d)][0] is not None)
print()
print("=" * 78)
print("  tickers con debt_long en su ULTIMO periodo:  antes %d  ->  despues %d"
      % (antes, despues))
print("=" * 78)
