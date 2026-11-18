"""
evaluar_etapa2.py -- criterios 2 a 4 de la alerta de la lista de vigilancia
(LOCAL, solo lee). El criterio 1 (correccion) lo mide auditar_etapa2.py.

Evalua cada corte con la MISMA funcion que usa el snapshot para armar el
mensaje (intradia_db.evaluar_corte), sobre lo que hay en local:
    - barras: precios_intradia_30m (sync, o --reconstruir N en la VM);
    - listas: vigilancia_tickers / vigilancia_niveles (las de la rutina, o
      reconstruidas con push_vigilancia.py --desde --hasta).
A las 15:00 se asume que el mensaje de las 11:00 salio (en la historia no hay
registro de envios): se cuentan solo las novedades, como en vivo.

CRITERIOS (pre-registrados el 30/10/2026, antes de correr nada):
    2. por TIPO de evento (cruce / revierte / movimiento), la MEDIANA de lineas
       por corte esta entre 1 y 15. Un tipo que falla sale del mensaje; si
       fallan los tres, la alerta se duerme. Una regla que nunca dispara no es
       una regla (CLAUDE.md, alta y baja de estrategias).
    3. un solo mensaje por corte. Se cumple por construccion (el mensaje achica
       las listas antes que partirse): se informa en cuantos cortes tuvo que
       achicar.
    4. INFORMATIVO: de los eventos vistos en cada corte, cuantos seguian al
       cierre de la rueda (mismo lado del nivel con el close de precios_diarios;
       |z| >= umbral con el close). Dice si la alerta adelanta al mensaje de la
       noche o es ruido. NO mide rendimiento.
    5. el juicio del usuario despues de 5-10 ruedas reales (se escribe en el doc).

Detalle por evento en reportes/intradia/AAAAMMDD_etapa2/eventos.csv (fuera de git).

Uso:
    python scripts/intradia/evaluar_etapa2.py
    python scripts/intradia/evaluar_etapa2.py --desde 2026-09-15 --hasta 2026-10-29

Salida: 0 al menos un tipo pasa el criterio 2 / 2 ninguno / 1 error o sin datos.
"""

import argparse
import csv
import math
import os
import sys
from datetime import date, datetime
from statistics import median

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))
os.environ.pop("DATABASE_URL", None)   # LOCAL siempre

from sqlalchemy import text   # noqa: E402

from src.data.database import get_engine          # noqa: E402
from src.utils import intradia, sesion_nyse       # noqa: E402
from src.utils import vigilancia_intradia as vi   # noqa: E402

import intradia_db as idb     # noqa: E402

RANGO_C2 = (1, 15)
MIN_CORTES = 20


def cortes_disponibles(eng, desde, hasta) -> list:
    sql = f"SELECT DISTINCT fecha, corte FROM {intradia.TABLA}"
    cond, p = [], {}
    if desde:
        cond.append("fecha >= :d")
        p["d"] = desde
    if hasta:
        cond.append("fecha <= :h")
        p["h"] = hasta
    if cond:
        sql += " WHERE " + " AND ".join(cond)
    with eng.connect() as c:
        filas = c.execute(text(sql), p).fetchall()
    orden = {c: i for i, c in enumerate(sesion_nyse.CORTES)}
    return sorted(((f, c) for f, c in filas), key=lambda x: (x[0], orden[x[1]]))


def closes_de(eng, fecha) -> dict:
    with eng.connect() as c:
        return {r.ticker: float(r.close) for r in c.execute(text(
            "SELECT ticker, close FROM precios_diarios WHERE fecha = :f AND close IS NOT NULL"),
            {"f": fecha}).fetchall()}


def persiste(ev: dict, close_f, info: dict):
    """El evento del corte sigue al cierre? None si falta el close."""
    if close_f is None:
        return None
    if ev["tipo"] == "movimiento":
        t = info[ev["ticker"]]
        return abs(math.log(close_f / t["close_d"]) / t["sigma_d"]) >= vi.UMBRAL_Z
    return vi.lado(close_f, ev["valor"]) == ev["lado"]


def evaluar(eng, desde=None, hasta=None) -> dict:
    """Corre todos los cortes. Lo reusa auditar_etapa2.py."""
    out = {"cortes": [], "sin_lista": [], "eventos": []}
    for fecha, corte in cortes_disponibles(eng, desde, hasta):
        hora = sesion_nyse.hora_corte(fecha, corte)
        if hora is None:
            continue
        r = idb.evaluar_corte(eng, fecha, corte, hora, asumir_previo=True)
        if not r["ok"]:
            out["sin_lista"].append((fecha, corte, r["esperada"]))
            continue
        completo = vi._armar(fecha, corte, r["fecha_lista"], r["tickers"], r["nuevos"],
                             r["deshechos"], r["sin_dato"], r["hay_previo"], vi.MAX_LINEAS)
        info = {t["ticker"]: t for t in r["tickers"]}
        closes = closes_de(eng, fecha)
        for e in r["eventos"]:
            out["eventos"].append(dict(e, fecha=fecha, corte=corte,
                                       persiste=persiste(e, closes.get(e["ticker"]), info)))
        out["cortes"].append({"fecha": fecha, "corte": corte, "r": r,
                              "lineas": vi.lineas_por_tipo(r["nuevos"], r["deshechos"]),
                              "largo": len(r["texto"]),
                              "achico": len(completo) > vi.MAX_LEN_MENSAJE})
    return out


def _escribir_csv(eventos: list) -> str:
    d = os.path.join(ROOT, "reportes", "intradia", f"{datetime.now():%Y%m%d}_etapa2")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "eventos.csv")
    cols = ["fecha", "corte", "ticker", "tipo", "nivel", "valor", "lado", "precio",
            "dist_pct", "z", "var_pct", "persiste"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(eventos)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde")
    ap.add_argument("--hasta")
    args = ap.parse_args()
    desde = date.fromisoformat(args.desde) if args.desde else None
    hasta = date.fromisoformat(args.hasta) if args.hasta else None

    eng = get_engine()
    ev = evaluar(eng, desde, hasta)
    cortes = ev["cortes"]
    print("=" * 70)
    print("  EVALUACION ETAPA 2 -- alerta de la lista de vigilancia")
    print("=" * 70)
    if ev["sin_lista"]:
        print(f"  Cortes sin la lista de la rueda anterior: {len(ev['sin_lista'])}")
        faltan = sorted({str(e) for _, _, e in ev["sin_lista"]})
        print(f"    listas que faltan: {', '.join(faltan[:10])}{' ...' if len(faltan) > 10 else ''}")
        print("    reconstruir: python scripts/intradia/push_vigilancia.py --desde AAAA-MM-DD")
    if not cortes:
        print("  Ningun corte evaluable.")
        return 1
    ruedas = sorted({c["fecha"] for c in cortes})
    print(f"  Cortes evaluados: {len(cortes)} | ruedas {ruedas[0]} -> {ruedas[-1]} ({len(ruedas)})")
    vig = [len(c["r"]["tickers"]) for c in cortes]
    print(f"  Vigilados por lista: mediana {median(vig):.0f} | min {min(vig)} | max {max(vig)}")
    if len(cortes) < MIN_CORTES:
        print(f"  [OJO] muestra chica: menos de {MIN_CORTES} cortes.")

    print(f"\n  2. Lineas por corte, por tipo (mediana entre {RANGO_C2[0]} y {RANGO_C2[1]})")
    pasan = {}
    for corte in sesion_nyse.CORTES:
        sub = [c for c in cortes if c["corte"] == corte]
        if sub:
            print(f"     {corte} ({len(sub)} cortes):")
        for tipo in vi.TIPOS + ("deshechos",):
            if not sub or (tipo == "deshechos" and corte == sesion_nyse.APERTURA_90):
                continue
            xs = [c["lineas"][tipo] for c in sub]
            print(f"       {tipo:<11} mediana {median(xs):>4.1f} | min {min(xs)} | max {max(xs)}")
    for tipo in vi.TIPOS:
        xs = [c["lineas"][tipo] for c in cortes]
        m = median(xs)
        pasan[tipo] = RANGO_C2[0] <= m <= RANGO_C2[1]
        print(f"     {tipo:<11} todos los cortes: mediana {m:.1f} -> "
              f"{'PASA' if pasan[tipo] else 'NO PASA (sale del mensaje)'}")

    achicados = sum(1 for c in cortes if c["achico"])
    print(f"\n  3. Un mensaje por corte: PASA por construccion | largo max "
          f"{max(c['largo'] for c in cortes)} | cortes que achicaron listas: {achicados}")

    print("\n  4. INFORMATIVO: eventos que seguian al cierre de la rueda")
    for corte in sesion_nyse.CORTES:
        for tipo in vi.TIPOS:
            sub = [e for e in ev["eventos"] if e["corte"] == corte and e["tipo"] == tipo]
            con = [e for e in sub if e["persiste"] is not None]
            if not sub:
                continue
            pct = f"{sum(e['persiste'] for e in con) / len(con):.0%}" if con else "s/d"
            print(f"     {corte:<12} {tipo:<11} eventos {len(sub):>4} | con close {len(con):>4} "
                  f"| siguen al cierre {pct}")

    print(f"\n  Detalle: {_escribir_csv(ev['eventos'])}")
    vivos = [t for t, ok in pasan.items() if ok]
    print("\n" + "=" * 70)
    print(f"  Tipos que quedan en el mensaje: {', '.join(vivos) if vivos else 'NINGUNO -> se duerme'}")
    print("  Criterio 1: auditar_etapa2.py | criterio 5: juicio del usuario (docs/intradia.md)")
    print("=" * 70)
    return 0 if vivos else 2


if __name__ == "__main__":
    sys.exit(main())
