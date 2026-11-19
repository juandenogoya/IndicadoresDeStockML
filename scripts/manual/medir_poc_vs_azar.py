"""
medir_poc_vs_azar.py -- el POC contra el azar. SOLO LEE.

Contesta la pregunta pre-registrada en docs/perfil_precio.md sec. 4: cuando el
precio entra en el area de valor, hace algo distinguible de lo que hace al entrar
en una banda cualquiera del mismo ticker?

EL DISENO ESTA EN EL DOC Y SE ESCRIBIO ANTES QUE ESTE SCRIPT. Aca no se eligen
umbrales ni direcciones: se ejecutan los que ya estan fijados.

  evento      : el cierre entra en la banda habiendo estado 5 ruedas afuera; la
                banda es la de t-1 (la de t incluiria el dia del evento).
  controles   : A = tramo del mismo ancho en una franja visitada al azar;
                B = idem entre las franjas del cuartil superior de volumen.
                20 semillas cada uno. B es el que decide.
  metrica     : exceso a h ruedas contra el universo equal-weight de la rueda,
                media por evento con IC95 y el DIA como unidad.
  primarias   : ventana 52, h 10, area angosta (<10%), las DOS direcciones.
  pasa si     : el exceso cae fuera del p5-p95 del control B en la direccion
                esperada, el IC95 excluye el cero, y las dos cosas valen en
                seleccion Y en confirmacion.

Sale 0 si la corrida termino bien (pase o no pase la prueba), 1 si fallo.
"""

import argparse
import json
import os
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
load_dotenv(".env", override=True)
os.environ.pop("DATABASE_URL", None)          # LOCAL manda (CLAUDE.md)

from sqlalchemy import text

from src.data.database import get_engine
from src.utils import perfil_precio as pp
from src.utils import poc_eventos as pe
from src.utils.ft_salidas import ic95_por_dia

# -- parametros PRE-REGISTRADOS (docs/perfil_precio.md sec. 4) -----------------
VENTANAS = (52, 21)                  # 52 es la primaria
HORIZONTES = (10, 5)                 # 10 es el primario
ANCHO_ANGOSTA = 10.0                 # ancho_va_pct < 10 -> "angosta"
FUERA = 5
SEMILLAS = 20
DESDE = date(2021, 1, 1)
CORTE = date(2025, 1, 1)             # seleccion < CORTE <= confirmacion
MIN_RUEDAS_PRE_CORTE = 300
PRIMARIA = {"ventana": 52, "h": 10, "area": "angosta"}
SEED_BASE = 20261119

MODOS = (("visitado", "A"), ("denso", "B"))


def log(msg):
    print(msg, flush=True)


# -- datos ---------------------------------------------------------------------

def cargar(engine) -> pd.DataFrame:
    q = text("""
        SELECT p.ticker, p.fecha, p.high, p.low, p.close, p.volume
        FROM precios_diarios p
        JOIN activos a ON a.ticker = p.ticker AND a.activo = TRUE
        WHERE p.fecha >= :desde
        ORDER BY p.ticker, p.fecha
    """)
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(q, {"desde": DESDE}).fetchall(),
                          columns=["ticker", "fecha", "high", "low", "close", "volume"])
    for col in ("high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def universo_forward(df: pd.DataFrame, horizontes) -> dict:
    """{h: Series indexada por fecha} con el retorno equal-weight del universo."""
    out = {}
    for h in horizontes:
        df = df.copy()
        df[f"_f{h}"] = df.groupby("ticker")["close"].transform(
            lambda s: s.shift(-h) / s - 1.0)
        out[h] = df.groupby("fecha")[f"_f{h}"].mean()
    return out


# -- bandas por rueda ----------------------------------------------------------

def bandas_de_ticker(high, low, volumen, ventana, rngs, paso=pp.PASO,
                     cobertura=pp.COBERTURA):
    """
    Para cada rueda: la banda del area de valor y las 2 x SEMILLAS de control.
    Devuelve {fuente: (val, vah)} con arrays de largo n, mas `ancho` y `saltos`.

    El perfil se calcula UNA vez por rueda y lo comparten todas las fuentes: las
    de control solo mueven la banda sobre el mismo perfil.
    """
    n = len(high)
    fuentes = ["poc"] + [f"{et}{s}" for _, et in MODOS for s in range(SEMILLAS)]
    val = {f: np.full(n, np.nan) for f in fuentes}
    vah = {f: np.full(n, np.nan) for f in fuentes}
    ancho = np.full(n, np.nan)
    saltos = 0

    for t in range(ventana - 1, n):
        ini = t + 1 - ventana
        k0, p = pp.perfil_bins(high[ini:t + 1], low[ini:t + 1], volumen[ini:t + 1], paso)
        if k0 is None or p.sum() <= 0:
            continue
        k, a, b = pp._area_valor(p, cobertura)
        nb = b - a + 1
        v, w = pe.banda_de_tramo(k0, a, nb, paso)
        val["poc"][t], vah["poc"][t] = v, w
        ancho[t] = (w / v - 1.0) * 100.0

        for modo, et in MODOS:
            cand = pe.posiciones_control(p, a, b, modo)
            if cand.size == 0:
                saltos += SEMILLAS
                continue
            for s in range(SEMILLAS):
                sel = int(rngs[(et, s)].choice(cand))
                cv, cw = pe.banda_de_tramo(k0, sel, nb, paso)
                val[f"{et}{s}"][t], vah[f"{et}{s}"][t] = cv, cw
    return val, vah, ancho, saltos


# -- corrida -------------------------------------------------------------------

def eventos_de_ticker(tk, d, ventana, uni_fwd):
    """Filas de evento de un ticker y una ventana, para todas las fuentes."""
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    close = d["close"].to_numpy(float)
    vol = d["volume"].to_numpy(float)
    fechas = d["fecha"].to_numpy()

    rngs = {(et, s): np.random.default_rng([SEED_BASE, abs(hash(tk)) % (2**31), i, s])
            for i, (_, et) in enumerate(MODOS) for s in range(SEMILLAS)}
    val, vah, ancho, saltos = bandas_de_ticker(high, low, vol, ventana, rngs)

    fwd_t = {h: pe.retorno_forward(close, h) for h in HORIZONTES}
    fwd_u = {h: pd.Series(uni_fwd[h]).reindex(pd.Index(fechas)).to_numpy(float)
             for h in HORIZONTES}

    filas = []
    for fuente in val:
        for e in pe.detectar_eventos(close, val[fuente], vah[fuente], FUERA):
            fila = {"ticker": tk, "fuente": fuente, "ventana": ventana,
                    "fecha": fechas[e.i], "direccion": e.direccion,
                    "ancho_pct": float(ancho[e.i - 1])}
            ok = False
            for h in HORIZONTES:
                rt, ru = fwd_t[h][e.i], fwd_u[h][e.i]
                fila[f"exceso_{h}"] = (float((rt - ru) * 100.0)
                                       if np.isfinite(rt) and np.isfinite(ru) else np.nan)
                ok = ok or np.isfinite(fila[f"exceso_{h}"])
            if ok:
                filas.append(fila)
    return filas, saltos


def resumir(ev: pd.DataFrame, ventana, h, area, direccion) -> dict:
    """Un bloque de resultados: POC contra la banda p5-p95 de cada control."""
    col = f"exceso_{h}"
    sel = ev[(ev["ventana"] == ventana) & (ev["direccion"] == direccion)]
    sel = sel[sel["ancho_pct"] < ANCHO_ANGOSTA] if area == "angosta" \
        else sel[sel["ancho_pct"] >= ANCHO_ANGOSTA]
    sel = sel[np.isfinite(sel[col])]

    out = {"ventana": ventana, "h": h, "area": area, "direccion": direccion}
    for periodo, mask in (("seleccion", sel["fecha"] < CORTE),
                          ("confirmacion", sel["fecha"] >= CORTE)):
        d = sel[mask]
        poc = d[d["fuente"] == "poc"]
        r = ic95_por_dia(poc[col], poc["fecha"])
        bloque = {"n_eventos": int(r["ops"]), "n_dias": int(r["dias"]),
                  "media_pp": r["media"], "ic_lo": r["lo"], "ic_hi": r["hi"]}
        for _, et in MODOS:
            medias = []
            for s in range(SEMILLAS):
                c = d[d["fuente"] == f"{et}{s}"]
                if len(c):
                    medias.append(ic95_por_dia(c[col], c["fecha"])["media"])
            medias = [m for m in medias if np.isfinite(m)]
            bloque[f"ctrl_{et}"] = {
                "n_semillas": len(medias),
                "n_eventos_medio": float(np.mean([
                    (d["fuente"] == f"{et}{s}").sum() for s in range(SEMILLAS)])),
                "p5": float(np.percentile(medias, 5)) if medias else np.nan,
                "p50": float(np.percentile(medias, 50)) if medias else np.nan,
                "p95": float(np.percentile(medias, 95)) if medias else np.nan,
                "posicion_poc": pe.fuera_de_banda(r["media"], medias),
            }
        out[periodo] = bloque
    return out


def veredicto(bloque: dict) -> dict:
    """Las tres condiciones de la sec. 4.7, por separado y en conjunto."""
    esperado = "arriba" if bloque["direccion"] == pe.DESDE_ARRIBA else "abajo"
    cond = {}
    for periodo in ("seleccion", "confirmacion"):
        b = bloque[periodo]
        fuera_b = b["ctrl_B"]["posicion_poc"] == esperado
        ic = b["ic_lo"], b["ic_hi"]
        excluye = np.isfinite(ic[0]) and np.isfinite(ic[1]) and (ic[0] > 0 or ic[1] < 0)
        signo = (b["media_pp"] > 0) if esperado == "arriba" else (b["media_pp"] < 0)
        cond[periodo] = {"fuera_del_control_B": bool(fuera_b),
                         "ic95_excluye_cero": bool(excluye),
                         "signo_esperado": bool(signo)}
    pasa = all(c["fuera_del_control_B"] and c["ic95_excluye_cero"] and c["signo_esperado"]
               for c in cond.values())
    return {"condiciones": cond, "PASA": bool(pasa)}


def texto(res: list) -> str:
    out = []
    for r in res:
        es_prim = (r["ventana"] == PRIMARIA["ventana"] and r["h"] == PRIMARIA["h"]
                   and r["area"] == PRIMARIA["area"])
        tag = "PRIMARIA" if es_prim else "secundaria"
        out.append(f"\n{'=' * 78}\n{tag} | ventana {r['ventana']} | h {r['h']} | "
                   f"area {r['area']} | {r['direccion']}")
        for periodo in ("seleccion", "confirmacion"):
            b = r[periodo]
            out.append(f"  {periodo:13s} POC  n={b['n_eventos']:6d} ({b['n_dias']:4d} dias)  "
                       f"media {b['media_pp']:+7.3f} pp  IC95 [{b['ic_lo']:+7.3f}; {b['ic_hi']:+7.3f}]")
            for _, et in MODOS:
                c = b[f"ctrl_{et}"]
                out.append(f"  {'':13s} ctrl {et}  n~{c['n_eventos_medio']:6.0f}            "
                           f"p5 {c['p5']:+7.3f}  p50 {c['p50']:+7.3f}  p95 {c['p95']:+7.3f}"
                           f"   -> POC {c['posicion_poc']}")
        v = r["veredicto"]
        out.append(f"  --> {'PASA' if v['PASA'] else 'NO PASA'}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="coma-separados, para una prueba corta")
    ap.add_argument("--dry-run", action="store_true", help="no escribe archivos")
    ap.add_argument("--etiqueta", default="poc")
    a = ap.parse_args()

    log("Cargando precios (LOCAL)...")
    df = cargar(get_engine())
    if a.tickers:
        df = df[df["ticker"].isin([t.strip().upper() for t in a.tickers.split(",")])]
    log(f"  {df['ticker'].nunique()} tickers, {len(df):,} filas")

    # universo pre-registrado: historia suficiente ANTES del corte
    pre = df[df["fecha"] < CORTE].groupby("ticker").size()
    elegibles = set(pre[pre >= MIN_RUEDAS_PRE_CORTE].index)
    df = df[df["ticker"].isin(elegibles)]
    log(f"  universo con >= {MIN_RUEDAS_PRE_CORTE} ruedas pre-{CORTE}: {len(elegibles)}")
    if not elegibles:
        log("Sin tickers elegibles."); return 1

    log("Retorno equal-weight del universo por rueda...")
    uni = universo_forward(df, HORIZONTES)

    filas, saltos = [], 0
    for ventana in VENTANAS:
        log(f"Ventana {ventana}: perfiles + {1 + 2 * SEMILLAS} fuentes por ticker...")
        for i, (tk, d) in enumerate(df.groupby("ticker", sort=True), 1):
            f, s = eventos_de_ticker(tk, d.sort_values("fecha"), ventana, uni)
            filas.extend(f); saltos += s
            if i % 40 == 0:
                log(f"  {i}/{len(elegibles)}")
    ev = pd.DataFrame(filas)
    log(f"\nEventos: {len(ev):,} ({(ev['fuente'] == 'poc').sum():,} del POC). "
        f"Ruedas-semilla sin posicion de control: {saltos:,}")

    res = []
    for ventana in VENTANAS:
        for h in HORIZONTES:
            for area in ("angosta", "ancha"):
                for direccion in (pe.DESDE_ARRIBA, pe.DESDE_ABAJO):
                    r = resumir(ev, ventana, h, area, direccion)
                    r["veredicto"] = veredicto(r)
                    res.append(r)
    # las primarias primero
    res.sort(key=lambda r: (not (r["ventana"] == PRIMARIA["ventana"]
                                 and r["h"] == PRIMARIA["h"]
                                 and r["area"] == PRIMARIA["area"]),
                            r["ventana"], r["h"], r["area"], r["direccion"]))

    salida = texto(res)
    print(salida)

    prim = [r for r in res
            if r["ventana"] == PRIMARIA["ventana"] and r["h"] == PRIMARIA["h"]
            and r["area"] == PRIMARIA["area"]]
    pasan = [r["direccion"] for r in prim if r["veredicto"]["PASA"]]
    log(f"\n{'=' * 78}\nPRIMARIAS que pasan: {pasan if pasan else 'NINGUNA'}")

    if not a.dry_run:
        dst = os.path.join("reportes", "poc_vs_azar",
                           f"{datetime.now():%Y%m%d}_{a.etiqueta}")
        os.makedirs(dst, exist_ok=True)
        ev.to_csv(os.path.join(dst, "eventos.csv"), index=False)
        with open(os.path.join(dst, "resultado.json"), "w", encoding="utf-8") as fh:
            json.dump({"parametros": {"ventanas": VENTANAS, "horizontes": HORIZONTES,
                                      "ancho_angosta": ANCHO_ANGOSTA, "fuera": FUERA,
                                      "semillas": SEMILLAS, "corte": str(CORTE),
                                      "min_ruedas_pre_corte": MIN_RUEDAS_PRE_CORTE,
                                      "seed_base": SEED_BASE, "primaria": PRIMARIA,
                                      "tickers": len(elegibles)},
                       "resultados": res}, fh, indent=2, default=str)
        with open(os.path.join(dst, "resumen.txt"), "w", encoding="utf-8") as fh:
            fh.write(salida)
        log(f"\nEscrito en {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
