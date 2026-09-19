"""
auditar_features_tablas.py
Audita features_velas, features_estructura y features_precio_accion contra el OHLCV
de precios_diarios. Solo LEE. Correrlo despues de un backfill, de una correccion de
split o de tocar velas.py / estructura.py / precio_accion.py.

POR QUE EXISTE (docs/estructura_velas.md sec. 12)
    "La tabla reproduce el codigo" y "el codigo calcula lo correcto" son dos preguntas
    distintas, y la segunda no la contesta ningun test unitario sobre datos sinteticos:
    hace falta chequear la definicion contra el OHLCV real, con una implementacion
    INDEPENDIENTE del modulo. Asi aparecio que el 71% de las envolventes de
    features_precio_accion no envuelven.

QUE MIDE
    A. REPRODUCIBILIDAD: recomputa las 3 tablas con su modulo desde precios_diarios y
       compara contra lo persistido. Detecta datos rancios, escala de split vieja y
       errores de persistencia.
    B. DEFINICIONES DE VELAS: reimplementadas a mano (sin importar velas.py ni
       precio_accion.py) y aplicadas sobre las filas marcadas en cada tabla.
    C. ESTRUCTURA: el swing marcado es un swing real, dias_* en rango, BOS y CHoCH
       excluyentes, e INVARIANCIA numerica (la fila de t no cambia con barras nuevas).
    D. VALORES INVENTADOS en features_precio_accion: tendencia_velas = -5 y
       rango_expansion = 0 donde no hay dato para calcularlos (costuras de backfill).

CODIGO DE SALIDA
    0  features_velas y features_estructura pasan todos sus chequeos y las 3 tablas
       reproducen.
    1  algo que DEBE pasar fallo.
    Los defectos de features_precio_accion son CONOCIDOS y documentados: se informan
    con su porcentaje pero no cambian el codigo de salida. Sus patrones estan
    reemplazados por features_velas.

Uso:
    python scripts/manual/auditar_features_tablas.py
    python scripts/manual/auditar_features_tablas.py --tickers 40 --semilla 7
    python scripts/manual/auditar_features_tablas.py --tickers KLAC,CRWD,AAPL
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# LOCAL es la fuente de verdad: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.database import query_df  # noqa: E402
from src.indicators import estructura as est  # noqa: E402
from src.indicators.precio_accion import _ALL_FEAT_COLS, _calcular_grupo  # noqa: E402
from src.indicators.velas import COLUMNAS as COLS_VELAS  # noqa: E402
from src.indicators.velas import calcular_velas  # noqa: E402

TOL_NUM = 1.5e-4          # las columnas de precio_accion son NUMERIC(x,4)
TOL_EST = 1e-6            # features_estructura guarda double precision
EPS = 1e-9
MIN_RUEDAS = 600
SIEMPRE = ("KLAC", "CRWD")   # los dos tickers con split corregido (escala)
CORTES_INVARIANCIA = 6


def log(msg=""):
    print(msg, flush=True)


class Chequeos:
    """Acumula (cumple, total) por chequeo y si el chequeo es obligatorio."""

    def __init__(self):
        self.d = {}

    def add(self, clave, cumple, total, obligatorio=True):
        c, t, _ = self.d.get(clave, (0, 0, obligatorio))
        self.d[clave] = (c + int(cumple), t + int(total), obligatorio)

    def fallas_obligatorias(self):
        return [k for k, (c, t, o) in self.d.items() if o and t > 0 and c < t]

    def imprimir(self, prefijo):
        log(f"  {'chequeo':<52} {'cumple':>9} {'de':>9} {'%':>7}")
        for k in sorted(x for x in self.d if x.startswith(prefijo)):
            c, t, o = self.d[k]
            marca = "" if (not o or c == t) else "  <-- FALLA"
            log(f"  {k:<52} {c:>9,} {t:>9,} {100.0 * c / max(t, 1):>6.1f}%{marca}")


def elegir_tickers(arg: str, semilla: int) -> list:
    if not arg.isdigit():
        return sorted({t.strip().upper() for t in arg.split(",") if t.strip()})
    uni = query_df("""SELECT ticker FROM precios_diarios GROUP BY ticker
                      HAVING COUNT(*) >= :m ORDER BY ticker""", params={"m": MIN_RUEDAS})
    rng = np.random.default_rng(semilla)
    base = rng.choice(uni["ticker"].to_numpy(), size=min(int(arg), len(uni)), replace=False)
    return sorted(set(base.tolist()) | set(SIEMPRE))


def cargar_precios(tk: str) -> pd.DataFrame:
    px = query_df("""SELECT fecha, open, high, low, close, volume FROM precios_diarios
                     WHERE ticker = :t ORDER BY fecha""", params={"t": tk})
    px["fecha"] = pd.to_datetime(px["fecha"])
    return px


def cargar_tabla(tabla: str, cols: list, tk: str) -> pd.DataFrame:
    db = query_df(f"SELECT fecha, {', '.join(cols)} FROM {tabla} WHERE ticker = :t ORDER BY fecha",
                  params={"t": tk})
    db["fecha"] = pd.to_datetime(db["fecha"])
    return db.set_index("fecha")


def comparar(calc: pd.DataFrame, db: pd.DataFrame, cols: list, tol: float, ch: Chequeos, tabla: str):
    com = calc.index.intersection(db.index)
    for c in cols:
        a = pd.to_numeric(calc.loc[com, c], errors="coerce")
        b = pd.to_numeric(db.loc[com, c], errors="coerce")
        mal = int((((a - b).abs() > tol) | (a.isna() != b.isna())).sum())
        ch.add(f"A.{tabla} reproduce (celdas fila x columna)", len(com) - mal, len(com))


# --- A. reproducibilidad ------------------------------------------------------

def auditar_reproducibilidad(tk, px, ch):
    comparar(calcular_velas(px).set_index("fecha"),
             cargar_tabla("features_velas", COLS_VELAS, tk), COLS_VELAS, 0.5, ch, "features_velas")

    cols_e = est.columnas(est.VENTANAS_TABLA)
    comparar(est.calcular_estructura(px, ventanas=est.VENTANAS_TABLA).set_index("fecha"),
             cargar_tabla("features_estructura", cols_e, tk), cols_e, TOL_EST, ch,
             "features_estructura")

    # precio_accion se calcula sobre el JOIN con indicadores_tecnicos (atr14, vol_relativo)
    ind = query_df("""SELECT fecha, atr14, vol_relativo FROM indicadores_tecnicos
                      WHERE ticker = :t ORDER BY fecha""", params={"t": tk})
    ind["fecha"] = pd.to_datetime(ind["fecha"])
    base = px.merge(ind, on="fecha", how="inner").assign(ticker=tk)
    comparar(_calcular_grupo(base).set_index("fecha"),
             cargar_tabla("features_precio_accion", list(_ALL_FEAT_COLS), tk),
             list(_ALL_FEAT_COLS), TOL_NUM, ch, "features_precio_accion")


# --- B. definiciones de velas (implementacion independiente) -----------------

def _anterior(x, k=1):
    return np.concatenate([np.full(k, np.nan), np.asarray(x, dtype=float)[:-k]])


def auditar_definiciones(tk, px, ch):
    o, h, lo, c = (px[x].astype(float).to_numpy() for x in ("open", "high", "low", "close"))
    rango = np.where(h - lo > 0, h - lo, np.nan)
    cuerpo = np.abs(c - o) / rango
    s_sup = (h - np.maximum(o, c)) / rango
    s_inf = (np.minimum(o, c) - lo) / rango
    alc, baj = c > o, c < o
    c1, o1 = _anterior(c), _anterior(o)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret5 = c1 / _anterior(c, 6) - 1.0
    pos = pd.Series(np.arange(len(px)), index=px["fecha"])

    def filas(db, col):
        i = pos.reindex(db.index[db[col].to_numpy() == 1]).dropna().astype(int).to_numpy()
        return i

    with np.errstate(invalid="ignore"):
        # features_velas: la definicion clasica del docstring de velas.py
        v = cargar_tabla("features_velas", COLS_VELAS, tk)
        j = filas(v, "patron_engulfing_bull")
        ch.add("B.velas engulfing_bull envuelve de verdad",
               (alc[j] & (c1[j] < o1[j]) & (o[j] <= c1[j]) & (c[j] >= o1[j])
                & ((o[j] < c1[j]) | (c[j] > o1[j]))).sum(), j.size)
        j = filas(v, "patron_engulfing_bear")
        ch.add("B.velas engulfing_bear envuelve de verdad",
               (baj[j] & (c1[j] > o1[j]) & (o[j] >= c1[j]) & (c[j] <= o1[j])
                & ((o[j] > c1[j]) | (c[j] < o1[j]))).sum(), j.size)
        j = filas(v, "patron_hammer")
        ch.add("B.velas hammer cumple la forma",
               ((cuerpo[j] <= 0.30 + EPS) & (s_inf[j] + EPS >= 2 * cuerpo[j])
                & (s_sup[j] <= 0.10 + EPS)).sum(), j.size)
        ch.add("B.velas hammer viene tras una CAIDA", (ret5[j] < 0).sum(), j.size)
        j = filas(v, "patron_hanging_man")
        ch.add("B.velas hanging_man viene tras una SUBA", (ret5[j] > 0).sum(), j.size)
        j = filas(v, "patron_marubozu_bull")
        ch.add("B.velas marubozu_bull es alcista", alc[j].sum(), j.size)
        formas = v[["patron_doji", "patron_hammer", "patron_hanging_man",
                    "patron_shooting_star", "patron_inverted_hammer"]].to_numpy().sum(axis=1)
        ch.add("B.velas una sola etiqueta de forma por vela", (formas <= 1).sum(), len(v))

        # features_precio_accion: MISMOS chequeos, defectos CONOCIDOS (no obligatorios)
        p = cargar_tabla("features_precio_accion",
                         ["patron_engulfing_bull", "patron_engulfing_bear", "patron_hammer",
                          "patron_doji", "patron_shooting_star", "patron_marubozu"], tk)
        j = filas(p, "patron_engulfing_bull")
        ch.add("B.precio_accion engulfing_bull envuelve de verdad",
               (alc[j] & (c1[j] < o1[j]) & (o[j] <= c1[j]) & (c[j] >= o1[j])).sum(), j.size, False)
        j = filas(p, "patron_engulfing_bear")
        ch.add("B.precio_accion engulfing_bear envuelve de verdad",
               (baj[j] & (c1[j] > o1[j]) & (o[j] >= c1[j]) & (c[j] <= o1[j])).sum(), j.size, False)
        j = filas(p, "patron_hammer")
        ch.add("B.precio_accion hammer con sombra superior <= 10%",
               (s_sup[j] <= 0.10 + EPS).sum(), j.size, False)
        ch.add("B.precio_accion hammer tras CAIDA (no hanging man)", (ret5[j] < 0).sum(), j.size, False)
        dj = p["patron_doji"].to_numpy() == 1
        sol = dj & ((p["patron_hammer"].to_numpy() == 1) | (p["patron_shooting_star"].to_numpy() == 1))
        ch.add("B.precio_accion doji sin doble etiqueta", (dj & ~sol).sum(), dj.sum(), False)
        ch.add("B.precio_accion marubozu distingue direccion", 0,
               int((p["patron_marubozu"].to_numpy() == 1).sum()), False)


# --- C. estructura ------------------------------------------------------------

def auditar_estructura(tk, px, ch, rng):
    h = px["high"].astype(float).to_numpy()
    lo = px["low"].astype(float).to_numpy()
    pos = pd.Series(np.arange(len(px)), index=px["fecha"])
    for n in est.VENTANAS_TABLA:
        db = cargar_tabla("features_estructura",
                          [f"is_sh_{n}", f"is_sl_{n}", f"dias_sh_{n}", f"bos_bull_{n}",
                           f"choch_bull_{n}", f"bos_bear_{n}", f"choch_bear_{n}"], tk)
        for tipo, serie, cmp_izq, cmp_der in (("sh", h, np.greater, np.greater_equal),
                                              ("sl", lo, np.less, np.less_equal)):
            i = pos.reindex(db.index[db[f"is_{tipo}_{n}"].to_numpy() == 1]).dropna().astype(int).to_numpy()
            i = i[i >= 2 * n]
            p = i - n                      # la barra del swing
            ext = np.max if tipo == "sh" else np.min
            izq = np.array([ext(serie[x - n:x]) for x in p])
            der = np.array([ext(serie[x + 1:x + n + 1]) for x in p])
            ok = cmp_izq(serie[p], izq) & cmp_der(serie[p], der) if p.size else np.array([])
            ch.add(f"C.estructura is_{tipo}_{n} marca un swing REAL", ok.sum(), p.size)
        d = pd.to_numeric(db[f"dias_sh_{n}"], errors="coerce").dropna().to_numpy()
        ch.add(f"C.estructura dias_sh_{n} entre {n} y {est.TOPE_DIAS_DIARIO}",
               ((d >= n) & (d <= est.TOPE_DIAS_DIARIO)).sum(), d.size)
        for lado in ("bull", "bear"):
            s = db[f"bos_{lado}_{n}"].to_numpy() + db[f"choch_{lado}_{n}"].to_numpy()
            ch.add(f"C.estructura bos_{lado}/choch_{lado} excluyentes N={n}", (s <= 1).sum(), len(db))

    # invariancia: la fila de t calculada con datos[:t+1] == la de la serie completa
    completo = est.calcular_estructura(px, ventanas=est.VENTANAS_TABLA).set_index("fecha")
    cols = est.columnas(est.VENTANAS_TABLA)
    if len(px) > 300:
        for t in rng.choice(range(300, len(px)), size=min(CORTES_INVARIANCIA, len(px) - 300),
                            replace=False):
            parcial = est.calcular_estructura(px.iloc[:t + 1], ventanas=est.VENTANAS_TABLA).iloc[-1]
            ref = completo.loc[px["fecha"].iloc[t]]
            igual = all((pd.isna(ref[c]) and pd.isna(parcial[c]))
                        or abs(float(ref[c]) - float(parcial[c])) < EPS for c in cols)
            ch.add("C.estructura INVARIANCIA (la fila de t no cambia)", igual, 1)


# --- D. valores inventados (toda la tabla) ------------------------------------

def auditar_inventados():
    log("")
    log("-" * 100)
    log("D. VALORES INVENTADOS en features_precio_accion (toda la tabla; defecto CONOCIDO)")
    log("-" * 100)
    x = query_df("""SELECT COUNT(*) n,
        SUM(CASE WHEN velas_alcistas_5d IS NULL AND tendencia_velas = -5 THEN 1 ELSE 0 END) tv,
        SUM(CASE WHEN velas_alcistas_5d IS NULL AND rango_expansion = 0 THEN 1 ELSE 0 END) re,
        SUM(CASE WHEN velas_alcistas_5d IS NOT NULL
                  AND tendencia_velas = 2 * velas_alcistas_5d - 5 THEN 1 ELSE 0 END) afin,
        SUM(CASE WHEN velas_alcistas_5d IS NOT NULL THEN 1 ELSE 0 END) con_dato
        FROM features_precio_accion""").iloc[0]
    n = int(x["n"])
    log(f"  tendencia_velas = -5 sin dato para calcularlo : {int(x['tv']):>6,} filas "
        f"({100.0 * int(x['tv']) / n:.3f}%)")
    log(f"  rango_expansion = 0 sin dato para calcularlo  : {int(x['re']):>6,} filas "
        f"({100.0 * int(x['re']) / n:.3f}%)")
    log(f"  tendencia_velas == 2*velas_alcistas_5d - 5    : {int(x['afin']):,} de "
        f"{int(x['con_dato']):,} ({100.0 * int(x['afin']) / max(int(x['con_dato']), 1):.2f}%) "
        "-> transformacion afin, cero informacion")
    f = query_df("""SELECT fecha::text f, COUNT(*) n FROM features_precio_accion
                    WHERE velas_alcistas_5d IS NULL AND tendencia_velas = -5
                    GROUP BY 1 ORDER BY 2 DESC LIMIT 6""")
    nf = query_df("""SELECT COUNT(DISTINCT fecha) f FROM features_precio_accion
                     WHERE velas_alcistas_5d IS NULL AND tendencia_velas = -5""").iloc[0]["f"]
    log(f"  repartidas en {int(nf)} fechas (costuras de cada backfill, no el arranque de la serie):")
    for _, r in f.iterrows():
        log(f"    {r['f']}  {int(r['n']):>4} tickers")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audita las 3 tablas de features contra el OHLCV")
    ap.add_argument("--tickers", default="25",
                    help="cantidad al azar (se suman KLAC y CRWD) o lista separada por coma")
    ap.add_argument("--semilla", type=int, default=23)
    args = ap.parse_args()

    tks = elegir_tickers(args.tickers, args.semilla)
    rng = np.random.default_rng(args.semilla + 1)
    log("=" * 100)
    log("AUDITORIA DE features_velas / features_estructura / features_precio_accion vs OHLCV")
    log("=" * 100)
    log(f"  {len(tks)} tickers: {', '.join(tks)}")

    ch = Chequeos()
    for tk in tks:
        px = cargar_precios(tk)
        if len(px) < 30:
            log(f"  {tk}: sin precios suficientes, se omite")
            continue
        auditar_reproducibilidad(tk, px, ch)
        auditar_definiciones(tk, px, ch)
        auditar_estructura(tk, px, ch, rng)

    for pref, tit in (("A.", "A. REPRODUCIBILIDAD (recomputar desde OHLCV vs lo guardado)"),
                      ("B.velas", "B. DEFINICIONES -- features_velas (deben dar 100%)"),
                      ("B.precio_accion", "B. DEFINICIONES -- features_precio_accion (defectos CONOCIDOS)"),
                      ("C.", "C. ESTRUCTURA (deben dar 100%)")):
        log("")
        log("-" * 100)
        log(tit)
        log("-" * 100)
        ch.imprimir(pref)
    auditar_inventados()

    fallas = ch.fallas_obligatorias()
    log("")
    log("=" * 100)
    if fallas:
        log(f"RESULTADO: {len(fallas)} chequeo(s) obligatorio(s) FALLAN:")
        for k in fallas:
            log(f"  - {k}")
        return 1
    log("RESULTADO: OK. features_velas y features_estructura son correctas y las 3 tablas")
    log("reproducen desde el OHLCV. Los defectos de features_precio_accion son los conocidos")
    log("(docs/estructura_velas.md sec. 12): sus patrones no se usan para decisiones nuevas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
