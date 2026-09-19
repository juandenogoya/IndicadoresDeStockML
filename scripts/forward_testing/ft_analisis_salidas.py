"""
ft_analisis_salidas.py
Analisis de las SALIDAS de las estrategias FT (docs/forward_testing/ANALISIS_SALIDAS.md).
Solo LEE (DB local). Guarda cada corrida en un directorio con fecha:
reportes/analisis_salidas/AAAAMMDD_<etiqueta>/  (log + CSV + parametros.json).

POR QUE EXISTE
    Las entradas no distinguen (ML en AUC 0,51; las senales de entrada de SMC solas no
    le ganan al universo) y lo unico que paso un backtest fue SMC CON sus salidas. Si la
    entrada vale poco, la salida arma el resultado. Supuesto de trabajo: la entrada es
    correcta y no se toca; la pregunta es si salir ese dia fue mejor que seguir adentro.

SECCIONES (--seccion; por defecto todas)
    panorama        que hizo el precio despues de cada salida real, por estrategia y
                    tipo de salida, contra el universo y en desvios del propio ticker;
                    referencia: salir un dia cualquiera al azar
    balances        las salidas por balance, eventos UNICOS, contra todos los balances
                    del universo en el mismo tramo
    tech_sector_v1  anatomia de la regla (combinaciones), la ventana del bug del score
                    0,0, motivos y que condicion disparo cada salida
    p0              (--seccion p0p1) fidelidad: el score del analisis contra el que vio el
                    bot, las entradas de FT y del motor, y la regla ACTUAL re-simulada sobre
                    las entradas reales de FT contra sus salidas reales
    p1              (--seccion p0p1) PASO 1 pre-registrado: entradas del motor 2021-2026
                    (en memoria, sin escribir en bt_hist_*) fijas, salida re-simulada con
                    cada variante, regla de lectura; y lo mismo sobre las entradas reales
                    del FT de control (diagnostico)
    p2              PASO 2 pre-registrado (sec. 8): grilla de pesos, 589 reglas de salida
                    distintas re-simuladas sobre las mismas entradas; la medicion principal
                    es lo que hizo el precio DESPUES de la salida; seleccion 2021-2024,
                    confirmacion 2025-2026 y FT de control

VENTANAS QUE NO MIDEN LA REGLA (se excluyen)
    - Salidas antes del arreglo `fix_score_cero_salida` (ft_cambios, 29/5/2026) en las
      estrategias que toco (4, 6, 8, 9): la consulta de salida no traia el close y el
      score daba 0 todos los dias.
    - motivo ESTRATEGIA_DISCONTINUADA (salida artificial) y *_SPLIT_FIX (incidente de
      splits del 21/7).

Uso:
    python scripts/forward_testing/ft_analisis_salidas.py --etiqueta exploratorio
    python scripts/forward_testing/ft_analisis_salidas.py --seccion tech_sector_v1
"""

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# LOCAL es la fuente de verdad: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.database import query_df  # noqa: E402
from src.utils import ft_salidas as fs  # noqa: E402

HORIZONTES = (5, 10, 20)
H_CLASIFICAR = 10
VOL_RUEDAS = 60                 # ruedas para la volatilidad diaria del ticker hasta D
MIN_OPS_TABLA = 8
CAMBIOS_QUE_INVALIDAN = ("fix_score_cero_salida",)
ID_TECH_SECTOR_V1 = 4
DIR_BASE = os.path.join(ROOT, "reportes", "analisis_salidas")


class Salida:
    """print + archivo. Un log por seccion."""

    def __init__(self, ruta):
        self.f = open(ruta, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")

    def titulo(self, txt):
        self("")
        self("=" * 104)
        self(txt)
        self("=" * 104)

    def cerrar(self):
        self.f.close()


def _git():
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return c + (" (con cambios sin commit)" if sucio else "")
    except Exception:
        return "desconocido"


# --- datos ----------------------------------------------------------------------

def cargar_precios(desde="2026-01-01"):
    px = query_df("SELECT ticker, fecha, close FROM precios_diarios WHERE fecha >= :d",
                  params={"d": desde})
    px["fecha"] = pd.to_datetime(px["fecha"])
    P = px.pivot(index="fecha", columns="ticker", values="close").sort_index().astype(float)
    sig = P.pct_change().rolling(VOL_RUEDAS, min_periods=40).std()
    fwd = {h: P.shift(-h) / P - 1.0 for h in HORIZONTES}
    uni = {h: fwd[h].mean(axis=1) for h in HORIZONTES}
    return P, sig, fwd, uni


def ventanas_invalidas():
    """{estrategia_id: fecha_efectiva} de los cambios que invalidan la historia previa."""
    c = query_df("SELECT clave, fecha_efectiva, estrategias FROM ft_cambios WHERE clave = ANY(:k)",
                 params={"k": list(CAMBIOS_QUE_INVALIDAN)})
    out = {}
    for r in c.itertuples():
        for e in r.estrategias:
            f = pd.Timestamp(r.fecha_efectiva)
            out[int(e)] = max(out.get(int(e), f), f)
    return out


def cargar_salidas():
    op = query_df("""SELECT o.id, o.estrategia_id, e.nombre, o.ticker, o.fecha_datos,
                            o.fecha_datos_salida, o.motivo_salida, o.pnl_pct
                     FROM ft_operaciones o JOIN ft_estrategias e ON e.id = o.estrategia_id
                     WHERE o.fecha_salida IS NOT NULL""")
    for c in ("fecha_datos", "fecha_datos_salida"):
        op[c] = pd.to_datetime(op[c])
    op["pnl_pct"] = op["pnl_pct"].astype(float)
    op["familia"] = op["motivo_salida"].map(fs.familia_salida)
    inval = ventanas_invalidas()
    corte = op["estrategia_id"].map(inval)
    op["invalida_bug"] = corte.notna() & (op["fecha_datos_salida"] < corte)
    op["artificial"] = op["motivo_salida"].eq("ESTRATEGIA_DISCONTINUADA") | op["familia"].eq("SPLIT_FIX")
    return op, inval


def medir_post(op, P, sig, fwd, uni):
    """Agrega ex{h} (exceso en %) y z{h} a cada salida valida."""
    filas = []
    for r in op.itertuples():
        d = {"id": r.id}
        D, t = r.fecha_datos_salida, r.ticker
        ok = t in P.columns and D in P.index
        s = sig.at[D, t] if ok else np.nan
        for h in HORIZONTES:
            ex, z = (fs.exceso_z(fwd[h].at[D, t], uni[h].at[D], s, h) if ok else (None, None))
            d[f"ex{h}"] = ex * 100 if ex is not None else np.nan
            d[f"z{h}"] = z if z is not None else np.nan
        filas.append(d)
    return op.merge(pd.DataFrame(filas), on="id", how="left")


def _fmt_ic(r):
    if np.isnan(r["lo"]):
        return f"{r['media']:+6.2f} [   sin IC    ]"
    return f"{r['media']:+6.2f} [{r['lo']:+6.2f};{r['hi']:+6.2f}]"


# --- secciones ------------------------------------------------------------------

def seccion_panorama(op, P, sig, fwd, uni, out, dirout):
    out.titulo("PANORAMA -- que hizo el precio DESPUES de cada salida real de FT")
    val = op[~op["invalida_bug"] & ~op["artificial"]].copy()
    out(f"  salidas cerradas: {len(op):,} | excluidas por el bug del score 0,0: "
        f"{int(op['invalida_bug'].sum()):,} | artificiales: {int(op['artificial'].sum()):,} | "
        f"validas: {len(val):,}")
    val = medir_post(val, P, sig, fwd, uni)
    for h in HORIZONTES:
        out(f"  con ventana completa a {h:>2} ruedas: {int(val[f'ex{h}'].notna().sum()):,}")
    out("")
    out(f"  exceso = retorno del ticker menos el universo equal-weight, misma ventana, en %.")
    out(f"  IC95 con el DIA de salida como unidad. Clasificacion a {H_CLASIFICAR} ruedas, en "
        f"desvios del ticker: 'a tiempo' < -{fs.UMBRAL_Z} | 'temprano' > +{fs.UMBRAL_Z}.")
    out(f"  {'estrategia':<27} {'salida':<12} {'ops':>4} {'dias':>4} {'ex5':>6} "
        f"{'ex10 [IC95 por dia]':>23} {'ex20':>6} {'a tiempo':>9} {'temprano':>9} {'indif':>6}")
    for (e, f), g in val.groupby(["nombre", "familia"]):
        gh = g.dropna(subset=[f"z{H_CLASIFICAR}"])
        if len(gh) < MIN_OPS_TABLA:
            continue
        ic = fs.ic95_por_dia(gh[f"ex{H_CLASIFICAR}"], gh["fecha_datos_salida"])
        cl = gh[f"z{H_CLASIFICAR}"].map(fs.clasificar)
        out(f"  {e:<27} {f:<12} {len(gh):>4} {ic['dias']:>4} {g['ex5'].mean():>+6.2f} "
            f"{_fmt_ic(ic):>23} {g['ex20'].mean():>+6.2f} "
            f"{100 * (cl == 'a_tiempo').mean():>8.0f}% {100 * (cl == 'temprano').mean():>8.0f}% "
            f"{100 * (cl == 'indiferente').mean():>5.0f}%")

    # referencia: salir un dia cualquiera, todo el universo, mismo tramo
    h = H_CLASIFICAR
    desde = val["fecha_datos_salida"].min()
    ex = fwd[h].sub(uni[h], axis=0)
    z = (ex / (sig * np.sqrt(h))).loc[desde:].to_numpy().ravel()
    z = z[~np.isnan(z)]
    out("")
    out(f"  REFERENCIA, salir al azar (todo el universo, todos los dias desde {desde:%Y-%m-%d}, "
        f"n={len(z):,}): a tiempo {100 * (z < -fs.UMBRAL_Z).mean():.0f}% | temprano "
        f"{100 * (z > fs.UMBRAL_Z).mean():.0f}% | indiferente "
        f"{100 * (np.abs(z) <= fs.UMBRAL_Z).mean():.0f}%")
    val.drop(columns=["invalida_bug", "artificial"]).to_csv(
        os.path.join(dirout, "panorama_salidas.csv"), index=False)
    out(f"  detalle por salida: {os.path.join(dirout, 'panorama_salidas.csv')}")


def seccion_balances(op, P, fwd, uni, out):
    out.titulo("BALANCES -- las salidas por balance, contadas como EVENTOS UNICOS")
    h = H_CLASIFICAR
    ex = fwd[h].sub(uni[h], axis=0) * 100
    b = op[(op["familia"] == "BALANCE") & ~op["invalida_bug"]]
    ev = b[["ticker", "fecha_datos_salida"]].drop_duplicates()
    v = np.array([ex.at[r.fecha_datos_salida, r.ticker] for r in ev.itertuples()
                  if r.fecha_datos_salida in ex.index and r.ticker in ex.columns])
    v = v[~np.isnan(v)]
    out(f"  salidas por balance: {len(b)} operaciones = {len(ev)} eventos unicos (ticker, rueda)")
    out(f"  exceso a {h} ruedas de esos eventos : media {v.mean():+.2f}% | mediana "
        f"{np.median(v):+.2f}% | desvio {v.std():.2f}")

    eh = query_df("""SELECT ticker, announcement_date a, report_time FROM earnings_historico
                     WHERE announcement_date BETWEEN :d AND :h""",
                  params={"d": str(b["fecha_datos_salida"].min())[:10],
                          "h": str(P.index[-h - 1])[:10]})
    eh["a"] = pd.to_datetime(eh["a"])
    idx, w = P.index, []
    for r in eh.itertuples():
        if r.ticker not in P.columns:
            continue
        pos = idx.searchsorted(r.a)
        pre = pos - 1 if str(r.report_time).lower().startswith("pre") else pos
        if 0 <= pre < len(idx):
            x = ex.at[idx[pre], r.ticker]
            if not np.isnan(x):
                w.append(x)
    w = np.array(w)
    out(f"  TODOS los balances del universo en el mismo tramo ({len(w)}), desde la rueda previa "
        f"a la reaccion: media {w.mean():+.2f}% | mediana {np.median(w):+.2f}% | desvio {w.std():.2f}")
    out("  -> si las medianas se parecen, lo que parece 'salio a tiempo' es la temporada de")
    out("     balances, no la regla. Esta salida se juzga por el riesgo que evita (el desvio).")


def seccion_tech_sector_v1(op, inval, out, dirout):
    out.titulo("TECH_SECTOR_v1 -- anatomia de la salida por score")
    out(f"  Umbrales del bot: entrada score >= {fs.SCORE_ENTRADA_V1} | salida score <= "
        f"{fs.SCORE_SALIDA_V1}. Pesos de src/strategies/scoring.py.")
    out(f"  Valores posibles del score: {fs.valores_posibles()}")
    out("  -> no hay ningun score entre 3,5 y 4,0: salir con <= 3,5 es exactamente dejar de")
    out("     cumplir la entrada. Entrada = SMA200 y SMA50 y 2 de 3 entre SMA21 / MACD / RSI.")
    out("")
    out(f"  {'SMA200 SMA50 SMA21 MACD RSI':<30} {'score':>6} {'entra':>6} "
        + " ".join(f"{('sale ' + v):>15}" for v in fs.VARIANTES))
    vistos = set()
    for c in sorted(fs.combinaciones(), key=lambda c: (-fs.score(c), fs.etiqueta_estado(c))):
        if not c["sma200"]:
            k = "(debajo de SMA200)"
            if k in vistos:
                continue
            vistos.add(k)
        else:
            k = fs.etiqueta_estado(c)
        out(f"  {k:<30} {fs.score(c):>6.1f} {('si' if fs.cumple_entrada(c) else '-'):>6} "
            + " ".join(f"{('SALE' if fs.sale(c, v) else '-'):>15}" for v in fs.VARIANTES))

    ts = op[op["estrategia_id"] == ID_TECH_SECTOR_V1].copy()
    fix = inval.get(ID_TECH_SECTOR_V1)
    ind = query_df("""SELECT i.ticker, i.fecha, p.close, i.sma21, i.sma50, i.sma200,
                             i.rsi14, i.macd, i.macd_signal
                      FROM indicadores_tecnicos i
                      JOIN precios_diarios p ON p.ticker = i.ticker AND p.fecha = i.fecha
                      WHERE i.fecha >= :d ORDER BY i.ticker, i.fecha""",
                   params={"d": str(ts["fecha_datos"].min() - pd.Timedelta(days=15))[:10]})
    ind["fecha"] = pd.to_datetime(ind["fecha"])
    ind["prev"] = ind.groupby("ticker")["fecha"].shift(1)
    cond, prev_de = {}, {}
    for r in ind.itertuples():
        cond[(r.ticker, r.fecha)] = fs.condiciones_desde_fila(
            {"close": r.close, "sma21": r.sma21, "sma50": r.sma50, "sma200": r.sma200,
             "rsi14": r.rsi14, "macd": r.macd, "macd_signal": r.macd_signal})
        prev_de[(r.ticker, r.fecha)] = r.prev

    out.titulo("TECH_SECTOR_v1 -- la ventana del bug del score 0,0 (fix_score_cero_salida)")
    cal = pd.Series(np.arange(len(ind["fecha"].unique())), index=np.sort(ind["fecha"].unique()))
    ts["ruedas"] = [cal.get(b, np.nan) - cal.get(a, np.nan)
                    for a, b in zip(ts["fecha_datos"], ts["fecha_datos_salida"])]
    cero = ts[ts["motivo_salida"] == "SCORE_DEGRADADO_0.0"]
    for nom, g in (("antes del fix", cero[cero["fecha_datos_salida"] < fix]),
                   ("despues del fix", cero[cero["fecha_datos_salida"] >= fix])):
        sc = np.array([fs.score(cond[(r.ticker, r.fecha_datos_salida)]) for r in g.itertuples()
                       if (r.ticker, r.fecha_datos_salida) in cond])
        out(f"  {nom:<16} salidas 'SCORE_0.0': {len(g):>4} | score REAL >= 4 (no debian salir): "
            f"{int((sc >= 4).sum()):>4} | score real 0: {int((sc == 0).sum()):>3} | "
            f"mediana de ruedas en posicion: {g['ruedas'].median():.0f}")
    out(f"  -> antes del {fix:%Y-%m-%d} la salida de la v1 NO es la regla: la historia valida "
        "empieza ahi.")

    val = ts[ts["fecha_datos_salida"] >= fix]
    out.titulo(f"TECH_SECTOR_v1 -- salidas VALIDAS (desde {fix:%Y-%m-%d})")
    out(f"  {len(val)} salidas en {val['fecha_datos_salida'].nunique()} ruedas, "
        f"{val['fecha_datos_salida'].min():%Y-%m-%d} -> {val['fecha_datos_salida'].max():%Y-%m-%d}")
    for f, g in val.groupby("familia"):
        out(f"    {f:<12} {len(g):>4}  resultado medio {g['pnl_pct'].mean():>+6.2f}%  "
            f"mediana de ruedas en posicion {g['ruedas'].median():.0f}")

    # que condicion disparo cada salida por score
    pos = query_df("""SELECT operacion_id, fecha_datos FROM ft_posiciones_diarias
                      WHERE estrategia_id = :e AND fecha_datos IS NOT NULL""",
                   params={"e": ID_TECH_SECTOR_V1})
    pos["fecha_datos"] = pd.to_datetime(pos["fecha_datos"])
    evaluadas = set(zip(pos["operacion_id"], pos["fecha_datos"]))
    filas = []
    for r in val[val["familia"] == "SCORE"].itertuples():
        k = (r.ticker, r.fecha_datos_salida)
        if k not in cond or pd.isna(prev_de.get(k)):
            continue
        kp = (r.ticker, prev_de[k])
        g = fs.gatillos(cond[kp], cond[k]) if kp in cond else ["sin dato previo"]
        filas.append({"id": r.id, "ticker": r.ticker, "fecha_datos_salida": r.fecha_datos_salida,
                      "gatillo": " + ".join(g), "n_gatillos": len(g), "pnl_pct": r.pnl_pct,
                      "estado_salida": fs.etiqueta_estado(cond[k]),
                      "bot_evaluo_rueda_previa": (r.id, prev_de[k]) in evaluadas})
    gt = pd.DataFrame(filas)
    gt.to_csv(os.path.join(dirout, "tech_sector_v1_gatillos.csv"), index=False)

    out.titulo("TECH_SECTOR_v1 -- que condicion disparo cada salida por score (vs la rueda anterior)")
    out(f"  {len(gt)} salidas por score validas")
    out(f"  {'gatillo':<74} {'n':>4} {'%':>5} {'resultado':>10}")
    for k, g in sorted(gt.groupby("gatillo"), key=lambda x: -len(x[1])):
        out(f"  {k:<74} {len(g):>4} {100 * len(g) / len(gt):>4.0f}% {g['pnl_pct'].mean():>+9.2f}%")
    solo = gt[gt["n_gatillos"] == 1]
    out("")
    out("  cuantas veces aparece cada condicion (sola o combinada):")
    for nombre in ("pierde SMA21", "pierde SMA50", "pierde SMA200", "pierde MACD",
                   "RSI sale por arriba (>68)", "RSI sale por abajo (<45)",
                   "sin cambio en la ultima rueda"):
        m = gt["gatillo"].str.contains(nombre, regex=False)
        s = solo["gatillo"].eq(nombre)
        out(f"    {nombre:<32} {int(m.sum()):>4} ({100 * m.mean():>3.0f}%) | como UNICO gatillo "
            f"{int(s.sum()):>4}")
    sc = gt[gt["gatillo"] == "sin cambio en la ultima rueda"]
    out("")
    out(f"  'sin cambio en la ultima rueda' ({len(sc)}): el bot evaluo esa posicion en la rueda "
        f"previa en {int(sc['bot_evaluo_rueda_previa'].sum())} casos y NO la evaluo en "
        f"{int((~sc['bot_evaluo_rueda_previa']).sum())}")
    out("  -> si no la evaluo, la condicion ya se habia perdido y la salida llego con atraso")
    out("     (dias sin rutina). Es ruido de la operacion, no de la regla.")
    out(f"  detalle por salida: {os.path.join(dirout, 'tech_sector_v1_gatillos.csv')}")


# --- paso 0 y paso 1 de TECH_SECTOR_v1: re-simular la salida con la entrada fija ---
# Pre-registro: docs/forward_testing/ANALISIS_SALIDAS.md sec. 6. Las entradas del
# backtest salen del motor (scripts/backtesting_historico) llamado EN MEMORIA: no escribe
# en bt_hist_*. La salida se re-simula con las reglas de FT (stop y take profit fijos
# desde la entrada, contra el close; balance con earnings_historico), no con las del
# motor, que no filtra balances y recalcula el stop con el ATR de cada dia.

DESDE_BACKTEST = "2021-09-01"
DIR_MOTOR = os.path.join(ROOT, "scripts", "backtesting_historico")


class Series:
    """Serie por ticker: fechas, close, condiciones por rueda y marca de balance."""

    def __init__(self, desde="2021-01-01"):
        px = query_df("""SELECT p.ticker, p.fecha, p.close, i.sma21, i.sma50, i.sma200,
                                i.rsi14, i.macd, i.macd_signal, i.atr14
                         FROM precios_diarios p
                         LEFT JOIN indicadores_tecnicos i ON i.ticker = p.ticker AND i.fecha = p.fecha
                         WHERE p.fecha >= :d ORDER BY p.ticker, p.fecha""", params={"d": desde})
        px["fecha"] = pd.to_datetime(px["fecha"])
        eh = query_df("SELECT ticker, announcement_date a FROM earnings_historico "
                      "WHERE announcement_date IS NOT NULL")
        eh["a"] = pd.to_datetime(eh["a"])
        anuncios = eh.groupby("ticker")["a"].apply(list).to_dict()
        self.s = {}
        for tk, g in px.groupby("ticker", sort=False):
            fechas = list(g["fecha"])
            conds = []
            for r in g.itertuples():
                if any(pd.isna(v) for v in (r.close, r.sma21, r.sma50, r.sma200, r.rsi14,
                                            r.macd, r.macd_signal)):
                    conds.append(None)
                else:
                    conds.append(fs.condiciones_desde_fila(
                        {"close": r.close, "sma21": r.sma21, "sma50": r.sma50, "sma200": r.sma200,
                         "rsi14": r.rsi14, "macd": r.macd, "macd_signal": r.macd_signal}))
            self.s[tk] = {
                "fechas": fechas,
                "idx": {f: i for i, f in enumerate(fechas)},
                "close": g["close"].astype(float).tolist(),
                "atr": g["atr14"].astype(float).tolist(),
                "conds": conds,
                "sale": {v: fs.score_por_rueda(conds, v) for v in fs.VARIANTES},
                "balance": fs.ruedas_de_balance(fechas, anuncios.get(tk, [])),
            }
        self.P = px.pivot(index="fecha", columns="ticker", values="close").sort_index().astype(float)
        self._uni = {}

    def universo(self, fe, fsal):
        """Retorno equal-weight del universo entre dos ruedas (tickers con precio en ambas)."""
        k = (fe, fsal)
        if k not in self._uni:
            a, b = self.P.loc[fe], self.P.loc[fsal]
            r = (b / a - 1.0).dropna()
            self._uni[k] = float(r.mean()) if len(r) else np.nan
        return self._uni[k]


def resimular(ser, entradas):
    """Una fila por (entrada, variante). `entradas`: ticker, fecha_entrada, precio_entrada,
    stop, take (+ columnas que se arrastran)."""
    filas = []
    for e in entradas.itertuples():
        s = ser.s.get(e.ticker)
        if s is None or e.fecha_entrada not in s["idx"]:
            continue
        i = s["idx"][e.fecha_entrada]
        for v in fs.VARIANTES:
            j, motivo = fs.primera_salida(s["close"], s["sale"][v], s["balance"], i,
                                          e.stop, e.take)
            fila = {"clave": e.clave, "ticker": e.ticker, "fecha_entrada": e.fecha_entrada,
                    "sector": getattr(e, "sector", None), "variante": v, "motivo": motivo}
            if j is None:
                fila.update(censurada=True)
            else:
                ret = s["close"][j] / e.precio_entrada - 1.0
                uni = ser.universo(e.fecha_entrada, s["fechas"][j])
                fila.update(censurada=False, fecha_salida=s["fechas"][j], ruedas=j - i,
                            ret_pct=100 * ret, exceso_pct=100 * (ret - uni))
            filas.append(fila)
    return pd.DataFrame(filas)


def correr_motor(desde, hasta):
    """Entradas del motor con la regla ACTUAL, en memoria (sin escribir en la DB)."""
    sys.path.insert(0, DIR_MOTOR)
    from bt_data_loader import BtDataLoader            # noqa: E402
    from ft_backtesting_runner import run_tech_sector  # noqa: E402
    from src.data.database import get_engine          # noqa: E402
    loader = BtDataLoader(get_engine(), pd.Timestamp(desde).date(), pd.Timestamp(hasta).date(),
                          "tecnico_sectorial")
    loader.cargar()
    pm = run_tech_sector(0, loader, "tecnico_sectorial", True, False)
    ops = pd.DataFrame(pm.operaciones_cerradas)
    ops["fecha_entrada"] = pd.to_datetime(ops["fecha_entrada"])
    return ops


def seccion_p0(out, dirout, ser, inval):
    fix = inval[ID_TECH_SECTOR_V1]
    out.titulo("PASO 0 -- el motor y la re-simulacion reproducen a TECH_SECTOR_v1 de FT?")

    # (a) el score que calcula el analisis es el que vio el bot
    pos = query_df("""SELECT operacion_id, ticker, fecha_datos, tech_score FROM ft_posiciones_diarias
                      WHERE estrategia_id = :e AND fecha_datos >= :f AND tech_score IS NOT NULL""",
                   params={"e": ID_TECH_SECTOR_V1, "f": str(fix)[:10]})
    pos["fecha_datos"] = pd.to_datetime(pos["fecha_datos"])
    igual = tot = 0
    for r in pos.itertuples():
        s = ser.s.get(r.ticker)
        if s is None or r.fecha_datos not in s["idx"]:
            continue
        c = s["conds"][s["idx"][r.fecha_datos]]
        if c is None:
            continue
        tot += 1
        igual += int(abs(fs.score(c) - float(r.tech_score)) < 1e-6)
    out(f"  (a) score del analisis vs el que registro el bot (ft_posiciones_diarias, desde "
        f"{fix:%Y-%m-%d}): igual en {igual:,} de {tot:,} ({100 * igual / max(tot, 1):.1f}%)")

    # (b) las entradas reales de FT cumplian la regla de entrada; y las del motor
    ent = query_df("""SELECT id, ticker, fecha_datos, precio_entrada, stop_loss, take_profit,
                             fecha_datos_salida, motivo_salida
                      FROM ft_operaciones WHERE estrategia_id = :e AND fecha_datos >= :f""",
                   params={"e": ID_TECH_SECTOR_V1, "f": str(fix)[:10]})
    for c in ("fecha_datos", "fecha_datos_salida"):
        ent[c] = pd.to_datetime(ent[c])
    ok = [ser.s[t]["conds"][ser.s[t]["idx"][f]] for t, f in zip(ent["ticker"], ent["fecha_datos"])
          if t in ser.s and f in ser.s[t]["idx"]]
    cumple = sum(1 for c in ok if c is not None and fs.cumple_entrada(c))
    out(f"  (b) entradas reales de FT desde {fix:%Y-%m-%d}: {len(ent)} | con score >= 4 en su rueda: "
        f"{cumple} de {len(ok)}")
    mot = correr_motor(str(fix)[:10], str(ser.P.index.max())[:10])
    a = set(zip(ent["ticker"], ent["fecha_datos"]))
    b = set(zip(mot["ticker"], mot["fecha_entrada"]))
    out(f"      motor en la misma ventana: {len(b)} entradas | coinciden (ticker y rueda) {len(a & b)} "
        f"| solo FT {len(a - b)} | solo motor {len(b - a)}")
    out("      (divergencia esperable: FT arranco con posiciones abiertas, bloquea entradas por balance,")
    out("       tiene dias sin rutina y otro tamano de posicion; el motor no. Las entradas del motor")
    out("       son 'las que haria una estrategia con esta regla', no una copia de FT.)")

    # (c) la re-simulacion de la regla ACTUAL sobre las entradas reales reproduce las salidas reales
    cerr = ent.dropna(subset=["fecha_datos_salida"]).copy()
    cerr = cerr[~cerr["motivo_salida"].astype(str).str.contains("SPLIT_FIX")]
    cerr["clave"] = cerr["id"]
    cerr["fecha_entrada"] = cerr["fecha_datos"]
    cerr["precio_entrada"] = [ser.s[t]["close"][ser.s[t]["idx"][f]]
                              if t in ser.s and f in ser.s[t]["idx"] else np.nan
                              for t, f in zip(cerr["ticker"], cerr["fecha_datos"])]
    cerr["stop"] = cerr["stop_loss"].astype(float)
    cerr["take"] = cerr["take_profit"].astype(float)
    rs = resimular(ser, cerr.dropna(subset=["precio_entrada"]))
    rs = rs[rs["variante"] == "actual"].merge(
        cerr[["clave", "fecha_datos_salida", "motivo_salida"]], on="clave")
    rs["fam_real"] = rs["motivo_salida"].map(fs.familia_salida)
    evaluadas = query_df("""SELECT operacion_id, fecha_datos FROM ft_posiciones_diarias
                            WHERE estrategia_id = :e AND fecha_datos IS NOT NULL""",
                         params={"e": ID_TECH_SECTOR_V1})
    evaluadas = set(zip(evaluadas["operacion_id"], pd.to_datetime(evaluadas["fecha_datos"])))
    igual_r = rs["fecha_salida"].eq(rs["fecha_datos_salida"])
    igual_m = rs["motivo"].eq(rs["fam_real"])
    antes = rs["fecha_salida"] < rs["fecha_datos_salida"]
    atraso = antes & ~pd.Series([(k, f) in evaluadas for k, f in zip(rs["clave"], rs["fecha_salida"])],
                                index=rs.index)
    out(f"  (c) regla ACTUAL re-simulada sobre las {len(rs)} entradas reales cerradas de FT:")
    out(f"      misma rueda de salida: {int(igual_r.sum())} ({100 * igual_r.mean():.0f}%) | misma rueda "
        f"y mismo motivo: {int((igual_r & igual_m).sum())} ({100 * (igual_r & igual_m).mean():.0f}%)")
    out(f"      la re-simulacion sale ANTES que FT: {int(antes.sum())}, de las cuales en {int(atraso.sum())} "
        f"el bot NO evaluo esa rueda (atraso de la rutina, no de la regla)")
    otros = rs[~igual_r & ~atraso]
    out(f"      diferencias no explicadas por la rutina: {len(otros)}")
    if len(otros):
        for r in otros.head(12).itertuples():
            out(f"        {r.ticker:<6} entrada {r.fecha_entrada:%Y-%m-%d} | re-simulada "
                f"{r.fecha_salida:%Y-%m-%d} {r.motivo:<11} | FT {r.fecha_datos_salida:%Y-%m-%d} "
                f"{r.motivo_salida}")
    rs.to_csv(os.path.join(dirout, "p0_resimulacion_vs_ft.csv"), index=False)
    return cerr


def _tabla_variantes(out, rs, titulo):
    """Metricas por variante sobre la muestra comun (sin censura en ninguna variante)."""
    cens = rs.groupby("clave")["censurada"].any()
    comun = rs[rs["clave"].isin(cens[~cens].index)].copy()
    out(f"  {titulo}: {comun['clave'].nunique():,} operaciones con salida en las 4 variantes "
        f"({int(cens.sum())} censuradas en alguna, excluidas)")
    out(f"  {'variante':<10} {'exceso medio':>13} {'exc/rueda':>10} {'ruedas':>7} {'ganadoras':>10} "
        f"{'p5 retorno':>11} {'retorno medio':>14}")
    for v in fs.VARIANTES:
        g = comun[comun["variante"] == v]
        out(f"  {v:<10} {g['exceso_pct'].mean():>+12.2f}% {(g['exceso_pct'] / g['ruedas']).mean():>+9.3f}% "
            f"{g['ruedas'].mean():>7.1f} {100 * (g['ret_pct'] > 0).mean():>9.0f}% "
            f"{fs._percentil(g['ret_pct'].tolist(), 5):>+10.2f}% {g['ret_pct'].mean():>+13.2f}%")
    return comun


def seccion_p1(out, dirout, ser, ent_ft):
    hasta = str(ser.P.index.max())[:10]
    out.titulo(f"PASO 1 -- PARAMETROS: sin SMA21 / sin MACD / sin RSI  ({DESDE_BACKTEST} -> {hasta})")
    out("  Pre-registro: docs/forward_testing/ANALISIS_SALIDAS.md sec. 6. Entradas del motor con la")
    out("  regla ACTUAL, fijas; salida re-simulada con las reglas de FT para cada variante.")
    mot = correr_motor(DESDE_BACKTEST, hasta)
    mot = mot.reset_index(drop=True)
    mot["clave"] = mot.index
    mot["stop"] = mot["stop_loss"].astype(float)
    mot["take"] = mot["take_profit"].astype(float)
    mot["precio_entrada"] = mot["precio_entrada"].astype(float)
    out(f"  entradas del motor: {len(mot):,} ({mot['fecha_entrada'].min():%Y-%m-%d} -> "
        f"{mot['fecha_entrada'].max():%Y-%m-%d}, {mot['ticker'].nunique()} tickers)")
    rs = resimular(ser, mot)
    rs.to_csv(os.path.join(dirout, "p1_resimulacion_backtest.csv"), index=False)
    out("")
    comun = _tabla_variantes(out, rs, "BACKTEST")

    base = comun[comun["variante"] == "actual"].set_index("clave")
    out("")
    out("  REGLA DE LECTURA (diferencia PAREADA variante - actual, exceso por operacion; IC95 con la")
    out("  rueda de entrada como unidad; >= 4 de 6 anios positivos; p5 no cae mas de 1 punto)")
    resultados = {}
    for v in ("sin_sma21", "sin_macd", "sin_rsi"):
        g = comun[comun["variante"] == v].set_index("clave").loc[base.index]
        dif = (g["exceso_pct"] - base["exceso_pct"]).tolist()
        dias = list(base["fecha_entrada"])
        anios = [f.year for f in base["fecha_entrada"]]
        r = fs.evaluar_variante(dif, dias, anios, g["ret_pct"].tolist(), base["ret_pct"].tolist())
        resultados[v] = r
        cambian = int((g["fecha_salida"] != base["fecha_salida"]).sum())
        ic = r["ic"]
        out("")
        out(f"  {v}{'  (HIPOTESIS PRINCIPAL)' if v == 'sin_sma21' else '  (comparacion)'}: "
            f"cambia la salida en {cambian:,} de {len(base):,} operaciones")
        out(f"    diferencia media {ic['media']:+.3f} pp  IC95 [{ic['lo']:+.3f}; {ic['hi']:+.3f}]  "
            f"({ic['dias']} ruedas de entrada) -> {'SI' if r['c1_ic_sobre_cero'] else 'no'}")
        out("    por anio: " + " | ".join(f"{a} {m:+.2f}" for a, m in r["por_anio"].items())
            + f"  -> {r['anios_positivos']}/{r['anios_total']} positivos "
            f"{'SI' if r['c2_anios'] else 'no'}")
        out(f"    cola (p5 del retorno): actual {r['p5_actual']:+.2f}% | variante {r['p5_variante']:+.2f}% "
            f"-> {'SI' if r['c3_cola'] else 'no'}")
        out(f"    ==> {'PASA' if r['pasa'] else 'NO PASA'}")

    out("")
    out("  por sector (informativo, NO decide: es la pregunta 4) -- diferencia media sin_sma21 - actual:")
    g = comun[comun["variante"] == "sin_sma21"].set_index("clave").loc[base.index]
    d = (g["exceso_pct"] - base["exceso_pct"])
    for sec, x in d.groupby(base["sector"]).mean().sort_values().items():
        out(f"    {str(sec):<24} {x:+.2f} pp  (n={int((base['sector'] == sec).sum())})")

    # sobre las entradas REALES del FT de control (diagnostico, no decide)
    out.titulo("PASO 1 -- las mismas variantes sobre las entradas REALES de FT (control, diagnostico)")
    ent = ent_ft.copy()
    ent["sector"] = None
    rs_ft = resimular(ser, ent.dropna(subset=["precio_entrada"]))
    rs_ft.to_csv(os.path.join(dirout, "p1_resimulacion_ft.csv"), index=False)
    comun_ft = _tabla_variantes(out, rs_ft, "FT desde el 29/5")
    b2 = comun_ft[comun_ft["variante"] == "actual"].set_index("clave")
    for v in ("sin_sma21", "sin_macd", "sin_rsi"):
        g2 = comun_ft[comun_ft["variante"] == v].set_index("clave").loc[b2.index]
        ic = fs.ic95_por_dia((g2["exceso_pct"] - b2["exceso_pct"]).tolist(), list(b2["fecha_entrada"]))
        out(f"    {v:<10} diferencia pareada {ic['media']:+.3f} pp IC95 [{ic['lo']:+.3f}; {ic['hi']:+.3f}] "
            f"({ic['dias']} ruedas)")
    out("  -> 64 ruedas de un solo tipo de mercado: sirve para ver si FT apunta al mismo lado que")
    out("     el backtest, no para decidir.")
    json.dump({v: {"pasa": r["pasa"], "dif_media": r["ic"]["media"], "ic95": [r["ic"]["lo"], r["ic"]["hi"]],
                   "anios_positivos": r["anios_positivos"], "p5_actual": r["p5_actual"],
                   "p5_variante": r["p5_variante"]} for v, r in resultados.items()},
              open(os.path.join(dirout, "p1_resultado.json"), "w", encoding="utf-8"), indent=2,
              default=float)


# --- paso 2: grilla de pesos, midiendo la SALIDA en si (sec. 8 del doc) --------------

FECHA_CONFIRMACION = pd.Timestamp("2025-01-01")
TOP_SELECCION = 3

# Palancas nombradas por el usuario el 19/9 (se buscan en la grilla y se informan aparte).
PALANCAS_USUARIO = [
    ("SMA21 peso 0 (quitar SMA21)", "obligatoria", {"sma21": 0}),
    ("RSI peso 0 (quitar RSI)", "obligatoria", {"rsi": 0}),
    ("MACD peso 0 (quitar MACD)", "obligatoria", {"macd": 0}),
    ("SMA50 peso 3 (mas peso a SMA50)", "obligatoria", {"sma50": 3}),
    ("RSI peso 2 (mas peso al RSI)", "obligatoria", {"rsi": 2}),
    ("SMA200 pesa 1 (no obligatoria) y RSI 2", 1, {"rsi": 2}),
    ("MACD peso 1 (hoy 1,5)", "obligatoria", {"macd": 1}),
]


class Matrices:
    """Re-simulacion de TODAS las reglas sobre las mismas entradas. Una fila por entrada,
    una columna por regla: dia de salida y metricas."""

    def __init__(self, ser, entradas, bits):
        P = ser.P
        r1 = P.ffill().pct_change(fill_method=None)  # = pct_change() de siempre, sin el aviso
        U = np.cumprod(1.0 + r1.mean(axis=1, skipna=True).fillna(0.0).to_numpy())
        pos_global = {f: k for k, f in enumerate(P.index)}
        T, R = len(entradas), bits.shape[0]
        shape = (T, R)
        self.EX = np.full(shape, -1, dtype=np.int64)
        self.RUEDAS = np.full(shape, np.nan)
        self.RET = np.full(shape, np.nan)
        self.TRAMO = np.full(shape, np.nan)
        self.POST = {h: np.full(shape, np.nan) for h in HORIZONTES}
        self.Z10 = np.full(shape, np.nan)
        self.fecha = np.array(entradas["fecha_entrada"].values)
        self.fila_valida = np.zeros(T, dtype=bool)
        cache = {}
        for t, e in enumerate(entradas.itertuples()):
            s = ser.s.get(e.ticker)
            if s is None or e.fecha_entrada not in s["idx"]:
                continue
            if e.ticker not in cache:
                cl = np.array(s["close"], dtype=float)
                ret = np.concatenate([[np.nan], cl[1:] / cl[:-1] - 1.0])
                cache[e.ticker] = {
                    "cl": cl,
                    "est": np.array([fs.ESTADO_SIN_DATOS if c is None else fs.indice_estado(c)
                                     for c in s["conds"]], dtype=np.int64),
                    "bal": np.array(s["balance"], dtype=bool),
                    "sig": pd.Series(ret).rolling(VOL_RUEDAS, min_periods=40).std().to_numpy(),
                    "g": np.array([pos_global[f] for f in s["fechas"]], dtype=np.int64),
                }
            d = cache[e.ticker]
            cl, n = d["cl"], len(d["cl"])
            i = s["idx"][e.fecha_entrada]
            pe = float(e.precio_entrada)
            fut = cl[i + 1:]
            with np.errstate(invalid="ignore"):
                duro = d["bal"][i + 1:] | (fut <= e.stop) | (fut >= e.take)
            duro &= ~np.isnan(fut)
            h = i + 1 + int(np.argmax(duro)) if duro.any() else None
            fin = (h + 1) if h is not None else n
            st = d["est"][i + 1:fin]
            if len(st) == 0:
                continue  # entrada en la ultima rueda: sin salida posible (censurada, como en primera_salida)
            M = bits[:, st]
            tiene = M.any(axis=1)
            primera = M.argmax(axis=1)
            ex = np.where(tiene, i + 1 + primera, h if h is not None else -1)
            self.EX[t] = ex
            ok = ex >= 0
            if not ok.any():
                continue
            self.fila_valida[t] = True
            x = ex[ok]
            self.RUEDAS[t, ok] = x - i
            ret = cl[x] / pe - 1.0
            self.RET[t, ok] = 100 * ret
            self.TRAMO[t, ok] = 100 * (ret - (U[d["g"][x]] / U[d["g"][i]] - 1.0))
            for hh in HORIZONTES:
                y = x + hh
                dentro = y < n
                if not dentro.any():
                    continue
                xs, ys = x[dentro], y[dentro]
                exc = (cl[ys] / cl[xs] - 1.0) - (U[d["g"][ys]] / U[d["g"][xs]] - 1.0)
                cols = np.flatnonzero(ok)[dentro]
                self.POST[hh][t, cols] = 100 * exc
                if hh == H_CLASIFICAR:
                    self.Z10[t, cols] = exc / (d["sig"][xs] * np.sqrt(hh))


def _media_por_dia(valores, fechas):
    """Media de las medias por rueda de entrada (cada rueda pesa uno)."""
    ok = ~np.isnan(valores)
    if not ok.any():
        return np.nan
    s = pd.Series(valores[ok]).groupby(fechas[ok]).mean()
    return float(s.mean())


def _metricas_regla(Mx, r, a, filas):
    """Metricas de la regla r contra la actual a, sobre las filas (mascara de entradas)."""
    both = filas & ~np.isnan(Mx.RET[:, r]) & ~np.isnan(Mx.RET[:, a])
    post = filas & ~np.isnan(Mx.POST[H_CLASIFICAR][:, r]) & ~np.isnan(Mx.POST[H_CLASIFICAR][:, a])
    dpost = np.where(post, Mx.POST[H_CLASIFICAR][:, r] - Mx.POST[H_CLASIFICAR][:, a], np.nan)
    dtramo = np.where(both, Mx.TRAMO[:, r] - Mx.TRAMO[:, a], np.nan)
    z = Mx.Z10[filas, r]
    z = z[~np.isnan(z)]
    return {
        "ops": int(both.sum()),
        "cambian": int((both & (Mx.EX[:, r] != Mx.EX[:, a])).sum()),
        "post10": float(np.nanmean(Mx.POST[H_CLASIFICAR][post, r])) if post.any() else np.nan,
        "post5": float(np.nanmean(Mx.POST[5][filas, r])),
        "post20": float(np.nanmean(Mx.POST[20][filas, r])),
        "dif_post": _media_por_dia(dpost, Mx.fecha),
        "dif_tramo": _media_por_dia(dtramo, Mx.fecha),
        "p5": float(np.nanpercentile(Mx.RET[both, r], 5)) if both.any() else np.nan,
        "p5_actual": float(np.nanpercentile(Mx.RET[both, a], 5)) if both.any() else np.nan,
        "ruedas": float(np.nanmean(Mx.RUEDAS[both, r])) if both.any() else np.nan,
        "a_tiempo": float((z < -fs.UMBRAL_Z).mean()) if len(z) else np.nan,
        "temprano": float((z > fs.UMBRAL_Z).mean()) if len(z) else np.nan,
        "lateral": float((np.abs(z) <= fs.UMBRAL_Z).mean()) if len(z) else np.nan,
        "_dpost": dpost, "_dtramo": dtramo,
    }


def _linea(out, nombre, m):
    out(f"  {nombre:<48} {m['ops']:>5} {m['cambian']:>6} {m['ruedas']:>6.1f} "
        f"{m['post10']:>+7.2f} {100 * m['a_tiempo']:>5.0f}% {100 * m['temprano']:>5.0f}% "
        f"{100 * m['lateral']:>5.0f}% {m['dif_post']:>+8.3f} {m['dif_tramo']:>+8.3f} "
        f"{m['p5']:>+7.2f}")


def _encabezado(out):
    out(f"  {'regla':<48} {'ops':>5} {'cambia':>6} {'ruedas':>6} {'post10':>7} {'a_tie':>6} "
        f"{'tempr':>6} {'later':>6} {'dif_post':>8} {'dif_tram':>8} {'p5':>7}")


def seccion_p2(out, dirout, ser, ent_ft):
    hasta = str(ser.P.index.max())[:10]
    out.titulo(f"PASO 2 -- GRILLA DE PESOS: la SALIDA en si ({DESDE_BACKTEST} -> {hasta})")
    out("  Pre-registro: docs/forward_testing/ANALISIS_SALIDAS.md sec. 8.1. Entradas fijas del motor;")
    out("  cada regla re-simulada. post10 = exceso del ticker contra el universo en las 10 ruedas")
    out("  DESPUES de la salida; dif_post = post10 de la regla menos el de la actual en la misma")
    out("  operacion (negativo = sale en mejores momentos); dif_tram = lo mismo para el tramo")
    out("  entrada -> salida; p5 = percentil 5 del retorno por operacion. Medias por rueda de entrada.")

    grilla = fs.reglas_grilla()
    etiquetas = OrderedDict((m, fs.etiqueta_pesos(*cfgs[0]) + (f"  (+{len(cfgs) - 1} equivalentes)"
                                                                  if len(cfgs) > 1 else ""))
                            for m, cfgs in grilla.items())
    for v in ("sin_sma21", "sin_macd", "sin_rsi"):
        m = fs.mascara_variante(v)
        etiquetas[m] = (etiquetas[m] + f"  = {v}") if m in etiquetas else f"{v} (paso 1)"
    mascaras = list(etiquetas.keys())
    act = fs.mascara_variante("actual")
    a = mascaras.index(act)
    bits = fs.bits_reglas(mascaras)
    out(f"  reglas distintas: {len(mascaras)} (589 de la grilla + las del paso 1 que no estaban)")

    mot = correr_motor(DESDE_BACKTEST, hasta).reset_index(drop=True)
    mot["stop"] = mot["stop_loss"].astype(float)
    mot["take"] = mot["take_profit"].astype(float)
    mot["precio_entrada"] = mot["precio_entrada"].astype(float)
    Mx = Matrices(ser, mot, bits)

    # control interno: la re-simulacion vectorizada da las MISMAS salidas que
    # fs.primera_salida (la validada contra FT en el paso 0) para las reglas del paso 1
    igual = total = 0
    for v in fs.VARIANTES:
        r = mascaras.index(fs.mascara_variante(v))
        for t, e in enumerate(mot.itertuples()):
            s_ = ser.s.get(e.ticker)
            if s_ is None or e.fecha_entrada not in s_["idx"]:
                continue
            j, _ = fs.primera_salida(s_["close"], s_["sale"][v], s_["balance"],
                                     s_["idx"][e.fecha_entrada], e.stop, e.take)
            total += 1
            igual += int((j if j is not None else -1) == Mx.EX[t, r])
    out(f"  control interno: salida vectorizada = salida validada en {igual:,} de {total:,} "
        f"(4 reglas x entradas)")
    if igual != total:
        out("  [ERROR] la re-simulacion vectorizada no coincide: se frena.")
        return

    sel = Mx.fila_valida & (Mx.fecha < np.datetime64(FECHA_CONFIRMACION))
    conf = Mx.fila_valida & (Mx.fecha >= np.datetime64(FECHA_CONFIRMACION))
    out(f"  entradas: {len(mot):,} | seleccion (2021-10 -> 2024-12): {int(sel.sum()):,} | "
        f"confirmacion (2025-01 -> 2026-09): {int(conf.sum()):,}")

    filas_csv, met_sel = [], {}
    for r, m in enumerate(mascaras):
        ms = _metricas_regla(Mx, r, a, sel)
        mc = _metricas_regla(Mx, r, a, conf)
        met_sel[r] = ms
        filas_csv.append({"mascara": m, "regla": etiquetas[m],
                          "estados_que_salen": bin(m).count("1"),
                          **{f"sel_{k}": v for k, v in ms.items() if not k.startswith("_")},
                          **{f"conf_{k}": v for k, v in mc.items() if not k.startswith("_")}})
    pd.DataFrame(filas_csv).to_csv(os.path.join(dirout, "p2_grilla.csv"), index=False)

    out("")
    out("  LA REGLA ACTUAL (FT de control), seleccion y confirmacion:")
    _encabezado(out)
    _linea(out, "actual -- seleccion", met_sel[a])
    _linea(out, "actual -- confirmacion", _metricas_regla(Mx, a, a, conf))

    out("")
    out("  PALANCAS NOMBRADAS POR EL USUARIO (periodo de seleccion):")
    _encabezado(out)
    for nombre, w200, cambio in PALANCAS_USUARIO:
        m = fs.mascara_pesos(dict(fs.PESOS_ACTUALES, **cambio), w200)
        r = mascaras.index(m)
        _linea(out, nombre + ("  [= actual]" if m == act else ""), met_sel[r])
    for v in ("sin_sma21", "sin_macd", "sin_rsi"):
        _linea(out, f"{v} (paso 1: no puede disparar)", met_sel[mascaras.index(fs.mascara_variante(v))])

    cand = [r for r in range(len(mascaras)) if r != a and fs.es_candidata(
        met_sel[r]["dif_post"], met_sel[r]["dif_tramo"], met_sel[r]["p5"], met_sel[r]["p5_actual"])]
    out("")
    out(f"  SELECCION: {len(cand)} de {len(mascaras) - 1} reglas son candidatas "
        f"(dif_post < 0, dif_tramo >= 0, cola no cae mas de 1 punto)")
    mejora_post = sum(1 for r in range(len(mascaras)) if r != a and met_sel[r]["dif_post"] < 0)
    mejora_tramo = sum(1 for r in range(len(mascaras)) if r != a and met_sel[r]["dif_tramo"] >= 0)
    out(f"    dif_post < 0 en {mejora_post} reglas | dif_tramo >= 0 en {mejora_tramo} reglas")
    orden = sorted(cand, key=lambda r: met_sel[r]["dif_post"])
    out("  las 10 candidatas con dif_post mas negativa:")
    _encabezado(out)
    for r in orden[:10]:
        _linea(out, etiquetas[mascaras[r]][:48], met_sel[r])

    out.titulo("PASO 2 -- CONFIRMACION de las 3 elegidas (2025-01 -> 2026-09) y FT de control")
    elegidas = orden[:TOP_SELECCION]
    if not elegidas:
        out("  ninguna regla es candidata en la seleccion: no hay nada que confirmar.")
    ft = ent_ft.copy()
    ft = ft.dropna(subset=["precio_entrada"]).reset_index(drop=True)
    Mft = Matrices(ser, ft, bits)
    res_json = {}
    for k, r in enumerate(elegidas, 1):
        m = mascaras[r]
        mc = _metricas_regla(Mx, r, a, conf)
        icp = fs.ic95_por_dia(list(mc["_dpost"]), list(Mx.fecha))
        ict = fs.ic95_por_dia(list(mc["_dtramo"]), list(Mx.fecha))
        c_post = icp["hi"] == icp["hi"] and icp["hi"] < 0
        c_tramo = ict["media"] == ict["media"] and ict["media"] >= 0
        c_cola = (mc["p5_actual"] - mc["p5"]) <= fs.CAIDA_COLA_MAX_PP
        conf_ok = c_post and c_tramo and c_cola
        mf = _metricas_regla(Mft, r, a, Mft.fila_valida)
        icf = fs.ic95_por_dia(list(mf["_dpost"]), list(Mft.fecha))
        dif = fs.diferencias_de_regla(m, act)
        out("")
        out(f"  [{k}] {etiquetas[m]}")
        out(f"      deja de salir en: {dif['deja_de_salir'] or '-'}")
        out(f"      empieza a salir en: {dif['empieza_a_salir'] or '-'}")
        out(f"      seleccion : dif_post {met_sel[r]['dif_post']:+.3f} | dif_tramo {met_sel[r]['dif_tramo']:+.3f}")
        out(f"      confirmac.: dif_post {icp['media']:+.3f} IC95 [{icp['lo']:+.3f}; {icp['hi']:+.3f}] -> "
            f"{'SI' if c_post else 'no'} | dif_tramo {ict['media']:+.3f} -> {'SI' if c_tramo else 'no'} | "
            f"p5 {mc['p5_actual']:+.2f} -> {mc['p5']:+.2f} -> {'SI' if c_cola else 'no'}")
        out(f"      ==> {'CONFIRMADA' if conf_ok else 'NO SE CONFIRMA'}")
        out(f"      FT de control: dif_post {icf['media']:+.3f} IC95 [{icf['lo']:+.3f}; {icf['hi']:+.3f}] | "
            f"dif_tramo {mf['dif_tramo']:+.3f} | cambia {mf['cambian']} de {mf['ops']}")
        res_json[etiquetas[m]] = {"confirmada": conf_ok, "dif_post_conf": icp["media"],
                                  "ic_post_conf": [icp["lo"], icp["hi"]],
                                  "dif_tramo_conf": ict["media"], "ft_dif_post": icf["media"]}

    out("")
    out("  FT de control -- la regla actual y las palancas del usuario sobre las entradas REALES:")
    _encabezado(out)
    _linea(out, "actual (FT)", _metricas_regla(Mft, a, a, Mft.fila_valida))
    for nombre, w200, cambio in PALANCAS_USUARIO:
        m = fs.mascara_pesos(dict(fs.PESOS_ACTUALES, **cambio), w200)
        if m != act:
            _linea(out, nombre, _metricas_regla(Mft, mascaras.index(m), a, Mft.fila_valida))
    json.dump(res_json, open(os.path.join(dirout, "p2_resultado.json"), "w", encoding="utf-8"),
              indent=2, default=float)
    _complemento_p2(out, Mx, Mft, mascaras, a, act, pd.DataFrame(filas_csv))
    out(f"  detalle de las {len(mascaras)} reglas: {os.path.join(dirout, 'p2_grilla.csv')}")


def _complemento_p2(out, Mx, Mft, mascaras, a, act, grilla):
    """Lo que NO estaba pre-registrado y no cambia la lectura: IC95 de las palancas por
    periodo y por anio, si el orden de la seleccion se mantiene en la confirmacion, y que
    pasa con las reglas que pasan (a) y (b) si se ignora la cola."""
    out.titulo("PASO 2 -- COMPLEMENTO DESCRIPTIVO (no pre-registrado; no cambia la lectura)")
    sel = Mx.fila_valida & (Mx.fecha < np.datetime64(FECHA_CONFIRMACION))
    conf = Mx.fila_valida & (Mx.fecha >= np.datetime64(FECHA_CONFIRMACION))
    anio = pd.DatetimeIndex(Mx.fecha).year.values

    def ic(M, r, filas):
        m = _metricas_regla(M, r, a, filas)
        return (m, fs.ic95_por_dia(list(m["_dpost"][filas]), list(M.fecha[filas])),
                fs.ic95_por_dia(list(m["_dtramo"][filas]), list(M.fecha[filas])))

    def f(x):
        return f"{x['media']:+.3f} [{x['lo']:+.3f}; {x['hi']:+.3f}]"

    reglas = [(n, fs.mascara_pesos(dict(fs.PESOS_ACTUALES, **c), w)) for n, w, c in PALANCAS_USUARIO]
    reglas = [(n, m) for n, m in reglas if m != act]
    reglas += [(f"{v} (paso 1)", fs.mascara_variante(v)) for v in ("sin_sma21", "sin_macd", "sin_rsi")]
    out("  Diferencia pareada contra la actual, pp, IC95 por rueda de entrada. post = 10 ruedas")
    out("  despues de la salida (negativo = mejor); tramo = entrada -> salida (positivo = mejor).")
    for nombre, m in reglas:
        r = mascaras.index(m)
        out(f"  {nombre}")
        for lab, M, filas in (("seleccion 2021-24", Mx, sel), ("confirmac 2025-26", Mx, conf),
                              ("FT de control    ", Mft, Mft.fila_valida)):
            mm, p, t = ic(M, r, filas)
            out(f"    {lab}: post {f(p)} | tramo {f(t)} | p5 {mm['p5_actual']:+.2f} -> {mm['p5']:+.2f}"
                f" | ruedas {mm['ruedas']:.1f}")
        por_anio = []
        for y in sorted(set(anio[Mx.fila_valida])):
            _, p, t = ic(Mx, r, Mx.fila_valida & (anio == y))
            por_anio.append(f"{y} {p['media']:+.2f}/{t['media']:+.2f}")
        out("    por anio (post/tramo): " + " | ".join(por_anio))

    g = grilla[grilla["mascara"] != act].copy()
    g["caida_sel"] = g["sel_p5_actual"] - g["sel_p5"]
    g["caida_conf"] = g["conf_p5_actual"] - g["conf_p5"]
    ruedas_act = float(grilla.loc[grilla["mascara"] == act, "sel_ruedas"].iloc[0])
    largo = g["sel_ruedas"] > ruedas_act
    out("")
    out(f"  la grilla es casi UNA dimension, cuanto se queda: de {len(g)} reglas, {int(largo.sum())} se "
        f"quedan mas que la actual y {int((~largo).sum())} menos")
    out(f"    correlacion entre reglas de 'ruedas' con: caida del p5 {np.corrcoef(g['sel_ruedas'], g['caida_sel'])[0, 1]:+.2f}"
        f" | dif_post {np.corrcoef(g['sel_ruedas'], g['sel_dif_post'])[0, 1]:+.2f}"
        f" | dif_tramo {np.corrcoef(g['sel_ruedas'], g['sel_dif_tramo'])[0, 1]:+.2f}")
    out(f"    medias (sel -> conf): quedarse mas  post {g.loc[largo, 'sel_dif_post'].mean():+.3f} -> "
        f"{g.loc[largo, 'conf_dif_post'].mean():+.3f} | tramo {g.loc[largo, 'sel_dif_tramo'].mean():+.3f} -> "
        f"{g.loc[largo, 'conf_dif_tramo'].mean():+.3f} | caida p5 {g.loc[largo, 'caida_sel'].mean():+.2f} -> "
        f"{g.loc[largo, 'caida_conf'].mean():+.2f}")
    out(f"                          salir antes   post {g.loc[~largo, 'sel_dif_post'].mean():+.3f} -> "
        f"{g.loc[~largo, 'conf_dif_post'].mean():+.3f} | tramo {g.loc[~largo, 'sel_dif_tramo'].mean():+.3f} -> "
        f"{g.loc[~largo, 'conf_dif_tramo'].mean():+.3f} | caida p5 {g.loc[~largo, 'caida_sel'].mean():+.2f} -> "
        f"{g.loc[~largo, 'caida_conf'].mean():+.2f}")
    out(f"  el orden de la seleccion NO se mantiene: correlacion entre reglas sel vs conf de dif_post "
        f"{np.corrcoef(g['sel_dif_post'], g['conf_dif_post'])[0, 1]:+.2f}, de dif_tramo "
        f"{np.corrcoef(g['sel_dif_tramo'], g['conf_dif_tramo'])[0, 1]:+.2f}")
    ab = g[(g["sel_dif_post"] < 0) & (g["sel_dif_tramo"] >= 0)]
    out(f"  ignorando la cola: {len(ab)} reglas pasan (a) y (b) en la seleccion (caida del p5 entre "
        f"{ab['caida_sel'].min():.2f} y {ab['caida_sel'].max():.2f} pp); en la confirmacion dif_post va de "
        f"{ab['conf_dif_post'].min():+.3f} a {ab['conf_dif_post'].max():+.3f} y dif_tramo >= 0 en "
        f"{int((ab['conf_dif_tramo'] >= 0).sum())}")
    x = Mx.POST[H_CLASIFICAR][sel, a]
    x = x[~np.isnan(x)]
    out(f"  escala: post10 de la regla actual en la seleccion, desvio por operacion {x.std():.2f} pp; "
        f"todas las dif_post de la grilla caen entre {g['sel_dif_post'].min():+.3f} y "
        f"{g['sel_dif_post'].max():+.3f} pp")


SECCIONES = ("panorama", "balances", "tech_sector_v1")
SECCIONES_PASOS = ("p0", "p1", "p2")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analisis de salidas de FT (solo lee)")
    ap.add_argument("--seccion", choices=SECCIONES + SECCIONES_PASOS + ("todas", "p0p1"),
                    default="todas",
                    help="todas = las exploratorias; p0p1 = pasos 0 y 1 de TECH_SECTOR_v1")
    ap.add_argument("--etiqueta", default="exploratorio",
                    help="sufijo del directorio reportes/analisis_salidas/AAAAMMDD_<etiqueta>/")
    args = ap.parse_args()

    dirout = os.path.join(DIR_BASE, f"{date.today():%Y%m%d}_{args.etiqueta}")
    os.makedirs(dirout, exist_ok=True)
    elegidas = (SECCIONES if args.seccion == "todas" else
                ("p0", "p1") if args.seccion == "p0p1" else (args.seccion,))

    op, inval = cargar_salidas()
    P, sig, fwd, uni = cargar_precios()
    json.dump({
        "fecha_corrida": str(date.today()), "ultima_rueda_precios": str(P.index.max())[:10],
        "git": _git(), "secciones": list(elegidas), "horizontes": HORIZONTES,
        "h_clasificar": H_CLASIFICAR, "umbral_z": fs.UMBRAL_Z, "vol_ruedas": VOL_RUEDAS,
        "ventanas_excluidas": {str(k): str(v)[:10] for k, v in inval.items()},
        "cambios_que_invalidan": CAMBIOS_QUE_INVALIDAN,
        "score_entrada_v1": fs.SCORE_ENTRADA_V1, "score_salida_v1": fs.SCORE_SALIDA_V1,
    }, open(os.path.join(dirout, "parametros.json"), "w", encoding="utf-8"), indent=2)

    ser = Series() if any(x in SECCIONES_PASOS for x in elegidas) else None
    ent_ft = None
    for s in elegidas:
        out = Salida(os.path.join(dirout, f"{s}.log"))
        if s == "p0":
            ent_ft = seccion_p0(out, dirout, ser, inval)
        elif s in ("p1", "p2"):
            if ent_ft is None:
                ent_ft = seccion_p0(Salida(os.path.join(dirout, "p0.log")), dirout, ser, inval)
            (seccion_p1 if s == "p1" else seccion_p2)(out, dirout, ser, ent_ft)
        elif s == "panorama":
            seccion_panorama(op, P, sig, fwd, uni, out, dirout)
        elif s == "balances":
            seccion_balances(op, P, fwd, uni, out)
        else:
            seccion_tech_sector_v1(op, inval, out, dirout)
        out.cerrar()
    print(f"\nResultados en {dirout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
