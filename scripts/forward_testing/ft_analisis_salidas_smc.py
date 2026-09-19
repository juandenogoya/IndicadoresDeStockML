"""
ft_analisis_salidas_smc.py
Analisis de las SALIDAS de FT_SMC_v1 (docs/forward_testing/ANALISIS_SALIDAS.md sec. 10).
Solo LEE la DB local. Mismo metodo que TECH_SECTOR_v1 (ft_analisis_salidas.py, del que se
reusan el universo, las metricas pareadas y el IC por rueda). La regla de salida y la grilla
viven en src/utils/ft_salidas_smc.py (puro, con tests).

POR QUE UNA RECONSTRUCCION
    SMC_v1 lee features_market_structure, cuya HISTORIA mira 10 ruedas al futuro. En vivo el
    bot lee la ULTIMA fila (swings provisorios, sin futuro). Para re-simular se recalcula, rueda
    por rueda, el modulo viejo (_calcular_estructura_n) sobre las ultimas 250 barras hasta esa
    rueda: lo que el bot veia. Se guarda en reportes/analisis_salidas/cache/ (fuera de git) y se
    reusa mientras la ultima rueda de precios no cambie (--recalcular lo fuerza).

SECCIONES (--seccion)
    p0       fidelidad contra FT: la reconstruccion contra lo que el bot guardo al entrar,
             la regla actual re-simulada contra las salidas reales, el stop final; y la
             anatomia historica (motivos, cuantas veces CHoCH/estructura salen primero)
    grilla   las 96 combinaciones pre-registradas (doc sec. 10.4): seleccion 2021-2024,
             confirmacion 2025-2026, FT de control y muestra con tope de 5 posiciones
    todas    las dos (default)

Uso:
    python scripts/forward_testing/ft_analisis_salidas_smc.py --seccion p0 --etiqueta smc_v1
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import date
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "forward_testing"))

# LOCAL es la fuente de verdad: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ft_analisis_salidas as A  # noqa: E402
from scripts.forward_testing.ft_scoring import LOOKBACK_DIAS, calcular_score_estructura  # noqa: E402
from src.data.database import query_df  # noqa: E402
from src.indicators.market_structure import _calcular_estructura_n  # noqa: E402
from src.utils import ft_salidas as fs  # noqa: E402
from src.utils import ft_salidas_smc as fsm  # noqa: E402
from src.utils.ft_salidas_smc import Regla  # noqa: E402

ID_SMC_V1 = 3
DESDE = A.DESDE_BACKTEST                      # 2021-09-01, el mismo que TECH_SECTOR_v1
DESDE_PRECIOS = "2020-06-01"                  # 250 barras antes de DESDE para la ventana
VENTANA_ASOF = 250
DIR_CACHE = os.path.join(A.DIR_BASE, "cache")
COLS_ASOF = ["ticker", "fecha", "estructura_10", "choch_bear_10", "dist_sl_10_pct",
             "dist_sh_10_pct", "dist_sl_5_pct", "tuvo_choch_bull", "tuvo_bos_bull"]
TICKERS_CONTROL = ("AAPL", "JPM", "HOOD")
MARGEN_BALANCES = 5     # ruedas de margen antes del primer balance que falta en earnings_historico

PALANCAS = [
    ("sin time stop", Regla("trail10", True, True, None)),
    ("time stop 10 dias", Regla("trail10", True, True, 10)),
    ("time stop 15 dias", Regla("trail10", True, True, 15)),
    ("time stop 30 dias", Regla("trail10", True, True, 30)),
    ("time stop 45 dias", Regla("trail10", True, True, 45)),
    ("stop fijo (sin trailing)", Regla("fijo", True, True, 20)),
    ("trailing de 5 barras", Regla("trail5", True, True, 20)),
    ("sin stop", Regla("sin", True, True, 20)),
    ("sin CHoCH ni estructura rota", Regla("trail10", False, False, 20)),
    ("sin stop ni time stop", Regla("sin", True, True, None)),
]


# --- reconstruccion: lo que el bot veia cada rueda ------------------------------------

def _ultima_fila(sub: pd.DataFrame):
    d = _calcular_estructura_n(sub.copy(), 10)
    d = _calcular_estructura_n(d, 5)
    f = d["fecha"].iloc[-1]
    lb = d["fecha"] >= f - pd.Timedelta(days=LOOKBACK_DIAS)
    return (f, float(d["estructura_10"].iloc[-1]), float(d["choch_bear_10"].iloc[-1]),
            float(d["dist_sl_10_pct"].iloc[-1]), float(d["dist_sh_10_pct"].iloc[-1]),
            float(d["dist_sl_5_pct"].iloc[-1]),
            int(d.loc[lb, "choch_bull_10"].max()), int(d.loc[lb, "bos_bull_10"].max()))


def _asof_ticker(args):
    """Worker: una fila por rueda desde DESDE con la ULTIMA fila del modulo viejo calculado
    sobre las ultimas VENTANA_ASOF barras."""
    tk, fechas, high, low, close = args
    g = pd.DataFrame({"fecha": fechas, "high": high, "low": low, "close": close})
    k0 = int(np.searchsorted(g["fecha"].values, np.datetime64(DESDE)))
    return [(tk,) + _ultima_fila(g.iloc[max(0, t + 1 - VENTANA_ASOF): t + 1].reset_index(drop=True))
            for t in range(k0, len(g))]


def reconstruir(out, recalcular=False):
    px = query_df("""SELECT ticker, fecha, high, low, close FROM precios_diarios
                     WHERE fecha >= :d AND close > 0 AND high > 0 AND low > 0
                     ORDER BY ticker, fecha""", params={"d": DESDE_PRECIOS})
    px["fecha"] = pd.to_datetime(px["fecha"])
    ultima = px["fecha"].max()
    ruta = os.path.join(DIR_CACHE, f"smc_v1_asof_{ultima:%Y%m%d}.parquet")
    grupos = {tk: g.reset_index(drop=True) for tk, g in px.groupby("ticker")}

    # control: la ventana de 250 barras da lo mismo que la historia completa
    dif = tot = 0
    for tk in TICKERS_CONTROL:
        g = grupos.get(tk)
        if g is None:
            continue
        k0 = int(np.searchsorted(g["fecha"].values, np.datetime64(DESDE)))
        for t in np.linspace(k0, len(g) - 1, 40).astype(int):
            a = _ultima_fila(g.iloc[max(0, t + 1 - VENTANA_ASOF): t + 1].reset_index(drop=True))
            b = _ultima_fila(g.iloc[: t + 1].reset_index(drop=True))
            tot += 1
            dif += int(not np.allclose(np.nan_to_num(np.array(a[1:], float), nan=-9e9),
                                       np.nan_to_num(np.array(b[1:], float), nan=-9e9)))
    out(f"  control: ventana de {VENTANA_ASOF} barras = historia completa en {tot - dif} de {tot} "
        f"ruedas ({', '.join(TICKERS_CONTROL)})")
    if dif:
        out("  [ALERTA] la ventana no reproduce la historia completa")

    if os.path.exists(ruta) and not recalcular:
        out(f"  reconstruccion: {os.path.relpath(ruta, ROOT)} (ya calculada)")
        return pd.read_parquet(ruta), ultima
    t0 = time.time()
    tareas = [(tk, g["fecha"].values, g["high"].astype(float).values, g["low"].astype(float).values,
               g["close"].astype(float).values) for tk, g in grupos.items()]
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        partes = pool.map(_asof_ticker, tareas, chunksize=1)
    df = pd.DataFrame([f for p in partes for f in p], columns=COLS_ASOF)
    os.makedirs(DIR_CACHE, exist_ok=True)
    df.to_parquet(ruta, index=False)
    out(f"  reconstruccion: {len(df):,} filas, {df['ticker'].nunique()} tickers, "
        f"{time.time() - t0:.0f}s -> {os.path.relpath(ruta, ROOT)}")
    return df, ultima


# --- cobertura de balances ---------------------------------------------------------------

def fin_balances(eh: pd.DataFrame, ruedas: pd.DatetimeIndex):
    """Ultima rueda en que la marca de balance es confiable. earnings_historico se carga con
    cuota (Alpha Vantage) y puede ir atrasada: para cada ticker, el anuncio que deberia seguir
    al ultimo cargado se estima como el del mismo trimestre del anio anterior + 364 dias. Si
    alguno ya deberia haber ocurrido, la re-simulacion se corta MARGEN_BALANCES ruedas antes
    de la rueda en que el bot habria salido por ese balance. Devuelve (fin, detalle)."""
    ult = eh.groupby("ticker")["a"].max()
    esperados = []
    for tk, u in ult.items():
        e = eh.loc[eh["ticker"] == tk, "a"]
        hace_un_anio = e[(e > u - pd.Timedelta(days=300)) & (e <= u - pd.Timedelta(days=200))]
        if len(hace_un_anio):
            esperados.append((hace_un_anio.min() + pd.Timedelta(days=364), tk))
    faltan = sorted(x for x in esperados if x[0] <= ruedas.max())
    if not faltan:
        return ruedas.max(), "earnings_historico completa"
    x, tk = faltan[0]
    k = int(ruedas.searchsorted(x)) - 1          # rueda de salida por ese balance
    fin = ruedas[max(0, k - MARGEN_BALANCES)]
    return fin, (f"{len(faltan)} tickers con un balance esperado y no cargado; el primero {tk} "
                 f"~{x:%Y-%m-%d}")


# --- datos por ticker -------------------------------------------------------------------

class DatosSMC:
    """Por ticker: fechas, dia (ordinal), close, la reconstruccion, balance, score y senal
    de entrada. P = cierres del universo (el mismo indice equal-weight que TECH_SECTOR)."""

    def __init__(self, asof):
        px = query_df("SELECT ticker, fecha, close FROM precios_diarios WHERE fecha >= '2021-01-01' "
                      "ORDER BY ticker, fecha")
        px["fecha"] = pd.to_datetime(px["fecha"])
        fpa = query_df("""SELECT ticker, fecha, es_alcista, patron_engulfing_bull, patron_hammer, vol_spike
                          FROM features_precio_accion WHERE fecha >= :d""", params={"d": DESDE})
        fpa["fecha"] = pd.to_datetime(fpa["fecha"])
        for c in ("es_alcista", "patron_engulfing_bull", "patron_hammer", "vol_spike"):
            fpa[c] = fpa[c].fillna(0).astype(int)
        eh = query_df("SELECT ticker, announcement_date a FROM earnings_historico "
                      "WHERE announcement_date IS NOT NULL")
        eh["a"] = pd.to_datetime(eh["a"])
        anuncios = eh.groupby("ticker")["a"].apply(list).to_dict()
        self.eh = eh
        asof = asof.copy()
        asof["fecha"] = pd.to_datetime(asof["fecha"])

        self.P = px.pivot(index="fecha", columns="ticker", values="close").sort_index().astype(float)
        r1 = self.P.ffill().pct_change(fill_method=None)
        self.U = np.cumprod(1.0 + r1.mean(axis=1, skipna=True).fillna(0.0).to_numpy())
        pos = {f: k for k, f in enumerate(self.P.index)}
        self.fin, self.fin_detalle = fin_balances(eh, self.P.index)

        base = (px.merge(asof, on=["ticker", "fecha"], how="left", indicator="_asof")
                  .merge(fpa, on=["ticker", "fecha"], how="left", indicator="_fpa"))
        self.s = {}
        for tk, g in base.groupby("ticker", sort=True):
            g = g.reset_index(drop=True)
            fechas = list(g["fecha"])
            cl = g["close"].astype(float).to_numpy()
            datos = (g["_asof"] == "both").to_numpy()
            score = np.full(len(g), np.nan)
            for k in np.flatnonzero(datos & (g["_fpa"] == "both").to_numpy()):
                r = g.iloc[k]
                score[k] = calcular_score_estructura({
                    "tuvo_choch_bull": int(r["tuvo_choch_bull"]), "tuvo_bos_bull": int(r["tuvo_bos_bull"]),
                    "estructura_10": int(r["estructura_10"]), "choch_bear_10": int(r["choch_bear_10"]),
                    "dist_sl_10_pct": float(r["dist_sl_10_pct"]), "dist_sh_10_pct": float(r["dist_sh_10_pct"]),
                    "es_alcista": int(r["es_alcista"]), "vol_spike": int(r["vol_spike"]),
                    "patron_engulfing_bull": int(r["patron_engulfing_bull"]),
                    "patron_hammer": int(r["patron_hammer"]), "close": float(r["close"])})[0]
            ret = np.concatenate([[np.nan], cl[1:] / cl[:-1] - 1.0])
            balance = np.array(fs.ruedas_de_balance(fechas, anuncios.get(tk, [])), dtype=bool)
            self.s[tk] = {
                "fechas": fechas, "idx": {f: i for i, f in enumerate(fechas)},
                "dia": np.array([f.toordinal() for f in fechas], dtype=np.int64),
                "close": cl, "datos": datos, "balance": balance,
                "dist10": g["dist_sl_10_pct"].astype(float).to_numpy(),
                "dist5": g["dist_sl_5_pct"].astype(float).to_numpy(),
                "choch10": g["choch_bear_10"].astype(float).to_numpy(),
                "estr10": g["estructura_10"].astype(float).to_numpy(),
                "tuvo_choch": g["tuvo_choch_bull"].astype(float).to_numpy(),
                "tuvo_bos": g["tuvo_bos_bull"].astype(float).to_numpy(),
                "es_alcista": g["es_alcista"].astype(float).to_numpy(),
                "score": score, "senal": np.nan_to_num(score, nan=-1.0) >= 1,
                "sig": pd.Series(ret).rolling(A.VOL_RUEDAS, min_periods=40).std().to_numpy(),
                "g": np.array([pos[f] for f in fechas], dtype=np.int64),
                "n_fin": int(np.searchsorted(np.array(fechas, dtype="datetime64[ns]"),
                                             np.datetime64(self.fin), side="right")),
            }

    def recorte(self, tk, completo=False):
        """Las series de la regla hasta la ultima rueda con balances confiables (o completas)."""
        s = self.s[tk]
        n = len(s["close"]) if completo else s["n_fin"]
        return (s["dia"][:n], s["close"][:n], s["dist10"][:n], s["dist5"][:n], s["choch10"][:n],
                s["estr10"][:n], s["datos"][:n], s["balance"][:n])

    def salida(self, tk, i, regla=fsm.REGLA_ACTUAL, completo=False):
        return fsm.primera_salida(*self.recorte(tk, completo), i, regla)


def entradas_todas(D):
    """Todas las senales, una posicion por ticker a la vez con la salida actual."""
    filas = []
    for tk, s in D.s.items():
        desde = int(np.searchsorted(np.array(s["fechas"], dtype="datetime64[ns]"), np.datetime64(DESDE)))
        libre = (s["senal"] & ~s["balance"])[:s["n_fin"]]
        for i in fsm.entradas_por_ticker(libre, desde, lambda k, tk=tk: D.salida(tk, k)[0]):
            filas.append((tk, i, s["fechas"][i], s["score"][i]))
    return pd.DataFrame(filas, columns=["ticker", "i", "fecha_entrada", "score"])


def entradas_con_tope(D):
    """Las que habria tomado la cartera de 5 posiciones (score desc, desempate por ticker)."""
    cand, dias = {}, set()
    for tk, s in D.s.items():
        desde = int(np.searchsorted(np.array(s["fechas"], dtype="datetime64[ns]"), np.datetime64(DESDE)))
        dias.update(int(x) for x in s["dia"][desde:s["n_fin"]])
        for i in np.flatnonzero((s["senal"] & ~s["balance"])[:s["n_fin"]]):
            if i >= desde:
                cand.setdefault(int(s["dia"][i]), []).append((float(s["score"][i]), tk, (tk, int(i))))

    def sal(clave):
        tk, i = clave
        j = D.salida(tk, i)[0]
        return int(D.s[tk]["dia"][j]) if j is not None else None

    ent = fsm.cartera_con_tope(sorted(dias), cand, sal)
    return pd.DataFrame([(tk, i, D.s[tk]["fechas"][i], D.s[tk]["score"][i]) for tk, i in ent],
                        columns=["ticker", "i", "fecha_entrada", "score"])


def entradas_ft(D):
    op = query_df("""SELECT id, ticker, fecha_datos, fecha_datos_salida, motivo_salida, stop_loss,
                            score_entrada, detalle_entrada
                     FROM ft_operaciones WHERE estrategia_id = :e ORDER BY fecha_datos""",
                  params={"e": ID_SMC_V1})
    for c in ("fecha_datos", "fecha_datos_salida"):
        op[c] = pd.to_datetime(op[c])
    op["detalle"] = op["detalle_entrada"].apply(lambda d: d if isinstance(d, dict) else
                                                (json.loads(d) if d else {}))
    op["i"] = [D.s[t]["idx"].get(f, -1) if t in D.s else -1 for t, f in zip(op["ticker"], op["fecha_datos"])]
    # para la grilla, solo las que entraron con balances confiables
    op["i_grilla"] = [i if (i >= 0 and i < D.s[t]["n_fin"]) else -1 for t, i in zip(op["ticker"], op["i"])]
    op["fecha_entrada"] = op["fecha_datos"]
    return op


# --- metricas: las mismas matrices que TECH_SECTOR --------------------------------------

class MatricesSMC:
    """Una fila por entrada, una columna por regla: rueda y motivo de salida, y las metricas
    que lee ft_analisis_salidas._metricas_regla (RET, TRAMO, POST, Z10, RUEDAS)."""

    def __init__(self, D, ops, reglas):
        T, R = len(ops), len(reglas)
        self.EX = np.full((T, R), -1, dtype=np.int64)
        self.MOT = np.full((T, R), -1, dtype=np.int64)
        self.RUEDAS = np.full((T, R), np.nan)
        self.RET = np.full((T, R), np.nan)
        self.TRAMO = np.full((T, R), np.nan)
        self.POST = {h: np.full((T, R), np.nan) for h in A.HORIZONTES}
        self.Z10 = np.full((T, R), np.nan)
        self.fecha = np.array(pd.to_datetime(ops["fecha_entrada"]).values)
        self.fila_valida = np.zeros(T, dtype=bool)
        U = D.U
        for t, (tk, i) in enumerate(zip(ops["ticker"], ops["i"])):
            if tk not in D.s or i < 0:
                continue
            s = D.s[tk]
            idx, mot = fsm.salidas_reglas(*D.recorte(tk), int(i), reglas)
            self.EX[t], self.MOT[t] = idx, mot
            ok = idx >= 0
            if not ok.any():
                continue
            self.fila_valida[t] = True
            cl, g, n = s["close"], s["g"], len(s["close"])
            x = idx[ok]
            self.RUEDAS[t, ok] = x - i
            ret = cl[x] / cl[i] - 1.0
            self.RET[t, ok] = 100 * ret
            self.TRAMO[t, ok] = 100 * (ret - (U[g[x]] / U[g[i]] - 1.0))
            for hh in A.HORIZONTES:
                y = x + hh
                dentro = y < n
                if not dentro.any():
                    continue
                xs, ys = x[dentro], y[dentro]
                exc = (cl[ys] / cl[xs] - 1.0) - (U[g[ys]] / U[g[xs]] - 1.0)
                cols = np.flatnonzero(ok)[dentro]
                self.POST[hh][t, cols] = 100 * exc
                if hh == A.H_CLASIFICAR:
                    self.Z10[t, cols] = exc / (s["sig"][xs] * np.sqrt(hh))


def control_interno(D, ops, reglas, Mx):
    """La salida vectorizada = la de referencia rueda por rueda, en todas las reglas."""
    igual = total = 0
    for t, (tk, i) in enumerate(zip(ops["ticker"], ops["i"])):
        if tk not in D.s or i < 0:
            continue
        for k, r in enumerate(reglas):
            j, m, _ = D.salida(tk, int(i), r)
            total += 1
            igual += int(Mx.EX[t, k] == (j if j is not None else -1)
                         and Mx.MOT[t, k] == (fsm.MOTIVOS.index(m) if m else -1))
    return igual, total


# --- paso 0 ------------------------------------------------------------------------------

def seccion_p0(out, dirout, D, asof, ultima, ops, tope, ft):
    out.titulo("PASO 0 -- la reconstruccion y la re-simulacion reproducen a FT_SMC_v1?")

    # (0) la ultima rueda reconstruida = la tabla que lee el bot hoy
    tab = query_df("""SELECT DISTINCT ON (ticker) ticker, fecha, estructura_10, choch_bear_10, dist_sl_10_pct
                      FROM features_market_structure ORDER BY ticker, fecha DESC""")
    tab["fecha"] = pd.to_datetime(tab["fecha"])
    ult = asof[pd.to_datetime(asof["fecha"]) == ultima]
    m = ult.merge(tab, on=["ticker", "fecha"], suffixes=("", "_tab"))
    ok = ((m["estructura_10"] == m["estructura_10_tab"]) & (m["choch_bear_10"] == m["choch_bear_10_tab"])
          & ((m["dist_sl_10_pct"] - m["dist_sl_10_pct_tab"].astype(float)).abs() < 1e-3))
    out(f"  (0) rueda {ultima:%Y-%m-%d}: reconstruccion = features_market_structure (lo que ve el bot) "
        f"en {int(ok.sum())} de {len(m)} tickers")

    # (a) lo que el bot guardo al entrar
    cmp_ = {"dist_sl": 0, "estructura": 0, "tuvo_choch": 0, "tuvo_bos": 0, "vela": 0, "senal": 0, "score": 0}
    n = 0
    malas = []
    for r in ft.itertuples():
        if r.i < 0:
            continue
        s, d, i = D.s[r.ticker], r.detalle, r.i
        if not s["datos"][i]:
            continue
        n += 1
        c = {"dist_sl": abs(s["dist10"][i] - float(d.get("dist_sl_pct", np.nan))) < 0.01,
             "estructura": int(s["estr10"][i]) == int(d.get("estructura_10", -9)),
             "tuvo_choch": bool(s["tuvo_choch"][i]) == bool(d.get("tuvo_choch_bull")),
             "tuvo_bos": bool(s["tuvo_bos"][i]) == bool(d.get("tuvo_bos_bull")),
             "vela": bool(s["es_alcista"][i]) == bool(d.get("es_alcista")),
             "senal": bool(s["senal"][i]),
             "score": abs(np.nan_to_num(s["score"][i], nan=-9) - float(r.score_entrada or -9)) < 1e-9}
        for k, v in c.items():
            cmp_[k] += int(v)
        if not all(c.values()):
            malas.append((r.ticker, r.fecha_datos, [k for k, v in c.items() if not v]))
    out(f"  (a) {len(ft)} entradas de FT, {n} con reconstruccion en su rueda. Coincide con lo que guardo el bot:")
    out("      " + " | ".join(f"{k} {v}/{n}" for k, v in cmp_.items()))
    for tk, f, que in malas[:12]:
        out(f"        {tk:<6} {f:%Y-%m-%d} difiere en: {', '.join(que)}")

    # (b) la regla actual re-simulada sobre las cerradas
    cerr = ft[ft["fecha_datos_salida"].notna() & (ft["i"] >= 0)
              & ~ft["motivo_salida"].astype(str).str.contains("SPLIT_FIX")].copy()
    ev = query_df("SELECT operacion_id, fecha_datos FROM ft_posiciones_diarias WHERE estrategia_id = :e "
                  "AND fecha_datos IS NOT NULL", params={"e": ID_SMC_V1})
    evaluadas = set(zip(ev["operacion_id"], pd.to_datetime(ev["fecha_datos"])))
    filas = []
    for r in cerr.itertuples():
        j, mot, sl = D.salida(r.ticker, r.i, completo=True)
        fs_ = D.s[r.ticker]["fechas"][j] if j is not None else pd.NaT
        real = fsm.familia_ft(r.motivo_salida)
        anuncio = D.eh[(D.eh["ticker"] == r.ticker) & (D.eh["a"] > r.fecha_datos_salida)
                       & (D.eh["a"] <= r.fecha_datos_salida + pd.Timedelta(days=10))]
        if real == "BALANCE" and mot != "BALANCE" and anuncio.empty:
            clase = "balance que falta en earnings_historico"
        elif j is not None and fs_ == r.fecha_datos_salida:
            clase = "igual" if mot == real else "misma rueda, otro motivo"
        elif j is not None and fs_ < r.fecha_datos_salida and (r.id, fs_) not in evaluadas:
            clase = "rutina: el bot no evaluo esa rueda"
        elif mot == "TIEMPO" and real == "TIEMPO":
            clase = "time stop: reloj del bot vs dias de datos"
        else:
            clase = "otra"
        filas.append({"id": r.id, "ticker": r.ticker, "entrada": r.fecha_datos, "resim": fs_,
                      "motivo_resim": mot, "ft": r.fecha_datos_salida, "motivo_ft": r.motivo_salida,
                      "clase": clase, "sl_resim": sl, "sl_ft": float(r.stop_loss) if r.stop_loss else np.nan})
    rs = pd.DataFrame(filas)
    rs.to_csv(os.path.join(dirout, "p0_resimulacion_vs_ft.csv"), index=False)
    out(f"  (b) regla ACTUAL re-simulada sobre las {len(rs)} cerradas de FT (series completas):")
    for k, v in rs["clase"].value_counts().items():
        out(f"      {k:<45} {v:>3}")
    for r in rs[rs["clase"].isin(["otra", "misma rueda, otro motivo"])].head(12).itertuples():
        out(f"        {r.ticker:<6} entrada {r.entrada:%Y-%m-%d} | re-simulada "
            f"{(r.resim.strftime('%Y-%m-%d') if pd.notna(r.resim) else 'sin salida'):<10} {str(r.motivo_resim):<10} "
            f"| FT {r.ft:%Y-%m-%d} {r.motivo_ft}")

    # (c) el stop final
    mismo = rs[rs["clase"] == "igual"]
    igual_sl = ((mismo["sl_resim"] - mismo["sl_ft"]).abs() / mismo["sl_ft"] < 1e-3).sum()
    out(f"  (c) stop final re-simulado = el del bot (+-0,1%) en {int(igual_sl)} de {len(mismo)} "
        f"salidas iguales")

    # (d) anatomia sobre la historia
    out("")
    out(f"  corte por balances: la re-simulacion llega hasta {D.fin:%Y-%m-%d} ({D.fin_detalle})")
    out(f"  (d) ENTRADAS historicas ({DESDE} -> {D.fin:%Y-%m-%d}): todas las senales {len(ops):,} "
        f"({ops['ticker'].nunique()} tickers) | con tope de 5: {len(tope):,}")
    por_anio = ops.groupby(pd.to_datetime(ops["fecha_entrada"]).dt.year).size()
    out("      por anio: " + " | ".join(f"{a} {v}" for a, v in por_anio.items()))
    for nombre, regla in (("regla actual", fsm.REGLA_ACTUAL),
                          ("sin stop (CHoCH y estructura activas)", Regla("sin", True, True, 20))):
        mots, ruedas, stop_con_choch = [], [], 0
        for tk, i in zip(ops["ticker"], ops["i"]):
            j, m_, _ = D.salida(tk, int(i), regla)
            mots.append(m_ or "SIN_SALIDA")
            if j is not None:
                ruedas.append(j - i)
                if m_ == "STOP" and D.s[tk]["choch10"][j] == 1:
                    stop_con_choch += 1
        vc = pd.Series(mots).value_counts()
        out(f"      {nombre}: " + " | ".join(f"{k} {v} ({100 * v / len(mots):.0f}%)" for k, v in vc.items())
            + f" | ruedas medias {np.mean(ruedas):.1f}")
        if regla == fsm.REGLA_ACTUAL and vc.get("STOP", 0):
            out(f"        salidas por stop con CHoCH bajista esa misma rueda: {stop_con_choch} de {vc['STOP']}")
    return rs


# --- la grilla ---------------------------------------------------------------------------

def _grupos(Mx, reglas):
    """Reglas que dan la misma salida en todas las operaciones -> una sola medicion."""
    vistos, rep, miembros = {}, [], OrderedDict()
    for k in range(len(reglas)):
        clave = Mx.EX[:, k].tobytes()
        if clave not in vistos:
            vistos[clave] = k
            rep.append(k)
            miembros[k] = []
        miembros[vistos[clave]].append(k)
    return rep, miembros


def _nombre(reglas, miembros, k):
    extra = len(miembros[k]) - 1
    return fsm.etiqueta(reglas[k]) + (f"  (+{extra} iguales)" if extra else "")


def seccion_grilla(out, dirout, D, ultima, ops, tope, ft):
    out.titulo(f"GRILLA -- la SALIDA de SMC_v1 ({DESDE} -> {D.fin:%Y-%m-%d})")
    out(f"  corte por balances: {D.fin_detalle}; despues de {D.fin:%Y-%m-%d} no se re-simula (las")
    out("  ruedas posteriores solo se usan para medir que hizo el precio despues de salir).")
    out("  Pre-registro: docs/forward_testing/ANALISIS_SALIDAS.md sec. 10.4. Entradas fijas (todas las")
    out("  senales, una posicion por ticker); cada regla re-simulada. post10 = exceso contra el universo")
    out("  en las 10 ruedas DESPUES de la salida; dif_post / dif_tram = regla menos actual en la misma")
    out("  operacion (post: negativo = mejor; tramo: positivo = mejor); p5 = percentil 5 por operacion.")
    reglas = fsm.grilla()
    a = 0
    Mx = MatricesSMC(D, ops, reglas)
    igual, total = control_interno(D, ops, reglas, Mx)
    out(f"  control interno: salida vectorizada = referencia rueda por rueda en {igual:,} de {total:,} "
        f"(96 reglas x entradas)")
    if igual != total:
        out("  [ERROR] la re-simulacion vectorizada no coincide: se frena.")
        return
    rep, miembros = _grupos(Mx, reglas)
    out(f"  96 combinaciones -> {len(rep)} reglas distintas (las demas dan exactamente las mismas salidas)")
    sel = Mx.fila_valida & (Mx.fecha < np.datetime64(A.FECHA_CONFIRMACION))
    conf = Mx.fila_valida & (Mx.fecha >= np.datetime64(A.FECHA_CONFIRMACION))
    out(f"  entradas: {len(ops):,} | seleccion (2021-09 -> 2024-12): {int(sel.sum()):,} | "
        f"confirmacion (2025-01 -> {D.fin:%Y-%m}): {int(conf.sum()):,}")

    # PRIMERO: la regla actual, que hizo el precio despues de cada tipo de salida
    ftg = ft.assign(i=ft["i_grilla"])
    Mft = MatricesSMC(D, ftg, reglas)
    out(f"  FT de control: {int((ftg['i'] >= 0).sum())} de {len(ft)} entradas antes del corte")
    out("")
    out("  LA REGLA ACTUAL (FT de control): despues de cada tipo de salida (post10, pp contra el universo)")
    out(f"  {'muestra':<22} {'salida':<11} {'ops':>5} {'ruedas':>6} {'ret':>7} {'post10':>7} "
        f"{'a_tie':>6} {'tempr':>6} {'later':>6}")
    for lab, M, filas in (("seleccion 2021-24", Mx, sel), ("confirmacion 2025-26", Mx, conf),
                          ("FT desde 24/4", Mft, Mft.fila_valida)):
        for m_code, m_name in [(-9, "TODAS")] + list(enumerate(fsm.MOTIVOS)):
            f_ = filas & ((M.MOT[:, a] == m_code) if m_code >= 0 else (M.MOT[:, a] >= 0))
            if f_.sum() == 0:
                continue
            p = M.POST[A.H_CLASIFICAR][f_, a]
            z = M.Z10[f_, a]
            z = z[~np.isnan(z)]
            out(f"  {lab:<22} {m_name:<11} {int(f_.sum()):>5} {np.nanmean(M.RUEDAS[f_, a]):>6.1f} "
                f"{np.nanmean(M.RET[f_, a]):>+7.2f} {np.nanmean(p):>+7.2f} "
                f"{100 * (z < -fs.UMBRAL_Z).mean():>5.0f}% {100 * (z > fs.UMBRAL_Z).mean():>5.0f}% "
                f"{100 * (np.abs(z) <= fs.UMBRAL_Z).mean():>5.0f}%")

    met_sel = {k: A._metricas_regla(Mx, k, a, sel) for k in rep}
    met_conf = {k: A._metricas_regla(Mx, k, a, conf) for k in rep}
    filas_csv = [{"regla": fsm.etiqueta(reglas[k]), "iguales": len(miembros[k]),
                  "miembros": " / ".join(fsm.etiqueta(reglas[q]) for q in miembros[k]),
                  **{f"sel_{c}": v for c, v in met_sel[k].items() if not c.startswith("_")},
                  **{f"conf_{c}": v for c, v in met_conf[k].items() if not c.startswith("_")}}
                 for k in rep]
    pd.DataFrame(filas_csv).to_csv(os.path.join(dirout, "grilla.csv"), index=False)

    def rep_de(regla):
        k = reglas.index(regla)
        return next(q for q, ms in miembros.items() if k in ms)

    out("")
    out("  PALANCAS (periodo de seleccion):")
    A._encabezado(out)
    A._linea(out, "actual", met_sel[a])
    for nombre, regla in PALANCAS:
        k = rep_de(regla)
        A._linea(out, nombre + ("  [= actual]" if k == a else ""), met_sel[k])

    cand = [k for k in rep if k != a and fs.es_candidata(
        met_sel[k]["dif_post"], met_sel[k]["dif_tramo"], met_sel[k]["p5"], met_sel[k]["p5_actual"])]
    out("")
    out(f"  SELECCION: {len(cand)} de {len(rep) - 1} reglas son candidatas "
        f"(dif_post < 0, dif_tramo >= 0, cola no cae mas de 1 punto)")
    out(f"    dif_post < 0 en {sum(1 for k in rep if k != a and met_sel[k]['dif_post'] < 0)} | "
        f"dif_tramo >= 0 en {sum(1 for k in rep if k != a and met_sel[k]['dif_tramo'] >= 0)}")
    orden = sorted(cand, key=lambda k: met_sel[k]["dif_post"])
    if orden:
        out("  candidatas con dif_post mas negativa:")
        A._encabezado(out)
        for k in orden[:10]:
            A._linea(out, fsm.etiqueta(reglas[k]), met_sel[k])

    out.titulo(f"GRILLA -- CONFIRMACION (2025-01 -> {D.fin:%Y-%m-%d}), FT de control y muestra con tope")
    elegidas = orden[:A.TOP_SELECCION]
    if not elegidas:
        out("  ninguna regla es candidata en la seleccion: no hay nada que confirmar.")
    Mtope = MatricesSMC(D, tope, reglas)
    res = {}
    for n_, k in enumerate(elegidas, 1):
        mc = met_conf[k]
        icp = fs.ic95_por_dia(list(mc["_dpost"][conf]), list(Mx.fecha[conf]))
        ict = fs.ic95_por_dia(list(mc["_dtramo"][conf]), list(Mx.fecha[conf]))
        c_post = icp["hi"] == icp["hi"] and icp["hi"] < 0
        c_tramo = ict["media"] == ict["media"] and ict["media"] >= 0
        c_cola = (mc["p5_actual"] - mc["p5"]) <= fs.CAIDA_COLA_MAX_PP
        ok = c_post and c_tramo and c_cola
        mf = A._metricas_regla(Mft, k, a, Mft.fila_valida)
        icf = fs.ic95_por_dia(list(mf["_dpost"][Mft.fila_valida]), list(Mft.fecha[Mft.fila_valida]))
        mt = A._metricas_regla(Mtope, k, a, Mtope.fila_valida)
        out("")
        out(f"  [{n_}] {_nombre(reglas, miembros, k)}")
        out(f"      seleccion : dif_post {met_sel[k]['dif_post']:+.3f} | dif_tramo {met_sel[k]['dif_tramo']:+.3f}")
        out(f"      confirmac.: dif_post {icp['media']:+.3f} IC95 [{icp['lo']:+.3f}; {icp['hi']:+.3f}] -> "
            f"{'SI' if c_post else 'no'} | dif_tramo {ict['media']:+.3f} -> {'SI' if c_tramo else 'no'} | "
            f"p5 {mc['p5_actual']:+.2f} -> {mc['p5']:+.2f} -> {'SI' if c_cola else 'no'}")
        out(f"      ==> {'CONFIRMADA' if ok else 'NO SE CONFIRMA'}")
        out(f"      FT de control: dif_post {icf['media']:+.3f} IC95 [{icf['lo']:+.3f}; {icf['hi']:+.3f}] | "
            f"dif_tramo {mf['dif_tramo']:+.3f} | cambia {mf['cambian']} de {mf['ops']}")
        out(f"      con tope de 5: dif_post {mt['dif_post']:+.3f} | dif_tramo {mt['dif_tramo']:+.3f} | "
            f"p5 {mt['p5_actual']:+.2f} -> {mt['p5']:+.2f}")
        res[fsm.etiqueta(reglas[k])] = {"confirmada": ok, "dif_post_conf": icp["media"],
                                        "ic_post_conf": [icp["lo"], icp["hi"]],
                                        "dif_tramo_conf": ict["media"], "ft_dif_post": icf["media"]}
    json.dump(res, open(os.path.join(dirout, "grilla_resultado.json"), "w", encoding="utf-8"),
              indent=2, default=float)

    _complemento(out, reglas, miembros, rep, a, Mx, Mft, Mtope, sel, conf, met_sel, met_conf)
    out(f"  detalle de las {len(rep)} reglas distintas: {os.path.join(dirout, 'grilla.csv')}")


def _complemento(out, reglas, miembros, rep, a, Mx, Mft, Mtope, sel, conf, met_sel, met_conf):
    """Lo que no estaba pre-registrado y no cambia la lectura: IC95 de las palancas por
    periodo, en FT y con tope, por anio; y si el orden de la seleccion se sostiene."""
    out.titulo("GRILLA -- COMPLEMENTO DESCRIPTIVO (no pre-registrado; no cambia la lectura)")
    anio = pd.DatetimeIndex(Mx.fecha).year.values

    def f(x):
        return f"{x['media']:+.3f} [{x['lo']:+.3f}; {x['hi']:+.3f}]"

    def ic(M, k, filas):
        m = A._metricas_regla(M, k, a, filas)
        return (m, fs.ic95_por_dia(list(m["_dpost"][filas]), list(M.fecha[filas])),
                fs.ic95_por_dia(list(m["_dtramo"][filas]), list(M.fecha[filas])))

    out("  Diferencia pareada contra la actual, pp, IC95 por rueda de entrada. post = 10 ruedas")
    out("  despues de la salida (negativo = mejor); tramo = entrada -> salida (positivo = mejor).")
    for nombre, regla in PALANCAS:
        kk = reglas.index(regla)
        k = next(q for q, ms in miembros.items() if kk in ms)
        if k == a:
            out(f"  {nombre}: = regla actual")
            continue
        out(f"  {nombre}")
        for lab, M, filas in (("seleccion 2021-24", Mx, sel), ("confirmac 2025-26", Mx, conf),
                              ("FT de control    ", Mft, Mft.fila_valida),
                              ("con tope de 5    ", Mtope, Mtope.fila_valida)):
            mm, p, t = ic(M, k, filas)
            out(f"    {lab}: post {f(p)} | tramo {f(t)} | p5 {mm['p5_actual']:+.2f} -> {mm['p5']:+.2f}"
                f" | ruedas {mm['ruedas']:.1f}")
        por_anio = []
        for y in sorted(set(anio[Mx.fila_valida])):
            _, p, t = ic(Mx, k, Mx.fila_valida & (anio == y))
            por_anio.append(f"{y} {p['media']:+.2f}/{t['media']:+.2f}")
        out("    por anio (post/tramo): " + " | ".join(por_anio))

    otros = [k for k in rep if k != a]
    if len(otros) >= 3:
        sp = np.array([met_sel[k]["dif_post"] for k in otros])
        cp = np.array([met_conf[k]["dif_post"] for k in otros])
        st = np.array([met_sel[k]["dif_tramo"] for k in otros])
        ct = np.array([met_conf[k]["dif_tramo"] for k in otros])
        ru = np.array([met_sel[k]["ruedas"] for k in otros])
        caida = np.array([met_sel[k]["p5_actual"] - met_sel[k]["p5"] for k in otros])
        ok = ~np.isnan(sp) & ~np.isnan(cp) & ~np.isnan(st) & ~np.isnan(ct)
        out("")
        out(f"  correlacion entre reglas (seleccion): ruedas en posicion con caida del p5 "
            f"{np.corrcoef(ru[ok], caida[ok])[0, 1]:+.2f} | con dif_post {np.corrcoef(ru[ok], sp[ok])[0, 1]:+.2f}"
            f" | con dif_tramo {np.corrcoef(ru[ok], st[ok])[0, 1]:+.2f}")
        out(f"  el orden de la seleccion en la confirmacion: correlacion sel vs conf de dif_post "
            f"{np.corrcoef(sp[ok], cp[ok])[0, 1]:+.2f}, de dif_tramo {np.corrcoef(st[ok], ct[ok])[0, 1]:+.2f}")
        ab = [k for k in otros if met_sel[k]["dif_post"] < 0 and met_sel[k]["dif_tramo"] >= 0]
        if ab:
            out(f"  ignorando la cola: {len(ab)} reglas pasan (a) y (b) en la seleccion; en la confirmacion "
                f"dif_post va de {min(met_conf[k]['dif_post'] for k in ab):+.3f} a "
                f"{max(met_conf[k]['dif_post'] for k in ab):+.3f} y dif_tramo >= 0 en "
                f"{sum(1 for k in ab if met_conf[k]['dif_tramo'] >= 0)}")
    x = Mx.POST[A.H_CLASIFICAR][sel, a]
    x = x[~np.isnan(x)]
    out(f"  escala: post10 de la regla actual en la seleccion, desvio por operacion {x.std():.2f} pp")


# --- main --------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Analisis de salidas de FT_SMC_v1 (solo lee)")
    ap.add_argument("--seccion", choices=("p0", "grilla", "todas"), default="todas")
    ap.add_argument("--etiqueta", default="smc_v1",
                    help="sufijo del directorio reportes/analisis_salidas/AAAAMMDD_<etiqueta>/")
    ap.add_argument("--recalcular", action="store_true", help="rehace la reconstruccion")
    args = ap.parse_args()

    dirout = os.path.join(A.DIR_BASE, f"{date.today():%Y%m%d}_{args.etiqueta}")
    os.makedirs(dirout, exist_ok=True)
    out = A.Salida(os.path.join(dirout, f"{args.seccion}.log"))
    out.titulo("RECONSTRUCCION -- lo que FT_SMC_v1 veia cada rueda")
    asof, ultima = reconstruir(out, args.recalcular)
    json.dump({
        "fecha_corrida": str(date.today()), "ultima_rueda_precios": f"{ultima:%Y-%m-%d}",
        "git": A._git(), "seccion": args.seccion, "desde": DESDE, "ventana_asof": VENTANA_ASOF,
        "lookback_dias": LOOKBACK_DIAS, "fecha_confirmacion": str(A.FECHA_CONFIRMACION)[:10],
        "horizontes": A.HORIZONTES, "h_clasificar": A.H_CLASIFICAR, "umbral_z": fs.UMBRAL_Z,
        "regla_actual": fsm.etiqueta(fsm.REGLA_ACTUAL), "stops": fsm.STOPS,
        "tiempos": [t if t is not None else "sin" for t in fsm.TIEMPOS],
    }, open(os.path.join(dirout, "parametros.json"), "w", encoding="utf-8"), indent=2)

    D = DatosSMC(asof)
    ops = entradas_todas(D)
    tope = entradas_con_tope(D)
    ft = entradas_ft(D)
    ops.to_csv(os.path.join(dirout, "entradas_todas.csv"), index=False)
    tope.to_csv(os.path.join(dirout, "entradas_con_tope.csv"), index=False)
    if args.seccion in ("p0", "todas"):
        seccion_p0(out, dirout, D, asof, ultima, ops, tope, ft)
    if args.seccion in ("grilla", "todas"):
        seccion_grilla(out, dirout, D, ultima, ops, tope, ft)
    out.cerrar()
    print(f"\nResultados en {dirout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
