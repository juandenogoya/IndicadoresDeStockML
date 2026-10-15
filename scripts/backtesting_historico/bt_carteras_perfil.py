"""
bt_carteras_perfil.py
Backtest pre-registrado de las CARTERAS POR PERFIL
(docs/forward_testing/estrategias/CARTERAS_PERFIL.md). Solo LEE la DB local.

QUE HACE
    1. Perfil de riesgo de cada ticker MES A MES con el motor puro de produccion
       (perfil_metricas + perfil_riesgo), con datos hasta la rueda previa al mes y
       la barra en curso anclada a la fecha de la rueda (volatilidad_mtf la ancla
       al reloj: sin esto el perfil no es point-in-time).
    2. Elegibles por rueda de las familias reconstruibles:
         Tecnica = src/strategies/scoring.calcular_score_tecnico >= 4,0
         SMC     = ft_scoring.calcular_score_estructura >= SCORE_ENTRADA_MIN de
                   FT_SMC_v3, sobre features_estructura N5 o N3 (el maximo)
       ML NO entra (sus modelos vieron estos anios al entrenar).
    3. Seleccion x 20 semillas y azar dentro del perfil x 20 semillas, por cartera,
       con src/utils/carteras_perfil (SL/TP sobre el close, balance transversal).
    4. Regla de lectura pre-registrada.

VALIDACIONES (correrlas ANTES del backtest)
    --validar-perfil  el perfil recalculado al 2026-08-06 tiene que dar igual al
                      snapshot guardado en perfiles_ticker (sale 1 si no).
    --validar-smc     los elegibles de SMC reconstruidos tienen que coincidir con
                      los candidatos que registraron FT_SMC_v3_N5 / N3.

Uso:
    python scripts/backtesting_historico/bt_carteras_perfil.py --validar-perfil
    python scripts/backtesting_historico/bt_carteras_perfil.py --validar-smc
    python scripts/backtesting_historico/bt_carteras_perfil.py            (backtest)
    python scripts/backtesting_historico/bt_carteras_perfil.py --semillas 2 --hasta 2022-06-30
    python scripts/backtesting_historico/bt_carteras_perfil.py --salidas Arriesgada,Especulativa --con-seleccion
        (diagnostico de salidas V0-V4; con seleccion al azar por defecto)

Salida del backtest: reportes/carteras_perfil/AAAAMMDD_HHMM/ (fuera de git):
parametros.json, resultados.json, resumen.txt.
"""

import os
import sys
import json
import time
import bisect
import random
import argparse
from datetime import date, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "forward_testing"))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass
os.environ.pop("DATABASE_URL", None)

import pandas as pd                                                      # noqa: E402

from src.data.database import query_df                                   # noqa: E402
from src.utils import carteras_perfil as cp                              # noqa: E402
from src.utils import perfil_metricas, perfil_riesgo                     # noqa: E402
from src.utils import volatilidad_mtf as vm                              # noqa: E402
from src.strategies.scoring import calcular_score_tecnico                # noqa: E402
from src.indicators.earnings_filter import dias_habiles_hasta            # noqa: E402
from scripts.forward_testing.ft_scoring import (                         # noqa: E402
    calcular_score_estructura, LOOKBACK_DIAS)
from scripts.forward_testing.ft_bot_smc_v3 import SCORE_ENTRADA_MIN as SMC_MIN  # noqa: E402
from ft_analisis_salidas_smc import fin_balances                         # noqa: E402

DESDE = date(2021, 9, 1)
HASTA = date(2026, 9, 30)
SCORE_TECNICO_MIN = 4.0          # src/strategies/sectorial.py SectorialConfig.score_entrada
VENTANAS_SMC = (5, 3)
N_SEMILLAS = 20
SEMILLA_AZAR = 10_000            # offset: las semillas del azar no son las de la seleccion
BENCH = "ES=F"
MIN_COBERTURA = 0.5              # una fecha es rueda si tiene >= la mitad de los tickers vivos


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _d(x):
    """date puro. Ojo: Timestamp y datetime SON subclase de date."""
    return x.date() if isinstance(x, datetime) else x


# ── Perfil point-in-time ──────────────────────────────────────────────────────

def atr_pit(df, d_ref):
    """ATR% semanal y mensual excluyendo la barra que contiene d_ref (no el reloj)."""
    out = {}
    for clave, anchor, per in (("atr_pct_w", "W-FRI", vm.ATR_PERIOD_W),
                               ("atr_pct_m", "M", vm.ATR_PERIOD_M)):
        b = vm.resample_ohlc(df, anchor)
        if len(b):
            b = b[pd.to_datetime(b["fecha"]).dt.to_period(anchor)
                  < pd.Timestamp(d_ref).to_period(anchor)]
        out[clave] = vm.atr_pct(b, per)
    return out


def perfil_en(d_ref, corte, px_por_ticker, bench, sectores, tickers=None):
    """
    Perfil de cada ticker con datos <= corte, barras en curso ancladas a d_ref.
    Devuelve {ticker: clasificacion de perfil_riesgo}.
    """
    b = bench[bench["fecha"] <= corte]
    rows = []
    for tk in (tickers or sorted(px_por_ticker)):
        df = px_por_ticker.get(tk)
        if df is None:
            continue
        df = df[df["fecha"] <= corte]
        if len(df) < 2:
            continue
        met = perfil_metricas.metricas_ticker(df, b)
        met.update(atr_pit(df, d_ref))
        rows.append({"ticker": tk, "sector": sectores.get(tk), "metricas": met})
    return {c["ticker"]: c for c in perfil_riesgo.perfilar_universo(rows)}


# ── Carga ─────────────────────────────────────────────────────────────────────

class Datos:
    def __init__(self, hasta):
        t0 = time.time()
        # TODOS los tickers de activos, tambien los dados de baja (soft delete): usar
        # solo los vivos hoy es sesgo de supervivencia (WBD, baja 14/10/2026, estuvo en
        # el universo todo el periodo). Cada ticker participa desde su primer precio.
        act = query_df("SELECT ticker, sector FROM activos")
        self.sectores = dict(zip(act.ticker, act.sector))
        tks = list(self.sectores)
        p = {"t": tks, "h": hasta}

        px = query_df("""SELECT ticker, fecha, open, high, low, close FROM precios_diarios
                         WHERE ticker = ANY(:t) AND fecha <= :h""", p)
        px["fecha"] = px["fecha"].map(_d)
        for c in ("open", "high", "low", "close"):
            px[c] = px[c].astype(float)
        self.px = px
        self.px_por_ticker = {tk: g.sort_values("fecha").reset_index(drop=True)
                              for tk, g in px.groupby("ticker")}
        bench = query_df("SELECT fecha, close FROM futuros_diarios WHERE ticker = :b "
                         "AND fecha <= :h ORDER BY fecha", {"b": BENCH, "h": hasta})
        bench["fecha"] = bench["fecha"].map(_d)
        bench["close"] = bench["close"].astype(float)
        self.bench = bench

        ind = query_df("""SELECT ticker, fecha, sma21, sma50, sma200, rsi14, macd, macd_signal
                          FROM indicadores_tecnicos WHERE ticker = ANY(:t)
                          AND fecha BETWEEN '2021-06-01' AND :h""", p)
        ind["fecha"] = ind["fecha"].map(_d)
        self.ind = ind

        cols = ", ".join(f"choch_bull_{n}, bos_bull_{n}, estructura_{n}, choch_bear_{n}, "
                         f"dist_sh_{n}_pct, dist_sl_{n}_pct" for n in VENTANAS_SMC)
        est = query_df(f"""SELECT e.ticker, e.fecha, {cols},
                                  v.patron_engulfing_bull, v.patron_hammer,
                                  COALESCE(pa.vol_spike, 0) AS vol_spike
                           FROM features_estructura e
                           LEFT JOIN features_velas v ON v.ticker = e.ticker AND v.fecha = e.fecha
                           LEFT JOIN features_precio_accion pa
                                  ON pa.ticker = e.ticker AND pa.fecha = e.fecha
                           WHERE e.ticker = ANY(:t) AND e.fecha BETWEEN '2021-06-01' AND :h""", p)
        est["fecha"] = est["fecha"].map(_d)
        self.est = est

        eh = query_df("SELECT ticker, announcement_date AS a FROM earnings_historico "
                      "WHERE announcement_date IS NOT NULL AND ticker = ANY(:t)", p)
        eh["a"] = pd.to_datetime(eh["a"])
        self.eh = eh
        self.balances = {tk: sorted(_d(x) for x in g["a"]) for tk, g in eh.groupby("ticker")}
        log(f"Datos cargados en {time.time() - t0:.0f}s: {len(px)} precios, "
            f"{len(ind)} indicadores, {len(est)} filas de estructura")

    # -- ruedas --------------------------------------------------------------
    def ruedas(self, desde, hasta):
        primera = self.px.groupby("ticker")["fecha"].min()
        cnt = self.px.groupby("fecha").size()
        out = []
        for f, n in cnt.items():
            if f < desde or f > hasta:
                continue
            vivos = int((primera <= f).sum())
            if n >= MIN_COBERTURA * vivos:
                out.append(f)
        return sorted(out)

    # -- elegibles -------------------------------------------------------------
    def elegibles_tecnica(self):
        """{fecha: [(ticker, score, sector)]} con score tecnico >= 4,0."""
        m = self.ind.merge(self.px[["ticker", "fecha", "close"]], on=["ticker", "fecha"])
        out = {}
        for r in m.to_dict("records"):
            s, _ = calcular_score_tecnico(r)
            if s >= SCORE_TECNICO_MIN:
                out.setdefault(r["fecha"], []).append((r["ticker"], s, self.sectores.get(r["ticker"])))
        return out

    def elegibles_smc(self, ventanas=VENTANAS_SMC):
        """
        {fecha: {ticker: (score_max, {n: score})}} con score >= SCORE_ENTRADA_MIN en
        alguna ventana. Replica ft_scoring_estructura.sql_features_hoy rueda por rueda:
        evento CHoCH/BOS bull en [D - LOOKBACK_DIAS, D] (dias calendario), estado de D,
        es_alcista = close > open de D, mismas claves *_10 que lee el score.
        """
        e = self.est.merge(self.px[["ticker", "fecha", "open", "close"]],
                           on=["ticker", "fecha"], how="inner")
        e = e.sort_values(["ticker", "fecha"]).reset_index(drop=True)
        e["_ts"] = pd.to_datetime(e["fecha"])
        out = {}
        ventana_dias = f"{LOOKBACK_DIAS + 1}D"       # (D-13d, D] = fecha >= D - 12 dias
        for n in ventanas:
            for ev in ("choch_bull", "bos_bull"):
                col = f"{ev}_{n}"
                e[f"_tuvo_{ev}"] = (e.set_index("_ts").groupby("ticker")[col]
                                    .rolling(ventana_dias).max().fillna(0).to_numpy())
            cand = e[(e["_tuvo_choch_bull"] > 0) | (e["_tuvo_bos_bull"] > 0)]
            for r in cand.to_dict("records"):
                row = {
                    "tuvo_choch_bull": int(r["_tuvo_choch_bull"] > 0),
                    "tuvo_bos_bull": int(r["_tuvo_bos_bull"] > 0),
                    "estructura_10": r[f"estructura_{n}"],
                    "choch_bear_10": r[f"choch_bear_{n}"],
                    "dist_sl_10_pct": r[f"dist_sl_{n}_pct"],
                    "dist_sh_10_pct": r[f"dist_sh_{n}_pct"],
                    "es_alcista": int(r["close"] > r["open"]),
                    "vol_spike": r["vol_spike"],
                    "patron_engulfing_bull": r["patron_engulfing_bull"],
                    "patron_hammer": r["patron_hammer"],
                    "close": r["close"],
                }
                row = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
                s, _ = calcular_score_estructura(row)
                if s < SMC_MIN:
                    continue
                d = out.setdefault(r["fecha"], {})
                smax, por_n = d.get(r["ticker"], (-1, {}))
                por_n[n] = s
                d[r["ticker"]] = (max(smax, s), por_n)
        return out

    # -- balances ----------------------------------------------------------------
    def con_balance(self, fecha):
        """Tickers con balance en `fecha` o en la rueda habil siguiente (regla de
        earnings_filter: cierra la rueda anterior y bloquea con buffer de 1)."""
        out = set()
        for tk, fechas in self.balances.items():
            i = bisect.bisect_left(fechas, fecha)
            if i < len(fechas) and dias_habiles_hasta(fecha, fechas[i]) <= 1:
                out.add(tk)
        return out


# ── Validaciones ──────────────────────────────────────────────────────────────

def validar_perfil(datos):
    snap = query_df("SELECT ticker, sector, perfil, score_riesgo, fecha_datos "
                    "FROM perfiles_ticker WHERE fecha = '2026-08-06'")
    corte = _d(snap["fecha_datos"].max())
    sect = dict(zip(snap.ticker, snap.sector))
    pp = perfil_en(date(2026, 8, 6), corte, datos.px_por_ticker, datos.bench, sect,
                   tickers=list(snap.ticker))
    iguales = [r for r in snap.itertuples() if pp.get(r.ticker, {}).get("perfil") == r.perfil]
    dif = max(abs((pp[r.ticker]["score_riesgo"] or 0) - float(r.score_riesgo))
              for r in snap.itertuples() if r.ticker in pp)
    log(f"VALIDACION PERFIL: {len(iguales)}/{len(snap)} perfiles iguales al snapshot "
        f"2026-08-06; max diferencia de score {dif:.6f}")
    return 0 if len(iguales) == len(snap) and dif < 1e-6 else 1


def validar_smc(datos):
    """Compara los elegibles reconstruidos con los candidatos que FT_SMC_v3 registro.
    FT excluye lo que ya tiene en cartera y lo bloqueado por balance: lo que sobra en
    la reconstruccion tiene que estar explicado por eso."""
    smc = datos.elegibles_smc()
    fechas_px = sorted(set(datos.px["fecha"]))
    close = {(r.ticker, r.fecha): r.close for r in datos.px.itertuples()}

    def rueda_de_datos(f, filas):
        """FT es asincronico: la corrida del dia f decide con el ultimo cierre que
        HABIA en ese momento (CLAUDE.md "FT asincronico"). La rueda real se deduce
        igualando el precio que registro FT con el close de cada rueda candidata."""
        i = bisect.bisect_right(fechas_px, f) - 1
        opciones = fechas_px[max(0, i - 3):i + 1][::-1]
        mejor, aciertos = None, -1
        for r in opciones:
            ok = sum(1 for tk, p in filas if p is not None
                     and abs(close.get((tk, r), -1) - float(p)) < 1e-6)
            if ok > aciertos:
                mejor, aciertos = r, ok
        return mejor if aciertos > 0 else None

    res_total = {"coinciden": 0, "faltan": 0, "sobran_explicados": 0,
                 "sobran_sin_explicar": 0, "corridas_sin_rueda": 0}
    for n, nombre in ((5, "FT_SMC_v3_N5"), (3, "FT_SMC_v3_N3")):
        cand = query_df("""SELECT c.fecha, c.ticker, c.precio_apertura FROM ft_candidatos_diarios c
                           JOIN ft_estrategias e ON e.id = c.estrategia_id
                           WHERE e.nombre = :n""", {"n": nombre})
        cand["fecha"] = cand["fecha"].map(_d)
        abiertas = query_df("""SELECT o.ticker, o.fecha_entrada, o.fecha_salida
                               FROM ft_operaciones o JOIN ft_estrategias e ON e.id = o.estrategia_id
                               WHERE e.nombre = :n""", {"n": nombre})
        corridas = query_df("""SELECT DISTINCT c.fecha FROM ft_candidatos_diarios c
                               JOIN ft_estrategias e ON e.id = c.estrategia_id
                               WHERE e.activa AND c.fecha >= '2026-10-01'""")
        for f in sorted(_d(x) for x in corridas["fecha"]):
            filas = [(r.ticker, r.precio_apertura) for r in cand.itertuples() if r.fecha == f]
            ft = {tk for tk, _ in filas}
            if not ft:
                continue                      # sin candidatos no se puede fechar la corrida
            rueda = rueda_de_datos(f, filas)
            if rueda is None:
                res_total["corridas_sin_rueda"] += 1
                log(f"  {nombre} corrida {f}: ningun precio coincide con una rueda")
                continue
            recon = {tk for tk, (s, por_n) in smc.get(rueda, {}).items() if n in por_n}
            en_cartera = {r.ticker for r in abiertas.itertuples()
                          if _d(r.fecha_entrada) < f and (r.fecha_salida is None
                          or pd.isna(r.fecha_salida) or _d(r.fecha_salida) >= f)}
            bal = datos.con_balance(rueda)
            faltan = ft - recon
            sobran = recon - ft
            expl = sobran & (en_cartera | bal)
            res_total["coinciden"] += len(ft & recon)
            res_total["faltan"] += len(faltan)
            res_total["sobran_explicados"] += len(expl)
            res_total["sobran_sin_explicar"] += len(sobran - expl)
            if faltan or (sobran - expl):
                log(f"  {nombre} corrida {f} (rueda {rueda}): FT {sorted(ft)} | recon "
                    f"{sorted(recon)} | faltan {sorted(faltan)} | sobran sin explicar "
                    f"{sorted(sobran - expl)}")
    log(f"VALIDACION SMC: {res_total}")
    ok = res_total["faltan"] == 0 and res_total["sobran_sin_explicar"] == 0
    return 0 if ok else 1


# ── Backtest ──────────────────────────────────────────────────────────────────

def perfiles_mensuales(datos, ruedas):
    """{(anio, mes): {ticker: cartera}} con datos hasta la rueda previa al mes.
    Cache en reportes/carteras_perfil/cache/ por rango de ruedas y ultima fecha de
    precios (si entra una rueda nueva o cambia el rango, se recalcula)."""
    cache_dir = os.path.join(ROOT, "reportes", "carteras_perfil", "cache")
    clave_cache = f"perfiles_{ruedas[0]}_{ruedas[-1]}_{max(datos.px['fecha'])}.json"
    ruta = os.path.join(cache_dir, clave_cache)
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as fh:
            crudo = json.load(fh)
        log(f"Perfil mensual desde cache ({clave_cache})")
        return {tuple(int(x) for x in k.split("-")): v for k, v in crudo.items()}
    out = _perfiles_mensuales(datos, ruedas)
    os.makedirs(cache_dir, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump({f"{a}-{m}": v for (a, m), v in out.items()}, fh)
    return out


def _perfiles_mensuales(datos, ruedas):
    out, t0 = {}, time.time()
    todas = sorted(set(datos.px["fecha"]))
    for f in ruedas:
        clave = (f.year, f.month)
        if clave in out:
            continue
        i = bisect.bisect_left(todas, f)
        corte = todas[i - 1]
        pp = perfil_en(f, corte, datos.px_por_ticker, datos.bench, datos.sectores)
        out[clave] = {tk: c["perfil"] for tk, c in pp.items()}
    log(f"Perfil recalculado para {len(out)} meses en {time.time() - t0:.0f}s")
    return out


def listas(datos, ruedas, semillas):
    """Por semilla y cartera, las ruedas con los candidatos de la SELECCION y del
    AZAR. Mismas semillas y mismo orden de sorteos que el backtest pre-registrado:
    el diagnostico de salidas usa exactamente las mismas listas."""
    tec = datos.elegibles_tecnica()
    smc = datos.elegibles_smc()
    log(f"Elegibles: tecnica en {len(tec)} ruedas, SMC en {len(smc)} ruedas")
    perfiles = perfiles_mensuales(datos, ruedas)
    close = {f: dict(zip(g.ticker, g.close)) for f, g in datos.px.groupby("fecha")}
    balance = {f: datos.con_balance(f) for f in ruedas}
    elegibles = {f: {"Tecnica": tec.get(f, []),
                     "SMC": [(tk, s, datos.sectores.get(tk))
                             for tk, (s, _) in smc.get(f, {}).items()]}
                 for f in ruedas}
    out = []
    for s in range(semillas):
        rng_sel = random.Random(s)
        rng_azar = random.Random(SEMILLA_AZAR + s)
        r_sel = {c: [] for c in cp.CARTERAS}
        r_azar = {c: [] for c in cp.CARTERAS}
        for f in ruedas:
            perf = perfiles[(f.year, f.month)]
            sel = cp.seleccionar(elegibles[f], rng_sel)
            cart = cp.por_cartera(sel, perf, rng_sel)
            cl = close.get(f, {})
            for c in cp.CARTERAS:
                r_sel[c].append(cp.Rueda(f, cl, balance[f], cart[c]))
                universo = [tk for tk, p in perf.items() if p == c and tk in cl]
                r_azar[c].append(cp.Rueda(f, cl, balance[f],
                                          cp.candidatos_azar(universo, len(cart[c]), rng_azar)))
        out.append((r_sel, r_azar))
    return out, perfiles, close


def variantes(cartera):
    """Las 5 variantes del diagnostico de salidas, con el SL/TP de la cartera."""
    sl, tp = cp.SALIDAS[cartera]
    return {
        "V0 actual":            cp.Salida(balance=True, sl=sl, tp=tp, usar_default=False),
        "V1 sin balance":       cp.Salida(balance=False, sl=sl, tp=tp, usar_default=False),
        "V2 sin SL ni TP":      cp.Salida(balance=True, sl=None, tp=None, usar_default=False),
        "V3 sin TP":            cp.Salida(balance=True, sl=sl, tp=None, usar_default=False),
        "V4 solo time stop 60": cp.Salida(balance=False, sl=None, tp=None, time_stop=60,
                                          usar_default=False),
    }


def salidas_diagnostico(ls, cartera, con_seleccion=False):
    """Diagnostico pre-registrado: misma entrada y mismas listas de candidatos
    (azar por defecto; con_seleccion=True usa las de la seleccion, informativo),
    variando solo la regla de salida."""
    i = 0 if con_seleccion else 1
    return {v: [cp.resumen(cp.simular(par[i][cartera], cartera, salida=sal)) for par in ls]
            for v, sal in variantes(cartera).items()}


def leer_variante(var, v0, tol_dd=5.0, tol_p5=2.0):
    """Regla de lectura del diagnostico (pre-registrada en la ficha)."""
    med = lambda rs, k: median_ok([r.get(k) for r in rs])
    anios = sorted({a for r in v0 for a in r["tramos"]})
    ganados = sum(1 for a in anios
                  if median_ok([r["tramos"].get(a) for r in var])
                  > median_ok([r["tramos"].get(a) for r in v0]))
    c1 = med(var, "retorno_total_pct") > med(v0, "retorno_total_pct")
    c2 = ganados >= cp.TRAMOS_MIN_GANADOS
    dd_var, dd_v0 = med(var, "max_dd_pct"), med(v0, "max_dd_pct")
    p5_var, p5_v0 = med(var, "p5_op_pct"), med(v0, "p5_op_pct")
    c3 = (dd_var - dd_v0 <= tol_dd) and (p5_v0 - p5_var <= tol_p5)
    pares = sum(1 for a, b in zip(var, v0) if a["retorno_total_pct"] >= b["retorno_total_pct"])
    return {"mejora": c1 and c2 and c3, "c1_total": c1, "c2_tramos": ganados, "c3_cola": c3,
            "p_pareado": pares / len(v0)}


def informe_salidas(res, ew, ruedas, semillas, cartera, modo="AL AZAR"):
    v0 = res["V0 actual"]
    med = lambda rs, k: median_ok([r.get(k) for r in rs])
    anios = sorted({a for r in v0 for a in r["tramos"]})
    ew_tot = 1.0
    for a in anios:
        ew_tot *= 1 + ew.get(a, 0) / 100
    sl, tp = cp.SALIDAS[cartera]
    L = [f"DIAGNOSTICO DE SALIDAS - CARTERA {cartera.upper()} (SL/TP {sl:.0%}/{tp:.0%})  "
         f"{ruedas[0]} -> {ruedas[-1]}  ({semillas} semillas, seleccion {modo}, "
         f"mismas listas en todas)",
         f"Referencia: mantener todo el perfil (EW) {(ew_tot - 1) * 100:+.1f}%", ""]
    filas = [("retorno total % (mediana)", "retorno_total_pct"),
             ("retorno / exposicion", "retorno_por_exposicion_pct"),
             ("exposicion media %", "exposicion_media_pct"),
             ("max drawdown %", "max_dd_pct"), ("sortino", "sortino"),
             ("peor operacion %", "peor_op_pct"), ("p5 operaciones %", "p5_op_pct"),
             ("operaciones", "operaciones"), ("aciertos", "win_rate"),
             ("profit factor", "profit_factor"), ("dias en posicion (mediana)", "dias_mediana"),
             ("dias en posicion (p90)", "dias_p90")]
    nombres = list(res)
    L.append(f"{'':<28}" + "".join(f"{n[:14]:>16}" for n in nombres))
    for etiqueta, k in filas:
        L.append(f"{etiqueta:<28}" + "".join(f"{_fmt(med(res[n], k), 2):>16}" for n in nombres))
    L.append(f"{'banda total p25 / p75':<28}" + "".join(
        f"{_fmt(cp._banda([r['retorno_total_pct'] for r in res[n]])['p25'], 0):>8}/"
        f"{_fmt(cp._banda([r['retorno_total_pct'] for r in res[n]])['p75'], 0):<7}" for n in nombres))
    L.append("")
    L.append("Por tramo (mediana): " + "  ".join(f"{n[:2]}" for n in nombres) + "  | EW")
    for a in anios:
        L.append(f"  {a}  " + "  ".join(f"{_fmt(median_ok([r['tramos'].get(a) for r in res[n]])):>7}"
                                       for n in nombres) + f"  | {_fmt(ew.get(a)):>6}")
    L.append("")
    L.append("Motivos de salida (mediana):")
    for n in nombres:
        mot = {}
        for r in res[n]:
            for k, v in r["motivos"].items():
                mot.setdefault(k, []).append(v)
        L.append(f"  {n:<22} " + ", ".join(f"{k} {median_ok(v)}" for k, v in sorted(mot.items())))
    L.append("")
    L.append("Lectura contra V0 (mejora = total > V0 Y >= 4/6 tramos Y cola dentro de tolerancia):")
    lect = {}
    for n in nombres[1:]:
        le = leer_variante(res[n], v0)
        lect[n] = le
        L.append(f"  {n:<22} {'MEJORA' if le['mejora'] else 'no mejora'}  | total>V0 {le['c1_total']} "
                 f"| tramos {le['c2_tramos']}/6 | cola ok {le['c3_cola']} "
                 f"| P(var >= V0, pareado) {le['p_pareado'] * 100:.0f}%")
    return "\n".join(L), lect


def correr(datos, ruedas, semillas):
    ls, perfiles, close = listas(datos, ruedas, semillas)
    resultados = {c: {"seleccion": [], "azar": []} for c in cp.CARTERAS}
    for r_sel, r_azar in ls:
        for c in cp.CARTERAS:
            resultados[c]["seleccion"].append(cp.resumen(cp.simular(r_sel[c], c)))
            resultados[c]["azar"].append(cp.resumen(cp.simular(r_azar[c], c)))
    return resultados, perfiles, close


def universo_ew(perfiles, close, ruedas):
    """Retorno por anio del universo equal-weight de cada perfil (rebalanceo diario,
    perfil vigente del mes). Informativo."""
    out = {c: {} for c in cp.CARTERAS}
    nivel = {c: 1.0 for c in cp.CARTERAS}
    base = {c: 1.0 for c in cp.CARTERAS}          # nivel al cierre del anio anterior
    for prev, f in zip(ruedas, ruedas[1:]):
        perf = perfiles[(f.year, f.month)]
        for c in cp.CARTERAS:
            if f.year != prev.year:
                base[c] = nivel[c]
            rets = [close[f][tk] / close[prev][tk] - 1 for tk, p in perf.items()
                    if p == c and tk in close.get(f, {}) and tk in close.get(prev, {})]
            if rets:
                nivel[c] *= 1 + sum(rets) / len(rets)
            out[c][f.year] = (nivel[c] / base[c] - 1) * 100
    return out


def _fmt(x, d=1):
    return "s/d" if x is None else f"{x:,.{d}f}"


def informe(resultados, lecturas, ew, ruedas, semillas, fin_txt):
    L = [f"BACKTEST CARTERAS POR PERFIL  {ruedas[0]} -> {ruedas[-1]}  ({len(ruedas)} ruedas, "
         f"{semillas} semillas)", f"Corte por balances: {fin_txt}", "Sin costos. Sin ML.", ""]
    anios = max(1e-9, (ruedas[-1] - ruedas[0]).days / 365.25)
    for c in cp.CARTERAS:
        le = lecturas[c]
        L.append(f"=== {c.upper()}  SL/TP {cp.SALIDAS[c][0]:.0%}/{cp.SALIDAS[c][1]:.0%}  -> "
                 f"{'PASA' if le['pasa'] else 'NO PASA'} (gana plata: {le['gana_plata']}, "
                 f"tramos ganados al azar: {le['tramos_ganados']}/{le['n_tramos']})")
        bs, ba = le["banda_seleccion"], le["banda_azar"]
        L.append(f"  retorno total  seleccion: min {_fmt(bs['min'])} p25 {_fmt(bs['p25'])} "
                 f"MED {_fmt(bs['mediana'])} p75 {_fmt(bs['p75'])} max {_fmt(bs['max'])}")
        L.append(f"                 azar     : min {_fmt(ba['min'])} p25 {_fmt(ba['p25'])} "
                 f"MED {_fmt(ba['mediana'])} p75 {_fmt(ba['p75'])} max {_fmt(ba['max'])}")
        L.append(f"  P(azar >= seleccion) sobre los pares de semillas: {_fmt(le['solape'] * 100, 0)}%")
        L.append("  tramo   seleccion   azar   universo EW")
        for a, t in le["por_tramo"].items():
            L.append(f"  {a}  {_fmt(t['seleccion']):>9}  {_fmt(t['azar']):>6}  "
                     f"{_fmt(ew[c].get(a)):>9}   {'gana' if t['gana'] else ''}")
        med = lambda k, lado: median_ok([r.get(k) for r in resultados[c][lado]])
        for k, nombre in (("exposicion_media_pct", "exposicion media %"),
                          ("retorno_por_exposicion_pct", "retorno / exposicion"),
                          ("max_dd_pct", "max drawdown %"), ("sortino", "sortino"),
                          ("operaciones", "operaciones"), ("win_rate", "aciertos"),
                          ("profit_factor", "profit factor"),
                          ("dias_mediana", "dias en posicion (mediana)"),
                          ("dias_p90", "dias en posicion (p90)"),
                          ("abiertas_al_final", "abiertas al final"),
                          ("ruedas_llena_pct", "ruedas con cartera llena %")):
            L.append(f"  {nombre:<28} sel {_fmt(med(k, 'seleccion'), 2):>9} | "
                     f"azar {_fmt(med(k, 'azar'), 2):>9}")
        ops = med("operaciones", "seleccion")
        L.append(f"  operaciones por anio (seleccion, mediana): {_fmt(ops / anios if ops else None)}")
        mot = {}
        for r in resultados[c]["seleccion"]:
            for k, v in r["motivos"].items():
                mot.setdefault(k, []).append(v)
        L.append("  motivos de salida (mediana): " + ", ".join(
            f"{k} {median_ok(v)}" for k, v in sorted(mot.items())))
        L.append("")
    return "\n".join(L)


def median_ok(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    from statistics import median
    return median(xs)


def main():
    ap = argparse.ArgumentParser(description="Backtest de carteras por perfil")
    ap.add_argument("--validar-perfil", action="store_true")
    ap.add_argument("--validar-smc", action="store_true")
    ap.add_argument("--salidas", metavar="CARTERAS",
                    help="diagnostico pre-registrado de salidas V0-V4, ej. Arriesgada,Especulativa")
    ap.add_argument("--con-seleccion", action="store_true",
                    help="ademas, las mismas variantes con la seleccion (informativo)")
    ap.add_argument("--semillas", type=int, default=N_SEMILLAS)
    ap.add_argument("--desde", default=DESDE.isoformat())
    ap.add_argument("--hasta", default=HASTA.isoformat())
    args = ap.parse_args()
    desde, hasta = date.fromisoformat(args.desde), date.fromisoformat(args.hasta)

    if args.validar_smc:
        return validar_smc(Datos(date.today()))
    if args.validar_perfil:
        return validar_perfil(Datos(date(2026, 8, 6)))

    datos = Datos(hasta)
    ruedas = datos.ruedas(desde, hasta)
    fin, fin_txt = fin_balances(datos.eh, pd.DatetimeIndex(pd.to_datetime(ruedas)))
    fin = _d(fin)
    ruedas = [f for f in ruedas if f <= fin]
    log(f"Ruedas: {len(ruedas)} ({ruedas[0]} -> {ruedas[-1]}). Balances: {fin_txt}")

    if args.salidas:
        carteras = [c.strip().capitalize() for c in args.salidas.split(",")]
        ls, perfiles, close = listas(datos, ruedas, args.semillas)
        ew = universo_ew(perfiles, close, ruedas)
        dest = os.path.join(ROOT, "reportes", "carteras_perfil",
                            f"{datetime.now():%Y%m%d_%H%M}_salidas")
        os.makedirs(dest, exist_ok=True)
        modos = [("azar", False)] + ([("seleccion", True)] if args.con_seleccion else [])
        todo, textos = {}, []
        for c in carteras:
            for etiqueta, con_sel in modos:
                res = salidas_diagnostico(ls, c, con_sel)
                txt, lect = informe_salidas(res, ew[c], ruedas, args.semillas, c,
                                            "CON SELECCION (informativo)" if con_sel else "AL AZAR")
                print("\n" + txt)
                textos.append(txt)
                todo[f"{c}_{etiqueta}"] = {"lecturas": lect, "corridas": res}
        with open(os.path.join(dest, "parametros.json"), "w", encoding="utf-8") as fh:
            json.dump({"desde": str(ruedas[0]), "hasta": str(ruedas[-1]),
                       "semillas": args.semillas, "carteras": carteras,
                       "variantes": {c: {k: v.__dict__ for k, v in variantes(c).items()}
                                     for c in carteras}},
                      fh, indent=1, default=str)
        with open(os.path.join(dest, "resultados.json"), "w", encoding="utf-8") as fh:
            json.dump({"universo_ew": ew, "diagnosticos": todo}, fh, indent=1, default=str)
        with open(os.path.join(dest, "resumen.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(textos))
        log(f"Detalle: {dest}")
        return 0

    resultados, perfiles, close = correr(datos, ruedas, args.semillas)
    lecturas = {c: cp.leer(resultados[c]["seleccion"], resultados[c]["azar"]) for c in cp.CARTERAS}
    ew = universo_ew(perfiles, close, ruedas)
    txt = informe(resultados, lecturas, ew, ruedas, args.semillas, fin_txt)
    print("\n" + txt)

    dest = os.path.join(ROOT, "reportes", "carteras_perfil", f"{datetime.now():%Y%m%d_%H%M}")
    os.makedirs(dest, exist_ok=True)
    params = {"desde": str(ruedas[0]), "hasta": str(ruedas[-1]), "semillas": args.semillas,
              "salidas": cp.SALIDAS, "pct_compra": cp.PCT_COMPRA, "max_posiciones": cp.MAX_POSICIONES,
              "top_por_familia": cp.TOP_POR_FAMILIA, "max_por_sector": cp.MAX_POR_SECTOR,
              "score_tecnico_min": SCORE_TECNICO_MIN, "smc_min": SMC_MIN,
              "ventanas_smc": VENTANAS_SMC, "corte_balances": fin_txt}
    with open(os.path.join(dest, "parametros.json"), "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=1, default=str)
    with open(os.path.join(dest, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump({"lecturas": lecturas, "universo_ew": ew, "corridas": resultados},
                  fh, indent=1, default=str)
    with open(os.path.join(dest, "resumen.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    log(f"Detalle: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
