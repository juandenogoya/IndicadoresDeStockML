"""
ft_grilla_entradas.py
PASO 1 del analisis de ENTRADAS: las 166 reglas booleanas, con la salida y el reparto
de TECH_SECTOR_v1 fijos (docs/forward_testing/ANALISIS_ENTRADAS.md sec. 9).
Solo LEE. Escribe resultados en reportes/analisis_entradas/AAAAMMDD_paso1/.

MOTOR
    Reusa `ft_backtesting_runner.run_tech_sector`, el mismo que produjo el backtest de
    referencia, parametrizado con `filtro_entrada`. No se reimplementa la cartera: asi
    la fidelidad es por construccion y no algo que haya que demostrar. El loader se
    carga UNA vez y se reusa en las 186 corridas (166 reglas + 20 sorteos), que es lo
    que hace viable correrlas todas.

REANUDABLE
    Cada corrida se escribe apenas termina en `resultados.jsonl` y su serie de retornos
    diarios en `series/<clave>.npy`. Al arrancar, saltea lo ya hecho. Se puede cortar y
    retomar sin perder nada. `--status` dice cuanto falta.

CONDICION DE ARRANQUE (seccion `fidelidad`)
    La mascara de la v1 tiene que reproducir la corrida base -- misma cantidad de
    operaciones, mismo retorno, mismas metricas por anio. Si no reproduce, no se lee
    ninguna comparacion.

Uso:
    python scripts/forward_testing/ft_grilla_entradas.py --seccion fidelidad
    python scripts/forward_testing/ft_grilla_entradas.py --seccion grilla
    python scripts/forward_testing/ft_grilla_entradas.py --seccion lectura
    python scripts/forward_testing/ft_grilla_entradas.py --status
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "backtesting_historico"))

os.environ.pop("DATABASE_URL", None)

import numpy as np  # noqa: E402

import bt_scoring as sc  # noqa: E402
from bt_data_loader import BtDataLoader  # noqa: E402
from ft_backtesting_runner import (metricas_vs_universo,  # noqa: E402
                                   run_tech_sector)
from src.data.database import get_engine  # noqa: E402
from src.utils import ft_entradas as fe  # noqa: E402

DESDE = "2021-09-01"
HASTA = "2026-09-18"
SEMILLAS_SORTEO = tuple(range(20))          # control de aleatorizacion (sec. 9.3)
CORTE_SELECCION = date(2024, 12, 31)        # seleccion <= corte < confirmacion
DIR_BASE = os.path.join(ROOT, "reportes", "analisis_entradas")


# --- persistencia reanudable --------------------------------------------------

class Almacen:
    """resultados.jsonl + series/<clave>.npy. Idempotente por clave."""

    def __init__(self, dir_corrida):
        self.dir = dir_corrida
        self.series = os.path.join(dir_corrida, "series")
        os.makedirs(self.series, exist_ok=True)
        self.ruta = os.path.join(dir_corrida, "resultados.jsonl")
        self.hechas = {}
        if os.path.exists(self.ruta):
            with open(self.ruta, encoding="utf-8") as f:
                for linea in f:
                    linea = linea.strip()
                    if not linea:
                        continue
                    d = json.loads(linea)
                    self.hechas[d["clave"]] = d

    def guardar(self, res: dict, serie: np.ndarray):
        np.save(os.path.join(self.series, f"{res['clave']}.npy"), serie)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
        self.hechas[res["clave"]] = res

    def serie(self, clave) -> np.ndarray:
        return np.load(os.path.join(self.series, f"{clave}.npy"))


# --- una corrida --------------------------------------------------------------

def _filtro_mascara(mascara: int):
    """callable(score, detalle) -> bool con la mascara. SMA200 sigue siendo filtro."""
    def filtro(score, detalle):
        if not detalle["filtro_sma200"]:
            return False
        idx = fe.indice_estado(detalle["cond_sma50"], detalle["cond_sma21"],
                               detalle["cond_macd"], detalle["cond_rsi"])
        return bool((mascara >> idx) & 1)
    return filtro


def _orden_sorteo(semilla: int):
    """Baraja los candidatos antes del sort estable -> el desempate pasa a ser azar."""
    rng = np.random.default_rng(semilla)

    def orden(candidatos):
        idx = rng.permutation(len(candidatos))
        return [candidatos[i] for i in idx]
    return orden


def correr_una(loader, clave, etiqueta, filtro_entrada=None, orden=None) -> tuple:
    """Una corrida completa. Devuelve (dict de metricas, serie de retornos diarios)."""
    t0 = time.time()
    pm = run_tech_sector(bt_id=0, loader=loader, logica="tecnico_sectorial",
                         dry_run=True, verbose=False, filtro_entrada=filtro_entrada,
                         orden_candidatos=orden, registrar_candidatos=False)
    resumen = pm.resumen_metricas()
    vs = metricas_vs_universo(pm, loader)
    ops = pm.operaciones_cerradas

    fechas = list(vs["ret_diario_estrategia"].index)
    sel = np.array([f <= CORTE_SELECCION for f in fechas])
    ret_est = vs["ret_diario_estrategia"].to_numpy(dtype=float)

    gana = sum(1 for a in vs["por_anio"]
               if a["estr_por_expo_pct"] > a["uni_por_expo_pct"])
    n_anios = len(vs["por_anio"])
    candidatos_rueda = (len(ops) / len(fechas)) if fechas else 0.0

    res = {
        "clave": clave, "etiqueta": etiqueta,
        "retorno_total_pct": resumen["retorno_total_pct"],
        "drawdown_max_pct": resumen["drawdown_max_pct"],
        "operaciones": resumen["total_operaciones"],
        "win_rate_pct": resumen["win_rate_pct"],
        "profit_factor": resumen["profit_factor"],
        "retorno_medio_op_pct": (sum(o["pnl_pct"] or 0 for o in ops) / len(ops)
                                 if ops else 0.0),
        "dias_promedio_pos": resumen["dias_promedio_pos"],
        "exposicion_media_pct": vs["exposicion_media_pct"],
        "universo_total_pct": vs["universo_total_pct"],
        "anios_gana_universo_ajustado": gana, "anios": n_anios,
        "ops_por_anio": len(ops) / (len(fechas) / 252.0) if fechas else 0.0,
        "ops_por_rueda": candidatos_rueda,
        "ret_seleccion_pct": float((np.prod(1 + ret_est[sel]) - 1) * 100),
        "ret_confirmacion_pct": float((np.prod(1 + ret_est[~sel]) - 1) * 100),
        "por_anio": vs["por_anio"],
        "segundos": round(time.time() - t0, 1),
    }
    return res, ret_est


# --- secciones ----------------------------------------------------------------

def tareas(seccion: str):
    """(clave, etiqueta, filtro, orden) de cada corrida pendiente por seccion."""
    v1 = fe.mascara_v1()
    out = []
    if seccion in ("fidelidad", "grilla", "todas"):
        out.append((f"m{v1}", f"v1 | {fe.texto_regla(v1)}", _filtro_mascara(v1), None))
    if seccion in ("grilla", "todas"):
        for m in fe.reglas_monotonas():
            if m == v1:
                continue
            out.append((f"m{m}", fe.texto_regla(m), _filtro_mascara(m), None))
    if seccion in ("sorteo", "todas"):
        for s in SEMILLAS_SORTEO:
            out.append((f"s{s}", f"v1 con desempate al azar (semilla {s})",
                        _filtro_mascara(v1), _orden_sorteo(s)))
    return out


def seccion_fidelidad(loader, almacen, res_v1) -> bool:
    """La mascara de la v1 tiene que reproducir la corrida base del runner."""
    print()
    print("=" * 78)
    print("FIDELIDAD -- la mascara de la v1 contra el filtro original del runner")
    print("=" * 78)
    base, _ = correr_una(loader, "base_runner", "filtro original (score >= 4,0)")

    filas = [("operaciones", base["operaciones"], res_v1["operaciones"]),
             ("retorno total %", base["retorno_total_pct"], res_v1["retorno_total_pct"]),
             ("drawdown max %", base["drawdown_max_pct"], res_v1["drawdown_max_pct"]),
             ("exposicion %", base["exposicion_media_pct"],
              res_v1["exposicion_media_pct"]),
             ("retorno medio/op %", base["retorno_medio_op_pct"],
              res_v1["retorno_medio_op_pct"])]
    print(f"  {'metrica':<22} {'runner':>14} {'mascara v1':>14} {'dif':>10}")
    ok = True
    for nombre, a, b in filas:
        dif = float(b) - float(a)
        ok = ok and abs(dif) < 1e-6
        print(f"  {nombre:<22} {float(a):>14.4f} {float(b):>14.4f} {dif:>10.6f}")
    for a, b in zip(base["por_anio"], res_v1["por_anio"]):
        if abs(a["estr_por_expo_pct"] - b["estr_por_expo_pct"]) > 1e-6:
            ok = False
            print(f"  anio {a['anio']}: {a['estr_por_expo_pct']:.4f} vs "
                  f"{b['estr_por_expo_pct']:.4f}  DIFIERE")
    print()
    veredicto = ("OK, la mascara reproduce el filtro original" if ok
                 else "FALLA -- no leer ninguna comparacion")
    print(f"  VEREDICTO: {veredicto}")
    return ok


def _ic95_pareado(dif: np.ndarray) -> tuple:
    """Media e IC95 de una diferencia diaria pareada (t de Student). En puntos %."""
    from scipy.stats import t as _t

    n = len(dif)
    if n < 3:
        return 0.0, 0.0, 0.0
    media = float(np.mean(dif))
    se = float(np.std(dif, ddof=1) / np.sqrt(n))
    h = float(_t.ppf(0.975, n - 1) * se)
    return media * 100, (media - h) * 100, (media + h) * 100


def seccion_lectura(almacen, loader=None):
    """Aplica la regla de lectura PRE-REGISTRADA (sec. 9.5). Solo lee lo ya corrido."""
    from scipy.stats import spearmanr

    v1_clave = f"m{fe.mascara_v1()}"
    reglas = {k: v for k, v in almacen.hechas.items() if k.startswith("m")}
    sorteos = {k: v for k, v in almacen.hechas.items() if k.startswith("s")}
    if v1_clave not in reglas:
        print("Falta la corrida de la v1; no se puede leer nada.")
        return {}

    v1 = reglas[v1_clave]
    serie_v1 = almacen.serie(v1_clave)
    n = len(serie_v1)
    fechas = list(loader.trading_days)[-n:] if loader is not None else None
    sel = (np.array([f <= CORTE_SELECCION for f in fechas]) if fechas
           else np.arange(n) < int(n * 0.66))

    print()
    print("=" * 78)
    print("LECTURA -- regla PRE-REGISTRADA (ANALISIS_ENTRADAS.md sec. 9.5)")
    print("=" * 78)
    print(f"Reglas corridas: {len(reglas)} de {len(fe.reglas_monotonas())}   "
          f"Sorteos: {len(sorteos)} de {len(SEMILLAS_SORTEO)}")
    print(f"Seleccion: hasta {CORTE_SELECCION} ({int(sel.sum())} ruedas)   "
          f"Confirmacion: {int((~sel).sum())} ruedas")

    # --- banda del sorteo: el piso de ruido -----------------------------------
    banda = None
    if sorteos:
        difs = [s["retorno_total_pct"] - v1["retorno_total_pct"] for s in sorteos.values()]
        banda = (min(difs), max(difs))
        print()
        print("BANDA DEL SORTEO (v1 con desempate al azar, %d semillas)" % len(sorteos))
        print(f"  retorno total: min {min(s['retorno_total_pct'] for s in sorteos.values()):+.2f}%  "
              f"max {max(s['retorno_total_pct'] for s in sorteos.values()):+.2f}%  "
              f"v1 alfabetico {v1['retorno_total_pct']:+.2f}%")
        print(f"  diferencia contra la v1: [{banda[0]:+.2f} ; {banda[1]:+.2f}] pp")
        print("  -> una regla que le gane a la v1 por menos que esto no gano por la regla")

    # --- por regla -------------------------------------------------------------
    filas = []
    for clave, r in reglas.items():
        serie = almacen.serie(clave)
        if len(serie) != n:
            continue
        dif = serie - serie_v1
        m_sel, lo_sel, hi_sel = _ic95_pareado(dif[sel])
        m_con, lo_con, hi_con = _ic95_pareado(dif[~sel])
        expo = r["exposicion_media_pct"]
        filas.append({
            "clave": clave, "etiqueta": r["etiqueta"],
            "retorno_total_pct": r["retorno_total_pct"],
            "ret_por_expo": r["retorno_total_pct"] / expo * 100 if expo else float("nan"),
            "exposicion_media_pct": expo, "operaciones": r["operaciones"],
            "ops_por_anio": r["ops_por_anio"], "drawdown_max_pct": r["drawdown_max_pct"],
            "dif_sel_pp": m_sel, "ic_sel_lo": lo_sel, "ic_sel_hi": hi_sel,
            "dif_con_pp": m_con, "ic_con_lo": lo_con, "ic_con_hi": hi_con,
            "ret_seleccion_pct": r["ret_seleccion_pct"],
            "ret_confirmacion_pct": r["ret_confirmacion_pct"],
            "anios_gana_universo": r["anios_gana_universo_ajustado"],
            "anios": r["anios"],
        })

    v1_expo = v1["exposicion_media_pct"]
    v1_rpe = v1["retorno_total_pct"] / v1_expo * 100
    for f in filas:
        excluye_cero = (f["ic_sel_lo"] > 0) or (f["ic_sel_hi"] < 0)
        c1 = f["ret_por_expo"] > v1_rpe and excluye_cero and f["dif_sel_pp"] > 0
        c2 = f["dif_con_pp"] > 0
        c3 = (banda is None) or (f["retorno_total_pct"] - v1["retorno_total_pct"]
                                 > banda[1])
        c4 = f["anios_gana_universo"] >= 4
        f.update({"c1_seleccion": c1, "c2_confirmacion": c2, "c3_supera_sorteo": c3,
                  "c4_universo": c4, "pasa": bool(c1 and c2 and c3 and c4)})

    filas.sort(key=lambda x: -x["dif_sel_pp"])
    pasan = [f for f in filas if f["pasa"]]

    print()
    print(f"{'regla':<46} {'ret%':>8} {'r/expo':>8} {'expo':>5} {'ops/a':>6} "
          f"{'dif sel':>8} {'dif conf':>9} {'anios':>6} {'pasa':>5}")
    for f in filas[:20]:
        print(f"{f['etiqueta'][:44]:<46} {f['retorno_total_pct']:>+8.2f} "
              f"{f['ret_por_expo']:>+8.1f} {f['exposicion_media_pct']:>4.0f}% "
              f"{f['ops_por_anio']:>6.0f} {f['dif_sel_pp']:>+8.4f} "
              f"{f['dif_con_pp']:>+9.4f} {f['anios_gana_universo']:>3}/{f['anios']} "
              f"{'SI' if f['pasa'] else '.':>5}")
    print()
    print(f"La v1 (control): ret {v1['retorno_total_pct']:+.2f}%  r/expo {v1_rpe:+.1f}  "
          f"expo {v1_expo:.0f}%  ops/anio {v1['ops_por_anio']:.0f}  "
          f"anios {v1['anios_gana_universo_ajustado']}/{v1['anios']}")
    print(f"REGLAS QUE PASAN LAS CUATRO CONDICIONES: {len(pasan)} de {len(filas)}")
    for f in pasan:
        print(f"  {f['etiqueta']}")

    # --- el orden entre reglas, ?sobrevive al cambio de periodo? ---------------
    rho = None
    if len(filas) > 10:
        a = [f["ret_seleccion_pct"] for f in filas]
        b = [f["ret_confirmacion_pct"] for f in filas]
        rho = float(spearmanr(a, b).statistic)
        print()
        print(f"Correlacion de orden seleccion vs confirmacion (Spearman): {rho:+.3f}")
        print("  -> si es baja, ninguna conclusion sobre una regla individual es fiable")

    import csv
    ruta = os.path.join(almacen.dir, "lectura.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\nDetalle por regla: {ruta}")
    return {"reglas": len(filas), "pasan": len(pasan), "banda_sorteo": banda,
            "spearman_sel_conf": rho}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seccion", default="todas",
                    choices=["todas", "fidelidad", "grilla", "sorteo",
                             "lectura"])
    ap.add_argument("--etiqueta", default="paso1")
    ap.add_argument("--limite", type=int, default=None,
                    help="correr solo las primeras N pendientes (para medir el ritmo)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    dir_corrida = os.path.join(DIR_BASE,
                               f"{date.today().strftime('%Y%m%d')}_{args.etiqueta}")
    os.makedirs(dir_corrida, exist_ok=True)
    almacen = Almacen(dir_corrida)
    pendientes = [t for t in tareas(args.seccion) if t[0] not in almacen.hechas]

    if args.seccion == "lectura":
        loader = BtDataLoader(get_engine(), date.fromisoformat(DESDE),
                              date.fromisoformat(HASTA), "tecnico_sectorial")
        loader.cargar()
        seccion_lectura(almacen, loader)
        return

    if args.status:
        total = len(tareas("todas"))
        print(f"Corrida  : {dir_corrida}")
        print(f"Hechas   : {len(almacen.hechas)} de {total}")
        print(f"Pendientes de la seccion '{args.seccion}': {len(pendientes)}")
        return

    print(f"Cargando datos {DESDE} -> {HASTA} (una sola vez para todas las corridas)...")
    t0 = time.time()
    loader = BtDataLoader(get_engine(), date.fromisoformat(DESDE),
                          date.fromisoformat(HASTA), "tecnico_sectorial")
    loader.cargar()
    print(f"  listo en {time.time() - t0:.0f} s | {len(loader.trading_days)} ruedas")

    with open(os.path.join(dir_corrida, "parametros.json"), "w", encoding="utf-8") as f:
        json.dump({"desde": DESDE, "hasta": HASTA, "seccion": args.seccion,
                   "corte_seleccion": str(CORTE_SELECCION),
                   "semillas_sorteo": list(SEMILLAS_SORTEO),
                   "reglas": len(fe.reglas_monotonas()),
                   "mascara_v1": fe.mascara_v1(),
                   "score_entrada_runner": sc.SCORE_ENTRADA_TECH,
                   "score_salida_runner": sc.SCORE_SALIDA_TECH,
                   "git_commit": _git_commit(),
                   "corrida": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2, ensure_ascii=False)

    if args.limite:
        pendientes = pendientes[:args.limite]

    print(f"Corridas pendientes: {len(pendientes)}")
    for i, (clave, etiqueta, filtro, orden) in enumerate(pendientes, 1):
        res, serie = correr_una(loader, clave, etiqueta, filtro, orden)
        almacen.guardar(res, serie)
        print(f"  [{i}/{len(pendientes)}] {clave:<10} {etiqueta[:48]:<48} "
              f"ret {res['retorno_total_pct']:>+7.2f}%  ops {res['operaciones']:>5}  "
              f"expo {res['exposicion_media_pct']:>4.0f}%  {res['segundos']:>5.1f}s",
              flush=True)

    if args.seccion in ("fidelidad", "todas"):
        clave_v1 = f"m{fe.mascara_v1()}"
        if clave_v1 in almacen.hechas:
            seccion_fidelidad(loader, almacen, almacen.hechas[clave_v1])

    print()
    print(f"Resultados en {dir_corrida}")


if __name__ == "__main__":
    main()
