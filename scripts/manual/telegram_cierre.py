"""
telegram_cierre.py
"Cierre por sector" por Telegram (29/9/2026, EN EVALUACION: no esta en la rutina).

QUE MANDA
    El cierre de la rueda leido por sector: mediana del dia y de 5 ruedas,
    suben/bajan, % sobre SMA50 y veredictos por sector; el sector mas fuerte y
    el mas debil; los tickers que cruzaron un muro de OI, un swing N5 confirmado
    o una SMA (nivel de la rueda ANTERIOR); y que vigilar en la proxima rueda
    (cerca de un nivel, balance en 5 ruedas, cambio de veredicto). Estado, no
    pronostico. Definiciones y criterios pre-registrados: docs/cierre_grupos.md.

RUEDAS
    Rueda D = la ultima fecha de precios_diarios con al menos la mitad del
    universo (o --rueda). La rueda ANTERIOR es la fecha previa con el mismo
    criterio, igual para todos los tickers (no un shift por ticker: con un hueco
    compararia contra una rueda vieja).

LOCAL-only: borra DATABASE_URL para que get_engine caiga a local.

Uso:
    python scripts/manual/telegram_cierre.py --dry-run            imprime, no envia
    python scripts/manual/telegram_cierre.py                      envia
    python scripts/manual/telegram_cierre.py --rueda 2026-09-25 --dry-run
    python scripts/manual/telegram_cierre.py --evaluar 20         criterios 2-4 sobre 20 ruedas

Codigo de salida:
    0 = enviado / impreso / evaluado
    2 = nada para mandar (sin datos para la rueda)
    1 = error de DB o de envio
"""

import os
import sys
import json
import argparse
from datetime import date, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass
os.environ.pop("DATABASE_URL", None)

from src.data.database import query_df                  # noqa: E402
from src.data.universo import get_universo_sectores     # noqa: E402
from src.utils import cierre_grupos as cg               # noqa: E402
from src.utils.foto_ticker import ruedas_hasta          # noqa: E402
from src.utils.trading_calendar import trading_days_between  # noqa: E402

MIN_COBERTURA = 0.5     # una fecha es "rueda del universo" con >= la mitad de los tickers


def _d(x):
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if hasattr(x, "date"):
        return x.date()
    return date.fromisoformat(str(x)[:10])


def _nn(v):
    """None para NaN/NaT de pandas."""
    if v is None:
        return None
    try:
        if v != v:
            return None
    except (TypeError, ValueError):
        pass
    return v


def ruedas_universo(tickers, hasta=None):
    """Fechas de precios_diarios con al menos MIN_COBERTURA del universo, ascendente."""
    df = query_df("""
        SELECT fecha, COUNT(*) AS n FROM precios_diarios
        WHERE ticker = ANY(:t) AND (CAST(:h AS DATE) IS NULL OR fecha <= :h)
        GROUP BY fecha ORDER BY fecha
    """, {"t": list(tickers), "h": hasta})
    minimo = MIN_COBERTURA * len(tickers)
    return [_d(r.fecha) for r in df.itertuples() if r.n >= minimo]


def cargar_insumos(tickers, fechas):
    """Todo lo que hace falta para las `fechas` pedidas, indexado por (ticker, fecha)."""
    p = {"t": list(tickers), "f": list(fechas)}
    precios = query_df("""
        SELECT p.ticker, p.fecha, p.close, i.sma50, i.sma200
        FROM precios_diarios p
        LEFT JOIN indicadores_tecnicos i ON i.ticker = p.ticker AND i.fecha = p.fecha
        WHERE p.ticker = ANY(:t) AND p.fecha = ANY(:f)
    """, p)
    muros = query_df("""
        SELECT ticker, fecha, soporte_strike, resistencia_strike,
               soporte_dist_pct, resistencia_dist_pct
        FROM opciones_pcr_plazo_diario
        WHERE ventana = 'corto' AND ticker = ANY(:t) AND fecha = ANY(:f)
    """, p)
    estr = query_df("""
        SELECT ticker, fecha, bos_bull_5, choch_bull_5, bos_bear_5, choch_bear_5,
               dist_sh_5_pct, dist_sl_5_pct
        FROM features_estructura
        WHERE ticker = ANY(:t) AND fecha = ANY(:f)
    """, p)
    vered = query_df("""
        SELECT ticker, fecha, veredicto FROM veredictos_universo_diario
        WHERE ticker = ANY(:t) AND fecha = ANY(:f)
    """, p)
    earn = query_df("""
        SELECT ticker, announcement_date AS f FROM earnings_historico
        WHERE ticker = ANY(:t) AND announcement_date IS NOT NULL
        UNION
        SELECT ticker, earnings_date AS f FROM earnings_calendar
        WHERE ticker = ANY(:t) AND earnings_date IS NOT NULL
    """, {"t": list(tickers)})

    def idx(df):
        return {(r["ticker"], _d(r["fecha"])): r for r in df.to_dict("records")}

    fechas_earn = {}
    for r in earn.itertuples():
        fechas_earn.setdefault(r.ticker, []).append(_d(r.f))
    for v in fechas_earn.values():
        v.sort()
    fechas_muros = {_d(f) for f in muros["fecha"]} if not muros.empty else set()
    return {"precios": idx(precios), "muros": idx(muros), "estr": idx(estr),
            "vered": idx(vered), "earn": fechas_earn, "fechas_muros": fechas_muros}


def _proximo_balance(fechas, rueda):
    for f in fechas or []:
        if f > rueda:
            return f
    return None


def armar_filas(sectores, ins, rueda, rueda_ant, rueda_5):
    """Filas del motor para la rueda: una por ticker con precio en `rueda`."""
    P, M, E, V = ins["precios"], ins["muros"], ins["estr"], ins["vered"]
    filas = []
    for tk, sector in sectores.items():
        hoy = P.get((tk, rueda))
        if hoy is None or _nn(hoy["close"]) is None:
            continue
        ant = P.get((tk, rueda_ant)) or {}
        r5 = P.get((tk, rueda_5)) or {}
        m_ant = M.get((tk, rueda_ant)) or {}
        m_hoy = M.get((tk, rueda)) or {}
        e = E.get((tk, rueda)) or {}
        fb = _proximo_balance(ins["earn"].get(tk), rueda)
        filas.append({
            "ticker": tk, "sector": sector,
            "close": _nn(hoy["close"]), "close_ant": _nn(ant.get("close")),
            "close_5": _nn(r5.get("close")),
            "sma50": _nn(hoy["sma50"]), "sma50_ant": _nn(ant.get("sma50")),
            "sma200": _nn(hoy["sma200"]), "sma200_ant": _nn(ant.get("sma200")),
            "sop_ant": _nn(m_ant.get("soporte_strike")),
            "res_ant": _nn(m_ant.get("resistencia_strike")),
            "sop_dist": _nn(m_hoy.get("soporte_dist_pct")),
            "res_dist": _nn(m_hoy.get("resistencia_dist_pct")),
            "swing_arriba": int(bool(_nn(e.get("bos_bull_5")) or _nn(e.get("choch_bull_5")))) if e else None,
            "swing_abajo": int(bool(_nn(e.get("bos_bear_5")) or _nn(e.get("choch_bear_5")))) if e else None,
            "dist_sh": _nn(e.get("dist_sh_5_pct")), "dist_sl": _nn(e.get("dist_sl_5_pct")),
            "veredicto": _nn((V.get((tk, rueda)) or {}).get("veredicto")),
            "veredicto_ant": _nn((V.get((tk, rueda_ant)) or {}).get("veredicto")),
            "earnings_ruedas": ruedas_hasta(rueda, fb) if fb else None,
            "earnings_fecha": fb,
        })
    return filas


def cargar_resultados(n_ruedas=1, hasta=None):
    """
    Construye el resultado del motor para las ultimas `n_ruedas` (terminando en
    `hasta` o en la ultima rueda). Devuelve [(resultado, filas)] ascendente.
    Lo reusa el auditor.
    """
    sectores = get_universo_sectores()
    todas = ruedas_universo(sectores.keys(), hasta)
    k = cg.RUEDAS_VAR_N
    if len(todas) < k + 2:
        return []
    objetivo = todas[-n_ruedas:] if n_ruedas <= len(todas) - k - 1 else todas[k + 1:]
    pos = {f: i for i, f in enumerate(todas)}
    necesarias = set()
    for f in objetivo:
        i = pos[f]
        necesarias.update({todas[i], todas[i - 1], todas[i - k]})
    ins = cargar_insumos(sectores.keys(), sorted(necesarias))

    out = []
    for f in objetivo:
        i = pos[f]
        filas = armar_filas(sectores, ins, f, todas[i - 1], todas[i - k])
        # dias habiles NYSE entre las dos ruedas que no llegaron a ser rueda
        # del universo (hueco en precios_diarios, ej. 2026-09-22 con 60 de 200)
        salteadas = [d for d in trading_days_between(todas[i - 1], f)
                     if todas[i - 1] < d < f]
        res = cg.construir(f, todas[i - 1], filas, n_universo=len(sectores),
                           oi_disponible=todas[i - 1] in ins["fechas_muros"],
                           ruedas_salteadas=salteadas)
        out.append((res, filas))
    return out


def _evaluar(n, hasta):
    pares = cargar_resultados(n, hasta)
    if not pares:
        print("Sin ruedas suficientes.")
        return 2
    resultados = [r for r, _ in pares]
    ev = cg.evaluar(resultados)
    ver = cg.veredicto_criterios(ev)

    print(f"Evaluacion sobre {ev['n_ruedas']} ruedas: "
          f"{resultados[0]['rueda']} -> {resultados[-1]['rueda']}\n")
    print("rueda       con_dato  OI  SWING  SMA  oi_ant  fuerte / debil                 msgs  largo")
    for r, (nm, largo) in zip(resultados, ev["mensajes"]):
        c = {t: len(cg.tickers_con_cruce(r["cruces"], t)) for t in cg.TIPOS_CRUCE}
        oi = str(c["OI"]) if r["oi_disponible"] else "-"
        fd = f"{cg.sector_corto(r['grupos']['fuerte'])} / {cg.sector_corto(r['grupos']['debil'])}"
        print(f"{r['rueda']}  {r['n_con_dato']:>8}  {oi:>3}  {c['SWING']:>5}  {c['SMA']:>3}  "
              f"{'si' if r['oi_disponible'] else 'NO':>6}  {fd:<30} {nm:>4}  {largo:>5}")

    print("\nCriterio 2 - mediana diaria de tickers con cruce (pasa entre 2 y 30):")
    for t in cg.TIPOS_CRUCE:
        print(f"  {t:<6} mediana {ev['mediana_cruces'][t]}  "
              f"(n ruedas {len(ev['cruces_por_dia'][t])})  -> "
              f"{'PASA' if ver['c2_por_tipo'][t] else 'NO PASA'}")
    print("Criterio 3 - sector mas repetido (pasa si <= 80%):")
    for k in ("fuerte", "debil"):
        e = ev[k]
        print(f"  {k:<7} {e['mas_repetido']} en {e['share']:.0%} de las ruedas")
    print(f"  -> {'PASA' if ver['c3'] else 'NO PASA'}")
    print(f"Criterio 4 - <= 2 mensajes de <= {cg.MAX_LEN_MENSAJE} chars: "
          f"{'PASA' if ver['c4'] else 'NO PASA'} "
          f"(max {max(n for n, _ in ev['mensajes'])} mensajes, "
          f"max {max(l for _, l in ev['mensajes'])} chars)")

    dest = os.path.join(ROOT, "reportes", "cierre_grupos",
                        f"{date.today():%Y%m%d}_eval_{ev['n_ruedas']}")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "evaluacion.json"), "w", encoding="utf-8") as fh:
        json.dump({"evaluacion": ev, "criterios": ver,
                   "parametros": {"MIN_N_RANKING": cg.MIN_N_RANKING,
                                  "UMBRAL_CERCA_PCT": cg.UMBRAL_CERCA_PCT,
                                  "RUEDAS_EARNINGS": cg.RUEDAS_EARNINGS,
                                  "MIN_COBERTURA": MIN_COBERTURA}},
                  fh, default=str, indent=1)
    with open(os.path.join(dest, "mensajes.txt"), "w", encoding="utf-8") as fh:
        for r in resultados:
            fh.write(f"===== {r['rueda']} =====\n")
            fh.write("\n\n".join(cg.mensajes(r)) + "\n\n")
    print(f"\nDetalle: {dest}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Cierre por sector por Telegram")
    ap.add_argument("--dry-run", action="store_true", help="imprime y no envia")
    ap.add_argument("--rueda", help="YYYY-MM-DD (default: ultima rueda del universo)")
    ap.add_argument("--evaluar", type=int, metavar="N",
                    help="evalua los criterios 2-4 sobre N ruedas (no envia)")
    args = ap.parse_args()
    hasta = date.fromisoformat(args.rueda) if args.rueda else None

    try:
        if args.evaluar:
            return _evaluar(args.evaluar, hasta)
        pares = cargar_resultados(1, hasta)
    except Exception as e:
        print(f"[ERROR] DB: {e}")
        return 1
    if not pares or not pares[-1][1]:
        print("Sin datos para la rueda pedida: nada para mandar.")
        return 2
    res = pares[-1][0]
    if hasta and res["rueda"] != hasta:
        print(f"OJO: {hasta} no es rueda del universo; se usa {res['rueda']}.")

    mensajes = cg.mensajes(res)
    print(f"Rueda: {res['rueda']} (anterior {res['rueda_ant']}) | "
          f"tickers: {res['n_con_dato']} | mensajes: {len(mensajes)}")
    for i, m in enumerate(mensajes, 1):
        print(f"\n----- mensaje {i}/{len(mensajes)} ({len(m)} chars) -----")
        print(m)
    if args.dry_run:
        return 0

    from src.pipeline.telegram_notifier import _send
    ok = all([_send(m) for m in mensajes])
    print("Telegram: enviado" if ok else "Telegram: FALLO el envio")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
