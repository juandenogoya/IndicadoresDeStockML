"""
ft_bot_tecnico.py
Forward-testing — Replica del Bot 2 Tecnico SMA/MACD/RSI (Alpaca).

Logica (3 capas de scoring, max 5.5 pts):
    Capa 1 (obligatoria): precio > SMA200
    Capa 2 (tendencia)  : precio > SMA50 (2.0pts), precio > SMA21 (1.0pt)
    Capa 3 (momentum)   : MACD > Signal + hist>0 (1.5pts), RSI 45-68 (1.0pt)

    ENTRADA : score >= SCORE_ENTRADA (4.0)
    SALIDA  : score <= SCORE_SALIDA  (3.5) — exit primario
              SL = entrada - 2 * ATR14     — exit emergencia
              TP = entrada + 4 * ATR14     — exit emergencia
              Earnings manana              — prioridad absoluta

Diferencias vs Bot Alpaca:
    - Precio de ejecucion = cierre del dia (precios_diarios), sin llamadas a Alpaca
    - Escribe en ft_operaciones / ft_estrategias / ft_metricas_diarias
    - No envia ordenes al broker

Uso:
    python scripts/forward_testing/ft_bot_tecnico.py
    python scripts/forward_testing/ft_bot_tecnico.py --dry-run
"""

import sys
import os
import argparse
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from src.indicators.earnings_filter import tickers_a_cerrar_hoy, tickers_a_bloquear_entrada
from scripts.forward_testing.ft_scoring import calcular_score_tecnico, obtener_indicadores_hoy

from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_posiciones_abiertas,
    obtener_precios_cierre_todos, abrir_operacion, cerrar_operacion,
    registrar_metricas_diarias, calcular_qty_ft, calcular_cash_desplegable,
    registrar_candidatos_diarios,
)

# ── Parametros de la estrategia ───────────────────────────────────────────────
NOMBRE_ESTRATEGIA = "FT_TECH_v1"
SCORE_ENTRADA     = 4.0
SCORE_SALIDA      = 3.5
SCORE_MAXIMO      = 5.5
ATR_MULT_SL       = 2.0
ATR_MULT_TP       = 4.0
MAX_POSICIONES    = 5
MAX_DEPLOY_PCT    = 0.80     # nunca desplegar mas del 80% del capital
RIESGO_POR_TRADE  = 0.15


# ── Evaluacion de cierres ─────────────────────────────────────────────────────

def evaluar_cierres(
    posiciones:   list[dict],
    precios:      dict[str, float],
    earnings_map: dict[str, date],
    indicadores_map: dict[str, dict],
) -> list[dict]:
    """
    Prioridades de salida:
        P1. Earnings manana  (prioridad absoluta)
        P2. Score degradado  (<= SCORE_SALIDA)  -- exit primario
        P3. Stop loss ATR    -- exit emergencia
        P4. Take profit ATR  -- exit emergencia
    """
    if not posiciones:
        return []

    a_cerrar = []

    for pos in posiciones:
        ticker         = pos["ticker"]
        precio_entrada = float(pos["precio_entrada"])
        precio_actual  = precios.get(ticker)
        sl             = float(pos["stop_loss"])  if pos.get("stop_loss")  else None
        tp             = float(pos["take_profit"]) if pos.get("take_profit") else None
        ind            = indicadores_map.get(ticker)

        if not precio_actual:
            log(f"  [WARN] Sin precio para {ticker}, se omite.")
            continue

        motivo = None

        # P1: Earnings
        if ticker in earnings_map:
            motivo = "EARNINGS_MANANA"

        # P2: Score degradado (exit primario)
        elif ind:
            score_actual, _ = calcular_score_tecnico(ind)
            if score_actual <= SCORE_SALIDA:
                motivo = f"SCORE_DEGRADADO_{score_actual:.1f}"

        # P3: Stop loss ATR (emergencia)
        if not motivo and sl and precio_actual <= sl:
            motivo = "STOP_LOSS_ATR"

        # P4: Take profit ATR (emergencia)
        if not motivo and tp and precio_actual >= tp:
            motivo = "TAKE_PROFIT_ATR"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar


# ── Evaluacion de entradas ────────────────────────────────────────────────────

def evaluar_entradas(
    posiciones:          list[dict],
    estrategia:          dict,
    indicadores:         list[dict],
    precios:             dict[str, float],
    tickers_bloqueados:  set,
    tickers_cerrados:    set,
) -> list[dict]:
    """
    Escanea indicadores tecnicos y retorna candidatos para abrir posicion.
    Los tickers cerrados en la misma corrida quedan excluidos (misma data,
    decision opuesta imposible por diseno de scoring).
    """
    tickers_abiertos = {p["ticker"] for p in posiciones}
    excluidos        = tickers_bloqueados | tickers_abiertos | tickers_cerrados
    capital_actual   = float(estrategia["capital_actual"])
    slots_libres     = MAX_POSICIONES - len(posiciones)

    if slots_libres <= 0:
        log("  Maximo de posiciones alcanzado, sin entradas.")
        return []

    # Cash efectivo respetando el techo de despliegue del 80%
    cash = calcular_cash_desplegable(estrategia, MAX_DEPLOY_PCT)
    if cash <= 0:
        pct_actual = float(estrategia["capital_inmovilizado"]) / capital_actual * 100
        log(f"  Techo de despliegue alcanzado ({pct_actual:.1f}% desplegado, "
            f"max={MAX_DEPLOY_PCT*100:.0f}%). Sin entradas.")
        return []

    candidatos = []

    for ind in indicadores:
        ticker = ind["ticker"]
        if ticker in excluidos:
            continue

        score, detalle = calcular_score_tecnico(ind)
        if score < SCORE_ENTRADA:
            continue

        candidatos.append({"ticker": ticker, "score": score,
                           "detalle": detalle, "ind": ind})

    if not candidatos:
        return []

    candidatos.sort(key=lambda x: x["score"], reverse=True)

    a_abrir = []

    for c in candidatos:
        if len(a_abrir) >= slots_libres:
            break

        ticker = c["ticker"]
        precio = precios.get(ticker)
        if not precio:
            continue

        atr = float(c["ind"].get("atr14") or 0)
        if not atr:
            log(f"  [SKIP] {ticker}: sin ATR14 disponible.")
            continue

        qty = calcular_qty_ft(cash, precio, RIESGO_POR_TRADE, capital_actual)
        if qty < 1:
            log(f"  [SKIP] {ticker}: capital insuficiente.")
            continue

        capital_trade = round(precio * qty, 2)
        if capital_trade > cash:
            log(f"  [SKIP] {ticker}: trade ({capital_trade:.2f}) supera cash ({cash:.2f}).")
            continue

        sl = round(precio - ATR_MULT_SL * atr, 4)
        tp = round(precio + ATR_MULT_TP * atr, 4)

        a_abrir.append({
            "ticker":  ticker,
            "precio":  precio,
            "qty":     qty,
            "capital": capital_trade,
            "sl":      sl,
            "tp":      tp,
            "score":   c["score"],
            "detalle": c["detalle"],
            "atr":     round(atr, 4),
        })

        cash -= capital_trade

    return a_abrir


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 55

    log(sep)
    log(f"FT Bot Tecnico | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(sep)

    # 1. Cargar estrategia
    estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)
    if not estrategia:
        log(f"[ERROR] Estrategia '{NOMBRE_ESTRATEGIA}' no encontrada o inactiva.")
        return

    eid = estrategia["id"]
    log(f"Estrategia id={eid} | capital={estrategia['capital_actual']:,.2f} | "
        f"cash={estrategia['cash_disponible']:,.2f}")

    # 2. Indicadores y precios (una query cada uno)
    indicadores     = obtener_indicadores_hoy()
    indicadores_map = {i["ticker"]: i for i in indicadores}
    precios         = obtener_precios_cierre_todos()
    log(f"Indicadores cargados: {len(indicadores)} tickers")

    # 3. Posiciones abiertas
    posiciones = obtener_posiciones_abiertas(eid)
    log(f"Posiciones abiertas: {len(posiciones)}")

    # 4. Filtro earnings para posiciones abiertas
    tickers_pos     = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar hoy): {list(earnings_cierre.keys())}")

    # 5. Evaluar cierres
    log(sep)
    log("CIERRES:")
    a_cerrar = evaluar_cierres(posiciones, precios, earnings_cierre, indicadores_map)

    tickers_cerrados = set()

    if not a_cerrar:
        log("  Sin posiciones a cerrar.")
    else:
        for c in a_cerrar:
            ticker = c["ticker"]
            precio = c["precio_cierre"]
            motivo = c["motivo"]
            pnl_est = round((precio - float(c["precio_entrada"])) * int(c["cantidad"]), 2)
            log(f"  CERRAR {ticker} | precio={precio:.2f} | "
                f"pnl_est={pnl_est:+.2f} | motivo={motivo}")

            tickers_cerrados.add(ticker)

            if not dry_run:
                resultado = cerrar_operacion(c["id"], eid, precio, motivo, hoy)
                log(f"    -> pnl={resultado.get('pnl', 0):+.2f} "
                    f"({resultado.get('pnl_pct', 0):+.2f}%)")

    # 6. Recargar posiciones y estrategia
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_abiertas(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 7. Filtro earnings para candidatos de entrada
    tickers_candidatos = [i["ticker"] for i in indicadores]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_candidatos) if tickers_candidatos else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {len(tickers_bloqueados)} tickers")

    # 8. Evaluar entradas
    log(sep)
    log("ENTRADAS:")
    a_abrir = evaluar_entradas(
        posiciones, estrategia, indicadores,
        precios, tickers_bloqueados, tickers_cerrados,
    )

    if not a_abrir:
        log("  Sin senales de entrada.")
    else:
        for e in a_abrir:
            log(f"  ABRIR {e['ticker']} | precio={e['precio']:.2f} | "
                f"qty={e['qty']} | capital={e['capital']:.2f} | "
                f"score={e['score']:.1f}/{SCORE_MAXIMO} | "
                f"sl={e['sl']:.2f} | tp={e['tp']:.2f} | atr={e['atr']:.4f}")

            if not dry_run:
                detalle = {
                    **e["detalle"],
                    "score":        e["score"],
                    "score_maximo": SCORE_MAXIMO,
                    "atr":          e["atr"],
                    "sl_mult_atr":  ATR_MULT_SL,
                    "tp_mult_atr":  ATR_MULT_TP,
                }
                op_id = abrir_operacion(
                    estrategia_id=eid,
                    ticker=e["ticker"],
                    fecha=hoy,
                    precio=e["precio"],
                    cantidad=e["qty"],
                    stop_loss=e["sl"],
                    take_profit=e["tp"],
                    score=e["score"],
                    detalle=detalle,
                )
                if op_id:
                    log(f"    -> operacion id={op_id} registrada.")

    # 9. Guardar candidatos del dia (abiertos + oportunidades)
    log(sep)
    log("CANDIDATOS:")
    tickers_abiertos_hoy = {p["ticker"] for p in posiciones}
    tickers_que_abrimos  = {e["ticker"] for e in a_abrir}

    candidatos_log = []
    for ind in indicadores:
        t = ind["ticker"]
        if t in tickers_abiertos_hoy or t in tickers_bloqueados:
            continue
        score, _ = calcular_score_tecnico(ind)
        if score < SCORE_ENTRADA:
            continue
        entro = t in tickers_que_abrimos
        candidatos_log.append({
            "ticker":      t,
            "score":       float(score),
            "entro":       entro,
            "motivo_skip": None if entro else "CAPITAL_O_POSICIONES",
        })

    if candidatos_log:
        log(f"  {len(candidatos_log)} candidatos qualifying — "
            f"{sum(1 for c in candidatos_log if c['entro'])} abiertos, "
            f"{sum(1 for c in candidatos_log if not c['entro'])} oportunidades")
        if not dry_run:
            registrar_candidatos_diarios(eid, hoy, candidatos_log)
    else:
        log("  Sin candidatos qualifying hoy.")

    # 10. Metricas del dia
    log(sep)
    if not dry_run:
        registrar_metricas_diarias(eid, hoy)
    else:
        log("[DRY RUN] Metricas no registradas.")

    log("Completado.")
    log(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FT Bot Tecnico")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
