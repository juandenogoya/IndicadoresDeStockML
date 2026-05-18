"""
ft_bot_smc.py
Forward-testing — Replica del Bot 3 Estructura SMC CHoCH/BOS (Alpaca).

Logica:
    ENTRADA : CHoCH/BOS bull en lookback (~10 dias habiles)
              + estructura_10 >= 0 (no rota)
              + choch_bear_10 = 0 (sin cambio bajista hoy)
              + es_alcista = 1 (vela alcista hoy)
              + dist_sl_10_pct entre 1% y 8%
              Score de calidad 0-3 pts para rankear candidatos

    SALIDA  : P1 - Trailing SL roto (precio <= stop_loss)
              P2 - CHoCH bajista hoy (choch_bear_10 = 1)
              P3 - Estructura rota (estructura_10 = -1)
              P4 - Time stop (>= 20 dias calendario)
              Earnings manana = prioridad absoluta (sobre P1-P4)

    SL      : Trailing estructural (swing low), solo sube, nunca baja
    TP      : Sin take profit fijo (filosofia: salida estructural pura)

Diferencias vs Bot Alpaca:
    - Precio de ejecucion = cierre del dia (precios_diarios), sin llamadas a Alpaca
    - Escribe en ft_operaciones / ft_estrategias / ft_metricas_diarias
    - No envia ordenes al broker

Uso:
    python scripts/forward_testing/ft_bot_smc.py
    python scripts/forward_testing/ft_bot_smc.py --dry-run
"""

import sys
import os
import argparse
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Entorno FT: forzar conexion a la DB LOCAL (ver scripts/forward_testing/ft_env.py)
from scripts.forward_testing.ft_env import configurar_entorno_local
configurar_entorno_local()

from src.indicators.earnings_filter import tickers_a_cerrar_hoy, tickers_a_bloquear_entrada
from scripts.forward_testing.ft_scoring import (
    calcular_score_estructura,
    obtener_features_hoy,
    obtener_estructura_tickers,
    calcular_actualizaciones_sl,
    _swing_low_precio,
)

from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_posiciones_abiertas,
    obtener_precios_cierre_todos, abrir_operacion, cerrar_operacion,
    actualizar_stop_loss, registrar_metricas_diarias, calcular_qty_ft,
    calcular_cash_desplegable, registrar_candidatos_diarios,
    registrar_estado_posiciones, backfill_retornos_candidatos,
)

# ── Parametros de la estrategia ───────────────────────────────────────────────
NOMBRE_ESTRATEGIA   = "FT_SMC_v1"
SCORE_ENTRADA_MIN   = 1
SCORE_MAXIMO        = 3
MIN_SL_DIST_PCT     = 1.0
MAX_SL_DIST_PCT     = 8.0
DIAS_MAX_POS        = 20
MAX_POSICIONES      = 5
MAX_DEPLOY_PCT      = 0.80     # nunca desplegar mas del 80% del capital
RIESGO_POR_TRADE    = 0.15


# ── Evaluacion de cierres ─────────────────────────────────────────────────────

def evaluar_cierres(
    posiciones:   list[dict],
    precios:      dict[str, float],
    earnings_map: dict[str, date],
) -> list[dict]:
    """
    Prioridades de salida:
        P0. Earnings manana     (prioridad absoluta, antes que todo)
        P1. Trailing SL roto    (precio <= stop_loss)
        P2. CHoCH bajista hoy   (choch_bear_10 = 1)
        P3. Estructura rota     (estructura_10 = -1)
        P4. Time stop           (>= DIAS_MAX_POS dias calendario)
    """
    if not posiciones:
        return []

    tickers   = [p["ticker"] for p in posiciones]
    datos_map = obtener_estructura_tickers(tickers)
    a_cerrar  = []

    for pos in posiciones:
        ticker        = pos["ticker"]
        datos_hoy     = datos_map.get(ticker)
        precio_actual = precios.get(ticker)
        stop_loss     = float(pos["stop_loss"]) if pos.get("stop_loss") else None
        fecha_entrada = pos["fecha_entrada"]
        dias_abierta  = (date.today() - fecha_entrada).days if fecha_entrada else 0

        if not precio_actual:
            log(f"  [WARN] Sin precio para {ticker}, se omite.")
            continue

        motivo = None

        # P0: Earnings (prioridad absoluta)
        if ticker in earnings_map:
            motivo = "EARNINGS_MANANA"

        elif datos_hoy:
            estructura_10 = int(datos_hoy.get("estructura_10", 0) or 0)
            choch_bear_10 = int(datos_hoy.get("choch_bear_10", 0) or 0)

            # P1: Trailing SL roto
            if stop_loss and precio_actual <= stop_loss:
                motivo = "TRAILING_SL"

            # P2: CHoCH bajista
            elif choch_bear_10 == 1:
                motivo = "CHOCH_BEAR"

            # P3: Estructura rota
            elif estructura_10 == -1:
                motivo = "ESTRUCTURA_ROTA"

            # P4: Time stop
            elif dias_abierta >= DIAS_MAX_POS:
                motivo = f"TIME_STOP_{dias_abierta}D"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar


# ── Trailing SL ───────────────────────────────────────────────────────────────

def procesar_trailing_sl(
    posiciones: list[dict],
    precios:    dict[str, float],
    dry_run:    bool = False,
) -> list[dict]:
    """
    Recalcula el SL estructural para cada posicion abierta.
    Actualiza en DB solo si el nuevo SL es mayor al actual (solo sube).
    Retorna la lista de posiciones con SL actualizado en memoria.
    """
    if not posiciones:
        return posiciones

    updates = calcular_actualizaciones_sl(posiciones)
    if not updates:
        log("  Trailing SL: sin actualizaciones.")
        return posiciones

    pos_map = {p["id"]: p for p in posiciones}

    for u in updates:
        op_id    = u["id"]
        nuevo_sl = u["nuevo_sl"]
        anterior = u["sl_anterior"]
        log(f"  Trailing SL {u['ticker']}: {anterior:.4f} -> {nuevo_sl:.4f} "
            f"(+{u['delta_pct']:.2f}%)")

        if not dry_run:
            actualizar_stop_loss(op_id, nuevo_sl)

        # Actualizar en memoria para que evaluar_cierres use el SL nuevo
        if op_id in pos_map:
            pos_map[op_id] = {**pos_map[op_id], "stop_loss": nuevo_sl}

    return list(pos_map.values())


# ── Evaluacion de entradas ────────────────────────────────────────────────────

def evaluar_entradas(
    posiciones:         list[dict],
    estrategia:         dict,
    precios:            dict[str, float],
    tickers_bloqueados: set,
) -> list[dict]:
    """
    Escanea features estructurales y retorna candidatos para abrir posicion.
    """
    tickers_abiertos = {p["ticker"] for p in posiciones}
    excluidos        = tickers_bloqueados | tickers_abiertos
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

    features   = obtener_features_hoy()
    candidatos = []

    for row in features:
        ticker = row["ticker"]
        if ticker in excluidos:
            continue

        score, detalle = calcular_score_estructura(row)
        if score < SCORE_ENTRADA_MIN:   # incluye score=-1 (filtro obligatorio fallo)
            continue

        candidatos.append({"ticker": ticker, "score": score,
                           "detalle": detalle, "row": row})

    if not candidatos:
        return []

    candidatos.sort(key=lambda x: x["score"], reverse=True)
    a_abrir = []

    for c in candidatos:
        if len(a_abrir) >= slots_libres:
            break

        ticker     = c["ticker"]
        row        = c["row"]
        det        = c["detalle"]
        precio     = precios.get(ticker)

        if not precio:
            continue

        dist_sl_pct = det["dist_sl_pct"]
        swing_low   = _swing_low_precio(precio, dist_sl_pct)

        if swing_low <= 0:
            continue

        # Verificar que SL sigue siendo valido con precio de cierre real
        dist_real = (precio - swing_low) / swing_low * 100
        if not (MIN_SL_DIST_PCT <= dist_real <= MAX_SL_DIST_PCT + 2.0):
            log(f"  [SKIP] {ticker}: dist_sl_real={dist_real:.1f}% fuera de rango.")
            continue

        qty = calcular_qty_ft(cash, precio, RIESGO_POR_TRADE, capital_actual)
        if qty < 1:
            log(f"  [SKIP] {ticker}: capital insuficiente.")
            continue

        capital_trade = round(precio * qty, 2)
        if capital_trade > cash:
            log(f"  [SKIP] {ticker}: trade ({capital_trade:.2f}) supera cash ({cash:.2f}).")
            continue

        evento = "CHoCH_BULL" if det["tuvo_choch_bull"] else "BOS_BULL"
        atr    = float(row.get("atr14") or 0)

        a_abrir.append({
            "ticker":        ticker,
            "precio":        precio,
            "qty":           qty,
            "capital":       capital_trade,
            "sl":            swing_low,
            "tp":            None,
            "score":         c["score"],
            "detalle":       det,
            "evento":        evento,
            "atr":           round(atr, 4),
            "dist_sl_pct":   round(dist_sl_pct, 2),
        })

        cash -= capital_trade

    return a_abrir


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 55

    log(sep)
    log(f"FT Bot SMC Estructura | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(sep)

    # 1. Cargar estrategia
    estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)
    if not estrategia:
        log(f"[ERROR] Estrategia '{NOMBRE_ESTRATEGIA}' no encontrada o inactiva.")
        return

    eid = estrategia["id"]
    log(f"Estrategia id={eid} | capital={estrategia['capital_actual']:,.2f} | "
        f"cash={estrategia['cash_disponible']:,.2f}")

    # 2. Precios de cierre
    precios = obtener_precios_cierre_todos()
    log(f"Precios cargados: {len(precios)} tickers")

    # 3. Posiciones abiertas
    posiciones = obtener_posiciones_abiertas(eid)
    log(f"Posiciones abiertas: {len(posiciones)}")

    # 4. Filtro earnings para posiciones abiertas
    tickers_pos     = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar hoy): {list(earnings_cierre.keys())}")

    # 5. Actualizar trailing SL (antes de evaluar cierres)
    log(sep)
    log("TRAILING SL:")
    posiciones = procesar_trailing_sl(posiciones, precios, dry_run)

    # 6. Evaluar cierres
    log(sep)
    log("CIERRES:")
    a_cerrar = evaluar_cierres(posiciones, precios, earnings_cierre)

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

            if not dry_run:
                resultado = cerrar_operacion(c["id"], eid, precio, motivo, hoy)
                log(f"    -> pnl={resultado.get('pnl', 0):+.2f} "
                    f"({resultado.get('pnl_pct', 0):+.2f}%)")

    # 7. Recargar posiciones y estrategia
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_abiertas(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 8. Filtro earnings para candidatos de entrada
    features_hoy       = obtener_features_hoy()
    tickers_candidatos = [f["ticker"] for f in features_hoy]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_candidatos) if tickers_candidatos else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {len(tickers_bloqueados)} tickers")

    # 9. Evaluar entradas
    log(sep)
    log("ENTRADAS:")
    a_abrir = evaluar_entradas(posiciones, estrategia, precios, tickers_bloqueados)

    if not a_abrir:
        log("  Sin senales de entrada.")
    else:
        for e in a_abrir:
            log(f"  ABRIR {e['ticker']} | evento={e['evento']} | "
                f"precio={e['precio']:.2f} | qty={e['qty']} | "
                f"capital={e['capital']:.2f} | score={e['score']:.0f}/{SCORE_MAXIMO} | "
                f"sl={e['sl']:.4f} | dist_sl={e['dist_sl_pct']:.1f}%")

            if not dry_run:
                detalle = {
                    **e["detalle"],
                    "evento":       e["evento"],
                    "score":        e["score"],
                    "dist_sl_pct":  e["dist_sl_pct"],
                    "atr":          e["atr"],
                }
                op_id = abrir_operacion(
                    estrategia_id=eid,
                    ticker=e["ticker"],
                    fecha=hoy,
                    precio=e["precio"],
                    cantidad=e["qty"],
                    stop_loss=e["sl"],
                    take_profit=None,
                    score=e["score"],
                    detalle=detalle,
                )
                if op_id:
                    log(f"    -> operacion id={op_id} registrada.")

    # 10. Guardar candidatos del dia (abiertos + oportunidades)
    log(sep)
    log("CANDIDATOS:")
    tickers_abiertos_hoy = {p["ticker"] for p in posiciones}
    tickers_que_abrimos  = {e["ticker"] for e in a_abrir}

    candidatos_log = []
    for row in features_hoy:
        t = row["ticker"]
        if t in tickers_abiertos_hoy or t in tickers_bloqueados:
            continue
        score, _ = calcular_score_estructura(row)
        if score < SCORE_ENTRADA_MIN:
            continue
        entro = t in tickers_que_abrimos
        candidatos_log.append({
            "ticker":          t,
            "score":           float(score),
            "entro":           entro,
            "motivo_skip":     None if entro else "CAPITAL_O_POSICIONES",
            "precio_apertura": precios.get(t),
        })

    if candidatos_log:
        log(f"  {len(candidatos_log)} candidatos qualifying — "
            f"{sum(1 for c in candidatos_log if c['entro'])} abiertos, "
            f"{sum(1 for c in candidatos_log if not c['entro'])} oportunidades")
        if not dry_run:
            registrar_candidatos_diarios(eid, hoy, candidatos_log)
    else:
        log("  Sin candidatos qualifying hoy.")

    # 11. Metricas del dia
    log(sep)
    if not dry_run:
        registrar_metricas_diarias(eid, hoy)
    else:
        log("[DRY RUN] Metricas no registradas.")

    # 12. Observacion diaria: snapshots de posiciones + retornos contrafactuales
    if not dry_run:
        registrar_estado_posiciones(eid, hoy, precios)
        backfill_retornos_candidatos(eid, hoy, precios)

    log("Completado.")
    log(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FT Bot SMC Estructura")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
