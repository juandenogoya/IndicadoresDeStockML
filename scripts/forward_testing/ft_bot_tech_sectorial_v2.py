"""
ft_bot_tech_sectorial_v2.py
Forward-testing — AT Tecnico Sectorial con retencion condicional y rotacion.

Estrategia FT_TECH_SECTOR_v2:
    Extension de v1 con dos cambios en la logica de SALIDA:

    1. RETENCION CONDICIONAL: no cierra si score=0 pero el mercado muestra
       acumulacion activa (candle_score_5d >= 0 O up_vol_5d >= 2).
       Cierra solo cuando los tres criterios convergen negativamente:
         tech_score = 0 AND candle_score_5d < 0 AND up_vol_5d < 2
       Motivo: SCORE_DEGRADADO_SIN_MOMENTUM

    2. ROTACION INTRASECTORIAL: si un sector esta lleno (5/5) y existe un
       candidato con score >= peor_score_actual + 1.0, cierra el peor y
       abre el candidato.
       Motivo cierre: ROTACION_SECTORIAL

    ENTRADA: identica a v1 (tech_score >= 4.0, SMA200, etc.)
    SIZING : identico a v1 (~$2.222 por posicion)

Uso:
    python scripts/forward_testing/ft_bot_tech_sectorial_v2.py
    python scripts/forward_testing/ft_bot_tech_sectorial_v2.py --dry-run
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

from sqlalchemy import text
from src.data.database import get_engine
from src.indicators.earnings_filter import tickers_a_cerrar_hoy, tickers_a_bloquear_entrada
from scripts.forward_testing.ft_scoring import (
    calcular_score_tecnico,
    obtener_candle_score_5d,
)
from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_precios_cierre_todos,
    abrir_operacion, cerrar_operacion,
    registrar_metricas_diarias, registrar_candidatos_diarios,
    registrar_estado_posiciones, backfill_retornos_candidatos,
)

# ── Parametros de la estrategia ───────────────────────────────────────────────

NOMBRE_ESTRATEGIA = "FT_TECH_SECTOR_v2"

SECTORES_ACTIVOS = [
    "Technology", "Consumer Cyclical", "Financial Services",
    "Industrials", "Basic Materials", "Healthcare",
    "Energy", "Communication Services", "Consumer Defensive",
]

CAPITAL_TOTAL      = 100_000.0
N_SECTORES         = len(SECTORES_ACTIVOS)                   # 9
SECTOR_BUDGET      = round(CAPITAL_TOTAL / N_SECTORES, 2)    # 11.111,11
MAX_POS_SECTOR     = 5
POSITION_PCT       = 0.20
POSITION_SIZE      = round(SECTOR_BUDGET * POSITION_PCT, 2)  # ~2.222,22

SCORE_ENTRADA      = 4.0
SCORE_MAXIMO       = 5.5
ATR_MULT_SL        = 2.0
ATR_MULT_TP        = 4.0

# v2: parametros de retencion y rotacion
RETENCION_CANDLE_MIN  = 0      # candle_score_5d >= 0  -> retener (condicion OR)
RETENCION_UP_VOL_MIN  = 2      # up_vol_5d >= 2        -> retener (condicion OR)
ROTACION_DELTA_SCORE  = 1.0    # diferencia minima de score para rotar


# ── Queries especificas ───────────────────────────────────────────────────────

def obtener_posiciones_con_sector(estrategia_id: int) -> list[dict]:
    """Posiciones abiertas enriquecidas con sector (JOIN activos)."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT o.id, o.ticker, o.lado, o.fecha_entrada,
                   o.precio_entrada, o.cantidad, o.capital_entrada,
                   o.stop_loss, o.take_profit, o.score_entrada, o.detalle_entrada,
                   COALESCE(a.sector, 'Unknown') AS sector
            FROM ft_operaciones o
            LEFT JOIN activos a ON a.ticker = o.ticker
            WHERE o.estrategia_id = :eid AND o.fecha_salida IS NULL
            ORDER BY a.sector, o.fecha_entrada
        """), {"eid": estrategia_id}).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_indicadores_con_sector() -> list[dict]:
    """
    Indicadores tecnicos del ultimo dia para los 9 sectores activos.
    JOIN con activos (sector) y precios_diarios (close).
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT i.ticker, a.sector,
                   p.close,
                   i.sma21, i.sma50, i.sma200,
                   i.rsi14, i.macd, i.macd_signal,
                   (i.macd - i.macd_signal) AS macd_hist,
                   i.atr14,
                   i.adx, i.vol_relativo
            FROM (
                SELECT DISTINCT ON (ticker)
                       ticker, fecha,
                       sma21, sma50, sma200,
                       rsi14, macd, macd_signal,
                       atr14, adx, vol_relativo
                FROM indicadores_tecnicos
                ORDER BY ticker, fecha DESC
            ) i
            JOIN precios_diarios p
              ON p.ticker = i.ticker AND p.fecha = i.fecha
            JOIN activos a ON a.ticker = i.ticker
            WHERE a.activo = TRUE
              AND a.sector = ANY(:sectores)
            ORDER BY a.sector, i.ticker
        """), {"sectores": SECTORES_ACTIVOS}).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_estado_tecnico_tickers(tickers: list[str]) -> dict[str, dict]:
    """Indicadores actuales para evaluar score actual de posiciones abiertas."""
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (ticker)
                   ticker, sma21, sma50, sma200,
                   rsi14, macd, macd_signal,
                   (macd - macd_signal) AS macd_hist,
                   atr14
            FROM indicadores_tecnicos
            WHERE ticker = ANY(:tickers)
            ORDER BY ticker, fecha DESC
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


def obtener_up_vol_5d_tickers(tickers: list[str]) -> dict[str, int]:
    """
    Calcula up_vol_5d para cada ticker: dias alcistas con volumen confirmado
    en los ultimos 5 dias de features_precio_accion.
    Retorna {ticker: up_vol_5d}.
    """
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            WITH ranked AS (
                SELECT ticker, es_alcista, vol_price_confirm,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
                FROM features_precio_accion
                WHERE ticker = ANY(:tickers)
            )
            SELECT ticker,
                   SUM(CASE WHEN es_alcista = 1 AND vol_price_confirm = 1
                            THEN 1 ELSE 0 END) AS up_vol_5d
            FROM ranked
            WHERE rn <= 5
            GROUP BY ticker
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: int(r.up_vol_5d or 0) for r in rows}


def _precio_close_para_indicadores(tickers: list[str]) -> dict[str, float]:
    """Agrega el close mas reciente al diccionario de indicadores."""
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (ticker) ticker, close
            FROM precios_diarios
            WHERE ticker = ANY(:tickers)
            ORDER BY ticker, fecha DESC
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: float(r.close) for r in rows}


# ── Evaluacion de cierres (v2) ────────────────────────────────────────────────

def evaluar_cierres(
    posiciones:      list[dict],
    precios:         dict[str, float],
    earnings_map:    dict[str, date],
    indicadores_map: dict[str, dict],
    candle_scores:   dict[str, float],
    up_vol_map:      dict[str, int],
) -> list[dict]:
    """
    Prioridades de salida v2:
        P0. Earnings manana        -> EARNINGS_MANANA
        P1. Stop loss ATR          -> STOP_LOSS_ATR
        P2. Take profit ATR        -> TAKE_PROFIT_ATR
        P3. Score degradado con    -> SCORE_DEGRADADO_SIN_MOMENTUM
            confirmacion de senales
            (tech=0 AND candle<0 AND up_vol<2)

    Retencion (NO cerrar) si score=0 pero:
        candle_score_5d >= 0  OR  up_vol_5d >= 2
    """
    if not posiciones:
        return []

    a_cerrar = []

    for pos in posiciones:
        ticker         = pos["ticker"]
        precio_actual  = precios.get(ticker)
        sl             = float(pos["stop_loss"])  if pos.get("stop_loss")  else None
        tp             = float(pos["take_profit"]) if pos.get("take_profit") else None
        ind            = indicadores_map.get(ticker)

        if not precio_actual:
            log(f"  [WARN] Sin precio para {ticker}, se omite.")
            continue

        motivo = None

        # P0: Earnings
        if ticker in earnings_map:
            motivo = "EARNINGS_MANANA"

        # P1: Stop loss ATR
        elif sl and precio_actual <= sl:
            motivo = "STOP_LOSS_ATR"

        # P2: Take profit ATR
        elif tp and precio_actual >= tp:
            motivo = "TAKE_PROFIT_ATR"

        # P3: Score degradado con confirmacion de mercado
        elif ind:
            score_actual, _ = calcular_score_tecnico(ind)
            if score_actual == 0.0:
                cs5d   = candle_scores.get(ticker, 0.0) or 0.0
                upv5   = up_vol_map.get(ticker, 0) or 0
                cerrar = cs5d < RETENCION_CANDLE_MIN and upv5 < RETENCION_UP_VOL_MIN
                if cerrar:
                    motivo = "SCORE_DEGRADADO_SIN_MOMENTUM"
                else:
                    log(f"  [RETENER] {ticker}: score=0 pero "
                        f"candle={cs5d:.1f} upvol={upv5} -> "
                        f"{'candle OK' if cs5d >= RETENCION_CANDLE_MIN else 'vol OK'}")

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar


# ── Evaluacion de entradas + rotacion por sector (v2) ─────────────────────────

def evaluar_entradas_y_rotacion_sectorial(
    posiciones:         list[dict],
    indicadores:        list[dict],
    precios:            dict[str, float],
    tickers_bloqueados: set,
    tickers_cerrados:   set,
    candle_scores:      dict[str, float],
    up_vol_map:         dict[str, int],
    indicadores_map:    dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """
    Para cada sector:
        1. Si tiene slots libres: entradas normales (score >= SCORE_ENTRADA)
        2. Si esta lleno (5/5): evaluar rotacion
           - Si existe candidato con score >= peor_score_actual + ROTACION_DELTA_SCORE
           - Cierra el peor, abre el candidato

    Retorna (a_abrir, a_rotar)
        a_abrir : list de dicts con datos de nuevas entradas
        a_rotar : list de dicts con (pos_cerrar, candidato_abrir)
    """
    tickers_abiertos = {p["ticker"] for p in posiciones}
    excluidos        = tickers_bloqueados | tickers_abiertos | tickers_cerrados

    # Capital y slots ya usados por sector
    sector_deployed = {}
    sector_n_open   = {}
    for pos in posiciones:
        s = pos.get("sector", "Unknown")
        sector_deployed[s] = sector_deployed.get(s, 0.0) + float(pos["capital_entrada"])
        sector_n_open[s]   = sector_n_open.get(s, 0) + 1

    # Score actual de cada posicion abierta (para comparar en rotacion)
    score_actual_map = {}
    for pos in posiciones:
        t   = pos["ticker"]
        ind = indicadores_map.get(t)
        if ind:
            sc, _ = calcular_score_tecnico(ind)
            score_actual_map[t] = sc
        else:
            score_actual_map[t] = float(pos.get("score_entrada") or 0)

    # Candidatos qualifying por sector
    candidatos_por_sector: dict[str, list] = {s: [] for s in SECTORES_ACTIVOS}

    for ind in indicadores:
        t = ind["ticker"]
        s = ind.get("sector", "Unknown")
        if t in excluidos or s not in SECTORES_ACTIVOS:
            continue
        score, detalle = calcular_score_tecnico(ind)
        if score < SCORE_ENTRADA:
            continue
        candidatos_por_sector[s].append({
            "ticker":  t,
            "sector":  s,
            "score":   score,
            "detalle": detalle,
            "ind":     ind,
        })

    for s in candidatos_por_sector:
        candidatos_por_sector[s].sort(key=lambda x: x["score"], reverse=True)

    a_abrir = []
    a_rotar = []

    for sector in SECTORES_ACTIVOS:
        candidatos = candidatos_por_sector[sector]
        deployed   = sector_deployed.get(sector, 0.0)
        n_open     = sector_n_open.get(sector, 0)
        available  = round(SECTOR_BUDGET - deployed, 2)
        slots      = MAX_POS_SECTOR - n_open

        if not candidatos:
            continue

        # ── Caso A: sector con slots libres — entradas normales ───────────────
        if slots > 0 and available > 0:
            log(f"  [{sector}] {len(candidatos)} candidatos | "
                f"slots={slots} | disponible=${available:,.2f}")
            avail_local = available

            for c in candidatos:
                if slots <= 0 or avail_local <= 0:
                    break

                ticker = c["ticker"]
                precio = precios.get(ticker)
                if not precio:
                    continue

                atr = float(c["ind"].get("atr14") or 0)
                if not atr:
                    log(f"    [SKIP] {ticker}: sin ATR14.")
                    continue

                qty = int(POSITION_SIZE / precio)
                if qty < 1:
                    log(f"    [SKIP] {ticker}: precio ${precio:.2f} > "
                        f"position_size ${POSITION_SIZE:,.2f}.")
                    continue

                capital_trade = round(precio * qty, 2)
                if capital_trade > avail_local:
                    log(f"    [SKIP] {ticker}: trade ${capital_trade:,.2f} > "
                        f"disponible ${avail_local:,.2f}.")
                    continue

                sl = round(precio - ATR_MULT_SL * atr, 4)
                tp = round(precio + ATR_MULT_TP * atr, 4)

                a_abrir.append({
                    "ticker":  ticker,
                    "sector":  sector,
                    "precio":  precio,
                    "qty":     qty,
                    "capital": capital_trade,
                    "sl":      sl,
                    "tp":      tp,
                    "score":   c["score"],
                    "detalle": c["detalle"],
                    "atr":     round(atr, 4),
                })

                avail_local -= capital_trade
                slots       -= 1

        # ── Caso B: sector lleno — evaluar rotacion ───────────────────────────
        elif slots == 0 and candidatos:
            # Mejor candidato externo
            mejor_candidato = candidatos[0]

            # Peor posicion abierta en este sector
            pos_sector = [p for p in posiciones if p.get("sector") == sector]
            if not pos_sector:
                continue

            peor_pos   = min(pos_sector, key=lambda p: score_actual_map.get(p["ticker"], 0))
            peor_score = score_actual_map.get(peor_pos["ticker"], 0)

            delta = mejor_candidato["score"] - peor_score
            if delta >= ROTACION_DELTA_SCORE:
                log(f"  [{sector}] ROTACION: cerrar {peor_pos['ticker']} "
                    f"(score_actual={peor_score:.1f}) -> abrir "
                    f"{mejor_candidato['ticker']} (score={mejor_candidato['score']:.1f}) "
                    f"delta={delta:.1f}")

                ticker_nuevo = mejor_candidato["ticker"]
                precio_nuevo = precios.get(ticker_nuevo)
                atr_nuevo    = float(mejor_candidato["ind"].get("atr14") or 0)

                if precio_nuevo and atr_nuevo:
                    qty_nuevo     = int(POSITION_SIZE / precio_nuevo)
                    capital_nuevo = round(precio_nuevo * qty_nuevo, 2)

                    if qty_nuevo >= 1:
                        sl_nuevo = round(precio_nuevo - ATR_MULT_SL * atr_nuevo, 4)
                        tp_nuevo = round(precio_nuevo + ATR_MULT_TP * atr_nuevo, 4)

                        a_rotar.append({
                            "pos_cerrar": peor_pos,
                            "candidato":  {
                                "ticker":  ticker_nuevo,
                                "sector":  sector,
                                "precio":  precio_nuevo,
                                "qty":     qty_nuevo,
                                "capital": capital_nuevo,
                                "sl":      sl_nuevo,
                                "tp":      tp_nuevo,
                                "score":   mejor_candidato["score"],
                                "detalle": mejor_candidato["detalle"],
                                "atr":     round(atr_nuevo, 4),
                            },
                        })
                    else:
                        log(f"    [SKIP ROT] {ticker_nuevo}: sin qty (precio alto).")
                else:
                    log(f"    [SKIP ROT] {ticker_nuevo}: sin precio o ATR.")
            else:
                log(f"  [{sector}] Sin rotacion: delta={delta:.1f} < {ROTACION_DELTA_SCORE}")

    return a_abrir, a_rotar


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 60

    log(sep)
    log(f"FT Bot Tech Sectorial v2 | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(f"  {N_SECTORES} sectores x ${SECTOR_BUDGET:,.2f} | "
        f"Position size: ${POSITION_SIZE:,.2f} | Max {MAX_POS_SECTOR} pos/sector")
    log(f"  Retencion: candle >= {RETENCION_CANDLE_MIN} OR up_vol >= {RETENCION_UP_VOL_MIN}")
    log(f"  Rotacion delta minimo: {ROTACION_DELTA_SCORE}")
    log(sep)

    # 1. Cargar estrategia
    estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)
    if not estrategia:
        log(f"[ERROR] Estrategia '{NOMBRE_ESTRATEGIA}' no encontrada o inactiva.")
        log("Hint: registrar en ft_estrategias con logica='tecnico_sectorial_v2'")
        return

    eid = estrategia["id"]
    log(f"Estrategia id={eid} | capital={estrategia['capital_actual']:,.2f} | "
        f"cash={estrategia['cash_disponible']:,.2f}")

    # 2. Precios de cierre
    precios = obtener_precios_cierre_todos()
    log(f"Precios cargados: {len(precios)} tickers")

    # 3. Indicadores con sector
    indicadores = obtener_indicadores_con_sector()
    log(f"Indicadores cargados: {len(indicadores)} tickers en {N_SECTORES} sectores")

    # 4. Posiciones abiertas con sector
    posiciones = obtener_posiciones_con_sector(eid)
    log(f"Posiciones abiertas: {len(posiciones)}")

    if posiciones:
        from collections import Counter
        sector_count = Counter(p["sector"] for p in posiciones)
        for s, n in sorted(sector_count.items()):
            dep = sum(float(p["capital_entrada"]) for p in posiciones if p["sector"] == s)
            log(f"  {s}: {n} pos | ${dep:,.2f} / ${SECTOR_BUDGET:,.2f}")

    # 5. Senales de mercado para posiciones abiertas (retencion)
    tickers_pos = [p["ticker"] for p in posiciones]
    log(sep)
    log("SENALES DE MERCADO (retencion):")
    candle_scores   = obtener_candle_score_5d()
    indicadores_map = obtener_estado_tecnico_tickers(tickers_pos)
    up_vol_map      = obtener_up_vol_5d_tickers(tickers_pos)

    # 6. Filtro earnings para posiciones abiertas
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar hoy): {list(earnings_cierre.keys())}")

    # 7. Evaluar cierres
    log(sep)
    log("CIERRES:")
    a_cerrar         = evaluar_cierres(
        posiciones, precios, earnings_cierre,
        indicadores_map, candle_scores, up_vol_map,
    )
    tickers_cerrados = set()

    if not a_cerrar:
        log("  Sin posiciones a cerrar.")
    else:
        for c in a_cerrar:
            ticker = c["ticker"]
            precio = c["precio_cierre"]
            motivo = c["motivo"]
            pnl_est = round((precio - float(c["precio_entrada"])) * int(c["cantidad"]), 2)
            log(f"  CERRAR {ticker} [{c.get('sector','-')}] | "
                f"precio={precio:.2f} | pnl_est={pnl_est:+.2f} | motivo={motivo}")

            tickers_cerrados.add(ticker)

            if not dry_run:
                resultado = cerrar_operacion(c["id"], eid, precio, motivo, hoy)
                log(f"    -> pnl={resultado.get('pnl', 0):+.2f} "
                    f"({resultado.get('pnl_pct', 0):+.2f}%)")

    # 8. Recargar tras cierres
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_con_sector(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 9. Filtro earnings para candidatos de entrada
    tickers_candidatos = [i["ticker"] for i in indicadores]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_candidatos) if tickers_candidatos else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {len(tickers_bloqueados)} tickers")

    # 10. Indicadores actuales (para calcular score actual de posiciones — rotacion)
    todos_tickers_ind   = [i["ticker"] for i in indicadores]
    indicadores_map_pos = obtener_estado_tecnico_tickers(todos_tickers_ind)
    up_vol_todos        = obtener_up_vol_5d_tickers(todos_tickers_ind)

    # 11. Evaluar entradas normales + rotacion
    log(sep)
    log("ENTRADAS Y ROTACION POR SECTOR:")
    a_abrir, a_rotar = evaluar_entradas_y_rotacion_sectorial(
        posiciones, indicadores, precios,
        tickers_bloqueados, tickers_cerrados,
        candle_scores, up_vol_todos, indicadores_map_pos,
    )

    # 11a. Registrar rotaciones
    if not a_rotar:
        log("  Sin rotaciones.")
    else:
        for rot in a_rotar:
            pos_cerrar = rot["pos_cerrar"]
            cand       = rot["candidato"]
            precio_sal = precios.get(pos_cerrar["ticker"])
            if not precio_sal:
                log(f"  [SKIP ROT] Sin precio para cerrar {pos_cerrar['ticker']}")
                continue

            pnl_est = round(
                (precio_sal - float(pos_cerrar["precio_entrada"])) * int(pos_cerrar["cantidad"]), 2
            )
            log(f"  ROTAR [{cand['sector']}] cerrar {pos_cerrar['ticker']} "
                f"pnl_est={pnl_est:+.2f} -> abrir {cand['ticker']} "
                f"score={cand['score']:.1f}")

            if not dry_run:
                resultado = cerrar_operacion(
                    pos_cerrar["id"], eid, precio_sal, "ROTACION_SECTORIAL", hoy
                )
                log(f"    -> cierre pnl={resultado.get('pnl', 0):+.2f} "
                    f"({resultado.get('pnl_pct', 0):+.2f}%)")

                detalle_cand = {
                    **cand["detalle"],
                    "sector":        cand["sector"],
                    "score":         cand["score"],
                    "score_maximo":  SCORE_MAXIMO,
                    "atr":           cand["atr"],
                    "sl_mult_atr":   ATR_MULT_SL,
                    "tp_mult_atr":   ATR_MULT_TP,
                    "sector_budget": SECTOR_BUDGET,
                    "position_size": POSITION_SIZE,
                    "rotacion":      True,
                }
                op_id = abrir_operacion(
                    estrategia_id=eid,
                    ticker=cand["ticker"],
                    fecha=hoy,
                    precio=cand["precio"],
                    cantidad=cand["qty"],
                    stop_loss=cand["sl"],
                    take_profit=cand["tp"],
                    score=cand["score"],
                    detalle=detalle_cand,
                )
                if op_id:
                    log(f"    -> apertura id={op_id} registrada.")

    # 11b. Registrar entradas normales
    if not a_abrir:
        log("  Sin entradas normales.")
    else:
        for e in a_abrir:
            log(f"  ABRIR {e['ticker']} [{e['sector']}] | "
                f"precio={e['precio']:.2f} | qty={e['qty']} | "
                f"capital={e['capital']:,.2f} | score={e['score']:.1f}/{SCORE_MAXIMO} | "
                f"sl={e['sl']:.2f} | tp={e['tp']:.2f} | atr={e['atr']:.4f}")

            if not dry_run:
                detalle = {
                    **e["detalle"],
                    "sector":        e["sector"],
                    "score":         e["score"],
                    "score_maximo":  SCORE_MAXIMO,
                    "atr":           e["atr"],
                    "sl_mult_atr":   ATR_MULT_SL,
                    "tp_mult_atr":   ATR_MULT_TP,
                    "sector_budget": SECTOR_BUDGET,
                    "position_size": POSITION_SIZE,
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

    # 12. Candidatos del dia
    log(sep)
    log("CANDIDATOS:")
    tickers_abiertos_hoy = {p["ticker"] for p in posiciones}
    tickers_que_abrimos  = {e["ticker"] for e in a_abrir}
    tickers_rotados      = {rot["candidato"]["ticker"] for rot in a_rotar}

    candidatos_log = []
    for ind in indicadores:
        t = ind["ticker"]
        s = ind.get("sector", "Unknown")
        if t in tickers_abiertos_hoy or t in tickers_bloqueados:
            continue
        score, _ = calcular_score_tecnico(ind)
        if score < SCORE_ENTRADA:
            continue
        entro = t in tickers_que_abrimos or t in tickers_rotados
        if not entro:
            # Determinar motivo_skip
            n_open = sum(1 for p in posiciones if p.get("sector") == s)
            motivo = f"ROTACION_NO_APLICA_{s}" if n_open >= MAX_POS_SECTOR else f"CAPITAL_O_SLOTS_{s}"
        else:
            motivo = None
        candidatos_log.append({
            "ticker":          t,
            "score":           float(score),
            "entro":           entro,
            "motivo_skip":     motivo,
            "precio_apertura": precios.get(t),
        })

    if candidatos_log:
        n_entro = sum(1 for c in candidatos_log if c["entro"])
        log(f"  {len(candidatos_log)} qualifying — {n_entro} abiertos, "
            f"{len(candidatos_log)-n_entro} oportunidades")
        if not dry_run:
            registrar_candidatos_diarios(eid, hoy, candidatos_log)
    else:
        log("  Sin candidatos qualifying hoy.")

    # 13. Metricas del dia
    log(sep)
    if not dry_run:
        registrar_metricas_diarias(eid, hoy)
    else:
        log("[DRY RUN] Metricas no registradas.")

    # 14. Snapshots de posiciones + retornos contrafactuales
    if not dry_run:
        posiciones_final = obtener_posiciones_con_sector(eid)
        registrar_estado_posiciones(eid, hoy, precios)
        backfill_retornos_candidatos(eid, hoy, precios)

    log("Completado.")
    log(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FT Bot Tech Sectorial v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
