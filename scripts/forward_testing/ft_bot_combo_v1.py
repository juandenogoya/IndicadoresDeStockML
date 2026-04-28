"""
ft_bot_combo_v1.py
Forward-testing — AT Tecnico sectorial con scoring de velas como desempate.

Estrategia FT_COMBO_v1:
    Base identica a FT_TECH_SECTOR_v1:
        Capital $100.000 / 9 sectores / $11.111 por sector.
        Max 5 posiciones por sector al 20% del presupuesto sectorial.
        SL = 2x ATR14 | TP = 4x ATR14.

    Diferencia clave vs FT_TECH_SECTOR_v1:
        Cuando dos o mas tickers tienen el mismo score tecnico,
        el desempate lo gana el que tiene mejor score de velas
        acumulado en los ultimos 5 dias de trading.

        Orden de ranking dentro de cada sector:
            1. tech_score DESC          (primario  — SMA/MACD/RSI, max 5.5)
            2. candle_score_5d DESC     (desempate — velas 5 dias, tipico -12 a +17)

        Adicionalmente: si candle_score_5d < CANDLE_SCORE_MIN (-3),
        el ticker se excluye aunque el tech_score sea 5.5.
        Significa que en los ultimos 5 dias las senales de distribucion
        superan a las de acumulacion — el tecnico puede estar rezagado.

    Scoring de velas (calculado en ft_scoring.obtener_candle_score_5d):
        Por dia (+/-):
            es_alcista              +0.5 / -0.5  base direccional
            patron_engulfing_bull   +1.5  (absorbe oferta previa)
            patron_hammer           +1.0  (rechazo de zona baja)
            patron_marubozu bull    +1.0  (fuerza compradora total)
            patron_engulfing_bear   -1.5  (absorbe demanda previa)
            patron_shooting_star    -1.0  (rechazo de zona alta)
            patron_marubozu bear    -0.5  (fuerza vendedora)
            vol_price_confirm       +1.0  (volumen confirma movimiento)
            vol_price_diverge       -0.5  (volumen diverge del movimiento)
            vol_spike solo          +0.5  (atencion elevada)
        Una sola vez (estructura de mercado, ultimo dia):
            bos_bull_5              +2.0
            choch_bull_5            +1.5
            bos_bear_5              -2.0
            choch_bear_5            -3.0

Diferencias vs Bot Alpaca:
    - Precio de ejecucion = cierre del dia (precios_diarios), sin Alpaca
    - Escribe en ft_operaciones / ft_estrategias / ft_metricas_diarias
    - No envia ordenes al broker

Uso:
    python scripts/forward_testing/ft_bot_combo_v1.py
    python scripts/forward_testing/ft_bot_combo_v1.py --dry-run
"""

import sys
import os
import argparse
from datetime import date
from collections import Counter

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
)

# ── Parametros de la estrategia ───────────────────────────────────────────────

NOMBRE_ESTRATEGIA = "FT_COMBO_v1"

SECTORES_ACTIVOS = [
    "Technology", "Consumer Cyclical", "Financial Services",
    "Industrials", "Basic Materials", "Healthcare",
    "Energy", "Communication Services", "Consumer Defensive",
]

CAPITAL_TOTAL  = 100_000.0
N_SECTORES     = len(SECTORES_ACTIVOS)           # 9
SECTOR_BUDGET  = round(CAPITAL_TOTAL / N_SECTORES, 2)  # 11.111,11
MAX_POS_SECTOR = 5
POSITION_PCT   = 0.20
POSITION_SIZE  = round(SECTOR_BUDGET * POSITION_PCT, 2)  # ~2.222,22

SCORE_ENTRADA     = 4.0
SCORE_SALIDA      = 3.5
SCORE_MAXIMO      = 5.5
ATR_MULT_SL       = 2.0
ATR_MULT_TP       = 4.0

# Umbral minimo de candle score: si un ticker tiene score < este valor
# se excluye aunque el score tecnico sea perfecto (distribucion activa)
CANDLE_SCORE_MIN = -3.0


# ── Queries especificas de este bot ──────────────────────────────────────────

def obtener_posiciones_con_sector(estrategia_id: int) -> list:
    """Posiciones abiertas enriquecidas con el sector del ticker."""
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


def obtener_indicadores_con_sector() -> list:
    """
    Indicadores tecnicos del ultimo dia para los 9 sectores activos.
    Devuelve close, sma21, sma50, sma200, rsi14, macd, macd_signal, atr14.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT i.ticker, a.sector,
                   p.close,
                   i.sma21, i.sma50, i.sma200,
                   i.rsi14, i.macd, i.macd_signal,
                   (i.macd - i.macd_signal) AS macd_hist,
                   i.atr14
            FROM (
                SELECT DISTINCT ON (ticker)
                       ticker, fecha,
                       sma21, sma50, sma200,
                       rsi14, macd, macd_signal,
                       atr14
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


def obtener_estado_tecnico_tickers(tickers: list) -> dict:
    """Indicadores actuales para evaluar score de salida de posiciones abiertas."""
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (i.ticker)
                   i.ticker,
                   p.close,
                   i.sma21, i.sma50, i.sma200,
                   i.rsi14, i.macd, i.macd_signal,
                   i.atr14
            FROM indicadores_tecnicos i
            JOIN precios_diarios p
              ON p.ticker = i.ticker AND p.fecha = i.fecha
            WHERE i.ticker = ANY(:tickers)
            ORDER BY i.ticker, i.fecha DESC
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


# ── Evaluacion de cierres ─────────────────────────────────────────────────────

def evaluar_cierres(
    posiciones:      list,
    precios:         dict,
    earnings_map:    dict,
    indicadores_map: dict,
) -> list:
    """
    Prioridades de salida (identicas a FT_TECH_SECTOR_v1):
        P1. Earnings manana  (prioridad absoluta)
        P2. Score degradado  (<= SCORE_SALIDA) — exit primario
        P3. Stop loss ATR    — exit emergencia
        P4. Take profit ATR  — exit emergencia
    """
    if not posiciones:
        return []

    a_cerrar = []

    for pos in posiciones:
        ticker         = pos["ticker"]
        precio_actual  = precios.get(ticker)
        sl             = float(pos["stop_loss"])   if pos.get("stop_loss")   else None
        tp             = float(pos["take_profit"]) if pos.get("take_profit") else None
        ind            = indicadores_map.get(ticker)

        if not precio_actual:
            log(f"  [WARN] Sin precio para {ticker}, se omite.")
            continue

        motivo = None

        if ticker in earnings_map:
            motivo = "EARNINGS_MANANA"
        elif ind:
            score_actual, _ = calcular_score_tecnico(ind)
            if score_actual <= SCORE_SALIDA:
                motivo = f"SCORE_DEGRADADO_{score_actual:.1f}"

        if not motivo and sl and precio_actual <= sl:
            motivo = "STOP_LOSS_ATR"
        if not motivo and tp and precio_actual >= tp:
            motivo = "TAKE_PROFIT_ATR"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar


# ── Evaluacion de entradas por sector ─────────────────────────────────────────

def evaluar_entradas_combo(
    posiciones:         list,
    indicadores:        list,
    candle_scores:      dict,
    precios:            dict,
    tickers_bloqueados: set,
    tickers_cerrados:   set,
) -> list:
    """
    Para cada sector:
        1. Filtra candidatos con tech_score >= SCORE_ENTRADA
        2. Descarta tickers con candle_score_5d < CANDLE_SCORE_MIN
        3. Rankea por (tech_score DESC, candle_score_5d DESC)
        4. Abre los mejores hasta completar slots o presupuesto sectorial
    """
    tickers_abiertos = {p["ticker"] for p in posiciones}
    excluidos        = tickers_bloqueados | tickers_abiertos | tickers_cerrados

    # Capital y slots abiertos por sector
    sector_deployed = {}
    sector_n_open   = {}
    for pos in posiciones:
        s = pos.get("sector", "Unknown")
        sector_deployed[s] = sector_deployed.get(s, 0.0) + float(pos["capital_entrada"])
        sector_n_open[s]   = sector_n_open.get(s, 0) + 1

    # Candidatos qualifying por sector
    candidatos_por_sector = {s: [] for s in SECTORES_ACTIVOS}

    for ind in indicadores:
        t = ind["ticker"]
        s = ind.get("sector", "Unknown")
        if t in excluidos or s not in SECTORES_ACTIVOS:
            continue

        tech_score, detalle = calcular_score_tecnico(ind)
        if tech_score < SCORE_ENTRADA:
            continue

        candle_score = candle_scores.get(t, 0.0)

        # Umbral minimo de velas: excluir si distribucion activa
        if candle_score < CANDLE_SCORE_MIN:
            continue

        candidatos_por_sector[s].append({
            "ticker":       t,
            "sector":       s,
            "tech_score":   tech_score,
            "candle_score": candle_score,
            "detalle":      detalle,
            "ind":          ind,
        })

    # Ranking por (tech_score DESC, candle_score DESC) dentro de cada sector
    for s in candidatos_por_sector:
        candidatos_por_sector[s].sort(
            key=lambda x: (x["tech_score"], x["candle_score"]),
            reverse=True,
        )

    a_abrir = []

    for sector in SECTORES_ACTIVOS:
        candidatos = candidatos_por_sector[sector]
        deployed   = sector_deployed.get(sector, 0.0)
        n_open     = sector_n_open.get(sector, 0)
        available  = round(SECTOR_BUDGET - deployed, 2)
        slots      = MAX_POS_SECTOR - n_open

        if not candidatos:
            continue
        if slots <= 0:
            log(f"  [{sector}] Max posiciones ({MAX_POS_SECTOR}) alcanzado.")
            continue
        if available <= 0:
            log(f"  [{sector}] Presupuesto agotado ({deployed:,.2f}/{SECTOR_BUDGET:,.2f}).")
            continue

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
                "ticker":       ticker,
                "sector":       sector,
                "precio":       precio,
                "qty":          qty,
                "capital":      capital_trade,
                "sl":           sl,
                "tp":           tp,
                "tech_score":   c["tech_score"],
                "candle_score": c["candle_score"],
                "detalle":      c["detalle"],
                "atr":          round(atr, 4),
            })

            avail_local -= capital_trade
            slots       -= 1

    return a_abrir


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 62

    log(sep)
    log(f"FT Bot COMBO v1 | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(f"  {N_SECTORES} sectores x ${SECTOR_BUDGET:,.2f} | "
        f"Position: ${POSITION_SIZE:,.2f} | Max {MAX_POS_SECTOR}/sector")
    log(f"  Ranking: tech_score DESC, candle_score_5d DESC | "
        f"Filtro candle: >= {CANDLE_SCORE_MIN}")
    log(sep)

    # 1. Cargar estrategia
    estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)
    if not estrategia:
        log(f"[ERROR] Estrategia '{NOMBRE_ESTRATEGIA}' no encontrada o inactiva.")
        log("Hint: python scripts/forward_testing/ft_setup_estrategias.py")
        return

    eid = estrategia["id"]
    log(f"Estrategia id={eid} | capital={estrategia['capital_actual']:,.2f} | "
        f"cash={estrategia['cash_disponible']:,.2f}")

    # 2. Datos de mercado (3 queries en paralelo conceptual — se ejecutan secuencialmente)
    precios      = obtener_precios_cierre_todos()
    indicadores  = obtener_indicadores_con_sector()
    candle_scores = obtener_candle_score_5d()   # dict {ticker: score}

    log(f"Precios: {len(precios)} tickers | "
        f"Indicadores: {len(indicadores)} | "
        f"Candle scores: {len(candle_scores)} tickers")

    # 3. Posiciones abiertas con sector
    posiciones = obtener_posiciones_con_sector(eid)
    log(f"Posiciones abiertas: {len(posiciones)}")

    if posiciones:
        sector_count = Counter(p["sector"] for p in posiciones)
        for s, n in sorted(sector_count.items()):
            dep = sum(float(p["capital_entrada"])
                      for p in posiciones if p["sector"] == s)
            log(f"  {s}: {n} pos | ${dep:,.2f} / ${SECTOR_BUDGET:,.2f}")

    # 4. Filtro earnings — posiciones abiertas
    tickers_pos     = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar): {list(earnings_cierre.keys())}")

    # 5. Evaluar cierres
    log(sep)
    log("CIERRES:")
    tickers_pos_all = [p["ticker"] for p in posiciones]
    indicadores_map = obtener_estado_tecnico_tickers(tickers_pos_all)
    a_cerrar        = evaluar_cierres(posiciones, precios, earnings_cierre, indicadores_map)
    tickers_cerrados = set()

    if not a_cerrar:
        log("  Sin posiciones a cerrar.")
    else:
        for c in a_cerrar:
            ticker  = c["ticker"]
            precio  = c["precio_cierre"]
            motivo  = c["motivo"]
            pnl_est = round(
                (precio - float(c["precio_entrada"])) * int(c["cantidad"]), 2
            )
            log(f"  CERRAR {ticker} [{c.get('sector','-')}] | "
                f"precio={precio:.2f} | pnl_est={pnl_est:+.2f} | motivo={motivo}")
            tickers_cerrados.add(ticker)

            if not dry_run:
                resultado = cerrar_operacion(c["id"], eid, precio, motivo, hoy)
                log(f"    -> pnl={resultado.get('pnl', 0):+.2f} "
                    f"({resultado.get('pnl_pct', 0):+.2f}%)")

    # 6. Recargar tras cierres
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_con_sector(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 7. Filtro earnings — candidatos de entrada
    tickers_candidatos = [i["ticker"] for i in indicadores]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_candidatos) if tickers_candidatos else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {len(tickers_bloqueados)} tickers")

    # 8. Evaluar entradas por sector
    log(sep)
    log("ENTRADAS POR SECTOR:")
    a_abrir = evaluar_entradas_combo(
        posiciones, indicadores, candle_scores,
        precios, tickers_bloqueados, tickers_cerrados,
    )

    # Estadistica de candle scores de candidatos
    candle_vals = [candle_scores.get(i["ticker"], 0.0) for i in indicadores]
    excluidos_candle = sum(
        1 for i in indicadores
        if calcular_score_tecnico(i)[0] >= SCORE_ENTRADA
        and candle_scores.get(i["ticker"], 0.0) < CANDLE_SCORE_MIN
    )
    if excluidos_candle:
        log(f"  [INFO] {excluidos_candle} tickers excluidos por candle_score < {CANDLE_SCORE_MIN}")

    if not a_abrir:
        log("  Sin entradas.")
    else:
        for e in a_abrir:
            log(f"  ABRIR {e['ticker']} [{e['sector']}] | "
                f"tech={e['tech_score']:.1f}/{SCORE_MAXIMO} | "
                f"candle={e['candle_score']:+.1f} | "
                f"precio={e['precio']:.2f} | qty={e['qty']} | "
                f"capital={e['capital']:,.2f} | "
                f"sl={e['sl']:.2f} | tp={e['tp']:.2f}")

            if not dry_run:
                detalle = {
                    **e["detalle"],
                    "sector":        e["sector"],
                    "tech_score":    e["tech_score"],
                    "candle_score":  e["candle_score"],
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
                    score=e["tech_score"],
                    detalle=detalle,
                )
                if op_id:
                    log(f"    -> operacion id={op_id} registrada.")

    # 9. Guardar candidatos del dia
    log(sep)
    log("CANDIDATOS:")
    tickers_abiertos_hoy = {p["ticker"] for p in posiciones}
    tickers_que_abrimos  = {e["ticker"] for e in a_abrir}

    candidatos_log = []
    for ind in indicadores:
        t = ind["ticker"]
        s = ind.get("sector", "Unknown")
        if t in tickers_abiertos_hoy or t in tickers_bloqueados:
            continue
        tech_score, _ = calcular_score_tecnico(ind)
        if tech_score < SCORE_ENTRADA:
            continue
        candle_score = candle_scores.get(t, 0.0)
        # Guardamos TODOS los qualifying tecnicos (incluso los excluidos por candle)
        entro = t in tickers_que_abrimos
        motivo_skip = None
        if not entro:
            if candle_score < CANDLE_SCORE_MIN:
                motivo_skip = f"CANDLE_SCORE_{candle_score:.1f}"
            else:
                motivo_skip = f"CAPITAL_O_SLOTS_{s}"
        candidatos_log.append({
            "ticker":      t,
            "score":       float(tech_score),
            "entro":       entro,
            "motivo_skip": motivo_skip,
        })

    if candidatos_log:
        n_entro = sum(1 for c in candidatos_log if c["entro"])
        n_candle_filter = sum(
            1 for c in candidatos_log
            if (c.get("motivo_skip") or "").startswith("CANDLE_SCORE")
        )
        log(f"  {len(candidatos_log)} qualifying — {n_entro} abiertos | "
            f"{n_candle_filter} filtrados por candle | "
            f"{len(candidatos_log)-n_entro-n_candle_filter} por capital/slots")
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
    parser = argparse.ArgumentParser(description="FT Bot COMBO v1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
