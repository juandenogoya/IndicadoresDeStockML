"""
ft_bot_tech_sectorial_oiexit_v1.py
Forward-testing — Entrada TECH_SECTOR_v1 + salida por OI walls / corrida.

Estrategia FT_TECH_SECTOR_OIEXIT_v1 (#10):
    ENTRADA: IDENTICA a TECH_SECTOR_v1 (score tecnico >= 4.0, particion
             sectorial, sin opciones). La unica diferencia en la apertura es
             que el SL inicial sale del put wall de OI (fallback 2x ATR) y NO
             se fija un TP de 4x ATR.

    SALIDA (lo nuevo):
        SL_inicial = put wall de OI (ventana medio, zona -10%, liquidez
                     relativa al ticker); si no hay wall valido -> 2x ATR.
        R = precio_entrada - SL_inicial

        FASE 1 (entrada -> +3R):  escalones de proteccion
            pico >= +1R -> SL = entrada (breakeven)
            pico >= +2R -> SL = +1R
            SL tocado -> cerrar (SL_PROTECCION)

        TRANSICION (+3R): se libera el techo, piso de SL = +2R

        FASE 2 (post +3R):
            Backstop (siempre): Chandelier = max(close) - 2.5xATR
            Quorum 3 de 4:  cierre<SMA21 / candle_5d<=-2 /
                            PCR_VOL mayoria bajista / divergencia volumen

        Earnings: transversal (prioridad absoluta).

    El SL muta dia a dia (solo hacia arriba). El estado se DERIVA del pico de
    R alcanzado (MAX(close) desde la entrada); no se almacena.

Ref: docs/forward_testing/estrategias/TECH_SECTOR_OIEXIT_v1.md

Uso:
    python scripts/forward_testing/ft_bot_tech_sectorial_oiexit_v1.py
    python scripts/forward_testing/ft_bot_tech_sectorial_oiexit_v1.py --dry-run
"""

import sys
import os
import argparse
import statistics
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Entorno FT: forzar conexion a la DB LOCAL (ver scripts/forward_testing/ft_env.py)
from scripts.forward_testing.ft_env import configurar_entorno_local
configurar_entorno_local()

from sqlalchemy import text
from src.data.database import get_engine
from src.indicators.earnings_filter import tickers_a_cerrar_hoy, tickers_a_bloquear_entrada
from scripts.forward_testing.ft_scoring import calcular_score_tecnico, obtener_candle_score_5d
from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_precios_cierre_todos,
    abrir_operacion, cerrar_operacion, actualizar_stop_loss,
    registrar_metricas_diarias, registrar_candidatos_diarios,
    registrar_estado_posiciones, backfill_retornos_candidatos,
)

# ── Parametros de la estrategia ───────────────────────────────────────────────

NOMBRE_ESTRATEGIA = "FT_TECH_SECTOR_OIEXIT_v1"

SECTORES_ACTIVOS = [
    "Technology", "Consumer Cyclical", "Financial Services",
    "Industrials", "Basic Materials", "Healthcare",
    "Energy", "Communication Services", "Consumer Defensive",
]

CAPITAL_TOTAL   = 100_000.0
N_SECTORES      = len(SECTORES_ACTIVOS)
SECTOR_BUDGET   = round(CAPITAL_TOTAL / N_SECTORES, 2)
MAX_POS_SECTOR  = 5
POSITION_PCT    = 0.20
POSITION_SIZE   = round(SECTOR_BUDGET * POSITION_PCT, 2)

SCORE_ENTRADA   = 4.0
SCORE_MAXIMO    = 5.5

# Put wall (SL inicial)
WALL_VENTANA_MIN     = 15
WALL_VENTANA_MAX     = 45
WALL_ZONA_PCT        = 0.10     # zona de busqueda: hasta -10% del precio
WALL_LIQ_MULT_MED    = 3.0      # wall_oi >= 3 x mediana de la zona
WALL_LIQ_MIN_ABS     = 1000     # piso de OI absoluto
WALL_DIST_MIN_PCT    = 0.02     # wall a no menos de 2% del precio
SL_FALLBACK_ATR_MULT = 2.0

# Escalones de R y corrida
R_BREAKEVEN     = 1.0
R_LOCK          = 2.0
R_LIBERA        = 3.0
R_PISO_FASE2    = 2.0
CHANDELIER_ATR_MULT = 2.5

# Quorum de salida (Fase 2)
QUORUM_MIN          = 3
CANDLE_SALIDA_MAX   = -2.0
PCR_VOL_SALIDA_MAX  = 1         # pcr_score <= 1 (mayoria bajista)
DIV_DIST_MAX20_MIN  = -1.0      # precio dentro del 1% del max 20d
DIV_VOL_RATIO_MAX   = 0.8       # volumen < 0.8 del promedio 5d

# PCR_VOL por ventana
PCR_VOL_UMBRAL_ALCISTA = 1.0
MIN_VOL_POR_VENTANA    = 500


# ── Entrada: queries (identicas a TECH_SECTOR_v1) ─────────────────────────────

def obtener_indicadores_con_sector() -> list[dict]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT i.ticker, a.sector, p.close,
                   i.sma21, i.sma50, i.sma200,
                   i.rsi14, i.macd, i.macd_signal,
                   (i.macd - i.macd_signal) AS macd_hist,
                   i.atr14, i.adx, i.vol_relativo
            FROM (
                SELECT DISTINCT ON (ticker)
                       ticker, fecha, sma21, sma50, sma200,
                       rsi14, macd, macd_signal, atr14, adx, vol_relativo
                FROM indicadores_tecnicos
                ORDER BY ticker, fecha DESC
            ) i
            JOIN precios_diarios p ON p.ticker = i.ticker AND p.fecha = i.fecha
            JOIN activos a ON a.ticker = i.ticker
            WHERE a.activo = TRUE AND a.sector = ANY(:sectores)
            ORDER BY a.sector, i.ticker
        """), {"sectores": SECTORES_ACTIVOS}).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_posiciones_con_sector(estrategia_id: int) -> list[dict]:
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


# ── Put walls (SL inicial) ────────────────────────────────────────────────────

def obtener_put_walls(tickers: list[str]) -> dict[str, dict]:
    """
    Para cada ticker calcula el put wall en la ventana medio (15-45d):
    el strike con mayor put OI en la zona [precio*(1-zona), precio), validado
    por liquidez RELATIVA al ticker.

    Retorna {ticker: {wall_strike, wall_oi, precio, valido, dist_pct}}.
    """
    if not tickers:
        return {}

    engine = get_engine()
    with engine.connect() as conn:
        fila = conn.execute(text("""
            SELECT MAX(fecha_snapshot) AS f FROM opciones_snapshot
            WHERE ticker = ANY(:tickers)
        """), {"tickers": tickers}).fetchone()
        if not fila or not fila.f:
            return {}
        fecha_snap = fila.f

        rows = conn.execute(text("""
            SELECT ticker, strike,
                   SUM(COALESCE(open_interest, 0)) AS put_oi,
                   MAX(precio_subyacente) AS precio
            FROM opciones_snapshot
            WHERE ticker = ANY(:tickers)
              AND fecha_snapshot = :fecha_snap
              AND tipo = 'put'
              AND (vencimiento - :fecha_snap) BETWEEN :vmin AND :vmax
            GROUP BY ticker, strike
        """), {
            "tickers": tickers, "fecha_snap": fecha_snap,
            "vmin": WALL_VENTANA_MIN, "vmax": WALL_VENTANA_MAX,
        }).fetchall()

    por_ticker: dict[str, list] = {}
    precio_map: dict[str, float] = {}
    for r in rows:
        if r.precio:
            precio_map[r.ticker] = float(r.precio)
        por_ticker.setdefault(r.ticker, []).append(
            (float(r.strike), int(r.put_oi or 0))
        )

    resultado = {}
    for t in tickers:
        precio = precio_map.get(t)
        if not precio:
            resultado[t] = {"valido": False}
            continue
        zona_min = precio * (1 - WALL_ZONA_PCT)
        cands = [(s, oi) for (s, oi) in por_ticker.get(t, [])
                 if zona_min <= s < precio and oi > 0]
        if not cands:
            resultado[t] = {"valido": False, "precio": precio}
            continue

        mediana = statistics.median([oi for _, oi in cands])
        wall_strike, wall_oi = max(cands, key=lambda x: x[1])
        dist = (precio - wall_strike) / precio
        valido = (
            wall_oi >= WALL_LIQ_MULT_MED * mediana
            and wall_oi >= WALL_LIQ_MIN_ABS
            and dist >= WALL_DIST_MIN_PCT
        )
        resultado[t] = {
            "wall_strike": round(wall_strike, 4),
            "wall_oi":     wall_oi,
            "precio":      precio,
            "dist_pct":    round(dist * 100, 2),
            "valido":      valido,
        }
    return resultado


# ── PCR_VOL por ventana (para el quorum de salida) ────────────────────────────

def obtener_pcr_vol_por_ventanas(tickers: list[str]) -> dict[str, dict]:
    """PCR_VOL por ventana (corto/medio/largo). Retorna {ticker: {pcr_score, valido}}."""
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        fila = conn.execute(text("""
            SELECT MAX(fecha_snapshot) AS f FROM opciones_snapshot
            WHERE ticker = ANY(:tickers)
        """), {"tickers": tickers}).fetchone()
        if not fila or not fila.f:
            return {}
        fecha_snap = fila.f
        rows = conn.execute(text("""
            SELECT ticker, (vencimiento - :fecha_snap) AS dias,
                   SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END) AS call_vol,
                   SUM(CASE WHEN tipo='put'  THEN COALESCE(volumen,0) ELSE 0 END) AS put_vol
            FROM opciones_snapshot
            WHERE ticker = ANY(:tickers) AND fecha_snapshot = :fecha_snap
              AND (vencimiento - :fecha_snap) BETWEEN 1 AND 90
            GROUP BY ticker, (vencimiento - :fecha_snap)
        """), {"tickers": tickers, "fecha_snap": fecha_snap}).fetchall()

    acum: dict[str, dict] = {}
    for r in rows:
        d = int(r.dias)
        cv, pv = int(r.call_vol or 0), int(r.put_vol or 0)
        a = acum.setdefault(r.ticker, {"c_c": 0, "c_p": 0, "m_c": 0, "m_p": 0, "l_c": 0, "l_p": 0})
        if 1 <= d <= 14:
            a["c_c"] += cv; a["c_p"] += pv
        elif 15 <= d <= 45:
            a["m_c"] += cv; a["m_p"] += pv
        elif 46 <= d <= 90:
            a["l_c"] += cv; a["l_p"] += pv

    def _ver(cv, pv):
        if (cv + pv) < MIN_VOL_POR_VENTANA or cv == 0:
            return None
        return "A" if (pv / cv) < PCR_VOL_UMBRAL_ALCISTA else "B"

    resultado = {}
    for t in tickers:
        d = acum.get(t)
        if not d:
            resultado[t] = {"pcr_score": 0, "valido": False}
            continue
        vs = [_ver(d["c_c"], d["c_p"]), _ver(d["m_c"], d["m_p"]), _ver(d["l_c"], d["l_p"])]
        valido = all(v is not None for v in vs)
        resultado[t] = {
            "pcr_score": sum(1 for v in vs if v == "A"),
            "valido":    valido,
        }
    return resultado


# ── Contexto de salida (SMA21, ATR, divergencia) ──────────────────────────────

def obtener_contexto_salida(tickers: list[str]) -> dict[str, dict]:
    """sma21, atr14 (indicadores) + dist_max_20d, vol_ratio_5d (features)."""
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT i.ticker, i.sma21, i.atr14,
                   f.dist_max_20d, f.vol_ratio_5d
            FROM (
                SELECT DISTINCT ON (ticker) ticker, sma21, atr14
                FROM indicadores_tecnicos
                WHERE ticker = ANY(:tickers)
                ORDER BY ticker, fecha DESC
            ) i
            LEFT JOIN (
                SELECT DISTINCT ON (ticker) ticker, dist_max_20d, vol_ratio_5d
                FROM features_precio_accion
                WHERE ticker = ANY(:tickers)
                ORDER BY ticker, fecha DESC
            ) f ON f.ticker = i.ticker
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


def obtener_max_close_desde(posiciones: list[dict]) -> dict[str, float]:
    """Para cada posicion abierta: MAX(close) desde su fecha_entrada (pico)."""
    if not posiciones:
        return {}
    engine = get_engine()
    out = {}
    with engine.connect() as conn:
        for p in posiciones:
            row = conn.execute(text("""
                SELECT MAX(close) AS mx FROM precios_diarios
                WHERE ticker = :t AND fecha >= :fe
            """), {"t": p["ticker"], "fe": p["fecha_entrada"]}).fetchone()
            if row and row.mx is not None:
                out[p["ticker"]] = float(row.mx)
    return out


# ── Evaluacion de cierres (maquina de estados) ────────────────────────────────

def evaluar_cierres(posiciones, precios, earnings_map, contexto, pcr_map,
                    candle_scores, max_close):
    """
    Retorna (a_cerrar, sl_updates).
      a_cerrar:   lista de dicts con motivo de cierre.
      sl_updates: lista de (op_id, nuevo_sl) para mutar el SL (solo sube).
    """
    a_cerrar, sl_updates = [], []

    for pos in posiciones:
        ticker  = pos["ticker"]
        precio  = precios.get(ticker)
        if not precio:
            log(f"  [WARN] Sin precio para {ticker}, se omite.")
            continue

        # P0 — Earnings (prioridad absoluta)
        if ticker in earnings_map:
            a_cerrar.append({**pos, "precio_cierre": precio, "motivo": "EARNINGS_MANANA"})
            continue

        entrada = float(pos["precio_entrada"])
        det     = pos.get("detalle_entrada") or {}
        sl_ini  = det.get("sl_inicial")
        r       = det.get("r_value")
        sl_actual = float(pos["stop_loss"]) if pos.get("stop_loss") else None

        # Sin datos de R (no deberia pasar): usar SL actual como unica defensa
        if sl_ini is None or r is None or r <= 0:
            if sl_actual and precio <= sl_actual:
                a_cerrar.append({**pos, "precio_cierre": precio, "motivo": "SL_PROTECCION"})
            continue

        r       = float(r)
        ctx     = contexto.get(ticker, {})
        atr     = float(ctx["atr14"]) if ctx.get("atr14") else None
        pico    = max_close.get(ticker, precio)

        nivel_3r = entrada + R_LIBERA * r

        # ── FASE 1: proteccion por escalones ──────────────────────────────────
        if pico < nivel_3r:
            nuevo_sl = sl_ini
            if pico >= entrada + R_LOCK * r:
                nuevo_sl = entrada + 1.0 * r
            elif pico >= entrada + R_BREAKEVEN * r:
                nuevo_sl = entrada
            # solo sube
            if sl_actual is not None:
                nuevo_sl = max(nuevo_sl, sl_actual)
            sl_updates.append((pos["id"], round(nuevo_sl, 4)))

            if precio <= nuevo_sl:
                a_cerrar.append({**pos, "precio_cierre": precio, "motivo": "SL_PROTECCION"})
            continue

        # ── FASE 2: corrida ───────────────────────────────────────────────────
        piso      = entrada + R_PISO_FASE2 * r
        chandelier = pico - CHANDELIER_ATR_MULT * atr if atr else piso
        hard_sl   = max(piso, chandelier)
        if sl_actual is not None:
            hard_sl = max(hard_sl, sl_actual)
        sl_updates.append((pos["id"], round(hard_sl, 4)))

        if precio <= hard_sl:
            a_cerrar.append({**pos, "precio_cierre": precio, "motivo": "BACKSTOP_CHANDELIER"})
            continue

        # Quorum 3 de 4 (señal no evaluable = no dispara)
        senales = 0
        sma21 = ctx.get("sma21")
        if sma21 is not None and precio < float(sma21):
            senales += 1
        cs = candle_scores.get(ticker)
        if cs is not None and float(cs) <= CANDLE_SALIDA_MAX:
            senales += 1
        pcr = pcr_map.get(ticker, {})
        if pcr.get("valido") and pcr.get("pcr_score", 99) <= PCR_VOL_SALIDA_MAX:
            senales += 1
        dmax = ctx.get("dist_max_20d")
        vratio = ctx.get("vol_ratio_5d")
        if (dmax is not None and vratio is not None
                and float(dmax) >= DIV_DIST_MAX20_MIN
                and float(vratio) < DIV_VOL_RATIO_MAX):
            senales += 1

        if senales >= QUORUM_MIN:
            a_cerrar.append({
                **pos, "precio_cierre": precio,
                "motivo": f"AGOTAMIENTO_QUORUM_{senales}de4",
            })

    return a_cerrar, sl_updates


# ── Evaluacion de entradas (identica a v1, SL = put wall) ─────────────────────

def evaluar_entradas_sectorial(posiciones, indicadores, precios, put_walls,
                               tickers_bloqueados, tickers_cerrados):
    tickers_abiertos = {p["ticker"] for p in posiciones}
    excluidos        = tickers_bloqueados | tickers_abiertos | tickers_cerrados

    sector_deployed, sector_n_open = {}, {}
    for pos in posiciones:
        s = pos.get("sector", "Unknown")
        sector_deployed[s] = sector_deployed.get(s, 0.0) + float(pos["capital_entrada"])
        sector_n_open[s]   = sector_n_open.get(s, 0) + 1

    candidatos_por_sector: dict[str, list] = {s: [] for s in SECTORES_ACTIVOS}
    for ind in indicadores:
        t = ind["ticker"]
        s = ind.get("sector", "Unknown")
        if t in excluidos or s not in SECTORES_ACTIVOS:
            continue
        score, detalle = calcular_score_tecnico(ind)
        if score < SCORE_ENTRADA:
            continue
        candidatos_por_sector[s].append(
            {"ticker": t, "sector": s, "score": score, "detalle": detalle, "ind": ind}
        )
    for s in candidatos_por_sector:
        candidatos_por_sector[s].sort(key=lambda x: x["score"], reverse=True)

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

        log(f"  [{sector}] {len(candidatos)} candidatos | slots={slots} | "
            f"disponible=${available:,.2f}")
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
                log(f"    [SKIP] {ticker}: precio ${precio:.2f} > position_size.")
                continue
            capital_trade = round(precio * qty, 2)
            if capital_trade > avail_local:
                log(f"    [SKIP] {ticker}: trade ${capital_trade:,.2f} > "
                    f"disponible ${avail_local:,.2f}.")
                continue

            # SL inicial: put wall si es valido y queda debajo del precio; si no, ATR
            wall = put_walls.get(ticker, {})
            if wall.get("valido") and wall.get("wall_strike", 0) < precio:
                sl_ini    = float(wall["wall_strike"])
                sl_source = "put_wall"
            else:
                sl_ini    = round(precio - SL_FALLBACK_ATR_MULT * atr, 4)
                sl_source = "atr"

            r_value = round(precio - sl_ini, 4)
            if r_value <= 0:
                sl_ini    = round(precio - SL_FALLBACK_ATR_MULT * atr, 4)
                sl_source = "atr"
                r_value   = round(precio - sl_ini, 4)

            tp_ref = round(precio + R_LIBERA * r_value, 4)   # +3R de referencia

            a_abrir.append({
                "ticker": ticker, "sector": sector, "precio": precio,
                "qty": qty, "capital": capital_trade,
                "sl_ini": sl_ini, "r_value": r_value, "sl_source": sl_source,
                "tp_ref": tp_ref, "score": c["score"], "detalle": c["detalle"],
                "atr": round(atr, 4), "wall": wall,
            })
            avail_local -= capital_trade
            slots       -= 1

    return a_abrir


# ── Runner ────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 60

    log(sep)
    log(f"FT Bot Tech Sectorial OIEXIT v1 | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(f"  {N_SECTORES} sectores x ${SECTOR_BUDGET:,.2f} | Position ${POSITION_SIZE:,.2f}")
    log(f"  SL=put wall (medio, -{WALL_ZONA_PCT:.0%}) | corrida +3R + Chandelier {CHANDELIER_ATR_MULT}xATR + quorum 3/4")
    log(sep)

    estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)
    if not estrategia:
        log(f"[ERROR] Estrategia '{NOMBRE_ESTRATEGIA}' no encontrada o inactiva.")
        log("Hint: python scripts/forward_testing/ft_setup_estrategias.py")
        return
    eid = estrategia["id"]
    log(f"Estrategia id={eid} | capital={estrategia['capital_actual']:,.2f} | "
        f"cash={estrategia['cash_disponible']:,.2f}")

    precios     = obtener_precios_cierre_todos()
    indicadores = obtener_indicadores_con_sector()
    posiciones  = obtener_posiciones_con_sector(eid)
    log(f"Precios: {len(precios)} | Indicadores: {len(indicadores)} | "
        f"Posiciones abiertas: {len(posiciones)}")

    # ── Datos para la salida ──────────────────────────────────────────────────
    tickers_pos     = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    contexto        = obtener_contexto_salida(tickers_pos)
    pcr_map_sal     = obtener_pcr_vol_por_ventanas(tickers_pos)
    candle_scores   = obtener_candle_score_5d()
    max_close       = obtener_max_close_desde(posiciones)

    # ── Cierres ───────────────────────────────────────────────────────────────
    log(sep)
    log("CIERRES:")
    a_cerrar, sl_updates = evaluar_cierres(
        posiciones, precios, earnings_cierre, contexto, pcr_map_sal,
        candle_scores, max_close,
    )

    # Mutar SL (trailing) de las posiciones que siguen abiertas
    tickers_a_cerrar = {c["ticker"] for c in a_cerrar}
    n_sl = 0
    for op_id, nuevo_sl in sl_updates:
        pos = next((p for p in posiciones if p["id"] == op_id), None)
        if not pos or pos["ticker"] in tickers_a_cerrar:
            continue
        sl_prev = float(pos["stop_loss"]) if pos.get("stop_loss") else None
        if sl_prev is None or nuevo_sl > sl_prev + 1e-9:
            if not dry_run:
                actualizar_stop_loss(op_id, nuevo_sl)
            n_sl += 1
    if n_sl:
        log(f"  SL actualizado (trailing) en {n_sl} posiciones.")

    tickers_cerrados = set()
    if not a_cerrar:
        log("  Sin posiciones a cerrar.")
    else:
        for c in a_cerrar:
            ticker, precio, motivo = c["ticker"], c["precio_cierre"], c["motivo"]
            pnl_est = round((precio - float(c["precio_entrada"])) * int(c["cantidad"]), 2)
            log(f"  CERRAR {ticker} [{c.get('sector','-')}] | precio={precio:.2f} | "
                f"pnl_est={pnl_est:+.2f} | motivo={motivo}")
            tickers_cerrados.add(ticker)
            if not dry_run:
                res = cerrar_operacion(c["id"], eid, precio, motivo, hoy)
                log(f"    -> pnl={res.get('pnl', 0):+.2f} ({res.get('pnl_pct', 0):+.2f}%)")

    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_con_sector(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # ── Entradas ──────────────────────────────────────────────────────────────
    tickers_cand       = [i["ticker"] for i in indicadores]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_cand) if tickers_cand else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {len(tickers_bloqueados)} tickers")

    # Put walls solo para los candidatos qualifying (ahorra queries)
    qualifying = []
    for ind in indicadores:
        if ind["ticker"] in (tickers_bloqueados | {p["ticker"] for p in posiciones} | tickers_cerrados):
            continue
        score, _ = calcular_score_tecnico(ind)
        if score >= SCORE_ENTRADA:
            qualifying.append(ind["ticker"])
    put_walls = obtener_put_walls(qualifying)

    log(sep)
    log("ENTRADAS POR SECTOR:")
    a_abrir = evaluar_entradas_sectorial(
        posiciones, indicadores, precios, put_walls,
        tickers_bloqueados, tickers_cerrados,
    )

    if not a_abrir:
        log("  Sin entradas.")
    else:
        for e in a_abrir:
            log(f"  ABRIR {e['ticker']} [{e['sector']}] | precio={e['precio']:.2f} | "
                f"qty={e['qty']} | score={e['score']:.1f}/{SCORE_MAXIMO} | "
                f"SL={e['sl_ini']:.2f} ({e['sl_source']}) | R={e['r_value']:.2f} | "
                f"TP_ref(+3R)={e['tp_ref']:.2f}")
            if not dry_run:
                detalle = {
                    **e["detalle"],
                    "sector":        e["sector"],
                    "score":         e["score"],
                    "score_maximo":  SCORE_MAXIMO,
                    "atr":           e["atr"],
                    "sl_inicial":    e["sl_ini"],
                    "r_value":       e["r_value"],
                    "sl_source":     e["sl_source"],
                    "wall_strike":   e["wall"].get("wall_strike"),
                    "wall_oi":       e["wall"].get("wall_oi"),
                    "tp_ref_3r":     e["tp_ref"],
                    "sector_budget": SECTOR_BUDGET,
                    "position_size": POSITION_SIZE,
                }
                op_id = abrir_operacion(
                    estrategia_id=eid, ticker=e["ticker"], fecha=hoy,
                    precio=e["precio"], cantidad=e["qty"],
                    stop_loss=e["sl_ini"], take_profit=e["tp_ref"],
                    score=e["score"], detalle=detalle,
                )
                if op_id:
                    log(f"    -> operacion id={op_id} registrada.")

    # ── Candidatos del dia ────────────────────────────────────────────────────
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
        score, _ = calcular_score_tecnico(ind)
        if score < SCORE_ENTRADA:
            continue
        entro = t in tickers_que_abrimos
        candidatos_log.append({
            "ticker": t, "score": float(score), "entro": entro,
            "motivo_skip": None if entro else f"CAPITAL_O_SLOTS_{s}",
            "precio_apertura": precios.get(t),
        })
    if candidatos_log:
        n_entro = sum(1 for c in candidatos_log if c["entro"])
        log(f"  {len(candidatos_log)} qualifying - {n_entro} abiertos, "
            f"{len(candidatos_log)-n_entro} oportunidades")
        if not dry_run:
            registrar_candidatos_diarios(eid, hoy, candidatos_log)
    else:
        log("  Sin candidatos qualifying hoy.")

    # ── Metricas y observacion ────────────────────────────────────────────────
    log(sep)
    if not dry_run:
        registrar_metricas_diarias(eid, hoy)
        registrar_estado_posiciones(eid, hoy, precios)
        backfill_retornos_candidatos(eid, hoy, precios)
    else:
        log("[DRY RUN] Metricas, posiciones y retornos no registrados.")

    log("Completado.")
    log(sep)


def main():
    parser = argparse.ArgumentParser(
        description="FT Bot Tech Sectorial + salida OI walls / corrida"
    )
    parser.add_argument("--dry-run", action="store_true", help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
