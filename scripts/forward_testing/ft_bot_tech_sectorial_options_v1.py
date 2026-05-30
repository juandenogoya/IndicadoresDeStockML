"""
ft_bot_tech_sectorial_options_v1.py
Forward-testing — AT Tecnico Sectorial con confirmacion de opciones (PCR_OI).

Estrategia FT_TECH_SECTOR_OPTIONS_v1:
    Combina TECH_SECTOR_v1 con una capa de confirmacion basada en el
    posicionamiento real del mercado de opciones (PCR_OI por ventana).

    ENTRADA — dos gates en serie:
        Gate 1: tech_score >= 4.0 (identico a v1)
        Gate 2: pcr_score >= 2/3 ventanas con PCR_OI < 1.0 (alcistas)
                Si el ticker no tiene liquidez suficiente en alguna ventana
                -> descartado (motivo: SIN_LIQUIDEZ_OPCIONES)

    VENTANAS de vencimiento (dias al vencimiento desde fecha_snapshot):
        Corto:  1 – 14 dias   (weeklies, posicionamiento tactico)
        Medio:  15 – 45 dias  (mensuales, mayor OI, mas confiable)
        Largo:  46 – 90 dias  (institucional, medio plazo)

    PCR_OI por ventana = sum(put_OI) / sum(call_OI) en esa ventana
        < 1.0  -> Alcista (A)
        >= 1.0 -> Bajista  (B)

    pcr_score = cantidad de ventanas Alcistas (0-3)
    Minimo de liquidez por ventana: sum(call_OI + put_OI) >= 500

    RANKING dentro del sector:
        Primario:   tech_score DESC
        Secundario: pcr_score DESC (tiebreaker)

    SALIDA:
        P0. Earnings manana                              -> EARNINGS_MANANA
        P1. precio <= SL (2x ATR)                       -> STOP_LOSS_ATR
        P2. precio >= TP (4x ATR)                       -> TAKE_PROFIT_ATR
        P3. tech_score <= 3.5 AND pcr_score = 0         -> SCORE_DEGRADADO_OPCIONES
        P3b. tech_score <= 3.5 AND pcr_score >= 1       -> RETENCION_OPCIONES (no cierra)

Uso:
    python scripts/forward_testing/ft_bot_tech_sectorial_options_v1.py
    python scripts/forward_testing/ft_bot_tech_sectorial_options_v1.py --dry-run
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
from scripts.forward_testing.ft_scoring import calcular_score_tecnico
from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_precios_cierre_todos,
    abrir_operacion, cerrar_operacion,
    registrar_metricas_diarias, registrar_candidatos_diarios,
    registrar_estado_posiciones, backfill_retornos_candidatos,
)

# ── Parametros de la estrategia ───────────────────────────────────────────────

NOMBRE_ESTRATEGIA = "FT_TECH_SECTOR_OPTIONS_v1"

SECTORES_ACTIVOS = [
    "Technology", "Consumer Cyclical", "Financial Services",
    "Industrials", "Basic Materials", "Healthcare",
    "Energy", "Communication Services", "Consumer Defensive",
]

CAPITAL_TOTAL     = 100_000.0
N_SECTORES        = len(SECTORES_ACTIVOS)
SECTOR_BUDGET     = round(CAPITAL_TOTAL / N_SECTORES, 2)   # 11.111,11
MAX_POS_SECTOR    = 5
POSITION_SIZE     = round(SECTOR_BUDGET / MAX_POS_SECTOR, 2)  # 2.222,22

SCORE_ENTRADA     = 4.0
SCORE_SALIDA      = 3.5
SCORE_MAXIMO      = 5.5
ATR_MULT_SL       = 2.0
ATR_MULT_TP       = 4.0

# Ventanas de vencimiento (dias al vencimiento)
VENTANA_CORTO_MIN  = 1
VENTANA_CORTO_MAX  = 14
VENTANA_MEDIO_MIN  = 15
VENTANA_MEDIO_MAX  = 45
VENTANA_LARGO_MIN  = 46
VENTANA_LARGO_MAX  = 90

# Parametros de opciones
PCR_OI_UMBRAL_ALCISTA  = 1.0   # < 1.0 = Alcista, >= 1.0 = Bajista
MIN_OI_POR_VENTANA     = 500   # OI minimo (call+put) para ventana valida
PCR_SCORE_ENTRADA_MIN  = 2     # al menos 2/3 ventanas Alcistas para entrar
PCR_SCORE_SALIDA_MAX   = 0     # 0/3 = todas Bajistas -> cerrar (con tech degradado)
PCR_SCORE_RETENCION    = 1     # >= 1 ventana Alcista -> retener


# ── Opciones: PCR_OI por ventana ─────────────────────────────────────────────

def obtener_pcr_oi_por_ventanas(tickers: list[str]) -> dict[str, dict]:
    """
    Calcula el PCR_OI para cada ventana de vencimiento de cada ticker.
    Usa el ultimo fecha_snapshot disponible en opciones_snapshot.

    Para cada ventana (corto/medio/largo):
        PCR_OI = sum(put_OI) / sum(call_OI)   para los contratos en ese rango de dias

    Veredicto por ventana:
        PCR_OI < 1.0   -> 'A' (Alcista)
        PCR_OI >= 1.0  -> 'B' (Bajista)
        OI < MIN_OI    -> None (sin liquidez suficiente)

    Retorna:
        {
          ticker: {
            'corto':   'A' | 'B' | None,
            'medio':   'A' | 'B' | None,
            'largo':   'A' | 'B' | None,
            'pcr_score': int (0-3, cantidad de ventanas Alcistas),
            'valido':    bool (False si alguna ventana no tiene liquidez),
            'detalle': { 'pcr_corto': float|None, 'pcr_medio': float|None, ... }
          }
        }
    """
    if not tickers:
        return {}

    engine = get_engine()
    with engine.connect() as conn:
        # Ultimo snapshot disponible (una sola fecha para todos los tickers)
        row_fecha = conn.execute(text("""
            SELECT MAX(fecha_snapshot) AS max_fecha
            FROM opciones_snapshot
            WHERE ticker = ANY(:tickers)
        """), {"tickers": tickers}).fetchone()

        if not row_fecha or not row_fecha.max_fecha:
            log("  [WARN] Sin datos de opciones_snapshot para los tickers dados.")
            return {}

        fecha_snap = row_fecha.max_fecha

        # OI por ventana para todos los tickers en una sola query
        rows = conn.execute(text("""
            SELECT
                ticker,
                (vencimiento - :fecha_snap)                        AS dias_al_venc,
                SUM(CASE WHEN tipo = 'call' THEN COALESCE(open_interest, 0) ELSE 0 END) AS call_oi,
                SUM(CASE WHEN tipo = 'put'  THEN COALESCE(open_interest, 0) ELSE 0 END) AS put_oi
            FROM opciones_snapshot
            WHERE ticker = ANY(:tickers)
              AND fecha_snapshot = :fecha_snap
              AND (vencimiento - :fecha_snap) BETWEEN :vmin AND :vmax
            GROUP BY ticker, (vencimiento - :fecha_snap)
        """), {
            "tickers":   tickers,
            "fecha_snap": fecha_snap,
            "vmin":       VENTANA_CORTO_MIN,
            "vmax":       VENTANA_LARGO_MAX,
        }).fetchall()

    # Acumular OI por ticker y ventana
    acum: dict[str, dict] = {}
    for r in rows:
        t    = r.ticker
        dias = int(r.dias_al_venc)
        co   = int(r.call_oi or 0)
        po   = int(r.put_oi  or 0)

        if t not in acum:
            acum[t] = {
                "corto_call": 0, "corto_put": 0,
                "medio_call": 0, "medio_put": 0,
                "largo_call": 0, "largo_put": 0,
            }

        if VENTANA_CORTO_MIN <= dias <= VENTANA_CORTO_MAX:
            acum[t]["corto_call"] += co
            acum[t]["corto_put"]  += po
        elif VENTANA_MEDIO_MIN <= dias <= VENTANA_MEDIO_MAX:
            acum[t]["medio_call"] += co
            acum[t]["medio_put"]  += po
        elif VENTANA_LARGO_MIN <= dias <= VENTANA_LARGO_MAX:
            acum[t]["largo_call"] += co
            acum[t]["largo_put"]  += po

    # Calcular veredicto por ventana
    resultado = {}
    for t in tickers:
        datos = acum.get(t)
        if not datos:
            # Sin datos de opciones para este ticker
            resultado[t] = {
                "corto": None, "medio": None, "largo": None,
                "pcr_score": 0, "valido": False,
                "detalle": {"pcr_corto": None, "pcr_medio": None, "pcr_largo": None},
            }
            continue

        def _veredicto(call_oi, put_oi):
            total = call_oi + put_oi
            if total < MIN_OI_POR_VENTANA or call_oi == 0:
                return None, None   # sin liquidez
            pcr = round(put_oi / call_oi, 4)
            v   = "A" if pcr < PCR_OI_UMBRAL_ALCISTA else "B"
            return v, pcr

        v_corto, pcr_corto = _veredicto(datos["corto_call"], datos["corto_put"])
        v_medio, pcr_medio = _veredicto(datos["medio_call"], datos["medio_put"])
        v_largo, pcr_largo = _veredicto(datos["largo_call"], datos["largo_put"])

        # Si CUALQUIER ventana no tiene liquidez -> ticker invalido
        valido = all(v is not None for v in (v_corto, v_medio, v_largo))

        pcr_score = sum(1 for v in (v_corto, v_medio, v_largo) if v == "A")

        resultado[t] = {
            "corto":     v_corto,
            "medio":     v_medio,
            "largo":     v_largo,
            "pcr_score": pcr_score,
            "valido":    valido,
            "detalle": {
                "pcr_corto": pcr_corto,
                "pcr_medio": pcr_medio,
                "pcr_largo": pcr_largo,
                "fecha_snap": str(fecha_snap),
            },
        }

    return resultado


# ── Queries de indicadores/posiciones ────────────────────────────────────────

def obtener_posiciones_con_sector(estrategia_id: int) -> list[dict]:
    """Posiciones abiertas con sector del ticker (JOIN activos)."""
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
    """Indicadores tecnicos del ultimo dia para los 9 sectores activos."""
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
                       rsi14, macd, macd_signal, atr14
                FROM indicadores_tecnicos
                ORDER BY ticker, fecha DESC
            ) i
            JOIN precios_diarios p ON p.ticker = i.ticker AND p.fecha = i.fecha
            JOIN activos a         ON a.ticker = i.ticker
            WHERE a.activo = TRUE AND a.sector = ANY(:sectores)
            ORDER BY a.sector, i.ticker
        """), {"sectores": SECTORES_ACTIVOS}).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_estado_tecnico_tickers(tickers: list[str]) -> dict[str, dict]:
    """
    Indicadores actuales para posiciones abiertas (evaluar score de salida).
    Incluye close (via JOIN precios_diarios) para que calcular_score_tecnico
    pueda evaluar la Capa 1 (close > sma200). Sin close el score siempre
    devolvia 0.0 y la decision quedaba dominada por el filtro PCR.
    """
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (i.ticker)
                   i.ticker, p.close,
                   i.sma21, i.sma50, i.sma200,
                   i.rsi14, i.macd, i.macd_signal,
                   (i.macd - i.macd_signal) AS macd_hist,
                   i.atr14
            FROM indicadores_tecnicos i
            JOIN precios_diarios p ON p.ticker = i.ticker AND p.fecha = i.fecha
            WHERE i.ticker = ANY(:tickers)
            ORDER BY i.ticker, i.fecha DESC
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


# ── Evaluacion de cierres ─────────────────────────────────────────────────────

def evaluar_cierres(
    posiciones:      list[dict],
    precios:         dict[str, float],
    earnings_map:    dict[str, date],
    indicadores_map: dict[str, dict],
    pcr_map:         dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """
    Evalua que posiciones cerrar segun las prioridades definidas.
    Retorna (a_cerrar, tickers_retenidos_por_opciones).

    P0. Earnings manana           -> EARNINGS_MANANA
    P1. precio <= SL              -> STOP_LOSS_ATR
    P2. precio >= TP              -> TAKE_PROFIT_ATR
    P3. score <= 3.5 AND pcr = 0  -> SCORE_DEGRADADO_OPCIONES
    P3b. score <= 3.5 AND pcr >= 1 -> RETENCION_OPCIONES (no cierra)
    """
    a_cerrar   = []
    retenidos  = []

    for pos in posiciones:
        ticker    = pos["ticker"]
        precio    = precios.get(ticker)
        sl        = float(pos["stop_loss"])  if pos.get("stop_loss")  else None
        tp        = float(pos["take_profit"]) if pos.get("take_profit") else None
        ind       = indicadores_map.get(ticker)
        pcr_info  = pcr_map.get(ticker, {})

        if not precio:
            log(f"  [WARN] Sin precio para {ticker}, se omite.")
            continue

        motivo = None

        # P0: Earnings
        if ticker in earnings_map:
            motivo = "EARNINGS_MANANA"

        # P1: Stop loss
        elif sl and precio <= sl:
            motivo = "STOP_LOSS_ATR"

        # P2: Take profit
        elif tp and precio >= tp:
            motivo = "TAKE_PROFIT_ATR"

        # P3 / P3b: Score degradado + confirmacion de opciones
        elif ind:
            score_actual, _ = calcular_score_tecnico(ind)
            if score_actual <= SCORE_SALIDA:
                pcr_score = pcr_info.get("pcr_score", 0)
                pcr_valido = pcr_info.get("valido", False)

                if not pcr_valido or pcr_score <= PCR_SCORE_SALIDA_MAX:
                    # Sin datos de opciones O todas las ventanas bajistas -> cerrar
                    motivo = "SCORE_DEGRADADO_OPCIONES"
                else:
                    # Al menos 1 ventana de opciones es Alcista -> retener
                    v_corto = pcr_info.get("corto", "?")
                    v_medio = pcr_info.get("medio", "?")
                    v_largo = pcr_info.get("largo", "?")
                    log(f"  [RETENCION_OPCIONES] {ticker}: "
                        f"score={score_actual:.1f} pcr_score={pcr_score}/3 "
                        f"({v_corto}/{v_medio}/{v_largo})")
                    retenidos.append(ticker)

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio,
                "motivo":        motivo,
            })

    return a_cerrar, retenidos


# ── Evaluacion de entradas por sector ─────────────────────────────────────────

def evaluar_entradas_sectorial(
    posiciones:         list[dict],
    indicadores:        list[dict],
    precios:            dict[str, float],
    pcr_map:            dict[str, dict],
    tickers_bloqueados: set,
    tickers_cerrados:   set,
) -> tuple[list[dict], list[dict]]:
    """
    Para cada sector con slots disponibles:
        Gate 1: tech_score >= SCORE_ENTRADA
        Gate 2: pcr_score >= PCR_SCORE_ENTRADA_MIN Y ticker valido en opciones
        Ranking: tech_score DESC, pcr_score DESC

    Retorna (a_abrir, candidatos_log) donde candidatos_log incluye
    los rechazados con su motivo_skip para registro en ft_candidatos_diarios.
    """
    tickers_abiertos = {p["ticker"] for p in posiciones}
    excluidos        = tickers_bloqueados | tickers_abiertos | tickers_cerrados

    sector_deployed = {}
    sector_n_open   = {}
    for pos in posiciones:
        s = pos.get("sector", "Unknown")
        sector_deployed[s] = sector_deployed.get(s, 0.0) + float(pos["capital_entrada"])
        sector_n_open[s]   = sector_n_open.get(s, 0) + 1

    # Evaluar todos los candidatos tecnicamente qualifying
    candidatos_por_sector: dict[str, list] = {s: [] for s in SECTORES_ACTIVOS}
    candidatos_log = []

    for ind in indicadores:
        t = ind["ticker"]
        s = ind.get("sector", "Unknown")
        if t in excluidos or s not in SECTORES_ACTIVOS:
            continue

        score, detalle = calcular_score_tecnico(ind)
        if score < SCORE_ENTRADA:
            continue

        # Gate 2: opciones
        pcr_info  = pcr_map.get(t, {})
        pcr_valido = pcr_info.get("valido", False)
        pcr_score  = pcr_info.get("pcr_score", 0)

        if not pcr_valido:
            candidatos_log.append({
                "ticker":          t,
                "score":           float(score),
                "entro":           False,
                "motivo_skip":     "SIN_LIQUIDEZ_OPCIONES",
                "precio_apertura": precios.get(t),
            })
            continue

        if pcr_score < PCR_SCORE_ENTRADA_MIN:
            v_corto = pcr_info.get("corto", "?")
            v_medio = pcr_info.get("medio", "?")
            v_largo = pcr_info.get("largo", "?")
            candidatos_log.append({
                "ticker":          t,
                "score":           float(score),
                "entro":           False,
                "motivo_skip":     f"PCR_CONSENSO_{v_corto}{v_medio}{v_largo}",
                "precio_apertura": precios.get(t),
            })
            continue

        # Pasa ambos gates -> candidato qualifying
        candidatos_por_sector[s].append({
            "ticker":    t,
            "sector":    s,
            "score":     score,
            "pcr_score": pcr_score,
            "detalle":   detalle,
            "pcr_info":  pcr_info,
            "ind":       ind,
        })

    # Ordenar por sector: tech_score DESC, pcr_score DESC
    for s in candidatos_por_sector:
        candidatos_por_sector[s].sort(
            key=lambda x: (x["score"], x["pcr_score"]), reverse=True
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
            log(f"  [{sector}] Lleno ({MAX_POS_SECTOR}/{MAX_POS_SECTOR}).")
            continue
        if available <= 0:
            log(f"  [{sector}] Sin capital disponible.")
            continue

        log(f"  [{sector}] {len(candidatos)} qualifying | slots={slots} | "
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

            sl = round(precio - ATR_MULT_SL * atr, 4)
            tp = round(precio + ATR_MULT_TP * atr, 4)

            pcr_info = c["pcr_info"]
            a_abrir.append({
                "ticker":    ticker,
                "sector":    sector,
                "precio":    precio,
                "qty":       qty,
                "capital":   capital_trade,
                "sl":        sl,
                "tp":        tp,
                "score":     c["score"],
                "pcr_score": c["pcr_score"],
                "pcr_info":  pcr_info,
                "detalle":   c["detalle"],
                "atr":       round(atr, 4),
            })

            candidatos_log.append({
                "ticker":          ticker,
                "score":           float(c["score"]),
                "entro":           True,
                "motivo_skip":     None,
                "precio_apertura": precio,
            })

            avail_local -= capital_trade
            slots       -= 1

        # Los candidatos que no entraron por slots/capital
        tickers_que_abrimos = {e["ticker"] for e in a_abrir}
        for c in candidatos:
            if c["ticker"] not in tickers_que_abrimos:
                if not any(cl["ticker"] == c["ticker"] for cl in candidatos_log):
                    candidatos_log.append({
                        "ticker":          c["ticker"],
                        "score":           float(c["score"]),
                        "entro":           False,
                        "motivo_skip":     f"CAPITAL_O_SLOTS_{sector}",
                        "precio_apertura": precios.get(c["ticker"]),
                    })

    return a_abrir, candidatos_log


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 60

    log(sep)
    log(f"FT Bot Tech Sectorial Options v1 | {hoy} "
        f"{'[DRY RUN]' if dry_run else ''}")
    log(f"  {N_SECTORES} sectores x ${SECTOR_BUDGET:,.2f} | "
        f"Position size: ${POSITION_SIZE:,.2f} | Max {MAX_POS_SECTOR} pos/sector")
    log(f"  Gate 2: pcr_score >= {PCR_SCORE_ENTRADA_MIN}/3 | "
        f"Min OI/ventana: {MIN_OI_POR_VENTANA}")
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

    # 2. Datos base
    precios     = obtener_precios_cierre_todos()
    indicadores = obtener_indicadores_con_sector()
    posiciones  = obtener_posiciones_con_sector(eid)

    log(f"Precios: {len(precios)} tickers | "
        f"Indicadores: {len(indicadores)} | "
        f"Posiciones abiertas: {len(posiciones)}")

    # 3. PCR_OI por ventana para TODOS los tickers del universo
    # (necesario tanto para cierres como para entradas)
    log(sep)
    log("CALCULANDO PCR_OI POR VENTANAS...")
    todos_tickers = list({i["ticker"] for i in indicadores})
    pcr_map = obtener_pcr_oi_por_ventanas(todos_tickers)

    n_validos  = sum(1 for v in pcr_map.values() if v.get("valido"))
    n_invalidos = len(pcr_map) - n_validos
    log(f"  PCR calculado: {n_validos} validos | {n_invalidos} sin liquidez suficiente")

    # 4. Earnings
    tickers_pos     = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar hoy): {list(earnings_cierre.keys())}")

    # 5. Indicadores actuales para posiciones abiertas
    indicadores_map = obtener_estado_tecnico_tickers(tickers_pos)

    # 6. Evaluar cierres
    log(sep)
    log("CIERRES:")
    a_cerrar, retenidos = evaluar_cierres(
        posiciones, precios, earnings_cierre, indicadores_map, pcr_map
    )
    tickers_cerrados = set()

    if not a_cerrar:
        log("  Sin posiciones a cerrar.")
    else:
        for c in a_cerrar:
            ticker = c["ticker"]
            precio = c["precio_cierre"]
            motivo = c["motivo"]
            pnl_est = round(
                (precio - float(c["precio_entrada"])) * int(c["cantidad"]), 2
            )
            pcr_inf = pcr_map.get(ticker, {})
            log(f"  CERRAR {ticker} [{c.get('sector','-')}] | "
                f"precio={precio:.2f} | pnl_est={pnl_est:+.2f} | "
                f"motivo={motivo} | pcr={pcr_inf.get('pcr_score','?')}/3 "
                f"({pcr_inf.get('corto','?')}/{pcr_inf.get('medio','?')}/{pcr_inf.get('largo','?')})")
            tickers_cerrados.add(ticker)

            if not dry_run:
                resultado = cerrar_operacion(c["id"], eid, precio, motivo, hoy)
                log(f"    -> pnl={resultado.get('pnl', 0):+.2f} "
                    f"({resultado.get('pnl_pct', 0):+.2f}%)")

    if retenidos:
        log(f"  Retenciones por opciones: {retenidos}")

    # 7. Recargar tras cierres
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_con_sector(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 8. Bloqueos por earnings proximos
    tickers_cand = [i["ticker"] for i in indicadores]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_cand) if tickers_cand else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {len(tickers_bloqueados)} tickers")

    # 9. Evaluar entradas
    log(sep)
    log("ENTRADAS POR SECTOR:")
    a_abrir, candidatos_log = evaluar_entradas_sectorial(
        posiciones, indicadores, precios, pcr_map,
        tickers_bloqueados, tickers_cerrados,
    )

    if not a_abrir:
        log("  Sin entradas.")
    else:
        for e in a_abrir:
            pcr_d = e["pcr_info"].get("detalle", {})
            log(f"  ABRIR {e['ticker']} [{e['sector']}] | "
                f"precio={e['precio']:.2f} | qty={e['qty']} | "
                f"capital={e['capital']:,.2f} | "
                f"tech={e['score']:.1f}/{SCORE_MAXIMO} | "
                f"pcr={e['pcr_score']}/3 "
                f"({e['pcr_info'].get('corto','?')}/"
                f"{e['pcr_info'].get('medio','?')}/"
                f"{e['pcr_info'].get('largo','?')}) | "
                f"sl={e['sl']:.2f} | tp={e['tp']:.2f}")

            if not dry_run:
                detalle = {
                    **e["detalle"],
                    "sector":          e["sector"],
                    "score":           e["score"],
                    "score_maximo":    SCORE_MAXIMO,
                    "atr":             e["atr"],
                    "sl_mult_atr":     ATR_MULT_SL,
                    "tp_mult_atr":     ATR_MULT_TP,
                    "pcr_score":       e["pcr_score"],
                    "pcr_corto":       e["pcr_info"].get("corto"),
                    "pcr_medio":       e["pcr_info"].get("medio"),
                    "pcr_largo":       e["pcr_info"].get("largo"),
                    "pcr_oi_corto":    pcr_d.get("pcr_corto"),
                    "pcr_oi_medio":    pcr_d.get("pcr_medio"),
                    "pcr_oi_largo":    pcr_d.get("pcr_largo"),
                    "pcr_fecha_snap":  pcr_d.get("fecha_snap"),
                    "sector_budget":   SECTOR_BUDGET,
                    "position_size":   POSITION_SIZE,
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

    # 10. Resumen de candidatos rechazados
    n_sin_liq  = sum(1 for c in candidatos_log if c.get("motivo_skip") == "SIN_LIQUIDEZ_OPCIONES")
    n_pcr_rej  = sum(1 for c in candidatos_log if (c.get("motivo_skip") or "").startswith("PCR_CONSENSO"))
    n_slots    = sum(1 for c in candidatos_log if (c.get("motivo_skip") or "").startswith("CAPITAL_O_SLOTS"))
    n_entro    = sum(1 for c in candidatos_log if c.get("entro"))
    log(sep)
    log("CANDIDATOS:")
    log(f"  {len(candidatos_log)} evaluados | {n_entro} abrieron | "
        f"{n_sin_liq} sin liquidez opciones | "
        f"{n_pcr_rej} rechazo PCR | "
        f"{n_slots} slots/capital")

    if not dry_run and candidatos_log:
        registrar_candidatos_diarios(eid, hoy, candidatos_log)

    # 11. Metricas del dia
    log(sep)
    if not dry_run:
        registrar_metricas_diarias(eid, hoy)
        registrar_estado_posiciones(eid, hoy, precios)
        backfill_retornos_candidatos(eid, hoy, precios)
    else:
        log("[DRY RUN] Metricas, posiciones y retornos no registrados.")

    log("Completado.")
    log(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FT Bot Tech Sectorial + Confirmacion Opciones PCR_OI"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
