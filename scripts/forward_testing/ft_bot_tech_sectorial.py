"""
ft_bot_tech_sectorial.py
Forward-testing — AT Tecnico con particion sectorial.

Estrategia FT_TECH_SECTOR_v1:
    Capital $100.000 dividido en 9 sectores de $11.111 cada uno.
    Cada sector opera de forma independiente con su propio presupuesto.

    ENTRADA : score tecnico >= 4.0 (misma logica que FT_TECH_v1)
              Ranking por score dentro de cada sector.
              Max 5 posiciones por sector al 20% del presupuesto sectorial.
              100% deployable — la diversificacion la controla la particion.

    SALIDA  : P1. Earnings manana (prioridad absoluta)
              P2. Score degradado (<= 3.5) — exit primario
              P3. Stop loss 2 x ATR14  — exit emergencia
              P4. Take profit 4 x ATR14 — exit emergencia

    SIZING  : ~$2.222 por posicion (20% de $11.111 sector budget)
              qty = floor(POSITION_SIZE / precio_cierre)

Diferencias vs FT_TECH_v1:
    - Sin techo global del 80% (la particion sectorial ya diversifica)
    - Position sizing fijo en dolares (no % del capital total)
    - Hasta 9 x 5 = 45 posiciones teoricas (en practica muchas menos)
    - Capital ocioso por sector: si un sector no tiene candidatos, esos
      $11.111 quedan liquidos y NO se reasignan a otros sectores.

Uso:
    python scripts/forward_testing/ft_bot_tech_sectorial.py
    python scripts/forward_testing/ft_bot_tech_sectorial.py --dry-run
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
# Cerebro de decision COMPARTIDO FT <-> Alpaca (Plan B, Tarea 16)
from src.strategies.sectorial import (
    ConfigSectorial, evaluar_cierres_sectorial, evaluar_entradas_sectorial,
)

from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_precios_cierre_todos,
    abrir_operacion, cerrar_operacion,
    registrar_metricas_diarias, registrar_candidatos_diarios,
    registrar_estado_posiciones, backfill_retornos_candidatos,
)

# ── Parametros de la estrategia ───────────────────────────────────────────────

NOMBRE_ESTRATEGIA = "FT_TECH_SECTOR_v1"

SECTORES_ACTIVOS = [
    "Technology", "Consumer Cyclical", "Financial Services",
    "Industrials", "Basic Materials", "Healthcare",
    "Energy", "Communication Services", "Consumer Defensive",
]

CAPITAL_TOTAL      = 100_000.0
N_SECTORES         = len(SECTORES_ACTIVOS)                   # 9
SECTOR_BUDGET      = round(CAPITAL_TOTAL / N_SECTORES, 2)    # 11.111,11
MAX_POS_SECTOR     = 5
POSITION_PCT       = 0.20                                    # 20% del presupuesto sectorial
POSITION_SIZE      = round(SECTOR_BUDGET * POSITION_PCT, 2)  # ~2.222,22

SCORE_ENTRADA      = 4.0
SCORE_SALIDA       = 3.5
SCORE_MAXIMO       = 5.5
ATR_MULT_SL        = 2.0
ATR_MULT_TP        = 4.0

# Config del cerebro compartido (variante v1: solo tecnico, sin opciones)
CFG = ConfigSectorial(
    nombre=NOMBRE_ESTRATEGIA,
    sectores_activos=SECTORES_ACTIVOS,
    sector_budget=SECTOR_BUDGET,
    max_pos_sector=MAX_POS_SECTOR,
    position_size=POSITION_SIZE,
    score_entrada=SCORE_ENTRADA,
    score_salida=SCORE_SALIDA,
    atr_mult_sl=ATR_MULT_SL,
    atr_mult_tp=ATR_MULT_TP,
    score_maximo=SCORE_MAXIMO,
    usar_opciones=False,
)


# ── Queries especificas de este bot ──────────────────────────────────────────

def obtener_posiciones_con_sector(estrategia_id: int) -> list[dict]:
    """
    Posiciones abiertas enriquecidas con el sector del ticker (JOIN activos).
    """
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
    Indicadores tecnicos del ultimo dia disponible para los 9 sectores activos.
    JOIN con activos (sector) y precios_diarios (close).
    Devuelve las columnas que calcular_score_tecnico() necesita:
        close, sma21, sma50, sma200, rsi14, macd, macd_signal, atr14
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
    """
    Para posiciones abiertas: trae los indicadores actuales para evaluar el
    score de salida.

    IMPORTANTE: devuelve close (via JOIN precios_diarios) + SMAs ABSOLUTAS
    (sma21/50/200), no dist_sma*. calcular_score_tecnico necesita esos
    campos exactos para evaluar la Capa 1 (close > sma200). Antes esta query
    devolvia dist_sma* y sin close, lo que forzaba score = 0.0 siempre y
    disparaba SCORE_DEGRADADO_0.0 todos los dias.
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


# ── Decision: la logica vive en src/strategies/sectorial.py (CFG con ──────────
#    usar_opciones=False). Aca quedan solo los adapters de datos (queries) y el
#    runner. evaluar_cierres_sectorial / evaluar_entradas_sectorial se importan.


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 60

    log(sep)
    log(f"FT Bot Tecnico Sectorial | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(f"  {N_SECTORES} sectores x ${SECTOR_BUDGET:,.2f} | "
        f"Position size: ${POSITION_SIZE:,.2f} | Max {MAX_POS_SECTOR} pos/sector")
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

    # 2. Precios de cierre
    precios = obtener_precios_cierre_todos()
    log(f"Precios cargados: {len(precios)} tickers")

    # 3. Indicadores con sector
    indicadores = obtener_indicadores_con_sector()
    log(f"Indicadores cargados: {len(indicadores)} tickers en {N_SECTORES} sectores")

    # 4. Posiciones abiertas con sector
    posiciones = obtener_posiciones_con_sector(eid)
    log(f"Posiciones abiertas: {len(posiciones)}")

    # Resumen por sector
    if posiciones:
        from collections import Counter
        sector_count = Counter(p["sector"] for p in posiciones)
        for s, n in sorted(sector_count.items()):
            dep = sum(float(p["capital_entrada"]) for p in posiciones if p["sector"] == s)
            log(f"  {s}: {n} pos | ${dep:,.2f} / ${SECTOR_BUDGET:,.2f}")

    # 5. Filtro earnings para posiciones abiertas
    tickers_pos     = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar hoy): {list(earnings_cierre.keys())}")

    # 6. Evaluar cierres
    log(sep)
    log("CIERRES:")
    tickers_pos_all   = [p["ticker"] for p in posiciones]
    indicadores_map   = obtener_estado_tecnico_tickers(tickers_pos_all)
    a_cerrar, _       = evaluar_cierres_sectorial(
        posiciones, precios, earnings_cierre, indicadores_map, CFG, log=log,
    )
    tickers_cerrados  = set()

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

    # 7. Recargar posiciones y estrategia tras cierres
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_con_sector(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 8. Filtro earnings para candidatos de entrada
    tickers_candidatos = [i["ticker"] for i in indicadores]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_candidatos) if tickers_candidatos else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {len(tickers_bloqueados)} tickers")

    # 9. Evaluar entradas por sector
    log(sep)
    log("ENTRADAS POR SECTOR:")
    a_abrir, _ = evaluar_entradas_sectorial(
        posiciones, indicadores, precios,
        tickers_bloqueados, tickers_cerrados, CFG, log=log,
    )

    if not a_abrir:
        log("  Sin entradas.")
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

    # 10. Guardar candidatos del dia (abiertos + oportunidades)
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
            "ticker":          t,
            "score":           float(score),
            "entro":           entro,
            "motivo_skip":     None if entro else f"CAPITAL_O_SLOTS_{s}",
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

    # 11. Registrar metricas del dia
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
    parser = argparse.ArgumentParser(description="FT Bot Tecnico Sectorial")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
