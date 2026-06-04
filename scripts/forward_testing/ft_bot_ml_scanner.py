"""
ft_bot_ml_scanner.py
Forward-testing — Replica del Bot 1 ML Scanner (Alpaca).

Logica:
    ENTRADA : alertas_scanner con nivel=COMPRA_FUERTE y score >= SCORE_MIN
    SALIDA  : score degradado (nivel ya no es COMPRA_FUERTE o score < minimo)
              SL fijo 5% sobre precio de entrada
              TP fijo 10% sobre precio de entrada
              Earnings manana (prioridad absoluta)

Diferencias vs Bot Alpaca:
    - Precio de ejecucion = cierre del dia (precios_diarios), sin llamadas a Alpaca
    - Escribe en ft_operaciones / ft_estrategias / ft_metricas_diarias
    - No envia ordenes al broker

Uso:
    python scripts/forward_testing/ft_bot_ml_scanner.py
    python scripts/forward_testing/ft_bot_ml_scanner.py --dry-run
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
# Cerebro de decision COMPARTIDO FT <-> Alpaca (Plan B, Tarea 16)
from src.strategies.ml_scanner import (
    ConfigML, evaluar_cierres_ml, evaluar_entradas_ml,
)

from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_posiciones_abiertas,
    obtener_precios_cierre_todos, abrir_operacion, cerrar_operacion,
    registrar_metricas_diarias, calcular_qty_ft, calcular_cash_desplegable,
    registrar_candidatos_diarios, registrar_estado_posiciones,
    backfill_retornos_candidatos,
)

# ── Parametros de la estrategia ───────────────────────────────────────────────
NOMBRE_ESTRATEGIA = "FT_ML_SCANNER_v1"
SCORE_MIN         = 65
NIVEL_MIN         = "COMPRA_FUERTE"
MAX_POSICIONES    = 5
MAX_DEPLOY_PCT    = 0.80     # nunca desplegar mas del 80% del capital
RIESGO_POR_TRADE  = 0.15     # 15% del capital actual por trade
SL_PCT            = 0.05     # Stop loss 5%
TP_PCT            = 0.10     # Take profit 10%

# Config del cerebro compartido
CFG_ML = ConfigML(
    nombre=NOMBRE_ESTRATEGIA,
    score_min=SCORE_MIN,
    nivel_min=NIVEL_MIN,
    max_posiciones=MAX_POSICIONES,
    max_deploy_pct=MAX_DEPLOY_PCT,
    riesgo_por_trade=RIESGO_POR_TRADE,
    sl_pct=SL_PCT,
    tp_pct=TP_PCT,
)


# ── Datos de scanner ──────────────────────────────────────────────────────────

def obtener_senales_scanner() -> list[dict]:
    """
    Lee las alertas del ultimo dia de scan.
    Retorna filas con alert_nivel=COMPRA_FUERTE y alert_score >= SCORE_MIN,
    ordenadas por score desc.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker, alert_nivel AS nivel, alert_score AS score, scan_fecha
            FROM alertas_scanner
            WHERE scan_fecha = (SELECT MAX(scan_fecha) FROM alertas_scanner)
              AND alert_nivel = :nivel
              AND alert_score >= :score_min
            ORDER BY alert_score DESC
        """), {"nivel": NIVEL_MIN, "score_min": SCORE_MIN}).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_estado_scanner_tickers(tickers: list[str]) -> dict[str, dict]:
    """
    Para los tickers dados, retorna el estado actual del scanner (ultimo scan).
    Usado para evaluar si mantener posiciones abiertas.
    """
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (ticker)
                ticker, alert_nivel AS nivel, alert_score AS score, scan_fecha
            FROM alertas_scanner
            WHERE ticker = ANY(:tickers)
            ORDER BY ticker, scan_fecha DESC
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


# ── Decision: la logica vive en src/strategies/ml_scanner.py (CFG_ML). Aca ────
#    quedan solo los adapters de datos (obtener_senales_scanner / obtener_estado_
#    scanner_tickers) y el runner. evaluar_cierres_ml / evaluar_entradas_ml se
#    importan del modulo compartido.


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 55

    log(sep)
    log(f"FT Bot ML Scanner | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(sep)

    # 1. Cargar estrategia
    estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)
    if not estrategia:
        log(f"[ERROR] Estrategia '{NOMBRE_ESTRATEGIA}' no encontrada o inactiva.")
        log("Hint: insertar primero en ft_estrategias con activa=TRUE.")
        return

    eid = estrategia["id"]
    log(f"Estrategia id={eid} | capital={estrategia['capital_actual']:,.2f} | "
        f"cash={estrategia['cash_disponible']:,.2f}")

    # 2. Precios de cierre (una sola query para todo)
    precios = obtener_precios_cierre_todos()
    log(f"Precios cargados: {len(precios)} tickers")

    # 3. Posiciones abiertas actuales
    posiciones = obtener_posiciones_abiertas(eid)
    log(f"Posiciones abiertas: {len(posiciones)}")

    # 4. Filtro earnings para posiciones abiertas
    tickers_pos = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar hoy): {list(earnings_cierre.keys())}")

    # 5. Evaluar cierres
    log(sep)
    log("CIERRES:")
    scanner_map = obtener_estado_scanner_tickers([p["ticker"] for p in posiciones])
    a_cerrar = evaluar_cierres_ml(
        posiciones, precios, earnings_cierre, scanner_map, CFG_ML, log=log,
    )

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

    # 6. Recargar posiciones y estrategia despues de cierres
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_abiertas(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 7. Filtro earnings para candidatos de entrada
    todas_las_senales = obtener_senales_scanner()
    tickers_candidatos = [s["ticker"] for s in todas_las_senales]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_candidatos) if tickers_candidatos else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {tickers_bloqueados}")

    # 8. Evaluar entradas
    log(sep)
    log("ENTRADAS:")
    cash = calcular_cash_desplegable(estrategia, MAX_DEPLOY_PCT)
    a_abrir = evaluar_entradas_ml(
        posiciones, todas_las_senales, precios, tickers_bloqueados,
        cash=cash,
        capital_actual=float(estrategia["capital_actual"]),
        capital_inmovilizado=float(estrategia["capital_inmovilizado"]),
        cfg=CFG_ML, log=log,
    )

    if not a_abrir:
        log("  Sin senales de entrada.")
    else:
        for e in a_abrir:
            log(f"  ABRIR {e['ticker']} | precio={e['precio']:.2f} | "
                f"qty={e['qty']} | capital={e['capital']:.2f} | "
                f"score={e['score']:.0f} | sl={e['sl']:.2f} | tp={e['tp']:.2f}")

            if not dry_run:
                detalle = {
                    "nivel":      e["nivel"],
                    "score_ml":   e["score"],
                    "scan_fecha": e["scan_fecha"],
                    "sl_pct":     SL_PCT,
                    "tp_pct":     TP_PCT,
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
    tickers_abiertos_hoy  = {p["ticker"] for p in posiciones}
    tickers_que_abrimos   = {e["ticker"] for e in a_abrir}

    candidatos_log = []
    for s in todas_las_senales:
        t = s["ticker"]
        if t in tickers_abiertos_hoy:
            continue   # ya tenia posicion abierta
        if t in tickers_bloqueados:
            continue   # bloqueado por earnings
        entro = t in tickers_que_abrimos
        candidatos_log.append({
            "ticker":          t,
            "score":           float(s["score"]),
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

    # 10. Registrar metricas del dia
    log(sep)
    if not dry_run:
        registrar_metricas_diarias(eid, hoy)
    else:
        log("[DRY RUN] Metricas no registradas.")

    # 11. Observacion diaria: snapshots de posiciones + retornos contrafactuales
    if not dry_run:
        registrar_estado_posiciones(eid, hoy, precios)
        backfill_retornos_candidatos(eid, hoy, precios)

    log("Completado.")
    log(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FT Bot ML Scanner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
