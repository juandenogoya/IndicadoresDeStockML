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

from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_posiciones_abiertas,
    obtener_precios_cierre_todos, abrir_operacion, cerrar_operacion,
    registrar_metricas_diarias, calcular_qty_ft, calcular_cash_desplegable,
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


# ── Evaluacion de cierres ─────────────────────────────────────────────────────

def evaluar_cierres(
    posiciones:   list[dict],
    precios:      dict[str, float],
    earnings_map: dict[str, date],
) -> list[dict]:
    """
    Evalua posiciones abiertas y detecta condiciones de salida.

    Prioridades:
        P1. Earnings manana     (prioridad absoluta)
        P2. Score degradado     (nivel ya no es COMPRA_FUERTE o score < minimo)
        P3. Stop loss 5%
        P4. Take profit 10%
    """
    if not posiciones:
        return []

    tickers      = [p["ticker"] for p in posiciones]
    scanner_map  = obtener_estado_scanner_tickers(tickers)
    a_cerrar     = []

    for pos in posiciones:
        ticker         = pos["ticker"]
        precio_entrada = float(pos["precio_entrada"])
        precio_actual  = precios.get(ticker)
        sl             = round(precio_entrada * (1 - SL_PCT), 4)
        tp             = round(precio_entrada * (1 + TP_PCT), 4)

        if not precio_actual:
            log(f"  [WARN] Sin precio para {ticker}, se omite evaluacion.")
            continue

        motivo = None

        # P1: Earnings
        if ticker in earnings_map:
            motivo = "EARNINGS_MANANA"

        # P2: Score degradado
        elif ticker not in scanner_map:
            motivo = "SCORE_DEGRADADO_sin_senal"
        else:
            scan = scanner_map[ticker]
            if scan["nivel"] != NIVEL_MIN or float(scan["score"]) < SCORE_MIN:
                motivo = f"SCORE_DEGRADADO_{scan['nivel']}_{scan['score']:.0f}"

        # P3: Stop loss
        if not motivo and precio_actual <= sl:
            motivo = "STOP_LOSS"

        # P4: Take profit
        if not motivo and precio_actual >= tp:
            motivo = "TAKE_PROFIT"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar


# ── Evaluacion de entradas ────────────────────────────────────────────────────

def evaluar_entradas(
    posiciones:        list[dict],
    estrategia:        dict,
    precios:           dict[str, float],
    tickers_bloqueados: set,
) -> list[dict]:
    """
    Evalua senales del scanner y retorna candidatos para abrir posicion.
    """
    tickers_abiertos = {p["ticker"] for p in posiciones}
    bloqueados       = tickers_bloqueados | tickers_abiertos
    senales          = obtener_senales_scanner()
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

    a_abrir = []

    for senal in senales:
        if len(a_abrir) >= slots_libres:
            break

        ticker = senal["ticker"]
        if ticker in bloqueados:
            continue

        precio = precios.get(ticker)
        if not precio:
            continue

        qty = calcular_qty_ft(cash, precio, RIESGO_POR_TRADE, capital_actual)
        if qty < 1:
            log(f"  [SKIP] {ticker}: capital insuficiente (precio={precio:.2f})")
            continue

        capital_trade = round(precio * qty, 2)
        if capital_trade > cash:
            log(f"  [SKIP] {ticker}: trade ({capital_trade:.2f}) supera cash ({cash:.2f})")
            continue

        sl = round(precio * (1 - SL_PCT), 4)
        tp = round(precio * (1 + TP_PCT), 4)

        a_abrir.append({
            "ticker":      ticker,
            "precio":      precio,
            "qty":         qty,
            "capital":     capital_trade,
            "sl":          sl,
            "tp":          tp,
            "score":       float(senal["score"]),
            "nivel":       senal["nivel"],
            "scan_fecha":  str(senal["scan_fecha"]),
        })

        # Descontar del cash proyectado para los siguientes candidatos
        cash -= capital_trade

    return a_abrir


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
    a_abrir = evaluar_entradas(posiciones, estrategia, precios, tickers_bloqueados)

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

    # 9. Registrar metricas del dia
    log(sep)
    if not dry_run:
        registrar_metricas_diarias(eid, hoy)
    else:
        log("[DRY RUN] Metricas no registradas.")

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
