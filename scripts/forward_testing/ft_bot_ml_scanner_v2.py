"""
ft_bot_ml_scanner_v2.py
Forward-testing -- ML_SCANNER_v1 con el modelo ML v2 (Etapa 3e, 13/9/2026).

Misma estrategia que FT_ML_SCANNER_v1: mismas reglas, mismos parametros y el
mismo cerebro compartido (src/strategies/ml_scanner.py). Lo UNICO distinto es de
donde sale la senal: lee alert_nivel_v2 / alert_score_v2 de alertas_scanner, que
el scanner calcula en paralelo con el modelo v2 (RF calibrado, 196 tickers;
docs/ml_reentrenamiento.md sec. 8c). FT_ML_SCANNER_v1 sigue como control y las
dos se comparan en la Etapa 4.

    ENTRADA : alert_nivel_v2 = COMPRA_FUERTE y alert_score_v2 >= SCORE_MIN
    SALIDA  : earnings manana / score degradado (v2) / SL 5% / TP 10%

Por que un archivo aparte y no parametrizar el de la v1: la v1 es el control del
experimento y su codigo no se toca mientras dure la comparacion.

GUARD: si la ultima corrida del scanner no trae la v2 (artefacto ausente o que no
valido contra su metadata), el bot NO opera y sale con codigo 1. Usar la ultima
fila con v2 de otra corrida seria decidir con una senal vieja, en silencio.

Uso:
    python scripts/forward_testing/ft_bot_ml_scanner_v2.py
    python scripts/forward_testing/ft_bot_ml_scanner_v2.py --dry-run
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
# Cerebro de decision COMPARTIDO con FT_ML_SCANNER_v1
from src.strategies.ml_scanner import (
    ConfigML, evaluar_cierres_ml, evaluar_entradas_ml,
)

from scripts.forward_testing.ft_utils import (
    log, cargar_estrategia, obtener_posiciones_abiertas,
    obtener_precios_cierre_todos, abrir_operacion, cerrar_operacion,
    registrar_metricas_diarias, calcular_cash_desplegable,
    registrar_candidatos_diarios, registrar_estado_posiciones,
    backfill_retornos_candidatos,
)

# ── Parametros: IDENTICOS a FT_ML_SCANNER_v1 ──────────────────────────────────
NOMBRE_ESTRATEGIA = "FT_ML_SCANNER_v2"
SCORE_MIN         = 65
NIVEL_MIN         = "COMPRA_FUERTE"
MAX_POSICIONES    = 5
MAX_DEPLOY_PCT    = 0.80     # nunca desplegar mas del 80% del capital
RIESGO_POR_TRADE  = 0.15     # 15% del capital actual por trade
SL_PCT            = 0.05     # Stop loss 5%
TP_PCT            = 0.10     # Take profit 10%

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


# ── Datos de scanner (columnas _v2) ───────────────────────────────────────────
# El "ultimo scan" se toma por MAX(scan_fecha), igual que la v1: a proposito, para
# que las dos lean la misma corrida del scanner.

def estado_v2_ultimo_scan() -> dict:
    """Cuantas filas de la ultima corrida del scanner traen la v2."""
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT scan_fecha, COUNT(*) AS filas, COUNT(alert_nivel_v2) AS con_v2,
                   MAX(ml_modelo_v2) AS modelo
            FROM alertas_scanner
            WHERE scan_fecha = (SELECT MAX(scan_fecha) FROM alertas_scanner)
            GROUP BY scan_fecha
        """)).fetchone()
    if not r:
        return {"scan_fecha": None, "filas": 0, "con_v2": 0, "modelo": None}
    return dict(r._mapping)


def obtener_senales_scanner() -> list[dict]:
    """
    Senales de la ultima corrida: alert_nivel_v2 = COMPRA_FUERTE y
    alert_score_v2 >= SCORE_MIN, ordenadas por score desc.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker, alert_nivel_v2 AS nivel, alert_score_v2 AS score, scan_fecha,
                   ml_prob_v2, ml_modelo_v2
            FROM alertas_scanner
            WHERE scan_fecha = (SELECT MAX(scan_fecha) FROM alertas_scanner)
              AND alert_nivel_v2 = :nivel
              AND alert_score_v2 >= :score_min
            ORDER BY alert_score_v2 DESC
        """), {"nivel": NIVEL_MIN, "score_min": SCORE_MIN}).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_estado_scanner_tickers(tickers: list[str]) -> dict[str, dict]:
    """
    Estado v2 mas reciente de cada ticker con posicion abierta. Toma la ultima
    fila CON v2, igual que la v1 toma la ultima fila que exista: si el scanner
    fallo en un ticker, se usa su corrida anterior en vez de cerrar por "sin senal".
    """
    if not tickers:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (ticker)
                ticker, alert_nivel_v2 AS nivel, alert_score_v2 AS score, scan_fecha
            FROM alertas_scanner
            WHERE ticker = ANY(:tickers)
              AND alert_nivel_v2 IS NOT NULL
            ORDER BY ticker, scan_fecha DESC
        """), {"tickers": tickers}).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> int:
    hoy = date.today()
    sep = "-" * 55

    log(sep)
    log(f"FT Bot ML Scanner v2 | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(sep)

    # 1. Cargar estrategia
    estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)
    if not estrategia:
        log(f"[ERROR] Estrategia '{NOMBRE_ESTRATEGIA}' no encontrada o inactiva.")
        log("Hint: python scripts/forward_testing/ft_setup_estrategias.py")
        return 1

    eid = estrategia["id"]
    log(f"Estrategia id={eid} | capital={estrategia['capital_actual']:,.2f} | "
        f"cash={estrategia['cash_disponible']:,.2f}")

    # 2. Guard: la ultima corrida del scanner tiene que traer la v2
    est_v2 = estado_v2_ultimo_scan()
    if est_v2["con_v2"] == 0:
        log(f"[ERROR] La ultima corrida del scanner ({est_v2['scan_fecha']}) no trae la v2 "
            f"({est_v2['filas']} filas, 0 con alert_nivel_v2). No se opera: correr el Paso 3 "
            f"con models_ml_v2/ en su lugar.")
        return 1
    log(f"Scanner v2: {est_v2['con_v2']}/{est_v2['filas']} filas | modelo {est_v2['modelo']} "
        f"| scan {est_v2['scan_fecha']}")
    if est_v2["con_v2"] < est_v2["filas"]:
        log(f"[WARN] {est_v2['filas'] - est_v2['con_v2']} tickers sin v2 en la ultima corrida.")

    # 3. Precios de cierre (una sola query para todo)
    precios = obtener_precios_cierre_todos()
    log(f"Precios cargados: {len(precios)} tickers")

    # 4. Posiciones abiertas actuales
    posiciones = obtener_posiciones_abiertas(eid)
    log(f"Posiciones abiertas: {len(posiciones)}")

    # 5. Filtro earnings para posiciones abiertas
    tickers_pos = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar hoy): {list(earnings_cierre.keys())}")

    # 6. Evaluar cierres
    log(sep)
    log("CIERRES:")
    scanner_map = obtener_estado_scanner_tickers(tickers_pos)
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

    # 7. Recargar posiciones y estrategia despues de cierres
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_abiertas(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 8. Filtro earnings para candidatos de entrada
    todas_las_senales = obtener_senales_scanner()
    senal_por_ticker = {s["ticker"]: s for s in todas_las_senales}
    tickers_candidatos = list(senal_por_ticker)
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_candidatos) if tickers_candidatos else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {tickers_bloqueados}")

    # 9. Evaluar entradas
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
            s = senal_por_ticker[e["ticker"]]
            log(f"  ABRIR {e['ticker']} | precio={e['precio']:.2f} | "
                f"qty={e['qty']} | capital={e['capital']:.2f} | "
                f"score={e['score']:.0f} | prob_v2={float(s['ml_prob_v2']):.3f} | "
                f"sl={e['sl']:.2f} | tp={e['tp']:.2f}")

            if not dry_run:
                detalle = {
                    "nivel":        e["nivel"],
                    "score_ml":     e["score"],
                    "scan_fecha":   e["scan_fecha"],
                    "sl_pct":       SL_PCT,
                    "tp_pct":       TP_PCT,
                    # Para atribuir despues por que difiere de la v1 (Etapa 4)
                    "ml_prob_v2":   float(s["ml_prob_v2"]),
                    "ml_modelo_v2": s["ml_modelo_v2"],
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
        log(f"  {len(candidatos_log)} candidatos qualifying -- "
            f"{sum(1 for c in candidatos_log if c['entro'])} abiertos, "
            f"{sum(1 for c in candidatos_log if not c['entro'])} oportunidades")
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
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FT Bot ML Scanner v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
