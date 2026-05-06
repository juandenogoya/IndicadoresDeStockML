"""
ft_bot_smc_v2.py
Forward-testing — SMC CHoCH/BOS con filtro de contexto y salida por agotamiento.

Estrategia FT_SMC_v2:
    Extension de SMC_v1 con dos cambios:

    1. FILTRO DE ENTRADA (OR): no entra si el mercado no confirma momentum.
       Requiere AL MENOS UNA:
         lateral_ratio   > 1.0   (mercado trending, no lateral)
         candle_score_5d > 0     (momentum de velas positivo en 5 dias)
       Candidatos rechazados: motivo_skip = 'FILTRO_CONTEXTO_SMC'

    2. SALIDA POR AGOTAMIENTO (AND): cierra cuando convergen tres senales negativas.
       Cierra si TODAS:
         up_vol_5d       = 0     (ningun dia alcista con volumen en ultimos 5)
         candle_score_5d < -2    (deterioro material de velas)
         lateral_ratio   < 0.5   (mercado lateral sin rango definido)
       Motivo: AGOTAMIENTO_SEÑAL
       El time stop de 20 dias de v1 es ELIMINADO.

    SALIDAS MANTENIDAS DE v1:
        - Trailing SL estructural (solo sube)
        - CHOCH_BEAR (choch_bear_10 = 1)
        - ESTRUCTURA_ROTA (estructura_10 = -1)
        - EARNINGS_MANANA (prioridad absoluta)

    ENTRADAS: identicas a v1 excepto por el filtro de contexto nuevo.

Uso:
    python scripts/forward_testing/ft_bot_smc_v2.py
    python scripts/forward_testing/ft_bot_smc_v2.py --dry-run
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
from scripts.forward_testing.ft_scoring import (
    calcular_score_estructura,
    obtener_features_hoy,
    obtener_estructura_tickers,
    calcular_actualizaciones_sl,
    obtener_candle_score_5d,
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
NOMBRE_ESTRATEGIA = "FT_SMC_v2"
SCORE_ENTRADA_MIN = 1
SCORE_MAXIMO      = 3
MIN_SL_DIST_PCT   = 1.0
MAX_SL_DIST_PCT   = 8.0
MAX_POSICIONES    = 5
MAX_DEPLOY_PCT    = 0.80
RIESGO_POR_TRADE  = 0.15

# v2: parametros de filtro de entrada (OR)
FILTRO_LATERAL_RATIO_MIN    = 1.0    # lateral_ratio > 1.0 -> trending (OK para entrar)
FILTRO_CANDLE_SCORE_MIN     = 0      # candle_score_5d > 0 -> momentum positivo (OK para entrar)

# v2: parametros de agotamiento (AND)
AGOTAMIENTO_UP_VOL_MAX      = 0      # up_vol_5d = 0   (ningun dia alcista con vol)
AGOTAMIENTO_CANDLE_MAX      = -2     # candle_score < -2 (deterioro material)
AGOTAMIENTO_LATERAL_MAX     = 0.5    # lateral_ratio < 0.5 (lateral estrecho)


# ── Helpers de mercado (v2) ───────────────────────────────────────────────────

def obtener_contexto_senales_tickers(tickers: list[str]) -> dict[str, dict]:
    """
    Calcula up_vol_5d y lateral_ratio para una lista de tickers.
    Usado para: evaluar agotamiento (posiciones abiertas) y
                evaluar filtro de contexto (candidatos de entrada).

    Retorna {ticker: {up_vol_5d, lateral_ratio, atr14, rango_5d_abs}}
    """
    if not tickers:
        return {}

    engine = get_engine()

    with engine.connect() as conn:
        # up_vol_5d desde features_precio_accion
        vol_rows = conn.execute(text("""
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
        up_vol_map = {r.ticker: int(r.up_vol_5d or 0) for r in vol_rows}

        # rango_5d_abs desde precios_diarios
        rango_rows = conn.execute(text("""
            WITH ranked AS (
                SELECT ticker, close,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
                FROM precios_diarios
                WHERE ticker = ANY(:tickers)
            )
            SELECT ticker, MAX(close) - MIN(close) AS rango_5d_abs
            FROM ranked
            WHERE rn <= 5
            GROUP BY ticker
        """), {"tickers": tickers}).fetchall()
        rango_map = {r.ticker: float(r.rango_5d_abs or 0) for r in rango_rows}

        # atr14 desde indicadores_tecnicos
        atr_rows = conn.execute(text("""
            SELECT DISTINCT ON (ticker) ticker, atr14
            FROM indicadores_tecnicos
            WHERE ticker = ANY(:tickers)
            ORDER BY ticker, fecha DESC
        """), {"tickers": tickers}).fetchall()
        atr_map = {r.ticker: float(r.atr14 or 0) for r in atr_rows}

    resultado = {}
    for t in tickers:
        rango   = rango_map.get(t, 0.0)
        atr     = atr_map.get(t, 0.0)
        lat_rat = round(rango / atr, 4) if atr > 0 else None
        resultado[t] = {
            "up_vol_5d":    up_vol_map.get(t, 0),
            "rango_5d_abs": rango,
            "atr14":        atr,
            "lateral_ratio": lat_rat,
        }

    return resultado


# ── Trailing SL ───────────────────────────────────────────────────────────────

def procesar_trailing_sl(
    posiciones: list[dict],
    precios:    dict[str, float],
    dry_run:    bool = False,
) -> list[dict]:
    """
    Recalcula el SL estructural para cada posicion abierta (solo sube).
    Identica a v1.
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

        if op_id in pos_map:
            pos_map[op_id] = {**pos_map[op_id], "stop_loss": nuevo_sl}

    return list(pos_map.values())


# ── Evaluacion de cierres (v2) ────────────────────────────────────────────────

def evaluar_cierres(
    posiciones:    list[dict],
    precios:       dict[str, float],
    earnings_map:  dict[str, date],
    candle_scores: dict[str, float],
    contexto_map:  dict[str, dict],
) -> list[dict]:
    """
    Prioridades de salida v2:
        P0. Earnings manana     -> EARNINGS_MANANA
        P1. Trailing SL roto    -> TRAILING_SL
        P2. CHoCH bajista       -> CHOCH_BEAR
        P3. Estructura rota     -> ESTRUCTURA_ROTA
        P4. Agotamiento senal   -> AGOTAMIENTO_SEÑAL
            up_vol_5d = 0 AND candle_score_5d < -2 AND lateral_ratio < 0.5
        [TIME_STOP ELIMINADO]
    """
    if not posiciones:
        return []

    tickers   = [p["ticker"] for p in posiciones]
    datos_map = obtener_estructura_tickers(tickers)
    a_cerrar  = []

    for pos in posiciones:
        ticker     = pos["ticker"]
        datos_hoy  = datos_map.get(ticker)
        precio_act = precios.get(ticker)
        stop_loss  = float(pos["stop_loss"]) if pos.get("stop_loss") else None

        if not precio_act:
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
            if stop_loss and precio_act <= stop_loss:
                motivo = "TRAILING_SL"

            # P2: CHoCH bajista
            elif choch_bear_10 == 1:
                motivo = "CHOCH_BEAR"

            # P3: Estructura rota
            elif estructura_10 == -1:
                motivo = "ESTRUCTURA_ROTA"

            # P4: Agotamiento de senal (v2 — reemplaza time stop)
            else:
                cs5d   = candle_scores.get(ticker, 0.0) or 0.0
                ctx    = contexto_map.get(ticker, {})
                upv5   = ctx.get("up_vol_5d", 0) or 0
                lat    = ctx.get("lateral_ratio")

                if (upv5 <= AGOTAMIENTO_UP_VOL_MAX
                        and cs5d < AGOTAMIENTO_CANDLE_MAX
                        and lat is not None
                        and lat < AGOTAMIENTO_LATERAL_MAX):
                    motivo = "AGOTAMIENTO_SEÑAL"
                    log(f"  [AGOTAMIENTO] {ticker}: up_vol={upv5} "
                        f"candle={cs5d:.1f} lateral={lat:.3f}")

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_act,
                "motivo":        motivo,
            })

    return a_cerrar


# ── Evaluacion de entradas (v2) ───────────────────────────────────────────────

def evaluar_entradas(
    posiciones:         list[dict],
    estrategia:         dict,
    precios:            dict[str, float],
    tickers_bloqueados: set,
    candle_scores:      dict[str, float],
    contexto_map:       dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """
    Escanea features estructurales y retorna:
        (a_abrir, rechazados_filtro)

    Filtro de contexto (OR): al menos una condicion debe cumplirse:
        lateral_ratio   > FILTRO_LATERAL_RATIO_MIN   (mercado trending)
        candle_score_5d > FILTRO_CANDLE_SCORE_MIN     (momentum positivo)

    Si ninguna se cumple: candidato rechazado con FILTRO_CONTEXTO_SMC.

    Candidatos rechazados se registran en ft_candidatos_diarios para
    evaluar si el filtro discrimina correctamente (analisis post-hoc).
    """
    tickers_abiertos = {p["ticker"] for p in posiciones}
    excluidos        = tickers_bloqueados | tickers_abiertos
    capital_actual   = float(estrategia["capital_actual"])
    slots_libres     = MAX_POSICIONES - len(posiciones)

    if slots_libres <= 0:
        log("  Maximo de posiciones alcanzado, sin entradas.")
        return [], []

    cash = calcular_cash_desplegable(estrategia, MAX_DEPLOY_PCT)
    if cash <= 0:
        pct_actual = float(estrategia["capital_inmovilizado"]) / capital_actual * 100
        log(f"  Techo de despliegue alcanzado ({pct_actual:.1f}% desplegado). Sin entradas.")
        return [], []

    features    = obtener_features_hoy()
    candidatos  = []
    rechazados  = []   # filtro de contexto

    for row in features:
        ticker = row["ticker"]
        if ticker in excluidos:
            continue

        score, detalle = calcular_score_estructura(row)
        if score < SCORE_ENTRADA_MIN:
            continue

        # Filtro de contexto v2 (OR)
        cs5d = candle_scores.get(ticker, 0.0) or 0.0
        ctx  = contexto_map.get(ticker, {})
        lat  = ctx.get("lateral_ratio")

        pasa_candle  = cs5d > FILTRO_CANDLE_SCORE_MIN
        pasa_lateral = lat is not None and lat > FILTRO_LATERAL_RATIO_MIN

        if not pasa_candle and not pasa_lateral:
            rechazados.append({
                "ticker":          ticker,
                "score":           float(score),
                "entro":           False,
                "motivo_skip":     "FILTRO_CONTEXTO_SMC",
                "precio_apertura": precios.get(ticker),
            })
            lat_log = f"{lat:.3f}" if lat is not None else "N/A"
            log(f"  [FILTRO_CONTEXTO] {ticker}: candle={cs5d:.1f} "
                f"lateral={lat_log} -> rechazado")
            continue

        candidatos.append({
            "ticker": ticker, "score": score,
            "detalle": detalle, "row": row,
            "candle_score_5d": cs5d, "lateral_ratio": lat,
        })

    if not candidatos:
        return [], rechazados

    candidatos.sort(key=lambda x: x["score"], reverse=True)
    a_abrir = []

    for c in candidatos:
        if len(a_abrir) >= slots_libres:
            break

        ticker = c["ticker"]
        row    = c["row"]
        det    = c["detalle"]
        precio = precios.get(ticker)

        if not precio:
            continue

        dist_sl_pct = det["dist_sl_pct"]
        swing_low   = _swing_low_precio(precio, dist_sl_pct)

        if swing_low <= 0:
            continue

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
            "ticker":             ticker,
            "precio":             precio,
            "qty":                qty,
            "capital":            capital_trade,
            "sl":                 swing_low,
            "tp":                 None,
            "score":              c["score"],
            "detalle":            det,
            "evento":             evento,
            "atr":                round(atr, 4),
            "dist_sl_pct":        round(dist_sl_pct, 2),
            "candle_score_5d":    c["candle_score_5d"],
            "lateral_ratio":      c["lateral_ratio"],
        })

        cash -= capital_trade

    return a_abrir, rechazados


# ── Runner principal ──────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    hoy = date.today()
    sep = "-" * 55

    log(sep)
    log(f"FT Bot SMC v2 | {hoy} {'[DRY RUN]' if dry_run else ''}")
    log(f"  Filtro entrada (OR): lateral>{FILTRO_LATERAL_RATIO_MIN} OR candle>{FILTRO_CANDLE_SCORE_MIN}")
    log(f"  Agotamiento (AND): up_vol<={AGOTAMIENTO_UP_VOL_MAX} AND "
        f"candle<{AGOTAMIENTO_CANDLE_MAX} AND lateral<{AGOTAMIENTO_LATERAL_MAX}")
    log(sep)

    # 1. Cargar estrategia
    estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)
    if not estrategia:
        log(f"[ERROR] Estrategia '{NOMBRE_ESTRATEGIA}' no encontrada o inactiva.")
        log("Hint: registrar en ft_estrategias con logica='smc_estructura_v2'")
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

    # 4. Senales de mercado para posiciones abiertas (agotamiento)
    tickers_pos    = [p["ticker"] for p in posiciones]
    candle_scores  = obtener_candle_score_5d()
    contexto_pos   = obtener_contexto_senales_tickers(tickers_pos)

    if posiciones:
        log("  Senales actuales por posicion:")
        for p in posiciones:
            t   = p["ticker"]
            cs  = candle_scores.get(t, 0.0) or 0.0
            ctx = contexto_pos.get(t, {})
            uv  = ctx.get("up_vol_5d", "?")
            lr  = ctx.get("lateral_ratio")
            lr_s = f"{lr:.3f}" if lr is not None else "N/A"
            log(f"    {t}: candle={cs:.1f} up_vol={uv} lateral={lr_s}")

    # 5. Filtro earnings para posiciones abiertas
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    if earnings_cierre:
        log(f"Earnings manana (cerrar hoy): {list(earnings_cierre.keys())}")

    # 6. Actualizar trailing SL (antes de evaluar cierres)
    log(sep)
    log("TRAILING SL:")
    posiciones = procesar_trailing_sl(posiciones, precios, dry_run)

    # 7. Evaluar cierres
    log(sep)
    log("CIERRES:")
    a_cerrar = evaluar_cierres(
        posiciones, precios, earnings_cierre,
        candle_scores, contexto_pos,
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

    # 8. Recargar posiciones y estrategia
    if not dry_run and a_cerrar:
        posiciones = obtener_posiciones_abiertas(eid)
        estrategia = cargar_estrategia(NOMBRE_ESTRATEGIA)

    # 9. Filtro earnings para candidatos de entrada
    features_hoy       = obtener_features_hoy()
    tickers_candidatos = [f["ticker"] for f in features_hoy]
    earnings_bloqueo   = tickers_a_bloquear_entrada(tickers_candidatos) if tickers_candidatos else {}
    tickers_bloqueados = set(earnings_bloqueo.keys())
    if tickers_bloqueados:
        log(f"Bloqueados por earnings proximos: {len(tickers_bloqueados)} tickers")

    # 10. Contexto de mercado para candidatos (filtro entrada)
    contexto_cand = obtener_contexto_senales_tickers(tickers_candidatos)

    # 11. Evaluar entradas
    log(sep)
    log("ENTRADAS:")
    a_abrir, rechazados_filtro = evaluar_entradas(
        posiciones, estrategia, precios,
        tickers_bloqueados, candle_scores, contexto_cand,
    )

    if not a_abrir:
        log("  Sin senales de entrada.")
    else:
        for e in a_abrir:
            lat_s = f"{e['lateral_ratio']:.3f}" if e['lateral_ratio'] is not None else "N/A"
            log(f"  ABRIR {e['ticker']} | evento={e['evento']} | "
                f"precio={e['precio']:.2f} | qty={e['qty']} | "
                f"capital={e['capital']:.2f} | score={e['score']:.0f}/{SCORE_MAXIMO} | "
                f"sl={e['sl']:.4f} | dist_sl={e['dist_sl_pct']:.1f}% | "
                f"candle={e['candle_score_5d']:.1f} | lateral={lat_s}")

            if not dry_run:
                detalle = {
                    **e["detalle"],
                    "evento":            e["evento"],
                    "score":             e["score"],
                    "dist_sl_pct":       e["dist_sl_pct"],
                    "atr":               e["atr"],
                    "candle_score_5d":   e["candle_score_5d"],
                    "lateral_ratio":     e["lateral_ratio"],
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

    if rechazados_filtro:
        log(f"  Filtro contexto: {len(rechazados_filtro)} candidatos rechazados")
        for r in rechazados_filtro:
            log(f"    FILTRADO {r['ticker']} (score={r['score']:.0f})")

    # 12. Candidatos del dia (qualifying + rechazados por filtro)
    log(sep)
    log("CANDIDATOS:")
    tickers_abiertos_hoy = {p["ticker"] for p in posiciones}
    tickers_que_abrimos  = {e["ticker"] for e in a_abrir}

    candidatos_log = []

    # Agregar primero los rechazados por filtro contexto
    for r in rechazados_filtro:
        if r["ticker"] not in tickers_abiertos_hoy and r["ticker"] not in tickers_bloqueados:
            candidatos_log.append(r)

    # Luego los que pasaron el filtro estructural (qualifying)
    for row in features_hoy:
        t = row["ticker"]
        if t in tickers_abiertos_hoy or t in tickers_bloqueados:
            continue
        # Si ya esta en rechazados_filtro, no duplicar
        if any(c["ticker"] == t for c in rechazados_filtro):
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
        n_entro    = sum(1 for c in candidatos_log if c["entro"])
        n_filtrado = sum(1 for c in candidatos_log if c.get("motivo_skip") == "FILTRO_CONTEXTO_SMC")
        n_no_entro = len(candidatos_log) - n_entro
        log(f"  {len(candidatos_log)} qualifying — {n_entro} abiertos, "
            f"{n_filtrado} filtro contexto, {n_no_entro - n_filtrado} otras razones")
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
        registrar_estado_posiciones(eid, hoy, precios)
        backfill_retornos_candidatos(eid, hoy, precios)

    log("Completado.")
    log(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FT Bot SMC Estructura v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalua sin escribir en DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
