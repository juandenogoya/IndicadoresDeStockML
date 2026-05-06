"""
ft_backtesting_runner.py
Motor de backtesting historico para las 3 estrategias.

Uso:
    python scripts/backtesting_historico/ft_backtesting_runner.py \
        --estrategia TECH_SECTOR_v1 \
        --desde 2025-06-01 --hasta 2025-12-31 \
        [--dry-run] [--verbose] [--reset]

Flujo:
    1. INIT: verificar schema, registrar instancia en bt_hist_estrategias
    2. BULK LOAD: cargar todas las tablas fuente en DataFrames
    3. LOOP POR DIA: sin queries DB
       a. Cierres: evaluar posiciones abiertas
       b. Entradas: evaluar candidatos y rankear
       c. Metricas: equity del dia
    4. ESCRITURA FINAL: INSERT masivo de operaciones, metricas, candidatos
"""

import sys
import os
import argparse
import json
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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

from bt_data_loader import BtDataLoader
import bt_scoring as sc
from bt_position_manager import BtPositionManager


# ── Mapeo de estrategia -> logica DB ──────────────────────────────────────────

ESTRATEGIAS = {
    "TECH_SECTOR_v1": {
        "logica":      "tecnico_sectorial",
        "descripcion": "Scoring tecnico (SMA/MACD/RSI) con sector partitioning 9x$11k",
    },
    "COMBO_v1": {
        "logica":      "combo_tech_candle",
        "descripcion": "Tecnico sectorial + candle score 5d como tiebreaker y filtro",
    },
    "SMC_v1": {
        "logica":      "smc_estructura",
        "descripcion": "CHoCH/BOS + estructura 10d + trailing SL estructural",
    },
}

SECTORES_TECH = [
    "Technology", "Consumer Cyclical", "Financial Services", "Industrials",
    "Basic Materials", "Healthcare", "Energy", "Communication Services",
    "Consumer Defensive",
]


def log(msg: str, verbose: bool = True):
    if verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


# ── Registro de instancia ─────────────────────────────────────────────────────

def registrar_instancia(conn, estrategia: str, desde: date, hasta: date,
                         n_tickers: int, dry_run: bool) -> int:
    """Inserta o retorna el ID de la instancia en bt_hist_estrategias."""
    cfg  = ESTRATEGIAS[estrategia]
    nombre = f"{estrategia}_{desde}_{hasta}"

    row = conn.execute(text("""
        SELECT id FROM bt_hist_estrategias
        WHERE nombre = :nombre AND fecha_bt_inicio = :desde AND fecha_bt_fin = :hasta
    """), {"nombre": nombre, "desde": desde, "hasta": hasta}).fetchone()

    if row:
        return row.id

    result = conn.execute(text("""
        INSERT INTO bt_hist_estrategias
            (nombre, descripcion, logica, capital_inicial,
             fecha_bt_inicio, fecha_bt_fin,
             sin_earnings_filter, universo_tickers,
             precio_ejecucion, estado, ejecutado_en)
        VALUES
            (:nombre, :desc, :logica, :capital,
             :desde, :hasta,
             TRUE, :n_tickers,
             'close_dia_senal', 'ejecutando', NOW())
        RETURNING id
    """), {
        "nombre":    nombre,
        "desc":      cfg["descripcion"],
        "logica":    cfg["logica"],
        "capital":   sc.CAPITAL_INICIAL,
        "desde":     desde,
        "hasta":     hasta,
        "n_tickers": n_tickers,
    })
    conn.commit()
    return result.scalar()


def reset_instancia(conn, estrategia: str, desde: date, hasta: date):
    """Borra resultados anteriores de la instancia para volver a correrla."""
    nombre = f"{estrategia}_{desde}_{hasta}"
    row = conn.execute(text("""
        SELECT id FROM bt_hist_estrategias WHERE nombre = :n AND fecha_bt_inicio = :d AND fecha_bt_fin = :h
    """), {"n": nombre, "d": desde, "h": hasta}).fetchone()

    if not row:
        return

    bt_id = row.id
    conn.execute(text("DELETE FROM bt_hist_candidatos       WHERE bt_id = :id"), {"id": bt_id})
    conn.execute(text("DELETE FROM bt_hist_metricas_diarias WHERE bt_id = :id"), {"id": bt_id})
    conn.execute(text("DELETE FROM bt_hist_operaciones      WHERE bt_id = :id"), {"id": bt_id})
    conn.execute(text("DELETE FROM bt_hist_estrategias      WHERE id    = :id"), {"id": bt_id})
    conn.commit()
    print(f"[RESET] Instancia '{nombre}' eliminada.")


# ── Escritura final ───────────────────────────────────────────────────────────

def escribir_resultados(engine, bt_id: int, pm: BtPositionManager, verbose: bool):
    """
    INSERT masivo usando psycopg2 execute_values — una llamada por tabla.
    Mucho mas rapido que executemany para miles de filas.
    """
    from psycopg2.extras import execute_values
    import psycopg2.extras

    ops        = pm.operaciones_cerradas
    metricas   = pm.metricas_diarias
    candidatos = pm.candidatos
    resumen    = pm.resumen_metricas()

    log(f"Escribiendo {len(ops)} operaciones, {len(metricas)} dias, "
        f"{len(candidatos)} candidatos...", verbose)

    raw = engine.raw_connection()
    try:
        cur = raw.cursor()

        # Operaciones
        if ops:
            rows_ops = [
                (
                    op["bt_id"], op["ticker"], op["lado"],
                    op["fecha_entrada"], op["precio_entrada"],
                    op["cantidad"], op["capital_entrada"],
                    op.get("stop_loss"), op.get("take_profit"),
                    op.get("score_entrada"),
                    psycopg2.extras.Json(op.get("detalle_entrada") or {}),
                    op.get("fecha_salida"), op.get("precio_salida"),
                    op.get("pnl"), op.get("pnl_pct"),
                    op.get("motivo_salida"), op.get("dias_abierta"),
                )
                for op in ops
            ]
            execute_values(cur, """
                INSERT INTO bt_hist_operaciones
                    (bt_id, ticker, lado, fecha_entrada, precio_entrada,
                     cantidad, capital_entrada, stop_loss, take_profit,
                     score_entrada, detalle_entrada,
                     fecha_salida, precio_salida, pnl, pnl_pct,
                     motivo_salida, dias_abierta)
                VALUES %s
            """, rows_ops, page_size=500)

        # Metricas diarias
        if metricas:
            rows_met = [
                (
                    m["bt_id"], m["fecha"], m["capital_total"],
                    m["cash_disponible"], m["capital_inmovilizado"],
                    m["posiciones_abiertas"], m["operaciones_cerradas_dia"],
                    m["pnl_dia"], m["retorno_acumulado_pct"], m["drawdown_pct"],
                )
                for m in metricas
            ]
            execute_values(cur, """
                INSERT INTO bt_hist_metricas_diarias
                    (bt_id, fecha, capital_total, cash_disponible,
                     capital_inmovilizado, posiciones_abiertas,
                     operaciones_cerradas_dia, pnl_dia,
                     retorno_acumulado_pct, drawdown_pct)
                VALUES %s
                ON CONFLICT (bt_id, fecha) DO NOTHING
            """, rows_met, page_size=500)

        # Candidatos
        if candidatos:
            rows_cand = [
                (
                    c["bt_id"], c["fecha"], c["ticker"], c["score"],
                    c["entro"], c.get("motivo_skip"), c.get("precio_apertura"),
                )
                for c in candidatos
            ]
            execute_values(cur, """
                INSERT INTO bt_hist_candidatos
                    (bt_id, fecha, ticker, score, entro, motivo_skip, precio_apertura)
                VALUES %s
                ON CONFLICT (bt_id, fecha, ticker) DO NOTHING
            """, rows_cand, page_size=500)

        # Actualizar resumen en bt_hist_estrategias
        cur.execute("""
            UPDATE bt_hist_estrategias SET
                capital_final       = %s,
                dias_habiles        = %s,
                retorno_total_pct   = %s,
                drawdown_max_pct    = %s,
                win_rate_pct        = %s,
                profit_factor       = %s,
                total_operaciones   = %s,
                dias_promedio_pos   = %s,
                sharpe_simplificado = %s,
                estado              = 'completado',
                ejecutado_en        = NOW()
            WHERE id = %s
        """, (
            pm.capital_total(),
            len(metricas),
            resumen["retorno_total_pct"],
            resumen["drawdown_max_pct"],
            resumen["win_rate_pct"],
            resumen["profit_factor"],
            resumen["total_operaciones"],
            resumen["dias_promedio_pos"],
            resumen["sharpe_simplificado"],
            bt_id,
        ))

        raw.commit()
        cur.close()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()

    log("Escritura completada.", verbose)


# ── Runners por estrategia ────────────────────────────────────────────────────

def _dias_habiles_entre(fecha_a: date, fecha_b: date, trading_days: list) -> int:
    return sum(1 for d in trading_days if fecha_a <= d <= fecha_b)


def run_tech_sector(bt_id: int, loader: BtDataLoader, logica: str,
                    dry_run: bool, verbose: bool) -> BtPositionManager:
    """Runner para TECH_SECTOR_v1 y COMBO_v1."""
    pm = BtPositionManager(bt_id, logica, sc.CAPITAL_INICIAL)
    pm.init_sectores(SECTORES_TECH)

    es_combo = (logica == "combo_tech_candle")
    candle_scores: dict[str, float] = {}

    for fecha in loader.trading_days:
        ops_cerradas_dia = 0

        # ── a) CIERRES ──────────────────────────────────────────────────────
        indicadores_hoy = {
            r["ticker"]: r
            for r in loader.get_indicadores_fecha(fecha)
        }
        precios_hoy = loader.get_close(fecha)

        for ticker in list(pm.posiciones_abiertas.keys()):
            pos = pm.posiciones_abiertas[ticker]
            close = precios_hoy.get(ticker)
            if close is None:
                continue

            ind = indicadores_hoy.get(ticker)
            if ind is None:
                continue

            score_hoy, _ = sc.calcular_score_tecnico(ind)
            atr14 = float(ind.get("atr14", 0) or 0)
            p_entrada = pos["precio_entrada"]
            sl = p_entrada - sc.ATR_MULT_SL * atr14
            tp = p_entrada + sc.ATR_MULT_TP * atr14

            motivo = None
            precio_sal = close

            if score_hoy <= sc.SCORE_SALIDA_TECH:
                motivo = "SCORE_DEGRADADO"
            elif close <= sl:
                motivo = "STOP_LOSS_ATR"
                precio_sal = sl
            elif close >= tp:
                motivo = "TAKE_PROFIT_ATR"
                precio_sal = tp

            if motivo:
                dias_hab = _dias_habiles_entre(pos["fecha_entrada"], fecha, loader.trading_days)
                pm.cerrar(ticker, fecha, precio_sal, motivo, dias_hab)
                ops_cerradas_dia += 1
                if verbose:
                    log(f"  CIERRE {ticker} {motivo} pnl={precio_sal - p_entrada:.2f}", verbose)

        # ── b) ENTRADAS ──────────────────────────────────────────────────────
        if es_combo:
            candle_scores = loader.get_candle_score_5d(fecha)

        candidatos = []
        for r in loader.get_indicadores_fecha(fecha):
            ticker = r["ticker"]
            if ticker in pm.posiciones_abiertas:
                continue
            score, detalle = sc.calcular_score_tecnico(r)
            if score < sc.SCORE_ENTRADA_TECH:
                continue

            if es_combo:
                cs = candle_scores.get(ticker, 0.0)
                if cs <= sc.CANDLE_SCORE_EXCLUIR_COMBO:
                    pm.registrar_candidato(bt_id, fecha, ticker, score, False,
                                          f"candle_score_bajo ({cs:.1f})", r.get("close", 0))
                    continue
            else:
                cs = 0.0

            candidatos.append({
                "ticker":   ticker,
                "score":    score,
                "cs":       cs,
                "detalle":  detalle,
                "close":    float(r.get("close", 0) or 0),
                "atr14":    float(r.get("atr14",  0) or 0),
                "sector":   r.get("sector"),
            })

        # Rankear: score DESC, luego candle_score DESC para COMBO
        candidatos.sort(key=lambda x: (-x["score"], -x["cs"]))

        for c in candidatos:
            ticker = c["ticker"]
            sector = c["sector"]
            close  = c["close"]
            atr14  = c["atr14"]

            if not sector:
                pm.registrar_candidato(bt_id, fecha, ticker, c["score"], False,
                                       "sin_sector", close)
                continue

            budget = pm.capital_disponible_sector(sector)
            capital_trade = round(budget * sc.POSITION_PCT, 2)
            if capital_trade <= 0:
                pm.registrar_candidato(bt_id, fecha, ticker, c["score"], False,
                                       "sin_capital_sector", close)
                continue

            puede, motivo_skip = pm.puede_entrar_sector(sector, capital_trade)
            if not puede:
                pm.registrar_candidato(bt_id, fecha, ticker, c["score"], False, motivo_skip, close)
                continue

            sl = close - sc.ATR_MULT_SL * atr14
            tp = close + sc.ATR_MULT_TP * atr14

            pm.abrir(ticker, fecha, close, sl, c["score"], c["detalle"],
                     atr14=atr14, take_profit=tp, sector=sector)
            pm.registrar_candidato(bt_id, fecha, ticker, c["score"], True, "", close)

            if verbose:
                log(f"  ENTRADA {ticker} [{sector}] score={c['score']} close={close:.2f}", verbose)

        # ── c) METRICAS ──────────────────────────────────────────────────────
        pm.registrar_dia(fecha, precios_hoy, ops_cerradas_dia)

    # Cerrar posiciones abiertas al fin del periodo
    precios_fin = loader.get_close(loader.trading_days[-1])
    pm.cerrar_todas_fin_periodo(loader.trading_days[-1], precios_fin, loader.trading_days)

    return pm


def run_smc(bt_id: int, loader: BtDataLoader,
            dry_run: bool, verbose: bool) -> BtPositionManager:
    """Runner para SMC_v1."""
    pm = BtPositionManager(bt_id, "smc_estructura", sc.CAPITAL_INICIAL)

    for fecha in loader.trading_days:
        ops_cerradas_dia = 0
        precios_hoy = loader.get_close(fecha)

        # ── a) CIERRES ──────────────────────────────────────────────────────
        fms_hoy = loader.get_fms_estado(fecha)

        for ticker in list(pm.posiciones_abiertas.keys()):
            pos    = pm.posiciones_abiertas[ticker]
            close  = precios_hoy.get(ticker)
            estado = fms_hoy.get(ticker, {})

            if close is None:
                continue

            sl_actual = pos.get("stop_loss") or 0.0
            dias_hab  = _dias_habiles_entre(pos["fecha_entrada"], fecha, loader.trading_days)

            # Actualizar trailing SL antes de evaluar salida
            dist_sl_pct = float(estado.get("dist_sl_10_pct", 0) or 0)
            if dist_sl_pct > 0:
                nuevo_sl = sc.swing_low_precio(close, dist_sl_pct)
                pm.actualizar_sl(ticker, nuevo_sl)
                sl_actual = pm.posiciones_abiertas.get(ticker, {}).get("stop_loss", sl_actual)

            motivo    = None
            precio_sal = close

            # P1: Trailing SL
            if sl_actual > 0 and close <= sl_actual:
                motivo    = "TRAILING_SL"
                precio_sal = sl_actual
            # P2: CHoCH bajista
            elif int(estado.get("choch_bear_10", 0) or 0) == 1:
                motivo = "CHOCH_BEAR"
            # P3: Estructura rota
            elif int(estado.get("estructura_10", 0) or 0) == -1:
                motivo = "ESTRUCTURA_ROTA"
            # P4: Time stop
            elif dias_hab >= sc.DIAS_MAX_SMC:
                motivo = "TIME_STOP_20D"

            if motivo:
                pm.cerrar(ticker, fecha, precio_sal, motivo, dias_hab)
                ops_cerradas_dia += 1
                if verbose:
                    pnl_ap = precio_sal - pos["precio_entrada"]
                    log(f"  CIERRE {ticker} {motivo} pnl_aprox={pnl_ap:.2f}", verbose)

        # ── b) ENTRADAS ──────────────────────────────────────────────────────
        candidatos_smc = loader.get_features_smc_entrada(fecha, sc.LOOKBACK_DIAS)
        candidatos_validos = []

        for row in candidatos_smc:
            ticker = row["ticker"]
            if ticker in pm.posiciones_abiertas:
                continue
            score, detalle = sc.calcular_score_estructura(row)
            if score < sc.SCORE_ENTRADA_MIN_SMC:
                if score != -1:
                    pm.registrar_candidato(bt_id, fecha, ticker, score, False,
                                           "score_bajo", row.get("close", 0))
                continue
            candidatos_validos.append({
                "ticker":  ticker,
                "score":   score,
                "detalle": detalle,
                "close":   float(row.get("close", 0) or 0),
                "dist_sl": float(row.get("dist_sl_10_pct", 0) or 0),
            })

        # Rankear por score DESC
        candidatos_validos.sort(key=lambda x: -x["score"])

        for c in candidatos_validos:
            ticker   = c["ticker"]
            close    = c["close"]
            dist_sl  = c["dist_sl"]
            sl_precio = sc.swing_low_precio(close, dist_sl)

            capital_trade = round(pm._cash * sc.RIESGO_POR_TRADE_SMC, 2)
            puede, motivo_skip = pm.puede_entrar_smc(capital_trade)
            if not puede:
                pm.registrar_candidato(bt_id, fecha, ticker, c["score"], False, motivo_skip, close)
                continue

            pm.abrir(ticker, fecha, close, sl_precio, c["score"], c["detalle"])
            pm.registrar_candidato(bt_id, fecha, ticker, c["score"], True, "", close)

            if verbose:
                log(f"  ENTRADA {ticker} SMC score={c['score']} close={close:.2f} SL={sl_precio:.2f}", verbose)

        # ── c) METRICAS ──────────────────────────────────────────────────────
        pm.registrar_dia(fecha, precios_hoy, ops_cerradas_dia)

    # Cerrar al fin del periodo
    precios_fin = loader.get_close(loader.trading_days[-1])
    pm.cerrar_todas_fin_periodo(loader.trading_days[-1], precios_fin, loader.trading_days)

    return pm


# ── Imprimir resumen en consola ───────────────────────────────────────────────

def imprimir_resumen(pm: BtPositionManager, estrategia: str, desde: date, hasta: date):
    ops     = pm.operaciones_cerradas
    resumen = pm.resumen_metricas()
    print()
    print("=" * 60)
    print(f"RESUMEN: {estrategia}  [{desde} -> {hasta}]")
    print("=" * 60)
    print(f"  Capital inicial:    $100,000.00")
    print(f"  Capital final:      ${pm.capital_total():>12,.2f}")
    print(f"  Retorno total:      {resumen['retorno_total_pct']:>+8.2f}%")
    print(f"  Drawdown maximo:    {resumen['drawdown_max_pct']:>8.2f}%")
    print(f"  Win rate:           {resumen['win_rate_pct']:>8.2f}%")
    print(f"  Profit factor:      {resumen['profit_factor']:>8.4f}")
    print(f"  Total operaciones:  {resumen['total_operaciones']:>8d}")
    print(f"  Dias prom posicion: {resumen['dias_promedio_pos']:>8.1f} dias habiles")
    print(f"  Sharpe simplif.:    {resumen['sharpe_simplificado']:>8.4f}")

    # Distribucion de motivos de salida
    motivos: dict[str, int] = {}
    for o in ops:
        m = o.get("motivo_salida") or "?"
        motivos[m] = motivos.get(m, 0) + 1
    if motivos:
        print()
        print("  Motivos de salida:")
        for m, n in sorted(motivos.items(), key=lambda x: -x[1]):
            print(f"    {m:<25s} {n:>4d}")
    print("=" * 60)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Motor de backtesting historico para TECH_SECTOR_v1, COMBO_v1, SMC_v1"
    )
    parser.add_argument("--estrategia", required=True,
                        choices=list(ESTRATEGIAS.keys()),
                        help="Estrategia a simular")
    parser.add_argument("--desde", required=True,
                        help="Fecha de inicio YYYY-MM-DD")
    parser.add_argument("--hasta", required=True,
                        help="Fecha de fin YYYY-MM-DD")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Simula sin escribir a DB")
    parser.add_argument("--verbose",  action="store_true",
                        help="Log detallado por dia")
    parser.add_argument("--reset",    action="store_true",
                        help="Borra resultados anteriores y vuelve a correr")
    args = parser.parse_args()

    desde    = date.fromisoformat(args.desde)
    hasta    = date.fromisoformat(args.hasta)
    estrategia = args.estrategia
    logica     = ESTRATEGIAS[estrategia]["logica"]
    dry_run    = args.dry_run
    verbose    = args.verbose or dry_run

    engine = get_engine()

    # ── RESET ────────────────────────────────────────────────────────────────
    if args.reset:
        with engine.connect() as conn:
            reset_instancia(conn, estrategia, desde, hasta)

    # ── VERIFICAR SCHEMA ─────────────────────────────────────────────────────
    with engine.connect() as conn:
        existe = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'bt_hist_estrategias' AND table_schema = 'public'
        """)).scalar()
    if not existe:
        print("ERROR: tablas bt_hist_* no existen. Correr bt_create_schema.py primero.")
        sys.exit(1)

    # ── BULK LOAD ────────────────────────────────────────────────────────────
    print()
    log(f"Iniciando BT: {estrategia}  [{desde} -> {hasta}]  dry_run={dry_run}", True)
    loader = BtDataLoader(engine, desde, hasta, logica)
    loader.cargar()

    n_tickers = len(loader.sector_map) if logica in ("tecnico_sectorial", "combo_tech_candle") \
                else len(loader.get_close(loader.trading_days[0]))

    # ── REGISTRAR INSTANCIA ──────────────────────────────────────────────────
    bt_id = -1
    if not dry_run:
        with engine.connect() as conn:
            bt_id = registrar_instancia(conn, estrategia, desde, hasta, n_tickers, dry_run)
        log(f"Instancia registrada: bt_id={bt_id}", True)
    else:
        bt_id = 0
        log("DRY RUN: no se escribe a DB", True)

    # ── LOOP ─────────────────────────────────────────────────────────────────
    log(f"Iniciando loop sobre {len(loader.trading_days)} dias habiles...", True)
    t0 = datetime.now()

    if logica in ("tecnico_sectorial", "combo_tech_candle"):
        pm = run_tech_sector(bt_id, loader, logica, dry_run, verbose)
    else:
        pm = run_smc(bt_id, loader, dry_run, verbose)

    elapsed = (datetime.now() - t0).total_seconds()
    log(f"Loop completado en {elapsed:.1f}s", True)

    # ── ESCRITURA ────────────────────────────────────────────────────────────
    if not dry_run:
        escribir_resultados(engine, bt_id, pm, verbose=True)
    else:
        log("DRY RUN: resultados calculados en memoria, sin escritura.", True)

    # ── RESUMEN ──────────────────────────────────────────────────────────────
    imprimir_resumen(pm, estrategia, desde, hasta)


if __name__ == "__main__":
    main()
