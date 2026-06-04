"""
bot_ml.py -- Bot 1 Alpaca (Plan B, Tarea 16): estrategia ML_SCANNER_v1.

"El bot solo opera": lee la masticada senales_bot_diaria (Railway), decide con el
cerebro COMPARTIDO src/strategies/ml_scanner.py (homologable a FT_ML_SCANNER_v1)
y ejecuta via src/trading/ejecucion_bot.py (cuenta Alpaca "" / tablas
posiciones_bot / operaciones_bot).

Entorno: carga .env + .env.local (override) -> DATABASE_URL=Railway (masticada +
estado del bot) y credenciales Alpaca. En GitHub Actions vienen de secrets.

Uso:
    python scripts/alpaca/bot_ml.py --dry-run    (no envia ordenes ni escribe DB)
    python scripts/alpaca/bot_ml.py              (ejecucion real -- requiere creds)
"""

import sys
import os
import argparse
from datetime import datetime

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

from src.strategies.ml_scanner import ConfigML, evaluar_cierres_ml, evaluar_entradas_ml
from src.trading import senales_adapter as sa, ejecucion_bot as eb
from src.indicators.earnings_filter import tickers_a_cerrar_hoy, tickers_a_bloquear_entrada


CFG = ConfigML(
    nombre="ML_SCANNER_v1",
    score_min=65,
    nivel_min="COMPRA_FUERTE",
    max_posiciones=5,
    max_deploy_pct=0.80,
    riesgo_por_trade=0.15,
    sl_pct=0.05,
    tp_pct=0.10,
)
BCFG = eb.BotAlpacaConfig(
    nombre="ML_SCANNER_v1",
    account_suffix="",
    tabla_pos="posiciones_bot",
    tabla_oper="operaciones_bot",
)
CAPITAL_BASE = 100_000.0   # fallback si no hay cuenta Alpaca (dry-run offline)


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _estado_capital():
    """(capital_actual, cash_desplegable, capital_inmovilizado) desde Alpaca; fallback baseline."""
    try:
        from src.trading import alpaca_client
        acc    = alpaca_client.get_account_info(BCFG.account_suffix)
        equity = float(acc["equity"])
        cash   = float(acc["cash"])
    except Exception as e:
        log(f"  [WARN] Sin cuenta Alpaca ({e}); uso baseline ${CAPITAL_BASE:,.0f}.")
        equity, cash = CAPITAL_BASE, CAPITAL_BASE
    inmovilizado = max(0.0, equity - cash)
    cash_desp    = min(cash, max(0.0, equity * CFG.max_deploy_pct - inmovilizado))
    return equity, cash_desp, inmovilizado


def run(dry_run: bool = False, ignore_frescura: bool = False):
    sep = "-" * 60
    log(sep)
    log(f"BOT ALPACA 1 -- ML_SCANNER_v1 {'[DRY RUN]' if dry_run else '[REAL]'}")
    log(sep)

    filas, fecha = sa.leer_senales()
    if not filas:
        log("[ERROR] senales_bot_diaria vacia. Abort (no se opera sin senales).")
        return
    log(f"Masticada: {len(filas)} tickers | fecha {fecha}")

    # Guard de frescura: no operar sobre senales rancias (sin push del dia)
    fecha_esp = sa.fecha_esperada_hoy()
    if not ignore_frescura and not sa.senales_frescas(fecha, fecha_esp):
        log(f"[GUARD FRESCURA] masticada={fecha} != esperada={fecha_esp}. "
            f"NO se opera (sin push del dia / datos rancios). --ignore-frescura para forzar.")
        return
    log(f"  Guard frescura: {'IGNORADO' if ignore_frescura else f'OK (esperada {fecha_esp})'}")

    precios     = sa.construir_precios(filas)
    scanner_map = {
        r["ticker"]: {"nivel": r["alert_nivel"],
                      "score": float(r["alert_score"]) if r["alert_score"] is not None else 0.0}
        for r in filas
    }
    posiciones = eb.get_posiciones_abiertas(BCFG)
    log(f"Posiciones abiertas: {len(posiciones)}")

    # ── Cierres ──
    log("CIERRES:")
    tickers_pos     = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    a_cerrar = evaluar_cierres_ml(posiciones, precios, earnings_cierre, scanner_map, CFG, log=log)
    if not a_cerrar:
        log("  Sin cierres.")
    else:
        eb.aplicar_cierres(BCFG, a_cerrar, dry_run=dry_run, log=log)
        if not dry_run:
            posiciones = eb.get_posiciones_abiertas(BCFG)

    # ── Entradas ──
    log("ENTRADAS:")
    ml_senales = sa.construir_ml_senales(filas, CFG.nivel_min, CFG.score_min, fecha)
    log(f"  Senales {CFG.nivel_min} score>={CFG.score_min:.0f}: {len(ml_senales)}")
    bloqueados = set(tickers_a_bloquear_entrada([s["ticker"] for s in ml_senales]) or {})
    if bloqueados:
        log(f"  Bloqueados por earnings: {bloqueados}")

    equity, cash_desp, inmovilizado = _estado_capital()
    log(f"  Capital: equity=${equity:,.0f} | cash_desplegable=${cash_desp:,.0f}")

    a_abrir = evaluar_entradas_ml(
        posiciones, ml_senales, precios, bloqueados,
        cash=cash_desp, capital_actual=equity, capital_inmovilizado=inmovilizado,
        cfg=CFG, log=log,
    )
    if not a_abrir:
        log("  Sin entradas.")
    else:
        eb.aplicar_entradas(BCFG, a_abrir, dry_run=dry_run, log=log)

    log("Completado.")
    log(sep)


def main():
    ap = argparse.ArgumentParser(description="Bot Alpaca 1 -- ML_SCANNER_v1")
    ap.add_argument("--dry-run", action="store_true", help="No envia ordenes ni escribe DB")
    ap.add_argument("--ignore-frescura", action="store_true",
                    help="Omite el guard de frescura (operar con la masticada actual)")
    args = ap.parse_args()
    run(dry_run=args.dry_run, ignore_frescura=args.ignore_frescura)


if __name__ == "__main__":
    main()
