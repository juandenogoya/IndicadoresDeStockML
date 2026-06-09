"""
bot_options.py -- Bot 3 Alpaca (Plan B, Tarea 16): TECH_SECTOR_OPTIONS_v2.

Lee la masticada senales_bot_diaria, decide con el cerebro COMPARTIDO
src/strategies/sectorial.py (usar_opciones=True -> gate PCR_VOL, homologable a
FT_TECH_SECTOR_OPTIONS_v2) y ejecuta via ejecucion_bot (cuenta "_3" / tablas
posiciones_bot_candle / operaciones_bot_candle; nombres heredados del reset).

v1 (bot_tech_sector) vs v2 (este) aisla el aporte del dato de opciones.

Uso:
    python scripts/alpaca/bot_options.py --dry-run
    python scripts/alpaca/bot_options.py
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

from src.strategies.sectorial import (
    ConfigSectorial, evaluar_cierres_sectorial, evaluar_entradas_sectorial,
)
from src.trading import senales_adapter as sa, ejecucion_bot as eb
from src.indicators.earnings_filter import tickers_a_cerrar_hoy, tickers_a_bloquear_entrada


SECTORES_ACTIVOS = [
    "Technology", "Consumer Cyclical", "Financial Services",
    "Industrials", "Basic Materials", "Healthcare",
    "Energy", "Communication Services", "Consumer Defensive",
]
CAPITAL_TOTAL  = 100_000.0
SECTOR_BUDGET  = round(CAPITAL_TOTAL / len(SECTORES_ACTIVOS), 2)
MAX_POS_SECTOR = 5
POSITION_SIZE  = round(SECTOR_BUDGET / MAX_POS_SECTOR, 2)

CFG = ConfigSectorial(
    nombre="TECH_SECTOR_OPTIONS_v2",
    sectores_activos=SECTORES_ACTIVOS,
    sector_budget=SECTOR_BUDGET,
    max_pos_sector=MAX_POS_SECTOR,
    position_size=POSITION_SIZE,
    score_entrada=4.0, score_salida=3.5,
    atr_mult_sl=2.0, atr_mult_tp=4.0,
    usar_opciones=True,
    pcr_score_entrada_min=2,
    pcr_score_salida_max=0,
)
BCFG = eb.BotAlpacaConfig(
    nombre="TECH_SECTOR_OPTIONS_v2",
    account_suffix="_3",
    tabla_pos="posiciones_bot_candle",
    tabla_oper="operaciones_bot_candle",
)


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(dry_run: bool = False, ignore_frescura: bool = False):
    sep = "-" * 60
    log(sep)
    log(f"BOT ALPACA 3 -- TECH_SECTOR_OPTIONS_v2 {'[DRY RUN]' if dry_run else '[REAL]'}")
    log(f"  {len(SECTORES_ACTIVOS)} sectores x ${SECTOR_BUDGET:,.2f} | "
        f"pos ${POSITION_SIZE:,.2f} | gate PCR>={CFG.pcr_score_entrada_min}/3")
    log(sep)

    filas, fecha = sa.leer_senales()
    if not filas:
        log("[ERROR] senales_bot_diaria vacia. Abort.")
        return
    log(f"Masticada: {len(filas)} tickers | fecha {fecha}")

    fecha_esp = sa.fecha_esperada_hoy()
    if not ignore_frescura and not sa.senales_frescas(fecha, fecha_esp):
        log(f"[GUARD FRESCURA] masticada={fecha} != esperada={fecha_esp}. "
            f"NO se opera (sin push del dia / datos rancios). --ignore-frescura para forzar.")
        return
    log(f"  Guard frescura: {'IGNORADO' if ignore_frescura else f'OK (esperada {fecha_esp})'}")

    indicadores     = filas
    precios         = sa.construir_precios(filas)
    sec_por_ticker  = sa.sector_por_ticker(filas)
    pcr_map         = sa.construir_pcr_map(filas)
    indicadores_map = {r["ticker"]: r for r in filas}
    posiciones      = eb.get_posiciones_abiertas(BCFG, sector_por_ticker=sec_por_ticker)
    n_pcr_validos   = sum(1 for v in pcr_map.values() if v.get("valido"))
    log(f"Posiciones abiertas: {len(posiciones)} | PCR validos: {n_pcr_validos}")

    # ── Cierres ──
    log("CIERRES:")
    tickers_pos     = [p["ticker"] for p in posiciones]
    earnings_cierre = tickers_a_cerrar_hoy(tickers_pos) if tickers_pos else {}
    a_cerrar, retenidos = evaluar_cierres_sectorial(
        posiciones, precios, earnings_cierre, indicadores_map, CFG,
        pcr_map=pcr_map, log=log,
    )
    tickers_cerrados = {c["ticker"] for c in a_cerrar}
    if retenidos:
        log(f"  Retenciones por opciones: {retenidos}")
    if not a_cerrar:
        log("  Sin cierres.")
    else:
        eb.aplicar_cierres(BCFG, a_cerrar, dry_run=dry_run, log=log)
        if not dry_run:
            posiciones = eb.get_posiciones_abiertas(BCFG, sector_por_ticker=sec_por_ticker)

    # ── Entradas ──
    log("ENTRADAS POR SECTOR:")
    bloqueados = set(tickers_a_bloquear_entrada([i["ticker"] for i in indicadores]) or {})
    if bloqueados:
        log(f"  Bloqueados por earnings: {len(bloqueados)} tickers")
    a_abrir, _ = evaluar_entradas_sectorial(
        posiciones, indicadores, precios, bloqueados, tickers_cerrados, CFG,
        pcr_map=pcr_map, log=log,
    )
    if not a_abrir:
        log("  Sin entradas.")
    else:
        eb.aplicar_entradas(BCFG, a_abrir, dry_run=dry_run, log=log)

    # Resumen por Telegram (incluye retenciones por opciones; no-op en dry-run)
    eb.notificar_telegram(BCFG, a_abrir, a_cerrar, retenidos=retenidos,
                          fecha=fecha, dry_run=dry_run, log=log)

    log("Completado.")
    log(sep)


def main():
    ap = argparse.ArgumentParser(description="Bot Alpaca 3 -- TECH_SECTOR_OPTIONS_v2")
    ap.add_argument("--dry-run", action="store_true", help="No envia ordenes ni escribe DB")
    ap.add_argument("--ignore-frescura", action="store_true",
                    help="Omite el guard de frescura (operar con la masticada actual)")
    args = ap.parse_args()
    run(dry_run=args.dry_run, ignore_frescura=args.ignore_frescura)


if __name__ == "__main__":
    main()
