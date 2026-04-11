"""
30_trading_bot.py
Bot de trading algoritmico sobre Alpaca (paper/live).
Lee senales del scanner ML y ejecuta ordenes.

Uso:
    python scripts/30_trading_bot.py                  # modo normal
    python scripts/30_trading_bot.py --dry-run        # simula sin enviar ordenes
    python scripts/30_trading_bot.py --solo-cierres   # solo evalua cierres
    python scripts/30_trading_bot.py --init           # inicializa tablas en DB
    python scripts/30_trading_bot.py --status         # muestra estado de cuenta y posiciones

Filtros configurables en .env.local:
    BOT_MAX_POSICIONES=5
    BOT_RIESGO_POR_TRADE=0.02
    BOT_SCORE_MIN_COMPRA=65
    BOT_NIVEL_MIN_COMPRA=COMPRA_FUERTE
    BOT_STOP_LOSS_PCT=0.05
    BOT_TAKE_PROFIT_PCT=0.10
    ALPACA_PAPER=true
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cargar variables de entorno — ROOT del proyecto
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    env_main  = os.path.join(ROOT, ".env")
    env_local = os.path.join(ROOT, ".env.local")
    if os.path.exists(env_main):
        load_dotenv(env_main)
    if os.path.exists(env_local):
        load_dotenv(env_local, override=True)  # .env.local sobreescribe .env
except ImportError:
    pass

from src.trading import alpaca_client, portfolio, strategy, risk


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cmd_status():
    """Muestra estado de cuenta y posiciones abiertas."""
    log("Consultando cuenta Alpaca...")
    cuenta = alpaca_client.get_account_info()
    modo   = "PAPER" if cuenta["paper"] else "LIVE"

    print(f"""
  ================================================
    CUENTA ALPACA — {modo}
  ================================================
    Equity          : ${cuenta['equity']:,.2f}
    Buying Power    : ${cuenta['buying_power']:,.2f}
    Cash            : ${cuenta['cash']:,.2f}
    Portfolio Value : ${cuenta['portfolio_value']:,.2f}
  ================================================
""")
    posiciones = alpaca_client.get_open_positions()
    if not posiciones:
        print("  Sin posiciones abiertas.")
    else:
        print(f"  {len(posiciones)} posicion(es) abierta(s):\n")
        for p in posiciones:
            signo = "+" if p["unrealized_pl"] >= 0 else ""
            print(f"    {p['ticker']:6s}  qty={p['qty']:.0f}  "
                  f"entrada=${p['avg_entry']:.2f}  "
                  f"actual=${p['current_price']:.2f}  "
                  f"PnL={signo}{p['unrealized_pl']:.2f} "
                  f"({signo}{p['unrealized_plpc']*100:.1f}%)")


def cmd_init():
    """Inicializa tablas del bot en Railway."""
    log("Inicializando tablas del bot...")
    portfolio.init_tabla()
    log("OK.")


def cmd_run(dry_run: bool = False, solo_cierres: bool = False, usar_filtro_mtf: bool = False):
    """Ciclo principal: evalua cierres y entradas."""

    separador = "=" * 60
    log(separador)
    log(f"  TRADING BOT — {'DRY RUN' if dry_run else 'EJECUCION REAL'}")
    log(f"  Filtro MTF : {'SI' if usar_filtro_mtf else 'NO'}")
    log(f"  Score min  : {risk.SCORE_MIN_COMPRA}")
    log(f"  Nivel min  : {risk.NIVEL_MIN_COMPRA}")
    log(f"  Max posic. : {risk.MAX_POSICIONES}")
    log(f"  Distribucion: {risk.MODO_DISTRIBUCION}")
    log(f"  Exposicion : {risk.MAX_EXPOSICION*100:.0f}% max")
    log(f"  Stop Loss  : {risk.STOP_LOSS_PCT*100:.0f}%")
    log(f"  Take Profit: {risk.TAKE_PROFIT_PCT*100:.0f}%")
    log(separador)

    # ── 1. Info de cuenta ──────────────────────────────────────
    cuenta = alpaca_client.get_account_info()
    equity = cuenta["equity"]
    log(f"  Equity: ${equity:,.2f} | Buying Power: ${cuenta['buying_power']:,.2f}")

    # ── 2. Posiciones abiertas en DB ───────────────────────────
    posiciones_db = portfolio.get_posiciones_abiertas()
    log(f"  Posiciones abiertas en DB: {len(posiciones_db)}")

    # ── 3. Evaluar cierres ─────────────────────────────────────
    log("\n  [CIERRES]")
    a_cerrar = strategy.evaluar_cierres(posiciones_db, usar_filtro_mtf)

    if not a_cerrar:
        log("  Sin posiciones para cerrar.")
    else:
        for c in a_cerrar:
            ticker = c["ticker"]
            motivo = c["motivo"]
            precio = c["precio_cierre"]
            log(f"  CERRAR {ticker} @ ${precio:.2f} — {motivo}")

            if not dry_run:
                try:
                    alpaca_client.close_position(ticker)
                    resultado = portfolio.registrar_cierre(c["id"], precio, motivo)
                    signo = "+" if resultado["pnl"] >= 0 else ""
                    log(f"    -> PnL: {signo}${resultado['pnl']:.2f} ({signo}{resultado['pnl_pct']*100:.1f}%)")
                except Exception as e:
                    log(f"    ERROR cerrando {ticker}: {e}")
            else:
                log("    [DRY RUN] orden no enviada.")

    if solo_cierres:
        log("\n  Modo solo-cierres. Fin.")
        return

    # Actualizar posiciones tras cierres
    posiciones_db = portfolio.get_posiciones_abiertas()

    # ── 4. Evaluar entradas ────────────────────────────────────
    log("\n  [ENTRADAS]")
    a_abrir = strategy.evaluar_entradas(posiciones_db, equity, usar_filtro_mtf)

    if not a_abrir:
        log("  Sin senales validas para abrir posicion hoy.")
    else:
        for entrada in a_abrir:
            ticker = entrada["ticker"]
            qty    = entrada["qty"]
            precio = entrada["precio"]
            score  = entrada["score"]
            nivel  = entrada["nivel"]
            t1w    = entrada.get("tendencia_1w", "-")
            t1m    = entrada.get("tendencia_1m", "-")

            capital = entrada.get("capital", precio * qty)
            pct     = entrada.get("pct_equity", capital / equity * 100)
            log(f"  COMPRAR {ticker}  qty={qty}  precio=${precio:.2f}  "
                f"capital=${capital:,.0f} ({pct:.1f}%)  score={score:.0f}  nivel={nivel}")
            log(f"    SL=${entrada['stop_loss']:.2f}  TP=${entrada['take_profit']:.2f}")

            if not dry_run:
                try:
                    orden = alpaca_client.place_market_order(ticker, qty, "buy")
                    pos_id = portfolio.registrar_entrada(
                        ticker          = ticker,
                        precio_entrada  = precio,
                        qty             = qty,
                        stop_loss       = entrada["stop_loss"],
                        take_profit     = entrada["take_profit"],
                        score_entrada   = score,
                        nivel_entrada   = nivel,
                        tendencia_1w    = entrada.get("tendencia_1w"),
                        tendencia_1m    = entrada.get("tendencia_1m"),
                        alpaca_order_id = orden["order_id"],
                    )
                    log(f"    -> Orden enviada: {orden['order_id']} | DB id: {pos_id}")
                except Exception as e:
                    log(f"    ERROR comprando {ticker}: {e}")
            else:
                log("    [DRY RUN] orden no enviada.")

    # ── 5. Notificacion Telegram ───────────────────────────────
    if not dry_run:
        _enviar_resumen_telegram(a_cerrar, a_abrir, equity)

    log("\n  Bot finalizado.")
    log(separador)


def _enviar_resumen_telegram(cierres: list, entradas: list, equity: float):
    """Envia resumen de operaciones del dia por Telegram."""
    try:
        import os, requests
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return

        lineas = ["<b>BOT ALPACA — Resumen del dia</b>"]

        if entradas:
            lineas.append(f"\n<b>COMPRAS ({len(entradas)})</b>")
            for e in entradas:
                capital = e.get("capital", e["precio"] * e["qty"])
                pct     = e.get("pct_equity", capital / equity * 100)
                lineas.append(
                    f"  BUY {e['ticker']}  x{e['qty']}  ${e['precio']:.2f}"
                    f"  (${capital:,.0f} / {pct:.1f}%)"
                    f"  score={e['score']:.0f}"
                )
        else:
            lineas.append("\nSin compras hoy.")

        if cierres:
            lineas.append(f"\n<b>CIERRES ({len(cierres)})</b>")
            for c in cierres:
                lineas.append(f"  SELL {c['ticker']}  {c['motivo']}  @ ${c['precio_cierre']:.2f}")

        msg = "\n".join(lineas)
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log(f"  [WARN] Telegram no enviado: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bot de trading Alpaca")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Simula sin enviar ordenes reales")
    parser.add_argument("--solo-cierres", action="store_true",
                        help="Solo evalua cierres, no abre posiciones nuevas")
    parser.add_argument("--filtro-mtf",   action="store_true",
                        help="Activa filtro MTF (requiere tendencia 1W y 1M alcistas)")
    parser.add_argument("--init",         action="store_true",
                        help="Inicializa tablas del bot en la DB")
    parser.add_argument("--status",       action="store_true",
                        help="Muestra estado de cuenta y posiciones")
    args = parser.parse_args()

    if args.init:
        cmd_init()
    elif args.status:
        cmd_status()
    else:
        cmd_run(
            dry_run       = args.dry_run,
            solo_cierres  = args.solo_cierres,
            usar_filtro_mtf = args.filtro_mtf,
        )


if __name__ == "__main__":
    main()
