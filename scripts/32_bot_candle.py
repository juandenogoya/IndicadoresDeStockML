"""
32_bot_candle.py
Bot de trading — Estrategia SMC Estructura (cuenta Alpaca #3).

Estrategia: Smart Money Concepts (CHoCH / BOS)
    Entrada : CHoCH o BOS alcista en ultimos ~10 dias habiles
              + estructura sostenida (estructura_10 >= 0)
              + vela alcista de confirmacion hoy
              + SL estructural <= 8% del precio
    Salida  : Estructura pura (Filosofia B — sin TP fijo, sin SL price-based)
              CHoCH bajista | estructura_10=-1 | Time stop 20 dias

Uso:
    python scripts/32_bot_candle.py             # ejecucion real
    python scripts/32_bot_candle.py --dry-run   # simula sin enviar ordenes
    python scripts/32_bot_candle.py --status    # estado cuenta
    python scripts/32_bot_candle.py --init      # inicializa tablas DB
    python scripts/32_bot_candle.py --liquidar  # cierra todas las posiciones abiertas (cambio estrategia)
"""

import sys
import os
import argparse
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from src.trading import alpaca_client, risk
from src.indicators.earnings_filter import tickers_a_cerrar_hoy, tickers_a_bloquear_entrada
from src.trading.strategy_structure import (
    evaluar_entradas_estructura,
    evaluar_cierres_estructura,
    SCORE_ENTRADA,
    SCORE_MAXIMO,
    MAX_SL_DISTANCE_PCT,
    DIAS_MAX_POS,
)

SUFFIX            = "_3"
TABLA_POSICIONES  = "posiciones_bot_candle"
TABLA_OPERACIONES = "operaciones_bot_candle"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# Init tablas
# ─────────────────────────────────────────────────────────────

def cmd_init():
    from src.data.database import get_engine
    from sqlalchemy import text
    log("Inicializando tablas bot estructura (SMC)...")
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLA_POSICIONES} (
                id              SERIAL PRIMARY KEY,
                ticker          VARCHAR(20) NOT NULL,
                fecha_entrada   DATE NOT NULL,
                precio_entrada  NUMERIC(10,4) NOT NULL,
                qty             INTEGER NOT NULL,
                stop_loss       NUMERIC(10,4),
                take_profit     NUMERIC(10,4),
                atr_entrada     NUMERIC(10,4),
                score_entrada   NUMERIC(5,2),
                nivel_entrada   VARCHAR(60),
                alpaca_order_id VARCHAR(100),
                estado          VARCHAR(20) DEFAULT 'ABIERTA',
                fecha_cierre    DATE,
                precio_cierre   NUMERIC(10,4),
                pnl             NUMERIC(10,4),
                pnl_pct         NUMERIC(6,4),
                motivo_cierre   VARCHAR(50),
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLA_OPERACIONES} (
                id              SERIAL PRIMARY KEY,
                ticker          VARCHAR(20) NOT NULL,
                fecha_entrada   DATE,
                fecha_cierre    DATE,
                precio_entrada  NUMERIC(10,4),
                precio_cierre   NUMERIC(10,4),
                qty             INTEGER,
                pnl             NUMERIC(10,4),
                pnl_pct         NUMERIC(6,4),
                motivo_cierre   VARCHAR(50),
                score_entrada   NUMERIC(5,2),
                nivel_entrada   VARCHAR(60),
                dias_abierta    INTEGER,
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
    log(f"Tablas {TABLA_POSICIONES} y {TABLA_OPERACIONES} OK.")


# ─────────────────────────────────────────────────────────────
# Helpers DB
# ─────────────────────────────────────────────────────────────

def get_posiciones_abiertas() -> list[dict]:
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, ticker, fecha_entrada, precio_entrada,
                   qty, stop_loss, take_profit, atr_entrada,
                   score_entrada, nivel_entrada
            FROM {TABLA_POSICIONES}
            WHERE estado = 'ABIERTA'
            ORDER BY fecha_entrada
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def registrar_entrada(ticker, precio, qty, sl, tp, atr, score, nivel, order_id) -> int:
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text(f"""
            INSERT INTO {TABLA_POSICIONES}
                (ticker, fecha_entrada, precio_entrada, qty,
                 stop_loss, take_profit, atr_entrada,
                 score_entrada, nivel_entrada, alpaca_order_id, estado)
            VALUES
                (:ticker, :fecha, :precio, :qty,
                 :sl, :tp, :atr,
                 :score, :nivel, :order_id, 'ABIERTA')
            RETURNING id
        """), {
            "ticker": ticker, "fecha": date.today(),
            "precio": precio, "qty": qty,
            "sl": sl, "tp": tp, "atr": atr,
            "score": score, "nivel": nivel,
            "order_id": order_id,
        })
        conn.commit()
        return r.scalar()


def registrar_cierre(pos_id, precio_cierre, motivo) -> dict:
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        pos = conn.execute(text(f"""
            SELECT ticker, fecha_entrada, precio_entrada, qty,
                   score_entrada, nivel_entrada
            FROM {TABLA_POSICIONES} WHERE id = :id
        """), {"id": pos_id}).fetchone()
        if not pos:
            raise ValueError(f"Posicion {pos_id} no encontrada.")

        pnl     = round((precio_cierre - float(pos.precio_entrada)) * pos.qty, 4)
        pnl_pct = round((precio_cierre - float(pos.precio_entrada)) / float(pos.precio_entrada), 4)
        dias_ab = (date.today() - pos.fecha_entrada).days

        conn.execute(text(f"""
            UPDATE {TABLA_POSICIONES} SET
                estado = 'CERRADA', fecha_cierre = :fecha,
                precio_cierre = :precio, pnl = :pnl,
                pnl_pct = :pnl_pct, motivo_cierre = :motivo
            WHERE id = :id
        """), {
            "fecha": date.today(), "precio": precio_cierre,
            "pnl": pnl, "pnl_pct": pnl_pct,
            "motivo": motivo, "id": pos_id,
        })
        conn.execute(text(f"""
            INSERT INTO {TABLA_OPERACIONES}
                (ticker, fecha_entrada, fecha_cierre, precio_entrada, precio_cierre,
                 qty, pnl, pnl_pct, motivo_cierre, score_entrada, nivel_entrada, dias_abierta)
            VALUES
                (:ticker, :f_entrada, :f_cierre, :p_entrada, :p_cierre,
                 :qty, :pnl, :pnl_pct, :motivo, :score, :nivel, :dias)
        """), {
            "ticker":    pos.ticker,
            "f_entrada": pos.fecha_entrada, "f_cierre": date.today(),
            "p_entrada": pos.precio_entrada, "p_cierre": precio_cierre,
            "qty": pos.qty, "pnl": pnl, "pnl_pct": pnl_pct,
            "motivo": motivo, "score": pos.score_entrada,
            "nivel": pos.nivel_entrada, "dias": dias_ab,
        })
        conn.commit()
    return {"pnl": pnl, "pnl_pct": pnl_pct}


# ─────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────

def cmd_status():
    log("Consultando cuenta Alpaca #3 (SMC Estructura)...")
    cuenta = alpaca_client.get_account_info(SUFFIX)
    modo   = "PAPER" if cuenta["paper"] else "LIVE"
    print(f"""
  ================================================
    CUENTA ALPACA #3 — {modo} (Estrategia SMC)
  ================================================
    Equity          : ${cuenta['equity']:,.2f}
    Buying Power    : ${cuenta['buying_power']:,.2f}
    Cash            : ${cuenta['cash']:,.2f}
    Portfolio Value : ${cuenta['portfolio_value']:,.2f}
  ================================================
""")
    posiciones = get_posiciones_abiertas()
    if not posiciones:
        print("  Sin posiciones abiertas.")
    else:
        print(f"  {len(posiciones)} posicion(es) abierta(s):\n")
        for p in posiciones:
            entrada  = float(p["precio_entrada"])
            sl       = float(p["stop_loss"]) if p.get("stop_loss") else 0
            dias     = (date.today() - p["fecha_entrada"]).days
            dist_sl  = round((entrada - sl) / entrada * 100, 1) if sl and entrada else 0
            print(f"    {p['ticker']:6s}  qty={p['qty']}  "
                  f"entrada=${entrada:.2f}  SL=${sl:.2f}({dist_sl:.1f}%)  "
                  f"score={p['score_entrada']}  "
                  f"dias={dias}/{DIAS_MAX_POS}  "
                  f"nivel={p['nivel_entrada']}")


# ─────────────────────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────────────────────

def cmd_run(dry_run: bool = False):
    separador = "=" * 60
    log(separador)
    log(f"  BOT SMC ESTRUCTURA — {'DRY RUN' if dry_run else 'EJECUCION REAL'}")
    log(f"  Entrada    : score >= {SCORE_ENTRADA} / {SCORE_MAXIMO}  SL_max={MAX_SL_DISTANCE_PCT}%")
    log(f"  Salida     : Estructura pura — CHoCH bear | estructura=-1 | Time stop")
    log(f"  Time stop  : {DIAS_MAX_POS} dias")
    log(f"  Max posic. : {risk.MAX_POSICIONES}")
    log(f"  Por trade  : {risk.RIESGO_POR_TRADE*100:.0f}% del equity")
    log(separador)

    cuenta = alpaca_client.get_account_info(SUFFIX)
    equity = cuenta["equity"]
    log(f"  Equity: ${equity:,.2f} | Buying Power: ${cuenta['buying_power']:,.2f}")

    posiciones = get_posiciones_abiertas()
    log(f"  Posiciones abiertas: {len(posiciones)}")

    # ── Paso 0: Earnings exit (prioridad absoluta) ───────────
    log("\n  [EARNINGS EXIT]")
    tickers_open    = [p["ticker"] for p in posiciones]
    earnings_closes = tickers_a_cerrar_hoy(tickers_open)

    if not earnings_closes:
        log("  Sin posiciones con earnings proximos.")
    else:
        for ticker, earnings_date in earnings_closes.items():
            pos = next((p for p in posiciones if p["ticker"] == ticker), None)
            if not pos:
                continue
            try:
                precio = alpaca_client.get_latest_price(ticker, suffix=SUFFIX)
            except Exception:
                precio = float(pos["precio_entrada"])
            motivo = f"EARNINGS_EXIT_{earnings_date}"
            log(f"  CERRAR {ticker} @ ${precio:.2f} — {motivo}")
            if not dry_run:
                try:
                    alpaca_client.close_position(ticker, SUFFIX)
                    resultado = registrar_cierre(pos["id"], precio, motivo)
                    signo = "+" if resultado["pnl"] >= 0 else ""
                    log(f"    -> PnL: {signo}${resultado['pnl']:.2f} ({signo}{resultado['pnl_pct']*100:.1f}%)")
                except Exception as e:
                    log(f"    ERROR: {e}")
            else:
                log("    [DRY RUN] orden no enviada.")

    # Actualizar posiciones post-earnings
    posiciones = get_posiciones_abiertas()

    # ── Paso 1: Cierres ───────────────────────────────────────
    log("\n  [CIERRES]")
    a_cerrar = evaluar_cierres_estructura(posiciones)

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
                    alpaca_client.close_position(ticker, SUFFIX)
                    resultado = registrar_cierre(c["id"], precio, motivo)
                    signo = "+" if resultado["pnl"] >= 0 else ""
                    log(f"    -> PnL: {signo}${resultado['pnl']:.2f} "
                        f"({signo}{resultado['pnl_pct']*100:.1f}%)")
                except Exception as e:
                    log(f"    ERROR: {e}")
            else:
                log("    [DRY RUN] orden no enviada.")

    # Actualizar posiciones post-cierres
    posiciones = get_posiciones_abiertas()

    # ── Paso 2: Entradas ──────────────────────────────────────
    log("\n  [ENTRADAS]")
    a_abrir = evaluar_entradas_estructura(posiciones, equity)

    # Bloquear candidatos con earnings proximos
    tickers_candidatos  = [e["ticker"] for e in a_abrir]
    bloqueados_earnings = tickers_a_bloquear_entrada(tickers_candidatos)
    if bloqueados_earnings:
        for ticker, fecha in bloqueados_earnings.items():
            log(f"  BLOQUEADO {ticker} — earnings {fecha}")
    a_abrir = [e for e in a_abrir if e["ticker"] not in bloqueados_earnings]

    if not a_abrir:
        log("  Sin setups estructurales validos hoy.")
    else:
        for entrada in a_abrir:
            ticker   = entrada["ticker"]
            qty      = entrada["qty"]
            precio   = entrada["precio"]
            score    = entrada["score"]
            capital  = entrada["capital"]
            pct      = entrada["pct_equity"]
            sl       = entrada["stop_loss"]
            dist_sl  = entrada["dist_sl_pct"]
            dist_sh  = entrada["dist_sh_pct"]
            log(f"  COMPRAR {ticker}  qty={qty}  precio=${precio:.2f}  "
                f"capital=${capital:,.0f} ({pct:.1f}%)  score={score:.1f}/{SCORE_MAXIMO}")
            log(f"    Evento={entrada['evento']}  "
                f"estruc={entrada['estructura_10']}  "
                f"SL=${sl:.2f} (-{dist_sl:.1f}%)  "
                f"dist_SH={dist_sh:+.1f}%  "
                f"vol={entrada['vol_spike']}  eng={entrada['eng_bull']}")

            if not dry_run:
                try:
                    orden  = alpaca_client.place_market_order(ticker, qty, "buy", SUFFIX)
                    pos_id = registrar_entrada(
                        ticker   = ticker,
                        precio   = precio,
                        qty      = qty,
                        sl       = sl,
                        tp       = entrada["take_profit"],   # None (sin TP fijo)
                        atr      = entrada["atr"],
                        score    = score,
                        nivel    = entrada["nivel"],
                        order_id = orden["order_id"],
                    )
                    log(f"    -> Orden: {orden['order_id']} | DB id: {pos_id}")
                except Exception as e:
                    log(f"    ERROR: {e}")
            else:
                log("    [DRY RUN] orden no enviada.")

    # ── Telegram ──────────────────────────────────────────────
    if not dry_run:
        _telegram(a_cerrar, a_abrir, equity)

    log("\n  Bot SMC Estructura finalizado.")
    log(separador)


def _telegram(cierres, entradas, equity):
    try:
        import requests
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return

        lineas = ["<b>BOT SMC ESTRUCTURA — Resumen</b>"]

        if entradas:
            lineas.append(f"\n<b>COMPRAS ({len(entradas)})</b>")
            for e in entradas:
                lineas.append(
                    f"  BUY {e['ticker']}  x{e['qty']}  ${e['precio']:.2f}"
                    f"  {e['evento']}  dist_SL={e['dist_sl_pct']:.1f}%"
                    f"  score={e['score']:.1f}/{SCORE_MAXIMO}"
                )
        else:
            lineas.append("\nSin compras hoy.")

        if cierres:
            lineas.append(f"\n<b>CIERRES ({len(cierres)})</b>")
            for c in cierres:
                lineas.append(
                    f"  SELL {c['ticker']}  {c['motivo']}  @ ${c['precio_cierre']:.2f}"
                )

        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": "\n".join(lineas), "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log(f"  [WARN] Telegram: {e}")


def cmd_liquidar():
    """
    Cierra TODAS las posiciones abiertas en Alpaca #3 y las marca como
    CERRADA en DB con motivo ESTRATEGIA_NUEVA.
    Usar una sola vez al migrar de la estrategia vieja a la nueva.
    """
    log("=" * 60)
    log("  LIQUIDACION TOTAL — cambio de estrategia (Alpaca #3)")
    log("=" * 60)

    posiciones = get_posiciones_abiertas()
    if not posiciones:
        log("  Sin posiciones abiertas. Nada que liquidar.")
        return

    log(f"  {len(posiciones)} posicion(es) a cerrar:\n")
    for pos in posiciones:
        ticker = pos["ticker"]
        log(f"  Cerrando {ticker} (id={pos['id']}, entrada=${float(pos['precio_entrada']):.2f})...")
        try:
            alpaca_client.close_position(ticker, SUFFIX)
            precio_actual = alpaca_client.get_latest_price(ticker, suffix=SUFFIX)
        except Exception as e:
            log(f"    [WARN] Alpaca: {e} — registrando cierre igual con precio estimado")
            precio_actual = float(pos["precio_entrada"])

        try:
            resultado = registrar_cierre(pos["id"], precio_actual, "ESTRATEGIA_NUEVA")
            signo = "+" if resultado["pnl"] >= 0 else ""
            log(f"    -> PnL: {signo}${resultado['pnl']:.2f} "
                f"({signo}{resultado['pnl_pct']*100:.1f}%)")
        except Exception as e:
            log(f"    ERROR DB: {e}")

    log("\n  Liquidacion completa. Ahora corre --dry-run para verificar la nueva estrategia.")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Bot SMC Estructura — Alpaca #3")
    parser.add_argument("--dry-run",  action="store_true", help="Simula sin enviar ordenes")
    parser.add_argument("--status",   action="store_true", help="Estado cuenta y posiciones")
    parser.add_argument("--init",     action="store_true", help="Inicializa tablas en DB")
    parser.add_argument("--liquidar", action="store_true", help="Cierra todas las posiciones (cambio estrategia)")
    args = parser.parse_args()

    if args.init:
        cmd_init()
    elif args.status:
        cmd_status()
    elif args.liquidar:
        cmd_liquidar()
    else:
        cmd_run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
