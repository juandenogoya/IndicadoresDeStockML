"""
31_bot_technical.py
Bot de trading — Estrategia Tecnica 3 capas (cuenta Alpaca #2).

Estrategia:
    Capa 1 (filtro): Precio > SMA200  (obligatorio)
    Capa 2 (tendencia): Precio > SMA50 (2pts) + Precio > SMA21 (1pt)
    Capa 3 (momentum): MACD > Signal + hist>0 (1.5pts) + RSI 45-68 (1pt)
    Entrada: score >= 4.0 / 5.5
    Salida:  SMA200 rota | SMA50 rota | MACD neg 2d | RSI<40 | SL5% | TP10%

Uso:
    python scripts/31_bot_technical.py            # ejecucion real
    python scripts/31_bot_technical.py --dry-run  # simula sin enviar ordenes
    python scripts/31_bot_technical.py --status   # estado cuenta
    python scripts/31_bot_technical.py --init     # inicializa tablas DB
"""

import sys
import os
import argparse
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Cargar variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from src.trading import alpaca_client, risk
from src.trading.strategy_technical import (
    evaluar_entradas_tech,
    evaluar_cierres_tech,
    SCORE_ENTRADA,
    SCORE_MAXIMO,
    RSI_MIN, RSI_MAX,
)
from src.trading.portfolio import (
    init_tabla,
    get_posiciones_abiertas,
    registrar_entrada,
    registrar_cierre,
    ticker_tiene_posicion,
)

# Sufijo para cuenta Alpaca #2
SUFFIX = "_2"
# Tablas propias del bot tecnico
TABLA_POSICIONES  = "posiciones_bot_tech"
TABLA_OPERACIONES = "operaciones_bot_tech"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# Init tablas propias
# ─────────────────────────────────────────────────────────────

def cmd_init():
    """Inicializa tablas del bot tecnico en Railway."""
    from src.data.database import get_engine
    from sqlalchemy import text
    log("Inicializando tablas bot tecnico...")
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
                score_entrada   NUMERIC(5,2),
                nivel_entrada   VARCHAR(40),
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
                nivel_entrada   VARCHAR(40),
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
    log("Tablas posiciones_bot_tech y operaciones_bot_tech OK.")


# ─────────────────────────────────────────────────────────────
# Helpers DB con tablas propias
# ─────────────────────────────────────────────────────────────

def get_posiciones_tech() -> list[dict]:
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, ticker, fecha_entrada, precio_entrada,
                   qty, stop_loss, take_profit, score_entrada, nivel_entrada
            FROM {TABLA_POSICIONES}
            WHERE estado = 'ABIERTA'
            ORDER BY fecha_entrada
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def registrar_entrada_tech(ticker, precio, qty, sl, tp, score, nivel, order_id) -> int:
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text(f"""
            INSERT INTO {TABLA_POSICIONES}
                (ticker, fecha_entrada, precio_entrada, qty,
                 stop_loss, take_profit, score_entrada, nivel_entrada,
                 alpaca_order_id, estado)
            VALUES
                (:ticker, :fecha, :precio, :qty,
                 :sl, :tp, :score, :nivel,
                 :order_id, 'ABIERTA')
            RETURNING id
        """), {
            "ticker": ticker, "fecha": date.today(),
            "precio": precio, "qty": qty,
            "sl": sl, "tp": tp,
            "score": score, "nivel": nivel,
            "order_id": order_id,
        })
        conn.commit()
        return r.scalar()


def registrar_cierre_tech(pos_id, precio_cierre, motivo) -> dict:
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        pos = conn.execute(text(f"""
            SELECT ticker, precio_entrada, qty, score_entrada, nivel_entrada
            FROM {TABLA_POSICIONES} WHERE id = :id
        """), {"id": pos_id}).fetchone()
        if not pos:
            raise ValueError(f"Posicion {pos_id} no encontrada.")

        pnl     = round((precio_cierre - float(pos.precio_entrada)) * pos.qty, 4)
        pnl_pct = round((precio_cierre - float(pos.precio_entrada)) / float(pos.precio_entrada), 4)

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
                (ticker, fecha_cierre, precio_entrada, precio_cierre,
                 qty, pnl, pnl_pct, motivo_cierre, score_entrada, nivel_entrada)
            VALUES
                (:ticker, :fecha, :precio_entrada, :precio_cierre,
                 :qty, :pnl, :pnl_pct, :motivo, :score, :nivel)
        """), {
            "ticker": pos.ticker, "fecha": date.today(),
            "precio_entrada": pos.precio_entrada, "precio_cierre": precio_cierre,
            "qty": pos.qty, "pnl": pnl, "pnl_pct": pnl_pct,
            "motivo": motivo, "score": pos.score_entrada, "nivel": pos.nivel_entrada,
        })
        conn.commit()
    return {"pnl": pnl, "pnl_pct": pnl_pct}


# ─────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────

def cmd_status():
    log("Consultando cuenta Alpaca #2 (Tecnico)...")
    cuenta = alpaca_client.get_account_info(SUFFIX)
    modo   = "PAPER" if cuenta["paper"] else "LIVE"
    print(f"""
  ================================================
    CUENTA ALPACA #2 — {modo} (Estrategia Tecnica)
  ================================================
    Equity          : ${cuenta['equity']:,.2f}
    Buying Power    : ${cuenta['buying_power']:,.2f}
    Cash            : ${cuenta['cash']:,.2f}
    Portfolio Value : ${cuenta['portfolio_value']:,.2f}
  ================================================
""")
    posiciones = get_posiciones_tech()
    if not posiciones:
        print("  Sin posiciones abiertas.")
    else:
        print(f"  {len(posiciones)} posicion(es) abierta(s):\n")
        for p in posiciones:
            print(f"    {p['ticker']:6s}  qty={p['qty']}  "
                  f"entrada=${float(p['precio_entrada']):.2f}  "
                  f"score={p['score_entrada']}  nivel={p['nivel_entrada']}")


# ─────────────────────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────────────────────

def cmd_run(dry_run: bool = False):
    separador = "=" * 60
    log(separador)
    log(f"  BOT TECNICO 3 CAPAS — {'DRY RUN' if dry_run else 'EJECUCION REAL'}")
    log(f"  Entrada    : score >= {SCORE_ENTRADA} / {SCORE_MAXIMO}")
    log(f"  RSI rango  : {RSI_MIN} - {RSI_MAX}")
    log(f"  Max posic. : {risk.MAX_POSICIONES}")
    log(f"  Por trade  : {risk.RIESGO_POR_TRADE*100:.0f}% del equity")
    log(f"  Stop Loss  : {risk.STOP_LOSS_PCT*100:.0f}%")
    log(f"  Take Profit: {risk.TAKE_PROFIT_PCT*100:.0f}%")
    log(separador)

    cuenta = alpaca_client.get_account_info(SUFFIX)
    equity = cuenta["equity"]
    log(f"  Equity: ${equity:,.2f} | Buying Power: ${cuenta['buying_power']:,.2f}")

    posiciones = get_posiciones_tech()
    log(f"  Posiciones abiertas: {len(posiciones)}")

    # ── Cierres ───────────────────────────────────────────────
    log("\n  [CIERRES]")
    a_cerrar = evaluar_cierres_tech(posiciones)

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
                    resultado = registrar_cierre_tech(c["id"], precio, motivo)
                    signo = "+" if resultado["pnl"] >= 0 else ""
                    log(f"    -> PnL: {signo}${resultado['pnl']:.2f} "
                        f"({signo}{resultado['pnl_pct']*100:.1f}%)")
                except Exception as e:
                    log(f"    ERROR: {e}")
            else:
                log("    [DRY RUN] orden no enviada.")

    # Actualizar posiciones
    posiciones = get_posiciones_tech()

    # ── Entradas ──────────────────────────────────────────────
    log("\n  [ENTRADAS]")
    a_abrir = evaluar_entradas_tech(posiciones, equity)

    if not a_abrir:
        log("  Sin senales tecnicas validas hoy.")
    else:
        for entrada in a_abrir:
            ticker  = entrada["ticker"]
            qty     = entrada["qty"]
            precio  = entrada["precio"]
            score   = entrada["score"]
            capital = entrada["capital"]
            pct     = entrada["pct_equity"]
            log(f"  COMPRAR {ticker}  qty={qty}  precio=${precio:.2f}  "
                f"capital=${capital:,.0f} ({pct:.1f}%)  score={score:.1f}/{SCORE_MAXIMO}")
            log(f"    SMA50={'OK' if entrada['cond_sma50'] else 'NO'}  "
                f"SMA21={'OK' if entrada['cond_sma21'] else 'NO'}  "
                f"MACD={'OK' if entrada['cond_macd'] else 'NO'}  "
                f"RSI={entrada['rsi']:.0f}  "
                f"SL=${entrada['stop_loss']:.2f}  TP=${entrada['take_profit']:.2f}")

            if not dry_run:
                try:
                    orden = alpaca_client.place_market_order(ticker, qty, "buy", SUFFIX)
                    pos_id = registrar_entrada_tech(
                        ticker   = ticker,
                        precio   = precio,
                        qty      = qty,
                        sl       = entrada["stop_loss"],
                        tp       = entrada["take_profit"],
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

    log("\n  Bot tecnico finalizado.")
    log(separador)


def _telegram(cierres, entradas, equity):
    try:
        import os, requests
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        lineas = ["<b>BOT TECNICO 3 CAPAS — Resumen</b>"]
        if entradas:
            lineas.append(f"\n<b>COMPRAS ({len(entradas)})</b>")
            for e in entradas:
                lineas.append(
                    f"  BUY {e['ticker']}  x{e['qty']}  ${e['precio']:.2f}"
                    f"  score={e['score']:.1f}/{SCORE_MAXIMO}"
                )
        else:
            lineas.append("\nSin compras hoy.")
        if cierres:
            lineas.append(f"\n<b>CIERRES ({len(cierres)})</b>")
            for c in cierres:
                lineas.append(f"  SELL {c['ticker']}  {c['motivo']}  @ ${c['precio_cierre']:.2f}")
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": "\n".join(lineas), "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log(f"  [WARN] Telegram: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bot Tecnico 3 Capas — Alpaca #2")
    parser.add_argument("--dry-run", action="store_true", help="Simula sin enviar ordenes")
    parser.add_argument("--status",  action="store_true", help="Estado cuenta y posiciones")
    parser.add_argument("--init",    action="store_true", help="Inicializa tablas en DB")
    args = parser.parse_args()

    if args.init:
        cmd_init()
    elif args.status:
        cmd_status()
    else:
        cmd_run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
