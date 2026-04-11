"""
33_bot_52w.py
Bot de trading — Estrategia Ruptura de Maximo 52 Semanas (cuenta Alpaca #4).

Basado en George & Hwang (2004): proximidad al maximo de 52 semanas
predice retornos futuros positivos.

Estrategia:
    Filtros obligatorios: dist_max_52w >= -8% Y close > SMA200
    Scoring (min 3/4):
        +1  Aproximacion  : dentro del 3% del maximo
        +1  Ruptura real  : precio por encima del maximo 52w
        +1  Volumen       : vol_relativo >= 1.5 o vol_spike
        +1  SMA50         : precio sobre SMA50
    Salida:
        SL   = entrada - 2.0 x ATR14
        TP   = entrada + 6.0 x ATR14  (ratio 1:3)
        SMA50 rota 2 dias consecutivos -> salir
        Time stop = 60 dias habiles

Uso:
    python scripts/33_bot_52w.py            # ejecucion real
    python scripts/33_bot_52w.py --dry-run  # simula sin enviar ordenes
    python scripts/33_bot_52w.py --status   # estado cuenta
    python scripts/33_bot_52w.py --init     # inicializa tablas DB
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
from src.trading.strategy_52w import (
    evaluar_entradas_52w,
    evaluar_cierres_52w,
    SCORE_ENTRADA,
    SCORE_MAXIMO,
    ATR_MULT_SL,
    ATR_MULT_TP,
    DIAS_MAX_POS,
    DIST_FILTRO,
)

SUFFIX            = "_4"
TABLA_POSICIONES  = "posiciones_bot_52w"
TABLA_OPERACIONES = "operaciones_bot_52w"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# Init tablas
# ─────────────────────────────────────────────────────────────

def cmd_init():
    from src.data.database import get_engine
    from sqlalchemy import text
    log("Inicializando tablas bot 52w...")
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
                max_52w_entrada NUMERIC(10,4),
                dist_52w_entrada NUMERIC(8,4),
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

def get_posiciones_52w() -> list[dict]:
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, ticker, fecha_entrada, precio_entrada,
                   qty, stop_loss, take_profit, atr_entrada,
                   max_52w_entrada, dist_52w_entrada,
                   score_entrada, nivel_entrada
            FROM {TABLA_POSICIONES}
            WHERE estado = 'ABIERTA'
            ORDER BY fecha_entrada
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def registrar_entrada_52w(ticker, precio, qty, sl, tp, atr, max_52w, dist_52w, score, nivel, order_id) -> int:
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text(f"""
            INSERT INTO {TABLA_POSICIONES}
                (ticker, fecha_entrada, precio_entrada, qty,
                 stop_loss, take_profit, atr_entrada,
                 max_52w_entrada, dist_52w_entrada,
                 score_entrada, nivel_entrada, alpaca_order_id, estado)
            VALUES
                (:ticker, :fecha, :precio, :qty,
                 :sl, :tp, :atr,
                 :max_52w, :dist_52w,
                 :score, :nivel, :order_id, 'ABIERTA')
            RETURNING id
        """), {
            "ticker": ticker, "fecha": date.today(),
            "precio": precio, "qty": qty,
            "sl": sl, "tp": tp, "atr": atr,
            "max_52w": max_52w, "dist_52w": dist_52w,
            "score": score, "nivel": nivel, "order_id": order_id,
        })
        conn.commit()
        return r.scalar()


def registrar_cierre_52w(pos_id, precio_cierre, motivo) -> dict:
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
            "ticker": pos.ticker,
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
    log("Consultando cuenta Alpaca #4 (52W)...")
    cuenta = alpaca_client.get_account_info(SUFFIX)
    modo   = "PAPER" if cuenta["paper"] else "LIVE"
    print(f"""
  ================================================
    CUENTA ALPACA #4 — {modo} (Estrategia 52W High)
  ================================================
    Equity          : ${cuenta['equity']:,.2f}
    Buying Power    : ${cuenta['buying_power']:,.2f}
    Cash            : ${cuenta['cash']:,.2f}
    Portfolio Value : ${cuenta['portfolio_value']:,.2f}
  ================================================
""")
    posiciones = get_posiciones_52w()
    if not posiciones:
        print("  Sin posiciones abiertas.")
    else:
        print(f"  {len(posiciones)} posicion(es) abierta(s):\n")
        for p in posiciones:
            entrada = float(p["precio_entrada"])
            dias    = (date.today() - p["fecha_entrada"]).days
            dist    = float(p["dist_52w_entrada"]) if p.get("dist_52w_entrada") else 0
            print(f"    {p['ticker']:6s}  qty={p['qty']}  "
                  f"entrada=${entrada:.2f}  "
                  f"dist_52w={dist:+.1f}%  "
                  f"score={p['score_entrada']}  "
                  f"dias={dias}/{DIAS_MAX_POS}")


# ─────────────────────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────────────────────

def cmd_run(dry_run: bool = False):
    separador = "=" * 60
    log(separador)
    log(f"  BOT 52W HIGH BREAKOUT — {'DRY RUN' if dry_run else 'EJECUCION REAL'}")
    log(f"  Filtro     : dist_max_52w >= {DIST_FILTRO*100:.0f}% + close > SMA200")
    log(f"  Entrada    : score >= {SCORE_ENTRADA} / {SCORE_MAXIMO}")
    log(f"  SL / TP    : {ATR_MULT_SL}x ATR / {ATR_MULT_TP}x ATR  (ratio 1:3)")
    log(f"  Time stop  : {DIAS_MAX_POS} dias")
    log(f"  Max posic. : {risk.MAX_POSICIONES}")
    log(f"  Por trade  : {risk.RIESGO_POR_TRADE*100:.0f}% del equity")
    log(separador)

    cuenta = alpaca_client.get_account_info(SUFFIX)
    equity = cuenta["equity"]
    log(f"  Equity: ${equity:,.2f} | Buying Power: ${cuenta['buying_power']:,.2f}")

    posiciones = get_posiciones_52w()
    log(f"  Posiciones abiertas: {len(posiciones)}")

    # ── Cierres ───────────────────────────────────────────────
    log("\n  [CIERRES]")
    a_cerrar = evaluar_cierres_52w(posiciones)

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
                    resultado = registrar_cierre_52w(c["id"], precio, motivo)
                    signo = "+" if resultado["pnl"] >= 0 else ""
                    log(f"    -> PnL: {signo}${resultado['pnl']:.2f} "
                        f"({signo}{resultado['pnl_pct']*100:.1f}%)")
                except Exception as e:
                    log(f"    ERROR: {e}")
            else:
                log("    [DRY RUN] orden no enviada.")

    posiciones = get_posiciones_52w()

    # ── Entradas ──────────────────────────────────────────────
    log("\n  [ENTRADAS]")
    a_abrir = evaluar_entradas_52w(posiciones, equity)

    if not a_abrir:
        log("  Sin rupturas de 52w validas hoy.")
    else:
        for entrada in a_abrir:
            ticker  = entrada["ticker"]
            qty     = entrada["qty"]
            precio  = entrada["precio"]
            score   = entrada["score"]
            capital = entrada["capital"]
            pct     = entrada["pct_equity"]
            dist    = entrada["dist_max_52w"]
            log(f"  COMPRAR {ticker}  qty={qty}  precio=${precio:.2f}  "
                f"capital=${capital:,.0f} ({pct:.1f}%)  score={score:.1f}/{SCORE_MAXIMO}")
            log(f"    Tipo={entrada['tipo']}  "
                f"dist_52w={dist:+.1f}%  max_52w=${entrada['max_52w']:.2f}  "
                f"VolRel={entrada['vol_relativo']:.1f}x  "
                f"ATR=${entrada['atr']:.2f}  "
                f"SL=${entrada['stop_loss']:.2f}  TP=${entrada['take_profit']:.2f}")

            if not dry_run:
                try:
                    orden  = alpaca_client.place_market_order(ticker, qty, "buy", SUFFIX)
                    pos_id = registrar_entrada_52w(
                        ticker   = ticker,
                        precio   = precio,
                        qty      = qty,
                        sl       = entrada["stop_loss"],
                        tp       = entrada["take_profit"],
                        atr      = entrada["atr"],
                        max_52w  = entrada["max_52w"],
                        dist_52w = entrada["dist_max_52w"],
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

    log("\n  Bot 52W finalizado.")
    log(separador)


def _telegram(cierres, entradas, equity):
    try:
        import requests
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        lineas = ["<b>BOT 52W HIGH BREAKOUT — Resumen</b>"]
        if entradas:
            lineas.append(f"\n<b>COMPRAS ({len(entradas)})</b>")
            for e in entradas:
                dist = e["dist_max_52w"]
                tipo_str = "RUPTURA" if dist > 0 else f"APROX ({dist:+.1f}%)"
                lineas.append(
                    f"  BUY {e['ticker']}  x{e['qty']}  ${e['precio']:.2f}"
                    f"  {tipo_str}  score={e['score']:.1f}/{SCORE_MAXIMO}"
                )
        else:
            lineas.append("\nSin rupturas hoy.")
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


def main():
    parser = argparse.ArgumentParser(description="Bot 52W High Breakout — Alpaca #4")
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
