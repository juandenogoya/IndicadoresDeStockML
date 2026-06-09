"""
ejecucion_bot.py -- Execution adapter de los bots Alpaca (Plan B, Tarea 16).

Generaliza el viejo portfolio.py: parametrizado por bot (cuenta Alpaca + par de
tablas de estado). Toma las decisiones del cerebro compartido (src/strategies)
y las EJECUTA: ordenes via alpaca_client + persistencia en
posiciones_bot{X}/operaciones_bot{X} (Railway).

Mapeo de los 3 bots (reusa tablas existentes del reset; nombres heredados):
    Bot 1 ML            -> cuenta ""   | posiciones_bot        / operaciones_bot
    Bot 2 TECH_SECTOR_v1-> cuenta "_2" | posiciones_bot_tech   / operaciones_bot_tech
    Bot 3 OPTIONS_v2    -> cuenta "_3" | posiciones_bot_candle / operaciones_bot_candle

En GitHub Actions get_engine() -> Railway (DATABASE_URL secret); ahi viven tanto
la masticada como las tablas de estado del bot. No hay conexion dual de este lado.

Nombres de tabla: provienen de BotAlpacaConfig (constantes del codigo, NO input
de usuario) -> interpolacion en el SQL es segura.
"""

from dataclasses import dataclass
from datetime import date

from sqlalchemy import text

from src.data.database import get_engine


def _alpaca():
    """Import LAZY del SDK de Alpaca: solo se necesita para EJECUTAR de verdad.
    El dry-run y la decision no dependen de alpaca-py (no instalado en local)."""
    from src.trading import alpaca_client
    return alpaca_client


@dataclass
class BotAlpacaConfig:
    """Identidad de ejecucion de un bot: cuenta Alpaca + tablas de estado."""
    nombre:         str
    account_suffix: str   # sufijo de credenciales Alpaca ("" / "_2" / "_3")
    tabla_pos:      str   # posiciones_bot{X}
    tabla_oper:     str   # operaciones_bot{X}


def _noop(*_a, **_k):
    pass


# ── Lectura de estado ─────────────────────────────────────────────────────────

def get_posiciones_abiertas(bcfg: BotAlpacaConfig, sector_por_ticker: dict = None) -> list:
    """
    Posiciones ABIERTA del bot, en la forma que espera el cerebro compartido.
    Enriquece con capital_entrada (precio*qty), cantidad (alias de qty) y sector
    (de la masticada, para la logica sectorial).
    """
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, ticker, fecha_entrada, precio_entrada, qty,
                   stop_loss, take_profit, score_entrada, nivel_entrada
            FROM {bcfg.tabla_pos}
            WHERE estado = 'ABIERTA'
            ORDER BY fecha_entrada
        """)).fetchall()

    out = []
    for r in rows:
        d = dict(r._mapping)
        precio = float(d["precio_entrada"]) if d["precio_entrada"] is not None else 0.0
        qty    = int(d["qty"]) if d["qty"] is not None else 0
        d["capital_entrada"] = round(precio * qty, 2)
        d["cantidad"]        = qty
        if sector_por_ticker is not None:
            d["sector"] = sector_por_ticker.get(d["ticker"], "Unknown")
        out.append(d)
    return out


# ── Persistencia ──────────────────────────────────────────────────────────────

def registrar_entrada(bcfg, ticker, precio, qty, sl, tp, score, nivel, order_id) -> int:
    """Inserta una posicion ABIERTA. Retorna el id."""
    eng = get_engine()
    with eng.connect() as conn:
        r = conn.execute(text(f"""
            INSERT INTO {bcfg.tabla_pos} (
                ticker, fecha_entrada, precio_entrada, qty,
                stop_loss, take_profit, score_entrada, nivel_entrada,
                alpaca_order_id, estado
            ) VALUES (
                :ticker, :fecha, :precio, :qty,
                :sl, :tp, :score, :nivel, :order_id, 'ABIERTA'
            ) RETURNING id
        """), {
            "ticker": ticker, "fecha": date.today(), "precio": precio, "qty": qty,
            "sl": sl, "tp": tp, "score": score, "nivel": nivel, "order_id": order_id,
        })
        conn.commit()
        return r.scalar()


def registrar_cierre(bcfg, posicion_id, precio_cierre, motivo) -> dict:
    """Cierra la posicion (PnL) y la archiva en operaciones_bot{X}."""
    eng = get_engine()
    with eng.connect() as conn:
        pos = conn.execute(text(f"""
            SELECT ticker, fecha_entrada, precio_entrada, qty, score_entrada, nivel_entrada
            FROM {bcfg.tabla_pos} WHERE id = :id
        """), {"id": posicion_id}).fetchone()
        if not pos:
            raise ValueError(f"Posicion {posicion_id} no encontrada en {bcfg.tabla_pos}.")

        pe  = float(pos.precio_entrada)
        pnl = round((precio_cierre - pe) * pos.qty, 4)
        pnl_pct = round((precio_cierre - pe) / pe, 4) if pe else 0.0

        conn.execute(text(f"""
            UPDATE {bcfg.tabla_pos} SET
                estado='CERRADA', fecha_cierre=:f, precio_cierre=:p,
                pnl=:pnl, pnl_pct=:pct, motivo_cierre=:m
            WHERE id=:id
        """), {"f": date.today(), "p": precio_cierre, "pnl": pnl, "pct": pnl_pct,
               "m": motivo, "id": posicion_id})

        conn.execute(text(f"""
            INSERT INTO {bcfg.tabla_oper} (
                ticker, fecha_entrada, fecha_cierre,
                precio_entrada, precio_cierre, qty,
                pnl, pnl_pct, motivo_cierre, score_entrada, nivel_entrada
            ) VALUES (
                :ticker, :fe, :fc, :pe, :pc, :qty, :pnl, :pct, :m, :score, :nivel
            )
        """), {
            "ticker": pos.ticker, "fe": pos.fecha_entrada, "fc": date.today(),
            "pe": pos.precio_entrada, "pc": precio_cierre, "qty": pos.qty,
            "pnl": pnl, "pct": pnl_pct, "m": motivo,
            "score": pos.score_entrada, "nivel": pos.nivel_entrada,
        })
        conn.commit()
    return {"pnl": pnl, "pnl_pct": pnl_pct, "motivo": motivo}


# ── Aplicar decisiones (ordenes + persistencia) ───────────────────────────────

def _precio_ejecucion(bcfg, ticker, fallback, dry_run):
    """Precio para registrar/ejecutar: live de Alpaca; fallback = close de la masticada."""
    if dry_run:
        return fallback
    try:
        return _alpaca().get_latest_price(ticker, suffix=bcfg.account_suffix)
    except Exception:
        return fallback


def aplicar_cierres(bcfg, a_cerrar, dry_run=True, log=_noop):
    """Cierra en Alpaca y archiva. a_cerrar: dicts con id, ticker, precio_cierre, motivo."""
    for c in a_cerrar:
        ticker = c["ticker"]
        precio = _precio_ejecucion(bcfg, ticker, c.get("precio_cierre"), dry_run)
        motivo = c["motivo"]
        pe     = float(c.get("precio_entrada") or 0)
        qty    = int(c.get("cantidad") or c.get("qty") or 0)
        pnl_est = round((precio - pe) * qty, 2) if pe else 0.0
        log(f"  CERRAR {ticker} @ ${precio:.2f} | motivo={motivo} | pnl_est={pnl_est:+.2f}")
        if dry_run:
            log("    [DRY RUN] orden no enviada.")
            continue
        try:
            _alpaca().close_position(ticker, suffix=bcfg.account_suffix)
            res = registrar_cierre(bcfg, c["id"], precio, motivo)
            log(f"    -> PnL: {res['pnl']:+.2f} ({res['pnl_pct']*100:+.1f}%)")
        except Exception as e:
            log(f"    ERROR cerrando {ticker}: {e}")


def aplicar_entradas(bcfg, a_abrir, dry_run=True, log=_noop):
    """Abre en Alpaca y registra. a_abrir: dicts con ticker, precio, qty, sl, tp, score, +nivel|sector."""
    for e in a_abrir:
        ticker = e["ticker"]
        qty    = e["qty"]
        precio = _precio_ejecucion(bcfg, ticker, e["precio"], dry_run)
        nivel  = e.get("nivel") or e.get("sector")   # ML guarda nivel; sectorial, sector
        sl, tp = e["sl"], e["tp"]
        log(f"  COMPRAR {ticker} x{qty} @ ${precio:.2f} | score={e['score']:.1f} | "
            f"nivel={nivel} | SL=${sl:.2f} TP=${tp:.2f}")
        if dry_run:
            log("    [DRY RUN] orden no enviada.")
            continue
        try:
            orden = _alpaca().place_market_order(ticker, qty, "buy", suffix=bcfg.account_suffix)
            pid = registrar_entrada(bcfg, ticker, precio, qty, sl, tp, e["score"],
                                    nivel, orden["order_id"])
            log(f"    -> Orden {orden['order_id']} | DB id={pid}")
        except Exception as e2:
            log(f"    ERROR comprando {ticker}: {e2}")


def notificar_telegram(bcfg, a_abrir, a_cerrar, retenidos=None, fecha=None,
                       dry_run=True, log=_noop):
    """
    Resumen diario del bot por Telegram (HTML). No-op en dry-run o sin credenciales.
    Reusa src/pipeline/telegram_notifier._send_long (import lazy). Fail-safe: si algo
    falla, NO interrumpe el bot.
    """
    if dry_run:
        return
    try:
        from src.pipeline.telegram_notifier import _send_long
    except Exception as ex:
        log(f"  [WARN] Telegram no disponible: {ex}")
        return

    a_abrir  = a_abrir or []
    a_cerrar = a_cerrar or []
    fecha = fecha or date.today()

    lineas = [f"<b>BOT {bcfg.nombre}</b> -- {fecha}"]

    if a_cerrar:
        lineas.append(f"\n<b>CIERRES ({len(a_cerrar)})</b>")
        for c in a_cerrar:
            p = c.get("precio_cierre")
            ps = f" @ ${float(p):.2f}" if p is not None else ""
            lineas.append(f"  SELL {c['ticker']} [{c.get('motivo','')}]{ps}")

    if a_abrir:
        lineas.append(f"\n<b>COMPRAS ({len(a_abrir)})</b>")
        for e in a_abrir:
            try:
                pr = float(e["precio"])
            except Exception:
                pr = 0.0
            lineas.append(f"  BUY {e['ticker']} x{e['qty']} @ ${pr:.2f} "
                          f"(score {float(e.get('score',0)):.1f})")

    if not a_abrir and not a_cerrar:
        lineas.append("\nSin operaciones hoy.")

    if retenidos:
        lineas.append(f"\n<b>RETENCION opciones ({len(retenidos)})</b>: "
                      + ", ".join(retenidos))

    try:
        ok = _send_long("\n".join(lineas))
        log(f"  Telegram: {'enviado' if ok else 'no enviado'}")
    except Exception as ex:
        log(f"  [WARN] Telegram fallo: {ex}")
