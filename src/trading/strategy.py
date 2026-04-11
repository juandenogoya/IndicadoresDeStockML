"""
strategy.py
Logica de estrategia: lee senales del scanner ML y decide
si abrir/cerrar posiciones segun parametros de riesgo.
"""

from datetime import date
from sqlalchemy import text
from src.data.database import get_engine
from src.trading import risk, alpaca_client, portfolio


def obtener_senales_hoy() -> list[dict]:
    """
    Lee alertas del ultimo scan disponible (ultimo dia de trading).
    En fin de semana retorna el viernes; en feriados, el ultimo dia habil.
    """
    engine = get_engine()
    with engine.connect() as conn:
        # Buscar la fecha mas reciente del scanner
        ultima = conn.execute(text("""
            SELECT DATE(scan_fecha) as fecha
            FROM alertas_scanner
            ORDER BY scan_fecha DESC
            LIMIT 1
        """)).scalar()

        if not ultima:
            return []

        rows = conn.execute(text("""
            SELECT
                a.ticker,
                a.alert_score   AS score,
                a.alert_nivel,
                a.ml_prob_ganancia,
                a.sector,
                a.precio_cierre
            FROM alertas_scanner a
            WHERE DATE(a.scan_fecha) = :fecha
            ORDER BY a.alert_score DESC
        """), {"fecha": ultima}).fetchall()

    return [dict(r._mapping) for r in rows]


def evaluar_cierres(
    posiciones: list[dict],
    usar_filtro_mtf: bool = False,
) -> list[dict]:
    """
    Evalua posiciones abiertas y decide si cerrar por:
    - Stop loss alcanzado
    - Take profit alcanzado
    - Senal de venta del scanner

    Retorna lista de posiciones a cerrar con motivo.
    """
    a_cerrar = []
    senales_hoy = {s["ticker"]: s for s in obtener_senales_hoy()}

    for pos in posiciones:
        ticker = pos["ticker"]
        try:
            precio_actual = alpaca_client.get_latest_price(ticker)
        except Exception as e:
            print(f"  WARN: no se pudo obtener precio de {ticker}: {e}")
            continue

        stop_loss   = float(pos["stop_loss"])   if pos["stop_loss"]   else None
        take_profit = float(pos["take_profit"]) if pos["take_profit"] else None

        # Stop loss
        if stop_loss and precio_actual <= stop_loss:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo": "STOP_LOSS",
            })
            continue

        # Take profit
        if take_profit and precio_actual >= take_profit:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo": "TAKE_PROFIT",
            })
            continue

        # Senal de venta del scanner ML
        senal = senales_hoy.get(ticker)
        if senal and senal["alert_nivel"] in ("VENTA", "VENTA_FUERTE"):
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo": "SENAL_VENTA",
            })

    return a_cerrar


def evaluar_entradas(
    posiciones_actuales: list[dict],
    equity: float,
    usar_filtro_mtf: bool = False,
) -> list[dict]:
    """
    Evalua senales del dia y selecciona candidatos para abrir posicion.
    Distribuye el capital segun BOT_MODO_DISTRIBUCION.
    Retorna lista de dicts con datos de entrada.
    """
    tickers_abiertos = {p["ticker"] for p in posiciones_actuales}
    senales_todas    = obtener_senales_hoy()

    # ── 1. Filtrar candidatos validos ──────────────────────────
    candidatos = []
    for senal in senales_todas:
        ticker = senal["ticker"]
        if ticker in tickers_abiertos:
            continue

        cumple, _ = risk.cumple_filtros_entrada(
            alert_nivel     = senal["alert_nivel"],
            score           = float(senal["score"]),
            usar_filtro_mtf = False,
        )
        if not cumple:
            continue

        # Respetar limite de posiciones simultaneas
        if len(posiciones_actuales) + len(candidatos) >= risk.MAX_POSICIONES:
            break

        candidatos.append(senal)

    if not candidatos:
        return []

    # ── 2. Distribuir capital entre candidatos ─────────────────
    distribucion = risk.distribuir_capital(equity, candidatos, risk.MODO_DISTRIBUCION)

    # ── 3. Construir lista de entradas con precios reales ──────
    a_abrir = []
    for senal in candidatos:
        ticker = senal["ticker"]
        capital_asignado = distribucion[ticker]

        try:
            precio = alpaca_client.get_latest_price(ticker)
        except Exception:
            precio = float(senal["precio_cierre"]) if senal.get("precio_cierre") else None
            if not precio:
                print(f"  WARN: sin precio para {ticker}, se omite.")
                continue

        qty = risk.calcular_qty(equity, precio, capital_asignado)
        if qty < 1:
            continue

        a_abrir.append({
            "ticker":       ticker,
            "precio":       precio,
            "qty":          qty,
            "capital":      round(precio * qty, 2),
            "pct_equity":   round(precio * qty / equity * 100, 1),
            "stop_loss":    risk.calcular_stop_loss(precio),
            "take_profit":  risk.calcular_take_profit(precio),
            "score":        float(senal["score"]),
            "nivel":        senal["alert_nivel"],
            "tendencia_1w": None,
            "tendencia_1m": None,
        })

    return a_abrir
