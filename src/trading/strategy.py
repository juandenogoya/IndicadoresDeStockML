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
    Retorna lista de dicts con datos de entrada.
    """
    tickers_abiertos  = {p["ticker"] for p in posiciones_actuales}
    capital_invertido = sum(
        float(p["qty"]) * alpaca_client.get_latest_price(p["ticker"])
        for p in posiciones_actuales
    ) if posiciones_actuales else 0.0

    senales = obtener_senales_hoy()
    a_abrir = []

    for senal in senales:
        ticker = senal["ticker"]

        # Ya tiene posicion abierta
        if ticker in tickers_abiertos:
            continue

        # Filtros de entrada
        cumple, motivo = risk.cumple_filtros_entrada(
            alert_nivel  = senal["alert_nivel"],
            score        = float(senal["score"]),
            usar_filtro_mtf = False,  # MTF no persiste en DB aun
        )
        if not cumple:
            continue

        # Precio actual de mercado (o precio de cierre si mercado cerrado)
        try:
            precio = alpaca_client.get_latest_price(ticker)
        except Exception:
            precio = float(senal["precio_cierre"]) if senal.get("precio_cierre") else None
            if not precio:
                print(f"  WARN: no se pudo obtener precio de {ticker}")
                continue

        # Verificar limites de portafolio
        puede, motivo_riesgo = risk.puede_abrir_posicion(
            n_posiciones_actuales = len(posiciones_actuales) + len(a_abrir),
            equity                = equity,
            capital_invertido     = capital_invertido,
            precio                = precio,
        )
        if not puede:
            print(f"  SKIP {ticker}: {motivo_riesgo}")
            break  # Si llego al limite, no evaluo mas

        qty = risk.calcular_qty(equity, precio)
        a_abrir.append({
            "ticker":       ticker,
            "precio":       precio,
            "qty":          qty,
            "stop_loss":    risk.calcular_stop_loss(precio),
            "take_profit":  risk.calcular_take_profit(precio),
            "score":        float(senal["score"]),
            "nivel":        senal["alert_nivel"],
            "tendencia_1w": None,
            "tendencia_1m": None,
        })
        capital_invertido += precio * qty

    return a_abrir
