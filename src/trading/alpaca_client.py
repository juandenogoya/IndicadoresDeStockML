"""
alpaca_client.py
Wrapper de conexion con Alpaca (paper y live).
Lee ALPACA_API_KEY y ALPACA_SECRET_KEY desde .env o variables de entorno.
"""

import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


def get_client() -> TradingClient:
    """Retorna cliente Alpaca (paper si ALPACA_PAPER=true)."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret   = os.getenv("ALPACA_SECRET_KEY")
    paper    = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    if not api_key or not secret:
        raise ValueError(
            "Faltan ALPACA_API_KEY y/o ALPACA_SECRET_KEY. "
            "Agregalas en .env.local"
        )
    return TradingClient(api_key, secret, paper=paper)


def get_data_client() -> StockHistoricalDataClient:
    """Retorna cliente de datos de mercado."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret   = os.getenv("ALPACA_SECRET_KEY")
    return StockHistoricalDataClient(api_key, secret)


def get_account_info() -> dict:
    """Retorna informacion de la cuenta: equity, buying_power, cash."""
    client = get_client()
    acc = client.get_account()
    return {
        "equity":        float(acc.equity),
        "buying_power":  float(acc.buying_power),
        "cash":          float(acc.cash),
        "portfolio_value": float(acc.portfolio_value),
        "paper":         acc.account_number.startswith("PA"),
    }


def get_open_positions() -> list[dict]:
    """Retorna lista de posiciones abiertas."""
    client = get_client()
    positions = client.get_all_positions()
    result = []
    for p in positions:
        result.append({
            "ticker":       p.symbol,
            "qty":          float(p.qty),
            "avg_entry":    float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
        })
    return result


def get_latest_price(ticker: str) -> float:
    """Retorna el ultimo precio de mercado de un ticker."""
    data_client = get_data_client()
    req = StockLatestQuoteRequest(symbol_or_symbols=[ticker])
    quotes = data_client.get_stock_latest_quote(req)
    q = quotes[ticker]
    # Usar mid price (ask+bid)/2, o ask si no hay bid
    if q.bid_price and q.ask_price:
        return round((float(q.ask_price) + float(q.bid_price)) / 2, 2)
    return float(q.ask_price or q.bid_price)


def place_market_order(ticker: str, qty: float, side: str) -> dict:
    """
    Envia una orden de mercado.
    side: 'buy' o 'sell'
    """
    client = get_client()
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = client.submit_order(req)
    return {
        "order_id":  str(order.id),
        "ticker":    order.symbol,
        "qty":       float(order.qty),
        "side":      order.side.value,
        "status":    order.status.value,
        "type":      "market",
    }


def place_limit_order(ticker: str, qty: float, side: str, limit_price: float) -> dict:
    """
    Envia una orden limitada.
    side: 'buy' o 'sell'
    """
    client = get_client()
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    req = LimitOrderRequest(
        symbol=ticker,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
    )
    order = client.submit_order(req)
    return {
        "order_id":    str(order.id),
        "ticker":      order.symbol,
        "qty":         float(order.qty),
        "side":        order.side.value,
        "limit_price": float(order.limit_price),
        "status":      order.status.value,
        "type":        "limit",
    }


def cancel_order(order_id: str) -> bool:
    """Cancela una orden pendiente. Retorna True si exitoso."""
    client = get_client()
    try:
        client.cancel_order_by_id(order_id)
        return True
    except Exception:
        return False


def close_position(ticker: str) -> dict:
    """Cierra completamente la posicion de un ticker."""
    client = get_client()
    order = client.close_position(ticker)
    return {
        "order_id": str(order.id),
        "ticker":   order.symbol,
        "side":     order.side.value,
        "status":   order.status.value,
    }
