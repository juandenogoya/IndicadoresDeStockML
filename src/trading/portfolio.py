"""
portfolio.py
Gestion de posiciones abiertas del bot en Railway.
Tabla: posiciones_bot (registro propio, independiente de Alpaca)
"""

from datetime import date
from sqlalchemy import text
from src.data.database import get_engine


def init_tabla():
    """Crea la tabla posiciones_bot si no existe."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS posiciones_bot (
                id              SERIAL PRIMARY KEY,
                ticker          VARCHAR(20) NOT NULL,
                fecha_entrada   DATE NOT NULL,
                precio_entrada  NUMERIC(10,4) NOT NULL,
                qty             INTEGER NOT NULL,
                stop_loss       NUMERIC(10,4),
                take_profit     NUMERIC(10,4),
                score_entrada   NUMERIC(5,2),
                nivel_entrada   VARCHAR(30),
                tendencia_1w    VARCHAR(20),
                tendencia_1m    VARCHAR(20),
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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS operaciones_bot (
                id              SERIAL PRIMARY KEY,
                ticker          VARCHAR(20) NOT NULL,
                fecha_entrada   DATE NOT NULL,
                fecha_cierre    DATE,
                precio_entrada  NUMERIC(10,4),
                precio_cierre   NUMERIC(10,4),
                qty             INTEGER,
                pnl             NUMERIC(10,4),
                pnl_pct         NUMERIC(6,4),
                motivo_cierre   VARCHAR(50),
                score_entrada   NUMERIC(5,2),
                nivel_entrada   VARCHAR(30),
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
    print("  Tablas posiciones_bot y operaciones_bot OK.")


def get_posiciones_abiertas() -> list[dict]:
    """Retorna posiciones con estado=ABIERTA."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, ticker, fecha_entrada, precio_entrada,
                   qty, stop_loss, take_profit, score_entrada,
                   nivel_entrada, alpaca_order_id
            FROM posiciones_bot
            WHERE estado = 'ABIERTA'
            ORDER BY fecha_entrada
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def ticker_tiene_posicion(ticker: str) -> bool:
    """Verifica si ya hay una posicion abierta para el ticker."""
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT COUNT(*) FROM posiciones_bot
            WHERE ticker = :ticker AND estado = 'ABIERTA'
        """), {"ticker": ticker}).scalar()
    return r > 0


def registrar_entrada(
    ticker: str,
    precio_entrada: float,
    qty: int,
    stop_loss: float,
    take_profit: float,
    score_entrada: float,
    nivel_entrada: str,
    tendencia_1w: str = None,
    tendencia_1m: str = None,
    alpaca_order_id: str = None,
) -> int:
    """Registra una nueva posicion abierta. Retorna el id."""
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text("""
            INSERT INTO posiciones_bot (
                ticker, fecha_entrada, precio_entrada, qty,
                stop_loss, take_profit, score_entrada, nivel_entrada,
                tendencia_1w, tendencia_1m, alpaca_order_id, estado
            ) VALUES (
                :ticker, :fecha, :precio, :qty,
                :sl, :tp, :score, :nivel,
                :t1w, :t1m, :order_id, 'ABIERTA'
            ) RETURNING id
        """), {
            "ticker":   ticker,
            "fecha":    date.today(),
            "precio":   precio_entrada,
            "qty":      qty,
            "sl":       stop_loss,
            "tp":       take_profit,
            "score":    score_entrada,
            "nivel":    nivel_entrada,
            "t1w":      tendencia_1w,
            "t1m":      tendencia_1m,
            "order_id": alpaca_order_id,
        })
        conn.commit()
        return r.scalar()


def registrar_cierre(
    posicion_id: int,
    precio_cierre: float,
    motivo_cierre: str,
) -> dict:
    """
    Cierra una posicion: calcula PnL y la mueve a operaciones_bot.
    motivo_cierre: 'STOP_LOSS' | 'TAKE_PROFIT' | 'SENAL_VENTA' | 'MANUAL'
    """
    engine = get_engine()
    with engine.connect() as conn:
        # Leer posicion
        pos = conn.execute(text("""
            SELECT ticker, precio_entrada, qty, score_entrada, nivel_entrada
            FROM posiciones_bot WHERE id = :id
        """), {"id": posicion_id}).fetchone()

        if not pos:
            raise ValueError(f"Posicion {posicion_id} no encontrada.")

        pnl     = round((precio_cierre - float(pos.precio_entrada)) * pos.qty, 4)
        pnl_pct = round((precio_cierre - float(pos.precio_entrada)) / float(pos.precio_entrada), 4)

        # Actualizar posicion
        conn.execute(text("""
            UPDATE posiciones_bot SET
                estado = 'CERRADA',
                fecha_cierre = :fecha,
                precio_cierre = :precio,
                pnl = :pnl,
                pnl_pct = :pnl_pct,
                motivo_cierre = :motivo
            WHERE id = :id
        """), {
            "fecha":  date.today(),
            "precio": precio_cierre,
            "pnl":    pnl,
            "pnl_pct": pnl_pct,
            "motivo": motivo_cierre,
            "id":     posicion_id,
        })

        # Registrar en historial
        conn.execute(text("""
            INSERT INTO operaciones_bot (
                ticker, fecha_entrada, fecha_cierre,
                precio_entrada, precio_cierre, qty,
                pnl, pnl_pct, motivo_cierre,
                score_entrada, nivel_entrada
            ) VALUES (
                :ticker, :fecha_entrada, :fecha_cierre,
                :precio_entrada, :precio_cierre, :qty,
                :pnl, :pnl_pct, :motivo,
                :score, :nivel
            )
        """), {
            "ticker":         pos.ticker,
            "fecha_entrada":  None,   # se carga desde posiciones_bot en produccion
            "fecha_cierre":   date.today(),
            "precio_entrada": pos.precio_entrada,
            "precio_cierre":  precio_cierre,
            "qty":            pos.qty,
            "pnl":            pnl,
            "pnl_pct":        pnl_pct,
            "motivo":         motivo_cierre,
            "score":          pos.score_entrada,
            "nivel":          pos.nivel_entrada,
        })
        conn.commit()

    return {"pnl": pnl, "pnl_pct": pnl_pct, "motivo": motivo_cierre}
