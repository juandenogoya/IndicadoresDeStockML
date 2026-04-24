"""
ft_utils.py
Funciones comunes para todos los bots de forward-testing.

Responsabilidades:
    - Cargar y actualizar instancias de estrategia (ft_estrategias)
    - Abrir y cerrar operaciones virtuales (ft_operaciones)
    - Actualizar trailing SL
    - Registrar metricas diarias (ft_metricas_diarias)
    - Obtener precios de cierre desde precios_diarios

Principio: solo escribe en tablas ft_*. Lee cualquier tabla existente.
"""

import json
from datetime import date, datetime
from sqlalchemy import text
from src.data.database import get_engine


# ── Logging ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Estrategia ───────────────────────────────────────────────────────────────

def cargar_estrategia(nombre: str) -> dict | None:
    """
    Carga una instancia de estrategia por nombre.
    Retorna None si no existe o no esta activa.
    """
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, nombre, logica, parametros,
                   capital_inicial, capital_actual,
                   cash_disponible, capital_inmovilizado,
                   activa, fecha_inicio
            FROM ft_estrategias
            WHERE nombre = :nombre AND activa = TRUE
        """), {"nombre": nombre}).fetchone()

    if not row:
        return None
    return dict(row._mapping)


def actualizar_capital_estrategia(conn, estrategia_id: int) -> None:
    """
    Recalcula y actualiza capital_actual, cash_disponible y capital_inmovilizado
    en ft_estrategias a partir del estado real de ft_operaciones.

    - capital_inmovilizado = suma de capital_entrada de posiciones abiertas
    - capital_actual       = cash_disponible + capital_inmovilizado
    - cash_disponible      = capital_inicial + pnl_acumulado - inmovilizado_actual

    Se llama despues de cada apertura o cierre de operacion.
    """
    conn.execute(text("""
        UPDATE ft_estrategias e
        SET
            capital_inmovilizado = COALESCE(sub.inmovilizado, 0),
            capital_actual       = e.cash_disponible + COALESCE(sub.inmovilizado, 0)
        FROM (
            SELECT estrategia_id, SUM(capital_entrada) AS inmovilizado
            FROM ft_operaciones
            WHERE estrategia_id = :eid AND fecha_salida IS NULL
            GROUP BY estrategia_id
        ) sub
        WHERE e.id = :eid AND sub.estrategia_id = :eid
    """), {"eid": estrategia_id})

    # Si no hay posiciones abiertas, inmovilizado = 0
    conn.execute(text("""
        UPDATE ft_estrategias
        SET capital_inmovilizado = 0,
            capital_actual       = cash_disponible
        WHERE id = :eid
          AND NOT EXISTS (
              SELECT 1 FROM ft_operaciones
              WHERE estrategia_id = :eid AND fecha_salida IS NULL
          )
    """), {"eid": estrategia_id})


# ── Precios ───────────────────────────────────────────────────────────────────

def obtener_precio_cierre(ticker: str) -> float | None:
    """Ultimo precio de cierre disponible para un ticker."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT close FROM precios_diarios
            WHERE ticker = :ticker
            ORDER BY fecha DESC
            LIMIT 1
        """), {"ticker": ticker}).fetchone()
    return float(row.close) if row else None


def obtener_precios_cierre_todos() -> dict[str, float]:
    """
    Retorna {ticker: close} con el ultimo cierre disponible para todos los tickers.
    Usado para evaluar posiciones abiertas sin hacer N queries individuales.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (ticker) ticker, close
            FROM precios_diarios
            ORDER BY ticker, fecha DESC
        """)).fetchall()
    return {r.ticker: float(r.close) for r in rows}


# ── Posiciones ────────────────────────────────────────────────────────────────

def obtener_posiciones_abiertas(estrategia_id: int) -> list[dict]:
    """Retorna todas las posiciones abiertas de una estrategia."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, ticker, lado, fecha_entrada,
                   precio_entrada, cantidad, capital_entrada,
                   stop_loss, take_profit, score_entrada, detalle_entrada
            FROM ft_operaciones
            WHERE estrategia_id = :eid AND fecha_salida IS NULL
            ORDER BY fecha_entrada
        """), {"eid": estrategia_id}).fetchall()
    return [dict(r._mapping) for r in rows]


# ── Abrir operacion ───────────────────────────────────────────────────────────

def abrir_operacion(
    estrategia_id: int,
    ticker:        str,
    fecha:         date,
    precio:        float,
    cantidad:      int,
    stop_loss:     float | None,
    take_profit:   float | None,
    score:         float,
    detalle:       dict,
    lado:          str = "long",
) -> int:
    """
    Registra una nueva operacion virtual y descuenta capital del cash disponible.

    Logica de capital al abrir:
        capital_entrada  = precio * cantidad
        cash_disponible -= capital_entrada
        capital_inmovilizado += capital_entrada
        capital_actual queda igual (cash -> inmovilizado)

    Retorna el id de la nueva operacion.
    """
    capital_entrada = round(precio * cantidad, 2)
    engine = get_engine()

    with engine.connect() as conn:
        # Verificar cash suficiente
        row = conn.execute(text(
            "SELECT cash_disponible FROM ft_estrategias WHERE id = :eid"
        ), {"eid": estrategia_id}).fetchone()

        if not row or float(row.cash_disponible) < capital_entrada:
            log(f"  [SKIP] {ticker}: cash insuficiente para abrir ({capital_entrada:.2f})")
            return 0

        # Insertar operacion
        result = conn.execute(text("""
            INSERT INTO ft_operaciones (
                estrategia_id, ticker, lado,
                fecha_entrada, precio_entrada, cantidad, capital_entrada,
                stop_loss, take_profit, score_entrada, detalle_entrada
            ) VALUES (
                :eid, :ticker, :lado,
                :fecha, :precio, :cantidad, :capital,
                :sl, :tp, :score, :detalle
            )
            RETURNING id
        """), {
            "eid":     estrategia_id,
            "ticker":  ticker,
            "lado":    lado,
            "fecha":   fecha,
            "precio":  precio,
            "cantidad": cantidad,
            "capital": capital_entrada,
            "sl":      stop_loss,
            "tp":      take_profit,
            "score":   score,
            "detalle": json.dumps(detalle),
        })
        op_id = result.fetchone()[0]

        # Actualizar cash de la estrategia
        conn.execute(text("""
            UPDATE ft_estrategias
            SET cash_disponible      = cash_disponible - :capital,
                capital_inmovilizado = capital_inmovilizado + :capital
            WHERE id = :eid
        """), {"capital": capital_entrada, "eid": estrategia_id})

        conn.commit()

    return op_id


# ── Cerrar operacion ──────────────────────────────────────────────────────────

def cerrar_operacion(
    op_id:         int,
    estrategia_id: int,
    precio_salida: float,
    motivo:        str,
    fecha:         date,
) -> dict:
    """
    Cierra una operacion virtual y realiza el PnL en el cash disponible.

    Logica de capital al cerrar:
        pnl              = (precio_salida - precio_entrada) * cantidad
        capital_salida   = precio_salida * cantidad
        cash_disponible += capital_salida      (recibe lo que vale hoy)
        capital_inmovilizado -= capital_entrada (libera lo que habia bloqueado)
        capital_actual = cash + inmovilizado   (actualizado)

    Retorna dict con {ticker, pnl, pnl_pct, motivo}.
    """
    engine = get_engine()

    with engine.connect() as conn:
        op = conn.execute(text("""
            SELECT ticker, precio_entrada, cantidad, capital_entrada
            FROM ft_operaciones
            WHERE id = :op_id AND fecha_salida IS NULL
        """), {"op_id": op_id}).fetchone()

        if not op:
            log(f"  [WARN] Operacion {op_id} no encontrada o ya cerrada.")
            return {}

        precio_entrada  = float(op.precio_entrada)
        cantidad        = int(op.cantidad)
        capital_entrada = float(op.capital_entrada)
        ticker          = op.ticker

        pnl             = round((precio_salida - precio_entrada) * cantidad, 2)
        pnl_pct         = round((precio_salida / precio_entrada - 1) * 100, 4)
        capital_salida  = round(precio_salida * cantidad, 2)

        # Actualizar operacion
        conn.execute(text("""
            UPDATE ft_operaciones
            SET fecha_salida  = :fecha,
                precio_salida = :precio_salida,
                pnl           = :pnl,
                pnl_pct       = :pnl_pct,
                motivo_salida = :motivo
            WHERE id = :op_id
        """), {
            "fecha":         fecha,
            "precio_salida": precio_salida,
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "motivo":        motivo,
            "op_id":         op_id,
        })

        # Actualizar capital de la estrategia
        conn.execute(text("""
            UPDATE ft_estrategias
            SET cash_disponible      = cash_disponible + :capital_salida,
                capital_inmovilizado = capital_inmovilizado - :capital_entrada,
                capital_actual       = cash_disponible + :capital_salida
                                       + (capital_inmovilizado - :capital_entrada)
            WHERE id = :eid
        """), {
            "capital_salida":  capital_salida,
            "capital_entrada": capital_entrada,
            "eid":             estrategia_id,
        })

        conn.commit()

    return {
        "ticker":  ticker,
        "pnl":     pnl,
        "pnl_pct": pnl_pct,
        "motivo":  motivo,
    }


# ── Actualizar Stop Loss (trailing) ──────────────────────────────────────────

def actualizar_stop_loss(op_id: int, nuevo_sl: float) -> None:
    """Actualiza el stop loss de una operacion abierta (uso: trailing SL)."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE ft_operaciones
            SET stop_loss = :sl
            WHERE id = :op_id AND fecha_salida IS NULL
        """), {"sl": nuevo_sl, "op_id": op_id})
        conn.commit()


# ── Metricas diarias ──────────────────────────────────────────────────────────

def registrar_metricas_diarias(estrategia_id: int, fecha: date) -> None:
    """
    Registra (o actualiza) el snapshot diario de una estrategia.
    Idempotente: si ya existe la fila para esa fecha la sobreescribe.
    """
    engine = get_engine()

    with engine.connect() as conn:
        # Estado actual de la estrategia
        est = conn.execute(text("""
            SELECT capital_inicial, capital_actual,
                   cash_disponible, capital_inmovilizado
            FROM ft_estrategias
            WHERE id = :eid
        """), {"eid": estrategia_id}).fetchone()

        if not est:
            log(f"  [WARN] Estrategia {estrategia_id} no encontrada para metricas.")
            return

        capital_inicial      = float(est.capital_inicial)
        capital_actual       = float(est.capital_actual)
        cash_disponible      = float(est.cash_disponible)
        capital_inmovilizado = float(est.capital_inmovilizado)

        # Posiciones abiertas
        n_abiertas = conn.execute(text("""
            SELECT COUNT(*) FROM ft_operaciones
            WHERE estrategia_id = :eid AND fecha_salida IS NULL
        """), {"eid": estrategia_id}).scalar() or 0

        # Operaciones cerradas hoy y PnL del dia
        row_hoy = conn.execute(text("""
            SELECT COUNT(*) AS n, COALESCE(SUM(pnl), 0) AS pnl_total
            FROM ft_operaciones
            WHERE estrategia_id = :eid AND fecha_salida = :fecha
        """), {"eid": estrategia_id, "fecha": fecha}).fetchone()

        ops_cerradas_hoy = int(row_hoy.n)
        pnl_dia          = float(row_hoy.pnl_total)
        retorno_acum     = round((capital_actual - capital_inicial) / capital_inicial * 100, 4)

        # INSERT / UPDATE (idempotente)
        conn.execute(text("""
            INSERT INTO ft_metricas_diarias (
                estrategia_id, fecha,
                capital_total, cash_disponible, capital_inmovilizado,
                posiciones_abiertas, operaciones_cerradas_dia,
                pnl_dia, retorno_acumulado_pct
            ) VALUES (
                :eid, :fecha,
                :capital_total, :cash, :inmovilizado,
                :n_abiertas, :ops_cerradas,
                :pnl_dia, :retorno
            )
            ON CONFLICT (estrategia_id, fecha) DO UPDATE SET
                capital_total            = EXCLUDED.capital_total,
                cash_disponible          = EXCLUDED.cash_disponible,
                capital_inmovilizado     = EXCLUDED.capital_inmovilizado,
                posiciones_abiertas      = EXCLUDED.posiciones_abiertas,
                operaciones_cerradas_dia = EXCLUDED.operaciones_cerradas_dia,
                pnl_dia                  = EXCLUDED.pnl_dia,
                retorno_acumulado_pct    = EXCLUDED.retorno_acumulado_pct
        """), {
            "eid":          estrategia_id,
            "fecha":        fecha,
            "capital_total": capital_actual,
            "cash":          cash_disponible,
            "inmovilizado":  capital_inmovilizado,
            "n_abiertas":    n_abiertas,
            "ops_cerradas":  ops_cerradas_hoy,
            "pnl_dia":       pnl_dia,
            "retorno":       retorno_acum,
        })

        conn.commit()

    log(f"  [METRICAS] capital={capital_actual:,.2f} | "
        f"cash={cash_disponible:,.2f} | "
        f"inmov={capital_inmovilizado:,.2f} | "
        f"retorno={retorno_acum:+.2f}%")


# ── Position sizing ───────────────────────────────────────────────────────────

# ── Candidatos diarios (oportunidades) ───────────────────────────────────────

def registrar_candidatos_diarios(
    estrategia_id: int,
    fecha:         date,
    candidatos:    list[dict],
) -> None:
    """
    Registra todos los candidatos de entrada del dia en ft_candidatos_diarios.
    Incluye tanto los que entraron (entro=True) como los que no pudieron
    entrar por limite de capital o posiciones (entro=False).

    Idempotente via ON CONFLICT DO UPDATE.

    candidatos: lista de dicts con keys:
        ticker      : str
        score       : float
        entro       : bool    (True si se abrio posicion ese dia)
        motivo_skip : str | None  (razon por la que no entro, o None si entro)
    """
    if not candidatos:
        return

    engine = get_engine()
    with engine.connect() as conn:
        for c in candidatos:
            conn.execute(text("""
                INSERT INTO ft_candidatos_diarios
                    (estrategia_id, fecha, ticker, score, entro, motivo_skip)
                VALUES
                    (:eid, :fecha, :ticker, :score, :entro, :motivo_skip)
                ON CONFLICT (estrategia_id, fecha, ticker) DO UPDATE SET
                    score       = EXCLUDED.score,
                    entro       = EXCLUDED.entro,
                    motivo_skip = EXCLUDED.motivo_skip
            """), {
                "eid":         estrategia_id,
                "fecha":       fecha,
                "ticker":      c["ticker"],
                "score":       c["score"],
                "entro":       c["entro"],
                "motivo_skip": c.get("motivo_skip"),
            })
        conn.commit()

    n_entro = sum(1 for c in candidatos if c["entro"])
    n_oport = len(candidatos) - n_entro
    log(f"  [CANDIDATOS] {len(candidatos)} guardados — "
        f"{n_entro} abiertos, {n_oport} oportunidades")


def calcular_cash_desplegable(
    estrategia:     dict,
    max_deploy_pct: float = 0.80,
) -> float:
    """
    Retorna el cash efectivamente disponible para nuevas posiciones,
    respetando el limite maximo de despliegue sobre el capital_actual.

    Logica:
        max_desplegable = capital_actual * max_deploy_pct  (ej: 80.000 sobre 100.000)
        headroom        = max_desplegable - capital_inmovilizado
        resultado       = min(cash_disponible, headroom)

    Si el portfolio ya supera el techo (ej: posiciones valen mas por PnL), retorna 0.
    """
    capital_actual       = float(estrategia["capital_actual"])
    capital_inmovilizado = float(estrategia["capital_inmovilizado"])
    cash_disponible      = float(estrategia["cash_disponible"])

    max_desplegable = capital_actual * max_deploy_pct
    headroom        = max(0.0, max_desplegable - capital_inmovilizado)
    return min(cash_disponible, headroom)


def calcular_qty_ft(
    cash_disponible: float,
    precio:          float,
    riesgo_pct:      float = 0.15,
    capital_actual:  float = None,
) -> int:
    """
    Calcula la cantidad de shares para una operacion FT.

    Usa el MENOR entre:
        - capital_actual * riesgo_pct  (15% del capital total)
        - cash_disponible              (no gastar mas de lo disponible)

    Retorna 0 si no alcanza para 1 share.
    """
    base          = capital_actual if capital_actual else cash_disponible
    capital_trade = min(base * riesgo_pct, cash_disponible)
    qty           = int(capital_trade / precio)
    return max(qty, 0)
