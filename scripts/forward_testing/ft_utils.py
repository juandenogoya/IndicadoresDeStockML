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
        ticker          : str
        score           : float
        entro           : bool    (True si se abrio posicion ese dia)
        motivo_skip     : str | None
        precio_apertura : float | None  (close del dia — referencia para calcular retornos futuros)
    """
    if not candidatos:
        return

    engine = get_engine()
    with engine.connect() as conn:
        for c in candidatos:
            conn.execute(text("""
                INSERT INTO ft_candidatos_diarios
                    (estrategia_id, fecha, ticker, score, entro,
                     motivo_skip, precio_apertura)
                VALUES
                    (:eid, :fecha, :ticker, :score, :entro,
                     :motivo_skip, :precio_apertura)
                ON CONFLICT (estrategia_id, fecha, ticker) DO UPDATE SET
                    score           = EXCLUDED.score,
                    entro           = EXCLUDED.entro,
                    motivo_skip     = EXCLUDED.motivo_skip,
                    precio_apertura = EXCLUDED.precio_apertura
            """), {
                "eid":             estrategia_id,
                "fecha":           fecha,
                "ticker":          c["ticker"],
                "score":           c["score"],
                "entro":           c["entro"],
                "motivo_skip":     c.get("motivo_skip"),
                "precio_apertura": c.get("precio_apertura"),
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


# ── Helpers calendario ────────────────────────────────────────────────────────

def _hace_n_habiles(d: date, n: int) -> date:
    """Retorna la fecha que fue n dias habiles antes de d."""
    from src.utils.trading_calendar import prev_trading_day
    for _ in range(n):
        d = prev_trading_day(d)
    return d


# ── Observacion diaria de posiciones ─────────────────────────────────────────

def registrar_estado_posiciones(
    estrategia_id: int,
    fecha:         date,
    precios:       dict[str, float],
) -> None:
    """
    Inserta snapshots diarios en ft_posiciones_diarias.

    Cubre 5 estados del ciclo de vida de una operacion:
        'abierta'  - posicion activa en <fecha>
        'cierre'   - posicion cerrada exactamente en <fecha>
        'post_5d'  - operacion cerrada exactamente 5 habiles antes de <fecha>
        'post_10d' - operacion cerrada exactamente 10 habiles antes de <fecha>
        'post_20d' - operacion cerrada exactamente 20 habiles antes de <fecha>

    Los estados post_Nd permiten evaluar si el cierre fue oportuno:
    si el precio sube despues de cerrar, el exit fue prematuro.

    Idempotente via ON CONFLICT (operacion_id, fecha) DO NOTHING.
    """
    from src.utils.trading_calendar import trading_days_between
    from scripts.forward_testing.ft_scoring import (
        obtener_candle_score_5d,
        obtener_indicadores_hoy,
        calcular_score_tecnico,
    )

    fecha_5  = _hace_n_habiles(fecha, 5)
    fecha_10 = _hace_n_habiles(fecha, 10)
    fecha_20 = _hace_n_habiles(fecha, 20)

    engine = get_engine()

    # ── Recopilar operaciones de interes ─────────────────────────────────────
    with engine.connect() as conn:
        abiertas = conn.execute(text("""
            SELECT id, ticker, fecha_entrada, precio_entrada
            FROM ft_operaciones
            WHERE estrategia_id = :eid AND fecha_salida IS NULL
        """), {"eid": estrategia_id}).fetchall()

        cerradas_hoy = conn.execute(text("""
            SELECT id, ticker, fecha_entrada, precio_entrada, motivo_salida
            FROM ft_operaciones
            WHERE estrategia_id = :eid AND fecha_salida = :fecha
        """), {"eid": estrategia_id, "fecha": fecha}).fetchall()

        counterfactual = []
        for estado, fs in [
            ("post_5d",  fecha_5),
            ("post_10d", fecha_10),
            ("post_20d", fecha_20),
        ]:
            ops = conn.execute(text("""
                SELECT id, ticker, fecha_entrada, precio_entrada
                FROM ft_operaciones
                WHERE estrategia_id = :eid AND fecha_salida = :fs
            """), {"eid": estrategia_id, "fs": fs}).fetchall()
            for op in ops:
                counterfactual.append((estado, op))

    # ── Tickers involucrados ──────────────────────────────────────────────────
    all_tickers = set()
    for op in list(abiertas) + list(cerradas_hoy):
        all_tickers.add(op.ticker)
    for _, op in counterfactual:
        all_tickers.add(op.ticker)

    if not all_tickers:
        return

    tickers_list = list(all_tickers)

    # ── Batch: rango 5d y vol desde tablas de features ───────────────────────
    with engine.connect() as conn:
        rango_rows = conn.execute(text("""
            WITH ranked AS (
                SELECT ticker, close,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
                FROM precios_diarios
                WHERE ticker = ANY(:tickers)
            )
            SELECT ticker, MAX(close) - MIN(close) AS rango_5d_abs
            FROM ranked
            WHERE rn <= 5
            GROUP BY ticker
        """), {"tickers": tickers_list}).fetchall()
        rango_map = {r.ticker: float(r.rango_5d_abs) for r in rango_rows}

        fms_rows = conn.execute(text("""
            WITH ranked AS (
                SELECT ticker, es_alcista, vol_price_confirm, vol_price_diverge,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
                FROM features_market_structure
                WHERE ticker = ANY(:tickers)
            )
            SELECT ticker,
                   MAX(CASE WHEN rn = 1 THEN vol_price_confirm ELSE NULL END) AS vpc,
                   MAX(CASE WHEN rn = 1 THEN vol_price_diverge ELSE NULL END) AS vpd,
                   SUM(CASE WHEN es_alcista = 1
                             AND vol_price_confirm = 1 THEN 1 ELSE 0 END) AS up_vol_5d
            FROM ranked
            WHERE rn <= 5
            GROUP BY ticker
        """), {"tickers": tickers_list}).fetchall()
        fms_map = {r.ticker: dict(r._mapping) for r in fms_rows}

    # ── Scores de velas y tecnicos (queries batch) ────────────────────────────
    candle_scores   = obtener_candle_score_5d()
    indicadores_lst = obtener_indicadores_hoy()
    ind_map         = {i["ticker"]: i for i in indicadores_lst}

    def _tech_score(ticker):
        ind = ind_map.get(ticker)
        if not ind:
            return None
        score, _ = calcular_score_tecnico(ind)
        return float(score)

    def _atr(ticker):
        ind = ind_map.get(ticker) or {}
        v = ind.get("atr14")
        return float(v) if v else None

    # ── Construir filas ───────────────────────────────────────────────────────
    def _make_fila(op, estado, motivo_sal=None):
        ticker         = op.ticker
        precio_entrada = float(op.precio_entrada)
        precio_cierre  = precios.get(ticker)

        retorno_pct = None
        if precio_cierre and precio_entrada:
            retorno_pct = round((precio_cierre / precio_entrada - 1) * 100, 4)

        dias = None
        try:
            entrada = op.fecha_entrada
            if hasattr(entrada, "date"):
                entrada = entrada.date()
            dias = trading_days_between(entrada, fecha)
        except Exception:
            pass

        atr14        = _atr(ticker)
        rango_5d_abs = rango_map.get(ticker)
        rango_5d_pct  = None
        lateral_ratio = None
        if rango_5d_abs is not None and precio_entrada > 0:
            rango_5d_pct = round(rango_5d_abs / precio_entrada * 100, 4)
        if rango_5d_abs is not None and atr14 and atr14 > 0:
            lateral_ratio = round(rango_5d_abs / atr14, 4)

        fms = fms_map.get(ticker, {})
        return {
            "estrategia_id":      estrategia_id,
            "operacion_id":       op.id,
            "ticker":             ticker,
            "fecha":              fecha,
            "estado":             estado,
            "precio_cierre":      precio_cierre,
            "precio_entrada_ref": precio_entrada,
            "retorno_pct":        retorno_pct,
            "dias_abierta":       dias,
            "tech_score":         _tech_score(ticker),
            "candle_score_5d":    candle_scores.get(ticker),
            "rango_5d_pct":       rango_5d_pct,
            "atr14":              atr14,
            "lateral_ratio":      lateral_ratio,
            "vol_price_confirm":  fms.get("vpc"),
            "vol_price_diverge":  fms.get("vpd"),
            "up_vol_5d":          fms.get("up_vol_5d"),
            "motivo_salida":      motivo_sal,
        }

    filas = []
    for op in abiertas:
        filas.append(_make_fila(op, "abierta"))
    for op in cerradas_hoy:
        filas.append(_make_fila(op, "cierre",
                                getattr(op, "motivo_salida", None)))
    for estado, op in counterfactual:
        filas.append(_make_fila(op, estado))

    if not filas:
        return

    # ── Insertar en batch ─────────────────────────────────────────────────────
    with engine.connect() as conn:
        for f in filas:
            conn.execute(text("""
                INSERT INTO ft_posiciones_diarias (
                    estrategia_id, operacion_id, ticker, fecha, estado,
                    precio_cierre, precio_entrada_ref, retorno_pct,
                    dias_abierta, tech_score, candle_score_5d,
                    rango_5d_pct, atr14, lateral_ratio,
                    vol_price_confirm, vol_price_diverge, up_vol_5d,
                    motivo_salida
                ) VALUES (
                    :estrategia_id, :operacion_id, :ticker, :fecha, :estado,
                    :precio_cierre, :precio_entrada_ref, :retorno_pct,
                    :dias_abierta, :tech_score, :candle_score_5d,
                    :rango_5d_pct, :atr14, :lateral_ratio,
                    :vol_price_confirm, :vol_price_diverge, :up_vol_5d,
                    :motivo_salida
                )
                ON CONFLICT (operacion_id, fecha) DO NOTHING
            """), f)
        conn.commit()

    n_ab = sum(1 for f in filas if f["estado"] == "abierta")
    n_ci = sum(1 for f in filas if f["estado"] == "cierre")
    n_po = len(filas) - n_ab - n_ci
    log(f"  [POSICIONES] {len(filas)} snapshots — "
        f"{n_ab} abiertas, {n_ci} cierres, {n_po} post-cierre")


# ── Backfill retornos de candidatos ───────────────────────────────────────────

def backfill_retornos_candidatos(
    estrategia_id: int,
    fecha:         date,
    precios:       dict[str, float],
) -> None:
    """
    Calcula y persiste retorno_5d, retorno_10d, retorno_20d
    en ft_candidatos_diarios para candidatos identificados hace N dias habiles.

    Logica:
        Para N en {5, 10, 20}:
            Busca candidatos de la fecha que fue N habiles antes de <fecha>
            con precio_apertura guardado pero retorno_Nd aun NULL.
            Retorno = (precio_actual - precio_apertura) / precio_apertura * 100

    Permite comparar: "si no entre en X, cuanto me habria ganado/perdido N dias despues?"
    Idempotente: solo actualiza filas con retorno_Nd IS NULL.
    """
    engine = get_engine()

    for n, col in [(5, "retorno_5d"), (10, "retorno_10d"), (20, "retorno_20d")]:
        fecha_n = _hace_n_habiles(fecha, n)

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT ticker, precio_apertura
                FROM ft_candidatos_diarios
                WHERE estrategia_id = :eid
                  AND fecha = :fecha_n
                  AND precio_apertura IS NOT NULL
                  AND {col} IS NULL
            """), {"eid": estrategia_id, "fecha_n": fecha_n}).fetchall()

            if not rows:
                continue

            actualizados = 0
            for row in rows:
                precio_actual   = precios.get(row.ticker)
                precio_apertura = float(row.precio_apertura)
                if not precio_actual or precio_apertura == 0:
                    continue
                retorno = round(
                    (precio_actual - precio_apertura) / precio_apertura * 100, 4
                )
                conn.execute(text(f"""
                    UPDATE ft_candidatos_diarios
                    SET {col} = :retorno
                    WHERE estrategia_id = :eid
                      AND fecha = :fecha_n
                      AND ticker = :ticker
                """), {
                    "retorno": retorno,
                    "eid":     estrategia_id,
                    "fecha_n": fecha_n,
                    "ticker":  row.ticker,
                })
                actualizados += 1

            if actualizados:
                conn.commit()
                log(f"  [RETORNOS] {col}: {actualizados} candidatos "
                    f"actualizados (ref={fecha_n})")
