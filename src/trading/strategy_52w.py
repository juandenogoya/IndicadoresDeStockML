"""
strategy_52w.py
Estrategia de Ruptura de Maximo de 52 Semanas.

Basada en George & Hwang (2004): la proximidad al maximo de 52 semanas
predice retornos futuros positivos. Inversores anclan el precio al maximo
historico reciente — al romperse, se produce aceleracion.

Filtros obligatorios:
    dist_max_52w >= -8%    (dentro del 8% del maximo)
    close > sma200         (tendencia alcista de largo plazo)

Scoring (0-4 pts, minimo 3 para operar):
    +1  Aproximacion      : dist_max_52w >= -3% (muy cerca del maximo)
    +1  Ruptura real      : dist_max_52w >  0%  (nuevo maximo de 52 semanas)
    +1  Volumen confirma  : vol_relativo >= 1.5 o vol_spike = 1
    +1  Tendencia media   : close > sma50

Salidas:
    SL   = entrada - 2.0 x ATR14
    TP   = entrada + 6.0 x ATR14  (ratio 1:3)
    SMA50 rota 2 dias consecutivos -> salir (soporte perdido)
    Time stop = 60 dias habiles
"""

from datetime import date
from sqlalchemy import text
from src.data.database import get_engine
from src.trading import alpaca_client, risk

# ── Parametros de scoring ─────────────────────────────────────
PTS_APROX      = 1.0
PTS_RUPTURA    = 1.0
PTS_VOLUMEN    = 1.0
PTS_SMA50      = 1.0
SCORE_MAXIMO   = PTS_APROX + PTS_RUPTURA + PTS_VOLUMEN + PTS_SMA50  # 4.0
SCORE_ENTRADA  = 3.0

# Thresholds
DIST_FILTRO    = -0.08   # dentro del 8% del maximo (filtro obligatorio)
DIST_APROX     = -0.03   # dentro del 3% = aproximacion
DIST_RUPTURA   =  0.00   # por encima = ruptura real
VOL_RELATIVO   =  1.5    # volumen 50% mayor al promedio

# Parametros de salida
ATR_MULT_SL    = 2.0     # SL mas amplio para hold largo
ATR_MULT_TP    = 6.0     # TP ambicioso (semanas/meses)
DIAS_MAX_POS   = 60      # time-stop: 60 dias habiles
SMA50_DIAS_NEG = 2       # dias bajo SMA50 para confirmar salida


# ─────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────

def obtener_datos_52w() -> list[dict]:
    """
    Calcula el maximo de 52 semanas y une con indicadores tecnicos
    y features de precio-accion para cada ticker.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            WITH precios_recientes AS (
                SELECT DISTINCT ON (ticker)
                    ticker, fecha, close
                FROM precios_diarios
                ORDER BY ticker, fecha DESC
            ),
            max_52w AS (
                SELECT ticker,
                       MAX(close) AS max_52w,
                       MIN(close) AS min_52w
                FROM precios_diarios
                WHERE fecha >= CURRENT_DATE - INTERVAL '52 weeks'
                GROUP BY ticker
            )
            SELECT
                pr.ticker,
                pr.fecha,
                pr.close,
                m.max_52w,
                m.min_52w,
                ROUND((pr.close - m.max_52w) / m.max_52w, 6) AS dist_max_52w,
                ROUND((pr.close - m.min_52w) / m.min_52w, 6) AS dist_min_52w,
                i.sma50,
                i.sma200,
                i.rsi14,
                i.atr14,
                i.vol_relativo,
                pa.vol_spike,
                pa.es_alcista,
                pa.tendencia_velas,
                pa.pos_rango_20d
            FROM precios_recientes pr
            JOIN max_52w m ON m.ticker = pr.ticker
            JOIN indicadores_tecnicos i
              ON i.ticker = pr.ticker AND i.fecha = pr.fecha
            LEFT JOIN features_precio_accion pa
              ON pa.ticker = pr.ticker AND pa.fecha = pr.fecha
            ORDER BY dist_max_52w DESC
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def obtener_historial_sma50(ticker: str, dias: int = 3) -> list[dict]:
    """
    Retorna los ultimos N cierres con SMA50 para detectar ruptura consecutiva.
    Orden: mas reciente primero.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.close, i.sma50
            FROM precios_diarios p
            JOIN indicadores_tecnicos i
              ON i.ticker = p.ticker AND i.fecha = p.fecha
            WHERE p.ticker = :ticker
            ORDER BY p.fecha DESC
            LIMIT :dias
        """), {"ticker": ticker, "dias": dias}).fetchall()
    return [{"close": float(r.close), "sma50": float(r.sma50)} for r in rows]


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────

def calcular_score_52w(row: dict) -> tuple[float, dict]:
    """
    Calcula el score de ruptura de 52 semanas.
    Retorna (score, detalle). Score=0 si filtros obligatorios fallan.
    """
    close        = float(row.get("close",       0) or 0)
    max_52w      = float(row.get("max_52w",     0) or 0)
    dist_max_52w = float(row.get("dist_max_52w",  -1) or -1)
    sma50        = float(row.get("sma50",        0) or 0)
    sma200       = float(row.get("sma200",       0) or 0)
    vol_rel      = float(row.get("vol_relativo", 0) or 0)
    vol_spike    = int(row.get("vol_spike",      0) or 0)

    detalle = {
        "filtro_dist":   dist_max_52w >= DIST_FILTRO,
        "filtro_sma200": close > sma200 if sma200 else False,
        "cond_aprox":    dist_max_52w >= DIST_APROX,
        "cond_ruptura":  dist_max_52w >  DIST_RUPTURA,
        "cond_volumen":  vol_rel >= VOL_RELATIVO or bool(vol_spike),
        "cond_sma50":    close > sma50 if sma50 else False,
        "dist_max_52w":  round(dist_max_52w * 100, 2),   # en %
        "max_52w":       round(max_52w, 2),
        "close":         round(close, 2),
        "vol_relativo":  round(vol_rel, 2),
        "sma50":         round(sma50, 2),
        "sma200":        round(sma200, 2),
    }

    # Filtros obligatorios
    if not detalle["filtro_dist"] or not detalle["filtro_sma200"]:
        return 0.0, detalle

    score = 0.0
    if detalle["cond_aprox"]:   score += PTS_APROX
    if detalle["cond_ruptura"]: score += PTS_RUPTURA
    if detalle["cond_volumen"]: score += PTS_VOLUMEN
    if detalle["cond_sma50"]:   score += PTS_SMA50

    return round(score, 2), detalle


# ─────────────────────────────────────────────────────────────
# Logica de entradas
# ─────────────────────────────────────────────────────────────

def evaluar_entradas_52w(
    posiciones_actuales: list[dict],
    equity: float,
) -> list[dict]:
    """
    Evalua todos los tickers y retorna candidatos para abrir posicion
    basados en ruptura del maximo de 52 semanas.
    """
    tickers_abiertos = {p["ticker"] for p in posiciones_actuales}
    datos            = obtener_datos_52w()
    candidatos       = []

    for row in datos:
        ticker = row["ticker"]
        if ticker in tickers_abiertos:
            continue

        score, detalle = calcular_score_52w(row)

        if score < SCORE_ENTRADA:
            continue

        if len(posiciones_actuales) + len(candidatos) >= risk.MAX_POSICIONES:
            break

        candidatos.append({"ticker": ticker, "score": score, "detalle": detalle, "row": row})

    if not candidatos:
        return []

    # Ordenar por dist_max_52w descendente (los mas cercanos/ruptura primero)
    candidatos.sort(key=lambda x: x["detalle"]["dist_max_52w"], reverse=True)

    a_abrir = []
    for c in candidatos:
        ticker = c["ticker"]
        row    = c["row"]
        det    = c["detalle"]

        try:
            precio = alpaca_client.get_latest_price(ticker, suffix="_4")
        except Exception:
            precio = det["close"]
            if not precio:
                continue

        atr = float(row.get("atr14") or 0)
        if not atr:
            continue

        stop_loss   = round(precio - ATR_MULT_SL * atr, 4)
        take_profit = round(precio + ATR_MULT_TP * atr, 4)

        qty = risk.calcular_qty(equity, precio, equity * risk.RIESGO_POR_TRADE)
        if qty < 1:
            continue

        tipo = "RUPTURA" if det["cond_ruptura"] else "APROX"
        nivel = f"52W_{tipo}_{c['score']:.0f}/{SCORE_MAXIMO:.0f}"

        a_abrir.append({
            "ticker":       ticker,
            "precio":       precio,
            "qty":          qty,
            "capital":      round(precio * qty, 2),
            "pct_equity":   round(precio * qty / equity * 100, 1),
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "atr":          round(atr, 4),
            "score":        c["score"],
            "nivel":        nivel,
            "tipo":         tipo,
            "dist_max_52w": det["dist_max_52w"],
            "max_52w":      det["max_52w"],
            "vol_relativo": det["vol_relativo"],
            "cond_aprox":   det["cond_aprox"],
            "cond_ruptura": det["cond_ruptura"],
            "cond_volumen": det["cond_volumen"],
            "cond_sma50":   det["cond_sma50"],
        })

    return a_abrir


# ─────────────────────────────────────────────────────────────
# Logica de salidas
# ─────────────────────────────────────────────────────────────

def evaluar_cierres_52w(posiciones: list[dict]) -> list[dict]:
    """
    Evalua posiciones abiertas y detecta condiciones de salida.

    Prioridades:
        1. SL dinamico (precio <= stop_loss)
        2. TP dinamico (precio >= take_profit)
        3. SMA50 rota 2 dias consecutivos (soporte perdido)
        4. Time stop (>= DIAS_MAX_POS dias)
    """
    if not posiciones:
        return []

    datos_map = {row["ticker"]: row for row in obtener_datos_52w()}

    a_cerrar = []

    for pos in posiciones:
        ticker = pos["ticker"]
        datos  = datos_map.get(ticker)
        if not datos:
            continue

        try:
            precio_actual = alpaca_client.get_latest_price(ticker, suffix="_4")
        except Exception:
            precio_actual = float(datos.get("close") or 0)
            if not precio_actual:
                continue

        stop_loss   = float(pos["stop_loss"])   if pos.get("stop_loss")   else None
        take_profit = float(pos["take_profit"]) if pos.get("take_profit") else None
        fecha_entrada = pos["fecha_entrada"]
        dias_abierta  = (date.today() - fecha_entrada).days if fecha_entrada else 0

        motivo = None

        # ── Prioridad 1: SL ──────────────────────────────────
        if stop_loss and precio_actual <= stop_loss:
            motivo = "STOP_LOSS"

        # ── Prioridad 2: TP ──────────────────────────────────
        elif take_profit and precio_actual >= take_profit:
            motivo = "TAKE_PROFIT"

        # ── Prioridad 3: SMA50 rota dias consecutivos ────────
        else:
            historial = obtener_historial_sma50(ticker, SMA50_DIAS_NEG)
            if (len(historial) >= SMA50_DIAS_NEG and
                    all(h["close"] < h["sma50"] for h in historial)):
                motivo = f"SMA50_ROTA_{SMA50_DIAS_NEG}D"

        # ── Prioridad 4: Time stop ───────────────────────────
        if not motivo and dias_abierta >= DIAS_MAX_POS:
            motivo = f"TIME_STOP_{dias_abierta}D"

        if motivo:
            a_cerrar.append({
                **pos,
                "precio_cierre": precio_actual,
                "motivo":        motivo,
            })

    return a_cerrar
