"""
senales_adapter.py -- Data adapter de los bots Alpaca (Plan B, Tarea 16).

Lee la tabla MASTICADA senales_bot_diaria (Railway, via get_engine -> en GH
Actions DATABASE_URL apunta a Railway) y arma los inputs que espera el cerebro
de decision COMPARTIDO (src/strategies):

    leer_senales()         -> (filas, fecha)   1 fila por ticker del dia
    construir_pcr_map()    -> {ticker: {corto, medio, largo, pcr_score, valido}}
    construir_ml_senales() -> [{ticker, nivel, score, scan_fecha}] (COMPRA_FUERTE+)
    construir_precios()    -> {ticker: close}   precio de DECISION (homologable a FT)
    sector_por_ticker()    -> {ticker: sector}

Las filas de la masticada YA tienen la forma de `ind` para el cerebro tecnico
(ticker, sector, close, sma21/50/200, rsi14, macd, macd_signal, atr14): se pasan
tal cual como `indicadores`.

Guard de frescura: senales_frescas() compara la fecha de la masticada con la
ultima rueda esperada. El guard COMPLETO (que aborta la operacion) es Paso 4;
aca queda la primitiva.
"""

from sqlalchemy import text

from src.data.database import get_engine


def leer_senales(fecha=None) -> tuple:
    """
    Lee las filas de senales_bot_diaria para `fecha` (o la ultima disponible).
    Retorna (filas: list[dict], fecha).
    """
    eng = get_engine()
    with eng.connect() as conn:
        if fecha is None:
            fecha = conn.execute(
                text("SELECT MAX(fecha) FROM senales_bot_diaria")
            ).scalar()
        if fecha is None:
            return [], None
        rows = conn.execute(text("""
            SELECT ticker, fecha, close, sector,
                   alert_nivel, alert_score,
                   sma21, sma50, sma200, rsi14, macd, macd_signal, atr14,
                   pcr_score, pcr_valido, pcr_corto, pcr_medio, pcr_largo
            FROM senales_bot_diaria
            WHERE fecha = :f
            ORDER BY ticker
        """), {"f": fecha}).fetchall()
    return [dict(r._mapping) for r in rows], fecha


def construir_pcr_map(filas) -> dict:
    """{ticker: {corto, medio, largo, pcr_score, valido}} para el cerebro de opciones."""
    out = {}
    for r in filas:
        out[r["ticker"]] = {
            "corto":     r["pcr_corto"],
            "medio":     r["pcr_medio"],
            "largo":     r["pcr_largo"],
            "pcr_score": int(r["pcr_score"]) if r["pcr_score"] is not None else 0,
            "valido":    bool(r["pcr_valido"]),
        }
    return out


def construir_ml_senales(filas, nivel_min: str, score_min: float, fecha) -> list:
    """
    Senales ML filtradas (nivel == nivel_min y score >= score_min), ordenadas por
    score DESC -- misma forma que obtenia obtener_senales_scanner() en FT.
    """
    out = []
    for r in filas:
        nivel = r["alert_nivel"]
        score = r["alert_score"]
        if nivel == nivel_min and score is not None and float(score) >= score_min:
            out.append({
                "ticker":     r["ticker"],
                "nivel":      nivel,
                "score":      float(score),
                "scan_fecha": str(fecha),
            })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def construir_precios(filas) -> dict:
    """{ticker: close} -- precio de DECISION (la ejecucion usa live de Alpaca)."""
    return {r["ticker"]: float(r["close"]) for r in filas if r["close"] is not None}


def sector_por_ticker(filas) -> dict:
    """{ticker: sector} para enriquecer las posiciones abiertas."""
    return {r["ticker"]: r["sector"] for r in filas}


def fecha_esperada_hoy(hoy=None):
    """
    Ultima rueda cuyo cierre deberia estar en la masticada cuando el bot corre.

    El bot corre intradia (15:15 UTC) sobre el cierre de la rueda ANTERIOR
    (lag 1 dia, igual que FT): el productor arma la masticada la noche previa,
    tras ft_run_diario. Por eso la fecha esperada = ultima rueda estrictamente
    anterior a hoy (prev_trading_day), tanto en dia habil como en feriado/finde.
    """
    from datetime import date as _date
    from src.utils.trading_calendar import prev_trading_day
    return prev_trading_day(hoy or _date.today())


def senales_frescas(fecha_masticada, fecha_esperada) -> bool:
    """
    Guard de frescura (Paso 4): True si la masticada es del dia esperado. Si no
    se pasa fecha_esperada, no puede juzgar -> True (no bloquea).
    """
    if fecha_esperada is None or fecha_masticada is None:
        return True
    return str(fecha_masticada) == str(fecha_esperada)
