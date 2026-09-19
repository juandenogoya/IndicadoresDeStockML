"""
ft_scoring_estructura.py
Insumos de la estrategia SMC leidos de la historia SIN informacion futura:
`features_estructura` (swings CONFIRMADOS, src/indicators/estructura.py) y
`features_velas` (patrones clasicos con contexto, src/indicators/velas.py).

POR QUE EXISTE (docs/estructura_velas.md sec. 4 y 9.3)
    ft_scoring.obtener_features_hoy lee `features_market_structure`, cuya historia
    mira N ruedas al futuro (swings con ventana centrada) y cuya ULTIMA fila trae
    swings PROVISIONALES, que pueden desaparecer al dia siguiente. En vivo eso no
    es ver el futuro -- el bot lee la ultima fila -- pero si es una senal inestable.
    El backtest 2021-09 -> 2026-09 midio que la regla de SMC_v1 con N=10 confirmado
    rinde +13,5% (contra +61% con la historia que mira al futuro), y que confirmando
    rapido, N=5 y N=3, rinde ~+58-60%. De ahi salen FT_SMC_v3_N5 y FT_SMC_v3_N3.

QUE HACE DISTINTO A ft_scoring.py
    1. Fuente: features_estructura + features_velas en vez de
       features_market_structure + features_precio_accion.
    2. Ventana N parametrizable (3, 5 o 10; las tres estan persistidas).
    3. El lookback de eventos se ancla a la ULTIMA RUEDA DE DATOS de la tabla, no
       a CURRENT_DATE. La rutina es manual: con el reloj, una corrida atrasada
       mira menos ruedas de las que dice mirar (misma familia que fecha_datos vs
       fecha_entrada, CLAUDE.md "FT asincronico").
    4. es_alcista se DERIVA del precio (close > open). features_velas no la guarda:
       es una propiedad de la vela, no un patron.

LO QUE NO CAMBIA
    El scoring es el MISMO: estas queries devuelven las claves con el sufijo _10 que
    espera ft_scoring.calcular_score_estructura (aliasing en SQL), asi que la logica
    de decision no se duplica ni se toca. La ventana real viaja en la clave
    `ventana` de cada fila, para que quede en el detalle de la operacion.
    Toda diferencia de resultado contra FT_SMC_v1 es atribuible a la FUENTE, no a
    una reimplementacion del score.

Funciones:
    obtener_features_hoy(ventana)            -> list[dict]
    obtener_estructura_tickers(tickers, ventana) -> dict[str, dict]
    calcular_actualizaciones_sl(posiciones, ventana) -> list[dict]
"""

from sqlalchemy import text

from src.data.database import get_engine
from src.indicators.estructura import VENTANAS_TABLA
from scripts.forward_testing.ft_scoring import (
    LOOKBACK_DIAS,
    _swing_low_precio,
)

TABLA_ESTRUCTURA = "features_estructura"
TABLA_VELAS = "features_velas"

# Contrato con ft_scoring.calcular_score_estructura: las claves que lee de cada
# fila. El sufijo _10 es historico (la v1 solo tenia N=10); aca la ventana real
# viaja en la clave `ventana`. Lo verifica tests/test_ft_scoring_estructura.py.
CLAVES_SCORE = (
    "tuvo_choch_bull", "tuvo_bos_bull", "estructura_10", "choch_bear_10",
    "dist_sl_10_pct", "dist_sh_10_pct", "es_alcista", "vol_spike",
    "patron_engulfing_bull", "patron_hammer", "close",
)


def _validar(ventana: int) -> int:
    n = int(ventana)
    if n not in VENTANAS_TABLA:
        raise ValueError(
            f"ventana={n} no esta en {TABLA_ESTRUCTURA} (persistidas: {VENTANAS_TABLA}). "
            f"Agregarla en estructura.VENTANAS_TABLA y correr "
            f"compute_estructura_velas.py --crear --completo")
    return n


def sql_features_hoy(n: int) -> str:
    """SQL de obtener_features_hoy. Afuera para poder testear el aliasing sin DB."""
    return f"""
            WITH ultima_rueda AS (
                SELECT MAX(fecha) AS fecha FROM {TABLA_ESTRUCTURA}
            ),
            eventos AS (
                SELECT
                    e.ticker,
                    MAX(CASE WHEN e.choch_bull_{n} = 1 THEN 1 ELSE 0 END) AS tuvo_choch_bull,
                    MAX(CASE WHEN e.bos_bull_{n}   = 1 THEN 1 ELSE 0 END) AS tuvo_bos_bull
                FROM {TABLA_ESTRUCTURA} e, ultima_rueda u
                WHERE e.fecha >= u.fecha - INTERVAL '{LOOKBACK_DIAS} days'
                GROUP BY e.ticker
                HAVING MAX(CASE WHEN e.choch_bull_{n} = 1 THEN 1 ELSE 0 END) = 1
                    OR MAX(CASE WHEN e.bos_bull_{n}   = 1 THEN 1 ELSE 0 END) = 1
            ),
            estado_actual AS (
                SELECT DISTINCT ON (es.ticker)
                    es.ticker,
                    es.fecha,
                    es.estructura_{n}     AS estructura_10,
                    es.choch_bull_{n}     AS choch_bull_10,
                    es.choch_bear_{n}     AS choch_bear_10,
                    es.bos_bull_{n}       AS bos_bull_10,
                    es.dist_sh_{n}_pct    AS dist_sh_10_pct,
                    es.dist_sl_{n}_pct    AS dist_sl_10_pct,
                    es.dias_sh_{n}        AS dias_sh_10,
                    es.dias_sl_{n}        AS dias_sl_10
                FROM {TABLA_ESTRUCTURA} es
                ORDER BY es.ticker, es.fecha DESC
            )
            SELECT
                ea.*,
                {n}                                       AS ventana,
                ev.tuvo_choch_bull,
                ev.tuvo_bos_bull,
                CASE WHEN p.close > p.open THEN 1 ELSE 0 END AS es_alcista,
                vl.patron_engulfing_bull,
                vl.patron_hammer,
                COALESCE(pa.vol_spike, 0)                 AS vol_spike,
                pa.up_vol_5d,
                i.rsi14,
                i.atr14,
                i.sma200,
                p.close
            FROM estado_actual ea
            JOIN eventos ev
              ON ev.ticker = ea.ticker
            JOIN {TABLA_VELAS} vl
              ON vl.ticker = ea.ticker AND vl.fecha = ea.fecha
            JOIN indicadores_tecnicos i
              ON i.ticker = ea.ticker AND i.fecha = ea.fecha
            JOIN precios_diarios p
              ON p.ticker = ea.ticker AND p.fecha = ea.fecha
            LEFT JOIN features_precio_accion pa
              ON pa.ticker = ea.ticker AND pa.fecha = ea.fecha
    """


def obtener_features_hoy(ventana: int) -> list:
    """
    Tickers con CHoCH o BOS bull CONFIRMADO en el lookback, con su estado actual.

    Las columnas de la ventana N se devuelven con los nombres _10 que espera
    calcular_score_estructura. `ventana` lleva el N real.
    """
    n = _validar(ventana)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql_features_hoy(n))).fetchall()
    return [dict(r._mapping) for r in rows]


def sql_estructura_tickers(n: int) -> str:
    """SQL de obtener_estructura_tickers. Afuera para testear el aliasing sin DB."""
    return f"""
            SELECT DISTINCT ON (es.ticker)
                es.ticker,
                es.fecha,
                {n}                    AS ventana,
                es.estructura_{n}      AS estructura_10,
                es.choch_bear_{n}      AS choch_bear_10,
                es.bos_bear_{n}        AS bos_bear_10,
                es.dist_sl_{n}_pct     AS dist_sl_10_pct,
                es.dist_sh_{n}_pct     AS dist_sh_10_pct,
                p.close
            FROM {TABLA_ESTRUCTURA} es
            JOIN precios_diarios p
              ON p.ticker = es.ticker AND p.fecha = es.fecha
            WHERE es.ticker = ANY(:tickers)
            ORDER BY es.ticker, es.fecha DESC
    """


def obtener_estructura_tickers(tickers: list, ventana: int) -> dict:
    """
    Ultima fila de features_estructura + precio para los tickers dados
    (gestion de posiciones abiertas: cierres y trailing SL).
    """
    if not tickers:
        return {}
    n = _validar(ventana)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql_estructura_tickers(n)),
                            {"tickers": list(tickers)}).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


def calcular_actualizaciones_sl(posiciones: list, ventana: int) -> list:
    """
    Trailing SL estructural sobre el ultimo swing low CONFIRMADO. El SL solo sube.
    Misma formula que ft_scoring.calcular_actualizaciones_sl; cambia la fuente.
    """
    if not posiciones:
        return []

    n = _validar(ventana)
    datos_map = obtener_estructura_tickers([p["ticker"] for p in posiciones], n)
    updates = []

    for pos in posiciones:
        ticker = pos["ticker"]
        datos_hoy = datos_map.get(ticker)
        if not datos_hoy:
            continue

        sl_actual = float(pos.get("stop_loss") or 0)
        if sl_actual <= 0:
            continue

        close = float(datos_hoy.get("close", 0) or 0)
        dist_sl_pct = float(datos_hoy.get("dist_sl_10_pct", 0) or 0)

        if close <= 0 or dist_sl_pct <= 0:
            continue

        nuevo_sl = _swing_low_precio(close, dist_sl_pct)
        if nuevo_sl <= sl_actual:
            continue

        updates.append({
            "id":          pos["id"],
            "ticker":      ticker,
            "nuevo_sl":    nuevo_sl,
            "sl_anterior": sl_actual,
            "delta_pct":   round((nuevo_sl - sl_actual) / sl_actual * 100, 2),
        })

    return updates
