"""
earnings_filter.py
Filtro de earnings transversal a todas las estrategias.

Principio: la volatilidad de un balance es impredecible (sube, baja o neutro).
           El objetivo es maximizar ingresos con control de riesgo.
           Se preserva el capital cerrando ANTES del evento, sin importar
           si la posicion esta en ganancia o perdida.

Reglas:
    CIERRE   : si hoy == dia_habil_anterior(earnings_date) -> cerrar posicion
               Prioridad absoluta sobre cualquier logica de estrategia.

    BLOQUEO  : si earnings_date <= hoy + DIAS_BUFFER_ENTRADA dias habiles
               -> no abrir posicion nueva en ese ticker.

Casos de borde manejados automaticamente:
    - Earnings lunes  -> cerrar viernes
    - Earnings post-feriado -> cerrar ultimo dia habil antes del feriado
    - Error yfinance (sin datos) -> no cerrar (fail-safe, no falso positivo)
"""

import os
import yfinance as yf
from datetime import date, timedelta
from functools import lru_cache


# ── Parametros configurables ─────────────────────────────────
DIAS_BUFFER_ENTRADA = int(os.getenv("BOT_EARNINGS_BUFFER_DIAS", "1"))
# 1 = bloquear si earnings es manana o hoy
# 2 = bloquear si earnings es en 2 dias habiles o menos


# ── Calendario NYSE ──────────────────────────────────────────

def _observado(d: date) -> date:
    """Ajusta un feriado que cae en fin de semana al dia habil mas cercano."""
    if d.weekday() == 5:   # sabado -> viernes
        return d - timedelta(days=1)
    if d.weekday() == 6:   # domingo -> lunes
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Retorna el n-esimo dia de la semana (0=lunes) de un mes/anio."""
    first  = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset) + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Retorna el ultimo dia de la semana (0=lunes) de un mes/anio."""
    last   = date(year, month + 1, 1) - timedelta(days=1) if month < 12 \
             else date(year + 1, 1, 1) - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _easter(year: int) -> date:
    """Algoritmo de Butcher para calcular la fecha de Pascua."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=10)
def get_nyse_holidays(year: int) -> frozenset:
    """Retorna el conjunto de feriados NYSE para un anio dado."""
    h = set()
    h.add(_observado(date(year, 1, 1)))          # New Year's Day
    h.add(_nth_weekday(year, 1, 0, 3))           # MLK Day (3er lunes enero)
    h.add(_nth_weekday(year, 2, 0, 3))           # Presidents Day (3er lunes febrero)
    h.add(_easter(year) - timedelta(days=2))     # Good Friday
    h.add(_last_weekday(year, 5, 0))             # Memorial Day (ultimo lunes mayo)
    h.add(_observado(date(year, 6, 19)))         # Juneteenth
    h.add(_observado(date(year, 7, 4)))          # Independence Day
    h.add(_nth_weekday(year, 9, 0, 1))           # Labor Day (1er lunes septiembre)
    h.add(_nth_weekday(year, 11, 3, 4))          # Thanksgiving (4to jueves noviembre)
    h.add(_observado(date(year, 12, 25)))        # Christmas
    return frozenset(h)


def es_dia_habil(d: date) -> bool:
    """True si d es dia habil NYSE (no fin de semana, no feriado)."""
    return d.weekday() < 5 and d not in get_nyse_holidays(d.year)


def dia_habil_anterior(d: date) -> date:
    """Dia habil NYSE inmediato anterior a d (sin incluir d)."""
    candidato = d - timedelta(days=1)
    while not es_dia_habil(candidato):
        candidato -= timedelta(days=1)
    return candidato


def dias_habiles_hasta(desde: date, hasta: date) -> int:
    """Cantidad de dias habiles NYSE entre desde (exclusive) y hasta (inclusive)."""
    count = 0
    d = desde
    while d < hasta:
        d += timedelta(days=1)
        if es_dia_habil(d):
            count += 1
    return count


# ── Datos de earnings (yfinance) ─────────────────────────────

def _obtener_earnings_fecha(ticker: str) -> date | None:
    """
    Consulta yfinance para la proxima fecha de earnings de un ticker.
    Retorna None si no hay dato o falla la consulta (fail-safe).
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if not cal:
            return None
        earnings_dates = cal.get("Earnings Date", [])
        if not earnings_dates:
            return None
        first = earnings_dates[0]
        if hasattr(first, "date"):
            return first.date()
        if hasattr(first, "to_pydatetime"):
            return first.to_pydatetime().date()
        return date.fromisoformat(str(first)[:10])
    except Exception:
        return None


def obtener_earnings_map(tickers: list[str]) -> dict[str, date]:
    """
    Retorna {ticker: earnings_date} para los tickers que tienen
    fecha de earnings proxima disponible en yfinance.
    Tickers sin dato son omitidos (comportamiento fail-safe).
    """
    resultado = {}
    for ticker in tickers:
        fecha = _obtener_earnings_fecha(ticker)
        if fecha:
            resultado[ticker] = fecha
    return resultado


# ── Logica de filtrado ────────────────────────────────────────

def tickers_a_cerrar_hoy(
    tickers: list[str],
    fecha_hoy: date = None,
) -> dict[str, date]:
    """
    De la lista de tickers (posiciones abiertas), retorna los que deben
    cerrarse HOY porque el dia siguiente habil es el dia de earnings.

    Retorna {ticker: earnings_date}.
    Si yfinance no tiene dato para un ticker -> NO se incluye (fail-safe).

    Ejemplo:
        earnings NFLX = lunes 20/4  -> dia_habil_anterior = viernes 17/4
        Si hoy == 17/4 -> NFLX debe cerrarse hoy.
    """
    hoy          = fecha_hoy or date.today()
    earnings_map = obtener_earnings_map(tickers)
    a_cerrar     = {}

    for ticker, earnings_date in earnings_map.items():
        dia_cierre = dia_habil_anterior(earnings_date)
        if hoy >= dia_cierre:
            a_cerrar[ticker] = earnings_date

    return a_cerrar


def tickers_a_bloquear_entrada(
    tickers: list[str],
    fecha_hoy: date = None,
    dias_buffer: int = None,
) -> dict[str, date]:
    """
    De la lista de candidatos a entrar, retorna los que deben bloquearse
    porque tienen earnings demasiado proximos.

    dias_buffer (default: DIAS_BUFFER_ENTRADA del .env):
        1 -> bloquear si earnings es manana o hoy
        2 -> bloquear si earnings es en <= 2 dias habiles

    Retorna {ticker: earnings_date}.
    """
    hoy    = fecha_hoy or date.today()
    buffer = dias_buffer if dias_buffer is not None else DIAS_BUFFER_ENTRADA

    earnings_map = obtener_earnings_map(tickers)
    bloqueados   = {}

    for ticker, earnings_date in earnings_map.items():
        habiles_restantes = dias_habiles_hasta(hoy, earnings_date)
        if habiles_restantes <= buffer:
            bloqueados[ticker] = earnings_date

    return bloqueados
