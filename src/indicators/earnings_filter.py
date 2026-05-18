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

Fuente de datos:
    obtener_earnings_map() lee de la tabla earnings_calendar via get_engine().
    Esa tabla la pobla semanalmente scripts/refresh_earnings_calendar.py
    (unico componente que llama a yfinance para earnings).

    earnings_filter NO llama a yfinance en linea: hacerlo en cada corrida de
    cada bot generaba ~1.500+ llamadas/dia y rate limit por IP.

Casos de borde manejados automaticamente:
    - Earnings lunes  -> cerrar viernes
    - Earnings post-feriado -> cerrar ultimo dia habil antes del feriado
    - earnings_date NULL / ticker ausente -> omitido del map (fail-safe:
      la estrategia no aplica filtro de earnings a ese ticker)
    - earnings_calendar vacia / inaccesible -> map vacio + WARNING: los bots
      corren igual, sin filtro de earnings, no se cortan.
"""

import os
import yfinance as yf
from datetime import date, datetime, timedelta
from functools import lru_cache


# ── Parametros configurables ─────────────────────────────────
DIAS_BUFFER_ENTRADA = int(os.getenv("BOT_EARNINGS_BUFFER_DIAS", "1"))
# 1 = bloquear si earnings es manana o hoy
# 2 = bloquear si earnings es en 2 dias habiles o menos

DIAS_STALE_WARNING = 10
# Si earnings_calendar no se refresca hace mas de N dias -> WARNING (se usa igual)


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


# ── Datos de earnings ────────────────────────────────────────

def _obtener_earnings_fecha(ticker: str) -> date | None:
    """
    Consulta yfinance para la proxima fecha de earnings de un ticker.
    Retorna None si no hay dato o falla la consulta (fail-safe).

    NOTA: la usa scripts/refresh_earnings_calendar.py para poblar la tabla
    earnings_calendar. Los bots NO la llaman directamente: consumen la tabla
    via obtener_earnings_map().
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
    Retorna {ticker: earnings_date} leyendo de la tabla earnings_calendar.

    Lee de la DB a la que apunte get_engine() (local para bots FT, Railway
    para bots Alpaca). NUNCA llama a yfinance.

    Comportamiento fail-safe:
        - Ticker con earnings_date NULL o ausente de la tabla -> omitido
          del map. La estrategia no le aplica filtro de earnings.
        - Tabla vacia, inexistente o inaccesible -> map vacio + WARNING.
          Los bots corren igual, sin filtro de earnings (no se cortan).
        - Tabla con datos viejos (> DIAS_STALE_WARNING dias) -> se usa
          igual + WARNING para que se refresque.
    """
    if not tickers:
        return {}

    from sqlalchemy import text
    from src.data.database import get_engine

    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT ticker, earnings_date, fecha_actualizacion
                FROM earnings_calendar
                WHERE ticker = ANY(:tickers)
                  AND earnings_date IS NOT NULL
            """), {"tickers": list(tickers)}).fetchall()
    except Exception as e:
        print(f"  [earnings_filter] WARNING: no se pudo leer earnings_calendar "
              f"({type(e).__name__}). Filtro de earnings INACTIVO esta corrida.")
        return {}

    if not rows:
        print("  [earnings_filter] WARNING: earnings_calendar sin fechas para "
              "estos tickers. Filtro de earnings INACTIVO esta corrida.")
        return {}

    # Aviso de staleness (la data se usa igual)
    actualizaciones = [r.fecha_actualizacion for r in rows if r.fecha_actualizacion]
    if actualizaciones:
        dias = (datetime.now() - max(actualizaciones)).days
        if dias > DIAS_STALE_WARNING:
            print(f"  [earnings_filter] WARNING: earnings_calendar desactualizada "
                  f"({dias} dias). Conviene correr refresh_earnings_calendar.py.")

    return {r.ticker: r.earnings_date for r in rows}


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
