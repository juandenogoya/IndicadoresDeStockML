"""
trading_calendar.py
Calendario de dias habiles NYSE (2025-2027).

Cubre: New Year, MLK, Presidents Day, Good Friday, Memorial Day,
       Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas.

Funciones publicas:
    is_trading_day(d)           -> bool
    holiday_name(d)             -> str | None
    prev_trading_day(d)         -> date
    next_trading_day(d)         -> date
    trading_days_between(a, b)  -> list[date]
    describe_date(d)            -> str   (para logs / bats)
"""

from datetime import date, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Feriados NYSE — mercado cerrado todo el dia
# ---------------------------------------------------------------------------

NYSE_HOLIDAYS: dict[date, str] = {

    # ── 2025 ─────────────────────────────────────────────────────────────
    date(2025, 1,  1):  "New Year's Day",
    date(2025, 1, 20):  "MLK Day",
    date(2025, 2, 17):  "Presidents Day",
    date(2025, 4, 18):  "Good Friday",
    date(2025, 5, 26):  "Memorial Day",
    date(2025, 6, 19):  "Juneteenth",
    date(2025, 7,  4):  "Independence Day",
    date(2025, 9,  1):  "Labor Day",
    date(2025, 11, 27): "Thanksgiving",
    date(2025, 12, 25): "Christmas",

    # ── 2026 ─────────────────────────────────────────────────────────────
    date(2026, 1,  1):  "New Year's Day",
    date(2026, 1, 19):  "MLK Day",
    date(2026, 2, 16):  "Presidents Day",
    date(2026, 4,  3):  "Good Friday",        # Easter: 5 Apr
    date(2026, 5, 25):  "Memorial Day",
    date(2026, 6, 19):  "Juneteenth",
    date(2026, 7,  3):  "Independence Day (observed)",  # 4-Jul cae Sabado
    date(2026, 9,  7):  "Labor Day",
    date(2026, 11, 26): "Thanksgiving",
    date(2026, 12, 25): "Christmas",

    # ── 2027 ─────────────────────────────────────────────────────────────
    date(2027, 1,  1):  "New Year's Day",
    date(2027, 1, 18):  "MLK Day",
    date(2027, 2, 15):  "Presidents Day",
    date(2027, 3, 26):  "Good Friday",        # Easter: 28 Mar
    date(2027, 5, 31):  "Memorial Day",
    date(2027, 6, 18):  "Juneteenth (observed)",   # 19-Jun cae Sabado
    date(2027, 7,  5):  "Independence Day (observed)",  # 4-Jul cae Domingo
    date(2027, 9,  6):  "Labor Day",
    date(2027, 11, 25): "Thanksgiving",
    date(2027, 12, 24): "Christmas (observed)",    # 25-Dic cae Sabado
}

_WEEKDAY_ES = ["Lunes", "Martes", "Miercoles", "Jueves",
               "Viernes", "Sabado", "Domingo"]


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def is_trading_day(d: date) -> bool:
    """True si NYSE opera ese dia (Lunes-Viernes, excluye feriados)."""
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def holiday_name(d: date) -> Optional[str]:
    """Nombre del feriado si la fecha es feriado NYSE, None si no."""
    return NYSE_HOLIDAYS.get(d)


def prev_trading_day(d: date) -> date:
    """Ultimo dia habil estrictamente anterior a d."""
    d = d - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def next_trading_day(d: date) -> date:
    """Primer dia habil estrictamente posterior a d."""
    d = d + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def trading_days_between(start: date, end: date) -> list:
    """Lista de dias habiles entre start y end (ambos inclusive)."""
    days = []
    d = start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def describe_date(d: date) -> str:
    """
    Descripcion legible de una fecha para logs y bats.
    Ejemplos:
        '2026-04-27 Lunes  [DIA HABIL]'
        '2026-04-25 Sabado [FIN DE SEMANA]'
        '2026-04-03 Viernes [FERIADO: Good Friday]'
    """
    dow = _WEEKDAY_ES[d.weekday()]
    if d.weekday() >= 5:
        razon = "FIN DE SEMANA"
    elif d in NYSE_HOLIDAYS:
        razon = "FERIADO: " + NYSE_HOLIDAYS[d]
    else:
        razon = "DIA HABIL"
    return f"{d}  {dow:<10}  [{razon}]"


# ---------------------------------------------------------------------------
# CLI rapido: python -m src.utils.trading_calendar [YYYY-MM-DD]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import date as _date

    if len(sys.argv) > 1:
        try:
            d = _date.fromisoformat(sys.argv[1])
        except ValueError:
            print(f"Formato invalido: {sys.argv[1]}  (usar YYYY-MM-DD)")
            sys.exit(1)
    else:
        d = _date.today()

    print()
    print(f"  Fecha consultada : {describe_date(d)}")
    print(f"  Dia habil previo : {prev_trading_day(d)}")
    print(f"  Dia habil sig.   : {next_trading_day(d)}")
    print()
