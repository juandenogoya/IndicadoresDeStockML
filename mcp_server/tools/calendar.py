"""
Tools de calendario bursatil NYSE.

Importa exclusivamente funciones puras de src.utils.trading_calendar.
Sin side effects, sin escritura a DB.

Requiere que PYTHONPATH apunte al repo root (configurado por el cliente
MCP o por CWD al correr desde el repo root en desarrollo).

Tools exportadas:
  check_trading_day(date: str) -> dict
  get_last_trading_day() -> dict
"""

import re
from datetime import date, datetime

from src.utils.trading_calendar import (
    holiday_name,
    is_trading_day,
    next_trading_day,
    prev_trading_day,
)

# Formato esperado para fechas de entrada
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Annotations comunes a todas las tools de este modulo
CALENDAR_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


def _parse_date(date_str: str) -> date:
    """
    Parsea un string YYYY-MM-DD a datetime.date.

    Lanza ValueError con mensaje claro si el formato es invalido o la
    fecha no existe en el calendario (ej. 2026-02-30).
    """
    if not _DATE_RE.match(date_str):
        raise ValueError(
            f"Formato de fecha invalido: '{date_str}'. "
            "Usar YYYY-MM-DD (ej. 2026-05-09)."
        )
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Fecha inexistente en el calendario: '{date_str}'. "
            "Verificar dia, mes y anio."
        )


def check_trading_day(date: str) -> dict:
    """
    Verifica si una fecha es dia habil bursatil NYSE.

    IMPORTANTE: llamar esta tool antes de razonar sobre cualquier
    ventana temporal. Nunca asumir si una fecha es habil.

    Args:
        date: Fecha a verificar en formato YYYY-MM-DD. Ej: "2026-05-09".

    Returns:
        dict con los siguientes campos:
          - date (str): La fecha consultada en formato YYYY-MM-DD.
          - is_trading_day (bool): True si NYSE opera ese dia.
          - reason (str | None): Razon si NO es habil:
              "weekend"          -> sabado o domingo
              "<nombre feriado>" -> feriado NYSE (ej. "Christmas")
              None               -> es dia habil (is_trading_day=True)
          - next_trading_day (str | None): Proximo dia habil en formato
              YYYY-MM-DD. None si la fecha consultada ya es habil.

    Ejemplos de retorno:

      Dia habil (viernes):
        {"date": "2026-05-08", "is_trading_day": True,
         "reason": None, "next_trading_day": None}

      Sabado:
        {"date": "2026-05-09", "is_trading_day": False,
         "reason": "weekend", "next_trading_day": "2026-05-11"}

      Feriado:
        {"date": "2026-12-25", "is_trading_day": False,
         "reason": "Christmas", "next_trading_day": "2026-12-28"}

    Raises:
        ValueError: Si el formato no es YYYY-MM-DD o la fecha es
            inexistente en el calendario (ej. 2026-02-30).
    """
    d = _parse_date(date)

    if is_trading_day(d):
        return {
            "date": date,
            "is_trading_day": True,
            "reason": None,
            "next_trading_day": None,
        }

    # Determinar razon de no habilidad
    if d.weekday() >= 5:
        reason = "weekend"
    else:
        # Es dia de semana pero feriado
        reason = holiday_name(d)  # str con el nombre del feriado

    next_td = next_trading_day(d)

    return {
        "date": date,
        "is_trading_day": False,
        "reason": reason,
        "next_trading_day": next_td.isoformat(),
    }


def get_last_trading_day() -> dict:
    """
    Retorna el ultimo dia habil NYSE hasta hoy (inclusive).

    Si hoy es dia habil, retorna hoy. Si hoy es fin de semana o
    feriado, retorna el viernes anterior (o el dia habil previo).

    Usar antes de construir cualquier ventana temporal: el "ultimo
    dia con datos" siempre es el ultimo dia habil, no necesariamente
    ayer ni el dia de hoy.

    Returns:
        dict con los siguientes campos:
          - last_trading_day (str): Ultimo dia habil en formato YYYY-MM-DD.
          - today (str): Fecha de hoy en formato YYYY-MM-DD.
          - today_is_trading_day (bool): True si hoy es dia habil.

    Ejemplo de retorno (consultado un domingo):
        {"last_trading_day": "2026-05-08",
         "today": "2026-05-10",
         "today_is_trading_day": False}

    Ejemplo de retorno (consultado un martes habil):
        {"last_trading_day": "2026-05-12",
         "today": "2026-05-12",
         "today_is_trading_day": True}
    """
    today = date.today()
    today_is_td = is_trading_day(today)

    if today_is_td:
        last_td = today
    else:
        last_td = prev_trading_day(today)

    return {
        "last_trading_day": last_td.isoformat(),
        "today": today.isoformat(),
        "today_is_trading_day": today_is_td,
    }
