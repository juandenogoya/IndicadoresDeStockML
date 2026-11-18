"""
sesion_nyse.py -- horario de la rueda NYSE y cortes intradiarios.

Modulo PURO (stdlib + zoneinfo): no importa DB, config ni pipeline. Usa
src/utils/trading_calendar.py para los feriados, SIN modificarlo.

POR QUE EXISTE. trading_calendar sabe que dias opera NYSE, pero no a que hora
abre y cierra ni cuales ruedas cierran a las 13:00 ET. El snapshot intradiario
(docs/intradia.md) corta en momentos RELATIVOS a la sesion, no en una hora fija
de pared: Argentina no cambia de hora y NY si, asi que "12:00 ARG" es 1,5 h
despues de la apertura de marzo a noviembre y 30 min en invierno.

CORTES (decididos 30/10/2026):
    apertura_90 = apertura + 90 min  -> 11:00 ET
    cierre_60   = cierre   - 60 min  -> 15:00 ET (12:00 ET en media rueda)

    | periodo                     | apertura_90        | cierre_60          |
    |-----------------------------|--------------------|--------------------|
    | horario de verano EE.UU.    | 15:00 UTC / 12 ARG | 19:00 UTC / 16 ARG |
    | invierno                    | 16:00 UTC / 13 ARG | 20:00 UTC / 17 ARG |
    | media rueda (invierno)      | 16:00 UTC / 13 ARG | 17:00 UTC / 14 ARG |

DISPARO. El cron no puede expresar "relativo a la sesion", asi que dispara en
TODOS los horarios UTC posibles (CRON_MINUTOS x CRON_HORAS_UTC) y el script
pregunta `cortes_vencidos(ahora)`: un corte esta vencido entre DEMORA_MIN y
VENTANA_MIN minutos despues de su hora. La DEMORA deja cerrar la barra de 30 min
que termina en el corte (sin ella se leeria la barra EN CURSO). La VENTANA
alcanza para los dos disparos de la hora (:05 y :20 = intento + reintento) y no
para el de la hora siguiente, que pertenece al otro horario (verano/invierno).
tests/test_sesion_nyse.py verifica que esta combinacion cubre cada corte de
2026-2027 exactamente con 2 disparos y que el crontab dice lo mismo.

COBERTURA. El calendario de feriados llega hasta COBERTURA_HASTA. Mas alla,
trading_calendar trataria un feriado como dia habil sin avisar -- un guard que
nunca falla y devuelve un numero plausible (patron de scan_fecha / features_
sector en CLAUDE.md). Por eso `cortes()` devuelve {} fuera de cobertura y
`con_cobertura()` permite avisarlo.

Funciones publicas:
    con_cobertura(d)              -> bool
    sesion(d)                     -> (inicio_utc, fin_utc) | None
    cortes(d)                     -> {corte: hora_utc}  ({} si no hay rueda)
    hora_corte(d, corte)          -> datetime utc | None
    cortes_vencidos(ahora_utc)    -> [(fecha, corte, hora_utc)]
    fecha_ny(instante_utc)        -> date (rueda de NY de ese instante)
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from src.utils.trading_calendar import NYSE_HOLIDAYS, is_trading_day

TZ_NY = ZoneInfo("America/New_York")

APERTURA = time(9, 30)
CIERRE = time(16, 0)
CIERRE_ANTICIPADO = time(13, 0)

# Medias ruedas NYSE (cierre 13:00 ET). Fuente: comunicado oficial "NYSE Group
# Announces 2026, 2027 and 2028 Holiday and Early Closings Calendar" (23/12/2025)
# y el de 2025-2027 (8/11/2024). Solo se cargan las fechas CONFIRMADAS contra el
# comunicado: un calendario de terceros lista medias ruedas que NYSE no tiene
# (18/6, 4/9 y 25/11/2026). Las de 2025 anteriores a diciembre no se cargaron:
# no se pudieron verificar y quedan fuera de la ventana de reconstruccion de
# Yahoo (~60 dias de barras de 30 min), asi que no las usa nadie.
# 2027: no hay media rueda de Nochebuena (el 24/12/2027 es feriado).
CIERRES_ANTICIPADOS: dict[date, str] = {
    date(2025, 12, 24): "Christmas Eve",
    date(2026, 11, 27): "Day after Thanksgiving",
    date(2026, 12, 24): "Christmas Eve",
    date(2027, 11, 26): "Day after Thanksgiving",
}

# Ultimo dia que cubre el calendario de feriados (fin del ultimo anio cargado).
COBERTURA_HASTA = date(max(NYSE_HOLIDAYS).year, 12, 31)

APERTURA_90 = "apertura_90"
CIERRE_60 = "cierre_60"
CORTES = (APERTURA_90, CIERRE_60)
MIN_DESDE_APERTURA = 90
MIN_ANTES_CIERRE = 60

BARRA_MIN = 30          # tamano de la barra intradiaria que se captura
DEMORA_MIN = 3          # minutos despues del corte antes de leer (cierra la barra)
VENTANA_MIN = 45        # hasta cuando, despues del corte, se lo sigue intentando

# Horarios del cron en Oracle (UTC, lunes a viernes). Deben coincidir con
# scripts/intradia/oracle_crontab_intradia.txt: lo verifica un test.
CRON_MINUTOS = (5, 20)
CRON_HORAS_UTC = (15, 16, 17, 19, 20)


def con_cobertura(d: date) -> bool:
    """True si el calendario de feriados cubre la fecha."""
    return d <= COBERTURA_HASTA


def _a_utc(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=TZ_NY).astimezone(timezone.utc)


def sesion(d: date) -> Optional[tuple[datetime, datetime]]:
    """(inicio, fin) de la rueda regular en UTC, o None si no hay rueda."""
    if not con_cobertura(d) or not is_trading_day(d):
        return None
    fin = CIERRE_ANTICIPADO if d in CIERRES_ANTICIPADOS else CIERRE
    return _a_utc(d, APERTURA), _a_utc(d, fin)


def cortes(d: date) -> dict[str, datetime]:
    """Hora UTC de cada corte de la rueda d. {} si no hay rueda o sin cobertura."""
    s = sesion(d)
    if s is None:
        return {}
    inicio, fin = s
    return {
        APERTURA_90: inicio + timedelta(minutes=MIN_DESDE_APERTURA),
        CIERRE_60: fin - timedelta(minutes=MIN_ANTES_CIERRE),
    }


def hora_corte(d: date, corte: str) -> Optional[datetime]:
    """Hora UTC de un corte puntual, o None si no hay rueda ese dia."""
    if corte not in CORTES:
        raise ValueError(f"corte desconocido: {corte!r} (validos: {CORTES})")
    return cortes(d).get(corte)


def fecha_ny(instante: datetime) -> date:
    """Fecha calendario de NY de un instante (aware). La rueda se nombra asi."""
    if instante.tzinfo is None:
        raise ValueError("instante sin zona horaria: pasar un datetime aware")
    return instante.astimezone(TZ_NY).date()


def cortes_vencidos(ahora: datetime) -> list[tuple[date, str, datetime]]:
    """
    Cortes que corresponde capturar en `ahora`: su hora quedo entre DEMORA_MIN y
    VENTANA_MIN minutos atras. Normalmente 0 o 1. El llamador decide si ya estan
    capturados (el reintento de los :20 encuentra el de los :05 y no hace nada).
    """
    d = fecha_ny(ahora)
    vencidos = []
    for corte, hora in cortes(d).items():
        transcurrido = (ahora - hora).total_seconds() / 60.0
        if DEMORA_MIN <= transcurrido <= VENTANA_MIN:
            vencidos.append((d, corte, hora))
    return vencidos
