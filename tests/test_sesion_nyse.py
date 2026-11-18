"""
test_sesion_nyse.py -- horario de la rueda y cortes intradiarios
(src/utils/sesion_nyse.py, docs/intradia.md).

Casos anclados en el cambio de hora de EE.UU. del 1/11/2026 (Argentina no
cambia) y en las medias ruedas del comunicado oficial de NYSE.
"""

import os
import re
from datetime import date, datetime, timedelta, timezone

import pytest

from src.utils import sesion_nyse as sn
from src.utils.trading_calendar import is_trading_day, trading_days_between

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_cortes_con_horario_de_verano():
    # Viernes 30/10/2026: NY en UTC-4.
    c = sn.cortes(date(2026, 10, 30))
    assert c[sn.APERTURA_90] == utc(2026, 10, 30, 15)
    assert c[sn.CIERRE_60] == utc(2026, 10, 30, 19)


def test_cortes_en_invierno():
    # Lunes 2/11/2026, primer dia habil despues del cambio de hora: UTC-5.
    c = sn.cortes(date(2026, 11, 2))
    assert c[sn.APERTURA_90] == utc(2026, 11, 2, 16)
    assert c[sn.CIERRE_60] == utc(2026, 11, 2, 20)


def test_media_rueda_corta_el_cierre_a_las_12_et():
    c = sn.cortes(date(2026, 11, 27))
    assert c[sn.APERTURA_90] == utc(2026, 11, 27, 16)   # 11:00 ET, igual que siempre
    assert c[sn.CIERRE_60] == utc(2026, 11, 27, 17)     # 12:00 ET = 13:00 - 60 min
    inicio, fin = sn.sesion(date(2026, 11, 27))
    assert fin == utc(2026, 11, 27, 18)


def test_sin_rueda_no_hay_cortes():
    assert sn.cortes(date(2026, 11, 26)) == {}   # Thanksgiving
    assert sn.cortes(date(2026, 10, 31)) == {}   # sabado
    assert sn.sesion(date(2026, 11, 26)) is None


def test_fuera_de_cobertura_no_inventa_ruedas():
    # 2028 no esta cargado en trading_calendar: un feriado de 2028 pasaria por
    # dia habil. Mejor ningun corte que uno falso.
    assert not sn.con_cobertura(date(2028, 1, 3))
    assert sn.cortes(date(2028, 1, 3)) == {}
    assert sn.con_cobertura(date(2027, 12, 31))


def test_medias_ruedas_son_dias_habiles():
    for d in sn.CIERRES_ANTICIPADOS:
        assert is_trading_day(d), d
        assert sn.con_cobertura(d)


def test_hora_corte_rechaza_corte_desconocido():
    with pytest.raises(ValueError):
        sn.hora_corte(date(2026, 11, 2), "mediodia")


def test_fecha_ny_exige_aware_y_usa_la_fecha_de_ny():
    # 01:00 UTC del 3/11 son las 20:00 del 2/11 en NY.
    assert sn.fecha_ny(utc(2026, 11, 3, 1)) == date(2026, 11, 2)
    with pytest.raises(ValueError):
        sn.fecha_ny(datetime(2026, 11, 3, 1))


@pytest.mark.parametrize("ahora, esperado", [
    (utc(2026, 11, 2, 16, 5), [sn.APERTURA_90]),
    (utc(2026, 11, 2, 16, 20), [sn.APERTURA_90]),   # reintento
    (utc(2026, 11, 2, 16, 1), []),                  # la barra 10:30-11:00 recien cierra
    (utc(2026, 11, 2, 17, 5), []),                  # fuera de la ventana
    (utc(2026, 11, 2, 15, 5), []),                  # horario de verano: no aplica en invierno
    (utc(2026, 11, 2, 20, 5), [sn.CIERRE_60]),
    (utc(2026, 10, 30, 19, 20), [sn.CIERRE_60]),
    (utc(2026, 10, 30, 16, 5), []),                 # 65 min despues de la apertura de verano
    (utc(2026, 11, 27, 17, 5), [sn.CIERRE_60]),     # media rueda
    (utc(2026, 11, 26, 16, 5), []),                 # feriado
])
def test_cortes_vencidos(ahora, esperado):
    assert [c for _, c, _ in sn.cortes_vencidos(ahora)] == esperado


def _disparos(d):
    """Instantes UTC en que dispara el cron un dia dado."""
    return [datetime(d.year, d.month, d.day, h, m, tzinfo=timezone.utc)
            for h in sn.CRON_HORAS_UTC for m in sn.CRON_MINUTOS]


def test_el_cron_cubre_cada_corte_2026_2027_con_intento_y_reintento():
    """
    Cada corte de cada rueda cubierta cae exactamente en 2 disparos del cron
    (intento + reintento) y ningun disparo ve dos cortes a la vez. Si se cambia
    la ventana, la demora o el horario del cron sin pensar en el otro, falla aca.
    """
    for d in trading_days_between(date(2026, 1, 1), date(2027, 12, 31)):
        vistos = {}
        for t in _disparos(d):
            v = sn.cortes_vencidos(t)
            assert len(v) <= 1, (d, t, v)
            for _, corte, _ in v:
                vistos[corte] = vistos.get(corte, 0) + 1
        assert vistos == {sn.APERTURA_90: 2, sn.CIERRE_60: 2}, (d, vistos)


def test_el_crontab_coincide_con_las_constantes():
    ruta = os.path.join(ROOT, "scripts", "intradia", "oracle_crontab_intradia.txt")
    lineas = [l for l in open(ruta, encoding="utf-8") if re.match(r"^\d", l)]
    assert len(lineas) == 1
    minutos, horas, dom, mes, dow = lineas[0].split()[:5]
    assert tuple(int(x) for x in minutos.split(",")) == sn.CRON_MINUTOS
    assert tuple(int(x) for x in horas.split(",")) == sn.CRON_HORAS_UTC
    assert (dom, mes, dow) == ("*", "*", "1-5")
    assert "snapshot_intradia.py" in lineas[0]
    assert "origin main-y9tr06" in lineas[0]
