"""
Tests del bloque "Setups de Vela" del Telegram del scanner (telegram_notifier).

Desde el 16/10/2026 los patrones salen de features_velas (definicion clasica,
src/indicators/velas.py) por ticker + precio_fecha, y NO de los patron_* de
features_pa (precio_accion.py: la envolvente solo compara tamano de cuerpo).

Caso real que lo motivo: TSM el 15/10/2026 salio como "Envolvente Alc." con un
cuerpo 415,17-422,25 que no envuelve al previo 430,00-436,25.
"""

from datetime import date, datetime

import pandas as pd

from src.indicators import velas
from src.pipeline import telegram_notifier as tn

# OHLCV de TSM en precios_diarios (coincide con TradingView).
TSM = pd.DataFrame({
    "fecha": pd.to_datetime(["2026-10-13", "2026-10-14", "2026-10-15"]),
    "open":  [439.515, 436.25, 415.17],
    "high":  [445.495, 440.39, 423.66],
    "low":   [437.40, 428.06, 407.13],
    "close": [445.12, 430.00, 422.25],
})


def _resultado_tsm(**extra):
    r = {
        "ticker": "TSM",
        "scan_fecha": datetime(2026, 10, 16, 21, 0, 0),
        "precio_fecha": date(2026, 10, 15),
        "alert_nivel": "NEUTRAL",
        "alert_score": 50,
        "vol_spike": 0,
        # Lo que trae features_pa (precio_accion.py) para esa rueda: el falso
        # positivo. El formatter NO lo tiene que leer.
        "patron_engulfing_bull": 1,
        "patron_hammer": 0,
        "patron_shooting_star": 0,
        "patron_engulfing_bear": 0,
    }
    r.update(extra)
    return r


def _fila_velas(df, fecha):
    out = velas.calcular_velas(df)
    fila = out[out["fecha"] == pd.Timestamp(fecha)].iloc[0]
    return {p: int(fila[p]) for p in tn._PATRONES_TELEGRAM}


def test_tsm_15_10_no_es_envolvente_clasica():
    fila = _fila_velas(TSM, "2026-10-15")
    assert fila["patron_engulfing_bull"] == 0
    assert all(v == 0 for v in fila.values())


def test_tsm_15_10_no_sale_en_setups_de_vela():
    velas_map = {("TSM", date(2026, 10, 15)): _fila_velas(TSM, "2026-10-15")}
    texto = tn.formatear_mensaje2_velas([_resultado_tsm()], velas_map)
    assert "Envolvente" not in texto
    assert texto == ""


def test_ignora_los_patrones_viejos_del_resultado():
    # Sin fila de features_velas no hay patron, aunque features_pa diga 1.
    assert tn.formatear_mensaje2_velas([_resultado_tsm()], {}) == ""
    assert tn.formatear_mensaje2_velas([_resultado_tsm()]) == ""


def test_la_clave_es_precio_fecha_no_scan_fecha():
    # Fila marcada en la rueda del scan (16/10), no en la de los datos (15/10).
    marcada = {p: 0 for p in tn._PATRONES_TELEGRAM}
    marcada["patron_engulfing_bull"] = 1
    por_scan = {("TSM", date(2026, 10, 16)): marcada}
    assert tn.formatear_mensaje2_velas([_resultado_tsm()], por_scan) == ""

    por_datos = {("TSM", date(2026, 10, 15)): marcada}
    texto = tn.formatear_mensaje2_velas([_resultado_tsm()], por_datos)
    assert "<b>TSM</b>  Envolvente Alc." in texto


def test_envolvente_que_envuelve_si_sale():
    # Previa roja 430-436,25; actual abre debajo del cierre previo y cierra
    # arriba de la apertura previa.
    df = TSM.copy()
    df.loc[2, ["open", "high", "low", "close"]] = [428.0, 440.0, 427.0, 438.0]
    fila = _fila_velas(df, "2026-10-15")
    assert fila["patron_engulfing_bull"] == 1
    texto = tn.formatear_mensaje2_velas(
        [_resultado_tsm(vol_spike=1)], {("TSM", date(2026, 10, 15)): fila})
    assert "Envolvente Alc." in texto
    assert "\U0001F514" in texto


def test_precio_fecha_como_timestamp():
    marcada = {p: 0 for p in tn._PATRONES_TELEGRAM}
    marcada["patron_shooting_star"] = 1
    r = _resultado_tsm(precio_fecha=pd.Timestamp("2026-10-15"))
    texto = tn.formatear_mensaje2_velas([r], {("TSM", date(2026, 10, 15)): marcada})
    assert "Shooting Star" in texto
