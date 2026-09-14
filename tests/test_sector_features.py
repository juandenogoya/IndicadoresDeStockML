"""
Tests del modo incremental de src/indicators/sector_features.py (13/9/2026).

El Paso 2 diario recalcula solo las ultimas ruedas. La garantia que importa es
que esas filas sean IDENTICAS a las de la corrida completa: si el incremental
recortara la historia ANTES de calcular, retorno_5d (y su z-score) saldria NaN
o distinto en las primeras ruedas del recorte, y nada fallaria.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.indicators import sector_features as sf


def _datos(n_fechas=15):
    """Formato de cargar_datos_completos: 2 sectores, 7 tickers, fecha datetime."""
    rng = np.random.default_rng(7)
    fechas = [date(2026, 8, 3) + timedelta(days=i) for i in range(n_fechas)]
    filas = []
    for sector, tickers in (("Technology", ["AAA", "BBB", "CCC", "DDD"]),
                            ("Energy", ["EEE", "FFF", "GGG"])):
        for t in tickers:
            close = 100.0
            for f in fechas:
                close *= 1 + rng.normal(0, 0.02)
                score = float(rng.choice([0.3, 0.65, 0.8]))
                filas.append({
                    "ticker": t, "nombre": t, "sector": sector, "fecha": f,
                    "close": close, "rsi14": rng.uniform(20, 80),
                    "macd_hist": rng.normal(), "dist_sma50": rng.normal(0, 5),
                    "adx": rng.uniform(10, 40), "vol_relativo": rng.uniform(0.5, 2),
                    "score_ponderado": score,
                    "senal": "LONG" if score >= 0.6 else "NEUTRAL",
                })
    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df, fechas


@pytest.fixture
def datos(monkeypatch):
    df, fechas = _datos()
    monkeypatch.setattr(sf, "cargar_datos_completos", lambda: df.copy())
    escritos = []
    monkeypatch.setattr(sf, "upsert_features_sector", lambda d: escritos.append(d.copy()))
    return fechas, escritos


def test_incremental_igual_a_completo_en_las_ruedas_recalculadas(datos):
    fechas, _ = datos
    completo = sf.procesar_features_sector(guardar_db=False, verbose=False)
    desde = fechas[10]
    inc = sf.procesar_features_sector(guardar_db=False, desde=desde, verbose=False)

    esperado = completo[completo["fecha"] >= desde].reset_index(drop=True)
    assert sorted(set(inc["fecha"])) == fechas[10:]
    pd.testing.assert_frame_equal(inc, esperado)


def test_la_primera_rueda_del_recorte_usa_la_historia_previa(datos):
    fechas, _ = datos
    inc = sf.procesar_features_sector(guardar_db=False, desde=fechas[10], verbose=False)
    primera = inc[inc["fecha"] == fechas[10]]
    assert len(primera) == 7
    assert primera["z_retorno_5d_sector"].notna().all()


def test_persiste_solo_desde(datos):
    fechas, escritos = datos
    sf.procesar_features_sector(guardar_db=True, desde=fechas[12], verbose=False)
    assert len(escritos) == 1
    assert min(escritos[0]["fecha"]) == fechas[12]


def test_desde_acepta_texto(datos):
    fechas, _ = datos
    a = sf.procesar_features_sector(guardar_db=False, desde=fechas[12], verbose=False)
    b = sf.procesar_features_sector(guardar_db=False, desde=str(fechas[12]), verbose=False)
    pd.testing.assert_frame_equal(a, b)


def test_desde_posterior_a_los_datos_no_escribe(datos):
    fechas, escritos = datos
    out = sf.procesar_features_sector(guardar_db=True, desde=fechas[-1] + timedelta(days=5),
                                      verbose=False)
    assert out.empty
    assert escritos == []
