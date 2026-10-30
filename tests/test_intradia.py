"""
test_intradia.py -- normalizacion de barras de 30 min, agregados del corte,
cobertura y spool (src/utils/intradia.py, docs/intradia.md).

Datos sinteticos con la forma de yahooquery history(interval='30m'): MultiIndex
(symbol, date). La forma REAL del indice (aware vs naive) la verifica la
primera corrida --dry-run en la VM; aca se prueban las dos.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.utils import intradia as it
from src.utils import sesion_nyse as sn

FECHA = date(2026, 11, 2)                          # invierno: apertura 14:30 UTC
INICIO, _FIN = sn.sesion(FECHA)
H_APERTURA = sn.hora_corte(FECHA, sn.APERTURA_90)  # 16:00 UTC = 11:00 ET
H_CIERRE = sn.hora_corte(FECHA, sn.CIERRE_60)      # 20:00 UTC = 15:00 ET
CAPT = H_APERTURA + timedelta(minutes=5)


def _df(barras_por_ticker, aware=True):
    """barras_por_ticker = {ticker: [(hh, mm ET, close, vol), ...]} del 2/11/2026."""
    idx, filas = [], []
    for t, barras in barras_por_ticker.items():
        for hh, mm, close, vol in barras:
            ts = pd.Timestamp(2026, 11, 2, hh, mm)
            if aware:
                ts = ts.tz_localize("America/New_York")
            idx.append((t, ts))
            filas.append({"open": close - 1, "high": close + 1, "low": close - 2,
                          "close": close, "volume": vol})
    mi = pd.MultiIndex.from_tuples(idx, names=["symbol", "date"])
    return pd.DataFrame(filas, index=mi)


SESION_HASTA_11 = [(9, 30, 100.0, 1000), (10, 0, 101.0, 800), (10, 30, 102.0, 600)]


def _norm(df, tickers, hora=H_APERTURA, corte=sn.APERTURA_90):
    return it.normalizar_barras(df, tickers, FECHA, corte, INICIO, hora, CAPT)


def test_descarta_barra_en_curso_y_premarket():
    barras = [(9, 0, 99.0, 50)] + SESION_HASTA_11 + [(11, 0, 103.0, 10)]
    filas = _norm(_df({"AAPL": barras}), ["AAPL"])
    assert [f["inicio_barra"] for f in filas] == [
        datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc),
        datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 11, 2, 15, 30, tzinfo=timezone.utc),
    ]
    assert all(f["fecha"] == FECHA and f["corte"] == sn.APERTURA_90 for f in filas)
    assert all(f["hora_corte"] == H_APERTURA and f["capturado_en"] == CAPT for f in filas)


def test_indice_naive_se_interpreta_como_hora_de_ny():
    a = _norm(_df({"AAPL": SESION_HASTA_11}, aware=True), ["AAPL"])
    b = _norm(_df({"AAPL": SESION_HASTA_11}, aware=False), ["AAPL"])
    assert [f["inicio_barra"] for f in a] == [f["inicio_barra"] for f in b]


def test_indice_en_utc_tambien_sirve():
    df = _df({"AAPL": SESION_HASTA_11})
    df.index = df.index.set_levels(df.index.levels[1].tz_convert("UTC"), level=1)
    assert len(_norm(df, ["AAPL"])) == 3


def test_otra_rueda_no_entra():
    df = _df({"AAPL": SESION_HASTA_11})
    df.index = df.index.set_levels(df.index.levels[1] - pd.Timedelta(days=3), level=1)
    assert _norm(df, ["AAPL"]) == []


def test_barra_sin_close_se_descarta_y_volumen_nan_queda_none():
    df = _df({"AAPL": SESION_HASTA_11})
    df.iloc[1, df.columns.get_loc("close")] = float("nan")
    df.iloc[2, df.columns.get_loc("volume")] = float("nan")
    filas = _norm(df, ["AAPL"])
    assert len(filas) == 2
    assert filas[-1]["volumen"] is None


def test_indice_plano_de_un_solo_ticker():
    df = _df({"AAPL": SESION_HASTA_11}).xs("AAPL", level=0)
    assert len(_norm(df, ["AAPL"])) == 3


def test_entradas_vacias():
    assert _norm(pd.DataFrame(), ["AAPL"]) == []
    assert _norm(None, ["AAPL"]) == []
    with pytest.raises(ValueError):
        it.normalizar_barras(pd.DataFrame(), ["AAPL"], FECHA, "mediodia", INICIO, H_APERTURA, CAPT)


def test_barras_esperadas():
    assert it.barras_esperadas(INICIO, H_APERTURA) == 3
    assert it.barras_esperadas(INICIO, H_CIERRE) == 11
    media = date(2026, 11, 27)
    ini, _ = sn.sesion(media)
    assert it.barras_esperadas(ini, sn.hora_corte(media, sn.CIERRE_60)) == 5


def test_agregar_corte():
    filas = _norm(_df({"AAPL": SESION_HASTA_11}), ["AAPL"])
    a = it.agregar_corte(list(reversed(filas)))
    assert a["open"] == 99.0          # open de la primera barra
    assert a["close"] == 102.0        # close de la ultima
    assert a["high"] == 103.0 and a["low"] == 98.0
    assert a["volumen"] == 2400 and a["n_barras"] == 3
    assert it.agregar_corte([]) is None


def test_cobertura_exige_la_barra_que_termina_en_el_corte():
    df = _df({
        "AAPL": SESION_HASTA_11,
        "MSFT": SESION_HASTA_11[:2],      # le falta la de 10:30
    })
    filas = _norm(df, ["AAPL", "MSFT", "NVDA"])
    cob = it.cobertura(filas, ["AAPL", "MSFT", "NVDA"], H_APERTURA)
    assert cob["completos"] == ["AAPL"]
    assert cob["sin_ultima_barra"] == ["MSFT"]
    assert cob["sin_datos"] == ["NVDA"]
    assert cob["pct"] == pytest.approx(1 / 3)


def test_spool_ida_y_vuelta(tmp_path):
    filas = _norm(_df({"AAPL": SESION_HASTA_11, "MSFT": SESION_HASTA_11}), ["AAPL", "MSFT"])
    filas[0]["volumen"] = None
    p = it.escribir_spool(filas, FECHA, sn.APERTURA_90, dir_spool=str(tmp_path))
    assert it.listar_spools(str(tmp_path)) == [p]
    assert it.leer_spool(p) == filas
    # Un reintento con la DB caida no pisa el spool anterior.
    p2 = it.escribir_spool(filas, FECHA, sn.APERTURA_90, dir_spool=str(tmp_path))
    assert p2 != p and len(it.listar_spools(str(tmp_path))) == 2


def test_valores_db_en_el_orden_del_upsert():
    filas = _norm(_df({"AAPL": SESION_HASTA_11}), ["AAPL"])
    v = it.valores_db(filas)
    assert len(v[0]) == len(it.COLUMNAS)
    assert v[0][it.COLUMNAS.index("ticker")] == "AAPL"
    assert v[0][it.COLUMNAS.index("close")] == 100.0
    for col in it.COLUMNAS:
        assert col in it.SQL_UPSERT and col in it.DDL_TABLA
