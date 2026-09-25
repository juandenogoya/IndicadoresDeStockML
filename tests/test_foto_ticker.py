"""Tests de src/utils/foto_ticker.py (foto al cierre para Telegram)."""

from datetime import date, timedelta

from src.utils import foto_ticker as ft
from src.utils.trading_calendar import trading_days_between


# --- familias y seleccion -------------------------------------------------------

def test_familia_sectorial_no_cae_en_tecnico():
    assert ft.familia_de("tecnico") == "TECH"
    assert ft.familia_de("tecnico_sectorial_options_v2") == "TECH_SECTOR"
    assert ft.familia_de("tecnico_sectorial_oiexit_v1") == "TECH_SECTOR"
    assert ft.familia_de("ml_scanner") == "ML"
    assert ft.familia_de("smc_estructura_confirmada") == "SMC"
    assert ft.familia_de(None) is None


def _c(ticker, estrategia, familia, score):
    return {"ticker": ticker, "estrategia": estrategia, "familia": familia, "score": score}


def test_confluencia_cuenta_familias_no_instancias():
    cands = [_c("AAA", f"TS_{i}", "TECH_SECTOR", 5.0) for i in range(5)]
    cands += [_c("BBB", "ML_v1", "ML", 70), _c("BBB", "SMC_v1", "SMC", 3.0)]
    sel = ft.seleccionar(cands)
    # AAA esta en 5 estrategias pero de UNA familia: no es confluencia
    assert [c["ticker"] for c in sel["confluencia"]] == ["BBB"]
    assert sel["confluencia"][0]["familias"] == ["ML", "SMC"]


def test_confluencia_ordena_por_cantidad_de_familias():
    cands = [
        _c("DOS", "ML_v1", "ML", 1), _c("DOS", "SMC_v1", "SMC", 1),
        _c("TRES", "ML_v1", "ML", 1), _c("TRES", "SMC_v1", "SMC", 1),
        _c("TRES", "TECH_v1", "TECH", 1),
    ]
    sel = ft.seleccionar(cands)
    assert [c["ticker"] for c in sel["confluencia"]] == ["TRES", "DOS"]


def test_top_por_familia_por_score_y_orden_sin_duplicados():
    cands = [_c(t, "ML_v1", "ML", s) for t, s in [("A", 60), ("B", 80), ("C", 70), ("D", 50)]]
    cands += [_c("B", "SMC_v1", "SMC", 2.0), _c("E", "SMC_v1", "SMC", 4.0)]
    sel = ft.seleccionar(cands, top_n=2)
    assert [f["ticker"] for f in sel["top"]["ML"]] == ["B", "C"]
    assert [f["ticker"] for f in sel["top"]["SMC"]] == ["E", "B"]
    # B primero (confluencia) y no se repite
    assert sel["orden"] == ["B", "C", "E"]


def test_tope_de_tickers():
    cands = [_c(f"T{i:02d}", "ML_v1", "ML", i) for i in range(30)]
    sel = ft.seleccionar(cands, top_n=30, max_tickers=4)
    assert len(sel["orden"]) == 4


# --- volumen --------------------------------------------------------------------

def test_vol_relativo_usa_mediana_de_las_previas_sin_la_propia():
    vols = [100.0] * 200 + [1_000_000.0] * 5 + [100.0] * 60 + [250.0]
    # los 5 picos no mueven la mediana; el dia propio no entra en la base
    assert ft.vol_relativo(vols, len(vols) - 1) == 2.5


def test_vol_relativo_con_historia_corta_es_none():
    vols = [100.0] * 50 + [200.0]
    assert ft.vol_relativo(vols, 50) is None


# --- foto -----------------------------------------------------------------------

def _serie(n=300, close0=100.0, paso=0.5):
    d0 = date(2025, 6, 2)
    dias = trading_days_between(d0, d0 + timedelta(days=n * 2))[:n]
    filas = []
    for i, d in enumerate(dias):
        c = close0 + paso * i
        filas.append({"fecha": d, "close": c, "volume": 1000.0, "rsi14": 50.0,
                      "macd": 1.0, "macd_signal": 0.5,
                      "sma50": c - 5, "sma200": c - 20})
    return filas


def test_foto_basica():
    filas = _serie()
    foto = ft.construir_foto(filas)
    assert foto["fecha"] == filas[-1]["fecha"]
    assert len(foto["ruedas"]) == 6       # 5 previas + la de datos
    assert foto["rsi_estado"] == "Neutral" and foto["macd"] == "Compra"
    assert foto["lado50"] == "sobre" and foto["lado200"] == "sobre"
    assert foto["vol_x"] == 1.0
    esperado = (filas[-1]["close"] / filas[-6]["close"] - 1) * 100
    assert abs(foto["var_n_pct"] - esperado) < 1e-9


def test_foto_marca_cambios_de_estado():
    filas = _serie()
    filas[-2].update(macd=0.2, macd_signal=0.5, rsi14=66.0)   # venta, sobrecompra
    filas[-1].update(macd=0.9, macd_signal=0.5, rsi14=60.0,   # compra, neutral
                     sma50=filas[-1]["close"] + 1)            # cruza SMA50 abajo
    foto = ft.construir_foto(filas)
    ult = foto["ruedas"][-1]["cambios"]
    assert "MACD->C" in ult
    assert "RSI->N" in ult
    assert "SMA50->bajo" in ult
    assert "RSI->SC" in foto["ruedas"][-2]["cambios"]


def test_foto_tolera_indicadores_faltantes_y_nan():
    filas = _serie()
    filas[-1].update(rsi14=None, macd=float("nan"), sma200=None)
    foto = ft.construir_foto(filas)
    assert foto["rsi_estado"] is None and foto["macd"] is None
    assert foto["dist200"] is None
    txt = ft.bloque_ticker("X", foto)
    assert "RSI s/d" in txt and "SMA200 s/d" in txt


def test_foto_con_pocas_filas_es_none():
    assert ft.construir_foto(_serie(n=6)) is None


# --- earnings -------------------------------------------------------------------

def test_ruedas_hasta_cuenta_habiles():
    jue = date(2026, 9, 24)
    assert ft.ruedas_hasta(jue, jue) == 0
    assert ft.ruedas_hasta(jue, date(2026, 9, 25)) == 1     # viernes
    assert ft.ruedas_hasta(jue, date(2026, 9, 28)) == 2     # lunes (salta finde)
    assert ft.ruedas_hasta(date(2026, 11, 25), date(2026, 11, 27)) == 1  # Thanksgiving


def test_ruedas_hasta_fecha_pasada_o_sin_fecha():
    assert ft.ruedas_hasta(date(2026, 9, 24), date(2026, 9, 1)) is None
    assert ft.ruedas_hasta(date(2026, 9, 24), None) is None


# --- texto ----------------------------------------------------------------------

def test_bloque_es_ascii_y_html_valido():
    filas = _serie()
    foto = ft.construir_foto(filas)
    txt = ft.bloque_ticker("BRK-B", foto, familias=["ML", "SMC"], sector="Tech & Co",
                           earnings_fecha=filas[-1]["fecha"] + timedelta(days=14),
                           rueda_ref=filas[-1]["fecha"])
    txt.encode("ascii")
    assert "&amp;" in txt and "Tech & Co" not in txt
    assert txt.count("<pre>") == 1 and txt.count("</pre>") == 1
    assert "Earnings: en" in txt and "[ML, SMC | " in txt
    assert "OJO" not in txt


def test_bloque_avisa_rueda_distinta():
    filas = _serie()
    foto = ft.construir_foto(filas)
    txt = ft.bloque_ticker("X", foto, rueda_ref=filas[-1]["fecha"] + timedelta(days=1))
    assert "OJO: ultimo dato" in txt


def test_numero_formato_ar():
    assert ft._num(1234.5) == "1.234,50"
    assert ft._sg(-2.345) == "-2,3%"
    assert ft._sg(0.0) == "+0,0%"


def test_empaquetar_no_parte_bloques():
    bloques = ["a" * 1500, "b" * 1500, "c" * 1500, "d" * 5000]
    msgs = ft.empaquetar(bloques, max_len=4000)
    assert msgs[0] == "a" * 1500 + "\n\n" + "b" * 1500
    assert msgs[1] == "c" * 1500
    assert msgs[2] == "d" * 5000
    assert "".join(msgs).replace("\n", "") == "".join(bloques)
