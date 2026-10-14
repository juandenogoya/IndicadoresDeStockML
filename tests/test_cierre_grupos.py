"""Tests de src/utils/cierre_grupos.py (cierre por sector para Telegram)."""

from datetime import date

from src.utils import cierre_grupos as cg

D = date(2026, 9, 28)
D_ANT = date(2026, 9, 25)


def _r(ticker, sector="Technology", close=100.0, close_ant=100.0, close_5=100.0, **kw):
    base = {"ticker": ticker, "sector": sector, "close": close, "close_ant": close_ant,
            "close_5": close_5, "sma50": None, "sma50_ant": None, "sma200": None,
            "sma200_ant": None, "sop_ant": None, "res_ant": None, "sop_dist": None,
            "res_dist": None, "swing_arriba": 0, "swing_abajo": 0, "dist_sh": None,
            "dist_sl": None, "veredicto": None, "veredicto_ant": None,
            "earnings_ruedas": None, "earnings_fecha": None}
    base.update(kw)
    return base


# --- grupos ---------------------------------------------------------------------

def test_mediana_y_no_promedio():
    # un ticker con +50% no arrastra la mediana del sector
    filas = [_r("A", close=101), _r("B", close=102), _r("C", close=150)]
    g = cg.resumen_grupos(filas, min_n=1)["grupos"][0]
    assert abs(g["var_dia"] - 2.0) < 1e-9
    assert g["suben"] == 3 and g["bajan"] == 0


def test_sector_chico_no_compite_en_el_ranking():
    filas = [_r(f"T{i}", "Technology", close=99) for i in range(5)]
    filas += [_r(f"E{i}", "Energy", close=101) for i in range(5)]
    filas += [_r("U1", "Utilities", close=110)]            # n=1, sube mas que todos
    res = cg.resumen_grupos(filas)
    assert res["fuerte"] == "Energy"
    assert res["debil"] == "Technology"
    ult = res["grupos"][-1]
    assert ult["sector"] == "Utilities" and not ult["rankeable"]


def test_un_solo_sector_rankeable_no_tiene_debil():
    filas = [_r(f"T{i}", close=101) for i in range(5)]
    res = cg.resumen_grupos(filas)
    assert res["fuerte"] == "Technology" and res["debil"] is None


def test_sobre_sma50_ignora_tickers_sin_sma():
    filas = [_r("A", sma50=90), _r("B", sma50=110), _r("C", sma50=None)]
    g = cg.resumen_grupos(filas, min_n=1)["grupos"][0]
    assert g["sobre_sma50_pct"] == 50.0


def test_veredictos_por_sector():
    filas = [_r("A", veredicto="ALCISTA"), _r("B", veredicto="BAJISTA"), _r("C")]
    g = cg.resumen_grupos(filas, min_n=1)["grupos"][0]
    assert g["veredictos"] == {"ALCISTA": 1, "NEUTRAL": 0, "BAJISTA": 1}


# --- cruces ---------------------------------------------------------------------

def test_muro_se_compara_contra_la_rueda_anterior():
    # soporte de ayer 98: el cierre de hoy 97 lo perdio
    # (el soporte de HOY estaria debajo de 97 por construccion y nunca daria cruce)
    filas = [_r("A", close=97, sop_ant=98, res_ant=105),
             _r("B", close=106, sop_ant=98, res_ant=105),
             _r("C", close=100, sop_ant=98, res_ant=105)]
    cr = cg.cruces(filas)
    assert cr["OI"]["abajo"] == [("A", 98.0)]
    assert cr["OI"]["arriba"] == [("B", 105.0)]


def test_muro_faltante_no_cruza():
    cr = cg.cruces([_r("A", close=50, sop_ant=None, res_ant=None)])
    assert cr["OI"] == {"arriba": [], "abajo": []}


def test_swing_usa_bos_y_choch():
    filas = [_r("A", swing_abajo=1), _r("B", swing_arriba=1), _r("C", swing_abajo=None)]
    cr = cg.cruces(filas)
    assert cr["SWING"]["abajo"] == [("A", None)]
    assert cr["SWING"]["arriba"] == [("B", None)]


def test_sma_cambio_de_lado():
    filas = [
        _r("A", close_ant=101, sma50_ant=100, close=99, sma50=100),     # baja la 50
        _r("B", close_ant=99, sma200_ant=100, close=101, sma200=100),   # sube la 200
        _r("C", close_ant=101, sma50_ant=100, close=102, sma50=100),    # sin cruce
        _r("D", close_ant=None, sma50_ant=100, close=90, sma50=100),    # sin dato previo
    ]
    cr = cg.cruces(filas)
    assert cr["SMA"]["abajo"] == [("A", "SMA50")]
    assert cr["SMA"]["arriba"] == [("B", "SMA200")]


def test_tickers_con_cruce_cuenta_distintos():
    filas = [_r("A", close_ant=101, sma50_ant=100, sma200_ant=100,
                close=99, sma50=100, sma200=100)]
    cr = cg.cruces(filas)
    assert len(cr["SMA"]["abajo"]) == 2
    assert cg.tickers_con_cruce(cr, "SMA") == {"A"}


# --- vigilar --------------------------------------------------------------------

def test_cerca_toma_el_nivel_mas_cercano_y_respeta_umbral():
    filas = [_r("A", sop_dist=-0.4, res_dist=0.9),
             _r("B", sop_dist=-1.0),                 # justo en el umbral: no entra
             _r("C", dist_sl=0.2, dist_sh=-0.1)]
    v = cg.vigilar(filas)
    assert v["cerca_oi"] == [("A", "sop", -0.4)]
    assert v["cerca_swing"] == [("C", "SH", -0.1)]


def test_earnings_y_veredictos():
    filas = [_r("A", earnings_ruedas=2, earnings_fecha=date(2026, 9, 30)),
             _r("B", earnings_ruedas=0),             # en la rueda de datos: ya paso
             _r("C", earnings_ruedas=6),
             _r("D", veredicto_ant="NEUTRAL", veredicto="ALCISTA"),
             _r("E", veredicto_ant="NEUTRAL", veredicto="NEUTRAL")]
    v = cg.vigilar(filas)
    assert v["earnings"] == [("A", 2, date(2026, 9, 30))]
    assert v["veredictos"] == [("D", "NEUTRAL", "ALCISTA")]


# --- texto ----------------------------------------------------------------------

def _universo():
    sectores = ["Technology", "Energy", "Healthcare", "Real Estate"]
    filas = []
    for s_i, s in enumerate(sectores):
        for i in range(12 if s != "Real Estate" else 3):
            filas.append(_r(f"{s[:2].upper()}{i}", s, close=100 + s_i + i * 0.1,
                            sma50=100, sma50_ant=100, close_ant=100,
                            sop_ant=99, res_ant=101 + s_i, sop_dist=-0.5,
                            swing_abajo=i % 5 == 0, dist_sl=0.3,
                            veredicto="NEUTRAL", veredicto_ant="ALCISTA",
                            earnings_ruedas=i % 7, earnings_fecha=date(2026, 10, 1)))
    return filas


def test_mensaje_ascii_y_dentro_del_largo():
    res = cg.construir(D, D_ANT, _universo(), n_universo=200)
    msgs = cg.mensajes(res)
    assert 1 <= len(msgs) <= 2
    for m in msgs:
        assert len(m) <= cg.MAX_LEN_MENSAJE
        m.encode("ascii")
    txt = "\n".join(msgs)
    assert "39 de 200 tickers con dato" in txt
    assert "Real Estate*" in txt
    assert "y " in txt and " mas" in txt           # las listas se recortan
    assert "Estado, no pronostico" in txt


def test_hueco_de_precios_se_avisa():
    res = cg.construir(date(2026, 9, 23), date(2026, 9, 21), _universo(),
                       ruedas_salteadas=[date(2026, 9, 22)])
    txt = cg.bloque_grupos(res)
    assert "hueco en precios_diarios (22/09)" in txt
    assert "rueda del 21/09" in txt
    assert "hueco" not in cg.bloque_grupos(cg.construir(D, D_ANT, _universo()))


def test_oi_no_disponible_se_dice():
    res = cg.construir(D, D_ANT, _universo(), oi_disponible=False)
    txt = cg.bloque_cruces(res)
    assert "hueco en opciones" in txt


# --- evaluacion -----------------------------------------------------------------

def test_evaluar_y_criterios():
    filas = _universo()
    resultados = [cg.construir(D, D_ANT, filas) for _ in range(5)]
    ev = cg.evaluar(resultados)
    assert ev["n_ruedas"] == 5
    # el mismo sector fuerte las 5 ruedas -> share 1.0 -> el criterio 3 falla
    assert ev["fuerte"]["share"] == 1.0
    ver = cg.veredicto_criterios(ev)
    assert ver["c3"] is False
    assert ver["c4"] is True


def test_evaluar_excluye_oi_de_ruedas_sin_muros():
    filas = _universo()
    resultados = [cg.construir(D, D_ANT, filas, oi_disponible=False),
                  cg.construir(D, D_ANT, filas)]
    ev = cg.evaluar(resultados)
    assert len(ev["cruces_por_dia"]["OI"]) == 1
    assert len(ev["cruces_por_dia"]["SMA"]) == 2


def test_sin_veredictos_de_la_rueda_anterior_no_es_ningun_cambio():
    """13/10/2026: FT no corrio la rueda del 12/10 y el mensaje decia
    "Cambio de veredicto (0): ninguno" sin tener contra que comparar."""
    filas = [_r("A", veredicto="ALCISTA"), _r("B", veredicto="NEUTRAL")]
    v = cg.vigilar(filas)
    assert v["veredictos"] == [] and v["veredictos_comparables"] == 0
    filas = [_r("A", veredicto_ant="ALCISTA", veredicto="ALCISTA")]
    assert cg.vigilar(filas)["veredictos_comparables"] == 1
