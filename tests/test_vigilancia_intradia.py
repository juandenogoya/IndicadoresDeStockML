"""
test_vigilancia_intradia.py -- lista de vigilancia, eventos del corte,
novedades 11:00 -> 15:00 y mensaje (src/utils/vigilancia_intradia.py,
docs/intradia.md sec. 9).
"""

import math
from datetime import date, datetime, timezone

import pytest

from src.utils import vigilancia_intradia as vi

D = date(2026, 10, 30)
GEN = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def _res(oi_abajo=(), oi_arriba=(), sw_abajo=(), sw_arriba=(), sma_arriba=(), sma_abajo=(),
         cerca_oi=(), cerca_sw=(), veredictos=()):
    return {
        "cruces": {
            "OI": {"abajo": list(oi_abajo), "arriba": list(oi_arriba)},
            "SWING": {"abajo": list(sw_abajo), "arriba": list(sw_arriba)},
            "SMA": {"arriba": list(sma_arriba), "abajo": list(sma_abajo)},
        },
        "vigilar": {"cerca_oi": list(cerca_oi), "cerca_swing": list(cerca_sw),
                    "earnings": [("ZZZ", 2, D)], "veredictos": list(veredictos),
                    "veredictos_comparables": 10},
    }


def _fila(tk, close, close_ant=None, sma21=None, sma50=None, sma200=None):
    return {"ticker": tk, "close": close, "close_ant": close_ant, "sma21": sma21,
            "sma50": sma50, "sma200": sma200}


# ── utilidades ────────────────────────────────────────────────────────────────

def test_lado():
    assert vi.lado(101, 100) == "sobre"
    assert vi.lado(99, 100) == "bajo"
    assert vi.lado(100, 100) is None and vi.lado(None, 100) is None


def test_sigma_log_exige_historia():
    assert vi.sigma_log([100.0] * 50) is None
    closes = [100.0 * (1.01 if i % 2 else 0.99) ** (i % 2) for i in range(200)]
    s = vi.sigma_log(closes)
    assert s is not None and s > 0
    # usa solo las ultimas VENTANA_SIGMA ruedas
    largo = [1000.0] * 500 + closes
    assert vi.sigma_log(largo) == pytest.approx(vi.sigma_log([1000.0] * 53 + closes))


# ── motivos ───────────────────────────────────────────────────────────────────

def test_motivos_foto_respeta_el_orden_con_tope():
    sel = {"confluencia": [{"ticker": "AAA", "familias": ["ML", "SMC"], "n_estrategias": 3}],
           "top": {"ML": [{"ticker": "AAA"}, {"ticker": "BBB"}], "TECH": [{"ticker": "CCC"}]},
           "orden": ["AAA", "BBB"]}
    m = vi.motivos_foto(sel)
    assert m == {"AAA": ["foto ML+SMC", "foto top ML"], "BBB": ["foto top ML"]}


def test_motivos_cierre_sin_balances():
    res = _res(oi_abajo=[("AAA", 150.0)], sma_arriba=[("BBB", "SMA50")],
               cerca_oi=[("CCC", "res", 0.5)], veredictos=[("DDD", "NEUTRAL", "ALCISTA")])
    m = vi.motivos_cierre(res)
    assert m["AAA"] == ["perdio sop OI"]
    assert m["BBB"] == ["paso sobre SMA50"]
    assert m["CCC"] == ["cerca res OI"]
    assert m["DDD"] == ["veredicto N>A"]
    assert "ZZZ" not in m        # balances proximos: decision de no vigilarlos


# ── armar_lista ───────────────────────────────────────────────────────────────

def _lista_basica():
    filas = [_fila("AAA", 148.0, close_ant=152.0, sma21=146.0, sma50=140.0, sma200=120.0)]
    res = _res(oi_abajo=[("AAA", 150.0)], sw_abajo=[("AAA", None)])
    motivos = {"AAA": ["foto TECH+SMC", "perdio sop OI"]}
    strikes = {"AAA": (145.0, 155.0)}
    return vi.armar_lista(D, filas, res, motivos, strikes, {"AAA": 0.02}, GEN)


def test_armar_lista_niveles_y_lados():
    tickers, niveles = _lista_basica()
    assert tickers == [{"fecha": D, "ticker": "AAA", "close_d": 148.0, "sigma_d": 0.02,
                        "motivos": "foto TECH+SMC; perdio sop OI", "origen": "push",
                        "generado_en": GEN}]
    n = {x["nivel"]: x for x in niveles}
    assert n["OI_SOP"]["valor"] == 145.0 and n["OI_SOP"]["lado_d"] == "sobre"
    assert n["OI_RES"]["lado_d"] == "bajo"
    assert n["OI_SOP_ANT"] == {"fecha": D, "ticker": "AAA", "nivel": "OI_SOP_ANT", "valor": 150.0,
                               "lado_d": "bajo", "cruzado_en_d": True}
    assert n["SMA21"]["valor"] == 146.0 and n["SMA21"]["lado_d"] == "sobre"
    assert not n["SMA50"]["cruzado_en_d"]
    assert "OI_RES_ANT" not in n


def test_los_swings_ya_no_son_niveles():
    """Decidido el 3/11/2026: los swings N5 salen del modelo intradiario."""
    _, niveles = _lista_basica()
    assert not [x for x in niveles if x["nivel"].startswith("SW_")]
    assert not [k for k in vi.NIVELES if k.startswith("SW_")]


def test_nivel_cruzado_reemplaza_al_de_d_si_coincide():
    filas = [_fila("AAA", 148.0)]
    res = _res(oi_abajo=[("AAA", 150.0)])
    tickers, niveles = vi.armar_lista(D, filas, res, {"AAA": ["x"]}, {"AAA": (150.0, None)},
                                      {}, GEN)
    noms = [x["nivel"] for x in niveles]
    assert "OI_SOP_ANT" in noms and "OI_SOP" not in noms


def test_sma_cruzada_en_d_se_marca_en_su_fila():
    filas = [_fila("BBB", 101.0, sma21=99.0, sma50=100.0)]
    res = _res(sma_arriba=[("BBB", "SMA50")])
    _, niveles = vi.armar_lista(D, filas, res, {"BBB": ["paso sobre SMA50"]}, {}, {}, GEN)
    n = {x["nivel"]: x for x in niveles}
    assert n["SMA50"] == {"fecha": D, "ticker": "BBB", "nivel": "SMA50", "valor": 100.0,
                          "lado_d": "sobre", "cruzado_en_d": True}
    assert n["SMA21"]["valor"] == 99.0 and not n["SMA21"]["cruzado_en_d"]


def test_ticker_sin_close_queda_afuera():
    tickers, niveles = vi.armar_lista(D, [], _res(), {"AAA": ["x"]}, {}, {}, GEN)
    assert tickers == [] and niveles == []


# ── eventos ───────────────────────────────────────────────────────────────────

def test_eventos_estado_y_movimiento():
    tickers, niveles = _lista_basica()
    # 151: recupera el soporte perdido ayer (150) y pasa sobre la SMA21 (146);
    # sigue sobre el soporte de D (145) y debajo de la resistencia (155).
    evs = vi.eventos(tickers, niveles, {"AAA": 151.0})
    assert {e["tipo"] for e in evs} == {"estado"}
    niv = {e["nivel"] for e in evs}
    assert "OI_SOP_ANT" in niv and "OI_SOP" not in niv
    ant = [e for e in evs if e["nivel"] == "OI_SOP_ANT"][0]
    assert ant["cruzado_en_d"] and ant["lado"] == "sobre"
    assert ant["var_pct"] == pytest.approx((151 / 148 - 1) * 100)
    # 144: pierde el soporte de D y queda bajo las tres medias que estaban debajo
    evs = vi.eventos(tickers, niveles, {"AAA": 144.0})
    sop = [e for e in evs if e["nivel"] == "OI_SOP"][0]
    assert sop["tipo"] == "estado" and not sop["cruzado_en_d"] and sop["lado"] == "bajo"
    # movimiento: ln(144/148)/0,02 = -1,37 -> no; con 140 = -2,78 -> si
    assert not [e for e in evs if e["tipo"] == "movimiento"]
    evs = vi.eventos(tickers, niveles, {"AAA": 140.0})
    mov = [e for e in evs if e["tipo"] == "movimiento"][0]
    assert mov["z"] == pytest.approx(math.log(140 / 148) / 0.02)
    assert mov["var_pct"] == pytest.approx((140 / 148 - 1) * 100)


def test_sin_precio_del_corte_no_hay_eventos():
    tickers, niveles = _lista_basica()
    assert vi.eventos(tickers, niveles, {}) == []


def test_sin_sigma_no_hay_movimiento():
    tickers, niveles = _lista_basica()
    tickers[0]["sigma_d"] = None
    assert not [e for e in vi.eventos(tickers, niveles, {"AAA": 100.0}) if e["tipo"] == "movimiento"]


def test_novedades():
    a = {"ticker": "AAA", "tipo": "estado", "nivel": "OI_SOP"}
    b = {"ticker": "BBB", "tipo": "movimiento", "nivel": None}
    c = {"ticker": "CCC", "tipo": "estado", "nivel": "SMA50"}
    nuevos, deshechos = vi.novedades([a, c], [a, b])
    assert nuevos == [c] and deshechos == [b]
    nuevos, deshechos = vi.novedades([a], None)
    assert nuevos == [a] and deshechos == []


# ── mensaje ───────────────────────────────────────────────────────────────────

def test_mensaje_sin_novedades_lo_dice():
    tickers, _ = _lista_basica()
    txt = vi.mensaje(date(2026, 11, 2), "apertura_90", D, tickers, [], [], [])
    assert "1 vigilados, sin cambios de estado." in txt
    assert "Estado, no pronostico." in txt
    txt = vi.mensaje(date(2026, 11, 2), "cierre_60", D, tickers, [], [], [], hay_previo=True)
    assert "sin cambios de estado desde las 11:00." in txt


def test_mensaje_con_cambios_y_deshechos():
    tickers, niveles = _lista_basica()
    evs = vi.eventos(tickers, niveles, {"AAA": 151.0})
    txt = vi.mensaje(date(2026, 11, 2), "cierre_60", D, tickers, evs,
                     [{"ticker": "AAA", "tipo": "movimiento", "nivel": None, "var_pct": -3.0,
                       "z": -2.1, "valor": None, "lado": None}], ["BBB"], hay_previo=True)
    assert "Nuevos: cambian de estado (1)" in txt       # UNA linea: un solo ticker
    assert "recupera el soporte OI 150,00 que perdio ayer" in txt
    assert "SMA21" not in txt      # a 151 ya estaba sobre la SMA21 (146): no cambio
    assert "Ya no mueven como a las 11:00 (1)" in txt
    assert "sin dato del corte: 1" in txt
    assert "<i>(foto TECH+SMC +)</i>" in txt


def test_una_linea_por_ticker_aunque_cambien_varios_estados():
    """El mensaje agrupa: 3 cambios del mismo ticker son UNA linea."""
    tickers, niveles = _lista_basica()
    tickers[0]["sigma_d"] = 0.20          # sigma grande: que no dispare `movimiento`
    evs = vi.eventos(tickers, niveles, {"AAA": 121.0})   # queda bajo SMA21, SMA50 y el soporte
    assert not [e for e in evs if e["tipo"] == "movimiento"]
    assert len({e["nivel"] for e in evs if e["tipo"] == "estado"}) >= 3
    txt = vi.mensaje(date(2026, 11, 2), "apertura_90", D, tickers, evs, [], [])
    assert "cambian de estado (1)" in txt
    assert txt.count("<b>AAA</b>") == 1


def test_la_nota_del_z_sale_solo_si_hay_movimiento():
    tickers, niveles = _lista_basica()
    solo_estado = vi.eventos(tickers, niveles, {"AAA": 144.0})
    assert "fuera de escala" not in vi.mensaje(
        date(2026, 11, 2), "apertura_90", D, tickers, solo_estado, [], [])
    con_mov = vi.eventos(tickers, niveles, {"AAA": 140.0})
    assert "fuera de escala" in vi.mensaje(
        date(2026, 11, 2), "apertura_90", D, tickers, con_mov, [], [])


def test_mensaje_largo_achica_listas_y_no_se_parte():
    tickers = [{"ticker": f"T{i:03d}", "motivos": "foto top ML; cerca sop OI"} for i in range(300)]
    evs = [{"ticker": f"T{i:03d}", "tipo": "estado", "nivel": "SMA200", "valor": 100.0,
            "lado": "bajo", "cruzado_en_d": False, "precio": 99.0, "var_pct": -1.5,
            "dist_pct": -1.0} for i in range(300)]
    txt = vi.mensaje(date(2026, 11, 2), "apertura_90", D, tickers, evs, [], [])
    assert len(txt) <= vi.MAX_LEN_MENSAJE
    assert "cambian de estado (300)" in txt and "mas" in txt
    assert txt.endswith("<i>Estado, no pronostico.</i>")


def test_mensaje_sin_lista():
    txt = vi.mensaje_sin_lista(date(2026, 11, 2), "apertura_90", D, date(2026, 10, 28))
    assert "rueda 30/10 no subida (la ultima es del 28/10)" in txt


def test_ddl_tiene_las_columnas():
    for col in vi.COLS_LISTAS + vi.COLS_TICKERS + vi.COLS_NIVELES + vi.COLS_ENVIOS:
        assert col in vi.DDL_TABLAS
