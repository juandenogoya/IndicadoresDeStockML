"""
test_ft_comparar.py -- por que difieren ML_SCANNER_v1 y v2 (Etapa 3f).

Casos sinteticos y deterministas (sin DB). Los dos que motivan el diseno: una
rueda con muchas senales exclusivas es UNA observacion de mercado (INSUFICIENTE
aunque sobren senales), y el exceso se mide contra el universo entero de la rueda,
tenga o no la v2.
"""

from datetime import date, datetime, timedelta

import pytest

from src.utils import ft_comparar as fc

D0 = date(2026, 9, 1)
CF = "COMPRA_FUERTE"


def d(i):
    return D0 + timedelta(days=i)


def alterna(i, a):
    return a if i % 2 == 0 else -a


def fila(ticker, i, n1="NEUTRAL", s1=50, n2="NEUTRAL", s2=50, sector="Tech",
         p1=0.5, p2=0.5, scan=None):
    return {"ticker": ticker, "sector": sector, "precio_fecha": d(i),
            "scan_fecha": scan or datetime(2026, 9, 1, 21, 0) + timedelta(days=i),
            "nivel_v1": n1, "score_v1": s1, "prob_v1": p1,
            "nivel_v2": n2, "score_v2": s2, "prob_v2": p2}


def universo_con_exclusivas(n_ruedas, por_rueda, ret_v2=0.05, ret_v1=-0.05, h=5):
    """
    Por rueda: `por_rueda` senales solo v2, otras tantas solo v1 y la misma
    cantidad de neutrales. Ruido de signo opuesto en v2 y v1: el promedio del
    universo queda en cero y el exceso conserva la varianza.
    """
    ruedas = [d(i) for i in range(n_ruedas + h + 1)]
    filas, closes = [], {}
    for i in range(n_ruedas):
        for k in range(por_rueda):
            a = alterna(i + k, 0.01)
            for tipo, r, (n1, s1), (n2, s2) in (
                    ("V2", ret_v2 + a, ("COMPRA", 70), (CF, 80)),
                    ("V1", ret_v1 - a, (CF, 80), ("COMPRA", 70)),
                    ("N", 0.0, ("NEUTRAL", 50), ("NEUTRAL", 50))):
                t = f"{tipo}{i}_{k}"
                filas.append(fila(t, i, n1, s1, n2, s2))
                closes[t] = {d(i): 100.0, d(i + h): 100.0 * (1 + r)}
    return filas, closes, ruedas


# ── Senales ───────────────────────────────────────────────────────────────────

def test_senal_es_la_regla_de_entrada_de_las_dos_estrategias():
    assert fc.es_senal(CF, 80)
    assert fc.es_senal("COMPRA_FUERTE   ", 80)       # CHAR(n) con relleno
    assert not fc.es_senal(CF, 64)
    assert not fc.es_senal("COMPRA", 90)
    assert not fc.es_senal(CF, None)
    assert not fc.es_senal(None, 80)


def test_grupo_de_cada_fila():
    assert fc.grupo(fila("A", 0, CF, 80, CF, 80)) == "ambas"
    assert fc.grupo(fila("A", 0, CF, 80, "COMPRA", 70)) == "solo_v1"
    assert fc.grupo(fila("A", 0, "COMPRA", 70, CF, 80)) == "solo_v2"
    assert fc.grupo(fila("A", 0)) is None
    assert fc.grupo(fila("A", 0, CF, 80, None, None)) == "solo_v1"


def test_filas_por_rueda_se_queda_con_la_ultima_corrida():
    temprana = fila("T", 0, CF, 80, CF, 80, scan=datetime(2026, 9, 1, 20, 0))
    tardia = fila("T", 0, scan=datetime(2026, 9, 1, 22, 0))
    sin_reloj = fila("T", 0, CF, 80, CF, 80)
    sin_reloj["scan_fecha"] = None
    assert fc.filas_por_rueda([tardia, temprana]) == [tardia]
    assert fc.filas_por_rueda([sin_reloj, tardia]) == [tardia]
    assert len(fc.filas_por_rueda([fila("T", 0), fila("T", 1), fila("U", 0)])) == 3


def test_retorno_cuenta_ruedas_del_mercado_no_las_del_ticker():
    ruedas = [d(i) for i in range(7)]
    idx = fc.indice_ruedas(ruedas)
    closes = {"T": {d(0): 100.0, d(1): 101.0, d(3): 110.0, d(5): 120.0}}   # sin d2
    assert fc.retorno_adelante(closes, ruedas, idx, "T", d(0), 2) is None   # falta d2
    assert fc.retorno_adelante(closes, ruedas, idx, "T", d(0), 3) == pytest.approx(0.10)
    assert fc.retorno_adelante(closes, ruedas, idx, "T", d(1), 2) == pytest.approx(110 / 101 - 1)
    assert fc.retorno_adelante(closes, ruedas, idx, "T", d(5), 2) is None   # no hay d7
    assert fc.retorno_adelante(closes, ruedas, idx, "X", d(0), 1) is None


def test_exceso_contra_el_universo_entero_de_la_rueda_con_o_sin_v2():
    ruedas = [d(i) for i in range(6)]
    con_v2 = fila("X", 0, CF, 80, CF, 80)
    sin_v2 = fila("Y", 0, CF, 80, None, None)
    closes = {"X": {d(0): 100.0, d(5): 110.0}, "Y": {d(0): 100.0, d(5): 100.0}}
    nuevas, medias = fc.agregar_retornos([con_v2, sin_v2], closes, ruedas, horizontes=(5,))
    assert medias[(d(0), 5)] == pytest.approx(0.05)
    assert nuevas[0]["exc_5"] == pytest.approx(0.05)
    assert nuevas[1]["exc_5"] == pytest.approx(-0.05)

    res = fc.comparar([con_v2, sin_v2], closes, ruedas, entrenados_v1={"X", "Y"},
                      horizontes=(5,))
    assert res["filas"] == 1 and res["filas_sin_v2"] == 1
    assert res["solapamiento"]["total"]["ambas"] == 1


def test_solapamiento_y_jaccard_por_rueda():
    filas = [fila("A", 0, CF, 80, CF, 80), fila("B", 0, CF, 80, "COMPRA", 70),
             fila("C", 0, "COMPRA", 70, CF, 80), fila("D", 0, "COMPRA", 70, CF, 80),
             fila("E", 0), fila("F", 1)]
    s = fc.solapamiento(filas)
    r0, r1 = s["ruedas"]
    assert (r0["filas"], r0["v1"], r0["v2"], r0["ambas"], r0["solo_v1"], r0["solo_v2"]) == (5, 2, 3, 1, 1, 2)
    assert r0["jaccard"] == pytest.approx(0.25)
    assert r1["jaccard"] is None
    assert s["total"]["n_ruedas"] == 2
    assert s["total"]["jaccard"] == pytest.approx(0.25)
    assert s["total"]["jaccard_medio_diario"] == pytest.approx(0.25)


def test_muchas_exclusivas_de_una_sola_rueda_son_insuficientes():
    filas, closes, ruedas = universo_con_exclusivas(n_ruedas=1, por_rueda=14)
    res = fc.comparar(filas, closes, ruedas, entrenados_v1=set(), horizontes=(5,))
    solo_v2 = res["senales"]["grupos"]["solo_v2"]
    assert solo_v2["n"] == 14
    assert solo_v2["horizontes"][5]["exceso"]["veredicto"] == fc.INSUFICIENTE
    assert solo_v2["horizontes"][5]["exceso"]["media"] is None
    cmp = res["senales"]["v2_vs_v1"][5]
    assert cmp["veredicto"] == fc.INSUFICIENTE
    assert cmp["diferencia"] is None and cmp["ic95_lo"] is None
    assert (cmp["n_ruedas_v1"], cmp["n_ruedas_v2"]) == (1, 1)


def test_exclusivas_de_la_v2_que_rinden_mas_dan_a_favor_de_v2():
    filas, closes, ruedas = universo_con_exclusivas(n_ruedas=6, por_rueda=2)
    res = fc.comparar(filas, closes, ruedas, entrenados_v1=set(), horizontes=(5,))
    g = res["senales"]["grupos"]
    assert g["solo_v2"]["horizontes"][5]["exceso"]["media"] == pytest.approx(5.0)
    assert g["solo_v1"]["horizontes"][5]["exceso"]["veredicto"] == fc.NEGATIVO
    cmp = res["senales"]["v2_vs_v1"][5]
    assert cmp["veredicto"] == fc.A_FAVOR_V2
    assert cmp["diferencia"] == pytest.approx(10.0)
    assert cmp["ic95_lo"] > 0


def test_sin_ventana_completa_la_senal_cuenta_pero_sin_retorno():
    filas, closes, ruedas = universo_con_exclusivas(n_ruedas=6, por_rueda=2)
    res = fc.comparar(filas, closes, ruedas, entrenados_v1=set(), horizontes=(5, 20))
    h20 = res["senales"]["grupos"]["solo_v2"]["horizontes"][20]
    assert h20["sin_retorno"] == 12
    assert h20["exceso"]["veredicto"] == fc.INSUFICIENTE


def test_atribucion_de_las_exclusivas():
    filas = [
        fila("NUEVO", 0, "COMPRA", 70, CF, 78, sector="Health", p1=0.60, p2=0.80),
        fila("VIEJO", 0, "NEUTRAL", 55, CF, 85, sector="Health", p1=0.50, p2=0.70),
        fila("SOLO1", 0, CF, 76, "COMPRA", 68, sector="Tech", p1=0.76, p2=0.55),
        fila("AMBAS", 0, CF, 80, CF, 80),
    ]
    a = fc.atribucion(filas, entrenados_v1={"VIEJO", "SOLO1", "AMBAS"})
    assert a["universo_sin_entrenar_pct"] == pytest.approx(25.0)
    v2 = a["grupos"]["solo_v2"]
    assert (v2["n"], v2["sin_entrenar_v1"]) == (2, 1)
    assert v2["sin_entrenar_pct"] == pytest.approx(50.0)
    assert dict(v2["nivel_otra"]) == {"COMPRA": 1, "NEUTRAL": 1}
    assert v2["dif_score"] == pytest.approx((8 + 30) / 2)
    assert v2["prob_v2"] == pytest.approx(0.75)
    assert v2["sectores"] == [("Health", 2)]
    assert dict(a["grupos"]["solo_v1"]["nivel_otra"]) == {"COMPRA": 1}
    assert a["grupos"]["ambas"]["nivel_otra"] == []


# ── Operaciones, oportunidades y cartera ──────────────────────────────────────

def op(ticker, e, s, pnl=None, pct=None, motivo=None):
    return {"ticker": ticker, "f_entrada": d(e), "f_salida": None if s is None else d(s),
            "pnl": pnl, "pnl_pct": pct, "motivo_salida": motivo}


def test_operaciones_compartidas_motivos_permanencia_y_split_fix():
    ruedas = [d(i) for i in range(11)]
    v1 = [op("AAPL", 0, 3, 10, 1.0, "SCORE_DEGRADADO_COMPRA_67"),
          op("MSFT", 4, 6, -5, -0.5, "STOP_LOSS"),
          op("KLAC", 1, 2, -900, -9.0, "STOP_LOSS_SPLIT_FIX")]
    v2 = [op("AAPL", 3, None),                       # abierta: se solapa hasta hoy
          op("MSFT", 1, 2, 3, 0.3, "TAKE_PROFIT")]   # antes de la del v1: no se solapa
    r = fc.operaciones_comparadas(v1, v2, ruedas, hoy=d(10))
    o1, o2 = r["v1"], r["v2"]
    assert (o1["n"], o1["abiertas"], o1["cerradas"], o1["contrafactuales"]) == (3, 0, 2, 1)
    assert (o1["compartidas"], o1["exclusivas"]) == (1, 2)
    assert dict(o1["motivos"]) == {"SCORE_DEGRADADO": 1, "STOP_LOSS": 1}
    assert o1["ruedas_media"] == pytest.approx(2.5)
    assert o1["trade"]["n"] == 2
    assert o1["tickers_exclusivos"] == ["KLAC", "MSFT"]
    assert (o2["n"], o2["abiertas"], o2["cerradas"], o2["compartidas"]) == (2, 1, 1, 1)
    assert r["expectancy_v2_vs_v1"]["veredicto"] == fc.INSUFICIENTE


def test_oportunidades_afuera_contra_adentro():
    cands = []
    for i in range(6):
        for k in range(2):
            a = alterna(i + k, 0.005)
            cands.append({"version": "v2", "ticker": f"A{i}{k}", "precio_fecha": d(i),
                          "entro": True, "exc_5": -0.02 + a})
            cands.append({"version": "v2", "ticker": f"F{i}{k}", "precio_fecha": d(i),
                          "entro": False, "exc_5": 0.03 - a})
    r = fc.resumen_oportunidades(cands, horizontes=(5,))
    h = r["v2"]["horizontes"][5]
    assert h["afuera_vs_adentro"]["veredicto"] == fc.AFUERA_MEJOR
    assert h["afuera_vs_adentro"]["diferencia"] == pytest.approx(5.0)
    assert r["v2"]["entraron"] == 12
    assert r["v1"]["horizontes"][5]["afuera_vs_adentro"]["veredicto"] == fc.INSUFICIENTE


def test_cartera_pareada_en_puntos_por_mes():
    def series(n):
        e1 = e2 = 100_000.0
        s1, s2 = [(d(0), e1)], [(d(0), e2)]
        for i in range(1, n + 1):
            r1 = alterna(i, 0.003)
            e1 *= 1 + r1
            e2 *= 1 + r1 + 0.001 + alterna(i, 0.0002)
            s1.append((d(i), e1))
            s2.append((d(i), e2))
        return s1, s2

    s1, s2 = series(24)
    r = fc.cartera_pareada(s1, s2)
    assert r["n"] == 24
    assert r["media"] == pytest.approx(2.1)
    assert r["veredicto"] == fc.A_FAVOR_V2
    assert r["retorno_v2_pct"] > r["retorno_v1_pct"]

    corto = fc.cartera_pareada(*series(10))
    assert corto["veredicto"] == fc.INSUFICIENTE and corto["media"] is None


# ── Texto ─────────────────────────────────────────────────────────────────────

def test_tablas_en_ascii_y_sin_numero_cuando_es_insuficiente():
    filas, closes, ruedas = universo_con_exclusivas(n_ruedas=1, por_rueda=3)
    res = fc.comparar(filas, closes, ruedas, entrenados_v1=set(), horizontes=(5, 20),
                      operaciones={"v1": [], "v2": []}, candidatos=[],
                      equity={"v1": [], "v2": []}, hoy=ruedas[-1])
    ts = fc.tablas(res)
    assert [t["clave"] for t in ts] == ["solapamiento", "senales", "atribucion",
                                        "operaciones", "oportunidades", "cartera"]
    for t in ts:
        for texto in [t["titulo"], t["nota"], *t["columnas"], *(c for f in t["filas"] for c in f)]:
            assert isinstance(texto, str)
            texto.encode("ascii")
            if texto.startswith(fc.INSUFICIENTE):
                assert "[" not in texto
    senales = next(t for t in ts if t["clave"] == "senales")
    assert senales["filas"][-1][-1].startswith(fc.INSUFICIENTE)


def test_tablas_con_muestra_publican_el_veredicto():
    filas, closes, ruedas = universo_con_exclusivas(n_ruedas=6, por_rueda=2)
    res = fc.comparar(filas, closes, ruedas, entrenados_v1=set(), horizontes=(5,))
    senales = next(t for t in fc.tablas(res) if t["clave"] == "senales")
    assert senales["filas"][-1][-1].endswith(fc.A_FAVOR_V2)
    assert senales["filas"][-1][-1].startswith("+10.00 [")
