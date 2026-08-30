"""
test_acciones_series.py -- tests del modulo PURO src/utils/acciones_series.py.

Casos SINTETICOS: no tocan la DB ni la red. Los numeros de los casos "reales"
salen de mediciones sobre el cache de SEC y yahooquery, documentadas en
docs/fuentes_fundamentales.md.

Lo que protegen: que NUNCA se mezclen dos bases de split en una misma serie.
Un conteo pre-split pegado a un precio corregido da un market cap dividido por
el factor del split, y de ahi un PER con un orden de magnitud de error.
"""

import pytest

from src.utils import acciones_series as A


# --------------------------------------------------------------- helpers --
def _p(fecha, shares, periodo="q"):
    return {"fecha": fecha, "shares": float(shares), "periodo": periodo}


# KLAC: Yahoo re-expresa a base actual (1.320M para 2025); SEC portada queda en
# la base de su momento (132M pre-split 10:1).
YAHOO_KLAC = [_p("2023-06-30", 1_367_500_000, "a"),
              _p("2024-06-30", 1_344_200_000, "a"),
              _p("2025-06-30", 1_320_200_000, "a"),
              _p("2026-06-30", 1_307_000_000, "q")]
SEC_KLAC = [_p("2023-07-21", 136_750_000), _p("2024-07-19", 134_420_000),
            _p("2025-07-21", 131_961_370), _p("2026-08-03", 1_306_546_783)]

# Un ticker sin splits: las dos fuentes coinciden.
YAHOO_LIMPIO = [_p("2023-12-31", 1_000_000_000, "a"),
                _p("2024-12-31", 980_000_000, "a"),
                _p("2025-12-31", 960_000_000, "a")]
SEC_LIMPIO = [_p("2021-04-20", 1_060_000_000), _p("2022-04-20", 1_030_000_000),
              _p("2023-04-20", 1_010_000_000), _p("2024-01-20", 998_000_000),
              _p("2025-01-20", 978_000_000)]


# ------------------------------------------------------------- validacion --
def test_validar_detecta_base_distinta():
    """EL caso KLAC: Yahoo 1.320M contra SEC 132M para el mismo trimestre."""
    v = A.validar_base(YAHOO_KLAC, SEC_KLAC)
    assert v["ok"] is False
    assert v["ratio_max"] > 9


def test_validar_acepta_misma_base():
    v = A.validar_base(YAHOO_LIMPIO, SEC_LIMPIO)
    assert v["ok"] is True
    assert v["n_pares"] >= 2


def test_validar_sin_solapamiento_no_es_ok():
    """No poder afirmar nada no es lo mismo que estar bien."""
    v = A.validar_base([_p("2026-01-01", 100)], [_p("2019-01-01", 100)])
    assert v["ok"] is False
    assert v["motivo"] == "sin solapamiento"


def test_validar_tolera_la_deriva_por_recompras():
    """Las fechas difieren semanas y entre medio hay recompras reales."""
    y = [_p("2025-12-31", 1_000_000_000, "a")]
    s = [_p("2026-01-25", 970_000_000)]        # -3%
    assert A.validar_base(y, s)["ok"] is True


def test_validar_no_tolera_el_split_mas_chico():
    """1.5:1 es el split mas chico del universo; tiene que quedar afuera."""
    y = [_p("2025-12-31", 1_500_000_000, "a")]
    s = [_p("2026-01-25", 1_000_000_000)]
    assert A.validar_base(y, s)["ok"] is False


# ----------------------------------------------------------- sin_saltos --
def test_sin_saltos_detecta_split():
    ok, motivo = A.sin_saltos([_p("2026-04-27", 130_627_521),
                               _p("2026-08-03", 1_306_546_783)])
    assert ok is False and "x10" in motivo


def test_sin_saltos_ignora_deriva_normal():
    ok, _ = A.sin_saltos([_p("2024-01-01", 1_000_000_000),
                          _p("2025-01-01", 950_000_000),
                          _p("2026-01-01", 900_000_000)])
    assert ok is True


# ----------------------------------------------------------- construir --
def test_no_extiende_cuando_las_bases_diferen():
    """KLAC: se queda con Yahoo y dice por que. NO elige una fuente."""
    serie, diag = A.construir(YAHOO_KLAC, SEC_KLAC, desde="2021-01-01")
    assert diag["extendido"] is False
    assert {p["fuente"] for p in serie} == {"yahooquery"}
    assert "ratio yahoo/sec" in diag["motivo"]


def test_extiende_cuando_la_base_esta_validada():
    serie, diag = A.construir(YAHOO_LIMPIO, SEC_LIMPIO, desde="2021-01-01")
    assert diag["extendido"] is True
    assert diag["desde_efectivo"] == "2021-04-20"
    assert diag["n_sec_usados"] == 3           # 2021, 2022, 2023-04
    assert serie[0]["fuente"] == "sec_portada"
    assert serie[-1]["fuente"] == "yahooquery"


def test_yahoo_manda_donde_llega():
    """SEC nunca pisa a Yahoo: solo extiende hacia atras."""
    serie, _ = A.construir(YAHOO_LIMPIO, SEC_LIMPIO, desde="2021-01-01")
    posteriores = [p for p in serie if p["fecha"] >= "2023-12-31"]
    assert all(p["fuente"] == "yahooquery" for p in posteriores)


def test_respeta_el_desde():
    serie, diag = A.construir(YAHOO_LIMPIO, SEC_LIMPIO, desde="2023-01-01")
    assert all(p["fecha"] >= "2023-01-01" for p in serie)
    assert diag["n_sec_usados"] == 1           # solo 2023-04


def test_un_split_entre_el_ultimo_sec_y_el_arranque_de_yahoo_frena_la_extension():
    """
    El hueco peligroso: si el split cae justo en el tramo sin datos, mirar solo
    los puntos previos no lo ve. Por eso el chequeo incluye el primer punto
    posterior al corte.
    """
    yahoo = [_p("2024-12-31", 2_000_000_000, "a"),
             _p("2025-12-31", 1_980_000_000, "a")]
    sec = [_p("2022-06-30", 990_000_000),      # pre-split
           _p("2023-06-30", 985_000_000),      # pre-split
           _p("2025-01-15", 1_990_000_000)]    # post-split, solapa con yahoo
    serie, diag = A.construir(yahoo, sec, desde="2021-01-01")
    assert diag["extendido"] is False
    assert {p["fuente"] for p in serie} == {"yahooquery"}


def test_sin_yahoo_no_hay_serie():
    """SEC sola no se usa: no hay contra que validar su base."""
    serie, diag = A.construir([], SEC_LIMPIO, desde="2021-01-01")
    assert serie == []
    assert diag["motivo"] == "sin datos de yahoo"


def test_serie_ordenada_y_sin_duplicados():
    serie, _ = A.construir(YAHOO_LIMPIO, SEC_LIMPIO, desde="2021-01-01")
    fechas = [p["fecha"] for p in serie]
    assert fechas == sorted(fechas)
    assert len(fechas) == len(set(fechas))


# ---------------------------------------------------------------- as-of --
def test_asof_es_escalon_no_interpola():
    """
    Decision explicita: entre dos puntos el conteo NO se mueve. Interpolar
    bajaria el error medio pero inventaria dato y borraria las
    discontinuidades reales, que son la cola del error (p99 11,35%).
    """
    serie, _ = A.construir(YAHOO_LIMPIO, SEC_LIMPIO, desde="2021-01-01")
    # Entre el punto de Yahoo de 2023-12-31 y el de 2024-12-31 no se mueve,
    # aunque SEC tenga un punto intermedio: Yahoo manda donde llega.
    a = A.asof(serie, "2024-06-15")["shares"]
    b = A.asof(serie, "2024-12-30")["shares"]
    assert a == b == 1_000_000_000


def test_asof_none_antes_del_primer_punto():
    serie, _ = A.construir(YAHOO_LIMPIO, SEC_LIMPIO, desde="2021-01-01")
    assert A.asof(serie, "2020-01-01") is None


def test_recorrer_asof_equivale_a_asof():
    serie, _ = A.construir(YAHOO_LIMPIO, SEC_LIMPIO, desde="2021-01-01")
    fechas = ["2020-01-01", "2021-04-20", "2022-06-01", "2025-12-31",
              "2026-08-27"]
    assert [p for _, p in A.recorrer_asof(serie, fechas)] == \
           [A.asof(serie, f) for f in fechas]


# --------------------------------------------------------------- aparear --
def test_aparear_toma_el_mas_cercano():
    a = [_p("2025-06-30", 100)]
    b = [_p("2025-05-01", 90), _p("2025-07-21", 99)]
    pares = A.aparear(a, b)
    assert len(pares) == 1
    assert pares[0][1]["shares"] == 99


def test_aparear_descarta_lo_lejano():
    a = [_p("2025-06-30", 100)]
    b = [_p("2025-01-01", 90)]
    assert A.aparear(a, b) == []


# ------------------------------------------------------- empalme continuo --
def test_empalme_con_salto_de_split_se_rechaza():
    """
    EL caso WMT, reproducido con su forma real (split 3:1 de 2024).

    Yahoo arranca en 2023-01 ya en la base NUEVA; la serie SEC esta en la
    vieja. Las dos guardas previas aprueban igual:
      - validar_base: los unicos pares que caen dentro de los 60 dias son de
        2025, o sea POST split, y ahi las dos fuentes coinciden.
      - sin_saltos: el tramo SEC que se agrega es todo pre-split y no tiene
        ningun salto interno.
    El escalon x2,99 esta justo en la juntura, que es donde nadie miraba.
    """
    yahoo = [{"fecha": "2023-01-31", "shares": 8_100_000_000},
             {"fecha": "2025-04-30", "shares": 7_986_000_000}]
    sec = [{"fecha": "2022-07-31", "shares": 2_745_000_000},
           {"fecha": "2022-10-31", "shares": 2_711_000_000},
           {"fecha": "2023-04-30", "shares": 2_704_000_000},
           {"fecha": "2025-04-30", "shares": 7_986_000_000}]
    serie, diag = A.construir(yahoo, sec, desde="2021-01-01")
    assert diag["validacion"]["ok"] is True, "la validacion vieja aprueba: ese es el punto"
    assert diag["extendido"] is False
    assert "empalme" in diag["motivo"]
    assert {p["fuente"] for p in serie} == {"yahooquery"}


def test_empalme_normal_se_acepta():
    """Un trimestre de recompras mueve unidades porcentuales, no multiplos."""
    yahoo = [{"fecha": "2023-01-31", "shares": 990_000_000},
             {"fecha": "2024-01-31", "shares": 975_000_000}]
    sec = [{"fecha": "2022-04-30", "shares": 1_010_000_000},
           {"fecha": "2022-10-31", "shares": 1_000_000_000},
           {"fecha": "2024-01-31", "shares": 975_000_000}]
    serie, diag = A.construir(yahoo, sec, desde="2021-01-01")
    assert diag["extendido"] is True
    assert diag["n_sec_usados"] == 2
    assert serie[0]["fecha"] == "2022-04-30"
