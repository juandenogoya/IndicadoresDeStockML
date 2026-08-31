"""
test_fundamentales_ttm.py -- tests del modulo PURO src/utils/fundamentales_ttm.py.

Casos SINTETICOS: no tocan la DB ni la red. Cada test construye la serie
trimestral minima que reproduce un problema concreto.

Los dos errores que estos tests existen para impedir:
  1. sumar 4 trimestres que NO abarcan un ano (hueco en la serie) -- el
     resultado es plausible y esta mal, en silencio;
  2. indexar la serie por period_end en vez de filed_primero -- mete lookahead
     en toda serie historica de multiplos.
"""

import pytest

from src.utils import fundamentales_ttm as ttm


# --------------------------------------------------------------- helpers --
# Cierres trimestrales calendario y su filed tipico (~35 dias despues).
CIERRES = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
           "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
FILED = ["2024-05-05", "2024-08-05", "2024-11-05", "2025-02-05",
         "2025-05-05", "2025-08-05", "2025-11-05", "2026-02-05"]


def _q(period_end, filed=None, **valores):
    fila = {"period_end": period_end, "filed_primero": filed,
            "filed_ultimo": filed, "fiscal_year": None, "fiscal_quarter": None}
    fila.update(valores)
    return fila


def _serie(n=8, **por_concepto):
    """
    Serie de n trimestres consecutivos. por_concepto: {concepto: [v1..vn]}.
    """
    filas = []
    for i in range(n):
        vals = {c: v[i] for c, v in por_concepto.items()}
        filas.append(_q(CIERRES[i], FILED[i], **vals))
    return filas


# ------------------------------------------------------------------- TTM --
def test_ttm_suma_cuatro_trimestres():
    serie = _serie(8, revenue=[10, 20, 30, 40, 50, 60, 70, 80])
    out = ttm.serie_ttm(serie)
    assert out[3]["revenue_ttm"] == 100    # 10+20+30+40
    assert out[7]["revenue_ttm"] == 260    # 50+60+70+80


def test_primeros_tres_periodos_sin_ttm_pero_con_fila():
    """La fila existe igual: hay que poder distinguir 'sin dato' de 'sin fila'."""
    out = ttm.serie_ttm(_serie(8, revenue=[10] * 8))
    assert len(out) == 8
    for i in range(3):
        assert out[i]["revenue_ttm"] is None
        assert out[i]["ventana_ok"] is False
    assert out[3]["ventana_ok"] is True


def test_hueco_en_la_serie_invalida_la_ventana():
    """
    EL test central. Faltando 2024-09-30, los 4 ultimos periodos consecutivos
    abarcan 5 trimestres. rolling(4).sum() sumaria igual y daria un numero
    creible. Aca la ventana se rechaza.
    """
    filas = [_q("2024-03-31", "2024-05-05", revenue=10),
             _q("2024-06-30", "2024-08-05", revenue=20),
             # falta 2024-09-30
             _q("2024-12-31", "2025-02-05", revenue=40),
             _q("2025-03-31", "2025-05-05", revenue=50)]
    out = ttm.serie_ttm(filas)
    assert out[-1]["ventana_ok"] is False
    assert out[-1]["revenue_ttm"] is None


def test_la_serie_se_recupera_despues_del_hueco():
    """El hueco invalida las ventanas que lo tocan, no la serie entera."""
    filas = ([_q("2023-12-31", "2024-02-05", revenue=5)]
             + [_q(c, f, revenue=10) for c, f in zip(CIERRES[:2], FILED[:2])]
             # falta 2024-09-30
             + [_q(c, f, revenue=10) for c, f in zip(CIERRES[3:], FILED[3:])])
    out = ttm.serie_ttm(filas)
    ok = [f["ventana_ok"] for f in out]
    assert ok[-1] is True          # 4 trimestres limpios de 2025
    assert any(v is False for v in ok[3:6])


def test_concepto_faltante_en_un_trimestre_anula_solo_ese_ttm():
    serie = _serie(4, revenue=[10, 20, 30, 40], cfo=[1, 2, None, 4])
    out = ttm.serie_ttm(serie)
    assert out[3]["revenue_ttm"] == 100
    assert out[3]["cfo_ttm"] is None       # 3 de 4 no es un ano


def test_eps_se_suma():
    """
    Contraintuitivo pero correcto: el EPS acumulado publicado coincide con la
    suma de los trimestrales. Tratarlo como promedio ponderado da absurdos.
    """
    serie = _serie(4, eps_diluted=[1.0, 1.5, 2.0, 1.85])
    out = ttm.serie_ttm(serie)
    assert out[3]["eps_diluted_ttm"] == pytest.approx(6.35)


def test_shares_no_se_suman():
    serie = _serie(4, shares_diluted=[100, 101, 102, 103])
    out = ttm.serie_ttm(serie)
    assert "shares_diluted_ttm" not in out[3]
    assert out[3]["shares_diluted"] == 103      # el del ultimo Q


def test_instantes_toman_la_foto_del_ultimo_trimestre():
    serie = _serie(4, assets=[900, 950, 1000, 1100], equity=[100, 110, 120, 130])
    out = ttm.serie_ttm(serie)
    assert out[3]["assets"] == 1100
    assert out[3]["equity"] == 130
    assert "assets_ttm" not in out[3]


def test_ventana_desde_marca_el_primer_trimestre():
    out = ttm.serie_ttm(_serie(8, revenue=[10] * 8))
    assert out[3]["ventana_desde"] == "2024-03-31"
    assert out[7]["ventana_desde"] == "2025-03-31"


def test_entrada_desordenada_se_ordena():
    serie = _serie(4, revenue=[10, 20, 30, 40])
    out = ttm.serie_ttm(list(reversed(serie)))
    assert [f["period_end"] for f in out] == CIERRES[:4]
    assert out[3]["revenue_ttm"] == 100


def test_ejercicio_fiscal_no_calendario():
    """Cierres a fin de septiembre (tipo AAPL): la ventana tiene que validar."""
    cierres = ["2024-12-28", "2025-03-29", "2025-06-28", "2025-09-27"]
    filas = [_q(c, None, revenue=10) for c in cierres]
    for f in filas:
        f["filed_primero"] = "2026-01-01"
    out = ttm.serie_ttm(filas)
    assert out[3]["ventana_ok"] is True
    assert out[3]["revenue_ttm"] == 40


def test_serie_vacia():
    assert ttm.serie_ttm([]) == []


def test_periodos_sin_period_end_se_descartan():
    filas = [_q(None, "2025-01-01", revenue=99)] + _serie(4, revenue=[10] * 4)
    out = ttm.serie_ttm(filas)
    assert len(out) == 4


# ------------------------------------------------------------ derivados --
def test_ebitda_es_ebit_mas_da():
    d = ttm.derivados({"operating_income_ttm": 1000.0, "d_and_a_ttm": 250.0})
    assert d["ebitda_ttm"] == 1250.0


def test_ebitda_none_si_falta_da():
    d = ttm.derivados({"operating_income_ttm": 1000.0})
    assert d["ebitda_ttm"] is None


def test_fcf_resta_capex_positivo():
    """capex viene POSITIVO en XBRL (es un 'Payments...')."""
    d = ttm.derivados({"cfo_ttm": 1000.0, "capex_ttm": 300.0})
    assert d["fcf_ttm"] == 700.0


def test_net_debt_none_si_no_hay_ninguna_deuda_tageada():
    """Ausencia en XBRL no es prueba de cero. Convertir 'no se' en 0 infla el EV."""
    d = ttm.derivados({"cash": 500.0})
    assert d["net_debt"] is None


def test_net_debt_con_una_sola_deuda_cuenta_la_otra_como_cero():
    """Sin contexto de la serie, la deuda ausente vale cero."""
    d = ttm.derivados({"debt_long": 1000.0, "cash": 300.0})
    assert d["net_debt"] == 700.0


def test_no_se_asume_cero_la_deuda_que_la_empresa_si_tagea():
    """
    "No esta en este trimestre" y "esta empresa no tiene esa deuda" se ven
    iguales en una fila sola, y no son lo mismo. La API de companyfacts
    descarta los hechos dimensionados, y varias empresas grandes pasaron a
    declarar su deuda de largo plazo solo con dimensiones: Verizon tiene 8
    hechos de LongTermDebt y ninguno desde 2025. Con la regla vieja el
    net_debt de AT&T salia NEGATIVO, como si tuviera caja neta, y eso se
    propagaba al EV sin que nada avisara.
    """
    d = ttm.derivados({"debt_short": 200.0, "cash": 300.0},
                      deudas_conocidas={"debt_short", "debt_long"})
    assert d["net_debt"] is None


def test_la_empresa_sin_deuda_larga_conserva_su_net_debt():
    """
    Si NUNCA tagea el concepto, la ausencia si significa cero y el EV se
    calcula igual. Sin esta mitad de la regla, cualquier empresa sin deuda de
    largo plazo perderia su EV.
    """
    d = ttm.derivados({"debt_short": 200.0, "cash": 300.0},
                      deudas_conocidas={"debt_short"})
    assert d["net_debt"] == -100.0


def test_enriquecer_deduce_las_deudas_conocidas_de_la_serie_entera():
    """
    La deteccion mira TODA la serie: el trimestre viejo que si tagea la deuda
    larga es lo que delata que el nuevo la tiene y no la declaro.

    Lo que se hace con esa deteccion cambio: antes el trimestre incompleto
    perdia su net_debt; ahora ARRASTRA el ultimo valor conocido y declara la
    edad. La regla que NO cambio es la que importa: nunca se asume cero.
    """
    filas = [_q("2025-03-31", debt_short=200.0, debt_long=5000.0, cash=300.0),
             _q("2025-06-30", debt_short=200.0, cash=300.0)]
    out = ttm.enriquecer(filas)
    assert out[0]["net_debt"] == 4900.0
    assert out[0]["net_debt_q"] == 0
    assert out[1]["net_debt"] == 4900.0, "arrastra los 5000, no asume 0"
    assert out[1]["net_debt_q"] == 1


def test_el_arrastre_vence_y_no_se_estira_para_siempre():
    """
    MAX_ARRASTRE_Q trimestres despues, el valor deja de ser una estimacion y
    pasa a ser una suposicion. Ahi vuelve el hueco.

    El caso real es DE (17 trimestres sin declarar su deuda larga) y ORCL (16):
    con el tope quedan afuera, que es lo correcto.
    """
    filas = [_q("2021-03-31", debt_short=200.0, debt_long=5000.0, cash=300.0)]
    for i, fecha in enumerate(("2021-06-30", "2021-09-30", "2021-12-31",
                               "2022-03-31", "2022-06-30", "2022-09-30")):
        filas.append(_q(fecha, debt_short=200.0, cash=300.0))
    out = ttm.enriquecer(filas)

    for fila in out[1:1 + ttm.MAX_ARRASTRE_Q]:
        assert fila["net_debt"] == 4900.0
    for fila in out[1 + ttm.MAX_ARRASTRE_Q:]:
        assert fila["net_debt"] is None, "pasado el tope, hueco visible"
        assert fila["net_debt_q"] is None


def test_el_arrastre_no_va_hacia_atras():
    """
    Solo hacia adelante. Meterle a un trimestre viejo una deuda declarada
    despues seria lookahead -- el mismo error que el modulo evita en el as-of.
    """
    filas = [_q("2025-03-31", debt_short=200.0, cash=300.0),
             _q("2025-06-30", debt_short=200.0, debt_long=5000.0, cash=300.0)]
    out = ttm.enriquecer(filas)
    assert out[0]["net_debt"] is None, "en marzo esa deuda todavia no existia"
    assert out[1]["net_debt"] == 4900.0


def test_el_arrastre_no_depende_del_orden_de_entrada():
    """
    enriquecer ordena por period_end antes de arrastrar. Si dependiera del
    orden en que llegan las filas, el resultado cambiaria segun la query.
    """
    a = _q("2025-03-31", debt_short=200.0, debt_long=5000.0, cash=300.0)
    b = _q("2025-06-30", debt_short=200.0, cash=300.0)
    directo = ttm.enriquecer([a, b])
    invertido = ttm.enriquecer([b, a])
    assert directo[1]["net_debt"] == 4900.0
    assert invertido[0]["net_debt"] == 4900.0, "b sigue siendo el que arrastra"
    assert invertido[1]["net_debt"] == 4900.0


def test_sin_valor_previo_no_se_inventa_nada():
    """
    Arrastrar exige tener QUE arrastrar. Una serie que nunca declaro la deuda
    larga en un periodo anterior no produce net_debt en el primero.
    """
    filas = [_q("2025-03-31", debt_short=200.0, cash=300.0),
             _q("2025-06-30", debt_short=200.0, debt_long=5000.0, cash=300.0)]
    out = ttm.enriquecer(filas)
    assert out[0]["net_debt"] is None


def test_net_debt_none_sin_cash():
    d = ttm.derivados({"debt_short": 100.0, "debt_long": 900.0})
    assert d["net_debt"] is None


def test_net_debt_puede_ser_negativo():
    """Caja neta: valido, resta del EV."""
    d = ttm.derivados({"debt_long": 100.0, "cash": 900.0})
    assert d["net_debt"] == -800.0


def test_bvps_usa_shares_out():
    d = ttm.derivados({"equity": 1000.0, "shares_out": 100.0,
                       "shares_diluted": 110.0})
    assert d["book_value_per_share"] == 10.0
    assert d["shares_fuente"] == "shares_out"


def test_shares_cae_a_diluted_si_falta_out():
    d = ttm.derivados({"equity": 1000.0, "shares_diluted": 200.0})
    assert d["shares"] == 200.0
    assert d["shares_fuente"] == "shares_diluted"
    assert d["book_value_per_share"] == 5.0


def test_bvps_none_sin_acciones():
    d = ttm.derivados({"equity": 1000.0})
    assert d["book_value_per_share"] is None


def test_eps_ttm_cae_a_basic():
    d = ttm.derivados({"eps_basic_ttm": 4.2})
    assert d["eps_ttm"] == 4.2
    d2 = ttm.derivados({"eps_diluted_ttm": 4.0, "eps_basic_ttm": 4.2})
    assert d2["eps_ttm"] == 4.0


def test_enriquecer_no_muta_la_entrada():
    filas = ttm.serie_ttm(_serie(4, operating_income=[100] * 4,
                                 d_and_a=[10] * 4))
    ricas = ttm.enriquecer(filas)
    assert "ebitda_ttm" not in filas[3]
    assert ricas[3]["ebitda_ttm"] == 440.0


# ---------------------------------------------------------------- as-of --
def test_asof_devuelve_el_ultimo_publicado():
    idx = ttm.indice_asof(_serie(4, revenue=[10, 20, 30, 40]))
    assert ttm.asof(idx, "2024-08-20")["period_end"] == "2024-06-30"
    assert ttm.asof(idx, "2024-11-05")["period_end"] == "2024-09-30"   # mismo dia


def test_asof_none_antes_de_la_primera_publicacion():
    idx = ttm.indice_asof(_serie(4, revenue=[10] * 4))
    assert ttm.asof(idx, "2024-01-15") is None


def test_asof_usa_filed_no_period_end():
    """
    EL otro test central. El 2024-04-15 el trimestre cerrado el 2024-03-31
    todavia NO se habia publicado (sale el 5/5). Indexar por period_end lo
    daria por disponible tres semanas antes: lookahead.
    """
    idx = ttm.indice_asof([_q("2023-12-31", "2024-02-05", revenue=5),
                           _q("2024-03-31", "2024-05-05", revenue=10)])
    fila = ttm.asof(idx, "2024-04-15")
    assert fila["period_end"] == "2023-12-31"


def test_asof_descarta_filas_sin_filed_primero():
    idx = ttm.indice_asof([_q("2024-03-31", None, revenue=10),
                           _q("2024-06-30", "2024-08-05", revenue=20)])
    assert ttm.asof(idx, "2024-07-01") is None
    assert ttm.asof(idx, "2024-09-01")["period_end"] == "2024-06-30"


def test_asof_empate_mismo_dia_gana_el_periodo_mayor():
    """El 10-K publica el Q4 y el FY el mismo dia."""
    idx = ttm.indice_asof([_q("2024-09-30", "2025-02-05", revenue=30),
                           _q("2024-12-31", "2025-02-05", revenue=40)])
    assert ttm.asof(idx, "2025-02-05")["period_end"] == "2024-12-31"


def test_recorrer_asof_equivale_a_asof():
    filas = _serie(8, revenue=[10] * 8)
    idx = ttm.indice_asof(filas)
    fechas = ["2024-01-01", "2024-05-05", "2024-06-01", "2024-11-04",
              "2024-11-05", "2026-06-01"]
    esperado = [ttm.asof(idx, f) for f in fechas]
    obtenido = [fila for _, fila in ttm.recorrer_asof(filas, fechas)]
    assert obtenido == esperado


def test_recorrer_asof_sin_filas():
    assert ttm.recorrer_asof([], ["2025-01-01"]) == [("2025-01-01", None)]
