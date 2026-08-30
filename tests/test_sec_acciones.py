"""
test_sec_acciones.py -- tests del modulo PURO src/utils/sec_acciones.py.

Casos SINTETICOS: no tocan la DB ni la red. Los numeros de los casos "reales"
(KLAC, BA) salen de los hechos observados en el cache de companyfacts y estan
documentados en docs/fuentes_fundamentales.md.

Lo que estos tests protegen: que la serie de acciones sea POINT-IN-TIME. Si se
cuela un conteo re-expresado por un split, el market cap historico queda
multiplicado por el factor del split y todos los multiplos con el.
"""

import pytest

from src.utils import sec_acciones as SA


# --------------------------------------------------------------- helpers --
def _cf(hechos, taxonomia="dei", tag=SA.TAG_PORTADA, unidad="shares"):
    return {"facts": {taxonomia: {tag: {"units": {unidad: hechos}}}}}


def _h(end, val, filed, accn="a-1", form="10-Q"):
    return {"end": end, "val": val, "filed": filed, "accn": accn, "form": form}


# ---------------------------------------------------------------- serie --
def test_una_entrada_por_portada_ordenada():
    cf = _cf([_h("2026-04-27", 130_627_521, "2026-04-30"),
              _h("2026-01-26", 131_076_611, "2026-01-30"),
              _h("2026-08-03", 1_306_546_783, "2026-08-06")])
    s = SA.serie_portada(cf)
    assert [p["fecha"] for p in s] == ["2026-01-26", "2026-04-27", "2026-08-03"]
    assert s[-1]["shares"] == 1_306_546_783


def test_split_real_queda_en_la_serie():
    """
    EL caso KLAC. El salto 10x del 3/8/2026 es un split REAL y tiene que estar:
    el precio salta igual ese dia, asi que el market cap queda bien. Lo que no
    puede pasar es que ese valor aparezca en una fecha ANTERIOR.
    """
    cf = _cf([_h("2026-04-27", 130_627_521, "2026-04-30"),
              _h("2026-08-03", 1_306_546_783, "2026-08-06")])
    s = SA.serie_portada(cf)
    assert SA.acciones_asof(s, "2026-06-15")["shares"] == 130_627_521
    assert SA.acciones_asof(s, "2026-08-20")["shares"] == 1_306_546_783


def test_redeclaracion_de_la_misma_fecha_gana_el_filed_mas_temprano():
    """
    Al reves que el normalizador, y a proposito: aca interesa lo que estaba
    publicado entonces, no el valor vigente hoy. Una re-declaracion posterior
    de la misma fecha de portada ya viene en otra base de split.
    """
    cf = _cf([_h("2025-06-30", 132_023_000, "2025-08-08"),
              _h("2025-06-30", 1_320_227_000, "2026-08-06")])
    s = SA.serie_portada(cf)
    assert len(s) == 1
    assert s[0]["shares"] == 132_023_000
    assert s[0]["filed"] == "2025-08-08"


def test_solo_lee_dei_no_us_gaap():
    """
    us-gaap:CommonStockSharesOutstanding mide otra cosa (acciones EMITIDAS: en
    BA da 1.012.261.159 constante contra 754-790M reales) y ademas se
    re-expresa. No entra ni como respaldo.
    """
    cf = _cf([_h("2025-12-31", 1_012_261_159, "2026-01-30")],
             taxonomia="us-gaap", tag="CommonStockSharesOutstanding")
    assert SA.serie_portada(cf) == []


def test_sin_el_tag_devuelve_vacio():
    assert SA.serie_portada({"facts": {"dei": {}}}) == []
    assert SA.serie_portada({}) == []
    assert SA.serie_portada(None) == []


def test_descarta_valores_con_error_de_unidad():
    """Conteos reportados en miles/millones en vez de unidades (AA, AXP, KO)."""
    cf = _cf([_h("2024-03-31", 179, "2024-04-05"),          # miles de millones?
              _h("2024-06-30", 179_559_688, "2024-07-05"),
              _h("2024-09-30", 1e14, "2024-10-05")])
    s = SA.serie_portada(cf)
    assert [p["shares"] for p in s] == [179_559_688]


def test_descarta_hechos_incompletos():
    cf = _cf([{"end": None, "val": 100_000_000, "filed": "2024-01-01"},
              {"end": "2024-03-31", "val": None, "filed": "2024-04-01"},
              {"end": "2024-06-30", "val": 100_000_000, "filed": None},
              _h("2024-09-30", 100_000_000, "2024-10-01")])
    s = SA.serie_portada(cf)
    assert [p["fecha"] for p in s] == ["2024-09-30"]


def test_desde_recorta():
    cf = _cf([_h("2019-03-31", 100_000_000, "2019-04-05"),
              _h("2024-03-31", 120_000_000, "2024-04-05")])
    s = SA.serie_portada(cf, desde="2020-01-01")
    assert [p["fecha"] for p in s] == ["2024-03-31"]


def test_toma_todas_las_unidades():
    """El tag puede venir bajo 'shares' o bajo otra clave de unidad."""
    cf = {"facts": {"dei": {SA.TAG_PORTADA: {"units": {
        "shares": [_h("2024-03-31", 100_000_000, "2024-04-05")],
        "pure": [_h("2024-06-30", 101_000_000, "2024-07-05")]}}}}}
    assert len(SA.serie_portada(cf)) == 2


# ---------------------------------------------------------------- as-of --
def test_asof_es_escalon():
    s = SA.serie_portada(_cf([_h("2024-04-27", 100_000_000, "2024-05-01"),
                              _h("2024-07-27", 99_000_000, "2024-08-01")]))
    assert SA.acciones_asof(s, "2024-04-27")["shares"] == 100_000_000
    assert SA.acciones_asof(s, "2024-06-15")["shares"] == 100_000_000
    assert SA.acciones_asof(s, "2024-07-26")["shares"] == 100_000_000
    assert SA.acciones_asof(s, "2024-07-27")["shares"] == 99_000_000
    assert SA.acciones_asof(s, "2030-01-01")["shares"] == 99_000_000


def test_asof_none_antes_de_la_primera_portada():
    s = SA.serie_portada(_cf([_h("2024-04-27", 100_000_000, "2024-05-01")]))
    assert SA.acciones_asof(s, "2024-01-15") is None
    assert SA.acciones_asof([], "2024-01-15") is None


def test_asof_acepta_date():
    from datetime import date
    s = SA.serie_portada(_cf([_h("2024-04-27", 100_000_000, "2024-05-01")]))
    assert SA.acciones_asof(s, date(2024, 6, 1))["shares"] == 100_000_000


def test_recorrer_asof_equivale_a_asof():
    s = SA.serie_portada(_cf([_h("2024-04-27", 100_000_000, "2024-05-01"),
                              _h("2024-07-27", 99_000_000, "2024-08-01"),
                              _h("2024-10-27", 98_000_000, "2024-11-01")]))
    fechas = ["2024-01-01", "2024-04-27", "2024-05-15", "2024-07-27",
              "2024-12-31"]
    assert [p for _, p in SA.recorrer_asof(s, fechas)] == \
           [SA.acciones_asof(s, f) for f in fechas]


def test_recorrer_asof_sin_serie():
    assert SA.recorrer_asof([], ["2025-01-01"]) == [("2025-01-01", None)]


# -------------------------------------------------------------- saltos --
def test_saltos_detecta_el_split():
    s = SA.serie_portada(_cf([_h("2026-04-27", 130_627_521, "2026-04-30"),
                              _h("2026-08-03", 1_306_546_783, "2026-08-06")]))
    sal = SA.saltos(s)
    assert len(sal) == 1
    assert sal[0]["ratio"] == pytest.approx(10.0, rel=1e-3)
    assert sal[0]["hasta"] == "2026-08-03"


def test_saltos_ignora_variacion_normal():
    s = SA.serie_portada(_cf([_h("2024-04-27", 100_000_000, "2024-05-01"),
                              _h("2024-07-27", 97_000_000, "2024-08-01")]))
    assert SA.saltos(s) == []


def test_saltos_detecta_split_inverso():
    s = SA.serie_portada(_cf([_h("2024-04-27", 100_000_000, "2024-05-01"),
                              _h("2024-07-27", 10_000_000, "2024-08-01")]))
    sal = SA.saltos(s)
    assert len(sal) == 1
    assert sal[0]["ratio"] == pytest.approx(0.1)


# ------------------------------------------------- invariante de portada --
def test_portada_posterior_a_su_filing_se_descarta():
    """
    Caso real AAL: un 10-Q declara portada 2027-07-17 habiendose presentado el
    2026-07-23 (anio mal tipeado). Sin el filtro esa fecha se vuelve el fin de
    la serie y su conteo queda vigente para siempre.
    """
    cf = _cf([_h("2026-04-27", 660_000_000, "2026-04-30"),
              _h("2027-07-17", 661_969_951, "2026-07-23")])
    s = SA.serie_portada(cf)
    assert [p["fecha"] for p in s] == ["2026-04-27"]


def test_portada_el_mismo_dia_del_filing_se_acepta():
    cf = _cf([_h("2026-04-30", 660_000_000, "2026-04-30")])
    assert len(SA.serie_portada(cf)) == 1


# ------------------------------------------------------------- respaldo --
def _cf_prom(hechos, tag="WeightedAverageNumberOfDilutedSharesOutstanding"):
    return {"facts": {"us-gaap": {tag: {"units": {"shares": hechos}}}}}


def _hq(end, val, filed, start):
    h = _h(end, val, filed)
    h["start"] = start
    return h


def test_respaldo_solo_toma_trimestres_no_acumulados():
    """El promedio ponderado de 9 meses no es el del trimestre."""
    cf = _cf_prom([_hq("2025-03-31", 100_000_000, "2025-05-01", "2025-01-01"),
                   _hq("2025-09-30", 102_000_000, "2025-11-01", "2025-01-01")])
    s = SA.serie_promedio(cf)
    assert [p["fecha"] for p in s] == ["2025-03-31"]


def test_respaldo_toma_el_primer_filed_no_el_reexpresado():
    """
    EL punto del respaldo. CRWD 2025-07-31: 250M en el 10-Q de 2025 y 999M
    re-expresado post-split 4:1 por el 10-Q de 2026. Gana el de 2025, que es el
    que esta en la misma base que el precio de julio 2025.
    """
    cf = _cf_prom([_hq("2025-07-31", 250_000_000, "2025-08-28", "2025-05-01"),
                   _hq("2025-07-31", 999_634_000, "2026-08-27", "2025-05-01")])
    s = SA.serie_promedio(cf)
    assert len(s) == 1
    assert s[0]["shares"] == 250_000_000
    assert s[0]["fuente"] == "promedio_diluido"


def test_respaldo_cae_a_basico_donde_falta_diluido():
    cf = {"facts": {"us-gaap": {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
            _hq("2025-03-31", 100_000_000, "2025-05-01", "2025-01-01")]}},
        "WeightedAverageNumberOfSharesOutstandingBasic": {"units": {"shares": [
            _hq("2025-03-31", 99_000_000, "2025-05-01", "2025-01-01"),
            _hq("2025-06-30", 98_000_000, "2025-08-01", "2025-04-01")]}}}}}
    s = SA.serie_promedio(cf)
    assert {p["fecha"]: p["shares"] for p in s} == {
        "2025-03-31": 100_000_000,   # diluido gana
        "2025-06-30": 98_000_000}    # basico rellena


def test_serie_acciones_prefiere_portada():
    cf = _cf([_h("2026-04-27", 130_000_000, "2026-04-30")])
    cf["facts"]["us-gaap"] = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
            _hq("2026-03-31", 129_000_000, "2026-04-30", "2026-01-01")]}}}
    s = SA.serie_acciones(cf)
    assert [p["fuente"] for p in s] == ["portada"]


def test_serie_acciones_usa_respaldo_sin_portada():
    """
    Ultimo recurso: los filers de clases multiples que ademas no publican el
    conteo del balance (F, MA, META, NKE, UPS...).
    """
    cf = _cf_prom([_hq("2026-03-31", 129_000_000, "2026-04-30", "2026-01-01")])
    s = SA.serie_acciones(cf)
    assert [p["fuente"] for p in s] == ["promedio_diluido"]


def test_serie_acciones_no_mezcla_las_dos_fuentes():
    """Alternar entre portada y promedio meteria escalones que no ocurrieron."""
    cf = _cf([_h("2026-04-27", 130_000_000, "2026-04-30")])
    cf["facts"]["us-gaap"] = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
            _hq("2025-03-31", 200_000_000, "2025-05-01", "2025-01-01")]}}}
    s = SA.serie_acciones(cf)
    assert len({p["fuente"] for p in s}) == 1


# ------------------------------------------------ nivel 2: el balance --
def _cf_bal(hechos):
    return {"facts": {SA.TAXONOMIA_BALANCE:
                      {SA.TAG_BALANCE: {"units": {"shares": hechos}}}}}


def test_balance_se_usa_cuando_no_hay_portada():
    """
    Los 21 filers de clases multiples no tienen portada: companyfacts descarta
    los hechos dimensionales. La mayoria SI publica el conteo consolidado en el
    balance -- medido, ratio 1.0000 contra yahooquery en 77 de 94 tickers.
    """
    cf = _cf_bal([_h("2026-06-30", 12_230_000_000, "2026-07-23")])
    s = SA.serie_acciones(cf)
    assert [p["fuente"] for p in s] == ["balance"]
    assert s[0]["shares"] == 12_230_000_000


def test_la_portada_le_gana_al_balance():
    """
    La portada no se re-expresa ante un split y el balance SI. Donde estan las
    dos, gana la portada -- si se invierte, la serie historica queda en la base
    de hoy y el market cap viejo sale multiplicado por el factor del split.
    """
    cf = _cf([_h("2026-04-27", 130_627_521, "2026-04-30")])
    cf["facts"][SA.TAXONOMIA_BALANCE] = {
        SA.TAG_BALANCE: {"units": {"shares": [
            _h("2026-04-27", 1_306_546_783, "2026-04-30")]}}}
    s = SA.serie_acciones(cf)
    assert [p["fuente"] for p in s] == ["portada"]
    assert s[0]["shares"] == 130_627_521


def test_el_balance_le_gana_al_promedio():
    """
    El balance es un conteo A UNA FECHA, igual que la portada. El promedio
    ponderado es otra magnitud (todo el trimestre) y por eso va ultimo.
    """
    cf = _cf_bal([_h("2026-03-31", 130_000_000, "2026-04-30")])
    cf["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
        "units": {"shares": [
            _hq("2026-03-31", 129_000_000, "2026-04-30", "2026-01-01")]}}
    s = SA.serie_acciones(cf)
    assert [p["fuente"] for p in s] == ["balance"]


def test_serie_acciones_no_mezcla_los_tres_niveles():
    cf = _cf([_h("2026-04-27", 130_000_000, "2026-04-30")])
    cf["facts"][SA.TAXONOMIA_BALANCE] = {
        SA.TAG_BALANCE: {"units": {"shares": [
            _h("2025-06-30", 128_000_000, "2025-07-25")]}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
            _hq("2024-03-31", 200_000_000, "2024-05-01", "2024-01-01")]}}}
    s = SA.serie_acciones(cf)
    assert len({p["fuente"] for p in s}) == 1


def test_las_acciones_EMITIDAS_no_cuentan_como_en_circulacion():
    """
    CommonStockSharesIssued incluye las que estan en tesoreria y NO es sinonimo
    del conteo en circulacion. Medido: en DELL da 1.11..1.31 contra yahooquery
    y en HSY 1.08..1.09, mientras CommonStockSharesOutstanding da 1.0000 en los
    dos. Si alguien lo agrega como sinonimo, el market cap se infla en silencio.
    """
    cf = {"facts": {SA.TAXONOMIA_BALANCE: {"CommonStockSharesIssued":
          {"units": {"shares": [_h("2026-06-30", 999_000_000, "2026-07-23")]}}}}}
    assert SA.serie_balance(cf) == []
    assert SA.serie_acciones(cf) == []


def test_series_por_nivel_ofrece_los_tres_sin_elegir():
    """
    EL caso TRIP: portada desde 2025-05 (2 puntos aca) contra balance desde
    2021-03. Elegir en este modulo seria elegir a ciegas -- solo se sabe cual
    nivel sirve despues de validarlo contra yahooquery. Asi que se ofrecen los
    tres, etiquetados, y decide quien puede.
    """
    cf = _cf([_h("2025-05-01", 140_000_000, "2025-05-06")])
    cf["facts"][SA.TAXONOMIA_BALANCE] = {
        SA.TAG_BALANCE: {"units": {"shares": [
            _h("2021-03-31", 135_000_000, "2021-05-04"),
            _h("2025-05-01", 140_000_000, "2025-05-06")]}}}
    niveles = SA.series_por_nivel(cf)
    assert [p["fecha"] for p in niveles["portada"]] == ["2025-05-01"]
    assert len(niveles["balance"]) == 2
    assert niveles["balance"][0]["fecha"] == "2021-03-31"


def test_series_por_nivel_devuelve_las_tres_claves_siempre():
    """Quien consume itera sobre las claves; que falte una es un KeyError."""
    niveles = SA.series_por_nivel({"facts": {}})
    assert set(niveles) == set(SA.NIVELES)
    assert all(v == [] for v in niveles.values())


def test_series_por_nivel_ya_viene_despicada():
    """
    El filtrado de escala y picos tiene que estar aplicado POR NIVEL. Si se
    hiciera despues de elegir, el nivel que se descarta nunca se limpia y quien
    lo levante se come el error de unidad.
    """
    cf = _cf_bal([_h("2021-03-31", 100_000_000, "2021-05-04"),
                  _h("2021-06-30", 100_000, "2021-08-04"),
                  _h("2021-09-30", 101_000_000, "2021-11-04")])
    fechas = [p["fecha"] for p in SA.series_por_nivel(cf)["balance"]]
    assert "2021-06-30" not in fechas


# ------------------------------------------------------------ despicado --
def _s(*vals):
    """Serie sintetica trimestral a partir de conteos."""
    fechas = ["2023-02-05", "2023-05-05", "2023-08-03", "2023-11-02",
              "2024-02-09", "2024-05-03"]
    return [{"fecha": f, "shares": float(v)} for f, v in zip(fechas, vals)]


def test_despicar_saca_error_de_unidad():
    """Caso real HL: 617.339.547.000 entre dos valores de ~615M."""
    s = _s(610_000_000, 612_636_803, 617_339_547_000, 618_232_871, 620_000_000)
    out = SA.despicar(s)
    assert [p["shares"] for p in out] == [610_000_000, 612_636_803,
                                          618_232_871, 620_000_000]


def test_despicar_saca_conteo_parcial():
    """Caso real WFC: 1.823.028.137 entre 3.752M y 3.632M."""
    s = _s(3_700_000_000, 3_752_223_519, 1_823_028_137, 3_631_639_714,
           3_600_000_000)
    assert 1_823_028_137 not in [p["shares"] for p in SA.despicar(s)]


def test_despicar_no_toca_un_split_real():
    """
    EL test que hace util a la funcion. AAPL 4:1: el nivel queda DESPLAZADO,
    los vecinos no concuerdan entre si, el punto se conserva.
    """
    s = _s(4_270_000_000, 4_275_634_000, 17_001_802_000, 17_000_000_000,
           16_900_000_000)
    assert [p["shares"] for p in SA.despicar(s)] == [p["shares"] for p in s]


def test_despicar_no_toca_split_inverso():
    s = _s(633_653_119, 633_000_000, 316_940_010, 316_000_000, 315_000_000)
    assert len(SA.despicar(s)) == 5


def test_despicar_conserva_los_extremos():
    """Sin dos vecinos no hay con que juzgar; inventar un criterio seria peor."""
    s = _s(1e12, 100_000_000, 101_000_000, 102_000_000)
    out = SA.despicar(s)
    assert out[0]["shares"] == 1e12


def test_despicar_series_cortas():
    assert SA.despicar([]) == []
    assert len(SA.despicar(_s(100_000_000, 1e12))) == 2


def test_despicar_no_muta_la_entrada():
    s = _s(610_000_000, 612_636_803, 617_339_547_000, 618_232_871)
    SA.despicar(s)
    assert len(s) == 4


def test_fuera_de_escala_saca_rachas_de_error_de_unidad():
    """
    Caso real HL: DOS trimestres seguidos en ~6,3e11 contra una serie de ~620M.
    Con dos puntos malos consecutivos los vecinos de cada uno ya no concuerdan,
    asi que el despicado local no alcanza.
    """
    s = _s(610_000_000, 625_117_775, 626_290_204_000, 629_715_867_000,
           637_015_436, 640_000_000)
    out = SA._fuera_de_escala(s)
    assert [p["shares"] for p in out] == [610_000_000, 625_117_775,
                                          637_015_436, 640_000_000]


def test_fuera_de_escala_no_toca_un_split():
    """x100 es deliberadamente flojo: el split mas grande del universo es 20:1."""
    s = _s(508_720_481, 508_000_000, 10_187_554_818, 10_200_000_000,
           10_250_000_000)
    assert len(SA._fuera_de_escala(s)) == 5


def test_serie_acciones_aplica_los_dos_filtros():
    cf = _cf([_h("2023-02-05", 610_000_000, "2023-02-10"),
              _h("2023-05-05", 625_117_775, "2023-05-10"),
              _h("2023-08-03", 626_290_204_000, "2023-08-10"),   # racha
              _h("2023-11-02", 629_715_867_000, "2023-11-10"),   # racha
              _h("2024-02-09", 637_015_436, "2024-02-14"),
              _h("2024-05-03", 1_000_000, "2024-05-10"),          # pico aislado
              _h("2024-08-02", 640_000_000, "2024-08-10")])
    vals = [p["shares"] for p in SA.serie_acciones(cf)]
    assert vals == [610_000_000, 625_117_775, 637_015_436, 640_000_000]
