"""
Tests de la logica pura de la vista "Comparativo Fundamental".

Se prueba lo que puede MENTIR EN SILENCIO -- el formateo de nulos, el calculo
de la variacion de cada multiplo y el diagnostico de por que falta un dato --,
no el layout de Streamlit. El motor de valuacion ya tiene sus propios tests
en test_valuacion_implicita.py; aca solo se verifica que la vista no lo
deforme al mostrarlo.

El caso central es la ASIMETRIA: PER y P/S escalan lineal con el precio y
EV/EBITDA no, porque el EV lleva la deuda neta adentro. Si eso se rompiera,
la vista mostraria numeros plausibles y equivocados, que es la peor falla
posible en una pantalla que se usa para decidir.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("streamlit", reason="la vista importa streamlit")

from dashboard import comparativo as C            # noqa: E402
from src.utils import valuacion_implicita as V    # noqa: E402


# ------------------------------------------------------------------ formato

def test_los_nulos_se_muestran_como_guion_y_no_como_cero():
    """Un cero es un dato; un hueco no. Confundirlos es el error a evitar."""
    assert C._n(None) == "--"
    assert C._n(float("nan")) == "--"
    assert C._n(0) == "0.00"
    assert C._pct(None) == "--"
    assert C._pct(0.0) == "0.0%"


def test_delta_no_divide_por_cero_ni_por_nulos():
    assert C._delta(None, 10) is None
    assert C._delta(10, None) is None
    assert C._delta(0, 10) is None
    assert C._delta(10.0, 12.0) == pytest.approx(0.2)


# ------------------------------------------------------- la asimetria del EV

def _base(net_debt):
    """Empresa de juguete: 100 acciones a $10, y la deuda que se pida."""
    return {"close": 10.0, "shares": 100.0, "net_debt": net_debt,
            "net_income_ttm": 50.0, "revenue_ttm": 500.0,
            "ebitda_ttm": 100.0, "equity": 400.0, "fcf_ttm": 40.0}


def _variaciones(net_debt, subida=0.20):
    base = _base(net_debt)
    hoy = V.escenario(base, {})
    tesis = V.escenario(base, {}, variacion=subida)
    return {m: C._delta(hoy["multiplos"].get(m), tesis["multiplos"].get(m))
            for m in ("pe_ratio", "ps_ratio", "ev_ebitda")}


def test_per_y_ps_escalan_lineal_con_el_precio():
    d = _variaciones(net_debt=500.0)
    assert d["pe_ratio"] == pytest.approx(0.20)
    assert d["ps_ratio"] == pytest.approx(0.20)


def test_con_deuda_el_ev_ebitda_se_mueve_MENOS_que_el_per():
    """
    El caso T (AT&T) en chico: +20% de precio movio su PER +20,0% y su
    EV/EBITDA solo +11,6%, porque arrastra 126.000 MM de deuda neta.
    """
    d = _variaciones(net_debt=500.0)          # deuda = 50% del market cap
    assert d["ev_ebitda"] < d["pe_ratio"]
    # market cap 1000 -> 1200; EV 1500 -> 1700; 1700/1500 - 1 = 13,3%
    assert d["ev_ebitda"] == pytest.approx(1700.0 / 1500.0 - 1.0)


def test_con_caja_neta_el_efecto_se_AMPLIFICA():
    """
    La otra mitad, que es la contraintuitiva: con deuda neta NEGATIVA el
    EV/EBITDA se mueve MAS que el PER, no menos. Si la vista solo contemplara
    el caso endeudado, aca mostraria una nota al reves.
    """
    d = _variaciones(net_debt=-500.0)
    assert d["ev_ebitda"] > d["pe_ratio"]
    # market cap 1000 -> 1200; EV 500 -> 700; 700/500 - 1 = 40%
    assert d["ev_ebitda"] == pytest.approx(0.40)


def test_sin_deuda_neta_no_hay_ev_ebitda():
    """Preferimos el hueco visible al numero calculado con deuda parcial."""
    base = _base(net_debt=None)
    assert V.escenario(base, {})["multiplos"]["ev_ebitda"] is None


# -------------------------------------------------------------- diagnostico

def test_el_motivo_distingue_las_dos_causas_del_hueco_de_ev():
    """
    Se arreglan distinto: una amplia el mapeo de tags de D&A, la otra necesita
    los hechos dimensionados. Decir "sin dato" a secas no deja accionar.
    """
    sin_ebitda = dict(_base(net_debt=500.0), ebitda_ttm=None)
    assert "EBITDA" in C._motivo_sin_dato("ev_ebitda", sin_ebitda)

    sin_deuda = _base(net_debt=None)
    motivo = C._motivo_sin_dato("ev_ebitda", sin_deuda)
    assert "deuda neta" in motivo
    assert "dimensiones" in motivo


def test_el_per_negativo_se_explica_en_vez_de_mostrarse():
    """Un PER con resultado negativo no es 'barato': es otra categoria."""
    base = dict(_base(net_debt=0.0), net_income_ttm=-50.0)
    assert V.escenario(base, {})["multiplos"]["pe_ratio"] is None
    assert "no es 'barato'" in C._motivo_sin_dato("pe_ratio", base)
