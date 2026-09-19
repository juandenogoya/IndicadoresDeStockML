"""Tests de src/utils/earnings_cobertura.py (quien debe un balance)."""

from datetime import date, timedelta

import pytest

from src.utils import earnings_cobertura as ec

HOY = date(2026, 9, 19)


def trimestral(n=8, ultimo=date(2026, 8, 3), cada=91):
    """n anuncios trimestrales terminando en `ultimo`."""
    return [ultimo - timedelta(days=cada * k) for k in range(n)][::-1]


# --- cadencia ------------------------------------------------------------------

def test_cadencia_propia_es_la_mediana_de_los_gaps():
    cad, propia = ec.cadencia(trimestral())
    assert (cad, propia) == (91.0, True)


def test_cadencia_ignora_un_intervalo_raro():
    # un cambio de cierre fiscal deja un gap de 150; la mediana no se mueve
    f = trimestral(n=6)
    f[3] = f[3] + timedelta(days=59)
    cad, propia = ec.cadencia(f)
    assert propia and 85 <= cad <= 97


def test_cadencia_semestral():
    cad, propia = ec.cadencia(trimestral(n=6, cada=182))
    assert (cad, propia) == (182.0, True)


def test_sin_historia_suficiente_cae_al_default():
    assert ec.cadencia([]) == (float(ec.CADENCIA_DEFAULT), False)
    assert ec.cadencia([date(2026, 5, 1)]) == (float(ec.CADENCIA_DEFAULT), False)
    assert ec.cadencia([date(2026, 5, 1), date(2026, 8, 1)])[1] is False


def test_fechas_repetidas_o_desordenadas_no_rompen():
    f = trimestral()
    assert ec.cadencia(f + f[::-1]) == ec.cadencia(f)


# --- debe un balance -----------------------------------------------------------

def test_al_dia_no_debe():
    e = ec.estado("AAPL", trimestral(ultimo=date(2026, 7, 30)), HOY)
    assert e.dias == 51 and not e.debe


def test_atrasado_debe():
    e = ec.estado("BA", trimestral(ultimo=date(2026, 4, 22)), HOY)
    assert e.debe and e.dias == 150


def test_el_margen_es_la_frontera():
    cad = 91
    justo = HOY - timedelta(days=int(cad * ec.MARGEN))          # 104 dias
    assert not ec.estado("X", trimestral(ultimo=justo), HOY).debe
    pasado = HOY - timedelta(days=int(cad * ec.MARGEN) + 1)
    assert ec.estado("X", trimestral(ultimo=pasado), HOY).debe


def test_semestral_no_se_marca_a_los_100_dias():
    # con cadencia fija de 91 dias esto daria un falso positivo
    e = ec.estado("HMY", trimestral(n=6, ultimo=date(2026, 6, 10), cada=182), HOY)
    assert e.cadencia == 182 and not e.debe


def test_sin_ninguna_fila_debe():
    e = ec.estado("HOOD", [], HOY)
    assert e.debe and e.ultimo is None and e.dias is None


# --- cobertura del universo ----------------------------------------------------

def _universo():
    return {
        "AAPL": trimestral(ultimo=date(2026, 7, 30)),      # al dia
        "MSFT": trimestral(ultimo=date(2026, 7, 29)),      # al dia
        "BA":   trimestral(ultimo=date(2026, 4, 22)),      # debe (150 d)
        "F":    trimestral(ultimo=date(2026, 4, 29)),      # debe (143 d)
        "HOOD": [],                                        # sin historia
    }


def test_cobertura_separa_sin_historia_de_atrasados():
    c = ec.cobertura(_universo(), HOY)
    assert c.total == 5
    assert c.sin_historia == ["HOOD"]
    assert [e.ticker for e in c.deben] == ["BA", "F"]       # el mas atrasado primero
    assert c.al_dia_hasta == date(2026, 7, 30)
    assert c.pendientes == 3 and c.corridas == 1 and not c.al_dia


def test_cobertura_al_dia():
    c = ec.cobertura({"AAPL": trimestral(ultimo=date(2026, 7, 30))}, HOY)
    assert c.al_dia and c.pendientes == 0 and c.corridas == 0


def test_corridas_redondea_para_arriba():
    hist = {f"T{i}": [] for i in range(ec.MAX_CALLS_DIA + 1)}
    assert ec.cobertura(hist, HOY).corridas == 2


def test_a_traer_pone_primero_a_los_que_no_tienen_nada():
    assert ec.a_traer(ec.cobertura(_universo(), HOY)) == ["HOOD", "BA", "F"]


def test_resumen_es_ascii_y_dice_lo_que_falta():
    lineas = ec.resumen(ec.cobertura(_universo(), HOY))
    texto = " ".join(lineas)
    assert texto.isascii()
    assert "2026-07-30" in texto and "BA" in texto and "3 llamadas" in texto


def test_resumen_al_dia_no_pide_corridas():
    lineas = ec.resumen(ec.cobertura({"AAPL": trimestral(ultimo=date(2026, 9, 1))}, HOY))
    assert any("Al dia" in x for x in lineas)
    assert not any("llamadas" in x for x in lineas)


@pytest.mark.parametrize("margen", [1.0, 1.15, 1.5])
def test_margen_mas_laxo_marca_menos(margen):
    c = ec.cobertura(_universo(), HOY, margen=margen)
    assert len(c.deben) <= 2
