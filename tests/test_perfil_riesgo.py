"""
Tests del clasificador de perfil de riesgo (src/utils/perfil_riesgo.py).

PUROS: universo sintetico, sin DB. Enfoque DATA-DRIVEN (percentil del universo):
    - prior sectorial (sector -> caja base) y fallback,
    - percentil por eje y caja por cuartil,
    - composite de percentiles (peso igual) + correccion +/-1,
    - flag de excepcion (desacuerdo 2+ cajas),
    - caso sin metricas (queda en el prior),
    - ranking intra-caja.
"""

import pytest

from src.utils import perfil_riesgo as pr


def _met(w, m, b, dd):
    return {"atr_pct_w": w, "atr_pct_m": m, "beta": b, "max_dd_1a": dd}


# Universo sintetico: gradiente claro de riesgo, sector NEUTRO (Moderada base)
# para aislar el efecto del cuantitativo salvo donde se testea el sector.
def _universo_gradiente(sector="Industrials"):
    rows = []
    for i in range(12):  # 12 tickers de vol creciente
        rows.append({
            "ticker": f"T{i:02d}", "sector": sector,
            "metricas": _met(3.0 + i, 5.0 + 2 * i, -0.3 + 0.3 * i, 5.0 + 4 * i),
        })
    return rows


# --- caja base (prior) ------------------------------------------------------

def test_sector_prior_staple_conservadora():
    base, fuente = pr.caja_base("Consumer Defensive")
    assert base == pr.CONSERVADORA and fuente == "sector"


def test_sector_prior_tech_arriesgada():
    base, fuente = pr.caja_base("Technology")
    assert base == pr.ARRIESGADA and fuente == "sector"


def test_sector_desconocido_cae_a_fallback():
    base, fuente = pr.caja_base("Sector Inexistente")
    assert base == pr.SECTOR_BASE_FALLBACK and fuente == "fallback"


# --- percentil y cuartil ----------------------------------------------------

def test_rank_percentil_extremos_y_medio():
    pob = [1, 2, 3, 4, 5]
    assert pr.rank_percentil(1, pob) == 10.0    # menor: midrank bajo
    assert pr.rank_percentil(5, pob) == 90.0    # mayor: midrank alto
    assert pr.rank_percentil(3, pob) == 50.0    # centro


def test_rank_percentil_none():
    assert pr.rank_percentil(None, [1, 2, 3]) is None
    assert pr.rank_percentil(5, []) is None


def test_caja_por_cuartil():
    assert pr.caja_por_cuartil(10) == pr.CONSERVADORA
    assert pr.caja_por_cuartil(40) == pr.MODERADA
    assert pr.caja_por_cuartil(60) == pr.ARRIESGADA
    assert pr.caja_por_cuartil(90) == pr.ESPECULATIVA
    assert pr.caja_por_cuartil(None) is None


# --- perfilado del universo -------------------------------------------------

def test_gradiente_mapea_de_menor_a_mayor_riesgo():
    res = pr.perfilar_universo(_universo_gradiente())
    by_tk = {c["ticker"]: c for c in res}
    # el mas tranquilo cae por debajo del mas riesgoso en caja cuantitativa
    assert by_tk["T00"]["caja_cuant"] < by_tk["T11"]["caja_cuant"]
    # score es percentil: el mas riesgoso ~tope, el mas calmo ~piso
    assert by_tk["T11"]["score_riesgo"] > by_tk["T00"]["score_riesgo"]


def test_perfil_puro_no_capa_el_movimiento():
    # Sector Conservadora (base 0); el ticker mas riesgoso cae en cuartil alto
    # (cuant 3) y el PERFIL respeta el comportamiento completo -> Especulativa.
    rows = _universo_gradiente(sector="Consumer Defensive")
    res = pr.perfilar_universo(rows)
    top = max(res, key=lambda c: c["score_riesgo"])
    assert top["caja_base"] == pr.CONSERVADORA
    assert top["caja_cuant"] == pr.ESPECULATIVA
    assert top["perfil_ordinal"] == pr.ESPECULATIVA   # perfil = cuant, sin capar
    assert top["movio"] == 3                           # despegue completo vs sector


def test_excepcion_cuando_desacuerdo_fuerte():
    rows = _universo_gradiente(sector="Consumer Defensive")
    res = pr.perfilar_universo(rows)
    top = max(res, key=lambda c: c["score_riesgo"])
    assert top["excepcion"] is True  # base 0 vs cuant 3 -> diff 3 (>=2)


def test_sin_metricas_queda_en_prior():
    rows = _universo_gradiente()
    rows.append({"ticker": "NA", "sector": "Consumer Defensive", "metricas": {}})
    res = pr.perfilar_universo(rows)
    na = next(c for c in res if c["ticker"] == "NA")
    assert na["sin_cuant"] is True
    assert na["perfil"] == "Conservadora"
    assert na["movio"] == 0 and na["score_riesgo"] is None


def test_renormaliza_sin_beta():
    # Un ticker sin beta se clasifica con los ejes restantes, no rompe.
    rows = _universo_gradiente()
    rows[3]["metricas"]["beta"] = None
    res = pr.perfilar_universo(rows)
    c = res[3] if res[3]["ticker"] == rows[3]["ticker"] else \
        next(x for x in res if x["ticker"] == rows[3]["ticker"])
    assert "beta" not in c["pct_ejes"]
    assert c["score_riesgo"] is not None


# --- ranking intra-caja -----------------------------------------------------

def test_rankea_intra_caja():
    res = pr.perfilar_universo(_universo_gradiente())
    # dentro de cada caja, rank 1 tiene el score mas alto y pct 100
    por_caja = {}
    for c in res:
        if c["score_riesgo"] is not None:
            por_caja.setdefault(c["perfil_ordinal"], []).append(c)
    for items in por_caja.values():
        top = max(items, key=lambda x: x["score_riesgo"])
        assert top["rank_en_caja"] == 1
        assert top["pct_en_caja"] == 100.0


def test_rank_none_si_sin_score():
    rows = [{"ticker": "NA", "sector": "Consumer Defensive", "metricas": {}}]
    res = pr.perfilar_universo(rows)
    assert res[0]["rank_en_caja"] is None and res[0]["pct_en_caja"] is None
