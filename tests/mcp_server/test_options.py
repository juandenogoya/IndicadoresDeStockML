"""
Tests para mcp_server/tools/options.py

Unitarios:
  - Helpers internos: _moneyness_label, _build_pcr_por_vencimiento, _tend_row
  - get_options_analysis: ticker sin datos, error de DB, estructura de respuesta,
    clampeo de dias_historia, separacion calls/puts en acumulacion_oi

Integracion (@pytest.mark.integration):
  - get_options_analysis con ticker real (AAPL)
  - get_options_analysis con ticker sin datos de opciones
"""

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server.tools.options import (
    MAX_DIAS_HISTORIA,
    _build_pcr_por_vencimiento,
    _moneyness_label,
    _tend_row,
    get_options_analysis,
)


# ── Tests de helpers ──────────────────────────────────────────────────────────

class TestMoneynessLabel:
    def test_atm_call(self):
        assert _moneyness_label(1.5, "call") == "ATM"

    def test_atm_put(self):
        assert _moneyness_label(-1.9, "put") == "ATM"

    def test_atm_exacto_cero(self):
        assert _moneyness_label(0.0, "call") == "ATM"

    def test_call_otm(self):
        # strike > precio → OTM para call
        assert _moneyness_label(5.0, "call") == "OTM"

    def test_call_itm(self):
        # strike < precio → ITM para call
        assert _moneyness_label(-5.0, "call") == "ITM"

    def test_put_itm(self):
        # strike > precio → ITM para put (price below strike)
        assert _moneyness_label(5.0, "put") == "ITM"

    def test_put_otm(self):
        # strike < precio → OTM para put
        assert _moneyness_label(-5.0, "put") == "OTM"

    def test_none_devuelve_na(self):
        assert _moneyness_label(None, "call") == "N/A"

    def test_limite_exacto_2pct_es_atm(self):
        assert _moneyness_label(2.0, "call") == "ATM"
        assert _moneyness_label(-2.0, "put") == "ATM"

    def test_sobre_limite_no_es_atm(self):
        assert _moneyness_label(2.01, "call") == "OTM"


class TestBuildPcrPorVencimiento:
    def _make_row(self, venc, fecha_snap, call_vol, put_vol, call_oi, put_oi, dias=30):
        return {
            "vencimiento":    date.fromisoformat(venc),
            "fecha_snapshot": date.fromisoformat(fecha_snap),
            "dias_a_venc":    dias,
            "call_vol":       call_vol,
            "put_vol":        put_vol,
            "call_oi":        call_oi,
            "put_oi":         put_oi,
        }

    def test_vencimientos_ordenados_asc(self):
        rows = [
            self._make_row("2026-07-17", "2026-05-15", 100, 120, 500, 600),
            self._make_row("2026-06-19", "2026-05-15", 200, 180, 800, 750),
        ]
        result = _build_pcr_por_vencimiento(rows)
        assert result[0]["vencimiento"] == "2026-06-19"
        assert result[1]["vencimiento"] == "2026-07-17"

    def test_serie_anidada_por_vencimiento(self):
        rows = [
            self._make_row("2026-06-19", "2026-05-15", 100, 120, 500, 600),
            self._make_row("2026-06-19", "2026-05-14", 90,  110, 480, 580),
        ]
        result = _build_pcr_por_vencimiento(rows)
        assert len(result) == 1
        assert len(result[0]["serie"]) == 2

    def test_pcr_calculado_correctamente(self):
        rows = [self._make_row("2026-06-19", "2026-05-15", 100, 140, 500, 600)]
        result = _build_pcr_por_vencimiento(rows)
        serie = result[0]["serie"][0]
        assert serie["pcr_vol"] == pytest.approx(1.4)
        assert serie["pcr_oi"]  == pytest.approx(1.2)

    def test_pcr_none_cuando_call_es_cero(self):
        rows = [self._make_row("2026-06-19", "2026-05-15", 0, 100, 0, 500)]
        result = _build_pcr_por_vencimiento(rows)
        serie = result[0]["serie"][0]
        assert serie["pcr_vol"] is None
        assert serie["pcr_oi"]  is None

    def test_vencimiento_como_iso_string(self):
        rows = [self._make_row("2026-06-19", "2026-05-15", 100, 120, 500, 600)]
        result = _build_pcr_por_vencimiento(rows)
        assert result[0]["vencimiento"] == "2026-06-19"
        assert result[0]["serie"][0]["fecha"] == "2026-05-15"


class TestTendRow:
    def test_iv_skew_calculado(self):
        row = {
            "fecha": date(2026, 5, 15),
            "iv_call_avg": Decimal("0.28"),
            "iv_put_avg":  Decimal("0.31"),
            "call_vol": 100000, "put_vol": 120000,
            "pcr_vol": Decimal("1.2"), "call_oi": 400000,
            "put_oi": 480000, "pcr_oi": Decimal("1.2"),
            "precio_sub": Decimal("195.5"), "n_contratos": 500,
            "vol_total_zscore": None, "pcr_vol_zscore": None,
            "iv_zscore": None, "vol_relativo": None, "percentil_vol": None,
        }
        result = _tend_row(row)
        assert result["iv_skew"] == pytest.approx(0.03, abs=1e-4)
        assert isinstance(result["iv_call_avg"], float)
        assert result["fecha"] == "2026-05-15"

    def test_iv_skew_none_cuando_faltan_datos(self):
        row = {
            "fecha": date(2026, 5, 15),
            "iv_call_avg": None, "iv_put_avg": None,
            "call_vol": 0, "put_vol": 0, "pcr_vol": None,
            "call_oi": 0, "put_oi": 0, "pcr_oi": None,
            "precio_sub": None, "n_contratos": 0,
            "vol_total_zscore": None, "pcr_vol_zscore": None,
            "iv_zscore": None, "vol_relativo": None, "percentil_vol": None,
        }
        result = _tend_row(row)
        assert result["iv_skew"] is None


# ── Tests unitarios de get_options_analysis ───────────────────────────────────

def _make_pool_mock(fetchrow_val, fetch_side_effect):
    """
    Construye (pool, conn) listos para parchear get_pool().
    pool debe ser MagicMock (no AsyncMock) para que pool.acquire()
    devuelva el context manager directamente sin envolverlo en coroutine.
    """
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_val)
    # side_effect como lista: cada llamada consume el siguiente elemento
    conn.fetch = AsyncMock(side_effect=fetch_side_effect if fetch_side_effect else None)

    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__  = AsyncMock(return_value=False)

    pool = MagicMock()          # MagicMock, no AsyncMock — igual que mock_conn
    pool.acquire.return_value = acquire_cm
    return pool, conn


@pytest.mark.asyncio
async def test_ticker_sin_datos_devuelve_error():
    pool, _ = _make_pool_mock(
        fetchrow_val={"max_fecha": None, "cutoff_fecha": None},
        fetch_side_effect=[],
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("FAKE")
    assert "error" in result
    assert "FAKE" in result["error"]


@pytest.mark.asyncio
async def test_db_error_devuelve_error():
    pool, conn = _make_pool_mock(
        fetchrow_val=None,
        fetch_side_effect=[],
    )
    conn.fetchrow.side_effect = Exception("connection refused")
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL")
    assert "error" in result
    assert "AAPL" in result["error"]


@pytest.mark.asyncio
async def test_estructura_respuesta_valida():
    """Verifica que el envelope tiene todas las claves esperadas."""
    pool, _ = _make_pool_mock(
        fetchrow_val={
            "max_fecha":    date(2026, 5, 15),
            "cutoff_fecha": date(2026, 4, 18),
        },
        fetch_side_effect=[[], [], []],  # tend, pcr, acum — todos vacios
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL")

    assert "error" not in result
    assert result["ticker"]          == "AAPL"
    assert result["fecha_snapshot"]  == "2026-05-15"
    assert result["cutoff_fecha"]    == "2026-04-18"
    assert result["dias_historia"]   == 20
    assert "tendencia_diaria"        in result
    assert "pcr_por_vencimiento"     in result
    assert "acumulacion_oi"          in result
    assert "top_calls_por_delta_oi"  in result["acumulacion_oi"]
    assert "top_puts_por_delta_oi"   in result["acumulacion_oi"]


@pytest.mark.asyncio
async def test_dias_historia_se_clampea_al_maximo():
    pool, _ = _make_pool_mock(
        fetchrow_val={
            "max_fecha":    date(2026, 5, 15),
            "cutoff_fecha": date(2026, 4, 18),
        },
        fetch_side_effect=[[], [], []],
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL", dias_historia=999)
    assert result["dias_historia"] == MAX_DIAS_HISTORIA


@pytest.mark.asyncio
async def test_acumulacion_separada_por_tipo():
    """Las filas de acumulacion_oi se separan correctamente en calls y puts."""
    acum_rows = [
        {
            "vencimiento": date(2026, 6, 19), "tipo": "call", "strike": Decimal("200"),
            "oi_inicio": 5000, "oi_fin": 12000, "delta_oi": 7000,
            "iv_actual": Decimal("0.32"), "precio_subyacente": Decimal("195.5"),
            "moneyness_pct": Decimal("2.3"),
        },
        {
            "vencimiento": date(2026, 5, 31), "tipo": "put", "strike": Decimal("185"),
            "oi_inicio": 2000, "oi_fin": 8000, "delta_oi": 6000,
            "iv_actual": Decimal("0.41"), "precio_subyacente": Decimal("195.5"),
            "moneyness_pct": Decimal("-5.4"),
        },
    ]
    pool, _ = _make_pool_mock(
        fetchrow_val={
            "max_fecha":    date(2026, 5, 15),
            "cutoff_fecha": date(2026, 4, 18),
        },
        fetch_side_effect=[[], [], acum_rows],
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL")

    calls = result["acumulacion_oi"]["top_calls_por_delta_oi"]
    puts  = result["acumulacion_oi"]["top_puts_por_delta_oi"]
    assert len(calls) == 1
    assert len(puts)  == 1
    assert calls[0]["tipo"] == "call"
    assert calls[0]["moneyness_label"] == "OTM"    # strike 200 > precio 195.5
    assert puts[0]["tipo"]  == "put"
    assert puts[0]["moneyness_label"]  == "OTM"    # strike 185 < precio 195.5


@pytest.mark.asyncio
async def test_precio_sub_del_primer_dia_de_tendencia():
    tend_row_mock = {
        "fecha": date(2026, 5, 15),
        "call_vol": 100000, "put_vol": 120000,
        "pcr_vol": Decimal("1.2"), "call_oi": 400000, "put_oi": 480000,
        "pcr_oi": Decimal("1.2"), "iv_call_avg": Decimal("0.28"),
        "iv_put_avg": Decimal("0.31"), "precio_sub": Decimal("195.50"),
        "n_contratos": 500, "vol_total_zscore": None, "pcr_vol_zscore": None,
        "iv_zscore": None, "vol_relativo": None, "percentil_vol": None,
    }
    pool, _ = _make_pool_mock(
        fetchrow_val={
            "max_fecha":    date(2026, 5, 15),
            "cutoff_fecha": date(2026, 4, 18),
        },
        fetch_side_effect=[[tend_row_mock], [], []],
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL")

    assert result["precio_subyacente"] == pytest.approx(195.50)


# ── Tests de integracion ──────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_options_analysis_integration():
    """Verifica estructura completa con ticker real."""
    result = await get_options_analysis("AAPL", dias_historia=10)

    assert "error" not in result, f"Error inesperado: {result.get('error')}"
    assert result["ticker"] == "AAPL"
    assert result["precio_subyacente"] is not None
    assert isinstance(result["tendencia_diaria"], list)
    assert isinstance(result["pcr_por_vencimiento"], list)
    assert "top_calls_por_delta_oi" in result["acumulacion_oi"]
    assert "top_puts_por_delta_oi"  in result["acumulacion_oi"]

    # Validar estructura de una fila de tendencia
    if result["tendencia_diaria"]:
        row = result["tendencia_diaria"][0]
        assert "fecha"      in row
        assert "pcr_vol"    in row
        assert "iv_skew"    in row

    # Validar estructura de pcr_por_vencimiento
    if result["pcr_por_vencimiento"]:
        venc = result["pcr_por_vencimiento"][0]
        assert "vencimiento" in venc
        assert "dias_a_venc" in venc
        assert "serie"       in venc
        if venc["serie"]:
            s = venc["serie"][0]
            assert "fecha"    in s
            assert "pcr_vol"  in s

    # Validar que dias_a_venc > 0 (solo contratos vivos)
    for venc in result["pcr_por_vencimiento"]:
        assert venc["dias_a_venc"] > 0, "Se filtraron contratos vencidos"

    # Validar moneyness_label en acumulacion
    for row in result["acumulacion_oi"]["top_calls_por_delta_oi"]:
        assert row["moneyness_label"] in ("ATM", "OTM", "ITM", "N/A")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_options_analysis_ticker_sin_datos():
    """Un ticker sin datos de opciones debe retornar error claro."""
    result = await get_options_analysis("TICKER_INEXISTENTE_XYZ")
    assert "error" in result
