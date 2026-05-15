"""
Tests para mcp_server/tools/stocks.py

Unitarios (mock_conn):
  - Validacion de formato de fecha (YYYYMMDD obligatorio)
  - hasta anterior a desde
  - Clampeo de periodo > 360 dias
  - Periodo valido <= 360 dias: envelope sin warning
  - Conversion Decimal -> float y date -> ISO string
  - Ticker sin datos: envelope con data=[]
  - Error de DB: retorna {"error": "..."}
  - Tests identicos para los 4 tools (comparten _fetch_stock_data)

Integracion (@pytest.mark.integration, requiere DATABASE_URL):
  - Un test por tool con ticker y rango conocidos
"""

from decimal import Decimal
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from mcp_server.tools.stocks import (
    MAX_DAYS,
    get_market_structure,
    get_price_action,
    get_price_history,
    get_technical_indicators,
    _parse_date,
    _clamp_range,
    _row_to_dict,
)


# ── Helpers internos ──────────────────────────────────────────────────────────

class TestParseFecha:
    def test_formato_valido(self):
        assert _parse_date("20250115") == date(2025, 1, 15)

    def test_formato_con_guiones_falla(self):
        with pytest.raises(ValueError):
            _parse_date("2025-01-15")

    def test_formato_barras_falla(self):
        with pytest.raises(ValueError):
            _parse_date("15/01/2025")

    def test_fecha_inexistente_falla(self):
        with pytest.raises(ValueError):
            _parse_date("20250132")  # enero no tiene 32 dias

    def test_string_vacio_falla(self):
        with pytest.raises(ValueError):
            _parse_date("")


class TestClampRange:
    def test_periodo_valido_no_recorta(self):
        desde = date(2025, 1, 1)
        hasta = date(2025, 6, 1)  # 151 dias
        hasta_s, dias_sol, dias_serv, warning = _clamp_range(desde, hasta)
        assert hasta_s == hasta
        assert dias_sol == dias_serv == 151
        assert warning is None

    def test_periodo_exacto_no_recorta(self):
        desde = date(2025, 1, 1)
        hasta = date(2025, 1, 1)  # 0 dias (mismo dia)
        hasta_s, _, _, warning = _clamp_range(desde, hasta)
        assert hasta_s == hasta
        assert warning is None

    def test_periodo_excede_recorta(self):
        from datetime import timedelta
        desde = date(2024, 1, 1)
        hasta = date(2025, 6, 1)  # >360 dias
        hasta_s, dias_sol, dias_serv, warning = _clamp_range(desde, hasta)
        assert hasta_s == desde + timedelta(days=MAX_DAYS)  # calculado, no hardcodeado
        assert dias_serv == MAX_DAYS
        assert dias_sol > MAX_DAYS
        assert warning is not None
        assert "360" in warning
        assert "20240101" in warning

    def test_warning_contiene_fechas_yyyymmdd(self):
        desde = date(2023, 3, 15)
        hasta = date(2024, 5, 1)
        _, _, _, warning = _clamp_range(desde, hasta)
        assert "20230315" in warning


class TestRowToDict:
    """
    _row_to_dict recibe cualquier objeto con .items() (asyncpg.Record o dict).
    Los tests usan dicts directamente — mismo contrato que asyncpg.Record.
    """

    def test_decimal_se_convierte_a_float(self):
        row = {"close": Decimal("123.45"), "volume": 1000}
        result = _row_to_dict(row)
        assert isinstance(result["close"], float)
        assert result["close"] == pytest.approx(123.45)
        assert result["volume"] == 1000

    def test_date_se_convierte_a_iso(self):
        row = {"fecha": date(2025, 3, 15), "close": Decimal("100.0")}
        result = _row_to_dict(row)
        assert result["fecha"] == "2025-03-15"
        assert isinstance(result["close"], float)

    def test_none_pasa_directo(self):
        row = {"adj_close": None}
        result = _row_to_dict(row)
        assert result["adj_close"] is None


# ── Tests unitarios de get_price_history (representativos de los 4 tools) ────

TOOL_CASES = [
    ("get_price_history",        get_price_history,        "mcp_server.tools.stocks.get_pool"),
    ("get_technical_indicators", get_technical_indicators, "mcp_server.tools.stocks.get_pool"),
    ("get_price_action",         get_price_action,         "mcp_server.tools.stocks.get_pool"),
    ("get_market_structure",     get_market_structure,     "mcp_server.tools.stocks.get_pool"),
]


def _make_pool_mock(rows):
    """Construye un pool mock que devuelve `rows` en conn.fetch."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.fetchrow = AsyncMock(return_value=None)

    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = AsyncMock()
    pool.acquire.return_value = acquire_cm
    return pool


@pytest.mark.asyncio
async def test_fecha_desde_invalida():
    result = await get_price_history("AAPL", "2025-01-01", "20250131")
    assert "error" in result
    assert "desde" in result["error"]


@pytest.mark.asyncio
async def test_fecha_hasta_invalida():
    result = await get_price_history("AAPL", "20250101", "31/01/2025")
    assert "error" in result
    assert "hasta" in result["error"]


@pytest.mark.asyncio
async def test_hasta_antes_de_desde():
    result = await get_price_history("AAPL", "20250201", "20250101")
    assert "error" in result
    assert "anterior" in result["error"]


@pytest.mark.asyncio
async def test_periodo_valido_envelope_sin_warning(mock_conn):
    pool, conn = mock_conn
    conn.fetch.return_value = []  # ticker sin datos: envelope vacio

    with patch("mcp_server.tools.stocks.get_pool", AsyncMock(return_value=pool)):
        result = await get_price_history("AAPL", "20250101", "20250331")

    assert "error" not in result
    assert result["warning"] is None
    assert result["dias_solicitados"] == result["dias_servidos"]
    assert result["ticker"] == "AAPL"
    assert result["desde_servido"] == "20250101"
    assert result["hasta_servido"] == "20250331"
    assert result["data"] == []


@pytest.mark.asyncio
async def test_periodo_excede_warning_en_envelope(mock_conn):
    pool, conn = mock_conn
    conn.fetch.return_value = []

    with patch("mcp_server.tools.stocks.get_pool", AsyncMock(return_value=pool)):
        # 2 anos de rango -> debe recortarse
        result = await get_price_history("AAPL", "20230101", "20250101")

    assert "error" not in result
    assert result["warning"] is not None
    assert "360" in result["warning"]
    assert result["dias_servidos"] == MAX_DAYS
    assert result["dias_solicitados"] > MAX_DAYS
    # hasta_servido debe ser desde + MAX_DAYS dias
    from datetime import timedelta
    esperado = (date(2023, 1, 1) + timedelta(days=MAX_DAYS)).strftime("%Y%m%d")
    assert result["hasta_servido"] == esperado


@pytest.mark.asyncio
async def test_conversion_decimal_a_float_en_data(mock_conn):
    """Los valores Decimal de asyncpg deben salir como float en data."""
    pool, conn = mock_conn

    # Usamos un dict directamente: mismo contrato .items() que asyncpg.Record
    fake_row = {
        "ticker": "AAPL",
        "fecha": date(2025, 1, 15),
        "close": Decimal("182.34"),
        "volume": 50000000,
    }
    conn.fetch.return_value = [fake_row]

    with patch("mcp_server.tools.stocks.get_pool", AsyncMock(return_value=pool)):
        result = await get_price_history("AAPL", "20250101", "20250131")

    assert "error" not in result
    assert len(result["data"]) == 1
    row = result["data"][0]
    assert isinstance(row["close"], float)
    assert row["close"] == pytest.approx(182.34)
    assert row["fecha"] == "2025-01-15"
    assert row["volume"] == 50000000


@pytest.mark.asyncio
async def test_db_error_devuelve_error_legible(mock_conn):
    pool, conn = mock_conn
    conn.fetch.side_effect = Exception("connection timeout")

    with patch("mcp_server.tools.stocks.get_pool", AsyncMock(return_value=pool)):
        result = await get_price_history("AAPL", "20250101", "20250131")

    assert "error" in result
    assert "AAPL" in result["error"]
    assert "connection timeout" in result["error"]


# Tests analogos para los otros 3 tools (validan la misma logica compartida)

@pytest.mark.parametrize("tool_func,patch_path", [
    (get_technical_indicators, "mcp_server.tools.stocks.get_pool"),
    (get_price_action,         "mcp_server.tools.stocks.get_pool"),
    (get_market_structure,     "mcp_server.tools.stocks.get_pool"),
])
@pytest.mark.asyncio
async def test_fecha_invalida_otros_tools(tool_func, patch_path):
    result = await tool_func("AAPL", "BAD_DATE", "20250131")
    assert "error" in result


@pytest.mark.parametrize("tool_func,patch_path", [
    (get_technical_indicators, "mcp_server.tools.stocks.get_pool"),
    (get_price_action,         "mcp_server.tools.stocks.get_pool"),
    (get_market_structure,     "mcp_server.tools.stocks.get_pool"),
])
@pytest.mark.asyncio
async def test_envelope_valido_otros_tools(tool_func, patch_path, mock_conn):
    pool, conn = mock_conn
    conn.fetch.return_value = []

    with patch(patch_path, AsyncMock(return_value=pool)):
        result = await tool_func("MSFT", "20250101", "20250228")

    assert "error" not in result
    assert result["ticker"] == "MSFT"
    assert result["warning"] is None
    assert "data" in result


# ── Tests de integracion ──────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_price_history_integration():
    """Verifica que AAPL tiene precios en enero 2025."""
    result = await get_price_history("AAPL", "20250102", "20250131")
    assert "error" not in result
    assert result["ticker"] == "AAPL"
    assert len(result["data"]) > 0
    row = result["data"][0]
    assert "close" in row
    assert isinstance(row["close"], float)
    assert "fecha" in row


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_technical_indicators_integration():
    """Verifica que AAPL tiene indicadores tecnicos en enero 2025."""
    result = await get_technical_indicators("AAPL", "20250102", "20250131")
    assert "error" not in result
    assert len(result["data"]) > 0
    row = result["data"][0]
    assert "rsi14" in row
    assert "sma50" in row
    assert "macd" in row


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_price_action_integration():
    """Verifica que AAPL tiene features de price action en enero 2025."""
    result = await get_price_action("AAPL", "20250102", "20250131")
    assert "error" not in result
    assert len(result["data"]) > 0
    row = result["data"][0]
    assert "patron_doji" in row
    assert "es_alcista" in row


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_market_structure_integration():
    """Verifica que AAPL tiene features de market structure en enero 2025."""
    result = await get_market_structure("AAPL", "20250102", "20250131")
    assert "error" not in result
    assert len(result["data"]) > 0
    row = result["data"][0]
    assert "bos_bull_5" in row
    assert "choch_bear_10" in row


@pytest.mark.integration
@pytest.mark.asyncio
async def test_periodo_excede_recortado_integration():
    """Verifica que un rango de 2 anos se recorta a 360 dias en DB real."""
    result = await get_price_history("AAPL", "20220101", "20240101")
    assert "error" not in result
    assert result["warning"] is not None
    assert result["dias_servidos"] == MAX_DAYS
    assert len(result["data"]) > 0
