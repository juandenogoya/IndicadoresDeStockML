"""
Tests para mcp_server/tools/exploration.py

Unitarios: mockean get_pool(), no requieren DB ni asyncpg instalado.
Integracion (@pytest.mark.integration): requieren DATABASE_URL en env
  y asyncpg instalado. Se skipean automaticamente si DATABASE_URL falta.

Correr desde el repo root:
  pytest tests/mcp_server/test_exploration.py -v            # solo unitarios
  pytest tests/mcp_server/test_exploration.py -v -m integration  # solo integ
  pytest tests/mcp_server/test_exploration.py -v --co       # listar tests
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from mcp_server.tools.exploration import (
    describe_table,
    list_tables,
    list_tickers,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_records(*dicts):
    """Simula asyncpg.Record: dicts planos que soportan dict(row)."""
    return list(dicts)


# ── list_tables: tests unitarios ──────────────────────────────────────────────

class TestListTables:

    @pytest.mark.asyncio
    async def test_sin_filtro_devuelve_todas(self, mock_conn):
        pool, conn = mock_conn
        conn.fetch.return_value = make_records(
            {"table_name": "activos",         "row_count_estimate": 199},
            {"table_name": "precios_diarios", "row_count_estimate": 206035},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tables()

        assert len(result) == 2
        assert result[0]["table_name"] == "activos"
        assert result[1]["row_count_estimate"] == 206035

    @pytest.mark.asyncio
    async def test_con_pattern_pasa_parametro(self, mock_conn):
        pool, conn = mock_conn
        conn.fetch.return_value = make_records(
            {"table_name": "precios_diarios",  "row_count_estimate": 206035},
            {"table_name": "precios_semanales", "row_count_estimate": 9800},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tables("precios%")

        # Verifica que el pattern llego a conn.fetch como primer argumento
        call_args = conn.fetch.call_args
        assert call_args[0][1] == "precios%"
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_reltuples_negativo_es_none(self, mock_conn):
        """reltuples=-1 (no analizada) debe aparecer como None en el resultado."""
        pool, conn = mock_conn
        conn.fetch.return_value = make_records(
            {"table_name": "nueva_tabla", "row_count_estimate": None},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tables()

        assert result[0]["row_count_estimate"] is None

    @pytest.mark.asyncio
    async def test_resultado_es_lista_de_dicts(self, mock_conn):
        """Cada elemento del resultado debe ser un dict (no asyncpg.Record)."""
        pool, conn = mock_conn
        conn.fetch.return_value = make_records(
            {"table_name": "activos", "row_count_estimate": 199},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tables()

        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    @pytest.mark.asyncio
    async def test_db_error_devuelve_error_legible_postgres(self, mock_conn):
        """Error de DB (cualquier Exception) -> lista con dict "error", no crash."""
        pool, conn = mock_conn
        # Simula error de tipo postgres
        conn.fetch.side_effect = Exception("SSL connection has been closed unexpectedly")

        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tables()

        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        assert "SSL" in result[0]["error"] or "No se pudo" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_db_error_devuelve_error_legible_oserror(self, mock_conn):
        """Error de red/socket (OSError) -> lista con dict "error", no crash."""
        pool, conn = mock_conn
        conn.fetch.side_effect = OSError("Connection refused")

        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tables()

        assert isinstance(result, list)
        assert "error" in result[0]
        assert len(result[0]["error"]) > 0


# ── describe_table: tests unitarios ───────────────────────────────────────────

class TestDescribeTable:

    @pytest.mark.asyncio
    async def test_tabla_existente_retorna_schema_completo(self, mock_conn):
        pool, conn = mock_conn
        # fetchrow: existence check
        conn.fetchrow.return_value = {"row_count_estimate": 206035}
        # fetch: [columnas, indices] en orden
        conn.fetch.side_effect = [
            make_records(
                {"name": "fecha",  "type": "date",    "nullable": False, "default": None},
                {"name": "ticker", "type": "character varying(20)",
                 "nullable": False, "default": None},
                {"name": "close",  "type": "double precision",
                 "nullable": True,  "default": None},
            ),
            make_records(
                {"name": "precios_diarios_pkey",
                 "definition": "CREATE UNIQUE INDEX precios_diarios_pkey ON precios_diarios"},
            ),
        ]
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await describe_table("precios_diarios")

        assert result["table_name"] == "precios_diarios"
        assert result["row_count_estimate"] == 206035
        assert len(result["columns"]) == 3
        assert result["columns"][0]["name"] == "fecha"
        assert len(result["indexes"]) == 1

    @pytest.mark.asyncio
    async def test_tabla_no_existente_retorna_error(self, mock_conn):
        pool, conn = mock_conn
        conn.fetchrow.return_value = None  # tabla no encontrada en pg_class

        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await describe_table("tabla_inexistente")

        assert "error" in result
        assert "tabla_inexistente" in result["error"]
        assert "list_tables" in result["error"]

    @pytest.mark.asyncio
    async def test_column_default_no_se_limpia(self, mock_conn):
        """column_default se retorna tal cual, incluyendo nextval(...)."""
        pool, conn = mock_conn
        conn.fetchrow.return_value = {"row_count_estimate": 100}
        conn.fetch.side_effect = [
            make_records(
                {"name": "id", "type": "integer", "nullable": False,
                 "default": "nextval('alertas_scanner_id_seq'::regclass)"},
            ),
            make_records(),  # sin indices
        ]
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await describe_table("alertas_scanner")

        id_col = result["columns"][0]
        assert id_col["default"] == "nextval('alertas_scanner_id_seq'::regclass)"

    @pytest.mark.asyncio
    async def test_retorno_tiene_claves_requeridas(self, mock_conn):
        pool, conn = mock_conn
        conn.fetchrow.return_value = {"row_count_estimate": 0}
        conn.fetch.side_effect = [make_records(), make_records()]

        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await describe_table("tabla_vacia")

        required_keys = {"table_name", "row_count_estimate", "columns", "indexes"}
        assert required_keys.issubset(result.keys())

    @pytest.mark.asyncio
    async def test_db_error_devuelve_error_legible_postgres(self, mock_conn):
        pool, conn = mock_conn
        conn.fetchrow.side_effect = Exception("terminating connection due to administrator command")

        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await describe_table("precios_diarios")

        assert "error" in result
        assert len(result["error"]) > 0

    @pytest.mark.asyncio
    async def test_db_error_devuelve_error_legible_oserror(self, mock_conn):
        pool, conn = mock_conn
        conn.fetchrow.side_effect = OSError("Network is unreachable")

        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await describe_table("precios_diarios")

        assert "error" in result
        assert "precios_diarios" in result["error"]


# ── list_tickers: tests unitarios ─────────────────────────────────────────────

class TestListTickers:

    @pytest.mark.asyncio
    async def test_sin_filtros_devuelve_todos(self, mock_conn):
        pool, conn = mock_conn
        conn.fetch.return_value = make_records(
            {"ticker": "AAPL", "sector": "Technology",
             "industry": "Consumer Electronics",
             "modelo_asignado": "rf_v3", "activo": True},
            {"ticker": "BAC",  "sector": "Financials",
             "industry": "Banks", "modelo_asignado": None, "activo": True},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tickers()

        assert len(result) == 2
        assert result[0]["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_cada_elemento_tiene_claves_del_contrato(self, mock_conn):
        pool, conn = mock_conn
        conn.fetch.return_value = make_records(
            {"ticker": "NVDA", "sector": "Technology",
             "industry": "Semiconductors",
             "modelo_asignado": "rf_v3", "activo": True},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tickers()

        required = {"ticker", "sector", "industry", "modelo_asignado", "activo"}
        assert required.issubset(result[0].keys())

    @pytest.mark.asyncio
    async def test_filtro_sector_pasa_parametro(self, mock_conn):
        pool, conn = mock_conn
        conn.fetch.return_value = make_records(
            {"ticker": "JPM", "sector": "Financials",
             "industry": "Banks",
             "modelo_asignado": None, "activo": True},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tickers(sector="Financials")

        call_args = conn.fetch.call_args[0]
        assert call_args[1] == "Financials"   # $1 = sector
        assert call_args[2] is False          # $2 = only_ml_active default
        assert result[0]["sector"] == "Financials"

    @pytest.mark.asyncio
    async def test_only_ml_active_filtra_por_modelo(self, mock_conn):
        """
        Verifica el COMPORTAMIENTO: todos los tickers retornados tienen
        modelo_asignado IS NOT NULL (no solo que $2=True llego a la query).
        """
        pool, conn = mock_conn
        # El mock simula que la query ya filtro: solo viene el que tiene modelo
        conn.fetch.return_value = make_records(
            {"ticker": "AAPL", "sector": "Technology",
             "industry": "Consumer Electronics",
             "modelo_asignado": "rf_v3", "activo": True},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tickers(only_ml_active=True)

        assert len(result) == 1
        assert result[0]["modelo_asignado"] is not None

    @pytest.mark.asyncio
    async def test_db_error_devuelve_error_legible_postgres(self, mock_conn):
        pool, conn = mock_conn
        conn.fetch.side_effect = Exception("could not connect to server: Connection refused")

        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tickers()

        assert "error" in result[0]
        assert len(result[0]["error"]) > 0

    @pytest.mark.asyncio
    async def test_db_error_devuelve_error_legible_oserror(self, mock_conn):
        pool, conn = mock_conn
        conn.fetch.side_effect = OSError("No route to host")

        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tickers()

        assert "error" in result[0]

    @pytest.mark.asyncio
    async def test_resultado_es_lista_de_dicts(self, mock_conn):
        pool, conn = mock_conn
        conn.fetch.return_value = make_records(
            {"ticker": "MSFT", "sector": "Technology",
             "industry": "Software", "modelo_asignado": None, "activo": True},
        )
        with patch("mcp_server.tools.exploration.get_pool",
                   AsyncMock(return_value=pool)):
            result = await list_tickers()

        assert isinstance(result, list)
        assert isinstance(result[0], dict)


# ── Tests de integracion (requieren DATABASE_URL + asyncpg) ───────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_tables_integration():
    """Verifica que precios_diarios esta en la DB real."""
    result = await list_tables()
    table_names = [r["table_name"] for r in result if "table_name" in r]
    assert "precios_diarios" in table_names
    assert "activos" in table_names
    assert len(table_names) >= 10  # al menos 10 tablas


@pytest.mark.integration
@pytest.mark.asyncio
async def test_describe_precios_diarios_integration():
    """Verifica que precios_diarios tiene las columnas esperadas."""
    result = await describe_table("precios_diarios")
    assert "error" not in result
    col_names = [c["name"] for c in result["columns"]]
    for expected in ("fecha", "ticker", "close"):
        assert expected in col_names, f"columna '{expected}' no encontrada"
    assert result["row_count_estimate"] is not None
    assert result["row_count_estimate"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_tickers_integration():
    """Verifica que hay al menos 199 tickers y todos tienen clave ticker."""
    result = await list_tickers()
    assert len(result) >= 199
    for row in result:
        assert "ticker" in row
        assert "error" not in row
