"""
Fixtures y configuracion compartida para todos los tests del proyecto.

Este conftest vive en tests/ (directorio sin __init__.py), lo que hace
que pytest NO inserte tests/ en sys.path al cargarlo. Combinado con
pythonpath=. en pytest.ini, repo_root/ queda en sys.path correctamente
y mcp_server.* se resuelve al paquete fuente (no a tests/mcp_server/).

Incluye:
  - clear_settings_cache: limpia lru_cache de get_settings() entre tests.
  - mock_conn: fixture de pool/conn mockeado para tests unitarios de tools.
  - Registro del marker "integration" y auto-skip si no hay DATABASE_URL.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_server.config import get_settings
import mcp_server.db.pool as _pool_module


# ── Cache de settings ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_db_pool():
    """
    Cierra y resetea el singleton del pool de asyncpg despues de cada test.

    pytest-asyncio (strict, function scope) crea un nuevo event loop por test.
    Si el pool fue creado en el loop del test anterior, el siguiente test falla
    con "Event loop is closed" o "connection was closed in the middle of operation".

    Estrategia:
      1. Captura la referencia al pool actual (puede ser None en unit tests).
      2. Resetea _pool = None ANTES de cerrar, para que cualquier llamada
         concurrente no vea el pool a medio cerrar.
      3. Intenta cerrar el pool con asyncio.run() (crea un event loop nuevo
         solo para el cleanup). Best-effort: si falla, el GC lo limpia.
    """
    yield
    import asyncio
    pool = _pool_module._pool
    _pool_module._pool = None
    if pool is not None:
        try:
            asyncio.run(pool.close())
        except Exception:
            pass  # best-effort: el GC cerrara los sockets eventualmente


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """
    Limpia el cache de get_settings() antes y despues de cada test.

    Evita que un test que setea DATABASE_URL en el env contamine el
    siguiente test, o que el cache retenga un valor invalido.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Marker de integracion ─────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: test que requiere conexion a DB real (DATABASE_URL en env)",
    )


def pytest_collection_modifyitems(items):
    """Skipea @pytest.mark.integration si DATABASE_URL no esta en el entorno."""
    if os.environ.get("DATABASE_URL"):
        return

    skip_no_db = pytest.mark.skip(
        reason="DATABASE_URL no esta en el entorno -- skip integration tests"
    )
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_no_db)


# ── Fixture de pool mockeado ──────────────────────────────────────────────────

@pytest.fixture
def mock_conn():
    """
    Retorna (pool_mock, conn_mock) listos para parchear get_pool().

    conn_mock.fetch y conn_mock.fetchrow son AsyncMock: se les puede
    setear .return_value o .side_effect para controlar los resultados.

    Uso tipico:
        async def test_algo(mock_conn):
            pool, conn = mock_conn
            conn.fetch.return_value = [{"col": "val"}]
            with patch("mcp_server.tools.exploration.get_pool",
                       AsyncMock(return_value=pool)):
                result = await list_tables()
    """
    conn = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchrow = AsyncMock()

    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm

    return pool, conn
