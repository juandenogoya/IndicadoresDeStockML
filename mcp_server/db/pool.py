"""
Pool de conexiones asyncpg -- singleton por proceso.

El import de asyncpg es DIFERIDO (dentro de _init_pool) para que este
modulo pueda importarse sin asyncpg instalado. Util para tests unitarios
que mockean get_pool() y nunca llegan a _init_pool().

Funciones publicas:
  get_pool()    -> Pool   async, lazy init en el primer llamado
  close_pool()  -> None   async, cierra y resetea el singleton
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Solo para type checkers; no ejecuta en runtime
    import asyncpg

_pool: asyncpg.Pool | None = None  # type: ignore[name-defined]


async def get_pool():
    """
    Retorna el pool asyncpg, inicializandolo si es la primera llamada.

    Lazy: el pool se crea al primer uso de una tool de DB, no al
    arrancar el server. Esto permite que ping() responda aunque la
    DB no este disponible.

    Raises:
        ConnectionError: si DATABASE_URL es invalida o la DB no responde.
        pydantic.ValidationError: si DATABASE_URL no esta en el entorno.
    """
    global _pool
    if _pool is None:
        _pool = await _init_pool()
    return _pool


async def _init_pool():
    """
    Crea el pool asyncpg leyendo DATABASE_URL de Settings.

    min_size=1, max_size=5: alineado con CONNECTION LIMIT 5 del rol
    mcp_reader. statement_timeout y read_only ya estan en el rol via
    ALTER ROLE; no se duplican aca.

    El import de asyncpg es diferido para no romper imports en entornos
    donde asyncpg no esta instalado (ej. tests unitarios con mocks).
    """
    import asyncpg  # deferred: solo se ejecuta al conectar por primera vez

    from mcp_server.config import get_settings

    dsn = get_settings().database_url
    try:
        return await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    except OSError as exc:
        raise ConnectionError(
            f"No se pudo conectar a la DB (error de red/socket): {exc}"
        ) from exc
    except asyncpg.PostgresError as exc:
        raise ConnectionError(
            f"No se pudo conectar a la DB (error PostgreSQL): {exc}"
        ) from exc
    except Exception as exc:
        raise ConnectionError(
            f"No se pudo inicializar el pool de DB: {exc}"
        ) from exc


async def close_pool() -> None:
    """
    Cierra el pool y resetea el singleton.

    Llamar al shutdown del server para liberar conexiones limpias.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
