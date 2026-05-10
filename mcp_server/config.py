"""
Configuracion del MCP server via pydantic-settings.

Lee variables de entorno seteadas por el cliente MCP (Gemini CLI,
Claude Desktop, etc.). NO lee .env directamente: el cliente es
responsable de inyectar las variables via su settings.json.

Variables:
  DATABASE_URL          (requerida) DSN del rol mcp_reader.
                        Formato: postgresql://mcp_reader:<pwd>@host:port/db
  QUERIES_CATALOG_PATH  (opcional) Ruta al repo ~/queries-catalog/.
  LOG_LEVEL             (opcional) DEBUG | INFO | WARNING. Default: INFO.

Si DATABASE_URL no esta en el entorno, Settings() lanza ValidationError
con mensaje claro al primer intento de conexion (no al importar).
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Requerida: falla con ValidationError si ausente o vacia
    database_url: str

    # Opcionales
    queries_catalog_path: str | None = None
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=None,        # NO leer .env -- el cliente MCP setea el env
        case_sensitive=False, # DATABASE_URL y database_url son equivalentes
    )

    @field_validator("database_url")
    @classmethod
    def database_url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "DATABASE_URL no puede estar vacia. "
                "Configurar en settings.json del cliente MCP."
            )
        return v

    @field_validator("log_level")
    @classmethod
    def log_level_valid(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL debe ser uno de {valid}, recibido: {v!r}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna la instancia de Settings (singleton via lru_cache).

    La validacion ocurre en el primer llamado, no al importar el modulo.
    Esto permite que el server arranque y responda ping aunque DATABASE_URL
    no este configurada: el error aparece al primer uso de una tool de DB.

    Para limpiar el cache en tests: get_settings.cache_clear()
    """
    return Settings()
