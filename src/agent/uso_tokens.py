"""
Registro de consumo de tokens del chat (tabla local llm_uso_tokens).

Una fila por consulta con los tokens REALES que reporta la API (entrada/salida).
Reutilizable por cualquier frontend del orquestador (dashboard hoy; Telegram a
futuro). Escribe en la DB LOCAL con un engine propio (nunca el rol mcp_reader,
que es SELECT-only). Todas las operaciones son best-effort: si fallan, devuelven
False/[] y NUNCA rompen el chat.
"""

from functools import lru_cache

from dotenv import dotenv_values
from sqlalchemy import create_engine, text

from src.agent.config import REPO_ROOT

USUARIO_DEFAULT = "local"  # multiusuario a futuro (cuotas por usuario)


@lru_cache(maxsize=1)
def _engine():
    """Engine a la DB local (lee .env sin tocar os.environ)."""
    env = dotenv_values(REPO_ROOT / ".env")
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    name = env.get("DB_NAME", "activos_ml")
    user = env.get("DB_USER", "postgres")
    pwd = env.get("DB_PASSWORD", "")
    return create_engine(
        f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}",
        pool_pre_ping=True,
    )


_INSERT = text("""
    INSERT INTO llm_uso_tokens
        (usuario, modelo, tokens_entrada, tokens_salida, tokens_total,
         n_rondas, tools, pregunta)
    VALUES
        (:usuario, :modelo, :tin, :tout, :ttot, :n, :tools, :pregunta)
""")

_RESUMEN = text("""
    SELECT fecha,
           COUNT(*)                          AS consultas,
           COALESCE(SUM(tokens_entrada), 0)  AS entrada,
           COALESCE(SUM(tokens_salida), 0)   AS salida,
           COALESCE(SUM(tokens_total), 0)    AS total
    FROM llm_uso_tokens
    WHERE (:usuario IS NULL OR usuario = :usuario)
    GROUP BY fecha
    ORDER BY fecha DESC
    LIMIT :dias
""")

_DETALLE = text("""
    SELECT ts, modelo, tokens_entrada, tokens_salida, tokens_total,
           n_rondas, tools, pregunta
    FROM llm_uso_tokens
    WHERE fecha = :fecha AND (:usuario IS NULL OR usuario = :usuario)
    ORDER BY ts DESC
""")


def registrar_uso(usuario: str, modelo, tokens_in, tokens_out, tokens_total,
                  n_rondas, tools, pregunta) -> bool:
    """Inserta una fila de consumo. Best-effort: devuelve True/False, no levanta."""
    try:
        with _engine().begin() as conn:
            conn.execute(_INSERT, {
                "usuario": usuario or USUARIO_DEFAULT,
                "modelo": modelo,
                "tin": int(tokens_in or 0),
                "tout": int(tokens_out or 0),
                "ttot": int(tokens_total or 0),
                "n": int(n_rondas or 0),
                "tools": tools,
                "pregunta": pregunta,
            })
        return True
    except Exception:
        return False


def resumen_por_fecha(usuario: str | None = None, dias: int = 30) -> list[dict]:
    """Consumo agregado por fecha (ultimos `dias` dias con actividad)."""
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                _RESUMEN, {"usuario": usuario, "dias": dias}
            ).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        return []


def detalle_por_fecha(fecha, usuario: str | None = None) -> list[dict]:
    """Consultas individuales de un dia (para auditar cual gasto mas)."""
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                _DETALLE, {"fecha": fecha, "usuario": usuario}
            ).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        return []
