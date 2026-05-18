"""
ft_env.py
Helper de entorno para los scripts de Forward Testing.

Proposito:
    Garantizar que TODOS los scripts FT (bots, setup, utilidades) apunten
    de forma determinista a la DB LOCAL, nunca a Railway.

Contexto (Plan C, 14/5/2026):
    LOCAL PostgreSQL = fuente de verdad para Forward Testing.
    Railway = solo opciones_snapshot (irrecuperable). FT no debe escribir
    en Railway: no es eficiente ni funcional pagar procesamiento de datos
    de prueba en infraestructura remota.

Como funciona get_engine() (src/data/database.py):
    1. Chequea os.getenv("DATABASE_URL") PRIMERO.
       Si esta seteada -> usa esa conexion = Railway.
    2. Si NO esta seteada -> cae a DB_CONFIG (DB_HOST/PORT/USER/PASSWORD
       del .env) = local.

    El problema historico: los scripts FT cargaban .env.local con
    override=True, que setea DATABASE_URL=Railway. Resultado: los bots
    escribian en Railway sin querer.

Solucion:
    configurar_entorno_local() carga SOLO el .env del proyecto (que tiene
    DB_HOST, DB_USER, etc. apuntando a local) y elimina explicitamente
    DATABASE_URL de os.environ. Asi get_engine() siempre cae a local.

Uso (al inicio de cada script FT, despues del sys.path.insert):
    from scripts.forward_testing.ft_env import configurar_entorno_local
    configurar_entorno_local()
"""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def configurar_entorno_local(verbose: bool = False) -> str:
    """
    Configura el entorno para que get_engine() apunte a la DB local.

    Pasos:
        1. Carga el .env del proyecto (DB_HOST, DB_PORT, DB_USER, etc.).
           NO carga .env.local (ahi vive DATABASE_URL=Railway).
        2. Elimina DATABASE_URL de os.environ si quedo seteada por el
           shell, un import previo o el propio .env.

    Es idempotente: se puede llamar varias veces sin efecto adverso.

    Args:
        verbose: si True, imprime el host de DB resuelto (util para logs).

    Returns:
        El DB_HOST resuelto (string), para confirmacion/logging.
    """
    # 1. Cargar .env del proyecto (sin .env.local)
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(ROOT, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
    except ImportError:
        # python-dotenv no instalado: se asume que el entorno ya trae
        # DB_HOST/DB_USER/etc. via variables de shell.
        pass

    # 2. Eliminar DATABASE_URL -> fuerza a get_engine() a usar DB_CONFIG local
    os.environ.pop("DATABASE_URL", None)

    db_host = os.getenv("DB_HOST", "localhost")
    if verbose:
        print(f"[ft_env] Entorno FT configurado a LOCAL (DB_HOST={db_host})")
    return db_host
