"""
universo.py -- Fuente de verdad UNICA del universo de tickers.

Antes el universo se servia de dos lugares desincronizados:
  - la tabla `activos` (respeta activo=TRUE) -> la leen recovery, cron, etc.
  - la lista hardcodeada config.ALL_TICKERS -> la leian snapshot opciones,
    scanner, refresh fundamentales/pais.
Resultado: dar de alta/baja un ticker requeria tocar dos lugares y, peor, los
procesos que leian config NO veian el alta/baja hecho en la tabla.

Este modulo centraliza la lectura desde `activos`. Asi, modificar la tabla
(activo=TRUE/FALSE, o un INSERT/DELETE) se refleja en TODO el pipeline vivo.

Detalles:
- Lee de la DB a la que apunte get_engine() (local o Railway). `activos` vive
  en AMBAS, sincronizada; cada script lee la suya segun su entorno.
- Fallback de seguridad: si `activos` esta vacia o inaccesible, cae a
  config.ALL_TICKERS con un WARN. El switch nunca deja al pipeline sin universo.
- config.ALL_TICKERS NO se elimina: lo usan scripts legacy ML/BT y queda como
  red de seguridad.
"""

from sqlalchemy import text

from src.data.database import get_engine


def get_universo(solo_activos: bool = True) -> list:
    """
    Lista de tickers del universo desde la tabla `activos` (fuente de verdad),
    ordenada alfabeticamente.

    Args:
        solo_activos: True (default) -> solo activo=TRUE (el universo vivo);
                      False -> incluye los dados de baja (activo=FALSE).

    Returns:
        list[str] de tickers. Fallback a config.ALL_TICKERS si `activos` no
        responde o esta vacia.
    """
    try:
        eng = get_engine()
        sql = "SELECT ticker FROM activos"
        if solo_activos:
            sql += " WHERE activo = TRUE"
        sql += " ORDER BY ticker"
        with eng.connect() as conn:
            tickers = [r[0] for r in conn.execute(text(sql)).fetchall()]
        if tickers:
            return tickers
        print("  [universo][WARN] tabla activos vacia -> fallback a config.ALL_TICKERS")
    except Exception as e:
        print(f"  [universo][WARN] no se pudo leer activos ({e}) "
              f"-> fallback a config.ALL_TICKERS")

    from src.utils.config import ALL_TICKERS
    return sorted(set(ALL_TICKERS))


def get_universo_sectores(solo_activos: bool = True) -> dict:
    """
    Mapa {ticker: sector} desde `activos`, para los consumidores que agrupan por
    sector. Fallback a config.TICKER_SECTOR (parcial: solo los tickers ML viejos).
    """
    try:
        eng = get_engine()
        sql = "SELECT ticker, sector FROM activos"
        if solo_activos:
            sql += " WHERE activo = TRUE"
        with eng.connect() as conn:
            out = {r[0]: r[1] for r in conn.execute(text(sql)).fetchall()}
        if out:
            return out
        print("  [universo][WARN] tabla activos vacia -> fallback a config.TICKER_SECTOR")
    except Exception as e:
        print(f"  [universo][WARN] no se pudo leer activos sectores ({e}) "
              f"-> fallback a config.TICKER_SECTOR")

    from src.utils.config import TICKER_SECTOR
    return dict(TICKER_SECTOR)
