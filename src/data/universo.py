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

CADENA DE FALLBACK (3 niveles, 20/6/2026 -- incidente Railway):
  1. tabla `activos`  -- fuente de verdad. Cada lectura exitosa REFRESCA el cache.
  2. cache en disco   -- espejo de la ultima lectura buena de ESA maquina.
  3. config.ALL_TICKERS -- ultimo recurso, solo si nunca hubo cache.

El nivel 2 existe porque el nivel 3 se DESINCRONIZA EN SILENCIO: `universo.py add`
escribe en `activos` y no toca config, asi que cada alta aleja la lista
hardcodeada de la realidad (HOOD, alta del 18/6/2026, nunca entro a ALL_TICKERS
-> con Railway caido el snapshot habria capturado 199 de 200 tickers sin avisar).
El cache se auto-cura: no depende de que nadie se acuerde de editar el config.

El cache vive en data/cache/ (gitignored) y es POR MAQUINA a proposito: refleja
la DB que ve ese host (Oracle->Railway, Windows->local), no un estado compartido.
"""

import json
import os
from datetime import datetime, timezone

from sqlalchemy import text

from src.data.database import get_engine

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_PATH = os.path.join(_ROOT, "data", "cache", "universo.json")


# ── Cache en disco ────────────────────────────────────────────────────────────

def _guardar_cache(tickers: list, sectores: dict):
    """
    Persiste la ultima lectura buena. Best-effort: que falle el cache NUNCA debe
    romper al llamador (el dato ya lo tenemos en memoria).

    Escritura atomica (tmp + replace) para que un corte no deje un JSON truncado
    que despues no parsee.
    """
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        payload = {
            "actualizado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n": len(tickers),
            "tickers": tickers,
            "sectores": sectores,
        }
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True, indent=1)
        os.replace(tmp, CACHE_PATH)
    except Exception as e:
        print(f"  [universo][WARN] no se pudo escribir el cache ({e})")


def _leer_cache() -> dict:
    """Devuelve el payload del cache, o {} si no existe / esta corrupto."""
    try:
        with open(CACHE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("tickers"):
            return data
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  [universo][WARN] cache ilegible ({e})")
    return {}


def _antiguedad(data: dict) -> str:
    """Texto legible de cuan viejo es el cache (para que el WARN sea accionable)."""
    try:
        ts = datetime.fromisoformat(data["actualizado"])
        dias = (datetime.now(timezone.utc) - ts).days
        return f"{data['actualizado']} ({dias}d)"
    except Exception:
        return "fecha desconocida"


def _leer_activos(solo_activos: bool) -> tuple:
    """Lee (tickers, {ticker: sector}) de `activos` en UNA query."""
    eng = get_engine()
    sql = "SELECT ticker, sector FROM activos"
    if solo_activos:
        sql += " WHERE activo = TRUE"
    sql += " ORDER BY ticker"
    with eng.connect() as conn:
        filas = conn.execute(text(sql)).fetchall()
    return [r[0] for r in filas], {r[0]: r[1] for r in filas}


# ── API publica ───────────────────────────────────────────────────────────────

def get_universo(solo_activos: bool = True) -> list:
    """
    Lista de tickers del universo desde la tabla `activos` (fuente de verdad),
    ordenada alfabeticamente.

    Args:
        solo_activos: True (default) -> solo activo=TRUE (el universo vivo);
                      False -> incluye los dados de baja (activo=FALSE).

    Returns:
        list[str] de tickers. Ver CADENA DE FALLBACK en el docstring del modulo.
    """
    try:
        tickers, sectores = _leer_activos(solo_activos)
        if tickers:
            # Solo cacheamos el universo VIVO: es el que consumen los procesos
            # que necesitan el fallback. solo_activos=False es para auditoria.
            if solo_activos:
                _guardar_cache(tickers, sectores)
            return tickers
        print("  [universo][WARN] tabla activos vacia")
    except Exception as e:
        print(f"  [universo][WARN] no se pudo leer activos ({e})")

    cache = _leer_cache()
    if cache:
        print(f"  [universo] usando CACHE en disco: {cache['n']} tickers, "
              f"actualizado {_antiguedad(cache)}")
        return list(cache["tickers"])

    from src.utils.config import ALL_TICKERS
    print("  [universo][WARN] sin cache -> fallback a config.ALL_TICKERS "
          "(lista hardcodeada, puede estar desactualizada vs `activos`)")
    return sorted(set(ALL_TICKERS))


def get_universo_sectores(solo_activos: bool = True) -> dict:
    """
    Mapa {ticker: sector} desde `activos`, para los consumidores que agrupan por
    sector. Misma cadena de fallback que get_universo().
    """
    try:
        tickers, sectores = _leer_activos(solo_activos)
        if sectores:
            if solo_activos:
                _guardar_cache(tickers, sectores)
            return sectores
        print("  [universo][WARN] tabla activos vacia")
    except Exception as e:
        print(f"  [universo][WARN] no se pudo leer activos sectores ({e})")

    cache = _leer_cache()
    if cache.get("sectores"):
        print(f"  [universo] sectores desde CACHE en disco: "
              f"actualizado {_antiguedad(cache)}")
        return dict(cache["sectores"])

    from src.utils.config import TICKER_SECTOR
    print("  [universo][WARN] sin cache -> fallback a config.TICKER_SECTOR "
          "(parcial: solo los tickers ML viejos)")
    return dict(TICKER_SECTOR)


def refrescar_cache() -> dict:
    """
    Fuerza una lectura de `activos` y reescribe el cache. Util para sembrarlo en
    una maquina nueva, o desde un host que SI ve la DB para dejar lista la red de
    seguridad antes de un corte.

    Returns: el payload cacheado. Levanta si la DB no responde (aca el fallback
    NO tiene sentido: el objetivo es justamente refrescar contra la fuente).
    """
    tickers, sectores = _leer_activos(solo_activos=True)
    if not tickers:
        raise RuntimeError("tabla activos vacia: no hay nada que cachear")
    _guardar_cache(tickers, sectores)
    return _leer_cache()
