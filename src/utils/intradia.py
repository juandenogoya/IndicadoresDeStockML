"""
intradia.py -- barras intradiarias de 30 min para el snapshot de los cortes.

Modulo PURO (pandas + stdlib): no importa DB, config ni pipeline. Lo usan el
snapshot de Oracle (scripts/intradia/snapshot_intradia.py), el sync a local y la
evaluacion de la etapa 1. Diseno y decisiones: docs/intradia.md.

QUE SE GUARDA. Cada captura guarda las barras de 30 min de la rueda desde la
apertura hasta el corte, en crudo; el OHLCV acumulado hasta el corte sale de
`agregar_corte()`, no de otra tabla. Los indicadores todavia no estan decididos:
guardar solo un resumen decidiria hoy que se pierde (ej. comparar el volumen de
las 11:00 ET contra lo normal de ESA hora pide el volumen de cada barra).

POR CAPTURA, no por barra. La clave es (ticker, corte, inicio_barra): la barra de
las 9:30 queda dos veces, como la vio el corte apertura_90 y como la vio el
cierre_60. Asi la fila es un registro POINT-IN-TIME de lo que el sistema sabia en
cada corte (si Yahoo revisa una barra, el cierre_60 no pisa lo que vio la
apertura) y la diferencia entre las dos copias mide esas revisiones.

TRAMPAS QUE RESUELVE:
- Barra EN CURSO: yahooquery (y IBKR) devuelven la barra que todavia no cerro.
  Solo se guardan barras que TERMINAN en el corte o antes. Misma trampa que
  yahooquery_loader con la barra diaria de hoy (CLAUDE.md, regla 10).
- Zonas horarias: el indice de yahooquery puede venir aware (hora de NY) o
  naive. Naive se interpreta como hora de NY (la del exchange de estos tickers);
  es un SUPUESTO que la primera corrida en la VM verifica (docs/intradia.md).
  Cada entrada se normaliza por separado: pandas 3.x no coacciona un indice con
  zonas mezcladas (incidente 18/6/2026 del loader diario).
- `fecha` = la rueda de NY a la que pertenece la barra, nunca la fecha de la
  captura ni la de UTC (familia de scan_fecha / fecha_datos en CLAUDE.md).

SPOOL: igual que el de opciones (src/utils/opciones_spool.py), un archivo en el
directorio = dato PENDIENTE de persistir. Se escribe ANTES de la DB y se borra
cuando la DB confirmo. Aca el lote es chico (~2.200 filas) y llega de una sola
descarga, asi que se escribe de una vez en vez de en streaming.
"""

import csv
import gzip
import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd

from src.utils.sesion_nyse import BARRA_MIN, CORTES, TZ_NY

TABLA = "precios_intradia_30m"

DDL_TABLA = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    ticker        VARCHAR(20)   NOT NULL,
    fecha         DATE          NOT NULL,   -- rueda de NY de la barra
    corte         VARCHAR(16)   NOT NULL,   -- captura que la guardo: apertura_90 | cierre_60
    inicio_barra  TIMESTAMPTZ   NOT NULL,   -- inicio de la barra de 30 min
    open          NUMERIC(14,4),
    high          NUMERIC(14,4),
    low           NUMERIC(14,4),
    close         NUMERIC(14,4),
    volumen       BIGINT,
    hora_corte    TIMESTAMPTZ   NOT NULL,   -- instante del corte (ultima barra termina ahi)
    capturado_en  TIMESTAMPTZ   NOT NULL,   -- cuando corrio la captura (reloj, NO fecha de datos)
    PRIMARY KEY (ticker, corte, inicio_barra)
);
CREATE INDEX IF NOT EXISTS idx_{TABLA}_fecha_corte ON {TABLA} (fecha, corte);
"""

COLUMNAS = [
    "ticker", "fecha", "corte", "inicio_barra",
    "open", "high", "low", "close", "volumen",
    "hora_corte", "capturado_en",
]

# ON CONFLICT sobre la PK completa (CLAUDE.md: requiere unique index FULL).
# Un reintento del MISMO corte actualiza con lo ultimo que dio Yahoo.
SQL_UPSERT = f"""
    INSERT INTO {TABLA} ({", ".join(COLUMNAS)}) VALUES %s
    ON CONFLICT (ticker, corte, inicio_barra) DO UPDATE SET
        fecha        = EXCLUDED.fecha,
        open         = EXCLUDED.open,
        high         = EXCLUDED.high,
        low          = EXCLUDED.low,
        close        = EXCLUDED.close,
        volumen      = EXCLUDED.volumen,
        hora_corte   = EXCLUDED.hora_corte,
        capturado_en = EXCLUDED.capturado_en
"""

DIR_SPOOL_DEFAULT = os.path.join("data", "intradia_spool")


# ── Normalizacion de la descarga ──────────────────────────────────────────────

def _instante_utc(v) -> Optional[datetime]:
    """Una entrada del indice de yahooquery -> datetime aware en UTC."""
    try:
        ts = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(TZ_NY)
    return ts.tz_convert("UTC").to_pydatetime()


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _por_ticker(df: pd.DataFrame, tickers: list[str]) -> Iterable[tuple[str, pd.DataFrame]]:
    """
    Separa la descarga por ticker. yahooquery devuelve MultiIndex (symbol, date)
    si hubo datos para algun ticker, o indice plano si se pidio uno solo.
    """
    if isinstance(df.index, pd.MultiIndex):
        for sym in df.index.get_level_values(0).unique():
            yield str(sym), df.xs(sym, level=0)
    elif len(tickers) == 1:
        yield tickers[0], df


def normalizar_barras(df, tickers: list[str], fecha: date, corte: str,
                      inicio_sesion: datetime, hora_corte: datetime,
                      capturado_en: datetime) -> list[dict]:
    """
    Descarga cruda de yahooquery (history interval=30m) -> filas para la tabla.

    Se quedan solo las barras de la rueda `fecha` que arrancan en la apertura o
    despues y TERMINAN en el corte o antes. Descarta la barra en curso, el
    pre-market, otras ruedas y las barras sin close.
    """
    if corte not in CORTES:
        raise ValueError(f"corte desconocido: {corte!r}")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    df = df.rename(columns=str.lower)
    ultimo_inicio = hora_corte - timedelta(minutes=BARRA_MIN)
    filas = []
    for ticker, sub in _por_ticker(df, tickers):
        for idx, row in sub.iterrows():
            inicio = _instante_utc(idx)
            if inicio is None or inicio < inicio_sesion or inicio > ultimo_inicio:
                continue
            close = _num(row.get("close"))
            if close is None:
                continue
            vol = _num(row.get("volume"))
            filas.append({
                "ticker": ticker,
                "fecha": fecha,
                "corte": corte,
                "inicio_barra": inicio,
                "open": _num(row.get("open")),
                "high": _num(row.get("high")),
                "low": _num(row.get("low")),
                "close": close,
                "volumen": int(vol) if vol is not None else None,
                "hora_corte": hora_corte,
                "capturado_en": capturado_en,
            })
    filas.sort(key=lambda f: (f["ticker"], f["inicio_barra"]))
    return filas


# ── Agregados y cobertura ─────────────────────────────────────────────────────

def barras_esperadas(inicio_sesion: datetime, hora_corte: datetime) -> int:
    """Barras de 30 min entre la apertura y el corte (3, 11 o 5 en media rueda)."""
    return int((hora_corte - inicio_sesion).total_seconds() // (BARRA_MIN * 60))


def agregar_corte(filas: list[dict]) -> Optional[dict]:
    """
    OHLCV de la sesion hasta el corte, desde las barras de UN ticker y UN corte.
    open = el de la primera barra, close = el de la ultima. None si no hay barras.
    """
    if not filas:
        return None
    fs = sorted(filas, key=lambda f: f["inicio_barra"])
    highs = [f["high"] for f in fs if f["high"] is not None]
    lows = [f["low"] for f in fs if f["low"] is not None]
    vols = [f["volumen"] for f in fs if f["volumen"] is not None]
    return {
        "open": fs[0]["open"],
        "high": max(highs) if highs else None,
        "low": min(lows) if lows else None,
        "close": fs[-1]["close"],
        "volumen": sum(vols) if vols else None,
        "n_barras": len(fs),
        "ultima_barra": fs[-1]["inicio_barra"],
    }


def cobertura(filas: list[dict], universo: list[str], hora_corte: datetime) -> dict:
    """
    Cuantos tickers del universo tienen la barra que TERMINA en el corte.

    Un ticker sin esa barra no sirve para el corte aunque tenga las anteriores:
    su "precio del corte" seria de media hora o mas atras.
    """
    ultimo_inicio = hora_corte - timedelta(minutes=BARRA_MIN)
    con_datos = {f["ticker"] for f in filas}
    completos = {f["ticker"] for f in filas if f["inicio_barra"] == ultimo_inicio}
    univ = sorted(set(universo))
    n = len(univ)
    return {
        "esperados": n,
        "completos": sorted(completos & set(univ)),
        "sin_ultima_barra": sorted((con_datos - completos) & set(univ)),
        "sin_datos": sorted(set(univ) - con_datos),
        "pct": (len(completos & set(univ)) / n) if n else 0.0,
    }


def precios_del_corte(filas: list[dict], hora_corte: datetime) -> dict:
    """
    {ticker: precio del corte} = close de la barra que TERMINA en el corte.
    Un ticker sin esa barra no tiene precio del corte (mismo criterio que
    `cobertura`): su ultimo close seria de media hora o mas atras.
    """
    ultimo_inicio = hora_corte - timedelta(minutes=BARRA_MIN)
    return {f["ticker"]: f["close"] for f in filas
            if f["inicio_barra"] == ultimo_inicio and f["close"] is not None}


# ── Spool en disco ────────────────────────────────────────────────────────────

def ruta_spool(fecha: date, corte: str, dir_spool: str = DIR_SPOOL_DEFAULT) -> str:
    return os.path.join(dir_spool, f"intradia_{fecha.isoformat()}_{corte}.csv.gz")


def escribir_spool(filas: list[dict], fecha: date, corte: str,
                   dir_spool: str = DIR_SPOOL_DEFAULT) -> str:
    """
    Vuelca las filas a disco ANTES de la DB. Si ya hay un spool de ese corte
    (reintento con la DB todavia caida) se escribe al lado con sufijo de hora:
    el upsert es idempotente y el ultimo en aplicarse gana.
    """
    os.makedirs(dir_spool, exist_ok=True)
    path = ruta_spool(fecha, corte, dir_spool)
    if os.path.exists(path):
        sufijo = datetime.now(timezone.utc).strftime("%H%M%S")
        path = path.replace(".csv.gz", f"_{sufijo}.csv.gz")
    # newline="" es requisito de csv en Windows (evita \r\r\n).
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, extrasaction="ignore")
        w.writeheader()
        for f in filas:
            w.writerow({k: ("" if f.get(k) is None else
                            f[k].isoformat() if isinstance(f[k], (date, datetime)) else f[k])
                        for k in COLUMNAS})
    return path


def leer_spool(path: str) -> list[dict]:
    """Lee un spool con los tipos restaurados (el CSV es todo texto)."""
    def dt(v):
        return datetime.fromisoformat(v) if v else None

    def fl(v):
        return float(v) if v != "" else None

    filas = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            filas.append({
                "ticker": r["ticker"],
                "fecha": date.fromisoformat(r["fecha"]),
                "corte": r["corte"],
                "inicio_barra": dt(r["inicio_barra"]),
                "open": fl(r["open"]),
                "high": fl(r["high"]),
                "low": fl(r["low"]),
                "close": fl(r["close"]),
                "volumen": int(r["volumen"]) if r["volumen"] != "" else None,
                "hora_corte": dt(r["hora_corte"]),
                "capturado_en": dt(r["capturado_en"]),
            })
    return filas


def listar_spools(dir_spool: str = DIR_SPOOL_DEFAULT) -> list[str]:
    """Spools PENDIENTES (ver invariante), en orden de nombre (fecha, corte)."""
    if not os.path.isdir(dir_spool):
        return []
    return sorted(
        os.path.join(dir_spool, f) for f in os.listdir(dir_spool)
        if f.startswith("intradia_") and f.endswith(".csv.gz")
    )


def valores_db(filas: list[dict]) -> list[tuple]:
    """Filas -> tuplas en el orden de COLUMNAS, para execute_values con SQL_UPSERT."""
    return [tuple(f[c] for c in COLUMNAS) for f in filas]
