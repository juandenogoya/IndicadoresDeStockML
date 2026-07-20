"""
opciones_spool.py -- red de seguridad en disco para el snapshot de opciones.

MOTIVO (incidente 2026-07-20): Railway se detuvo por limite de consumo. El
snapshot de opciones (33_opciones_snapshot.py) persistia cada ticker
directamente a la DB; con la DB caida, cada ticker levantaba excepcion y la
chain YA DESCARGADA se descartaba. Como Yahoo solo sirve la chain VIGENTE, ese
dato es IRRECUPERABLE al dia siguiente.

Este modulo desacopla CAPTURAR de ALMACENAR: el snapshot vuelca las filas a un
.csv.gz en disco ANTES de intentar la DB. Si la DB falla, el archivo queda y se
reinyecta despues con scripts/manual/replay_opciones_spool.py.

INVARIANTE: un archivo presente en el directorio de spool = dato PENDIENTE de
persistir. Cuando la DB confirma la escritura completa, el spool se borra.
Asi "hay archivos en el spool" es, por si solo, la senal de que algo quedo sin
sincronizar (lo usa el replay y sirve para alertar).

Modulo PURO: no importa DB, config ni nada del pipeline. Escritura en streaming
(no acumula en memoria) para que un corte a mitad de corrida conserve lo bajado
hasta ese momento.
"""

import csv
import gzip
import os
from datetime import date, datetime
from typing import Iterator, Optional

# Orden fijo de columnas del crudo de opciones_snapshot. NO reordenar: el
# replay lee posicionalmente por nombre, pero mantener el orden hace los
# archivos diffeables y estables entre versiones.
COLUMNAS = [
    "fecha_snapshot", "ticker", "vencimiento", "tipo", "strike",
    "volumen", "open_interest", "iv", "bid", "ask",
    "precio_subyacente", "hv_20d",
]

DIR_SPOOL_DEFAULT = os.path.join("data", "opciones_spool")


def ruta_spool(fecha: date, dir_spool: str = DIR_SPOOL_DEFAULT) -> str:
    """Ruta del archivo de spool de una fecha de snapshot."""
    return os.path.join(dir_spool, f"opciones_{fecha.isoformat()}.csv.gz")


class SpoolWriter:
    """
    Escritor incremental del crudo de opciones a .csv.gz.

    Uso:
        with SpoolWriter(fecha) as sp:
            for ticker in tickers:
                sp.write(filas_del_ticker)
        print(sp.filas, sp.path)

    Si el archivo de la fecha YA existe (rerun del mismo dia), se escribe a un
    archivo con sufijo de hora para no pisar la captura previa. El replay toma
    todos los archivos del directorio y el upsert es idempotente
    (ON CONFLICT DO NOTHING), asi que duplicar no rompe nada.
    """

    def __init__(self, fecha: date, dir_spool: str = DIR_SPOOL_DEFAULT):
        self.fecha = fecha
        self.dir_spool = dir_spool
        os.makedirs(dir_spool, exist_ok=True)

        path = ruta_spool(fecha, dir_spool)
        if os.path.exists(path):
            sufijo = datetime.now().strftime("%H%M%S")
            path = os.path.join(dir_spool, f"opciones_{fecha.isoformat()}_{sufijo}.csv.gz")
        self.path = path

        self.filas = 0
        self._fh = None
        self._writer = None

    def __enter__(self):
        self.abrir()
        return self

    def __exit__(self, *exc):
        self.cerrar()
        return False

    def abrir(self):
        # newline="" es requisito de csv en Windows (evita \r\r\n).
        self._fh = gzip.open(self.path, "wt", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNAS,
                                      extrasaction="ignore")
        self._writer.writeheader()

    def write(self, filas: list[dict]) -> int:
        """Vuelca las filas de un ticker y hace flush (durabilidad por ticker)."""
        if not filas or self._writer is None:
            return 0
        self._writer.writerows(filas)
        self._fh.flush()
        self.filas += len(filas)
        return len(filas)

    def cerrar(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

    def descartar(self):
        """Borra el spool. Se llama SOLO cuando la DB confirmo todo (ver invariante)."""
        self.cerrar()
        if os.path.exists(self.path):
            os.remove(self.path)


def _parse_num(v: str, tipo):
    """'' -> None; si no, castea. El CSV serializa None como cadena vacia."""
    if v is None or v == "":
        return None
    try:
        return tipo(v)
    except (TypeError, ValueError):
        return None


def leer_spool(path: str) -> Iterator[dict]:
    """
    Lee un .csv.gz de spool y devuelve filas listas para el upsert, con los
    tipos restaurados (el CSV es todo texto).
    """
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            yield {
                "fecha_snapshot":    row["fecha_snapshot"] or None,
                "ticker":            row["ticker"],
                "vencimiento":       row["vencimiento"] or None,
                "tipo":              row["tipo"],
                "strike":            _parse_num(row["strike"], float),
                "volumen":           _parse_num(row["volumen"], int),
                "open_interest":     _parse_num(row["open_interest"], int),
                "iv":                _parse_num(row["iv"], float),
                "bid":               _parse_num(row["bid"], float),
                "ask":               _parse_num(row["ask"], float),
                "precio_subyacente": _parse_num(row["precio_subyacente"], float),
                "hv_20d":            _parse_num(row["hv_20d"], float),
            }


def listar_spools(dir_spool: str = DIR_SPOOL_DEFAULT) -> list[str]:
    """Spools PENDIENTES (ver invariante), ordenados por fecha de captura."""
    if not os.path.isdir(dir_spool):
        return []
    archivos = [
        os.path.join(dir_spool, f)
        for f in os.listdir(dir_spool)
        if f.startswith("opciones_") and f.endswith(".csv.gz")
    ]
    return sorted(archivos)


def fecha_de_spool(path: str) -> Optional[date]:
    """Extrae la fecha del nombre del archivo ('opciones_2026-07-20[_HHMMSS].csv.gz')."""
    base = os.path.basename(path).replace(".csv.gz", "")
    partes = base.split("_")
    if len(partes) < 2:
        return None
    try:
        return date.fromisoformat(partes[1])
    except ValueError:
        return None
