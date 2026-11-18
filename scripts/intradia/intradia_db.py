"""
intradia_db.py -- lecturas y escrituras de la linea intradiaria, contra el
engine que se le pase (Railway en el snapshot de Oracle, LOCAL en la
evaluacion). La logica vive en los modulos puros (src/utils/intradia.py y
src/utils/vigilancia_intradia.py); aca solo SQL y la composicion de un corte.

`evaluar_corte` es la UNICA definicion de "que dice el mensaje de un corte":
la usan el snapshot (para enviarlo) y evaluar_etapa2.py (para medirlo), asi
las dos no pueden divergir.
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import text

from src.utils import intradia, sesion_nyse
from src.utils import vigilancia_intradia as vi
from src.utils.trading_calendar import prev_trading_day


def _float(v):
    return None if v is None else float(v)


def ultima_lista(eng, antes_de: date) -> Optional[date]:
    """
    Rueda de la lista de vigilancia mas reciente ANTERIOR a `antes_de`. Sale de
    la cabecera (vigilancia_listas) y no de los tickers: una lista VACIA existe.
    """
    with eng.connect() as c:
        return c.execute(text(f"SELECT MAX(fecha) FROM {vi.TABLA_LISTAS} WHERE fecha < :f"),
                         {"f": antes_de}).scalar()


def cargar_lista(eng, rueda: date) -> tuple[list, list]:
    with eng.connect() as c:
        tks = c.execute(text(
            f"SELECT {', '.join(vi.COLS_TICKERS)} FROM {vi.TABLA_TICKERS} "
            "WHERE fecha = :f ORDER BY ticker"), {"f": rueda}).mappings().all()
        nvs = c.execute(text(
            f"SELECT {', '.join(vi.COLS_NIVELES)} FROM {vi.TABLA_NIVELES} "
            "WHERE fecha = :f ORDER BY ticker, nivel"), {"f": rueda}).mappings().all()
    tickers = [dict(r, close_d=_float(r["close_d"]), sigma_d=_float(r["sigma_d"])) for r in tks]
    niveles = [dict(r, valor=_float(r["valor"])) for r in nvs]
    return tickers, niveles


def cargar_filas_corte(eng, fecha: date, corte: str) -> list[dict]:
    with eng.connect() as c:
        rows = c.execute(text(
            f"SELECT {', '.join(intradia.COLUMNAS)} FROM {intradia.TABLA} "
            "WHERE fecha = :f AND corte = :c"), {"f": fecha, "c": corte}).mappings().all()
    return [dict(r, close=_float(r["close"])) for r in rows]


def envio_previo(eng, fecha: date, corte: str) -> Optional[dict]:
    with eng.connect() as c:
        r = c.execute(text(
            f"SELECT {', '.join(vi.COLS_ENVIOS)} FROM {vi.TABLA_ENVIOS} "
            "WHERE fecha = :f AND corte = :c"), {"f": fecha, "c": corte}).mappings().first()
    return dict(r) if r else None


SQL_UPSERT_ENVIO = f"""
    INSERT INTO {vi.TABLA_ENVIOS} ({", ".join(vi.COLS_ENVIOS)})
    VALUES ({", ".join(":" + c for c in vi.COLS_ENVIOS)})
    ON CONFLICT (fecha, corte) DO UPDATE SET
        fecha_lista = EXCLUDED.fecha_lista, n_vigilados = EXCLUDED.n_vigilados,
        n_eventos = EXCLUDED.n_eventos, texto = EXCLUDED.texto,
        enviado_en = EXCLUDED.enviado_en
"""


def registrar_envio(eng, fila: dict):
    with eng.begin() as c:
        c.execute(text(SQL_UPSERT_ENVIO), fila)


def evaluar_corte(eng, fecha: date, corte: str, hora: datetime,
                  filas: Optional[list] = None,
                  asumir_previo: Optional[bool] = None) -> dict:
    """
    Todo lo que dice el mensaje de un corte.

    filas: barras del corte ya en memoria (el snapshot recien capturado); None
           = se leen de la DB.
    asumir_previo (solo cierre_60): si el mensaje de las 11:00 se considera
           enviado. None = se mira vigilancia_envios (lo que paso de verdad);
           la evaluacion historica pasa True (no hay registro de envios).

    La lista tiene que ser de la rueda habil anterior: con una lista vieja se
    compararia contra niveles de otra rueda, y eso se dice en vez de evaluarlo.
    """
    esperada = prev_trading_day(fecha)
    disponible = ultima_lista(eng, fecha)
    base = {"fecha": fecha, "corte": corte, "esperada": esperada, "fecha_lista": disponible}
    if disponible != esperada:
        return dict(base, ok=False, fecha_lista=None, tickers=[], eventos=[], nuevos=[],
                    deshechos=[], sin_dato=[], hay_previo=None,
                    texto=vi.mensaje_sin_lista(fecha, corte, esperada, disponible))

    tickers, niveles = cargar_lista(eng, disponible)
    if filas is None:
        filas = cargar_filas_corte(eng, fecha, corte)
    precios = intradia.precios_del_corte(filas, hora)
    evs = vi.eventos(tickers, niveles, precios)

    previo, hay_previo = None, None
    if corte == sesion_nyse.CIERRE_60:
        if asumir_previo is None:
            env = envio_previo(eng, fecha, sesion_nyse.APERTURA_90)
            asumir_previo = bool(env and env["fecha_lista"] == disponible)
        h11 = sesion_nyse.hora_corte(fecha, sesion_nyse.APERTURA_90)
        p11 = intradia.precios_del_corte(
            cargar_filas_corte(eng, fecha, sesion_nyse.APERTURA_90), h11)
        hay_previo = bool(asumir_previo and p11)
        if hay_previo:
            previo = vi.eventos(tickers, niveles, p11)
    nuevos, deshechos = vi.novedades(evs, previo)
    sin_dato = sorted(t["ticker"] for t in tickers if t["ticker"] not in precios)
    texto = vi.mensaje(fecha, corte, disponible, tickers, nuevos, deshechos, sin_dato,
                       hay_previo=hay_previo)
    return dict(base, ok=True, tickers=tickers, niveles=niveles, precios=precios,
                eventos=evs, nuevos=nuevos, deshechos=deshechos, sin_dato=sin_dato,
                hay_previo=hay_previo, texto=texto)
