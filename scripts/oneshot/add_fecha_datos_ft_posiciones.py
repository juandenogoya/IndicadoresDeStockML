"""
add_fecha_datos_ft_posiciones.py
Agrega y backfillea `fecha_datos` en ft_posiciones_diarias.

PROBLEMA (auditoria de copias del precio, 10/9/2026):
    ft_posiciones_diarias.fecha es la fecha en que CORRIO el bot (registro), pero
    precio_cierre -- y con el, retorno, rango, ATR y scores de la fila -- es el del
    ULTIMO CIERRE DISPONIBLE en ese momento. Medido sobre 23.240 filas: 98,8% es el
    close de una rueda ANTERIOR a `fecha` y 0,4% el del mismo dia. Es la misma trampa
    que fecha_entrada / fecha_datos en ft_operaciones (docs/forward_testing/METRICAS.md):
    cruzar `fecha` con precios_diarios o indicadores_tecnicos lee el dia equivocado,
    en silencio.

METODO DE BACKFILL:
    1. Match EXACTO y UNICO: la rueda <= fecha (hasta 15 ruedas atras) cuyo close
       coincide con precio_cierre (el close redondeado a 4 decimales).
       NO la tolerancia de un centavo de add_fecha_datos_ft_operaciones.py: en
       tickers baratos dos ruedas seguidas difieren en un centavo y el primer
       borrador elegia la mas reciente (LAC 3,14 el 13/7 y 3,15 el 14/7). Si dos
       ruedas tienen el MISMO close (WBD 26,12 el 6/7 y el 7/7) el precio no dice
       cual se uso: no vota.
    2. Lo que no matchea (o es ambiguo) toma la MODA de su corrida: todas las filas
       de una misma (estrategia, fecha) se escribieron con los datos de la misma
       noche. Aca caen los precios de Railway previos a la migracion de FT (mayo) y
       las filas de KLAC/CRWD anteriores a la correccion del split, en escala vieja.
       La moda NO pisa un match unico: en una corrida con precios cargados a medias
       (hueco del 2026-08-28) cada ticker puede tener su propia fecha.
    3. Si la corrida no tiene ningun voto, la moda de las demas estrategias de esa
       misma fecha.
    CONTROL: el dia de la ENTRADA y el del CIERRE, la fila tiene que coincidir con
    ft_operaciones.fecha_datos / fecha_datos_salida, que se resolvieron por otro lado.

    No toca ninguna otra columna: las 39 filas contaminadas por el split quedan como
    estan (decision 12/9/2026, ver METRICAS.md "Splits").

HACIA ADELANTE:
    ft_utils.registrar_estado_posiciones() la escribe: MAX(fecha) del ticker en
    precios_diarios, que es la fecha del close que usa como precio_cierre. Los 10 bots
    no cambian. Correr este script con --apply ANTES del proximo ft_run_diario: sin la
    columna, el INSERT de registrar_estado_posiciones falla.

Uso:
    python scripts/oneshot/add_fecha_datos_ft_posiciones.py            # dry run
    python scripts/oneshot/add_fecha_datos_ft_posiciones.py --apply
"""

import os
import sys
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "forward_testing"))

from ft_env import configurar_entorno_local  # noqa: E402
configurar_entorno_local()

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402

VENTANA = 15    # ruedas hacia atras para buscar el close que matchea
TOL = 0.0001    # precio_cierre = close redondeado a 4 decimales (dif real < 0,00005)

DDL = """
ALTER TABLE ft_posiciones_diarias
    ADD COLUMN IF NOT EXISTS fecha_datos DATE;

COMMENT ON COLUMN ft_posiciones_diarias.fecha_datos IS
    'Fecha del close usado como precio_cierre (el ultimo disponible cuando corrio '
    'el bot). La columna fecha es la de REGISTRO. Para cruzar con precios_diarios '
    'o indicadores_tecnicos hay que usar esta.';
"""


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _hay(x):
    return x is not None and not (isinstance(x, float) and pd.isna(x))


def _a_date(serie):
    return pd.to_datetime(serie).dt.date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="escribe (default: dry run)")
    args = ap.parse_args()

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")

    with engine.connect() as conn:
        pos = pd.read_sql(text("""
            SELECT id, estrategia_id, operacion_id, ticker, fecha, estado, precio_cierre
            FROM ft_posiciones_diarias ORDER BY id
        """), conn)
        pos["fecha"] = _a_date(pos["fecha"])
        desde = min(pos["fecha"]) - timedelta(days=40)
        px = pd.read_sql(text("""
            SELECT ticker, fecha, close FROM precios_diarios WHERE fecha >= :d
        """), conn, params={"d": desde})
        ops = pd.read_sql(text("""
            SELECT id AS operacion_id, fecha_entrada, fecha_datos AS fd_entrada,
                   fecha_salida, fecha_datos_salida AS fd_salida
            FROM ft_operaciones
        """), conn)

    px["fecha"] = _a_date(px["fecha"])
    dias = sorted(set(px["fecha"]))
    idx = {d: i for i, d in enumerate(dias)}
    closes = {(t, f): float(c) for t, f, c in px.itertuples(index=False)}
    log(f"Filas: {len(pos)} | ruedas de precios desde {dias[0]}")

    def votar(ticker, precio, fecha_reg):
        """
        (fecha, motivo): la rueda <= fecha_reg cuyo close == precio si es UNICA.
        fecha None con motivo 'sin_match' o 'ambiguo' (mas de una rueda con ese close).
        """
        if not _hay(precio):
            return None, "sin_precio"
        i = bisect_right(dias, fecha_reg) - 1
        hallados = []
        for j in range(i, max(i - VENTANA - 1, -1), -1):
            c = closes.get((ticker, dias[j]))
            if c is not None and abs(c - float(precio)) < TOL:
                hallados.append(dias[j])
        if len(hallados) == 1:
            return hallados[0], "match"
        return None, ("ambiguo" if hallados else "sin_match")

    votos = [votar(r.ticker, r.precio_cierre, r.fecha) for r in pos.itertuples()]

    corrida = defaultdict(list)
    for r, (m, _) in zip(pos.itertuples(), votos):
        if m is not None:
            corrida[(r.estrategia_id, r.fecha)].append(m)
    moda_corrida = {k: Counter(v).most_common(1)[0][0] for k, v in corrida.items()}
    por_fecha = defaultdict(list)
    for (_, f), fd in moda_corrida.items():
        por_fecha[f].append(fd)
    moda_fecha = {f: Counter(v).most_common(1)[0][0] for f, v in por_fecha.items()}

    fechas_datos, metodos, motivos = [], [], []
    for r, (m, motivo) in zip(pos.itertuples(), votos):
        if m is not None:
            fd, met = m, "match"
        elif (r.estrategia_id, r.fecha) in moda_corrida:
            fd, met = moda_corrida[(r.estrategia_id, r.fecha)], "moda_corrida"
        elif r.fecha in moda_fecha:
            fd, met = moda_fecha[r.fecha], "moda_fecha"
        else:
            fd, met = None, "sin_resolver"
        fechas_datos.append(fd)
        metodos.append(met)
        motivos.append(motivo)
    pos["fecha_datos"] = fechas_datos
    pos["metodo"] = metodos
    pos["motivo"] = motivos

    # ── Reporte ──────────────────────────────────────────────────────────────
    print()
    print("=== Metodo (y por que no hubo match unico) ===")
    print(pos.groupby(["metodo", "motivo"]).size().to_string())

    def desfase(r):
        if r.fecha_datos is None:
            return None
        return (bisect_right(dias, r.fecha) - 1) - idx[r.fecha_datos]

    des = pd.Series([desfase(r) for r in pos.itertuples()]).dropna().astype(int)
    print()
    print("=== Desfase (ruedas entre el dato y la fecha de registro, %) ===")
    print((des.value_counts(normalize=True).sort_index() * 100).round(1).to_string())

    # Control contra ft_operaciones (resuelto por otro metodo).
    for c in ("fecha_entrada", "fd_entrada", "fecha_salida", "fd_salida"):
        ops[c] = _a_date(ops[c])
    m = pos.merge(ops, on="operacion_id", how="left")
    ent = m[(m["estado"] == "abierta") & (m["fecha"] == m["fecha_entrada"]) & m["fd_entrada"].notna()]
    sal = m[(m["estado"] == "cierre") & (m["fecha"] == m["fecha_salida"]) & m["fd_salida"].notna()]
    malas = pd.concat([ent[ent["fecha_datos"] != ent["fd_entrada"]].assign(ref=ent["fd_entrada"]),
                       sal[sal["fecha_datos"] != sal["fd_salida"]].assign(ref=sal["fd_salida"])])
    print()
    print("=== Control contra ft_operaciones ===")
    print(f"  dia de ENTRADA: {(ent['fecha_datos'] == ent['fd_entrada']).sum()} de {len(ent)} coinciden")
    print(f"  dia de CIERRE : {(sal['fecha_datos'] == sal['fd_salida']).sum()} de {len(sal)} coinciden")
    if len(malas):
        malas["mes"] = [f.strftime("%Y-%m") for f in malas["fecha"]]
        print("  diferencias por mes:", malas["mes"].value_counts().sort_index().to_dict())
        print(malas[["id", "estrategia_id", "ticker", "fecha", "estado", "fecha_datos",
                     "ref", "metodo", "motivo"]].tail(12).to_string(index=False))

    if not args.apply:
        print()
        log("[DRY RUN] no se escribio nada. Usar --apply.")
        return 0

    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.execute(text("UPDATE ft_posiciones_diarias SET fecha_datos = :fd WHERE id = :id"),
                     [{"fd": r.fecha_datos, "id": int(r.id)}
                      for r in pos.itertuples() if r.fecha_datos is not None])
        conn.commit()
        total, con = conn.execute(text(
            "SELECT COUNT(*), COUNT(fecha_datos) FROM ft_posiciones_diarias")).fetchone()
    log(f"Backfill aplicado: {con} de {total} filas con fecha_datos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
