"""
add_fecha_datos_ft_operaciones.py
Agrega y backfillea fecha_datos / fecha_datos_salida en ft_operaciones.

PROBLEMA (2026-07-21):
    `fecha_entrada` es la fecha en que se REGISTRO la operacion, no la del dato
    con que se decidio. El sistema es asincronico por diseno: los bots operan
    con el OHLCV del ultimo cierre disponible. Pero el desfase NO es fijo --
    depende de cuan rancia estaba la data cuando corrio el bot (la rutina
    nocturna es manual):

        mismo dia habil   326 ops (18.0%)   <- la recovery ya habia corrido
        1 dia habil antes 1246 ops (68.8%)  <- el caso tipico
        2 dias antes        84 ops ( 4.6%)  <- se salteo una noche
        5 dias antes        55 ops ( 3.0%)  <- se salteo una semana

    Consecuencias de no tener la fecha del dato:
    - La equity marca por primera vez la posicion el dia del REGISTRO, metiendo
      varios dias de movimiento como un salto de un solo dia -> infla la
      volatilidad y distorsiona el drawdown (~8% de las ops).
    - Cualquier analisis que cruce fecha_entrada con indicadores_tecnicos de esa
      misma fecha lee EL DIA EQUIVOCADO. Silencioso y sistematico.

METODO DE BACKFILL:
    Se deriva por matching de precio contra el close... pero eso falla para las
    salidas por SL/TP, cuyo precio es el nivel del stop y no un cierre.

    Solucion: **todas las operaciones de una misma corrida comparten la misma
    fecha de dato**. Entonces se agrupa por (estrategia_id, fecha de registro),
    se juntan los "votos" de las ops que SI matchean un close, y la moda del
    grupo se aplica a todo el grupo -- incluidas las de SL/TP.

HACIA ADELANTE:
    ft_utils.abrir_operacion()/cerrar_operacion() lo escriben solos: el precio
    que usan los bots ES el ultimo close disponible, asi que la fecha del dato
    es MAX(fecha) de ese ticker en precios_diarios al momento de registrar.
    Los 10 bots NO necesitan cambios (todos pasan por ft_utils).

Uso:
    python scripts/oneshot/add_fecha_datos_ft_operaciones.py            # dry run
    python scripts/oneshot/add_fecha_datos_ft_operaciones.py --apply
"""

import os
import sys
import argparse
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "forward_testing"))

from ft_env import configurar_entorno_local  # noqa: E402
configurar_entorno_local()

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402

VENTANA = 15   # dias habiles hacia atras para buscar el close que matchea
TOL = 0.011    # tolerancia de precio (NUMERIC(12,4) redondeado)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


DDL = """
ALTER TABLE ft_operaciones
    ADD COLUMN IF NOT EXISTS fecha_datos        DATE,
    ADD COLUMN IF NOT EXISTS fecha_datos_salida DATE;

COMMENT ON COLUMN ft_operaciones.fecha_datos IS
    'Fecha del OHLCV con el que se decidio y valuo la ENTRADA. El sistema es '
    'asincronico: no coincide con fecha_entrada (que es cuando se registro). '
    'Es la fecha correcta para cruzar con precios_diarios/indicadores_tecnicos.';
COMMENT ON COLUMN ft_operaciones.fecha_datos_salida IS
    'Idem para la SALIDA. NULL mientras la posicion este abierta.';
"""


def dia_habil_le(f, dias):
    prev = [d for d in dias if d <= f]
    return prev[-1] if prev else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")

    with engine.connect() as conn:
        ops = pd.read_sql(text("""
            SELECT id, estrategia_id, ticker, fecha_entrada, precio_entrada,
                   fecha_salida, precio_salida, motivo_salida
            FROM ft_operaciones ORDER BY id
        """), conn)
        px = pd.read_sql(text("""
            SELECT ticker, fecha, close FROM precios_diarios
            WHERE fecha >= '2026-03-01'
        """), conn)

    px["fecha"] = pd.to_datetime(px["fecha"])
    px["close"] = pd.to_numeric(px["close"])
    for c in ("fecha_entrada", "fecha_salida"):
        ops[c] = pd.to_datetime(ops[c])
    for c in ("precio_entrada", "precio_salida"):
        ops[c] = pd.to_numeric(ops[c])

    dias = sorted(px["fecha"].unique())
    idx = {d: i for i, d in enumerate(dias)}
    pxi = px.set_index(["ticker", "fecha"])["close"]

    def votar(ticker, precio, fecha_reg):
        """Dia habil cuyo close == precio. None si no matchea (SL/TP intradia)."""
        if pd.isna(precio) or pd.isna(fecha_reg):
            return None
        base = dia_habil_le(fecha_reg, dias)
        if base is None:
            return None
        i = idx[base]
        for off in range(0, VENTANA + 1):
            j = i - off
            if j < 0:
                break
            try:
                if abs(float(pxi.loc[(ticker, dias[j])]) - float(precio)) < TOL:
                    return dias[j]
            except KeyError:
                continue
        return None

    # ── Votos por corrida ────────────────────────────────────────────────
    # Una "corrida" = (estrategia, fecha de registro). Todas sus ops comparten
    # la misma fecha de dato.
    votos = {}
    for _, o in ops.iterrows():
        v = votar(o.ticker, o.precio_entrada, o.fecha_entrada)
        if v is not None:
            votos.setdefault((o.estrategia_id, o.fecha_entrada), []).append(v)
        if pd.notna(o.fecha_salida):
            v = votar(o.ticker, o.precio_salida, o.fecha_salida)
            if v is not None:
                votos.setdefault((o.estrategia_id, o.fecha_salida), []).append(v)

    # moda por corrida
    fecha_corrida = {}
    for k, vs in votos.items():
        fecha_corrida[k] = pd.Series(vs).mode().iloc[0]

    # fallback global: si una corrida no tiene ningun voto propio, se usa la
    # moda de las OTRAS estrategias que corrieron esa misma fecha.
    por_fecha = {}
    for (eid, f), fd in fecha_corrida.items():
        por_fecha.setdefault(f, []).append(fd)
    fecha_global = {f: pd.Series(v).mode().iloc[0] for f, v in por_fecha.items()}

    def resolver(eid, f):
        if pd.isna(f):
            return None
        if (eid, f) in fecha_corrida:
            return fecha_corrida[(eid, f)]
        return fecha_global.get(f)

    ops["fecha_datos"] = [resolver(o.estrategia_id, o.fecha_entrada)
                          for _, o in ops.iterrows()]
    ops["fecha_datos_salida"] = [resolver(o.estrategia_id, o.fecha_salida)
                                 for _, o in ops.iterrows()]

    # ── Reporte ──────────────────────────────────────────────────────────
    n = len(ops)
    res_e = ops["fecha_datos"].notna().sum()
    cerr = ops["fecha_salida"].notna()
    res_s = ops.loc[cerr, "fecha_datos_salida"].notna().sum()
    log(f"Operaciones: {n} | entradas resueltas: {res_e} ({res_e/n*100:.1f}%) "
        f"| salidas resueltas: {res_s}/{cerr.sum()} ({res_s/cerr.sum()*100:.1f}%)")

    d = ops.dropna(subset=["fecha_datos"]).copy()
    d["off"] = [(idx[dia_habil_le(r.fecha_entrada, dias)] - idx[r.fecha_datos])
                if dia_habil_le(r.fecha_entrada, dias) in idx and r.fecha_datos in idx
                else None for _, r in d.iterrows()]
    print("\n=== Desfase resultante (dias habiles) entradas ===")
    print(d["off"].value_counts().sort_index().to_string())

    print("\n=== Muestra ===")
    m = ops[["id", "estrategia_id", "ticker", "fecha_entrada", "fecha_datos",
             "fecha_salida", "fecha_datos_salida", "motivo_salida"]].head(8)
    print(m.to_string(index=False))

    if not args.apply:
        print()
        log("[DRY RUN] no se escribio nada. Usar --apply.")
        return 0

    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    log("Columnas creadas.")

    with engine.connect() as conn:
        for _, o in ops.iterrows():
            conn.execute(text("""
                UPDATE ft_operaciones
                SET fecha_datos = :fd, fecha_datos_salida = :fds
                WHERE id = :id
            """), {
                "fd":  o.fecha_datos.date() if pd.notna(o.fecha_datos) else None,
                "fds": o.fecha_datos_salida.date() if pd.notna(o.fecha_datos_salida) else None,
                "id":  int(o.id),
            })
        conn.commit()
    log(f"Backfill aplicado sobre {n} operaciones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
