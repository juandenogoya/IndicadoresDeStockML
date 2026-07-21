"""
fix_ft_ops_split.py
Corrige las operaciones de Forward Testing distorsionadas por los splits de
KLAC (10:1, 2026-06-11) y CRWD (4:1, 2026-06-30).

QUE PASO:
    precios_diarios no se re-ajustaba hacia atras ante un split (corregido con
    scripts/manual/splits.py). Las posiciones que estaban abiertas cuando ocurrio
    el split se valuaron y cerraron al precio POST-split con la cantidad
    PRE-split -> perdidas ficticias de -72% a -88%.

QUE CORRIGE ESTE SCRIPT (aritmetica):
    precio_entrada /= ratio ; cantidad *= ratio
    capital_entrada queda IGUAL: P*N == (P/r)*(N*r). El titular de N acciones
    pre-split pasa a tener N*r acciones al precio P/r; es la misma posicion.
    Luego recomputa pnl y pnl_pct, y ajusta el cash de la estrategia por la
    diferencia (el producto de la venta era r veces menor de lo que debia).

QUE **NO** PUEDE CORREGIR (importante):
    Las SALIDAS son contrafacticas. El bot cerro esas posiciones porque "vio"
    un derrumbe del 88%: motivos STOP_LOSS_ATR y SL_PROTECCION que jamas se
    habrian disparado con los precios correctos. Con la aritmetica arreglada el
    PnL de esos trades pasa a ser correcto PARA LA SALIDA QUE OCURRIO, pero la
    estrategia real habria seguido en posicion.

    O sea: esto arregla la contabilidad, no la historia. Las 8 operaciones
    quedan documentadas en el JOURNAL como no representativas de la logica de
    salida de sus estrategias -- mismo criterio que se uso con el bug de
    score=0.0 (JOURNAL 2026-05-30).

Uso:
    python scripts/oneshot/fix_ft_ops_split.py             # dry run
    python scripts/oneshot/fix_ft_ops_split.py --apply
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

SPLITS = {"KLAC": {"ratio": 10, "fecha": "2026-06-11"},
          "CRWD": {"ratio": 4,  "fecha": "2026-06-30"}}
BACKUP_DIR = os.path.join(ROOT, "data", "backups")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def afectadas(engine):
    """
    Ops que tenian la posicion ABIERTA cuando ocurrio el split.

    fecha_entrada <= fecha_split porque el sistema es asincronico: una entrada
    fechada el dia del split se ejecuto al cierre del dia habil ANTERIOR, o sea
    a precio pre-split.
    """
    filas = []
    with engine.connect() as conn:
        for tk, s in SPLITS.items():
            df = pd.read_sql(text("""
                SELECT o.id, o.estrategia_id, e.nombre, o.ticker,
                       o.fecha_entrada, o.fecha_salida,
                       o.precio_entrada, o.cantidad, o.capital_entrada,
                       o.precio_salida, o.pnl, o.pnl_pct, o.motivo_salida
                FROM ft_operaciones o JOIN ft_estrategias e ON e.id = o.estrategia_id
                WHERE o.ticker = :tk
                  AND o.fecha_entrada <= :f
                  AND (o.fecha_salida IS NULL OR o.fecha_salida >= :f)
                ORDER BY o.id
            """), conn, params={"tk": tk, "f": s["fecha"]})
            df["ratio"] = s["ratio"]
            filas.append(df)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="escribe (default: dry run)")
    args = ap.parse_args()

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")

    df = afectadas(engine)
    if df.empty:
        log("No hay operaciones afectadas.")
        return 0

    for c in ("precio_entrada", "cantidad", "capital_entrada", "precio_salida",
              "pnl", "pnl_pct", "ratio"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["precio_entrada_new"] = (df.precio_entrada / df.ratio).round(4)
    df["cantidad_new"] = (df.cantidad * df.ratio).astype(int)
    df["pnl_new"] = ((df.precio_salida - df.precio_entrada_new)
                     * df.cantidad_new).round(2)
    df["pnl_pct_new"] = ((df.precio_salida / df.precio_entrada_new - 1) * 100).round(4)
    df["delta_pnl"] = (df.pnl_new - df.pnl).round(2)

    log(f"Operaciones afectadas: {len(df)}")
    print()
    print(f"{'id':>5} {'est':>3} {'tk':<5} {'entrada':>10} {'->':>9} {'cant':>4} {'->':>5} "
          f"{'pnl':>10} {'->':>10} {'delta':>10}  motivo")
    for _, r in df.iterrows():
        print(f"{int(r.id):>5} {int(r.estrategia_id):>3} {r.ticker:<5} "
              f"{r.precio_entrada:>10.2f} {r.precio_entrada_new:>9.2f} "
              f"{int(r.cantidad):>4} {r.cantidad_new:>5} "
              f"{r.pnl:>10.2f} {r.pnl_new:>10.2f} {r.delta_pnl:>+10.2f}  {r.motivo_salida}")

    print()
    log(f"Delta total de PnL: {df.delta_pnl.sum():+,.2f}")
    print()
    imp = df.groupby(["estrategia_id", "nombre"])["delta_pnl"].sum().reset_index()
    with engine.connect() as conn:
        est = pd.read_sql(text(
            "SELECT id, capital_inicial, capital_actual FROM ft_estrategias"), conn
        ).set_index("id")
    imp = imp.set_index("estrategia_id").join(est)
    imp["ret_antes"] = ((imp.capital_actual.astype(float)
                         / imp.capital_inicial.astype(float) - 1) * 100).round(2)
    imp["ret_despues"] = (((imp.capital_actual.astype(float) + imp.delta_pnl)
                           / imp.capital_inicial.astype(float) - 1) * 100).round(2)
    print(imp[["nombre", "delta_pnl", "ret_antes", "ret_despues"]].to_string())

    if not args.apply:
        print()
        log("[DRY RUN] no se escribio nada. Usar --apply para corregir.")
        return 0

    # Backup
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ruta = os.path.join(BACKUP_DIR, f"ft_operaciones_split_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with engine.connect() as conn:
        pd.read_sql(text("""
            SELECT * FROM ft_operaciones WHERE id = ANY(:ids)
        """), conn, params={"ids": df.id.astype(int).tolist()}).to_csv(ruta, index=False)
    log(f"Backup: {os.path.relpath(ruta, ROOT)}")

    with engine.connect() as conn:
        for _, r in df.iterrows():
            # El motivo se marca con sufijo _SPLIT_FIX porque quedo MINTIENDO:
            # un SL_PROTECCION que produce +19.9% es imposible. Sin la marca,
            # estas 8 salidas contaminarian el analisis por motivo_salida
            # (metricas_por_motivo) como si fueran evidencia sobre la calidad
            # de esas reglas de salida, cuando en realidad las disparo el
            # derrumbe falso. Marcado, quedan en su propio bucket y se ven.
            motivo = r.motivo_salida
            if motivo and not motivo.endswith("_SPLIT_FIX"):
                motivo = f"{motivo}_SPLIT_FIX"
            conn.execute(text("""
                UPDATE ft_operaciones
                SET precio_entrada = :pe, cantidad = :cant,
                    pnl = :pnl, pnl_pct = :pnl_pct, motivo_salida = :motivo
                WHERE id = :id
            """), {"pe": float(r.precio_entrada_new), "cant": int(r.cantidad_new),
                   "pnl": float(r.pnl_new), "pnl_pct": float(r.pnl_pct_new),
                   "motivo": motivo, "id": int(r.id)})

        # El cash recibio r veces menos de lo que debia al vender.
        for eid, delta in df.groupby("estrategia_id")["delta_pnl"].sum().items():
            conn.execute(text("""
                UPDATE ft_estrategias
                SET cash_disponible = cash_disponible + :d,
                    capital_actual  = capital_actual  + :d
                WHERE id = :eid
            """), {"d": float(delta), "eid": int(eid)})
            log(f"  estrategia {eid}: cash y capital {delta:+,.2f}")
        conn.commit()

    log("Correccion aplicada.")
    log("RECORDAR: las salidas de estas operaciones siguen siendo contrafacticas "
        "(se dispararon por el derrumbe falso). Documentar en el JOURNAL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
