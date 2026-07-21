"""
fix_ft_ops_split_sltp.py
Repara stop_loss / take_profit de las operaciones corregidas por split.

QUE PASO:
    `fix_ft_ops_split.py` corrigio precio_entrada y cantidad de las 8 ops
    afectadas por los splits de KLAC y CRWD, pero **se olvido de stop_loss y
    take_profit**. Quedaron en la escala VIEJA:

        id 820  KLAC  precio_entrada 201.14  stop_loss 1847.31   <- x10 de mas

    El PnL de esas operaciones esta bien (no depende del SL), pero el registro
    es internamente incoherente y rompe cualquier analisis de distancia al
    stop o de expectancy en R -- justo la metrica que METRICAS.md documenta
    como reconstruible para las 10 estrategias.

    Lo detecto la auditoria de calidad de datos, con el chequeo
    "SL/TP incoherentes (long: SL < entrada < TP)".

IDEMPOTENTE: solo toca las filas donde el SL sigue estando fuera de escala
(stop_loss > precio_entrada * 3). Correrlo dos veces no divide dos veces.

Uso:
    python scripts/oneshot/fix_ft_ops_split_sltp.py            # dry run
    python scripts/oneshot/fix_ft_ops_split_sltp.py --apply
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

RATIOS = {"KLAC": 10, "CRWD": 4}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")

    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT id, estrategia_id, ticker, precio_entrada, stop_loss, take_profit
            FROM ft_operaciones
            WHERE motivo_salida LIKE '%%_SPLIT_FIX'
              AND stop_loss > precio_entrada * 3
            ORDER BY id
        """), conn)

    if df.empty:
        log("Nada que reparar: todos los SL/TP ya estan en escala.")
        return 0

    for c in ("precio_entrada", "stop_loss", "take_profit"):
        df[c] = pd.to_numeric(df[c])
    df["ratio"] = df.ticker.map(RATIOS)
    df["sl_new"] = (df.stop_loss / df.ratio).round(4)
    df["tp_new"] = (df.take_profit / df.ratio).round(4)
    df["dist_sl_new_pct"] = ((1 - df.sl_new / df.precio_entrada) * 100).round(2)

    log(f"Operaciones a reparar: {len(df)}")
    print()
    print(f"{'id':>5} {'tk':<5} {'entrada':>10} {'SL viejo':>11} {'SL nuevo':>10} "
          f"{'TP viejo':>11} {'TP nuevo':>10} {'dist SL':>8}")
    for _, r in df.iterrows():
        print(f"{int(r.id):>5} {r.ticker:<5} {r.precio_entrada:>10.4f} "
              f"{r.stop_loss:>11.4f} {r.sl_new:>10.4f} "
              f"{r.take_profit:>11.4f} {r.tp_new:>10.4f} {r.dist_sl_new_pct:>7.2f}%")

    if not args.apply:
        print()
        log("[DRY RUN] no se escribio nada. Usar --apply.")
        return 0

    with engine.connect() as conn:
        for _, r in df.iterrows():
            conn.execute(text("""
                UPDATE ft_operaciones
                SET stop_loss = :sl, take_profit = :tp
                WHERE id = :id
            """), {"sl": float(r.sl_new), "tp": float(r.tp_new), "id": int(r.id)})
        conn.commit()
    log(f"Reparadas {len(df)} operaciones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
