"""
add_multiplos_px_columns.py  (one-shot, 2/6/2026)
Agrega a fundamentales_ratios_q las columnas de multiplos recalculados al CIERRE
del dia (*_px), conservando intactos los pe_ratio/pb_ratio/ps_ratio/ev_ebitda
originales de Yahoo (foto del Q).

Tambien agrega shares_out: las acciones en circulacion del ultimo Q (del balance),
para que compute_multiplos_px pueda recalcular P/S y EV/EBITDA sin releer el
balance cada dia.

Idempotente (ADD COLUMN IF NOT EXISTS). Correr una sola vez en LOCAL.
    venv\\Scripts\\python scripts\\oneshot\\add_multiplos_px_columns.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text
from src.data.database import get_engine

DDL = """
ALTER TABLE fundamentales_ratios_q
    ADD COLUMN IF NOT EXISTS pe_ratio_px   NUMERIC(14,4),
    ADD COLUMN IF NOT EXISTS pb_ratio_px   NUMERIC(14,4),
    ADD COLUMN IF NOT EXISTS ps_ratio_px   NUMERIC(14,4),
    ADD COLUMN IF NOT EXISTS ev_ebitda_px  NUMERIC(14,4),
    ADD COLUMN IF NOT EXISTS precio_px     NUMERIC(14,4),
    ADD COLUMN IF NOT EXISTS fecha_px      DATE,
    ADD COLUMN IF NOT EXISTS shares_out    NUMERIC(20,2);
"""


def main():
    eng = get_engine()
    print("engine host:", eng.url.host, "| db:", eng.url.database)
    if eng.url.host not in ("localhost", "127.0.0.1"):
        print("ABORT: este one-shot es LOCAL-only (fundamentales viven en local).")
        sys.exit(1)
    with eng.connect() as c:
        c.execute(text(DDL))
        c.commit()
    print("OK: columnas *_px + shares_out agregadas a fundamentales_ratios_q (idempotente).")


if __name__ == "__main__":
    main()
