"""
create_universo_cambios_table.py  (one-shot, Tarea 14)

Crea la tabla `universo_cambios` en LOCAL y RAILWAY. Registra cada alta/baja de
ticker del universo, para reproducibilidad (point-in-time) y honestidad de
analisis historicos: "este sector tenia N tickers en tal fecha".

Dual-DB porque `activos` (y por ende el universo) vive en ambas; el log de
cambios acompana.

Uso:
    python scripts/oneshot/create_universo_cambios_table.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text
from scripts.push_senales_bot import _engine_local, _engine_railway

DDL = """
CREATE TABLE IF NOT EXISTS universo_cambios (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(20)  NOT NULL,
    accion      VARCHAR(10)  NOT NULL,   -- 'ALTA' | 'BAJA'
    fecha       DATE         NOT NULL,
    sector      VARCHAR(60),
    motivo      TEXT,
    detalle     JSONB,
    created_at  TIMESTAMP DEFAULT NOW()
);
"""
DDL_IDX = "CREATE INDEX IF NOT EXISTS idx_universo_cambios_ticker ON universo_cambios(ticker);"


def main():
    for nombre, eng in [("LOCAL", _engine_local()), ("RAILWAY", _engine_railway())]:
        with eng.begin() as conn:
            conn.execute(text(DDL))
            conn.execute(text(DDL_IDX))
        print(f"  universo_cambios OK en {nombre}.")


if __name__ == "__main__":
    main()
