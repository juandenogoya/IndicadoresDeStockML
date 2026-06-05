"""
backup_opciones_local.py  (one-shot, Tarea 16 Paso 6)

Backup off-site de las tablas de OPCIONES desde la DB LOCAL a archivos parquet
en backups/ (que vive en OneDrive -> se sincroniza fuera de la maquina).

Por que: tras el cleanup de Railway, las opciones >30d quedan solo en local. El
chain de opciones es IRRECUPERABLE (Yahoo solo sirve el vigente). Este export es
el seguro contra perder el disco local. Se corre ANTES del --execute del cleanup.

Lee LOCAL (Plan C: copia completa) en chunks (memoria acotada) y escribe parquet.

Uso:
    python scripts/oneshot/backup_opciones_local.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import text

from scripts.push_senales_bot import _engine_local

TABLAS = [
    "opciones_snapshot",
    "opciones_resumen_diario",
    "opciones_zscore_diario",
    "opciones_pcr_plazo_diario",
    "opciones_sector_pcr_plazo_diario",
    "opciones_sector_zscore_diario",
]
CHUNK = 200_000


def main():
    eng = _engine_local()
    outdir = os.path.join(ROOT, "backups", f"opciones_local_{date.today():%Y%m%d}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Backup -> {outdir}")
    print("-" * 60)

    for t in TABLAS:
        path = os.path.join(outdir, f"{t}.parquet")
        writer = None
        total = 0
        with eng.connect().execution_options(stream_results=True) as conn:
            for chunk in pd.read_sql_query(text(f"SELECT * FROM {t}"), conn, chunksize=CHUNK):
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="snappy")
                else:
                    # Distintos chunks pueden inferir int vs double (NULLs) -> unificar
                    # al schema del primero para que el parquet sea consistente.
                    table = table.cast(writer.schema, safe=False)
                writer.write_table(table)
                total += len(chunk)
        if writer is not None:
            writer.close()
        mb = os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else 0
        print(f"  {t:36} {total:>10,} filas -> {mb:>7.1f} MB")

    print("-" * 60)
    print("Backup completado. Verificar que los .parquet esten en OneDrive sincronizado.")


if __name__ == "__main__":
    main()
