"""
auditar_etapa2.py -- criterio 1 de la alerta de la lista de vigilancia
(LOCAL, solo lee): los eventos de cada corte, recalculados en SQL sin el codigo
del motor, coinciden al 100% con los de src/utils/vigilancia_intradia.py.

Mismo papel que auditar_cierre_grupos.py: si el motor y una implementacion
independiente no dicen lo mismo, el mensaje no es confiable. Compara TODOS los
eventos del corte (no solo las novedades: esas son una resta de conjuntos).

Uso:
    python scripts/intradia/auditar_etapa2.py [--desde AAAA-MM-DD] [--hasta AAAA-MM-DD]

Salida: 0 coincide el 100% / 2 hay diferencias / 1 error o sin datos.
"""

import argparse
import os
import sys
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import evaluar_etapa2 as ev2   # noqa: E402  (carga .env y fuerza LOCAL)

from sqlalchemy import text   # noqa: E402

from src.data.database import get_engine          # noqa: E402
from src.utils import sesion_nyse                 # noqa: E402
from src.utils import vigilancia_intradia as vi   # noqa: E402

SQL_EVENTOS = """
WITH p AS (
    SELECT ticker, CAST(close AS DOUBLE PRECISION) AS p
    FROM precios_intradia_30m
    WHERE fecha = :f AND corte = :c AND inicio_barra = :u AND close IS NOT NULL
)
SELECT n.ticker,
       CASE WHEN n.cruzado_en_d THEN 'revierte' ELSE 'cruce' END AS tipo,
       n.nivel
FROM p
JOIN vigilancia_niveles n ON n.ticker = p.ticker AND n.fecha = :fl
JOIN vigilancia_tickers t ON t.ticker = n.ticker AND t.fecha = n.fecha
WHERE (n.lado_d = 'sobre' AND p.p < n.valor) OR (n.lado_d = 'bajo' AND p.p > n.valor)
UNION ALL
SELECT t.ticker, 'movimiento' AS tipo, NULL AS nivel
FROM p
JOIN vigilancia_tickers t ON t.ticker = p.ticker AND t.fecha = :fl
WHERE t.sigma_d IS NOT NULL AND p.p > 0 AND t.close_d > 0
  AND ABS(LN(p.p / t.close_d) / t.sigma_d) >= :z
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde")
    ap.add_argument("--hasta")
    args = ap.parse_args()
    desde = date.fromisoformat(args.desde) if args.desde else None
    hasta = date.fromisoformat(args.hasta) if args.hasta else None

    eng = get_engine()
    res = ev2.evaluar(eng, desde, hasta)
    if not res["cortes"]:
        print("  Ningun corte evaluable.")
        return 1

    n_ev, difs = 0, []
    for c in res["cortes"]:
        r = c["r"]
        hora = sesion_nyse.hora_corte(c["fecha"], c["corte"])
        with eng.connect() as conn:
            sql = {(x.ticker, x.tipo, x.nivel) for x in conn.execute(text(SQL_EVENTOS), {
                "f": c["fecha"], "c": c["corte"], "fl": r["fecha_lista"],
                "u": hora - timedelta(minutes=sesion_nyse.BARRA_MIN), "z": vi.UMBRAL_Z})}
        motor = {vi.clave(e) for e in r["eventos"]}
        n_ev += len(motor)
        if sql != motor:
            difs.append((c["fecha"], c["corte"], sorted(motor - sql, key=str),
                         sorted(sql - motor, key=str)))

    print(f"  Cortes auditados: {len(res['cortes'])} | eventos del motor: {n_ev}")
    for f, corte, solo_motor, solo_sql in difs[:20]:
        print(f"  [DIF] {f} {corte}: solo motor {solo_motor[:5]} | solo SQL {solo_sql[:5]}")
    print(f"  Criterio 1: {'PASA (100%)' if not difs else f'NO PASA ({len(difs)} cortes difieren)'}")
    return 0 if not difs else 2


if __name__ == "__main__":
    sys.exit(main())
