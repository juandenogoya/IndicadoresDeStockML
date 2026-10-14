"""
auditar_cierre_grupos.py
Criterio 1 de docs/cierre_grupos.md: recalcula el cierre por sector con SQL
INDEPENDIENTE (sin el codigo de alineacion del script ni el motor) y lo compara
contra lo que produce telegram_cierre.cargar_resultados.

Compara, por rueda y por sector: n, mediana de var dia, mediana de var 5r,
suben, bajan, % sobre SMA50; y el conjunto (ticker, direccion, detalle) de cada
tipo de cruce (OI, SWING, SMA). Tolerancia 1e-6. Solo lee.

Uso:
    python scripts/manual/auditar_cierre_grupos.py            (20 ruedas)
    python scripts/manual/auditar_cierre_grupos.py --ruedas 5 --hasta 2026-09-25

Salida: 0 = coincide todo / 1 = hay diferencias o error.
"""

import os
import sys
import argparse
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "manual"))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass
os.environ.pop("DATABASE_URL", None)

from src.data.database import query_df                  # noqa: E402
import telegram_cierre as tc                            # noqa: E402

TOL = 1e-6

# Ruedas del universo numeradas; D, D-1 y D-5 salen del numero de rueda.
_RUEDAS = """
    ru AS (
        SELECT fecha, ROW_NUMBER() OVER (ORDER BY fecha) AS rn
        FROM (SELECT p.fecha FROM precios_diarios p
              JOIN activos a ON a.ticker = p.ticker AND a.activo
              WHERE (CAST(:h AS DATE) IS NULL OR p.fecha <= :h)
              GROUP BY p.fecha
              HAVING COUNT(*) >= 0.5 * (SELECT COUNT(*) FROM activos WHERE activo)) x
    ),
    obj AS (
        SELECT d.fecha AS d, a.fecha AS d_ant, c.fecha AS d5
        FROM ru d JOIN ru a ON a.rn = d.rn - 1 JOIN ru c ON c.rn = d.rn - 5
        WHERE d.rn > (SELECT MAX(rn) FROM ru) - :n
    ),
    base AS (
        SELECT o.d, o.d_ant, t.ticker, t.sector,
               p.close, pa.close AS close_ant, p5.close AS close_5,
               i.sma50, i.sma200, ia.sma50 AS sma50_ant, ia.sma200 AS sma200_ant
        FROM obj o
        JOIN activos t ON t.activo
        JOIN precios_diarios p ON p.ticker = t.ticker AND p.fecha = o.d
        LEFT JOIN precios_diarios pa ON pa.ticker = t.ticker AND pa.fecha = o.d_ant
        LEFT JOIN precios_diarios p5 ON p5.ticker = t.ticker AND p5.fecha = o.d5
        LEFT JOIN indicadores_tecnicos i ON i.ticker = t.ticker AND i.fecha = o.d
        LEFT JOIN indicadores_tecnicos ia ON ia.ticker = t.ticker AND ia.fecha = o.d_ant
        WHERE p.close IS NOT NULL
    )
"""

SQL_GRUPOS = "WITH" + _RUEDAS + """
SELECT d, sector, COUNT(*) AS n,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (close / NULLIF(close_ant, 0) - 1) * 100)
           FILTER (WHERE close_ant > 0) AS var_dia,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (close / NULLIF(close_5, 0) - 1) * 100)
           FILTER (WHERE close_5 > 0) AS var_5r,
       COUNT(*) FILTER (WHERE close_ant > 0 AND close > close_ant) AS suben,
       COUNT(*) FILTER (WHERE close_ant > 0 AND close < close_ant) AS bajan,
       100.0 * COUNT(*) FILTER (WHERE sma50 IS NOT NULL AND close > sma50)
             / NULLIF(COUNT(*) FILTER (WHERE sma50 IS NOT NULL), 0) AS sobre_sma50_pct
FROM base GROUP BY d, sector
"""

SQL_CRUCES = "WITH" + _RUEDAS + """
SELECT b.d, 'OI' AS tipo, 'abajo' AS lado, b.ticker, m.soporte_strike::float AS det
FROM base b JOIN opciones_pcr_plazo_diario m
  ON m.ticker = b.ticker AND m.fecha = b.d_ant AND m.ventana = 'corto'
WHERE b.close < m.soporte_strike
UNION ALL
SELECT b.d, 'OI', 'arriba', b.ticker, m.resistencia_strike::float
FROM base b JOIN opciones_pcr_plazo_diario m
  ON m.ticker = b.ticker AND m.fecha = b.d_ant AND m.ventana = 'corto'
WHERE b.close > m.resistencia_strike
UNION ALL
SELECT b.d, 'SWING', 'arriba', b.ticker, NULL
FROM base b JOIN features_estructura e ON e.ticker = b.ticker AND e.fecha = b.d
WHERE COALESCE(e.bos_bull_5, 0) <> 0 OR COALESCE(e.choch_bull_5, 0) <> 0
UNION ALL
SELECT b.d, 'SWING', 'abajo', b.ticker, NULL
FROM base b JOIN features_estructura e ON e.ticker = b.ticker AND e.fecha = b.d
WHERE COALESCE(e.bos_bear_5, 0) <> 0 OR COALESCE(e.choch_bear_5, 0) <> 0
UNION ALL
SELECT d, 'SMA', CASE WHEN close > sma50 THEN 'arriba' ELSE 'abajo' END, ticker, 50
FROM base
WHERE close_ant IS NOT NULL AND sma50_ant IS NOT NULL AND sma50 IS NOT NULL
  AND (close_ant > sma50_ant) <> (close > sma50)
UNION ALL
SELECT d, 'SMA', CASE WHEN close > sma200 THEN 'arriba' ELSE 'abajo' END, ticker, 200
FROM base
WHERE close_ant IS NOT NULL AND sma200_ant IS NOT NULL AND sma200 IS NOT NULL
  AND (close_ant > sma200_ant) <> (close > sma200)
"""


def _cerca(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOL


def _num(v):
    if v is None:
        return None
    v = float(v)
    return None if v != v else v


def main():
    ap = argparse.ArgumentParser(description="Auditor del cierre por sector")
    ap.add_argument("--ruedas", type=int, default=20)
    ap.add_argument("--hasta", help="YYYY-MM-DD")
    args = ap.parse_args()
    hasta = date.fromisoformat(args.hasta) if args.hasta else None
    prm = {"n": args.ruedas, "h": hasta}

    try:
        sql_g = query_df(SQL_GRUPOS, prm)
        sql_c = query_df(SQL_CRUCES, prm)
        pares = tc.cargar_resultados(args.ruedas, hasta)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    esperado_g = {(tc._d(r["d"]), r["sector"]): r for r in sql_g.to_dict("records")}
    esperado_c = {}
    for r in sql_c.to_dict("records"):
        det = r["det"]
        if r["tipo"] == "SMA":
            det = f"SMA{int(det)}"
        elif det is not None and det == det:
            det = round(float(det), 6)
        else:
            det = None
        esperado_c.setdefault((tc._d(r["d"]), r["tipo"]), set()).add((r["ticker"], r["lado"], det))

    difs = []
    n_cmp_g = n_cmp_c = 0
    ruedas_sql = sorted({k[0] for k in esperado_g})
    ruedas_mot = [res["rueda"] for res, _ in pares]
    if ruedas_sql != ruedas_mot:
        difs.append(f"ruedas distintas: SQL {ruedas_sql} vs motor {ruedas_mot}")

    for res, _ in pares:
        d = res["rueda"]
        for g in res["grupos"]["grupos"]:
            e = esperado_g.get((d, g["sector"]))
            n_cmp_g += 1
            if e is None:
                difs.append(f"{d} {g['sector']}: no esta en SQL")
                continue
            for campo in ("n", "var_dia", "var_5r", "suben", "bajan", "sobre_sma50_pct"):
                if not _cerca(g[campo], _num(e[campo])):
                    difs.append(f"{d} {g['sector']} {campo}: motor {g[campo]} vs SQL {e[campo]}")
        sect_sql = {k[1] for k in esperado_g if k[0] == d}
        sect_mot = {g["sector"] for g in res["grupos"]["grupos"]}
        if sect_sql != sect_mot:
            difs.append(f"{d} sectores: motor {sorted(sect_mot)} vs SQL {sorted(sect_sql)}")

        for tipo in ("OI", "SWING", "SMA"):
            mot = set()
            for lado in ("arriba", "abajo"):
                for tk, det in res["cruces"][tipo][lado]:
                    if isinstance(det, float):
                        det = round(det, 6)
                    mot.add((tk, lado, det))
            if tipo == "OI" and not res["oi_disponible"]:
                mot_ok = not mot and not esperado_c.get((d, tipo))
                if not mot_ok:
                    difs.append(f"{d} OI sin muros pero hay cruces")
                continue
            sql = esperado_c.get((d, tipo), set())
            n_cmp_c += 1
            if mot != sql:
                difs.append(f"{d} {tipo}: solo motor {sorted(mot - sql)} | solo SQL {sorted(sql - mot)}")

    print(f"Ruedas: {len(pares)} ({ruedas_mot[0] if ruedas_mot else '-'} -> "
          f"{ruedas_mot[-1] if ruedas_mot else '-'})")
    print(f"Comparados: {n_cmp_g} sector-ruedas x 6 campos, {n_cmp_c} conjuntos de cruces")
    if difs:
        print(f"DIFERENCIAS ({len(difs)}):")
        for x in difs[:40]:
            print("  " + x)
        return 1
    print("Coincide el 100% (tolerancia 1e-6).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
