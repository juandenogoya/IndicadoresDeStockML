"""
cleanup_railway_planB.py  (one-shot, Tarea 16 Paso 6)

Limpieza de storage de Railway habilitada por Plan B: con los bots leyendo solo
la masticada senales_bot_diaria, las tablas CRUDAS de mercado US dejan de tener
consumidor vivo en Railway (local es la copia autoritativa bajo Plan C).

Hace dos cosas:
  (A) DROP de crudas congeladas (US mercado + futuros + backtest + FT). Local las
      tiene con igual o mas datos (verificado).
  (B) RETENTION de opciones: KEEP las tablas (las escribe el snapshot, las lee el
      sync), pero purga filas viejas. snapshot a N dias, derivadas a M dias.

SEGURIDAD:
  - --dry-run (DEFAULT): no toca NADA. Reporta tamanos, filas, que dropearia/
    purgaria, el chequeo de FKs y la cobertura de local. Hay que pasar --execute
    explicitamente para que actue.
  - Pre-check de cobertura: para cada tabla de opciones, compara rango de fechas y
    conteo local vs Railway. Si local NO cubre, ABORTA el purge de esa tabla.
  - Chequeo de FKs: si alguna tabla que se MANTIENE referencia a una que se dropea,
    lo reporta (con --execute usa DROP ... CASCADE, que removeria esas FKs).
  - El backup off-site (export de opciones local a OneDrive) es un paso SEPARADO
    (scripts/oneshot/backup_opciones_local.py) que se corre ANTES de --execute.

Uso:
    python scripts/oneshot/cleanup_railway_planB.py                 # dry-run
    python scripts/oneshot/cleanup_railway_planB.py --execute       # real
    python scripts/oneshot/cleanup_railway_planB.py --execute --no-vacuum
    python scripts/oneshot/cleanup_railway_planB.py --dias-snapshot 30 --dias-derivadas 90
"""

import os
import sys
import argparse
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text
from scripts.push_senales_bot import _engine_railway, _engine_local


# Tablas a DROPEAR (local es fuente autoritativa). Orden hijo->padre por si hay FKs.
DROP_TABLES = [
    # US mercado (crudas congeladas)
    "indicadores_tecnicos", "features_market_structure", "features_precio_accion",
    "features_ml", "features_sector", "scoring_tecnico", "ticker_zscore_diario",
    "alertas_scanner", "features_regimen_macro",
    "precios_diarios",                      # padre de varias -> al final
    # futuros
    "indicadores_tecnicos_futuros", "futuros_diarios",
    # backtest historico
    "bt_hist_candidatos", "bt_hist_operaciones", "bt_hist_metricas_diarias",
    "bt_hist_estrategias",
    # forward testing (local es fuente de verdad desde 18/5)
    "ft_operaciones", "ft_candidatos_diarios", "ft_posiciones_diarias",
    "ft_metricas_diarias", "ft_estrategias",
]

# RETENTION de opciones: tabla -> (columna_fecha, dias_a_mantener)
def _retention(dias_snapshot, dias_derivadas):
    return {
        "opciones_snapshot":                ("fecha_snapshot", dias_snapshot),
        "opciones_resumen_diario":          ("fecha", dias_derivadas),
        "opciones_zscore_diario":           ("fecha", dias_derivadas),
        "opciones_pcr_plazo_diario":        ("fecha", dias_derivadas),
        "opciones_sector_pcr_plazo_diario": ("fecha", dias_derivadas),
        "opciones_sector_zscore_diario":    ("fecha", dias_derivadas),
    }


def _size(conn, tabla):
    return conn.execute(text(
        "SELECT pg_size_pretty(pg_total_relation_size(:t)), pg_total_relation_size(:t)"
    ), {"t": tabla}).fetchone()


def _existe(conn, tabla):
    return conn.execute(text(
        "SELECT to_regclass(:t) IS NOT NULL"
    ), {"t": tabla}).scalar()


def chequear_fks(conn, drop_tables):
    """FKs desde una tabla NO-drop hacia una tabla drop (las que CASCADE removeria)."""
    rows = conn.execute(text("""
        SELECT tc.table_name AS referencing, ccu.table_name AS referenced
        FROM information_schema.table_constraints tc
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND ccu.table_name = ANY(:drops)
          AND tc.table_name <> ALL(:drops)
    """), {"drops": drop_tables}).fetchall()
    return [(r[0], r[1]) for r in rows]


def main():
    ap = argparse.ArgumentParser(description="Limpieza Railway Plan B (Tarea 16 Paso 6)")
    ap.add_argument("--execute", action="store_true",
                    help="EJECUTA los cambios. Sin esto, solo dry-run (no toca nada).")
    ap.add_argument("--dias-snapshot", type=int, default=30,
                    help="Dias a mantener en opciones_snapshot (default 30)")
    ap.add_argument("--dias-derivadas", type=int, default=90,
                    help="Dias a mantener en opciones derivadas (default 90)")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="No correr VACUUM FULL tras el purge (mas rapido, no libera disco)")
    args = ap.parse_args()

    dry = not args.execute
    hoy = date.today()
    cut_snap = hoy - timedelta(days=args.dias_snapshot)
    cut_der  = hoy - timedelta(days=args.dias_derivadas)
    retention = _retention(args.dias_snapshot, args.dias_derivadas)

    sep = "=" * 72
    print(sep)
    print(f"CLEANUP RAILWAY PLAN B  -- {'DRY RUN (no toca nada)' if dry else 'EJECUCION REAL'}")
    print(f"  Retention: snapshot {args.dias_snapshot}d (corte {cut_snap}) | "
          f"derivadas {args.dias_derivadas}d (corte {cut_der})")
    print(sep)

    eng_r = _engine_railway()
    eng_l = _engine_local()

    # ── (A) DROP candidatas ──
    print("\n[A] DROP de crudas congeladas (local = copia autoritativa)")
    print("-" * 72)
    total_drop = 0
    presentes = []
    with eng_r.connect() as c:
        for t in DROP_TABLES:
            if not _existe(c, t):
                print(f"  {t:36} (no existe en Railway, skip)")
                continue
            sz, b = _size(c, t)
            n = c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            total_drop += b
            presentes.append(t)
            print(f"  {t:36} {sz:>10} {n:>12,} filas")
        print("-" * 72)
        print(f"  {'TOTAL a liberar por DROP':36} {total_drop/1024/1024:>9.0f} MB | {len(presentes)} tablas")

        fks = chequear_fks(c, presentes)
        if fks:
            print("\n  [!] FKs desde tablas que se MANTIENEN hacia tablas drop (CASCADE las quita):")
            for ref, refd in fks:
                print(f"      {ref} -> {refd}")
        else:
            print("  [OK] Ninguna tabla mantenida referencia a las tablas drop.")

    # ── (B) RETENTION opciones + cobertura local ──
    print("\n[B] RETENTION opciones (KEEP tabla, purgar viejas) + cobertura local")
    print("-" * 72)
    total_purge_rows = 0
    coberturas_ok = {}
    with eng_r.connect() as cr, eng_l.connect() as cl:
        for t, (col, dias) in retention.items():
            if not _existe(cr, t):
                print(f"  {t:34} (no existe, skip)")
                continue
            corte = hoy - timedelta(days=dias)
            r_tot = cr.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            r_del = cr.execute(text(f"SELECT COUNT(*) FROM {t} WHERE {col} < :c"), {"c": corte}).scalar()
            r_min, r_max = cr.execute(text(f"SELECT MIN({col}), MAX({col}) FROM {t}")).fetchone()
            # cobertura local
            try:
                l_tot = cl.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                l_min, l_max = cl.execute(text(f"SELECT MIN({col}), MAX({col}) FROM {t}")).fetchone()
                cubre = (l_min is not None and r_min is not None
                         and str(l_min) <= str(r_min) and l_tot >= r_tot)
            except Exception as e:
                l_tot, l_min, l_max, cubre = "err", "-", "-", False
            coberturas_ok[t] = cubre
            total_purge_rows += r_del
            flag = "OK" if cubre else "!! local NO cubre"
            print(f"  {t}")
            print(f"     Railway: {r_tot:>10,} filas | rango {r_min}..{r_max} | a borrar (<{corte}): {r_del:,}")
            print(f"     Local  : {str(l_tot):>10} filas | rango {l_min}..{l_max} | cobertura: {flag}")

    print("-" * 72)
    print(f"  Filas a purgar (opciones): {total_purge_rows:,}")

    if dry:
        print("\n" + sep)
        print("DRY RUN: no se toco nada. Revisa el reporte. Para ejecutar de verdad:")
        print("  1) correr el backup off-site (scripts/oneshot/backup_opciones_local.py)")
        print("  2) python scripts/oneshot/cleanup_railway_planB.py --execute")
        print(sep)
        return

    # ── EJECUCION REAL ──
    if any(not ok for ok in coberturas_ok.values()):
        print("\n[ABORT] Alguna tabla de opciones NO esta cubierta por local. "
              "No se purga nada para no perder datos. Revisar.")
        return

    print("\n[EXEC] Dropeando crudas...")
    if presentes:
        lista = ", ".join(presentes)
        with eng_r.begin() as c:
            c.execute(text(f"DROP TABLE IF EXISTS {lista} CASCADE"))
        print(f"  DROP OK: {len(presentes)} tablas.")

    print("[EXEC] Purgando opciones viejas...")
    purgadas = []
    with eng_r.begin() as c:
        for t, (col, dias) in retention.items():
            if not _existe(c, t):
                continue
            corte = hoy - timedelta(days=dias)
            res = c.execute(text(f"DELETE FROM {t} WHERE {col} < :c"), {"c": corte})
            print(f"  {t}: {res.rowcount:,} filas borradas (<{corte})")
            purgadas.append(t)

    if not args.no_vacuum:
        print("[EXEC] VACUUM FULL (libera disco; puede tardar)...")
        raw = eng_r.raw_connection()
        try:
            raw.set_isolation_level(0)  # autocommit (VACUUM no corre en transaccion)
            cur = raw.cursor()
            for t in purgadas:
                print(f"  VACUUM FULL {t} ...", flush=True)
                cur.execute(f"VACUUM FULL {t}")
            cur.close()
        finally:
            raw.close()

    print("\n" + sep)
    print("CLEANUP COMPLETADO.")
    print(sep)


if __name__ == "__main__":
    main()
