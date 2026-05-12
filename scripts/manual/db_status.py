"""
db_status.py
Estado de todas las tablas de datos -- diagnostico rapido.

Uso:
    python scripts/manual/db_status.py                    (Railway, default)
    python scripts/manual/db_status.py --target railway   (idem)
    python scripts/manual/db_status.py --target local     (PostgreSQL local)
    python scripts/manual/db_status.py --dias 20

Target:
    railway : usa DATABASE_URL de .env.local (Railway prod)
    local   : usa DB_HOST/PORT/NAME/USER/PASSWORD de .env (PostgreSQL local)
"""

import os
import sys
import argparse
from datetime import date, datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from sqlalchemy import create_engine, text


def get_engine(target: str = "railway"):
    """
    Retorna el engine SQLAlchemy correspondiente al target solicitado.

    target='railway' : DATABASE_URL (definida en .env.local)
    target='local'   : compone DSN desde DB_HOST/PORT/NAME/USER/PASSWORD (.env)
    """
    if target == "local":
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")
        name = os.environ.get("DB_NAME", "activos_ml")
        user = os.environ.get("DB_USER", "postgres")
        pwd  = os.environ.get("DB_PASSWORD", "")
        url  = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
        return create_engine(url)
    # default: railway
    return create_engine(os.environ["DATABASE_URL"])


def dias_habiles(desde, hasta):
    """Lista de dias L-V entre dos fechas (sin filtrar feriados US)."""
    dias = []
    d = desde
    while d <= hasta:
        if d.weekday() < 5:
            dias.append(d)
        d += timedelta(days=1)
    return dias


def tag_estado(condicion, ok="OK", mal="!! PROBLEMA"):
    return ok if condicion else mal


def sep(titulo):
    print(f"\n  {titulo}")
    print("  " + "-" * 56)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dias", type=int, default=14,
                        help="Ventana de dias a revisar (default: 14)")
    parser.add_argument("--target", choices=["railway", "local"], default="railway",
                        help="DB a consultar: 'railway' (default) o 'local'")
    args = parser.parse_args()

    hoy   = date.today()
    desde = hoy - timedelta(days=args.dias)

    print()
    print("=" * 60)
    print(f"  DB STATUS  |  TARGET: {args.target.upper():<8} |  {hoy}  |  {datetime.now().strftime('%H:%M')}")
    print("=" * 60)

    engine = get_engine(args.target)
    habiles = dias_habiles(desde, hoy - timedelta(days=1))
    habiles_recientes = habiles[-10:]   # ultimos 10 dias habiles

    with engine.connect() as conn:

        # ── OPCIONES SNAPSHOT ─────────────────────────────────────
        sep("OPCIONES SNAPSHOT")
        try:
            rows = conn.execute(text("""
                SELECT fecha_snapshot,
                       COUNT(*)               AS filas,
                       COUNT(DISTINCT ticker)  AS tickers
                FROM opciones_snapshot
                WHERE fecha_snapshot >= :desde
                GROUP BY fecha_snapshot
                ORDER BY fecha_snapshot
            """), {"desde": desde}).fetchall()
            db_snap = {r[0]: (r[1], r[2]) for r in rows}
            ok = falta = 0
            print(f"  {'fecha':<14} {'filas':>10}  {'tickers':>8}  estado")
            for d in habiles_recientes:
                if d in db_snap:
                    filas, tickers = db_snap[d]
                    estado = "OK" if filas >= 10_000 else "!! POCOS DATOS"
                    print(f"  {str(d):<14} {filas:>10,}  {tickers:>8}  {estado}")
                    ok += 1
                else:
                    print(f"  {str(d):<14} {'---':>10}  {'---':>8}  !! FALTA")
                    falta += 1
            estado_cob = "OK" if falta == 0 else f"!! {falta} dia(s) sin datos"
            print(f"\n  Cobertura: {ok}/{ok+falta} dias habiles  [{estado_cob}]")
        except Exception as e:
            print(f"  ERROR: {e}")

        # ── OPCIONES RESUMEN DIARIO ───────────────────────────────
        sep("OPCIONES RESUMEN DIARIO")
        try:
            rows2 = conn.execute(text("""
                SELECT fecha, COUNT(*) AS tickers
                FROM opciones_resumen_diario
                WHERE fecha >= :desde
                GROUP BY fecha
                ORDER BY fecha
            """), {"desde": desde}).fetchall()
            db_res = {r[0]: r[1] for r in rows2}
            print(f"  {'fecha':<14} {'tickers':>8}  estado")
            for d in habiles_recientes:
                if d in db_res:
                    t = db_res[d]
                    estado = "OK" if t >= 100 else "!! POCOS TICKERS"
                    print(f"  {str(d):<14} {t:>8}  {estado}")
                else:
                    print(f"  {str(d):<14} {'---':>8}  !! FALTA")
        except Exception as e:
            print(f"  ERROR: {e}")

        # ── PRECIOS DIARIOS ───────────────────────────────────────
        sep("PRECIOS DIARIOS (Scanner)")
        try:
            total_t = conn.execute(text(
                "SELECT COUNT(DISTINCT ticker) FROM precios_diarios"
            )).scalar()
            max_f = conn.execute(text(
                "SELECT MAX(fecha) FROM precios_diarios"
            )).scalar()
            dias_atraso = (hoy - max_f).days if max_f else 999
            print(f"  Total tickers : {total_t}")
            print(f"  Fecha max     : {max_f}  [{tag_estado(dias_atraso <= 3)}]")
            umbral_stale = hoy - timedelta(days=3)
            stale = conn.execute(text("""
                SELECT ticker, MAX(fecha) AS ultimo
                FROM precios_diarios
                GROUP BY ticker
                HAVING MAX(fecha) < :umbral
                ORDER BY MAX(fecha), ticker
            """), {"umbral": umbral_stale}).fetchall()
            if stale:
                print(f"  Tickers desactualizados (> 3 dias):")
                for r in stale:
                    print(f"    !! {r[0]:<8}  ultimo: {r[1]}")
            else:
                print(f"  Tickers desactualizados : ninguno  [OK]")
        except Exception as e:
            print(f"  ERROR: {e}")

        # ── ALERTAS SCANNER ───────────────────────────────────────
        sep("ALERTAS SCANNER")
        try:
            rows3 = conn.execute(text("""
                SELECT DATE(scan_fecha)                AS dia,
                       COUNT(DISTINCT ticker)          AS tickers,
                       SUM(CASE WHEN alert_nivel != 'NEUTRAL'
                                THEN 1 ELSE 0 END)     AS alertas
                FROM alertas_scanner
                WHERE scan_fecha >= :desde
                GROUP BY DATE(scan_fecha)
                ORDER BY dia DESC
                LIMIT 7
            """), {"desde": desde}).fetchall()
            if rows3:
                print(f"  {'fecha':<14} {'tickers':>8}  {'alertas':>8}")
                for r in rows3:
                    print(f"  {str(r[0]):<14} {r[1]:>8}  {r[2]:>8}")
            else:
                print("  Sin datos recientes")
        except Exception as e:
            print(f"  ERROR: {e}")

        # ── FUTUROS DE INDICES ────────────────────────────────────
        sep("FUTUROS DE INDICES")
        try:
            rows4 = conn.execute(text("""
                SELECT ticker,
                       MAX(fecha)  AS ultimo,
                       COUNT(*)    AS total_dias
                FROM futuros_diarios
                GROUP BY ticker
                ORDER BY ticker
            """)).fetchall()
            fut_map = {r[0]: (r[1], r[2]) for r in rows4}
            for s in ["ES=F", "YM=F", "NQ=F", "RTY=F"]:
                if s in fut_map:
                    ultimo, total = fut_map[s]
                    dias_atraso = (hoy - ultimo).days if ultimo else 999
                    estado = "OK" if dias_atraso <= 3 else f"!! {dias_atraso} dias de atraso"
                    print(f"  {s:<8} ultimo={ultimo}  total={total:,} dias  [{estado}]")
                else:
                    print(f"  {s:<8} !! SIN DATOS EN DB")
        except Exception as e:
            print(f"  ERROR: {e}")

        # ── MODULO ARGENTINA (IOL) ────────────────────────────────
        sep("ARGENTINA IOL -- SCANNER AR")
        try:
            tabla_ar = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'alertas_scanner_ar')"
            )).scalar()
            if not tabla_ar:
                print("  Tabla alertas_scanner_ar NO EXISTE -- correr init_tablas_ar.py")
            else:
                rows_ar = conn.execute(text("""
                    SELECT DATE(scan_fecha)           AS dia,
                           COUNT(DISTINCT ticker)     AS tickers,
                           SUM(CASE WHEN alert_nivel = 'COMPRA_FUERTE'
                                    THEN 1 ELSE 0 END) AS fuertes,
                           SUM(CASE WHEN alert_nivel = 'COMPRA'
                                    THEN 1 ELSE 0 END) AS compras,
                           SUM(CASE WHEN alert_nivel = 'VENTA'
                                    THEN 1 ELSE 0 END) AS ventas
                    FROM alertas_scanner_ar
                    WHERE scan_fecha >= :desde
                    GROUP BY DATE(scan_fecha)
                    ORDER BY dia DESC
                    LIMIT 7
                """), {"desde": desde}).fetchall()
                if rows_ar:
                    print(f"  {'fecha':<14} {'tickers':>8}  {'fuertes':>8}  {'compras':>8}  {'ventas':>7}")
                    for r in rows_ar:
                        print(f"  {str(r[0]):<14} {r[1]:>8}  {r[2]:>8}  {r[3]:>8}  {r[4]:>7}")
                else:
                    print("  Sin datos recientes en alertas_scanner_ar")
                total_ar = conn.execute(text(
                    "SELECT COUNT(*) FROM activos_ar WHERE activo = true"
                )).scalar()
                print(f"\n  Universo activos_ar: {total_ar} tickers activos")
        except Exception as e:
            print(f"  ERROR: {e}")

        sep("ARGENTINA IOL -- OPCIONES AR GREGAS")
        try:
            tabla_opc = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'opciones_ar_gregas')"
            )).scalar()
            if not tabla_opc:
                print("  Tabla opciones_ar_gregas NO EXISTE -- correr init_tablas_ar.py")
            else:
                rows_opc = conn.execute(text("""
                    SELECT fecha,
                           COUNT(DISTINCT ticker_subyacente) AS tickers,
                           COUNT(*)                          AS strikes,
                           ROUND(AVG(implied_volatility)::numeric, 4) AS iv_prom
                    FROM opciones_ar_gregas
                    WHERE fecha >= :desde
                    GROUP BY fecha
                    ORDER BY fecha DESC
                    LIMIT 7
                """), {"desde": desde}).fetchall()
                if rows_opc:
                    print(f"  {'fecha':<14} {'tickers':>8}  {'strikes':>8}  {'IV_prom':>8}")
                    for r in rows_opc:
                        print(f"  {str(r[0]):<14} {r[1]:>8}  {r[2]:>8}  {str(r[3]):>8}")
                else:
                    print("  Sin datos en opciones_ar_gregas (se construye dia a dia)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
