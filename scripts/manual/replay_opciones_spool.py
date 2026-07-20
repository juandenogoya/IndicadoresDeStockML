"""
replay_opciones_spool.py -- reinyecta a la DB los snapshots de opciones que
quedaron en disco porque la DB no estaba disponible.

CONTEXTO: 33_opciones_snapshot.py vuelca la chain cruda a
data/opciones_spool/opciones_YYYY-MM-DD.csv.gz ANTES de intentar la DB. Si la DB
falla (ej. Railway detenido por limite de consumo, incidente 2026-07-20), el
archivo queda pendiente. Este script lo carga y borra el spool al confirmar.

INVARIANTE: archivo presente en el spool = dato pendiente. Este script es el
unico que lo levanta.

El upsert es ON CONFLICT DO NOTHING (idempotente): correrlo de mas no duplica
ni pisa nada.

Uso:
    python scripts/manual/replay_opciones_spool.py --list
    python scripts/manual/replay_opciones_spool.py --dry-run
    python scripts/manual/replay_opciones_spool.py --target railway
    python scripts/manual/replay_opciones_spool.py --target local
    python scripts/manual/replay_opciones_spool.py --keep     # no borra el spool
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BATCH = 5000


def _cargar_env(target: str):
    """
    Selecciona la DB destino ANTES de importar src.data.database.

    Patron del proyecto: get_engine() mira DATABASE_URL primero (=Railway, viene
    de .env.local); si no esta, cae a DB_CONFIG (=local). Para forzar local hay
    que eliminar DATABASE_URL de os.environ.
    """
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    if target == "railway":
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
        if not os.getenv("DATABASE_URL"):
            print("  [ERROR] target=railway pero no hay DATABASE_URL en .env.local")
            sys.exit(1)
    else:
        os.environ.pop("DATABASE_URL", None)


SQL_INSERT = """
    INSERT INTO opciones_snapshot (
        fecha_snapshot, ticker, vencimiento, tipo, strike,
        volumen, open_interest, iv, bid, ask,
        precio_subyacente, hv_20d
    ) VALUES %s
    ON CONFLICT (fecha_snapshot, ticker, vencimiento, tipo, strike)
    DO NOTHING
"""

COLS = ["fecha_snapshot", "ticker", "vencimiento", "tipo", "strike",
        "volumen", "open_interest", "iv", "bid", "ask",
        "precio_subyacente", "hv_20d"]


def replay_archivo(path: str, dry_run: bool, keep: bool) -> bool:
    """Carga un spool. Devuelve True si quedo persistido (o dry-run OK)."""
    import psycopg2.extras
    from sqlalchemy import text
    from src.data.database import get_engine, get_connection
    from src.utils.opciones_spool import leer_spool, fecha_de_spool

    fecha = fecha_de_spool(path)
    filas = list(leer_spool(path))
    print(f"\n  {os.path.basename(path)}  ->  {len(filas):,} contratos  (fecha {fecha})")

    if not filas:
        print("    vacio -> descarto")
        if not dry_run and not keep:
            os.remove(path)
        return True

    if dry_run:
        tickers = len({f['ticker'] for f in filas})
        print(f"    [DRY RUN] {tickers} tickers, no escribo")
        return True

    valores = [tuple(f[c] for c in COLS) for f in filas]
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(valores), BATCH):
                    psycopg2.extras.execute_values(
                        cur, SQL_INSERT, valores[i:i + BATCH], page_size=500)
                    print(f"    ... {min(i + BATCH, len(valores)):,}/{len(valores):,}",
                          end="\r", flush=True)
    except Exception as e:
        print(f"\n    [ERROR] no se pudo escribir: {str(e)[:160]}")
        print(f"    El spool SE CONSERVA. Reintentar cuando la DB este arriba.")
        return False

    # Verificacion: contar lo que quedo realmente en la DB para esa fecha.
    eng = get_engine()
    with eng.connect() as conn:
        en_db = conn.execute(
            text("SELECT COUNT(*) FROM opciones_snapshot WHERE fecha_snapshot = :f"),
            {"f": fecha}).scalar() or 0
    print(f"\n    OK -> {en_db:,} contratos en DB para {fecha}")

    if en_db < len(filas):
        # Puede ser legitimo (duplicados internos del spool) pero conviene mirarlo.
        print(f"    [WARN] en DB hay menos filas que en el spool "
              f"({en_db:,} < {len(filas):,}). CONSERVO el spool por las dudas.")
        return False

    if keep:
        print("    --keep: spool conservado")
    else:
        os.remove(path)
        print("    spool borrado (dato ya en DB)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Reinyecta spools de opciones a la DB")
    ap.add_argument("--target", choices=["local", "railway"], default="railway",
                    help="DB destino (default: railway, que es donde vive el crudo)")
    ap.add_argument("--spool-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="No escribe")
    ap.add_argument("--list", action="store_true", help="Solo lista pendientes")
    ap.add_argument("--keep", action="store_true", help="No borra el spool tras cargarlo")
    args = ap.parse_args()

    _cargar_env(args.target)

    from src.utils.opciones_spool import listar_spools, DIR_SPOOL_DEFAULT
    dir_spool = args.spool_dir or os.path.join(ROOT, DIR_SPOOL_DEFAULT)
    pendientes = listar_spools(dir_spool)

    print("=" * 66)
    print(f"  REPLAY SPOOL OPCIONES  |  target={args.target}")
    print(f"  dir: {dir_spool}")
    print("=" * 66)

    if not pendientes:
        print("  Sin spools pendientes. Nada que hacer.")
        return

    print(f"  Pendientes: {len(pendientes)}")
    for p in pendientes:
        mb = os.path.getsize(p) / 1e6
        print(f"    - {os.path.basename(p)}  ({mb:.1f} MB)")

    if args.list:
        return

    ok = sum(replay_archivo(p, args.dry_run, args.keep) for p in pendientes)
    print("\n" + "=" * 66)
    print(f"  {ok}/{len(pendientes)} spools procesados.")
    if ok < len(pendientes):
        print("  Quedan pendientes -> reintentar cuando la DB este disponible.")
        sys.exit(1)
    print("=" * 66)


if __name__ == "__main__":
    main()
