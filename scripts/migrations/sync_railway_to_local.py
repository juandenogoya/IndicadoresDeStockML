"""
sync_railway_to_local.py
Sincroniza la DB local con Railway (fuente de verdad de produccion).

Estrategia por tabla:
  - activos            : INSERT 74 tickers faltantes (ON CONFLICT DO NOTHING)
  - precios_diarios    : pull solo los 74 tickers nuevos (local ya es mas reciente para los 125)
  - features_pa        : idem
  - features_ms        : idem
  - alertas_scanner    : pull desde April 14 en adelante (Railway tiene hasta May 6)
  - ticker_zscore      : tabla no existe en local -> crear + sync completo
  - futuros_diarios    : tabla no existe en local -> crear + sync completo

NO sincroniza:
  - features_1w        : Railway esta vacia, local tiene datos reales -> skip
  - opciones_*         : arquitectura separada, pendiente decision
  - tablas de backtest : son locales por diseno

Uso:
    python scripts/migrations/sync_railway_to_local.py
    python scripts/migrations/sync_railway_to_local.py --dry-run
    python scripts/migrations/sync_railway_to_local.py --tabla precios_diarios
    python scripts/migrations/sync_railway_to_local.py --tabla alertas_scanner
"""

import sys
import os
import argparse
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

SEP = "=" * 65


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Conexiones ────────────────────────────────────────────────────────────────

def _parse_env_file(path: str) -> dict:
    """Lee un archivo .env y retorna dict clave->valor (sin tocar os.environ)."""
    result = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                result[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return result


def get_railway_engine():
    """Engine SQLAlchemy apuntando a Railway (lee .env.local directamente)."""
    env = _parse_env_file(os.path.join(ROOT, ".env.local"))
    db_url = env.get("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError("DATABASE_URL no encontrado en .env.local")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, connect_args={"sslmode": "require"})


def get_local_engine():
    """
    Engine SQLAlchemy apuntando a DB local (lee .env directamente,
    sin tocar os.environ para evitar contaminacion con .env.local).
    """
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    name = env.get("DB_NAME", "activos_ml")
    user = env.get("DB_USER", "postgres")
    pwd  = env.get("DB_PASSWORD", "")
    url  = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url)


# ── Helpers ───────────────────────────────────────────────────────────────────

def query_scalar(engine, sql: str):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def tablas_locales(local_eng) -> list:
    sql = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """
    with local_eng.connect() as conn:
        return [r[0] for r in conn.execute(text(sql)).fetchall()]


# ── Sync: activos ─────────────────────────────────────────────────────────────

def sync_activos(rail_eng, local_eng, dry_run: bool):
    log("activos -- buscando tickers faltantes...")

    df_rail = pd.read_sql(
        "SELECT ticker, sector, industry, activo FROM activos ORDER BY ticker",
        rail_eng
    )
    df_local = pd.read_sql("SELECT ticker FROM activos", local_eng)

    nuevos = df_rail[~df_rail["ticker"].isin(df_local["ticker"])]
    log(f"  Railway: {len(df_rail)} | Local: {len(df_local)} | Nuevos: {len(nuevos)}")

    if nuevos.empty:
        log("  OK - sin cambios necesarios")
        return 0

    if dry_run:
        log(f"  DRY-RUN: insertaria {len(nuevos)} tickers: {list(nuevos['ticker'])}")
        return len(nuevos)

    with local_eng.connect() as conn:
        for _, row in nuevos.iterrows():
            conn.execute(text("""
                INSERT INTO activos (ticker, sector, industry, activo)
                VALUES (:ticker, :sector, :industry, :activo)
                ON CONFLICT (ticker) DO NOTHING
            """), {
                "ticker":   row["ticker"],
                "sector":   row.get("sector"),
                "industry": row.get("industry"),
                "activo":   bool(row["activo"]) if row["activo"] is not None else True,
            })
        conn.commit()

    log(f"  OK - {len(nuevos)} tickers insertados")
    return len(nuevos)


# ── Sync: tablas ticker+fecha (precios, features) ────────────────────────────

def sync_ticker_fecha(
    rail_eng, local_eng, tabla: str,
    col_fecha: str, dry_run: bool,
    solo_tickers_nuevos: bool = True
):
    """
    Sincroniza una tabla con clave (ticker, fecha).
    Si solo_tickers_nuevos=True, solo trae filas de tickers que no existen en local.
    """
    log(f"{tabla} ({'solo nuevos' if solo_tickers_nuevos else 'incremental fecha'})...")

    # Tickers que YA existen en local
    tickers_local = pd.read_sql(
        f"SELECT DISTINCT ticker FROM {tabla}", local_eng
    )["ticker"].tolist()

    # Tickers disponibles en Railway
    tickers_rail = pd.read_sql(
        f"SELECT DISTINCT ticker FROM {tabla}", rail_eng
    )["ticker"].tolist()

    if solo_tickers_nuevos:
        tickers_pull = [t for t in tickers_rail if t not in tickers_local]
        if not tickers_pull:
            log(f"  OK - no hay tickers nuevos en Railway para {tabla}")
            return 0
        log(f"  Tickers a importar: {len(tickers_pull)}")
        tickers_str = "'" + "','".join(tickers_pull) + "'"
        where = f"ticker IN ({tickers_str})"
    else:
        # Incremental por fecha (para alertas_scanner etc)
        max_local = query_scalar(local_eng, f"SELECT MAX({col_fecha}) FROM {tabla}")
        if max_local:
            where = f"{col_fecha} > '{max_local}'"
            log(f"  Pull desde {max_local}")
        else:
            where = "1=1"
            log(f"  Pull completo (tabla local vacia)")

    df = pd.read_sql(f"SELECT * FROM {tabla} WHERE {where}", rail_eng)
    log(f"  Filas a insertar: {len(df)}")

    if df.empty:
        log(f"  OK - sin datos nuevos")
        return 0

    if dry_run:
        log(f"  DRY-RUN: insertaria {len(df)} filas en {tabla}")
        return len(df)

    # Dropear columna 'id' (serial de Railway) para que local auto-asigne su propio id
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # INSERT con pandas (tabla ya existe, schema compatible)
    df.to_sql(tabla, local_eng, if_exists="append", index=False, method="multi", chunksize=500)
    log(f"  OK - {len(df)} filas insertadas en {tabla}")
    return len(df)


# ── Sync: alertas_scanner ─────────────────────────────────────────────────────

def sync_alertas(rail_eng, local_eng, dry_run: bool):
    """
    Incremental por precio_fecha (la columna usada en el unique index).
    Usa INSERT ... ON CONFLICT DO NOTHING para evitar duplicados.
    """
    log("alertas_scanner -- incremental por precio_fecha...")

    max_local = query_scalar(local_eng, "SELECT MAX(precio_fecha) FROM alertas_scanner")
    log(f"  Max precio_fecha local: {max_local}")

    where = f"precio_fecha > '{max_local}'" if max_local else "1=1"
    df = pd.read_sql(
        f"SELECT * FROM alertas_scanner WHERE {where} ORDER BY precio_fecha, ticker",
        rail_eng
    )
    log(f"  Filas candidatas: {len(df)}")

    if df.empty:
        log("  OK - sin alertas nuevas")
        return 0

    if dry_run:
        log(f"  DRY-RUN: insertaria hasta {len(df)} filas")
        return len(df)

    # Dropear id para que local auto-asigne
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # INSERT con ON CONFLICT DO NOTHING via psycopg2 directo
    cols = list(df.columns)
    cols_str = ", ".join(cols)
    placeholders = ", ".join([f"%({c})s" for c in cols])
    sql = (
        f"INSERT INTO alertas_scanner ({cols_str}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT DO NOTHING"
    )

    local_env = _parse_env_file(os.path.join(ROOT, ".env"))
    conn = psycopg2.connect(
        host=local_env.get("DB_HOST", "localhost"),
        port=int(local_env.get("DB_PORT", 5432)),
        dbname=local_env.get("DB_NAME", "activos_ml"),
        user=local_env.get("DB_USER", "postgres"),
        password=local_env.get("DB_PASSWORD", ""),
    )
    inserted = 0
    try:
        cur = conn.cursor()
        for batch_start in range(0, len(df), 200):
            batch = df.iloc[batch_start:batch_start + 200]
            records = batch.where(pd.notnull(batch), None).to_dict("records")
            psycopg2.extras.execute_batch(cur, sql, records, page_size=200)
            inserted += cur.rowcount if cur.rowcount > 0 else 0
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    log(f"  OK - {inserted} alertas insertadas (de {len(df)} candidatas)")
    return inserted


# ── Sync: tablas que no existen en local (crear + poblar) ────────────────────

def sync_tabla_nueva(rail_eng, local_eng, tabla: str, col_fecha: str, dry_run: bool, dias: int = None):
    """
    Trae una tabla completa de Railway que no existe en local.
    Si dias != None, limita a los ultimos N dias.
    """
    log(f"{tabla} -- tabla nueva en local...")

    where = "1=1"
    if dias:
        where = f"{col_fecha} >= CURRENT_DATE - INTERVAL '{dias} days'"
        log(f"  Limitando a ultimos {dias} dias")

    df = pd.read_sql(f"SELECT * FROM {tabla} WHERE {where} ORDER BY {col_fecha}", rail_eng)
    log(f"  Filas disponibles: {len(df)}")

    if df.empty:
        log(f"  Sin datos en Railway para {tabla}")
        return 0

    if dry_run:
        log(f"  DRY-RUN: crearia tabla y cargaria {len(df)} filas")
        return len(df)

    # Crear tabla + insertar (pandas infiere schema basico)
    df.to_sql(tabla, local_eng, if_exists="replace", index=False, method="multi", chunksize=500)
    log(f"  OK - tabla {tabla} creada con {len(df)} filas")
    return len(df)


# ── Main ──────────────────────────────────────────────────────────────────────

TABLAS_DISPONIBLES = [
    "activos",
    "precios_diarios",
    "features_precio_accion",
    "features_market_structure",
    "alertas_scanner",
    "ticker_zscore_diario",
    "futuros_diarios",
]


def main():
    parser = argparse.ArgumentParser(description="Sync Railway -> Local DB")
    parser.add_argument("--dry-run", action="store_true", help="Simula sin escribir")
    parser.add_argument("--tabla", choices=TABLAS_DISPONIBLES,
                        help="Sincronizar solo esta tabla")
    args = parser.parse_args()
    dry_run = args.dry_run

    print()
    print(SEP)
    print(f"  SYNC  Railway -> Local  |  {date.today()}")
    if dry_run:
        print(f"  MODO: DRY-RUN (sin escritura)")
    if args.tabla:
        print(f"  Tabla: {args.tabla}")
    print(SEP)
    print()

    rail_eng  = get_railway_engine()
    local_eng = get_local_engine()
    tablas_ok = tablas_locales(local_eng)

    resultados = {}

    def correr(nombre, fn):
        if args.tabla and args.tabla != nombre:
            return
        print()
        try:
            n = fn()
            resultados[nombre] = ("OK", n)
        except Exception as e:
            log(f"  ERROR en {nombre}: {e}")
            resultados[nombre] = ("ERROR", str(e))

    # 1. activos
    correr("activos", lambda: sync_activos(rail_eng, local_eng, dry_run))

    # 2. precios_diarios (solo 74 tickers nuevos — local ya es mas reciente para los 125)
    correr("precios_diarios", lambda: sync_ticker_fecha(
        rail_eng, local_eng, "precios_diarios", "fecha", dry_run,
        solo_tickers_nuevos=True
    ))

    # 3. features_precio_accion (solo tickers nuevos)
    correr("features_precio_accion", lambda: sync_ticker_fecha(
        rail_eng, local_eng, "features_precio_accion", "fecha", dry_run,
        solo_tickers_nuevos=True
    ))

    # 4. features_market_structure (solo tickers nuevos)
    correr("features_market_structure", lambda: sync_ticker_fecha(
        rail_eng, local_eng, "features_market_structure", "fecha", dry_run,
        solo_tickers_nuevos=True
    ))

    # 5. alertas_scanner (incremental por fecha — Railway tiene hasta May 6)
    correr("alertas_scanner", lambda: sync_alertas(rail_eng, local_eng, dry_run))

    # 6. ticker_zscore_diario (no existe en local — crear + sync completo)
    correr("ticker_zscore_diario", lambda: sync_tabla_nueva(
        rail_eng, local_eng, "ticker_zscore_diario", "fecha", dry_run
    ))

    # 7. futuros_diarios (no existe en local — crear + sync, ultimos 365 dias)
    correr("futuros_diarios", lambda: sync_tabla_nueva(
        rail_eng, local_eng, "futuros_diarios", "fecha", dry_run, dias=365
    ))

    # ── Resumen ──
    print()
    print(SEP)
    print("  RESUMEN:")
    for tabla, (estado, detalle) in resultados.items():
        sufijo = f"{detalle} filas" if isinstance(detalle, int) else detalle
        print(f"  {tabla:<35} {estado}  {sufijo}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
