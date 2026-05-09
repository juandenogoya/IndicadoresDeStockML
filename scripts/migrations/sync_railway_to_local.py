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

import json
import math
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

def _serializar_jsonb(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas JSONB (dict/list) a JSON string para compatibilidad con psycopg2."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if not sample.empty and isinstance(sample.iloc[0], (dict, list)):
                df = df.copy()
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )
                log(f"    columna '{col}' serializada como JSON string")
    return df


def sync_tabla_nueva(rail_eng, local_eng, tabla: str, col_fecha, dry_run: bool, dias: int = None):
    """
    Sincroniza una tabla desde Railway hacia local.

    Estrategia automatica:
      - col_fecha=None  -> SIEMPRE reemplaza (tablas pequeñas / cabeceras, ej ft_estrategias)
      - col_fecha=str   -> INCREMENTAL si tabla local ya tiene datos (append desde max fecha)
                           CARGA INICIAL si tabla local esta vacia o no existe (replace)

    Si dias != None, limita la carga inicial a los ultimos N dias.
    """
    log(f"{tabla}...")

    # Detectar estado local
    count_local = 0
    max_local   = None
    try:
        with local_eng.connect() as conn:
            if col_fecha:
                r = conn.execute(
                    text(f"SELECT COUNT(*), MAX({col_fecha}) FROM {tabla}")
                ).fetchone()
                count_local, max_local = int(r[0]), r[1]
            else:
                count_local = int(conn.execute(
                    text(f"SELECT COUNT(*) FROM {tabla}")
                ).scalar() or 0)
    except Exception:
        count_local = 0  # tabla no existe todavia

    # Decidir modo
    if col_fecha is None:
        # Siempre reemplaza (tabla de referencia pequena)
        where        = "1=1"
        if_exists    = "replace"
        log(f"  Modo: replace completo ({count_local} filas locales actuales)")
    elif count_local == 0:
        # Carga inicial
        where = "1=1"
        if dias:
            where = f"{col_fecha} >= CURRENT_DATE - INTERVAL '{dias} days'"
            log(f"  Modo: carga inicial (ultimos {dias} dias)")
        else:
            log(f"  Modo: carga inicial (tabla vacia o no existe)")
        if_exists = "replace"
    else:
        # Incremental: solo filas nuevas
        if max_local:
            where     = f"{col_fecha} > '{max_local}'"
            if_exists = "append"
            log(f"  Modo: incremental desde {max_local} ({count_local:,} filas locales)")
        else:
            # col_fecha existe pero todos los valores son NULL -> replace
            where     = "1=1"
            if_exists = "replace"
            log(f"  Modo: replace (max fecha local es NULL)")

    order = f"ORDER BY {col_fecha}" if col_fecha else ""
    df = pd.read_sql(f"SELECT * FROM {tabla} WHERE {where} {order}", rail_eng)
    log(f"  Filas a insertar: {len(df)}")

    if df.empty:
        log(f"  OK - sin datos nuevos")
        return 0

    if dry_run:
        log(f"  DRY-RUN: {if_exists} {len(df)} filas en {tabla}")
        return len(df)

    df = _serializar_jsonb(df)

    df.to_sql(tabla, local_eng, if_exists=if_exists, index=False, method="multi", chunksize=500)
    accion = "insertadas" if if_exists == "append" else "cargadas"
    log(f"  OK - {len(df):,} filas {accion}")
    return len(df)


def sync_grupo(rail_eng, local_eng, tablas_conf: list, dry_run: bool) -> int:
    """
    Sincroniza un grupo de tablas nuevas en bloque.
    tablas_conf: lista de (tabla, col_fecha) donde col_fecha puede ser None.
    Retorna total de filas insertadas.
    """
    total = 0
    for tabla, col_fecha in tablas_conf:
        try:
            n = sync_tabla_nueva(rail_eng, local_eng, tabla, col_fecha, dry_run)
            total += n
        except Exception as e:
            log(f"  ERROR en {tabla}: {e}")
            raise
    return total


# ── Sync: opciones ───────────────────────────────────────────────────────────

def _opciones_local_conn(local_env: dict):
    """Conexion psycopg2 directa a local (para inserts con ON CONFLICT)."""
    return psycopg2.connect(
        host=local_env.get("DB_HOST", "localhost"),
        port=int(local_env.get("DB_PORT", 5432)),
        dbname=local_env.get("DB_NAME", "activos_ml"),
        user=local_env.get("DB_USER", "postgres"),
        password=local_env.get("DB_PASSWORD", ""),
    )


def _get_local_sqlalchemy_engine(local_env: dict):
    """SQLAlchemy engine para local construido desde el dict del .env (sin tocar os.environ)."""
    host = local_env.get("DB_HOST", "localhost")
    port = local_env.get("DB_PORT", "5432")
    name = local_env.get("DB_NAME", "activos_ml")
    user = local_env.get("DB_USER", "postgres")
    pwd  = local_env.get("DB_PASSWORD", "")
    url  = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url)


def _create_opciones_tables_local(local_eng):
    """Crea las tres tablas de opciones en local si no existen."""
    ddl_statements = [
        # opciones_snapshot
        """
        CREATE TABLE IF NOT EXISTS opciones_snapshot (
            id               SERIAL PRIMARY KEY,
            fecha_snapshot   DATE          NOT NULL,
            ticker           VARCHAR(20)   NOT NULL,
            vencimiento      DATE          NOT NULL,
            tipo             VARCHAR(4)    NOT NULL,
            strike           NUMERIC(12,2) NOT NULL,
            volumen          INTEGER,
            open_interest    INTEGER,
            iv               NUMERIC(8,4),
            bid              NUMERIC(10,4),
            ask              NUMERIC(10,4),
            precio_subyacente NUMERIC(12,4),
            hv_20d           NUMERIC(8,4),
            created_at       TIMESTAMP DEFAULT NOW(),
            CONSTRAINT opciones_snapshot_uniq
                UNIQUE (fecha_snapshot, ticker, vencimiento, tipo, strike)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_opciones_fecha      ON opciones_snapshot (fecha_snapshot)",
        "CREATE INDEX IF NOT EXISTS idx_opciones_ticker     ON opciones_snapshot (ticker)",
        "CREATE INDEX IF NOT EXISTS idx_opciones_tipo       ON opciones_snapshot (tipo)",
        "CREATE INDEX IF NOT EXISTS idx_opciones_vencimiento ON opciones_snapshot (vencimiento)",

        # opciones_resumen_diario
        """
        CREATE TABLE IF NOT EXISTS opciones_resumen_diario (
            id           SERIAL PRIMARY KEY,
            fecha        DATE          NOT NULL,
            ticker       VARCHAR(20)   NOT NULL,
            call_vol     BIGINT,
            put_vol      BIGINT,
            pcr_vol      NUMERIC(8,4),
            call_oi      BIGINT,
            put_oi       BIGINT,
            pcr_oi       NUMERIC(8,4),
            iv_call_avg  NUMERIC(8,4),
            iv_put_avg   NUMERIC(8,4),
            n_contratos  INTEGER,
            max_oi_strike NUMERIC(12,2),
            max_oi_venc  DATE,
            precio_sub   NUMERIC(12,4),
            created_at   TIMESTAMP DEFAULT NOW(),
            CONSTRAINT opciones_resumen_diario_uniq UNIQUE (fecha, ticker)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_resumen_fecha  ON opciones_resumen_diario (fecha)",
        "CREATE INDEX IF NOT EXISTS idx_resumen_ticker ON opciones_resumen_diario (ticker)",

        # opciones_zscore_diario
        """
        CREATE TABLE IF NOT EXISTS opciones_zscore_diario (
            id               SERIAL PRIMARY KEY,
            ticker           VARCHAR(20)   NOT NULL,
            fecha            DATE          NOT NULL,
            sector           VARCHAR(100),
            industry         VARCHAR(100),
            vol_calls        BIGINT,
            vol_puts         BIGINT,
            vol_total        BIGINT,
            pcr_vol          NUMERIC(8,4),
            iv_avg           NUMERIC(8,4),
            vol_calls_zscore NUMERIC(8,4),
            vol_puts_zscore  NUMERIC(8,4),
            vol_total_zscore NUMERIC(8,4),
            pcr_vol_zscore   NUMERIC(8,4),
            iv_zscore        NUMERIC(8,4),
            vol_relativo     NUMERIC(8,4),
            percentil_vol    INTEGER,
            ventana_dias     INTEGER,
            vol_media        NUMERIC(12,2),
            vol_std          NUMERIC(12,2),
            created_at       TIMESTAMP DEFAULT NOW(),
            CONSTRAINT opciones_zscore_diario_uniq UNIQUE (ticker, fecha)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_opciones_zscore_fecha   ON opciones_zscore_diario (fecha)",
        "CREATE INDEX IF NOT EXISTS idx_opciones_zscore_ticker  ON opciones_zscore_diario (ticker)",
        "CREATE INDEX IF NOT EXISTS idx_opciones_zscore_sector  ON opciones_zscore_diario (sector)",
    ]
    with local_eng.connect() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt.strip()))
        conn.commit()
    log("  Tablas opciones creadas/verificadas en local")


def _sync_opciones_incremental(
    rail_eng, local_env: dict, tabla: str,
    col_fecha: str, unique_cols: list,
    dry_run: bool, chunksize: int = 1000
) -> int:
    """
    Sync incremental de una tabla de opciones:
    - Detecta max fecha en local
    - Descarga desde Railway todo lo posterior
    - INSERT con ON CONFLICT DO NOTHING usando psycopg2 directo
    """
    # Max fecha local (si la tabla no existe aun, max_local = None)
    max_local = None
    conn_check = _opciones_local_conn(local_env)
    try:
        cur = conn_check.cursor()
        cur.execute(f"SELECT MAX({col_fecha}) FROM {tabla}")
        max_local = cur.fetchone()[0]
    except Exception:
        pass  # tabla no existe todavia (primera vez o dry-run)
    finally:
        conn_check.close()

    log(f"  Max {col_fecha} local: {max_local}")
    where = f"{col_fecha} > '{max_local}'" if max_local else "1=1"

    # Contar filas a traer
    total = pd.read_sql(f"SELECT COUNT(*) as n FROM {tabla} WHERE {where}", rail_eng).iloc[0]["n"]
    log(f"  Filas a insertar: {total}")

    if total == 0:
        log("  OK - sin datos nuevos")
        return 0

    if dry_run:
        log(f"  DRY-RUN: insertaria {total} filas en {tabla}")
        return int(total)

    if max_local is None:
        # ── CARGA INICIAL: tabla vacia ────────────────────────────────────────
        # Usamos pandas to_sql que convierte NaN -> NULL automaticamente via
        # SQLAlchemy, evitando problemas de tipos con columnas int/smallint.
        log(f"  Carga inicial via to_sql (NaN->NULL automatico)...")
        local_eng_obj = _get_local_sqlalchemy_engine(local_env)
        df_full = pd.read_sql(
            f"SELECT * FROM {tabla} WHERE {where} ORDER BY {col_fecha}",
            rail_eng
        )
        if "id" in df_full.columns:
            df_full = df_full.drop(columns=["id"])
        df_full.to_sql(
            tabla, local_eng_obj,
            if_exists="append", index=False,
            method="multi", chunksize=chunksize
        )
        log(f"  OK - {len(df_full)} filas insertadas en {tabla}")
        return len(df_full)

    # ── INCREMENTAL: tabla con datos, solo filas nuevas ───────────────────────
    # Usamos execute_batch con ON CONFLICT DO NOTHING.
    # NaN en columnas float se limpia manualmente via math.isnan().
    inserted = 0
    offset = 0
    while offset < total:
        df_chunk = pd.read_sql(
            f"SELECT * FROM {tabla} WHERE {where} ORDER BY {col_fecha} LIMIT {chunksize} OFFSET {offset}",
            rail_eng
        )
        if df_chunk.empty:
            break

        if "id" in df_chunk.columns:
            df_chunk = df_chunk.drop(columns=["id"])

        cols = list(df_chunk.columns)
        cols_str = ", ".join(cols)
        placeholders = ", ".join([f"%({c})s" for c in cols])
        sql = (
            f"INSERT INTO {tabla} ({cols_str}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT DO NOTHING"
        )

        # Limpiar NaN/inf -> None en todos los valores del dict
        # (pandas guarda None como NaN en columnas float al hacer to_dict)
        raw_records = df_chunk.to_dict("records")
        records = [
            {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
             for k, v in rec.items()}
            for rec in raw_records
        ]

        conn = _opciones_local_conn(local_env)
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
            inserted += cur.rowcount if cur.rowcount > 0 else 0
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        offset += chunksize
        log(f"    chunk {offset}/{total} ...")

    log(f"  OK - {inserted} filas insertadas en {tabla}")
    return inserted


def sync_opciones(rail_eng, local_eng, dry_run: bool):
    """Crea tablas de opciones en local si no existen y sincroniza desde Railway."""
    log("opciones -- creando tablas si no existen...")
    if not dry_run:
        _create_opciones_tables_local(local_eng)

    local_env = _parse_env_file(os.path.join(ROOT, ".env"))

    # opciones_resumen_diario (pequena, ~2600 filas)
    log("opciones_resumen_diario...")
    n1 = _sync_opciones_incremental(
        rail_eng, local_env, "opciones_resumen_diario", "fecha",
        ["fecha", "ticker"], dry_run
    )

    # opciones_zscore_diario (pequena, ~2600 filas)
    log("opciones_zscore_diario...")
    n2 = _sync_opciones_incremental(
        rail_eng, local_env, "opciones_zscore_diario", "fecha",
        ["ticker", "fecha"], dry_run
    )

    # opciones_snapshot (grande, ~1.2M filas) — chunks de 5000
    log("opciones_snapshot (puede tardar varios minutos)...")
    n3 = _sync_opciones_incremental(
        rail_eng, local_env, "opciones_snapshot", "fecha_snapshot",
        ["fecha_snapshot", "ticker", "vencimiento", "tipo", "strike"],
        dry_run, chunksize=5000
    )

    total = n1 + n2 + n3
    log(f"  opciones total: resumen={n1} zscore={n2} snapshot={n3}")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

TABLAS_DISPONIBLES = [
    "activos",
    "precios_diarios",
    "features_precio_accion",
    "features_market_structure",
    "alertas_scanner",
    "ticker_zscore_diario",
    "futuros_diarios",
    "opciones",
    "forward_testing",
    "bt_historico",
    "features_ml",
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

    # 6. ticker_zscore_diario (incremental por fecha)
    correr("ticker_zscore_diario", lambda: sync_tabla_nueva(
        rail_eng, local_eng, "ticker_zscore_diario", "fecha", dry_run
    ))

    # 7. futuros_diarios (incremental; carga inicial limitada a 365 dias)
    correr("futuros_diarios", lambda: sync_tabla_nueva(
        rail_eng, local_eng, "futuros_diarios", "fecha", dry_run, dias=365
    ))

    # 8. opciones (3 tablas: resumen_diario + zscore_diario + snapshot)
    correr("opciones", lambda: sync_opciones(rail_eng, local_eng, dry_run))

    # 9. forward_testing (5 tablas FT — Railway es fuente de verdad)
    #    ft_estrategias: col_fecha=None -> siempre replace (capital cambia cada dia)
    correr("forward_testing", lambda: sync_grupo(rail_eng, local_eng, [
        ("ft_estrategias",       None),           # replace: capital_actual cambia diario
        ("ft_candidatos_diarios","fecha"),         # incremental
        ("ft_operaciones",       "fecha_entrada"), # incremental
        ("ft_posiciones_diarias","fecha"),         # incremental
        ("ft_metricas_diarias",  "fecha"),         # incremental
    ], dry_run))

    # 10. bt_historico (4 tablas — resultados de backtesting historico)
    #     bt_hist_estrategias: col_fecha=None -> replace (pequeña, estatica)
    correr("bt_historico", lambda: sync_grupo(rail_eng, local_eng, [
        ("bt_hist_estrategias",     None),         # replace: 6 filas, estatica
        ("bt_hist_candidatos",      "fecha"),       # incremental
        ("bt_hist_operaciones",     "fecha_entrada"), # incremental
        ("bt_hist_metricas_diarias","fecha"),       # incremental
    ], dry_run))

    # 11. features_ml (4 tablas — entrenamiento ML + regimen macro + futuros)
    correr("features_ml", lambda: sync_grupo(rail_eng, local_eng, [
        ("features_ml",                "fecha"),
        ("features_sector",            "fecha"),
        ("features_regimen_macro",     "fecha"),
        ("indicadores_tecnicos_futuros","fecha"),
    ], dry_run))

    # ── Resumen ──
    errores = [t for t, (e, _) in resultados.items() if e == "ERROR"]
    print()
    print(SEP)
    print("  RESUMEN:")
    for tabla, (estado, detalle) in resultados.items():
        sufijo = f"{detalle:,} filas" if isinstance(detalle, int) else str(detalle)[:80]
        marca  = "OK  " if estado == "OK" else "FAIL"
        print(f"  [{marca}] {tabla:<35} {sufijo}")
    print(SEP)
    if errores:
        print(f"  {len(errores)} tabla(s) con error: {', '.join(errores)}")
        print(SEP)
    print()
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
