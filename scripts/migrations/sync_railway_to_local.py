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
  - forward_testing    : FT corre 100% en local (Plan C). Local es la fuente
                         de verdad de ft_*. Sincronizar desde Railway pisaria
                         datos vivos con datos viejos. Migracion puntual:
                         scripts/migrations/migrate_ft_railway_to_local.py
  - tablas de backtest : son locales por diseno

SI sincroniza earnings_calendar (replace completo): tabla de estado actual
poblada en Railway por scripts/refresh_earnings_calendar.py.

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

def sync_earnings_historico(rail_eng, local_eng, dry_run: bool):
    """
    Trae earnings_historico Railway -> local (sync FINAL del backfill hecho en
    Oracle). Merge idempotente por (ticker, fiscal_period_end) con ON CONFLICT
    DO NOTHING: preserva el schema/PK local y las filas ya cargadas, suma el
    resto. Una vez traido todo, la tabla Railway se puede dropear (transito).
    """
    log("earnings_historico -- merge idempotente (ticker, fiscal_period_end)...")

    # Si la tabla no existe en Railway aun (backfill no arrancado), no hay nada.
    try:
        df = pd.read_sql(
            "SELECT ticker, fiscal_period_end, announcement_date, report_time, "
            "fetched_at FROM earnings_historico ORDER BY ticker, fiscal_period_end",
            rail_eng)
    except Exception as e:
        log(f"  Railway sin earnings_historico ({str(e)[:60]}). Nada que traer.")
        return 0

    log(f"  Filas en Railway: {len(df)} ({df['ticker'].nunique()} tickers)")
    if df.empty:
        log("  OK - Railway vacio")
        return 0
    if dry_run:
        log(f"  DRY-RUN: mergearia hasta {len(df)} filas")
        return len(df)

    sql = (
        "INSERT INTO earnings_historico "
        "(ticker, fiscal_period_end, announcement_date, report_time, fetched_at) "
        "VALUES (%(ticker)s, %(fiscal_period_end)s, %(announcement_date)s, "
        "%(report_time)s, %(fetched_at)s) "
        "ON CONFLICT (ticker, fiscal_period_end) DO NOTHING"
    )
    local_env = _parse_env_file(os.path.join(ROOT, ".env"))
    conn = _opciones_local_conn(local_env)
    inserted = 0
    try:
        cur = conn.cursor()
        records = df.where(pd.notnull(df), None).to_dict("records")
        psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
        conn.commit()
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM earnings_historico")
        inserted, n_tk = cur.fetchone()
        log(f"  OK - local ahora: {inserted} filas / {n_tk} tickers")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return inserted


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

        # opciones_sector_zscore_diario (z-scores agregados por sector)
        """
        CREATE TABLE IF NOT EXISTS opciones_sector_zscore_diario (
            id                      SERIAL PRIMARY KEY,
            fecha                   DATE          NOT NULL,
            sector                  VARCHAR(100)  NOT NULL,
            n_tickers               SMALLINT,
            vol_calls_sector        BIGINT,
            vol_puts_sector         BIGINT,
            vol_total_sector        BIGINT,
            pcr_vol_sector          NUMERIC(8,4),
            pcr_vol_sector_zscore   NUMERIC(6,2),
            pcr_vol_sector_media    NUMERIC(8,4),
            pcr_vol_sector_std      NUMERIC(8,4),
            vol_total_sector_zscore NUMERIC(6,2),
            vol_total_sector_media  NUMERIC(18,2),
            vol_total_sector_std    NUMERIC(18,2),
            ventana_dias            SMALLINT,
            created_at              TIMESTAMP DEFAULT NOW(),
            CONSTRAINT opciones_sector_zscore_diario_uniq UNIQUE (fecha, sector)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_opc_sector_zscore_fecha  ON opciones_sector_zscore_diario (fecha)",
        "CREATE INDEX IF NOT EXISTS idx_opc_sector_zscore_sector ON opciones_sector_zscore_diario (sector)",

        # opciones_pcr_plazo_diario (PCR + muros OI por ventana, por ticker)
        """
        CREATE TABLE IF NOT EXISTS opciones_pcr_plazo_diario (
            id                    SERIAL PRIMARY KEY,
            fecha                 DATE          NOT NULL,
            ticker                VARCHAR(20)   NOT NULL,
            ventana               VARCHAR(10)   NOT NULL,
            dte_min               SMALLINT,
            dte_max               SMALLINT,
            call_vol              BIGINT,
            put_vol               BIGINT,
            pcr_vol               NUMERIC(8,4),
            call_oi               BIGINT,
            put_oi                BIGINT,
            pcr_oi                NUMERIC(8,4),
            veredicto_oi          CHAR(1),
            precio_sub            NUMERIC(12,4),
            soporte_strike        NUMERIC(12,4),
            soporte_oi            BIGINT,
            soporte_dist_pct      NUMERIC(6,2),
            soporte_fuerza        NUMERIC(5,1),
            resistencia_strike    NUMERIC(12,4),
            resistencia_oi        BIGINT,
            resistencia_dist_pct  NUMERIC(6,2),
            resistencia_fuerza    NUMERIC(5,1),
            expected_move         NUMERIC(12,4),
            zona_pct              NUMERIC(6,2),
            n_contratos           INTEGER,
            created_at            TIMESTAMP DEFAULT NOW(),
            CONSTRAINT opciones_pcr_plazo_diario_uniq UNIQUE (fecha, ticker, ventana)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pcr_plazo_fecha   ON opciones_pcr_plazo_diario (fecha)",
        "CREATE INDEX IF NOT EXISTS idx_pcr_plazo_ticker  ON opciones_pcr_plazo_diario (ticker)",
        "CREATE INDEX IF NOT EXISTS idx_pcr_plazo_ventana ON opciones_pcr_plazo_diario (ventana)",
        # Columnas v2 (muros mejorados) para tablas locales ya existentes (idempotente)
        "ALTER TABLE opciones_pcr_plazo_diario ADD COLUMN IF NOT EXISTS soporte_fuerza     NUMERIC(5,1)",
        "ALTER TABLE opciones_pcr_plazo_diario ADD COLUMN IF NOT EXISTS resistencia_fuerza NUMERIC(5,1)",
        "ALTER TABLE opciones_pcr_plazo_diario ADD COLUMN IF NOT EXISTS expected_move      NUMERIC(12,4)",
        "ALTER TABLE opciones_pcr_plazo_diario ADD COLUMN IF NOT EXISTS zona_pct           NUMERIC(6,2)",

        # opciones_sector_pcr_plazo_diario (PCR sectorial por ventana + zscore)
        """
        CREATE TABLE IF NOT EXISTS opciones_sector_pcr_plazo_diario (
            id                      SERIAL PRIMARY KEY,
            fecha                   DATE          NOT NULL,
            sector                  VARCHAR(100)  NOT NULL,
            ventana                 VARCHAR(10)   NOT NULL,
            dte_min                 SMALLINT,
            dte_max                 SMALLINT,
            n_tickers               SMALLINT,
            call_vol_sector         BIGINT,
            put_vol_sector          BIGINT,
            pcr_vol_sector          NUMERIC(8,4),
            call_oi_sector          BIGINT,
            put_oi_sector           BIGINT,
            pcr_oi_sector           NUMERIC(8,4),
            veredicto_oi            CHAR(1),
            pcr_vol_sector_zscore   NUMERIC(6,2),
            pcr_vol_sector_media    NUMERIC(8,4),
            pcr_vol_sector_std      NUMERIC(8,4),
            ventana_dias            SMALLINT,
            created_at              TIMESTAMP DEFAULT NOW(),
            CONSTRAINT opciones_sector_pcr_plazo_diario_uniq UNIQUE (fecha, sector, ventana)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_sec_pcr_plazo_fecha   ON opciones_sector_pcr_plazo_diario (fecha)",
        "CREATE INDEX IF NOT EXISTS idx_sec_pcr_plazo_sector  ON opciones_sector_pcr_plazo_diario (sector)",
        "CREATE INDEX IF NOT EXISTS idx_sec_pcr_plazo_ventana ON opciones_sector_pcr_plazo_diario (ventana)",
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
    # RE-PULL desde (e INCLUYENDO) el ultimo dia local con '>='.
    # ON CONFLICT DO NOTHING hace el insert idempotente: completa dias que
    # quedaron parciales (ej: un sync interrumpido a mitad) sin duplicar.
    # Antes usabamos '>' estricto -> un dia parcial quedaba atrapado para
    # siempre porque max_local ya apuntaba a esa fecha.
    where = f"{col_fecha} >= '{max_local}'" if max_local else "1=1"

    # Contar filas a traer
    total = pd.read_sql(f"SELECT COUNT(*) as n FROM {tabla} WHERE {where}", rail_eng).iloc[0]["n"]
    log(f"  Filas candidatas (re-pull {where}): {total}")

    if total == 0:
        log("  OK - sin datos nuevos")
        return 0

    if dry_run:
        log(f"  DRY-RUN: upsert {total} filas en {tabla}")
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
        log(f"  OK - {len(df_full):,} filas insertadas en {tabla}")
        return len(df_full)

    # ── INCREMENTAL / RE-PULL ─────────────────────────────────────────────────
    # Cargamos TODO el set >= max_local de una sola vez (es chico: 1-2 dias) y
    # lo insertamos en chunks. NO usamos paginacion LIMIT/OFFSET: con muchas
    # filas compartiendo la misma fecha, 'ORDER BY fecha' tiene empates y el
    # OFFSET no es determinista -> saltea/duplica filas. Cargar el set entero
    # y chunquear el INSERT elimina el problema.
    df_full = pd.read_sql(
        f"SELECT * FROM {tabla} WHERE {where} ORDER BY {col_fecha}",
        rail_eng
    )
    if "id" in df_full.columns:
        df_full = df_full.drop(columns=["id"])

    # Conteo local antes (para reportar inserciones reales)
    before = query_scalar(
        _get_local_sqlalchemy_engine(local_env),
        f"SELECT COUNT(*) FROM {tabla} WHERE {where}"
    )

    cols = list(df_full.columns)
    cols_str = ", ".join(cols)
    placeholders = ", ".join([f"%({c})s" for c in cols])
    sql = (
        f"INSERT INTO {tabla} ({cols_str}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT DO NOTHING"
    )

    conn = _opciones_local_conn(local_env)
    try:
        cur = conn.cursor()
        for start in range(0, len(df_full), chunksize):
            chunk = df_full.iloc[start:start + chunksize]
            raw_records = chunk.to_dict("records")
            records = [
                {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                 for k, v in rec.items()}
                for rec in raw_records
            ]
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
            conn.commit()
            log(f"    {min(start + chunksize, len(df_full)):,}/{len(df_full):,} ...")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    after = query_scalar(
        _get_local_sqlalchemy_engine(local_env),
        f"SELECT COUNT(*) FROM {tabla} WHERE {where}"
    )
    inserted = (after or 0) - (before or 0)
    log(f"  OK - {inserted:,} filas nuevas insertadas (de {len(df_full):,} candidatas; resto ya existia)")
    return inserted


def sync_opciones(rail_eng, local_eng, dry_run: bool):
    """
    Sincroniza SOLO opciones_snapshot (crudo) desde Railway.

    Tarea 17 (migracion Plan C "snapshot nube solo-crudo"): la nube captura
    unicamente el crudo; las derivadas (resumen/zscore/pcr_plazo/sector_*) ya NO
    viven en Railway -> se computan en LOCAL via compute_opciones_derivadas.py
    (paso [0b] de ft_run_diario, tras este sync). Por eso aca solo se trae el crudo.
    """
    log("opciones -- creando tablas si no existen...")
    if not dry_run:
        _create_opciones_tables_local(local_eng)  # crea snapshot + derivadas en local

    local_env = _parse_env_file(os.path.join(ROOT, ".env"))

    # opciones_snapshot (crudo) -- unica tabla de opciones que vive en Railway.
    log("opciones_snapshot (puede tardar varios minutos)...")
    n3 = _sync_opciones_incremental(
        rail_eng, local_env, "opciones_snapshot", "fecha_snapshot",
        ["fecha_snapshot", "ticker", "vencimiento", "tipo", "strike"],
        dry_run, chunksize=5000
    )

    log(f"  opciones total: snapshot={n3} "
        f"(derivadas se computan en LOCAL, no se sincronizan desde Railway)")
    return n3


# ── Sync: earnings_calendar ──────────────────────────────────────────────────

def sync_earnings_calendar(rail_eng, local_eng, dry_run: bool):
    """
    Sincroniza earnings_calendar Railway -> local en modo REPLACE.

    earnings_calendar es una tabla de ESTADO ACTUAL (una fila por ticker con
    su proxima fecha de earnings). No acumula historico: se pisa entera cada
    vez. Se usa TRUNCATE + INSERT para preservar el schema local (PK, indice,
    tipos) -- a diferencia de to_sql(if_exists='replace') que lo degrada.
    """
    log("earnings_calendar -- replace completo Railway -> local...")

    ddl_tabla = """
        CREATE TABLE IF NOT EXISTS earnings_calendar (
            ticker              VARCHAR(20) PRIMARY KEY,
            earnings_date       DATE,
            fecha_actualizacion TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """
    ddl_idx = ("CREATE INDEX IF NOT EXISTS idx_earnings_calendar_date "
               "ON earnings_calendar (earnings_date)")

    df = pd.read_sql(
        "SELECT ticker, earnings_date, fecha_actualizacion FROM earnings_calendar",
        rail_eng
    )
    log(f"  Railway: {len(df)} filas")

    if dry_run:
        log(f"  DRY-RUN: reemplazaria la tabla local con {len(df)} filas")
        return len(df)

    with local_eng.connect() as conn:
        conn.execute(text(ddl_tabla.strip()))
        conn.execute(text(ddl_idx))
        conn.execute(text("TRUNCATE earnings_calendar"))
        if not df.empty:
            sql = text("""
                INSERT INTO earnings_calendar
                    (ticker, earnings_date, fecha_actualizacion)
                VALUES (:ticker, :earnings_date, :fecha_actualizacion)
            """)
            registros = []
            for _, row in df.iterrows():
                ed = row["earnings_date"]
                fa = row["fecha_actualizacion"]
                registros.append({
                    "ticker":              row["ticker"],
                    "earnings_date":       None if pd.isna(ed) else ed,
                    "fecha_actualizacion": None if pd.isna(fa) else fa,
                })
            conn.execute(sql, registros)
        conn.commit()

    log(f"  OK - {len(df)} filas (local reemplazado)")
    return len(df)


# ── Sync: forward_testing (guard) ─────────────────────────────────────────────

def sync_forward_testing_guard(dry_run: bool):
    """
    Forward Testing corre 100% en LOCAL (Plan C). La DB local es la fuente de
    verdad de las tablas ft_*. Sincronizarlas desde Railway las pisaria con
    datos congelados -> se omite a proposito.

    Para una migracion puntual Railway -> local usar:
        scripts/migrations/migrate_ft_railway_to_local.py
    """
    log("forward_testing -- SKIP: FT corre en local, no se sincroniza desde Railway.")
    log("  (migracion puntual: scripts/migrations/migrate_ft_railway_to_local.py)")
    return 0


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
    "earnings_calendar",
    "earnings_historico",
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

    # 8b. earnings_calendar (replace completo — tabla de estado actual)
    correr("earnings_calendar",
           lambda: sync_earnings_calendar(rail_eng, local_eng, dry_run))

    # 8c. earnings_historico (sync FINAL del backfill de Oracle; merge idempotente).
    #     NO esta en el ciclo normal: se corre a mano con --tabla earnings_historico
    #     cuando el backfill en Railway esta completo. Aca queda para el modo full.
    correr("earnings_historico",
           lambda: sync_earnings_historico(rail_eng, local_eng, dry_run))

    # 9. forward_testing — NO se sincroniza: FT corre en local (ver guard)
    correr("forward_testing", lambda: sync_forward_testing_guard(dry_run))

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
