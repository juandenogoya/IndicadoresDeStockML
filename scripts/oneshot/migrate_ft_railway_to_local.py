"""
migrate_ft_railway_to_local.py
Migracion de un solo uso: trae las 5 tablas ft_* de Railway a Local.

Contexto:
    Los bots de Forward Testing venian escribiendo en Railway por error
    (cargaban .env.local -> DATABASE_URL=Railway). Plan C define que FT
    debe correr 100% en local. Esta migracion mueve la fuente de verdad
    de FT a local.

Problema que resuelve:
    Las tablas ft_* locales tienen schema DEGRADADO: un sync previo con
    pandas to_sql(if_exists='replace') las recreo sin SERIAL id, sin PK,
    sin constraints ni tipos JSONB. Ademas su data esta stale (12 dias).

Estrategia:
    1. Backup de las tablas ft_* locales actuales a CSV.
    2. DROP + CREATE cada tabla local con el schema REAL de Railway.
    3. Carga completa desde Railway preservando id y timestamps.
    4. Reset de secuencias id a MAX(id)+1.
    5. Todo en UNA transaccion: si algo falla, rollback total.
    6. Verificacion de conteos local == Railway.

Uso:
    python scripts/migrations/migrate_ft_railway_to_local.py --dry-run
    python scripts/migrations/migrate_ft_railway_to_local.py --confirm
"""

import sys
import os
import argparse
import json
import math
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text

SEP = "=" * 68


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Entorno ──────────────────────────────────────────────────────────────────

def _parse_env_file(path: str) -> dict:
    """Lee un .env y retorna dict clave->valor (sin tocar os.environ)."""
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
    """SQLAlchemy engine -> Railway (lee .env.local directamente)."""
    env = _parse_env_file(os.path.join(ROOT, ".env.local"))
    db_url = env.get("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError("DATABASE_URL no encontrado en .env.local")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, connect_args={"sslmode": "require"})


def get_local_conn():
    """Conexion psycopg2 directa -> DB local (lee .env directamente)."""
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    return psycopg2.connect(
        host=env.get("DB_HOST", "localhost"),
        port=int(env.get("DB_PORT", 5432)),
        dbname=env.get("DB_NAME", "activos_ml"),
        user=env.get("DB_USER", "postgres"),
        password=env.get("DB_PASSWORD", ""),
    )


def get_local_engine():
    """SQLAlchemy engine -> DB local (para lecturas con pandas)."""
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    name = env.get("DB_NAME", "activos_ml")
    user = env.get("DB_USER", "postgres")
    pwd  = env.get("DB_PASSWORD", "")
    return create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}")


# ── Schema real de las tablas ft_* (tomado de Railway) ───────────────────────

DDL = {
    "ft_estrategias": [
        """
        CREATE TABLE ft_estrategias (
            id                   SERIAL PRIMARY KEY,
            nombre               TEXT NOT NULL UNIQUE,
            descripcion          TEXT,
            logica               TEXT NOT NULL,
            parametros           JSONB,
            capital_inicial      NUMERIC NOT NULL DEFAULT 100000,
            capital_actual       NUMERIC NOT NULL DEFAULT 100000,
            cash_disponible      NUMERIC NOT NULL DEFAULT 100000,
            capital_inmovilizado NUMERIC NOT NULL DEFAULT 0,
            activa               BOOLEAN NOT NULL DEFAULT TRUE,
            fecha_inicio         DATE,
            fecha_fin            DATE,
            creado_en            TIMESTAMP DEFAULT NOW()
        )
        """,
    ],
    "ft_operaciones": [
        """
        CREATE TABLE ft_operaciones (
            id              SERIAL PRIMARY KEY,
            estrategia_id   INTEGER NOT NULL,
            ticker          TEXT NOT NULL,
            lado            TEXT NOT NULL DEFAULT 'long',
            fecha_entrada   DATE NOT NULL,
            precio_entrada  NUMERIC NOT NULL,
            cantidad        INTEGER NOT NULL,
            capital_entrada NUMERIC NOT NULL,
            stop_loss       NUMERIC,
            take_profit     NUMERIC,
            score_entrada   NUMERIC,
            detalle_entrada JSONB,
            fecha_salida    DATE,
            precio_salida   NUMERIC,
            pnl             NUMERIC,
            pnl_pct         NUMERIC,
            motivo_salida   TEXT,
            creado_en       TIMESTAMP DEFAULT NOW()
        )
        """,
        "CREATE INDEX idx_ft_op_estrategia ON ft_operaciones (estrategia_id)",
        "CREATE INDEX idx_ft_op_ticker     ON ft_operaciones (ticker)",
        "CREATE INDEX idx_ft_op_abiertas   ON ft_operaciones (estrategia_id) "
        "WHERE fecha_salida IS NULL",
    ],
    "ft_candidatos_diarios": [
        """
        CREATE TABLE ft_candidatos_diarios (
            id              SERIAL PRIMARY KEY,
            estrategia_id   INTEGER NOT NULL,
            fecha           DATE NOT NULL,
            ticker          VARCHAR(20) NOT NULL,
            score           NUMERIC NOT NULL,
            entro           BOOLEAN NOT NULL DEFAULT FALSE,
            motivo_skip     VARCHAR(60),
            precio_apertura NUMERIC,
            retorno_5d      NUMERIC,
            retorno_10d     NUMERIC,
            retorno_20d     NUMERIC,
            CONSTRAINT ft_candidatos_diarios_estrategia_id_fecha_ticker_key
                UNIQUE (estrategia_id, fecha, ticker)
        )
        """,
        "CREATE INDEX idx_ft_cand_eid_fecha ON ft_candidatos_diarios "
        "(estrategia_id, fecha DESC)",
    ],
    "ft_metricas_diarias": [
        """
        CREATE TABLE ft_metricas_diarias (
            estrategia_id            INTEGER NOT NULL,
            fecha                    DATE NOT NULL,
            capital_total            NUMERIC NOT NULL,
            cash_disponible          NUMERIC NOT NULL,
            capital_inmovilizado     NUMERIC NOT NULL,
            posiciones_abiertas      INTEGER NOT NULL DEFAULT 0,
            operaciones_cerradas_dia INTEGER NOT NULL DEFAULT 0,
            pnl_dia                  NUMERIC NOT NULL DEFAULT 0,
            retorno_acumulado_pct    NUMERIC,
            PRIMARY KEY (estrategia_id, fecha)
        )
        """,
        "CREATE INDEX idx_ft_metricas_fecha ON ft_metricas_diarias (fecha)",
    ],
    "ft_posiciones_diarias": [
        """
        CREATE TABLE ft_posiciones_diarias (
            id                 SERIAL PRIMARY KEY,
            estrategia_id      INTEGER NOT NULL,
            operacion_id       INTEGER NOT NULL,
            ticker             VARCHAR(20) NOT NULL,
            fecha              DATE NOT NULL,
            estado             VARCHAR(20) NOT NULL,
            precio_cierre      NUMERIC,
            precio_entrada_ref NUMERIC,
            retorno_pct        NUMERIC,
            dias_abierta       SMALLINT,
            tech_score         NUMERIC,
            candle_score_5d    NUMERIC,
            rango_5d_pct       NUMERIC,
            atr14              NUMERIC,
            lateral_ratio      NUMERIC,
            vol_price_confirm  SMALLINT,
            vol_price_diverge  SMALLINT,
            up_vol_5d          SMALLINT,
            motivo_salida      VARCHAR(40),
            CONSTRAINT ft_posiciones_diarias_operacion_id_fecha_key
                UNIQUE (operacion_id, fecha)
        )
        """,
        "CREATE INDEX idx_ft_pd_eid_fecha ON ft_posiciones_diarias "
        "(estrategia_id, fecha DESC)",
        "CREATE INDEX idx_ft_pd_estado    ON ft_posiciones_diarias (estado)",
        "CREATE INDEX idx_ft_pd_opid      ON ft_posiciones_diarias (operacion_id)",
    ],
}

# Orden de procesamiento (padre primero, por convencion; no hay FKs declaradas)
TABLAS = [
    "ft_estrategias",
    "ft_operaciones",
    "ft_candidatos_diarios",
    "ft_metricas_diarias",
    "ft_posiciones_diarias",
]

# Columnas JSONB por tabla (necesitan psycopg2.extras.Json al insertar)
JSONB_COLS = {
    "ft_estrategias":  {"parametros"},
    "ft_operaciones":  {"detalle_entrada"},
}

# Tablas con secuencia id que hay que resetear
TABLAS_CON_SECUENCIA = {
    "ft_estrategias":        "ft_estrategias_id_seq",
    "ft_operaciones":        "ft_operaciones_id_seq",
    "ft_candidatos_diarios": "ft_candidatos_diarios_id_seq",
    "ft_posiciones_diarias": "ft_posiciones_diarias_id_seq",
}


# ── Backup ───────────────────────────────────────────────────────────────────

def backup_local(local_eng) -> str:
    """Vuelca las tablas ft_* locales actuales a CSV. Retorna el directorio."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = os.path.join(os.path.dirname(__file__), f"backup_ft_local_{ts}")
    os.makedirs(bdir, exist_ok=True)
    log(f"Backup de tablas ft_* locales -> {bdir}")
    for t in TABLAS:
        try:
            df = pd.read_sql(f"SELECT * FROM {t}", local_eng)
            path = os.path.join(bdir, f"{t}.csv")
            df.to_csv(path, index=False)
            log(f"  {t}: {len(df)} filas -> {os.path.basename(path)}")
        except Exception as e:
            log(f"  {t}: [WARN] no se pudo respaldar ({e})")
    return bdir


# ── Migracion ────────────────────────────────────────────────────────────────

def _limpiar_valor(v, es_jsonb: bool):
    """Normaliza un valor de pandas para psycopg2."""
    if es_jsonb:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str):
            # ya viene serializado
            try:
                v = json.loads(v)
            except (ValueError, TypeError):
                return None
        return psycopg2.extras.Json(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    # pandas NaT
    if v is pd.NaT:
        return None
    return v


def migrar_tabla(cur, rail_eng, tabla: str, dry_run: bool) -> tuple[int, int]:
    """
    Migra una tabla. Retorna (filas_railway, filas_insertadas_local).
    Opera dentro de la transaccion del cursor dado (no hace commit).
    """
    df = pd.read_sql(f"SELECT * FROM {tabla}", rail_eng)
    n_rail = len(df)
    log(f"  {tabla}: {n_rail} filas en Railway")

    if dry_run:
        return n_rail, 0

    # DROP + CREATE con schema real
    cur.execute(f"DROP TABLE IF EXISTS {tabla} CASCADE")
    for stmt in DDL[tabla]:
        cur.execute(stmt.strip())

    if n_rail == 0:
        return 0, 0

    # Insertar preservando todas las columnas (incluido id)
    cols    = list(df.columns)
    jsonb   = JSONB_COLS.get(tabla, set())
    col_sql = ", ".join(cols)
    ph      = ", ".join([f"%s" for _ in cols])
    insert  = f"INSERT INTO {tabla} ({col_sql}) VALUES ({ph})"

    registros = []
    for _, row in df.iterrows():
        registros.append(tuple(
            _limpiar_valor(row[c], c in jsonb) for c in cols
        ))

    psycopg2.extras.execute_batch(cur, insert, registros, page_size=500)

    # Reset de secuencia si corresponde
    seq = TABLAS_CON_SECUENCIA.get(tabla)
    if seq:
        cur.execute(
            f"SELECT setval('{seq}', (SELECT COALESCE(MAX(id), 1) FROM {tabla}))"
        )

    cur.execute(f"SELECT COUNT(*) FROM {tabla}")
    n_local = cur.fetchone()[0]
    return n_rail, n_local


def main():
    parser = argparse.ArgumentParser(
        description="Migra tablas ft_* de Railway a Local (un solo uso)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula: lee Railway, no escribe en local")
    parser.add_argument("--confirm", action="store_true",
                        help="Ejecuta la migracion real")
    args = parser.parse_args()

    if not args.dry_run and not args.confirm:
        print("Falta --dry-run o --confirm")
        sys.exit(2)
    dry_run = args.dry_run and not args.confirm

    print()
    print(SEP)
    print(f"  MIGRACION FT  Railway -> Local  |  {date.today()}")
    print(f"  MODO: {'DRY-RUN (sin escritura)' if dry_run else 'REAL (--confirm)'}")
    print(SEP)
    print()

    rail_eng  = get_railway_engine()
    local_eng = get_local_engine()

    # 1. Backup (siempre, incluso en dry-run)
    bdir = backup_local(local_eng)
    print()

    # 2. Migracion en UNA transaccion
    conn = get_local_conn()
    conn.autocommit = False
    resultados = {}
    try:
        cur = conn.cursor()
        log("Migrando tablas (transaccion unica)...")
        for t in TABLAS:
            n_rail, n_local = migrar_tabla(cur, rail_eng, t, dry_run)
            resultados[t] = (n_rail, n_local)

        if dry_run:
            conn.rollback()
            log("DRY-RUN: rollback, local sin cambios.")
        else:
            conn.commit()
            log("COMMIT: migracion aplicada.")
    except Exception as e:
        conn.rollback()
        log(f"[ERROR] {e}")
        log("ROLLBACK: local quedo intacto.")
        conn.close()
        sys.exit(1)
    finally:
        if not conn.closed:
            conn.close()

    # 3. Verificacion
    print()
    print(SEP)
    print("  RESUMEN:")
    ok = True
    for t in TABLAS:
        n_rail, n_local = resultados[t]
        if dry_run:
            print(f"  [DRY ] {t:<26} Railway={n_rail}")
        else:
            marca = "OK  " if n_rail == n_local else "FAIL"
            if n_rail != n_local:
                ok = False
            print(f"  [{marca}] {t:<26} Railway={n_rail}  Local={n_local}")
    print(SEP)
    print(f"  Backup local previo en: {bdir}")
    print(SEP)
    print()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
