"""
sync_local_to_railway.py
Sincroniza Railway con los datos frescos del PostgreSQL local.

Direccion inversa de sync_railway_to_local.py: se usa cuando paso 1+2+3
del pipeline corren localmente (rate limit en otras IPs) y necesitamos
subir los resultados a la fuente de verdad (Railway).

Pasos disponibles:
  --paso precios       : precios_diarios + futuros_diarios
  --paso indicadores   : indicadores_tecnicos + indicadores_tecnicos_futuros
  --paso features      : features_precio_accion + features_market_structure
                         + features_regimen_macro + ticker_zscore_diario
  --paso scanner       : alertas_scanner

Estrategia comun por tabla:
  1. Lee MAX(col_fecha) en Railway
  2. Pull desde local WHERE col_fecha > max_rail (o todo si Railway vacia)
  3. Drop columna 'id' para que Railway auto-asigne via sequence
  4. Reset sequence de Railway al MAX(id) actual (evita conflictos silenciosos)
  5. INSERT con ON CONFLICT DO NOTHING en chunks de 500 filas
  6. Muestra MAX(col_fecha) Railway antes/despues como verificacion

Uso:
    python scripts/migrations/sync_local_to_railway.py --paso precios
    python scripts/migrations/sync_local_to_railway.py --paso precios --dry-run
    python scripts/migrations/sync_local_to_railway.py --paso indicadores
    python scripts/migrations/sync_local_to_railway.py --paso features
    python scripts/migrations/sync_local_to_railway.py --paso scanner
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

SEP = "=" * 65
CHUNKSIZE = 500


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Lectura de .env (sin tocar os.environ) ────────────────────────────────────

def _parse_env_file(path: str) -> dict:
    """Lee un archivo .env y retorna dict clave->valor."""
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


# ── Engines ────────────────────────────────────────────────────────────────────

def get_railway_engine():
    """Engine SQLAlchemy apuntando a Railway (DATABASE_URL en .env.local)."""
    env = _parse_env_file(os.path.join(ROOT, ".env.local"))
    db_url = env.get("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError("DATABASE_URL no encontrado en .env.local")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, connect_args={"sslmode": "require"})


def get_local_engine():
    """Engine SQLAlchemy apuntando a DB local (DB_HOST/etc en .env)."""
    env = _parse_env_file(os.path.join(ROOT, ".env"))
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    name = env.get("DB_NAME", "activos_ml")
    user = env.get("DB_USER", "postgres")
    pwd  = env.get("DB_PASSWORD", "")
    url  = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url)


def _railway_psycopg2_conn():
    """Conexion psycopg2 directa a Railway (para INSERTs con ON CONFLICT)."""
    env = _parse_env_file(os.path.join(ROOT, ".env.local"))
    db_url = env.get("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError("DATABASE_URL no encontrado en .env.local")
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(db_url, sslmode="require")


# ── Helpers ────────────────────────────────────────────────────────────────────

def query_scalar(engine, sql: str):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def _clean_nan(records: list) -> list:
    """Convierte NaN/inf -> None en valores float dentro de records."""
    return [
        {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
         for k, v in rec.items()}
        for rec in records
    ]


def _serializar_jsonb(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas con dict/list a JSON string (compatibilidad psycopg2)."""
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


def _reset_sequence(rail_eng, tabla: str):
    """
    Resetea la secuencia 'id' al MAX(id) de Railway.
    Previene conflictos PK silenciosos (ON CONFLICT DO NOTHING los oculta).
    Best-effort: si la tabla no tiene columna id o secuencia, solo loguea WARN.
    """
    try:
        sql = f"""
            SELECT setval(
                pg_get_serial_sequence('{tabla}', 'id'),
                COALESCE((SELECT MAX(id) FROM {tabla}), 1),
                true
            )
        """
        with rail_eng.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
        log(f"  Sequence 'id' reseteada para {tabla}")
    except Exception as e:
        log(f"  WARN: no se pudo resetear sequence ({str(e)[:80]})")


def _insert_chunks(records: list, tabla: str, cols: list, chunk_size: int = CHUNKSIZE) -> int:
    """
    Inserta records en Railway via psycopg2 + execute_batch + ON CONFLICT DO NOTHING.
    Retorna cantidad total enviada (no necesariamente persistida si hay duplicados).
    """
    cols_str = ", ".join(cols)
    placeholders = ", ".join([f"%({c})s" for c in cols])
    sql = (
        f"INSERT INTO {tabla} ({cols_str}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT DO NOTHING"
    )

    conn = _railway_psycopg2_conn()
    sent = 0
    try:
        cur = conn.cursor()
        for batch_start in range(0, len(records), chunk_size):
            batch = records[batch_start:batch_start + chunk_size]
            psycopg2.extras.execute_batch(cur, sql, batch, page_size=chunk_size)
            sent += len(batch)
            log(f"    chunk {sent}/{len(records)}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return sent


# ── Sync incremental generico (local -> Railway) ──────────────────────────────

def sync_incremental(
    local_eng, rail_eng, tabla: str,
    col_fecha: str, dry_run: bool
) -> dict:
    """
    Sincroniza una tabla local -> Railway de forma incremental.

    Estrategia:
      1. Lee MAX(col_fecha) en Railway y local
      2. Si local <= Railway, no hace nada (Railway ya esta al dia)
      3. Pull local WHERE col_fecha > max_rail
      4. Reset sequence id en Railway (best-effort)
      5. INSERT en chunks con ON CONFLICT DO NOTHING
      6. Verifica MAX(col_fecha) Railway despues
    """
    log(f"{tabla}...")

    try:
        max_rail = query_scalar(rail_eng, f"SELECT MAX({col_fecha}) FROM {tabla}")
    except Exception as e:
        log(f"  ERROR leyendo Railway.{tabla}: {str(e)[:100]}")
        raise

    try:
        max_local = query_scalar(local_eng, f"SELECT MAX({col_fecha}) FROM {tabla}")
    except Exception as e:
        log(f"  ERROR leyendo local.{tabla}: {str(e)[:100]}")
        raise

    log(f"  MAX({col_fecha}) Railway: {max_rail}")
    log(f"  MAX({col_fecha}) Local:   {max_local}")

    if max_rail and max_local and max_local <= max_rail:
        log(f"  OK - Railway ya esta al dia (sin filas mas nuevas en local)")
        return {"tabla": tabla, "filas": 0, "estado": "SKIP"}

    where = f"{col_fecha} > '{max_rail}'" if max_rail else "1=1"
    df = pd.read_sql(
        f"SELECT * FROM {tabla} WHERE {where} ORDER BY {col_fecha}",
        local_eng
    )
    log(f"  Filas locales nuevas: {len(df)}")

    if df.empty:
        log(f"  OK - sin filas nuevas")
        return {"tabla": tabla, "filas": 0, "estado": "OK"}

    if dry_run:
        log(f"  DRY-RUN: enviaria {len(df)} filas a Railway.{tabla}")
        return {"tabla": tabla, "filas": len(df), "estado": "DRY-RUN"}

    # Drop id para que Railway auto-asigne via sequence
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Serializa JSONB si hay columnas dict/list
    df = _serializar_jsonb(df)

    # Reset sequence ANTES del insert (previene conflictos PK silenciosos)
    _reset_sequence(rail_eng, tabla)

    # Limpia NaN/inf -> None
    raw = df.to_dict("records")
    records = _clean_nan(raw)

    sent = _insert_chunks(records, tabla, list(df.columns))
    log(f"  OK - {sent} filas enviadas a Railway.{tabla}")

    # Verifica MAX despues
    try:
        max_rail_post = query_scalar(rail_eng, f"SELECT MAX({col_fecha}) FROM {tabla}")
        log(f"  MAX({col_fecha}) Railway despues: {max_rail_post}")
    except Exception:
        pass

    return {"tabla": tabla, "filas": sent, "estado": "OK"}


# ── Pasos ──────────────────────────────────────────────────────────────────────

PASO_CONFIG = {
    "precios": [
        ("precios_diarios",   "fecha"),
        ("futuros_diarios",   "fecha"),
    ],
    "indicadores": [
        ("indicadores_tecnicos",          "fecha"),
        ("indicadores_tecnicos_futuros",  "fecha"),
    ],
    "features": [
        ("features_precio_accion",     "fecha"),
        ("features_market_structure",  "fecha"),
        ("features_regimen_macro",     "fecha"),
        ("ticker_zscore_diario",       "fecha"),
    ],
    "scanner": [
        ("alertas_scanner",  "scan_fecha"),
    ],
}


def correr_paso(paso: str, local_eng, rail_eng, dry_run: bool) -> list:
    if paso not in PASO_CONFIG:
        raise ValueError(f"Paso desconocido: {paso}. Validos: {list(PASO_CONFIG)}")

    resultados = []
    for tabla, col_fecha in PASO_CONFIG[paso]:
        try:
            r = sync_incremental(local_eng, rail_eng, tabla, col_fecha, dry_run)
            resultados.append(r)
        except Exception as e:
            log(f"  ERROR en {tabla}: {str(e)[:200]}")
            resultados.append({"tabla": tabla, "filas": 0, "estado": f"ERROR: {str(e)[:80]}"})
            # No reraise para mostrar resumen completo
            break

    return resultados


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sync Local -> Railway DB (paso a paso)"
    )
    parser.add_argument("--paso", choices=list(PASO_CONFIG.keys()), required=True,
                        help="Paso a sincronizar")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sin escribir en Railway")
    args = parser.parse_args()

    print()
    print(SEP)
    print(f"  SYNC  Local -> Railway  |  paso: {args.paso}  |  {date.today()}")
    if args.dry_run:
        print(f"  MODO: DRY-RUN (sin escritura en Railway)")
    print(SEP)
    print()

    local_eng = get_local_engine()
    rail_eng  = get_railway_engine()

    resultados = correr_paso(args.paso, local_eng, rail_eng, args.dry_run)

    # ── Resumen ──
    print()
    print(SEP)
    print(f"  RESUMEN paso '{args.paso}':")
    for r in resultados:
        estado = r["estado"]
        if estado.startswith("ERROR"):
            marca = "FAIL"
        elif estado in ("OK", "SKIP", "DRY-RUN"):
            marca = "OK  "
        else:
            marca = "??  "
        print(f"  [{marca}] {r['tabla']:<35} {r['filas']:>7} filas  ({estado})")
    print(SEP)
    print()

    errores = [r for r in resultados if r["estado"].startswith("ERROR")]
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
