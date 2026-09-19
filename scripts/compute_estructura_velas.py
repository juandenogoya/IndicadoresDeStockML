"""
compute_estructura_velas.py
Tarea 23, Fase 2: historia HONESTA de estructura de mercado y patrones de vela, en
tablas paralelas a las viejas (LOCAL).

    features_estructura  <- src/indicators/estructura.py (swings confirmados;
                            las 24 columnas de features_market_structure para
                            N=5 y N=10, mas 12 de N=3 desde el 17/9/2026 para
                            las estrategias FT_SMC_v3_N3/N5)
    features_velas       <- src/indicators/velas.py (patrones clasicos con contexto)

Por que tablas nuevas y no reemplazar las viejas (docs/estructura_velas.md sec. 9):
    features_market_structure mira N ruedas al futuro, pero la leen los modelos
    ML v1/v2 (en vivo recalculan con el modulo viejo), el score del scanner y las
    estrategias FT durante la Etapa 4. Cambiarla cambiaria sus entradas en medio
    del experimento. Estas tablas no tienen consumidores todavia: son la base del
    modelo v3 (Fase 3) y de la migracion de consumidores (Fase 4).

Por que alcanza con escribir las ruedas nuevas:
    Los dos modulos son INVARIANTES (tests/test_estructura.py, tests/test_velas.py):
    la fila de una fecha no cambia cuando llegan barras nuevas. El calculo se hace
    sobre la historia completa de cada ticker (barato: segundos) y se persiste desde
    la primera rueda que falta, con RUEDAS_SOLAPE de margen por si el Paso 1 quedo
    PARCIAL. Un ticker sin filas se escribe entero (altas de universo). Solo una
    correccion de precios_diarios (splits.py corregir) cambia la historia: ahi va
    --completo.

Donde corre: Paso 2 de la rutina (cron_diario --step features, paso 2c), despues de
features_sector. NO es insumo de ninguna decision: si falla, el Paso 2 avisa y sigue.

Uso:
    python scripts/compute_estructura_velas.py --crear --completo   # carga inicial
    python scripts/compute_estructura_velas.py                      # incremental
    python scripts/compute_estructura_velas.py --tickers KLAC,CRWD --completo
    python scripts/compute_estructura_velas.py --dry-run
    python scripts/compute_estructura_velas.py --status
"""

import argparse
import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psycopg2.extras  # noqa: E402

from src.indicators import estructura, velas  # noqa: E402

TABLA_ESTRUCTURA = "features_estructura"
TABLA_VELAS = "features_velas"
RUEDAS_SOLAPE = 10

# Tipos generados desde las listas de columnas de los modulos: el esquema no se
# puede desincronizar del calculo.
# VENTANAS_TABLA (3, 5, 10) y no VENTANAS (5, 10): N=3 es para las estrategias
# FT_SMC_v3_N3/N5 (ver docs/estructura_velas.md sec. 9.3).
_VENTANAS_EST = estructura.VENTANAS_TABLA
_COLS_EST = estructura.columnas(_VENTANAS_EST)
_ENTERAS_EST = (set(estructura.columnas_enteras(_VENTANAS_EST))
                | {c for c in _COLS_EST if c.startswith("dias_")})
_COLS_VEL = velas.COLUMNAS


def _ddl(tabla: str, columnas, enteras) -> str:
    defs = ",\n    ".join(
        f"{c} {'SMALLINT' if c in enteras else 'DOUBLE PRECISION'}" for c in columnas)
    # PRIMARY KEY (ticker, fecha): indice unico FULL, sirve para ON CONFLICT.
    return f"""
CREATE TABLE IF NOT EXISTS {tabla} (
    ticker       VARCHAR(20) NOT NULL,
    fecha        DATE        NOT NULL,
    {defs},
    calculado_en TIMESTAMP   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, fecha)
)"""


DDL = [
    _ddl(TABLA_ESTRUCTURA, _COLS_EST, _ENTERAS_EST),
    f"CREATE INDEX IF NOT EXISTS idx_{TABLA_ESTRUCTURA}_fecha ON {TABLA_ESTRUCTURA} (fecha)",
    _ddl(TABLA_VELAS, _COLS_VEL, set(_COLS_VEL)),
    f"CREATE INDEX IF NOT EXISTS idx_{TABLA_VELAS}_fecha ON {TABLA_VELAS} (fecha)",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _sql_columnas_faltantes(tabla: str, columnas, enteras) -> list:
    """ADD COLUMN IF NOT EXISTS por columna: agregar una ventana nueva (N=3, 17/9/2026)
    no pisa la tabla ni pierde la historia ya calculada. Idempotente."""
    tipo = lambda c: "SMALLINT" if c in enteras else "DOUBLE PRECISION"  # noqa: E731
    return [f"ALTER TABLE {tabla} ADD COLUMN IF NOT EXISTS {c} {tipo(c)}"
            for c in columnas]


def crear_tablas() -> None:
    from src.data.database import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            for sql in DDL:
                cur.execute(sql)
            for sql in _sql_columnas_faltantes(TABLA_ESTRUCTURA, _COLS_EST, _ENTERAS_EST):
                cur.execute(sql)
            for sql in _sql_columnas_faltantes(TABLA_VELAS, _COLS_VEL, set(_COLS_VEL)):
                cur.execute(sql)
        conn.commit()


def _tablas_existen() -> bool:
    from src.data.database import query_df
    df = query_df("""
        SELECT COUNT(*) AS n FROM information_schema.tables
        WHERE table_name IN (:a, :b)
    """, params={"a": TABLA_ESTRUCTURA, "b": TABLA_VELAS})
    return int(df["n"].iloc[0]) == 2


def _registros(df: pd.DataFrame, columnas, enteras) -> list:
    """DataFrame -> tuplas con tipos nativos (NaN -> None)."""
    out = []
    vals = df[["ticker", "fecha"] + list(columnas)].to_numpy(dtype=object)
    idx_enteras = {i + 2 for i, c in enumerate(columnas) if c in enteras}
    for fila in vals:
        rec = [str(fila[0]), pd.Timestamp(fila[1]).date()]
        for i in range(2, len(fila)):
            v = fila[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec.append(None)
            elif i in idx_enteras:
                rec.append(int(v))
            else:
                rec.append(float(v))
        out.append(tuple(rec))
    return out


def _upsert(conn, tabla: str, columnas, registros: list) -> None:
    cols = ["ticker", "fecha"] + list(columnas)
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in columnas)
    sql = (f"INSERT INTO {tabla} ({', '.join(cols)}) VALUES %s "
           f"ON CONFLICT (ticker, fecha) DO UPDATE SET {sets}, calculado_en = NOW()")
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, registros, page_size=2000)


def computar(completo: bool = False, tickers=None, dry_run: bool = False,
             ruedas_solape: int = RUEDAS_SOLAPE, verbose: bool = True) -> dict:
    """
    Calcula ambas tablas y persiste las filas que faltan (o todas con completo=True).

    Usa src.data.database (query_df / get_connection): escribe en la misma DB que
    el resto del paso que lo llama. Standalone, main() fuerza LOCAL.

    Returns:
        {"tickers", "filas_estructura", "filas_velas", "desde_min", "ultima"}
    """
    from src.data.database import get_connection, query_df

    if not dry_run and not _tablas_existen():
        raise RuntimeError(f"faltan {TABLA_ESTRUCTURA}/{TABLA_VELAS}: correr "
                           f"scripts/compute_estructura_velas.py --crear --completo")

    filtro, params = "", {}
    if tickers:
        filtro, params = "AND ticker = ANY(:tk)", {"tk": list(tickers)}
    precios = query_df(f"""
        SELECT ticker, fecha, open, high, low, close FROM precios_diarios
        WHERE close > 0 AND high > 0 AND low > 0 AND open > 0 {filtro}
        ORDER BY ticker, fecha
    """, params=params)
    if precios.empty:
        raise ValueError("precios_diarios sin filas para calcular")
    precios["fecha"] = pd.to_datetime(precios["fecha"])

    # Desde donde escribir, por ticker.
    ruedas = sorted(precios["fecha"].unique())
    corte_solape = pd.Timestamp(ruedas[max(0, len(ruedas) - ruedas_solape)])
    ultima_guardada = {}
    if not completo and _tablas_existen():
        u = query_df(f"""
            SELECT e.ticker, LEAST(e.ultima, COALESCE(v.ultima, e.ultima)) AS ultima
            FROM (SELECT ticker, MAX(fecha) AS ultima FROM {TABLA_ESTRUCTURA} GROUP BY ticker) e
            LEFT JOIN (SELECT ticker, MAX(fecha) AS ultima FROM {TABLA_VELAS} GROUP BY ticker) v
              ON v.ticker = e.ticker
        """)
        ultima_guardada = {r.ticker: pd.Timestamp(r.ultima) for r in u.itertuples()}

    partes_est, partes_vel, desdes = [], [], []
    for tk, g in precios.groupby("ticker", sort=True):
        est = estructura.calcular_estructura(g, ventanas=_VENTANAS_EST)
        vel = velas.calcular_velas(g)
        if completo or tk not in ultima_guardada:
            desde = g["fecha"].min()
        else:
            desde = min(ultima_guardada[tk] + pd.Timedelta(days=1), corte_solape)
        desdes.append(desde)
        partes_est.append(est[est["fecha"] >= desde])
        partes_vel.append(vel[vel["fecha"] >= desde])

    df_est = pd.concat(partes_est, ignore_index=True)
    df_vel = pd.concat(partes_vel, ignore_index=True)
    stats = {
        "tickers": precios["ticker"].nunique(),
        "filas_estructura": len(df_est),
        "filas_velas": len(df_vel),
        "desde_min": min(desdes).date(),
        "ultima": precios["fecha"].max().date(),
    }
    if verbose:
        log(f"  {stats['tickers']} tickers | a escribir: estructura {stats['filas_estructura']:,}"
            f" / velas {stats['filas_velas']:,} filas | desde {stats['desde_min']} | "
            f"ultima rueda {stats['ultima']}")
    if dry_run:
        return stats

    reg_est = _registros(df_est, _COLS_EST, _ENTERAS_EST)
    reg_vel = _registros(df_vel, _COLS_VEL, set(_COLS_VEL))
    with get_connection() as conn:
        _upsert(conn, TABLA_ESTRUCTURA, _COLS_EST, reg_est)
        _upsert(conn, TABLA_VELAS, _COLS_VEL, reg_vel)
        conn.commit()
    return stats


def mostrar_status() -> None:
    from src.data.database import query_df
    for tabla in (TABLA_ESTRUCTURA, TABLA_VELAS):
        df = query_df(f"""
            SELECT COUNT(*) AS filas, COUNT(DISTINCT ticker) AS tickers,
                   MIN(fecha) AS desde, MAX(fecha) AS hasta, MAX(calculado_en) AS calculado
            FROM {tabla}
        """)
        r = df.iloc[0]
        log(f"{tabla}: {int(r.filas):,} filas | {int(r.tickers)} tickers | "
            f"{r.desde} -> {r.hasta} | ultimo calculo {r.calculado}")


def main() -> int:
    # LOCAL-only: con DATABASE_URL seteada get_connection caeria a Railway.
    os.environ.pop("DATABASE_URL", None)

    ap = argparse.ArgumentParser(description="Estructura y velas point-in-time (Tarea 23).")
    ap.add_argument("--crear", action="store_true", help="crea las tablas si no existen")
    ap.add_argument("--completo", action="store_true", help="reescribe toda la historia")
    ap.add_argument("--tickers", help="lista separada por comas")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.crear:
        crear_tablas()
        log(f"tablas {TABLA_ESTRUCTURA} y {TABLA_VELAS} listas")
    if args.status:
        mostrar_status()
        return 0

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    inicio = datetime.now()
    computar(completo=args.completo, tickers=tickers, dry_run=args.dry_run)
    log(f"{'dry run' if args.dry_run else 'OK'} en {(datetime.now() - inicio).seconds}s")
    if not args.dry_run:
        mostrar_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
