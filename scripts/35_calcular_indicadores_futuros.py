"""
35_calcular_indicadores_futuros.py
Calcula indicadores tecnicos sobre futuros_diarios y los persiste en
indicadores_tecnicos_futuros.

Reutiliza calcular_indicadores() de src/indicators/technical.py —
misma logica que sobre acciones, aplicada a los 12 futuros de referencia.

Uso:
    python scripts/35_calcular_indicadores_futuros.py           # recalcula ultimos 10 dias
    python scripts/35_calcular_indicadores_futuros.py --init    # crea tabla + calcula todo
    python scripts/35_calcular_indicadores_futuros.py --status  # resumen de fechas en DB
"""

import sys
import os
import argparse
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _setup_target_env(target: str):
    """
    Configura os.environ para que get_engine() apunte al target deseado.
    DEBE correr ANTES de importar src.data.database: config.py construye
    DB_CONFIG al momento del import segun DATABASE_URL en os.environ.
      target='local'   -> elimina DATABASE_URL -> get_engine usa DB_CONFIG local
      target='railway' -> carga DATABASE_URL desde .env.local (Oracle/produccion)
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    if target == "railway":
        envlocal = os.path.join(ROOT, ".env.local")
        if os.path.exists(envlocal):
            load_dotenv(envlocal, override=True)
    else:
        os.environ.pop("DATABASE_URL", None)


# Resolver --target ANTES de importar modulos que tocan la DB
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--target", choices=["local", "railway"], default="local")
_setup_target_env(_pre.parse_known_args()[0].target)

import pandas as pd
import psycopg2.extras
from sqlalchemy import text

from src.data.database import get_engine, get_connection
from src.indicators.technical import calcular_indicadores


# ── Constantes ────────────────────────────────────────────────────────────────

DIAS_UPDATE = 10    # ventana de upsert en modo diario (cubre fines de semana + feriados)

FUTUROS = {
    "YM=F":  "Dow Jones Mini (US30)",
    "ES=F":  "S&P 500 E-mini",
    "NQ=F":  "Nasdaq-100 E-mini",
    "RTY=F": "Russell 2000 E-mini",
    "GC=F":  "Gold futures",
    "SI=F":  "Silver futures",
    "UB=F":  "Ultra T-Bond 30Y",
    "TN=F":  "Ultra 10-Year Note",
    "ZL=F":  "Soybean Oil",
    "ZM=F":  "Soybean Meal",
    "XAE=F": "E-mini Energy Select Sector",
    "XAU=F": "E-mini Utilities Select Sector",
}


# ── DDL ───────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS indicadores_tecnicos_futuros (
    id          SERIAL        PRIMARY KEY,
    ticker      VARCHAR(10)   NOT NULL,
    fecha       DATE          NOT NULL,
    sma21       NUMERIC(14,4),
    sma50       NUMERIC(14,4),
    sma200      NUMERIC(14,4),
    dist_sma21  NUMERIC(10,4),
    dist_sma50  NUMERIC(10,4),
    dist_sma200 NUMERIC(10,4),
    rsi14       NUMERIC(8,4),
    macd        NUMERIC(14,6),
    macd_signal NUMERIC(14,6),
    macd_hist   NUMERIC(14,6),
    atr14       NUMERIC(14,4),
    bb_upper    NUMERIC(14,4),
    bb_middle   NUMERIC(14,4),
    bb_lower    NUMERIC(14,4),
    obv         NUMERIC(20,2),
    vol_relativo NUMERIC(10,4),
    adx         NUMERIC(8,4),
    momentum    NUMERIC(14,4),
    created_at  TIMESTAMP     DEFAULT NOW(),
    updated_at  TIMESTAMP     DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_itf_ticker ON indicadores_tecnicos_futuros (ticker);
CREATE INDEX IF NOT EXISTS idx_itf_fecha  ON indicadores_tecnicos_futuros (fecha);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def init_tabla():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    log("Tabla indicadores_tecnicos_futuros lista.")


def cargar_ohlcv(ticker: str) -> pd.DataFrame:
    """Lee toda la historia de un futuro desde futuros_diarios."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT fecha, open, high, low, close, volume FROM futuros_diarios "
                 "WHERE ticker = :ticker ORDER BY fecha ASC"),
            conn,
            params={"ticker": ticker},
        )
    return df


def upsert_indicadores_futuros(df: pd.DataFrame):
    """Upsert bulk en indicadores_tecnicos_futuros."""
    if df.empty:
        return 0

    cols = [
        "ticker", "fecha", "sma21", "sma50", "sma200",
        "dist_sma21", "dist_sma50", "dist_sma200",
        "rsi14", "macd", "macd_signal", "macd_hist",
        "atr14", "bb_upper", "bb_middle", "bb_lower",
        "obv", "vol_relativo", "adx", "momentum",
    ]
    records = df[cols].to_dict(orient="records")

    SQL = """
        INSERT INTO indicadores_tecnicos_futuros
            (ticker, fecha, sma21, sma50, sma200,
             dist_sma21, dist_sma50, dist_sma200,
             rsi14, macd, macd_signal, macd_hist,
             atr14, bb_upper, bb_middle, bb_lower,
             obv, vol_relativo, adx, momentum)
        VALUES
            (%(ticker)s, %(fecha)s, %(sma21)s, %(sma50)s, %(sma200)s,
             %(dist_sma21)s, %(dist_sma50)s, %(dist_sma200)s,
             %(rsi14)s, %(macd)s, %(macd_signal)s, %(macd_hist)s,
             %(atr14)s, %(bb_upper)s, %(bb_middle)s, %(bb_lower)s,
             %(obv)s, %(vol_relativo)s, %(adx)s, %(momentum)s)
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            sma21        = EXCLUDED.sma21,
            sma50        = EXCLUDED.sma50,
            sma200       = EXCLUDED.sma200,
            dist_sma21   = EXCLUDED.dist_sma21,
            dist_sma50   = EXCLUDED.dist_sma50,
            dist_sma200  = EXCLUDED.dist_sma200,
            rsi14        = EXCLUDED.rsi14,
            macd         = EXCLUDED.macd,
            macd_signal  = EXCLUDED.macd_signal,
            macd_hist    = EXCLUDED.macd_hist,
            atr14        = EXCLUDED.atr14,
            bb_upper     = EXCLUDED.bb_upper,
            bb_middle    = EXCLUDED.bb_middle,
            bb_lower     = EXCLUDED.bb_lower,
            obv          = EXCLUDED.obv,
            vol_relativo = EXCLUDED.vol_relativo,
            adx          = EXCLUDED.adx,
            momentum     = EXCLUDED.momentum,
            updated_at   = NOW()
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, SQL, records, page_size=500)
    return len(records)


def procesar_ticker(ticker: str, solo_recientes: bool = False) -> int:
    """
    Calcula indicadores para un ticker y hace upsert.
    Lee el historico completo (necesario para SMA200), pero si solo_recientes=True
    solo persiste los ultimos DIAS_UPDATE dias.
    """
    df_ohlcv = cargar_ohlcv(ticker)
    if df_ohlcv.empty:
        return 0

    df_ind = calcular_indicadores(df_ohlcv, ticker)
    if df_ind.empty:
        return 0

    if solo_recientes:
        corte = date.today() - timedelta(days=DIAS_UPDATE)
        df_ind = df_ind[pd.to_datetime(df_ind["fecha"]).dt.date >= corte]

    return upsert_indicadores_futuros(df_ind)


# ── Runners ───────────────────────────────────────────────────────────────────

def cmd_init():
    """Crea la tabla y calcula indicadores sobre todo el historial disponible."""
    init_tabla()
    log(f"Calculando indicadores para {len(FUTUROS)} futuros (historial completo)...")
    log("=" * 55)

    total = 0
    for ticker, nombre in FUTUROS.items():
        n = procesar_ticker(ticker, solo_recientes=False)
        log(f"  {ticker:<8s}  {nombre:<38s}  {n:4d} filas")
        total += n

    log("=" * 55)
    log(f"Total upsert: {total:,} filas")


def cmd_update():
    """
    Recalcula indicadores para los ultimos DIAS_UPDATE dias de cada futuro.
    Lee el historial completo para asegurar SMA200 correcto, pero solo
    persiste la ventana reciente (idempotente por UNIQUE constraint).
    """
    total = 0
    errores = 0
    for ticker in FUTUROS:
        try:
            n = procesar_ticker(ticker, solo_recientes=True)
            total += n
        except Exception as e:
            log(f"  ERROR {ticker}: {e}")
            errores += 1

    log(f"  Indicadores futuros: {total} filas ({len(FUTUROS) - errores}/{len(FUTUROS)} OK)")
    return total


def cmd_status():
    """Muestra el rango de fechas y ultimo RSI/dist_sma50 por ticker."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                ticker,
                COUNT(*)            AS n,
                MIN(fecha)          AS desde,
                MAX(fecha)          AS hasta,
                ROUND(AVG(rsi14)::numeric, 1)       AS rsi_avg,
                ROUND(AVG(dist_sma50)::numeric, 2)  AS dist_sma50_avg
            FROM indicadores_tecnicos_futuros
            GROUP BY ticker
            ORDER BY ticker
        """)).fetchall()

    if not rows:
        print("  Sin datos en indicadores_tecnicos_futuros.")
        return

    print()
    print(f"  {'TICKER':<8s}  {'FILAS':>5s}  {'DESDE':<12s}  {'HASTA':<12s}  "
          f"{'RSI_AVG':>8s}  {'DSMA50_AVG':>10s}  NOMBRE")
    print("  " + "-" * 75)
    for r in rows:
        nombre = FUTUROS.get(r[0], "")
        print(f"  {r[0]:<8s}  {r[1]:>5d}  {str(r[2]):<12s}  {str(r[3]):<12s}  "
              f"{float(r[4] or 0):>8.1f}  {float(r[5] or 0):>10.2f}  {nombre}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Indicadores tecnicos sobre futuros de referencia"
    )
    parser.add_argument("--init",   action="store_true",
                        help="Crea tabla y calcula indicadores sobre historial completo")
    parser.add_argument("--status", action="store_true",
                        help="Muestra resumen de datos en DB")
    parser.add_argument("--target", choices=["local", "railway"], default="local",
                        help="DB destino: local (default) o railway (Oracle/produccion)")
    args = parser.parse_args()

    if args.init:
        cmd_init()
    elif args.status:
        cmd_status()
    else:
        cmd_update()


if __name__ == "__main__":
    main()
