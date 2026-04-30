"""
36_features_regimen_macro.py
Construye features_regimen_macro: tabla pivotada por fecha con indicadores
de contexto macro derivados de los 12 futuros de referencia.

Esquema: UNA fila por fecha (no por ticker). Joinable a features de acciones
por fecha para enriquecer el feature store con contexto de mercado global.

Columnas por instrumento (prefijo = ticker sin '=F'):
    {t}_rsi14       RSI 14 dias
    {t}_dist_sma50  Distancia % al SMA50 (regimen principal)
    {t}_dist_sma200 Distancia % al SMA200 (tendencia larga)
    {t}_adx         Fuerza de tendencia
    {t}_ret5d       Retorno 5 dias (%) — calculado desde futuros_diarios
    {t}_sobre_sma50 1 si precio > SMA50, 0 si no

Prefijos: es, nq, ym, rty, gc, si, ub, tn, zl, zm, xae, xau
  (gc=Gold, si=Silver, ub=T-Bond 30Y, tn=10Y Note,
   xae=Energy sector, xau=Utilities sector)

Senales cruzadas (cross-market):
    mercado_alcista      ES > SMA50 (1/0)
    mercado_tend_fuerte  ES ADX > 25 (1/0)
    nq_liderando         NQ ret5d > ES ret5d (tech leads = risk-on) (1/0)
    russell_rezagado     RTY ret5d < ES ret5d - 1.0% (small caps lag) (1/0)
    oro_risk_off         GC > SMA50 AND GC RSI > 60 (1/0)
    bonos_fuertes        UB > SMA50 (tasas largas a la baja) (1/0)
    energia_fuerte       XAE > SMA50 (sector energia alcista) (1/0)
    utilities_fuerte     XAU > SMA50 (sector defensivo alcista) (1/0)
    risk_off_score       0-4: oro_risk_off + bonos_fuertes + (1-mercado_alcista) + russell_rezagado

Uso:
    python scripts/36_features_regimen_macro.py           # recalcula ultimos 15 dias
    python scripts/36_features_regimen_macro.py --init    # crea tabla + calcula todo
    python scripts/36_features_regimen_macro.py --status  # resumen DB
"""

import sys
import os
import argparse
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

import pandas as pd
import numpy as np
import psycopg2.extras
from sqlalchemy import text

from src.data.database import get_engine, get_connection


# ── Mapeo ticker -> prefijo de columna ────────────────────────────────────────

TICKER_PREFIX = {
    "ES=F":  "es",
    "NQ=F":  "nq",
    "YM=F":  "ym",
    "RTY=F": "rty",
    "GC=F":  "gc",
    "SI=F":  "si",
    "UB=F":  "ub",
    "TN=F":  "tn",
    "ZL=F":  "zl",
    "ZM=F":  "zm",
    "XAE=F": "xae",
    "XAU=F": "xau",
}

DIAS_UPDATE = 15   # ventana de upsert en modo diario


# ── DDL ───────────────────────────────────────────────────────────────────────

def _columnas_ddl() -> str:
    """Genera las columnas por instrumento para el DDL."""
    cols = []
    for pfx in TICKER_PREFIX.values():
        cols += [
            f"    {pfx}_rsi14       NUMERIC(8,4)",
            f"    {pfx}_dist_sma50  NUMERIC(10,4)",
            f"    {pfx}_dist_sma200 NUMERIC(10,4)",
            f"    {pfx}_adx         NUMERIC(8,4)",
            f"    {pfx}_ret5d       NUMERIC(10,4)",
            f"    {pfx}_sobre_sma50 SMALLINT",
        ]
    return ",\n".join(cols)


DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS features_regimen_macro (
    id                  SERIAL      PRIMARY KEY,
    fecha               DATE        NOT NULL UNIQUE,
{col_instrumentos},
    -- Senales cruzadas
    mercado_alcista     SMALLINT,
    mercado_tend_fuerte SMALLINT,
    nq_liderando        SMALLINT,
    russell_rezagado    SMALLINT,
    oro_risk_off        SMALLINT,
    bonos_fuertes       SMALLINT,
    energia_fuerte      SMALLINT,
    utilities_fuerte    SMALLINT,
    risk_off_score      SMALLINT,
    created_at          TIMESTAMP   DEFAULT NOW(),
    updated_at          TIMESTAMP   DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_frm_fecha ON features_regimen_macro (fecha);
"""

DDL = DDL_TEMPLATE.format(col_instrumentos=_columnas_ddl())


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def init_tabla():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    log("Tabla features_regimen_macro lista.")


def _safe_int(val) -> int:
    """Convierte a int (0/1) manejando NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return int(bool(val))


# ── Carga de datos fuente ─────────────────────────────────────────────────────

def cargar_indicadores() -> pd.DataFrame:
    """Lee indicadores_tecnicos_futuros para todos los tickers."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT ticker, fecha, rsi14, dist_sma50, dist_sma200, adx, sma50, sma200
                FROM indicadores_tecnicos_futuros
                ORDER BY fecha ASC, ticker ASC
            """),
            conn,
        )
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def cargar_precios() -> pd.DataFrame:
    """Lee futuros_diarios para calcular retornos."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT ticker, fecha, close
                FROM futuros_diarios
                ORDER BY fecha ASC, ticker ASC
            """),
            conn,
        )
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


# ── Calculo de features ───────────────────────────────────────────────────────

def calcular_retornos(df_precios: pd.DataFrame) -> pd.DataFrame:
    """Calcula retorno_5d por ticker desde futuros_diarios."""
    resultados = []
    for ticker, grp in df_precios.groupby("ticker"):
        grp = grp.sort_values("fecha").copy()
        grp["ret5d"] = (grp["close"].pct_change(5) * 100).round(4)
        resultados.append(grp[["ticker", "fecha", "ret5d"]])
    return pd.concat(resultados, ignore_index=True)


def construir_features(df_ind: pd.DataFrame, df_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Pivota indicadores y retornos de (ticker, fecha) a (fecha) con columnas por instrumento.
    Agrega senales cruzadas.
    """
    # Merge indicadores + retornos
    df = df_ind.merge(df_ret, on=["ticker", "fecha"], how="left")

    # Pivot: una fila por fecha, columnas por ticker
    fechas = sorted(df["fecha"].unique())
    rows = []

    for fecha in fechas:
        dia = df[df["fecha"] == fecha].set_index("ticker")
        row = {"fecha": fecha.date()}

        # Features por instrumento
        for ticker, pfx in TICKER_PREFIX.items():
            if ticker not in dia.index:
                row[f"{pfx}_rsi14"]       = None
                row[f"{pfx}_dist_sma50"]  = None
                row[f"{pfx}_dist_sma200"] = None
                row[f"{pfx}_adx"]         = None
                row[f"{pfx}_ret5d"]       = None
                row[f"{pfx}_sobre_sma50"] = None
                continue

            r = dia.loc[ticker]
            row[f"{pfx}_rsi14"]       = float(r["rsi14"])       if pd.notna(r["rsi14"])       else None
            row[f"{pfx}_dist_sma50"]  = float(r["dist_sma50"])  if pd.notna(r["dist_sma50"])  else None
            row[f"{pfx}_dist_sma200"] = float(r["dist_sma200"]) if pd.notna(r["dist_sma200"]) else None
            row[f"{pfx}_adx"]         = float(r["adx"])         if pd.notna(r["adx"])         else None
            row[f"{pfx}_ret5d"]       = float(r["ret5d"])       if pd.notna(r["ret5d"])       else None
            sobre = int(r["dist_sma50"] > 0) if pd.notna(r["dist_sma50"]) else None
            row[f"{pfx}_sobre_sma50"] = sobre

        # ── Senales cruzadas ──────────────────────────────────────────────────
        es_sobre  = row.get("es_sobre_sma50")
        es_adx    = row.get("es_adx")
        es_ret5d  = row.get("es_ret5d")
        nq_ret5d  = row.get("nq_ret5d")
        rty_ret5d = row.get("rty_ret5d")
        gc_sobre  = row.get("gc_sobre_sma50")
        gc_rsi    = row.get("gc_rsi14")
        ub_sobre  = row.get("ub_sobre_sma50")
        xae_sobre = row.get("xae_sobre_sma50")
        xau_sobre = row.get("xau_sobre_sma50")

        mercado_alcista     = _safe_int(es_sobre == 1)
        mercado_tend_fuerte = _safe_int(es_adx is not None and es_adx > 25)

        nq_liderando = _safe_int(
            nq_ret5d is not None and es_ret5d is not None and nq_ret5d > es_ret5d
        )
        russell_rezagado = _safe_int(
            rty_ret5d is not None and es_ret5d is not None and rty_ret5d < (es_ret5d - 1.0)
        )
        oro_risk_off  = _safe_int(gc_sobre == 1 and gc_rsi is not None and gc_rsi > 60)
        bonos_fuertes = _safe_int(ub_sobre == 1)
        energia_fuerte   = _safe_int(xae_sobre == 1)
        utilities_fuerte = _safe_int(xau_sobre == 1)

        risk_off_score = (
            oro_risk_off
            + bonos_fuertes
            + (1 - mercado_alcista)   # mercado bajista suma 1 al score defensivo
            + russell_rezagado
        )

        row["mercado_alcista"]     = mercado_alcista
        row["mercado_tend_fuerte"] = mercado_tend_fuerte
        row["nq_liderando"]        = nq_liderando
        row["russell_rezagado"]    = russell_rezagado
        row["oro_risk_off"]        = oro_risk_off
        row["bonos_fuertes"]       = bonos_fuertes
        row["energia_fuerte"]      = energia_fuerte
        row["utilities_fuerte"]    = utilities_fuerte
        row["risk_off_score"]      = risk_off_score

        rows.append(row)

    return pd.DataFrame(rows)


# ── Persistencia ──────────────────────────────────────────────────────────────

def _build_upsert_sql(cols: list[str]) -> str:
    col_list  = ", ".join(cols)
    val_list  = ", ".join(f"%({c})s" for c in cols)
    update    = ",\n            ".join(
        f"{c} = EXCLUDED.{c}" for c in cols if c != "fecha"
    )
    return f"""
        INSERT INTO features_regimen_macro ({col_list})
        VALUES ({val_list})
        ON CONFLICT (fecha) DO UPDATE SET
            {update},
            updated_at = NOW()
    """


def _clean_record(rec: dict) -> dict:
    """Convierte numpy scalars y nan a tipos Python nativos para psycopg2."""
    import math
    out = {}
    for k, v in rec.items():
        if v is None:
            out[k] = None
        elif isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif hasattr(v, "item"):           # numpy scalar → Python nativo
            out[k] = v.item()
        else:
            out[k] = v
    return out


def persistir(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols    = [c for c in df.columns]
    records = [_clean_record(r) for r in df.to_dict(orient="records")]
    sql     = _build_upsert_sql(cols)
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
    return len(records)


# ── Runners ───────────────────────────────────────────────────────────────────

def cmd_init():
    init_tabla()
    log("Cargando indicadores y precios...")
    df_ind = cargar_indicadores()
    df_ret = calcular_retornos(cargar_precios())

    log("Construyendo features pivotadas...")
    df_feat = construir_features(df_ind, df_ret)

    n = persistir(df_feat)
    log(f"Total upsert: {n:,} filas ({df_feat['fecha'].min()} -> {df_feat['fecha'].max()})")


def cmd_update():
    """Recalcula solo los ultimos DIAS_UPDATE dias (idempotente)."""
    df_ind = cargar_indicadores()
    df_ret = calcular_retornos(cargar_precios())

    corte = date.today() - timedelta(days=DIAS_UPDATE)
    df_ind = df_ind[df_ind["fecha"].dt.date >= corte]

    if df_ind.empty:
        log("  Sin datos recientes para actualizar.")
        return 0

    # Retornos: necesitamos el historial completo para calcular ret5d correctamente
    # Filtramos solo despues de calcular
    df_ret_reciente = df_ret[df_ret["fecha"].dt.date >= corte]

    df_feat = construir_features(df_ind, df_ret_reciente)
    n = persistir(df_feat)
    log(f"  Regimen macro: {n} filas actualizadas")
    return n


def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                MIN(fecha)               AS desde,
                MAX(fecha)               AS hasta,
                COUNT(*)                 AS n,
                ROUND(AVG(es_rsi14)::numeric, 1)         AS es_rsi_avg,
                ROUND(AVG(gc_rsi14)::numeric, 1)         AS gc_rsi_avg,
                ROUND(AVG(ub_dist_sma50)::numeric, 2)    AS ub_dsma50_avg,
                ROUND(AVG(risk_off_score)::numeric, 2)   AS risk_off_avg,
                SUM(mercado_alcista)                     AS dias_alcistas,
                SUM(oro_risk_off)                        AS dias_oro_riskoff
            FROM features_regimen_macro
        """)).fetchone()

    if not rows or rows[0] is None:
        print("  Sin datos en features_regimen_macro.")
        return

    print()
    print(f"  Periodo    : {rows[0]}  ->  {rows[1]}  ({rows[2]} dias)")
    print(f"  ES RSI avg : {rows[3]}")
    print(f"  GC RSI avg : {rows[4]}  (oro)")
    print(f"  UB dist SMA50 avg: {rows[5]}%  (bonos)")
    print(f"  Risk-off score avg: {rows[6]}  (0-4)")
    print(f"  Dias mercado alcista : {rows[7]}/{rows[2]}")
    print(f"  Dias oro risk-off    : {rows[8]}/{rows[2]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Features de regimen macro desde futuros de referencia"
    )
    parser.add_argument("--init",   action="store_true",
                        help="Crea tabla y calcula historial completo")
    parser.add_argument("--status", action="store_true",
                        help="Resumen de datos en DB")
    args = parser.parse_args()

    if args.init:
        cmd_init()
    elif args.status:
        cmd_status()
    else:
        cmd_update()


if __name__ == "__main__":
    main()
