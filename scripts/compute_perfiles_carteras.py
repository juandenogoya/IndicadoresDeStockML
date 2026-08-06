"""
compute_perfiles_carteras.py
Computa y persiste el perfil de riesgo de cada ticker (Fase 3 del perfilado de
carteras). Lee precios_diarios + futuros ES (benchmark) + activos, corre el motor
PURO Fase 1 (perfil_metricas) + Fase 2 (perfil_riesgo) sobre todo el universo y
hace UPSERT en perfiles_ticker.

Cadencia MENSUAL (el perfil es una propiedad estable del instrumento, no una
senal diaria). Standalone, NO va en el recovery diario.

LOCAL-only (Plan C): fuente de verdad de OHLCV es local; la tabla vive en local.

Modelo (ver docs/perfiles_carteras.md):
    - perfil = comportamiento cuantitativo puro (percentil composite -> cuartil).
    - sector = contexto (caja_base) + flag de excepcion.
    - benchmark = futuros ES (ES=F). CONSTRAINT: arranca 2025-05-09 -> beta ~1a.

Uso:
    python scripts/compute_perfiles_carteras.py --dry-run   # calcula y muestra, no escribe
    python scripts/compute_perfiles_carteras.py             # calcula y UPSERT (fecha=hoy)
    python scripts/compute_perfiles_carteras.py --fecha 2026-08-06
"""

import sys
import os
import argparse
from collections import Counter
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd
from sqlalchemy import text

from scripts.oneshot.create_fundamentales_tables import get_local_engine
from src.utils import perfil_metricas, perfil_riesgo

BENCH_TICKER = "ES=F"
SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cargar_datos(eng):
    """Universo (ticker/sector/industry), precios OHLC por ticker y benchmark."""
    with eng.connect() as c:
        universo = pd.read_sql(text(
            "SELECT ticker, sector, industry FROM activos WHERE activo = TRUE "
            "ORDER BY ticker"), c)
        precios = pd.read_sql(text(
            "SELECT ticker, fecha, open, high, low, close FROM precios_diarios"), c)
        bench = pd.read_sql(text(
            "SELECT fecha, close FROM futuros_diarios WHERE ticker = :b "
            "ORDER BY fecha"), c, params={"b": BENCH_TICKER})
    return universo, precios, bench


def construir_clasificaciones(universo, precios, bench):
    """Corre Fase 1 (metricas) + Fase 2 (clasificacion) sobre el universo."""
    px_by_tk = {tk: g.sort_values("fecha").reset_index(drop=True)
                for tk, g in precios.groupby("ticker")}
    fecha_datos = {tk: g["fecha"].max() for tk, g in px_by_tk.items()}

    rows = []
    for _, r in universo.iterrows():
        tk = r["ticker"]
        df = px_by_tk.get(tk)
        met = perfil_metricas.metricas_ticker(df, bench) if df is not None else {}
        rows.append({"ticker": tk, "sector": r["sector"],
                     "industry": r["industry"], "metricas": met})

    clasifs = perfil_riesgo.perfilar_universo(rows)
    # adjunta lo que el clasificador puro NO devuelve: industry (no la usa para
    # clasificar), metricas crudas y fecha_datos.
    met_by_tk = {r["ticker"]: r["metricas"] for r in rows}
    ind_by_tk = {r["ticker"]: r["industry"] for r in rows}
    for c in clasifs:
        c["_met"] = met_by_tk.get(c["ticker"], {})
        c["industry"] = ind_by_tk.get(c["ticker"])
        c["_fecha_datos"] = fecha_datos.get(c["ticker"])
    return clasifs


def _num(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _fila(c, fecha):
    met = c["_met"]
    pe = c.get("pct_ejes", {})
    return {
        "ticker": c["ticker"], "fecha": fecha,
        "sector": c["sector"], "industry": c.get("industry"),
        "perfil": c["perfil"], "perfil_ordinal": c["perfil_ordinal"],
        "caja_base": c["caja_base"], "caja_base_fuente": c["caja_base_fuente"],
        "caja_cuant": c["caja_cuant"], "score_riesgo": _num(c["score_riesgo"]),
        "movio": c["movio"], "excepcion": c["excepcion"], "sin_cuant": c["sin_cuant"],
        "rank_en_caja": c.get("rank_en_caja"), "n_en_caja": c.get("n_en_caja"),
        "pct_en_caja": c.get("pct_en_caja"),
        "atr_pct_d": _num(met.get("atr_pct_d")), "atr_pct_w": _num(met.get("atr_pct_w")),
        "atr_pct_m": _num(met.get("atr_pct_m")), "beta": _num(met.get("beta")),
        "max_dd_1a": _num(met.get("max_dd_1a")), "max_dd_hist": _num(met.get("max_dd_hist")),
        "pct_atr_pct_w": pe.get("atr_pct_w"), "pct_atr_pct_m": pe.get("atr_pct_m"),
        "pct_beta": pe.get("beta"), "pct_max_dd_1a": pe.get("max_dd_1a"),
        "fecha_datos": c.get("_fecha_datos"),
    }


UPSERT = text("""
INSERT INTO perfiles_ticker (
    ticker, fecha, sector, industry, perfil, perfil_ordinal, caja_base,
    caja_base_fuente, caja_cuant, score_riesgo, movio, excepcion, sin_cuant,
    rank_en_caja, n_en_caja, pct_en_caja, atr_pct_d, atr_pct_w, atr_pct_m,
    beta, max_dd_1a, max_dd_hist, pct_atr_pct_w, pct_atr_pct_m, pct_beta,
    pct_max_dd_1a, fecha_datos, computed_at
) VALUES (
    :ticker, :fecha, :sector, :industry, :perfil, :perfil_ordinal, :caja_base,
    :caja_base_fuente, :caja_cuant, :score_riesgo, :movio, :excepcion, :sin_cuant,
    :rank_en_caja, :n_en_caja, :pct_en_caja, :atr_pct_d, :atr_pct_w, :atr_pct_m,
    :beta, :max_dd_1a, :max_dd_hist, :pct_atr_pct_w, :pct_atr_pct_m, :pct_beta,
    :pct_max_dd_1a, :fecha_datos, now()
)
ON CONFLICT (ticker, fecha) DO UPDATE SET
    sector = EXCLUDED.sector, industry = EXCLUDED.industry,
    perfil = EXCLUDED.perfil, perfil_ordinal = EXCLUDED.perfil_ordinal,
    caja_base = EXCLUDED.caja_base, caja_base_fuente = EXCLUDED.caja_base_fuente,
    caja_cuant = EXCLUDED.caja_cuant, score_riesgo = EXCLUDED.score_riesgo,
    movio = EXCLUDED.movio, excepcion = EXCLUDED.excepcion,
    sin_cuant = EXCLUDED.sin_cuant, rank_en_caja = EXCLUDED.rank_en_caja,
    n_en_caja = EXCLUDED.n_en_caja, pct_en_caja = EXCLUDED.pct_en_caja,
    atr_pct_d = EXCLUDED.atr_pct_d, atr_pct_w = EXCLUDED.atr_pct_w,
    atr_pct_m = EXCLUDED.atr_pct_m, beta = EXCLUDED.beta,
    max_dd_1a = EXCLUDED.max_dd_1a, max_dd_hist = EXCLUDED.max_dd_hist,
    pct_atr_pct_w = EXCLUDED.pct_atr_pct_w, pct_atr_pct_m = EXCLUDED.pct_atr_pct_m,
    pct_beta = EXCLUDED.pct_beta, pct_max_dd_1a = EXCLUDED.pct_max_dd_1a,
    fecha_datos = EXCLUDED.fecha_datos, computed_at = now()
""")


def resumen(clasifs):
    dist = Counter(c["perfil"] for c in clasifs)
    log("Distribucion:")
    for p in ["Conservadora", "Moderada", "Arriesgada", "Especulativa"]:
        log(f"   {p:13} {dist.get(p, 0):3}")
    n_exc = sum(1 for c in clasifs if c["excepcion"])
    n_sc = sum(1 for c in clasifs if c["sin_cuant"])
    log(f"   excepciones: {n_exc}  |  sin_cuant: {n_sc}  |  total: {len(clasifs)}")


def main():
    ap = argparse.ArgumentParser(description="Computa y persiste perfiles_ticker")
    ap.add_argument("--fecha", help="fecha del snapshot YYYY-MM-DD (default hoy)")
    ap.add_argument("--dry-run", action="store_true", help="calcula y muestra, no escribe")
    args = ap.parse_args()

    fecha = (datetime.strptime(args.fecha, "%Y-%m-%d").date()
             if args.fecha else date.today())

    log(SEP)
    log(f"Computo de perfiles_ticker | fecha snapshot = {fecha} | LOCAL")
    log(SEP)

    eng = get_local_engine()
    log("Cargando universo, precios y benchmark...")
    universo, precios, bench = cargar_datos(eng)
    log(f"Universo: {len(universo)} | precios: {len(precios)} filas | "
        f"benchmark {BENCH_TICKER}: {len(bench)} ruedas "
        f"({bench['fecha'].min()}..{bench['fecha'].max()})" if len(bench)
        else f"Universo: {len(universo)} | SIN benchmark {BENCH_TICKER}")

    log("Corriendo Fase 1 (metricas) + Fase 2 (clasificacion)...")
    clasifs = construir_clasificaciones(universo, precios, bench)
    resumen(clasifs)

    if args.dry_run:
        log("[DRY RUN] no se escribe nada.")
        return

    filas = [_fila(c, fecha) for c in clasifs]
    log(f"UPSERT de {len(filas)} filas en perfiles_ticker...")
    with eng.begin() as c:
        for f in filas:
            c.execute(UPSERT, f)
    log("Listo.")


if __name__ == "__main__":
    main()
