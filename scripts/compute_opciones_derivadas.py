"""
compute_opciones_derivadas.py
Computa las tablas DERIVADAS de opciones en LOCAL desde el crudo (opciones_snapshot).

Parte de la migracion "snapshot nube solo-crudo" (AGENDA Tarea 17): la nube captura
solo el crudo (opciones_snapshot); este script computa en LOCAL todo lo derivado para
una fecha, desde el crudo ya synced:
  0. Precio de referencia     resuelve el precio por ticker y lo DIAGNOSTICA: cuantos
                              toman el close de precios_diarios (y cuantos llevados a la
                              escala del dia por un split posterior), cuantos caen al
                              precio de la captura, cuantos quedan sin precio, y si los
                              dos divergen (> 1%). Lo aplica el paso 2
  1. HV (hv_20d)              desde precios_diarios LOCAL -> UPDATE opciones_snapshot
  2. opciones_resumen_diario  (recompute desde el crudo; precio_sub = precio de referencia)
  3. opciones_zscore_diario + opciones_sector_zscore_diario   (src.utils.zscore_pipeline)
  4. opciones_pcr_plazo_diario + opciones_sector_pcr_plazo_diario (src.utils.opciones_plazo;
     zona de los muros y expected move sobre el precio de referencia)

Precio de referencia (10/9/2026, src/utils/precio_referencia.py): el close de la rueda en
precios_diarios manda, llevado a la escala de ESE dia con los splits reales posteriores
(registro splits_aplicados, lo escribe scripts/manual/splits.py corregir);
opciones_snapshot.precio_subyacente solo tapa el hueco. La fuente
usada queda en `precio_fuente` de opciones_resumen_diario y opciones_pcr_plazo_diario.
El paso 0 es ademas un detector de huecos de precios_diarios: si la rueda no esta
cargada, sus tickers caen al precio de la captura y se avisa.

Target SIEMPRE local: NO carga .env.local, usa get_local_engine() (lee .env directo).
Todas las funciones reciben el engine local explicito. Idempotente (UPSERT / UPDATE).

Uso:
    python scripts/compute_opciones_derivadas.py                     # ultima fecha en snapshot local
    python scripts/compute_opciones_derivadas.py --fecha 2026-06-05
    python scripts/compute_opciones_derivadas.py --desde 2026-04-18  # todas las fechas >= desde, en orden
"""

import sys
import os
import math
import argparse
from collections import defaultdict
from datetime import date, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import text
# get_local_engine arma el engine LOCAL leyendo .env directo, sin tocar os.environ
# (no carga .env.local -> DATABASE_URL queda sin setear -> todo apunta a local).
from scripts.migrations.sync_railway_to_local import get_local_engine
from src.utils import precio_referencia as pr

SEP = "=" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── 1. HV ──────────────────────────────────────────────────────────────────────

def computar_hv(engine, fecha: date) -> tuple[int, int]:
    """
    Computa hv_20d (vol historica anualizada 20d) por ticker desde precios_diarios
    LOCAL, anclada en la fecha del snapshot (no CURRENT_DATE -> backfilleable).
    Hace UPDATE opciones_snapshot SET hv_20d para todas las filas de esa fecha/ticker.
    Retorna (tickers_con_hv, filas_actualizadas).
    """
    with engine.connect() as conn:
        tickers = [r[0] for r in conn.execute(text(
            "SELECT DISTINCT ticker FROM opciones_snapshot WHERE fecha_snapshot = :f"
        ), {"f": fecha}).fetchall()]
        if not tickers:
            return 0, 0
        rows = conn.execute(text("""
            SELECT ticker, close
            FROM   precios_diarios
            WHERE  ticker = ANY(:tickers)
              AND  close > 0
              AND  fecha <= CAST(:f AS date)
              AND  fecha >= CAST(:f AS date) - INTERVAL '40 days'
            ORDER  BY ticker, fecha ASC
        """), {"tickers": tickers, "f": fecha}).fetchall()

    closes = defaultdict(list)
    for t, c in rows:
        closes[t].append(float(c))

    hv_map: dict[str, float] = {}
    for t, cl in closes.items():
        if len(cl) < 5:
            continue
        c = cl[-21:]
        lr = [math.log(c[i] / c[i - 1]) for i in range(1, len(c))]
        n = len(lr)
        mean = sum(lr) / n
        var = sum((r - mean) ** 2 for r in lr) / (n - 1) if n > 1 else 0.0
        hv_map[t] = round(math.sqrt(var) * math.sqrt(252), 6)

    filas = 0
    with engine.begin() as conn:
        for t, hv in hv_map.items():
            r = conn.execute(text(
                "UPDATE opciones_snapshot SET hv_20d = :hv "
                "WHERE ticker = :t AND fecha_snapshot = :f"
            ), {"hv": hv, "t": t, "f": fecha})
            filas += r.rowcount or 0
    return len(hv_map), filas


# ── 2. resumen_diario (recompute desde el crudo) ───────────────────────────────

# Mismo agregado que cmd_backfill_resumen() de 33_opciones_snapshot.py, por :fecha.
# precio_sub entra con el precio de la CAPTURA y precio_fuente en NULL: los fija
# despues computar_resumen() con precio_referencia.resolver_precio. Asi la regla
# (incluida la excepcion de escala de split) vive en Python en un solo lugar y no
# en una replica SQL que se desincronice.
SQL_RESUMEN = """
    WITH oi_per_key AS (
        SELECT fecha_snapshot, ticker, strike, vencimiento,
               SUM(COALESCE(open_interest, 0)) AS total_oi
        FROM   opciones_snapshot
        WHERE  fecha_snapshot = :fecha
        GROUP  BY fecha_snapshot, ticker, strike, vencimiento
    ),
    top_strike AS (
        SELECT DISTINCT ON (ticker)
               ticker, strike AS max_oi_strike, vencimiento AS max_oi_venc
        FROM   oi_per_key
        ORDER  BY ticker, total_oi DESC
    ),
    agg AS (
        SELECT
            fecha_snapshot AS fecha, ticker,
            SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END)       AS call_vol,
            SUM(CASE WHEN tipo='put'  THEN COALESCE(volumen,0) ELSE 0 END)       AS put_vol,
            SUM(CASE WHEN tipo='call' THEN COALESCE(open_interest,0) ELSE 0 END) AS call_oi,
            SUM(CASE WHEN tipo='put'  THEN COALESCE(open_interest,0) ELSE 0 END) AS put_oi,
            CASE WHEN SUM(CASE WHEN tipo='call' AND iv IS NOT NULL THEN COALESCE(open_interest,0) ELSE 0 END) > 0
                 THEN ROUND(SUM(CASE WHEN tipo='call' AND iv IS NOT NULL THEN iv*COALESCE(open_interest,0) ELSE 0 END)
                       / SUM(CASE WHEN tipo='call' AND iv IS NOT NULL THEN COALESCE(open_interest,0) ELSE 0 END)::NUMERIC, 6)
                 ELSE NULL END AS iv_call_avg,
            CASE WHEN SUM(CASE WHEN tipo='put' AND iv IS NOT NULL THEN COALESCE(open_interest,0) ELSE 0 END) > 0
                 THEN ROUND(SUM(CASE WHEN tipo='put' AND iv IS NOT NULL THEN iv*COALESCE(open_interest,0) ELSE 0 END)
                       / SUM(CASE WHEN tipo='put' AND iv IS NOT NULL THEN COALESCE(open_interest,0) ELSE 0 END)::NUMERIC, 6)
                 ELSE NULL END AS iv_put_avg,
            COUNT(*)               AS n_contratos,
            MAX(precio_subyacente) AS precio_snap   -- de la CAPTURA (fallback)
        FROM opciones_snapshot
        WHERE fecha_snapshot = :fecha
        GROUP BY fecha_snapshot, ticker
    )
    INSERT INTO opciones_resumen_diario (
        fecha, ticker, call_vol, put_vol, pcr_vol, call_oi, put_oi, pcr_oi,
        iv_call_avg, iv_put_avg, n_contratos, max_oi_strike, max_oi_venc, precio_sub,
        precio_fuente
    )
    SELECT a.fecha, a.ticker, a.call_vol, a.put_vol,
           CASE WHEN a.call_vol > 0 THEN ROUND(a.put_vol::NUMERIC/a.call_vol, 4) ELSE NULL END,
           a.call_oi, a.put_oi,
           CASE WHEN a.call_oi > 0 THEN ROUND(a.put_oi::NUMERIC/a.call_oi, 4) ELSE NULL END,
           a.iv_call_avg, a.iv_put_avg, a.n_contratos, t.max_oi_strike, t.max_oi_venc,
           a.precio_snap, NULL
    FROM agg a LEFT JOIN top_strike t USING (ticker)
    ON CONFLICT (fecha, ticker) DO UPDATE SET
        call_vol=EXCLUDED.call_vol, put_vol=EXCLUDED.put_vol, pcr_vol=EXCLUDED.pcr_vol,
        call_oi=EXCLUDED.call_oi, put_oi=EXCLUDED.put_oi, pcr_oi=EXCLUDED.pcr_oi,
        iv_call_avg=EXCLUDED.iv_call_avg, iv_put_avg=EXCLUDED.iv_put_avg,
        n_contratos=EXCLUDED.n_contratos, max_oi_strike=EXCLUDED.max_oi_strike,
        max_oi_venc=EXCLUDED.max_oi_venc, precio_sub=EXCLUDED.precio_sub,
        precio_fuente=EXCLUDED.precio_fuente
"""


def computar_resumen(engine, fecha: date, resueltos: dict) -> int:
    """
    Recompute del resumen y, en la misma transaccion, el precio de referencia.
    `resueltos` = {ticker: (precio, fuente)} de chequear_precio_referencia(): la
    misma regla que usan los muros en opciones_plazo.
    """
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE opciones_resumen_diario ADD COLUMN IF NOT EXISTS precio_fuente VARCHAR(16)"))
        r = conn.execute(text(SQL_RESUMEN), {"fecha": fecha})
        n = r.rowcount or 0
        if resueltos:
            conn.execute(
                text("UPDATE opciones_resumen_diario SET precio_sub = :p, precio_fuente = :fu "
                     "WHERE fecha = :fecha AND ticker = :t"),
                [{"p": p, "fu": fu, "fecha": fecha, "t": t} for t, (p, fu) in resueltos.items()])
        return n


# ── 0. Precio de referencia (diagnostico) ──────────────────────────────────────

def chequear_precio_referencia(engine, fecha: date) -> dict:
    """
    Aplica la regla de src/utils/precio_referencia.py a los tickers del snapshot
    de `fecha` y devuelve los precios resueltos, el reparto por fuente y las
    divergencias. NO escribe: el paso 2 persiste `resueltos` y el paso 4 aplica
    la misma funcion por su cuenta.

    Sirve ademas de detector de huecos de precios_diarios: si una rueda no esta
    cargada, sus tickers caen al precio de la captura (caso 2026-08-28: 157 de
    200 tickers sin close).
    """
    with engine.connect() as conn:
        snaps = {t: px for t, px in conn.execute(text(
            "SELECT ticker, MAX(precio_subyacente) FROM opciones_snapshot "
            "WHERE fecha_snapshot = :f GROUP BY ticker"), {"f": fecha}).fetchall()}
        closes = {}
        if snaps:
            closes = {t: c for t, c in conn.execute(text(
                "SELECT ticker, close FROM precios_diarios "
                "WHERE fecha = :f AND ticker = ANY(:tks)"),
                {"f": fecha, "tks": list(snaps)}).fetchall()}
    from src.utils.opciones_plazo import cargar_factores_escala
    factores = {t: fa for t, fa in cargar_factores_escala(engine, fecha).items() if t in snaps}
    resueltos = pr.resolver_mapa(closes, snaps, tickers=list(snaps), factores=factores)
    return {
        "n": len(snaps),
        "resueltos": resueltos,
        "factores": factores,
        "fuentes": pr.contar_fuentes(resueltos),
        "captura_sin_precio": sum(1 for v in snaps.values()
                                  if pr.resolver_precio(None, v)[0] is None),
        "divergencias": pr.medir_divergencias(closes, snaps, factores=factores),
        "sin_registro": pr.escalas_sin_registro(closes, snaps, factores=factores),
    }


def _log_precio_referencia(chk: dict, fecha: date):
    f = chk["fuentes"]
    n_dueno = f[pr.FUENTE_DIARIO] + f[pr.FUENTE_DIARIO_ESCALA]
    log(f"  precio referencia : {chk['n']} tickers | precios_diarios {n_dueno}"
        f" (en escala de split: {f[pr.FUENTE_DIARIO_ESCALA]}) | captura (hueco) {f[pr.FUENTE_SNAPSHOT]}"
        f" | sin precio {f['sin_precio']}")
    if chk["captura_sin_precio"]:
        log(f"  [INFO] la captura vino sin precio en {chk['captura_sin_precio']}/{chk['n']} tickers"
            f" (no afecta a los que tienen close en precios_diarios)")
    if f[pr.FUENTE_DIARIO_ESCALA]:
        muestra = ", ".join(f"{t} x{fa:g}" for t, fa in sorted(chk["factores"].items()))
        log(f"  [INFO] {f[pr.FUENTE_DIARIO_ESCALA]} tickers con split posterior: el close se lleva a la"
            f" escala de ese dia (la de los strikes): {muestra}")
    if f[pr.FUENTE_SNAPSHOT]:
        log(f"  [WARN] {f[pr.FUENTE_SNAPSHOT]} tickers sin close en precios_diarios para {fecha}:"
            f" usan el precio de la captura. Si la rueda deberia estar cargada, falta el recovery de precios.")
    if f["sin_precio"]:
        log(f"  [WARN] {f['sin_precio']} tickers sin ningun precio para {fecha}:"
            f" quedan sin muros ni expected move.")
    div = chk["divergencias"]
    if div:
        muestra = ", ".join(f"{t} {c:.2f} vs {s:.2f} ({d * 100:.1f}%)" for t, c, s, d in div[:5])
        log(f"  [WARN] {len(div)} tickers con close y captura divergentes"
            f" (>{pr.TOL_DIVERGENCIA * 100:.0f}%): {muestra}")
    sr = chk["sin_registro"]
    if sr:
        muestra = ", ".join(f"{t} {c:.2f} vs {s:.2f} (x{k:g})" for t, c, s, k in sr[:5])
        log(f"  [WARN] {len(sr)} tickers donde captura y close difieren por un split EXACTO que"
            f" splits_aplicados no explica (split sin corregir con splits.py, o corregido a mano"
            f" sin registrar): {muestra}")


# ── Main ────────────────────────────────────────────────────────────────────────

def computar_fecha(engine, fecha: date):
    """Corre los pasos 0-4 para UNA fecha. Cada paso aisla su error (no corta los demas)."""
    print()
    print(SEP)
    print(f"  COMPUTE OPCIONES DERIVADAS (LOCAL)  |  fecha = {fecha}")
    print(SEP)

    # 0. Precio de referencia (resuelve y diagnostica; lo persiste el paso 2)
    resueltos = {}
    try:
        chk = chequear_precio_referencia(engine, fecha)
        resueltos = chk["resueltos"]
        _log_precio_referencia(chk, fecha)
    except Exception as e:
        log(f"  [ERROR] precio referencia: {e}")

    # 1. HV
    try:
        n_hv, n_filas = computar_hv(engine, fecha)
        log(f"  HV_20d            : {n_hv} tickers -> UPDATE {n_filas:,} filas snapshot")
    except Exception as e:
        log(f"  [ERROR] HV: {e}")

    # 2. resumen
    try:
        n_res = computar_resumen(engine, fecha, resueltos)
        log(f"  resumen_diario    : {n_res} tickers")
    except Exception as e:
        log(f"  [ERROR] resumen: {e}")

    # 3. zscore + sector
    try:
        from src.utils.zscore_pipeline import (
            calcular_zscore_opciones, calcular_zscore_opciones_sector, init_tablas
        )
        init_tablas(engine)
        n_z = calcular_zscore_opciones(fecha, engine)
        log(f"  zscore opciones   : {n_z} tickers")
        n_zs = calcular_zscore_opciones_sector(fecha, engine)
        log(f"  zscore sector     : {n_zs} sectores")
    except Exception as e:
        log(f"  [ERROR] zscore: {e}")

    # 4. pcr_plazo + sector
    try:
        from src.utils.opciones_plazo import (
            calcular_pcr_plazo, calcular_pcr_sector_plazo, init_tabla, init_tabla_sector
        )
        init_tabla(engine)
        init_tabla_sector(engine)
        n_p = calcular_pcr_plazo(fecha, engine)
        log(f"  pcr_plazo         : {n_p} filas")
        n_ps = calcular_pcr_sector_plazo(fecha, engine)
        log(f"  pcr_plazo sector  : {n_ps} filas")
    except Exception as e:
        log(f"  [ERROR] pcr_plazo: {e}")

    print(SEP)
    log("  Completado.")
    print()


def fechas_desde(engine, desde: date) -> list:
    """Fechas del snapshot local >= desde, ASCENDENTES (los z-scores usan la historia previa)."""
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text(
            "SELECT DISTINCT fecha_snapshot FROM opciones_snapshot "
            "WHERE fecha_snapshot >= :d ORDER BY fecha_snapshot"), {"d": desde}).fetchall()]


def run(fecha: date = None, desde: date = None):
    engine = get_local_engine()

    if desde is not None:
        fechas = fechas_desde(engine, desde)
        if not fechas:
            log(f"No hay fechas en opciones_snapshot local >= {desde}. Nada que computar.")
            return
        log(f"Recalculo de {len(fechas)} fechas ({fechas[0]} .. {fechas[-1]}), en orden.")
        for f in fechas:
            computar_fecha(engine, f)
        return

    if fecha is None:
        with engine.connect() as conn:
            fecha = conn.execute(text("SELECT MAX(fecha_snapshot) FROM opciones_snapshot")).scalar()
    if fecha is None:
        log("No hay datos en opciones_snapshot local. Nada que computar.")
        return
    computar_fecha(engine, fecha)


def main():
    parser = argparse.ArgumentParser(description="Computa derivadas de opciones en LOCAL desde el crudo")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--fecha", help="YYYY-MM-DD (default: ultima fecha en opciones_snapshot local)")
    grupo.add_argument("--desde", help="YYYY-MM-DD: recalcula TODAS las fechas >= desde, en orden "
                                       "(los z-scores usan la historia previa)")
    args = parser.parse_args()
    fecha = date.fromisoformat(args.fecha) if args.fecha else None
    desde = date.fromisoformat(args.desde) if args.desde else None
    run(fecha, desde)


if __name__ == "__main__":
    main()
