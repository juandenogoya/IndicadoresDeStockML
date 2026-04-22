"""
34_opciones_analisis.py
Analisis de sentimiento sobre datos de opciones.

Fuente principal: opciones_resumen_diario (~199 filas/dia, acumulado desde 18/4/2026)
Fuente detalle:   opciones_snapshot       (~85-90k filas/dia, nivel contrato)

Analisis disponibles:
    --analisis                Nivel 0+1: analisis macro por ticker (PCR, IV skew, sesgo,
                              grid DTE x moneyness con VOL%, OI%, IV ponderado, mov.esperado)
    --drill                   Nivel 2: detalle de strikes individuales (DTE, prima, IV/HV, V/OI)
    --vol-ticker              L1: Volumen call/put por ticker (ranking universo)
    --vol-vencimiento         L2: Volumen call/put por ticker + vencimiento
    --vol-strike              L3: Volumen call/put por ticker + vencimiento + strike
    --vol-sector              L1/L2: Volumen agregado por sector
    --vol-industria           L1/L2: Volumen agregado por industria

Uso analisis por ticker:
    python scripts/34_opciones_analisis.py --analisis --ticker NVDA
    python scripts/34_opciones_analisis.py --analisis --ticker NVDA --fecha 2026-04-20
    python scripts/34_opciones_analisis.py --analisis --ticker NVDA --atm-pct 5

    python scripts/34_opciones_analisis.py --drill --ticker NVDA --tipo call --dte 30
    python scripts/34_opciones_analisis.py --drill --ticker NVDA --tipo put  --dte 60
    python scripts/34_opciones_analisis.py --drill --ticker NVDA --tipo call --dte 30 --top 10

Uso analisis universo:
    python scripts/34_opciones_analisis.py --vol-ticker
    python scripts/34_opciones_analisis.py --vol-ticker --fecha 2026-04-18
    python scripts/34_opciones_analisis.py --vol-ticker --top 20

    python scripts/34_opciones_analisis.py --vol-vencimiento --ticker BAC
    python scripts/34_opciones_analisis.py --vol-vencimiento --venc 2026-05-15
    python scripts/34_opciones_analisis.py --vol-vencimiento --venc 2026-05-15 --top 20

    python scripts/34_opciones_analisis.py --vol-strike --ticker BAC --venc 2026-05-15
    python scripts/34_opciones_analisis.py --vol-strike --ticker BAC

    python scripts/34_opciones_analisis.py --vol-sector
    python scripts/34_opciones_analisis.py --vol-sector --sector Technology

    python scripts/34_opciones_analisis.py --vol-industria
    python scripts/34_opciones_analisis.py --vol-industria --industria Semiconductors
"""

import sys
import os
import argparse
import math
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

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

from sqlalchemy import text
from src.data.database import get_engine


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _fmt_vol(n: Optional[int]) -> str:
    if n is None:
        return "   N/A"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _fmt_pcr(v) -> str:
    if v is None:
        return "  N/A"
    return f"{float(v):.2f}"


def _ultima_fecha(engine) -> Optional[date]:
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT MAX(fecha) FROM opciones_resumen_diario"
        )).fetchone()
    return row[0] if row else None


# ── L1: Volumen call/put por ticker ──────────────────────────────────────────

def cmd_vol_ticker(fecha: Optional[date] = None, top: int = 0):
    """
    Ranking de tickers por volumen total de opciones (call + put) del dia.
    Muestra call_vol, put_vol, total_vol y PCR_vol.

    PCR_vol = put_vol / call_vol.
        > 1.0  ->  mas puts que calls  (sesgo bajista / hedging)
        < 1.0  ->  mas calls que puts  (sesgo alcista / especulacion)
    """
    engine = get_engine()

    if fecha is None:
        fecha = _ultima_fecha(engine)
        if fecha is None:
            log("Sin datos en opciones_resumen_diario.")
            return

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                ticker,
                call_vol,
                put_vol,
                COALESCE(call_vol, 0) + COALESCE(put_vol, 0)  AS total_vol,
                pcr_vol
            FROM  opciones_resumen_diario
            WHERE fecha = :fecha
            ORDER BY total_vol DESC
        """), {"fecha": fecha}).fetchall()

    if not rows:
        log(f"Sin datos para {fecha}.")
        return

    if top > 0:
        rows = rows[:top]

    separador = "-" * 52
    print()
    print(f"  VOLUMEN OPCIONES POR TICKER  |  {fecha}")
    if top > 0:
        print(f"  Top {top} por volumen total")
    print(f"  {'#':>3s}  {'TICKER':<8s}  {'CALL_VOL':>9s}  {'PUT_VOL':>9s}  {'TOTAL':>9s}  {'PCR_vol':>7s}")
    print("  " + separador)

    for i, r in enumerate(rows, 1):
        ticker    = r[0]
        call_vol  = r[1]
        put_vol   = r[2]
        total_vol = r[3]
        pcr_vol   = r[4]

        # Indicador visual de sesgo
        if pcr_vol is None:
            sesgo = " "
        elif float(pcr_vol) >= 1.5:
            sesgo = "v"    # sesgo bajista fuerte
        elif float(pcr_vol) >= 1.0:
            sesgo = "-"    # sesgo bajista leve
        else:
            sesgo = "^"    # sesgo alcista

        print(
            f"  {i:>3d}  {ticker:<8s}  "
            f"{_fmt_vol(call_vol):>9s}  "
            f"{_fmt_vol(put_vol):>9s}  "
            f"{_fmt_vol(total_vol):>9s}  "
            f"{_fmt_pcr(pcr_vol):>7s} {sesgo}"
        )

    print("  " + separador)

    # Totales agregados
    total_call = sum(r[1] or 0 for r in rows)
    total_put  = sum(r[2] or 0 for r in rows)
    total_all  = total_call + total_put
    pcr_agg    = round(total_put / total_call, 2) if total_call > 0 else None

    print(
        f"  {'':>3s}  {'TOTAL':<8s}  "
        f"{_fmt_vol(total_call):>9s}  "
        f"{_fmt_vol(total_put):>9s}  "
        f"{_fmt_vol(total_all):>9s}  "
        f"{_fmt_pcr(pcr_agg):>7s}"
    )
    print()
    print(f"  Tickers mostrados: {len(rows)}")
    print(f"  Leyenda PCR_vol:  ^ alcista (<1.0)  - bajista leve (1.0-1.5)  v bajista fuerte (>=1.5)")
    print()


# ── L2: Volumen call/put por ticker + vencimiento ────────────────────────────

def cmd_vol_vencimiento(
    fecha: Optional[date] = None,
    ticker: Optional[str] = None,
    venc: Optional[date] = None,
    top: int = 0,
):
    """
    Volumen call/put desglosado por (ticker, vencimiento).

    Modos:
        --ticker BAC            -> todos los vencimientos de BAC ese dia
        --venc 2026-05-16       -> todos los tickers para ese vencimiento (ranking)
        --ticker BAC --venc ... -> un ticker + un vencimiento especifico

    Util para:
        - Ver en que vencimientos se concentra el flujo de un ticker
        - Detectar rotacion de interes entre vencimientos de un dia al otro
        - Comparar actividad entre tickers para un mismo vencimiento
    """
    engine = get_engine()

    if fecha is None:
        fecha = _ultima_fecha(engine)
        if fecha is None:
            log("Sin datos en opciones_resumen_diario.")
            return

    # Construir WHERE dinamico
    filtros = ["fecha_snapshot = :fecha"]
    params: dict = {"fecha": fecha}

    if ticker:
        filtros.append("ticker = :ticker")
        params["ticker"] = ticker.upper()

    if venc:
        filtros.append("vencimiento = :venc")
        params["venc"] = venc

    where = " AND ".join(filtros)

    # ORDER BY: si se filtra por ticker -> por vencimiento asc; sino -> por volumen desc
    if ticker is not None and venc is None:
        order_clause = "ORDER BY ticker ASC, vencimiento ASC"
    else:
        order_clause = "ORDER BY total_vol DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            WITH base AS (
                SELECT
                    ticker,
                    vencimiento,
                    SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END) AS call_vol,
                    SUM(CASE WHEN tipo='put'  THEN COALESCE(volumen,0) ELSE 0 END) AS put_vol,
                    SUM(COALESCE(volumen,0))                                        AS total_vol,
                    CASE
                        WHEN SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END) > 0
                        THEN ROUND(
                            SUM(CASE WHEN tipo='put' THEN COALESCE(volumen,0) ELSE 0 END)::NUMERIC /
                            SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END), 2)
                        ELSE NULL
                    END AS pcr_vol,
                    (vencimiento - CAST(:fecha AS date)) AS dte
                FROM  opciones_snapshot
                WHERE {where}
                GROUP BY ticker, vencimiento
            )
            SELECT * FROM base
            {order_clause}
        """), params).fetchall()

    if not rows:
        log(f"Sin datos para los filtros indicados ({fecha}).")
        return

    if top > 0:
        rows = rows[:top]

    # Titulo
    titulo_partes = [f"VOLUMEN POR TICKER + VENCIMIENTO  |  {fecha}"]
    if ticker:
        titulo_partes.append(f"Ticker: {ticker.upper()}")
    if venc:
        titulo_partes.append(f"Vencimiento: {venc}")
    if top > 0:
        titulo_partes.append(f"Top {top}")

    separador = "-" * 62

    print()
    for p in titulo_partes:
        print(f"  {p}")
    print(f"  {'TICKER':<8s}  {'VENC':<12s}  {'DTE':>4s}  {'CALL_VOL':>9s}  {'PUT_VOL':>9s}  {'TOTAL':>9s}  {'PCR':>6s}")
    print("  " + separador)

    prev_ticker = None
    for r in rows:
        t         = r[0]
        v         = r[1]
        call_vol  = r[2]
        put_vol   = r[3]
        total_vol = r[4]
        pcr       = r[5]
        dte       = r[6]

        # Separador visual entre tickers cuando se listan multiples
        if ticker is None and venc is None and t != prev_ticker and prev_ticker is not None:
            print("  " + separador)
        prev_ticker = t

        if pcr is None:
            sesgo = " "
        elif float(pcr) >= 1.5:
            sesgo = "v"
        elif float(pcr) >= 1.0:
            sesgo = "-"
        else:
            sesgo = "^"

        # Resaltar vencimientos cortos (weeklies <= 14 DTE)
        dte_tag = f"{dte}w" if dte is not None and dte <= 14 else f"{dte} " if dte is not None else "  "

        print(
            f"  {t:<8s}  {str(v):<12s}  {dte_tag:>4s}  "
            f"{_fmt_vol(call_vol):>9s}  "
            f"{_fmt_vol(put_vol):>9s}  "
            f"{_fmt_vol(total_vol):>9s}  "
            f"{_fmt_pcr(pcr):>6s} {sesgo}"
        )

    print("  " + separador)
    total_call = sum(r[2] or 0 for r in rows)
    total_put  = sum(r[3] or 0 for r in rows)
    total_all  = total_call + total_put
    pcr_agg    = round(total_put / total_call, 2) if total_call > 0 else None

    label = ticker.upper() if ticker else ("TOTAL" if not venc else str(venc))
    print(
        f"  {label:<8s}  {'':12s}  {'':>4s}  "
        f"{_fmt_vol(total_call):>9s}  "
        f"{_fmt_vol(total_put):>9s}  "
        f"{_fmt_vol(total_all):>9s}  "
        f"{_fmt_pcr(pcr_agg):>6s}"
    )
    print()
    print(f"  Filas mostradas: {len(rows)}")
    print(f"  Leyenda DTE: 'w' = weekly (<=14 dias)  |  PCR: ^ <1.0  - 1.0-1.5  v >=1.5")
    print()


# ── L3: Volumen call/put por ticker + vencimiento + strike ───────────────────

def cmd_vol_strike(
    fecha: Optional[date] = None,
    ticker: Optional[str] = None,
    venc: Optional[date] = None,
    top: int = 0,
):
    """
    Volumen call/put al nivel mas granular: (ticker, vencimiento, strike).
    Solo incluye strikes que tuvieron volumen real en el dia (vol > 0).

    Muestra distancia porcentual del strike al precio actual del subyacente:
        ~0%  = ATM (at the money)
        +N%  = OTM para calls / ITM para puts  (strike sobre el precio)
        -N%  = ITM para calls / OTM para puts  (strike bajo el precio)

    Recomendado usar con --ticker (obligatorio) y opcionalmente --venc.
    Sin --venc muestra todos los vencimientos agrupados.
    """
    engine = get_engine()

    if not ticker:
        print("  ERROR: --vol-strike requiere --ticker (ej: --ticker BAC)")
        return

    if fecha is None:
        fecha = _ultima_fecha(engine)
        if fecha is None:
            log("Sin datos en opciones_resumen_diario.")
            return

    filtros = ["fecha_snapshot = :fecha", "ticker = :ticker"]
    params: dict = {"fecha": fecha, "ticker": ticker.upper()}

    if venc:
        filtros.append("vencimiento = :venc")
        params["venc"] = venc

    where = " AND ".join(filtros)

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            WITH base AS (
                SELECT
                    ticker,
                    vencimiento,
                    strike,
                    MAX(precio_subyacente)                                           AS precio_sub,
                    SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END)  AS call_vol,
                    SUM(CASE WHEN tipo='put'  THEN COALESCE(volumen,0) ELSE 0 END)  AS put_vol,
                    SUM(COALESCE(volumen,0))                                         AS total_vol,
                    CASE
                        WHEN SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END) > 0
                        THEN ROUND(
                            SUM(CASE WHEN tipo='put' THEN COALESCE(volumen,0) ELSE 0 END)::NUMERIC /
                            SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END), 2)
                        ELSE NULL
                    END AS pcr_vol,
                    (vencimiento - CAST(:fecha AS date))                             AS dte
                FROM opciones_snapshot
                WHERE {where}
                GROUP BY ticker, vencimiento, strike
            )
            SELECT
                ticker, vencimiento, strike, precio_sub,
                call_vol, put_vol, total_vol, pcr_vol, dte,
                CASE
                    WHEN precio_sub > 0
                    THEN ROUND((strike - precio_sub) / precio_sub * 100, 1)
                    ELSE NULL
                END AS dist_pct
            FROM base
            WHERE total_vol > 0
            ORDER BY vencimiento ASC, strike ASC
        """), params).fetchall()

    if not rows:
        log(f"Sin strikes con volumen para {ticker.upper()} ({fecha}).")
        return

    if top > 0:
        # top por volumen total dentro del resultado ya filtrado
        rows = sorted(rows, key=lambda r: r[6], reverse=True)[:top]
        rows = sorted(rows, key=lambda r: (r[1], r[2]))  # reordenar por venc+strike

    titulo_partes = [f"VOLUMEN POR STRIKE  |  {ticker.upper()}  |  {fecha}"]
    if venc:
        titulo_partes.append(f"Vencimiento: {venc}")
    if top > 0:
        titulo_partes.append(f"Top {top} strikes por volumen")

    separador = "-" * 66

    print()
    for p in titulo_partes:
        print(f"  {p}")

    precio_ref = rows[0][3] if rows else None
    if precio_ref:
        print(f"  Precio actual: {float(precio_ref):.2f}")

    print(f"  {'VENC':<12s}  {'DTE':>4s}  {'STRIKE':>8s}  {'DIST':>6s}  {'CALL_VOL':>9s}  {'PUT_VOL':>9s}  {'TOTAL':>9s}  {'PCR':>6s}")
    print("  " + separador)

    prev_venc = None
    for r in rows:
        v         = r[1]
        strike    = float(r[2])
        precio_sub = r[3]
        call_vol  = r[4]
        put_vol   = r[5]
        total_vol = r[6]
        pcr       = r[7]
        dte       = r[8]
        dist_pct  = r[9]

        # Separador entre vencimientos
        if v != prev_venc and prev_venc is not None:
            print("  " + separador)
        prev_venc = v

        dte_tag = f"{dte}w" if dte is not None and dte <= 14 else f"{dte} " if dte is not None else "  "

        # Indicador de moneyness
        if dist_pct is None:
            dist_str = "   N/A"
        else:
            d = float(dist_pct)
            if abs(d) <= 2.0:
                dist_str = f" ~{d:+.1f}%"   # ATM
            else:
                dist_str = f"  {d:+.1f}%"

        if pcr is None:
            sesgo = " "
        elif float(pcr) >= 1.5:
            sesgo = "v"
        elif float(pcr) >= 1.0:
            sesgo = "-"
        else:
            sesgo = "^"

        print(
            f"  {str(v):<12s}  {dte_tag:>4s}  "
            f"{strike:>8.2f}  {dist_str:>6s}  "
            f"{_fmt_vol(call_vol):>9s}  "
            f"{_fmt_vol(put_vol):>9s}  "
            f"{_fmt_vol(total_vol):>9s}  "
            f"{_fmt_pcr(pcr):>6s} {sesgo}"
        )

    print("  " + separador)
    total_call = sum(r[4] or 0 for r in rows)
    total_put  = sum(r[5] or 0 for r in rows)
    total_all  = total_call + total_put
    pcr_agg    = round(total_put / total_call, 2) if total_call > 0 else None

    print(
        f"  {ticker.upper():<12s}  {'':>4s}  {'':>8s}  {'':>6s}  "
        f"{_fmt_vol(total_call):>9s}  "
        f"{_fmt_vol(total_put):>9s}  "
        f"{_fmt_vol(total_all):>9s}  "
        f"{_fmt_pcr(pcr_agg):>6s}"
    )
    print()
    print(f"  Strikes mostrados: {len(rows)}")
    print(f"  Leyenda DIST: ~0% ATM  +N% strike sobre precio  -N% strike bajo precio")
    print(f"  Leyenda DTE:  'w' = weekly (<=14 dias)")
    print(f"  Leyenda PCR:  ^ <1.0  - 1.0-1.5  v >=1.5")
    print()


# ── L1/L2 por Sector o Industry (funcion compartida) ─────────────────────────

def _cmd_vol_grupo(
    grupo_col: str,             # 'sector' o 'industry'
    grupo_val: Optional[str],   # filtro opcional (ej: 'Technology')
    fecha: Optional[date],
    venc: Optional[date],
    top: int,
):
    """
    Agrega volumen call/put por sector o industry.

    L1 (sin --venc, sin filtro de grupo):
        Ranking de todos los grupos por volumen total del dia.
        Fuente: opciones_resumen_diario JOIN activos.

    L2 (con --venc O con filtro de grupo):
        Desglosa por vencimiento.
        - --venc FECHA:  todos los grupos para ese vencimiento
        - --sector X:    todos los vencimientos del grupo X
        Fuente: opciones_snapshot JOIN activos.
    """
    engine = get_engine()

    if fecha is None:
        fecha = _ultima_fecha(engine)
        if fecha is None:
            log("Sin datos disponibles.")
            return

    modo_l2 = venc is not None or grupo_val is not None
    label_col = grupo_col.capitalize()

    # ── L1: resumen diario agregado por grupo ────────────────────────────────
    if not modo_l2:
        filtros = ["r.fecha = :fecha", "a.activo = true", f"a.{grupo_col} IS NOT NULL"]
        params: dict = {"fecha": fecha}

        if grupo_val:
            filtros.append(f"a.{grupo_col} = :grupo_val")
            params["grupo_val"] = grupo_val

        where = " AND ".join(filtros)

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    a.{grupo_col}                               AS grupo,
                    SUM(r.call_vol)                             AS call_vol,
                    SUM(r.put_vol)                              AS put_vol,
                    SUM(COALESCE(r.call_vol,0) + COALESCE(r.put_vol,0)) AS total_vol,
                    CASE
                        WHEN SUM(r.call_vol) > 0
                        THEN ROUND(SUM(r.put_vol)::NUMERIC / SUM(r.call_vol), 2)
                        ELSE NULL
                    END AS pcr_vol,
                    COUNT(DISTINCT r.ticker)                    AS n_tickers
                FROM opciones_resumen_diario r
                JOIN activos a ON a.ticker = r.ticker
                WHERE {where}
                GROUP BY a.{grupo_col}
                ORDER BY total_vol DESC
            """), params).fetchall()

        if not rows:
            log(f"Sin datos para {fecha}.")
            return

        if top > 0:
            rows = rows[:top]

        titulo = f"VOLUMEN POR {label_col.upper()}  |  {fecha}  (L1)"
        sep = "-" * 62
        print()
        print(f"  {titulo}")
        print(f"  {label_col.upper():<35s}  {'N':>3s}  {'CALL_VOL':>9s}  {'PUT_VOL':>9s}  {'TOTAL':>9s}  {'PCR':>6s}")
        print("  " + sep)

        for r in rows:
            grupo    = str(r[0]) if r[0] else "N/A"
            call_vol = r[1]
            put_vol  = r[2]
            total    = r[3]
            pcr      = r[4]
            n        = r[5]

            sesgo = " " if pcr is None else ("v" if float(pcr) >= 1.5 else ("-" if float(pcr) >= 1.0 else "^"))
            print(
                f"  {grupo:<35s}  {n:>3d}  "
                f"{_fmt_vol(call_vol):>9s}  {_fmt_vol(put_vol):>9s}  "
                f"{_fmt_vol(total):>9s}  {_fmt_pcr(pcr):>6s} {sesgo}"
            )

        print("  " + sep)
        total_call = sum(r[1] or 0 for r in rows)
        total_put  = sum(r[2] or 0 for r in rows)
        total_all  = total_call + total_put
        pcr_agg    = round(total_put / total_call, 2) if total_call > 0 else None
        print(
            f"  {'TOTAL':<35s}  {'':>3s}  "
            f"{_fmt_vol(total_call):>9s}  {_fmt_vol(total_put):>9s}  "
            f"{_fmt_vol(total_all):>9s}  {_fmt_pcr(pcr_agg):>6s}"
        )
        print()
        print(f"  Grupos mostrados: {len(rows)}")
        print(f"  Leyenda PCR: ^ <1.0  - 1.0-1.5  v >=1.5")
        print()
        return

    # ── L2: snapshot agregado por (grupo, vencimiento) ───────────────────────
    filtros = ["s.fecha_snapshot = :fecha", "a.activo = true", f"a.{grupo_col} IS NOT NULL"]
    params = {"fecha": fecha}

    if venc:
        filtros.append("s.vencimiento = :venc")
        params["venc"] = venc

    if grupo_val:
        filtros.append(f"a.{grupo_col} = :grupo_val")
        params["grupo_val"] = grupo_val

    where = " AND ".join(filtros)

    # ORDER: si hay filtro de grupo -> por vencimiento asc; sino -> vol desc
    order = "grupo ASC, vencimiento ASC" if grupo_val and not venc else "total_vol DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            WITH base AS (
                SELECT
                    a.{grupo_col}                                                    AS grupo,
                    s.vencimiento,
                    SUM(CASE WHEN s.tipo='call' THEN COALESCE(s.volumen,0) ELSE 0 END) AS call_vol,
                    SUM(CASE WHEN s.tipo='put'  THEN COALESCE(s.volumen,0) ELSE 0 END) AS put_vol,
                    SUM(COALESCE(s.volumen,0))                                        AS total_vol,
                    CASE
                        WHEN SUM(CASE WHEN s.tipo='call' THEN COALESCE(s.volumen,0) ELSE 0 END) > 0
                        THEN ROUND(
                            SUM(CASE WHEN s.tipo='put' THEN COALESCE(s.volumen,0) ELSE 0 END)::NUMERIC /
                            SUM(CASE WHEN s.tipo='call' THEN COALESCE(s.volumen,0) ELSE 0 END), 2)
                        ELSE NULL
                    END AS pcr_vol,
                    (s.vencimiento - CAST(:fecha AS date))                           AS dte,
                    COUNT(DISTINCT s.ticker)                                         AS n_tickers
                FROM opciones_snapshot s
                JOIN activos a ON a.ticker = s.ticker
                WHERE {where}
                GROUP BY a.{grupo_col}, s.vencimiento
            )
            SELECT * FROM base
            ORDER BY {order}
        """), params).fetchall()

    if not rows:
        log(f"Sin datos para los filtros indicados ({fecha}).")
        return

    if top > 0:
        rows = sorted(rows, key=lambda r: r[4], reverse=True)[:top]
        rows = sorted(rows, key=lambda r: (r[0], r[1]))

    titulo_partes = [f"VOLUMEN POR {label_col.upper()} + VENCIMIENTO  |  {fecha}  (L2)"]
    if grupo_val:
        titulo_partes.append(f"{label_col}: {grupo_val}")
    if venc:
        titulo_partes.append(f"Vencimiento: {venc}")
    if top > 0:
        titulo_partes.append(f"Top {top}")

    sep = "-" * 70
    print()
    for p in titulo_partes:
        print(f"  {p}")
    print(f"  {label_col.upper():<35s}  {'VENC':<12s}  {'DTE':>4s}  {'N':>3s}  {'CALL_VOL':>9s}  {'PUT_VOL':>9s}  {'TOTAL':>9s}  {'PCR':>6s}")
    print("  " + sep)

    prev_grupo = None
    for r in rows:
        grupo    = str(r[0]) if r[0] else "N/A"
        v        = r[1]
        call_vol = r[2]
        put_vol  = r[3]
        total    = r[4]
        pcr      = r[5]
        dte      = r[6]
        n        = r[7]

        if grupo != prev_grupo and prev_grupo is not None:
            print("  " + sep)
        prev_grupo = grupo

        dte_tag = f"{dte}w" if dte is not None and dte <= 14 else f"{dte} " if dte is not None else "  "
        sesgo = " " if pcr is None else ("v" if float(pcr) >= 1.5 else ("-" if float(pcr) >= 1.0 else "^"))

        print(
            f"  {grupo:<35s}  {str(v):<12s}  {dte_tag:>4s}  {n:>3d}  "
            f"{_fmt_vol(call_vol):>9s}  {_fmt_vol(put_vol):>9s}  "
            f"{_fmt_vol(total):>9s}  {_fmt_pcr(pcr):>6s} {sesgo}"
        )

    print("  " + sep)
    total_call = sum(r[2] or 0 for r in rows)
    total_put  = sum(r[3] or 0 for r in rows)
    total_all  = total_call + total_put
    pcr_agg    = round(total_put / total_call, 2) if total_call > 0 else None
    print(
        f"  {'TOTAL':<35s}  {'':12s}  {'':>4s}  {'':>3s}  "
        f"{_fmt_vol(total_call):>9s}  {_fmt_vol(total_put):>9s}  "
        f"{_fmt_vol(total_all):>9s}  {_fmt_pcr(pcr_agg):>6s}"
    )
    print()
    print(f"  Filas mostradas: {len(rows)}")
    print(f"  Leyenda DTE: 'w' = weekly (<=14 dias)  |  PCR: ^ <1.0  - 1.0-1.5  v >=1.5")
    print()


# ── Analisis por Ticker: helpers ─────────────────────────────────────────────

ATM_PCT_DEFAULT = 3.0   # +/- 3% del precio subyacente define zona ATM

# Filtro de plausibilidad para IV de yfinance.
# yfinance devuelve valores erroneos (0%, 0.1%) o absurdos (>300%) con frecuencia.
# Solo usamos IV dentro de este rango para calculos ponderados.
IV_MIN_PLAUSIBLE = 0.05   # 5% anualizado (minimo realista para cualquier opcion activa)
IV_MAX_PLAUSIBLE = 2.00   # 200% anualizado (maximo realista para large/mid cap)

BUCKETS_DTE = ["<=30d", "31-60d", "61-90d"]
BUCKETS_MON = ["ITM", "ATM", "OTM"]
LABEL_DTE   = {"<=30d": "<=30 dias", "31-60d": "31-60 dias", "61-90d": "61-90 dias"}


def _moneyness(dist_pct: float, tipo: str, atm_pct: float) -> str:
    """
    Clasifica un contrato en ITM / ATM / OTM segun distancia % al precio actual.
    La logica es inversa para puts vs calls:
      Call ITM: strike < precio  (dist_pct < 0)
      Put  ITM: strike > precio  (dist_pct > 0)
    """
    if abs(dist_pct) <= atm_pct:
        return "ATM"
    if tipo == "call":
        return "ITM" if dist_pct < 0 else "OTM"
    else:
        return "ITM" if dist_pct > 0 else "OTM"


def _dte_bucket(dte: int) -> str:
    if dte <= 30:
        return "<=30d"
    if dte <= 60:
        return "31-60d"
    return "61-90d"


def _expected_move(precio_sub: float, iv_atm: float, avg_dte: float):
    """
    Movimiento esperado derivado de las opciones ATM:
      EM = precio * IV_ATM * sqrt(DTE / 365)
    Retorna (em_abs, em_pct_100) o (None, None) si faltan datos.
    """
    if not (precio_sub and iv_atm and avg_dte and avg_dte > 0):
        return None, None
    em_pct = iv_atm * math.sqrt(avg_dte / 365.0)
    return precio_sub * em_pct, em_pct * 100.0


def _sesgo(pcr_vol, otm_c_pct, otm_p_pct, iv_skew_atm):
    """
    Combina tres senales para producir un sesgo direccional:
      1. PCR_vol  : < 0.7 alcista / > 1.3 bajista
      2. OTM ratio: calls OTM% / puts OTM%  > 1.2 alcista / < 0.8 bajista
      3. IV Skew  : put_iv / call_iv  < 1.0 alcista / > 1.15 bajista

    Retorna (label, fuerza) donde label = ALCISTA / BAJISTA / LATERAL
    y fuerza = FUERTE / MODERADO / '' (lateral).
    """
    signals = []

    if pcr_vol is not None:
        if pcr_vol < 0.7:
            signals.append(1)
        elif pcr_vol > 1.3:
            signals.append(-1)
        else:
            signals.append(0)

    if otm_c_pct and otm_p_pct and otm_p_pct > 0:
        ratio = otm_c_pct / otm_p_pct
        if ratio > 1.2:
            signals.append(1)
        elif ratio < 0.8:
            signals.append(-1)
        else:
            signals.append(0)

    if iv_skew_atm is not None:
        if iv_skew_atm < 1.0:
            signals.append(1)
        elif iv_skew_atm > 1.15:
            signals.append(-1)
        else:
            signals.append(0)

    if not signals:
        return "N/D", ""

    score = sum(signals)
    n     = len(signals)

    if score == n:
        return "ALCISTA", "FUERTE"
    if score >= 1:
        return "ALCISTA", "MODERADO"
    if score == -n:
        return "BAJISTA", "FUERTE"
    if score <= -1:
        return "BAJISTA", "MODERADO"
    return "LATERAL", ""


def _iv_plausible(iv) -> bool:
    """True si IV esta dentro del rango plausible para yfinance (5%-300%)."""
    return iv is not None and IV_MIN_PLAUSIBLE <= iv <= IV_MAX_PLAUSIBLE


def _cell_nuevo():
    # iv_oi_weight: suma de OI solo de contratos con IV plausible (denominador real del promedio)
    # n_iv_plaus: contratos con IV plausible (para calcular cobertura)
    return {"vol": 0, "oi": 0, "iv_oi_sum": 0.0, "iv_oi_weight": 0,
            "dte_sum": 0.0, "dte_n": 0, "n": 0, "n_iv_plaus": 0}


# ── Nivel 0 + 1: Analisis macro por ticker ───────────────────────────────────

def cmd_analisis_ticker(
    ticker: str,
    fecha: Optional[date] = None,
    atm_pct: float = ATM_PCT_DEFAULT,
):
    """
    Nivel 0: header con metricas globales del ticker (PCR, IV skew, sesgo neto).
    Nivel 1: grid DTE-bucket x moneyness, separado por calls y puts.
             Muestra VOL%, OI%, IV ponderado por OI, ratio IV/HV.
             Al pie de cada bucket: movimiento esperado y sesgo.
    """
    engine  = get_engine()
    ticker  = ticker.upper()

    # Determinar fecha si no se indica
    if fecha is None:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT MAX(fecha_snapshot) FROM opciones_snapshot WHERE ticker = :t"
            ), {"t": ticker}).fetchone()
        fecha = row[0] if row else None
        if fecha is None:
            log(f"Sin datos para {ticker}.")
            return

    # Contratos del dia
    with engine.connect() as conn:
        raw = conn.execute(text("""
            SELECT
                tipo,
                strike,
                COALESCE(volumen, 0)        AS vol,
                COALESCE(open_interest, 0)  AS oi,
                iv,
                precio_subyacente,
                hv_20d,
                (vencimiento - CAST(:fecha AS date)) AS dte
            FROM opciones_snapshot
            WHERE ticker            = CAST(:ticker AS varchar)
              AND fecha_snapshot    = CAST(:fecha  AS date)
              AND (vencimiento - CAST(:fecha AS date)) > 0
        """), {"ticker": ticker, "fecha": fecha}).fetchall()

    if not raw:
        log(f"Sin contratos en opciones_snapshot para {ticker} en {fecha}.")
        return

    # Resumen diario (header)
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT pcr_vol, pcr_oi, precio_sub, max_oi_strike, max_oi_venc
            FROM opciones_resumen_diario
            WHERE ticker = CAST(:ticker AS varchar)
              AND fecha  = CAST(:fecha  AS date)
        """), {"ticker": ticker, "fecha": fecha}).fetchone()

    precio_sub = float(raw[0][5]) if raw[0][5] else None
    hv_20d     = float(raw[0][6]) if raw[0][6] else None

    # Agregar por (tipo, dte_bucket, moneyness)
    # agg[(tipo, dte_b, mon)] = {vol, oi, iv_oi_sum, dte_sum, dte_n, n}
    agg = {}

    for row in raw:
        tipo   = row[0]
        strike = float(row[1])
        vol    = int(row[2])
        oi     = int(row[3])
        iv     = float(row[4]) if row[4] is not None else None
        dte    = int(row[7])   if row[7] is not None else None

        if dte is None or dte <= 0:
            continue
        if precio_sub is None or precio_sub <= 0:
            continue

        dist_pct = (strike - precio_sub) / precio_sub * 100
        dte_b    = _dte_bucket(dte)
        mon      = _moneyness(dist_pct, tipo, atm_pct)
        key      = (tipo, dte_b, mon)

        if key not in agg:
            agg[key] = _cell_nuevo()
        c = agg[key]
        c["vol"]           += vol
        c["oi"]            += oi
        c["dte_sum"]       += dte
        c["dte_n"]         += 1
        c["n"]             += 1
        if _iv_plausible(iv) and oi > 0:
            c["iv_oi_sum"]    += iv * oi
            c["iv_oi_weight"] += oi
            c["n_iv_plaus"]   += 1

    # Totales por (tipo, dte_bucket) para calcular porcentajes
    totals = {}
    for (tipo, db, mon), c in agg.items():
        k = (tipo, db)
        if k not in totals:
            totals[k] = {"vol": 0, "oi": 0}
        totals[k]["vol"] += c["vol"]
        totals[k]["oi"]  += c["oi"]

    # ── Nivel 0 — Header ──────────────────────────────────────────────────────

    pcr_vol        = float(res[0]) if res and res[0] else None
    pcr_oi         = float(res[1]) if res and res[1] else None
    max_oi_strike  = float(res[3]) if res and res[3] else None
    max_oi_venc    = res[4]        if res            else None

    # IV skew ATM global: avg ponderado de todos los buckets DTE
    def _iv_w(tipo_f, mon_f):
        iv_sum, oi_w = 0.0, 0
        for db in BUCKETS_DTE:
            c = agg.get((tipo_f, db, mon_f), {})
            iv_sum += c.get("iv_oi_sum",    0.0)
            oi_w   += c.get("iv_oi_weight", 0)
        return iv_sum / oi_w if oi_w > 0 else None

    iv_c_atm = _iv_w("call", "ATM")
    iv_p_atm = _iv_w("put",  "ATM")
    iv_skew_atm = (iv_p_atm / iv_c_atm) if (iv_c_atm and iv_p_atm and iv_c_atm > 0) else None

    # OTM ratio global (usando primer bucket DTE con datos)
    otm_c_pct_g = otm_p_pct_g = None
    for db in BUCKETS_DTE:
        t_c = totals.get(("call", db), {}).get("vol", 0)
        t_p = totals.get(("put",  db), {}).get("vol", 0)
        if t_c > 0 and t_p > 0:
            otm_c_pct_g = agg.get(("call", db, "OTM"), {}).get("vol", 0) / t_c * 100
            otm_p_pct_g = agg.get(("put",  db, "OTM"), {}).get("vol", 0) / t_p * 100
            break

    sesgo_label, sesgo_fuerza = _sesgo(pcr_vol, otm_c_pct_g, otm_p_pct_g, iv_skew_atm)

    SEP_D = "=" * 78
    SEP_S = "-" * 78

    print()
    print(SEP_D)
    hv_str  = f"{hv_20d * 100:.1f}%" if hv_20d else "N/D"
    pr_str  = f"${precio_sub:.2f}"   if precio_sub else "N/D"
    print(f"  ANALISIS: {ticker:<8s} | Precio: {pr_str:<10s} | {fecha}  | HV_20d: {hv_str}")
    print(SEP_D)

    pv_f = ("^" if pcr_vol and pcr_vol < 1.0 else ("v" if pcr_vol and pcr_vol > 1.3 else "-")) if pcr_vol else " "
    print(
        f"  PCR_vol: {f'{pcr_vol:.2f}' if pcr_vol else 'N/D'} {pv_f}"
        f"  |  PCR_oi: {f'{pcr_oi:.2f}' if pcr_oi else 'N/D'}"
        f"  |  IV Skew ATM (put/call): {f'{iv_skew_atm:.2f}' if iv_skew_atm else 'N/D'}"
    )
    ses_str = f"{sesgo_label} {sesgo_fuerza}".strip()
    oi_str  = (f"  |  Max OI Strike: ${max_oi_strike:.2f} (venc {max_oi_venc})"
               if max_oi_strike else "")
    print(f"  Sesgo neto: {ses_str}{oi_str}")
    print(SEP_S)

    # ── Nivel 1 — Grid por DTE bucket ────────────────────────────────────────

    HDR_COLS = (
        f"  {'':8s} | "
        f"{'VOL%':>5s} {'OI%':>5s} {'IV_w':>6s} {'IV/HV':>5s} {'N':>4s} | "
        f"{'VOL%':>5s} {'OI%':>5s} {'IV_w':>6s} {'IV/HV':>5s} {'N':>4s}"
    )
    HDR_TIPOS = (
        f"  {'':8s} | "
        f"{'--- CALLS ---':^31s} | "
        f"{'--- PUTS ---':^31s}"
    )

    for db in BUCKETS_DTE:
        # Verificar si hay datos en este bucket
        has_data = any(
            agg.get((t, db, m), {}).get("vol", 0) > 0 or
            agg.get((t, db, m), {}).get("oi",  0) > 0
            for t in ("call", "put") for m in BUCKETS_MON
        )
        if not has_data:
            continue

        print(f"\n  DTE: {LABEL_DTE[db]}")
        print(HDR_TIPOS)
        print(HDR_COLS)
        print("  " + SEP_S)

        for mon in BUCKETS_MON:
            partes = []
            for tipo in ("call", "put"):
                c     = agg.get((tipo, db, mon), _cell_nuevo())
                t_vol = totals.get((tipo, db), {}).get("vol", 0)
                t_oi  = totals.get((tipo, db), {}).get("oi",  0)

                vol_pct = c["vol"] / t_vol * 100 if t_vol > 0 else 0.0
                oi_pct  = c["oi"]  / t_oi  * 100 if t_oi  > 0 else 0.0
                iv_w    = (c["iv_oi_sum"] / c["iv_oi_weight"]
                           if c["iv_oi_weight"] > 0 else None)
                iv_hv   = (iv_w / hv_20d) if (iv_w and hv_20d) else None

                # Indicador de cobertura IV (% contratos con IV plausible)
                iv_cov = c["n_iv_plaus"] / c["n"] * 100 if c["n"] > 0 else 0.0
                low_cov = iv_cov < 30 and c["n"] > 0  # aviso si cobertura < 30%

                if iv_w:
                    iv_str = f"{iv_w * 100:.1f}%" + ("!" if low_cov else " ")
                else:
                    iv_str = "  N/D "
                hv_str2 = f"{iv_hv:.2f}x" if iv_hv else "  N/D"

                partes.append(
                    f"{vol_pct:>4.1f}% {oi_pct:>4.1f}% {iv_str:>6s} {hv_str2:>5s} {c['n']:>4d}"
                )

            print(f"  {mon:<8s} | {partes[0]} | {partes[1]}")

        print("  " + SEP_S)

        # Metricas del bucket: PCR, IV skew, movimiento esperado, sesgo
        t_c_vol = totals.get(("call", db), {}).get("vol", 0)
        t_p_vol = totals.get(("put",  db), {}).get("vol", 0)
        pcr_db  = round(t_p_vol / t_c_vol, 2) if t_c_vol > 0 else None
        pcr_db_f = ("^" if pcr_db and pcr_db < 1.0 else ("v" if pcr_db and pcr_db > 1.3 else "-")) if pcr_db else " "

        # IV skew ATM del bucket (ponderado por OI, solo contratos con IV valido)
        c_atm_db  = agg.get(("call", db, "ATM"), _cell_nuevo())
        p_atm_db  = agg.get(("put",  db, "ATM"), _cell_nuevo())
        iv_c_db   = (c_atm_db["iv_oi_sum"] / c_atm_db["iv_oi_weight"]
                     if c_atm_db["iv_oi_weight"] > 0 else None)
        iv_p_db   = (p_atm_db["iv_oi_sum"] / p_atm_db["iv_oi_weight"]
                     if p_atm_db["iv_oi_weight"] > 0 else None)
        sk_db     = (iv_p_db / iv_c_db) if (iv_c_db and iv_p_db and iv_c_db > 0) else None

        # OTM ratio del bucket
        c_otm_pct = (agg.get(("call", db, "OTM"), {}).get("vol", 0) / t_c_vol * 100) if t_c_vol > 0 else 0
        p_otm_pct = (agg.get(("put",  db, "OTM"), {}).get("vol", 0) / t_p_vol * 100) if t_p_vol > 0 else 0

        sesgo_db, sf_db = _sesgo(pcr_db, c_otm_pct, p_otm_pct, sk_db)

        # Movimiento esperado (ATM calls, DTE promedio del bucket)
        avg_dte_db = (c_atm_db["dte_sum"] / c_atm_db["dte_n"]) if c_atm_db["dte_n"] > 0 else None
        em_abs, em_pct2 = _expected_move(precio_sub, iv_c_db, avg_dte_db)

        pcr_s  = f"{pcr_db:.2f} {pcr_db_f}" if pcr_db  else "N/D"
        sk_s   = f"{sk_db:.2f}"              if sk_db   else "N/D"
        em_s   = f"+/-${em_abs:.2f} (+/-{em_pct2:.1f}%)" if em_abs else "N/D"
        ses_db = f"{sesgo_db} {sf_db}".strip()

        # Cobertura IV del bucket ATM (calls + puts)
        n_atm_c = c_atm_db["n"]
        n_atm_p = p_atm_db["n"]
        cov_c   = (c_atm_db["n_iv_plaus"] / n_atm_c * 100) if n_atm_c > 0 else 0.0
        cov_p   = (p_atm_db["n_iv_plaus"] / n_atm_p * 100) if n_atm_p > 0 else 0.0
        cov_s   = f"(IV cov: C={cov_c:.0f}% P={cov_p:.0f}%)"

        print(f"  PCR_vol: {pcr_s}  |  IV Skew ATM: {sk_s}  {cov_s}  |  Mov.Esp.: {em_s}")
        print(f"  Sesgo {LABEL_DTE[db]}: {ses_db}")

    print()
    print(f"  Umbral ATM: +/-{atm_pct:.0f}%  |  Contratos en snapshot: {len(raw)}")
    print(f"  IV: ponderado por OI, filtro 5%-200% (yfinance)  |  '!' = cobertura IV < 30%")
    print()


# ── Nivel 2: Detalle por strike ───────────────────────────────────────────────

def cmd_drill_ticker(
    ticker: str,
    fecha: Optional[date] = None,
    dte_max: int = 30,
    tipo: str = "call",
    atm_pct: float = ATM_PCT_DEFAULT,
    top_n: int = 15,
):
    """
    Nivel 2: tabla de strikes individuales para un ticker / tipo / DTE <= N dias.
    Ordena por volumen dentro de cada bucket de moneyness.
    Muestra: strike, dist%, prima, IV, VOL, OI, IV/HV, V/OI.
    """
    engine = get_engine()
    ticker = ticker.upper()
    tipo   = tipo.lower()

    if fecha is None:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT MAX(fecha_snapshot) FROM opciones_snapshot WHERE ticker = :t"
            ), {"t": ticker}).fetchone()
        fecha = row[0] if row else None
        if fecha is None:
            log(f"Sin datos para {ticker}.")
            return

    with engine.connect() as conn:
        raw = conn.execute(text("""
            SELECT
                vencimiento,
                strike,
                COALESCE(volumen, 0)        AS vol,
                COALESCE(open_interest, 0)  AS oi,
                iv,
                COALESCE(bid, 0)            AS bid,
                COALESCE(ask, 0)            AS ask,
                precio_subyacente,
                hv_20d,
                (vencimiento - CAST(:fecha AS date)) AS dte
            FROM opciones_snapshot
            WHERE ticker            = CAST(:ticker AS varchar)
              AND fecha_snapshot    = CAST(:fecha  AS date)
              AND tipo              = CAST(:tipo   AS varchar)
              AND (vencimiento - CAST(:fecha AS date)) BETWEEN 1 AND :dte_max
            ORDER BY vencimiento ASC, strike ASC
        """), {
            "ticker": ticker,
            "fecha":  fecha,
            "tipo":   tipo,
            "dte_max": dte_max,
        }).fetchall()

    if not raw:
        log(f"Sin datos: {ticker} / {tipo.upper()} / DTE<={dte_max}d en {fecha}.")
        return

    precio_sub = float(raw[0][7]) if raw[0][7] else None
    hv_20d     = float(raw[0][8]) if raw[0][8] else None

    # Enriquecer con moneyness y metricas derivadas
    enriched = []
    for r in raw:
        strike   = float(r[1])
        vol, oi  = int(r[2]), int(r[3])
        iv_raw   = float(r[4]) if r[4] is not None else None
        iv       = iv_raw if _iv_plausible(iv_raw) else None   # filtro plausibilidad
        bid, ask = float(r[5]), float(r[6])
        dte      = int(r[9])
        prima    = (bid + ask) / 2.0
        dist_pct = (strike - precio_sub) / precio_sub * 100 if precio_sub else 0.0
        mon      = _moneyness(dist_pct, tipo, atm_pct)
        iv_hv    = (iv / hv_20d) if (iv and hv_20d) else None
        v_oi     = (vol / oi)    if oi > 0           else None
        enriched.append({
            "venc": r[0], "strike": strike, "vol": vol, "oi": oi,
            "iv": iv, "prima": prima, "dist_pct": dist_pct,
            "mon": mon, "dte": dte, "iv_hv": iv_hv, "v_oi": v_oi,
        })

    # Top N por volumen dentro de cada bucket, manteniendo orden venc+strike
    if top_n > 0:
        by_mon = defaultdict(list)
        for e in enriched:
            by_mon[e["mon"]].append(e)
        kept = []
        for mon in BUCKETS_MON:
            bucket = sorted(by_mon[mon], key=lambda x: x["vol"], reverse=True)[:top_n]
            kept.extend(bucket)
        enriched = sorted(kept, key=lambda x: (x["dte"], x["strike"]))

    # Agrupar por vencimiento para mostrar separadores por expiry
    enriched_by_venc = {}
    for e in enriched:
        venc_key = (e["dte"], e["venc"])
        if venc_key not in enriched_by_venc:
            enriched_by_venc[venc_key] = []
        enriched_by_venc[venc_key].append(e)

    # Print
    SEP   = "-" * 80
    SEP_V = "=" * 80
    tipo_label = "CALLS" if tipo == "call" else "PUTS"
    print()
    print(f"  DETALLE: {ticker}  |  {tipo_label}  |  DTE<={dte_max}d  |  {fecha}")
    if precio_sub:
        hv_label = f"  |  HV_20d: {hv_20d * 100:.1f}%" if hv_20d else ""
        print(f"  Precio: ${precio_sub:.2f}{hv_label}")
    print()
    print(
        f"  {'Mon':<5s}  {'Venc':<12s}  {'DTE':>3s}  {'Strike':>8s}  "
        f"{'Dist%':>6s}  {'Prima':>7s}  {'IV':>6s}  "
        f"{'VOL':>8s}  {'OI':>8s}  {'IV/HV':>6s}  {'V/OI':>5s}"
    )

    mon_totals = defaultdict(lambda: {"vol": 0, "oi": 0})
    t_vol_all  = 0
    t_oi_all   = 0

    for (dte_v, venc_v), group in sorted(enriched_by_venc.items()):
        # Separador por vencimiento
        venc_vol = sum(e["vol"] for e in group)
        venc_oi  = sum(e["oi"]  for e in group)
        pcr_v    = None
        if tipo == "call":
            pcr_label = ""  # el drill es mono-tipo, no hay PCR por contrato
        print("  " + SEP)
        print(f"  -- {str(venc_v)}  DTE={dte_v}  VOL={_fmt_vol(venc_vol)}  OI={_fmt_vol(venc_oi)}")
        print("  " + SEP)

        # Ordenar dentro del vencimiento por moneyness_order luego strike
        mon_order = {"ITM": 0, "ATM": 1, "OTM": 2}
        group_sorted = sorted(group, key=lambda x: (mon_order.get(x["mon"], 9), x["strike"]))

        for e in group_sorted:
            mon = e["mon"]
            mon_totals[mon]["vol"] += e["vol"]
            mon_totals[mon]["oi"]  += e["oi"]
            t_vol_all += e["vol"]
            t_oi_all  += e["oi"]

            iv_s  = f"{e['iv']*100:.1f}%"  if e["iv"]    else "   N/D"
            hv_s  = f"{e['iv_hv']:.1f}x"  if e["iv_hv"] else "   N/D"
            voi_s = f"{e['v_oi']:.2f}"     if e["v_oi"]  else "  N/D"

            print(
                f"  {mon:<5s}  {str(e['venc']):<12s}  {e['dte']:>3d}  {e['strike']:>8.2f}  "
                f"{e['dist_pct']:>+6.1f}%  ${e['prima']:>6.2f}  {iv_s:>6s}  "
                f"{_fmt_vol(e['vol']):>8s}  {_fmt_vol(e['oi']):>8s}  {hv_s:>6s}  {voi_s:>5s}"
            )

    # Totales por moneyness + grand total
    print("  " + SEP)
    for mon in BUCKETS_MON:
        mt = mon_totals[mon]
        if mt["vol"] > 0 or mt["oi"] > 0:
            print(
                f"  {'  ' + mon + ' total':<34s}"
                f"  {_fmt_vol(mt['vol']):>8s}  {_fmt_vol(mt['oi']):>8s}"
            )
    print("  " + SEP)
    print(
        f"  {'TOTAL':<34s}"
        f"  {_fmt_vol(t_vol_all):>8s}  {_fmt_vol(t_oi_all):>8s}"
    )
    print()
    print(
        f"  Umbral ATM: +/-{atm_pct:.0f}%  |  Strikes: {len(enriched)}"
        f"  |  IV: filtro 5%-200%  |  OI=dia anterior (T-1)  |  V/OI=vol_T/oi_(T-1), >0.3=flujo nuevo"
    )
    print()


def cmd_vol_sector(
    fecha: Optional[date] = None,
    sector: Optional[str] = None,
    venc: Optional[date] = None,
    top: int = 0,
):
    _cmd_vol_grupo("sector", sector, fecha, venc, top)


def cmd_vol_industria(
    fecha: Optional[date] = None,
    industria: Optional[str] = None,
    venc: Optional[date] = None,
    top: int = 0,
):
    _cmd_vol_grupo("industry", industria, fecha, venc, top)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analisis de sentimiento de opciones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Parametros comunes
    parser.add_argument(
        "--fecha",
        help="Fecha snapshot YYYY-MM-DD (default: ultima disponible en DB)",
        default=None,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Limitar a top N filas por volumen total (0 = todas)",
    )
    parser.add_argument(
        "--ticker",
        help="Filtrar por ticker especifico (ej: BAC)",
        default=None,
    )
    parser.add_argument(
        "--venc",
        help="Filtrar por fecha de vencimiento YYYY-MM-DD (ej: 2026-05-16)",
        default=None,
    )

    # Analisis disponibles
    parser.add_argument(
        "--vol-ticker",
        action="store_true",
        help="L1: Ranking de tickers por volumen call/put del dia",
    )
    parser.add_argument(
        "--vol-vencimiento",
        action="store_true",
        help="L2: Volumen call/put por ticker + vencimiento",
    )
    parser.add_argument(
        "--vol-strike",
        action="store_true",
        help="L3: Volumen call/put por ticker + vencimiento + strike (requiere --ticker)",
    )
    parser.add_argument(
        "--vol-sector",
        action="store_true",
        help="L1/L2: Volumen agregado por sector",
    )
    parser.add_argument(
        "--vol-industria",
        action="store_true",
        help="L1/L2: Volumen agregado por industria",
    )
    parser.add_argument(
        "--sector",
        help="Filtrar por sector (ej: Technology). Usar con --vol-sector",
        default=None,
    )
    parser.add_argument(
        "--industria",
        help="Filtrar por industria (ej: Semiconductors). Usar con --vol-industria",
        default=None,
    )

    # Analisis ticker individual (macro + drill)
    parser.add_argument(
        "--analisis",
        action="store_true",
        help="Nivel 0+1: analisis macro del ticker (requiere --ticker)",
    )
    parser.add_argument(
        "--drill",
        action="store_true",
        help="Nivel 2: detalle de strikes (requiere --ticker y --tipo)",
    )
    parser.add_argument(
        "--tipo",
        choices=["call", "put"],
        default="call",
        help="Tipo de opcion para --drill: call (default) o put",
    )
    parser.add_argument(
        "--dte",
        type=int,
        default=30,
        help="DTE maximo para --drill (default: 30 dias)",
    )
    parser.add_argument(
        "--atm-pct",
        type=float,
        default=ATM_PCT_DEFAULT,
        dest="atm_pct",
        help=f"Umbral porcentual para zona ATM (default: +/-{ATM_PCT_DEFAULT:.0f}%%)",
    )

    args = parser.parse_args()

    fecha = date.fromisoformat(args.fecha) if args.fecha else None
    venc  = date.fromisoformat(args.venc)  if args.venc  else None

    if args.vol_ticker:
        cmd_vol_ticker(fecha=fecha, top=args.top)
        return

    if args.vol_vencimiento:
        cmd_vol_vencimiento(fecha=fecha, ticker=args.ticker, venc=venc, top=args.top)
        return

    if args.vol_strike:
        cmd_vol_strike(fecha=fecha, ticker=args.ticker, venc=venc, top=args.top)
        return

    if args.vol_sector:
        cmd_vol_sector(fecha=fecha, sector=args.sector, venc=venc, top=args.top)
        return

    if args.vol_industria:
        cmd_vol_industria(fecha=fecha, industria=args.industria, venc=venc, top=args.top)
        return

    if args.analisis:
        if not args.ticker:
            print("  Error: --analisis requiere --ticker.")
            return
        cmd_analisis_ticker(ticker=args.ticker, fecha=fecha, atm_pct=args.atm_pct)
        return

    if args.drill:
        if not args.ticker:
            print("  Error: --drill requiere --ticker.")
            return
        cmd_drill_ticker(
            ticker=args.ticker,
            fecha=fecha,
            dte_max=args.dte,
            tipo=args.tipo,
            atm_pct=args.atm_pct,
            top_n=args.top if args.top > 0 else 15,
        )
        return

    # Sin flag -> mostrar ayuda
    parser.print_help()


if __name__ == "__main__":
    main()
