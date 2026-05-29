"""
30_resumen_semanal_1w.py
Resumen ejecutivo de la ultima semana completa para los 124 tickers.

Columnas por ticker:
    ticker, sector, fecha, n_dias, close
    vela        : ALCISTA/BAJISTA + body_pct
    patron      : figuras detectadas (HAMMER, ENGULF_BULL, etc.)
    vol         : VOL_SPIKE o vol_ratio_5d
    estructura  : estructura_10 (ALG/NEU/BAJ)
    rsi14       : valor + flag OB/OS
    macd_dir    : direccion del histograma MACD (+ / - / CRUCE)
    sma50_pct   : distancia % al SMA50
    sma200_pct  : distancia % al SMA200
    mejor_ee_es : mejor estrategia por PF en operaciones historicas
    pf          : profit factor de esa estrategia
    wr_pct      : win rate
    g_p         : ganancias / perdidas

Salida:
    - Tabla en consola (compacta)
    - CSV en /data/resumen_semanal_1w_FECHA.csv (si existe la carpeta)

Uso:
    python scripts/30_resumen_semanal_1w.py
    python scripts/30_resumen_semanal_1w.py --semana 2026-02-14   <- semana especifica
    python scripts/30_resumen_semanal_1w.py --sector Financials
    python scripts/30_resumen_semanal_1w.py --con-patron           <- solo tickers con patron
    python scripts/30_resumen_semanal_1w.py --min-ops 5            <- min ops para best strategy
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import query_df


# ─────────────────────────────────────────────────────────────
# Abreviaciones de sector
# ─────────────────────────────────────────────────────────────

SECTOR_SHORT = {
    "Financials":              "FIN",
    "Consumer Staples":        "STAP",
    "Consumer Discretionary":  "DISC",
    "Consumer Cyclical":       "CYCL",
    "Communication Services":  "COMM",
    "Technology":              "TECH",
    "Healthcare":              "HLTH",
    "Energy":                  "ENRG",
    "Industrials":             "IND",
    "Materials":               "MAT",
    "Real Estate":             "REAL",
    "Airlines":                "AIR",
    "Automotive":              "AUTO",
    "Mining":                  "MINE",
    "Brazil":                  "BRA",
    "China/SE Asia":           "ASIA",
}


def sector_abbr(s: str) -> str:
    if not s or str(s) == "None":
        return "----"
    return SECTOR_SHORT.get(s, s[:4].upper())


# ─────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────

def obtener_ultima_semana() -> str:
    """Retorna la fecha_semana mas reciente disponible en indicadores_tecnicos_1w."""
    r = query_df("SELECT MAX(fecha) AS ult FROM indicadores_tecnicos_1w")
    if r.empty or r.iloc[0]["ult"] is None:
        raise RuntimeError("indicadores_tecnicos_1w esta vacio.")
    return str(r.iloc[0]["ult"])


def cargar_resumen(fecha_semana: str, min_ops: int = 1) -> "pd.DataFrame":
    """
    Carga todos los datos necesarios para el resumen semanal en un solo query.

    Joins:
        precios_semanales           -> OHLCV + n_dias
        indicadores_tecnicos_1w     -> RSI, MACD, SMA50/200 + distancias
        features_precio_accion_1w   -> patron, body_pct, vol_spike
        features_market_structure_1w -> estructura_10
        activos                     -> sector
        resultados_bt_pa_1w (CTE)   -> mejor estrategia por ticker
    """
    sql = """
        WITH mejor_estrategia AS (
            SELECT DISTINCT ON (ticker)
                ticker,
                estrategia_entrada,
                estrategia_salida,
                total_operaciones,
                ganancias,
                perdidas,
                ROUND(win_rate * 100, 1)           AS wr_pct,
                ROUND(retorno_promedio_pct, 2)      AS ret_prom,
                ROUND(profit_factor::NUMERIC, 2)   AS pf
            FROM resultados_bt_pa_1w
            WHERE ticker IS NOT NULL
              AND total_operaciones >= :min_ops
            ORDER BY ticker, profit_factor DESC
        )
        SELECT
            -- Identificacion
            p.ticker,
            COALESCE(a.sector, 'N/D')              AS sector,
            p.fecha_semana                          AS fecha,
            p.n_dias,

            -- OHLCV
            p.open,
            p.high,
            p.low,
            p.close,
            p.volume,

            -- Indicadores tecnicos
            i.rsi14,
            i.macd,
            i.macd_signal,
            i.macd_hist,
            i.sma50,
            i.sma200,
            i.dist_sma50,
            i.dist_sma200,
            i.atr14,

            -- Features precio/accion
            pa.es_alcista,
            pa.body_pct,
            pa.body_ratio,
            pa.upper_shadow_pct,
            pa.lower_shadow_pct,
            pa.rango_diario_pct,
            pa.rango_rel_atr,
            pa.clv,
            pa.vol_spike,
            pa.vol_ratio_5d,
            pa.up_vol_5d,
            pa.patron_doji,
            pa.patron_hammer,
            pa.patron_engulfing_bull,
            pa.patron_shooting_star,
            pa.patron_marubozu,
            pa.patron_engulfing_bear,
            pa.inside_bar,
            pa.outside_bar,
            pa.vol_price_confirm,
            pa.vol_price_diverge,
            pa.dist_max_20d,
            pa.dist_min_20d,
            pa.pos_rango_20d,
            pa.tendencia_velas,

            -- Market structure
            ms.estructura_5,
            ms.estructura_10,
            ms.dist_sh_10_pct,
            ms.dist_sl_10_pct,
            ms.dias_sh_10,
            ms.dias_sl_10,
            ms.bos_bull_10,
            ms.bos_bear_10,
            ms.choch_bull_10,
            ms.choch_bear_10,

            -- Mejor estrategia historica
            me.estrategia_entrada   AS best_ee,
            me.estrategia_salida    AS best_es,
            me.total_operaciones    AS best_ops,
            me.ganancias            AS best_g,
            me.perdidas             AS best_p,
            me.wr_pct               AS best_wr,
            me.ret_prom             AS best_ret,
            me.pf                   AS best_pf

        FROM precios_semanales p
        JOIN  indicadores_tecnicos_1w          i   ON p.ticker = i.ticker  AND p.fecha_semana = i.fecha
        LEFT JOIN features_precio_accion_1w   pa   ON p.ticker = pa.ticker AND p.fecha_semana = pa.fecha
        LEFT JOIN features_market_structure_1w ms   ON p.ticker = ms.ticker AND p.fecha_semana = ms.fecha
        LEFT JOIN activos                      a    ON p.ticker = a.ticker
        LEFT JOIN mejor_estrategia             me   ON p.ticker = me.ticker
        WHERE p.fecha_semana = :fecha
        ORDER BY COALESCE(a.sector, 'ZZ'), p.ticker
    """
    df = query_df(sql, params={"fecha": fecha_semana, "min_ops": min_ops})
    return df


# ─────────────────────────────────────────────────────────────
# Formateo de columnas
# ─────────────────────────────────────────────────────────────

def _fmt_vela(row) -> str:
    """Estructura de la vela: ALCISTA/BAJISTA + body_pct."""
    es_alc   = row.get("es_alcista")
    body_pct = row.get("body_pct")
    if es_alc is None:
        return "  ----"
    dir_   = "ALC" if int(es_alc) == 1 else "BAJ"
    bp_str = f"{float(body_pct):+.1f}%" if body_pct is not None else "  ?%"
    return f"{dir_} {bp_str}"


def _fmt_patron(row) -> str:
    """Detecta y concatena todas las figuras activas."""
    patrones = []
    if row.get("patron_engulfing_bull") == 1:
        patrones.append("ENGULF+")
    if row.get("patron_hammer") == 1:
        patrones.append("HAMMER")
    if row.get("patron_engulfing_bear") == 1:
        patrones.append("ENGULF-")
    if row.get("patron_shooting_star") == 1:
        patrones.append("SHOOT*")
    if row.get("patron_doji") == 1:
        patrones.append("DOJI")
    if row.get("patron_marubozu") == 1:
        patrones.append("MARUB")
    if row.get("inside_bar") == 1:
        patrones.append("IB")
    if row.get("outside_bar") == 1:
        patrones.append("OB")
    return "+".join(patrones) if patrones else "-"


def _fmt_vol(row) -> str:
    """Indica si hay vol_spike o ratio relativo."""
    if row.get("vol_spike") == 1:
        vr = row.get("vol_ratio_5d")
        if vr is not None:
            return f"SPIKE x{float(vr):.1f}"
        return "SPIKE"
    vr = row.get("vol_ratio_5d")
    if vr is not None:
        return f"x{float(vr):.2f}"
    return "-"


def _fmt_estructura(row) -> str:
    """Estructura de mercado N=10."""
    e10 = row.get("estructura_10")
    e5  = row.get("estructura_5")
    if e10 is None:
        return "----"
    e10v = int(e10)
    e5v  = int(e5) if e5 is not None else 0
    txt10 = "ALG" if e10v == 1 else ("BAJ" if e10v == -1 else "NEU")
    txt5  = "alg" if e5v == 1  else ("baj" if e5v == -1  else "neu")

    # BOS/CHoCH esta semana
    flags = []
    if row.get("bos_bull_10") == 1:
        flags.append("BOS+")
    if row.get("bos_bear_10") == 1:
        flags.append("BOS-")
    if row.get("choch_bull_10") == 1:
        flags.append("CHoCH+")
    if row.get("choch_bear_10") == 1:
        flags.append("CHoCH-")

    base = f"{txt10}/{txt5}"
    return f"{base} {'+'.join(flags)}" if flags else base


def _fmt_rsi(row) -> str:
    """RSI con flag de extremos."""
    rsi = row.get("rsi14")
    if rsi is None:
        return "  --"
    v = float(rsi)
    flag = " OB" if v >= 70 else (" OS" if v <= 30 else "   ")
    return f"{v:5.1f}{flag}"


def _fmt_macd(row) -> str:
    """Indica signo del histograma MACD y si hubo cruce esta semana."""
    hist   = row.get("macd_hist")
    macd   = row.get("macd")
    signal = row.get("macd_signal")
    if hist is None:
        return "  --"
    h = float(hist)
    signo = "+" if h >= 0 else "-"
    # Cruce: macd y signal muy proximos (dentro del 1% del precio)
    if macd is not None and signal is not None:
        gap = abs(float(macd) - float(signal))
        close_p = row.get("close")
        if close_p and float(close_p) > 0:
            gap_pct = gap / float(close_p) * 100
            if gap_pct < 0.3:
                return f"{signo} CRUCE"
    return f"{signo} {abs(h):.3f}"


def _fmt_sma(pct) -> str:
    """Distancia % a SMA."""
    if pct is None:
        return "   --"
    v = float(pct)
    return f"{v:+6.1f}%"


def _fmt_best(row) -> str:
    """Mejor estrategia."""
    ee = row.get("best_ee")
    es = row.get("best_es")
    if ee is None or es is None:
        return "----/----"
    return f"{ee}/{es}"


def _tiene_senal_alcista(row) -> bool:
    """Verdadero si hay algun patron alcista activo."""
    return any([
        row.get("patron_engulfing_bull") == 1,
        row.get("patron_hammer") == 1,
        row.get("bos_bull_10") == 1,
        row.get("choch_bull_10") == 1,
        (row.get("vol_spike") == 1 and int(row.get("es_alcista", 0) or 0) == 1),
    ])


def _tiene_senal_bajista(row) -> bool:
    """Verdadero si hay algun patron bajista activo."""
    return any([
        row.get("patron_engulfing_bear") == 1,
        row.get("patron_shooting_star") == 1,
        row.get("bos_bear_10") == 1,
        row.get("choch_bear_10") == 1,
    ])


# ─────────────────────────────────────────────────────────────
# Impresion de tabla
# ─────────────────────────────────────────────────────────────

HEADER_FMT = (
    f"{'TICK':<6} {'SECT':<5} {'FECHA':<11} {'D':>1} {'CLOSE':>8} "
    f"{'VELA':<11} {'PATRON':<14} {'VOL':<10} "
    f"{'EST10':<12} {'RSI':>8} {'MACD':<10} "
    f"{'SMA50%':>7} {'SMA200%':>8} "
    f"{'BEST EE/ES':<10} {'PF':>6} {'WR%':>6} {'G/P':>6}"
)

ROW_FMT = (
    "{tick:<6} {sect:<5} {fecha:<11} {nd:>1} {close:>8.2f} "
    "{vela:<11} {patron:<14} {vol:<10} "
    "{est:<12} {rsi:<8} {macd:<10} "
    "{sma50:>7} {sma200:>8} "
    "{best:<10} {pf:>6} {wr:>6} {gp:>6}"
)


def imprimir_tabla(df_rows: list, titulo: str):
    """Imprime la tabla formateada en consola."""
    print(f"\n{titulo}")
    print("=" * len(HEADER_FMT))
    print(HEADER_FMT)
    print("-" * len(HEADER_FMT))

    sector_actual = None
    for r in df_rows:
        # Separador por sector
        s = r.get("sector", "")
        if s != sector_actual:
            sector_actual = s
            print(f"  -- {s} --")

        close = r.get("close")
        pf    = r.get("best_pf")
        wr    = r.get("best_wr")
        g     = r.get("best_g")
        p     = r.get("best_p")

        import math
        def _ok(v):
            """Verdadero si el valor es util (no None, no NaN)."""
            if v is None:
                return False
            try:
                return not math.isnan(float(v))
            except (TypeError, ValueError):
                return False

        print(ROW_FMT.format(
            tick   = r.get("ticker", ""),
            sect   = sector_abbr(s),
            fecha  = str(r.get("fecha", ""))[:10],
            nd     = int(r.get("n_dias", 0) or 0),
            close  = float(close) if _ok(close) else 0.0,
            vela   = _fmt_vela(r),
            patron = _fmt_patron(r),
            vol    = _fmt_vol(r),
            est    = _fmt_estructura(r),
            rsi    = _fmt_rsi(r),
            macd   = _fmt_macd(r),
            sma50  = _fmt_sma(r.get("dist_sma50")),
            sma200 = _fmt_sma(r.get("dist_sma200")),
            best   = _fmt_best(r),
            pf     = f"{float(pf):.2f}" if _ok(pf) else "  --",
            wr     = f"{float(wr):.1f}%" if _ok(wr) else "  --",
            gp     = f"{int(float(g))}/{int(float(p))}" if (_ok(g) and _ok(p)) else "--/--",
        ))

    print("=" * len(HEADER_FMT))


# ─────────────────────────────────────────────────────────────
# Seccion de señales destacadas
# ─────────────────────────────────────────────────────────────

def imprimir_senales_destacadas(df_rows: list):
    """Muestra por separado los tickers con senales alcistas y bajistas."""
    alcistas = [r for r in df_rows if _tiene_senal_alcista(r)]
    bajistas = [r for r in df_rows if _tiene_senal_bajista(r)]

    if alcistas:
        print(f"\n  SENALES ALCISTAS ({len(alcistas)} tickers):")
        print(f"  {'Ticker':<7} {'Patron':<20} {'Estructura':<14} {'RSI':>6} {'SMA50%':>7} {'Best':>10} {'PF':>6}")
        print("  " + "-" * 75)
        for r in sorted(alcistas, key=lambda x: str(x.get("sector", ""))):
            pf = r.get("best_pf")
            print(
                f"  {r['ticker']:<7} {_fmt_patron(r):<20} {_fmt_estructura(r):<14} "
                f"{_fmt_rsi(r):>6} {_fmt_sma(r.get('dist_sma50')):>7} "
                f"{_fmt_best(r):>10} {float(pf):>6.2f}" if pf else
                f"  {r['ticker']:<7} {_fmt_patron(r):<20} {_fmt_estructura(r):<14} "
                f"{_fmt_rsi(r):>6} {_fmt_sma(r.get('dist_sma50')):>7} "
                f"{_fmt_best(r):>10} {'--':>6}"
            )

    if bajistas:
        print(f"\n  SENALES BAJISTAS ({len(bajistas)} tickers):")
        print(f"  {'Ticker':<7} {'Patron':<20} {'Estructura':<14} {'RSI':>6} {'SMA50%':>7}")
        print("  " + "-" * 60)
        for r in sorted(bajistas, key=lambda x: str(x.get("sector", ""))):
            print(
                f"  {r['ticker']:<7} {_fmt_patron(r):<20} {_fmt_estructura(r):<14} "
                f"{_fmt_rsi(r):>6} {_fmt_sma(r.get('dist_sma50')):>7}"
            )


# ─────────────────────────────────────────────────────────────
# Exportar a CSV
# ─────────────────────────────────────────────────────────────

def exportar_csv(df, fecha_semana: str):
    """Exporta el DataFrame completo a CSV si existe la carpeta /data."""
    import pandas as pd

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data"
    )
    if not os.path.exists(data_dir):
        print(f"  [INFO] Carpeta /data no existe, saltando exportacion CSV.")
        return

    fname = os.path.join(data_dir, f"resumen_semanal_1w_{fecha_semana}.csv")
    df.to_csv(fname, index=False)
    print(f"  CSV exportado: {fname}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Resumen ejecutivo semanal 1W para los 124 tickers."
    )
    parser.add_argument(
        "--semana", type=str, default=None,
        help="Fecha de la semana (YYYY-MM-DD). Default: ultima disponible."
    )
    parser.add_argument(
        "--sector", type=str, default=None,
        help="Filtrar por sector (substring, case-insensitive)."
    )
    parser.add_argument(
        "--con-patron", action="store_true",
        help="Mostrar solo tickers con algun patron de vela activo."
    )
    parser.add_argument(
        "--min-ops", type=int, default=1,
        help="Minimo de operaciones para mostrar mejor estrategia. Default: 1."
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Exportar resultado a CSV en /data/"
    )
    args = parser.parse_args()

    inicio = datetime.now()

    print()
    print("=" * 70)
    print("  RESUMEN SEMANAL 1W -- DASHBOARD TECNICO")
    print("=" * 70)

    # ── Determinar semana a analizar ──────────────────────────
    if args.semana:
        fecha_semana = args.semana
        print(f"  Semana solicitada : {fecha_semana}")
    else:
        fecha_semana = obtener_ultima_semana()
        print(f"  Ultima semana     : {fecha_semana}")

    print(f"  Min ops best strat: {args.min_ops}")
    print(f"  Filtro sector     : {args.sector or 'Todos'}")
    print(f"  Solo con patron   : {args.con_patron}")

    # ── Cargar datos ──────────────────────────────────────────
    print(f"\n  Cargando datos...")
    df = cargar_resumen(fecha_semana, min_ops=args.min_ops)

    if df.empty:
        print(f"  [ERROR] Sin datos para la semana {fecha_semana}.")
        print(f"  Verifica que existan datos en indicadores_tecnicos_1w para esa fecha.")
        sys.exit(1)

    print(f"  Tickers cargados  : {len(df)}")

    # ── Verificar semanas con feriados ────────────────────────
    n_dias_unicos = df["n_dias"].value_counts().to_dict() if "n_dias" in df.columns else {}
    if any(k is not None and int(k) < 5 for k in n_dias_unicos.keys()):
        semanas_cortas = {k: v for k, v in n_dias_unicos.items() if k is not None and int(k) < 5}
        for dias, cnt in sorted(semanas_cortas.items()):
            print(f"  AVISO: {cnt} ticker(s) con solo {dias} dia(s) habiles esta semana (feriado).")

    # ── Filtros ───────────────────────────────────────────────
    df_rows = df.to_dict(orient="records")

    if args.sector:
        filtro = args.sector.lower()
        df_rows = [r for r in df_rows if filtro in str(r.get("sector", "")).lower()]
        print(f"  Despues filtro sector: {len(df_rows)} tickers")

    if args.con_patron:
        df_rows = [r for r in df_rows
                   if any(r.get(p) == 1 for p in [
                       "patron_hammer", "patron_engulfing_bull",
                       "patron_shooting_star", "patron_engulfing_bear",
                       "patron_doji", "patron_marubozu",
                       "inside_bar", "outside_bar",
                   ])]
        print(f"  Despues filtro patron: {len(df_rows)} tickers")

    if not df_rows:
        print("  Sin tickers con los filtros aplicados.")
        sys.exit(0)

    # ── Tabla principal ───────────────────────────────────────
    titulo = f"  SEMANA: {fecha_semana}  |  {len(df_rows)} tickers"
    imprimir_tabla(df_rows, titulo)

    # ── Senales destacadas ────────────────────────────────────
    print()
    print("=" * 70)
    print("  SENALES DESTACADAS DE LA SEMANA")
    print("=" * 70)
    imprimir_senales_destacadas(df_rows)

    # ── Estadisticas de la semana ─────────────────────────────
    import pandas as pd
    df_f = pd.DataFrame(df_rows)

    print()
    print("=" * 70)
    print("  ESTADISTICAS DE LA SEMANA")
    print("=" * 70)

    n_alc  = int((df_f["es_alcista"] == 1).sum()) if "es_alcista" in df_f else 0
    n_baj  = int((df_f["es_alcista"] == 0).sum()) if "es_alcista" in df_f else 0
    n_pat  = int(df_f[[c for c in df_f.columns if "patron_" in c or c in ["inside_bar", "outside_bar"]]]
                 .apply(lambda col: col == 1).any(axis=1).sum()) if any("patron_" in c for c in df_f.columns) else 0
    n_vol  = int((df_f.get("vol_spike", pd.Series(dtype=int)) == 1).sum())
    n_bos  = int((df_f.get("bos_bull_10", pd.Series(dtype=int)) == 1).sum() +
                 (df_f.get("bos_bear_10", pd.Series(dtype=int)) == 1).sum())
    n_choch= int((df_f.get("choch_bull_10", pd.Series(dtype=int)) == 1).sum() +
                 (df_f.get("choch_bear_10", pd.Series(dtype=int)) == 1).sum())

    rsi_col = df_f["rsi14"].dropna().astype(float) if "rsi14" in df_f else pd.Series(dtype=float)
    n_ob   = int((rsi_col >= 70).sum())
    n_os   = int((rsi_col <= 30).sum())

    sma50_col = df_f["dist_sma50"].dropna().astype(float) if "dist_sma50" in df_f else pd.Series(dtype=float)

    print(f"\n  Velas alcistas    : {n_alc} ({n_alc*100//len(df_rows)}%)")
    print(f"  Velas bajistas    : {n_baj} ({n_baj*100//len(df_rows)}%)")
    print(f"  Con patron activo : {n_pat}")
    print(f"  Vol Spike         : {n_vol}")
    print(f"  BOS esta semana   : {n_bos}")
    print(f"  CHoCH esta semana : {n_choch}")
    print(f"  RSI overbought    : {n_ob} (>= 70)")
    print(f"  RSI oversold      : {n_os} (<= 30)")
    if not rsi_col.empty:
        print(f"  RSI promedio      : {rsi_col.mean():.1f}")
    if not sma50_col.empty:
        print(f"  Dist SMA50 prom   : {sma50_col.mean():+.1f}%")
        n_sobre_sma50 = int((sma50_col > 0).sum())
        print(f"  Sobre SMA50       : {n_sobre_sma50}/{len(sma50_col)} tickers")

    # Sectores con mas senales alcistas
    alc_df = df_f[df_f["es_alcista"] == 1]
    if "sector" in alc_df.columns and not alc_df.empty:
        sec_alc = alc_df["sector"].value_counts().head(3)
        print(f"\n  Top sectores alcistas: ", end="")
        print(" | ".join([f"{s} ({n})" for s, n in sec_alc.items()]))

    # ── Exportar CSV ──────────────────────────────────────────
    if args.csv:
        exportar_csv(df, fecha_semana)

    fin = datetime.now()
    print()
    print("=" * 70)
    print(f"  Generado en {(fin-inicio).seconds}s  |  {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
