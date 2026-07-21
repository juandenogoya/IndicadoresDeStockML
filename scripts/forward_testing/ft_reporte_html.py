"""
ft_reporte_html.py
Genera un reporte HTML autocontenido con el estado de las estrategias de
Forward Testing. Reemplaza el dashboard de Streamlit Cloud por un artefacto
estatico, simple y sin infraestructura.

Caracteristicas:
    - Un unico archivo .html, sin servidor, sin JS externo.
    - Grafico de curvas de equity embebido como PNG (matplotlib + base64).
    - Tablas en HTML/CSS. Cero dependencias nuevas (matplotlib + pandas).
    - Lee la DB LOCAL (via ft_env). Pensado para abrir tras ft_run_diario.bat.
    - Salida por defecto en reportes/ -> se sincroniza via OneDrive.

Uso:
    python scripts/forward_testing/ft_reporte_html.py
    python scripts/forward_testing/ft_reporte_html.py --output ruta/al/reporte.html
"""

import sys
import os
import io
import base64
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Entorno FT: forzar conexion a la DB LOCAL (ver scripts/forward_testing/ft_env.py)
from scripts.forward_testing.ft_env import configurar_entorno_local
configurar_entorno_local()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text
from src.data.database import get_engine
from src.utils.ft_metricas import (
    resumen_riesgo, retornos_desde_equity, metricas_trade, RF_ANUAL_DEFAULT,
)

OUTPUT_DEFAULT = os.path.join(ROOT, "reportes", "ft_reporte.html")
MAX_CERRADAS_TABLA = 20   # operaciones cerradas a mostrar por estrategia

# Ventana comparable: fecha del fix del bug de score=0.0 (ver JOURNAL 2026-05-30).
# Antes de esta fecha las estrategias 4, 6, 8 y 9 cerraban posiciones por un
# score que siempre daba 0 -> su historia previa mide el bug, no la estrategia.
# Es la ventana en la que las 10 son comparables entre si.
VENTANA_COMPARABLE = "2026-05-30"


# ── Helpers de formato ────────────────────────────────────────────────────────

def fmt_usd(v, dec=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"${v:,.{dec}f}"


def fmt_pct(v, dec=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:+.{dec}f}%"


def fmt_pnl(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:+,.2f}"


def clase_signo(v):
    """Clase CSS segun signo (verde/rojo/neutro)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "neutro"
    if v > 0:
        return "pos"
    if v < 0:
        return "neg"
    return "neutro"


def esc(s):
    """Escape minimo para texto en HTML."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ── Queries ───────────────────────────────────────────────────────────────────

def cargar_datos():
    """Carga estrategias, operaciones y metricas diarias desde la DB local."""
    engine = get_engine()
    with engine.connect() as conn:
        estrategias = conn.execute(text("""
            SELECT id, nombre, logica, descripcion,
                   capital_inicial, capital_actual,
                   cash_disponible, capital_inmovilizado, fecha_inicio
            FROM ft_estrategias
            WHERE activa = TRUE
            ORDER BY id
        """)).fetchall()

        operaciones = conn.execute(text("""
            SELECT o.id, o.estrategia_id, o.ticker, o.cantidad,
                   o.capital_entrada, o.fecha_entrada, o.precio_entrada,
                   o.score_entrada, o.fecha_salida, o.precio_salida,
                   o.pnl, o.pnl_pct, o.motivo_salida,
                   -- Duracion sobre fecha_datos: fecha_entrada es la de
                   -- REGISTRO y el sistema es asincronico (ver METRICAS.md).
                   COALESCE(o.fecha_datos_salida, o.fecha_salida)
                     - COALESCE(o.fecha_datos, o.fecha_entrada) AS dias,
                   COALESCE(o.fecha_datos_salida, o.fecha_salida) AS f_salida_datos,
                   p.close AS precio_actual
            FROM ft_operaciones o
            LEFT JOIN (
                SELECT DISTINCT ON (ticker) ticker, close
                FROM precios_diarios
                ORDER BY ticker, fecha DESC
            ) p ON p.ticker = o.ticker AND o.fecha_salida IS NULL
            ORDER BY o.estrategia_id,
                     (o.fecha_salida IS NOT NULL),
                     o.fecha_entrada DESC
        """)).fetchall()

        # Equity MARCADA A MERCADO (ft_equity_diaria), no ft_metricas_diarias:
        # esa esta a COSTO de entrada y solo se mueve al cerrar una operacion
        # -> sobre ella el drawdown de posiciones abiertas es INVISIBLE.
        # Ver docs/forward_testing/METRICAS.md.
        metricas = conn.execute(text("""
            SELECT estrategia_id, fecha, equity, exposicion_pct
            FROM ft_equity_diaria
            ORDER BY estrategia_id, fecha
        """)).fetchall()

        # Benchmark: equiponderado del universo activo. Es el "comprar todo"
        # contra el que se mide si la SELECCION de tickers aporta algo.
        benchmark = conn.execute(text("""
            SELECT p.fecha, AVG(p.close / p0.close) AS idx
            FROM precios_diarios p
            JOIN (
                SELECT DISTINCT ON (ticker) ticker, close
                FROM precios_diarios
                WHERE fecha >= (SELECT MIN(fecha) FROM ft_equity_diaria)
                ORDER BY ticker, fecha
            ) p0 ON p0.ticker = p.ticker
            JOIN activos a ON a.ticker = p.ticker AND a.activo = TRUE
            WHERE p.fecha >= (SELECT MIN(fecha) FROM ft_equity_diaria)
            GROUP BY p.fecha ORDER BY p.fecha
        """)).fetchall()

    df_ops = pd.DataFrame([dict(r._mapping) for r in operaciones])
    df_met = pd.DataFrame([dict(r._mapping) for r in metricas])
    df_bch = pd.DataFrame([dict(r._mapping) for r in benchmark])
    ests   = [dict(r._mapping) for r in estrategias]

    # Normalizar fechas a Timestamp una sola vez: psycopg devuelve datetime.date
    # y comparar eso con pd.Timestamp explota ("Cannot compare Timestamp with
    # datetime.date").
    if not df_met.empty:
        df_met["fecha"] = pd.to_datetime(df_met["fecha"])
    if not df_ops.empty and "f_salida_datos" in df_ops.columns:
        df_ops["f_salida_datos"] = pd.to_datetime(df_ops["f_salida_datos"])

    return ests, df_ops, df_met, df_bch


# ── Metricas de riesgo ────────────────────────────────────────────────────────

def serie_benchmark(df_bch):
    """Serie del benchmark indexada por fecha. Vacia si no hay datos."""
    if df_bch is None or df_bch.empty:
        return None
    s = df_bch.copy()
    s["fecha"] = pd.to_datetime(s["fecha"])
    return s.set_index("fecha")["idx"].astype(float)


def calcular_riesgo(df_met_e, bench, desde=None):
    """
    Metricas de riesgo de una estrategia sobre su equity a mercado.

    El benchmark se reindexa a las fechas de ESTA estrategia: cada una arranca
    en fecha distinta y compararlas contra una ventana fija seria comparar
    periodos distintos.

    `desde` recorta la serie (ventana comparable post-fix).
    """
    if df_met_e is None or df_met_e.empty or len(df_met_e) < 2:
        return None

    sub = df_met_e.sort_values("fecha")
    if desde is not None:
        sub = sub[sub["fecha"] >= pd.Timestamp(desde)]
    if len(sub) < 3:
        return None
    equity = [float(v) for v in sub["equity"]]
    expos = [float(v) for v in sub["exposicion_pct"].dropna()]

    rb = None
    if bench is not None:
        b = bench.reindex(pd.to_datetime(sub["fecha"])).ffill()
        if b.notna().all():
            cand = retornos_desde_equity(b.tolist())
            if len(cand) == len(equity) - 1:
                rb = cand

    return resumen_riesgo(equity, retornos_bench=rb, exposiciones=expos)


# ── Calculo de stats por estrategia ───────────────────────────────────────────

def calcular_stats(est, df_ops_e):
    """Calcula las metricas de una estrategia a partir de sus operaciones."""
    capital_inicial = float(est["capital_inicial"])
    cash            = float(est["cash_disponible"])

    abiertas = df_ops_e[df_ops_e["fecha_salida"].isna()].copy()
    cerradas = df_ops_e[df_ops_e["fecha_salida"].notna()].copy()

    pnl_no_realizado = 0.0
    valor_mercado    = 0.0
    if not abiertas.empty:
        ab = abiertas.dropna(subset=["precio_actual"]).copy()
        if not ab.empty:
            ab["pa"] = ab["precio_actual"].astype(float)
            ab["pe"] = ab["precio_entrada"].astype(float)
            ab["q"]  = ab["cantidad"].astype(float)
            pnl_no_realizado = float(((ab["pa"] - ab["pe"]) * ab["q"]).sum())
            valor_mercado    = float((ab["pa"] * ab["q"]).sum())

    pnl_realizado = float(cerradas["pnl"].sum()) if not cerradas.empty else 0.0
    saldo         = cash + valor_mercado
    retorno_pct   = ((saldo - capital_inicial) / capital_inicial * 100
                     if capital_inicial else 0.0)

    n_cerradas = len(cerradas)
    ganadoras  = int((cerradas["pnl"] > 0).sum()) if n_cerradas else 0
    win_rate   = (ganadoras / n_cerradas * 100) if n_cerradas else None

    return {
        "capital_inicial":  capital_inicial,
        "cash":             cash,
        "saldo":            saldo,
        "retorno_pct":      retorno_pct,
        "pnl_realizado":    pnl_realizado,
        "pnl_no_realizado": pnl_no_realizado,
        "n_abiertas":       len(abiertas),
        "n_cerradas":       n_cerradas,
        "win_rate":         win_rate,
        "abiertas":         abiertas,
        "cerradas":         cerradas,
    }


# ── Grafico de equity ─────────────────────────────────────────────────────────

def grafico_equity(df_met, ests, bench=None):
    """Genera el grafico de curvas de equity como PNG base64. None si no hay datos."""
    if df_met.empty:
        return None

    nombre_por_id = {e["id"]: e["nombre"].replace("FT_", "") for e in ests}

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("tab10")

    for i, (eid, grupo) in enumerate(df_met.groupby("estrategia_id")):
        if eid not in nombre_por_id:
            continue
        g = grupo.sort_values("fecha")
        ax.plot(
            pd.to_datetime(g["fecha"]),
            g["equity"].astype(float),
            label=nombre_por_id[eid],
            color=cmap(i % 10),
            linewidth=1.6,
        )

    # Benchmark reescalado a 100k para que sea comparable a simple vista.
    if bench is not None and len(bench) > 1:
        ax.plot(bench.index, (bench / bench.iloc[0] * 100_000).values,
                label="Universo equiponderado", color="#111", linestyle="--",
                linewidth=2.0, zorder=1)

    ax.axhline(100_000, color="#999", linestyle=":", linewidth=0.9)
    ax.set_title("Equity marcada a mercado, por estrategia")
    ax.set_ylabel("Capital (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── Construccion del HTML ─────────────────────────────────────────────────────

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 24px;
       background: #f4f5f7; color: #1f2430; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
h2 { font-size: 17px; margin: 28px 0 8px 0; border-bottom: 2px solid #d0d4da;
     padding-bottom: 4px; }
.sub { color: #6b7280; font-size: 13px; margin-bottom: 18px; }
table { border-collapse: collapse; width: 100%; background: #fff;
        font-size: 13px; margin-bottom: 10px; }
th, td { padding: 6px 9px; text-align: right; border-bottom: 1px solid #e5e7eb; }
th { background: #2d3340; color: #fff; text-align: right; font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
tr:hover td { background: #f0f4f8; }
.pos { color: #1a8a3a; font-weight: 600; }
.neg { color: #d23b3b; font-weight: 600; }
.neutro { color: #6b7280; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
        padding: 14px 16px; margin-bottom: 22px; }
.metricas { font-size: 13px; color: #374151; margin: 4px 0 10px 0; }
.metricas b { color: #1f2430; }
.chart { text-align: center; margin: 10px 0 24px 0; }
.chart img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }
.tag { display: inline-block; background: #e8eaf0; color: #374151;
       border-radius: 4px; padding: 1px 7px; font-size: 11px; margin-left: 6px; }
.vacio { color: #9ca3af; font-style: italic; font-size: 12px; }
.nc { color: #9ca3af; font-size: 11px; }
.aviso { background: #fff8e1; border: 1px solid #f0d68a; border-left: 4px solid #e0a800;
         border-radius: 5px; padding: 11px 14px; margin: 10px 0 20px 0;
         font-size: 12.5px; color: #4a3c10; line-height: 1.55; }
.aviso b { color: #2c2408; }
.riesgo td.ic { color: #6b7280; font-size: 12px; }
"""


def tabla_riesgo(filas, clave="riesgo"):
    """
    Tabla de metricas de riesgo, calculadas SOLO desde ft_equity_diaria.

    Regla no negociable (METRICAS.md #8): todo Sharpe se muestra con su n y su
    IC95%; si el intervalo incluye cero se marca NO CONCLUYENTE. Sin eso, un
    Sharpe puntual con n<70 se lee como una superioridad que el dato no
    respalda.
    """
    th = ("<tr><th>Estrategia</th><th>Dias</th><th>Retorno</th><th>Max DD</th>"
          "<th>Volat.</th><th>Sortino</th><th>Sharpe</th><th>IC95% Sharpe</th>"
          "<th>Inf. Ratio</th><th>Beta</th><th>Exposic.</th></tr>")
    # Ordenado por Sortino: es el ratio primario del FT (long-only con stop =
    # distribucion asimetrica por diseno).
    def _orden(f):
        r = f.get(clave)
        if not r or r.get("sortino") is None:
            return -999
        return r["sortino"]

    trs = []
    for f in sorted(filas, key=_orden, reverse=True):
        r = f.get(clave)
        if not r or r.get("insuficiente"):
            trs.append(f"<tr><td>{esc(f['nombre'])}</td>"
                       f"<td colspan='10' class='vacio'>Serie insuficiente</td></tr>")
            continue

        sh = r.get("sharpe")
        if sh:
            sharpe_txt = f"{sh['sharpe']:+.2f}"
            ic_txt = f"[{sh['ic95_lo']:+.2f}, {sh['ic95_hi']:+.2f}]"
            if not sh["concluyente"]:
                ic_txt += " <span class='nc'>no concl.</span>"
        else:
            sharpe_txt, ic_txt = "-", "-"

        def num(v, dec=2, signo=False):
            if v is None:
                return "-"
            return f"{v:+.{dec}f}" if signo else f"{v:.{dec}f}"

        trs.append(
            f"<tr>"
            f"<td>{esc(f['nombre'])}</td>"
            f"<td>{r['n_dias']}</td>"
            f"<td class='{clase_signo(r['retorno_total_pct'])}'>{fmt_pct(r['retorno_total_pct'])}</td>"
            f"<td class='neg'>-{num(r['max_dd_pct'])}%</td>"
            f"<td>{num(r['volatilidad_anual'], 1)}%</td>"
            f"<td class='{clase_signo(r['sortino'] or 0)}'>{num(r['sortino'], 2, True)}</td>"
            f"<td class='{clase_signo(sh['sharpe'] if sh else 0)}'>{sharpe_txt}</td>"
            f"<td class='ic'>{ic_txt}</td>"
            f"<td class='{clase_signo(r['information_ratio'] or 0)}'>{num(r['information_ratio'], 2, True)}</td>"
            f"<td>{num(r['beta'])}</td>"
            f"<td>{num(r['exposicion_media'], 0)}%</td>"
            f"</tr>"
        )
    return f"<table class='riesgo'>{th}{''.join(trs)}</table>"


def tabla_trade(filas, clave="trade"):
    """Metricas a nivel operacion: aca esta el n grande y la senal real."""
    th = ("<tr><th>Estrategia</th><th>Oper.</th><th>Win rate</th>"
          "<th>Expectancy</th><th>Profit factor</th><th>Payoff</th>"
          "<th>Gan. media</th><th>Perd. media</th><th>Duracion</th></tr>")
    trs = []
    for f in sorted(filas, key=lambda x: -((x.get(clave) or {}).get("profit_factor") or 0)):
        m = f.get(clave) or {}
        if not m.get("n"):
            trs.append(f"<tr><td>{esc(f['nombre'])}</td>"
                       f"<td colspan='8' class='vacio'>Sin operaciones cerradas</td></tr>")
            continue
        pf = m["profit_factor"]
        pf_txt = f"{pf:.2f}" if pf is not None else "-"
        pf_cls = "pos" if (pf or 0) > 1 else "neg"
        po = m["payoff_ratio"]
        po_txt = f"{po:.2f}" if po is not None else "-"
        dur = m["duracion_media_dias"]
        dur_txt = f"{dur:.1f} d" if dur is not None else "-"
        trs.append(
            f"<tr>"
            f"<td>{esc(f['nombre'])}</td>"
            f"<td>{m['n']}</td>"
            f"<td>{m['win_rate']:.1f}%</td>"
            f"<td class='{clase_signo(m['expectancy_pct'] or 0)}'>{fmt_pct(m['expectancy_pct'] or 0)}</td>"
            f"<td class='{pf_cls}'>{pf_txt}</td>"
            f"<td>{po_txt}</td>"
            f"<td class='pos'>{fmt_pct(m['ganancia_media_pct'] or 0)}</td>"
            f"<td class='neg'>{fmt_pct(m['perdida_media_pct'] or 0)}</td>"
            f"<td>{dur_txt}</td>"
            f"</tr>"
        )
    return f"<table>{th}{''.join(trs)}</table>"


def tabla_comparativa(filas):
    """Tabla resumen de las 9 estrategias."""
    th = ("<tr><th>Estrategia</th><th>Retorno</th><th>Saldo estimado</th>"
          "<th>PnL realizado</th><th>PnL no realiz.</th>"
          "<th>Abiertas</th><th>Cerradas</th><th>Win rate</th></tr>")
    trs = []
    for f in filas:
        s = f["stats"]
        wr = f"{s['win_rate']:.0f}%" if s["win_rate"] is not None else "-"
        trs.append(
            f"<tr>"
            f"<td>{esc(f['nombre'])}</td>"
            f"<td class='{clase_signo(s['retorno_pct'])}'>{fmt_pct(s['retorno_pct'])}</td>"
            f"<td>{fmt_usd(s['saldo'])}</td>"
            f"<td class='{clase_signo(s['pnl_realizado'])}'>{fmt_pnl(s['pnl_realizado'])}</td>"
            f"<td class='{clase_signo(s['pnl_no_realizado'])}'>{fmt_pnl(s['pnl_no_realizado'])}</td>"
            f"<td>{s['n_abiertas']}</td>"
            f"<td>{s['n_cerradas']}</td>"
            f"<td>{wr}</td>"
            f"</tr>"
        )
    return f"<table>{th}{''.join(trs)}</table>"


def tabla_abiertas(abiertas):
    """Tabla de posiciones abiertas de una estrategia."""
    if abiertas.empty:
        return "<div class='vacio'>Sin posiciones abiertas.</div>"
    th = ("<tr><th>Ticker</th><th>Cant.</th><th>Capital</th><th>F. Entrada</th>"
          "<th>P. Entrada</th><th>P. Actual</th><th>PnL no realiz.</th>"
          "<th>PnL %</th><th>Score</th></tr>")
    trs = []
    for _, r in abiertas.iterrows():
        pe = float(r["precio_entrada"])
        q  = int(r["cantidad"])
        pa = float(r["precio_actual"]) if pd.notna(r["precio_actual"]) else None
        pnl   = (pa - pe) * q if pa is not None else None
        pnlpc = (pa / pe - 1) * 100 if pa is not None else None
        sc = f"{float(r['score_entrada']):.1f}" if pd.notna(r["score_entrada"]) else "-"
        trs.append(
            f"<tr><td>{esc(r['ticker'])}</td><td>{q}</td>"
            f"<td>{fmt_usd(float(r['capital_entrada']))}</td>"
            f"<td>{str(r['fecha_entrada'])[:10]}</td>"
            f"<td>{fmt_usd(pe)}</td>"
            f"<td>{fmt_usd(pa) if pa is not None else '-'}</td>"
            f"<td class='{clase_signo(pnl)}'>{fmt_pnl(pnl)}</td>"
            f"<td class='{clase_signo(pnlpc)}'>{fmt_pct(pnlpc)}</td>"
            f"<td>{sc}</td></tr>"
        )
    return f"<table>{th}{''.join(trs)}</table>"


def tabla_cerradas(cerradas):
    """Tabla de operaciones cerradas recientes de una estrategia."""
    if cerradas.empty:
        return "<div class='vacio'>Sin operaciones cerradas.</div>"
    total = len(cerradas)
    df = cerradas.sort_values("fecha_salida", ascending=False).head(MAX_CERRADAS_TABLA)
    th = ("<tr><th>Ticker</th><th>F. Entrada</th><th>F. Salida</th>"
          "<th>P. Entrada</th><th>P. Salida</th><th>PnL</th><th>PnL %</th>"
          "<th>Motivo</th></tr>")
    trs = []
    for _, r in df.iterrows():
        pnl   = float(r["pnl"])     if pd.notna(r["pnl"])     else None
        pnlpc = float(r["pnl_pct"]) if pd.notna(r["pnl_pct"]) else None
        trs.append(
            f"<tr><td>{esc(r['ticker'])}</td>"
            f"<td>{str(r['fecha_entrada'])[:10]}</td>"
            f"<td>{str(r['fecha_salida'])[:10]}</td>"
            f"<td>{fmt_usd(float(r['precio_entrada']))}</td>"
            f"<td>{fmt_usd(float(r['precio_salida'])) if pd.notna(r['precio_salida']) else '-'}</td>"
            f"<td class='{clase_signo(pnl)}'>{fmt_pnl(pnl)}</td>"
            f"<td class='{clase_signo(pnlpc)}'>{fmt_pct(pnlpc)}</td>"
            f"<td>{esc(r['motivo_salida'] or '-')}</td></tr>"
        )
    nota = (f"<div class='vacio'>Mostrando las {MAX_CERRADAS_TABLA} mas recientes "
            f"de {total} cerradas.</div>" if total > MAX_CERRADAS_TABLA else "")
    return f"<table>{th}{''.join(trs)}</table>{nota}"


def render_html(ests, df_ops, df_met, df_bch=None, desde=VENTANA_COMPARABLE):
    """Construye el documento HTML completo."""
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M")
    bench = serie_benchmark(df_bch)

    # Stats por estrategia
    filas = []
    for e in ests:
        df_e = (df_ops[df_ops["estrategia_id"] == e["id"]]
                if not df_ops.empty else pd.DataFrame())
        if df_e.empty:
            df_e = pd.DataFrame(columns=[
                "fecha_salida", "precio_actual", "precio_entrada", "cantidad",
                "pnl", "pnl_pct", "ticker", "capital_entrada", "fecha_entrada",
                "score_entrada", "precio_salida", "motivo_salida",
            ])
        stats = calcular_stats(e, df_e)

        df_met_e = (df_met[df_met["estrategia_id"] == e["id"]]
                    if not df_met.empty else pd.DataFrame())
        riesgo = calcular_riesgo(df_met_e, bench)
        riesgo_comp = calcular_riesgo(df_met_e, bench, desde=desde)

        cerr = stats["cerradas"]
        trade = metricas_trade(cerr.to_dict("records")) if not cerr.empty else {"n": 0}

        # Ventana comparable a nivel operacion: se filtra por la fecha del DATO
        # de salida, no por la de registro.
        trade_comp = {"n": 0}
        if not cerr.empty and "f_salida_datos" in cerr.columns:
            cc = cerr[pd.to_datetime(cerr["f_salida_datos"]) >= pd.Timestamp(desde)]
            if not cc.empty:
                trade_comp = metricas_trade(cc.to_dict("records"))

        filas.append({"nombre": e["nombre"].replace("FT_", ""),
                      "est": e, "stats": stats,
                      "riesgo": riesgo, "riesgo_comp": riesgo_comp,
                      "trade": trade, "trade_comp": trade_comp})

    # Resumen global
    saldo_total   = sum(f["stats"]["saldo"] for f in filas)
    inicial_total = sum(f["stats"]["capital_inicial"] for f in filas)
    ret_global    = ((saldo_total - inicial_total) / inicial_total * 100
                     if inicial_total else 0.0)

    # Grafico
    chart_b64 = grafico_equity(df_met, ests, bench)
    chart_html = (
        f"<div class='chart'><img src='data:image/png;base64,{chart_b64}'></div>"
        if chart_b64 else
        "<div class='vacio'>Sin datos en ft_equity_diaria. Correr "
        "ft_compute_equity.py.</div>"
    )

    n_bench = 0 if bench is None else len(bench)
    ret_bench = ((bench.iloc[-1] / bench.iloc[0] - 1) * 100
                 if bench is not None and n_bench > 1 else None)
    # Benchmark restringido a la ventana comparable
    ret_bench_comp = None
    if bench is not None:
        bc = bench[bench.index >= pd.Timestamp(desde)]
        if len(bc) > 1:
            ret_bench_comp = (bc.iloc[-1] / bc.iloc[0] - 1) * 100

    # Fuera del f-string: Python 3.11 no admite expresiones multilinea adentro.
    bench_txt = f" ({fmt_pct(ret_bench)} en el periodo)" if ret_bench is not None else ""
    rf_txt = f"{RF_ANUAL_DEFAULT * 100:.1f}%"
    bench_comp_txt = (f"<b class='{clase_signo(ret_bench_comp)}'>{fmt_pct(ret_bench_comp)}</b>"
                      if ret_bench_comp is not None else "n/d")
    bench_full_txt = (f"<b class='{clase_signo(ret_bench)}'>{fmt_pct(ret_bench)}</b>"
                      if ret_bench is not None else "n/d")

    # Bloques por estrategia
    bloques = []
    for f in filas:
        s, e = f["stats"], f["est"]
        bloques.append(f"""
        <div class="card">
          <h2>{esc(f['nombre'])} <span class="tag">{esc(e['logica'])}</span></h2>
          <div class="metricas">
            Capital inicial: <b>{fmt_usd(s['capital_inicial'])}</b> &nbsp;|&nbsp;
            Saldo estimado: <b>{fmt_usd(s['saldo'])}</b> &nbsp;|&nbsp;
            Retorno: <b class="{clase_signo(s['retorno_pct'])}">{fmt_pct(s['retorno_pct'])}</b>
            &nbsp;|&nbsp; Cash libre: <b>{fmt_usd(s['cash'])}</b><br>
            PnL realizado: <b class="{clase_signo(s['pnl_realizado'])}">{fmt_pnl(s['pnl_realizado'])}</b>
            &nbsp;|&nbsp;
            PnL no realizado: <b class="{clase_signo(s['pnl_no_realizado'])}">{fmt_pnl(s['pnl_no_realizado'])}</b>
            &nbsp;|&nbsp; Posiciones abiertas: <b>{s['n_abiertas']}</b>
            &nbsp;|&nbsp; Operaciones cerradas: <b>{s['n_cerradas']}</b>
          </div>
          <b>Posiciones abiertas</b>
          {tabla_abiertas(s['abiertas'])}
          <b>Operaciones cerradas</b>
          {tabla_cerradas(s['cerradas'])}
        </div>
        """)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forward Testing - Reporte</title>
<style>{CSS}</style>
</head>
<body>
  <h1>Forward Testing - Reporte de estrategias</h1>
  <div class="sub">
    Generado: {ahora} &nbsp;|&nbsp; {len(filas)} estrategias activas &nbsp;|&nbsp;
    Saldo total estimado: <b>{fmt_usd(saldo_total)}</b> &nbsp;|&nbsp;
    Retorno agregado: <b class="{clase_signo(ret_global)}">{fmt_pct(ret_global)}</b>
  </div>

  <h2>Resumen comparativo</h2>
  {tabla_comparativa(filas)}

  <h2>Equity marcada a mercado</h2>
  <div class="sub">
    Valor real de cada cartera dia a dia (cash + posiciones al cierre de ese dia).
    La linea negra punteada es el universo equiponderado{bench_txt}: comprar todo,
    sin seleccionar. Todo lo que quede por debajo de esa linea no esta aportando
    por elegir.
  </div>
  {chart_html}

  <h2>Riesgo &mdash; ventana comparable (desde {desde})</h2>
  <div class="aviso">
    <b>Por que esta es la tabla principal.</b> El {desde} se corrigio el bug del
    <code>SCORE_DEGRADADO_0.0</code>, que hacia que las estrategias 4, 6, 8 y 9
    cerraran posiciones por un score que siempre daba cero. Su historia anterior
    mide el bug, no la estrategia. Esta es la ventana en la que <b>las 10 son
    comparables entre si</b>.<br>
    Benchmark (universo equiponderado) en esta ventana: {bench_comp_txt}.
    <b>Leer los retornos contra ese numero, no contra cero</b>: en un mercado en
    baja, perder menos que el indice es agregar valor.<br><br>
    <b>Como leer las columnas.</b> El <b>max drawdown</b> es la peor caida desde
    un maximo previo; solo es calculable con la equity a mercado y es la metrica
    mas confiable de la tabla.
    El <b>Sharpe</b> va con su intervalo de confianza al 95%: con estas muestras
    el numero suelto no distingue habilidad de azar, y si el intervalo
    <b>incluye el cero</b> se marca <span class="nc">no concl.</span> y
    <b>no alcanza para justificar un cambio de estrategia</b>.
    Se ordena por <b>Sortino</b>, el ratio principal aca: las estrategias son
    long-only con stop, o sea asimetricas por diseno, y el Sharpe castiga las
    subidas fuertes que justamente se buscan. Tasa libre de riesgo: {rf_txt} anual.
    El <b>Information Ratio</b> mide el aporte por encima del benchmark: responde
    "esto le gana a comprar todo el universo?".
  </div>
  {tabla_riesgo(filas, "riesgo_comp")}

  <h2>Riesgo &mdash; historia completa</h2>
  <div class="sub">
    Toda la serie de cada estrategia. Benchmark del periodo: {bench_full_txt}.
    Util para ver el comportamiento en mas de un regimen de mercado, pero
    <b>no comparable entre estrategias</b>: incluye el periodo del bug para las
    cuatro afectadas, y cada estrategia arranca en una fecha distinta.
  </div>
  {tabla_riesgo(filas, "riesgo")}

  <h2>Metricas por operacion &mdash; desde {desde}</h2>
  <div class="sub">
    Con 30-70 dias de serie los ratios de arriba son ruidosos; aca el n es de
    decenas o centenas de operaciones y la senal es mas confiable.
    <b>Profit factor</b> = dolares ganados por cada dolar perdido (&gt;1 es rentable).
    Ordenado por profit factor.
  </div>
  {tabla_trade(filas, "trade_comp")}

  <h2>Metricas por operacion &mdash; historia completa</h2>
  <div class="sub">
    Incluye el periodo del bug de score para las estrategias 4, 6, 8 y 9.
  </div>
  {tabla_trade(filas, "trade")}

  <h2>Detalle por estrategia</h2>
  {''.join(bloques)}
</body>
</html>"""


# ── Runner ────────────────────────────────────────────────────────────────────

def run(output, desde=VENTANA_COMPARABLE):
    print(f"[ft_reporte_html] Leyendo DB local...")
    ests, df_ops, df_met, df_bch = cargar_datos()
    if not ests:
        print("[ft_reporte_html] No hay estrategias activas. Nada que reportar.")
        return

    if df_met.empty:
        print("[ft_reporte_html] [WARN] ft_equity_diaria vacia: sin metricas de "
              "riesgo ni grafico. Correr ft_compute_equity.py primero.")

    html = render_html(ests, df_ops, df_met, df_bch, desde=desde)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(html)

    n_ops = 0 if df_ops.empty else len(df_ops)
    n_eq = 0 if df_met.empty else len(df_met)
    print(f"[ft_reporte_html] {len(ests)} estrategias, {n_ops} operaciones, "
          f"{n_eq} dias de equity.")
    print(f"[ft_reporte_html] Reporte generado: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera el reporte HTML de Forward Testing"
    )
    parser.add_argument("--output", default=OUTPUT_DEFAULT,
                        help=f"Ruta del HTML de salida (default: {OUTPUT_DEFAULT})")
    parser.add_argument("--desde", default=VENTANA_COMPARABLE,
                        help=f"Inicio de la ventana comparable YYYY-MM-DD "
                             f"(default: {VENTANA_COMPARABLE}, fix del score=0.0)")
    args = parser.parse_args()
    run(args.output, desde=args.desde)


if __name__ == "__main__":
    main()
