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

OUTPUT_DEFAULT = os.path.join(ROOT, "reportes", "ft_reporte.html")
MAX_CERRADAS_TABLA = 20   # operaciones cerradas a mostrar por estrategia


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

        metricas = conn.execute(text("""
            SELECT estrategia_id, fecha, capital_total
            FROM ft_metricas_diarias
            ORDER BY estrategia_id, fecha
        """)).fetchall()

    df_ops = pd.DataFrame([dict(r._mapping) for r in operaciones])
    df_met = pd.DataFrame([dict(r._mapping) for r in metricas])
    ests   = [dict(r._mapping) for r in estrategias]
    return ests, df_ops, df_met


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

def grafico_equity(df_met, ests):
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
            g["capital_total"].astype(float),
            label=nombre_por_id[eid],
            color=cmap(i % 10),
            linewidth=1.6,
        )

    ax.axhline(100_000, color="#999", linestyle="--", linewidth=0.9)
    ax.set_title("Curvas de equity por estrategia (capital total)")
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
"""


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


def render_html(ests, df_ops, df_met):
    """Construye el documento HTML completo."""
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M")

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
        filas.append({"nombre": e["nombre"].replace("FT_", ""),
                       "est": e, "stats": calcular_stats(e, df_e)})

    # Resumen global
    saldo_total   = sum(f["stats"]["saldo"] for f in filas)
    inicial_total = sum(f["stats"]["capital_inicial"] for f in filas)
    ret_global    = ((saldo_total - inicial_total) / inicial_total * 100
                     if inicial_total else 0.0)

    # Grafico
    chart_b64 = grafico_equity(df_met, ests)
    chart_html = (
        f"<div class='chart'><img src='data:image/png;base64,{chart_b64}'></div>"
        if chart_b64 else
        "<div class='vacio'>Sin datos de ft_metricas_diarias para graficar.</div>"
    )

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

  <h2>Curvas de equity</h2>
  {chart_html}

  <h2>Detalle por estrategia</h2>
  {''.join(bloques)}
</body>
</html>"""


# ── Runner ────────────────────────────────────────────────────────────────────

def run(output):
    print(f"[ft_reporte_html] Leyendo DB local...")
    ests, df_ops, df_met = cargar_datos()
    if not ests:
        print("[ft_reporte_html] No hay estrategias activas. Nada que reportar.")
        return

    html = render_html(ests, df_ops, df_met)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(html)

    n_ops = 0 if df_ops.empty else len(df_ops)
    print(f"[ft_reporte_html] {len(ests)} estrategias, {n_ops} operaciones.")
    print(f"[ft_reporte_html] Reporte generado: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera el reporte HTML de Forward Testing"
    )
    parser.add_argument("--output", default=OUTPUT_DEFAULT,
                        help=f"Ruta del HTML de salida (default: {OUTPUT_DEFAULT})")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
