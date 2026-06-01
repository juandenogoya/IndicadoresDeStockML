"""
make_infografia_fundamental.py
Genera una infografia fundamental (PNG) para un ticker, con DATOS REALES y
COMPROBABLES de las tablas locales (fundamentales_*). NO usa LLM, NO usa
proyecciones (sin DCF ni fair value): es una FOTO del ultimo trimestre + la
comparacion vs pares del sector/region. Coherente con la filosofia del AF
en el proyecto (descriptivo, no predictivo).

A diferencia de make_infografia.py (que es tecnico y consume el MCP), este lee
DIRECTO de la DB local porque los fundamentales viven en local (Plan C).

Bloques (solo lo que podemos llenar honestamente):
  - Encabezado: ticker, sector/region, cierre USD, ultimo Q, moneda de reporte
  - KPIs margenes TTM: bruto / operativo / neto / FCF
  - Serie Revenue & FCF (los Q que haya, hasta 8)
  - Valuacion (PER/PB/PS/EV-EBITDA) con vs mediana del sector
  - Calidad (ROE/ROA/ROIC + margen neto) con vs mediana del sector
  - Solvencia (current ratio, deuda/patrimonio, caja neta)
  - Footer disclaimer

Uso:
    python make_infografia_fundamental.py AAPL
    python make_infografia_fundamental.py AAPL -o salida.png
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Forzar engine local ANTES de importar database (Plan C: fundamentales en local)
os.environ.pop("DATABASE_URL", None)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))

try:
    from jinja2 import Environment, FileSystemLoader
    from weasyprint import HTML
except ImportError as exc:
    print(f"ERROR: falta dependencia: {exc}", file=sys.stderr)
    print("Ejecutar: pip install -r scripts/reports/requirements.txt", file=sys.stderr)
    sys.exit(1)

from src.data.database import query_df

TEMPLATES_DIR = SCRIPT_DIR / "templates"
DEFAULT_HANDLE = "@juan_de_nogoya"


# ── Helpers de formato ────────────────────────────────────────────────────────

def _f(v):
    if v is None:
        return None
    try:
        fv = float(v)
        return None if fv != fv else fv  # NaN guard
    except (TypeError, ValueError):
        return None


def _jget(rj, key):
    """Lee una clave de raw_json (dict o str JSON). None si falta/NaN."""
    if rj is None:
        return None
    if isinstance(rj, str):
        try:
            import json as _json
            rj = _json.loads(rj)
        except Exception:
            return None
    v = rj.get(key) if isinstance(rj, dict) else None
    return _f(v)


def _pct(v, decimales=1):
    """Fraccion -> 'XX.X%'. None -> '-'."""
    v = _f(v)
    return "-" if v is None else f"{v * 100:.{decimales}f}%"


def _ratio(v):
    v = _f(v)
    return "-" if v is None else f"{v:.1f}"


def _money_b(v):
    """Absoluto -> miles de millones 'X.XB' / millones 'XXXM'. Moneda de reporte."""
    v = _f(v)
    if v is None:
        return "-"
    a = abs(v)
    signo = "-" if v < 0 else ""
    if a >= 1e9:
        return f"{signo}{a / 1e9:.1f}B"
    if a >= 1e6:
        return f"{signo}{a / 1e6:.0f}M"
    return f"{signo}{a:,.0f}"


def _signo_pct(v):
    """vs_median_pct (fraccion) -> '+XX%' / '-XX%'. None -> None."""
    v = _f(v)
    return None if v is None else f"{v * 100:+.0f}%"


# ── Carga de datos (todo de tablas locales) ───────────────────────────────────

def cargar_datos(ticker: str) -> dict:
    ticker = ticker.strip().upper()

    # Ratios del ultimo Q
    r = query_df(
        """
        SELECT * FROM fundamentales_ratios_q
        WHERE ticker = :t ORDER BY fiscal_period_end DESC LIMIT 1
        """,
        params={"t": ticker},
    )
    if r.empty:
        raise SystemExit(f"Sin datos fundamentales para {ticker}. "
                         f"Correr scripts/manual/refresh_fundamentales.bat")
    ratios = r.iloc[0].to_dict()

    # Region (ticker_pais)
    pais = query_df("SELECT region, country FROM ticker_pais WHERE ticker = :t",
                    params={"t": ticker})
    region = pais.iloc[0]["region"] if not pais.empty else None

    # Cierre USD (precios_diarios)
    px = query_df(
        "SELECT fecha, close FROM precios_diarios WHERE ticker = :t "
        "ORDER BY fecha DESC LIMIT 1",
        params={"t": ticker},
    )
    cierre = _f(px.iloc[0]["close"]) if not px.empty else None
    fecha_cierre = str(px.iloc[0]["fecha"]) if not px.empty else None

    # MarketCap (ultimo Q 3M de valuation) para el dividend yield
    mc = query_df(
        "SELECT market_cap FROM fundamentales_valuation_q WHERE ticker = :t "
        "AND period_type='3M' ORDER BY period_end DESC LIMIT 1",
        params={"t": ticker},
    )
    market_cap = _f(mc.iloc[0]["market_cap"]) if not mc.empty else None

    # Serie Revenue & FCF (hasta 8 Q, orden cronologico)
    serie = query_df(
        """
        SELECT i.fiscal_period_end AS fpe, i.total_revenue AS revenue,
               c.free_cash_flow AS fcf
        FROM fundamentales_income_q i
        LEFT JOIN fundamentales_cashflow_q c
          ON c.ticker = i.ticker AND c.fiscal_period_end = i.fiscal_period_end
        WHERE i.ticker = :t
        ORDER BY i.fiscal_period_end
        """,
        params={"t": ticker},
    )

    # Dividendos: CashDividendsPaid (cashflow, viene negativo) ultimos 8 Q +
    # serie de margenes (de ratios_q: net_margin_ttm y roe_ttm) para tendencia.
    divcf = query_df(
        """
        SELECT fiscal_period_end AS fpe, raw_json
        FROM fundamentales_cashflow_q WHERE ticker = :t
        ORDER BY fiscal_period_end DESC LIMIT 4
        """,
        params={"t": ticker},
    )
    serie_margen = query_df(
        """
        SELECT fiscal_period_end AS fpe, net_margin_ttm, roe_ttm
        FROM fundamentales_ratios_q WHERE ticker = :t
        ORDER BY fiscal_period_end
        """,
        params={"t": ticker},
    )

    # Comparacion vs sector (ultimo Q)
    vs = query_df(
        """
        SELECT metric, vs_median_pct, peer_median, percentile, peer_n, peer_basis,
               peer_region, low_sample
        FROM fundamentales_ticker_vs_sector
        WHERE ticker = :t AND fiscal_period_end = (
            SELECT MAX(fiscal_period_end) FROM fundamentales_ticker_vs_sector
            WHERE ticker = :t
        )
        """,
        params={"t": ticker},
    )
    vs_map = {row["metric"]: row for _, row in vs.iterrows()}
    peer_meta = {}
    if not vs.empty:
        first = vs.iloc[0]
        peer_meta = {"basis": first["peer_basis"], "region": first["peer_region"],
                     "n": int(first["peer_n"]) if first["peer_n"] is not None else None}

    return {
        "ratios": ratios, "region": region, "cierre": cierre,
        "fecha_cierre": fecha_cierre, "serie": serie, "vs_map": vs_map,
        "peer_meta": peer_meta, "divcf": divcf, "serie_margen": serie_margen,
        "market_cap": market_cap,
    }


# ── Mini graficos SVG (sin libs de plotting; control total del estilo) ────────

def _sparkline_dual(serie, w=560, h=210, pad=34, solo_revenue=False) -> str:
    """
    Mini grafico de dos series (revenue + fcf) normalizadas a su rango conjunto.
    Devuelve SVG embebible. Normalizamos a [0,1] sobre el min/max combinado para
    que ambas curvas compartan eje (la comparacion visual es de forma, no de
    nivel absoluto -- las magnitudes van en los KPIs).

    solo_revenue=True (banca): grafica solo ingresos (FCF no es metrica de banco).
    """
    rows = [(str(r["fpe"]), _f(r["revenue"]),
             None if solo_revenue else _f(r["fcf"])) for _, r in serie.iterrows()]
    rows = [r for r in rows if r[1] is not None or r[2] is not None]
    if len(rows) < 2:
        return '<div class="sin-datos">serie insuficiente</div>'

    revs = [r[1] for r in rows if r[1] is not None]
    fcfs = [r[2] for r in rows if r[2] is not None]
    vals = revs + fcfs
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(rows)

    def _x(i):
        return pad + (w - 2 * pad) * i / (n - 1)

    def _y(v):
        # invertir Y (SVG crece hacia abajo)
        return h - pad - (h - 2 * pad) * (v - lo) / rng

    def _poly(idx):
        pts = []
        for i, r in enumerate(rows):
            v = r[idx]
            if v is None:
                continue
            pts.append(f"{_x(i):.1f},{_y(v):.1f}")
        return " ".join(pts)

    # Linea de cero si el rango cruza 0 (FCF puede ser negativo)
    cero = ""
    if lo < 0 < hi:
        y0 = _y(0)
        cero = (f'<line x1="{pad}" y1="{y0:.1f}" x2="{w - pad}" y2="{y0:.1f}" '
                f'stroke="#bbb" stroke-width="1" stroke-dasharray="3,3"/>')

    # Etiquetas de primer y ultimo periodo (anio-mes)
    lbl0 = rows[0][0][:7]
    lbl1 = rows[-1][0][:7]

    return f'''<svg viewBox="0 0 {w} {h}" class="spark" xmlns="http://www.w3.org/2000/svg">
      {cero}
      <polyline points="{_poly(1)}" fill="none" stroke="#111" stroke-width="2.5"/>
      <polyline points="{_poly(2)}" fill="none" stroke="#9aa0a6" stroke-width="2.5"/>
      <text x="{pad}" y="{h - 6}" class="spark-lbl">{lbl0}</text>
      <text x="{w - pad}" y="{h - 6}" class="spark-lbl" text-anchor="end">{lbl1}</text>
    </svg>'''


def _sparkline_pct(serie_df, col, w=560, h=170, pad=34) -> str:
    """
    Mini grafico de UNA serie de porcentajes (margen neto o ROE) en el tiempo.
    Etiqueta el primer y ultimo valor con su %. Para la tendencia de margenes.
    """
    rows = [(str(r["fpe"]), _f(r[col])) for _, r in serie_df.iterrows()]
    rows = [r for r in rows if r[1] is not None]
    if len(rows) < 2:
        return '<div class="sin-datos">serie insuficiente</div>'
    vals = [r[1] for r in rows]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or abs(hi) or 1.0
    n = len(rows)

    def _x(i): return pad + (w - 2 * pad) * i / (n - 1)
    def _y(v): return h - pad - (h - 2 * pad) * (v - lo) / rng

    pts = " ".join(f"{_x(i):.1f},{_y(r[1]):.1f}" for i, r in enumerate(rows))
    # color de la linea segun tendencia (ultimo vs primero)
    sube = rows[-1][1] >= rows[0][1]
    color = "#1a7f37" if sube else "#cf222e"
    v0 = f"{rows[0][1]*100:.0f}%"; v1 = f"{rows[-1][1]*100:.0f}%"
    return f'''<svg viewBox="0 0 {w} {h}" class="spark" xmlns="http://www.w3.org/2000/svg">
      <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>
      <text x="{pad}" y="{h - 6}" class="spark-lbl">{v0}</text>
      <text x="{w - pad}" y="{h - 6}" class="spark-lbl" text-anchor="end">{v1}</text>
    </svg>'''


def _calc_dividendos(data) -> dict:
    """Yield TTM + payout TTM desde CashDividendsPaid (caja, 4Q) / MarketCap / NI.
    Devuelve {paga, yield_pct, payout_pct}. paga=False si no distribuye."""
    divcf = data.get("divcf")
    if divcf is None or divcf.empty:
        return {"paga": False}
    divs = [_jget(rj, "CashDividendsPaid") for rj in divcf["raw_json"]]
    divs = [abs(d) for d in divs if d is not None]
    div_ttm = sum(divs) if divs else 0.0
    if div_ttm <= 0:
        return {"paga": False}
    ni_ttm = _f(data["ratios"].get("net_income_ttm"))
    mcap = _f(data.get("market_cap"))
    yld = (div_ttm / mcap) if (mcap and mcap > 0) else None
    payout = (div_ttm / ni_ttm) if (ni_ttm and ni_ttm > 0) else None
    return {"paga": True, "yield_pct": yld, "payout_pct": payout}


# ── Construccion del contexto para el template ────────────────────────────────

def _fila_cmp(label, metric, valor_fmt, data, mejor_si_alto):
    """Arma una fila comparativa: valor + vs-mediana + color segun direccion."""
    vs = data["vs_map"].get(metric)
    vs_pct = _signo_pct(vs["vs_median_pct"]) if vs is not None else None
    color = None
    if vs is not None:
        vp = _f(vs["vs_median_pct"])
        if vp is not None and vp != 0:
            # color segun si "mejor" es alto o bajo para esta metrica
            mejor = (vp > 0) if mejor_si_alto else (vp < 0)
            color = "good" if mejor else "bad"
    return {"label": label, "valor": valor_fmt, "vs": vs_pct, "color": color}


def build_context(ticker: str, data: dict, handle: str) -> dict:
    r = data["ratios"]
    cur = r.get("reporting_currency") or "?"
    fpe = str(r.get("fiscal_period_end"))
    sector = r.get("sector") or "?"
    region = data["region"] or "?"

    pm = data["peer_meta"]
    if pm.get("basis") == "region":
        peer_txt = f"vs {sector} / {pm['region']} (n={pm['n']})"
    elif pm.get("basis") == "usa_fallback":
        peer_txt = f"vs {sector} / USA (n={pm['n']}) — sin pares regionales"
    elif pm.get("basis") == "custom":
        peer_txt = f"vs {sector} (n={pm['n']})"
    else:
        peer_txt = "pocas empresas del sector en seguimiento"

    es_fin = (r.get("profile") == "financiero")

    # Dividendos (ambos perfiles)
    _d = _calc_dividendos(data)
    if not _d["paga"]:
        div = {"paga": False}
    else:
        div = {
            "paga": True,
            "yield": _pct(_d["yield_pct"]) if _d["yield_pct"] is not None else "-",
            "payout": _pct(_d["payout_pct"]) if _d["payout_pct"] is not None else "-",
        }

    # Valuacion: igual en ambos perfiles (sin EV/EBITDA en banca: no aplica)
    valuacion = [
        _fila_cmp("PER",       "pe_ratio",  _ratio(r.get("pe_ratio")),  data, mejor_si_alto=False),
        _fila_cmp("P/B",       "pb_ratio",  _ratio(r.get("pb_ratio")),  data, mejor_si_alto=False),
        _fila_cmp("P/S",       "ps_ratio",  _ratio(r.get("ps_ratio")),  data, mejor_si_alto=False),
    ]
    if not es_fin:
        valuacion.append(
            _fila_cmp("EV/EBITDA", "ev_ebitda", _ratio(r.get("ev_ebitda")), data, mejor_si_alto=False))

    if es_fin:
        # Perfil FINANCIERO: KPIs y calidad propios de banca
        kpis = [
            {"label": "Margen neto",      "valor": _pct(r.get("net_margin_ttm"))},
            {"label": "ROE (TTM)",        "valor": _pct(r.get("roe_ttm"))},
            {"label": "ROTCE (TTM)",      "valor": _pct(r.get("rotce_ttm"))},
            {"label": "Ratio eficiencia", "valor": _pct(r.get("efficiency_ratio_ttm"))},
        ]
        kpis_foot = "metricas de banca sobre 12 meses (TTM)"
        calidad = [
            _fila_cmp("ROE",             "roe_ttm",             _pct(r.get("roe_ttm")),             data, mejor_si_alto=True),
            _fila_cmp("ROTCE",           "rotce_ttm",           _pct(r.get("rotce_ttm")),           data, mejor_si_alto=True),
            _fila_cmp("ROA",             "roa_ttm",             _pct(r.get("roa_ttm")),             data, mejor_si_alto=True),
            _fila_cmp("Margen neto",     "net_margin_ttm",      _pct(r.get("net_margin_ttm")),      data, mejor_si_alto=True),
            _fila_cmp("Ratio eficiencia","efficiency_ratio_ttm",_pct(r.get("efficiency_ratio_ttm")),data, mejor_si_alto=False),
        ]
        # Crecimiento (banca: FCF no aplica)
        crecimiento = [
            {"label": "Ingresos YoY", "valor": _signo_pct(r.get("revenue_yoy_pct")) or "-"},
            {"label": "Ut. neta YoY", "valor": _signo_pct(r.get("net_income_yoy_pct")) or "-"},
            {"label": "BPA YoY",      "valor": _signo_pct(r.get("eps_yoy_pct")) or "-"},
        ]
    else:
        # Perfil NO-FINANCIERO: margenes industriales + ROIC + FCF
        kpis = [
            {"label": "Margen bruto",      "valor": _pct(r.get("gross_margin_ttm"))},
            {"label": "Margen operativo",  "valor": _pct(r.get("operating_margin_ttm"))},
            {"label": "Margen neto",       "valor": _pct(r.get("net_margin_ttm"))},
            {"label": "Margen FCF",        "valor": _pct(r.get("fcf_margin_ttm"))},
        ]
        kpis_foot = "margenes sobre 12 meses (TTM)"
        calidad = [
            _fila_cmp("ROE",         "roe_ttm",        _pct(r.get("roe_ttm")),        data, mejor_si_alto=True),
            _fila_cmp("ROA",         "roa_ttm",        _pct(r.get("roa_ttm")),        data, mejor_si_alto=True),
            _fila_cmp("ROIC",        "roic_ttm",       _pct(r.get("roic_ttm")),       data, mejor_si_alto=True),
            _fila_cmp("Margen neto", "net_margin_ttm", _pct(r.get("net_margin_ttm")), data, mejor_si_alto=True),
        ]
        crecimiento = [
            {"label": "Ingresos YoY", "valor": _signo_pct(r.get("revenue_yoy_pct")) or "-"},
            {"label": "Ut. neta YoY", "valor": _signo_pct(r.get("net_income_yoy_pct")) or "-"},
            {"label": "BPA YoY",      "valor": _signo_pct(r.get("eps_yoy_pct")) or "-"},
            {"label": "FCF YoY",      "valor": _signo_pct(r.get("fcf_yoy_pct")) or "-"},
        ]

    # Solvencia. net_debt<0 => caja neta (se muestra como positivo "caja neta")
    nd = _f(r.get("net_debt"))
    if nd is None:
        caja_label, caja_val = "Deuda neta", "-"
    elif nd < 0:
        caja_label, caja_val = "Caja neta", f"{_money_b(-nd)} {cur}"
    else:
        caja_label, caja_val = "Deuda neta", f"{_money_b(nd)} {cur}"

    if es_fin:
        # Banca: sin liquidez corriente NI deuda neta (poco informativa:
        # incluye depositos/funding). Solo Deuda/Patrimonio.
        solvencia = [
            {"label": "Deuda / Patrimonio", "valor": _ratio(r.get("debt_to_equity"))},
        ]
    else:
        solvencia = [
            {"label": "Liquidez corriente", "valor": _ratio(r.get("current_ratio"))},
            {"label": "Deuda / Patrimonio", "valor": _ratio(r.get("debt_to_equity"))},
            {"label": caja_label,           "valor": caja_val},
        ]

    return {
        "ticker": ticker,
        "sector": sector,
        "region": region,
        "cierre": data["cierre"],
        "fecha_cierre": data["fecha_cierre"],
        "moneda": cur,
        "ultimo_q": fpe,
        "peer_txt": peer_txt,
        "es_financiero": es_fin,
        "moneda_no_usd": (cur not in ("USD", "?", None)),
        "kpis": kpis,
        "kpis_foot": kpis_foot,
        "calidad_titulo": "Calidad / Rentabilidad (banca)" if es_fin else "Calidad / Rentabilidad",
        "serie_titulo": "Ingresos" if es_fin else "Ingresos y FCF",
        "valuacion": valuacion,
        "calidad": calidad,
        "crecimiento": crecimiento,
        "solvencia": solvencia,
        "spark_svg": _sparkline_dual(data["serie"], solo_revenue=es_fin),
        "dividendos": div,
        "margen_titulo": "Tendencia ROE (TTM)" if es_fin else "Tendencia margen neto (TTM)",
        "margen_svg": _sparkline_pct(data.get("serie_margen"),
                                     "roe_ttm" if es_fin else "net_margin_ttm"),
        "handle": handle,
        "fecha_legible": datetime.now().strftime("%d/%m/%Y"),
    }


# ── Render ────────────────────────────────────────────────────────────────────

def render_html(ctx: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    tpl = env.get_template("infografia_fundamental.html")
    return tpl.render(**ctx)


def html_a_png(html_str: str, output_path: Path) -> None:
    html_obj = HTML(string=html_str, base_url=str(TEMPLATES_DIR))
    pdf_bytes = html_obj.write_pdf()
    import fitz  # PyMuPDF: PDF -> PNG sin deps de sistema
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x alta resolucion
    pix.save(str(output_path))
    doc.close()


def generar_infografia(ticker: str, output: str | None = None,
                       handle: str = DEFAULT_HANDLE) -> Path:
    """
    API publica reutilizable (ej. desde el dashboard): genera el PNG y devuelve
    su ruta. Misma logica que main() pero sin argparse ni prints.
    """
    ticker = ticker.upper()
    data = cargar_datos(ticker)
    ctx = build_context(ticker, data, handle)
    html_str = render_html(ctx)
    out = Path(output) if output else (
        SCRIPT_DIR / "output" / f"{ticker}_{data['fecha_cierre']}_fundamental.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    html_a_png(html_str, out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Infografia fundamental (datos reales locales)")
    ap.add_argument("ticker", help="Ticker (ej AAPL)")
    ap.add_argument("-o", "--output", default=None, help="Ruta PNG de salida")
    ap.add_argument("--handle", default=DEFAULT_HANDLE)
    args = ap.parse_args()

    out = generar_infografia(args.ticker, args.output, args.handle)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
