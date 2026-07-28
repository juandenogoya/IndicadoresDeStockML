"""
make_infografia_simple.py
Genera la infografia SIMPLE (PNG) de un ticker: 1 tarjeta 4:5 combinada
tecnico + fundamental, pensada para redes. Menos datos, mejor contados:
2 graficos (precio+muros, combo PER/ROE) + chips de opciones + cinta de veredicto.

Datos REALES de tablas locales (Plan C), sin LLM. Convive con las otras dos
infografias (tecnica del MCP y fundamental). Spec: docs/infografia_simple.md.

Reusa el "cerebro" del dashboard:
  - dashboard.sintesis_data.cargar_datos_ticker -> precio + opciones_plazo (muros
    con fuerza + pcr_vol + veredicto_oi).
  - src.utils.dashboard_sintesis.sintetizar -> veredicto (ALCISTA/NEUTRAL/BAJISTA).

Uso:
    python scripts/reports/make_infografia_simple.py NVDA
    python scripts/reports/make_infografia_simple.py JPM -o salida.png
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Forzar engine local ANTES de importar cualquier cosa que lea la DB (Plan C).
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
from dashboard.sintesis_data import cargar_datos_ticker
from src.utils.dashboard_sintesis import sintetizar

TEMPLATES_DIR = SCRIPT_DIR / "templates"
DEFAULT_HANDLE = "@juan_de_nogoya"

PLAZOS = ["corto", "medio", "largo"]
PLAZO_LBL = {"corto": "Corto", "medio": "Medio", "largo": "Largo"}
COLOR_ESTADO = {"ALCISTA": "#1a7f37", "BAJISTA": "#c1121f", "NEUTRAL": "#6b7280"}
DIAS_PRECIO = 88   # ~4 meses habiles

# Paleta (consistente con infografia_fundamental.css)
INK = "#1a1a1a"
VERDE = "#1a7f37"
ROJO = "#cf222e"
SECTOR_BAR = "#aac4e0"   # barras = PER mediano del sector (benchmark, tono suave)
SUAVE = "#9aa0a6"


# ── Helpers de formato ────────────────────────────────────────────────────────

def _f(v):
    if v is None:
        return None
    try:
        fv = float(v)
        return None if fv != fv else fv  # NaN guard
    except (TypeError, ValueError):
        return None


def _ratio(v):
    v = _f(v)
    if v is None:
        return "-"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_datos(ticker: str) -> dict:
    ticker = ticker.strip().upper()

    # Cerebro del dashboard: veredicto + opciones_plazo (muros con fuerza) + precio.
    datos = cargar_datos_ticker(ticker)
    sintesis = sintetizar(datos)

    precio = datos.get("precio") or {}
    if precio.get("close") is None:
        raise SystemExit(f"Sin precio para {ticker} (precios_diarios vacia?).")

    # Serie de precio ~4 meses (cronologico)
    px = query_df(
        "SELECT fecha, close FROM precios_diarios WHERE ticker = :t "
        "ORDER BY fecha DESC LIMIT :n",
        params={"t": ticker, "n": DIAS_PRECIO},
    )
    px = px.iloc[::-1].reset_index(drop=True)  # a cronologico

    # Valuacion: PER del ticker vs PER mediano de su SECTOR, por trimestre
    # calendario (los fiscal_period_end no estan alineados entre tickers -> se
    # bucketea por date_trunc('quarter') para comparar peras con peras).
    srow = query_df("SELECT sector FROM activos WHERE ticker = :t",
                    params={"t": ticker})
    sector = srow.iloc[0]["sector"] if not srow.empty else None
    serie_val = _serie_valuacion(ticker, sector)

    return {
        "ticker": ticker,
        "datos": datos,
        "sintesis": sintesis,
        "precio": precio,
        "px": px,
        "sector": sector,
        "serie_val": serie_val,
    }


def _cq_label(cq) -> str:
    """date (primer dia del trimestre) -> 'YY Qn' (ej. 26 Q1)."""
    q = (cq.month - 1) // 3 + 1
    return f"{cq.year % 100:02d} Q{q}"


def _serie_valuacion(ticker: str, sector, n: int = 6) -> list:
    """PER del ticker + PER mediano del sector, por trimestre calendario (ultimos
    n con dato del ticker, cronologico). Ambos de pe_ratio fiscal (misma base)."""
    per_t = query_df(
        """
        SELECT DISTINCT ON (date_trunc('quarter', fiscal_period_end))
               date_trunc('quarter', fiscal_period_end)::date AS cq, pe_ratio AS per
        FROM fundamentales_ratios_q
        WHERE ticker = :t AND pe_ratio IS NOT NULL
        ORDER BY date_trunc('quarter', fiscal_period_end) DESC, fiscal_period_end DESC
        """,
        params={"t": ticker},
    )
    if per_t.empty or not sector:
        return []
    per_s = query_df(
        """
        SELECT date_trunc('quarter', r.fiscal_period_end)::date AS cq,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY r.pe_ratio) AS per_med
        FROM fundamentales_ratios_q r
        JOIN activos a ON a.ticker = r.ticker AND a.activo = TRUE
        WHERE a.sector = :s AND r.pe_ratio IS NOT NULL
        GROUP BY 1
        """,
        params={"s": sector},
    )
    smap = {row["cq"]: _f(row["per_med"]) for _, row in per_s.iterrows()}
    rows = per_t.head(n).iloc[::-1]  # ultimos n, cronologico
    return [{"cq": r["cq"], "lbl": _cq_label(r["cq"]),
             "per_t": _f(r["per"]), "per_s": smap.get(r["cq"])}
            for _, r in rows.iterrows()]


def _mejor_muro(opciones_plazo: dict, lado: str) -> dict | None:
    """Muro (put_wall=soporte / call_wall=resistencia) de MAYOR fuerza entre las
    3 ventanas. Devuelve el dict del muro + su ventana. None si no hay ninguno."""
    best = None
    for v in PLAZOS:
        b = (opciones_plazo.get(v) or {}).get(lado)
        if not b or b.get("strike") is None:
            continue
        fz = _f(b.get("fuerza"))
        cand = {**b, "ventana": v, "_fz": fz if fz is not None else -1.0}
        if best is None or cand["_fz"] > best["_fz"]:
            best = cand
    return best


# ── Graficos SVG (inline, mismo patron que los sparklines) ────────────────────

def _chart_precio_muros(px, soporte, resistencia, w=1000, h=262, pad=42) -> str:
    """Linea de precio + soporte (verde) y resistencia (roja) como lineas
    horizontales etiquetadas. UN solo eje (precio)."""
    closes = [_f(c) for c in px["close"].tolist()]
    closes = [c for c in closes if c is not None]
    if len(closes) < 2:
        return '<div class="sin-datos">serie de precio insuficiente</div>'
    n = len(closes)

    niveles = []
    if soporte and soporte.get("strike") is not None:
        niveles.append(_f(soporte["strike"]))
    if resistencia and resistencia.get("strike") is not None:
        niveles.append(_f(resistencia["strike"]))
    lo = min(closes + [x for x in niveles if x is not None])
    hi = max(closes + [x for x in niveles if x is not None])
    rng = (hi - lo) or 1.0
    lo -= rng * 0.06
    hi += rng * 0.06
    rng = hi - lo

    def _x(i):
        return pad + (w - 2 * pad) * i / (n - 1)

    def _y(v):
        return h - pad - (h - 2 * pad) * (v - lo) / rng

    pts = " ".join(f"{_x(i):.1f},{_y(c):.1f}" for i, c in enumerate(closes))

    def _linea_nivel(nivel, color, etiqueta):
        if not nivel or nivel.get("strike") is None:
            return ""
        y = _y(_f(nivel["strike"]))
        return (
            f'<line x1="{pad}" y1="{y:.1f}" x2="{w - pad}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2.5" stroke-dasharray="7,5"/>'
            f'<rect x="{pad}" y="{y - 27:.1f}" width="392" height="25" rx="5" '
            f'fill="{color}"/>'
            f'<text x="{pad + 12}" y="{y - 8:.1f}" class="nivel-lbl">{etiqueta}</text>'
        )

    def _et(nivel, pref):
        st = _ratio(nivel["strike"])
        fz = _f(nivel.get("fuerza"))
        fz_s = f" - f{fz:.0f}%" if fz is not None else ""
        return f"{pref} {st}{fz_s} ({PLAZO_LBL[nivel['ventana']]})"

    res_svg = _linea_nivel(resistencia, ROJO, _et(resistencia, "Resist.")) if resistencia else ""
    sop_svg = _linea_nivel(soporte, VERDE, _et(soporte, "Soporte")) if soporte else ""

    # ultimo precio destacado
    ult = closes[-1]
    yu = _y(ult)
    xu = _x(n - 1)
    dot = (f'<circle cx="{xu:.1f}" cy="{yu:.1f}" r="7" fill="{INK}"/>'
           f'<text x="{xu - 10:.1f}" y="{yu - 14:.1f}" class="px-lbl" '
           f'text-anchor="end">{_ratio(ult)}</text>')

    f0 = str(px.iloc[0]["fecha"])[:10]
    f1 = str(px.iloc[-1]["fecha"])[:10]

    return f'''<svg viewBox="0 0 {w} {h}" class="chart" xmlns="http://www.w3.org/2000/svg">
      {res_svg}
      {sop_svg}
      <polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2.6"
                stroke-linejoin="round" stroke-linecap="round"/>
      {dot}
      <text x="{pad}" y="{h - 16}" class="ax-lbl">{f0}</text>
      <text x="{w - pad}" y="{h - 16}" class="ax-lbl" text-anchor="end">{f1}</text>
    </svg>'''


def _chart_per_sector(serie, w=1000, h=262) -> str:
    """PER del TICKER (linea) vs PER mediano del SECTOR (barras) en UN solo eje
    (los dos son PER -> comparables). Barras = benchmark del sector; linea = el
    ticker encima. Muestra si cotiza con premio/descuento vs sus pares."""
    rows = [r for r in serie
            if r.get("per_t") is not None or r.get("per_s") is not None]
    if len(rows) < 2:
        return '<div class="sin-datos">serie de valuacion insuficiente</div>'
    n = len(rows)
    vals = [r["per_t"] for r in rows if r.get("per_t") is not None] + \
           [r["per_s"] for r in rows if r.get("per_s") is not None]
    hi = (max(vals) or 1.0) * 1.14
    pad_l, pad_r, pad_t, pad_b = 18, 18, 22, 42

    def _x(i):
        return pad_l + (w - pad_l - pad_r) * (i + 0.5) / n

    def _y(v):
        return (h - pad_b) - (h - pad_t - pad_b) * v / hi

    y_base = _y(0.0)
    bw = (w - pad_l - pad_r) / n * 0.62

    # Barras = PER mediano del sector (benchmark)
    bars = ""
    for i, r in enumerate(rows):
        v = r.get("per_s")
        if v is None:
            continue
        x = _x(i) - bw / 2
        yv = _y(v)
        bars += (f'<rect x="{x:.1f}" y="{yv:.1f}" width="{bw:.1f}" '
                 f'height="{max(y_base - yv, 1):.1f}" rx="4" fill="{SECTOR_BAR}"/>')
        bars += (f'<text x="{_x(i):.1f}" y="{yv - 8:.1f}" class="s-lbl" '
                 f'text-anchor="middle">{v:.0f}</text>')

    # Linea = PER del ticker (encima)
    idxs = [i for i, r in enumerate(rows) if r.get("per_t") is not None]
    pts = " ".join(f"{_x(i):.1f},{_y(rows[i]['per_t']):.1f}" for i in idxs)
    marks = "".join(
        f'<circle cx="{_x(i):.1f}" cy="{_y(rows[i]["per_t"]):.1f}" r="5" fill="{INK}"/>'
        for i in idxs)
    # Solo el ultimo (PER actual del ticker); los del sector se etiquetan en cada
    # barra. Evita la colision de labels en el primer punto (line ~= bar).
    et = ""
    if idxs:
        i1 = idxs[-1]
        et = (f'<text x="{_x(i1):.1f}" y="{_y(rows[i1]["per_t"]) - 15:.1f}" '
              f'class="t-lbl" text-anchor="end">{rows[i1]["per_t"]:.0f}</text>')

    ejex = "".join(
        f'<text x="{_x(i):.1f}" y="{h - 10}" class="ax-lbl" text-anchor="middle">{r["lbl"]}</text>'
        for i, r in enumerate(rows))

    return f'''<svg viewBox="0 0 {w} {h}" class="chart" xmlns="http://www.w3.org/2000/svg">
      {bars}
      <polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2.8"
                stroke-linejoin="round"/>
      {marks}{et}{ejex}
    </svg>'''


# ── Chips de opciones ─────────────────────────────────────────────────────────

def _clase_pcr_vol(v):
    v = _f(v)
    if v is None:
        return "neutral", "-"
    if v < 0.7:
        return "alcista", f"{v:.2f}"
    if v > 1.0:
        return "bajista", f"{v:.2f}"
    return "neutral", f"{v:.2f}"


def _clase_oi(veredicto_oi):
    if veredicto_oi == "Alcista":
        return "alcista", "Alcista"
    if veredicto_oi == "Bajista":
        return "bajista", "Bajista"
    return "neutral", "-"


# ── Contexto para el template ─────────────────────────────────────────────────

def build_context(payload: dict, handle: str) -> dict:
    ticker = payload["ticker"]
    datos = payload["datos"]
    sintesis = payload["sintesis"]
    precio = payload["precio"]
    perfil = datos.get("perfil") or {}
    op = datos.get("opciones_plazo") or {}

    soporte = _mejor_muro(op, "put_wall")
    resistencia = _mejor_muro(op, "call_wall")

    filas_op = []
    hay_op = False
    for v in PLAZOS:
        b = op.get(v) or {}
        pcr_cls, pcr_txt = _clase_pcr_vol(b.get("pcr_vol"))
        oi_cls, oi_txt = _clase_oi(b.get("veredicto_oi"))
        if b.get("pcr_vol") is not None or b.get("veredicto_oi") is not None:
            hay_op = True
        filas_op.append({
            "plazo": PLAZO_LBL[v],
            "pcr_cls": pcr_cls, "pcr_txt": pcr_txt,
            "oi_cls": oi_cls, "oi_txt": oi_txt,
        })

    estado = sintesis.get("estado", "NEUTRAL")

    return {
        "ticker": ticker,
        "sector": perfil.get("sector") or "-",
        "cierre": _ratio(precio.get("close")),
        "fecha_cierre": str(precio.get("fecha"))[:10] if precio.get("fecha") else "-",
        "estado": estado,
        "estado_color": COLOR_ESTADO.get(estado, "#6b7280"),
        "frase": sintesis.get("frase", ""),
        "chart_precio": _chart_precio_muros(payload["px"], soporte, resistencia),
        "hay_muros": bool(soporte or resistencia),
        "filas_op": filas_op,
        "hay_op": hay_op,
        "chart_fund": _chart_per_sector(payload["serie_val"]),
        "handle": handle,
        "fecha_legible": datetime.now().strftime("%d/%m/%Y"),
    }


# ── Render ────────────────────────────────────────────────────────────────────

def render_html(ctx: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    tpl = env.get_template("infografia_simple.html")
    return tpl.render(**ctx)


def html_a_png(html_str: str, output_path: Path) -> None:
    html_obj = HTML(string=html_str, base_url=str(TEMPLATES_DIR))
    pdf_bytes = html_obj.write_pdf()
    import fitz  # PyMuPDF: PDF -> PNG
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x
    pix.save(str(output_path))
    doc.close()


def generar_infografia_simple(ticker: str, output: str | None = None,
                              handle: str = DEFAULT_HANDLE) -> Path:
    """API publica reutilizable (dashboard): genera el PNG y devuelve su ruta."""
    ticker = ticker.upper()
    payload = cargar_datos(ticker)
    ctx = build_context(payload, handle)
    html_str = render_html(ctx)
    out = Path(output) if output else (
        SCRIPT_DIR / "output" / f"{ticker}_{ctx['fecha_cierre']}_simple.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    html_a_png(html_str, out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Infografia simple (tecnico + fundamental)")
    ap.add_argument("ticker", help="Ticker (ej NVDA)")
    ap.add_argument("-o", "--output", default=None, help="Ruta PNG de salida")
    ap.add_argument("--handle", default=DEFAULT_HANDLE)
    args = ap.parse_args()
    out = generar_infografia_simple(args.ticker, args.output, args.handle)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
