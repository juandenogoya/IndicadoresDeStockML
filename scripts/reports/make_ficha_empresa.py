"""
make_ficha_empresa.py
Genera la FICHA DE EMPRESA (PNG): una tarjeta "presentacion de la empresa",
fondo oscuro, con los datos PROPIOS del ultimo trimestre reportado y su
comparacion contra el MISMO trimestre del ano anterior (YoY). SIN pares / sin
benchmark sectorial (decision del usuario 4/8/2026): la empresa contra si misma.

Datos REALES, LOCAL (Plan C), sin LLM, sin estimaciones:
  - fundamentales_ratios_q : YoY %, margenes, ROE/ROIC (o ROTCE/efficiency en
    bancos), FCF, solvencia, valuacion al cierre (_px), perfil, moneda.
  - fundamentales_income_q : el ABSOLUTO trimestral (ingresos y resultado neto
    del Q, no TTM) para el "Q vs mismo Q del ano anterior".
  - activos                : nombre de la empresa.

Adapta las tiles por `profile` (bancos: ROE/ROTCE/efficiency; no-bancos:
margenes/ROIC/FCF). Absolutos en la moneda de reporte de la empresa.

Uso:
    python scripts/reports/make_ficha_empresa.py MELI
    python scripts/reports/make_ficha_empresa.py JPM -o salida.png
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

TEMPLATES_DIR = SCRIPT_DIR / "templates"
DEFAULT_HANDLE = "@juan_de_nogoya"


# -- Helpers de formato -------------------------------------------------------

def _f(v):
    if v is None:
        return None
    try:
        fv = float(v)
        return None if fv != fv else fv  # NaN guard
    except (TypeError, ValueError):
        return None


def _sym(cur) -> str:
    return "US$" if cur == "USD" else (cur or "")


def _money(v, cur) -> str:
    """Abrevia a B/M/K en la moneda de reporte. Ej 8.85e9 -> 'US$ 8.85 B'."""
    v = _f(v)
    if v is None:
        return "-"
    neg = v < 0
    a = abs(v)
    if a >= 1e9:
        s = f"{a / 1e9:.2f} B"
    elif a >= 1e6:
        s = f"{a / 1e6:.1f} M"
    elif a >= 1e3:
        s = f"{a / 1e3:.1f} K"
    else:
        s = f"{a:.0f}"
    return f"{_sym(cur)} {'-' if neg else ''}{s}"


def _per_share(v, cur) -> str:
    v = _f(v)
    return "-" if v is None else f"{_sym(cur)} {v:.2f}"


def _pct(v) -> str:
    """Fraccion -> porcentaje. Ej 0.4386 -> '43.9%'."""
    v = _f(v)
    return "-" if v is None else f"{v * 100:.1f}%"


def _pct_yoy(v) -> str:
    """Fraccion -> delta YoY con signo. Ej 0.49 -> '+49.0% YoY'."""
    v = _f(v)
    return "-" if v is None else f"{v * 100:+.1f}% YoY"


def _mult(v) -> str:
    v = _f(v)
    return "-" if v is None else f"{v:.1f}x"


def _ratio(v) -> str:
    v = _f(v)
    return "-" if v is None else f"{v:.2f}"


def _cls(v) -> str:
    """Clase de color por signo (up/down/neutral) para deltas YoY."""
    v = _f(v)
    if v is None:
        return "neutral"
    return "up" if v > 0 else ("down" if v < 0 else "neutral")


def _tile(label, valor, sub=None, sub_cls="neutral") -> dict:
    return {"label": label, "valor": valor, "sub": sub, "sub_cls": sub_cls}


def _pref(v, either_px, fiscal) -> float | None:
    """Prefiere el multiplo al cierre (_px); si falta (ADR no-USD), usa el fiscal."""
    return _f(either_px) if _f(either_px) is not None else _f(fiscal)


# -- Carga de datos -----------------------------------------------------------

def cargar_datos(ticker: str) -> dict:
    ticker = ticker.strip().upper()

    # ANCLA = ultimo trimestre con estado de resultados REAL (revenue no NULL).
    # El trimestre mas reciente suele ser un "stub" recien reportado (yahooquery
    # trae el EPS antes que el income statement completo) -> tomar LIMIT 1 a secas
    # agarraria esa fila vacia. Se ancla al ultimo Q efectivamente reportado.
    inc = query_df(
        "SELECT fiscal_period_end, total_revenue, net_income "
        "FROM fundamentales_income_q WHERE ticker = :t AND total_revenue IS NOT NULL "
        "ORDER BY fiscal_period_end DESC LIMIT 1",
        params={"t": ticker},
    )
    if inc.empty:  # fallback: sin ningun income completo, tomo el ultimo que haya
        inc = query_df(
            "SELECT fiscal_period_end, total_revenue, net_income "
            "FROM fundamentales_income_q WHERE ticker = :t "
            "ORDER BY fiscal_period_end DESC LIMIT 1",
            params={"t": ticker},
        )
    income_q = inc.iloc[0].to_dict() if not inc.empty else {}
    anchor = income_q.get("fiscal_period_end")

    # Fila de fundamentales = ratios del trimestre ancla (fallback: el ultimo).
    row = {}
    if anchor is not None:
        r = query_df(
            "SELECT * FROM fundamentales_ratios_q WHERE ticker = :t "
            "AND fiscal_period_end = :f LIMIT 1",
            params={"t": ticker, "f": anchor},
        )
        row = r.iloc[0].to_dict() if not r.empty else {}
    if not row:
        r = query_df(
            "SELECT * FROM fundamentales_ratios_q WHERE ticker = :t "
            "ORDER BY fiscal_period_end DESC LIMIT 1", params={"t": ticker})
        if r.empty:
            raise SystemExit(f"Sin fundamentales para {ticker} (fundamentales_ratios_q vacia).")
        row = r.iloc[0].to_dict()

    # Valuacion = fila MAS reciente (el _px trae el precio al cierre de hoy, que
    # se recomputa sobre el ultimo trimestre; puede diferir del ancla).
    vr = query_df(
        "SELECT pe_ratio, pb_ratio, ps_ratio, ev_ebitda, "
        "pe_ratio_px, pb_ratio_px, ps_ratio_px, ev_ebitda_px, precio_px, fecha_px "
        "FROM fundamentales_ratios_q WHERE ticker = :t "
        "ORDER BY fiscal_period_end DESC LIMIT 1", params={"t": ticker})
    val_row = vr.iloc[0].to_dict() if not vr.empty else row

    nom = query_df("SELECT nombre FROM activos WHERE ticker = :t",
                   params={"t": ticker})
    nombre = nom.iloc[0]["nombre"] if not nom.empty else ""

    return {"ticker": ticker, "row": row, "val_row": val_row,
            "income_q": income_q, "nombre": nombre}


# -- Secciones (tiles) --------------------------------------------------------

def build_sections(row: dict, income_q: dict, val_row: dict) -> list:
    cur = row.get("reporting_currency")
    banco = (row.get("profile") == "financiero")
    S = []

    # 1. Resultado del trimestre (Q vs mismo Q del ano anterior)
    S.append({
        "titulo": "Resultado del trimestre  (variacion interanual)",
        "tiles": [
            _tile("Ingresos", _money(income_q.get("total_revenue"), cur),
                  _pct_yoy(row.get("revenue_yoy_pct")), _cls(row.get("revenue_yoy_pct"))),
            _tile("Resultado neto", _money(income_q.get("net_income"), cur),
                  _pct_yoy(row.get("net_income_yoy_pct")), _cls(row.get("net_income_yoy_pct"))),
            _tile("BPA (trimestre)", _per_share(row.get("eps_q"), cur),
                  _pct_yoy(row.get("eps_yoy_pct")), _cls(row.get("eps_yoy_pct"))),
        ],
    })

    # 2. Margenes y retornos (TTM) -- adaptado por perfil
    if banco:
        tiles = [
            _tile("ROE", _pct(row.get("roe_ttm")), "TTM"),
            _tile("ROTCE", _pct(row.get("rotce_ttm")), "TTM"),
            _tile("Efficiency ratio", _pct(row.get("efficiency_ratio_ttm")),
                  "menor es mejor"),
        ]
    else:
        tiles = [
            _tile("Margen bruto", _pct(row.get("gross_margin_ttm")), "TTM"),
            _tile("Margen operativo", _pct(row.get("operating_margin_ttm")),
                  _delta_pp(row.get("operating_margin_yoy_delta")),
                  _cls(row.get("operating_margin_yoy_delta"))),
            _tile("Margen neto", _pct(row.get("net_margin_ttm")),
                  _delta_pp(row.get("net_margin_yoy_delta")),
                  _cls(row.get("net_margin_yoy_delta"))),
            _tile("ROE", _pct(row.get("roe_ttm")), "TTM"),
            _tile("ROIC", _pct(row.get("roic_ttm")), "TTM"),
        ]
    S.append({"titulo": "Margenes y retornos  (TTM)", "tiles": tiles})

    # 3. Caja y solvencia (TTM) -- solo tiles con dato (bancos traen menos)
    caja = []
    if _f(row.get("fcf_ttm")) is not None:
        sub = f"margen {_pct(row.get('fcf_margin_ttm'))} . {_pct_yoy(row.get('fcf_yoy_pct'))}"
        caja.append(_tile("Flujo de caja libre", _money(row.get("fcf_ttm"), cur),
                          sub, _cls(row.get("fcf_yoy_pct"))))
    # Deuda neta: NO para bancos (TotalDebt-Cash no es una lectura de solvencia
    # significativa en un banco; su "deuda" incluye fondeo/depositos). Se muestra
    # solo en no-financieros. En bancos queda Deuda/Patrimonio (si bendecida).
    if not banco and _f(row.get("net_debt")) is not None:
        nd = _f(row.get("net_debt_to_ebitda_ttm"))
        sub = f"{nd:.2f}x EBITDA" if nd is not None else None
        caja.append(_tile("Deuda neta", _money(row.get("net_debt"), cur), sub))
    if _f(row.get("current_ratio")) is not None:
        caja.append(_tile("Current ratio", _ratio(row.get("current_ratio"))))
    if _f(row.get("debt_to_equity")) is not None:
        caja.append(_tile("Deuda / Patrimonio", _ratio(row.get("debt_to_equity"))))
    if caja:
        S.append({"titulo": "Caja y solvencia  (TTM)", "tiles": caja})

    # 4. Valuacion (al cierre) -- tira de contexto (provisional). Usa val_row
    # (fila mas reciente) porque el _px trae el precio al cierre de hoy.
    val = [
        _tile("PER", _mult(_pref(None, val_row.get("pe_ratio_px"), val_row.get("pe_ratio")))),
        _tile("P / Ventas", _mult(_pref(None, val_row.get("ps_ratio_px"), val_row.get("ps_ratio")))),
        _tile("EV / EBITDA", _mult(_pref(None, val_row.get("ev_ebitda_px"), val_row.get("ev_ebitda")))),
        _tile("P / VL", _mult(_pref(None, val_row.get("pb_ratio_px"), val_row.get("pb_ratio")))),
    ]
    fpx = val_row.get("fecha_px")
    ftxt = str(fpx)[:10] if fpx else "cierre"
    S.append({"titulo": f"Valuacion  (al cierre {ftxt})", "tiles": val})

    return S


def _delta_pp(v) -> str | None:
    """Delta de margen YoY en puntos porcentuales (ya viene como fraccion)."""
    v = _f(v)
    return None if v is None else f"{v * 100:+.1f} pp YoY"


def _q_label(fpe) -> str:
    d = fpe if hasattr(fpe, "month") else datetime.fromisoformat(str(fpe)[:10]).date()
    q = (d.month - 1) // 3 + 1
    return f"Q{q} {d.year} . cierre {d.strftime('%d/%m/%Y')}"


# -- Contexto -----------------------------------------------------------------

def build_context(payload: dict, handle: str) -> dict:
    row = payload["row"]
    val_row = payload["val_row"]
    cur = row.get("reporting_currency")
    sector = row.get("sector") or "-"
    industry = row.get("industry")
    sec_txt = f"{sector} . {industry}" if industry else sector

    # Nombre: ocultar si es igual al ticker (activos a veces guarda el ticker)
    nombre = payload["nombre"] or ""
    if nombre.strip().upper() == payload["ticker"]:
        nombre = ""

    return {
        "ticker": payload["ticker"],
        "nombre": nombre,
        "sector": sec_txt,
        "precio": _per_share(val_row.get("precio_px"), cur),
        "fecha_px": str(val_row.get("fecha_px"))[:10] if val_row.get("fecha_px") else "-",
        "periodo": _q_label(row.get("fiscal_period_end")),
        "moneda": cur or "-",
        "sections": build_sections(row, payload["income_q"], val_row),
        "handle": handle,
        "fecha_legible": datetime.now().strftime("%d/%m/%Y"),
    }


# -- Render -------------------------------------------------------------------

def render_html(ctx: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    tpl = env.get_template("ficha_empresa.html")
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


def generar_ficha_empresa(ticker: str, output: str | None = None,
                          handle: str = DEFAULT_HANDLE) -> Path:
    """API publica reutilizable (dashboard): genera el PNG y devuelve su ruta."""
    ticker = ticker.upper()
    payload = cargar_datos(ticker)
    ctx = build_context(payload, handle)
    html_str = render_html(ctx)
    out = Path(output) if output else (
        SCRIPT_DIR / "output" / f"{ticker}_{ctx['fecha_px']}_ficha.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    html_a_png(html_str, out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Ficha de empresa (presentacion, fondo oscuro)")
    ap.add_argument("ticker", help="Ticker (ej MELI)")
    ap.add_argument("-o", "--output", default=None, help="Ruta PNG de salida")
    ap.add_argument("--handle", default=DEFAULT_HANDLE)
    args = ap.parse_args()
    out = generar_ficha_empresa(args.ticker, args.output, args.handle)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
