"""
dashboard/export_jpg.py
Exporta el informe descriptivo a una imagen JPG (YYYYMMDD_TICKER.jpg).

Reusa la maquinaria de scripts/reports/: Jinja2 (HTML) -> WeasyPrint (PDF) ->
pymupdf (imagen). El contenido sale de dashboard/view.construir_vista(), la misma
fuente que usa el dashboard Streamlit (paridad garantizada).

Requiere las dependencias del venv del proyecto (weasyprint, pymupdf). Por eso el
dashboard se corre bajo el venv (ver run_dashboard.bat).
"""

import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jinja2 import Environment, FileSystemLoader

from dashboard.view import construir_vista
from dashboard.metricas import construir_papel

_DIR = Path(__file__).resolve().parent
_TEMPLATES = _DIR / "templates"
_OUTPUT = _DIR / "output"


def _fecha_archivo(datos: dict) -> str:
    """YYYYMMDD a partir de la fecha de cierre; hoy como fallback."""
    f = (datos.get("precio") or {}).get("fecha")
    try:
        return f.strftime("%Y%m%d")  # date/datetime
    except AttributeError:
        try:
            return str(f).replace("-", "")[:8] or date.today().strftime("%Y%m%d")
        except Exception:
            return date.today().strftime("%Y%m%d")


def _render_html(vista: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=True, trim_blocks=True, lstrip_blocks=True,
    )
    tpl = env.get_template("informe.html")
    return tpl.render(
        enc=vista["encabezado"],
        tec=vista["tecnico"],
        opciones=vista["opciones"],
        muros=vista["muros"],
        sector=vista["sector"],
        conclusion=vista["conclusion"],
    )


def _html_a_jpg(html: str, out_jpg: Path, dpi: int = 144) -> Path:
    """HTML -> PDF (WeasyPrint) -> JPG (pymupdf). Concatena paginas si hay >1."""
    from weasyprint import HTML, CSS
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    css = _TEMPLATES / "informe.css"
    pdf_bytes = HTML(string=html, base_url=str(_TEMPLATES)).write_pdf(
        stylesheets=[CSS(filename=str(css))]
    )

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = pymupdf.Matrix(zoom, zoom)
    try:
        if doc.page_count == 1:
            pix = doc[0].get_pixmap(matrix=mat, alpha=False)
            pix.save(str(out_jpg), jpg_quality=92)
        else:
            # Apilar verticalmente todas las paginas en una sola imagen.
            pixmaps = [doc[i].get_pixmap(matrix=mat, alpha=False)
                       for i in range(doc.page_count)]
            w = max(p.width for p in pixmaps)
            h = sum(p.height for p in pixmaps)
            canvas = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
            canvas.clear_with(255)
            y = 0
            for p in pixmaps:
                p.set_origin(0, y)
                canvas.copy(p, p.irect)
                y += p.height
            canvas.save(str(out_jpg), jpg_quality=92)
    finally:
        doc.close()
    return out_jpg


def generar_jpg(ticker: str, datos: dict, sintesis: dict,
                out_dir: Path = None) -> Path:
    """
    Genera la imagen JPG del informe y devuelve su ruta.
    Nombre: YYYYMMDD_TICKER.jpg (fecha = cierre del ticker).
    """
    vista = construir_vista(datos, sintesis)
    out_dir = Path(out_dir) if out_dir else _OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jpg = out_dir / f"{_fecha_archivo(datos)}_{ticker.upper()}.jpg"
    html = _render_html(vista)
    return _html_a_jpg(html, out_jpg)


def generar_papel_pdf(ticker: str, datos: dict, sintesis: dict,
                      out_dir: Path = None) -> Path:
    """
    Genera el PDF del papel de trabajo (documento metodologico, texto
    seleccionable) y devuelve su ruta. Nombre: YYYYMMDD_TICKER_papel.pdf.
    """
    from weasyprint import HTML, CSS

    vista = construir_vista(datos, sintesis)
    papel = construir_papel(datos, sintesis)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=True, trim_blocks=True, lstrip_blocks=True,
    )
    html = env.get_template("papel.html").render(enc=vista["encabezado"], papel=papel)

    out_dir = Path(out_dir) if out_dir else _OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"{_fecha_archivo(datos)}_{ticker.upper()}_papel.pdf"
    HTML(string=html, base_url=str(_TEMPLATES)).write_pdf(
        target=str(out_pdf), stylesheets=[CSS(filename=str(_TEMPLATES / "papel.css"))]
    )
    return out_pdf


# CLI util para probar sin Streamlit: python dashboard/export_jpg.py AAPL
if __name__ == "__main__":
    os.environ.pop("DATABASE_URL", None)
    from dashboard.sintesis_data import cargar_datos_ticker
    from src.utils.dashboard_sintesis import sintetizar

    tk = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()
    d = cargar_datos_ticker(tk)
    s = sintetizar(d)
    p = generar_jpg(tk, d, s)
    print(f"OK: {p}")
