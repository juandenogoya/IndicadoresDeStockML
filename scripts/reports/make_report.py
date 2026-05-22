"""
make_report.py
Genera un PDF de análisis técnico (formato shareable para X)
a partir de un archivo YAML con los datos del análisis.

Uso:
    python make_report.py <input.yaml> [-o output.pdf]

Si no se pasa --output, el PDF se genera junto al YAML con el mismo nombre.

Ejemplo:
    python make_report.py examples/ul_2026-05-16.yaml
    -> genera examples/ul_2026-05-16.pdf

Dependencias: ver requirements.txt en este directorio.
    pip install -r scripts/reports/requirements.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
    import markdown as md
    from jinja2 import Environment, FileSystemLoader
    from markupsafe import Markup
    from weasyprint import HTML, CSS
except ImportError as exc:
    print(f"ERROR: falta instalar dependencias: {exc}", file=sys.stderr)
    print("Ejecutá: pip install -r scripts/reports/requirements.txt", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR    = Path(__file__).resolve().parent
TEMPLATES_DIR = SCRIPT_DIR / "templates"


def cargar_yaml(path: Path) -> dict:
    """Carga y valida básicamente el YAML de input."""
    if not path.is_file():
        raise FileNotFoundError(f"no existe: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} no contiene un mapping YAML")

    # Campos requeridos mínimos para no fallar en el template
    requeridos = ["ticker", "fecha_legible", "handle", "precio",
                  "niveles", "sesgo", "indicadores", "opciones_ventanas"]
    faltantes = [k for k in requeridos if k not in data]
    if faltantes:
        raise ValueError(f"faltan campos requeridos en YAML: {faltantes}")

    # Defaults opcionales
    data.setdefault("nombre", "")
    data.setdefault("var_5d", None)
    data.setdefault("fecha", data["fecha_legible"])
    data.setdefault("strikes_calientes", [])
    data.setdefault("narrativa", "")
    data["sesgo"].setdefault("iv_skew_valor", None)

    return data


def cargar_narrativa(yaml_path: Path, data: dict) -> str:
    """
    Carga la narrativa desde un .md hermano del YAML.

    Si existe `<base>.md` al lado del YAML, se usa su contenido (preferido).
    Si no existe pero el YAML tenia un campo `narrativa`, se usa eso
    (backward-compat con el formato anterior).
    """
    md_path = yaml_path.with_suffix(".md")
    if md_path.is_file():
        return md_path.read_text(encoding="utf-8")
    return data.get("narrativa", "") or ""


# ── Preprocesador de markdown del LLM ─────────────────────────────────────────
# El output de Gemini CLI tiene patrones no-estandar de markdown:
#   - Prefijo "✦ " al inicio
#   - Indentacion uniforme de 2-3 espacios por linea
#   - Bullets con "* " en vez de "- "
#   - Secciones numeradas "1. Titulo" en vez de "## Titulo"
# Este preprocesador limpia esos patrones para que el parser markdown
# los reconozca correctamente.

_GEMINI_PREFIX = "✦"

def _preprocesar_markdown(texto: str) -> str:
    """Limpia el output del LLM para que sea markdown estandar."""
    if not texto:
        return texto

    texto = texto.strip()
    if texto.startswith(_GEMINI_PREFIX):
        texto = texto.lstrip(_GEMINI_PREFIX).lstrip()

    out = []
    for raw in texto.split("\n"):
        # Quitar indentacion uniforme -- el LLM indenta todo
        line = raw.lstrip()

        # Bullets: "* item" -> "- item"
        if line.startswith("* "):
            out.append("- " + line[2:])
            continue

        # Secciones numeradas cortas: "1. Titulo" -> "## Titulo"
        # Solo si no termina con ":" (eso es subtitulo, no seccion principal)
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m and len(line) < 80 and not line.rstrip().endswith(":"):
            out.append("## " + m.group(2))
            continue

        out.append(line)

    return "\n".join(out)


def _markdown_filter(text: str | None) -> Markup:
    """
    Convierte markdown a HTML seguro para inyectar en el template.
    Aplica el preprocesador antes (limpia output del LLM si vino crudo).
    """
    if not text:
        return Markup("")
    text = _preprocesar_markdown(text)
    html = md.markdown(
        text,
        extensions=["extra", "sane_lists"],
        output_format="html",
    )
    return Markup(html)


def renderizar_html(data: dict) -> str:
    """Renderiza el template Jinja2 con los datos cargados."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["markdown"] = _markdown_filter
    template = env.get_template("report.html")
    return template.render(**data)


def generar_pdf(html: str, output: Path) -> None:
    """Convierte el HTML renderizado a PDF aplicando el CSS local."""
    css_path = TEMPLATES_DIR / "styles.css"
    # base_url permite que <link rel="stylesheet" href="styles.css"> resuelva
    HTML(string=html, base_url=str(TEMPLATES_DIR)).write_pdf(
        target=str(output),
        stylesheets=[CSS(filename=str(css_path))],
    )


def generar_pngs(pdf_path: Path, dpi: int = 150) -> list[Path]:
    """
    Convierte cada pagina del PDF a un PNG (para subir a X, que no
    soporta PDF como adjunto).

    Naming: <pdf_stem>_p1.png, _p2.png, ...
    Resolucion default 150 DPI (~1240x1754 px en A4) -- sharp en mobile.
    """
    try:
        import pymupdf  # nombre nuevo del paquete
    except ImportError:
        import fitz as pymupdf   # fallback al alias clasico

    doc = pymupdf.open(str(pdf_path))
    paths: list[Path] = []
    zoom = dpi / 72                 # 72 es el DPI base de PDF
    mat = pymupdf.Matrix(zoom, zoom)
    try:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = pdf_path.with_name(f"{pdf_path.stem}_p{i}.png")
            pix.save(str(out))
            paths.append(out)
    finally:
        doc.close()
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera un PDF de análisis técnico desde un YAML.",
    )
    parser.add_argument("input", type=Path, help="archivo YAML con los datos")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="ruta del PDF de salida (default: mismo nombre que el YAML, .pdf)",
    )
    parser.add_argument(
        "--no-png", action="store_true",
        help="solo generar PDF (por defecto se generan tambien PNGs por pagina)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="resolucion de los PNGs (default 150 -- mas alto = mas grande)",
    )
    args = parser.parse_args()

    try:
        data = cargar_yaml(args.input)
    except Exception as exc:
        print(f"ERROR cargando YAML: {exc}", file=sys.stderr)
        return 1

    # Cargar narrativa desde .md hermano (preferido) o del propio YAML
    data["narrativa"] = cargar_narrativa(args.input, data)

    output = args.output or args.input.with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        html = renderizar_html(data)
        generar_pdf(html, output)
    except Exception as exc:
        print(f"ERROR generando PDF: {exc}", file=sys.stderr)
        return 2

    print(f"OK: PDF generado en {output}")

    # Generar PNGs por pagina (a menos que se pase --no-png)
    if not args.no_png:
        try:
            pngs = generar_pngs(output, dpi=args.dpi)
        except Exception as exc:
            print(f"ADVERTENCIA: PDF OK pero fallo la conversion a PNG: {exc}",
                  file=sys.stderr)
            return 0
        for p in pngs:
            print(f"OK: PNG generado en {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
