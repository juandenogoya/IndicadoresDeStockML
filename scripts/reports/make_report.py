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
import sys
from pathlib import Path

try:
    import yaml
    from jinja2 import Environment, FileSystemLoader
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
                  "veredicto", "niveles", "sesgo", "indicadores",
                  "opciones_ventanas", "conclusion"]
    faltantes = [k for k in requeridos if k not in data]
    if faltantes:
        raise ValueError(f"faltan campos requeridos en YAML: {faltantes}")

    # Defaults opcionales
    data.setdefault("nombre", "")
    data.setdefault("var_5d", None)
    data.setdefault("fecha", data["fecha_legible"])
    data.setdefault("strikes_calientes", [])
    data["sesgo"].setdefault("iv_skew_valor", None)

    return data


def renderizar_html(data: dict) -> str:
    """Renderiza el template Jinja2 con los datos cargados."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera un PDF de análisis técnico desde un YAML.",
    )
    parser.add_argument("input", type=Path, help="archivo YAML con los datos")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="ruta del PDF de salida (default: mismo nombre que el YAML, .pdf)",
    )
    args = parser.parse_args()

    try:
        data = cargar_yaml(args.input)
    except Exception as exc:
        print(f"ERROR cargando YAML: {exc}", file=sys.stderr)
        return 1

    output = args.output or args.input.with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        html = renderizar_html(data)
        generar_pdf(html, output)
    except Exception as exc:
        print(f"ERROR generando PDF: {exc}", file=sys.stderr)
        return 2

    print(f"OK: PDF generado en {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
