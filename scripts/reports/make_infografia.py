"""
make_infografia.py
Genera una infografía single-image (PNG) para subir a X.

Toma el mismo YAML que make_report.py (datos del MCP). NO usa el .md
de narrativa -- la infografía es 100% datos + reglas (sin LLM).

Uso:
    python make_infografia.py <input.yaml> [-o output.png]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

try:
    import yaml
    from jinja2 import Environment, FileSystemLoader
    from weasyprint import HTML, CSS
except ImportError as exc:
    print(f"ERROR: falta dependencia: {exc}", file=sys.stderr)
    print("Ejecutá: pip install -r scripts/reports/requirements.txt", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR    = Path(__file__).resolve().parent
TEMPLATES_DIR = SCRIPT_DIR / "templates"
DEFAULT_HANDLE = "@juan_de_nogoya"


def _es_ticker(arg: str) -> bool:
    """True si el argumento parece un ticker, no una ruta a un .yaml."""
    p = Path(arg)
    if p.suffix.lower() in (".yaml", ".yml"):
        return False
    if p.exists():
        return False
    return arg.isalnum() and len(arg) <= 6


def obtener_data(input_arg: str, handle: str) -> tuple[dict, Path]:
    """
    Devuelve (data, ruta_base_sin_extension).

    - Si input_arg es un ticker -> consulta el MCP y arma los datos.
    - Si es una ruta .yaml      -> carga el archivo.
    """
    if _es_ticker(input_arg):
        # Reusar la logica de build_yaml (fetch + transformacion)
        import build_yaml   # mismo directorio
        ticker = input_arg.upper()
        ov, opts = asyncio.run(build_yaml._fetch(ticker))
        data = build_yaml.build_data(ticker, ov, opts, handle)
        base = SCRIPT_DIR / "output" / f"{ticker}_{data['fecha']}"
        return data, base

    yaml_path = Path(input_arg)
    data = cargar_yaml(yaml_path)
    return data, yaml_path.with_suffix("")


# ── Carga del YAML ────────────────────────────────────────────────────────────

def cargar_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"no existe: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} no contiene un mapping YAML")

    requeridos = ["ticker", "fecha_legible", "handle", "precio",
                  "niveles", "sesgo", "indicadores", "opciones_ventanas"]
    faltantes = [k for k in requeridos if k not in data]
    if faltantes:
        raise ValueError(f"faltan campos requeridos en YAML: {faltantes}")

    data.setdefault("nombre", "")
    data.setdefault("sector", "")
    data.setdefault("var_5d", None)
    data.setdefault("fecha", data["fecha_legible"])
    data.setdefault("strikes_calientes", [])
    data.setdefault("indicadores_pa", None)   # opcional: estructura/patron/etc.
    data["sesgo"].setdefault("iv_skew_valor", None)

    return data


# ── Lógica de "frase combinada" (regla, sin LLM) ─────────────────────────────

def _normalizar_sesgo(s: str) -> str:
    """alcista/neutro/bajista/lateral/sin datos -> categoria base."""
    if not s:
        return "sin-datos"
    s = s.lower().strip()
    if "alcista" in s:
        return "alcista"
    if "bajista" in s:
        return "bajista"
    if "neutr" in s or "lateral" in s:
        return "neutro"
    return "sin-datos"


_MATRIZ_FRASE = {
    # (indicadores, opciones) -> conclusión cerrada (nunca "mirar X")
    ("alcista", "alcista"): "Confluencia alcista: técnico y opciones alineados al alza.",
    ("alcista", "bajista"): "Divergencia: técnico alcista pero opciones defensivas — cautela.",
    ("alcista", "neutro"):  "Sesgo alcista en técnico; opciones todavía sin confirmar.",
    ("bajista", "alcista"): "Divergencia: técnico débil pero opciones optimistas — posible giro.",
    ("bajista", "bajista"): "Confluencia bajista: técnico y opciones apuntan a la baja.",
    ("bajista", "neutro"):  "Sesgo bajista en técnico; opciones sin confirmar el movimiento.",
    ("neutro",  "alcista"): "Consolidación técnica con opciones alcistas: sesgo levemente positivo.",
    ("neutro",  "bajista"): "Consolidación técnica con opciones defensivas: sesgo levemente negativo.",
    ("neutro",  "neutro"):  "Sin convicción direccional: consolidación en técnico y opciones.",
}


def frase_combinada(sesgo: dict) -> str:
    """
    Genera una conclusión corta combinando los sesgos de indicadores y
    opciones (con matiz de IV skew). Reglas determinísticas, sin LLM.

    Toda frase es una CONCLUSION cerrada -- nunca deriva al lector a
    "mirar X", porque ese dato ya está en el panel.
    """
    i = _normalizar_sesgo(sesgo.get("indicadores"))
    o = _normalizar_sesgo(sesgo.get("opciones"))
    s = _normalizar_sesgo(sesgo.get("iv_skew"))

    base = _MATRIZ_FRASE.get((i, o))

    if base is None:
        # Algún sesgo sin datos
        if i == "sin-datos" and o == "sin-datos":
            return "Datos insuficientes para un veredicto combinado."
        if i == "sin-datos":
            return f"Opciones con sesgo {o}; técnico sin datos suficientes."
        if o == "sin-datos":
            return f"Técnico con sesgo {i}; opciones sin datos suficientes."
        return "Cuadro mixto entre técnico y opciones."

    # Matiz de IV skew: solo cuando agrega contraste a una lectura alcista
    if s == "bajista" and i == "alcista" and o == "alcista":
        base += " El IV skew marca cierta cobertura de fondo."

    return base


# ── Frase de Price Action (patrón + estructura) ───────────────────────────────

def _clasificar_patron(patron: str | None) -> str | None:
    """Clasifica el patrón de vela en alcista/bajista/indecision/momentum."""
    if not patron or "sin patr" in patron.lower():
        return None
    p = patron.lower()
    if p in ("engulfing_bull", "hammer"):
        return "alcista"
    if p in ("engulfing_bear", "shooting_star"):
        return "bajista"
    if p == "doji":
        return "indecision"
    if p == "marubozu":
        return "momentum"
    return None


def frase_price_action(pa: dict | None) -> str | None:
    """
    Conclusión corta combinando patrón de vela + estructura SMC.
    El caso mas util es patrón de reversion vs estructura opuesta:
    señala un posible giro NO confirmado estructuralmente.
    """
    if not pa:
        return None

    patron = _clasificar_patron(pa.get("patron"))
    estr   = _normalizar_sesgo(pa.get("estructura_5"))

    # Patrón de reversión contra la estructura (lo mas informativo)
    if patron == "alcista" and estr == "bajista":
        return "Señal de reversión alcista dentro de estructura bajista: posible giro sin confirmar."
    if patron == "bajista" and estr == "alcista":
        return "Señal de agotamiento en estructura alcista: posible techo en formación."

    # Patrón a favor de la estructura (continuación)
    if patron == "alcista" and estr == "alcista":
        return "Patrón alcista en estructura alcista: continuación al alza."
    if patron == "bajista" and estr == "bajista":
        return "Patrón bajista en estructura bajista: continuación a la baja."

    # Patrón en rango
    if patron == "alcista" and estr == "neutro":
        return "Patrón alcista en rango: posible salida al alza."
    if patron == "bajista" and estr == "neutro":
        return "Patrón bajista en rango: posible quiebre a la baja."

    # Patrones especiales
    if patron == "indecision":
        return "Vela de indecisión: el mercado no define dirección."
    if patron == "momentum":
        return f"Vela de fuerte momentum en estructura {estr}."

    # Sin patrón -> apoyarse en la estructura
    if estr == "alcista":
        return "Sin patrón de vela destacado; estructura alcista vigente."
    if estr == "bajista":
        return "Sin patrón de vela destacado; estructura bajista vigente."
    if estr == "neutro":
        return "Sin patrón ni estructura definida: consolidación."
    return None


# ── Rendering ────────────────────────────────────────────────────────────────

def renderizar_html(data: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("infografia.html")

    # Frases interpretativas (reglas, sin LLM)
    data["frase_combinada"] = frase_combinada(data.get("sesgo", {}))

    # Bloque PA: si vino info adicional la usamos; sino dejamos None
    if "pa" not in data:
        data["pa"] = None
    data["frase_pa"] = frase_price_action(data.get("pa"))

    return template.render(**data)


def generar_pdf(html: str, output: Path) -> None:
    css_path = TEMPLATES_DIR / "infografia.css"
    HTML(string=html, base_url=str(TEMPLATES_DIR)).write_pdf(
        target=str(output),
        stylesheets=[CSS(filename=str(css_path))],
    )


def generar_png(pdf_path: Path, dpi: int = 144) -> Path:
    """
    Convierte el PDF (1 sola pagina) a PNG.

    Nota: el PDF generado mide 1080x1500 px. A 144 DPI (96 base × 1.5)
    el PNG queda en ~1620x2250 px -- mas que suficiente para X.
    """
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    doc = pymupdf.open(str(pdf_path))
    zoom = dpi / 72
    mat = pymupdf.Matrix(zoom, zoom)
    try:
        page = doc[0]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out = pdf_path.with_suffix(".png")
        pix.save(str(out))
        return out
    finally:
        doc.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera infografía PNG (single-image) para X desde un YAML.",
    )
    parser.add_argument("input", help="ticker (ej: BAC) o ruta a un .yaml ya generado")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="ruta del PNG (default: <base>_ig.png en scripts/reports/output/)",
    )
    parser.add_argument(
        "--handle", default=DEFAULT_HANDLE,
        help=f"handle de X (solo aplica si pasás un ticker; default {DEFAULT_HANDLE})",
    )
    parser.add_argument(
        "--dpi", type=int, default=144,
        help="resolucion del PNG (default 144)",
    )
    parser.add_argument(
        "--keep-pdf", action="store_true",
        help="conservar el PDF intermedio (default: se borra)",
    )
    args = parser.parse_args()

    try:
        data, base = obtener_data(args.input, args.handle)
    except Exception as exc:
        print(f"ERROR obteniendo datos: {exc}", file=sys.stderr)
        return 1

    output_png = args.output or base.with_name(f"{base.name}_ig.png")
    output_pdf = output_png.with_suffix(".pdf")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        html = renderizar_html(data)
        generar_pdf(html, output_pdf)
        generar_png(output_pdf, dpi=args.dpi)
    except Exception as exc:
        print(f"ERROR generando infografía: {exc}", file=sys.stderr)
        return 2

    if not args.keep_pdf:
        output_pdf.unlink(missing_ok=True)

    print(f"OK: PNG generado en {output_png}")
    if args.keep_pdf:
        print(f"    (PDF conservado en {output_pdf})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
