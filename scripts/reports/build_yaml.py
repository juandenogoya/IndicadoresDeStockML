"""
build_yaml.py
Construye un YAML pre-cargado con los datos del análisis para make_report.py.

Llama directamente a las tools del MCP (get_ticker_overview + get_options_analysis)
y traduce las respuestas al formato que espera el template.

Deja `veredicto` y `conclusion` como TODO -- el usuario los completa después
(idealmente pidiéndoselos al LLM en Gemini CLI).

Uso:
    python build_yaml.py JPM
    python build_yaml.py JPM -o ruta/al/archivo.yaml
    python build_yaml.py JPM --handle @otro_handle

Output default: scripts/reports/output/<TICKER>_<YYYY-MM-DD>.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date
from pathlib import Path

try:
    import yaml
    from dotenv import load_dotenv
except ImportError as exc:
    print(f"ERROR: falta dependencia: {exc}", file=sys.stderr)
    print("Ejecutá: pip install -r scripts/reports/requirements.txt", file=sys.stderr)
    sys.exit(1)


# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]    # scripts/reports/ -> proyecto

# ── Cargar .env ANTES de importar mcp_server (pydantic-settings lee al import)
load_dotenv(PROJECT_ROOT / ".env")

# MCP espera DATABASE_URL; el .env del proyecto tiene MCP_READER_LOCAL_DSN.
# Aliasamos para que el pool del MCP funcione sin tocar el .env.
if not os.getenv("DATABASE_URL") and os.getenv("MCP_READER_LOCAL_DSN"):
    os.environ["DATABASE_URL"] = os.environ["MCP_READER_LOCAL_DSN"]

# Hacer importable mcp_server
sys.path.insert(0, str(PROJECT_ROOT))

# Importar tools del MCP
from mcp_server.tools.options  import get_options_analysis    # noqa: E402
from mcp_server.tools.overview import get_ticker_overview     # noqa: E402


DEFAULT_HANDLE = "@juan_de_nogoya"

_MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
          "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _fecha_legible(iso: str | None) -> str:
    """'2026-05-20' -> '20 mayo 2026'."""
    if not iso:
        return ""
    y, m, d = iso.split("-")
    return f"{int(d)} {_MESES[int(m) - 1]} {y}"


# ── Transformacion MCP -> YAML ────────────────────────────────────────────────

def build_data(ticker: str, ov: dict, opts: dict, handle: str) -> dict:
    """Combina overview + options analysis en el dict YAML del template."""
    if isinstance(opts, dict) and "error" in opts:
        raise RuntimeError(f"get_options_analysis: {opts['error']}")
    if isinstance(ov, dict) and "error" in ov:
        raise RuntimeError(f"get_ticker_overview: {ov['error']}")

    perfil   = ov.get("perfil")   or {}
    precio_s = ov.get("precio")   or {}
    tecnicos = ov.get("tecnicos") or {}
    señales  = tecnicos.get("señales") or {}

    precio     = precio_s.get("close")
    var_5d     = precio_s.get("variacion_5d_pct")
    fecha_snap = opts.get("fecha_snapshot")

    # ── Soporte/resistencia por ventana
    sr        = opts.get("soporte_resistencia") or {}
    ventanas  = sr.get("ventanas", [])

    # Tabla resumen niveles (pagina 1)
    niveles = []
    for v in ventanas:
        niveles.append({
            "ventana":     v["ventana"].capitalize(),
            "resistencia": (v["resistencia"]["zona"] if v.get("resistencia") else None),
            "soporte":     (v["soporte"]["zona"]     if v.get("soporte")     else None),
        })

    # ── Sesgo general
    # - Indicadores: la tendencia_sma sintetizada del overview
    # - Opciones: sesgo de la ventana corto plazo (lo mas relevante a corto)
    # - IV skew: directo del actual de tendencia_diaria
    sesgo_indicadores = señales.get("tendencia_sma", "sin datos")

    sesgo_opc = "sin datos"
    v_corto = next((v for v in ventanas if v["ventana"] == "corto"), None)
    if v_corto:
        sesgo_opc = v_corto.get("sesgo_pcr_oi", "sin datos")

    tend_diaria = opts.get("tendencia_diaria") or {}
    actual      = tend_diaria.get("actual") or {}
    iv_skew_v   = actual.get("iv_skew")
    iv_skew_lab = actual.get("sesgo_iv_skew", "sin datos")

    sesgo = {
        "indicadores":   sesgo_indicadores.upper(),
        "opciones":      sesgo_opc.upper(),
        "iv_skew":       iv_skew_lab.upper(),
        "iv_skew_valor": iv_skew_v,
    }

    # ── Indicadores (pagina 2)
    indicadores = {
        "rsi14":         tecnicos.get("rsi14") or 0.0,
        "rsi14_sesgo":   señales.get("rsi_estado", "sin datos"),
        "macd_hist":     tecnicos.get("macd_hist") or 0.0,
        "macd_sesgo":    señales.get("macd_direccion", "sin datos"),
        "sma_tendencia": señales.get("tendencia_sma", "sin datos"),
        # Distancia % del precio a cada SMA (positivo = precio sobre la media)
        "dist_sma21":    tecnicos.get("dist_sma21"),
        "dist_sma50":    tecnicos.get("dist_sma50"),
        "dist_sma200":   tecnicos.get("dist_sma200"),
        "adx":           tecnicos.get("adx") or 0.0,
        "adx_fuerza":    señales.get("adx_fuerza", "sin datos"),
    }

    # ── Opciones por ventana (pagina 2 -- detalle completo)
    opciones_ventanas = []
    for v in ventanas:
        opciones_ventanas.append({
            "ventana":      v["ventana"],
            "dias":         v["dias"],
            "valido":       v["valido"],
            "pcr_oi":       v.get("pcr_oi"),
            "sesgo_pcr_oi": v.get("sesgo_pcr_oi", "sin datos"),
            "resistencia":  v.get("resistencia"),
            "soporte":      v.get("soporte"),
        })

    # ── Price action + estructura (para el bloque PA de la infografía)
    pa_section = ov.get("price_action") or {}
    ms_section = ov.get("market_structure") or {}
    pa = {
        "patron":          pa_section.get("patron_activo") or "sin patrón destacado",
        "tendencia_velas": pa_section.get("tendencia_velas") or "—",
        "estructura_5":    ms_section.get("estructura_5") or "—",
    }

    # ── Strikes calientes (top 2 calls + top 2 puts por delta_oi)
    acum       = opts.get("acumulacion_oi") or {}
    top_calls  = (acum.get("top_calls_por_delta_oi") or [])[:2]
    top_puts   = (acum.get("top_puts_por_delta_oi")  or [])[:2]
    strikes_calientes = []
    for r in top_calls + top_puts:
        strikes_calientes.append({
            "tipo":        r["tipo"],
            "strike":      float(r["strike"]),
            "vencimiento": r["vencimiento"],
            "moneyness":   r.get("moneyness_label", "N/A"),
            "delta_oi":    int(r["delta_oi"]),
        })

    # ── Ensamblar YAML final
    return {
        "ticker":            ticker,
        "nombre":            "",   # ej. "Unilever"; completar a mano si querés
        "sector":            perfil.get("sector") or "",   # solo info, no se renderiza
        "fecha":             fecha_snap or date.today().isoformat(),
        "fecha_legible":     _fecha_legible(fecha_snap),
        "handle":            handle,
        "precio":            float(precio) if precio is not None else 0.0,
        "var_5d":            float(var_5d) if var_5d is not None else None,
        "niveles":           niveles,
        "sesgo":             sesgo,
        "indicadores":       indicadores,
        "opciones_ventanas": opciones_ventanas,
        "strikes_calientes": strikes_calientes,
        "pa":                pa,   # price action + estructura (infografía)
        # NOTA: la narrativa NO va en este YAML. Se edita en un archivo .md
        # separado (mismo nombre base) que make_report.py lee automaticamente.
    }


# ── Placeholder del .md de narrativa ─────────────────────────────────────────

_NARRATIVA_PLACEHOLDER = """## Análisis de opciones

TODO: pegá acá el análisis de opciones del LLM (output de Gemini tal cual).
El script limpia automáticamente bullets con `*`, indentación y secciones
numeradas tipo "1. Título" -> las convierte en formato markdown.

## Estructura / técnico

TODO: si aplica, análisis de SMC, Elliot Wave, Fibo, soportes/resistencias
de la grafica, etc.

## Conclusión

TODO: setup operativo concreto y niveles clave a vigilar.
"""


# ── Fetch concurrente ────────────────────────────────────────────────────────

async def _fetch(ticker: str) -> tuple[dict, dict]:
    """Llama a las 2 tools del MCP en paralelo."""
    ov, opts = await asyncio.gather(
        get_ticker_overview(ticker),
        get_options_analysis(ticker),
    )
    return ov, opts


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera YAML pre-cargado para make_report.py desde MCP+DB.",
    )
    parser.add_argument("ticker", help="Simbolo del ticker (ej: JPM, AAPL)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Ruta del YAML (default: scripts/reports/output/<TICKER>_<YYYY-MM-DD>.yaml)",
    )
    parser.add_argument(
        "--handle", default=DEFAULT_HANDLE,
        help=f"Handle de X (default: {DEFAULT_HANDLE})",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()

    print(f"[1/2] Consultando MCP (overview + options) para {ticker}...")
    try:
        ov, opts = asyncio.run(_fetch(ticker))
    except Exception as exc:
        print(f"ERROR consultando MCP/DB: {exc}", file=sys.stderr)
        return 1

    print("[2/2] Armando YAML...")
    try:
        data = build_data(ticker, ov, opts, args.handle)
    except Exception as exc:
        print(f"ERROR armando datos: {exc}", file=sys.stderr)
        return 2

    output = args.output
    if output is None:
        output_dir = SCRIPT_DIR / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{ticker}_{data['fecha']}.yaml"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    # Dumper sin anchors -- el YAML queda mas legible para editar
    class _NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    with output.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=_NoAliasDumper,
                  allow_unicode=True, sort_keys=False, width=100)

    # Generar el .md de narrativa al lado del YAML (si no existe ya).
    narrativa_path = output.with_suffix(".md")
    if not narrativa_path.exists():
        with narrativa_path.open("w", encoding="utf-8") as f:
            f.write(_NARRATIVA_PLACEHOLDER)
        creado_md = True
    else:
        creado_md = False

    print()
    print(f"OK: YAML generado en {output}")
    if creado_md:
        print(f"OK: MD  generado en {narrativa_path}  (placeholder para narrativa)")
    else:
        print(f"--  MD  ya existia: {narrativa_path}  (no lo sobreescribo)")
    print()
    print("Siguiente paso:")
    print(f"  1. Editá `{narrativa_path.name}` (pegá la respuesta del LLM tal cual).")
    print(f"  2. Generá el PDF:")
    print(f'     scripts\\reports\\make_report.bat "{output}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
