"""
app.py
App local (Streamlit) para generar reportes de análisis técnico.

Flujo:
  1. Escribís un ticker y "Traer datos" -> consulta el MCP.
  2. (Opcional) Pegás la respuesta del LLM en el box de narrativa.
  3. Botón "Generar Infografía" (solo datos) o "Generar PDF" (con narrativa).
  4. Preview en pantalla + botón de descarga.

Reusa build_yaml / make_report / make_infografia -- no reimplementa nada.

Lanzar:
  scripts\reports\app.bat
  (o: streamlit run scripts/reports/app.py)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Hacer importables los modulos hermanos (mismo directorio)
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import streamlit as st

import make_report         # noqa: E402
import make_infografia     # noqa: E402

HANDLE     = "@juan_de_nogoya"
OUTPUT_DIR = SCRIPT_DIR / "output"


def _fetch_data(ticker: str) -> dict:
    """
    Obtiene los datos corriendo build_yaml.py como SUBPROCESO.

    Por que subproceso y no importar/llamar directo: asyncpg (el driver de
    la DB) no se lleva bien con el modelo de event-loop/threads de Streamlit
    (da WinError 64 / "Future attached to a different loop"). Corriendo
    build_yaml.py aparte, el acceso async a la DB queda aislado en su propio
    proceso con su propio asyncio.run -- igual que cuando se corre el .bat.
    La app solo LEE el YAML resultante (sincrono, sin async).
    """
    # Forzar target LOCAL: si el proceso de Streamlit heredo DATABASE_URL
    # (p.ej. apuntando a Railway), la sacamos para que build_yaml aliasee el
    # DSN local (MCP_READER_LOCAL_DSN del .env). Patron documentado en CLAUDE.md.
    env = os.environ.copy()
    env.pop("DATABASE_URL", None)

    res = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "build_yaml.py"), ticker, "--handle", HANDLE],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(PROJECT_ROOT), env=env,
    )
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or "fallo build_yaml.py").strip())

    # build_yaml genera output/<TICKER>_<FECHA>.yaml -- tomar el mas reciente
    candidatos = sorted(OUTPUT_DIR.glob(f"{ticker}_*.yaml"),
                        key=lambda p: p.stat().st_mtime)
    if not candidatos:
        raise RuntimeError("build_yaml.py no genero el YAML esperado")

    return make_report.cargar_yaml(candidatos[-1])


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Generador de Reportes", page_icon="📊", layout="centered")
st.title("📊 Generador de Reportes")
st.caption("Análisis técnico para X — infografía (solo datos) o PDF (con narrativa del LLM)")

col_t, col_b = st.columns([3, 1])
ticker = col_t.text_input("Ticker", placeholder="BAC").strip().upper()
if col_b.button("Traer datos", use_container_width=True):
    if not ticker:
        st.warning("Escribí un ticker.")
    else:
        with st.spinner(f"Consultando MCP para {ticker}…"):
            try:
                st.session_state["data"] = _fetch_data(ticker)
                # Limpiar resultados previos al cambiar de ticker
                for k in ("ig_png", "ig_name", "pdf_bytes", "pdf_name", "pdf_pngs"):
                    st.session_state.pop(k, None)
                st.success(f"Datos de {ticker} cargados ({st.session_state['data']['fecha']}).")
            except Exception as exc:
                st.session_state.pop("data", None)
                st.error(f"Error consultando datos: {exc}")

data = st.session_state.get("data")

if data:
    # ── Vista previa de datos ──────────────────────────────────────────────
    with st.expander("Vista previa de datos", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"${data['precio']:.2f}",
                  f"{data['var_5d']:+.1f}% (5d)" if data.get("var_5d") is not None else None)
        c2.metric("Indicadores", data["sesgo"]["indicadores"])
        c3.metric("Opciones (PCR OI corto)", data["sesgo"]["opciones"])
        st.write(f"**IV Skew:** {data['sesgo']['iv_skew']}  |  "
                 f"**RSI:** {data['indicadores']['rsi14']:.1f}  |  "
                 f"**ADX:** {data['indicadores']['adx']:.1f}")

    st.divider()

    # ── Narrativa (solo para el PDF) ───────────────────────────────────────
    narrativa = st.text_area(
        "Narrativa (pegá la respuesta de Gemini — solo se usa en el PDF)",
        height=280,
        placeholder="## Análisis de opciones\n\n...\n\n## Conclusión\n\n...",
    )

    col1, col2 = st.columns(2)

    # ── Botón: Infografía ──────────────────────────────────────────────────
    if col1.button("🖼️  Generar Infografía", use_container_width=True):
        with st.spinner("Generando infografía…"):
            try:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                base = OUTPUT_DIR / f"{data['ticker']}_{data['fecha']}"
                png  = base.with_name(f"{base.name}_ig.png")
                pdf  = png.with_suffix(".pdf")
                html = make_infografia.renderizar_html(dict(data))
                make_infografia.generar_pdf(html, pdf)
                make_infografia.generar_png(pdf)
                pdf.unlink(missing_ok=True)
                st.session_state["ig_png"]  = png.read_bytes()
                st.session_state["ig_name"] = png.name
            except Exception as exc:
                st.error(f"Error generando infografía: {exc}")

    # ── Botón: PDF ─────────────────────────────────────────────────────────
    if col2.button("📄  Generar PDF", use_container_width=True):
        if not narrativa.strip():
            st.warning("Pegá la narrativa antes de generar el PDF.")
        else:
            with st.spinner("Generando PDF…"):
                try:
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    pdf = OUTPUT_DIR / f"{data['ticker']}_{data['fecha']}.pdf"
                    d = dict(data)
                    d["narrativa"] = narrativa
                    html = make_report.renderizar_html(d)
                    make_report.generar_pdf(html, pdf)
                    pngs = make_report.generar_pngs(pdf)
                    st.session_state["pdf_bytes"] = pdf.read_bytes()
                    st.session_state["pdf_name"]  = pdf.name
                    st.session_state["pdf_pngs"]  = [(p.name, p.read_bytes()) for p in pngs]
                except Exception as exc:
                    st.error(f"Error generando PDF: {exc}")

    # ── Resultados ─────────────────────────────────────────────────────────
    if st.session_state.get("ig_png"):
        st.divider()
        st.subheader("Infografía")
        st.image(st.session_state["ig_png"], use_container_width=True)
        st.download_button(
            "⬇️  Descargar infografía",
            st.session_state["ig_png"],
            file_name=st.session_state["ig_name"],
            mime="image/png",
            use_container_width=True,
        )

    if st.session_state.get("pdf_bytes"):
        st.divider()
        st.subheader("Reporte PDF")
        st.download_button(
            "⬇️  Descargar PDF",
            st.session_state["pdf_bytes"],
            file_name=st.session_state["pdf_name"],
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption("Páginas como PNG (para subir a X):")
        for nombre, b in st.session_state.get("pdf_pngs", []):
            st.image(b, caption=nombre, use_container_width=True)
            st.download_button(
                f"⬇️  {nombre}", b, file_name=nombre, mime="image/png",
                key=f"dl_{nombre}", use_container_width=True,
            )
else:
    st.info("Escribí un ticker y tocá **Traer datos** para empezar.")
