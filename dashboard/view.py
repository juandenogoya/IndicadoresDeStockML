"""
dashboard/view.py
View model del informe: transforma (datos, sintesis) en estructuras listas para
renderizar. FUENTE UNICA de contenido para que el dashboard Streamlit y la
exportacion a JPG muestren EXACTAMENTE lo mismo.

Puro: sin Streamlit, sin HTML, sin DB. Solo formateo + clasificacion.
"""

from src.utils.clasificacion_tecnica import (
    clasificar_rsi, clasificar_macd, clasificar_adx,
    clasificar_tendencia_sma, clasificar_estructura_smc,
)

DASH = "-"  # marcador de celda vacia (ASCII, regla de encoding del proyecto)
PLAZOS = ["corto", "medio", "largo"]
PLAZO_LBL = {"corto": "Corto", "medio": "Medio", "largo": "Largo"}
COLOR_ESTADO = {"ALCISTA": "#1a7f37", "BAJISTA": "#c1121f", "NEUTRAL": "#6b7280"}

# Particion DISJUNTA de reglas entre los dos sub-bloques de la conclusion, para
# que Rapida y Detallada no se repitan.
_TIPOS_RAPIDA  = {"opciones", "tecnico", "refuerzo", "contexto", "info"}
_TIPOS_VIGILAR = {"matiz", "riesgo", "oportunidad"}


def fmt(v, dec=2):
    if v is None:
        return DASH
    try:
        return f"{float(v):.{dec}f}"
    except (TypeError, ValueError):
        return str(v)


def _muro_cell(muro: dict | None) -> str:
    if not muro or muro.get("strike") is None:
        return DASH
    return f"{fmt(muro['strike'])} ({muro['dist_pct']:+.1f}%)"


def _inusual(z) -> str:
    if z is None:
        return DASH
    if z >= 1.0:
        return f"z {z:+.1f} (cobertura alta)"
    if z <= -1.0:
        return f"z {z:+.1f} (optimismo)"
    return f"z {z:+.1f} (normal)"


def _encabezado(datos, sintesis) -> dict:
    perfil = datos.get("perfil") or {}
    precio = datos.get("precio") or {}
    estado = sintesis["estado"]
    return {
        "ticker":   datos.get("ticker"),
        "sector":   perfil.get("sector") or "?",
        "industry": perfil.get("industry") or "",
        "close":    fmt(precio.get("close")),
        "fecha":    str(precio.get("fecha")) if precio.get("fecha") else DASH,
        "estado":   estado,
        "frase":    sintesis.get("frase", ""),
        "color":    COLOR_ESTADO.get(estado, "#6b7280"),
    }


def _tecnico(datos) -> dict:
    d = datos["tecnico"].get("diario") or {}
    w = datos["tecnico"].get("semanal") or {}
    smc = datos.get("estructura") or {}
    et_smc = DASH
    if smc:
        et_smc = clasificar_estructura_smc(
            smc.get("estructura_10"), smc.get("choch_bull_10"), smc.get("choch_bear_10"),
            smc.get("bos_bull_10"), smc.get("bos_bear_10"),
        )["etiqueta"]
    filas = [
        ("RSI",
         clasificar_rsi(d.get("rsi")) or DASH,
         clasificar_rsi(w.get("rsi")) or DASH),
        ("MACD",
         clasificar_macd(d.get("macd"), d.get("macd_signal")) or DASH,
         clasificar_macd(w.get("macd"), w.get("macd_signal")) or DASH),
        ("Tendencia (SMA)",
         clasificar_tendencia_sma(d.get("close"), d.get("sma21"), d.get("sma50"), d.get("sma200")) or DASH,
         DASH),
        ("Fuerza (ADX)", clasificar_adx(d.get("adx")) or DASH, DASH),
        ("Estructura (SMC)", et_smc or DASH, DASH),
    ]
    return {
        "fecha_d": str(d.get("fecha")) if d.get("fecha") else DASH,
        "fecha_w": str(w.get("fecha")) if w.get("fecha") else DASH,
        "filas":   filas,
    }


def _opciones(datos):
    op = datos.get("opciones_plazo") or {}
    if not op:
        return None
    return [(PLAZO_LBL[p], fmt((op.get(p) or {}).get("pcr_vol")),
             fmt((op.get(p) or {}).get("pcr_oi")),
             (op.get(p) or {}).get("veredicto_oi") or DASH) for p in PLAZOS]


def _muros(datos):
    op = datos.get("opciones_plazo") or {}
    if not op:
        return None
    return [(PLAZO_LBL[p], _muro_cell((op.get(p) or {}).get("put_wall")),
             _muro_cell((op.get(p) or {}).get("call_wall"))) for p in PLAZOS]


def _sector(datos):
    sec = datos.get("sector_plazo") or {}
    sector = (datos.get("perfil") or {}).get("sector") or "?"
    if not sec:
        return {"sector": sector, "filas": None}
    filas = [(PLAZO_LBL[p], fmt((sec.get(p) or {}).get("pcr_vol_sector")),
              (sec.get(p) or {}).get("veredicto_oi") or DASH,
              _inusual((sec.get(p) or {}).get("pcr_vol_sector_zscore"))) for p in PLAZOS]
    return {"sector": sector, "filas": filas}


def _conclusion(sintesis) -> dict:
    reglas = sintesis.get("reglas", [])
    votos = sintesis.get("votos", {})
    estado = sintesis["estado"]

    rapida = [r["mensaje"] for r in reglas if r["tipo"] in _TIPOS_RAPIDA][:4]
    if not rapida:
        rapida = ["Sin sesgo dominante; ver cuadros de arriba."]

    dim_lbl = {"A": "tecnico", "B": "opciones", "F": "estructura"}
    dims = [f"{dim_lbl[k]} {votos[k].lower()}" for k in ("A", "B", "F")
            if votos.get(k) and votos[k] != "Neutral"]
    detallada = f"El veredicto {estado} surge del cruce de dimensiones"
    detallada += f" ({', '.join(dims)})." if dims else "; ninguna dimension fija una direccion clara."
    vigilar = [r["mensaje"] for r in reglas if r["tipo"] in _TIPOS_VIGILAR]
    if vigilar:
        detallada += " A vigilar: " + " ".join(vigilar)
    return {"rapida": rapida, "detallada": detallada}


def construir_vista(datos: dict, sintesis: dict) -> dict:
    """Arma el view model completo del informe a partir de datos + sintesis."""
    return {
        "encabezado": _encabezado(datos, sintesis),
        "tecnico":    _tecnico(datos),
        "opciones":   _opciones(datos),
        "muros":      _muros(datos),
        "sector":     _sector(datos),
        "conclusion": _conclusion(sintesis),
    }
