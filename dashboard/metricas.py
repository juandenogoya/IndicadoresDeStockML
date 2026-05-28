"""
dashboard/metricas.py
Papel de trabajo (transparencia / Fase 2).

DICCIONARIO = definicion UNICA por metrica (formula, ventana, fuente, umbral).
Es la base de la capa de transparencia: explica el "como" de cada dato del informe.
Los umbrales/ventanas replican los del codigo (clasificacion_tecnica.py,
opciones_plazo.py) -- si cambian alli, actualizar aca tambien.

construir_papel(datos, sintesis) cruza esa definicion estatica con los CRUDOS en
vivo (los numeros que dieron este valor para este ticker) y devuelve secciones de
filas listas para la pestana del dashboard y el export PDF.

Puro: sin DB, sin Streamlit. ASCII en strings (regla de encoding del proyecto).
"""

from src.utils.clasificacion_tecnica import (
    clasificar_rsi, clasificar_macd, clasificar_adx,
    clasificar_tendencia_sma, clasificar_estructura_smc,
)
from dashboard.view import fmt, DASH, PLAZOS, PLAZO_LBL

# ── Diccionario de metricas (definicion estatica) ───────────────────────────────
# Cada entrada: formula, ventana, fuente (tabla.columna), umbral/interpretacion.
DICCIONARIO = {
    "rsi": {
        "formula": "RSI de Wilder (promedio de subas vs bajas).",
        "ventana": "14 ruedas (D) / 14 semanas (W)",
        "fuente":  "indicadores_tecnicos.rsi14 (D); calculado al vuelo sobre W-FRI (W)",
        "umbral":  "<35 Sobreventa | 35-65 Neutral | >65 Sobrecompra",
    },
    "macd": {
        "formula": "MACD = EMA12 - EMA26; signal = EMA9 del MACD.",
        "ventana": "12 / 26 / 9",
        "fuente":  "indicadores_tecnicos.macd, macd_signal (D); al vuelo (W)",
        "umbral":  "macd > signal = Compra | macd < signal = Venta",
    },
    "tendencia_sma": {
        "formula": "Alineacion de medias + posicion del close.",
        "ventana": "SMA 21 / 50 / 200",
        "fuente":  "indicadores_tecnicos.sma21/50/200 + precios_diarios.close",
        "umbral":  "close>todas y SMA21>=50>=200 = Alcista; inverso = Bajista; sino Lateral",
    },
    "adx": {
        "formula": "ADX de Wilder (fuerza de tendencia, sin direccion).",
        "ventana": "14 ruedas",
        "fuente":  "indicadores_tecnicos.adx",
        "umbral":  ">=25 Fuerte | 20-25 Moderada | <20 Debil",
    },
    "estructura_smc": {
        "formula": "SMC: el CHoCH (giro) tiene prioridad; sino vota la estructura_10.",
        "ventana": "10 (estrategico)",
        "fuente":  "features_market_structure.estructura_10, choch_*_10, bos_*_10",
        "umbral":  "CHoCH bull->Alcista, bear->Bajista; estructura 1/-1/0",
    },
    "pcr_vol": {
        "formula": "PCR_vol = put_vol / call_vol (volumen del dia) por ventana.",
        "ventana": "corto 1-14 | medio 15-45 | largo 46-90 dias al vencimiento",
        "fuente":  "opciones_pcr_plazo_diario.pcr_vol",
        "umbral":  "informativo (<1 = mas volumen en calls)",
    },
    "pcr_oi": {
        "formula": "PCR_oi = put_oi / call_oi (interes abierto) por ventana.",
        "ventana": "corto 1-14 | medio 15-45 | largo 46-90 dias",
        "fuente":  "opciones_pcr_plazo_diario.pcr_oi, veredicto_oi",
        "umbral":  "<1 Alcista | >=1 Bajista | OI total < 500 = sin liquidez (s/d)",
    },
    "muro_soporte": {
        "formula": "Strike con mayor PUT OI por debajo del precio (zona -10%).",
        "ventana": "por plazo; distancia recalculada vs close del dia",
        "fuente":  "opciones_pcr_plazo_diario.soporte_strike, soporte_oi",
        "umbral":  "valido si OI >= 3x mediana de la zona y >= 1000 y dist >= 2%",
    },
    "muro_resistencia": {
        "formula": "Strike con mayor CALL OI por encima del precio (zona +10%).",
        "ventana": "por plazo; distancia recalculada vs close del dia",
        "fuente":  "opciones_pcr_plazo_diario.resistencia_strike, resistencia_oi",
        "umbral":  "valido si OI >= 3x mediana de la zona y >= 1000 y dist >= 2%",
    },
    "pcr_vol_sec": {
        "formula": "PCR_vol sectorial = suma put_vol / suma call_vol de los tickers del sector.",
        "ventana": "por plazo",
        "fuente":  "opciones_sector_pcr_plazo_diario.pcr_vol_sector",
        "umbral":  "informativo; ver z-score para 'inusual'",
    },
    "z_sector": {
        "formula": "z = (PCR_vol_sec - media) / std, contra la historia del sector+plazo.",
        "ventana": "~60 ruedas",
        "fuente":  "opciones_sector_pcr_plazo_diario.pcr_vol_sector_zscore, _media, _std",
        "umbral":  "|z|>1 inusual; z>+1 cobertura (bajista atipico); z<-1 optimismo",
    },
    "veredicto_sec": {
        "formula": "PCR_oi sectorial agregado por ventana.",
        "ventana": "por plazo",
        "fuente":  "opciones_sector_pcr_plazo_diario.veredicto_oi",
        "umbral":  "<1 Alcista | >=1 Bajista",
    },
}


def _fila(metrica, valor, crudo, dic_key) -> dict:
    d = DICCIONARIO[dic_key]
    return {
        "metrica": metrica,
        "valor":   valor if valor not in (None, "") else DASH,
        "crudo":   crudo if crudo not in (None, "") else DASH,
        "formula": d["formula"],
        "ventana": d["ventana"],
        "fuente":  d["fuente"],
        "umbral":  d["umbral"],
    }


def _seccion_tecnico(datos) -> list:
    d = datos["tecnico"].get("diario") or {}
    w = datos["tecnico"].get("semanal") or {}
    smc = datos.get("estructura") or {}

    filas = []
    # Diario
    filas.append(_fila("RSI (D)", clasificar_rsi(d.get("rsi")),
                       f"RSI = {fmt(d.get('rsi'), 1)}", "rsi"))
    filas.append(_fila("MACD (D)", clasificar_macd(d.get("macd"), d.get("macd_signal")),
                       f"MACD {fmt(d.get('macd'))} vs signal {fmt(d.get('macd_signal'))}", "macd"))
    filas.append(_fila("Tendencia SMA (D)",
                       clasificar_tendencia_sma(d.get("close"), d.get("sma21"), d.get("sma50"), d.get("sma200")),
                       f"close {fmt(d.get('close'))}; SMA21 {fmt(d.get('sma21'))} / "
                       f"SMA50 {fmt(d.get('sma50'))} / SMA200 {fmt(d.get('sma200'))}", "tendencia_sma"))
    filas.append(_fila("Fuerza ADX (D)", clasificar_adx(d.get("adx")),
                       f"ADX = {fmt(d.get('adx'), 1)}", "adx"))
    if smc:
        et = clasificar_estructura_smc(
            smc.get("estructura_10"), smc.get("choch_bull_10"), smc.get("choch_bear_10"),
            smc.get("bos_bull_10"), smc.get("bos_bear_10"),
        )["etiqueta"]
        crudo = (f"estructura_10={smc.get('estructura_10')}; "
                 f"CHoCH b/s={int(bool(smc.get('choch_bull_10')))}/{int(bool(smc.get('choch_bear_10')))}; "
                 f"BOS b/s={int(bool(smc.get('bos_bull_10')))}/{int(bool(smc.get('bos_bear_10')))}")
    else:
        et, crudo = DASH, DASH
    filas.append(_fila("Estructura SMC", et, crudo, "estructura_smc"))
    # Semanal
    filas.append(_fila("RSI (W)", clasificar_rsi(w.get("rsi")),
                       f"RSI = {fmt(w.get('rsi'), 1)}", "rsi"))
    filas.append(_fila("MACD (W)", clasificar_macd(w.get("macd"), w.get("macd_signal")),
                       f"MACD {fmt(w.get('macd'))} vs signal {fmt(w.get('macd_signal'))}", "macd"))
    return filas


def _seccion_opciones(datos) -> list:
    op = datos.get("opciones_plazo") or {}
    filas = []
    for p in PLAZOS:
        b = op.get(p) or {}
        lbl = PLAZO_LBL[p]
        ver = {"Alcista": "Alcista", "Bajista": "Bajista"}.get(b.get("veredicto_oi"), "s/d")
        filas.append(_fila(f"PCR_vol ({lbl})", fmt(b.get("pcr_vol")),
                           f"put_vol {b.get('put_vol', DASH)} / call_vol {b.get('call_vol', DASH)}", "pcr_vol"))
        oi_total = None
        if b.get("put_oi") is not None and b.get("call_oi") is not None:
            oi_total = b["put_oi"] + b["call_oi"]
        filas.append(_fila(f"PCR_oi ({lbl})", f"{fmt(b.get('pcr_oi'))} ({ver})",
                           f"put_oi {b.get('put_oi', DASH)} / call_oi {b.get('call_oi', DASH)}"
                           f"; OI total {oi_total if oi_total is not None else DASH}", "pcr_oi"))
        pw, cw = b.get("put_wall"), b.get("call_wall")
        filas.append(_fila(f"Soporte ({lbl})",
                           f"{fmt(pw['strike'])} ({pw['dist_pct']:+.1f}%)" if pw else DASH,
                           f"strike {fmt(pw['strike'])} con OI {pw['oi']}" if pw else "sin muro valido",
                           "muro_soporte"))
        filas.append(_fila(f"Resistencia ({lbl})",
                           f"{fmt(cw['strike'])} ({cw['dist_pct']:+.1f}%)" if cw else DASH,
                           f"strike {fmt(cw['strike'])} con OI {cw['oi']}" if cw else "sin muro valido",
                           "muro_resistencia"))
    return filas


def _seccion_sector(datos) -> list:
    sec = datos.get("sector_plazo") or {}
    filas = []
    for p in PLAZOS:
        b = sec.get(p) or {}
        lbl = PLAZO_LBL[p]
        filas.append(_fila(f"PCR_vol_sec ({lbl})", fmt(b.get("pcr_vol_sector")),
                           f"sobre {b.get('n_tickers', DASH)} tickers", "pcr_vol_sec"))
        z = b.get("pcr_vol_sector_zscore")
        filas.append(_fila(f"z sector ({lbl})", f"{z:+.1f}" if z is not None else DASH,
                           f"media {fmt(b.get('pcr_vol_sector_media'))}, std {fmt(b.get('pcr_vol_sector_std'))}",
                           "z_sector"))
        filas.append(_fila(f"Veredicto sec ({lbl})", b.get("veredicto_oi") or DASH,
                           f"PCR_oi sec {fmt(b.get('pcr_oi_sector'))}", "veredicto_sec"))
    return filas


def construir_papel(datos: dict, sintesis: dict) -> list:
    """
    Devuelve las secciones del papel de trabajo:
    [{"seccion": str, "filas": [ {metrica, valor, crudo, formula, ventana, fuente, umbral} ]}]
    """
    return [
        {"seccion": "Tecnico", "filas": _seccion_tecnico(datos)},
        {"seccion": "Opciones por plazo (ticker)", "filas": _seccion_opciones(datos)},
        {"seccion": "Sector por plazo", "filas": _seccion_sector(datos)},
    ]
