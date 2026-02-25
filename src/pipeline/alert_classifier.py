"""
alert_classifier.py
Sistema de puntuacion compuesto (0-100) y clasificacion de alertas.

Escala:
    75 - 100  =>  COMPRA_FUERTE
    60 -  74  =>  COMPRA
    40 -  59  =>  NEUTRAL
    25 -  39  =>  VENTA
     0 -  24  =>  VENTA_FUERTE

Composicion del score (base = 50):
    +/- ML V3 P(ganancia)          max +30 / min -15
    +   Condiciones PA alcistas     max +15
    +   Score rule-based            max +10
    -   Senales bajistas            max -25
"""

from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────
# Umbrales
# ─────────────────────────────────────────────────────────────

_NIVELES = [
    (75, "COMPRA_FUERTE"),
    (60, "COMPRA"),
    (40, "NEUTRAL"),
    (25, "VENTA"),
    (0,  "VENTA_FUERTE"),
]


# ─────────────────────────────────────────────────────────────
# Puntuacion por componente
# ─────────────────────────────────────────────────────────────

def _puntos_ml(prob: float) -> Tuple[int, str]:
    """Contribucion del modelo ML V3 al score."""
    if prob >= 0.75:
        pts, desc = 30, f"ML fuerte alcista ({prob:.0%})"
    elif prob >= 0.65:
        pts, desc = 22, f"ML alcista ({prob:.0%})"
    elif prob >= 0.55:
        pts, desc = 14, f"ML leve alcista ({prob:.0%})"
    elif prob >= 0.45:
        pts, desc = 5,  f"ML neutro ({prob:.0%})"
    elif prob >= 0.35:
        pts, desc = -5, f"ML leve bajista ({prob:.0%})"
    else:
        pts, desc = -15, f"ML bajista ({prob:.0%})"
    return pts, desc


def _puntos_pa(ev1: int, ev2: int, ev3: int, ev4: int) -> Tuple[int, str]:
    """Contribucion de las condiciones PA al score."""
    n_met = ev1 + ev2 + ev3 + ev4
    condiciones = []
    if ev4: condiciones.append("EV4")
    if ev1: condiciones.append("EV1")
    if ev2: condiciones.append("EV2")
    if ev3: condiciones.append("EV3")

    if n_met >= 3:
        return 15, f"PA: alta confluencia ({', '.join(condiciones)})"
    elif n_met == 2:
        return 10, f"PA: {', '.join(condiciones)}"
    elif n_met == 1:
        return 5, f"PA: {', '.join(condiciones)}"
    return 0, "PA: sin condicion alcista"


def _puntos_score_rb(score: float) -> Tuple[int, str]:
    """Contribucion del scoring rule-based al score."""
    if score is None:
        return 0, "Score: sin dato"
    if score >= 0.75:
        return 10, f"Score tecnico alto ({score:.2f})"
    elif score >= 0.60:
        return 7, f"Score tecnico moderado ({score:.2f})"
    elif score >= 0.45:
        return 3, f"Score tecnico neutro ({score:.2f})"
    elif score >= 0.30:
        return -2, f"Score tecnico bajo ({score:.2f})"
    return -5, f"Score tecnico muy bajo ({score:.2f})"


def _puntos_bajistas(bear_bos10: int, bear_choch10: int,
                     bear_estructura: int) -> Tuple[int, str]:
    """Penalizacion por senales bajistas de market structure."""
    pts = 0
    desc_parts = []
    if bear_bos10:
        pts   -= 12
        desc_parts.append("BOS bajista")
    if bear_choch10:
        pts   -= 12
        desc_parts.append("CHoCH bajista")
    if bear_estructura:
        pts   -= 8
        desc_parts.append("estructura bajista")

    if desc_parts:
        return pts, "Bajistas: " + ", ".join(desc_parts)
    return 0, ""


# ─────────────────────────────────────────────────────────────
# Clasificador principal
# ─────────────────────────────────────────────────────────────

def clasificar_alerta(signals: Dict, meta: Dict) -> Tuple[float, str, str]:
    """
    Calcula el score compuesto y asigna el nivel de alerta.

    Args:
        signals: dict de evaluar_ticker() con ml_prob_ganancia, pa_ev*, bear_*
        meta:    dict de calcular_features_completas() con score_ponderado, etc.

    Returns:
        (alert_score, alert_nivel, alert_detalle)
        alert_score  : float 0-100
        alert_nivel  : str COMPRA_FUERTE | COMPRA | NEUTRAL | VENTA | VENTA_FUERTE
        alert_detalle: str descripcion legible
    """
    base = 50.0
    partes = []

    # ML
    prob = signals.get("ml_prob_ganancia", 0.5)
    pts_ml, desc_ml = _puntos_ml(prob)
    base += pts_ml
    partes.append(desc_ml)

    # PA
    ev1 = signals.get("pa_ev1", 0)
    ev2 = signals.get("pa_ev2", 0)
    ev3 = signals.get("pa_ev3", 0)
    ev4 = signals.get("pa_ev4", 0)
    pts_pa, desc_pa = _puntos_pa(ev1, ev2, ev3, ev4)
    base += pts_pa
    partes.append(desc_pa)

    # Score rule-based
    score_rb = meta.get("score_ponderado") if meta else None
    pts_rb, desc_rb = _puntos_score_rb(score_rb)
    base += pts_rb
    partes.append(desc_rb)

    # Senales bajistas
    b_bos  = signals.get("bear_bos10", 0)
    b_choch = signals.get("bear_choch10", 0)
    b_est  = signals.get("bear_estructura", 0)
    pts_bear, desc_bear = _puntos_bajistas(b_bos, b_choch, b_est)
    base += pts_bear
    if desc_bear:
        partes.append(desc_bear)

    # Clamp 0-100
    alert_score = round(max(0.0, min(100.0, base)), 1)

    # Determinar nivel
    alert_nivel = "NEUTRAL"
    for umbral, nivel in _NIVELES:
        if alert_score >= umbral:
            alert_nivel = nivel
            break

    # Detalle legible
    alert_detalle = " | ".join(p for p in partes if p)

    return alert_score, alert_nivel, alert_detalle
