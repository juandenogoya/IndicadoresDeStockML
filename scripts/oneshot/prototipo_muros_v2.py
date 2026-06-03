"""
prototipo_muros_v2.py
PROTOTIPO (Etapa A) -- compara muros de OI ACTUALES vs PROPUESTOS.
Solo LEE opciones_snapshot. NO escribe nada. No toca produccion.

Dos mejoras propuestas a debatir con datos:
  1. OI COMBINADO (call+put) por strike, direccional:
     - soporte    = strike con mayor OI total DEBAJO del precio
     - resistencia = strike con mayor OI total ARRIBA del precio
     (Hoy: soporte = solo put OI debajo; resistencia = solo call OI arriba.)
  2. ZONA DINAMICA por expected move (en vez de +/-10% fijo):
     EM = precio * IV_ATM * sqrt(DTE/365);  zona = +/- k * EM
     IV_ATM = IV promedio de contratos a +/-5% del precio (no el promedio
     simple, que el volatility smile distorsiona).

Uso: venv/Scripts/python.exe scripts/oneshot/prototipo_muros_v2.py
"""
import os
import sys
import math
import statistics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.pop("DATABASE_URL", None)  # forzar LOCAL

from src.data.database import get_engine
from sqlalchemy import text

VENTANAS = {"corto": (1, 14), "medio": (15, 45), "largo": (46, 90)}

# Validez del muro
WALL_LIQ_MULT_MED = 3.0
WALL_LIQ_MIN_ABS  = 1000

# Viejo
ZONA_PCT_VIEJO     = 0.10
DIST_MIN_VIEJO     = 0.02    # 2% fijo

# Nuevo (ajustado tras Etapa A, iteracion 3 -- SIN "3x mediana")
K_SIGMA        = 1.5    # multiplicador del expected move
CAP_ZONA       = 0.18   # tope de zona
DIST_MIN_FRAC  = 0.15   # dist_min = 15% de la zona (relativo)
DIST_MIN_PISO  = 0.005  # piso 0.5% (evita el strike ATM)
USAR_3X_MEDIANA = False # el muro nuevo NO usa 3x mediana (incompatible con OI combinado)

TICKERS = ["AAPL", "NVDA", "MSFT", "KO", "TSLA", "JPM"]
FECHA   = "2026-06-02"


def _validar(cands, precio, dist_min, usar_mediana=True):
    """Validez: (3x mediana opcional) + 1000 abs + dist >= dist_min.
    Devuelve tambien fuerza_pct = % del OI del lado concentrado en el muro."""
    if not cands:
        return None
    ws, wo = max(cands, key=lambda x: x[1])
    dist = abs(precio - ws) / precio
    ok_med = True
    if usar_mediana:
        mediana = statistics.median([oi for _, oi in cands])
        ok_med = wo >= WALL_LIQ_MULT_MED * mediana
    if ok_med and wo >= WALL_LIQ_MIN_ABS and dist >= dist_min:
        oi_total_lado = sum(oi for _, oi in cands)
        fuerza = round(wo / oi_total_lado * 100, 0) if oi_total_lado else None
        return {"strike": ws, "oi": int(wo), "dist_pct": round(dist * 100, 2), "fuerza": fuerza}
    return None


def muro_viejo(contratos, precio, lado):
    """Logica actual: soporte = put OI debajo; resistencia = call OI arriba; zona +/-10%, dist 2%."""
    zmin, zmax = precio * (1 - ZONA_PCT_VIEJO), precio * (1 + ZONA_PCT_VIEJO)
    if lado == "soporte":
        cands = [(c["strike"], c["oi"]) for c in contratos
                 if c["tipo"] == "put" and zmin <= c["strike"] < precio and c["oi"] > 0]
    else:
        cands = [(c["strike"], c["oi"]) for c in contratos
                 if c["tipo"] == "call" and precio < c["strike"] <= zmax and c["oi"] > 0]
    return _validar(cands, precio, DIST_MIN_VIEJO)


def muro_nuevo(contratos, precio, zona_pct, lado):
    """Propuesto ajustado: OI combinado; zona +/- min(k*EM, cap); dist_min relativo."""
    zmin, zmax = precio * (1 - zona_pct), precio * (1 + zona_pct)
    dist_min = max(DIST_MIN_PISO, DIST_MIN_FRAC * zona_pct)
    oi_por_strike = {}
    for c in contratos:
        if c["oi"] > 0:
            oi_por_strike[c["strike"]] = oi_por_strike.get(c["strike"], 0) + c["oi"]
    if lado == "soporte":
        cands = [(s, oi) for s, oi in oi_por_strike.items() if zmin <= s < precio]
    else:
        cands = [(s, oi) for s, oi in oi_por_strike.items() if precio < s <= zmax]
    return _validar(cands, precio, dist_min, usar_mediana=USAR_3X_MEDIANA)


def iv_atm(contratos, precio):
    """IV promedio de contratos con strike dentro de +/-5% del precio."""
    ivs = [c["iv"] for c in contratos
           if c["iv"] and abs(c["strike"] - precio) / precio <= 0.05]
    return sum(ivs) / len(ivs) if ivs else None


def dte_ponderado(contratos):
    """DTE promedio ponderado por OI (donde esta el interes real)."""
    tot = sum(c["oi"] for c in contratos)
    if tot == 0:
        return None
    return sum(c["dte"] * c["oi"] for c in contratos) / tot


def _fmt(m):
    if not m:
        return "-"
    fz = f", fuerza={m['fuerza']:.0f}%" if m.get("fuerza") is not None else ""
    return f"{m['strike']:.1f} ({m['dist_pct']:+.1f}%, OI={m['oi']:,}{fz})"


def main():
    eng = get_engine()
    # Contadores de cobertura (cuantos de los slots tienen muro)
    cov = {"viejo_sop": 0, "viejo_res": 0, "nuevo_sop": 0, "nuevo_res": 0, "slots": 0}
    capeadas = 0

    for ticker in TICKERS:
        with eng.connect() as c:
            rows = c.execute(text("""
                SELECT tipo, strike, open_interest AS oi, iv,
                       (vencimiento - fecha_snapshot) AS dte,
                       precio_subyacente
                FROM opciones_snapshot
                WHERE fecha_snapshot = :f AND ticker = :t
                  AND (vencimiento - fecha_snapshot) BETWEEN 1 AND 90
                  AND open_interest IS NOT NULL
            """), {"f": FECHA, "t": ticker}).fetchall()
        if not rows:
            print(f"\n{ticker}: sin datos"); continue
        precio = float(rows[0].precio_subyacente)

        print(f"\n{'='*78}\n{ticker}  (precio {precio:.2f})\n{'='*78}")
        for vent, (lo, hi) in VENTANAS.items():
            contratos = [{"tipo": r.tipo, "strike": float(r.strike), "oi": int(r.oi or 0),
                          "iv": float(r.iv) if r.iv else None, "dte": int(r.dte)}
                         for r in rows if lo <= int(r.dte) <= hi]
            if not contratos:
                print(f"  [{vent}] sin contratos"); continue

            iva = iv_atm(contratos, precio)
            dte = dte_ponderado(contratos)
            if iva and dte:
                em = precio * iva * math.sqrt(dte / 365.0)
                zona_raw = K_SIGMA * em / precio
                zona_new = min(zona_raw, CAP_ZONA)   # aplicar cap
            else:
                em, zona_raw, zona_new = None, None, ZONA_PCT_VIEJO

            sv = muro_viejo(contratos, precio, "soporte")
            rv = muro_viejo(contratos, precio, "resistencia")
            sn = muro_nuevo(contratos, precio, zona_new, "soporte")
            rn = muro_nuevo(contratos, precio, zona_new, "resistencia")

            cov["slots"] += 1
            cov["viejo_sop"] += 1 if sv else 0
            cov["viejo_res"] += 1 if rv else 0
            cov["nuevo_sop"] += 1 if sn else 0
            cov["nuevo_res"] += 1 if rn else 0
            cap_flag = ""
            if zona_raw and zona_raw > CAP_ZONA:
                capeadas += 1
                cap_flag = f"  [CAP {zona_raw*100:.0f}->{CAP_ZONA*100:.0f}%]"

            dmin = max(DIST_MIN_PISO, DIST_MIN_FRAC * zona_new) * 100
            em_s = f"EM={em:.1f} zona={zona_new*100:.1f}% dmin={dmin:.1f}%" if em else "EM=N/D"
            iv_s = f"IV_ATM={iva*100:.0f}%" if iva else "IV=N/D"
            print(f"  [{vent:5} {lo}-{hi}d]  {iv_s}  DTE={dte:.0f}  {em_s}{cap_flag}")
            print(f"     VIEJO  sop: {_fmt(sv):<34} res: {_fmt(rv)}")
            print(f"     NUEVO  sop: {_fmt(sn):<34} res: {_fmt(rn)}")

    print(f"\n{'='*78}\nRESUMEN DE COBERTURA ({cov['slots']} slots = {len(TICKERS)} tickers x 3 ventanas)")
    print(f"{'='*78}")
    s = cov["slots"]
    print(f"  Soporte    -- VIEJO: {cov['viejo_sop']}/{s} ({100*cov['viejo_sop']//s}%)   NUEVO: {cov['nuevo_sop']}/{s} ({100*cov['nuevo_sop']//s}%)")
    print(f"  Resistencia-- VIEJO: {cov['viejo_res']}/{s} ({100*cov['viejo_res']//s}%)   NUEVO: {cov['nuevo_res']}/{s} ({100*cov['nuevo_res']//s}%)")
    print(f"  Zonas capeadas a {CAP_ZONA*100:.0f}%: {capeadas}/{s}")


if __name__ == "__main__":
    main()
