"""
dashboard/radar.py
Radar de actividad inusual (familia E del spec). SCREENING del universo (no
analisis de un ticker): detecta anomalias de actividad en opciones del dia via
z-scores (cuantos desvios esta el valor de hoy vs la historia del propio ticker).

Logica PURA (sin DB, sin Streamlit). Recibe el dict de dashboard.sintesis_data.
cargar_radar() y devuelve filas clasificadas y rankeadas.

Decisiones de diseno (27-28/5/2026):
  - Senales: volumen de opciones, IV, y sesgo PCR.
  - Cruce accion+opciones (28/5): si el volumen de opciones Y el de la accion son
    ambos inusuales -> "Institucional probable". Requiere ticker_zscore_diario
    fresca (se recalcula en local; ver memory/dashboard.md). Si falta, no dispara.
  - Sin score unico: se muestran las z por separado + un TAG legible (no caja negra).
  - Guarda de liquidez por percentil_vol (escala 0-100) para filtrar ruido.
  - "Sector acompana": si el volumen del sector tambien es inusual (rotacion) vs
    solo el ticker (idiosincratico), usando vol_total_sector_zscore.

ASCII en strings (regla de encoding del proyecto).
"""

Z_INUSUAL_DEFAULT   = 2.0    # |z| para considerar una senal "inusual"
PERCENTIL_MIN       = 50.0   # guarda de liquidez: percentil_vol minimo (0-100)
Z_SECTOR            = 1.0    # z del sector para decir que "acompana"


def _clasificar(t: dict, z: float):
    """
    Devuelve (tags, magnitud) de un ticker. tags = lista de tipos de anomalia;
    magnitud = la z mas fuerte entre las disparadas. ([] si no dispara nada.)
    """
    tags, mags = [], []
    vol_z, iv_z, pcr_z = t.get("vol_z"), t.get("iv_z"), t.get("pcr_z")
    stock_z = t.get("stock_vol_z")

    if vol_z is not None and vol_z >= z:
        tags.append("Volumen inusual"); mags.append(vol_z)
    if iv_z is not None and iv_z >= z:
        tags.append("IV en alza"); mags.append(iv_z)
    # PCR = put/call. Baja fuerte = mas calls (sesgo alcista); sube fuerte = mas
    # puts (cobertura / sesgo defensivo).
    if pcr_z is not None and pcr_z <= -z:
        tags.append("Sesgo a calls"); mags.append(abs(pcr_z))
    if pcr_z is not None and pcr_z >= z:
        tags.append("Cobertura (puts)"); mags.append(pcr_z)
    # Cruce accion+opciones: volumen inusual en AMBOS = huella institucional probable.
    if (vol_z is not None and vol_z >= z) and (stock_z is not None and stock_z >= z):
        tags.append("Institucional probable")

    return tags, mags


def construir_radar(radar_data: dict, z: float = Z_INUSUAL_DEFAULT,
                    percentil_min: float = PERCENTIL_MIN,
                    z_sector: float = Z_SECTOR) -> list:
    """
    Construye las filas del radar a partir de cargar_radar().

    Returns:
        Lista de dicts ordenada por magnitud desc:
        {ticker, sector, tipo, magnitud, vol_z, iv_z, pcr_z, sector_acompana}
    """
    sector_z = radar_data.get("sector_z") or {}
    filas = []
    for t in radar_data.get("tickers", []):
        # Guarda de liquidez: si el volumen del dia no llega al percentil minimo
        # del propio ticker, lo descartamos (ruido de iliquidez relativa).
        pv = t.get("percentil_vol")
        if pv is not None and pv < percentil_min:
            continue

        tags, mags = _clasificar(t, z)
        if not tags:
            continue

        sz = sector_z.get(t.get("sector"))
        if sz is not None and sz >= z_sector:
            acompana = f"Si (z={sz:+.1f})"
        else:
            acompana = "No"

        filas.append({
            "ticker":          t.get("ticker"),
            "sector":          t.get("sector"),
            "tipo":            ", ".join(tags),
            "magnitud":        round(max(mags), 1),
            "vol_z":           round(t["vol_z"], 1) if t.get("vol_z") is not None else None,
            "iv_z":            round(t["iv_z"], 1) if t.get("iv_z") is not None else None,
            "pcr_z":           round(t["pcr_z"], 1) if t.get("pcr_z") is not None else None,
            "accion_z":        round(t["stock_vol_z"], 1) if t.get("stock_vol_z") is not None else None,
            "sector_acompana": acompana,
        })

    filas.sort(key=lambda r: r["magnitud"], reverse=True)
    return filas
