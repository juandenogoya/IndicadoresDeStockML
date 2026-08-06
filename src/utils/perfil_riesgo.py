"""
perfil_riesgo.py
Clasificador de perfil de riesgo por ticker (Fase 2 del perfilado de carteras).

Enfoque PERFIL PURO + CONTEXTO SECTORIAL (docs/perfiles_carteras.md seccion 4;
decision del usuario 5/8/2026):
    1. El PERFIL es el COMPORTAMIENTO cuantitativo puro (caja_cuant): como se
       movio de verdad el ticker, sin capar por el sector. Para un balde de
       RIESGO importa como zarandea, no la etiqueta GICS.
    2. El SECTOR queda como CONTEXTO (caja_base, prior) y como fuente del flag
       de EXCEPCION: cuando el comportamiento se despega 2+ cajas del sector, se
       marca (la senal valiosa: el ticker que no se comporta como su sector).
    3. `movio` reporta cuantas cajas se despega del sector (informativo, ya no
       limita la etiqueta). Se dejo de capar a +/-1 porque el caja_cuant es
       robusto (percentil sobre los 200, promedio de 4 metricas), no un umbral
       suelto ruidoso: capar una senal confiable perdia informacion.

    CAVEAT (historia corta): con ~15 meses de futuros ES el perfil es un espejo
    de la ventana reciente -> etiquetas mas moviles y algun nombre estructural-
    mente moderado, dormido, cae una caja abajo. Es transparente y se estabiliza
    solo cuando haya 3-5 anios de historia. Ver docs/perfiles_carteras.md.

CLASIFICACION DATA-DRIVEN (decision del usuario, 5/8/2026): el riesgo
cuantitativo NO usa umbrales absolutos tipeados a mano ni listas curadas de
tickers. Se deriva de la DISTRIBUCION del propio universo:
    - Cada eje (ATR%_w, ATR%_m, beta, drawdown_1a) -> PERCENTIL del ticker
      dentro del universo (0..100).
    - Composite = promedio de los percentiles de los ejes disponibles (peso
      IGUAL para todos: sin ponderaciones arbitrarias).
    - Caja cuantitativa por CUARTIL del composite: <25 / 25-50 / 50-75 / >=75.
Misma regla estadistica para los 200. Los unicos "parametros" son los cuartiles
(25/50/75), que son la particion natural en 4 cajas, no una eleccion a dedo.

Las 4 cajas son un ORDINAL 0..3 (Conservadora < Moderada < Arriesgada <
Especulativa). El composite percentil ademas rankea dentro de cada caja.

PURO: recibe una lista de (ticker, sector, metricas) y devuelve clasificaciones.
SIN config, SIN database, SIN side effects. Necesita el UNIVERSO entero (no un
ticker aislado) porque los cortes son relativos a la distribucion. La lectura de
metricas (Fase 1), del sector (activos) y la persistencia las hace la Fase 3.

Diseno: docs/perfiles_carteras.md.
"""

# --- Las 4 cajas (ordinal) --------------------------------------------------
CONSERVADORA = 0
MODERADA = 1
ARRIESGADA = 2
ESPECULATIVA = 3

PERFIL_NOMBRE = {
    CONSERVADORA: "Conservadora",
    MODERADA: "Moderada",
    ARRIESGADA: "Arriesgada",
    ESPECULATIVA: "Especulativa",
}

# --- Prior top-down: sector -> caja base ------------------------------------
# Mapeo aprobado por el usuario (5/8/2026) contra los 11 sectores del universo.
# Es una regla UNIFORME (un sector -> una caja), no un hardcodeo por ticker.
# Fallback = Moderada (el medio neutro) para un sector desconocido a futuro.
SECTOR_BASE = {
    "Consumer Defensive":     CONSERVADORA,
    "Utilities":              CONSERVADORA,
    "Healthcare":             MODERADA,
    "Real Estate":            MODERADA,
    "Financial Services":     MODERADA,
    "Industrials":            MODERADA,
    "Basic Materials":        MODERADA,
    "Energy":                 MODERADA,
    "Communication Services": ARRIESGADA,
    "Consumer Cyclical":      ARRIESGADA,
    "Technology":             ARRIESGADA,
}
SECTOR_BASE_FALLBACK = MODERADA

# Ejes de comportamiento (peso IGUAL). El diario (atr_pct_d) se deja fuera del
# composite: para swing de mediano/largo la vol de 1 rueda es la mas ruidosa; el
# semanal y mensual capturan corto y medio/largo. Ajustable si se decide sumarlo.
EJES = ("atr_pct_w", "atr_pct_m", "beta", "max_dd_1a")

# Cuartiles que parten el composite (percentil 0..100) en las 4 cajas.
CUARTILES = (25.0, 50.0, 75.0)


def caja_base(sector, sector_map=None):
    """
    Caja base (prior top-down) de un sector.

    Returns:
        (ordinal 0..3, fuente) con fuente in {"sector", "fallback"}.
    """
    sector_map = SECTOR_BASE if sector_map is None else sector_map
    if sector in sector_map:
        return sector_map[sector], "sector"
    return SECTOR_BASE_FALLBACK, "fallback"


def rank_percentil(valor, poblacion):
    """
    Percentil (0..100) de `valor` dentro de `poblacion` (midrank para empates).
    None si valor es None o la poblacion esta vacia.

    midrank: pct = 100 * (#menores + 0.5*#iguales) / N. Robusto a duplicados y
    homogeneo entre ejes.
    """
    if valor is None or not poblacion:
        return None
    n = len(poblacion)
    menores = sum(1 for x in poblacion if x < valor)
    iguales = sum(1 for x in poblacion if x == valor)
    return round(100.0 * (menores + 0.5 * iguales) / n, 2)


def caja_por_cuartil(pct, cuartiles=None):
    """percentil 0..100 -> caja 0..3 por cuartil. None si pct es None."""
    if pct is None:
        return None
    cuartiles = CUARTILES if cuartiles is None else cuartiles
    caja = 0
    for c in cuartiles:
        if pct >= c:
            caja += 1
        else:
            break
    return caja


def _poblaciones(rows, ejes):
    """Por eje, la lista de valores no-None del universo (para rankear)."""
    pobl = {e: [] for e in ejes}
    for r in rows:
        met = r.get("metricas") or {}
        for e in ejes:
            v = met.get(e)
            if v is not None:
                pobl[e].append(v)
    return pobl


def perfilar_universo(rows, ejes=EJES, sector_map=None, cuartiles=None):
    """
    Clasifica el UNIVERSO entero: el PERFIL es el riesgo cuantitativo puro
    (caja por cuartil del percentil composite dentro del universo). El sector
    queda como contexto (caja_base) y para el flag de excepcion.

    Args:
        rows: lista de dicts con:
            ticker    simbolo
            sector    sector de `activos`
            metricas  dict de Fase 1 (atr_pct_w/m, beta, max_dd_1a); ejes en
                      None se descartan del composite. Si TODOS None -> el ticker
                      queda en el prior (sin_cuant=True).
        ejes: ejes a promediar (default EJES; peso igual).

    Returns:
        lista de dicts (mismo orden que rows) con:
            ticker, sector
            perfil / perfil_ordinal          caja final
            caja_base / caja_base_nombre / caja_base_fuente
            caja_cuant                       0..3 por cuartil | None (= perfil)
            score_riesgo                     composite percentil 0..100 | None
            pct_ejes                         dict eje -> percentil del ticker
            movio                            cajas de despegue vs sector (con signo)
            excepcion                        bool (despegue 2+ cajas del sector)
            sin_cuant                        bool
            rank_en_caja / n_en_caja / pct_en_caja   (via rankear_intra_caja)
    """
    ejes = tuple(ejes)
    cuartiles = CUARTILES if cuartiles is None else cuartiles
    pobl = _poblaciones(rows, ejes)

    out = []
    for r in rows:
        ticker = r.get("ticker")
        sector = r.get("sector")
        met = r.get("metricas") or {}
        base, fuente = caja_base(sector, sector_map)

        pct_ejes = {}
        for e in ejes:
            p = rank_percentil(met.get(e), pobl[e])
            if p is not None:
                pct_ejes[e] = p

        if not pct_ejes:
            out.append({
                "ticker": ticker, "sector": sector,
                "perfil": PERFIL_NOMBRE[base], "perfil_ordinal": base,
                "caja_base": base, "caja_base_nombre": PERFIL_NOMBRE[base],
                "caja_base_fuente": fuente,
                "caja_cuant": None, "score_riesgo": None, "pct_ejes": {},
                "movio": 0, "excepcion": False, "sin_cuant": True,
            })
            continue

        score = round(sum(pct_ejes.values()) / len(pct_ejes), 2)
        cuant = caja_por_cuartil(score, cuartiles)
        diff = cuant - base          # cuanto se despega del sector (con signo)
        final = cuant                # PERFIL PURO: el comportamiento manda

        out.append({
            "ticker": ticker, "sector": sector,
            "perfil": PERFIL_NOMBRE[final], "perfil_ordinal": final,
            "caja_base": base, "caja_base_nombre": PERFIL_NOMBRE[base],
            "caja_base_fuente": fuente,
            "caja_cuant": cuant, "score_riesgo": score, "pct_ejes": pct_ejes,
            "movio": diff, "excepcion": abs(diff) >= 2, "sin_cuant": False,
        })

    return rankear_intra_caja(out)


def rankear_intra_caja(clasificaciones):
    """
    Agrega el ranking dentro de cada caja final (quien esta mas caliente).

    Muta cada dict agregando:
        rank_en_caja    1 = mas riesgoso de su caja (score mas alto)
        n_en_caja       cantidad de tickers con score en esa caja
        pct_en_caja     percentil 0..100 del score dentro de la caja

    Los tickers sin score (sin_cuant) quedan con rank/pct = None. Devuelve la
    misma lista (mutada) por comodidad.
    """
    por_caja = {}
    for c in clasificaciones:
        if c.get("score_riesgo") is None:
            c["rank_en_caja"] = None
            c["n_en_caja"] = None
            c["pct_en_caja"] = None
            continue
        por_caja.setdefault(c["perfil_ordinal"], []).append(c)

    for items in por_caja.values():
        items.sort(key=lambda x: x["score_riesgo"], reverse=True)
        n = len(items)
        for i, c in enumerate(items):
            c["rank_en_caja"] = i + 1
            c["n_en_caja"] = n
            c["pct_en_caja"] = (round(100.0 * (n - 1 - i) / (n - 1), 1)
                                if n > 1 else 100.0)

    return clasificaciones
