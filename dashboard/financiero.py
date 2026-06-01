"""
dashboard/financiero.py
Construye las tablas de la vista "Analisis Financiero" (logica pura, sin
Streamlit). Recibe los dicts de sintesis_data.cargar_financiero_ticker() /
cargar_screener_sector() y devuelve filas listas para mostrar.

Vista DESCRIPTIVA (igual que el resto del dashboard): muestra la foto
fundamental + comparacion con pares de la misma region. No vota ni recomienda.

Lee de fundamentales_ratios_q (ratios del ultimo Q) y de
fundamentales_ticker_vs_sector (vs mediana de pares). Ambas se pueblan con
scripts/compute_fundamentales_*.py (encadenados al refresh .bat).
"""

# -- Metadatos por metrica (etiqueta, formato, direccion "mejor") -------------
# fmt: 'ratio' (X.X) | 'pct' (valor*100 con %) | 'money' (absoluto, moneda)
# higher_better: True si mas alto = mejor; False si mas bajo = mejor (valuacion);
#                None si es neutro (absolutos descriptivos, sin juicio).

METRIC_META = {
    # Valuacion (mas bajo = mas barato = "mejor" para entrar)
    "pe_ratio":             ("PER",          "ratio", False),
    "pb_ratio":             ("P/B",          "ratio", False),
    "ps_ratio":             ("P/S",          "ratio", False),
    "ev_ebitda":            ("EV/EBITDA",    "ratio", False),
    "book_value_per_share": ("Valor libro/accion", "money", None),
    "eps_ttm":              ("BPA (TTM)",    "money", None),
    "eps_q":                ("BPA (Q)",      "money", None),
    # Calidad / rentabilidad (mas alto = mejor)
    "roe_ttm":              ("ROE (TTM)",    "pct", True),
    "roa_ttm":              ("ROA (TTM)",    "pct", True),
    "roic_ttm":             ("ROIC (TTM)",   "pct", True),
    "rotce_ttm":            ("ROTCE (TTM)",  "pct", True),   # banca
    "gross_margin_ttm":     ("Margen bruto", "pct", True),
    "operating_margin_ttm": ("Margen oper.", "pct", True),
    "net_margin_ttm":       ("Margen neto",  "pct", True),
    "efficiency_ratio_ttm": ("Ratio eficiencia", "pct", False),  # banca: mas bajo = mejor
    # Crecimiento (mas alto = mejor)
    "revenue_qoq_pct":      ("Ingresos QoQ", "pct", True),
    "revenue_yoy_pct":      ("Ingresos YoY", "pct", True),
    "net_income_qoq_pct":   ("Ut. neta QoQ", "pct", True),
    "net_income_yoy_pct":   ("Ut. neta YoY", "pct", True),
    "eps_qoq_pct":          ("BPA QoQ",      "pct", True),
    "eps_yoy_pct":          ("BPA YoY",      "pct", True),
    # Solvencia
    "current_ratio":        ("Liquidez corr.", "ratio", True),
    "debt_to_equity":       ("Deuda/Patrim.",  "ratio", False),
}

# Bloques del informe por ticker, SEGUN PERFIL (las financieras tienen
# dinamica contable distinta: ver docs/fundamentales_calculo.md).
BLOQUES_NO_FINANCIERO = {
    "Valuacion": ["pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
                  "book_value_per_share", "eps_ttm"],
    "Calidad / Rentabilidad": ["roe_ttm", "roa_ttm", "roic_ttm",
                               "gross_margin_ttm", "operating_margin_ttm",
                               "net_margin_ttm"],
    "Crecimiento": ["revenue_qoq_pct", "revenue_yoy_pct",
                    "net_income_qoq_pct", "net_income_yoy_pct",
                    "eps_qoq_pct", "eps_yoy_pct"],
    "Solvencia": ["current_ratio", "debt_to_equity"],
}
# Financieras: sin margenes industriales/ROIC/liquidez/WC. Con ROTCE + eficiencia.
BLOQUES_FINANCIERO = {
    "Valuacion": ["pe_ratio", "pb_ratio", "ps_ratio",
                  "book_value_per_share", "eps_ttm"],
    "Calidad / Rentabilidad (banca)": ["roe_ttm", "rotce_ttm", "roa_ttm",
                                       "net_margin_ttm", "efficiency_ratio_ttm"],
    "Crecimiento": ["revenue_yoy_pct", "net_income_qoq_pct",
                    "net_income_yoy_pct", "eps_qoq_pct", "eps_yoy_pct"],
    "Solvencia": ["debt_to_equity"],
}


def bloques_para(profile):
    """Devuelve el set de bloques segun el perfil del ticker."""
    return BLOQUES_FINANCIERO if profile == "financiero" else BLOQUES_NO_FINANCIERO

# Columnas del screener sectorial segun perfil dominante del sector
SCREENER_METRICS_FIN = ["pe_ratio", "pb_ratio", "roe_ttm", "rotce_ttm",
                        "net_margin_ttm", "efficiency_ratio_ttm", "revenue_yoy_pct"]
# Columnas del screener sectorial (metricas comparables)
SCREENER_METRICS = ["pe_ratio", "pb_ratio", "roe_ttm", "roic_ttm",
                    "net_margin_ttm", "revenue_yoy_pct"]


def _f(v):
    if v is None:
        return None
    try:
        fv = float(v)
        return None if fv != fv else fv  # NaN guard
    except (TypeError, ValueError):
        return None


def fmt_valor(metric, value):
    """Formatea un valor segun el tipo de metrica. '-' si None."""
    v = _f(value)
    if v is None:
        return "-"
    fmt = METRIC_META.get(metric, (None, "ratio", None))[1]
    if fmt == "pct":
        return f"{v * 100:.1f}%"
    if fmt == "money":
        return f"{v:,.2f}"
    return f"{v:.2f}"


def fmt_vs_median(value):
    """Formatea vs_median_pct (fraccional) como +XX% / -XX%."""
    v = _f(value)
    if v is None:
        return "-"
    return f"{v * 100:+.0f}%"


def evaluar_color(metric, value, peer_median):
    """
    Devuelve 'good' | 'bad' | None segun si el ticker esta mejor/peor que la
    mediana de su sector, respetando la direccion de la metrica.
    None si no hay comparacion o la metrica es neutra (absolutos).
    """
    v = _f(value)
    m = _f(peer_median)
    higher_better = METRIC_META.get(metric, (None, None, None))[2]
    if v is None or m is None or higher_better is None:
        return None
    if v == m:
        return None
    mejor = (v > m) if higher_better else (v < m)
    return "good" if mejor else "bad"


# -- Vista por ticker ---------------------------------------------------------

def construir_bloques_ticker(data: dict) -> list:
    """
    data = {ratios: {col: val}, vs_sector: {metric: {peer_median, vs_median_pct,
            percentile, peer_n, peer_basis, low_sample}}, ...}
    Devuelve [ {bloque, filas:[{metrica, valor, vs_mediana, percentil, color}]} ].
    """
    ratios = data.get("ratios") or {}
    vs = data.get("vs_sector") or {}
    profile = ratios.get("profile")
    bloques = []
    for nombre, metrics in bloques_para(profile).items():
        filas = []
        for m in metrics:
            label = METRIC_META[m][0]
            val = ratios.get(m)
            cmp = vs.get(m) or {}
            med = cmp.get("peer_median")
            pct = cmp.get("percentile")
            filas.append({
                "metrica":    label,
                "valor":      fmt_valor(m, val),
                "vs_mediana": fmt_vs_median(cmp.get("vs_median_pct")),
                "mediana":    fmt_valor(m, med),
                "percentil":  (f"{_f(pct) * 100:.0f}%" if _f(pct) is not None else "-"),
                "color":      evaluar_color(m, val, med),
            })
        bloques.append({"bloque": nombre, "filas": filas})
    return bloques


def texto_peer_basis(data: dict) -> str:
    """Leyenda sobre con quien se comparo (basis/region/n)."""
    meta = data.get("peer_meta") or {}
    basis = meta.get("peer_basis")
    region = meta.get("peer_region")
    n = meta.get("peer_n")
    sector = (data.get("ratios") or {}).get("sector")
    if basis == "region":
        return f"Comparado vs pares de {sector} / {region} (n={n})."
    if basis == "usa_fallback":
        return (f"Sin pares regionales suficientes -> comparado vs {sector} / USA "
                f"(n={n}). Leer con cautela: distinta geografia.")
    if basis == "custom":
        return f"Comparado vs {sector} / regiones elegidas (n={n})."
    return ("Pocas empresas del sector en seguimiento: sin comparacion sectorial "
            "confiable (se muestran los valores absolutos).")


# -- Screener sectorial -------------------------------------------------------

def construir_screener(rows: list) -> dict:
    """
    rows = lista de dicts (un ticker c/u) con sus ratios + columnas vs-sector.
    Devuelve {filas:[...], mediana:{...}, columnas:[...], perfil:...}. La fila
    mediana se calcula sobre valores no-null de cada metrica.

    Las columnas se eligen segun el perfil DOMINANTE de las filas: si la mayoria
    son financieras, se muestran metricas de banca (ROTCE/eficiencia) en vez de
    ROIC; si no, las estandar. Asi un sector mixto (ej Financial Services con
    JPM + V) usa el set del grupo mas numeroso.
    """
    n_fin = sum(1 for r in rows if r.get("profile") == "financiero")
    es_fin = n_fin > len(rows) / 2 if rows else False
    metrics = SCREENER_METRICS_FIN if es_fin else SCREENER_METRICS

    filas = []
    acumulado = {m: [] for m in metrics}

    for r in rows:
        fila = {"ticker": r.get("ticker")}
        for m in metrics:
            v = _f(r.get(m))
            fila[m] = v
            if v is not None:
                acumulado[m].append(v)
        filas.append(fila)

    # Mediana por metrica
    def _median(xs):
        if not xs:
            return None
        s = sorted(xs)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

    mediana = {m: _median(acumulado[m]) for m in metrics}

    # Formatear filas para display (y ordenar por PER asc por defecto)
    filas_fmt = []
    for f in filas:
        ff = {"ticker": f["ticker"]}
        for m in metrics:
            ff[METRIC_META[m][0]] = fmt_valor(m, f[m])
        ff["_pe_sort"] = f.get("pe_ratio") if f.get("pe_ratio") is not None else 1e12
        filas_fmt.append(ff)
    filas_fmt.sort(key=lambda x: x["_pe_sort"])
    for f in filas_fmt:
        f.pop("_pe_sort", None)

    fila_med = {"ticker": "— MEDIANA —"}
    for m in metrics:
        fila_med[METRIC_META[m][0]] = fmt_valor(m, mediana[m])

    columnas = ["ticker"] + [METRIC_META[m][0] for m in metrics]
    return {"filas": filas_fmt, "mediana": fila_med, "columnas": columnas,
            "n": len(filas), "perfil": "financiero" if es_fin else "no_financiero"}
