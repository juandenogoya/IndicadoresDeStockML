"""
mcp_server/tools/fundamentals.py
Analisis fundamental trimestral por ticker.

Tool unica: get_fundamentals(ticker, quarters)

Responde preguntas del tipo:
  "Cual es el ROE de JPM?"
  "Como viene el crecimiento de revenue de MU?"
  "NVDA esta cara contra su sector?"
  "Como dio el ultimo balance de XP?"

Fuentes (capa DERIVADA, recomputable, validada vs balances oficiales MU/XP/JPM):
  fundamentales_ratios_q         : 1 fila por (ticker, fiscal_period_end).
                                   Valuacion, rentabilidad, margenes, crecimiento
                                   QoQ/YoY, FCF, solvencia. Columna `profile`.
  fundamentales_ticker_vs_sector : formato long, 10 metricas del ultimo Q vs la
                                   mediana de PARES de la MISMA region.

Convenciones criticas (docs/fundamentales_calculo.md):
  - profile='financiero' (bancos + override curado, ej. XP): margenes
    industriales / ROIC / liquidez / working capital / FCF son NULL POR DISENO
    (no aplican a un balance bancario). En su lugar: rotce_ttm y
    efficiency_ratio_ttm. La tool lo explicita ("no_aplica_financiero").
  - Multiplos *_px = recalculados con el CIERRE del dia (numerador precio hoy,
    denominador TTM) -> son la version vigente. pe_ratio/pb_ratio/... son los
    del cierre del Q fiscal (Yahoo). La tool reporta ambos.
  - Los RATIOS son inmunes a escala/moneda. Los ABSOLUTOS (revenue_ttm, fcf_ttm,
    working_capital, net_debt, BVPS, EPS) quedan en la MONEDA DE REPORTE (29
    ADRs reportan en moneda local) -> no comparables cross-ticker sin FX.
  - Caveat ADR: BVPS (por accion ordinaria) y EPS de Yahoo (por ADR) estan en
    bases distintas; para comparar usar ratios.
  - ~4 tickers semestrales (HMY/RIO/UL/VOD) no tienen Q -> sin filas.
  - peer_basis: 'region' (pares de su region), 'usa_fallback' (bucket regional
    chico, se compara contra sector USA -- flag honesto), 'none' (sin peer-set).
"""

from datetime import date
from decimal import Decimal

from mcp_server.db.pool import get_pool
from mcp_server.db.queries import (
    SQL_FUNDAMENTALS_RATIOS,
    SQL_FUNDAMENTALS_VS_SECTOR,
)

FUNDAMENTALS_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

MAX_QUARTERS = 8

# Tickers con reporte semestral conocido (sin trimestres -> sin ratios)
_SEMESTRALES = {"HMY", "RIO", "UL", "VOD"}

_NO_APLICA = "no_aplica_financiero"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(val):
    """Decimal/None -> float/None (JSON-serializable)."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    return val


def _pct(val):
    """
    Fraccion -> porcentaje (0.167 -> 16.7). En la DB TODOS los rate-metrics
    (roe/margenes/crecimiento, incluso las columnas llamadas *_pct) se guardan
    como FRACCION. La tool normaliza a % para que el LLM no se confunda.
    """
    v = _f(val)
    return round(v * 100, 2) if v is not None else None


def _d(val):
    """date/None -> isoformat/None."""
    return val.isoformat() if isinstance(val, date) else val


def _bloque_valuacion(r) -> dict:
    """Multiplos: *_px (al cierre del dia) como vigente + fiscal de referencia."""
    return {
        "nota": "al_cierre = recalculado con el ultimo precio (vigente); "
                "fiscal = al cierre del trimestre reportado",
        "precio_usado": _f(r["precio_px"]),
        "fecha_precio": _d(r["fecha_px"]),
        "pe":        {"al_cierre": _f(r["pe_ratio_px"]), "fiscal": _f(r["pe_ratio"])},
        "pb":        {"al_cierre": _f(r["pb_ratio_px"]), "fiscal": _f(r["pb_ratio"])},
        "ps":        {"al_cierre": _f(r["ps_ratio_px"]), "fiscal": _f(r["ps_ratio"])},
        "ev_ebitda": {"al_cierre": _f(r["ev_ebitda_px"]), "fiscal": _f(r["ev_ebitda"])},
        "book_value_per_share": _f(r["book_value_per_share"]),
        "eps_ttm": _f(r["eps_ttm"]),
    }


def _bloque_rentabilidad(r, es_financiero: bool) -> dict:
    """Rentabilidad segun perfil. Financiero: ROTCE/efficiency en vez de margenes/ROIC."""
    out = {
        "roe_ttm_pct": _pct(r["roe_ttm"]),
        "roa_ttm_pct": _pct(r["roa_ttm"]),
    }
    if es_financiero:
        out.update({
            "rotce_ttm_pct": _pct(r["rotce_ttm"]),
            "efficiency_ratio_ttm_pct": _pct(r["efficiency_ratio_ttm"]),
            "roic_ttm_pct": _NO_APLICA,
            "margenes": _NO_APLICA,
        })
    else:
        out.update({
            "roic_ttm_pct": _pct(r["roic_ttm"]),
            "margen_bruto_ttm_pct": _pct(r["gross_margin_ttm"]),
            "margen_operativo_ttm_pct": _pct(r["operating_margin_ttm"]),
            "margen_neto_ttm_pct": _pct(r["net_margin_ttm"]),
            "margen_neto_yoy_delta_pp": _pct(r["net_margin_yoy_delta"]),
            "margen_operativo_yoy_delta_pp": _pct(r["operating_margin_yoy_delta"]),
            "opex_sobre_revenue_ttm_pct": _pct(r["opex_to_revenue_ttm"]),
        })
    return out


def _bloque_solvencia_fcf(r, es_financiero: bool, moneda: str) -> dict:
    if es_financiero:
        return {
            "nota": "liquidez/working capital/FCF no aplican a balance "
                    "financiero (estructura bancaria)",
            "debt_to_equity": _f(r["debt_to_equity"]),
        }
    return {
        "moneda_absolutos": moneda,
        "fcf_ttm": _f(r["fcf_ttm"]),
        "fcf_margin_ttm_pct": _pct(r["fcf_margin_ttm"]),
        "fcf_yoy_pct": _pct(r["fcf_yoy_pct"]),
        "current_ratio": _f(r["current_ratio"]),
        "working_capital": _f(r["working_capital"]),
        "debt_to_equity": _f(r["debt_to_equity"]),
        "net_debt": _f(r["net_debt"]),
        "net_debt_to_ebitda_ttm": _f(r["net_debt_to_ebitda_ttm"]),
    }


def _serie_trimestral(rows) -> list[dict]:
    """Trayectoria por trimestre (mas reciente primero): crecimiento + nivel."""
    serie = []
    for r in rows:
        serie.append({
            "trimestre_cierre": _d(r["fiscal_period_end"]),
            "revenue_ttm": _f(r["revenue_ttm"]),
            "revenue_qoq_pct": _pct(r["revenue_qoq_pct"]),
            "revenue_yoy_pct": _pct(r["revenue_yoy_pct"]),
            "eps_q": _f(r["eps_q"]),
            "eps_yoy_pct": _pct(r["eps_yoy_pct"]),
            "net_income_yoy_pct": _pct(r["net_income_yoy_pct"]),
            "net_margin_ttm_pct": _pct(r["net_margin_ttm"]),
            "roe_ttm_pct": _pct(r["roe_ttm"]),
        })
    return serie


def _vs_sector(rows) -> dict:
    """Comparativa vs pares regionales -> dict por metrica + leyenda peer_basis."""
    if not rows:
        return {"disponible": False,
                "motivo": "sin comparativa sectorial para este ticker"}

    basis = rows[0]["peer_basis"]
    leyenda = {
        "region":       "pares del MISMO sector y region",
        "usa_fallback": "pocas empresas en su region: comparado contra el "
                        "sector de USA (referencia imperfecta)",
        "none":         "sin peer-set (menos de 5 empresas comparables)",
    }.get(basis, basis)

    # Metricas guardadas como FRACCION (se muestran en %); los multiplos
    # (pe/pb/ps/ev_ebitda) son valores absolutos y quedan tal cual.
    _METRICAS_FRACCION = {
        "roe_ttm", "roa_ttm", "roic_ttm",
        "net_margin_ttm", "operating_margin_ttm", "revenue_yoy_pct",
    }

    metricas = {}
    for r in rows:
        if r["peer_basis"] == "none":
            continue
        es_frac = r["metric"] in _METRICAS_FRACCION
        conv = _pct if es_frac else _f
        metricas[r["metric"]] = {
            "valor": conv(r["value"]),
            "mediana_pares": conv(r["peer_median"]),
            "vs_mediana_pct": _pct(r["vs_median_pct"]),
            "percentil": _pct(r["percentile"]),
            "n_pares": r["peer_n"],
        }

    return {
        "disponible": basis != "none",
        "base_comparacion": leyenda,
        "sector": rows[0]["sector"],
        "region_ticker": rows[0]["ticker_region"],
        "region_pares": rows[0]["peer_region"],
        "trimestre": _d(rows[0]["fiscal_period_end"]),
        "metricas": metricas,
        "interpretacion": "vs_mediana_pct > 0 en valuacion (per/pb/ps/ev_ebitda) "
                          "= mas CARO que sus pares; > 0 en calidad/crecimiento "
                          "(roe/roa/roic/margenes/revenue_yoy) = MEJOR que sus pares",
    }


def _caveats(r, moneda: str, es_financiero: bool) -> list[str]:
    """Advertencias automaticas segun el caso."""
    out = []
    if moneda and moneda != "USD":
        out.append(
            f"Reporta en {moneda}: los valores ABSOLUTOS (revenue, FCF, net debt, "
            f"BVPS, EPS) estan en esa moneda, no comparables con tickers USD sin "
            f"conversion. Los RATIOS y porcentajes si son comparables."
        )
        out.append(
            "ADR: el EPS (por ADR) y el BVPS (por accion ordinaria) pueden estar "
            "en bases distintas; para comparar entre empresas usar ratios."
        )
    if es_financiero:
        out.append(
            "Perfil FINANCIERO (banco/broker): margenes industriales, ROIC, "
            "liquidez corriente y FCF no aplican a su estructura contable. "
            "Mirar ROE, ROTCE y efficiency ratio."
        )
    nq = r["n_quarters_available"]
    if nq is not None and int(nq) < 5:
        out.append(
            f"Solo {nq} trimestres de historia disponibles: los YoY (vs Q-4) "
            f"pueden faltar o ser menos confiables."
        )
    return out


# ── Tool publica ──────────────────────────────────────────────────────────────

async def get_fundamentals(ticker: str, quarters: int = 4) -> dict:
    """
    Analisis fundamental trimestral de un ticker: valuacion, rentabilidad,
    crecimiento, solvencia y comparativa contra los pares de su sector/region.

    Fuente: estados contables trimestrales (yahooquery, ultimos ~8 trimestres,
    validados contra balances oficiales) ya procesados en ratios. Vista
    DESCRIPTIVA del fundamental -- independiente del analisis tecnico/ML.

    Parametros:
      ticker   : ej. "JPM" (obligatorio)
      quarters : trimestres de trayectoria a incluir (default 4, max 8)

    Bloques de la respuesta:
      identidad     : sector, industria, perfil contable (financiero/no),
                      moneda de reporte, ultimo trimestre disponible
      valuacion     : PER, P/B, P/S, EV/EBITDA -- version "al_cierre"
                      (recalculada con el ultimo precio, la VIGENTE) y
                      "fiscal" (al cierre del trimestre) + BVPS + EPS TTM
      rentabilidad  : ROE/ROA (+margenes y ROIC si no es financiero;
                      ROTCE y efficiency ratio si es banco/broker)
      crecimiento   : serie por trimestre (mas reciente primero) con revenue,
                      EPS y net income QoQ/YoY -- muestra la TRAYECTORIA
      solvencia_fcf : FCF, current ratio, deuda/equity, net debt/EBITDA
      vs_sector     : cada metrica vs la MEDIANA de sus pares regionales
                      (percentil, n de pares, base de comparacion)
      caveats       : advertencias automaticas (moneda no-USD, ADR, perfil
                      financiero, poca historia)

    Perfil financiero (bancos, brokers): margenes/ROIC/liquidez/FCF figuran
    como "no_aplica_financiero" -- NO es un dato faltante, es que esos ratios
    no tienen sentido en un balance bancario.

    Ejemplos:
      get_fundamentals("JPM")              -> banco, ROE/ROTCE/efficiency
      get_fundamentals("MU", quarters=8)   -> trayectoria larga de un ciclico
      get_fundamentals("NVDA")             -> valuacion vs pares Tech-USA
    """
    # ── Validaciones ──────────────────────────────────────────────────────────
    if not ticker or not isinstance(ticker, str):
        return {"error": "ticker es obligatorio (ej. 'JPM')."}
    ticker = ticker.strip().upper()
    quarters = max(1, min(int(quarters), MAX_QUARTERS))

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(SQL_FUNDAMENTALS_RATIOS, ticker, quarters)
            vs_rows = await conn.fetch(SQL_FUNDAMENTALS_VS_SECTOR, ticker)

        if not rows:
            extra = (
                " Este ticker reporta SEMESTRALMENTE (sin trimestres) y la capa "
                "de ratios trimestrales no lo cubre."
                if ticker in _SEMESTRALES else
                " Verificar que el ticker pertenezca al universo (list_tickers) "
                "y que el refresh de fundamentales este corrido."
            )
            return {"error": f"Sin datos fundamentales para {ticker}.{extra}"}

        ultimo = rows[0]
        moneda = ultimo["reporting_currency"] or "USD"
        es_fin = (ultimo["profile"] == "financiero")

        return {
            "ticker": ticker,
            "fecha_consulta": date.today().isoformat(),
            "identidad": {
                "sector": ultimo["sector"],
                "industria": ultimo["industry"],
                "perfil_contable": ultimo["profile"],
                "moneda_reporte": moneda,
                "ultimo_trimestre": _d(ultimo["fiscal_period_end"]),
                "trimestres_disponibles": ultimo["n_quarters_available"],
            },
            "valuacion": _bloque_valuacion(ultimo),
            "rentabilidad": _bloque_rentabilidad(ultimo, es_fin),
            "crecimiento_trimestral": _serie_trimestral(rows),
            "solvencia_fcf": _bloque_solvencia_fcf(ultimo, es_fin, moneda),
            "vs_sector": _vs_sector(vs_rows),
            "caveats": _caveats(ultimo, moneda, es_fin),
        }

    except Exception as exc:
        return {"error": f"Error consultando fundamentales de {ticker}: {exc}"}
