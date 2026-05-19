"""
Tests para mcp_server/tools/screener.py

Unitarios:
  - Validaciones de parametros invalidos
  - Clampeo de limit
  - filtros_aplicados solo muestra filtros activos
  - Campos sintetizados: señal_smc, patron_activo
  - Conversion Decimal -> float, date -> str
  - solo_ml_activos False nunca aparece en filtros_aplicados
  - Resultado vacio (0 tickers)
  - Estructura del envelope de respuesta

Integracion (@pytest.mark.integration):
  - Sin filtros: devuelve resultados
  - Con filtro alert_nivel COMPRA_FUERTE
  - Con múltiples filtros combinados
  - Ticker inexistente por sector: resultado vacio
"""

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server.tools.screener import (
    MAX_LIMIT,
    SCREENER_ANNOTATIONS,
    _resultado_row,
    screen_tickers,
)


# ── Helper de mock ────────────────────────────────────────────────────────────

def _make_pool_mock(fetch_result: list):
    """
    Screener hace una sola llamada: conn.fetch(SQL_SCREEN_TICKERS, *params).
    fetch_result es la lista de filas que devuelve.
    """
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=fetch_result)

    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__  = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    return pool, conn


# Fila completa de ejemplo (simula asyncpg.Record via dict con .items())
def _make_row(
    ticker="AAPL",
    sector="Technology",
    industry="Consumer Electronics",
    modelo_asignado="rf_v3",
    rsi14=62.5,
    macd_hist=0.3,
    adx=28.0,
    sma200=175.0,
    vol_relativo=1.8,
    estructura_5=1,
    choch_bull_5=True,
    choch_bear_5=False,
    bos_bull_5=False,
    bos_bear_5=False,
    es_alcista=True,
    vol_spike=False,
    patron_engulfing_bull=True,
    patron_engulfing_bear=False,
    patron_hammer=False,
    patron_shooting_star=False,
    patron_marubozu=False,
    patron_doji=False,
    alert_score=72.5,
    alert_nivel="COMPRA_FUERTE",
    close=293.32,
    fecha_precio=date(2026, 5, 15),
):
    return {
        "ticker": ticker,
        "sector": sector,
        "industry": industry,
        "modelo_asignado": modelo_asignado,
        "rsi14": Decimal(str(rsi14)) if rsi14 is not None else None,
        "macd_hist": Decimal(str(macd_hist)) if macd_hist is not None else None,
        "adx": Decimal(str(adx)) if adx is not None else None,
        "sma200": Decimal(str(sma200)) if sma200 is not None else None,
        "vol_relativo": Decimal(str(vol_relativo)) if vol_relativo is not None else None,
        "estructura_5": estructura_5,
        "choch_bull_5": choch_bull_5,
        "choch_bear_5": choch_bear_5,
        "bos_bull_5": bos_bull_5,
        "bos_bear_5": bos_bear_5,
        "es_alcista": es_alcista,
        "vol_spike": vol_spike,
        "patron_engulfing_bull": patron_engulfing_bull,
        "patron_engulfing_bear": patron_engulfing_bear,
        "patron_hammer": patron_hammer,
        "patron_shooting_star": patron_shooting_star,
        "patron_marubozu": patron_marubozu,
        "patron_doji": patron_doji,
        "alert_score": Decimal(str(alert_score)) if alert_score is not None else None,
        "alert_nivel": alert_nivel,
        "close": Decimal(str(close)) if close is not None else None,
        "fecha_precio": fecha_precio,
    }


# ── Tests de _resultado_row ───────────────────────────────────────────────────

class TestResultadoRow:
    def test_señal_smc_choch_bull(self):
        r = _resultado_row(_make_row(choch_bull_5=True))
        assert r["señal_smc"] == "CHoCH alcista"

    def test_señal_smc_choch_bear(self):
        r = _resultado_row(_make_row(choch_bull_5=False, choch_bear_5=True))
        assert r["señal_smc"] == "CHoCH bajista"

    def test_señal_smc_bos_bull(self):
        r = _resultado_row(_make_row(
            choch_bull_5=False, choch_bear_5=False, bos_bull_5=True
        ))
        assert r["señal_smc"] == "BOS alcista"

    def test_señal_smc_bos_bear(self):
        r = _resultado_row(_make_row(
            choch_bull_5=False, choch_bear_5=False, bos_bull_5=False, bos_bear_5=True
        ))
        assert r["señal_smc"] == "BOS bajista"

    def test_señal_smc_none_sin_señales(self):
        r = _resultado_row(_make_row(
            choch_bull_5=False, choch_bear_5=False, bos_bull_5=False, bos_bear_5=False
        ))
        assert r["señal_smc"] is None

    def test_patron_engulfing_bull(self):
        r = _resultado_row(_make_row(patron_engulfing_bull=True))
        assert r["patron_activo"] == "engulfing_bull"

    def test_patron_hammer(self):
        r = _resultado_row(_make_row(
            patron_engulfing_bull=False, patron_engulfing_bear=False,
            patron_hammer=True,
        ))
        assert r["patron_activo"] == "hammer"

    def test_patron_none_sin_patrones(self):
        r = _resultado_row(_make_row(
            patron_engulfing_bull=False, patron_engulfing_bear=False,
            patron_hammer=False, patron_shooting_star=False,
            patron_marubozu=False, patron_doji=False,
        ))
        assert r["patron_activo"] is None

    def test_modelo_asignado_renombrado_a_modelo_ml(self):
        r = _resultado_row(_make_row(modelo_asignado="rf_v3"))
        assert "modelo_ml"        in r
        assert "modelo_asignado"  not in r
        assert r["modelo_ml"] == "rf_v3"

    def test_columnas_detalle_eliminadas(self):
        r = _resultado_row(_make_row())
        for col in ["choch_bull_5", "choch_bear_5", "bos_bull_5", "bos_bear_5",
                    "patron_engulfing_bull", "patron_engulfing_bear", "patron_hammer",
                    "patron_shooting_star", "patron_marubozu", "patron_doji", "sma200"]:
            assert col not in r, f"columna de detalle no eliminada: {col}"

    def test_decimal_convertido_a_float(self):
        r = _resultado_row(_make_row(rsi14=62.5, alert_score=72.5))
        assert isinstance(r["rsi14"],       float)
        assert isinstance(r["alert_score"], float)

    def test_date_convertida_a_iso_string(self):
        r = _resultado_row(_make_row(fecha_precio=date(2026, 5, 15)))
        assert r["fecha_precio"] == "2026-05-15"

    def test_estructura_5_etiquetada(self):
        assert _resultado_row(_make_row(estructura_5=1))["estructura_5"]  == "alcista"
        assert _resultado_row(_make_row(estructura_5=0))["estructura_5"]  == "neutral"
        assert _resultado_row(_make_row(estructura_5=-1))["estructura_5"] == "bajista"

    def test_estructura_5_none(self):
        r = _resultado_row(_make_row(estructura_5=None))
        assert r["estructura_5"] == "sin datos"

    def test_macd_direccion_agregado(self):
        assert _resultado_row(_make_row(macd_hist=0.3))["macd_direccion"]  == "alcista"
        assert _resultado_row(_make_row(macd_hist=-0.5))["macd_direccion"] == "bajista"

    def test_es_alcista_vol_spike_a_bool(self):
        r = _resultado_row(_make_row(es_alcista=True, vol_spike=False))
        assert r["es_alcista"] is True
        assert r["vol_spike"]  is False
        # tambien funciona si la DB los da como 0/1
        r2 = _resultado_row(_make_row(es_alcista=1, vol_spike=0))
        assert r2["es_alcista"] is True
        assert r2["vol_spike"]  is False


# ── Tests de validacion de parametros ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_macd_direccion_invalido():
    result = await screen_tickers(macd_direccion="lateral")
    assert "error" in result
    assert "macd_direccion" in result["error"]


@pytest.mark.asyncio
async def test_señal_smc_invalida():
    result = await screen_tickers(señal_smc="choch_super")
    assert "error" in result
    assert "señal_smc" in result["error"]


@pytest.mark.asyncio
async def test_patron_invalido():
    result = await screen_tickers(patron="doji_dragón")
    assert "error" in result
    assert "patron" in result["error"]


@pytest.mark.asyncio
async def test_alert_nivel_invalido():
    result = await screen_tickers(alert_nivel="SUPER_COMPRA")
    assert "error" in result
    assert "alert_nivel" in result["error"]


@pytest.mark.asyncio
async def test_ordenar_por_invalido():
    result = await screen_tickers(ordenar_por="precio")
    assert "error" in result
    assert "ordenar_por" in result["error"]


@pytest.mark.asyncio
async def test_estructura_5_invalido():
    result = await screen_tickers(estructura_5=2)
    assert "error" in result
    assert "estructura_5" in result["error"]


@pytest.mark.asyncio
async def test_multiples_errores_se_concatenan():
    result = await screen_tickers(macd_direccion="XX", patron="YY")
    assert "error" in result
    assert "macd_direccion" in result["error"]
    assert "patron"         in result["error"]


# ── Tests de limit ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_limit_clampeado_al_maximo():
    pool, conn = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers(limit=999)
    # verificar que el limit pasado a la DB fue MAX_LIMIT
    args = conn.fetch.call_args[0]   # positional args del fetch
    assert args[-1] == MAX_LIMIT     # $17 = limit


@pytest.mark.asyncio
async def test_limit_minimo_1():
    pool, conn = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers(limit=0)
    args = conn.fetch.call_args[0]
    assert args[-1] == 1


# ── Tests de filtros_aplicados ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_filtros_aplicados_solo_muestra_activos():
    pool, _ = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers(rsi_min=60, alert_nivel="COMPRA")

    fa = result["filtros_aplicados"]
    assert "rsi_min"     in fa
    assert "alert_nivel" in fa
    # parametros no pasados NO deben aparecer
    assert "rsi_max"          not in fa
    assert "sector"           not in fa
    assert "macd_direccion"   not in fa


@pytest.mark.asyncio
async def test_solo_ml_activos_false_no_aparece_en_filtros():
    """solo_ml_activos=False (default) no debe mostrarse en filtros_aplicados."""
    pool, _ = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers()
    assert "solo_ml_activos" not in result["filtros_aplicados"]


@pytest.mark.asyncio
async def test_solo_ml_activos_true_aparece_en_filtros():
    pool, _ = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers(solo_ml_activos=True)
    assert "solo_ml_activos" in result["filtros_aplicados"]


# ── Tests de estructura del resultado ────────────────────────────────────────

@pytest.mark.asyncio
async def test_envelope_estructura_completa():
    pool, _ = _make_pool_mock([_make_row()])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers()

    assert "error"            not in result
    assert "fecha_analisis"   in result
    assert "filtros_aplicados" in result
    assert "total_encontrados" in result
    assert "ordenado_por"     in result
    assert "limit"            in result
    assert "tickers"          in result


@pytest.mark.asyncio
async def test_resultado_vacio():
    pool, _ = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers(rsi_min=99)

    assert result["total_encontrados"] == 0
    assert result["tickers"] == []
    assert "error" not in result


@pytest.mark.asyncio
async def test_total_encontrados_refleja_lista():
    rows = [_make_row(ticker=f"T{i}") for i in range(5)]
    pool, _ = _make_pool_mock(rows)
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers()

    assert result["total_encontrados"] == 5
    assert len(result["tickers"]) == 5


@pytest.mark.asyncio
async def test_resultado_ticker_tiene_campos_requeridos():
    pool, _ = _make_pool_mock([_make_row()])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers()

    t = result["tickers"][0]
    for campo in ["ticker", "sector", "industry", "modelo_ml",
                  "fecha_precio", "close",
                  "rsi14", "macd_hist", "macd_direccion", "adx", "vol_relativo",
                  "estructura_5", "señal_smc",
                  "es_alcista", "vol_spike", "patron_activo",
                  "alert_score", "alert_nivel"]:
        assert campo in t, f"campo faltante: {campo}"


@pytest.mark.asyncio
async def test_ordenado_por_reflejado_en_respuesta():
    pool, _ = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers(ordenar_por="rsi14")
    assert result["ordenado_por"] == "rsi14"


# ── Tests de parametros pasados al SQL ────────────────────────────────────────

@pytest.mark.asyncio
async def test_parametros_pasados_en_orden_correcto():
    """Verifica que los 17 parametros se pasen en el orden correcto a asyncpg."""
    pool, conn = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        await screen_tickers(
            solo_ml_activos=True,
            sector="Technology",
            rsi_min=55.0,
            rsi_max=70.0,
            adx_min=25.0,
            vol_relativo_min=1.5,
            sobre_sma200=True,
            macd_direccion="alcista",
            estructura_5=1,
            señal_smc="choch_bull",
            es_alcista=True,
            vol_spike=True,
            patron="hammer",
            alert_nivel="COMPRA",
            alert_score_min=60.0,
            ordenar_por="adx",
            limit=10,
        )

    args = conn.fetch.call_args[0]   # (sql, $1, $2, ..., $17)
    # args[0] = SQL, args[1..17] = parametros
    assert args[1]  is True            # $1  solo_ml_activos
    assert args[2]  == "Technology"    # $2  sector
    assert args[3]  == 55.0            # $3  rsi_min
    assert args[4]  == 70.0            # $4  rsi_max
    assert args[5]  == 25.0            # $5  adx_min
    assert args[6]  == 1.5             # $6  vol_relativo_min
    assert args[7]  is True            # $7  sobre_sma200
    assert args[8]  == "alcista"       # $8  macd_direccion
    assert args[9]  == 1               # $9  estructura_5
    assert args[10] == "choch_bull"    # $10 señal_smc
    assert args[11] is True            # $11 es_alcista
    assert args[12] is True            # $12 vol_spike
    assert args[13] == "hammer"        # $13 patron
    assert args[14] == "COMPRA"        # $14 alert_nivel
    assert args[15] == 60.0            # $15 alert_score_min
    assert args[16] == "adx"           # $16 ordenar_por
    assert args[17] == 10              # $17 limit


@pytest.mark.asyncio
async def test_sin_filtros_pasa_none_y_false_correctamente():
    """Sin filtros, $1=False y $2..$15=None."""
    pool, conn = _make_pool_mock([])
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        await screen_tickers()

    args = conn.fetch.call_args[0]
    assert args[1]  is False   # $1 solo_ml_activos siempre bool
    assert args[2]  is None    # $2 sector
    assert args[3]  is None    # $3 rsi_min
    assert args[15] is None    # $15 alert_score_min


# ── Tests de error de DB ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_error_de_db_devuelve_error():
    pool, conn = _make_pool_mock([])
    conn.fetch.side_effect = Exception("connection refused")
    with patch("mcp_server.tools.screener.get_pool", AsyncMock(return_value=pool)):
        result = await screen_tickers()
    assert "error" in result
    assert "screener" in result["error"].lower()


# ── Tests de anotaciones ──────────────────────────────────────────────────────

def test_screener_annotations_readonly():
    assert SCREENER_ANNOTATIONS["readOnlyHint"]    is True
    assert SCREENER_ANNOTATIONS["destructiveHint"] is False


# ── Tests de integracion ──────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_screener_sin_filtros_devuelve_resultados():
    """Sin filtros debe devolver tickers (hasta limit=20 default)."""
    result = await screen_tickers()

    assert "error"          not in result
    assert result["total_encontrados"] > 0
    assert len(result["tickers"]) <= 20

    t = result["tickers"][0]
    assert "ticker"      in t
    assert "alert_score" in t


@pytest.mark.integration
@pytest.mark.asyncio
async def test_screener_filtro_alert_nivel():
    """Filtro por alert_nivel debe devolver solo tickers con ese nivel."""
    result = await screen_tickers(alert_nivel="COMPRA_FUERTE", limit=50)

    assert "error" not in result
    for t in result["tickers"]:
        assert t["alert_nivel"] == "COMPRA_FUERTE"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_screener_filtro_sector_technology():
    """Filtro por sector debe devolver solo ese sector."""
    result = await screen_tickers(sector="Technology", limit=50)

    assert "error" not in result
    for t in result["tickers"]:
        assert t["sector"] == "Technology"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_screener_multiples_filtros():
    """Combinacion de filtros: RSI alcista + estructura alcista."""
    result = await screen_tickers(
        rsi_min=55,
        estructura_5=1,
        es_alcista=True,
        limit=10,
    )

    assert "error" not in result
    fa = result["filtros_aplicados"]
    assert fa["rsi_min"]     == 55
    assert fa["estructura_5"] == 1
    assert fa["es_alcista"]  is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_screener_sector_inexistente_devuelve_vacio():
    """Sector que no existe en el universo devuelve lista vacia."""
    result = await screen_tickers(sector="SectorQueNoExisteXYZ")

    assert "error" not in result
    assert result["total_encontrados"] == 0
    assert result["tickers"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_screener_solo_ml_activos():
    """solo_ml_activos=True devuelve solo tickers con modelo RF."""
    result = await screen_tickers(solo_ml_activos=True, limit=50)

    assert "error" not in result
    for t in result["tickers"]:
        assert t["modelo_ml"] is not None
