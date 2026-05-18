"""
Tests para mcp_server/tools/overview.py

Unitarios:
  - Helpers de señales: _tendencia_sma, _rsi_estado, _macd_direccion,
    _bb_posicion, _adx_fuerza, _patron_activo, _senal_reciente_ms, _variaciones
  - get_ticker_overview: ticker no encontrado, DB error, estructura completa,
    precio sin datos, opciones no disponible, señales con datos parciales

Integracion (@pytest.mark.integration):
  - get_ticker_overview con ticker real (AAPL)
  - get_ticker_overview con ticker inexistente
"""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server.tools.overview import (
    OVERVIEW_ANNOTATIONS,
    _adx_fuerza,
    _bb_posicion,
    _macd_direccion,
    _moneyness_label,
    _patron_activo,
    _rsi_estado,
    _senal_reciente_ms,
    _tendencia_sma,
    _variaciones,
    get_ticker_overview,
)


# ── Helpers de mock ───────────────────────────────────────────────────────────

def _make_pool_mock(fetchrow_vals: list, fetch_vals: list):
    """
    Construye (pool, conn) con side_effects para simular la secuencia de
    llamadas de get_ticker_overview.

    fetchrow_vals: [perfil, tecnicos, pa, ms, vol, opciones_resumen]
    fetch_vals:    [precio_rows, top_oi_rows]  (top_oi solo si opciones != None)
    """
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=fetchrow_vals)
    conn.fetch    = AsyncMock(side_effect=fetch_vals)

    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__  = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_cm
    return pool, conn


# Fila de perfil minima
_PERFIL = {
    "sector": "Technology", "industry": "Consumer Electronics",
    "modelo_asignado": "rf_v3", "activo": True,
}

# Filas de precio (n dias, close decrece ~1% por dia — orden DESC por fecha)
_HOY = date(2026, 5, 15)

def _precio_rows(n=21, base=200.0):
    rows = []
    for i in range(n):
        c = round(base * (1 - i * 0.01), 2)
        rows.append({
            "fecha": _HOY - timedelta(days=i),
            "open": c - 0.5, "high": c + 1.0, "low": c - 1.0,
            "close": c, "volume": 50_000_000, "adj_close": c,
        })
    return rows

_TECNICOS = {
    "fecha": date(2026, 5, 15),
    "sma21": 190.0, "sma50": 185.0, "sma200": 175.0,
    "dist_sma21": 5.0, "dist_sma50": 8.0, "dist_sma200": 14.0,
    "rsi14": Decimal("62.5"),
    "momentum": Decimal("3.2"),
    "macd": Decimal("1.5"), "macd_signal": Decimal("1.2"), "macd_hist": Decimal("0.3"),
    "atr14": Decimal("3.0"),
    "bb_upper": Decimal("210.0"), "bb_middle": Decimal("200.0"),
    "bb_lower": Decimal("190.0"),
    "obv": 1_000_000,
    "vol_relativo": Decimal("1.2"),
    "adx": Decimal("28.0"),
}

_PA = {
    "fecha": date(2026, 5, 15),
    "body_pct": Decimal("0.8"), "body_ratio": Decimal("0.7"),
    "upper_shadow_pct": Decimal("0.3"), "lower_shadow_pct": Decimal("0.1"),
    "es_alcista": True,
    "gap_apertura_pct": Decimal("0.0"),
    "rango_diario_pct": Decimal("1.5"), "rango_rel_atr": Decimal("0.9"),
    "clv": Decimal("0.6"),
    "patron_doji": False, "patron_hammer": False, "patron_shooting_star": False,
    "patron_marubozu": False, "patron_engulfing_bull": True, "patron_engulfing_bear": False,
    "inside_bar": False, "outside_bar": False,
    "tendencia_velas": Decimal("0.6"),
    "vol_spike": False, "vol_price_confirm": True, "vol_price_diverge": False,
}

_MS = {
    "fecha": date(2026, 5, 15),
    "estructura_5": 1,
    "bos_bull_5": False, "bos_bear_5": False,
    "choch_bull_5": True, "choch_bear_5": False,
    "dist_sh_5_pct": Decimal("2.1"), "dist_sl_5_pct": Decimal("3.4"),
    "impulso_5_pct": Decimal("1.8"),
    "estructura_10": 1,
    "bos_bull_10": False, "bos_bear_10": False,
    "choch_bull_10": False, "choch_bear_10": False,
    "dist_sh_10_pct": Decimal("4.5"), "dist_sl_10_pct": Decimal("6.1"),
    "impulso_10_pct": Decimal("2.3"),
}

_VOL = {
    "ticker": "AAPL", "fecha": date(2026, 5, 15),
    "vol_zscore": Decimal("1.8"), "percentil_vol": Decimal("82.0"),
}

_OPC_RESUMEN = {
    "fecha": date(2026, 5, 15),
    "pcr_vol": Decimal("0.85"), "pcr_oi": Decimal("0.92"),
    "iv_call_avg": Decimal("0.25"), "iv_put_avg": Decimal("0.28"),
    "call_vol": 500_000, "put_vol": 425_000,
    "call_oi": 1_200_000, "put_oi": 1_100_000,
    "precio_sub": Decimal("200.0"), "n_contratos": 350,
}

_TOP_OI = [
    {"tipo": "call", "strike": Decimal("210.0"), "vencimiento": date(2026, 6, 19),
     "open_interest": 50_000, "iv": Decimal("0.26"), "moneyness_pct": Decimal("5.0")},
    {"tipo": "put",  "strike": Decimal("190.0"), "vencimiento": date(2026, 6, 19),
     "open_interest": 45_000, "iv": Decimal("0.29"), "moneyness_pct": Decimal("-5.0")},
]


# ── Tests de _tendencia_sma ───────────────────────────────────────────────────

class TestTendenciaSma:
    def test_alcista_fuerte(self):
        assert _tendencia_sma(210, 200, 190, 175) == "alcista fuerte"

    def test_alcista(self):
        # precio > sma50 y sma200, pero no alineacion perfecta
        assert _tendencia_sma(200, 180, 190, 175) == "alcista"

    def test_bajista_fuerte(self):
        assert _tendencia_sma(160, 170, 180, 190) == "bajista fuerte"

    def test_bajista(self):
        # precio < sma50 y sma200, sin alineacion perfecta
        assert _tendencia_sma(170, 190, 180, 175) == "bajista"

    def test_lateral(self):
        # precio < sma50 pero > sma200 (no cumple condicion alcista ni bajista)
        # close=177, sma21=175, sma50=180, sma200=175
        # close < sma50=True, close < sma200? 177<175=False -> no bajista -> lateral
        assert _tendencia_sma(177, 175, 180, 175) == "lateral"

    def test_sin_datos_none(self):
        assert _tendencia_sma(None, 200, 190, 175) == "sin datos"
        assert _tendencia_sma(200, None, 190, 175) == "sin datos"

    def test_alineacion_perfecta_con_igual_no_alcista_fuerte(self):
        # strict > no aplica si hay igualdad
        assert _tendencia_sma(200, 200, 190, 175) != "alcista fuerte"


# ── Tests de _rsi_estado ──────────────────────────────────────────────────────

class TestRsiEstado:
    def test_sobrecomprado(self):
        assert _rsi_estado(70) == "sobrecomprado"
        assert _rsi_estado(80) == "sobrecomprado"

    def test_sobrevendido(self):
        assert _rsi_estado(30) == "sobrevendido"
        assert _rsi_estado(20) == "sobrevendido"

    def test_alcista(self):
        assert _rsi_estado(60) == "alcista"
        assert _rsi_estado(55) == "alcista"

    def test_bajista(self):
        assert _rsi_estado(40) == "bajista"
        assert _rsi_estado(45) == "bajista"

    def test_neutral(self):
        assert _rsi_estado(50) == "neutral"
        assert _rsi_estado(52) == "neutral"

    def test_sin_datos(self):
        assert _rsi_estado(None) == "sin datos"

    def test_limite_exacto_70_es_sobrecomprado(self):
        assert _rsi_estado(70) == "sobrecomprado"

    def test_limite_exacto_55_es_alcista(self):
        assert _rsi_estado(55) == "alcista"


# ── Tests de _macd_direccion ──────────────────────────────────────────────────

class TestMacdDireccion:
    def test_alcista(self):
        assert _macd_direccion(0.5) == "alcista"

    def test_bajista(self):
        assert _macd_direccion(-0.3) == "bajista"

    def test_exacto_cero_es_bajista(self):
        # 0 no es > 0
        assert _macd_direccion(0) == "bajista"

    def test_sin_datos(self):
        assert _macd_direccion(None) == "sin datos"


# ── Tests de _bb_posicion ─────────────────────────────────────────────────────

class TestBbPosicion:
    def test_alta(self):
        # pos = (195 - 180) / (200 - 180) = 0.75 -> media-alta
        # pos = (199 - 180) / (200 - 180) = 0.95 -> alta
        assert _bb_posicion(199, 200, 180) == "alta"

    def test_media_alta(self):
        assert _bb_posicion(195, 200, 180) == "media-alta"

    def test_media(self):
        assert _bb_posicion(191, 200, 180) == "media"

    def test_media_baja(self):
        assert _bb_posicion(185, 200, 180) == "media-baja"

    def test_baja(self):
        assert _bb_posicion(181, 200, 180) == "baja"

    def test_sin_datos_none(self):
        assert _bb_posicion(None, 200, 180) == "sin datos"

    def test_sin_datos_rango_cero(self):
        assert _bb_posicion(200, 200, 200) == "sin datos"


# ── Tests de _adx_fuerza ──────────────────────────────────────────────────────

class TestAdxFuerza:
    def test_muy_fuerte(self):
        assert _adx_fuerza(45) == "muy fuerte"
        assert _adx_fuerza(40) == "muy fuerte"

    def test_fuerte(self):
        assert _adx_fuerza(30) == "fuerte"
        assert _adx_fuerza(25) == "fuerte"

    def test_moderada(self):
        assert _adx_fuerza(22) == "moderada"
        assert _adx_fuerza(20) == "moderada"

    def test_debil(self):
        assert _adx_fuerza(15) == "debil"

    def test_sin_datos(self):
        assert _adx_fuerza(None) == "sin datos"


# ── Tests de _patron_activo ───────────────────────────────────────────────────

class TestPatronActivo:
    def test_engulfing_bull(self):
        d = {"patron_engulfing_bull": True, "patron_doji": True}
        assert _patron_activo(d) == "engulfing_bull"  # prioridad sobre doji

    def test_doji_cuando_es_unico(self):
        d = {"patron_doji": True, "patron_engulfing_bull": False}
        assert _patron_activo(d) == "doji"

    def test_ninguno(self):
        d = {"patron_doji": False, "patron_hammer": False}
        assert _patron_activo(d) is None

    def test_hammer(self):
        d = {"patron_hammer": True, "patron_engulfing_bull": False,
             "patron_engulfing_bear": False}
        assert _patron_activo(d) == "hammer"

    def test_engulfing_bear_sobre_hammer(self):
        d = {"patron_engulfing_bear": True, "patron_hammer": True,
             "patron_engulfing_bull": False}
        assert _patron_activo(d) == "engulfing_bear"


# ── Tests de _senal_reciente_ms ───────────────────────────────────────────────

class TestSenalRecienteMs:
    def test_choch_bull_5_tiene_prioridad(self):
        d = {"choch_bull_5": True, "bos_bull_5": True, "choch_bull_10": True}
        assert _senal_reciente_ms(d) == "CHoCH alcista (ventana 5)"

    def test_bos_bear_5(self):
        d = {"choch_bull_5": False, "choch_bear_5": False,
             "bos_bull_5": False, "bos_bear_5": True}
        assert _senal_reciente_ms(d) == "BOS bajista (ventana 5)"

    def test_ventana_10_cuando_no_hay_5(self):
        d = {"choch_bull_5": False, "choch_bear_5": False,
             "bos_bull_5": False, "bos_bear_5": False,
             "bos_bull_10": True}
        assert _senal_reciente_ms(d) == "BOS alcista (ventana 10)"

    def test_none_sin_señales(self):
        d = {"choch_bull_5": False, "choch_bear_5": False,
             "bos_bull_5": False, "bos_bear_5": False,
             "choch_bull_10": False, "choch_bear_10": False,
             "bos_bull_10": False, "bos_bear_10": False}
        assert _senal_reciente_ms(d) is None

    def test_none_con_dict_none(self):
        assert _senal_reciente_ms(None) is None


# ── Tests de _variaciones ─────────────────────────────────────────────────────

class TestVariaciones:
    def test_tres_variaciones_con_21_filas(self):
        rows = _precio_rows(21, base=200.0)
        v1d, v5d, v20d = _variaciones(rows)
        assert v1d is not None
        assert v5d is not None
        assert v20d is not None
        # close[0]=200, close[1]=198 => (200/198 - 1)*100 ≈ 1.01
        assert v1d == pytest.approx(200.0 / 198.0 * 100 - 100, abs=0.02)

    def test_solo_1d_con_2_filas(self):
        rows = _precio_rows(2, base=100.0)
        v1d, v5d, v20d = _variaciones(rows)
        assert v1d is not None
        assert v5d is None
        assert v20d is None

    def test_vacio_devuelve_none(self):
        assert _variaciones([]) == (None, None, None)

    def test_close_cero_devuelve_none(self):
        rows = [{"close": 0.0, "adj_close": 0.0, "fecha": "2026-05-15"}]
        assert _variaciones(rows) == (None, None, None)

    def test_usa_adj_close_si_no_hay_close(self):
        rows = [
            {"close": None, "adj_close": 100.0},
            {"close": None, "adj_close": 98.0},
        ]
        v1d, _, _ = _variaciones(rows)
        assert v1d == pytest.approx((100.0 / 98.0 - 1) * 100, abs=0.01)


# ── Tests de _moneyness_label ─────────────────────────────────────────────────

class TestMoneynessLabel:
    def test_atm_dentro_de_2pct(self):
        assert _moneyness_label(1.5, "call") == "ATM"
        assert _moneyness_label(-1.9, "put") == "ATM"
        assert _moneyness_label(2.0, "call") == "ATM"

    def test_call_otm(self):
        assert _moneyness_label(5.0, "call") == "OTM"

    def test_call_itm(self):
        assert _moneyness_label(-5.0, "call") == "ITM"

    def test_put_itm(self):
        assert _moneyness_label(5.0, "put") == "ITM"

    def test_put_otm(self):
        assert _moneyness_label(-5.0, "put") == "OTM"

    def test_none_devuelve_na(self):
        assert _moneyness_label(None, "call") == "N/A"


# ── Tests de get_ticker_overview ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ticker_no_encontrado_devuelve_error():
    """Si el ticker no existe en activos, retorna {"error": ...}."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[None],        # perfil not found
        fetch_vals=[],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("FAKE_XYZ")

    assert "error" in result
    assert "FAKE_XYZ" in result["error"]


@pytest.mark.asyncio
async def test_db_error_en_perfil_devuelve_error():
    """Si falla la query de perfil, retorna {"error": ...}."""
    pool, conn = _make_pool_mock(
        fetchrow_vals=[Exception("connection refused")],
        fetch_vals=[],
    )
    conn.fetchrow.side_effect = Exception("connection refused")
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert "error_critico" in result or "error" in result


@pytest.mark.asyncio
async def test_estructura_completa_con_todos_los_datos():
    """Verifica que todas las secciones esten presentes con datos completos."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, _TECNICOS, _PA, _MS, _VOL, _OPC_RESUMEN],
        fetch_vals=[_precio_rows(21), _TOP_OI],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert "error" not in result
    assert result["ticker"] == "AAPL"
    assert result["fecha_analisis"] is not None

    # Todas las secciones presentes
    for seccion in ["perfil", "precio", "tecnicos", "price_action",
                    "market_structure", "volumen", "opciones"]:
        assert seccion in result, f"Falta seccion: {seccion}"


@pytest.mark.asyncio
async def test_perfil_mapeado_correctamente():
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, None, None, None, None, None],
        fetch_vals=[[], []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert result["perfil"]["sector"]    == "Technology"
    assert result["perfil"]["industry"]  == "Consumer Electronics"
    assert result["perfil"]["modelo_ml"] == "rf_v3"
    assert result["perfil"]["activo"]    is True


@pytest.mark.asyncio
async def test_precio_variaciones_calculadas():
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, None, None, None, None, None],
        fetch_vals=[_precio_rows(21), []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    p = result["precio"]
    assert "close" in p
    assert p["variacion_1d_pct"]  is not None
    assert p["variacion_5d_pct"]  is not None
    assert p["variacion_20d_pct"] is not None


@pytest.mark.asyncio
async def test_precio_sin_datos():
    """Si no hay filas de precio, la seccion dice disponible=False."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, None, None, None, None, None],
        fetch_vals=[[], []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert result["precio"].get("disponible") is False


@pytest.mark.asyncio
async def test_tecnicos_señales_derivadas():
    """Las señales categoricas se calculan correctamente."""
    # close=200 > sma21=190 > sma50=185 > sma200=175 -> alcista fuerte
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, _TECNICOS, None, None, None, None],
        fetch_vals=[_precio_rows(21, base=200.0), []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    señales = result["tecnicos"]["señales"]
    assert señales["tendencia_sma"]  == "alcista fuerte"
    assert señales["rsi_estado"]     == "alcista"      # rsi=62.5
    assert señales["macd_direccion"] == "alcista"      # hist=0.3 > 0
    assert señales["adx_fuerza"]     == "fuerte"       # adx=28


@pytest.mark.asyncio
async def test_price_action_patron_detectado():
    """patron_activo se agrega correctamente al dict de price action."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, None, _PA, None, None, None],
        fetch_vals=[[], []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert result["price_action"]["patron_activo"] == "engulfing_bull"


@pytest.mark.asyncio
async def test_market_structure_señal_reciente():
    """señal_reciente se agrega correctamente (CHoCH alcista ventana 5)."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, None, None, _MS, None, None],
        fetch_vals=[[], []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert result["market_structure"]["señal_reciente"] == "CHoCH alcista (ventana 5)"


@pytest.mark.asyncio
async def test_opciones_disponible_true_con_datos():
    """Si hay datos de opciones, disponible=True e iv_skew calculado."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, None, None, None, None, _OPC_RESUMEN],
        fetch_vals=[[], _TOP_OI],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    opc = result["opciones"]
    assert opc["disponible"] is True
    assert opc["iv_skew"] == pytest.approx(0.03, abs=1e-4)  # 0.28 - 0.25
    assert len(opc["top_calls_oi"]) == 1
    assert len(opc["top_puts_oi"])  == 1
    assert opc["top_calls_oi"][0]["moneyness_label"] == "OTM"  # strike 210 > 200
    assert opc["top_puts_oi"][0]["moneyness_label"]  == "OTM"  # strike 190 < 200


@pytest.mark.asyncio
async def test_opciones_no_disponible_sin_datos():
    """Si no hay datos de opciones, disponible=False."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, None, None, None, None, None],  # opciones=None
        fetch_vals=[[], []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert result["opciones"]["disponible"] is False
    assert "razon" in result["opciones"]


@pytest.mark.asyncio
async def test_opciones_top_oi_limitado_a_3():
    """top_calls_oi y top_puts_oi tienen maximo 3 elementos."""
    top_oi_5 = [
        {"tipo": "call", "strike": Decimal(f"{200 + i * 5}.0"),
         "vencimiento": date(2026, 6, 19), "open_interest": 50_000 - i * 5_000,
         "iv": Decimal("0.25"), "moneyness_pct": Decimal(f"{i*2}.5")}
        for i in range(5)
    ] + [
        {"tipo": "put", "strike": Decimal(f"{195 - i * 5}.0"),
         "vencimiento": date(2026, 6, 19), "open_interest": 45_000 - i * 5_000,
         "iv": Decimal("0.28"), "moneyness_pct": Decimal(f"-{i*2+1}.5")}
        for i in range(5)
    ]
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, None, None, None, None, _OPC_RESUMEN],
        fetch_vals=[[], top_oi_5],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert len(result["opciones"]["top_calls_oi"]) == 3
    assert len(result["opciones"]["top_puts_oi"])  == 3


@pytest.mark.asyncio
async def test_decimal_convertido_a_float():
    """Los Decimal de asyncpg se convierten a float en el resultado."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, _TECNICOS, None, None, None, None],
        fetch_vals=[_precio_rows(2), []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert isinstance(result["tecnicos"]["rsi14"], float)
    assert isinstance(result["tecnicos"]["macd_hist"], float)


@pytest.mark.asyncio
async def test_date_convertida_a_iso_string():
    """Las fechas date de asyncpg se convierten a ISO string."""
    pool, _ = _make_pool_mock(
        fetchrow_vals=[_PERFIL, _TECNICOS, None, None, None, None],
        fetch_vals=[_precio_rows(2), []],
    )
    with patch("mcp_server.tools.overview.get_pool", AsyncMock(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert result["tecnicos"]["fecha"] == "2026-05-15"
    assert result["precio"]["fecha"] == "2026-05-15"


@pytest.mark.asyncio
async def test_secciones_independientes_error_en_una_no_rompe_otras():
    """Un error en la seccion de tecnicos no impide devolver price_action."""
    from unittest.mock import AsyncMock as AM

    conn = AM()
    # perfil ok, tecnicos explota, pa ok, ms ok, vol ok, opciones None
    conn.fetchrow = AM(side_effect=[_PERFIL, Exception("DB error"), _PA, _MS, _VOL, None])
    conn.fetch    = AM(side_effect=[_precio_rows(2), []])

    acquire_cm = AM()
    acquire_cm.__aenter__ = AM(return_value=conn)
    acquire_cm.__aexit__  = AM(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire_cm

    with patch("mcp_server.tools.overview.get_pool", AM(return_value=pool)):
        result = await get_ticker_overview("AAPL")

    assert "error" in result.get("tecnicos", {})          # tecnicos fallo
    assert "fecha" in result.get("price_action", {})      # pa ok
    assert result.get("market_structure", {}).get("señal_reciente") is not None


# ── Tests de anotaciones ──────────────────────────────────────────────────────

def test_overview_annotations_readonly():
    assert OVERVIEW_ANNOTATIONS["readOnlyHint"]    is True
    assert OVERVIEW_ANNOTATIONS["destructiveHint"] is False


# ── Tests de integracion ──────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_overview_integration_aapl():
    """Verifica estructura completa con ticker real."""
    result = await get_ticker_overview("AAPL")

    assert "error" not in result, f"Error inesperado: {result.get('error')}"
    assert result["ticker"] == "AAPL"

    # Perfil
    assert "perfil" in result
    assert result["perfil"]["sector"] is not None

    # Precio
    assert "precio" in result
    p = result["precio"]
    if p.get("disponible") is not False:
        assert p["close"] is not None
        assert p["fecha"] is not None

    # Tecnicos con señales
    assert "tecnicos" in result
    t = result["tecnicos"]
    if t.get("disponible") is not False:
        señales = t.get("señales", {})
        assert señales["tendencia_sma"] in (
            "alcista fuerte", "alcista", "lateral", "bajista", "bajista fuerte", "sin datos"
        )
        assert señales["rsi_estado"] in (
            "sobrecomprado", "alcista", "neutral", "bajista", "sobrevendido", "sin datos"
        )

    # Price action
    assert "price_action" in result

    # Market structure
    assert "market_structure" in result

    # Opciones (puede o no tener datos)
    assert "opciones" in result
    opc = result["opciones"]
    assert "disponible" in opc
    if opc["disponible"]:
        assert "pcr_vol" in opc
        assert "iv_skew" in opc
        assert "top_calls_oi" in opc
        assert "top_puts_oi"  in opc
        assert len(opc["top_calls_oi"]) <= 3
        assert len(opc["top_puts_oi"])  <= 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_overview_ticker_inexistente():
    """Un ticker que no existe en activos debe retornar error claro."""
    result = await get_ticker_overview("TICKER_INEXISTENTE_XYZ")
    assert "error" in result
    assert "TICKER_INEXISTENTE_XYZ" in result["error"]
