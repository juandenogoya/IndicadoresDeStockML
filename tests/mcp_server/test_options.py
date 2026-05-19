"""
Tests para mcp_server/tools/options.py

Unitarios:
  - Helpers internos: _moneyness_label, _build_pcr_por_vencimiento, _tend_row
  - get_options_analysis: ticker sin datos, error de DB, estructura de respuesta,
    clampeo de dias_historia, separacion calls/puts en acumulacion_oi

Integracion (@pytest.mark.integration):
  - get_options_analysis con ticker real (AAPL)
  - get_options_analysis con ticker sin datos de opciones
"""

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server.tools.options import (
    MAX_DIAS_HISTORIA,
    _acum_row,
    _iv_skew,
    _moneyness_label,
    _resumen_pcr_por_vencimiento,
    _sesgo_iv_skew,
    _sesgo_pcr,
    _tend_actual,
    _tend_serie_row,
    get_options_analysis,
)


# ── Tests de helpers ──────────────────────────────────────────────────────────

class TestMoneynessLabel:
    def test_atm_call(self):
        assert _moneyness_label(1.5, "call") == "ATM"

    def test_atm_put(self):
        assert _moneyness_label(-1.9, "put") == "ATM"

    def test_atm_exacto_cero(self):
        assert _moneyness_label(0.0, "call") == "ATM"

    def test_call_otm(self):
        # strike > precio → OTM para call
        assert _moneyness_label(5.0, "call") == "OTM"

    def test_call_itm(self):
        # strike < precio → ITM para call
        assert _moneyness_label(-5.0, "call") == "ITM"

    def test_put_itm(self):
        # strike > precio → ITM para put (price below strike)
        assert _moneyness_label(5.0, "put") == "ITM"

    def test_put_otm(self):
        # strike < precio → OTM para put
        assert _moneyness_label(-5.0, "put") == "OTM"

    def test_none_devuelve_na(self):
        assert _moneyness_label(None, "call") == "N/A"

    def test_limite_exacto_2pct_es_atm(self):
        assert _moneyness_label(2.0, "call") == "ATM"
        assert _moneyness_label(-2.0, "put") == "ATM"

    def test_sobre_limite_no_es_atm(self):
        assert _moneyness_label(2.01, "call") == "OTM"


class TestSesgoPcr:
    def test_alcista(self):
        assert _sesgo_pcr(0.5) == "alcista"

    def test_neutro_limite_inferior(self):
        assert _sesgo_pcr(0.7) == "neutro"

    def test_neutro_limite_superior(self):
        assert _sesgo_pcr(1.0) == "neutro"

    def test_bajista(self):
        assert _sesgo_pcr(1.5) == "bajista"

    def test_none_devuelve_sin_datos(self):
        assert _sesgo_pcr(None) == "sin datos"


class TestResumenPcrPorVencimiento:
    def _make_row(self, venc, fecha_snap, call_vol, put_vol, call_oi, put_oi, dias=30):
        return {
            "vencimiento":    date.fromisoformat(venc),
            "fecha_snapshot": date.fromisoformat(fecha_snap),
            "dias_a_venc":    dias,
            "call_vol":       call_vol,
            "put_vol":        put_vol,
            "call_oi":        call_oi,
            "put_oi":         put_oi,
        }

    def test_vencimientos_ordenados_asc(self):
        rows = [
            self._make_row("2026-07-17", "2026-05-15", 100, 120, 500, 600),
            self._make_row("2026-06-19", "2026-05-15", 200, 180, 800, 750),
        ]
        result = _resumen_pcr_por_vencimiento(rows)
        assert result[0]["vencimiento"] == "2026-06-19"
        assert result[1]["vencimiento"] == "2026-07-17"

    def test_un_resumen_por_vencimiento(self):
        # Dos dias del mismo vencimiento colapsan en UN resumen
        rows = [
            self._make_row("2026-06-19", "2026-05-15", 100, 120, 500, 600),
            self._make_row("2026-06-19", "2026-05-14", 90,  110, 480, 580),
        ]
        result = _resumen_pcr_por_vencimiento(rows)
        assert len(result) == 1
        assert result[0]["dias_con_data"] == 2

    def test_pcr_oi_inicio_y_actual(self):
        # SQL ordena fecha DESC: primera fila = actual, ultima = inicio
        rows = [
            self._make_row("2026-06-19", "2026-05-15", 100, 140, 500, 600),  # actual
            self._make_row("2026-06-19", "2026-05-14", 90,  110, 400, 300),  # inicio
        ]
        result = _resumen_pcr_por_vencimiento(rows)[0]
        assert result["pcr_oi_actual"] == pytest.approx(1.2)   # 600/500
        assert result["pcr_oi_inicio"] == pytest.approx(0.75)  # 300/400

    def test_delta_oi(self):
        rows = [
            self._make_row("2026-06-19", "2026-05-15", 100, 140, 500, 600),  # actual
            self._make_row("2026-06-19", "2026-05-14", 90,  110, 400, 300),  # inicio
        ]
        result = _resumen_pcr_por_vencimiento(rows)[0]
        assert result["delta_call_oi"] == 100   # 500 - 400
        assert result["delta_put_oi"]  == 300   # 600 - 300

    def test_tendencia_cambio_de_sesgo(self):
        # inicio PCR OI 0.75 (neutro), actual 1.2 (bajista)
        rows = [
            self._make_row("2026-06-19", "2026-05-15", 100, 140, 500, 600),
            self._make_row("2026-06-19", "2026-05-14", 90,  110, 400, 300),
        ]
        result = _resumen_pcr_por_vencimiento(rows)[0]
        assert result["sesgo_actual"] == "bajista"
        assert result["tendencia"]    == "neutro -> bajista"

    def test_tendencia_estable(self):
        # ambos dias con PCR OI alcista
        rows = [
            self._make_row("2026-06-19", "2026-05-15", 100, 50, 1000, 400),
            self._make_row("2026-06-19", "2026-05-14", 90,  40, 900,  360),
        ]
        result = _resumen_pcr_por_vencimiento(rows)[0]
        assert result["sesgo_actual"] == "alcista"
        assert result["tendencia"]    == "estable en alcista"

    def test_pcr_oi_none_cuando_call_es_cero(self):
        rows = [self._make_row("2026-06-19", "2026-05-15", 0, 100, 0, 500)]
        result = _resumen_pcr_por_vencimiento(rows)[0]
        assert result["pcr_oi_actual"] is None
        assert result["sesgo_actual"]  == "sin datos"

    def test_vencimiento_como_iso_string(self):
        rows = [self._make_row("2026-06-19", "2026-05-15", 100, 120, 500, 600)]
        result = _resumen_pcr_por_vencimiento(rows)
        assert result[0]["vencimiento"] == "2026-06-19"

    def test_sin_serie_cruda_en_output(self):
        # El cambio de diseño: NO debe existir la clave 'serie'
        rows = [self._make_row("2026-06-19", "2026-05-15", 100, 120, 500, 600)]
        result = _resumen_pcr_por_vencimiento(rows)[0]
        assert "serie" not in result


class TestIvSkew:
    def test_skew_positivo(self):
        assert _iv_skew(0.31, 0.28) == pytest.approx(0.03, abs=1e-4)

    def test_skew_negativo(self):
        assert _iv_skew(0.20, 0.28) == pytest.approx(-0.08, abs=1e-4)

    def test_none_si_falta_put(self):
        assert _iv_skew(None, 0.28) is None

    def test_none_si_falta_call(self):
        assert _iv_skew(0.31, None) is None


def _tend_row_completo(**overrides):
    """Fila cruda de tendencia con todos los campos del SQL."""
    base = {
        "fecha": date(2026, 5, 15),
        "call_vol": 100000, "put_vol": 120000, "pcr_vol": Decimal("1.2"),
        "call_oi": 400000, "put_oi": 480000, "pcr_oi": Decimal("1.2"),
        "iv_call_avg": Decimal("0.28"), "iv_put_avg": Decimal("0.31"),
        "precio_sub": Decimal("195.5"), "n_contratos": 500,
        "vol_total_zscore": Decimal("1.4"), "pcr_vol_zscore": Decimal("0.5"),
        "iv_zscore": Decimal("-0.2"), "vol_relativo": Decimal("1.1"),
        "percentil_vol": Decimal("65"),
    }
    base.update(overrides)
    return base


class TestSesgoIvSkew:
    def test_bajista_skew_positivo(self):
        # skew > +0.05 -> puts mas caras -> bajista
        assert _sesgo_iv_skew(0.19) == "bajista"

    def test_alcista_skew_negativo(self):
        # skew < -0.05 -> calls mas caras -> alcista
        assert _sesgo_iv_skew(-0.12) == "alcista"

    def test_neutro_dentro_de_banda(self):
        assert _sesgo_iv_skew(0.03) == "neutro"
        assert _sesgo_iv_skew(-0.04) == "neutro"
        assert _sesgo_iv_skew(0.0) == "neutro"

    def test_none_sin_datos(self):
        assert _sesgo_iv_skew(None) == "sin datos"


class TestTendSerieRow:
    def test_campos_recortados(self):
        result = _tend_serie_row(_tend_row_completo())
        # 10 campos: los 7 recortados + sesgo_pcr_vol/oi + sesgo_iv_skew
        assert set(result.keys()) == {
            "fecha", "pcr_vol", "sesgo_pcr_vol", "pcr_oi", "sesgo_pcr_oi",
            "iv_skew", "sesgo_iv_skew",
            "vol_total_zscore", "pcr_vol_zscore", "iv_zscore",
        }

    def test_sesgo_iv_skew_etiquetado(self):
        # iv_put 0.40 - iv_call 0.28 = 0.12 > 0.05 -> bajista
        result = _tend_serie_row(
            _tend_row_completo(iv_put_avg=Decimal("0.40"), iv_call_avg=Decimal("0.28"))
        )
        assert result["sesgo_iv_skew"] == "bajista"

    def test_campos_redundantes_fuera(self):
        result = _tend_serie_row(_tend_row_completo())
        for campo in ("call_vol", "put_vol", "call_oi", "put_oi",
                      "precio_sub", "iv_call_avg", "iv_put_avg",
                      "vol_relativo", "percentil_vol"):
            assert campo not in result

    def test_iv_skew_calculado(self):
        result = _tend_serie_row(_tend_row_completo())
        assert result["iv_skew"] == pytest.approx(0.03, abs=1e-4)

    def test_fecha_iso_y_decimal_a_float(self):
        result = _tend_serie_row(_tend_row_completo())
        assert result["fecha"] == "2026-05-15"
        assert isinstance(result["pcr_vol"], float)

    def test_sesgo_pcr_etiquetado(self):
        # pcr_vol y pcr_oi = 1.2 en el helper -> bajista (>1.0)
        result = _tend_serie_row(_tend_row_completo())
        assert result["sesgo_pcr_vol"] == "bajista"
        assert result["sesgo_pcr_oi"]  == "bajista"

    def test_sesgo_pcr_alcista(self):
        # PCR 0.45 < 0.7 -> alcista (mas calls que puts)
        result = _tend_serie_row(
            _tend_row_completo(pcr_vol=Decimal("0.45"), pcr_oi=Decimal("0.45"))
        )
        assert result["sesgo_pcr_vol"] == "alcista"
        assert result["sesgo_pcr_oi"]  == "alcista"


class TestTendActual:
    def test_incluye_niveles_iv_y_contratos(self):
        result = _tend_actual(_tend_row_completo())
        assert result["iv_call_avg"] == pytest.approx(0.28)
        assert result["iv_put_avg"]  == pytest.approx(0.31)
        assert result["n_contratos"] == 500
        assert result["iv_skew"]     == pytest.approx(0.03, abs=1e-4)

    def test_iv_skew_none_cuando_faltan_datos(self):
        result = _tend_actual(
            _tend_row_completo(iv_call_avg=None, iv_put_avg=None)
        )
        assert result["iv_skew"] is None

    def test_sesgo_pcr_etiquetado(self):
        result = _tend_actual(_tend_row_completo())
        assert result["sesgo_pcr_vol"] == "bajista"
        assert result["sesgo_pcr_oi"]  == "bajista"

    def test_sesgo_iv_skew_presente(self):
        # helper: skew 0.31-0.28 = 0.03 -> dentro de banda -> neutro
        result = _tend_actual(_tend_row_completo())
        assert result["sesgo_iv_skew"] == "neutro"


class TestAcumRow:
    def _raw_row(self, **overrides):
        base = {
            "vencimiento":       date(2026, 6, 19),
            "tipo":              "call",
            "strike":            Decimal("200"),
            "oi_inicio":         5000,
            "oi_fin":            12000,
            "delta_oi":          7000,
            "iv_actual":         Decimal("0.32"),
            "precio_subyacente": Decimal("195.5"),
            "moneyness_pct":     Decimal("2.3"),
        }
        base.update(overrides)
        return base

    def test_campos_esperados(self):
        result = _acum_row(self._raw_row())
        assert set(result.keys()) == {
            "vencimiento", "tipo", "strike", "oi_fin", "delta_oi",
            "iv_actual", "moneyness_pct", "moneyness_label",
        }

    def test_campos_redundantes_fuera(self):
        result = _acum_row(self._raw_row())
        assert "precio_subyacente" not in result   # ya esta en top-level
        assert "oi_inicio"         not in result   # derivable: oi_fin - delta_oi

    def test_oi_fin_y_delta_se_mantienen(self):
        result = _acum_row(self._raw_row())
        assert result["oi_fin"]   == 12000
        assert result["delta_oi"] == 7000

    def test_moneyness_label_sintetizado(self):
        # strike 200 > precio 195.5 con +2.3% → OTM para call
        result = _acum_row(self._raw_row())
        assert result["moneyness_label"] == "OTM"

    def test_decimal_a_float(self):
        result = _acum_row(self._raw_row())
        assert isinstance(result["strike"], float)
        assert isinstance(result["iv_actual"], float)


# ── Tests unitarios de get_options_analysis ───────────────────────────────────

def _make_pool_mock(fetchrow_val, fetch_side_effect):
    """
    Construye (pool, conn) listos para parchear get_pool().
    pool debe ser MagicMock (no AsyncMock) para que pool.acquire()
    devuelva el context manager directamente sin envolverlo en coroutine.
    """
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_val)
    # side_effect como lista: cada llamada consume el siguiente elemento
    conn.fetch = AsyncMock(side_effect=fetch_side_effect if fetch_side_effect else None)

    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__  = AsyncMock(return_value=False)

    pool = MagicMock()          # MagicMock, no AsyncMock — igual que mock_conn
    pool.acquire.return_value = acquire_cm
    return pool, conn


@pytest.mark.asyncio
async def test_ticker_sin_datos_devuelve_error():
    pool, _ = _make_pool_mock(
        fetchrow_val={"max_fecha": None, "cutoff_fecha": None},
        fetch_side_effect=[],
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("FAKE")
    assert "error" in result
    assert "FAKE" in result["error"]


@pytest.mark.asyncio
async def test_db_error_devuelve_error():
    pool, conn = _make_pool_mock(
        fetchrow_val=None,
        fetch_side_effect=[],
    )
    conn.fetchrow.side_effect = Exception("connection refused")
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL")
    assert "error" in result
    assert "AAPL" in result["error"]


@pytest.mark.asyncio
async def test_estructura_respuesta_valida():
    """Verifica que el envelope tiene todas las claves esperadas."""
    pool, _ = _make_pool_mock(
        fetchrow_val={
            "max_fecha":    date(2026, 5, 15),
            "cutoff_fecha": date(2026, 4, 18),
        },
        fetch_side_effect=[[], [], []],  # tend, pcr, acum — todos vacios
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL")

    assert "error" not in result
    assert result["ticker"]          == "AAPL"
    assert result["fecha_snapshot"]  == "2026-05-15"
    assert result["cutoff_fecha"]    == "2026-04-18"
    assert result["dias_historia"]   == 20
    assert "tendencia_diaria"        in result
    assert "pcr_por_vencimiento"     in result
    assert "acumulacion_oi"          in result
    assert "top_calls_por_delta_oi"  in result["acumulacion_oi"]
    assert "top_puts_por_delta_oi"   in result["acumulacion_oi"]


@pytest.mark.asyncio
async def test_dias_historia_se_clampea_al_maximo():
    pool, _ = _make_pool_mock(
        fetchrow_val={
            "max_fecha":    date(2026, 5, 15),
            "cutoff_fecha": date(2026, 4, 18),
        },
        fetch_side_effect=[[], [], []],
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL", dias_historia=999)
    assert result["dias_historia"] == MAX_DIAS_HISTORIA


@pytest.mark.asyncio
async def test_acumulacion_separada_por_tipo():
    """Las filas de acumulacion_oi se separan correctamente en calls y puts."""
    acum_rows = [
        {
            "vencimiento": date(2026, 6, 19), "tipo": "call", "strike": Decimal("200"),
            "oi_inicio": 5000, "oi_fin": 12000, "delta_oi": 7000,
            "iv_actual": Decimal("0.32"), "precio_subyacente": Decimal("195.5"),
            "moneyness_pct": Decimal("2.3"),
        },
        {
            "vencimiento": date(2026, 5, 31), "tipo": "put", "strike": Decimal("185"),
            "oi_inicio": 2000, "oi_fin": 8000, "delta_oi": 6000,
            "iv_actual": Decimal("0.41"), "precio_subyacente": Decimal("195.5"),
            "moneyness_pct": Decimal("-5.4"),
        },
    ]
    pool, _ = _make_pool_mock(
        fetchrow_val={
            "max_fecha":    date(2026, 5, 15),
            "cutoff_fecha": date(2026, 4, 18),
        },
        fetch_side_effect=[[], [], acum_rows],
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL")

    calls = result["acumulacion_oi"]["top_calls_por_delta_oi"]
    puts  = result["acumulacion_oi"]["top_puts_por_delta_oi"]
    assert len(calls) == 1
    assert len(puts)  == 1
    assert calls[0]["tipo"] == "call"
    assert calls[0]["moneyness_label"] == "OTM"    # strike 200 > precio 195.5
    assert puts[0]["tipo"]  == "put"
    assert puts[0]["moneyness_label"]  == "OTM"    # strike 185 < precio 195.5


@pytest.mark.asyncio
async def test_precio_sub_del_primer_dia_de_tendencia():
    tend_row_mock = {
        "fecha": date(2026, 5, 15),
        "call_vol": 100000, "put_vol": 120000,
        "pcr_vol": Decimal("1.2"), "call_oi": 400000, "put_oi": 480000,
        "pcr_oi": Decimal("1.2"), "iv_call_avg": Decimal("0.28"),
        "iv_put_avg": Decimal("0.31"), "precio_sub": Decimal("195.50"),
        "n_contratos": 500, "vol_total_zscore": None, "pcr_vol_zscore": None,
        "iv_zscore": None, "vol_relativo": None, "percentil_vol": None,
    }
    pool, _ = _make_pool_mock(
        fetchrow_val={
            "max_fecha":    date(2026, 5, 15),
            "cutoff_fecha": date(2026, 4, 18),
        },
        fetch_side_effect=[[tend_row_mock], [], []],
    )
    with patch("mcp_server.tools.options.get_pool", AsyncMock(return_value=pool)):
        result = await get_options_analysis("AAPL")

    assert result["precio_subyacente"] == pytest.approx(195.50)


# ── Tests de integracion ──────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_options_analysis_integration():
    """Verifica estructura completa con ticker real."""
    result = await get_options_analysis("AAPL", dias_historia=10)

    assert "error" not in result, f"Error inesperado: {result.get('error')}"
    assert result["ticker"] == "AAPL"
    assert result["precio_subyacente"] is not None
    assert isinstance(result["tendencia_diaria"], dict)
    assert isinstance(result["pcr_por_vencimiento"], list)
    assert "top_calls_por_delta_oi" in result["acumulacion_oi"]
    assert "top_puts_por_delta_oi"  in result["acumulacion_oi"]

    # Validar estructura de tendencia_diaria: {actual, serie}
    td = result["tendencia_diaria"]
    assert "actual" in td
    assert "serie"  in td
    assert isinstance(td["serie"], list)
    if td["serie"]:
        row = td["serie"][0]
        assert "fecha"   in row
        assert "pcr_vol" in row
        assert "iv_skew" in row
        # campos redundantes ya no estan en la serie
        assert "call_vol" not in row
    if td["actual"]:
        assert "iv_call_avg" in td["actual"]
        assert "n_contratos" in td["actual"]

    # Validar estructura de pcr_por_vencimiento (resumen computado, sin serie)
    if result["pcr_por_vencimiento"]:
        venc = result["pcr_por_vencimiento"][0]
        assert "vencimiento"   in venc
        assert "dias_a_venc"   in venc
        assert "serie"     not in venc          # ya no se devuelve la serie cruda
        assert "pcr_oi_actual" in venc
        assert "delta_call_oi" in venc
        assert "sesgo_actual"  in venc
        assert "tendencia"     in venc

    # Validar que dias_a_venc > 0 (solo contratos vivos)
    for venc in result["pcr_por_vencimiento"]:
        assert venc["dias_a_venc"] > 0, "Se filtraron contratos vencidos"

    # Validar moneyness_label en acumulacion
    for row in result["acumulacion_oi"]["top_calls_por_delta_oi"]:
        assert row["moneyness_label"] in ("ATM", "OTM", "ITM", "N/A")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_options_analysis_ticker_sin_datos():
    """Un ticker sin datos de opciones debe retornar error claro."""
    result = await get_options_analysis("TICKER_INEXISTENTE_XYZ")
    assert "error" in result
