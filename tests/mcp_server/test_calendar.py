"""
Tests para mcp_server/tools/calendar.py

Correr desde el repo root:
  pytest tests/mcp_server/test_calendar.py -v

Requiere PYTHONPATH=. o correr desde el repo root (CWD en sys.path).
"""

from datetime import date

import pytest

from mcp_server.tools.calendar import check_trading_day, get_last_trading_day


# ── check_trading_day ─────────────────────────────────────────────────────────

class TestCheckTradingDay:

    def test_dia_habil_viernes(self):
        """2026-05-08 es viernes habil."""
        result = check_trading_day("2026-05-08")
        assert result["date"] == "2026-05-08"
        assert result["is_trading_day"] is True
        assert result["reason"] is None
        assert result["next_trading_day"] is None

    def test_sabado_no_habil(self):
        """2026-05-09 es sabado: no habil, proximo habil es lunes 2026-05-11."""
        result = check_trading_day("2026-05-09")
        assert result["date"] == "2026-05-09"
        assert result["is_trading_day"] is False
        assert result["reason"] == "weekend"
        assert result["next_trading_day"] == "2026-05-11"

    def test_domingo_no_habil(self):
        """2026-05-10 es domingo: no habil, proximo habil es lunes 2026-05-11."""
        result = check_trading_day("2026-05-10")
        assert result["is_trading_day"] is False
        assert result["reason"] == "weekend"
        assert result["next_trading_day"] == "2026-05-11"

    def test_feriado_christmas(self):
        """2026-12-25 es Christmas: feriado NYSE, razon debe contener el nombre."""
        result = check_trading_day("2026-12-25")
        assert result["is_trading_day"] is False
        assert result["reason"] is not None
        assert "Christmas" in result["reason"]
        # Proximo habil: lunes 2026-12-28
        assert result["next_trading_day"] == "2026-12-28"

    def test_feriado_independence_day_2026(self):
        """2026-07-03 es Independence Day (observed, porque 4-Jul cae sabado)."""
        result = check_trading_day("2026-07-03")
        assert result["is_trading_day"] is False
        assert result["reason"] is not None
        # Proximo habil: lunes 2026-07-06
        assert result["next_trading_day"] == "2026-07-06"

    def test_formato_invalido_barras(self):
        """Formato DD/MM/YYYY debe levantar ValueError con mensaje claro."""
        with pytest.raises(ValueError, match="Formato de fecha invalido"):
            check_trading_day("25/12/2026")

    def test_formato_invalido_sin_guiones(self):
        """Formato YYYYMMDD (sin guiones) debe levantar ValueError."""
        with pytest.raises(ValueError, match="Formato de fecha invalido"):
            check_trading_day("20261225")

    def test_fecha_inexistente(self):
        """2026-02-30 no existe en el calendario: debe levantar ValueError."""
        with pytest.raises(ValueError, match="inexistente"):
            check_trading_day("2026-02-30")

    def test_retorno_tiene_todas_las_claves(self):
        """El dict de retorno siempre tiene las 4 claves, habil o no."""
        expected_keys = {"date", "is_trading_day", "reason", "next_trading_day"}
        for d in ("2026-05-08", "2026-05-09", "2026-12-25"):
            result = check_trading_day(d)
            assert set(result.keys()) == expected_keys, f"Faltan claves para {d}"


# ── get_last_trading_day ──────────────────────────────────────────────────────

class TestGetLastTradingDay:

    def test_retorna_dict_con_claves_esperadas(self):
        """El retorno siempre tiene las 3 claves definidas en el contrato."""
        result = get_last_trading_day()
        assert "last_trading_day" in result
        assert "today" in result
        assert "today_is_trading_day" in result

    def test_last_trading_day_es_fecha_valida(self):
        """last_trading_day es parseable como YYYY-MM-DD."""
        result = get_last_trading_day()
        # Debe poder parsearse sin excepcion
        parsed = date.fromisoformat(result["last_trading_day"])
        assert parsed <= date.today()

    def test_today_coincide_con_fecha_actual(self):
        """today en el retorno debe coincidir con date.today()."""
        result = get_last_trading_day()
        assert result["today"] == date.today().isoformat()

    def test_last_trading_day_no_es_futuro(self):
        """El ultimo dia habil nunca puede ser mayor a hoy."""
        result = get_last_trading_day()
        last = date.fromisoformat(result["last_trading_day"])
        assert last <= date.today()

    def test_today_is_trading_day_es_bool(self):
        """today_is_trading_day debe ser bool, no int ni str."""
        result = get_last_trading_day()
        assert isinstance(result["today_is_trading_day"], bool)

    def test_consistencia_today_is_trading_day(self):
        """
        Si today_is_trading_day=True, last_trading_day debe ser hoy.
        Si today_is_trading_day=False, last_trading_day debe ser anterior a hoy.
        """
        result = get_last_trading_day()
        last = date.fromisoformat(result["last_trading_day"])
        today = date.fromisoformat(result["today"])
        if result["today_is_trading_day"]:
            assert last == today
        else:
            assert last < today
