"""
test_rutina.py -- la rutina diaria orquestada (src/utils/rutina.py) y el
ejecutor que guarda el log y respeta el codigo de salida
(scripts/manual/rutina_diaria.py).

Lo que importa probar es la POLITICA acordada con el usuario el 13/9/2026 (que
frena y que no) y que el ejecutor no repita el error del `tee`: devolver un
codigo que no es el del proceso.
"""

import importlib.util
import os
import sys

import pytest

from src.utils import rutina as R

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── Orden y politica ──────────────────────────────────────────────────────────

def test_orden_de_la_rutina_y_politica_acordada():
    assert [p.clave for p in R.PASOS] == ["sync", "paso1", "paso2", "paso3", "ft",
                                          "earnings"]
    politica = {p.clave: p.si_falla for p in R.PASOS}
    assert politica == {"sync": R.SEGUIR, "paso1": R.FRENAR, "paso2": R.FRENAR,
                        "paso3": R.FRENAR, "ft": R.INFORMAR,
                        "earnings": R.INFORMAR}


def test_earnings_va_ultimo_y_no_frena_a_nadie():
    """Nada de la rutina espera a earnings_historico: no es insumo de decisiones
    (el filtro de balances de los bots lee earnings_calendar) y es el paso mas
    lento por la cuota de Alpha Vantage."""
    assert R.PASOS[-1].clave == "earnings"
    assert R.debe_seguir(R.paso("earnings"), R.ERROR)
    # su fecha de DATOS es la del anuncio, no la de la corrida
    assert R.paso("earnings").columna == "announcement_date"


def test_retomar_desde_un_paso():
    assert [p.clave for p in R.pasos_desde("paso2")] == ["paso2", "paso3", "ft",
                                                        "earnings"]
    with pytest.raises(ValueError):
        R.pasos_desde("paso9")


def test_sync_que_falla_sigue_y_saltea_la_purga():
    res, notas = R.clasificar_sync(1, None)
    assert res == R.ERROR and any("purga" in n for n in notas)
    assert R.debe_seguir(R.paso("sync"), res)
    assert R.clasificar_sync(0, 1)[0] == R.PARCIAL
    assert R.clasificar_sync(0, 0) == (R.OK, [])


# ── Paso 1: parcial contra caida ──────────────────────────────────────────────

def _resumen(n_precios=0, n_futuros=0):
    return {"target_date": "2026-09-14",
            "pend_precios": {f"T{i:02d}": "2026-09-11" for i in range(n_precios)},
            "pend_futuros": {f"F{i}": "2026-09-11" for i in range(n_futuros)}}


def test_paso1_sin_resumen_y_con_error_es_una_caida():
    res, notas, pend = R.clasificar_paso1(1, None)
    assert res == R.ERROR and pend is None
    assert not R.debe_seguir(R.paso("paso1"), res)


def test_paso1_todo_al_dia():
    assert R.clasificar_paso1(0, _resumen())[0] == R.OK


def test_paso1_hasta_10_pendientes_sigue_con_aviso():
    res, notas, pend = R.clasificar_paso1(1, _resumen(10))
    assert res == R.PARCIAL and len(pend["precios"]) == 10
    assert R.debe_seguir(R.paso("paso1"), res)


def test_paso1_con_mas_de_10_pendientes_frena():
    res, _, _ = R.clasificar_paso1(1, _resumen(11))
    assert res == R.ERROR
    assert not R.debe_seguir(R.paso("paso1"), res)


def test_futuros_pendientes_avisan_pero_no_frenan():
    res, notas, _ = R.clasificar_paso1(1, _resumen(0, 2))
    assert res == R.PARCIAL and any("futuros" in n for n in notas)


# ── ft_run_diario ─────────────────────────────────────────────────────────────

def test_ft_distingue_guard_de_bot_con_error():
    assert R.clasificar_ft(0)[0] == R.OK
    assert R.clasificar_ft(R.FT_GUARD)[0] == R.ERROR
    assert R.clasificar_ft(R.FT_BOT_FALLO)[0] == R.PARCIAL


def test_una_interrupcion_frena_aunque_el_paso_diga_seguir():
    assert not R.debe_seguir(R.paso("sync"), R.INTERRUMPIDO)


# ── Resumen y Telegram ────────────────────────────────────────────────────────

def _corrida(frenada_en=None, resultado_paso2=R.OK):
    return {
        "rutina_id": "20260914_0930", "inicio": "2026-09-14 09:30:00", "log_dir": "logs/rutina/x",
        "frenada_en": frenada_en,
        "pasos": [
            {"clave": "sync", "titulo": "Sync opciones + purga Railway", "resultado": R.OK,
             "duracion_s": 72, "tabla": "opciones_snapshot",
             "rueda_antes": "2026-09-11", "rueda_despues": "2026-09-14"},
            {"clave": "paso1", "titulo": "Paso 1 - precios e indicadores", "resultado": R.PARCIAL,
             "duracion_s": 843, "notas": ["2 tickers sin la rueda 2026-09-14"],
             "pendientes": {"rueda": "2026-09-14",
                            "precios": {"SE": "2026-09-11", "A<B": None}, "futuros": {}},
             "huecos": {"precios_diarios": {"n": 1, "huecos": {"2026-09-10": ["KO"]}}}},
            {"clave": "paso2", "titulo": "Paso 2 - features", "resultado": resultado_paso2,
             "duracion_s": 290, "notas": ["termino con codigo 1"] if resultado_paso2 == R.ERROR else []},
        ],
    }


def test_telegram_detalla_los_tickers_pendientes_con_su_ultimo_dato():
    msg = R.mensaje_telegram(_corrida())
    assert "SE (ultimo dato 2026-09-11)" in msg
    assert "sin datos" in msg                 # ticker que nunca tuvo dato
    assert "A&lt;B" in msg and "A<B" not in msg  # parse_mode HTML: todo escapado
    assert "2026-09-10: 1 ticker (KO)" in msg
    assert R.estado_general(_corrida()) == "CON AVISOS"


def test_rutina_frenada_dice_como_retomar():
    c = _corrida(frenada_en="paso2", resultado_paso2=R.ERROR)
    txt = R.resumen_texto(c)
    assert "FRENADA en Paso 2 - features" in txt
    assert "--desde paso2" in txt and "cron_paso2_features.bat" in txt
    assert R.codigo_salida(c) == 1


def test_resumen_es_ascii_puro():
    R.resumen_texto(_corrida()).encode("ascii")


def test_codigos_de_salida():
    ok = _corrida()
    ok["pasos"] = [dict(p, resultado=R.OK, notas=[], pendientes=None, huecos=None) for p in ok["pasos"]]
    assert R.codigo_salida(ok) == 0
    assert R.codigo_salida(_corrida()) == 2


def test_fmt_duracion():
    assert R.fmt_duracion(None) == "-"
    assert R.fmt_duracion(42) == "42s"
    assert R.fmt_duracion(75) == "1m 15s"
    assert R.fmt_duracion(3725) == "1h 02m"


# ── Ejecutor: el codigo de salida es el del proceso ───────────────────────────

def _cargar_rutina_diaria():
    ruta = os.path.join(ROOT, "scripts", "manual", "rutina_diaria.py")
    spec = importlib.util.spec_from_file_location("rutina_diaria_test", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ejecutor_devuelve_el_codigo_real_y_guarda_toda_la_salida(tmp_path):
    rd = _cargar_rutina_diaria()
    log = rd.Log(str(tmp_path / "paso.log"))
    codigo = rd.ejecutar(
        [sys.executable, "-c",
         "import sys; print('salida normal'); print('salida de error', file=sys.stderr); sys.exit(3)"],
        log)
    log.cerrar()
    assert codigo == 3
    contenido = (tmp_path / "paso.log").read_text(encoding="utf-8")
    assert "salida normal" in contenido and "salida de error" in contenido
