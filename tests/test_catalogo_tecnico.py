"""
test_catalogo_tecnico.py
Guards del catalogo de indicadores y estrategias (src/utils/catalogo_tecnico.py).

El catalogo es DOCUMENTACION EJECUTABLE: describe codigo que vive en otro lado.
El riesgo no es que se rompa, es que se quede ATRAS en silencio -- se agrega una
estrategia y la planilla sigue mostrando 13. Estos tests existen para que eso
falle en la suite y no en el Excel.

Todo lo de aca es PURO: sin DB. Lo que necesita `ft_estrategias` lo valida
`exportar_catalogo.py --check`.
"""

import io
import os
import re

import pytest

from src.utils import catalogo_tecnico as cat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED = os.path.join(ROOT, "scripts", "forward_testing", "ft_setup_estrategias.py")


# ── Hoja Indicadores ──────────────────────────────────────────────────────────

def test_los_grupos_cubren_todos_los_indicadores():
    """Si se agrega un indicador fuera de un grupo, la hoja lo perderia."""
    assert sum(n for _, n in cat.GRUPOS) == len(cat.INDICADORES)


def test_ningun_indicador_tiene_campos_vacios():
    for ind in cat.INDICADORES:
        assert ind.sigla.strip(), ind
        assert ind.nombre.strip(), ind.sigla
        assert ind.funcion.strip(), ind.sigla
        assert ind.origen.strip(), ind.sigla


def test_no_hay_siglas_duplicadas_dentro_del_mismo_grupo():
    """Una sigla puede repetirse ENTRE grupos (los patrones de vela viejos y
    nuevos se llaman igual, y es justamente lo que hay que poder comparar),
    pero no dentro de uno."""
    i = 0
    for titulo, n in cat.GRUPOS:
        siglas = [x.sigla for x in cat.INDICADORES[i:i + n]]
        assert len(siglas) == len(set(siglas)), f"duplicado en {titulo}"
        i += n


# ── Hojas de estrategia ───────────────────────────────────────────────────────

def test_logicas_uso_y_cerebro_describen_las_mismas_familias():
    assert set(cat.LOGICAS) == set(cat.USO)
    assert set(cat.LOGICAS) == set(cat.CEREBRO)


def test_toda_logica_del_seed_esta_catalogada():
    """El guard que importa: una estrategia nueva no puede quedar sin ficha.

    Se lee el .py como TEXTO a proposito: importarlo ejecuta
    configurar_entorno_local(), que toca os.environ.
    """
    txt = io.open(SEED, encoding="utf-8").read()
    logicas = set(re.findall(r'"logica":\s*"([a-z0-9_]+)"', txt))
    assert logicas, "no se encontro ninguna logica en el seed: cambio el formato?"
    faltan = sorted(logicas - set(cat.LOGICAS))
    assert not faltan, f"logicas sin ficha en el catalogo: {faltan}"


def test_toda_estrategia_del_seed_tiene_bot_mapeado():
    txt = io.open(SEED, encoding="utf-8").read()
    nombres = set(re.findall(r'"nombre":\s*"(FT_[A-Za-z0-9_]+)"', txt))
    assert nombres
    faltan = sorted(nombres - set(cat.BOTS))
    assert not faltan, f"estrategias sin .py mapeado: {faltan}"


def test_los_bots_mapeados_existen_en_disco():
    for nombre, ruta in cat.BOTS.items():
        # La ruta puede traer aclaraciones: "...py --ventana 5" / "...py (nota)"
        archivo = ruta.split(" ")[0]
        assert os.path.isfile(os.path.join(ROOT, archivo)), f"{nombre}: {archivo}"


def test_todo_parametro_citado_tiene_descripcion():
    citados = {p for filas in cat.USO.values() for f in filas for p in f.params}
    faltan = sorted(citados - set(cat.PARAMETROS))
    assert not faltan, f"parametros sin descripcion: {faltan}"


def test_las_filas_de_uso_estan_completas():
    for logica, filas in cat.USO.items():
        assert filas, f"{logica} no tiene ni una fila de insumos"
        for f in filas:
            assert f.indicador.strip(), logica
            assert f.rol.strip(), f"{logica} / {f.indicador}"


def test_no_parametrizado_solo_explica_claves_reales():
    """Si una clave de NO_PARAMETRIZADO ya no se cita en ninguna fila, la
    explicacion quedo huerfana."""
    citados = {p for filas in cat.USO.values() for f in filas for p in f.params}
    huerfanas = sorted(set(cat.NO_PARAMETRIZADO) - citados)
    assert not huerfanas, f"explicaciones que ya no aplican: {huerfanas}"


# ── Hoja Scores ───────────────────────────────────────────────────────────────

def test_los_scores_tienen_condicion_y_capa():
    for nombre, filas in cat.SCORES.items():
        assert filas, nombre
        for f in filas:
            assert f.capa.strip(), nombre
            assert f.condicion.strip(), f"{nombre} / {f.capa}"


def test_el_score_tecnico_suma_55():
    """Los puntos escritos en la ficha tienen que dar el maximo declarado por
    el codigo que de verdad puntua."""
    from src.strategies.scoring import SCORE_MAXIMO_TECH

    suma = 0.0
    for f in cat.SCORE_TECNICO:
        if f.capa == "TOTAL":
            continue   # esa fila REPITE el maximo, no suma una condicion
        try:
            suma += float(f.puntos.replace("+", "").replace(",", "."))
        except ValueError:
            continue   # "obligatoria": habilita, no puntua
    assert suma == pytest.approx(SCORE_MAXIMO_TECH)


def test_la_banda_de_rsi_de_la_ficha_es_la_del_codigo():
    from src.strategies.scoring import RSI_MAX, RSI_MIN

    fila = [f for f in cat.SCORE_TECNICO if "RSI14" in f.condicion]
    assert len(fila) == 1
    assert str(int(RSI_MIN)) in fila[0].condicion
    assert str(int(RSI_MAX)) in fila[0].condicion


# ── Reglas del proyecto ───────────────────────────────────────────────────────

def test_el_catalogo_es_ascii_puro():
    """Regla 2 de CLAUDE.md: Windows cp1252. Este modulo es todo texto, asi que
    es el mas expuesto a que se cuele un acento o una flecha Unicode."""
    txt = io.open(os.path.join(ROOT, "src", "utils", "catalogo_tecnico.py"),
                  encoding="utf-8").read()
    malos = sorted({c for c in txt if ord(c) > 127})
    assert not malos, f"no-ASCII en el catalogo: {malos}"


def test_el_exportador_es_ascii_puro():
    txt = io.open(os.path.join(ROOT, "scripts", "manual", "exportar_catalogo.py"),
                  encoding="utf-8").read()
    malos = sorted({c for c in txt if ord(c) > 127})
    assert not malos, f"no-ASCII en el exportador: {malos}"


def test_el_metodo_no_queda_vacio():
    """La hoja Metodo es la que evita confundir ruido con mejora: si alguien la
    vacia, el archivo pierde su guardarrail."""
    assert len(cat.METODO) >= 10
    for regla, detalle in cat.METODO:
        assert regla.strip() and len(detalle) > 80, regla
