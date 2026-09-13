"""
test_ft_tramos.py -- medir un cambio de FT por tramos, con grupo de control.

Casos sinteticos y deterministas (sin DB). El caso central reproduce el motivo
del grupo de control: con 60% de exposicion, una estrategia SIN cambio real
"le gana" al universo cuando el mercado cae, y contra el control no.
"""

from datetime import date, timedelta

import pytest

from src.utils import ft_tramos as ft

D0 = date(2026, 1, 1)


def d(i):
    return D0 + timedelta(days=i)


def serie_desde_retornos(retornos, base=100_000.0):
    """[(fecha, equity)] con la primera fecha como base (sin retorno)."""
    eq = base
    out = [(d(0), eq)]
    for i, r in enumerate(retornos, start=1):
        eq *= 1.0 + r
        out.append((d(i), eq))
    return out


def alterna(i, a):
    """Ruido determinista de media exactamente cero en ventanas pares."""
    return a if i % 2 == 0 else -a


# ── Cortes y tramos ───────────────────────────────────────────────────────────

def test_cortes_solo_de_cambios_que_cambian_decisiones_y_la_incluyen():
    cambios = [
        {"fecha_efectiva": d(10), "estrategias": [1, 4], "cambia_decisiones": True},
        {"fecha_efectiva": d(20), "estrategias": [4], "cambia_decisiones": False},
        {"fecha_efectiva": d(30), "estrategias": [2], "cambia_decisiones": True},
        {"fecha_efectiva": d(10), "estrategias": [4], "cambia_decisiones": True},
    ]
    assert ft.cortes_de(4, cambios) == [d(10)]
    assert ft.cortes_de(2, cambios) == [d(30)]
    assert ft.cortes_de(9, cambios) == []


def test_el_retorno_del_dia_del_corte_pertenece_al_tramo_anterior():
    fechas = [d(i) for i in range(11)]          # 10 retornos: d1..d10
    t = ft.tramos(fechas, [d(4)])
    assert t == [{"desde": None, "hasta": d(4), "n_ruedas": 4},
                 {"desde": d(4), "hasta": None, "n_ruedas": 6}]


def test_corte_en_el_inicio_se_ignora_y_corte_futuro_abre_tramo_vacio():
    fechas = [d(i) for i in range(5)]
    assert len(ft.tramos(fechas, [d(0)])) == 1
    t = ft.tramos(fechas, [d(9)])
    assert t[-1] == {"desde": d(9), "hasta": None, "n_ruedas": 0}


def test_retornos_por_fecha_no_salta_huecos():
    r = ft.retornos_por_fecha([(d(0), 100), (d(1), 110), (d(2), None), (d(3), 121)])
    assert r == {d(1): pytest.approx(0.10)}


# ── Operaciones ───────────────────────────────────────────────────────────────

OPS = [
    {"estrategia_id": 1, "f_entrada": d(1), "f_salida": d(3), "pnl": 10, "pnl_pct": 1.0},
    {"estrategia_id": 1, "f_entrada": d(3), "f_salida": d(5), "pnl": -5, "pnl_pct": -0.5},   # cruza
    {"estrategia_id": 1, "f_entrada": d(5), "f_salida": d(7), "pnl": 20, "pnl_pct": 2.0},    # entra en el corte
    {"estrategia_id": 1, "f_entrada": d(1), "f_salida": d(5), "pnl": -9, "pnl_pct": -9.0},   # sale en el corte
    {"estrategia_id": 1, "f_entrada": d(6), "f_salida": None, "pnl": None, "pnl_pct": None},  # abierta
    {"estrategia_id": 1, "f_entrada": d(1), "f_salida": d(2), "pnl": -80, "pnl_pct": -80.0,
     "motivo_salida": "STOP_LOSS_ATR_SPLIT_FIX"},
]


def test_una_decision_con_el_dato_del_corte_ya_es_regla_nueva():
    antes, cf = ft.operaciones_de_tramo(OPS, None, d(5))
    despues, _ = ft.operaciones_de_tramo(OPS, d(5), None)
    assert [o["pnl"] for o in antes] == [10]
    assert [o["pnl"] for o in despues] == [20]
    assert cf == 1


def test_cruzan_el_corte_las_que_entraron_antes_y_salieron_en_o_despues():
    assert ft.operaciones_que_cruzan(OPS, d(5)) == 2


# ── Inferencia ────────────────────────────────────────────────────────────────

def test_t_critico_95_contra_tabla():
    assert ft.t_critico_95(10) == pytest.approx(2.228, abs=0.005)
    assert ft.t_critico_95(19) == pytest.approx(2.093, abs=0.002)
    assert ft.t_critico_95(2) == pytest.approx(4.303)
    assert ft.t_critico_95(10_000) == pytest.approx(1.960, abs=0.001)


def test_muestra_chica_es_insuficiente_y_no_publica_numeros():
    r = ft.comparar_medias([0.01] * 5, [0.02] * 30, minimo=20)
    assert r["veredicto"] == ft.INSUFICIENTE
    assert r["diferencia"] is None and r["ic95_lo"] is None
    assert (r["n_antes"], r["n_despues"]) == (5, 30)


def test_diferencia_clara_mejora_y_ruido_no_concluye():
    antes = [alterna(i, 0.01) for i in range(30)]
    mejora = ft.comparar_medias(antes, [0.02 + x for x in antes], minimo=20)
    assert mejora["veredicto"] == ft.MEJORA
    assert mejora["diferencia"] == pytest.approx(0.02)

    peor = ft.comparar_medias(antes, [x - 0.02 for x in antes], minimo=20)
    assert peor["veredicto"] == ft.EMPEORA

    ruido = ft.comparar_medias(antes, [x + 0.001 for x in antes], minimo=20)
    assert ruido["veredicto"] == ft.NO_CONCLUYENTE


def test_escala_multiplica_todos_los_numeros():
    antes = [alterna(i, 0.001) for i in range(20)]
    r1 = ft.comparar_medias(antes, [0.004 + x for x in antes], minimo=20)
    r2 = ft.comparar_medias(antes, [0.004 + x for x in antes], minimo=20, escala=2100)
    assert r2["diferencia"] == pytest.approx(r1["diferencia"] * 2100)
    assert r2["ic95_hi"] == pytest.approx(r1["ic95_hi"] * 2100)
    assert r2["veredicto"] == r1["veredicto"]


def test_diferencia_en_diferencias_resta_lo_que_le_paso_al_control():
    # Mercado del tramo: +2 por operacion antes del corte, -1 despues, para todos.
    base = [alterna(i, 1.0) for i in range(20)]
    antes_s, despues_s = [2 + x for x in base], [-1 + x for x in base]
    antes_c, despues_c = [2 + x for x in base], [-1 + x for x in base]

    # Sin control, cualquier estrategia "empeora" solo por el mercado.
    assert ft.comparar_medias(antes_s, despues_s, minimo=10)["veredicto"] == ft.EMPEORA

    placebo = ft.comparar_diferencias(antes_s, despues_s, antes_c, despues_c, minimo=10)
    assert placebo["veredicto"] == ft.NO_CONCLUYENTE
    assert placebo["diferencia"] == pytest.approx(0.0)

    real = ft.comparar_diferencias(antes_s, [x + 2 for x in despues_s],
                                   antes_c, despues_c, minimo=10)
    assert real["veredicto"] == ft.MEJORA
    assert real["diferencia"] == pytest.approx(2.0)

    control_chico = ft.comparar_diferencias(antes_s, despues_s, antes_c[:3], despues_c, minimo=10)
    assert control_chico["veredicto"] == ft.INSUFICIENTE
    assert control_chico["diferencia"] is None


def test_bootstrap_determinista_y_cubre_la_media():
    vals = [alterna(i, 0.01) + 0.002 for i in range(40)]
    media = lambda xs: sum(xs) / len(xs)  # noqa: E731
    ic1 = ft.ic95_bootstrap(vals, media, n_boot=500)
    ic2 = ft.ic95_bootstrap(vals, media, n_boot=500)
    assert ic1 == ic2
    assert ic1[0] < 0.002 < ic1[1]
    assert ft.ic95_bootstrap([1.0], media) is None
    assert ft.ic95_bootstrap(vals, lambda xs: None) is None


# ── Grupo de control ──────────────────────────────────────────────────────────

def test_control_excluye_a_las_que_cambiaron_sus_reglas_en_la_ventana():
    cambio = {"fecha_efectiva": d(30), "estrategias": [1], "cambia_decisiones": True}
    cambios = [cambio,
               {"fecha_efectiva": d(40), "estrategias": [2], "cambia_decisiones": True},
               {"fecha_efectiva": d(90), "estrategias": [3], "cambia_decisiones": True},
               {"fecha_efectiva": d(45), "estrategias": [4], "cambia_decisiones": False}]
    control, excl = ft.grupo_control(cambio, [1, 2, 3, 4], cambios, d(10), d(60))
    assert control == [3, 4]
    assert excl == [2]


# ── Evaluacion completa ───────────────────────────────────────────────────────

N_LADO = 30   # retornos por lado; par para que el ruido alternado tenga media 0


def _mercado(i):
    return 0.001 if i <= N_LADO else -0.01     # el mercado se cae despues del corte


def _escenario():
    exp = 0.6   # exposicion de las estrategias: el 40% restante es caja
    idx = range(1, 2 * N_LADO + 1)
    serie = lambda f: serie_desde_retornos([f(i) for i in idx])  # noqa: E731
    series = {
        # 1: el cambio le agrega +0,4% diario real
        1: serie(lambda i: exp * _mercado(i) + alterna(i, 0.0005) + (0.004 if i > N_LADO else 0)),
        # 2: afectada en el papel, sin efecto real
        2: serie(lambda i: exp * _mercado(i) + alterna(i, 0.0005)),
        # 3 y 4: control; sus ruidos se cancelan entre si
        3: serie(lambda i: exp * _mercado(i) + alterna(i, 0.002)),
        4: serie(lambda i: exp * _mercado(i) - alterna(i, 0.002)),
    }
    bench = serie(_mercado)
    cambio = {"clave": "x", "titulo": "t", "fecha_efectiva": d(N_LADO),
              "estrategias": [1, 2], "cambia_decisiones": True}
    return cambio, series, bench


def test_el_control_separa_el_efecto_del_cambio_del_efecto_mercado():
    cambio, series, bench = _escenario()
    res = ft.evaluar_cambio(cambio, [cambio], series, benchmark=bench)
    assert res["control"] == [3, 4]
    filas = {f["estrategia_id"]: f for f in res["filas"]}

    real = filas[1]
    assert real["antes"]["n_ruedas"] == N_LADO and real["despues"]["n_ruedas"] == N_LADO
    assert real["vs_control"]["veredicto"] == ft.MEJORA
    assert real["vs_control"]["diferencia"] == pytest.approx(0.004 * 21 * 100)

    # Sin efecto real: contra el universo "mejora" solo por tener 40% de caja
    # en una caida; contra el control, que tiene la misma caja, no.
    placebo = filas[2]
    assert placebo["vs_universo"]["veredicto"] == ft.MEJORA
    assert placebo["vs_control"]["veredicto"] == ft.NO_CONCLUYENTE
    assert placebo["vs_control"]["diferencia"] == pytest.approx(0.0, abs=1e-9)


def test_pocas_operaciones_dan_expectancy_insuficiente():
    cambio, series, bench = _escenario()
    ops = [{"estrategia_id": 1, "f_entrada": d(2), "f_salida": d(5), "pnl": 1, "pnl_pct": 1.0},
           {"estrategia_id": 1, "f_entrada": d(28), "f_salida": d(35), "pnl": 1, "pnl_pct": 1.0}]
    res = ft.evaluar_cambio(cambio, [cambio], series, benchmark=bench, operaciones=ops)
    fila = next(f for f in res["filas"] if f["estrategia_id"] == 1)
    assert fila["expectancy"]["veredicto"] == ft.INSUFICIENTE
    assert fila["expectancy_vs_control"]["veredicto"] == ft.INSUFICIENTE
    assert fila["antes"]["trade"]["n"] == 1
    assert fila["n_cruzan"] == 1


def test_estrategia_nacida_con_el_cambio_no_se_compara():
    cambio, series, bench = _escenario()
    series[5] = [(f, v) for f, v in series[3] if f >= d(N_LADO)]
    cambio = dict(cambio, estrategias=[1, 5])
    res = ft.evaluar_cambio(cambio, [cambio], series, benchmark=bench)
    assert res["nacidas_con_el_cambio"] == [5]
    assert [f["estrategia_id"] for f in res["filas"]] == [1]


def test_sin_control_la_comparacion_principal_queda_vacia():
    cambio, series, bench = _escenario()
    cambio = dict(cambio, estrategias=[1, 2, 3, 4])
    res = ft.evaluar_cambio(cambio, [cambio], series, benchmark=bench)
    assert res["control"] == []
    assert all(f["vs_control"] is None for f in res["filas"])


def test_tramo_vigente_cuenta_desde_el_ultimo_corte():
    cambio, series, _ = _escenario()
    ops = [{"estrategia_id": 1, "f_entrada": d(31), "f_salida": d(33), "pnl": 1, "pnl_pct": 1.0}]
    v = ft.tramo_vigente(1, [cambio], series[1], ops)
    assert v["corte"] == d(N_LADO) and v["n_ruedas"] == N_LADO
    assert v["n_ops"] == 1 and v["faltan_ops"] == ft.MIN_OPS_TRAMO - 1
    sin = ft.tramo_vigente(3, [cambio], series[3], ops)
    assert sin["corte"] is None and sin["desde"] == d(0)
