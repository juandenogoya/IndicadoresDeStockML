"""
ft_comparar.py
Por que difieren FT_ML_SCANNER_v1 y FT_ML_SCANNER_v2 (Etapa 3f). Modulo PURO.

Diseno: docs/forward_testing/METRICAS.md, seccion 13.

Las dos estrategias tienen las mismas reglas y cambia el modelo ML. La comparacion
de cartera dice CUAL rinde mas; este modulo explica POR QUE eligen distinto:

    1. Senales, sobre las MISMAS filas de alertas_scanner: misma rueda, mismo
       ticker, mismo price action y score tecnico; cambian la probabilidad ML y
       sus cortes. Por rueda: senales de las dos, solo de la v1 y solo de la v2,
       con retorno real a 5 y 20 ruedas y exceso contra el universo de esa rueda.
       No depende de lo que operaron: una senal exclusiva cuenta aunque el tope de
       posiciones la haya dejado afuera.
    2. Atribucion de las exclusivas: cobertura del entrenamiento de la v1, nivel
       que les dio la otra version, probabilidades y sector.
    3. Operaciones: tickers que operaron las dos a la vez contra exclusivos,
       motivos de salida, permanencia y expectancy.
    4. Oportunidades: candidatos que el tope de 5 posiciones dejo afuera, con su
       retorno contrafactual.
    5. Cartera: diferencia diaria de retorno v2 - v1, pareada (mismos dias).

PURO (mismo contrato que ft_tramos.py): sin DB, config, archivos ni red. Entran
listas y dicts, sale un dict. `tablas()` lo pasa a texto plano, el mismo para el
reporte .md y la seccion del HTML.

CONVENCION DE FECHAS: todo se ancla en la rueda de DATOS (`precio_fecha` de la
alerta, `fecha_datos` de la operacion). El retorno a N ruedas de una senal de la
rueda D es close(D+N) / close(D) - 1, con D+N contado sobre las ruedas del MERCADO
(no las del ticker: un hueco en su serie no corre la ventana) y exigiendo el close
de ese dia exacto.

SPLITS: se lee precios_diarios tal como esta. Un split corregido deja la serie
continua (splits.py divide la historia), asi que una ventana que lo cruza es valida.
Uno sin corregir contamina sus ventanas hasta que se corrige, y como todo se
recalcula en cada corrida, corregirlo arregla la historia sola.

INDEPENDENCIA: las senales de una misma rueda comparten mercado. Por eso la
comparacion principal usa el EXCESO contra el promedio del universo en esa rueda
(descuenta el factor comun) y exige, ademas de un minimo de senales, un minimo de
ruedas distintas. Aun asi el IC95 trata las senales como independientes: si varias
comparten sector y semana, queda angosto de mas.
"""

import math
import statistics
from collections import Counter

from src.utils import ft_tramos
from src.utils.ft_metricas import metricas_trade

# Senal = lo que abre posicion en las dos estrategias (src/strategies/ml_scanner).
NIVEL_SENAL = "COMPRA_FUERTE"
SCORE_MIN_SENAL = 65

HORIZONTES = (5, 20)
GRUPOS = ("ambas", "solo_v1", "solo_v2")
ETIQUETAS = {"ambas": "Ambas", "solo_v1": "Solo v1", "solo_v2": "Solo v2"}

# Minimos para publicar un numero. Senales y operaciones usan el de operaciones de
# la Etapa 1. Las senales exigen ademas ruedas distintas: el primer dia de la v2 ya
# trae 14 exclusivas de UNA sola rueda, que son una sola observacion de mercado.
MIN_SENALES = ft_tramos.MIN_OPS_TRAMO
MIN_RUEDAS_SENAL = 5
MIN_OPS = ft_tramos.MIN_OPS_TRAMO
MIN_RUEDAS_CARTERA = ft_tramos.MIN_RUEDAS_TRAMO

INSUFICIENTE = ft_tramos.INSUFICIENTE
NO_CONCLUYENTE = ft_tramos.NO_CONCLUYENTE
POSITIVO = "POSITIVO"
NEGATIVO = "NEGATIVO"
A_FAVOR_V2 = "A FAVOR DE V2"
A_FAVOR_V1 = "A FAVOR DE V1"
AFUERA_MEJOR = "AFUERA RINDIO MAS"
ADENTRO_MEJOR = "ADENTRO RINDIO MAS"

_CAMPOS_NUMERICOS = ("media_antes", "media_despues", "diferencia", "se", "gl",
                     "ic95_lo", "ic95_hi")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _txt(v):
    return None if v is None else str(v).strip()


def _media(xs):
    return statistics.fmean(xs) if xs else None


def _releer(res, si_mejora, si_empeora):
    """Renombra el veredicto de ft_tramos.comparar_medias (despues - antes)."""
    res = dict(res)
    res["veredicto"] = {ft_tramos.MEJORA: si_mejora,
                        ft_tramos.EMPEORA: si_empeora}.get(res["veredicto"], res["veredicto"])
    return res


def resumir(valores, minimo=MIN_SENALES, ruedas=None, min_ruedas=0, escala=100.0):
    """
    Media de una muestra con IC95 (t de Student, n-1 grados de libertad) y
    porcentaje de valores positivos.

    `ruedas`: la rueda de cada valor; si llegan, se exige `min_ruedas` distintas.
    Por debajo de cualquiera de los dos minimos: INSUFICIENTE y sin numeros.
    `escala` multiplica media e intervalo (x100 = fraccion a %).
    """
    n = len(valores)
    n_ruedas = len(set(ruedas)) if ruedas is not None else None
    out = {"n": n, "n_ruedas": n_ruedas, "minimo": minimo, "min_ruedas": min_ruedas,
           "media": None, "ic95_lo": None, "ic95_hi": None, "positivos_pct": None,
           "veredicto": INSUFICIENTE}
    if n < max(minimo, 2) or (n_ruedas is not None and n_ruedas < min_ruedas):
        return out
    m = statistics.fmean(valores)
    se = math.sqrt(statistics.variance(valores) / n)
    t = ft_tramos.t_critico_95(n - 1)
    lo, hi = m - t * se, m + t * se
    out.update({
        "media": m * escala, "ic95_lo": lo * escala, "ic95_hi": hi * escala,
        "positivos_pct": sum(1 for v in valores if v > 0) / n * 100,
        "veredicto": POSITIVO if lo > 0 else NEGATIVO if hi < 0 else NO_CONCLUYENTE,
    })
    return out


# ── Senales ───────────────────────────────────────────────────────────────────

def es_senal(nivel, score, nivel_min=NIVEL_SENAL, score_min=SCORE_MIN_SENAL):
    """La regla de entrada de las dos estrategias sobre (nivel, score)."""
    return _txt(nivel) == nivel_min and score is not None and float(score) >= score_min


def grupo(fila):
    """'ambas' | 'solo_v1' | 'solo_v2' | None (ninguna da senal)."""
    s1 = es_senal(fila.get("nivel_v1"), fila.get("score_v1"))
    s2 = es_senal(fila.get("nivel_v2"), fila.get("score_v2"))
    if s1 and s2:
        return "ambas"
    if s1:
        return "solo_v1"
    if s2:
        return "solo_v2"
    return None


def filas_por_rueda(filas):
    """
    Una fila por (ticker, precio_fecha): la de la ultima corrida (`scan_fecha`).
    Si el scanner corrio dos veces sobre la misma rueda, los bots leyeron la ultima.
    """
    ultima = {}
    for f in filas:
        k = (f["ticker"], f["precio_fecha"])
        previa = ultima.get(k)
        if (previa is None or previa.get("scan_fecha") is None
                or (f.get("scan_fecha") is not None and f["scan_fecha"] >= previa["scan_fecha"])):
            ultima[k] = f
    return [ultima[k] for k in sorted(ultima, key=lambda k: (k[1], k[0]))]


def indice_ruedas(ruedas):
    return {f: i for i, f in enumerate(sorted(ruedas))}


def retorno_adelante(closes, ruedas, idx, ticker, fecha, n):
    """
    close(D+N) / close(D) - 1, con D+N sobre las ruedas del mercado.

    closes: {ticker: {fecha: close}}; ruedas: lista ordenada; idx: indice_ruedas.
    None si falta la rueda D+N o falta el close de D o de D+N.
    """
    i = idx.get(fecha)
    if i is None or i + n >= len(ruedas):
        return None
    fin = ruedas[i + n]
    serie = closes.get(ticker) or {}
    c0, c1 = serie.get(fecha), serie.get(fin)
    if c0 is None or c1 is None or c0 <= 0:
        return None
    return c1 / c0 - 1.0


def agregar_retornos(filas, closes, ruedas, horizontes=HORIZONTES, medias=None):
    """
    Copia de cada fila con `ret_N` (retorno a N ruedas) y `exc_N` (menos el promedio
    del universo en esa rueda y ventana).

    `medias` {(rueda, N): promedio}: None = calcularlo de las mismas filas, que en
    ese caso tienen que ser el universo entero de cada rueda. Para los candidatos se
    pasan las del universo.

    Devuelve (filas_nuevas, medias).
    """
    rs = sorted(ruedas)
    idx = indice_ruedas(rs)
    nuevas = []
    for f in filas:
        g = dict(f)
        for n in horizontes:
            g[f"ret_{n}"] = retorno_adelante(closes, rs, idx, f["ticker"],
                                             f["precio_fecha"], n)
        nuevas.append(g)

    if medias is None:
        acum = {}
        for g in nuevas:
            for n in horizontes:
                r = g[f"ret_{n}"]
                if r is not None:
                    acum.setdefault((g["precio_fecha"], n), []).append(r)
        medias = {k: statistics.fmean(v) for k, v in acum.items()}

    for g in nuevas:
        for n in horizontes:
            r, m = g[f"ret_{n}"], medias.get((g["precio_fecha"], n))
            g[f"exc_{n}"] = None if r is None or m is None else r - m
    return nuevas, medias


def solapamiento(filas):
    """
    Por rueda: senales de cada version, compartidas, exclusivas y Jaccard
    (compartidas / union). Un dia sin senales de ninguna tiene Jaccard None.
    """
    por = {}
    for f in filas:
        d = por.setdefault(f["precio_fecha"], {g: 0 for g in GRUPOS} | {"filas": 0})
        d["filas"] += 1
        g = grupo(f)
        if g:
            d[g] += 1

    def fila(fecha, d):
        union = d["ambas"] + d["solo_v1"] + d["solo_v2"]
        return {"rueda": fecha, "filas": d["filas"],
                "v1": d["ambas"] + d["solo_v1"], "v2": d["ambas"] + d["solo_v2"],
                "ambas": d["ambas"], "solo_v1": d["solo_v1"], "solo_v2": d["solo_v2"],
                "jaccard": d["ambas"] / union if union else None}

    diarias = [fila(f, por[f]) for f in sorted(por)]
    tot = {g: sum(d[g] for d in por.values()) for g in GRUPOS} | {
        "filas": sum(d["filas"] for d in por.values())}
    total = fila(None, tot)
    jac = [d["jaccard"] for d in diarias if d["jaccard"] is not None]
    total.update({"n_ruedas": len(diarias), "jaccard_medio_diario": _media(jac)})
    return {"ruedas": diarias, "total": total}


def resumen_senales(filas, horizontes=HORIZONTES):
    """
    Por grupo y horizonte: retorno y exceso medios (%) con IC95. Y la comparacion
    que decide: exceso de las exclusivas de la v2 contra las de la v1. Las
    compartidas son iguales en las dos; la diferencia de eleccion vive entera en
    las exclusivas.
    """
    por_grupo = {g: [f for f in filas if grupo(f) == g] for g in GRUPOS}
    out = {"grupos": {}, "v2_vs_v1": {}}
    for g, fg in por_grupo.items():
        hs = {}
        for n in horizontes:
            con = [f for f in fg if f.get(f"exc_{n}") is not None]
            rds = [f["precio_fecha"] for f in con]
            hs[n] = {
                "retorno": resumir([f[f"ret_{n}"] for f in con], MIN_SENALES, rds, MIN_RUEDAS_SENAL),
                "exceso": resumir([f[f"exc_{n}"] for f in con], MIN_SENALES, rds, MIN_RUEDAS_SENAL),
                "sin_retorno": len(fg) - len(con),
            }
        out["grupos"][g] = {"n": len(fg), "horizontes": hs}

    for n in horizontes:
        lados = [[f for f in por_grupo[g] if f.get(f"exc_{n}") is not None]
                 for g in ("solo_v1", "solo_v2")]
        res = ft_tramos.comparar_medias([f[f"exc_{n}"] for f in lados[0]],
                                        [f[f"exc_{n}"] for f in lados[1]],
                                        MIN_SENALES, escala=100.0)
        ruedas_lado = [len({f["precio_fecha"] for f in lado}) for lado in lados]
        res["n_ruedas_v1"], res["n_ruedas_v2"] = ruedas_lado
        res["min_ruedas"] = MIN_RUEDAS_SENAL
        if min(ruedas_lado) < MIN_RUEDAS_SENAL:
            res.update({k: None for k in _CAMPOS_NUMERICOS})
            res["veredicto"] = INSUFICIENTE
        out["v2_vs_v1"][n] = _releer(res, A_FAVOR_V2, A_FAVOR_V1)
    return out


def atribucion(filas, entrenados_v1):
    """
    Por que una senal es exclusiva. Las dos versiones comparten todo el score menos
    los puntos ML, asi que la diferencia de score ES la diferencia de puntos ML.

    Por grupo:
        sin_entrenar_v1   tickers que la v1 no vio al entrenar (su probabilidad
                          sale de un modelo sin historia del ticker). Leer contra
                          `universo_sin_entrenar_pct`, la tasa base de las filas.
        nivel_otra        nivel que le dio la otra version: COMPRA es un desacuerdo
                          de borde, NEUTRAL o menos es un desacuerdo de fondo.
        prob_v1 / prob_v2 probabilidad media de cada modelo.
        dif_score         score v2 - score v1 medio (= puntos ML).
        sectores          [(sector, n)] de mayor a menor.
    """
    entrenados = set(entrenados_v1 or ())
    n_filas = len(filas)
    out = {"universo_filas": n_filas,
           "universo_sin_entrenar_pct": (sum(1 for f in filas if f["ticker"] not in entrenados)
                                         / n_filas * 100 if n_filas else None),
           "grupos": {}}
    for g, otra in (("ambas", None), ("solo_v1", "nivel_v2"), ("solo_v2", "nivel_v1")):
        fg = [f for f in filas if grupo(f) == g]
        n = len(fg)
        sin = sum(1 for f in fg if f["ticker"] not in entrenados)
        difs = [float(f["score_v2"]) - float(f["score_v1"]) for f in fg
                if f.get("score_v1") is not None and f.get("score_v2") is not None]
        out["grupos"][g] = {
            "n": n,
            "sin_entrenar_v1": sin,
            "sin_entrenar_pct": sin / n * 100 if n else None,
            "nivel_otra": (Counter(_txt(f.get(otra)) or "SIN NIVEL" for f in fg).most_common()
                           if otra else []),
            "prob_v1": _media([float(f["prob_v1"]) for f in fg if f.get("prob_v1") is not None]),
            "prob_v2": _media([float(f["prob_v2"]) for f in fg if f.get("prob_v2") is not None]),
            "dif_score": _media(difs),
            "sectores": Counter(_txt(f.get("sector")) or "SIN SECTOR" for f in fg).most_common(),
        }
    return out


# ── Operaciones ───────────────────────────────────────────────────────────────

def motivo_base(motivo):
    """SCORE_DEGRADADO_COMPRA_67 -> SCORE_DEGRADADO. El resto queda igual."""
    if not motivo:
        return None
    return "SCORE_DEGRADADO" if motivo.startswith("SCORE_DEGRADADO") else motivo


def _solapan(a, b, hoy):
    """Mismo ticker con posicion abierta en las dos al mismo tiempo."""
    if a["ticker"] != b["ticker"]:
        return False
    fin_a, fin_b = a.get("f_salida") or hoy, b.get("f_salida") or hoy
    if fin_a is None or fin_b is None:
        return False
    return a["f_entrada"] <= fin_b and b["f_entrada"] <= fin_a


def _pct_cerradas(ops):
    return [float(o["pnl_pct"]) for o in ops
            if o.get("f_salida") is not None and o.get("pnl_pct") is not None
            and not ft_tramos.es_contrafactual(o)]


def operaciones_comparadas(ops_v1, ops_v2, ruedas, hoy):
    """
    ops: [{ticker, f_entrada, f_salida (None si abierta), pnl, pnl_pct,
    motivo_salida}] con fechas de DATO.

    Una operacion es COMPARTIDA si la otra version tuvo el mismo ticker abierto en
    algun dia en comun (`hoy` cierra las abiertas). Permanencia en ruedas del
    mercado. Las `_SPLIT_FIX` no entran en las metricas (salida contrafactual).
    """
    idx = indice_ruedas(ruedas)
    out = {}
    for nombre, propias, otras in (("v1", ops_v1, ops_v2), ("v2", ops_v2, ops_v1)):
        filas = []
        for op in propias:
            f = dict(op)
            f["compartida"] = any(_solapan(op, o, hoy) for o in otras)
            fe, fs = op["f_entrada"], op.get("f_salida")
            f["ruedas"] = idx[fs] - idx[fe] if fs in idx and fe in idx else None
            filas.append(f)
        salidas = [f for f in filas if f.get("f_salida") is not None]
        cerradas = [f for f in salidas if not ft_tramos.es_contrafactual(f)]
        out[nombre] = {
            "n": len(filas),
            "abiertas": len(filas) - len(salidas),
            "cerradas": len(cerradas),
            "contrafactuales": len(salidas) - len(cerradas),
            "compartidas": sum(1 for f in filas if f["compartida"]),
            "exclusivas": sum(1 for f in filas if not f["compartida"]),
            "motivos": Counter(motivo_base(f.get("motivo_salida")) or "SIN MOTIVO"
                               for f in cerradas).most_common(),
            "ruedas_media": _media([f["ruedas"] for f in cerradas if f["ruedas"] is not None]),
            "trade": metricas_trade(cerradas),
            "trade_compartidas": metricas_trade([f for f in cerradas if f["compartida"]]),
            "trade_exclusivas": metricas_trade([f for f in cerradas if not f["compartida"]]),
            "tickers_exclusivos": sorted({f["ticker"] for f in filas if not f["compartida"]}),
        }
    out["expectancy_v2_vs_v1"] = _releer(
        ft_tramos.comparar_medias(_pct_cerradas(ops_v1), _pct_cerradas(ops_v2), MIN_OPS),
        A_FAVOR_V2, A_FAVOR_V1)
    return out


def resumen_oportunidades(candidatos, horizontes=HORIZONTES):
    """
    candidatos: filas con `version`, `entro`, `precio_fecha` y `exc_N`
    (agregar_retornos con las medias del universo). Por version y horizonte: exceso
    de los que entraron y de los que el tope dejo afuera, y afuera - adentro.
    """
    out = {}
    for v in ("v1", "v2"):
        propios = [c for c in candidatos if c.get("version") == v]
        hs = {}
        for n in horizontes:
            lados = {clave: [c for c in propios if bool(c.get("entro")) is entro
                             and c.get(f"exc_{n}") is not None]
                     for clave, entro in (("adentro", True), ("afuera", False))}
            hs[n] = {clave: resumir([c[f"exc_{n}"] for c in con], MIN_SENALES,
                                    [c["precio_fecha"] for c in con], MIN_RUEDAS_SENAL)
                     for clave, con in lados.items()}
            res = ft_tramos.comparar_medias([c[f"exc_{n}"] for c in lados["adentro"]],
                                            [c[f"exc_{n}"] for c in lados["afuera"]],
                                            MIN_SENALES, escala=100.0)
            hs[n]["afuera_vs_adentro"] = _releer(res, AFUERA_MEJOR, ADENTRO_MEJOR)
        out[v] = {"n": len(propios), "entraron": sum(1 for c in propios if c.get("entro")),
                  "horizontes": hs}
    return out


# ── Cartera ───────────────────────────────────────────────────────────────────

def cartera_pareada(serie_v1, serie_v2):
    """
    Diferencia diaria de retorno v2 - v1 en los dias que tienen las dos, en puntos
    por mes con IC95 (muestra pareada: el mismo mercado cada dia). Retorno
    acumulado de cada una sobre esos mismos dias.
    """
    r1 = ft_tramos.retornos_por_fecha(serie_v1 or [])
    r2 = ft_tramos.retornos_por_fecha(serie_v2 or [])
    comunes = sorted(set(r1) & set(r2))
    res = resumir([r2[f] - r1[f] for f in comunes], MIN_RUEDAS_CARTERA,
                  escala=ft_tramos.RUEDAS_MES * 100)
    res["veredicto"] = {POSITIVO: A_FAVOR_V2, NEGATIVO: A_FAVOR_V1}.get(res["veredicto"],
                                                                        res["veredicto"])
    res.update({
        "desde": comunes[0] if comunes else None,
        "hasta": comunes[-1] if comunes else None,
        "retorno_v1_pct": ft_tramos.acumulado_pct(r1[f] for f in comunes) if comunes else None,
        "retorno_v2_pct": ft_tramos.acumulado_pct(r2[f] for f in comunes) if comunes else None,
    })
    return res


# ── Todo junto ────────────────────────────────────────────────────────────────

def comparar(filas, closes, ruedas, entrenados_v1, operaciones=None, candidatos=None,
             equity=None, hoy=None, horizontes=HORIZONTES):
    """
    Analisis completo v1 contra v2.

    Args:
        filas:         alertas_scanner del periodo, TODAS las filas de cada rueda:
                       {ticker, sector, scan_fecha, precio_fecha, nivel_v1,
                       score_v1, prob_v1, nivel_v2, score_v2, prob_v2}
        closes:        {ticker: {fecha: close}} de precios_diarios
        ruedas:        ruedas del mercado, ordenables
        entrenados_v1: tickers que estaban en el entrenamiento de la v1
        operaciones:   {"v1": [...], "v2": [...]} (ver operaciones_comparadas)
        candidatos:    [{version, ticker, precio_fecha, entro}] de ft_candidatos_diarios
        equity:        {"v1": [(fecha, equity)], "v2": [...]} de ft_equity_diaria
        hoy:           ultima rueda, cierra las operaciones abiertas al solaparlas

    El exceso de cada fila se mide contra el promedio de TODAS las filas de su
    rueda, tengan o no la v2. Solo las filas con las dos versiones entran en la
    comparacion de senales.
    """
    unicas = filas_por_rueda(filas)
    con_ret, medias = agregar_retornos(unicas, closes, ruedas, horizontes)
    comparables = [f for f in con_ret if _txt(f.get("nivel_v2")) and _txt(f.get("nivel_v1"))]
    fechas = sorted({f["precio_fecha"] for f in comparables})

    out = {
        "filas": len(comparables),
        "filas_sin_v2": len(con_ret) - len(comparables),
        "desde": fechas[0] if fechas else None,
        "hasta": fechas[-1] if fechas else None,
        "horizontes": tuple(horizontes),
        "solapamiento": solapamiento(comparables),
        "senales": resumen_senales(comparables, horizontes),
        "atribucion": atribucion(comparables, entrenados_v1),
        "operaciones": None,
        "oportunidades": None,
        "cartera": None,
    }
    if operaciones is not None:
        out["operaciones"] = operaciones_comparadas(operaciones.get("v1", []),
                                                    operaciones.get("v2", []),
                                                    ruedas, hoy)
    if candidatos is not None:
        cands, _ = agregar_retornos(candidatos, closes, ruedas, horizontes, medias=medias)
        out["oportunidades"] = resumen_oportunidades(cands, horizontes)
    if equity is not None:
        out["cartera"] = cartera_pareada(equity.get("v1"), equity.get("v2"))
    return out


# ── Texto (mismo para el .md y el HTML) ───────────────────────────────────────

def _f(v, dec=2):
    return "-" if v is None else f"{v:+.{dec}f}"


def _pct(v):
    return "-" if v is None else f"{v * 100:.0f}%"


def texto_media(r, dec=2):
    """resumir() -> '+1.23 [-0.40, +2.86] NO CONCLUYENTE' o 'INSUFICIENTE (...)'."""
    if r is None:
        return "-"
    if r["veredicto"] == INSUFICIENTE:
        if r.get("n_ruedas") is not None and r.get("min_ruedas"):
            return (f"INSUFICIENTE (n {r['n']} en {r['n_ruedas']} ruedas; "
                    f"min {r['minimo']} en {r['min_ruedas']})")
        return f"INSUFICIENTE (n {r['n']}; min {r['minimo']})"
    return f"{_f(r['media'], dec)} [{_f(r['ic95_lo'], dec)}, {_f(r['ic95_hi'], dec)}] {r['veredicto']}"


def texto_diferencia(r, lados=("v1", "v2"), dec=2):
    """Una comparar_medias releida -> texto con los dos n si es INSUFICIENTE."""
    if r is None:
        return "-"
    if r["veredicto"] == INSUFICIENTE:
        extra = ""
        if "n_ruedas_v1" in r:
            extra = (f"; ruedas {r['n_ruedas_v1']} / {r['n_ruedas_v2']}, "
                     f"min {r['min_ruedas']}")
        return (f"INSUFICIENTE (n {lados[0]} {r['n_antes']} / {lados[1]} "
                f"{r['n_despues']}, min {r['minimo']}{extra})")
    return (f"{_f(r['diferencia'], dec)} [{_f(r['ic95_lo'], dec)}, "
            f"{_f(r['ic95_hi'], dec)}] {r['veredicto']}")


def _expectancy(trade):
    e = trade.get("expectancy_pct")
    return "-" if e is None else f"{e:+.2f} (n {trade['n']})"


def _pares(items, maximo=None):
    items = list(items)[:maximo] if maximo else list(items)
    return ", ".join(f"{k} {v}" for k, v in items) or "-"


def tablas(res, max_ruedas=10):
    """
    El resultado de comparar() como tablas de texto plano ASCII:
    [{clave, titulo, nota, columnas, filas}]. Las celdas son str.
    """
    hs = res["horizontes"]
    out = []

    sol, t = res["solapamiento"], res["solapamiento"]["total"]
    out.append({
        "clave": "solapamiento",
        "titulo": "Solapamiento de senales",
        "nota": (f"Senal = {NIVEL_SENAL} con score >= {SCORE_MIN_SENAL}, la regla de entrada "
                 f"de las dos. En {t['n_ruedas']} ruedas: v1 {t['v1']}, v2 {t['v2']}, "
                 f"ambas {t['ambas']}, solo v1 {t['solo_v1']}, solo v2 {t['solo_v2']}. "
                 f"Jaccard (compartidas / union): total {_pct(t['jaccard'])}, medio diario "
                 f"{_pct(t['jaccard_medio_diario'])}. Ultimas {max_ruedas} ruedas, la mas "
                 f"reciente arriba."),
        "columnas": ["Rueda", "Filas", "v1", "v2", "Ambas", "Solo v1", "Solo v2", "Jaccard"],
        "filas": [[str(d["rueda"]), str(d["filas"]), str(d["v1"]), str(d["v2"]),
                   str(d["ambas"]), str(d["solo_v1"]), str(d["solo_v2"]), _pct(d["jaccard"])]
                  for d in reversed(sol["ruedas"][-max_ruedas:])],
    })

    sen = res["senales"]
    filas = []
    for g in GRUPOS:
        info = sen["grupos"][g]
        fila = [ETIQUETAS[g], str(info["n"])]
        for n in hs:
            h = info["horizontes"][n]
            fila += [texto_media(h["retorno"]), texto_media(h["exceso"])]
        filas.append(fila)
    fila = ["Solo v2 - solo v1", ""]
    for n in hs:
        fila += ["-", texto_diferencia(sen["v2_vs_v1"][n])]
    filas.append(fila)
    out.append({
        "clave": "senales",
        "titulo": "Retorno de las senales",
        "nota": ("Retorno real a N ruedas desde la rueda de la senal, en %, con IC95. Exceso = "
                 "menos el promedio del universo en la misma rueda y ventana. Las compartidas "
                 "son iguales en las dos: la diferencia de eleccion esta en las exclusivas, y "
                 "la ultima fila es la comparacion que importa. Minimo "
                 f"{MIN_SENALES} senales en {MIN_RUEDAS_SENAL} ruedas distintas."),
        "columnas": ["Grupo", "Senales"] + [c for n in hs for c in (f"Retorno {n}r %",
                                                                     f"Exceso {n}r %")],
        "filas": filas,
    })

    atr = res["atribucion"]
    filas = []
    for g in GRUPOS:
        a = atr["grupos"][g]
        filas.append([
            ETIQUETAS[g], str(a["n"]),
            "-" if a["sin_entrenar_pct"] is None else f"{a['sin_entrenar_v1']} ({a['sin_entrenar_pct']:.0f}%)",
            _pares(a["nivel_otra"]) if g != "ambas" else "-",
            "-" if a["prob_v1"] is None else f"{a['prob_v1']:.3f}",
            "-" if a["prob_v2"] is None else f"{a['prob_v2']:.3f}",
            _f(a["dif_score"], 1),
            _pares(a["sectores"], 4),
        ])
    base = atr["universo_sin_entrenar_pct"]
    out.append({
        "clave": "atribucion",
        "titulo": "Por que una senal es de una sola",
        "nota": ("Las dos versiones comparten todo el score menos los puntos ML: la diferencia "
                 "de score ES la diferencia de puntos ML. 'Sin entrenar v1' = tickers que la v1 "
                 "no vio al entrenar; leerlo contra la tasa base de las filas comparables, "
                 f"{'-' if base is None else f'{base:.0f}%'}. 'Nivel en la otra': COMPRA es un "
                 "desacuerdo de borde; NEUTRAL o menos, de fondo."),
        "columnas": ["Grupo", "Senales", "Sin entrenar v1", "Nivel en la otra version",
                     "Prob v1", "Prob v2", "Score v2 - v1", "Sectores"],
        "filas": filas,
    })

    ops = res.get("operaciones")
    if ops is not None:
        filas = []
        for v in ("v1", "v2"):
            o = ops[v]
            filas.append([
                v, str(o["n"]), str(o["abiertas"]), str(o["cerradas"]),
                str(o["compartidas"]), str(o["exclusivas"]),
                "-" if o["trade"].get("win_rate") is None else f"{o['trade']['win_rate']:.0f}%",
                _expectancy(o["trade"]), _expectancy(o["trade_compartidas"]),
                _expectancy(o["trade_exclusivas"]),
                "-" if o["ruedas_media"] is None else f"{o['ruedas_media']:.1f}",
                _pares(o["motivos"]),
            ])
        contraf = ops["v1"]["contrafactuales"] + ops["v2"]["contrafactuales"]
        out.append({
            "clave": "operaciones",
            "titulo": "Operaciones",
            "nota": ("Compartida = la otra version tuvo el mismo ticker abierto algun dia en comun. "
                     "Expectancy = pnl % medio por operacion cerrada. Expectancy v2 - v1: "
                     f"{texto_diferencia(ops['expectancy_v2_vs_v1'])}."
                     + (f" {contraf} operaciones _SPLIT_FIX fuera de las metricas." if contraf else "")),
            "columnas": ["Version", "Oper.", "Abiertas", "Cerradas", "Compartidas", "Exclusivas",
                         "Win rate", "Expectancy %", "Exp. compartidas", "Exp. exclusivas",
                         "Ruedas media", "Motivos de salida"],
            "filas": filas,
        })

    opo = res.get("oportunidades")
    if opo is not None:
        filas = []
        for v in ("v1", "v2"):
            for n in hs:
                h = opo[v]["horizontes"][n]
                filas.append([v, f"{n}r", texto_media(h["adentro"]), texto_media(h["afuera"]),
                              texto_diferencia(h["afuera_vs_adentro"], ("adentro", "afuera"))])
        out.append({
            "clave": "oportunidades",
            "titulo": "Oportunidades: lo que dejo afuera el tope de posiciones",
            "nota": ("Candidatos que cumplian la regla de entrada. Exceso contra el universo, en %. "
                     "Si afuera rinde mas que adentro, el orden por score no elige bien dentro del "
                     "dia; con mas senales por dia (la v2) el tope pesa mas."),
            "columnas": ["Version", "Horizonte", "Entraron", "Quedaron afuera", "Afuera - adentro"],
            "filas": filas,
        })

    car = res.get("cartera")
    if car is not None:
        out.append({
            "clave": "cartera",
            "titulo": "Cartera",
            "nota": ("Diferencia diaria de retorno v2 - v1 sobre ft_equity_diaria en los dias que "
                     "tienen las dos, en puntos por mes con IC95. Muestra pareada: el mismo "
                     f"mercado cada dia. Minimo {MIN_RUEDAS_CARTERA} ruedas."),
            "columnas": ["Ruedas", "Desde", "Hasta", "Retorno v1", "Retorno v2",
                         "v2 - v1 (pp/mes)"],
            "filas": [[str(car["n"]), str(car["desde"] or "-"), str(car["hasta"] or "-"),
                       "-" if car["retorno_v1_pct"] is None else f"{car['retorno_v1_pct']:+.2f}%",
                       "-" if car["retorno_v2_pct"] is None else f"{car['retorno_v2_pct']:+.2f}%",
                       texto_media(car, dec=1)]],
        })
    return out
