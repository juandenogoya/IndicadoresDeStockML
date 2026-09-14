"""
ft_comparar_ml.py
Por que difieren FT_ML_SCANNER_v1 y FT_ML_SCANNER_v2 (Etapa 3f).

Lee la DB LOCAL, calcula con src/utils/ft_comparar.py y escribe
reportes/ft_comparar_ml.md. Solo lee. La misma comparacion sale en el reporte HTML
diario (seccion "ML v1 vs v2: por que difieren"), que usa cargar_insumos() de aca.

Los retornos de las senales salen de precios_diarios por precio_fecha: las columnas
retorno_Nd_real de alertas_scanner no se llenan desde mayo (el Paso 4 no esta en
la rutina).

Uso:
    python scripts/forward_testing/ft_comparar_ml.py
    python scripts/forward_testing/ft_comparar_ml.py --desde 2026-09-14
    python scripts/forward_testing/ft_comparar_ml.py --output ruta/archivo.md

Diseno: docs/forward_testing/METRICAS.md, seccion 13.
"""

import argparse
import os
import sys
from datetime import date, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Entorno FT: forzar conexion a la DB LOCAL (ver scripts/forward_testing/ft_env.py)
from scripts.forward_testing.ft_env import configurar_entorno_local
configurar_entorno_local()

from sqlalchemy import text
from src.data.database import get_engine
from src.utils import ft_comparar
from src.utils.trading_calendar import is_trading_day

NOMBRE_V1 = "FT_ML_SCANNER_v1"
NOMBRE_V2 = "FT_ML_SCANNER_v2"
OUTPUT_DEFAULT = os.path.join(ROOT, "reportes", "ft_comparar_ml.md")

# Tickers del entrenamiento de la v1. features_ml tenia 123: los que llegaban a dos
# anios de historia (docs/ml_reentrenamiento.md). La Etapa 3b la reconstruyo con 196,
# asi que el conjunto se reproduce por la historia de precios: MIN(fecha) hasta fin de
# 2021 da exactamente 123. activos.modelo_asignado NO sirve: da 125 (HOOD y LAC tienen
# modelo asignado y no tienen esa historia).
HISTORIA_V1_HASTA = date(2021, 12, 31)

LEYENDA = (
    "Entre corchetes, el IC95. INSUFICIENTE: la muestra no llega al minimo y el numero "
    "no se muestra. NO CONCLUYENTE: el intervalo incluye el cero. Las senales de una "
    "misma rueda comparten mercado: por eso se mide el exceso contra el universo y se "
    "exigen ruedas distintas, pero el IC las sigue tratando como independientes y queda "
    "algo angosto. Operaciones y oportunidades cuentan solo lo que entro desde la rueda "
    "de inicio: las posiciones que la v1 ya tenia abiertas quedan afuera, aunque si "
    "estan en su equity. Diseno: docs/forward_testing/METRICAS.md sec. 13."
)


def _t(v):
    return None if v is None else str(v).strip()


def _f(v):
    return None if v is None else float(v)


def cargar_insumos(desde=None):
    """
    Lee lo que necesita ft_comparar.comparar(). None si falta alguna de las dos
    estrategias. `desde` (date) = primera rueda de datos; None = inicio de la v2.
    """
    engine = get_engine()
    with engine.connect() as conn:
        ests = {r.nombre: dict(r._mapping) for r in conn.execute(text("""
            SELECT id, nombre, fecha_inicio FROM ft_estrategias WHERE nombre IN (:v1, :v2)
        """), {"v1": NOMBRE_V1, "v2": NOMBRE_V2})}
        if NOMBRE_V1 not in ests or NOMBRE_V2 not in ests:
            return None
        id_v1, id_v2 = ests[NOMBRE_V1]["id"], ests[NOMBRE_V2]["id"]
        if desde is None:
            desde = ests[NOMBRE_V2]["fecha_inicio"]
        p = {"desde": desde, "v1": id_v1, "v2": id_v2}

        filas = [{
            "ticker": _t(r.ticker), "sector": _t(r.sector), "scan_fecha": r.scan_fecha,
            "precio_fecha": r.precio_fecha,
            "nivel_v1": _t(r.nivel_v1), "score_v1": _f(r.score_v1), "prob_v1": _f(r.prob_v1),
            "nivel_v2": _t(r.nivel_v2), "score_v2": _f(r.score_v2), "prob_v2": _f(r.prob_v2),
        } for r in conn.execute(text("""
            SELECT ticker, sector, scan_fecha, precio_fecha,
                   alert_nivel AS nivel_v1, alert_score AS score_v1,
                   ml_prob_ganancia AS prob_v1,
                   alert_nivel_v2 AS nivel_v2, alert_score_v2 AS score_v2,
                   ml_prob_v2 AS prob_v2
            FROM alertas_scanner
            WHERE precio_fecha >= :desde
        """), p)]

        closes = {}
        for r in conn.execute(text("""
            SELECT ticker, fecha, close FROM precios_diarios
            WHERE fecha >= :desde AND close IS NOT NULL
        """), p):
            closes.setdefault(_t(r.ticker), {})[r.fecha] = float(r.close)
        # Ruedas del MERCADO: las ventanas se cuentan sobre estas, no sobre la serie
        # de cada ticker. Solo dias habiles NYSE (regla 1 del proyecto).
        ruedas = sorted({f for serie in closes.values() for f in serie if is_trading_day(f)})

        entrenados = {_t(r.ticker) for r in conn.execute(text("""
            SELECT ticker FROM precios_diarios GROUP BY ticker HAVING MIN(fecha) <= :hasta
        """), {"hasta": HISTORIA_V1_HASTA})}

        # Fechas del DATO, no de registro (CLAUDE.md, FT asincronico).
        operaciones = {"v1": [], "v2": []}
        for r in conn.execute(text("""
            SELECT estrategia_id, ticker,
                   COALESCE(fecha_datos, fecha_entrada) AS f_entrada,
                   CASE WHEN fecha_salida IS NULL THEN NULL
                        ELSE COALESCE(fecha_datos_salida, fecha_salida) END AS f_salida,
                   pnl, pnl_pct, motivo_salida
            FROM ft_operaciones
            WHERE estrategia_id IN (:v1, :v2)
              AND COALESCE(fecha_datos, fecha_entrada) >= :desde
            ORDER BY id
        """), p):
            operaciones["v1" if r.estrategia_id == id_v1 else "v2"].append({
                "ticker": _t(r.ticker), "f_entrada": r.f_entrada, "f_salida": r.f_salida,
                "pnl": _f(r.pnl), "pnl_pct": _f(r.pnl_pct), "motivo_salida": r.motivo_salida,
            })

        # ft_candidatos_diarios guarda el dia de REGISTRO. La rueda de datos es la de
        # la corrida del scanner que leyo el bot: la ultima alerta del ticker hasta ese
        # dia. Sus retorno_Nd propios no sirven aca: cuentan dias desde el registro y
        # solo se llenan si el bot corre justo N ruedas despues.
        candidatos = [{
            "version": "v1" if r.estrategia_id == id_v1 else "v2",
            "ticker": _t(r.ticker), "precio_fecha": r.precio_fecha, "entro": bool(r.entro),
        } for r in conn.execute(text("""
            SELECT c.estrategia_id, c.ticker, c.entro,
                   (SELECT MAX(a.precio_fecha) FROM alertas_scanner a
                    WHERE a.ticker = c.ticker AND a.scan_fecha < c.fecha + 1) AS precio_fecha
            FROM ft_candidatos_diarios c
            WHERE c.estrategia_id IN (:v1, :v2) AND c.fecha >= :desde
        """), p)]

        equity = {"v1": [], "v2": []}
        for r in conn.execute(text("""
            SELECT estrategia_id, fecha, equity FROM ft_equity_diaria
            WHERE estrategia_id IN (:v1, :v2) AND fecha >= :desde
            ORDER BY fecha
        """), p):
            equity["v1" if r.estrategia_id == id_v1 else "v2"].append((r.fecha, float(r.equity)))

    return {"desde": desde, "id_v1": id_v1, "id_v2": id_v2, "filas": filas,
            "closes": closes, "ruedas": ruedas,
            "entrenados_v1": entrenados, "operaciones": operaciones,
            "candidatos": candidatos, "equity": equity,
            "hoy": ruedas[-1] if ruedas else None}


def calcular(insumos):
    return ft_comparar.comparar(
        insumos["filas"], insumos["closes"], insumos["ruedas"], insumos["entrenados_v1"],
        operaciones=insumos["operaciones"], candidatos=insumos["candidatos"],
        equity=insumos["equity"], hoy=insumos["hoy"])


def render_md(res, insumos):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M")
    rango = f", ruedas {res['desde']} a {res['hasta']}" if res["desde"] else ""
    lineas = [
        "# ML_SCANNER v1 vs v2: por que difieren", "",
        f"Generado: {ahora} | Desde la rueda {insumos['desde']} | Filas con v1 y v2: "
        f"{res['filas']}{rango} | Filas sin v2: {res['filas_sin_v2']} | Tickers "
        f"entrenados en la v1: {len(insumos['entrenados_v1'])}", "",
    ]
    if res["filas"] == 0:
        lineas += ["> Todavia no hay filas de alertas_scanner con la v2 desde esa rueda.", ""]
    for i, t in enumerate(ft_comparar.tablas(res), 1):
        lineas += [f"## {i}. {t['titulo']}", "", t["nota"], ""]
        if t["filas"]:
            lineas.append("| " + " | ".join(t["columnas"]) + " |")
            lineas.append("|" + "---|" * len(t["columnas"]))
            lineas += ["| " + " | ".join(f) + " |" for f in t["filas"]]
        else:
            lineas.append("_Sin filas._")
        lineas.append("")
    lineas += ["## Como leer", "", LEYENDA, ""]
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser(
        description="Por que difieren FT_ML_SCANNER_v1 y v2 (solo lee la DB local)")
    ap.add_argument("--desde", type=date.fromisoformat, default=None,
                    help="Primera rueda de datos YYYY-MM-DD (default: inicio de la v2)")
    ap.add_argument("--output", default=OUTPUT_DEFAULT,
                    help=f"Ruta del .md (default: {OUTPUT_DEFAULT})")
    args = ap.parse_args()

    insumos = cargar_insumos(args.desde)
    if insumos is None:
        print(f"[ft_comparar_ml] [ERROR] Falta {NOMBRE_V1} o {NOMBRE_V2} en ft_estrategias.")
        return 1
    res = calcular(insumos)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(render_md(res, insumos))

    t = res["solapamiento"]["total"]
    print(f"[ft_comparar_ml] Desde {insumos['desde']}: {res['filas']} filas con v1 y v2 en "
          f"{t['n_ruedas']} ruedas | senales v1 {t['v1']} / v2 {t['v2']} / ambas {t['ambas']} "
          f"| filas sin v2 {res['filas_sin_v2']}")
    print(f"[ft_comparar_ml] Reporte: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
