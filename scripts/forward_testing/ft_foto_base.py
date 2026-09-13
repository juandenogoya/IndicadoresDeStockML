"""
ft_foto_base.py
Foto de BASE congelada del Forward Testing: como estaba cada estrategia en una
rueda dada, con su incertidumbre, para comparar contra ella despues de un cambio.

Por que un archivo y no alcanza con el reporte: el reporte se regenera todos los
dias y la historia se RECOMPUTA (ft_equity_diaria es una capa derivada). Si se
corrige un dato o un bug, el reporte muestra la historia ya corregida y deja de
existir la version que se veia antes. La foto conserva lo que se sabia ese dia.

Salida (nunca pisa una foto existente salvo --forzar):
    reportes/ft_foto_base_<rueda>.json   completo, para comparar por codigo
    reportes/ft_foto_base_<rueda>.md     legible
<rueda> = ultima fecha de ft_equity_diaria (la fecha del DATO, no la de corrida).

Por estrategia, en la historia completa y en la ventana comparable:
    retorno, benchmark de la misma ventana, max DD, volatilidad,
    Sortino con IC95 por bootstrap, Sharpe con IC95 (Lo 2002),
    metricas por operacion y tramo vigente segun ft_cambios.

Uso:
    python scripts/forward_testing/ft_foto_base.py
    python scripts/forward_testing/ft_foto_base.py --desde 2026-05-30 --forzar
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Importar el reporte configura el entorno local (ft_env) y trae las mismas
# lecturas y reglas de ventana: la foto y el reporte no pueden medir distinto.
from scripts.forward_testing.ft_reporte_html import (  # noqa: E402
    ROOT, VENTANA_COMPARABLE, cargar_datos, cargar_cambios, serie_benchmark,
    calcular_riesgo, series_tramos, operaciones_tramos,
)
import pandas as pd  # noqa: E402
from src.utils.ft_metricas import (  # noqa: E402
    metricas_trade, retornos_desde_equity, sortino, RF_ANUAL_DEFAULT,
)
from src.utils import ft_tramos  # noqa: E402

DIR_SALIDA = os.path.join(ROOT, "reportes")


def _sortino_valor(retornos):
    s = sortino(retornos)
    return s["sortino"] if s else None


def ventana(df_met_e, bench, cerradas, desde=None):
    """Riesgo + Sortino con IC95 + metricas de operacion de una ventana."""
    riesgo = calcular_riesgo(df_met_e, bench, desde=desde)
    sub = df_met_e.sort_values("fecha")
    if desde is not None:
        sub = sub[sub["fecha"] >= pd.Timestamp(desde)]
    r = retornos_desde_equity([float(v) for v in sub["equity"]])
    ic = ft_tramos.ic95_bootstrap(r, _sortino_valor) if len(r) >= 2 else None

    ops = cerradas
    if desde is not None and not ops.empty:
        ops = ops[ops["f_salida_datos"] >= pd.Timestamp(desde)]
    trade = metricas_trade(ops.to_dict("records")) if not ops.empty else {"n": 0}

    if riesgo:
        riesgo = dict(riesgo)
        riesgo["sortino_ic95"] = list(ic) if ic else None
        riesgo["desde"] = str(sub["fecha"].iloc[0].date()) if len(sub) else None
    return {"riesgo": riesgo, "trade": trade}


def construir(desde):
    ests, df_ops, df_met, df_bch = cargar_datos()
    cambios = cargar_cambios() or []
    bench = serie_benchmark(df_bch)
    series = series_tramos(df_met)
    ops_t = operaciones_tramos(df_ops)
    rueda = df_met["fecha"].max().date()

    filas = []
    for e in ests:
        eid = e["id"]
        df_met_e = df_met[df_met["estrategia_id"] == eid]
        if df_met_e.empty:
            continue
        df_e = df_ops[df_ops["estrategia_id"] == eid] if not df_ops.empty else df_ops
        cerradas = df_e[df_e["fecha_salida"].notna()] if not df_e.empty else df_e
        filas.append({
            "id": eid,
            "nombre": e["nombre"].replace("FT_", ""),
            "fecha_inicio": str(e["fecha_inicio"]),
            "completa": ventana(df_met_e, bench, cerradas),
            "comparable": ventana(df_met_e, bench, cerradas, desde),
            "tramo_vigente": ft_tramos.tramo_vigente(eid, cambios, series.get(eid), ops_t),
        })

    def ret_bench(desde_b=None):
        if bench is None:
            return None
        b = bench if desde_b is None else bench[bench.index >= pd.Timestamp(desde_b)]
        return (b.iloc[-1] / b.iloc[0] - 1) * 100 if len(b) > 1 else None

    return {
        "rueda": str(rueda),
        "generado_en": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ventana_comparable": desde,
        "rf_anual": RF_ANUAL_DEFAULT,
        "minimos_tramo": {"ruedas": ft_tramos.MIN_RUEDAS_TRAMO, "ops": ft_tramos.MIN_OPS_TRAMO},
        "benchmark_pct": {"completa": ret_bench(), "comparable": ret_bench(desde)},
        "estrategias": filas,
        "cambios": cambios,
    }


def _f(v, fmt="{:+.2f}"):
    return "-" if v is None else fmt.format(v)


def markdown(foto):
    out = [f"# Foto de base FT -- rueda {foto['rueda']}", "",
           f"Generada {foto['generado_en']}. Tasa libre de riesgo {foto['rf_anual'] * 100:.1f}%. "
           f"Sortino con IC95 por bootstrap; Sharpe con IC95 de Lo (2002). Un IC que incluye "
           f"el cero es NO CONCLUYENTE.", ""]
    bp = foto["benchmark_pct"]
    for clave, titulo, b in [("comparable", f"Ventana comparable (desde {foto['ventana_comparable']})",
                              bp["comparable"]),
                             ("completa", "Historia completa", bp["completa"])]:
        out += [f"## {titulo}", "", f"Universo equiponderado: {_f(b)}%", "",
                "| Estrategia | Dias | Retorno | Bench | Max DD | Sortino [IC95] | Sharpe [IC95] "
                "| Ops | Win | PF | Expectancy |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for f in foto["estrategias"]:
            r, t = f[clave]["riesgo"], f[clave]["trade"]
            if not r or r.get("insuficiente"):
                out.append(f"| {f['nombre']} | serie insuficiente |||||||||| ")
                continue
            ic = r.get("sortino_ic95")
            so = f"{_f(r['sortino'])} [{_f(ic[0])}, {_f(ic[1])}]" if ic else _f(r["sortino"])
            sh = r.get("sharpe")
            sh_txt = (f"{_f(sh['sharpe'])} [{_f(sh['ic95_lo'])}, {_f(sh['ic95_hi'])}]"
                      if sh else "-")
            n = t.get("n", 0)
            out.append(
                f"| {f['nombre']} | {r['n_dias']} | {_f(r['retorno_total_pct'])}% "
                f"| {_f(r.get('benchmark_retorno_pct'))}% | -{_f(r['max_dd_pct'], '{:.2f}')}% "
                f"| {so} | {sh_txt} | {n} "
                f"| {_f(t.get('win_rate'), '{:.1f}') if n else '-'}% "
                f"| {_f(t.get('profit_factor'), '{:.2f}') if n else '-'} "
                f"| {_f(t.get('expectancy_pct')) if n else '-'}% |")
        out.append("")

    out += ["## Tramo vigente por estrategia", "",
            f"Minimo para comparar un tramo: {foto['minimos_tramo']['ruedas']} ruedas y "
            f"{foto['minimos_tramo']['ops']} operaciones cerradas.", "",
            "| Estrategia | Desde | Ruedas | Ops cerradas |", "|---|---|---|---|"]
    for f in foto["estrategias"]:
        v = f["tramo_vigente"]
        if v:
            out.append(f"| {f['nombre']} | {v['desde']} | {v['n_ruedas']} | {v['n_ops']} |")
    out += ["", "## Cambios registrados (ft_cambios)", "",
            "| Fecha efectiva | Corta | Tipo | Clave | Estrategias |", "|---|---|---|---|---|"]
    for c in foto["cambios"]:
        out.append(f"| {c['fecha_efectiva']} | {'SI' if c['cambia_decisiones'] else '-'} "
                   f"| {c['tipo']} | {c['clave']} | {c['estrategias']} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Foto de base congelada del Forward Testing")
    ap.add_argument("--desde", default=VENTANA_COMPARABLE,
                    help=f"inicio de la ventana comparable (default {VENTANA_COMPARABLE})")
    ap.add_argument("--forzar", action="store_true", help="pisar una foto existente de esa rueda")
    args = ap.parse_args()

    foto = construir(args.desde)
    base = os.path.join(DIR_SALIDA, f"ft_foto_base_{foto['rueda']}")
    if os.path.exists(base + ".json") and not args.forzar:
        print(f"[ft_foto_base] Ya existe {base}.json. Una foto de base no se pisa: "
              f"--forzar si es a proposito.")
        return 1

    os.makedirs(DIR_SALIDA, exist_ok=True)
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(foto, fh, ensure_ascii=True, indent=2, default=str)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(markdown(foto))
    print(f"[ft_foto_base] {len(foto['estrategias'])} estrategias, rueda {foto['rueda']}, "
          f"{len(foto['cambios'])} cambios registrados.")
    print(f"[ft_foto_base] {base}.json")
    print(f"[ft_foto_base] {base}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
