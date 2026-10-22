"""
ft_analisis_fill.py -- Eran ALCANZABLES los precios con los que FT registro sus
operaciones?

SOLO LEE. No escribe en la DB, no toca ninguna estrategia, no va a ft_cambios.
Pre-registro, regla de lectura y resultados: docs/forward_testing/ANALISIS_FILL.md.

FT registra entrada y salida al close de `fecha_datos`, la ultima rueda que
precios_diarios tenia cuando corrio el bot. Ese close YA HABIA OCURRIDO al decidir.
Este script re-valua cada operacion al primer precio obtenible y mide la diferencia
PAREADA contra el fill registrado, con la RUEDA como unidad de independencia.

Las dos patas (entrada y salida) se mueven JUNTAS: re-valuar solo una mide un
artefacto que no existe. La logica vive en src/utils/ft_fill.py (puro, con tests).

Uso:
    python scripts/forward_testing/ft_analisis_fill.py
    python scripts/forward_testing/ft_analisis_fill.py --etiqueta prueba --sin-archivo

Codigos de salida: 0 la convencion queda confirmada (el fill NO es material) /
2 el fill ES material segun la regla pre-registrada / 1 error.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# LOCAL es la fuente de verdad: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

import pandas as pd  # noqa: E402

from src.data.database import query_df  # noqa: E402
from src.utils import ft_fill as ff  # noqa: E402

CAMBIOS_QUE_INVALIDAN = ("fix_score_cero_salida",)
DIR_BASE = os.path.join(ROOT, "reportes", "analisis_fill")

# La apertura de NYSE son las 09:30 ET. El reloj de `creado_en` es la hora LOCAL de
# la maquina (Argentina, UTC-3) y todo el periodo analizado (abr-oct 2026) cae en
# horario de verano de EE.UU. (ET = UTC-4), asi que 09:30 ET = 10:30 local.
HORA_APERTURA_LOCAL = time(10, 30)


class Salida:
    """print + archivo, como el resto de los analisis de FT."""

    def __init__(self, ruta=None):
        self.f = open(ruta, "w", encoding="utf-8") if ruta else None

    def __call__(self, txt=""):
        print(txt)
        if self.f:
            self.f.write(txt + "\n")

    def cerrar(self):
        if self.f:
            self.f.close()


def ventanas_invalidas():
    """{estrategia_id: fecha_efectiva} de los cambios que invalidan la historia."""
    c = query_df("SELECT clave, fecha_efectiva, estrategias FROM ft_cambios "
                 "WHERE clave = ANY(:k)", params={"k": list(CAMBIOS_QUE_INVALIDAN)})
    out = {}
    for r in c.itertuples():
        for e in r.estrategias:
            f = pd.Timestamp(r.fecha_efectiva)
            out[int(e)] = max(out.get(int(e), f), f)
    return out


def cargar():
    """Operaciones cerradas + apertura y close de la primera rueda posterior a
    cada pata. El LATERAL hace el 'primera rueda siguiente' en la DB."""
    return query_df("""
        SELECT o.id, o.estrategia_id, e.nombre, o.ticker, o.motivo_salida,
               o.fecha_datos, o.fecha_datos_salida, o.creado_en,
               o.precio_entrada, o.precio_salida, o.pnl_pct,
               ent.fecha AS ent_fecha_sig, ent.open AS ent_open_sig,
               ent.close AS ent_close_sig,
               sal.fecha AS sal_fecha_sig, sal.open AS sal_open_sig,
               sal.close AS sal_close_sig
        FROM ft_operaciones o
        JOIN ft_estrategias e ON e.id = o.estrategia_id
        LEFT JOIN LATERAL (
            SELECT p.fecha, p.open, p.close FROM precios_diarios p
            WHERE p.ticker = o.ticker AND p.fecha > o.fecha_datos
            ORDER BY p.fecha LIMIT 1
        ) ent ON TRUE
        LEFT JOIN LATERAL (
            SELECT p.fecha, p.open, p.close FROM precios_diarios p
            WHERE p.ticker = o.ticker AND p.fecha > o.fecha_datos_salida
            ORDER BY p.fecha LIMIT 1
        ) sal ON TRUE
        WHERE o.fecha_salida IS NOT NULL
    """)


def preparar(op, log):
    """Filtra con las exclusiones pre-registradas y arma los fills por operacion."""
    log(f"Operaciones cerradas en la DB: {len(op)}")

    excl = op["motivo_salida"].isin(ff.MOTIVOS_EXCLUIDOS)
    if excl.any():
        for m, n in op.loc[excl, "motivo_salida"].value_counts().items():
            log(f"  - excluidas por motivo {m}: {n}")
        op = op[~excl].copy()

    inval = ventanas_invalidas()
    if inval:
        corte = op["estrategia_id"].map(inval)
        fuera = corte.notna() & (pd.to_datetime(op["fecha_datos_salida"]) < corte)
        if fuera.any():
            log(f"  - excluidas por ventana invalida de ft_cambios "
                f"({', '.join(CAMBIOS_QUE_INVALIDAN)}): {int(fuera.sum())}")
            op = op[~fuera].copy()

    for c in ("precio_entrada", "precio_salida", "ent_open_sig", "ent_close_sig",
              "sal_open_sig", "sal_close_sig"):
        op[c] = pd.to_numeric(op[c], errors="coerce")

    filas = []
    sin_pata = 0
    for r in op.itertuples():
        fills = {
            "entrada": {"actual": r.precio_entrada,
                        "apertura_sig": r.ent_open_sig,
                        "cierre_sig": r.ent_close_sig},
            "salida": {"actual": r.precio_salida,
                       "apertura_sig": r.sal_open_sig,
                       "cierre_sig": r.sal_close_sig},
        }
        if ff.revaluar(fills, ff.CONTROL) is None:
            sin_pata += 1
            continue
        filas.append({
            "id": r.id, "estrategia": r.nombre, "ticker": r.ticker,
            "rueda_salida": pd.Timestamp(r.fecha_datos_salida).date(),
            "mes": pd.Timestamp(r.fecha_datos_salida).strftime("%Y-%m"),
            "creado_en": r.creado_en,
            "ent_fecha_sig": r.ent_fecha_sig, "sal_fecha_sig": r.sal_fecha_sig,
            "fills": fills,
        })
    if sin_pata:
        log(f"  - excluidas sin precio en alguna pata del control: {sin_pata}")
    log(f"Operaciones que entran al analisis: {len(filas)}")
    return filas


def diagnostico_supuesto(filas, log):
    """`apertura_sig` supone que el bot actuo ANTES de esa apertura. Cuantas
    operaciones cumplen el supuesto de verdad?"""
    cumple = falla = sin_dato = 0
    for f in filas:
        ce, fs_ = f["creado_en"], f["sal_fecha_sig"]
        if ce is None or pd.isna(ce) or fs_ is None or pd.isna(fs_):
            sin_dato += 1
            continue
        apertura = pd.Timestamp(pd.Timestamp(fs_).date()).replace(
            hour=HORA_APERTURA_LOCAL.hour, minute=HORA_APERTURA_LOCAL.minute)
        if pd.Timestamp(ce) < apertura:
            cumple += 1
        else:
            falla += 1
    n = len(filas)
    log("")
    log("SUPUESTO DE `apertura_sig` (el bot actuo antes de esa apertura)")
    log(f"  lo cumple          : {cumple} ({cumple / n:.1%})")
    log(f"  NO lo cumple       : {falla} ({falla / n:.1%})  -> para estas, la cota")
    log( "                       optimista subestima el efecto real")
    if sin_dato:
        log(f"  sin dato de hora   : {sin_dato}")
    return {"cumple": cumple, "no_cumple": falla, "sin_dato": sin_dato}


def totales_por_estrategia(filas, variante):
    """Suma de retornos por estrategia (en pp). None si alguna op no se puede."""
    acum = {}
    for f in filas:
        v = ff.revaluar(f["fills"], variante)
        if v is None:
            continue
        acum.setdefault(f["estrategia"], 0.0)
        acum[f["estrategia"]] += v
    return acum


def analizar(filas, variante, log):
    comp = [(f, ff.comparar_operacion(f["fills"], variante)) for f in filas]
    usables = [(f, c) for f, c in comp if c is not None]
    perdidas = len(comp) - len(usables)

    difs = [c["dif"] for _, c in usables]
    ruedas = [f["rueda_salida"] for f, _ in usables]
    meses = [f["mes"] for f, _ in usables]
    r_act = [c["actual"] for _, c in usables]
    r_var = [c["variante"] for _, c in usables]

    res = ff.resumen_variante(difs, ruedas, r_act, r_var)
    por_mes = ff.signo_por_mes(difs, meses)
    tot_a = totales_por_estrategia([f for f, _ in usables], ff.CONTROL)
    tot_v = totales_por_estrategia([f for f, _ in usables], variante)
    vuelta = ff.da_vuelta_el_signo(tot_a, tot_v)
    veredicto = ff.leer(res, por_mes, vuelta)

    log("")
    log("=" * 74)
    log(f"  VARIANTE: {variante}" + ("   (PRINCIPAL)" if variante == ff.PRINCIPAL else ""))
    log("=" * 74)
    if perdidas:
        log(f"  operaciones sin precio en esta variante (no entran): {perdidas}")
    log(f"  operaciones {res['ops']}  |  ruedas {res['ruedas']}")
    log("")
    log(f"  retorno medio por operacion   control  {res['ret_actual']:+.3f} %")
    log(f"                                variante {res['ret_variante']:+.3f} %")
    log(f"  DIFERENCIA PAREADA (por rueda)         {res['dif_media']:+.3f} pp"
        f"   IC95 [{res['ic95_lo']:+.3f}; {res['ic95_hi']:+.3f}]")
    log(f"                      (por operacion)    {res['dif_media_ops']:+.3f} pp")
    log("")
    log(f"  operaciones ganadoras    control {res['ganadoras_actual']:.1f} %"
        f"   -> variante {res['ganadoras_variante']:.1f} %")
    log(f"  cola (percentil 5)       control {res['p5_actual']:+.2f} %"
        f"   -> variante {res['p5_variante']:+.2f} %")

    log("")
    log("  diferencia media por mes:")
    for m, v in por_mes.items():
        log(f"    {m}   {v:+.3f} pp")

    log("")
    log("  retorno TOTAL por estrategia (suma de retornos, pp):")
    for k in sorted(tot_a, key=lambda x: -tot_a[x]):
        marca = "  <-- CAMBIA DE SIGNO" if k in vuelta else ""
        log(f"    {k:28s} {tot_a[k]:+9.2f}  ->  {tot_v.get(k, float('nan')):+9.2f}{marca}")

    log("")
    log("  REGLA DE LECTURA pre-registrada (las tres):")
    log(f"    1. IC95 excluye el cero          : {'SI' if veredicto['ic95_excluye_cero'] else 'NO'}")
    log(f"    2. material (|dif| >= {ff.FRACCION_MATERIAL:.0%} del control = "
        f"{veredicto['umbral_material']:.3f} pp, o da vuelta un signo): "
        f"{'SI' if veredicto['material'] else 'NO'}")
    log(f"       por magnitud {veredicto['material_por_magnitud']} | "
        f"por signo {veredicto['material_por_signo']}"
        + (f" ({', '.join(vuelta)})" if vuelta else ""))
    log(f"    3. mismo signo en >= {ff.MESES_MINIMOS} meses     : "
        f"{'SI' if veredicto['consistente'] else 'NO'} "
        f"({veredicto['meses_al_mismo_lado']} de {veredicto['meses_total']})")
    log("")
    log(f"    >>> {'EL FILL ES UN ARTEFACTO MATERIAL' if veredicto['pasa'] else 'NO PASA: la convencion queda CONFIRMADA por medicion'}")

    return {"resumen": res, "por_mes": por_mes, "veredicto": veredicto,
            "totales_control": tot_a, "totales_variante": tot_v,
            "signos_dados_vuelta": vuelta, "sin_precio": perdidas}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etiqueta", default="fill")
    ap.add_argument("--sin-archivo", action="store_true",
                    help="solo imprime, no escribe en reportes/")
    args = ap.parse_args()

    destino = None
    if not args.sin_archivo:
        destino = os.path.join(DIR_BASE, f"{date.today():%Y%m%d}_{args.etiqueta}")
        os.makedirs(destino, exist_ok=True)

    log = Salida(os.path.join(destino, "resumen.txt") if destino else None)
    try:
        log("=" * 74)
        log("  REALISMO DEL FILL DE FT -- eran alcanzables esos precios?")
        log(f"  corrida {date.today()}  |  commit {git_commit()}")
        log("  Pre-registro: docs/forward_testing/ANALISIS_FILL.md sec. 2")
        log("=" * 74)
        log("")

        filas = preparar(cargar(), log)
        if not filas:
            log("No quedaron operaciones. Nada que medir.")
            return 1

        sup = diagnostico_supuesto(filas, log)

        salidas = {}
        for v in ff.VARIANTES:
            if v == ff.CONTROL:
                continue
            salidas[v] = analizar(filas, v, log)

        pasa = salidas[ff.PRINCIPAL]["veredicto"]["pasa"]
        log("")
        log("=" * 74)
        log(f"  VEREDICTO (variante principal {ff.PRINCIPAL}): "
            f"{'EL FILL IMPORTA' if pasa else 'LA CONVENCION QUEDA CONFIRMADA'}")
        log("=" * 74)

        if destino:
            with open(os.path.join(destino, "parametros.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({
                    "corrida": str(date.today()), "commit": git_commit(),
                    "operaciones": len(filas),
                    "variantes": list(ff.VARIANTES), "principal": ff.PRINCIPAL,
                    "fraccion_material": ff.FRACCION_MATERIAL,
                    "meses_minimos": ff.MESES_MINIMOS,
                    "motivos_excluidos": list(ff.MOTIVOS_EXCLUIDOS),
                    "cambios_que_invalidan": list(CAMBIOS_QUE_INVALIDAN),
                    "hora_apertura_local": str(HORA_APERTURA_LOCAL),
                    "supuesto_apertura": sup,
                    "resultados": {k: {"resumen": v["resumen"],
                                       "por_mes": v["por_mes"],
                                       "veredicto": v["veredicto"],
                                       "signos_dados_vuelta": v["signos_dados_vuelta"]}
                                   for k, v in salidas.items()},
                }, fh, indent=2, default=str)
            log("")
            log(f"Salida en {destino}")

        return 2 if pasa else 0
    finally:
        log.cerrar()


if __name__ == "__main__":
    sys.exit(main())
