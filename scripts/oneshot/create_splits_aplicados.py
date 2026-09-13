"""
create_splits_aplicados.py
Crea `splits_aplicados` y la carga con los splits que YA estan reflejados en
precios_diarios.

POR QUE (12/9/2026):
    El factor de escala del precio de referencia de opciones leia polygon_splits,
    congelado desde el 30/8 (linea SEC/Polygon dormida): el proximo split no
    habria entrado. Y una lista de splits "de afuera" tiene otro riesgo: si el
    split esta listado pero precios_diarios todavia no se corrigio, el factor se
    aplicaria sobre una historia que sigue en la escala vieja (doble escala).
    `splits_aplicados` solo tiene splits YA reflejados, y desde ahora lo escribe
    scripts/manual/splits.py corregir en la misma transaccion que la correccion.
    El DDL vive ahi (DDL_SPLITS_APLICADOS) para no duplicarlo.

CARGA INICIAL:
    Fuente: eventos de split de Yahoo (yahooquery history, columna `splits`) para
    el universo activo, dentro de la historia de precios_diarios de cada ticker.
    Cada evento se VALIDA antes de registrarlo:
      - es un split de PRECIO: ratio real (>= 1,5 o <= 1/1,5) y ademas EXACTO
        (2, 3, 4, 5, 10, 20... o su inverso, precio_referencia.ratio_de_split).
        Yahoo tambien lista como "split" ajustes de precio por spinoff: DELL 1,973
        (VMware, 2021) y RTX 1,589 (Carrier/Otis, 2020) pasan el 1,5 pero no son
        splits. Quedan informados como ajustes.
      - la serie es CONTINUA en la ejecucion: close de la rueda previa / close de
        la ejecucion NO se parece al ratio. Si se parece, la historia sigue en dos
        escalas: NO se registra y se informa (hay que correr splits.py corregir).
    origen = 'corregido' si splits.py dejo backup en data/backups (KLAC y CRWD,
    21/7/2026): el corte y las filas corregidas se reconstruyen desde el backup.
    El resto, 'historia_ajustada' (la serie se bajo ya en la escala nueva).

    Informa las diferencias contra polygon_splits.

Uso:
    python scripts/oneshot/create_splits_aplicados.py            # dry run
    python scripts/oneshot/create_splits_aplicados.py --apply
"""

import os
import sys
import glob
import argparse
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "manual"))

# splits.py configura el entorno LOCAL al importarse (ft_env).
from splits import DDL_SPLITS_APLICADOS, registrar_split, log  # noqa: E402

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402
from src.utils import yfinance_lock  # noqa: E402
from src.utils.precio_referencia import es_split_real, ratio_de_split  # noqa: E402
from src.utils.yahooquery_loader import eventos_split  # noqa: E402

# close previo / close de la ejecucion dentro de +-15% del ratio = dos escalas.
# Holgado a proposito: KLAC sin corregir dio 8,86 siendo 10:1 (subio +12,9% real).
TOL_ESCALA = 0.15
LOTE = 50


def es_split_de_precio(ratio):
    """Ratio real (>= 1,5 o su inverso) Y exacto: descarta ajustes por spinoff."""
    return es_split_real(ratio) and ratio_de_split(1.0, ratio) is not None


def origen_desde_backup(conn, ticker, execution_date, ratio):
    """
    Si splits.py corrigio este split, su backup guarda la escala vieja: de ahi
    salen el corte y las filas corregidas. Sin backup que lo explique, la serie
    se bajo ya ajustada.
    """
    rutas = sorted(glob.glob(os.path.join(ROOT, "data", "backups", f"precios_{ticker}_*.csv")))
    if rutas:
        hoy = pd.read_sql(text("SELECT fecha, close FROM precios_diarios WHERE ticker = :t"),
                          conn, params={"t": ticker})
        hoy["fecha"] = pd.to_datetime(hoy["fecha"]).dt.date
    for ruta in rutas:
        bak = pd.read_csv(ruta, usecols=["fecha", "close"])
        bak["fecha"] = pd.to_datetime(bak["fecha"]).dt.date
        m = bak.merge(hoy, on="fecha", suffixes=("_bak", "_hoy"))
        m["r"] = m["close_bak"].astype(float) / m["close_hoy"].astype(float)
        corregidas = m[(m["r"] / float(ratio) - 1).abs() <= 0.02]
        if corregidas.empty:
            continue
        ultima = corregidas["fecha"].max()
        if not (execution_date - timedelta(days=30) <= ultima < execution_date):
            continue
        return {"origen": "corregido",
                "fecha_corte_db": m.loc[m["fecha"] > ultima, "fecha"].min(),
                "filas_corregidas": int(len(corregidas)),
                "backup": os.path.relpath(ruta, ROOT).replace(os.sep, "/")}
    return {"origen": "historia_ajustada", "fecha_corte_db": None,
            "filas_corregidas": None, "backup": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="escribe (default: dry run)")
    args = ap.parse_args()

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")
    if engine.url.host not in ("localhost", "127.0.0.1"):
        log("[ERROR] el registro vive en la DB LOCAL. Abortado.")
        return 1

    with engine.connect() as conn:
        hist = {r.ticker: (r.desde, r.hasta) for r in conn.execute(text("""
            SELECT p.ticker, MIN(p.fecha) AS desde, MAX(p.fecha) AS hasta
            FROM precios_diarios p
            JOIN activos a ON a.ticker = p.ticker AND a.activo
            GROUP BY p.ticker
        """))}
    tickers = sorted(hist)
    inicio = min(d for d, _ in hist.values())
    log(f"Universo con precios: {len(tickers)} tickers | historia desde {inicio}")

    yfinance_lock.acquire("create_splits_aplicados.py")
    eventos = {}
    for i in range(0, len(tickers), LOTE):
        eventos.update(eventos_split(tickers[i:i + LOTE], inicio, date.today()))
    log(f"Eventos de split en Yahoo: {sum(len(v) for v in eventos.values())} "
        f"en {len(eventos)} tickers")

    a_registrar, dos_escalas, fuera, ajustes = [], [], [], []
    with engine.connect() as conn:
        for tk in sorted(eventos):
            desde, hasta = hist[tk]
            for fe, ratio in eventos[tk]:
                if not es_split_de_precio(ratio):
                    ajustes.append((tk, fe, ratio))
                    continue
                if fe <= desde or fe > hasta:
                    fuera.append((tk, fe, ratio))
                    continue
                prev = conn.execute(text("""
                    SELECT close FROM precios_diarios
                    WHERE ticker = :t AND fecha < :f ORDER BY fecha DESC LIMIT 1
                """), {"t": tk, "f": fe}).scalar()
                post = conn.execute(text("""
                    SELECT close FROM precios_diarios
                    WHERE ticker = :t AND fecha >= :f ORDER BY fecha LIMIT 1
                """), {"t": tk, "f": fe}).scalar()
                salto = float(prev) / float(post)
                if abs(salto / ratio - 1) <= TOL_ESCALA:
                    dos_escalas.append((tk, fe, ratio, salto))
                    continue
                a_registrar.append({"ticker": tk, "execution_date": fe, "ratio": ratio,
                                    "salto": salto,
                                    **origen_desde_backup(conn, tk, fe, ratio)})

    print()
    log(f"A REGISTRAR: {len(a_registrar)}")
    for s in a_registrar:
        extra = (f" | corte {s['fecha_corte_db']}, {s['filas_corregidas']} filas, {s['backup']}"
                 if s["origen"] == "corregido" else "")
        print(f"  {s['ticker']:<6} {s['execution_date']}  x{s['ratio']:<6g} {s['origen']:<18} "
              f"previo/ejecucion {s['salto']:.3f}{extra}")
    if dos_escalas:
        print()
        log(f"[!] {len(dos_escalas)} split(s) con la historia TODAVIA en dos escalas "
            f"(NO se registran). Corregir con scripts/manual/splits.py corregir:")
        for tk, fe, r, sa in dos_escalas:
            print(f"  {tk:<6} {fe}  x{r:g}  previo/ejecucion {sa:.3f}")
    print()
    log(f"Fuera de la historia de precios_diarios (no se registran): {len(fuera)}"
        + (": " + ", ".join(f"{t} {f} x{r:g}" for t, f, r in fuera) if fuera else ""))
    if ajustes:
        log("Eventos que NO son split de precio (dividendo en acciones, spinoff): "
            + ", ".join(f"{t} {f} x{r:g}" for t, f, r in ajustes))

    # Contraste con polygon_splits (informativo).
    try:
        with engine.connect() as conn:
            poly = {(r.ticker, r.execution_date): float(r.ratio) for r in conn.execute(text(
                "SELECT ticker, execution_date, ratio FROM polygon_splits "
                "WHERE ratio >= 1.5 OR ratio <= 1/1.5"))}
    except Exception as e:
        poly = None
        log(f"polygon_splits no disponible para contrastar ({str(e)[:60]})")
    if poly is not None:
        yq = {(t, f): r for t, evs in eventos.items() for f, r in evs if es_split_real(r)}
        solo_poly = sorted(k for k in poly if k not in yq and k[0] in hist)
        solo_yq = sorted(k for k in yq if k not in poly)
        distinto = sorted(k for k in poly if k in yq and abs(poly[k] / yq[k] - 1) > 0.01)
        print()
        log(f"Contraste con polygon_splits: coinciden {len(set(poly) & set(yq)) - len(distinto)}"
            f" | solo polygon {len(solo_poly)} | solo Yahoo {len(solo_yq)}"
            f" | ratio distinto {len(distinto)}")
        for etiqueta, claves in (("solo polygon", solo_poly), ("solo Yahoo", solo_yq),
                                 ("ratio distinto", distinto)):
            for k in claves:
                print(f"  {etiqueta:<14} {k[0]:<6} {k[1]}  polygon x{poly.get(k, float('nan')):g}"
                      f" | yahoo x{yq.get(k, float('nan')):g}")

    if not args.apply:
        print()
        log("[DRY RUN] no se escribio nada. Usar --apply.")
        return 0

    with engine.connect() as conn:
        conn.execute(text(DDL_SPLITS_APLICADOS))
        for s in a_registrar:
            registrar_split(conn, s["ticker"], s["execution_date"], s["ratio"],
                            origen=s["origen"], fuente_fecha="yahoo",
                            fecha_corte_db=s["fecha_corte_db"],
                            filas_corregidas=s["filas_corregidas"], backup=s["backup"])
        conn.commit()
        n = conn.execute(text("SELECT COUNT(*) FROM splits_aplicados")).scalar()
    log(f"splits_aplicados: {n} filas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
