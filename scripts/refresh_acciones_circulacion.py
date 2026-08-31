"""
refresh_acciones_circulacion.py
Puebla `acciones_circulacion`: serie de acciones en circulacion por ticker, en
base de split ACTUAL, apareable con precios_diarios.

Fuente primaria: yahooquery `OrdinarySharesNumber` (trimestral + anual).
Extension hacia atras: serie SEC de portada, SOLO en los tickers donde se
VALIDA que no cambiaron de base. La logica de validacion y armado es pura y
vive en src/utils/acciones_series.py; aca solo estan la red y la DB.

LOCAL-only. Ningun consumidor de produccion lee esta tabla todavia.

Lo que hay que saber para leer la salida
----------------------------------------
- yahooquery sirve ~5 puntos TRIMESTRALES (ultimo anio) y ~4 ANUALES. Entre
  2023 y 2025 hay UN punto por anio. Medido contra la serie trimestral real, el
  error de esa granularidad es mediana 0,24% / p90 2,74% / p99 11,35%, con cola
  en fusiones y recompras agresivas (BG 33%, CAR 30%, SPGI 29%). El error va
  1:1 al market cap.
- El conteo se mantiene entre puntos (ESCALON). No se interpola: seria inventar
  dato y borraria justo esas discontinuidades.
- El sesgo tiene signo: un conteo viejo queda ALTO (las empresas recompran),
  medido +0,29% en media. Sobreestima el market cap.

Uso:
    python scripts/refresh_acciones_circulacion.py
    python scripts/refresh_acciones_circulacion.py --desde 2023-01-01
    python scripts/refresh_acciones_circulacion.py --tickers AAPL,KLAC --verbose
    python scripts/refresh_acciones_circulacion.py --dry-run
"""

import argparse
import os
import sys
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import psycopg2
import psycopg2.extras

from scripts.oneshot.create_fundamentales_tables import _parse_env_file
from src.utils import acciones_series as A
from src.utils.yfinance_lock import acquire as acquire_yf_lock
from src.utils.yfinance_lock import _release as release_yf_lock

SEP = "=" * 64
TABLA = "acciones_circulacion"

# 2021: el objetivo. Yahoo no llega (su punto mas viejo cae en 2022-12/2023-06),
# asi que a 2021 se llega SOLO extendiendo con SEC en los tickers validados.
DESDE_DEFECTO = "2021-01-01"

CAMPO = "OrdinarySharesNumber"
CHUNK = 20            # mismo tamano que refresh_fundamentales
PAUSA_CHUNK = 2.0
MAX_RETRIES = 3


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _conn(env):
    return psycopg2.connect(
        host=env.get("DB_HOST", "localhost"), port=int(env.get("DB_PORT", 5432)),
        dbname=env.get("DB_NAME", "activos_ml"), user=env.get("DB_USER", "postgres"),
        password=env.get("DB_PASSWORD", ""))


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# --------------------------------------------------------------- lectura --
def universo(env, tickers=None):
    with _conn(env) as cx:
        with cx.cursor() as cur:
            if tickers:
                cur.execute("SELECT ticker FROM activos WHERE ticker = ANY(%s) "
                            "ORDER BY ticker", (tickers,))
            else:
                cur.execute("SELECT ticker FROM activos WHERE activo IS TRUE "
                            "ORDER BY ticker")
            return [r[0] for r in cur.fetchall()]


# Orden de preferencia de las series de respaldo, de mas a menos fiel:
#   portada  -- dei, point-in-time, NO se re-expresa ante un split
#   balance  -- us-gaap, point-in-time, pero SI se re-expresa
#   promedio -- promedio ponderado del trimestre: otra magnitud
#   polygon  -- ULTIMO recurso, y a proposito: es la unica que sale de un
#               tercero en vez de del propio emisor, y la unica que cuesta
#               pedidos con cupo. Existe porque hay un caso que SEC no puede
#               cubrir de ninguna manera: los filers multiclase que tagean
#               cada conteo por clase. companyfacts descarta todo lo
#               dimensionado, asi que V (Visa) llega con CERO puntos en los
#               tres niveles -- no es que el nivel elegido sea malo, es que
#               no hay ninguno. Son 401 ruedas sin market cap.
PREFERENCIA_FUENTE = ("portada", "balance", "promedio_diluido", "polygon")

# Las de SEC llevan prefijo; polygon se nombra sola.
def _etiqueta(fuente, rebasada):
    base = fuente if fuente == "polygon" else "sec_" + fuente
    return base + ("_rb" if rebasada else "")


def series_sec(env):
    """
    {ticker: {fuente: [{fecha, shares, filed}]}} -- TODOS los niveles.

    `filed` viaja con cada punto porque A.rebasar corta por ahi, no por la
    fecha de periodo: un numero esta en la base vigente cuando se PRESENTO.
    Sin esta columna el rebase cae al fallback por `fecha` y rompe justo los
    trimestres que cierran antes del split y se presentan despues -- GOOG
    2022-06-30 es el caso testigo.

    Antes esto filtraba `WHERE fuente='portada'` y descartaba en silencio los
    respaldos que sec_acciones ya habia calculado y persistido. El efecto era
    invisible y caro: los 21 filers de clases multiples (GOOG, META, F, MA,
    NKE, UPS, V...) no tienen portada -- companyfacts descarta los hechos
    dimensionales -- asi que llegaban con CERO puntos, la validacion anotaba
    "sin solapamiento" y se quedaban sin extension. Medido antes del arreglo:
    11.491 ruedas sin market cap en 43 tickers, hasta el 41% de un ticker.
    """
    out = {}
    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute("SELECT ticker, fecha, shares, fuente, filed "
                        "FROM fundamentales_sec_acciones "
                        "ORDER BY ticker, fecha")
            for tk, f, sh, fuente, filed in cur.fetchall():
                out.setdefault(tk, {}).setdefault(fuente, []).append(
                    {"fecha": f.isoformat(), "shares": float(sh),
                     "filed": filed.isoformat() if filed else None})
    return out


def series_polygon(env):
    """
    {ticker: [{fecha, shares}]} desde polygon_acciones.

    USA `weighted_shares`, NO `share_class_shares`. La distincion es la unica
    razon por la que esta fuente sirve: `share_class_shares` es UNA clase, y en
    V da 1.635 MM contra 2.079 MM de yahoo (ratio 0,79) -- justo el error que
    esta tabla viene a evitar. `weighted_shares` es el total y cruza contra
    yahoo en 1,0136 y 1,0277 en las dos fechas comunes de V.

    No trae `filed`: el conteo esta fechado "as of" esa fecha, asi que para el
    rebase `fecha` ES el momento cuya base rige, y el fallback de A.rebasar
    hace lo correcto sin caso especial.

    Si la tabla no existe devuelve {} y el refresh sigue andando igual que
    antes: como los splits, es una MEJORA opcional y no un requisito.
    """
    out = {}
    try:
        with _conn(env) as cx:
            with cx.cursor() as cur:
                cur.execute("SELECT ticker, fecha, weighted_shares "
                            "FROM polygon_acciones "
                            "WHERE weighted_shares IS NOT NULL "
                            "ORDER BY ticker, fecha")
                for tk, f, sh in cur.fetchall():
                    out.setdefault(tk, []).append(
                        {"fecha": f.isoformat(), "shares": float(sh)})
    except Exception:
        return {}
    return out


def splits_por_ticker(env):
    """
    {ticker: [{fecha, ratio}]} desde polygon_splits.

    Si la tabla no existe todavia devuelve {} y todo sigue funcionando como
    antes: el rebase es una MEJORA opcional, no un requisito. Un entorno sin
    Polygon ingerido pierde las ruedas de base de split -- que es exactamente
    lo que pasaba hasta ahora -- pero no se rompe.
    """
    out = {}
    try:
        with _conn(env) as cx:
            with cx.cursor() as cur:
                cur.execute("SELECT ticker, execution_date, ratio FROM polygon_splits "
                            "WHERE ratio IS NOT NULL ORDER BY ticker, execution_date")
                for tk, fe, ratio in cur.fetchall():
                    out.setdefault(tk, []).append(
                        {"fecha": fe.isoformat(), "ratio": float(ratio)})
    except Exception:
        return {}
    return out


def construir_mejor(yah, niveles, desde, splits=None):
    """
    Prueba los niveles SEC en orden de preferencia y se queda con el PRIMERO
    que logra extender la serie hacia atras.

    Por que probar y no elegir de antemano: ninguna regla ciega gana. Elegir
    por preferencia deja a TRIP y LYFT con una portada de 4-6 puntos que
    arranca en 2025 (930 ruedas perdidas); elegir por cobertura rompe UPST y
    SNOW, que con la portada ya andaban. Lo unico que distingue un nivel util
    de uno inservible es si PASA LA VALIDACION contra yahooquery, y eso solo se
    sabe corriendola.

    Los niveles no se mezclan entre si: se elige uno y se usa entero.

    Dentro de cada nivel se prueban DOS variantes -- con rebase y sin el -- y
    tambien ahi decide la validacion. El motivo esta en el docstring de
    A.rebasar: Polygon publica en el mismo endpoint los splits reales y los
    ajustes de precio por spinoff, que no mueven el conteo. Aplicar un factor
    de spinoff corrompe la serie (HON: 0,500 -> 0,471); no aplicar un split
    real la pierde entera (WMT, NVDA, AMZN...). Como no hay forma confiable de
    distinguirlos por el ratio -- BBD 1,1 e ITUB 1,03 son bonificaciones que
    SI cuentan --, se corren las dos y gana la que valide.

    El orden importa: primero SIN rebase. Si la serie ya validaba, no se la
    toca; el rebase entra solo donde hace falta. Asi un ratio espurio nunca
    puede desplazar a una serie que ya estaba sana.

    Si ninguno extiende se devuelve el intento del nivel mas preferido, para
    que el diagnostico que se persiste hable del mejor dato disponible y no
    del ultimo que se probo.
    """
    primero = None
    for fuente in PREFERENCIA_FUENTE:
        base = niveles.get(fuente) or []
        if not base:
            continue
        for usar_rebase in (False, True):
            if usar_rebase and not splits:
                continue
            serie_sec, rb = ((base, []) if not usar_rebase
                             else A.rebasar(base, splits))
            if usar_rebase and not rb:
                continue          # el rebase no cambio nada: ya se probo
            etiqueta = _etiqueta(fuente, bool(rb))
            serie, diag = A.construir(yah, serie_sec, desde=desde,
                                      etiqueta=etiqueta)
            diag["rebase"] = len(rb)
            diag["fuente_sec"] = fuente
            if primero is None:
                primero = (serie, diag)
            if diag["extendido"]:
                return serie, diag
    if primero is not None:
        return primero
    serie, diag = A.construir(yah, [], desde=desde)
    diag["fuente_sec"] = None
    return serie, diag



# ------------------------------------------------------------------ red --
def _extraer(df, periodo):
    """DataFrame de yahooquery -> {ticker: [{fecha, shares, periodo}]}."""
    out = {}
    if df is None or isinstance(df, str) or not hasattr(df, "reset_index"):
        return out
    d = df.reset_index()
    if CAMPO not in d.columns or "asOfDate" not in d.columns:
        return out
    for _, r in d.iterrows():
        v = r.get(CAMPO)
        if v is None or v != v or float(v) <= 0:
            continue
        tk = r.get("symbol")
        fecha = r["asOfDate"]
        out.setdefault(tk, []).append(
            {"fecha": str(fecha)[:10], "shares": float(v), "periodo": periodo})
    return out


def bajar_chunk(tickers):
    """Trimestral + anual del chunk. Devuelve {ticker: [puntos]} unificado."""
    from yahooquery import Ticker
    t = Ticker(tickers, asynchronous=True, max_workers=8)
    datos = {}
    for freq, periodo in (("q", "q"), ("a", "a")):
        df = t.get_financial_data([CAMPO], frequency=freq, trailing=False)
        for tk, pts in _extraer(df, periodo).items():
            datos.setdefault(tk, []).extend(pts)
    # Una fecha, un punto: el trimestral gana sobre el anual (mismo dato, pero
    # el trimestral es el que se refresca mas seguido).
    for tk, pts in datos.items():
        por_fecha = {}
        for p in sorted(pts, key=lambda x: (x["fecha"], x["periodo"] != "q")):
            por_fecha.setdefault(p["fecha"], p)
        datos[tk] = [por_fecha[f] for f in sorted(por_fecha)]
    return datos


def bajar_chunk_con_retry(tickers, idx):
    espera = 30
    for intento in range(1, MAX_RETRIES + 1):
        try:
            datos = bajar_chunk(tickers)
        except Exception as e:
            log(f"  chunk {idx} intento {intento}: {type(e).__name__}: {e}")
            datos = {}
        # Vacio para TODO el chunk huele a rate limit (yahooquery no lo levanta
        # como excepcion: devuelve vacio). Ver CLAUDE.md.
        if datos:
            return datos
        if intento < MAX_RETRIES:
            log(f"  chunk {idx}: vacio, posible rate limit -> backoff {espera}s")
            time.sleep(espera)
            espera *= 2
    log(f"  chunk {idx}: FALLA tras {MAX_RETRIES} intentos")
    return {}


# ------------------------------------------------------------ escritura --
def reemplazar_serie(env, ticker, filas):
    """
    La serie se REGENERA entera por ticker: es una proyeccion de las fuentes,
    no un historico acumulado. Si un punto deja de calificar (cambio el
    veredicto de la validacion), tiene que desaparecer.
    """
    cols = ["ticker", "fecha", "shares", "periodo", "fuente"]
    ph = ", ".join(f"%({c})s" for c in cols)
    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLA} WHERE ticker=%s", (ticker,))
            if filas:
                psycopg2.extras.execute_batch(cur, (
                    f"INSERT INTO {TABLA} ({', '.join(cols)}) VALUES ({ph})"),
                    filas, page_size=200)
        cx.commit()
    return len(filas)


def upsert_validacion(env, fila):
    cols = ["ticker", "extendido", "desde_efectivo", "n_yahoo", "n_sec_usados",
            "n_pares", "ratio_min", "ratio_max", "motivo"]
    ph = ", ".join(f"%({c})s" for c in cols)
    setc = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "ticker")
    with _conn(env) as cx:
        with cx.cursor() as cur:
            cur.execute(f"INSERT INTO {TABLA}_validacion ({', '.join(cols)}) "
                        f"VALUES ({ph}) ON CONFLICT (ticker) DO UPDATE SET "
                        f"{setc}, updated_at=NOW()", fila)
        cx.commit()


# ----------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(
        description="Refresh de acciones_circulacion (LOCAL-only)")
    p.add_argument("--tickers", help="CSV; default: universo activo")
    p.add_argument("--desde", default=DESDE_DEFECTO,
                   help=f"Fecha objetivo de arranque (default {DESDE_DEFECTO}). "
                        f"Yahoo no llega solo; se alcanza extendiendo con SEC "
                        f"en los tickers validados.")
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--dry-run", action="store_true", help="No escribe en la DB")
    p.add_argument("--verbose", action="store_true", help="Detalle por ticker")
    args = p.parse_args()

    env = _parse_env_file(os.path.join(ROOT, ".env"))
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    print()
    print(SEP)
    print(f"  REFRESH ACCIONES EN CIRCULACION"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")
    print(SEP)
    print()

    lista = universo(env, tickers)
    log(f"tickers: {len(lista)}  |  objetivo: desde {args.desde}")

    acquire_yf_lock("refresh_acciones_circulacion")
    try:
        datos = {}
        chunks = list(_chunks(lista, args.chunk))
        for i, ch in enumerate(chunks, 1):
            log(f"chunk {i}/{len(chunks)} ({len(ch)} tickers)...")
            datos.update(bajar_chunk_con_retry(ch, i))
            if i < len(chunks):
                time.sleep(PAUSA_CHUNK)
    finally:
        release_yf_lock()

    log(f"yahooquery devolvio datos para {len(datos)} tickers")

    sec = series_sec(env)
    sp = splits_por_ticker(env)
    log(f"splits de Polygon: {sum(len(v) for v in sp.values())} "
        f"en {len(sp)} tickers")
    pol = series_polygon(env)
    log(f"acciones de Polygon: {sum(len(v) for v in pol.values())} puntos "
        f"en {len(pol)} tickers")
    n_filas = n_ext = 0
    sin_datos, no_ext = [], []
    for ticker in lista:
        yah = datos.get(ticker, [])
        # Polygon entra como un candidato mas del mismo diccionario: se prueba
        # ultimo y tiene que pasar la MISMA validacion que los niveles de SEC.
        niveles = dict(sec.get(ticker, {}))
        if pol.get(ticker):
            niveles["polygon"] = pol[ticker]
        serie, diag = construir_mejor(yah, niveles, args.desde,
                                      sp.get(ticker))
        if not serie:
            sin_datos.append(ticker)
            continue
        filas = [{"ticker": ticker, **p} for p in serie]
        if not args.dry_run:
            n_filas += reemplazar_serie(env, ticker, filas)
            val = diag["validacion"] or {}
            upsert_validacion(env, {
                "ticker": ticker, "extendido": diag["extendido"],
                "desde_efectivo": diag["desde_efectivo"],
                "n_yahoo": diag["n_yahoo"], "n_sec_usados": diag["n_sec_usados"],
                "n_pares": val.get("n_pares"), "ratio_min": val.get("ratio_min"),
                "ratio_max": val.get("ratio_max"), "motivo": diag["motivo"]})
        else:
            n_filas += len(filas)
        if diag["extendido"]:
            n_ext += 1
        else:
            no_ext.append(f"{ticker} ({diag['motivo']})")
        if args.verbose:
            log(f"  {ticker:<6} {len(serie):>2} pts  desde {diag['desde_efectivo']}"
                f"  {'EXTENDIDO' if diag['extendido'] else 'solo yahoo'}"
                f"  {diag['motivo'] or ''}")

    print()
    print(SEP)
    print(f"  OK  |  tickers: {len(lista) - len(sin_datos)}  |  puntos: {n_filas}"
          f"  |  extendidos a {args.desde}: {n_ext}")
    if sin_datos:
        print(f"  sin datos: {len(sin_datos)}  ->  {', '.join(sin_datos)}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
