"""
splits.py
Deteccion y correccion de splits no aplicados en precios_diarios.

PROBLEMA (detectado 2026-07-21):
    precios_diarios NO se re-ajusta hacia atras cuando un ticker hace split.
    El pipeline diario solo trae los dias nuevos (ya ajustados por Yahoo), asi
    que la historia previa queda en la escala VIEJA. Resultado: la serie de un
    ticker queda partida en dos escalas y todo lo que la cruza se rompe.

    Caso real: KLAC split 10:1 el 2026-06-11 y CRWD 4:1 el 2026-06-30. En FT
    ocho posiciones abiertas se cerraron al precio post-split con la cantidad
    pre-split -> -12.709,83 USD de perdidas FICTICIAS, con motivos de salida
    (STOP_LOSS_ATR, SL_PROTECCION) disparados por un derrumbe que no existio.
    Ademas toda SMA/RSI/ATR que cruce la fecha del split queda corrupta, y con
    ella las features de ML.

DETECCION (2 etapas, para no gastar red al pedo):
    1. Barrido local: variaciones diarias > umbral (default 25%) en
       precios_diarios. Baratisimo, sirve de alerta temprana en el pipeline.
    2. Verificacion contra Yahoo SOLO de los candidatos: si el ratio
       db/yahoo es constante y distinto de 1 antes de una fecha, ES un split.
       Un movimiento real de mercado da ratio 1 en todo el rango.

    La etapa 2 es la que evita falsos positivos: un -38% real (CAR, 22/4/2026)
    y un split se ven identicos en la etapa 1.

CORRECCION:
    Re-descarga el historial COMPLETO ya ajustado y lo upsertea (ON CONFLICT
    DO UPDATE), despues recomputa las tablas derivadas del ticker. No aplica
    un divisor a mano: re-bajar es autoritativo y cubre splits multiples.

    NO corrige ft_operaciones: eso es historia de trading y se trata aparte
    (ver docs/forward_testing/METRICAS.md).

Uso:
    python scripts/manual/splits.py detectar
    python scripts/manual/splits.py detectar --umbral 0.30 --desde 2026-01-01
    python scripts/manual/splits.py corregir KLAC CRWD --dry-run
    python scripts/manual/splits.py corregir KLAC CRWD --apply
    python scripts/manual/splits.py corregir KLAC --apply --no-derivadas
"""

import os
import sys
import argparse
from datetime import date, datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "forward_testing"))

from ft_env import configurar_entorno_local  # noqa: E402
configurar_entorno_local()

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402

BACKUP_DIR = os.path.join(ROOT, "data", "backups")

# Ratios de split "limpios" que aceptamos como plausibles (directos e inversos)
RATIOS_PLAUSIBLES = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30]
TOL_RATIO = 0.02      # 2% de tolerancia contra el ratio limpio
# Dispersion maxima RELATIVA del ratio db/yahoo para considerarlo constante.
# Relativa y no absoluta: sobre un ratio de 10 (split 10:1), un 0.01 absoluto
# serian 0.1% -- inalcanzable en cuanto hay ajustes por dividendos en el medio.
TOL_CONSTANTE = 0.03


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Etapa 1: barrido local ────────────────────────────────────────────────────

def detectar_candidatos(engine, umbral=0.25, desde=None):
    """Variaciones diarias > umbral en precios_diarios. Solo lectura."""
    desde = desde or (date.today() - timedelta(days=365))
    with engine.connect() as conn:
        px = pd.read_sql(text("""
            SELECT ticker, fecha, close FROM precios_diarios
            WHERE fecha >= :desde ORDER BY ticker, fecha
        """), conn, params={"desde": desde})

    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["prev"] = px.groupby("ticker")["close"].shift(1)
    px["chg"] = px["close"] / px["prev"] - 1
    cand = px[px["chg"].abs() > umbral].dropna(subset=["chg"]).copy()
    cand["ratio"] = cand["prev"] / cand["close"]
    cand["ratio_limpio"] = cand["ratio"].apply(_ratio_limpio)
    return cand.sort_values(["ticker", "fecha"])


def _ratio_limpio(r):
    """Devuelve el ratio de split limpio mas cercano, o None si no se parece."""
    for cand in RATIOS_PLAUSIBLES:
        for valor in (cand, 1.0 / cand):
            if abs(r - valor) / valor < TOL_RATIO:
                return round(valor, 4) if valor >= 1 else round(valor, 6)
    return None


# ── Etapa 2: verificacion contra Yahoo ────────────────────────────────────────

def verificar_split(engine, ticker, desde=None):
    """
    Compara el historial de la DB contra yahooquery (ajustado).

    Devuelve dict con: es_split, ratio, fecha_split, n_filas_mal.
    Un movimiento REAL de mercado da ratio 1.0 en todo el rango; un split
    deja un escalon: ratio constante != 1 antes de la fecha, 1 despues.
    """
    from src.utils.yahooquery_loader import download_batch

    # El lock de yfinance se toma UNA vez a nivel comando (es por proceso/IP,
    # no por llamada): pedirlo aca aborta en el segundo ticker del loop.
    desde = desde or date(2024, 1, 1)
    res = download_batch([ticker], start=desde, end=date.today())
    yq = res.get(ticker)
    if yq is None or yq.empty:
        return {"ticker": ticker, "es_split": None, "error": "yahooquery sin datos"}

    yq = yq.copy()
    yq.index = pd.to_datetime(yq.index)
    serie_yq = yq["Close"].astype(float)

    with engine.connect() as conn:
        db = pd.read_sql(text("""
            SELECT fecha, close FROM precios_diarios
            WHERE ticker = :tk AND fecha >= :desde ORDER BY fecha
        """), conn, params={"tk": ticker, "desde": desde})
    db["fecha"] = pd.to_datetime(db["fecha"])
    db["close"] = pd.to_numeric(db["close"], errors="coerce")

    comp = pd.DataFrame({"db": db.set_index("fecha")["close"], "yahoo": serie_yq}).dropna()
    if comp.empty:
        return {"ticker": ticker, "es_split": None, "error": "sin fechas comparables"}

    comp["ratio"] = comp["db"] / comp["yahoo"]
    mal = comp[(comp["ratio"] - 1.0).abs() > 0.01]

    if mal.empty:
        return {"ticker": ticker, "es_split": False, "n_filas_mal": 0,
                "detalle": "la DB coincide con Yahoo: movimiento real, no split"}

    ratio = float(mal["ratio"].median())
    disperso = ((mal["ratio"].max() - mal["ratio"].min()) / ratio) > TOL_CONSTANTE
    limpio = _ratio_limpio(ratio)
    fecha_split = comp[(comp["ratio"] - 1.0).abs() <= 0.01].index.min()

    # Un ratio CONSTANTE no alcanza para declarar split: tiene que ser ademas
    # un ratio PLAUSIBLE de split. ORCL (0.9841) y DELL (0.9744) dan constante
    # pero no existe un split de 0.98:1 -- es otra discrepancia (ajuste por
    # dividendos / backfill viejo). Sin este filtro el script recomendaba
    # "corregir ORCL DELL", o sea dividir datos sanos por 0.98 y romperlos.
    # Este script recomienda una accion DESTRUCTIVA: un falso positivo cuesta
    # mas que un falso negativo.
    es_split = (not disperso) and (limpio is not None)

    detalle = None
    if disperso:
        detalle = "ratio disperso: no es un cambio de escala limpio"
    elif limpio is None:
        detalle = (f"ratio {ratio:.4f} constante pero NO es un ratio de split "
                   f"plausible -- discrepancia de datos a investigar, NO corregir "
                   f"con divisor")

    return {
        "ticker":       ticker,
        "es_split":     es_split,
        "ratio":        round(ratio, 4),
        "ratio_limpio": limpio,
        "fecha_split":  fecha_split.date() if pd.notna(fecha_split) else None,
        "n_filas_mal":  len(mal),
        "rango_mal":    (mal.index.min().date(), mal.index.max().date()),
        "disperso":     disperso,
        "detalle":      detalle,
    }


# ── Correccion ────────────────────────────────────────────────────────────────

def backup_precios(engine, ticker):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT * FROM precios_diarios WHERE ticker = :tk ORDER BY fecha
        """), conn, params={"tk": ticker})
    ruta = os.path.join(
        BACKUP_DIR, f"precios_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    df.to_csv(ruta, index=False)
    log(f"   backup: {len(df)} filas -> {os.path.relpath(ruta, ROOT)}")
    return ruta


def corregir_ticker(engine, ticker, aplicar=False, derivadas=True, ratio=None,
                    fecha_split=None):
    """
    Aplica el divisor del split a las filas ANTERIORES a fecha_split.

    Por que divisor y no re-descargar la serie ajustada:
        precios_diarios guarda el close CRUDO tal como lo devolvio Yahoo el dia
        que se bajo -- nunca se re-ajusta por dividendos hacia atras. El `Close`
        de yahooquery, en cambio, viene ajustado por split Y por dividendos
        (verificado: KLAC da ratio 9.8369 en vez de 10 exacto; CRWD, que no paga
        dividendos, da 4.0000 clavado).

        Sobrescribir con la serie ajustada dejaria a estos tickers en una base
        distinta a la de los otros 198 -- cambiariamos un problema por otro, mas
        sutil. El divisor corrige EXACTAMENTE el split y no toca nada mas.

    OHLC y adj_close se dividen por el ratio; volumen se multiplica.
    """
    log(f"--- {ticker} ---")

    if ratio is None or fecha_split is None:
        log("   verificando contra Yahoo...")
        v = verificar_split(engine, ticker)
        if not v.get("es_split"):
            log(f"   no es un split pendiente ({v.get('detalle') or v.get('error') or 'ratio disperso'}). Se omite.")
            return False
        ratio = v["ratio_limpio"] or round(v["ratio"])
        fecha_split = v["fecha_split"]

    with engine.connect() as conn:
        n = conn.execute(text("""
            SELECT COUNT(*) FROM precios_diarios
            WHERE ticker = :tk AND fecha < :f
        """), {"tk": ticker, "f": fecha_split}).scalar()
        muestra = pd.read_sql(text("""
            SELECT fecha, open, high, low, close, volume FROM precios_diarios
            WHERE ticker = :tk AND fecha < :f ORDER BY fecha DESC LIMIT 2
        """), conn, params={"tk": ticker, "f": fecha_split})

    log(f"   split {ratio}:1 desde {fecha_split} | {n} filas a corregir (fecha < {fecha_split})")
    for _, r in muestra.iterrows():
        log(f"     {r['fecha']}: close {float(r['close']):.4f} -> "
            f"{float(r['close'])/ratio:.4f} | vol {int(r['volume']):,} -> "
            f"{int(r['volume']*ratio):,}")

    if not aplicar:
        log("   [DRY RUN] no se escribe nada. Usar --apply para corregir.")
        return True
    if n == 0:
        log("   nada que corregir.")
        return True

    backup_precios(engine, ticker)

    with engine.connect() as conn:
        res = conn.execute(text("""
            UPDATE precios_diarios
            SET open      = open      / :r,
                high      = high      / :r,
                low       = low       / :r,
                close     = close     / :r,
                adj_close = adj_close / :r,
                volume    = (volume * :r)::bigint
            WHERE ticker = :tk AND fecha < :f
        """), {"r": ratio, "tk": ticker, "f": fecha_split})
        conn.commit()
    log(f"   precios_diarios: {res.rowcount} filas corregidas.")

    if derivadas:
        log("   recomputando indicadores tecnicos...")
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT ticker, fecha, open, high, low, close, volume
                FROM precios_diarios WHERE ticker = :tk ORDER BY fecha
            """), conn, params={"tk": ticker})
        df["fecha"] = pd.to_datetime(df["fecha"])
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        from src.indicators.technical import procesar_indicadores_ticker
        procesar_indicadores_ticker(ticker, df, guardar_db=True)
        log("   OK (features y z-scores se recomputan al final, en bulk)")

    return True


def recomputar_derivadas_bulk(desde):
    """
    features (bulk) + z-scores. Se corren UNA vez despues de corregir todos
    los tickers: procesar_features_* recalcula el universo entero, asi que
    llamarlo por ticker seria tirar tiempo.
    """
    log("Recomputando features (precio_accion + market_structure, bulk)...")
    from src.indicators.precio_accion import procesar_features_precio_accion
    from src.indicators.market_structure import procesar_features_market_structure
    procesar_features_precio_accion()
    procesar_features_market_structure()

    log(f"Backfill z-scores de acciones desde {desde}...")
    from src.utils.zscore_pipeline import backfill_zscore_tickers
    n = backfill_zscore_tickers(desde=desde)
    log(f"   ticker_zscore_diario: {n} filas")


# ── Chequeo automatico (pipeline diario) ──────────────────────────────────────

def chequeo_diario(engine, dias=7, umbral=0.25, alertar=True, usar_lock=True,
                   max_verificar=6, log=log):
    """
    Chequeo desatendido para el pipeline diario. **Detecta y AVISA, NO corrige.**

    Por que no corrige solo: corregir reescribe precios_diarios, que es la fuente
    de verdad de todo el sistema, y el detector ya produjo falsos positivos una
    vez (ORCL/DELL, ratio ~0.98). Una correccion automatica sobre un falso
    positivo rompe datos sanos sin que nadie se entere. El costo de esperar a que
    una persona corra `corregir` es bajo; el de corregir mal, no.

    Barato por diseno: la etapa 1 es una query local sobre los ultimos `dias`.
    La etapa 2 (red) solo corre si hay candidatos, que en un dia normal son cero.

    Args:
        usar_lock: False cuando lo llama un script que YA tiene el lock de
                   yfinance (ej. recovery_incremental) -- pedirlo de nuevo
                   aborta el proceso entero.

    Returns dict: {candidatos, confirmados, discrepancias, error}
    """
    from datetime import timedelta
    res = {"candidatos": [], "confirmados": [], "discrepancias": [], "error": None}

    try:
        desde = date.today() - timedelta(days=dias)
        cand = detectar_candidatos(engine, umbral, desde)
        if cand.empty:
            log(f"SPLITS: sin variaciones > {umbral:.0%} en los ultimos {dias} dias. OK")
            return res

        cand = cand.copy()
        cand["mag"] = cand["chg"].abs()
        cand["es_caida"] = cand["chg"] < 0
        tickers = (cand.sort_values(["es_caida", "mag"], ascending=[False, False])
                   ["ticker"].drop_duplicates().tolist()[:max_verificar])
        res["candidatos"] = tickers
        log(f"SPLITS: {len(cand)} salto(s) > {umbral:.0%}; verificando {tickers} contra Yahoo...")

        if usar_lock:
            from src.utils import yfinance_lock
            yfinance_lock.acquire("splits.py chequeo_diario")

        for tk in tickers:
            try:
                r = verificar_split(engine, tk)
            except Exception as e:
                log(f"  {tk}: no se pudo verificar ({str(e)[:80]})")
                continue
            if r.get("es_split"):
                res["confirmados"].append(r)
                log(f"  {tk}: SPLIT CONFIRMADO {r['ratio_limpio']}:1 desde "
                    f"{r['fecha_split']} ({r['n_filas_mal']} filas)")
            elif r.get("detalle") and "NO es un ratio de split" in r["detalle"]:
                res["discrepancias"].append(r)
                log(f"  {tk}: discrepancia ratio {r['ratio']} (no es split)")
            else:
                log(f"  {tk}: movimiento real, OK")

    except Exception as e:
        res["error"] = str(e)[:200]
        log(f"SPLITS: ERROR en el chequeo (no critico): {res['error']}")
        return res

    if alertar and (res["confirmados"] or res["discrepancias"]):
        _alertar_telegram(res)
    return res


def _alertar_telegram(res):
    """Alerta de split pendiente. Silenciosa si Telegram no esta configurado."""
    try:
        from src.pipeline.telegram_notifier import _send
    except Exception:
        return

    lineas = ["*ALERTA: split sin aplicar en precios_diarios*", ""]
    for r in res["confirmados"]:
        lineas.append(
            f"*{r['ticker']}* - split {r['ratio_limpio']}:1 desde {r['fecha_split']}\n"
            f"  {r['n_filas_mal']} filas en la escala vieja")
    if res["confirmados"]:
        tks = " ".join(r["ticker"] for r in res["confirmados"])
        lineas += [
            "",
            "La historia previa quedo en la escala vieja: SMA/RSI/ATR rotos y "
            "stops que pueden dispararse solos sobre las posiciones abiertas.",
            "",
            f"`python scripts/manual/splits.py corregir {tks} --apply`",
        ]
    if res["discrepancias"]:
        tks = ", ".join(r["ticker"] for r in res["discrepancias"])
        lineas += ["", f"_Ademas, discrepancia contra Yahoo (NO es split, no "
                       f"corregir con divisor): {tks}_"]
    try:
        _send("\n".join(lineas))
    except Exception:
        pass


# ── Comandos ──────────────────────────────────────────────────────────────────

def cmd_detectar(args):
    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")
    desde = pd.Timestamp(args.desde).date() if args.desde else None

    cand = detectar_candidatos(engine, args.umbral, desde)
    if cand.empty:
        log(f"Sin variaciones diarias > {args.umbral:.0%}. Nada que revisar.")
        return 0

    log(f"Candidatos (variacion diaria > {args.umbral:.0%}): {len(cand)}")
    print()
    print(f"{'ticker':<8} {'fecha':<12} {'prev':>12} {'close':>12} {'var':>9} "
          f"{'ratio':>8} {'limpio':>8}")
    for _, r in cand.iterrows():
        limpio = r["ratio_limpio"] if r["ratio_limpio"] else "-"
        print(f"{r['ticker']:<8} {str(r['fecha']):<12} {r['prev']:>12.4f} "
              f"{r['close']:>12.4f} {r['chg']:>+8.1%} {r['ratio']:>8.3f} {str(limpio):>8}")

    # OJO: el ratio observado NO es el ratio del split salvo que el precio no
    # se haya movido ese dia. KLAC dio 8.856 siendo un split 10:1, porque el
    # 11/6 ademas subio +12.9% real. Por eso `ratio_limpio` es informativo y
    # NO se usa como filtro: el unico juez confiable es Yahoo (etapa 2).
    #
    # Se priorizan las CAIDAS (un split forward divide el precio) pero NO se
    # descartan las subidas: un split INVERSO multiplica el precio y se veria
    # como un salto hacia arriba. Filtrar solo caidas dejaba ese caso ciego.
    cand = cand.copy()
    cand["mag"] = cand["chg"].abs()
    cand["es_caida"] = cand["chg"] < 0
    sospechosos = (cand.sort_values(["es_caida", "mag"], ascending=[False, False])
                   ["ticker"].drop_duplicates().tolist())

    print()
    if not sospechosos:
        log("Sin caidas relevantes: no hay candidatos a split.")
        return 0

    if not args.verificar:
        log(f"Candidatos a verificar (caidas, mayor a menor): {sospechosos[:15]}")
        log("Correr con --verificar para confirmar contra Yahoo (etapa 2).")
        return 0

    tope = args.max_verificar
    if len(sospechosos) > tope:
        log(f"[WARN] {len(sospechosos)} candidatos; se verifican los {tope} "
            f"mayores. Subir con --max-verificar.")
        sospechosos = sospechosos[:tope]

    log(f"Verificando {len(sospechosos)} candidatos contra Yahoo...")
    from src.utils import yfinance_lock
    yfinance_lock.acquire("splits.py")

    confirmados = []
    revisar = []
    for tk in sospechosos:
        r = verificar_split(engine, tk)
        if r.get("es_split"):
            confirmados.append(r)
            log(f"  {tk}: SPLIT CONFIRMADO ratio {r['ratio']} "
                f"(limpio {r['ratio_limpio']}) desde {r['fecha_split']} "
                f"| {r['n_filas_mal']} filas mal")
        elif r.get("es_split") is False:
            det = r.get("detalle") or ""
            if "NO es un ratio de split" in det:
                revisar.append(r)
                log(f"  {tk}: [!] DISCREPANCIA (no split) ratio {r['ratio']} "
                    f"en {r['n_filas_mal']} filas -- {det}")
            else:
                log(f"  {tk}: OK -- {det}")
        else:
            log(f"  {tk}: INDETERMINADO ({r.get('error', 'ratio disperso')}) "
                f"-- revisar a mano")

    print()
    if revisar:
        log(f"[!] {len(revisar)} ticker(s) con discrepancia contra Yahoo que NO es "
            f"un split: {' '.join(r['ticker'] for r in revisar)}")
        log("    NO corregir con divisor. Investigar el origen (ajuste por "
            "dividendos, backfill viejo, fuente distinta).")
    if confirmados:
        tks = " ".join(c["ticker"] for c in confirmados)
        log(f"SPLITS SIN APLICAR: {tks}")
        log(f"Corregir con: python scripts/manual/splits.py corregir {tks} --apply")
        return 1
    log("Ningun split pendiente: los saltos restantes son movimientos reales.")
    return 0


def cmd_chequeo(args):
    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")
    res = chequeo_diario(engine, dias=args.dias, umbral=args.umbral,
                         alertar=not args.sin_alerta)
    if res["confirmados"]:
        tks = " ".join(r["ticker"] for r in res["confirmados"])
        log(f"[!] SPLITS SIN APLICAR: {tks}")
        log(f"    python scripts/manual/splits.py corregir {tks} --apply")
        return 1
    return 0


def cmd_corregir(args):
    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")
    log(f"Tickers: {args.tickers}  {'[APPLY]' if args.apply else '[DRY RUN]'}")

    from src.utils import yfinance_lock
    yfinance_lock.acquire("splits.py")

    ok = []
    for tk in args.tickers:
        if corregir_ticker(engine, tk.upper(), aplicar=args.apply,
                           derivadas=not args.no_derivadas,
                           ratio=args.ratio, fecha_split=args.fecha_split):
            ok.append(tk.upper())

    if args.apply and ok and not args.no_derivadas:
        with engine.connect() as conn:
            desde = conn.execute(text("""
                SELECT MIN(fecha) FROM precios_diarios WHERE ticker = ANY(:tk)
            """), {"tk": ok}).scalar()
        recomputar_derivadas_bulk(desde)

    log("Listo." if args.apply else "Dry run terminado (no se escribio nada).")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Deteccion y correccion de splits.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detectar", help="busca saltos tipo split en precios_diarios")
    d.add_argument("--umbral", type=float, default=0.25,
                   help="variacion diaria minima (default 0.25 = 25%%)")
    d.add_argument("--desde", help="fecha inicial del barrido (YYYY-MM-DD)")
    d.add_argument("--verificar", action="store_true",
                   help="etapa 2: confirma los candidatos contra Yahoo (usa red)")
    d.add_argument("--max-verificar", type=int, default=10,
                   help="tope de tickers a verificar contra Yahoo (default 10)")
    d.set_defaults(func=cmd_detectar)

    ch = sub.add_parser("chequeo", help="chequeo desatendido para el pipeline diario")
    ch.add_argument("--dias", type=int, default=7,
                    help="ventana de barrido en dias corridos (default 7)")
    ch.add_argument("--umbral", type=float, default=0.25)
    ch.add_argument("--sin-alerta", action="store_true",
                    help="no enviar Telegram (solo imprime)")
    ch.set_defaults(func=cmd_chequeo)

    c = sub.add_parser("corregir", help="re-baja el historial ajustado y recomputa")
    c.add_argument("tickers", nargs="+")
    c.add_argument("--apply", action="store_true", help="escribe (default: dry run)")
    c.add_argument("--dry-run", action="store_true", help="explicito, es el default")
    c.add_argument("--no-derivadas", action="store_true",
                   help="no recomputar indicadores/features/z-scores")
    c.add_argument("--ratio", type=float,
                   help="forzar el ratio del split (default: se verifica en Yahoo)")
    c.add_argument("--fecha-split", dest="fecha_split",
                   help="forzar la fecha del split YYYY-MM-DD (default: Yahoo)")
    c.set_defaults(func=cmd_corregir)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
