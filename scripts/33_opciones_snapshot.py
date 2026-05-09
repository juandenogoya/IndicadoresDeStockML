"""
33_opciones_snapshot.py
Recolector diario de opciones sobre acciones (yfinance).

Graba dos tablas por cada ejecucion:
    1. opciones_snapshot   — nivel contrato (~52k filas/dia)
       Vencimientos <= MAX_DTE dias, vol > 0 o OI >= MIN_OI.
       Uso: ML training, analisis de strikes, concentracion de OI.
       Retencion: 2 anos (contratos vencidos = labeled training data).

    2. opciones_resumen_diario — nivel ticker (124 filas/dia)
       Agregados: PCR_vol, PCR_oi, IV promedio ponderado por OI,
       strike con max OI combinado, n contratos activos.
       Uso: dashboards, queries PCR rapidas, series de tiempo de sentimiento.
       Retencion: indefinida (datos ya comprimidos, tamano trivial).

Ambas tablas se construyen de los mismos datos descargados de yfinance;
opciones_resumen_diario no genera ninguna llamada HTTP adicional.

Parametros configurables via .env:
    OPCIONES_MAX_DTE=90    vencimientos <= N dias desde hoy
    OPCIONES_MIN_OI=5      OI minimo para incluir contrato sin volumen del dia

Uso:
    python scripts/33_opciones_snapshot.py            # todos los tickers
    python scripts/33_opciones_snapshot.py --init     # crea las 2 tablas en DB
    python scripts/33_opciones_snapshot.py --ticker AAPL MSFT
    python scripts/33_opciones_snapshot.py --dry-run           # muestra sin guardar
    python scripts/33_opciones_snapshot.py --status            # muestra resumen de DB
    python scripts/33_opciones_snapshot.py --backfill-resumen  # recalcula resumen de fechas ya en snapshot
    python scripts/33_opciones_snapshot.py --validate          # check intraday calidad + Telegram
"""

import sys
import os
import argparse
import math
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

import yfinance as yf
import psycopg2.extras
from sqlalchemy import text
from src.data.database import get_engine, get_connection
from src.utils.config import ALL_TICKERS


# ── Parametros configurables ──────────────────────────────────────────────────

# Vencimientos dentro de los proximos N dias calendario.
# 90 dias captura weeklies + monthlies del trimestre.
MAX_DTE = int(os.getenv("OPCIONES_MAX_DTE", "90"))

# OI minimo para incluir un contrato que NO opero hoy (vol=0).
# Contratos con OI=1..4 son posiciones abandonadas / noise: no aportan
# informacion de sentimiento y saturan la tabla.
# vol > 0  siempre se incluye (se opero hoy, independiente del OI).
MIN_OI = int(os.getenv("OPCIONES_MIN_OI", "5"))


# ── DDL ───────────────────────────────────────────────────────────────────────

DDL_SNAPSHOT = """
CREATE TABLE IF NOT EXISTS opciones_snapshot (
    id                SERIAL        PRIMARY KEY,
    fecha_snapshot    DATE          NOT NULL,
    ticker            VARCHAR(20)   NOT NULL,
    vencimiento       DATE          NOT NULL,
    tipo              VARCHAR(4)    NOT NULL,     -- 'call' | 'put'
    strike            NUMERIC(12,4) NOT NULL,
    volumen           INTEGER,
    open_interest     INTEGER,
    iv                NUMERIC(10,6),              -- volatilidad implicita (decimal: 0.25 = 25%)
    bid               NUMERIC(12,4),
    ask               NUMERIC(12,4),
    precio_subyacente NUMERIC(12,4),
    hv_20d            NUMERIC(10,6),
    created_at        TIMESTAMP     DEFAULT NOW(),
    UNIQUE (fecha_snapshot, ticker, vencimiento, tipo, strike)
);

CREATE INDEX IF NOT EXISTS idx_opciones_fecha       ON opciones_snapshot (fecha_snapshot);
CREATE INDEX IF NOT EXISTS idx_opciones_ticker      ON opciones_snapshot (ticker);
CREATE INDEX IF NOT EXISTS idx_opciones_vencimiento ON opciones_snapshot (vencimiento);
CREATE INDEX IF NOT EXISTS idx_opciones_tipo        ON opciones_snapshot (tipo);
"""

DDL_RESUMEN = """
CREATE TABLE IF NOT EXISTS opciones_resumen_diario (
    id            SERIAL        PRIMARY KEY,
    fecha         DATE          NOT NULL,
    ticker        VARCHAR(20)   NOT NULL,

    -- Volumen del dia (resets diario)
    call_vol      BIGINT,
    put_vol       BIGINT,
    pcr_vol       NUMERIC(8,4),   -- put_vol / call_vol  (NULL si call_vol = 0)

    -- Open Interest acumulado
    call_oi       BIGINT,
    put_oi        BIGINT,
    pcr_oi        NUMERIC(8,4),   -- put_oi / call_oi   (NULL si call_oi = 0)

    -- IV promedio ponderado por OI
    iv_call_avg   NUMERIC(8,6),
    iv_put_avg    NUMERIC(8,6),

    -- Concentracion
    n_contratos   INTEGER,        -- total contratos activos en el snapshot
    max_oi_strike NUMERIC(12,4),  -- strike con mayor OI combinado call+put
    max_oi_venc   DATE,           -- vencimiento del strike con mayor OI

    -- Contexto
    precio_sub    NUMERIC(12,4),  -- ultimo close en precios_diarios

    created_at    TIMESTAMP     DEFAULT NOW(),
    UNIQUE (fecha, ticker)
);

CREATE INDEX IF NOT EXISTS idx_resumen_fecha  ON opciones_resumen_diario (fecha);
CREATE INDEX IF NOT EXISTS idx_resumen_ticker ON opciones_resumen_diario (ticker);
"""


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_tabla():
    """Crea opciones_snapshot y opciones_resumen_diario si no existen."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(DDL_SNAPSHOT))
        conn.execute(text(DDL_RESUMEN))
        conn.commit()
    log("Tablas opciones_snapshot y opciones_resumen_diario listas.")


# ── Precios y HV desde DB ─────────────────────────────────────────────────────

def _get_precios_subyacentes(tickers: list[str]) -> dict[str, float]:
    """Ultimo close disponible en precios_diarios por ticker."""
    engine = get_engine()
    placeholders = ", ".join(f"'{t}'" for t in tickers)
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT ON (ticker) ticker, close
            FROM   precios_diarios
            WHERE  ticker IN ({placeholders})
              AND  close > 0
            ORDER  BY ticker, fecha DESC
        """)).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def _get_hv_20d(tickers: list[str]) -> dict[str, float]:
    """
    Volatilidad historica anualizada a 20 dias para cada ticker.
    HV = std(log_returns_20d) * sqrt(252).  Retorna decimal (0.25 = 25%).
    """
    engine = get_engine()
    placeholders = ", ".join(f"'{t}'" for t in tickers)
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT ticker, close
            FROM   precios_diarios
            WHERE  ticker IN ({placeholders})
              AND  close  > 0
              AND  fecha  >= CURRENT_DATE - INTERVAL '35 days'
            ORDER  BY ticker, fecha ASC
        """)).fetchall()

    closes_by_ticker: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        closes_by_ticker[r[0]].append(float(r[1]))

    hv_map: dict[str, float] = {}
    for ticker, closes in closes_by_ticker.items():
        if len(closes) < 5:
            continue
        c = closes[-21:]
        log_ret = [math.log(c[i] / c[i - 1]) for i in range(1, len(c))]
        n    = len(log_ret)
        mean = sum(log_ret) / n
        var  = sum((r - mean) ** 2 for r in log_ret) / (n - 1) if n > 1 else 0.0
        hv_map[ticker] = round(math.sqrt(var) * math.sqrt(252), 6)

    return hv_map


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    f = _safe_float(val)
    return None if f is None else int(f)


# ── Colector por ticker ───────────────────────────────────────────────────────

def recolectar_ticker(
    ticker: str,
    fecha_snapshot: date,
    precio_subyacente: Optional[float],
    hv_20d: Optional[float],
    max_dte: int = None,
    min_oi: int = None,
) -> list[dict]:
    """
    Descarga la chain de opciones para un ticker.

    Filtros aplicados:
        - Vencimientos <= max_dte dias desde fecha_snapshot (default MAX_DTE)
        - Se incluye el contrato si:  vol > 0   (se opero hoy)
                                  OR  OI >= min_oi  (posicion relevante abierta)
          Contratos con vol=0 y OI < MIN_OI son noise (posiciones fantasma).
    """
    from datetime import timedelta
    _max_dte = max_dte or MAX_DTE
    _min_oi  = min_oi  if min_oi is not None else MIN_OI
    limite_vencimiento = fecha_snapshot + timedelta(days=_max_dte)

    try:
        yft  = yf.Ticker(ticker)
        exps = yft.options
        if not exps:
            return []
    except Exception:
        return []

    exps = [e for e in exps if date.fromisoformat(e) <= limite_vencimiento]
    if not exps:
        return []

    filas = []

    for exp_str in exps:
        try:
            exp_date = date.fromisoformat(exp_str)
            chain    = yft.option_chain(exp_str)
        except Exception:
            continue

        time.sleep(0.5)

        for tipo, df in [("call", chain.calls), ("put", chain.puts)]:
            if df is None or df.empty:
                continue

            for _, opt in df.iterrows():
                vol = _safe_int(opt.get("volume"))
                oi  = _safe_int(opt.get("openInterest"))

                # Incluir si: opero hoy  O  posicion >= MIN_OI
                active_today   = vol is not None and vol > 0
                relevant_oi    = oi  is not None and oi  >= _min_oi
                if not active_today and not relevant_oi:
                    continue

                filas.append({
                    "fecha_snapshot":    fecha_snapshot,
                    "ticker":            ticker,
                    "vencimiento":       exp_date,
                    "tipo":              tipo,
                    "strike":            float(opt["strike"]),
                    "volumen":           vol,
                    "open_interest":     oi,
                    "iv":                _safe_float(opt.get("impliedVolatility")),
                    "bid":               _safe_float(opt.get("bid")),
                    "ask":               _safe_float(opt.get("ask")),
                    "precio_subyacente": precio_subyacente,
                    "hv_20d":            hv_20d,
                })

    return filas


# ── Resumen diario (agrega en memoria, sin calls adicionales) ─────────────────

def _computar_resumen(
    ticker: str,
    fecha: date,
    filas: list[dict],
    precio_subyacente: Optional[float],
) -> dict:
    """
    Agrega las filas de un ticker en un unico registro de resumen.
    Se llama con los datos ya descargados — cero llamadas HTTP extra.
    """
    calls = [f for f in filas if f["tipo"] == "call"]
    puts  = [f for f in filas if f["tipo"] == "put"]

    call_vol = sum(f["volumen"]        or 0 for f in calls)
    put_vol  = sum(f["volumen"]        or 0 for f in puts)
    call_oi  = sum(f["open_interest"]  or 0 for f in calls)
    put_oi   = sum(f["open_interest"]  or 0 for f in puts)

    pcr_vol = round(put_vol / call_vol, 4) if call_vol > 0 else None
    pcr_oi  = round(put_oi  / call_oi,  4) if call_oi  > 0 else None

    def _iv_avg(subset: list[dict]) -> Optional[float]:
        """IV promedio ponderado por OI."""
        total_oi = sum(
            (f["open_interest"] or 0)
            for f in subset if f["iv"] is not None
        )
        if total_oi == 0:
            return None
        weighted = sum(
            (f["iv"] or 0.0) * (f["open_interest"] or 0)
            for f in subset if f["iv"] is not None
        )
        return round(weighted / total_oi, 6)

    # Strike con mayor OI combinado (call + put) por (strike, vencimiento)
    oi_by_key: dict[tuple, int] = defaultdict(int)
    for f in filas:
        oi_by_key[(f["strike"], f["vencimiento"])] += (f["open_interest"] or 0)

    if oi_by_key:
        top_key = max(oi_by_key, key=lambda k: oi_by_key[k])
        max_oi_strike = top_key[0]
        max_oi_venc   = top_key[1]
    else:
        max_oi_strike = None
        max_oi_venc   = None

    return {
        "fecha":        fecha,
        "ticker":       ticker,
        "call_vol":     call_vol,
        "put_vol":      put_vol,
        "pcr_vol":      pcr_vol,
        "call_oi":      call_oi,
        "put_oi":       put_oi,
        "pcr_oi":       pcr_oi,
        "iv_call_avg":  _iv_avg(calls),
        "iv_put_avg":   _iv_avg(puts),
        "n_contratos":  len(filas),
        "max_oi_strike": max_oi_strike,
        "max_oi_venc":  max_oi_venc,
        "precio_sub":   precio_subyacente,
    }


# ── Persistencia ──────────────────────────────────────────────────────────────

def persistir_filas(filas: list[dict]) -> int:
    """
    Upsert bulk de opciones_snapshot.
    ON CONFLICT DO NOTHING -> idempotente (rerun seguro).
    """
    if not filas:
        return 0

    SQL = """
        INSERT INTO opciones_snapshot (
            fecha_snapshot, ticker, vencimiento, tipo, strike,
            volumen, open_interest, iv, bid, ask,
            precio_subyacente, hv_20d
        ) VALUES %s
        ON CONFLICT (fecha_snapshot, ticker, vencimiento, tipo, strike)
        DO NOTHING
    """
    valores = [
        (
            f["fecha_snapshot"], f["ticker"],  f["vencimiento"], f["tipo"],  f["strike"],
            f["volumen"],        f["open_interest"], f["iv"],   f["bid"],    f["ask"],
            f["precio_subyacente"], f["hv_20d"],
        )
        for f in filas
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, SQL, valores, page_size=500)

    return len(filas)


def persistir_resumenes(resumenes: list[dict]) -> int:
    """
    Upsert bulk de opciones_resumen_diario.
    ON CONFLICT actualiza todos los campos (los agregados pueden variar en rerun
    si el ticker se proceso parcialmente en una corrida anterior).
    """
    if not resumenes:
        return 0

    SQL = """
        INSERT INTO opciones_resumen_diario (
            fecha, ticker,
            call_vol, put_vol, pcr_vol,
            call_oi,  put_oi,  pcr_oi,
            iv_call_avg, iv_put_avg,
            n_contratos, max_oi_strike, max_oi_venc,
            precio_sub
        ) VALUES %s
        ON CONFLICT (fecha, ticker) DO UPDATE SET
            call_vol      = EXCLUDED.call_vol,
            put_vol       = EXCLUDED.put_vol,
            pcr_vol       = EXCLUDED.pcr_vol,
            call_oi       = EXCLUDED.call_oi,
            put_oi        = EXCLUDED.put_oi,
            pcr_oi        = EXCLUDED.pcr_oi,
            iv_call_avg   = EXCLUDED.iv_call_avg,
            iv_put_avg    = EXCLUDED.iv_put_avg,
            n_contratos   = EXCLUDED.n_contratos,
            max_oi_strike = EXCLUDED.max_oi_strike,
            max_oi_venc   = EXCLUDED.max_oi_venc,
            precio_sub    = EXCLUDED.precio_sub
    """
    valores = [
        (
            r["fecha"],      r["ticker"],
            r["call_vol"],   r["put_vol"],  r["pcr_vol"],
            r["call_oi"],    r["put_oi"],   r["pcr_oi"],
            r["iv_call_avg"], r["iv_put_avg"],
            r["n_contratos"], r["max_oi_strike"], r["max_oi_venc"],
            r["precio_sub"],
        )
        for r in resumenes
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, SQL, valores, page_size=200)

    return len(resumenes)


# ── Runner ────────────────────────────────────────────────────────────────────

def _check_datos_existentes(fecha: date) -> int:
    """Retorna cuantas filas hay en opciones_snapshot para la fecha dada."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            return conn.execute(
                text("SELECT COUNT(*) FROM opciones_snapshot WHERE fecha_snapshot = :f"),
                {"f": fecha}
            ).scalar() or 0
    except Exception:
        return 0


def cmd_run(tickers: list[str], dry_run: bool = False,
            fecha_override: Optional[date] = None, intento: int = 1):
    fecha_hoy      = fecha_override or date.today()
    _skip_telegram = os.getenv("OPCIONES_SKIP_TELEGRAM", "0") == "1"

    log("=" * 60)
    log(f"  OPCIONES SNAPSHOT  |  {fecha_hoy}")
    if fecha_override:
        log(f"  FECHA OVERRIDE    : {fecha_override} (backfill manual)")
    log(f"  Tickers   : {len(tickers)}")
    log(f"  MAX_DTE   : {MAX_DTE} dias")
    log(f"  MIN_OI    : {MIN_OI} (umbral sin volumen)")
    log(f"  Modo      : {'DRY RUN' if dry_run else 'REAL'}")
    log(f"  Intento   : {intento}/3")
    log("=" * 60)

    # ── Check previo: datos ya existentes ────────────────────────────────────
    if not dry_run:
        filas_existentes = _check_datos_existentes(fecha_hoy)
        if filas_existentes >= _MIN_FILAS:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
            log(f"  Datos ya disponibles para {fecha_hoy}: {filas_existentes:,} filas — skip.")
            if not _skip_telegram:
                try:
                    from src.pipeline.telegram_notifier import _send as _tg_send
                    nl = "\n"
                    _tg_send(
                        f"INFO Opciones snapshot -- datos ya disponibles{nl}"
                        f"<i>{fecha_hoy} | {ts}</i>{nl}"
                        f"Intento {intento}/3 omitido: ya existen {filas_existentes:,} contratos en DB."
                    )
                except Exception as tg_err:
                    log(f"  [WARN] Telegram no disponible: {tg_err}")
            return

    log("  Cargando precios subyacentes y HV_20d desde DB...")
    precios = _get_precios_subyacentes(tickers)
    hvs     = _get_hv_20d(tickers)
    log(f"  Precios  : {len(precios)} tickers")
    log(f"  HV_20d   : {len(hvs)} tickers")

    total_filas  = 0
    sin_opciones = 0
    errores      = 0
    resumenes    = []

    for i, ticker in enumerate(tickers, 1):
        precio = precios.get(ticker)
        hv     = hvs.get(ticker)

        try:
            filas = recolectar_ticker(ticker, fecha_hoy, precio, hv)

            if not filas:
                log(f"  [{i:3d}/{len(tickers)}] {ticker:<8s}  sin opciones activas")
                sin_opciones += 1
                time.sleep(0.2)
                continue

            # Agregar resumen en memoria (sin calls adicionales)
            resumen = _computar_resumen(ticker, fecha_hoy, filas, precio)
            resumenes.append(resumen)

            if dry_run:
                pcr_v = f"{resumen['pcr_vol']:.2f}" if resumen["pcr_vol"] else "N/A"
                log(f"  [{i:3d}/{len(tickers)}] {ticker:<8s}  "
                    f"{len(filas):5d} filas  PCR_vol={pcr_v}  [DRY RUN]")
            else:
                n = persistir_filas(filas)
                pcr_v = f"{resumen['pcr_vol']:.2f}" if resumen["pcr_vol"] else "N/A"
                log(f"  [{i:3d}/{len(tickers)}] {ticker:<8s}  "
                    f"{n:5d} filas  PCR_vol={pcr_v}")
                total_filas += n

            time.sleep(0.3)

        except Exception as e:
            log(f"  [{i:3d}/{len(tickers)}] {ticker:<8s}  ERROR: {e}")
            errores += 1

    # Persistir resumenes de todos los tickers en un solo bulk
    if not dry_run and resumenes:
        n_res = persistir_resumenes(resumenes)
        log(f"")
        log(f"  Resumenes diarios: {n_res} tickers -> opciones_resumen_diario")

        # Calcular Z-scores de opciones para la fecha del snapshot
        try:
            from src.utils.zscore_pipeline import calcular_zscore_opciones, init_tablas
            engine = get_engine()
            init_tablas(engine)   # crea tabla si no existe (idempotente)
            n_z = calcular_zscore_opciones(fecha_hoy, engine)
            log(f"  Z-scores opciones: {n_z} tickers -> opciones_zscore_diario")
        except Exception as z_err:
            log(f"  [WARN] Z-score opciones no calculado: {z_err}")

    log("")
    log(f"  Filas snapshot   : {total_filas:,}")
    log(f"  Sin opciones     : {sin_opciones}")
    log(f"  Errores          : {errores}")
    log("  Completado.")

    if not dry_run:
        _post_run_check(fecha_hoy, total_filas, len(tickers), errores, sin_opciones, intento=intento)


# ── Chequeo post-escritura EOD ────────────────────────────────────────────────

# Umbrales
_MIN_FILAS       = 10_000   # con 199 tickers esperamos ~85k; < 10k es critico
_MAX_ERRORES     = 30       # > 30 tickers con error = problema sistemico (~15%)
_MAX_SIN_OPC     = 60       # > 60 tickers sin contratos = sospechoso (~30%)


def _post_run_check(fecha, total_filas, n_tickers, errores, sin_opciones, intento: int = 1):
    """
    Verifica la calidad del snapshot recien escrito en DB.
    - Envia Telegram siempre (OK, redundante o ISSUES), salvo modo manual.
    - Si hay issues criticos: exit(1) -> GH Actions marca FAILED.
    """
    import sys as _sys
    _skip_telegram = os.getenv("OPCIONES_SKIP_TELEGRAM", "0") == "1"

    def _tg(msg):
        if _skip_telegram:
            return
        try:
            from src.pipeline.telegram_notifier import _send as _tg_send
            _tg_send(msg)
        except Exception as tg_err:
            log(f"  [WARN] Telegram no disponible: {tg_err}")

    # Deteccion de dia no habil NYSE
    from datetime import date as _date
    from src.utils.trading_calendar import is_trading_day, describe_date
    fecha_obj = fecha if isinstance(fecha, _date) else _date.fromisoformat(str(fecha))
    if total_filas == 0 and errores == 0 and sin_opciones >= n_tickers * 0.95:
        if not is_trading_day(fecha_obj):
            ts   = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
            desc = describe_date(fecha_obj)
            log(f"  [POST-CHECK] {desc} -- mercado cerrado, sin datos esperado.")
            nl = "\n"
            _tg(f"INFO Snapshot EOD -- dia no habil{nl}<i>{ts}</i>{nl}Fecha: {desc}{nl}Sin opciones. Comportamiento normal.")
            return

    # Deteccion de run redundante / rate limit
    if total_filas == 0 and errores == 0:
        try:
            engine = get_engine()
            with engine.connect() as conn:
                filas_db = conn.execute(
                    text("SELECT COUNT(*) FROM opciones_snapshot WHERE fecha_snapshot = :f"),
                    {"f": fecha}
                ).scalar() or 0
        except Exception:
            filas_db = 0

        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        nl = "\n"
        if filas_db >= _MIN_FILAS:
            log(f"  [POST-CHECK] Datos ya existentes: {filas_db:,} filas para {fecha}.")
            _tg(f"INFO Opciones snapshot -- datos ya disponibles{nl}<i>{fecha} | {ts}</i>{nl}Intento {intento}/3: ya existen {filas_db:,} contratos en DB.")
            return

        # Rate limit de yfinance
        _proximos = {1: "02:00 UTC", 2: "06:00 UTC", 3: None}
        proximo   = _proximos.get(intento)
        log(f"  [POST-CHECK] Rate limit yfinance (intento {intento}/3). 0 filas escritas.")
        if proximo:
            _tg(f"WARN Opciones snapshot -- sin datos (intento {intento}/3){nl}<i>{fecha} | {ts}</i>{nl}yfinance sin contratos (rate limit).{nl}Proximo intento: <b>{proximo}</b>")
        else:
            _tg(f"ERROR Opciones snapshot -- todos los intentos fallaron{nl}<i>{fecha} | {ts}</i>{nl}3 intentos sin datos (23:00 / 02:00 / 11:00 UTC).{nl}Requiere revision manual.")
        if intento < 3:
            return

    issues   = []
    warnings = []

    if total_filas == 0:
        issues.append("0 filas escritas en opciones_snapshot (posible fallo yfinance o DB)")
    elif total_filas < _MIN_FILAS:
        issues.append(f"Solo {total_filas:,} filas (minimo {_MIN_FILAS:,} con {n_tickers} tickers)")

    if errores > _MAX_ERRORES:
        issues.append(f"{errores}/{n_tickers} tickers con error ({errores/n_tickers*100:.0f}%)")
    elif errores > 10:
        warnings.append(f"{errores}/{n_tickers} tickers con error")

    if sin_opciones > _MAX_SIN_OPC:
        warnings.append(f"{sin_opciones}/{n_tickers} sin contratos ({sin_opciones/n_tickers*100:.0f}%)")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    nl = "\n"
    if issues:
        iss_str = nl.join(f"  - {x}" for x in issues)
        _tg(f"ERROR Snapshot EOD FALLIDO{nl}<i>{fecha} | {ts}</i>{nl}{nl}<b>Issues:</b>{nl}{iss_str}{nl}Revisar GH Actions.")
    elif warnings:
        warn_str = nl.join(f"  - {x}" for x in warnings)
        _tg(f"WARN Snapshot EOD con advertencias{nl}<i>{fecha} | {ts}</i>{nl}{total_filas:,} filas | {n_tickers} tickers{nl}<b>Advertencias:</b>{nl}{warn_str}")
    else:
        _tg(f"OK Snapshot EOD OK{nl}<i>{fecha} | {ts}</i>{nl}{total_filas:,} filas | {n_tickers} tickers | {errores} errores")

    if issues:
        log("")
        log("  [POST-CHECK] ISSUES CRITICOS:")
        for x in issues:
            log(f"    - {x}")
        _sys.exit(1)



def cmd_backfill_resumen():
    """
    Recalcula opciones_resumen_diario para todas las fechas presentes en
    opciones_snapshot que aun no tienen entrada en resumen.
    Util para: (1) primera vez despues de --init, (2) recuperar dias que
    fallaron, (3) repoblar tras cambio de logica de calculo.

    No hace llamadas a yfinance: lee exclusivamente desde opciones_snapshot.
    """
    engine = get_engine()

    # Fechas en snapshot que no tienen resumen todavia
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT fecha_snapshot
            FROM   opciones_snapshot
            WHERE  fecha_snapshot NOT IN (
                SELECT DISTINCT fecha FROM opciones_resumen_diario
            )
            ORDER  BY fecha_snapshot
        """)).fetchall()

    fechas_pendientes = [r[0] for r in rows]

    if not fechas_pendientes:
        log("  opciones_resumen_diario ya esta al dia. Sin fechas pendientes.")
        return

    log(f"  Fechas pendientes: {[str(f) for f in fechas_pendientes]}")

    # Para cada fecha pendiente calculamos el resumen via SQL agregado
    # (evita traer 52k filas a Python — todo en la DB)
    SQL_BACKFILL = """
        WITH oi_per_key AS (
            SELECT fecha_snapshot, ticker, strike, vencimiento,
                   SUM(COALESCE(open_interest, 0)) AS total_oi
            FROM   opciones_snapshot
            WHERE  fecha_snapshot = :fecha
            GROUP  BY fecha_snapshot, ticker, strike, vencimiento
        ),
        top_strike AS (
            SELECT DISTINCT ON (ticker)
                   ticker,
                   strike      AS max_oi_strike,
                   vencimiento AS max_oi_venc
            FROM   oi_per_key
            ORDER  BY ticker, total_oi DESC
        ),
        agg AS (
            SELECT
                fecha_snapshot AS fecha,
                ticker,
                SUM(CASE WHEN tipo='call' THEN COALESCE(volumen,0) ELSE 0 END)        AS call_vol,
                SUM(CASE WHEN tipo='put'  THEN COALESCE(volumen,0) ELSE 0 END)        AS put_vol,
                SUM(CASE WHEN tipo='call' THEN COALESCE(open_interest,0) ELSE 0 END)  AS call_oi,
                SUM(CASE WHEN tipo='put'  THEN COALESCE(open_interest,0) ELSE 0 END)  AS put_oi,
                -- IV call ponderada por OI
                CASE
                    WHEN SUM(CASE WHEN tipo='call' AND iv IS NOT NULL
                             THEN COALESCE(open_interest,0) ELSE 0 END) > 0
                    THEN ROUND(
                        SUM(CASE WHEN tipo='call' AND iv IS NOT NULL
                            THEN iv * COALESCE(open_interest,0) ELSE 0 END)
                        / SUM(CASE WHEN tipo='call' AND iv IS NOT NULL
                              THEN COALESCE(open_interest,0) ELSE 0 END)::NUMERIC, 6)
                    ELSE NULL
                END AS iv_call_avg,
                -- IV put ponderada por OI
                CASE
                    WHEN SUM(CASE WHEN tipo='put' AND iv IS NOT NULL
                             THEN COALESCE(open_interest,0) ELSE 0 END) > 0
                    THEN ROUND(
                        SUM(CASE WHEN tipo='put' AND iv IS NOT NULL
                            THEN iv * COALESCE(open_interest,0) ELSE 0 END)
                        / SUM(CASE WHEN tipo='put' AND iv IS NOT NULL
                              THEN COALESCE(open_interest,0) ELSE 0 END)::NUMERIC, 6)
                    ELSE NULL
                END AS iv_put_avg,
                COUNT(*)                                                               AS n_contratos,
                MAX(precio_subyacente)                                                 AS precio_sub
            FROM opciones_snapshot
            WHERE fecha_snapshot = :fecha
            GROUP BY fecha_snapshot, ticker
        )
        INSERT INTO opciones_resumen_diario (
            fecha, ticker,
            call_vol, put_vol, pcr_vol,
            call_oi,  put_oi,  pcr_oi,
            iv_call_avg, iv_put_avg,
            n_contratos, max_oi_strike, max_oi_venc,
            precio_sub
        )
        SELECT
            a.fecha,
            a.ticker,
            a.call_vol,
            a.put_vol,
            CASE WHEN a.call_vol > 0
                 THEN ROUND(a.put_vol::NUMERIC / a.call_vol, 4) ELSE NULL END,
            a.call_oi,
            a.put_oi,
            CASE WHEN a.call_oi > 0
                 THEN ROUND(a.put_oi::NUMERIC / a.call_oi, 4) ELSE NULL END,
            a.iv_call_avg,
            a.iv_put_avg,
            a.n_contratos,
            t.max_oi_strike,
            t.max_oi_venc,
            a.precio_sub
        FROM       agg     a
        LEFT JOIN  top_strike t USING (ticker)
        ON CONFLICT (fecha, ticker) DO UPDATE SET
            call_vol      = EXCLUDED.call_vol,
            put_vol       = EXCLUDED.put_vol,
            pcr_vol       = EXCLUDED.pcr_vol,
            call_oi       = EXCLUDED.call_oi,
            put_oi        = EXCLUDED.put_oi,
            pcr_oi        = EXCLUDED.pcr_oi,
            iv_call_avg   = EXCLUDED.iv_call_avg,
            iv_put_avg    = EXCLUDED.iv_put_avg,
            n_contratos   = EXCLUDED.n_contratos,
            max_oi_strike = EXCLUDED.max_oi_strike,
            max_oi_venc   = EXCLUDED.max_oi_venc,
            precio_sub    = EXCLUDED.precio_sub
    """

    total = 0
    for fecha in fechas_pendientes:
        with engine.connect() as conn:
            result = conn.execute(text(SQL_BACKFILL), {"fecha": fecha})
            conn.commit()
            n = result.rowcount
        log(f"  {fecha}  ->  {n} tickers insertados")
        total += n

    log(f"  Backfill completado: {total} filas en opciones_resumen_diario")


def cmd_status():
    """Muestra metricas rapidas de ambas tablas."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            # opciones_snapshot
            snap = conn.execute(text("""
                SELECT MIN(fecha_snapshot) AS desde,
                       MAX(fecha_snapshot) AS hasta,
                       COUNT(*)            AS total_filas,
                       COUNT(DISTINCT fecha_snapshot) AS dias
                FROM opciones_snapshot
            """)).fetchone()

            # opciones_resumen_diario — ultimos 5 dias, top PCR_oi
            resumen_rows = conn.execute(text("""
                SELECT fecha, ticker, pcr_vol, pcr_oi, call_oi, put_oi, n_contratos
                FROM   opciones_resumen_diario
                WHERE  fecha = (SELECT MAX(fecha) FROM opciones_resumen_diario)
                ORDER  BY pcr_oi DESC NULLS LAST
                LIMIT  10
            """)).fetchall()

        if snap and snap[2]:
            print()
            print(f"  opciones_snapshot:")
            print(f"    Rango   : {snap[0]} -> {snap[1]}")
            print(f"    Dias    : {snap[3]}")
            print(f"    Total   : {snap[2]:,} filas")
            print()

        if resumen_rows:
            fecha_res = resumen_rows[0][0]
            print(f"  opciones_resumen_diario | {fecha_res}  (top 10 por PCR_oi):")
            print(f"  {'TICKER':<8s}  {'PCR_vol':>7s}  {'PCR_oi':>7s}  "
                  f"{'CALL_OI':>10s}  {'PUT_OI':>10s}  {'N':>6s}")
            print("  " + "-" * 58)
            for r in resumen_rows:
                pcr_v = f"{float(r[2]):.2f}" if r[2] else " N/A "
                pcr_o = f"{float(r[3]):.2f}" if r[3] else " N/A "
                print(f"  {r[1]:<8s}  {pcr_v:>7s}  {pcr_o:>7s}  "
                      f"{(r[4] or 0):>10,d}  {(r[5] or 0):>10,d}  {(r[6] or 0):>6d}")
        else:
            print("  Sin datos en opciones_resumen_diario.")
    except Exception as e:
        print(f"ERROR en cmd_status: {e}")


# ── Validacion intraday ──────────────────────────────────────────────────────

# Tickers representativos para validacion: mega cap + diversidad sectorial
VALIDATE_TICKERS = ["NVDA", "AAPL", "TSLA", "MSFT", "SPY", "BAC", "XOM", "TSM"]

def cmd_validate():
    """
    Valida calidad de datos de yfinance a mitad de jornada.

    Checks:
      1. Precio actual yfinance vs ultimo close en DB (detecta precios stale)
      2. Contratos activos por ticker (detecta si yfinance no devuelve datos)
      3. Cobertura IV (% contratos con IV plausible 5%-200%)
      4. PCR vol en rango razonable

    Exit code 0: todos los checks OK o solo warnings
    Exit code 1: uno o mas checks CRITICOS fallaron  -> GH Actions marca FAILED
    """
    import sys as _sys

    fecha_hoy = date.today()
    ts = datetime.now().strftime("%H:%M UTC")
    tickers = VALIDATE_TICKERS

    log("=" * 60)
    log(f"  VALIDACION INTRADAY  |  {fecha_hoy}  |  {ts}")
    log(f"  Tickers: {', '.join(tickers)}")
    log("=" * 60)

    precios_db = _get_precios_subyacentes(tickers)
    hvs        = _get_hv_20d(tickers)

    # ── Check 1: precio yfinance vs DB ────────────────────────────────────────
    log("\n  [CHECK 1] Precio actual yfinance vs ultimo close en DB")
    precio_errors = []
    for ticker in tickers:
        try:
            fi = yf.Ticker(ticker).fast_info
            precio_yf = getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None)
            precio_db = precios_db.get(ticker)
            if precio_yf and precio_db and precio_db > 0:
                diff = abs(float(precio_yf) - precio_db) / precio_db * 100
                if diff > 15:
                    estado = "ERROR (posible stale)"
                    precio_errors.append(ticker)
                elif diff > 5:
                    estado = "WARN"
                else:
                    estado = "OK"
                log(f"    {ticker:<6s}  yf=${float(precio_yf):8.2f}  "
                    f"db=${precio_db:8.2f}  diff={diff:5.1f}%  [{estado}]")
            else:
                log(f"    {ticker:<6s}  sin precio disponible [SKIP]")
        except Exception as e:
            log(f"    {ticker:<6s}  ERROR: {e}")

    # ── Check 2+3+4: opciones dry-run ────────────────────────────────────────
    log("\n  [CHECK 2-4] Contratos / IV coverage / PCR  (DRY RUN)")
    option_errors = []

    # Pre-check: verificar que yfinance sea accesible antes de iterar.
    # Rate limiting de GH Actions IPs es transitorio y no indica problema
    # de calidad de datos — el snapshot EOD corre a 01:00 UTC con mucho
    # menos trafico. Si hay rate-limit aqui, saltamos Check 2-4 con WARN.
    # Nota: yfinance puede retornar [] sin lanzar excepcion ante rate-limit;
    # por eso tambien verificamos que el ticker de referencia (NVDA) tenga
    # al menos un vencimiento disponible — si no, es señal de bloqueo.
    _yf_accessible = True
    _yf_error_msg  = ""
    try:
        _test_exps = yf.Ticker(tickers[0]).options
        if not _test_exps:
            _yf_accessible = False
            _yf_error_msg  = f"{tickers[0]} sin vencimientos (posible rate limit)"
    except Exception as _conn_err:
        _yf_accessible = False
        _yf_error_msg  = str(_conn_err)

    if not _yf_accessible:
        log(f"  yfinance no accesible: {_yf_error_msg}")
        log("  Check 2-4 omitido [WARN] — conectividad transitoria, no es problema de datos")

    if _yf_accessible:
        for ticker in tickers:
            precio = precios_db.get(ticker)
            hv     = hvs.get(ticker)
            try:
                filas = recolectar_ticker(ticker, fecha_hoy, precio, hv)
            except Exception as e:
                log(f"    {ticker:<6s}  ERROR recolectando: {e}")
                option_errors.append(f"{ticker}:error")
                time.sleep(3)
                continue

            if not filas:
                log(f"    {ticker:<6s}  0 contratos  [ERROR]")
                option_errors.append(f"{ticker}:0contratos")
                time.sleep(3)
                continue

            n = len(filas)
            iv_ok  = sum(1 for f in filas
                         if f.get("iv") and 0.05 <= f["iv"] <= 2.0)
            iv_cov = iv_ok / n * 100

            cvol = sum((f.get("volumen") or 0) for f in filas if f["tipo"] == "call")
            pvol = sum((f.get("volumen") or 0) for f in filas if f["tipo"] == "put")
            pcr  = pvol / cvol if cvol > 0 else None

            n_tag   = "OK"   if n >= 100   else ("WARN" if n >= 20   else "ERROR")
            iv_tag  = "OK"   if iv_cov >= 70 else ("WARN" if iv_cov >= 30 else "ERROR")
            pcr_tag = "OK"   if pcr and 0.1 <= pcr <= 5.0 else "WARN"
            pcr_str = f"{pcr:.2f}" if pcr else "N/A"

            if n_tag == "ERROR":
                option_errors.append(f"{ticker}:n={n}")
            if iv_tag == "ERROR":
                option_errors.append(f"{ticker}:iv_cov={iv_cov:.0f}%")

            log(f"    {ticker:<6s}  n={n:5d}[{n_tag}]  "
                f"IV_cov={iv_cov:4.0f}%[{iv_tag}]  PCR={pcr_str}[{pcr_tag}]")

            time.sleep(2)  # anti-rate-limit entre tickers

    # ── Resumen ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    critical = precio_errors + option_errors

    # ── Telegram ──────────────────────────────────────────────────────────────
    try:
        from src.pipeline.telegram_notifier import _send as _tg_send
        ts_msg = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        n_tickers = len(tickers)
        if not _yf_accessible:
            tg_text = (
                "\u26a0\ufe0f <b>Opciones Validacion — yfinance no accesible</b>\n"
                f"<i>{ts_msg}</i>\n\n"
                f"Motivo: {_yf_error_msg}\n"
                "Rate limit transitorio en GH Actions (1 PM ET = horario pico).\n"
                "Check 2-4 omitido. No indica problema de calidad de datos.\n"
                "El snapshot EOD (01:00 UTC) deberia correr sin inconvenientes."
            )
        elif not critical:
            tg_text = (
                "\u2705 <b>Opciones Validacion OK</b>\n"
                f"<i>{ts_msg}</i>\n"
                f"{n_tickers}/{n_tickers} tickers sin issues criticos. EOD snapshot OK."
            )
        else:
            issues_str = "\n".join(f"  - {iss}" for iss in critical)
            tg_text = (
                "\u26a0\ufe0f <b>Opciones Validacion ISSUES</b>\n"
                f"<i>{ts_msg}</i>\n\n"
                f"<b>Issues criticos ({len(critical)}):</b>\n"
                f"{issues_str}\n\n"
                "Verificar yfinance antes del snapshot EOD (01:00 UTC)."
            )
        _tg_send(tg_text)
    except Exception as tg_err:
        log(f"  [WARN] Telegram no disponible: {tg_err}")

    if not critical:
        if not _yf_accessible:
            log("  RESULTADO: WARN — yfinance rate-limited, Check 2-4 no ejecutado")
        else:
            log("  RESULTADO: OK — datos intraday validos para EOD snapshot")
        _sys.exit(0)
    else:
        log("  RESULTADO: ISSUES CRITICOS DETECTADOS:")
        for iss in critical:
            log(f"    - {iss}")
        log("  Verificar yfinance y DB antes del cierre de mercado.")
        _sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Snapshot diario de opciones (yfinance)")
    parser.add_argument("--init",    action="store_true",
                        help="Crea las tablas opciones_snapshot y opciones_resumen_diario")
    parser.add_argument("--dry-run", action="store_true",
                        help="Descarga datos pero no escribe en DB")
    parser.add_argument("--ticker",  nargs="+",
                        help="Tickers especificos (default: todos del pipeline)")
    parser.add_argument("--status",           action="store_true",
                        help="Muestra estadisticas de ambas tablas")
    parser.add_argument("--backfill-resumen", action="store_true",
                        help="Recalcula resumen para fechas en snapshot sin resumen aun")
    parser.add_argument("--validate", action="store_true",
                        help="Validacion intraday: check calidad datos yfinance (exit 1 si hay errores)")
    parser.add_argument("--fecha",
                        help="Fecha del snapshot YYYY-MM-DD (default: hoy). "
                             "Usar para backfill de dias perdidos.",
                        default=None)
    parser.add_argument("--intento", type=int, default=1, choices=[1, 2, 3],
                        help="Numero de intento del dia (1=23UTC, 2=02UTC, 3=11UTC)")
    args = parser.parse_args()

    if args.init:
        init_tabla()
        return

    if args.status:
        cmd_status()
        return

    if args.backfill_resumen:
        cmd_backfill_resumen()
        return

    if args.validate:
        cmd_validate()
        return

    fecha_override = date.fromisoformat(args.fecha) if args.fecha else None
    tickers = args.ticker if args.ticker else list(ALL_TICKERS)
    cmd_run(tickers, dry_run=args.dry_run, fecha_override=fecha_override, intento=args.intento)


if __name__ == "__main__":
    main()
