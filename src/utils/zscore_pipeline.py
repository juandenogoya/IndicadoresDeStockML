"""
zscore_pipeline.py
Calcula Z-scores diarios de volumen para opciones y acciones.

Tablas producidas:
    opciones_zscore_diario  -- Z-scores de opciones (vol calls/puts, PCR, IV)
    ticker_zscore_diario    -- Z-scores de precios/volumen de acciones

Logica de calculo:
    - Ventana deslizante de hasta 60 dias anteriores a la fecha de calculo
    (excluye el dia actual del historico para no contaminar la media).
    - Con menos de 2 dias de historia: Z-scores = NULL, ventana_dias = 0.
    - vol_relativo = vol_hoy / vol_media_historica.
    - percentil_vol = % de dias historicos con volumen MENOR al dia actual.

Uso diario (llamado desde los scripts de cron):
    from src.utils.zscore_pipeline import calcular_zscore_opciones, calcular_zscore_tickers
    n = calcular_zscore_opciones(fecha, engine)
    n = calcular_zscore_tickers(fecha, engine)

Backfill completo (via migration script):
    from src.utils.zscore_pipeline import backfill_zscore_tickers, backfill_zscore_opciones
    backfill_zscore_tickers(engine)
    backfill_zscore_opciones(engine)
"""

import math
from datetime import date
from typing import Optional

import psycopg2.extras
from sqlalchemy import text

from src.data.database import get_engine, get_connection


# ── DDL ───────────────────────────────────────────────────────────────────────

DDL_OPCIONES_ZSCORE = """
CREATE TABLE IF NOT EXISTS opciones_zscore_diario (
    id                  SERIAL PRIMARY KEY,
    ticker              VARCHAR(20)   NOT NULL,
    fecha               DATE          NOT NULL,

    -- Dimensiones de agrupacion (desnormalizadas para queries eficientes sin JOIN)
    sector              VARCHAR(100),
    industry            VARCHAR(100),

    -- Valores brutos de contexto (evita JOIN a opciones_resumen_diario)
    vol_calls           BIGINT,
    vol_puts            BIGINT,
    vol_total           BIGINT,
    pcr_vol             NUMERIC(8,4),
    iv_avg              NUMERIC(8,6),   -- promedio ponderado IV call + put

    -- Z-scores (ventana hasta 60 dias anteriores)
    vol_calls_zscore    NUMERIC(6,2),
    vol_puts_zscore     NUMERIC(6,2),
    vol_total_zscore    NUMERIC(6,2),
    pcr_vol_zscore      NUMERIC(6,2),   -- Z del ratio, no solo del volumen
    iv_zscore           NUMERIC(6,2),   -- spike de IV = anticipacion de movimiento

    -- Metricas complementarias
    vol_relativo        NUMERIC(8,4),   -- vol_total / vol_media_hist
    percentil_vol       SMALLINT,       -- 0-100

    -- Metadata del calculo (auditoria y debug)
    ventana_dias        SMALLINT,       -- dias usados en la ventana historica
    vol_media           NUMERIC(14,2),
    vol_std             NUMERIC(14,2),

    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_opciones_zscore_fecha   ON opciones_zscore_diario (fecha);
CREATE INDEX IF NOT EXISTS idx_opciones_zscore_ticker  ON opciones_zscore_diario (ticker);
CREATE INDEX IF NOT EXISTS idx_opciones_zscore_sector  ON opciones_zscore_diario (sector);
"""

DDL_TICKER_ZSCORE = """
CREATE TABLE IF NOT EXISTS ticker_zscore_diario (
    id                  SERIAL PRIMARY KEY,
    ticker              VARCHAR(20)   NOT NULL,
    fecha               DATE          NOT NULL,

    -- Dimensiones de agrupacion
    sector              VARCHAR(100),
    industry            VARCHAR(100),

    -- Valores brutos de contexto (evita JOIN a precios_diarios)
    volume              BIGINT,
    close               NUMERIC(12,4),
    retorno_diario      NUMERIC(8,6),   -- (close - close_prev) / close_prev

    -- Z-scores
    vol_zscore          NUMERIC(6,2),
    retorno_zscore      NUMERIC(6,2),   -- retorno del dia vs historia propia

    -- Metricas complementarias
    vol_relativo        NUMERIC(8,4),   -- volume / vol_media_hist
    percentil_vol       SMALLINT,       -- 0-100

    -- Metadata del calculo
    ventana_dias        SMALLINT,
    vol_media           NUMERIC(18,2),
    vol_std             NUMERIC(18,2),

    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_ticker_zscore_fecha   ON ticker_zscore_diario (fecha);
CREATE INDEX IF NOT EXISTS idx_ticker_zscore_ticker  ON ticker_zscore_diario (ticker);
CREATE INDEX IF NOT EXISTS idx_ticker_zscore_sector  ON ticker_zscore_diario (sector);
"""

# Z-scores AGREGADOS a nivel sector (no por ticker). Detecta sentimiento de
# opciones inusual en TODO un sector vs su propia historia -- util para ver
# si la tendencia de un ticker esta acompanada o contradicha por su sector.
# Clave: pcr_vol_sector se calcula SUM(puts)/SUM(calls) del sector, NO como
# promedio de los PCR por ticker (que sesgaria por tamano relativo).
DDL_OPCIONES_SECTOR_ZSCORE = """
CREATE TABLE IF NOT EXISTS opciones_sector_zscore_diario (
    id                      SERIAL PRIMARY KEY,
    fecha                   DATE          NOT NULL,
    sector                  VARCHAR(100)  NOT NULL,

    n_tickers               SMALLINT,     -- tickers del sector que aportaron

    -- Volumenes agregados del sector (suma de todos sus tickers)
    vol_calls_sector        BIGINT,
    vol_puts_sector         BIGINT,
    vol_total_sector        BIGINT,
    pcr_vol_sector          NUMERIC(8,4), -- SUM(puts) / SUM(calls)

    -- Z-scores temporales (ventana hasta 60 dias del propio sector)
    pcr_vol_sector_zscore   NUMERIC(6,2), -- sentimiento sectorial inusual
    pcr_vol_sector_media    NUMERIC(8,4),
    pcr_vol_sector_std      NUMERIC(8,4),
    vol_total_sector_zscore NUMERIC(6,2), -- spike de actividad sectorial
    vol_total_sector_media  NUMERIC(18,2),
    vol_total_sector_std    NUMERIC(18,2),

    ventana_dias            SMALLINT,
    created_at              TIMESTAMP DEFAULT NOW(),
    UNIQUE (fecha, sector)
);

CREATE INDEX IF NOT EXISTS idx_opc_sector_zscore_fecha  ON opciones_sector_zscore_diario (fecha);
CREATE INDEX IF NOT EXISTS idx_opc_sector_zscore_sector ON opciones_sector_zscore_diario (sector);
"""


def init_tablas(engine=None):
    """Crea las tablas de Z-scores si no existen."""
    eng = engine or get_engine()
    with eng.connect() as conn:
        conn.execute(text(DDL_OPCIONES_ZSCORE))
        conn.execute(text(DDL_TICKER_ZSCORE))
        conn.execute(text(DDL_OPCIONES_SECTOR_ZSCORE))
        conn.commit()


# ── Helpers estadisticos ──────────────────────────────────────────────────────

def _safe_round(val, decimals: int = 2) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _media_std(values: list) -> tuple:
    """Media y desviacion estandar muestral. Retorna (None, None) si < 2 valores."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n < 2:
        return None, None
    mean = sum(vals) / n
    var  = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var)


def _zscore(val, mean, std) -> Optional[float]:
    """Z = (val - mean) / std. Retorna None si std == 0 o inputs invalidos."""
    if val is None or mean is None or std is None or std == 0:
        return None
    try:
        return round((float(val) - float(mean)) / float(std), 2)
    except (TypeError, ValueError):
        return None


def _percentil(val, history: list) -> Optional[int]:
    """Percentil de val en la historia (0-100). None si historia vacia."""
    hist = [h for h in history if h is not None]
    if not hist or val is None:
        return None
    return round(sum(1 for h in hist if h < val) / len(hist) * 100)


# ── SQL de upsert compartido ──────────────────────────────────────────────────

_SQL_UPSERT_OPCIONES = """
    INSERT INTO opciones_zscore_diario (
        ticker, fecha, sector, industry,
        vol_calls, vol_puts, vol_total, pcr_vol, iv_avg,
        vol_calls_zscore, vol_puts_zscore, vol_total_zscore,
        pcr_vol_zscore, iv_zscore,
        vol_relativo, percentil_vol,
        ventana_dias, vol_media, vol_std
    ) VALUES %s
    ON CONFLICT (ticker, fecha) DO UPDATE SET
        sector           = EXCLUDED.sector,
        industry         = EXCLUDED.industry,
        vol_calls        = EXCLUDED.vol_calls,
        vol_puts         = EXCLUDED.vol_puts,
        vol_total        = EXCLUDED.vol_total,
        pcr_vol          = EXCLUDED.pcr_vol,
        iv_avg           = EXCLUDED.iv_avg,
        vol_calls_zscore = EXCLUDED.vol_calls_zscore,
        vol_puts_zscore  = EXCLUDED.vol_puts_zscore,
        vol_total_zscore = EXCLUDED.vol_total_zscore,
        pcr_vol_zscore   = EXCLUDED.pcr_vol_zscore,
        iv_zscore        = EXCLUDED.iv_zscore,
        vol_relativo     = EXCLUDED.vol_relativo,
        percentil_vol    = EXCLUDED.percentil_vol,
        ventana_dias     = EXCLUDED.ventana_dias,
        vol_media        = EXCLUDED.vol_media,
        vol_std          = EXCLUDED.vol_std
"""

_SQL_UPSERT_TICKERS = """
    INSERT INTO ticker_zscore_diario (
        ticker, fecha, sector, industry,
        volume, close, retorno_diario,
        vol_zscore, retorno_zscore,
        vol_relativo, percentil_vol,
        ventana_dias, vol_media, vol_std
    ) VALUES %s
    ON CONFLICT (ticker, fecha) DO UPDATE SET
        sector         = EXCLUDED.sector,
        industry       = EXCLUDED.industry,
        volume         = EXCLUDED.volume,
        close          = EXCLUDED.close,
        retorno_diario = EXCLUDED.retorno_diario,
        vol_zscore     = EXCLUDED.vol_zscore,
        retorno_zscore = EXCLUDED.retorno_zscore,
        vol_relativo   = EXCLUDED.vol_relativo,
        percentil_vol  = EXCLUDED.percentil_vol,
        ventana_dias   = EXCLUDED.ventana_dias,
        vol_media      = EXCLUDED.vol_media,
        vol_std        = EXCLUDED.vol_std
"""


# ── Calculo diario: opciones ──────────────────────────────────────────────────

def calcular_zscore_opciones(fecha: date, engine=None, ventana: int = 60) -> int:
    """
    Calcula Z-scores de opciones para todos los tickers con datos en `fecha`.
    Lee historia de opciones_resumen_diario (excluye fecha actual de la ventana).
    Retorna numero de filas insertadas/actualizadas en opciones_zscore_diario.
    """
    eng = engine or get_engine()

    # Tickers con datos en esta fecha + sector/industry desde activos
    with eng.connect() as conn:
        tickers_rows = conn.execute(text("""
            SELECT r.ticker, a.sector, a.industry
            FROM   opciones_resumen_diario r
            LEFT   JOIN activos a ON r.ticker = a.ticker
            WHERE  r.fecha = :fecha
            ORDER  BY r.ticker
        """), {"fecha": fecha}).fetchall()

    if not tickers_rows:
        return 0

    registros = []

    with eng.connect() as conn:
        for ticker, sector, industry in tickers_rows:
            # Datos del dia actual
            row = conn.execute(text("""
                SELECT call_vol, put_vol, pcr_vol, iv_call_avg, iv_put_avg,
                       call_oi, put_oi
                FROM   opciones_resumen_diario
                WHERE  ticker = :t AND fecha = :f
            """), {"t": ticker, "f": fecha}).fetchone()

            if not row:
                continue

            call_vol  = int(row[0] or 0)
            put_vol   = int(row[1] or 0)
            vol_total = call_vol + put_vol
            pcr_vol   = float(row[2]) if row[2] is not None else None
            iv_call   = float(row[3]) if row[3] is not None else None
            iv_put    = float(row[4]) if row[4] is not None else None
            call_oi   = int(row[5] or 0)
            put_oi    = int(row[6] or 0)
            total_oi  = call_oi + put_oi

            # IV promedio ponderado por OI (fallback a simple si OI = 0)
            if iv_call is not None and iv_put is not None:
                if total_oi > 0:
                    iv_avg = (iv_call * call_oi + iv_put * put_oi) / total_oi
                else:
                    iv_avg = (iv_call + iv_put) / 2
            elif iv_call is not None:
                iv_avg = iv_call
            elif iv_put is not None:
                iv_avg = iv_put
            else:
                iv_avg = None

            # Historia previa (excluye fecha actual)
            hist = conn.execute(text("""
                SELECT call_vol, put_vol, pcr_vol, iv_call_avg, iv_put_avg,
                       call_oi, put_oi
                FROM   opciones_resumen_diario
                WHERE  ticker = :t AND fecha < :f
                ORDER  BY fecha DESC
                LIMIT  :v
            """), {"t": ticker, "f": fecha, "v": ventana}).fetchall()

            if not hist:
                registros.append(_row_opciones_sin_hist(
                    ticker, fecha, sector, industry,
                    call_vol, put_vol, vol_total, pcr_vol, iv_avg
                ))
                continue

            h_call  = [int(r[0] or 0) for r in hist]
            h_put   = [int(r[1] or 0) for r in hist]
            h_total = [c + p for c, p in zip(h_call, h_put)]
            h_pcr   = [float(r[2]) for r in hist if r[2] is not None]
            h_iv    = []
            for r in hist:
                ic  = float(r[3]) if r[3] is not None else None
                ip  = float(r[4]) if r[4] is not None else None
                hoi = int(r[5] or 0) + int(r[6] or 0)
                if ic is not None and ip is not None:
                    h_iv.append((ic * int(r[5] or 0) + ip * int(r[6] or 0)) / hoi
                                if hoi > 0 else (ic + ip) / 2)
                elif ic is not None:
                    h_iv.append(ic)
                elif ip is not None:
                    h_iv.append(ip)

            mean_call,  std_call  = _media_std(h_call)
            mean_put,   std_put   = _media_std(h_put)
            mean_total, std_total = _media_std(h_total)
            mean_pcr,   std_pcr   = _media_std(h_pcr)
            mean_iv,    std_iv    = _media_std(h_iv)

            vol_rel = _safe_round(vol_total / mean_total, 4) if mean_total and mean_total > 0 else None

            registros.append((
                ticker, fecha, sector, industry,
                call_vol, put_vol, vol_total,
                pcr_vol, _safe_round(iv_avg, 6),
                _zscore(call_vol, mean_call, std_call),
                _zscore(put_vol,  mean_put,  std_put),
                _zscore(vol_total, mean_total, std_total),
                _zscore(pcr_vol,  mean_pcr,  std_pcr),
                _zscore(iv_avg,   mean_iv,   std_iv),
                vol_rel,
                _percentil(vol_total, h_total),
                len(hist),
                _safe_round(mean_total, 2),
                _safe_round(std_total,  2),
            ))

    if not registros:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _SQL_UPSERT_OPCIONES, registros, page_size=200)

    return len(registros)


def _row_opciones_sin_hist(ticker, fecha, sector, industry,
                           call_vol, put_vol, vol_total, pcr_vol, iv_avg) -> tuple:
    """Fila con Z-scores nulos para tickers sin historia previa."""
    return (
        ticker, fecha, sector, industry,
        call_vol, put_vol, vol_total,
        pcr_vol, _safe_round(iv_avg, 6),
        None, None, None, None, None,  # Z-scores
        None, None,                    # vol_relativo, percentil
        0, None, None,                 # ventana_dias, vol_media, vol_std
    )


# ── Calculo: opciones agregadas por sector ────────────────────────────────────

_SQL_UPSERT_OPCIONES_SECTOR = """
    INSERT INTO opciones_sector_zscore_diario (
        fecha, sector, n_tickers,
        vol_calls_sector, vol_puts_sector, vol_total_sector,
        pcr_vol_sector, pcr_vol_sector_zscore, pcr_vol_sector_media, pcr_vol_sector_std,
        vol_total_sector_zscore, vol_total_sector_media, vol_total_sector_std,
        ventana_dias
    ) VALUES %s
    ON CONFLICT (fecha, sector) DO UPDATE SET
        n_tickers               = EXCLUDED.n_tickers,
        vol_calls_sector        = EXCLUDED.vol_calls_sector,
        vol_puts_sector         = EXCLUDED.vol_puts_sector,
        vol_total_sector        = EXCLUDED.vol_total_sector,
        pcr_vol_sector          = EXCLUDED.pcr_vol_sector,
        pcr_vol_sector_zscore   = EXCLUDED.pcr_vol_sector_zscore,
        pcr_vol_sector_media    = EXCLUDED.pcr_vol_sector_media,
        pcr_vol_sector_std      = EXCLUDED.pcr_vol_sector_std,
        vol_total_sector_zscore = EXCLUDED.vol_total_sector_zscore,
        vol_total_sector_media  = EXCLUDED.vol_total_sector_media,
        vol_total_sector_std    = EXCLUDED.vol_total_sector_std,
        ventana_dias            = EXCLUDED.ventana_dias
"""


def calcular_zscore_opciones_sector(fecha: date, engine=None, ventana: int = 60) -> int:
    """
    Calcula el PCR_Vol y volumen agregados por SECTOR para `fecha`, mas sus
    Z-scores temporales contra la propia historia del sector.

    pcr_vol_sector = SUM(put_vol) / SUM(call_vol) de todos los tickers del
    sector (NO promedio de PCR por ticker -- eso sesgaria por tamano relativo).

    Lee de opciones_resumen_diario + activos (sector). Excluye `fecha` de la
    ventana historica para no contaminar la media.

    Retorna numero de filas insertadas/actualizadas.
    """
    eng = engine or get_engine()

    # Agregacion sectorial del dia actual
    with eng.connect() as conn:
        hoy_rows = conn.execute(text("""
            SELECT a.sector,
                   COUNT(*)                          AS n_tickers,
                   SUM(COALESCE(r.call_vol, 0))      AS call_vol,
                   SUM(COALESCE(r.put_vol,  0))      AS put_vol
            FROM   opciones_resumen_diario r
            JOIN   activos a ON r.ticker = a.ticker
            WHERE  r.fecha = :f AND a.sector IS NOT NULL
            GROUP  BY a.sector
            ORDER  BY a.sector
        """), {"f": fecha}).fetchall()

    if not hoy_rows:
        return 0

    registros = []

    with eng.connect() as conn:
        for sector, n_tickers, call_vol, put_vol in hoy_rows:
            call_vol  = int(call_vol or 0)
            put_vol   = int(put_vol or 0)
            vol_total = call_vol + put_vol
            pcr_vol   = _safe_round(put_vol / call_vol, 4) if call_vol > 0 else None

            # Historia sectorial previa: agrega por fecha < f para el mismo sector
            hist = conn.execute(text("""
                SELECT r.fecha,
                       SUM(COALESCE(r.call_vol, 0)) AS call_vol,
                       SUM(COALESCE(r.put_vol,  0)) AS put_vol
                FROM   opciones_resumen_diario r
                JOIN   activos a ON r.ticker = a.ticker
                WHERE  a.sector = :s AND r.fecha < :f
                GROUP  BY r.fecha
                ORDER  BY r.fecha DESC
                LIMIT  :v
            """), {"s": sector, "f": fecha, "v": ventana}).fetchall()

            if not hist:
                registros.append((
                    fecha, sector, int(n_tickers),
                    call_vol, put_vol, vol_total,
                    pcr_vol, None, None, None,   # pcr zscore/media/std
                    None, None, None,            # vol_total zscore/media/std
                    0,
                ))
                continue

            h_call  = [int(r[1] or 0) for r in hist]
            h_put   = [int(r[2] or 0) for r in hist]
            h_total = [c + p for c, p in zip(h_call, h_put)]
            h_pcr   = [round(p / c, 4) for c, p in zip(h_call, h_put) if c > 0]

            mean_total, std_total = _media_std(h_total)
            mean_pcr,   std_pcr   = _media_std(h_pcr)

            registros.append((
                fecha, sector, int(n_tickers),
                call_vol, put_vol, vol_total,
                pcr_vol,
                _zscore(pcr_vol, mean_pcr, std_pcr),
                _safe_round(mean_pcr, 4),
                _safe_round(std_pcr, 4),
                _zscore(vol_total, mean_total, std_total),
                _safe_round(mean_total, 2),
                _safe_round(std_total, 2),
                len(hist),
            ))

    if not registros:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _SQL_UPSERT_OPCIONES_SECTOR, registros, page_size=200)

    return len(registros)


def backfill_zscore_opciones_sector(engine=None, desde: date = None) -> int:
    """
    Backfill de opciones_sector_zscore_diario para todas las fechas presentes
    en opciones_resumen_diario. Reusa calcular_zscore_opciones_sector() por
    fecha (volumen chico: ~10 sectores x N fechas).

    Args:
        desde: si se especifica, solo procesa fechas >= desde.
    """
    eng = engine or get_engine()
    filtro = "WHERE fecha >= :desde" if desde else ""
    with eng.connect() as conn:
        fechas = conn.execute(
            text(f"SELECT DISTINCT fecha FROM opciones_resumen_diario {filtro} ORDER BY fecha"),
            {"desde": desde} if desde else {}
        ).fetchall()

    total = 0
    for (f,) in fechas:
        total += calcular_zscore_opciones_sector(f, engine=eng)
    return total


# ── Calculo diario: tickers ───────────────────────────────────────────────────

def calcular_zscore_tickers(fecha: date, engine=None, ventana: int = 60) -> int:
    """
    Calcula Z-scores de volumen y retorno para todos los tickers con datos en `fecha`.
    Lee historia de precios_diarios (excluye fecha actual de la ventana).
    Retorna numero de filas insertadas/actualizadas en ticker_zscore_diario.
    """
    eng = engine or get_engine()

    # Tickers con datos en esta fecha
    with eng.connect() as conn:
        tickers_rows = conn.execute(text("""
            SELECT p.ticker, a.sector, a.industry
            FROM   precios_diarios p
            LEFT   JOIN activos a ON p.ticker = a.ticker
            WHERE  p.fecha = :fecha AND p.volume > 0
            ORDER  BY p.ticker
        """), {"fecha": fecha}).fetchall()

    if not tickers_rows:
        return 0

    registros = []

    with eng.connect() as conn:
        for ticker, sector, industry in tickers_rows:
            # Datos del dia actual
            row = conn.execute(text("""
                SELECT close, volume
                FROM   precios_diarios
                WHERE  ticker = :t AND fecha = :f
            """), {"t": ticker, "f": fecha}).fetchone()

            if not row or not row[0] or not row[1]:
                continue

            close_hoy  = float(row[0])
            volume_hoy = int(row[1])

            # Historia previa (ventana + 1 para calcular retorno del dia anterior)
            hist = conn.execute(text("""
                SELECT fecha, close, volume
                FROM   precios_diarios
                WHERE  ticker = :t AND fecha < :f AND volume > 0
                ORDER  BY fecha DESC
                LIMIT  :v
            """), {"t": ticker, "f": fecha, "v": ventana + 1}).fetchall()

            if not hist:
                registros.append(_row_ticker_sin_hist(
                    ticker, fecha, sector, industry, volume_hoy, close_hoy
                ))
                continue

            h_closes  = [float(r[1]) for r in hist if r[1]]
            h_volumes = [int(r[2] or 0) for r in hist]

            # Retorno del dia: close_hoy vs close_ayer
            retorno_diario = None
            if h_closes:
                close_prev = h_closes[0]
                if close_prev > 0:
                    retorno_diario = round((close_hoy - close_prev) / close_prev, 6)

            # Serie de retornos historicos para Z-score de retorno
            h_retornos = []
            for i in range(len(h_closes) - 1):
                if h_closes[i + 1] > 0:
                    h_retornos.append((h_closes[i] - h_closes[i + 1]) / h_closes[i + 1])

            # Limitar historia al ventana real (sin el +1 extra)
            h_volumes_ventana = h_volumes[:ventana]
            ventana_dias = len(h_volumes_ventana)

            mean_vol, std_vol = _media_std(h_volumes_ventana)
            mean_ret, std_ret = _media_std(h_retornos)

            vol_rel = _safe_round(volume_hoy / mean_vol, 4) if mean_vol and mean_vol > 0 else None

            registros.append((
                ticker, fecha, sector, industry,
                volume_hoy, close_hoy, retorno_diario,
                _zscore(volume_hoy,    mean_vol, std_vol),
                _zscore(retorno_diario, mean_ret, std_ret),
                vol_rel,
                _percentil(volume_hoy, h_volumes_ventana),
                ventana_dias,
                _safe_round(mean_vol, 2),
                _safe_round(std_vol,  2),
            ))

    if not registros:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _SQL_UPSERT_TICKERS, registros, page_size=200)

    return len(registros)


def _row_ticker_sin_hist(ticker, fecha, sector, industry, volume, close) -> tuple:
    """Fila con Z-scores nulos para tickers sin historia previa."""
    return (
        ticker, fecha, sector, industry,
        volume, close, None,
        None, None,         # vol_zscore, retorno_zscore
        None, None,         # vol_relativo, percentil
        0, None, None,      # ventana_dias, vol_media, vol_std
    )


# ── Backfill completo via SQL window functions ────────────────────────────────

def backfill_zscore_tickers(engine=None, desde: date = None) -> int:
    """
    Backfill de ticker_zscore_diario para TODAS las fechas en precios_diarios.
    Usa window functions SQL para eficiencia: una sola query vs N queries por fecha.
    Retorna numero de filas insertadas/actualizadas.

    Args:
        desde: Si se especifica, solo procesa fechas >= desde.
    """
    eng = engine or get_engine()

    # Bug fix: el WHERE final esta sobre `stats s`, no sobre `prices p`.
    # Antes decia "AND p.fecha >= :desde" y daba "falta entrada para tabla p".
    filtro_desde = "AND s.fecha >= :desde" if desde else ""

    SQL = f"""
        WITH prices AS (
            SELECT p.ticker, p.fecha, p.close, p.volume,
                   a.sector, a.industry,
                   LAG(p.close) OVER (PARTITION BY p.ticker ORDER BY p.fecha) AS prev_close
            FROM   precios_diarios p
            LEFT   JOIN activos a ON p.ticker = a.ticker
            WHERE  p.volume > 0
        ),
        retornos AS (
            SELECT ticker, fecha, close, volume, sector, industry,
                   CASE WHEN prev_close > 0
                        THEN (close - prev_close) / prev_close
                        ELSE NULL
                   END AS retorno
            FROM prices
        ),
        stats AS (
            SELECT
                ticker, fecha, close, volume, sector, industry, retorno,
                AVG(volume::NUMERIC) OVER w       AS vol_media,
                STDDEV_SAMP(volume::NUMERIC) OVER w AS vol_std,
                COUNT(volume) OVER w              AS ventana_dias,
                AVG(retorno) OVER w               AS ret_media,
                STDDEV_SAMP(retorno) OVER w       AS ret_std
            FROM retornos
            WINDOW w AS (
                PARTITION BY ticker
                ORDER BY fecha
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
            )
        )
        INSERT INTO ticker_zscore_diario (
            ticker, fecha, sector, industry,
            volume, close, retorno_diario,
            vol_zscore, retorno_zscore,
            vol_relativo, percentil_vol,
            ventana_dias, vol_media, vol_std
        )
        SELECT
            s.ticker,
            s.fecha,
            s.sector,
            s.industry,
            s.volume,
            s.close,
            ROUND(s.retorno::NUMERIC, 6)                                       AS retorno_diario,
            CASE WHEN s.vol_std > 0
                 THEN ROUND((s.volume - s.vol_media) / s.vol_std, 2)
                 ELSE NULL END                                                  AS vol_zscore,
            CASE WHEN s.ret_std > 0 AND s.retorno IS NOT NULL
                 THEN ROUND((s.retorno - s.ret_media) / s.ret_std, 2)
                 ELSE NULL END                                                  AS retorno_zscore,
            CASE WHEN s.vol_media > 0
                 THEN ROUND(s.volume / s.vol_media, 4)
                 ELSE NULL END                                                  AS vol_relativo,
            NULL::SMALLINT                                                      AS percentil_vol,
            COALESCE(s.ventana_dias, 0)::SMALLINT                              AS ventana_dias,
            ROUND(s.vol_media, 2)                                               AS vol_media,
            ROUND(s.vol_std, 2)                                                 AS vol_std
        FROM stats s
        WHERE TRUE {filtro_desde}
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            sector         = EXCLUDED.sector,
            industry       = EXCLUDED.industry,
            volume         = EXCLUDED.volume,
            close          = EXCLUDED.close,
            retorno_diario = EXCLUDED.retorno_diario,
            vol_zscore     = EXCLUDED.vol_zscore,
            retorno_zscore = EXCLUDED.retorno_zscore,
            vol_relativo   = EXCLUDED.vol_relativo,
            ventana_dias   = EXCLUDED.ventana_dias,
            vol_media      = EXCLUDED.vol_media,
            vol_std        = EXCLUDED.vol_std
    """

    params = {"desde": desde} if desde else {}

    with eng.connect() as conn:
        result = conn.execute(text(SQL), params)
        conn.commit()
        return result.rowcount


def backfill_zscore_opciones(engine=None, desde: date = None) -> int:
    """
    Backfill de opciones_zscore_diario para TODAS las fechas en opciones_resumen_diario.
    Usa window functions SQL para eficiencia.
    Retorna numero de filas insertadas/actualizadas.

    Args:
        desde: Si se especifica, solo procesa fechas >= desde.
    """
    eng = engine or get_engine()

    filtro_desde = "AND r.fecha >= :desde" if desde else ""

    SQL = f"""
        WITH resumen AS (
            SELECT
                r.ticker, r.fecha,
                a.sector, a.industry,
                COALESCE(r.call_vol, 0)                     AS call_vol,
                COALESCE(r.put_vol,  0)                     AS put_vol,
                COALESCE(r.call_vol, 0) + COALESCE(r.put_vol, 0) AS vol_total,
                r.pcr_vol,
                -- IV ponderada por OI
                CASE
                    WHEN (COALESCE(r.call_oi,0) + COALESCE(r.put_oi,0)) > 0
                         AND r.iv_call_avg IS NOT NULL AND r.iv_put_avg IS NOT NULL
                    THEN (r.iv_call_avg * COALESCE(r.call_oi,0)
                          + r.iv_put_avg * COALESCE(r.put_oi,0))
                         / (COALESCE(r.call_oi,0) + COALESCE(r.put_oi,0))
                    WHEN r.iv_call_avg IS NOT NULL AND r.iv_put_avg IS NOT NULL
                    THEN (r.iv_call_avg + r.iv_put_avg) / 2
                    ELSE COALESCE(r.iv_call_avg, r.iv_put_avg)
                END AS iv_avg
            FROM opciones_resumen_diario r
            LEFT JOIN activos a ON r.ticker = a.ticker
            WHERE TRUE {filtro_desde}
        ),
        stats AS (
            SELECT
                ticker, fecha, sector, industry,
                call_vol, put_vol, vol_total, pcr_vol, iv_avg,
                AVG(call_vol::NUMERIC)  OVER w AS call_media,
                STDDEV_SAMP(call_vol::NUMERIC)  OVER w AS call_std,
                AVG(put_vol::NUMERIC)   OVER w AS put_media,
                STDDEV_SAMP(put_vol::NUMERIC)   OVER w AS put_std,
                AVG(vol_total::NUMERIC) OVER w AS total_media,
                STDDEV_SAMP(vol_total::NUMERIC) OVER w AS total_std,
                AVG(pcr_vol)            OVER w AS pcr_media,
                STDDEV_SAMP(pcr_vol)    OVER w AS pcr_std,
                AVG(iv_avg)             OVER w AS iv_media,
                STDDEV_SAMP(iv_avg)     OVER w AS iv_std,
                COUNT(vol_total) OVER w         AS ventana_dias
            FROM resumen
            WINDOW w AS (
                PARTITION BY ticker
                ORDER BY fecha
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
            )
        )
        INSERT INTO opciones_zscore_diario (
            ticker, fecha, sector, industry,
            vol_calls, vol_puts, vol_total, pcr_vol, iv_avg,
            vol_calls_zscore, vol_puts_zscore, vol_total_zscore,
            pcr_vol_zscore, iv_zscore,
            vol_relativo, percentil_vol,
            ventana_dias, vol_media, vol_std
        )
        SELECT
            ticker, fecha, sector, industry,
            call_vol, put_vol, vol_total,
            pcr_vol,
            ROUND(iv_avg::NUMERIC, 6)                                           AS iv_avg,
            CASE WHEN call_std  > 0 THEN ROUND((call_vol  - call_media)  / call_std,  2) ELSE NULL END,
            CASE WHEN put_std   > 0 THEN ROUND((put_vol   - put_media)   / put_std,   2) ELSE NULL END,
            CASE WHEN total_std > 0 THEN ROUND((vol_total - total_media) / total_std, 2) ELSE NULL END,
            CASE WHEN pcr_std   > 0 AND pcr_vol IS NOT NULL
                      THEN ROUND((pcr_vol - pcr_media) / pcr_std, 2) ELSE NULL END,
            CASE WHEN iv_std    > 0 AND iv_avg  IS NOT NULL
                      THEN ROUND((iv_avg  - iv_media)  / iv_std,  2) ELSE NULL END,
            CASE WHEN total_media > 0 THEN ROUND(vol_total / total_media, 4) ELSE NULL END,
            NULL::SMALLINT,
            COALESCE(ventana_dias, 0)::SMALLINT,
            ROUND(total_media::NUMERIC, 2),
            ROUND(total_std::NUMERIC,   2)
        FROM stats
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            sector           = EXCLUDED.sector,
            industry         = EXCLUDED.industry,
            vol_calls        = EXCLUDED.vol_calls,
            vol_puts         = EXCLUDED.vol_puts,
            vol_total        = EXCLUDED.vol_total,
            pcr_vol          = EXCLUDED.pcr_vol,
            iv_avg           = EXCLUDED.iv_avg,
            vol_calls_zscore = EXCLUDED.vol_calls_zscore,
            vol_puts_zscore  = EXCLUDED.vol_puts_zscore,
            vol_total_zscore = EXCLUDED.vol_total_zscore,
            pcr_vol_zscore   = EXCLUDED.pcr_vol_zscore,
            iv_zscore        = EXCLUDED.iv_zscore,
            vol_relativo     = EXCLUDED.vol_relativo,
            ventana_dias     = EXCLUDED.ventana_dias,
            vol_media        = EXCLUDED.vol_media,
            vol_std          = EXCLUDED.vol_std
    """

    params = {"desde": desde} if desde else {}

    with eng.connect() as conn:
        result = conn.execute(text(SQL), params)
        conn.commit()
        return result.rowcount
