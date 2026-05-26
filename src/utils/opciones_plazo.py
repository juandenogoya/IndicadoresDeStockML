"""
opciones_plazo.py
Capa de datos: PCR (volumen y OI) y muros de OI (soporte/resistencia)
segmentados por VENTANA de vencimiento, por ticker y fecha.

Homogeneidad con las estrategias FT (config y bots oiexit/options):
  Ventanas de vencimiento (dias al vencimiento desde fecha_snapshot):
    corto:  1 - 14   (weeklies, tactico)
    medio:  15 - 45  (mensuales, mayor OI)
    largo:  46 - 90  (institucional)

  Veredicto PCR_OI: < 1.0 = Alcista ('A'), >= 1.0 = Bajista ('B'),
                    OI total < MIN_OI = sin liquidez (None)

  Muro de OI (mismo criterio que ft_bot_tech_sectorial_oiexit_v1):
    - zona: hasta +/- 10% del precio subyacente
    - soporte    = strike con mayor PUT OI por debajo del precio
    - resistencia= strike con mayor CALL OI por encima del precio
    - valido si: OI >= 3 x mediana de la zona  Y  OI >= 1000 absoluto
                 Y distancia al precio >= 2%

Produce la tabla opciones_pcr_plazo_diario (formato largo: 1 fila por
ticker x fecha x ventana). Se alimenta de opciones_snapshot.

Uso diario (desde el snapshot, tras persistir contratos):
    from src.utils.opciones_plazo import calcular_pcr_plazo
    n = calcular_pcr_plazo(fecha_snapshot, engine)

Backfill:
    from src.utils.opciones_plazo import backfill_pcr_plazo
    backfill_pcr_plazo(engine, desde=date(2026,4,18))
"""

import statistics
from datetime import date
from typing import Optional

import psycopg2.extras
from sqlalchemy import text

from src.data.database import get_engine, get_connection


# ── Constantes (homogeneas con config.py y los bots FT) ───────────────────────
VENTANAS = {
    "corto": (1, 14),
    "medio": (15, 45),
    "largo": (46, 90),
}
MIN_OI_POR_VENTANA   = 500    # OI total minimo para emitir veredicto
PCR_UMBRAL_ALCISTA   = 1.0    # < 1.0 = Alcista, >= 1.0 = Bajista

# Muros de OI (identicos a ft_bot_tech_sectorial_oiexit_v1)
WALL_ZONA_PCT        = 0.10   # zona de busqueda: +/- 10% del precio
WALL_LIQ_MULT_MED    = 3.0    # OI >= 3 x mediana de la zona
WALL_LIQ_MIN_ABS     = 1000   # piso de OI absoluto
WALL_DIST_MIN_PCT    = 0.02   # muro a >= 2% del precio


# ── DDL ───────────────────────────────────────────────────────────────────────

DDL_PCR_PLAZO = """
CREATE TABLE IF NOT EXISTS opciones_pcr_plazo_diario (
    id                    SERIAL PRIMARY KEY,
    fecha                 DATE          NOT NULL,   -- = fecha_snapshot
    ticker                VARCHAR(20)   NOT NULL,
    ventana               VARCHAR(10)   NOT NULL,   -- 'corto' | 'medio' | 'largo'
    dte_min               SMALLINT,
    dte_max               SMALLINT,

    -- PCR por volumen
    call_vol              BIGINT,
    put_vol               BIGINT,
    pcr_vol               NUMERIC(8,4),

    -- PCR por open interest
    call_oi               BIGINT,
    put_oi                BIGINT,
    pcr_oi                NUMERIC(8,4),
    veredicto_oi          CHAR(1),                  -- 'A' | 'B' | NULL (sin liquidez)

    precio_sub            NUMERIC(12,4),

    -- Muros de OI (soporte = put wall debajo; resistencia = call wall arriba)
    soporte_strike        NUMERIC(12,4),
    soporte_oi            BIGINT,
    soporte_dist_pct      NUMERIC(6,2),
    resistencia_strike    NUMERIC(12,4),
    resistencia_oi        BIGINT,
    resistencia_dist_pct  NUMERIC(6,2),

    n_contratos           INTEGER,
    created_at            TIMESTAMP DEFAULT NOW(),
    UNIQUE (fecha, ticker, ventana)
);

CREATE INDEX IF NOT EXISTS idx_pcr_plazo_fecha   ON opciones_pcr_plazo_diario (fecha);
CREATE INDEX IF NOT EXISTS idx_pcr_plazo_ticker  ON opciones_pcr_plazo_diario (ticker);
CREATE INDEX IF NOT EXISTS idx_pcr_plazo_ventana ON opciones_pcr_plazo_diario (ventana);
"""


def init_tabla(engine=None):
    """Crea la tabla opciones_pcr_plazo_diario si no existe."""
    eng = engine or get_engine()
    with eng.connect() as conn:
        conn.execute(text(DDL_PCR_PLAZO))
        conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_round(val, decimals: int = 4) -> Optional[float]:
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def _pcr(put: int, call: int) -> Optional[float]:
    """PCR = put / call. None si call == 0 (sin denominador valido)."""
    return _safe_round(put / call, 4) if call and call > 0 else None


def _veredicto(pcr: Optional[float], oi_total: int) -> Optional[str]:
    """A (alcista) si pcr<1, B (bajista) si >=1, None si sin liquidez."""
    if oi_total < MIN_OI_POR_VENTANA or pcr is None:
        return None
    return "A" if pcr < PCR_UMBRAL_ALCISTA else "B"


def _muro(cands: list, precio: float, lado: str) -> dict:
    """
    Calcula el muro de OI (mismo criterio que oiexit).

    Args:
        cands: lista de (strike, oi) candidatos en la zona.
        precio: precio subyacente.
        lado: 'soporte' (strikes debajo) o 'resistencia' (strikes arriba).

    Returns:
        {strike, oi, dist_pct} si valido, o {} si no hay muro valido.
    """
    if not cands or not precio:
        return {}
    mediana = statistics.median([oi for _, oi in cands])
    wall_strike, wall_oi = max(cands, key=lambda x: x[1])
    dist = abs(precio - wall_strike) / precio
    valido = (
        wall_oi >= WALL_LIQ_MULT_MED * mediana
        and wall_oi >= WALL_LIQ_MIN_ABS
        and dist >= WALL_DIST_MIN_PCT
    )
    if not valido:
        return {}
    return {
        "strike":   round(wall_strike, 4),
        "oi":       int(wall_oi),
        "dist_pct": round(dist * 100, 2),
    }


# ── SQL de upsert ──────────────────────────────────────────────────────────────

_SQL_UPSERT = """
    INSERT INTO opciones_pcr_plazo_diario (
        fecha, ticker, ventana, dte_min, dte_max,
        call_vol, put_vol, pcr_vol,
        call_oi, put_oi, pcr_oi, veredicto_oi,
        precio_sub,
        soporte_strike, soporte_oi, soporte_dist_pct,
        resistencia_strike, resistencia_oi, resistencia_dist_pct,
        n_contratos
    ) VALUES %s
    ON CONFLICT (fecha, ticker, ventana) DO UPDATE SET
        dte_min              = EXCLUDED.dte_min,
        dte_max              = EXCLUDED.dte_max,
        call_vol             = EXCLUDED.call_vol,
        put_vol              = EXCLUDED.put_vol,
        pcr_vol              = EXCLUDED.pcr_vol,
        call_oi              = EXCLUDED.call_oi,
        put_oi               = EXCLUDED.put_oi,
        pcr_oi               = EXCLUDED.pcr_oi,
        veredicto_oi         = EXCLUDED.veredicto_oi,
        precio_sub           = EXCLUDED.precio_sub,
        soporte_strike       = EXCLUDED.soporte_strike,
        soporte_oi           = EXCLUDED.soporte_oi,
        soporte_dist_pct     = EXCLUDED.soporte_dist_pct,
        resistencia_strike   = EXCLUDED.resistencia_strike,
        resistencia_oi       = EXCLUDED.resistencia_oi,
        resistencia_dist_pct = EXCLUDED.resistencia_dist_pct,
        n_contratos          = EXCLUDED.n_contratos
"""


# ── Calculo diario ──────────────────────────────────────────────────────────────

def calcular_pcr_plazo(fecha: date, engine=None) -> int:
    """
    Calcula PCR_vol, PCR_OI, veredicto y muros S/R por ventana para todos los
    tickers con contratos en `fecha` (fecha_snapshot). Persiste en
    opciones_pcr_plazo_diario.

    Retorna numero de filas (ticker x ventana) insertadas/actualizadas.
    """
    eng = engine or get_engine()

    # Traemos el detalle por (ticker, strike, tipo, dias_al_venc) de la fecha.
    # dias_al_venc = vencimiento - fecha_snapshot. Filtramos a 1..90 dias.
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                ticker,
                strike,
                tipo,
                (vencimiento - :f)                 AS dte,
                SUM(COALESCE(volumen, 0))          AS vol,
                SUM(COALESCE(open_interest, 0))    AS oi,
                MAX(precio_subyacente)             AS precio
            FROM opciones_snapshot
            WHERE fecha_snapshot = :f
              AND (vencimiento - :f) BETWEEN 1 AND 90
            GROUP BY ticker, strike, tipo, (vencimiento - :f)
        """), {"f": fecha}).fetchall()

    if not rows:
        return 0

    # Estructura: acc[ticker][ventana] = {call_vol, put_vol, call_oi, put_oi,
    #   puts: [(strike, oi)], calls: [(strike, oi)], precio, n}
    acc: dict = {}
    precio_map: dict = {}

    for r in rows:
        ticker = r.ticker
        dte    = int(r.dte)
        tipo   = r.tipo
        strike = float(r.strike)
        vol    = int(r.vol or 0)
        oi     = int(r.oi or 0)
        if r.precio:
            precio_map[ticker] = float(r.precio)

        # A que ventana pertenece este DTE
        ventana = None
        for nombre, (lo, hi) in VENTANAS.items():
            if lo <= dte <= hi:
                ventana = nombre
                break
        if ventana is None:
            continue

        slot = acc.setdefault(ticker, {}).setdefault(ventana, {
            "call_vol": 0, "put_vol": 0, "call_oi": 0, "put_oi": 0,
            "puts": [], "calls": [], "n": 0,
        })
        slot["n"] += 1
        if tipo == "call":
            slot["call_vol"] += vol
            slot["call_oi"]  += oi
            if oi > 0:
                slot["calls"].append((strike, oi))
        else:  # put
            slot["put_vol"] += vol
            slot["put_oi"]  += oi
            if oi > 0:
                slot["puts"].append((strike, oi))

    # Construir registros
    registros = []
    for ticker, ventanas in acc.items():
        precio = precio_map.get(ticker)
        for ventana, slot in ventanas.items():
            lo, hi = VENTANAS[ventana]
            call_vol = slot["call_vol"]
            put_vol  = slot["put_vol"]
            call_oi  = slot["call_oi"]
            put_oi   = slot["put_oi"]
            oi_total = call_oi + put_oi

            pcr_vol = _pcr(put_vol, call_vol)
            pcr_oi  = _pcr(put_oi, call_oi)
            veredicto = _veredicto(pcr_oi, oi_total)

            # Muros: soporte = put wall debajo del precio; resistencia = call wall arriba
            sop = {}
            res = {}
            if precio:
                zona_inf = precio * (1 - WALL_ZONA_PCT)
                zona_sup = precio * (1 + WALL_ZONA_PCT)
                puts_zona  = [(s, oi) for (s, oi) in slot["puts"]  if zona_inf <= s < precio]
                calls_zona = [(s, oi) for (s, oi) in slot["calls"] if precio < s <= zona_sup]
                sop = _muro(puts_zona,  precio, "soporte")
                res = _muro(calls_zona, precio, "resistencia")

            registros.append((
                fecha, ticker, ventana, lo, hi,
                call_vol, put_vol, pcr_vol,
                call_oi, put_oi, pcr_oi, veredicto,
                _safe_round(precio, 4) if precio else None,
                sop.get("strike"), sop.get("oi"), sop.get("dist_pct"),
                res.get("strike"), res.get("oi"), res.get("dist_pct"),
                slot["n"],
            ))

    if not registros:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _SQL_UPSERT, registros, page_size=300)

    return len(registros)


def backfill_pcr_plazo(engine=None, desde: date = None) -> int:
    """
    Backfill de opciones_pcr_plazo_diario para todas las fechas presentes en
    opciones_snapshot. Reusa calcular_pcr_plazo() por fecha.

    Args:
        desde: si se especifica, solo procesa fechas >= desde.
    """
    eng = engine or get_engine()
    filtro = "WHERE fecha_snapshot >= :desde" if desde else ""
    with eng.connect() as conn:
        fechas = conn.execute(
            text(f"SELECT DISTINCT fecha_snapshot FROM opciones_snapshot {filtro} ORDER BY fecha_snapshot"),
            {"desde": desde} if desde else {}
        ).fetchall()

    total = 0
    for (f,) in fechas:
        total += calcular_pcr_plazo(f, engine=eng)
    return total


# ── PCR sectorial por ventana (deriva de opciones_pcr_plazo_diario) ────────────

DDL_SECTOR_PCR_PLAZO = """
CREATE TABLE IF NOT EXISTS opciones_sector_pcr_plazo_diario (
    id                      SERIAL PRIMARY KEY,
    fecha                   DATE          NOT NULL,
    sector                  VARCHAR(100)  NOT NULL,
    ventana                 VARCHAR(10)   NOT NULL,
    dte_min                 SMALLINT,
    dte_max                 SMALLINT,
    n_tickers               SMALLINT,

    call_vol_sector         BIGINT,
    put_vol_sector          BIGINT,
    pcr_vol_sector          NUMERIC(8,4),

    call_oi_sector          BIGINT,
    put_oi_sector           BIGINT,
    pcr_oi_sector           NUMERIC(8,4),
    veredicto_oi            CHAR(1),

    -- Z-score temporal del PCR_vol sectorial (vs historia del propio
    -- sector+ventana). Detecta sentimiento inusual a nivel sector/plazo.
    pcr_vol_sector_zscore   NUMERIC(6,2),
    pcr_vol_sector_media    NUMERIC(8,4),
    pcr_vol_sector_std      NUMERIC(8,4),
    ventana_dias            SMALLINT,

    created_at              TIMESTAMP DEFAULT NOW(),
    UNIQUE (fecha, sector, ventana)
);

CREATE INDEX IF NOT EXISTS idx_sec_pcr_plazo_fecha   ON opciones_sector_pcr_plazo_diario (fecha);
CREATE INDEX IF NOT EXISTS idx_sec_pcr_plazo_sector  ON opciones_sector_pcr_plazo_diario (sector);
CREATE INDEX IF NOT EXISTS idx_sec_pcr_plazo_ventana ON opciones_sector_pcr_plazo_diario (ventana);
"""


def init_tabla_sector(engine=None):
    """Crea la tabla opciones_sector_pcr_plazo_diario si no existe."""
    eng = engine or get_engine()
    with eng.connect() as conn:
        conn.execute(text(DDL_SECTOR_PCR_PLAZO))
        conn.commit()


_SQL_UPSERT_SECTOR = """
    INSERT INTO opciones_sector_pcr_plazo_diario (
        fecha, sector, ventana, dte_min, dte_max, n_tickers,
        call_vol_sector, put_vol_sector, pcr_vol_sector,
        call_oi_sector, put_oi_sector, pcr_oi_sector, veredicto_oi,
        pcr_vol_sector_zscore, pcr_vol_sector_media, pcr_vol_sector_std, ventana_dias
    ) VALUES %s
    ON CONFLICT (fecha, sector, ventana) DO UPDATE SET
        dte_min               = EXCLUDED.dte_min,
        dte_max               = EXCLUDED.dte_max,
        n_tickers             = EXCLUDED.n_tickers,
        call_vol_sector       = EXCLUDED.call_vol_sector,
        put_vol_sector        = EXCLUDED.put_vol_sector,
        pcr_vol_sector        = EXCLUDED.pcr_vol_sector,
        call_oi_sector        = EXCLUDED.call_oi_sector,
        put_oi_sector         = EXCLUDED.put_oi_sector,
        pcr_oi_sector         = EXCLUDED.pcr_oi_sector,
        veredicto_oi          = EXCLUDED.veredicto_oi,
        pcr_vol_sector_zscore = EXCLUDED.pcr_vol_sector_zscore,
        pcr_vol_sector_media  = EXCLUDED.pcr_vol_sector_media,
        pcr_vol_sector_std    = EXCLUDED.pcr_vol_sector_std,
        ventana_dias          = EXCLUDED.ventana_dias
"""


def calcular_pcr_sector_plazo(fecha: date, engine=None, ventana_zscore: int = 60) -> int:
    """
    Agrega opciones_pcr_plazo_diario por (sector, ventana) para `fecha` y
    calcula PCR_vol/OI sectorial + z-score temporal del PCR_vol sectorial.

    Deriva de la tabla por ticker (no recalcula desde opciones_snapshot).
    Prerequisito: calcular_pcr_plazo(fecha) ya corrio para esa fecha.

    Retorna numero de filas (sector x ventana) insertadas/actualizadas.
    """
    from src.utils.zscore_pipeline import _media_std, _zscore

    eng = engine or get_engine()

    # Agregacion sectorial del dia: suma vol/OI de los tickers de cada sector
    with eng.connect() as conn:
        hoy = conn.execute(text("""
            SELECT a.sector, p.ventana, p.dte_min, p.dte_max,
                   COUNT(*)                AS n_tickers,
                   SUM(p.call_vol)         AS call_vol,
                   SUM(p.put_vol)          AS put_vol,
                   SUM(p.call_oi)          AS call_oi,
                   SUM(p.put_oi)           AS put_oi
            FROM   opciones_pcr_plazo_diario p
            JOIN   activos a ON p.ticker = a.ticker
            WHERE  p.fecha = :f AND a.sector IS NOT NULL
            GROUP  BY a.sector, p.ventana, p.dte_min, p.dte_max
        """), {"f": fecha}).fetchall()

    if not hoy:
        return 0

    registros = []
    with eng.connect() as conn:
        for row in hoy:
            sector  = row.sector
            ventana = row.ventana
            call_vol = int(row.call_vol or 0)
            put_vol  = int(row.put_vol or 0)
            call_oi  = int(row.call_oi or 0)
            put_oi   = int(row.put_oi or 0)
            oi_total = call_oi + put_oi

            pcr_vol = _pcr(put_vol, call_vol)
            pcr_oi  = _pcr(put_oi, call_oi)
            veredicto = _veredicto(pcr_oi, oi_total)

            # Historia del PCR_vol sectorial para (sector, ventana), fechas < f
            hist = conn.execute(text("""
                SELECT p.fecha,
                       SUM(p.call_vol) AS call_vol,
                       SUM(p.put_vol)  AS put_vol
                FROM   opciones_pcr_plazo_diario p
                JOIN   activos a ON p.ticker = a.ticker
                WHERE  a.sector = :s AND p.ventana = :v AND p.fecha < :f
                GROUP  BY p.fecha
                ORDER  BY p.fecha DESC
                LIMIT  :n
            """), {"s": sector, "v": ventana, "f": fecha, "n": ventana_zscore}).fetchall()

            h_pcr = [_pcr(int(h.put_vol or 0), int(h.call_vol or 0)) for h in hist]
            h_pcr = [x for x in h_pcr if x is not None]
            mean_pcr, std_pcr = _media_std(h_pcr)

            registros.append((
                fecha, sector, ventana, int(row.dte_min), int(row.dte_max), int(row.n_tickers),
                call_vol, put_vol, pcr_vol,
                call_oi, put_oi, pcr_oi, veredicto,
                _zscore(pcr_vol, mean_pcr, std_pcr),
                _safe_round(mean_pcr, 4), _safe_round(std_pcr, 4),
                len(h_pcr),
            ))

    if not registros:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _SQL_UPSERT_SECTOR, registros, page_size=200)

    return len(registros)


def backfill_pcr_sector_plazo(engine=None, desde: date = None) -> int:
    """
    Backfill de opciones_sector_pcr_plazo_diario. Reusa calcular_pcr_sector_plazo
    por fecha. Prerequisito: opciones_pcr_plazo_diario ya poblada.
    """
    eng = engine or get_engine()
    filtro = "WHERE fecha >= :desde" if desde else ""
    with eng.connect() as conn:
        fechas = conn.execute(
            text(f"SELECT DISTINCT fecha FROM opciones_pcr_plazo_diario {filtro} ORDER BY fecha"),
            {"desde": desde} if desde else {}
        ).fetchall()

    total = 0
    for (f,) in fechas:
        total += calcular_pcr_sector_plazo(f, engine=eng)
    return total
