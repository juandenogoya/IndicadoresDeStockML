"""
push_senales_bot.py
Productor de la tabla masticada `senales_bot_diaria` (Plan B, Tarea 16).

QUE HACE
    Arma 1 fila por ticker del universo tradeable (activos.activo=TRUE) con las
    senales pre-computadas que consumen los 3 bots Alpaca (rediseno Plan B):
      - Bot 1 ML            -> alert_nivel, alert_score
      - Bot 2 TECH_SECTOR   -> close, sector, sma21/50/200, rsi14, macd, macd_signal, atr14
      - Bot 3 OPTIONS_v2    -> pcr_score, pcr_valido, pcr_corto/medio/largo (PCR_VOL)
    y la UPSERTea a Railway. El bot "solo opera": lee esta tabla, no las crudas.

CONEXION DUAL (el punto fino)
    - LEE de la DB LOCAL (fuente de verdad bajo Plan C): indicadores_tecnicos,
      precios_diarios, activos, alertas_scanner, opciones_snapshot.
    - ESCRIBE en RAILWAY (senales_bot_diaria), construyendo un engine dedicado
      desde DATABASE_URL de .env.local (no via get_engine() global).

CUANDO CORRERLO
    DESPUES de ft_run_diario.bat: ahi la data local ya esta fresca, incluido
    opciones_snapshot (que ft_run_diario sincroniza de Railway en su paso [0]).
    Es HERMANO de ft_run_diario, no downstream: sube SENALES, no las decisiones
    de FT. Standalone: correrlo solo re-pushea sin re-correr FT (requiere que la
    data local del dia ya este cargada).

NOTA (Paso 2 de Tarea 16)
    _pcr_vol_por_ventanas() recibe el engine como parametro: es la semilla del
    modulo de decision compartido FT<->Alpaca. La logica/umbrales replican
    obtener_pcr_vol_por_ventanas() de ft_bot_tech_sectorial_options_v2.py.

Uso:
    python scripts/push_senales_bot.py
    python scripts/push_senales_bot.py --dry-run     (no escribe en Railway)
"""

import os
import sys
import argparse
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, text


# ── Parametros de opciones (PCR sobre VOLUMEN diario) ─────────────────────────
# Replica de ft_bot_tech_sectorial_options_v2.py (se unifica en Paso 2).
VENTANA_CORTO_MIN, VENTANA_CORTO_MAX = 1, 14
VENTANA_MEDIO_MIN, VENTANA_MEDIO_MAX = 15, 45
VENTANA_LARGO_MIN, VENTANA_LARGO_MAX = 46, 90

PCR_VOL_UMBRAL_ALCISTA = 1.0   # < 1.0 = Alcista (A), >= 1.0 = Bajista (B)
MIN_VOL_POR_VENTANA    = 500   # volumen minimo (call+put) para ventana valida


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ── Conexiones ────────────────────────────────────────────────────────────────

def _leer_database_url_local() -> str:
    """
    Lee DATABASE_URL (Railway) directamente de .env.local SIN cargarlo en
    os.environ (asi get_engine() sigue cayendo a la DB local para los reads).
    """
    env_path = os.path.join(ROOT, ".env.local")
    if not os.path.exists(env_path):
        raise RuntimeError(
            ".env.local no encontrado: necesario para el destino Railway "
            "(DATABASE_URL). El productor lee local y escribe Railway."
        )
    with open(env_path, "r", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            if linea.startswith("DATABASE_URL"):
                # DATABASE_URL=postgres://...  (puede venir con o sin comillas)
                valor = linea.split("=", 1)[1].strip().strip('"').strip("'")
                return valor
    raise RuntimeError("DATABASE_URL no esta definida en .env.local.")


def _engine_railway():
    """Engine SQLAlchemy apuntando a Railway (mismo normalizado que get_engine)."""
    url = _leer_database_url_local().strip()
    url = url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url, connect_args={"sslmode": "require"}, pool_pre_ping=True)


def _engine_local():
    """Engine SQLAlchemy apuntando a la DB local (Plan C: fuente de verdad)."""
    # Reutiliza la logica de ft_env: carga .env del proyecto y elimina
    # DATABASE_URL de os.environ -> get_engine() cae a DB_CONFIG local.
    from scripts.forward_testing.ft_env import configurar_entorno_local
    configurar_entorno_local()
    from src.data.database import get_engine
    return get_engine()


# ── DDL: tabla masticada ──────────────────────────────────────────────────────

DDL_SENALES = """
CREATE TABLE IF NOT EXISTS senales_bot_diaria (
    ticker       VARCHAR(20)  NOT NULL,
    fecha        DATE         NOT NULL,
    close        NUMERIC(14,4),
    sector       VARCHAR(60),
    alert_nivel  VARCHAR(30),
    alert_score  NUMERIC(7,2),
    sma21        NUMERIC(14,4),
    sma50        NUMERIC(14,4),
    sma200       NUMERIC(14,4),
    rsi14        NUMERIC(7,4),
    macd         NUMERIC(14,6),
    macd_signal  NUMERIC(14,6),
    atr14        NUMERIC(14,6),
    pcr_score    SMALLINT,
    pcr_valido   BOOLEAN,
    pcr_corto    CHAR(1),
    pcr_medio    CHAR(1),
    pcr_largo    CHAR(1),
    updated_at   TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (ticker, fecha)
);
"""


def ensure_table(eng):
    """Crea senales_bot_diaria en Railway si no existe (idempotente)."""
    with eng.begin() as conn:
        conn.execute(text(DDL_SENALES))


# ── Lectura LOCAL: tecnico + close + sector ───────────────────────────────────

def obtener_tecnico_local(eng) -> list[dict]:
    """
    Ultimo dia de indicadores por ticker (universo activo) + close + sector.
    1 fila por ticker (DISTINCT ON), fecha = ultima fecha de indicadores.
    """
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (i.ticker)
                   i.ticker, i.fecha,
                   p.close,
                   COALESCE(a.sector, 'Unknown') AS sector,
                   i.sma21, i.sma50, i.sma200,
                   i.rsi14, i.macd, i.macd_signal, i.atr14
            FROM indicadores_tecnicos i
            JOIN precios_diarios p ON p.ticker = i.ticker AND p.fecha = i.fecha
            JOIN activos a         ON a.ticker = i.ticker
            WHERE a.activo = TRUE
            ORDER BY i.ticker, i.fecha DESC
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


# ── Lectura LOCAL: alertas ML (ultimo scan por ticker) ────────────────────────

def obtener_alertas_local(eng) -> dict[str, dict]:
    """Ultima alerta del scanner por ticker (alert_nivel, alert_score)."""
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (ticker)
                   ticker, alert_nivel, alert_score
            FROM alertas_scanner
            ORDER BY ticker, scan_fecha DESC
        """)).fetchall()
    return {r.ticker: dict(r._mapping) for r in rows}


# ── Lectura LOCAL: PCR_VOL por ventana ────────────────────────────────────────

def _pcr_vol_por_ventanas(eng, tickers: list[str]) -> dict[str, dict]:
    """
    PCR_VOL por ventana de vencimiento (corto/medio/largo) usando el ultimo
    fecha_snapshot disponible en opciones_snapshot (LOCAL bajo Plan C).

    Veredicto por ventana:
        PCR_VOL < 1.0  -> 'A' (Alcista)
        PCR_VOL >= 1.0 -> 'B' (Bajista)
        VOL < MIN_VOL  -> None (sin liquidez)

    Retorna {ticker: {corto, medio, largo, pcr_score(0-3), valido(bool)}}.
    Recibe el engine como parametro (semilla del modulo compartido, Paso 2).
    """
    if not tickers:
        return {}

    with eng.connect() as conn:
        row_fecha = conn.execute(text("""
            SELECT MAX(fecha_snapshot) AS max_fecha
            FROM opciones_snapshot
            WHERE ticker = ANY(:tickers)
        """), {"tickers": tickers}).fetchone()

        if not row_fecha or not row_fecha.max_fecha:
            log("  [WARN] Sin datos de opciones_snapshot para los tickers dados.")
            return {}

        fecha_snap = row_fecha.max_fecha

        rows = conn.execute(text("""
            SELECT
                ticker,
                (vencimiento - :fecha_snap) AS dias_al_venc,
                SUM(CASE WHEN tipo = 'call' THEN COALESCE(volumen, 0) ELSE 0 END) AS call_vol,
                SUM(CASE WHEN tipo = 'put'  THEN COALESCE(volumen, 0) ELSE 0 END) AS put_vol
            FROM opciones_snapshot
            WHERE ticker = ANY(:tickers)
              AND fecha_snapshot = :fecha_snap
              AND (vencimiento - :fecha_snap) BETWEEN :vmin AND :vmax
            GROUP BY ticker, (vencimiento - :fecha_snap)
        """), {
            "tickers":    tickers,
            "fecha_snap": fecha_snap,
            "vmin":       VENTANA_CORTO_MIN,
            "vmax":       VENTANA_LARGO_MAX,
        }).fetchall()

    acum: dict[str, dict] = {}
    for r in rows:
        t    = r.ticker
        dias = int(r.dias_al_venc)
        cv   = int(r.call_vol or 0)
        pv   = int(r.put_vol  or 0)
        if t not in acum:
            acum[t] = {"corto_call": 0, "corto_put": 0, "medio_call": 0,
                       "medio_put": 0, "largo_call": 0, "largo_put": 0}
        if VENTANA_CORTO_MIN <= dias <= VENTANA_CORTO_MAX:
            acum[t]["corto_call"] += cv
            acum[t]["corto_put"]  += pv
        elif VENTANA_MEDIO_MIN <= dias <= VENTANA_MEDIO_MAX:
            acum[t]["medio_call"] += cv
            acum[t]["medio_put"]  += pv
        elif VENTANA_LARGO_MIN <= dias <= VENTANA_LARGO_MAX:
            acum[t]["largo_call"] += cv
            acum[t]["largo_put"]  += pv

    def _veredicto(call_vol, put_vol):
        total = call_vol + put_vol
        if total < MIN_VOL_POR_VENTANA or call_vol == 0:
            return None
        pcr = put_vol / call_vol
        return "A" if pcr < PCR_VOL_UMBRAL_ALCISTA else "B"

    resultado = {}
    for t in tickers:
        datos = acum.get(t)
        if not datos:
            resultado[t] = {"corto": None, "medio": None, "largo": None,
                            "pcr_score": 0, "valido": False}
            continue
        v_corto = _veredicto(datos["corto_call"], datos["corto_put"])
        v_medio = _veredicto(datos["medio_call"], datos["medio_put"])
        v_largo = _veredicto(datos["largo_call"], datos["largo_put"])
        valido    = all(v is not None for v in (v_corto, v_medio, v_largo))
        pcr_score = sum(1 for v in (v_corto, v_medio, v_largo) if v == "A")
        resultado[t] = {"corto": v_corto, "medio": v_medio, "largo": v_largo,
                        "pcr_score": pcr_score, "valido": valido}
    return resultado


# ── Ensamblado + UPSERT ───────────────────────────────────────────────────────

UPSERT_SQL = """
INSERT INTO senales_bot_diaria
    (ticker, fecha, close, sector, alert_nivel, alert_score,
     sma21, sma50, sma200, rsi14, macd, macd_signal, atr14,
     pcr_score, pcr_valido, pcr_corto, pcr_medio, pcr_largo, updated_at)
VALUES
    (:ticker, :fecha, :close, :sector, :alert_nivel, :alert_score,
     :sma21, :sma50, :sma200, :rsi14, :macd, :macd_signal, :atr14,
     :pcr_score, :pcr_valido, :pcr_corto, :pcr_medio, :pcr_largo, NOW())
ON CONFLICT (ticker, fecha) DO UPDATE SET
    close       = EXCLUDED.close,
    sector      = EXCLUDED.sector,
    alert_nivel = EXCLUDED.alert_nivel,
    alert_score = EXCLUDED.alert_score,
    sma21       = EXCLUDED.sma21,
    sma50       = EXCLUDED.sma50,
    sma200      = EXCLUDED.sma200,
    rsi14       = EXCLUDED.rsi14,
    macd        = EXCLUDED.macd,
    macd_signal = EXCLUDED.macd_signal,
    atr14       = EXCLUDED.atr14,
    pcr_score   = EXCLUDED.pcr_score,
    pcr_valido  = EXCLUDED.pcr_valido,
    pcr_corto   = EXCLUDED.pcr_corto,
    pcr_medio   = EXCLUDED.pcr_medio,
    pcr_largo   = EXCLUDED.pcr_largo,
    updated_at  = NOW()
"""


def construir_filas(tecnico, alertas, pcr_map) -> list[dict]:
    """Une las 3 fuentes en 1 fila por ticker (forma = columnas de la masticada)."""
    filas = []
    for t in tecnico:
        ticker = t["ticker"]
        al  = alertas.get(ticker, {})
        pcr = pcr_map.get(ticker, {})
        filas.append({
            "ticker":      ticker,
            "fecha":       t["fecha"],
            "close":       t["close"],
            "sector":      t["sector"],
            "alert_nivel": al.get("alert_nivel"),
            "alert_score": al.get("alert_score"),
            "sma21":       t["sma21"],
            "sma50":       t["sma50"],
            "sma200":      t["sma200"],
            "rsi14":       t["rsi14"],
            "macd":        t["macd"],
            "macd_signal": t["macd_signal"],
            "atr14":       t["atr14"],
            "pcr_score":   pcr.get("pcr_score", 0),
            "pcr_valido":  pcr.get("valido", False),
            "pcr_corto":   pcr.get("corto"),
            "pcr_medio":   pcr.get("medio"),
            "pcr_largo":   pcr.get("largo"),
        })
    return filas


def upsert_railway(eng, filas: list[dict]):
    """UPSERT de todas las filas en una transaccion."""
    with eng.begin() as conn:
        conn.execute(text(UPSERT_SQL), filas)


# ── Runner ────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    sep = "-" * 62
    log(sep)
    log(f"PUSH senales_bot_diaria {'[DRY RUN]' if dry_run else ''}")
    log(sep)

    eng_local = _engine_local()
    log("Engine local OK (lectura).")

    # 1. Tecnico + close + sector (universo activo)
    tecnico = obtener_tecnico_local(eng_local)
    if not tecnico:
        log("[ERROR] Sin indicadores locales para el universo activo. Abort.")
        return
    fechas = {str(t["fecha"]) for t in tecnico}
    log(f"Tecnico: {len(tecnico)} tickers | fecha(s) datos: {sorted(fechas)}")

    # 2. Alertas ML (ultimo scan por ticker)
    alertas = obtener_alertas_local(eng_local)
    log(f"Alertas scanner: {len(alertas)} tickers con alerta")

    # 3. PCR_VOL por ventana
    tickers = [t["ticker"] for t in tecnico]
    pcr_map = _pcr_vol_por_ventanas(eng_local, tickers)
    n_validos = sum(1 for v in pcr_map.values() if v.get("valido"))
    log(f"PCR_VOL: {len(pcr_map)} tickers | {n_validos} con liquidez en las 3 ventanas")

    # 4. Ensamblar
    filas = construir_filas(tecnico, alertas, pcr_map)
    n_alert  = sum(1 for f in filas if f["alert_nivel"])
    n_cf     = sum(1 for f in filas if f["alert_nivel"] == "COMPRA_FUERTE")
    log(f"Filas armadas: {len(filas)} | con alerta: {n_alert} | COMPRA_FUERTE: {n_cf}")

    if dry_run:
        log("[DRY RUN] No se escribe en Railway. Muestra de 3 filas:")
        for f in filas[:3]:
            log(f"  {f['ticker']:6} | close={f['close']} | sec={f['sector']:22} | "
                f"alert={f['alert_nivel']}/{f['alert_score']} | "
                f"pcr={f['pcr_score']}/3 valido={f['pcr_valido']} "
                f"({f['pcr_corto']}/{f['pcr_medio']}/{f['pcr_largo']})")
        log("Completado (dry-run).")
        log(sep)
        return

    # 5. Destino Railway: asegurar tabla + UPSERT
    eng_rail = _engine_railway()
    ensure_table(eng_rail)
    log("Tabla senales_bot_diaria asegurada en Railway.")
    upsert_railway(eng_rail, filas)
    log(f"UPSERT OK: {len(filas)} filas en Railway.")
    log("Completado.")
    log(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Productor de senales_bot_diaria (local -> Railway)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Arma las filas y reporta, sin escribir en Railway")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
