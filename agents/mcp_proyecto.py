"""
MCP Server — Proyecto Indicadores y Machine Learning
Expone herramientas de consulta a la DB Railway para sesiones Claude Code.
Patron: engine por llamada -> query -> dispose (sin conexiones persistentes).
"""
import datetime
import decimal
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text

# Cargar credenciales desde el root del proyecto (parent de agents/)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env.local")
load_dotenv(_ROOT / ".env")

mcp = FastMCP("proyecto-tools")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_engine():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL no encontrada. Verificar .env.local")
    url = url.replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(
        url,
        connect_args={"sslmode": "require"},
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )


def _jv(val):
    """Convierte un valor DB a tipo JSON-serializable."""
    if val is None:
        return None
    if isinstance(val, (datetime.date, datetime.datetime)):
        return str(val)
    if isinstance(val, decimal.Decimal):
        return float(val)
    if isinstance(val, (int, float, str, bool)):
        return val
    return str(val)


def _row(row) -> dict:
    """Convierte una SQLAlchemy Row a dict serializable."""
    return {k: _jv(v) for k, v in row._mapping.items()}


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_pipeline_status() -> dict:
    """
    Estado actual del pipeline diario.
    Retorna: ultima fecha de precios, cantidad de tickers, ultimo scan y alertas de hoy.
    """
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            r = conn.execute(text("""
                SELECT MAX(fecha) AS ultima_fecha, COUNT(DISTINCT ticker) AS n_tickers
                FROM precios_diarios
            """)).fetchone()

            r2 = conn.execute(text("""
                SELECT alert_nivel, COUNT(*) AS n
                FROM alertas_scanner
                WHERE DATE(scan_fecha) = (SELECT MAX(DATE(scan_fecha)) FROM alertas_scanner)
                GROUP BY alert_nivel
            """)).fetchall()

            r3 = conn.execute(text("""
                SELECT MAX(DATE(scan_fecha)) AS ultimo_scan
                FROM alertas_scanner
            """)).fetchone()

        return {
            "ultima_fecha_precios": _jv(r.ultima_fecha) if r else None,
            "tickers_con_precios": int(r.n_tickers) if r else 0,
            "ultimo_scan": _jv(r3.ultimo_scan) if r3 else None,
            "alertas_ultimo_scan": {row.alert_nivel: int(row.n) for row in r2},
        }
    finally:
        engine.dispose()


@mcp.tool()
def get_alertas_hoy(nivel_min: str = "COMPRA", fecha: str = "") -> list:
    """
    Alertas del scanner ML.
    nivel_min: COMPRA_FUERTE | COMPRA | VENTA | NEUTRAL  (default: COMPRA)
    fecha: YYYY-MM-DD — si vacio usa la ultima fecha con alertas.
    Retorna lista con ticker, nivel, score, detalle, scan_fecha.
    """
    _PESO = {"COMPRA_FUERTE": 3, "COMPRA": 2, "VENTA": 1, "NEUTRAL": 0}
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            if not fecha:
                r = conn.execute(text(
                    "SELECT MAX(DATE(scan_fecha)) FROM alertas_scanner"
                )).fetchone()
                fecha = str(r[0]) if r and r[0] else None
            if not fecha:
                return []

            peso_min = _PESO.get(nivel_min.upper(), 2)
            niveles = [k for k, v in _PESO.items() if v >= peso_min]

            # Construir placeholders para IN
            placeholders = ", ".join(f":n{i}" for i in range(len(niveles)))
            params: dict = {"fecha": fecha}
            params.update({f"n{i}": n for i, n in enumerate(niveles)})

            rows = conn.execute(text(f"""
                SELECT ticker, alert_nivel, alert_score, alert_detalle, scan_fecha
                FROM alertas_scanner
                WHERE DATE(scan_fecha) = :fecha
                  AND alert_nivel IN ({placeholders})
                ORDER BY alert_score DESC
                LIMIT 50
            """), params).fetchall()

        return [_row(r) for r in rows]
    finally:
        engine.dispose()


@mcp.tool()
def get_opciones_resumen(ticker: str, dias: int = 5) -> list:
    """
    Resumen diario de opciones para un ticker: PCR volumen/OI, IV calls y puts.
    ticker: ej AAPL, NVDA
    dias: cuantos dias hacia atras (default 5, max 30)
    """
    dias = min(dias, 30)
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT fecha, pcr_vol, pcr_oi,
                       iv_call_avg, iv_put_avg,
                       call_vol, put_vol,
                       call_oi, put_oi
                FROM opciones_resumen_diario
                WHERE ticker = :ticker
                ORDER BY fecha DESC
                LIMIT :dias
            """), {"ticker": ticker.upper(), "dias": dias}).fetchall()
        return [_row(r) for r in rows]
    finally:
        engine.dispose()


@mcp.tool()
def get_precios_recientes(tickers: str, dias: int = 10) -> dict:
    """
    Precios OHLCV recientes para uno o varios tickers.
    tickers: separados por coma, ej "AAPL,NVDA,MSFT"
    dias: dias hacia atras por ticker (default 10, max 60)
    Retorna dict ticker -> lista de velas OHLCV ordenadas desc por fecha.
    """
    dias = min(dias, 60)
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return {}

    placeholders = ", ".join(f":t{i}" for i in range(len(ticker_list)))
    params: dict = {"limit": len(ticker_list) * dias}
    params.update({f"t{i}": t for i, t in enumerate(ticker_list)})

    engine = _get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT ticker, fecha, open, high, low, close, volume
                FROM precios_diarios
                WHERE ticker IN ({placeholders})
                ORDER BY ticker, fecha DESC
                LIMIT :limit
            """), params).fetchall()

        result: dict = {}
        counts: dict = {}
        for r in rows:
            t = r.ticker
            if t not in result:
                result[t] = []
                counts[t] = 0
            if counts[t] < dias:
                result[t].append(_row(r))
                counts[t] += 1
        return result
    finally:
        engine.dispose()


@mcp.tool()
def get_bt_resultados(ticker: str = "", estrategia_entrada: str = "", estrategia_salida: str = "") -> list:
    """
    Resultados de backtesting historico (resultados_bt_pa).
    Filtros opcionales: ticker, estrategia_entrada, estrategia_salida.
    Retorna top 30 por retorno_total_pct desc.
    """
    engine = _get_engine()
    try:
        conditions = []
        params: dict = {}
        if ticker:
            conditions.append("ticker = :ticker")
            params["ticker"] = ticker.upper()
        if estrategia_entrada:
            conditions.append("estrategia_entrada = :ee")
            params["ee"] = estrategia_entrada
        if estrategia_salida:
            conditions.append("estrategia_salida = :es")
            params["es"] = estrategia_salida

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT ticker, estrategia_entrada, estrategia_salida,
                       total_operaciones, win_rate, profit_factor,
                       retorno_total_pct, max_drawdown_pct,
                       retorno_promedio_pct, ganancias, perdidas
                FROM resultados_bt_pa
                {where}
                ORDER BY retorno_total_pct DESC
                LIMIT 30
            """), params).fetchall()

        return [_row(r) for r in rows]
    finally:
        engine.dispose()


@mcp.tool()
def get_ft_posiciones(estrategia: str = "") -> dict:
    """
    Posiciones abiertas y resumen de resultados del forward testing.
    estrategia: nombre exacto de la estrategia (opcional — si vacio devuelve todas).
    Retorna dict con posiciones_abiertas y resumen_estrategias.
    """
    engine = _get_engine()
    try:
        params: dict = {}
        where_join = ""
        if estrategia:
            where_join = "AND e.nombre = :estrategia"
            params["estrategia"] = estrategia

        with engine.connect() as conn:
            abiertas = conn.execute(text(f"""
                SELECT o.ticker, o.fecha_entrada, o.precio_entrada,
                       o.stop_loss, o.take_profit, o.pnl, o.pnl_pct,
                       e.nombre AS estrategia
                FROM ft_operaciones o
                JOIN ft_estrategias e ON o.estrategia_id = e.id
                WHERE o.fecha_salida IS NULL {where_join}
                ORDER BY o.fecha_entrada DESC
                LIMIT 20
            """), params).fetchall()

            resumen = conn.execute(text(f"""
                SELECT e.nombre AS estrategia,
                       COUNT(o.id) AS total_ops,
                       SUM(CASE WHEN o.pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       ROUND(AVG(o.pnl_pct)::numeric, 4) AS avg_pnl_pct,
                       ROUND(SUM(o.pnl)::numeric, 2) AS total_pnl
                FROM ft_operaciones o
                JOIN ft_estrategias e ON o.estrategia_id = e.id
                WHERE o.fecha_salida IS NOT NULL {where_join}
                GROUP BY e.nombre
                ORDER BY total_pnl DESC
            """), params).fetchall()

        return {
            "posiciones_abiertas": [_row(r) for r in abiertas],
            "resumen_estrategias": [_row(r) for r in resumen],
        }
    finally:
        engine.dispose()


@mcp.tool()
def get_indicadores_ticker(ticker: str) -> dict:
    """
    Indicadores tecnicos, features ML y z-scores para un ticker (ultimo dia disponible).
    ticker: simbolo del activo, ej AAPL
    Retorna: features_precio, features_market_structure, zscore, ultima_alerta.
    """
    t = ticker.strip().upper()
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            fpa = conn.execute(text("""
                SELECT * FROM features_precio_accion
                WHERE ticker = :t ORDER BY fecha DESC LIMIT 1
            """), {"t": t}).fetchone()

            fms = conn.execute(text("""
                SELECT * FROM features_market_structure
                WHERE ticker = :t ORDER BY fecha DESC LIMIT 1
            """), {"t": t}).fetchone()

            zs = conn.execute(text("""
                SELECT * FROM ticker_zscore_diario
                WHERE ticker = :t ORDER BY fecha DESC LIMIT 1
            """), {"t": t}).fetchone()

            alerta = conn.execute(text("""
                SELECT alert_nivel, alert_score, alert_detalle, scan_fecha
                FROM alertas_scanner
                WHERE ticker = :t ORDER BY scan_fecha DESC LIMIT 1
            """), {"t": t}).fetchone()

        return {
            "ticker": t,
            "features_precio": _row(fpa) if fpa else None,
            "features_market_structure": _row(fms) if fms else None,
            "zscore": _row(zs) if zs else None,
            "ultima_alerta": _row(alerta) if alerta else None,
        }
    finally:
        engine.dispose()


@mcp.tool()
def query_db(sql: str, limit: int = 100) -> list:
    """
    Query SELECT ad-hoc contra la DB Railway. Solo permite SELECT.
    sql: query SQL valida (solo SELECT)
    limit: max filas a retornar (max 200, default 100)
    Retorna lista de dicts con los resultados.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return [{"error": "Solo se permiten queries SELECT"}]

    limit = max(1, min(limit, 200))
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM ({sql}) _mcp_q LIMIT {limit}")
            ).fetchall()
        return [_row(r) for r in rows]
    except Exception as exc:
        return [{"error": str(exc)}]
    finally:
        engine.dispose()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
