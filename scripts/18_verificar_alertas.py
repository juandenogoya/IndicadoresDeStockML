"""
18_verificar_alertas.py
Verificacion post-facto de alertas del scanner.

Rellena los retornos reales en alertas_scanner comparando el precio de
cierre al momento de la alerta vs el precio real N dias habiles despues.

Horizontes:
    1d  -> primer dia habil despues de precio_fecha
    5d  -> 5 dias habiles (aprox 1 semana)
    20d -> 20 dias habiles (aprox 1 mes)

Fuente de precios:
    - Tickers en DB: consulta precios_diarios (primer fecha >= objetivo)
    - Tickers externos: descarga ventana puntual con yfinance

Logica de actualizacion:
    - Rellena cada horizonte en cuanto el tiempo ya paso
    - verificado=TRUE cuando retorno_20d_real queda rellenado
    - Idempotente: puede correrse diariamente sin problema

Uso:
    python scripts/18_verificar_alertas.py              # actualiza todo
    python scripts/18_verificar_alertas.py --reporte    # muestra tabla de aciertos
    python scripts/18_verificar_alertas.py --dias 5     # solo horizonte 5d
"""

import sys
import os
import argparse
import pandas as pd
import psycopg2.extras
from datetime import date, timedelta
from typing import Optional, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import query_df, get_connection


# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────

HORIZONTES = [1, 5, 20]          # dias habiles
MAX_OFFSET_EXTRA = 7              # dias calendario extra para buscar precio


# ─────────────────────────────────────────────────────────────
# Helpers de fechas
# ─────────────────────────────────────────────────────────────

def fecha_objetivo(base: date, n_dias_habiles: int) -> date:
    """
    Calcula la fecha objetivo N dias habiles despues de base.
    Usa pandas BDay (business days) que maneja fines de semana.
    No considera feriados especificos, pero es suficientemente preciso.
    """
    ts = pd.Timestamp(base) + pd.offsets.BDay(n_dias_habiles)
    return ts.date()


def dias_habiles_transcurridos(desde: date, hasta: date) -> int:
    """Cuenta dias habiles entre dos fechas (sin feriados)."""
    rng = pd.bdate_range(start=desde, end=hasta)
    return max(0, len(rng) - 1)   # -1 porque bdate_range incluye ambos extremos


# ─────────────────────────────────────────────────────────────
# Obtener precio de cierre en o despues de una fecha objetivo
# ─────────────────────────────────────────────────────────────

def _precio_desde_db(ticker: str, fecha_min: date, fecha_max: date) -> Optional[Dict]:
    """
    Busca el primer precio de cierre en [fecha_min, fecha_max] para el ticker en DB.
    Retorna dict {fecha, close} o None si no hay datos.
    """
    sql = """
        SELECT fecha, close
        FROM precios_diarios
        WHERE ticker = :ticker
          AND fecha >= :fecha_min
          AND fecha <= :fecha_max
          AND close > 0
        ORDER BY fecha ASC
        LIMIT 1
    """
    try:
        df = query_df(sql, params={
            "ticker":    ticker,
            "fecha_min": fecha_min,
            "fecha_max": fecha_max,
        })
        if df.empty:
            return None
        row = df.iloc[0]
        return {"fecha": row["fecha"], "close": float(row["close"])}
    except Exception:
        return None


_YF_CACHE: Dict[str, pd.DataFrame] = {}   # cache por ticker en la sesion


def _precio_desde_yfinance(ticker: str, fecha_min: date, fecha_max: date) -> Optional[Dict]:
    """
    Descarga o usa cache de precios yfinance para el ticker y
    busca el primer cierre en [fecha_min, fecha_max].
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    # Usar cache si ya descargamos este ticker
    if ticker not in _YF_CACHE:
        try:
            raw = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
            if raw.empty:
                _YF_CACHE[ticker] = pd.DataFrame()
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw = raw.reset_index()
            raw = raw.rename(columns={"date": "fecha", "Date": "fecha"})
            raw["fecha"] = pd.to_datetime(raw["fecha"]).dt.date
            _YF_CACHE[ticker] = raw[["fecha", "close"]].copy()
        except Exception:
            _YF_CACHE[ticker] = pd.DataFrame()
            return None

    df = _YF_CACHE[ticker]
    if df.empty:
        return None

    mask = (df["fecha"] >= fecha_min) & (df["fecha"] <= fecha_max)
    sub  = df[mask].sort_values("fecha")
    if sub.empty:
        return None

    row = sub.iloc[0]
    return {"fecha": row["fecha"], "close": float(row["close"])}


def obtener_precio(ticker: str, fecha_obj: date, en_db: bool) -> Optional[Dict]:
    """
    Obtiene el precio de cierre mas cercano en o despues de fecha_obj.
    Busca en una ventana de hasta MAX_OFFSET_EXTRA dias calendario.
    """
    fecha_max = fecha_obj + timedelta(days=MAX_OFFSET_EXTRA)
    today     = date.today()

    # No buscar precios futuros
    if fecha_obj > today:
        return None
    fecha_max = min(fecha_max, today)

    if en_db:
        resultado = _precio_desde_db(ticker, fecha_obj, fecha_max)
        if resultado:
            return resultado
        # Fallback a yfinance (puede que haya actualizado mas reciente)

    return _precio_desde_yfinance(ticker, fecha_obj, fecha_max)


# ─────────────────────────────────────────────────────────────
# Cargar alertas pendientes de verificacion
# ─────────────────────────────────────────────────────────────

def cargar_alertas_pendientes(solo_horizonte: Optional[int] = None) -> pd.DataFrame:
    """
    Carga alertas donde al menos un horizonte todavia no fue rellenado
    y el tiempo necesario ya transcurrio.
    """
    today = date.today()

    # Construir condicion dinamica segun que horizontes queremos verificar
    condiciones = []
    if solo_horizonte is None or solo_horizonte == 1:
        # Verificar 1d: ya paso 1 dia habil desde precio_fecha
        condiciones.append("(precio_1d_real IS NULL AND precio_fecha <= :cutoff_1d)")
    if solo_horizonte is None or solo_horizonte == 5:
        condiciones.append("(precio_5d_real IS NULL AND precio_fecha <= :cutoff_5d)")
    if solo_horizonte is None or solo_horizonte == 20:
        condiciones.append("(precio_20d_real IS NULL AND precio_fecha <= :cutoff_20d)")

    if not condiciones:
        return pd.DataFrame()

    where_cond = " OR ".join(condiciones)

    sql = f"""
        SELECT id, ticker, precio_fecha, precio_cierre,
               precio_1d_real, precio_5d_real, precio_20d_real,
               retorno_1d_real, retorno_5d_real, retorno_20d_real,
               verificado
        FROM alertas_scanner
        WHERE ({where_cond})
          AND precio_fecha IS NOT NULL
          AND precio_cierre IS NOT NULL
          AND precio_cierre > 0
        ORDER BY precio_fecha ASC, id ASC
    """
    cutoff_1d  = (pd.Timestamp(today) - pd.offsets.BDay(1)).date()
    cutoff_5d  = (pd.Timestamp(today) - pd.offsets.BDay(5)).date()
    cutoff_20d = (pd.Timestamp(today) - pd.offsets.BDay(20)).date()

    try:
        df = query_df(sql, params={
            "cutoff_1d":  cutoff_1d,
            "cutoff_5d":  cutoff_5d,
            "cutoff_20d": cutoff_20d,
        })
        return df
    except Exception as e:
        print(f"  [ERROR] No se pudieron cargar alertas: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Verificar si el ticker tiene precios en DB
# ─────────────────────────────────────────────────────────────

_EN_DB_CACHE: Dict[str, bool] = {}


def ticker_en_db(ticker: str) -> bool:
    if ticker in _EN_DB_CACHE:
        return _EN_DB_CACHE[ticker]
    sql = "SELECT 1 FROM activos WHERE ticker = :t LIMIT 1"
    try:
        df = query_df(sql, params={"t": ticker})
        result = not df.empty
    except Exception:
        result = False
    _EN_DB_CACHE[ticker] = result
    return result


# ─────────────────────────────────────────────────────────────
# Procesar y actualizar registros
# ─────────────────────────────────────────────────────────────

def verificar_alertas(solo_horizonte: Optional[int] = None) -> int:
    """
    Rellena retornos post-facto para todas las alertas pendientes.

    Returns:
        Numero de registros actualizados.
    """
    df = cargar_alertas_pendientes(solo_horizonte)

    if df.empty:
        print("  No hay alertas pendientes de verificacion.")
        return 0

    print(f"  Alertas pendientes: {len(df)}")
    actualizados = 0

    updates = []   # lista de dicts para el batch update

    for _, row in df.iterrows():
        alerta_id     = int(row["id"])
        ticker        = str(row["ticker"])
        precio_fecha  = row["precio_fecha"]
        precio_cierre = float(row["precio_cierre"])
        en_db         = ticker_en_db(ticker)

        if isinstance(precio_fecha, str):
            precio_fecha = date.fromisoformat(precio_fecha)
        elif hasattr(precio_fecha, "date"):
            precio_fecha = precio_fecha.date()

        update = {"id": alerta_id}

        # ── Horizonte 1d ──────────────────────────────────────
        if pd.isna(row["precio_1d_real"]):
            obj_1d = fecha_objetivo(precio_fecha, 1)
            if obj_1d <= date.today():
                p = obtener_precio(ticker, obj_1d, en_db)
                if p:
                    update["precio_1d_real"]  = round(p["close"], 4)
                    update["retorno_1d_real"] = round((p["close"] / precio_cierre - 1) * 100, 4)

        # ── Horizonte 5d ──────────────────────────────────────
        if pd.isna(row["precio_5d_real"]):
            obj_5d = fecha_objetivo(precio_fecha, 5)
            if obj_5d <= date.today():
                p = obtener_precio(ticker, obj_5d, en_db)
                if p:
                    update["precio_5d_real"]  = round(p["close"], 4)
                    update["retorno_5d_real"] = round((p["close"] / precio_cierre - 1) * 100, 4)

        # ── Horizonte 20d ─────────────────────────────────────
        if pd.isna(row["precio_20d_real"]):
            obj_20d = fecha_objetivo(precio_fecha, 20)
            if obj_20d <= date.today():
                p = obtener_precio(ticker, obj_20d, en_db)
                if p:
                    update["precio_20d_real"]  = round(p["close"], 4)
                    update["retorno_20d_real"] = round((p["close"] / precio_cierre - 1) * 100, 4)

        # ── Marcar verificado si ya tenemos retorno_20d ───────
        tiene_20d = (
            "retorno_20d_real" in update
            or not pd.isna(row.get("retorno_20d_real", float("nan")))
        )
        if tiene_20d:
            update["verificado"] = True

        if len(update) > 1:   # tiene algo ademas de 'id'
            updates.append(update)

    if not updates:
        print("  Nada nuevo para actualizar (precios futuros o datos no disponibles).")
        return 0

    # ── Persistir en DB (update por id) ─────────────────────
    actualizados = _persistir_actualizaciones(updates)
    print(f"  Registros actualizados: {actualizados}")
    return actualizados


def _persistir_actualizaciones(updates: List[Dict]) -> int:
    """Ejecuta los UPDATE en alertas_scanner para cada dict de updates."""
    count = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for upd in updates:
                alerta_id = upd.pop("id")
                if not upd:
                    continue

                set_clauses = ", ".join(f"{k} = %({k})s" for k in upd.keys())
                sql = f"""
                    UPDATE alertas_scanner
                    SET {set_clauses}
                    WHERE id = %(id)s
                """
                upd["id"] = alerta_id
                cur.execute(sql, upd)
                count += 1

    return count


# ─────────────────────────────────────────────────────────────
# Reporte de aciertos
# ─────────────────────────────────────────────────────────────

def imprimir_reporte():
    """
    Muestra un resumen estadistico de la precision del scanner
    usando las alertas ya verificadas.
    """
    sql = """
        SELECT
            alert_nivel,
            COUNT(*)                                              AS n_alertas,
            ROUND(AVG(retorno_1d_real)::NUMERIC,  2)             AS ret_1d_avg,
            ROUND(AVG(retorno_5d_real)::NUMERIC,  2)             AS ret_5d_avg,
            ROUND(AVG(retorno_20d_real)::NUMERIC, 2)             AS ret_20d_avg,
            ROUND(100.0 * SUM(CASE WHEN retorno_20d_real > 1 THEN 1 ELSE 0 END)
                        / NULLIF(COUNT(retorno_20d_real), 0), 1) AS win_rate_20d_pct,
            ROUND(AVG(ml_prob_ganancia)::NUMERIC * 100, 1)       AS ml_prob_avg_pct
        FROM alertas_scanner
        WHERE verificado = TRUE
        GROUP BY alert_nivel
        ORDER BY
            CASE alert_nivel
                WHEN 'COMPRA_FUERTE' THEN 1
                WHEN 'COMPRA'        THEN 2
                WHEN 'NEUTRAL'       THEN 3
                WHEN 'VENTA'         THEN 4
                WHEN 'VENTA_FUERTE'  THEN 5
                ELSE 6
            END
    """
    df = query_df(sql)

    if df.empty:
        print("\n  Sin alertas verificadas todavia. Ejecuta el script en unos dias.")
        return

    print("\n" + "=" * 75)
    print("  REPORTE DE ACIERTOS DEL SCANNER (alertas verificadas)")
    print("=" * 75)
    print(f"  {'NIVEL':<16}  {'N':>4}  {'Ret1d%':>7}  {'Ret5d%':>7}  "
          f"{'Ret20d%':>8}  {'WinRate20d%':>11}  {'ML%':>5}")
    print("-" * 75)

    for _, row in df.iterrows():
        nivel    = str(row["alert_nivel"])[:16]
        n        = int(row["n_alertas"])
        r1d      = f"{float(row['ret_1d_avg']):.2f}%"   if row["ret_1d_avg"]  is not None else "  N/A"
        r5d      = f"{float(row['ret_5d_avg']):.2f}%"   if row["ret_5d_avg"]  is not None else "  N/A"
        r20d     = f"{float(row['ret_20d_avg']):.2f}%"  if row["ret_20d_avg"] is not None else "  N/A"
        wr       = f"{float(row['win_rate_20d_pct']):.1f}%" if row["win_rate_20d_pct"] is not None else "  N/A"
        ml_avg   = f"{float(row['ml_prob_avg_pct']):.1f}%"  if row["ml_prob_avg_pct"] is not None else "  N/A"
        print(f"  {nivel:<16}  {n:>4}  {r1d:>7}  {r5d:>7}  {r20d:>8}  {wr:>11}  {ml_avg:>5}")

    print("=" * 75)
    print("  WinRate20d = % alertas con retorno_20d > +1%")

    # ── Detalle por ticker ─────────────────────────────────────
    sql2 = """
        SELECT ticker, alert_nivel,
               ROUND(retorno_20d_real::NUMERIC, 2)  AS ret_20d,
               ROUND(ml_prob_ganancia::NUMERIC * 100, 1) AS ml_pct,
               alert_score,
               precio_fecha
        FROM alertas_scanner
        WHERE verificado = TRUE
        ORDER BY precio_fecha DESC, alert_nivel, ticker
        LIMIT 50
    """
    df2 = query_df(sql2)
    if not df2.empty:
        print(f"\n  Ultimas {len(df2)} alertas verificadas:")
        print(f"  {'FECHA':<12} {'TICKER':<7} {'NIVEL':<14} "
              f"{'SCORE':>5} {'ML%':>5} {'Ret20d%':>8}")
        print("  " + "-" * 55)
        for _, r in df2.iterrows():
            fecha  = str(r["precio_fecha"])[:10]
            ticker = str(r["ticker"])
            nivel  = str(r["alert_nivel"])[:14]
            score  = f"{float(r['alert_score']):.0f}" if r["alert_score"] is not None else "?"
            ml_p   = f"{float(r['ml_pct']):.0f}%"     if r["ml_pct"]     is not None else "?"
            ret20  = f"{float(r['ret_20d']):+.2f}%"   if r["ret_20d"]    is not None else "N/A"
            print(f"  {fecha:<12} {ticker:<7} {nivel:<14} "
                  f"{score:>5} {ml_p:>5} {ret20:>8}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verificacion post-facto de alertas del scanner."
    )
    parser.add_argument(
        "--reporte", action="store_true",
        help="Muestra tabla de aciertos del scanner (alertas ya verificadas)"
    )
    parser.add_argument(
        "--dias", type=int, choices=[1, 5, 20], default=None,
        help="Verificar solo el horizonte indicado (1, 5 o 20 dias)"
    )
    parser.add_argument(
        "--solo-reporte", action="store_true",
        help="Solo muestra el reporte, sin actualizar precios"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  VERIFICACION POST-FACTO DE ALERTAS")
    print("=" * 60)

    if not args.solo_reporte:
        horizonte_txt = f"{args.dias}d" if args.dias else "1d + 5d + 20d"
        print(f"\n[1/2] Actualizando precios reales (horizontes: {horizonte_txt})...")
        n = verificar_alertas(solo_horizonte=args.dias)
        print(f"  Total: {n} alertas actualizadas.")
    else:
        print("\n  Modo solo-reporte: no se actualizan precios.")

    if args.reporte or args.solo_reporte:
        print("\n[2/2] Generando reporte de aciertos...")
        imprimir_reporte()
    else:
        print("\n  Para ver el reporte de aciertos ejecuta:")
        print("  python scripts/18_verificar_alertas.py --reporte")

    print("\nVerificacion completada.")


if __name__ == "__main__":
    main()
