"""
37_scanner_ar.py
Scanner diario para el mercado argentino (BCBA).

Flujo:
  1. Fetch de precios historicos desde IOL REST API (3 anos de historia)
  2. Calculo de indicadores tecnicos en ARS (sin conversion de moneda)
  3. Calculo de estructura de mercado (BOS/CHoCH N=10)
  4. Generacion de alert_score y alert_nivel
  5. Persistencia en alertas_scanner_ar (ON CONFLICT = upsert diario)

Diseño: todos los calculos en ARS original. IOL es la fuente de verdad.

Uso:
  python scripts/37_scanner_ar.py              # todos los tickers
  python scripts/37_scanner_ar.py --ticker GGAL YPFD
  python scripts/37_scanner_ar.py --dry-run    # muestra sin guardar
  python scripts/37_scanner_ar.py --status     # muestra ultimas alertas en DB
"""

import sys
import os
import argparse
import time
import warnings
from datetime import datetime, date, timedelta

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

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2.extras
from sqlalchemy import text
from src.data.database import get_engine, get_connection
from src.data.iol_client import IOLClient


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Indicadores tecnicos ──────────────────────────────────────────────────────

def calcular_sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def calcular_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calcular_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal


def calcular_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def calcular_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    plus_dm  = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm.values]  = 0
    minus_dm[minus_dm < plus_dm.values] = 0
    atr = calcular_atr(df, n)
    plus_di  = 100 * plus_dm.rolling(n).mean()  / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(n).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean()


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de indicadores a un DataFrame OHLCV."""
    df = df.copy()
    c = df["close"]

    df["sma20"]  = calcular_sma(c, 20)
    df["sma50"]  = calcular_sma(c, 50)
    df["sma200"] = calcular_sma(c, 200)

    df["dist_sma20"]  = (c - df["sma20"])  / df["sma20"]  * 100
    df["dist_sma50"]  = (c - df["sma50"])  / df["sma50"]  * 100
    df["dist_sma200"] = (c - df["sma200"]) / df["sma200"] * 100

    df["rsi14"] = calcular_rsi(c, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calcular_macd(c)
    df["atr14"] = calcular_atr(df, 14)
    df["adx"]   = calcular_adx(df, 14)

    return df


# ── Estructura de mercado (BOS/CHoCH N=10) ───────────────────────────────────

def detectar_swings(highs: np.ndarray, lows: np.ndarray, n: int = 10):
    """Retorna arrays con posiciones de swing highs y swing lows."""
    sh_idx, sl_idx = [], []
    for i in range(n, len(highs) - n):
        if highs[i] == max(highs[i-n:i+n+1]):
            sh_idx.append(i)
        if lows[i]  == min(lows[i-n:i+n+1]):
            sl_idx.append(i)
    return sh_idx, sl_idx


def calcular_estructura(df: pd.DataFrame, n: int = 10) -> dict:
    """Detecta BOS/CHoCH y retorna dict con features de estructura."""
    highs = df["high"].values
    lows  = df["low"].values
    closes = df["close"].values
    sh_idx, sl_idx = detectar_swings(highs, lows, n)

    ultimo_close = closes[-1]
    resultado = {
        "estructura_10": 0,
        "bos_bull_10":   0,
        "bos_bear_10":   0,
        "choch_bull_10": 0,
        "choch_bear_10": 0,
    }

    if len(sh_idx) < 2 or len(sl_idx) < 2:
        return resultado

    # Ultimos 2 swing highs y lows
    sh1, sh2 = highs[sh_idx[-1]], highs[sh_idx[-2]]
    sl1, sl2 = lows[sl_idx[-1]],  lows[sl_idx[-2]]

    # Estructura: HH/HL = alcista (1), LH/LL = bajista (-1)
    if sh1 > sh2 and sl1 > sl2:
        resultado["estructura_10"] = 1   # alcista
    elif sh1 < sh2 and sl1 < sl2:
        resultado["estructura_10"] = -1  # bajista

    # BOS alcista: cierre por encima del ultimo swing high
    if ultimo_close > highs[sh_idx[-1]]:
        resultado["bos_bull_10"] = 1
    # BOS bajista: cierre por debajo del ultimo swing low
    if ultimo_close < lows[sl_idx[-1]]:
        resultado["bos_bear_10"] = 1
    # CHoCH alcista: estructura era bajista y rompe al alza
    if resultado["estructura_10"] == -1 and resultado["bos_bull_10"]:
        resultado["choch_bull_10"] = 1
    # CHoCH bajista: estructura era alcista y rompe a la baja
    if resultado["estructura_10"] == 1 and resultado["bos_bear_10"]:
        resultado["choch_bear_10"] = 1

    return resultado


# ── Scoring y nivel de alerta ─────────────────────────────────────────────────

def calcular_score(row: pd.Series, estructura: dict) -> tuple:
    """
    Retorna (alert_score 0-100, condiciones_ok, score_ponderado 0-1).
    Pesos calibrados para mercado argentino en ARS.
    """
    condiciones = {
        "rsi":       30 <= row["rsi14"] <= 68,
        "macd":      row["macd_hist"] > 0,
        "sma20":     row["close"] > row["sma20"],
        "sma50":     row["close"] > row["sma50"],
        "sma200":    row["close"] > row["sma200"],
        "adx":       row["adx"] > 20,
        "estructura": estructura["estructura_10"] == 1,
        "bos_bull":  estructura["bos_bull_10"] == 1,
    }
    pesos = {
        "rsi": 0.15, "macd": 0.15, "sma20": 0.10,
        "sma50": 0.15, "sma200": 0.15, "adx": 0.10,
        "estructura": 0.10, "bos_bull": 0.10,
    }
    score_pond = sum(pesos[k] for k, v in condiciones.items() if v)
    conds_ok   = sum(1 for v in condiciones.values() if v)
    alert_score = round(score_pond * 100, 1)
    return alert_score, conds_ok, round(score_pond, 4)


def nivel_alerta(score: float) -> str:
    if score >= 80:
        return "COMPRA_FUERTE"
    if score >= 60:
        return "COMPRA"
    if score >= 35:
        return "NEUTRAL"
    return "VENTA"


def generar_detalle(row: pd.Series, estructura: dict, score: float) -> str:
    partes = []
    if row["rsi14"] < 35:
        partes.append(f"RSI oversold ({row['rsi14']:.1f})")
    elif row["rsi14"] > 65:
        partes.append(f"RSI overbought ({row['rsi14']:.1f})")
    if row["macd_hist"] > 0:
        partes.append("MACD positivo")
    if row["close"] > row["sma200"]:
        partes.append("sobre SMA200")
    if estructura["bos_bull_10"]:
        partes.append("BOS alcista")
    if estructura["choch_bull_10"]:
        partes.append("CHoCH alcista")
    if estructura["bos_bear_10"]:
        partes.append("BOS bajista")
    return " | ".join(partes) if partes else "Sin senales destacadas"


# ── Analisis de un ticker ─────────────────────────────────────────────────────

def analizar_ticker(client: IOLClient, ticker: str, sector: str) -> dict | None:
    """Retorna dict listo para insertar en alertas_scanner_ar, o None si falla."""
    try:
        df = client.get_price_history(
            symbol=ticker,
            market="BCBA",
            from_date="2023-01-01",
            to_date=date.today().strftime("%Y-%m-%d"),
        )
        if df is None or len(df) < 210:
            log(f"  {ticker}: datos insuficientes ({len(df) if df is not None else 0} barras)")
            return None

        df = calcular_indicadores(df)
        ultima = df.iloc[-1]

        if pd.isna(ultima["sma200"]) or pd.isna(ultima["rsi14"]):
            log(f"  {ticker}: indicadores incompletos en la ultima barra")
            return None

        estructura = calcular_estructura(df, n=10)
        alert_score, conds_ok, score_pond = calcular_score(ultima, estructura)

        return {
            "ticker":          ticker,
            "sector":          sector,
            "precio_cierre":   float(ultima["close"]),
            "precio_fecha":    ultima["fecha"],
            "atr14":           float(ultima["atr14"]) if not pd.isna(ultima["atr14"]) else None,
            "rsi14":           float(ultima["rsi14"]),
            "macd_hist":       float(ultima["macd_hist"]),
            "dist_sma20":      float(ultima["dist_sma20"]),
            "dist_sma50":      float(ultima["dist_sma50"]),
            "dist_sma200":     float(ultima["dist_sma200"]),
            "adx":             float(ultima["adx"]) if not pd.isna(ultima["adx"]) else None,
            "score_ponderado": score_pond,
            "condiciones_ok":  conds_ok,
            "estructura_10":   estructura["estructura_10"],
            "bos_bull_10":     estructura["bos_bull_10"],
            "bos_bear_10":     estructura["bos_bear_10"],
            "choch_bull_10":   estructura["choch_bull_10"],
            "choch_bear_10":   estructura["choch_bear_10"],
            "alert_score":     alert_score,
            "alert_nivel":     nivel_alerta(alert_score),
            "alert_detalle":   generar_detalle(ultima, estructura, alert_score),
        }

    except Exception as e:
        log(f"  {ticker}: ERROR — {e}")
        return None


# ── Persistencia ──────────────────────────────────────────────────────────────

SQL_UPSERT = """
    INSERT INTO alertas_scanner_ar (
        ticker, sector, precio_cierre, precio_fecha, atr14,
        rsi14, macd_hist, dist_sma20, dist_sma50, dist_sma200, adx,
        score_ponderado, condiciones_ok,
        estructura_10, bos_bull_10, bos_bear_10, choch_bull_10, choch_bear_10,
        alert_score, alert_nivel, alert_detalle
    ) VALUES (
        %(ticker)s, %(sector)s, %(precio_cierre)s, %(precio_fecha)s, %(atr14)s,
        %(rsi14)s, %(macd_hist)s, %(dist_sma20)s, %(dist_sma50)s, %(dist_sma200)s, %(adx)s,
        %(score_ponderado)s, %(condiciones_ok)s,
        %(estructura_10)s, %(bos_bull_10)s, %(bos_bear_10)s, %(choch_bull_10)s, %(choch_bear_10)s,
        %(alert_score)s, %(alert_nivel)s, %(alert_detalle)s
    )
    ON CONFLICT (ticker, date(scan_fecha))
    DO UPDATE SET
        precio_cierre   = EXCLUDED.precio_cierre,
        precio_fecha    = EXCLUDED.precio_fecha,
        atr14           = EXCLUDED.atr14,
        rsi14           = EXCLUDED.rsi14,
        macd_hist       = EXCLUDED.macd_hist,
        dist_sma20      = EXCLUDED.dist_sma20,
        dist_sma50      = EXCLUDED.dist_sma50,
        dist_sma200     = EXCLUDED.dist_sma200,
        adx             = EXCLUDED.adx,
        score_ponderado = EXCLUDED.score_ponderado,
        condiciones_ok  = EXCLUDED.condiciones_ok,
        estructura_10   = EXCLUDED.estructura_10,
        bos_bull_10     = EXCLUDED.bos_bull_10,
        bos_bear_10     = EXCLUDED.bos_bear_10,
        choch_bull_10   = EXCLUDED.choch_bull_10,
        choch_bear_10   = EXCLUDED.choch_bear_10,
        alert_score     = EXCLUDED.alert_score,
        alert_nivel     = EXCLUDED.alert_nivel,
        alert_detalle   = EXCLUDED.alert_detalle
"""


def persistir(alertas: list[dict]):
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, SQL_UPSERT, alertas, page_size=100)
    log(f"  Guardadas {len(alertas)} alertas en alertas_scanner_ar.")


# ── Comandos ──────────────────────────────────────────────────────────────────

def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker, alert_nivel, alert_score, precio_cierre,
                   precio_fecha, scan_fecha::date AS dia
            FROM alertas_scanner_ar
            WHERE scan_fecha >= NOW() - INTERVAL '7 days'
            ORDER BY alert_score DESC, ticker
        """)).fetchall()
    if not rows:
        log("Sin alertas en los ultimos 7 dias.")
        return
    log(f"Alertas recientes ({len(rows)}):")
    for r in rows:
        log(f"  {r[0]:<8} {r[1]:<15} score={r[2]:>5.1f}  cierre={r[3]:>10.2f}  "
            f"precio_fecha={r[4]}  scan={r[5]}")


def cmd_run(tickers_filter: list[str], dry_run: bool):
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT ticker, sector FROM activos_ar WHERE activo = true ORDER BY ticker"
        )).fetchall()

    activos = [(r[0], r[1] or "Sin sector") for r in rows]
    if tickers_filter:
        activos = [(t, s) for t, s in activos if t in tickers_filter]

    if not activos:
        log("No hay tickers AR activos en la DB. Ejecuta init_tablas_ar.py primero.")
        return

    log(f"Scanner AR: {len(activos)} tickers")
    client = IOLClient()
    alertas = []

    for i, (ticker, sector) in enumerate(activos, 1):
        log(f"  [{i:>2}/{len(activos)}] {ticker}...")
        resultado = analizar_ticker(client, ticker, sector)
        if resultado:
            alertas.append(resultado)
            log(f"         {resultado['alert_nivel']:<15} score={resultado['alert_score']:>5.1f}  "
                f"cierre={resultado['precio_cierre']:>10.2f} ARS")
        time.sleep(0.3)  # cortesia con la API

    log(f"\nResultados: {len(alertas)}/{len(activos)} tickers procesados.")

    if dry_run:
        log("DRY RUN: no se guardo nada.")
        return

    if alertas:
        persistir(alertas)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scanner diario mercado AR (BCBA).")
    parser.add_argument("--ticker", nargs="+", help="Filtrar tickers especificos")
    parser.add_argument("--dry-run", action="store_true", help="Calcula pero no guarda")
    parser.add_argument("--status",  action="store_true", help="Muestra alertas recientes en DB")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    else:
        cmd_run(
            tickers_filter=args.ticker or [],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
