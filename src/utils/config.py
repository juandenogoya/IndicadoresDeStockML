"""
config.py
Carga variables de entorno y define constantes globales del proyecto.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── PostgreSQL ────────────────────────────────────────────────
# Soporta DATABASE_URL (Railway/Render/Heroku) o variables individuales.
# DATABASE_URL tiene prioridad si esta definida en el entorno.
_DATABASE_URL = os.getenv("DATABASE_URL")

if _DATABASE_URL:
    # Parsear postgresql://user:pass@host:port/dbname
    # Railway a veces usa el prefijo "postgres://" (alias)
    from urllib.parse import urlparse
    _u = urlparse(_DATABASE_URL.replace("postgres://", "postgresql://", 1))
    DB_CONFIG = {
        "host":     _u.hostname,
        "port":     str(_u.port or 5432),
        "dbname":   _u.path.lstrip("/"),
        "user":     _u.username,
        "password": _u.password,
    }
else:
    DB_CONFIG = {
        "host":     os.getenv("DB_HOST", "localhost"),
        "port":     os.getenv("DB_PORT", "5432"),
        "dbname":   os.getenv("DB_NAME", "activos_ml"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

# ── API Keys ──────────────────────────────────────────────────
FMP_API_KEY          = os.getenv("FMP_API_KEY")
ALPHA_VANTAGE_KEY    = os.getenv("ALPHA_VANTAGE_API_KEY")
MARKETSTACK_KEY      = os.getenv("MARKETSTACK_API_KEY")
NEWSAPI_KEY          = os.getenv("NEWSAPI_KEY")
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID")

# ── Proyecto ──────────────────────────────────────────────────
START_DATE = os.getenv("START_DATE", "2020-01-01")

# ── Universo de Activos ───────────────────────────────────────
ACTIVOS = {
    "Financials": ["JPM", "BAC", "MS", "GS", "WFC", "AXP"],
    "Consumer Staples": ["KO", "PEP", "PG", "PM", "CL", "MO"],
    "Consumer Discretionary": ["TGT", "WMT", "COST", "HD", "LOW"],
    "Automotive": ["F", "GM", "STLA", "TM"],
}

# Lista plana de todos los tickers
ALL_TICKERS = [ticker for tickers in ACTIVOS.values() for ticker in tickers]

# Mapa inverso: ticker -> sector
TICKER_SECTOR = {
    ticker: sector
    for sector, tickers in ACTIVOS.items()
    for ticker in tickers
}

# ── Parámetros Indicadores Técnicos ───────────────────────────
SMA_PERIODS   = [21, 50, 200]
RSI_PERIOD    = 14
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
ATR_PERIOD    = 14
BB_PERIOD     = 20
BB_STD        = 2
ADX_PERIOD    = 14
VOL_MA_PERIOD = 20      # periodo para calcular volumen promedio

# ── Parámetros Scoring Rule-Based ────────────────────────────
SCORING_WEIGHTS = {
    "rsi":      0.20,
    "macd":     0.20,
    "sma21":    0.10,
    "sma50":    0.15,
    "sma200":   0.20,
    "momentum": 0.15,
}
SCORE_ENTRADA_UMBRAL = 0.60   # >= 60% para señal LONG
SCORE_SALIDA_UMBRAL  = 0.30   # <= 30% para cierre por señal

RSI_OVERSOLD  = 35
RSI_OVERBOUGHT = 65

# ── Parámetros Backtesting ────────────────────────────────────
TIMEOUT_DIAS  = 20            # dias máximo en posicion
UMBRAL_NEUTRO = 0.01          # +/- 1% define Ganancia / Perdida / Neutro

ESTRATEGIAS_ENTRADA = ["E1", "E2", "E3", "E4"]
ESTRATEGIAS_SALIDA  = ["S1", "S2", "S3", "S4"]

# ── Split de datos ────────────────────────────────────────────
TRAIN_RATIO = 0.70
TEST_RATIO  = 0.15
# Backtest = 1 - TRAIN_RATIO - TEST_RATIO = 0.15

# ── Configuración ML ─────────────────────────────────────────
# Sectores con modelo propio en el challenger sectorial.
# Automotive excluido: solo 3 tickers, datos insuficientes para
# un modelo sectorial robusto. Usará el Global Champion.
SECTORES_ML = ["Financials", "Consumer Staples", "Consumer Discretionary"]

# Directorio donde se serializan los modelos entrenados
_BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR    = os.path.join(_BASE_DIR, "models")
MODELS_V2_DIR = os.path.join(_BASE_DIR, "models_v2")
MODELS_V3_DIR = os.path.join(_BASE_DIR, "models_v3")
