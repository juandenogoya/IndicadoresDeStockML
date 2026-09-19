"""
bt_data_loader.py
Carga en bulk desde local PostgreSQL los datos historicos necesarios
para simular las 3 estrategias de backtesting.

Un unico viaje a la DB al inicio del BT.
Dentro del loop diario: cero queries, todo en memoria (pandas).

Estrategias y tablas que necesitan:
    TECH_SECTOR_v1 : indicadores_tecnicos + precios_diarios + activos
    COMBO_v1       : + features_precio_accion + features_market_structure
    SMC_v1         : precios_diarios + features_market_structure + features_precio_accion

HISTORIA (17/9/2026, Tarea 23 -- docs/estructura_velas.md):
    historia="vieja": features_market_structure + patrones de features_precio_accion,
        como se diseno. Esa historia de estructura mira N ruedas al futuro.
    historia="nueva": estructura con swings CONFIRMADOS (features_estructura, o
        src/indicators/estructura.py al vuelo si la ventana no es 5 ni 10) y patrones
        de features_velas. Las columnas de volumen siguen de features_precio_accion
        (no miran al futuro). `ventana` = N de la estructura que usa SMC (se expone con
        los nombres *_10 que lee la estrategia); COMBO usa siempre N=5.

LOCAL-only (Plan C): antes cargaba .env.local con override, que setea DATABASE_URL
y manda get_engine a Railway, donde las tablas de mercado estan congeladas.
"""

import sys
import os
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.environ.pop("DATABASE_URL", None)

from sqlalchemy import text
from src.data.database import get_engine
from src.utils.trading_calendar import is_trading_day

HISTORIAS = ("vieja", "nueva")
_COLS_SMC = ("estructura", "choch_bull", "choch_bear", "bos_bull", "dist_sl", "dist_sh")

# Dias calendario adicionales antes de `desde` para cubrir lookbacks:
#   SMC: 12 dias calendario de CHoCH/BOS
#   COMBO candle score: ~7 dias calendario = 5 dias habiles
BUFFER_DIAS = 20


def _log(msg: str):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _int(x) -> int:
    """Convierte a int manejando None y NaN (NaN es truthy en Python, int(NaN) falla)."""
    if x is None:
        return 0
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0
    return 0 if f != f else int(f)  # f != f es True solo para NaN


def _float(x) -> float:
    """Convierte a float manejando None y NaN."""
    if x is None:
        return 0.0
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


class BtDataLoader:
    """
    Carga y expone los datos historicos para la simulacion de backtesting.

    Uso:
        loader = BtDataLoader(engine, desde=date(2025,6,1), hasta=date(2025,12,31), logica="tecnico_sectorial")
        loader.cargar()
        precios_dia = loader.get_close(fecha)
        indicadores = loader.get_indicadores_fecha(fecha)
    """

    LOGICAS_TECH  = {"tecnico", "tecnico_sectorial", "combo_tech_candle"}
    LOGICAS_FMS   = {"smc_estructura", "combo_tech_candle"}
    LOGICAS_FPA   = {"smc_estructura", "combo_tech_candle"}
    LOGICAS_SECTOR = {"tecnico_sectorial", "combo_tech_candle"}

    def __init__(self, engine, desde: date, hasta: date, logica: str,
                 historia: str = "vieja", ventana: int = 10):
        if historia not in HISTORIAS:
            raise ValueError(f"historia debe ser una de {HISTORIAS}")
        if historia == "vieja" and ventana != 10:
            raise ValueError("la historia vieja solo tiene la ventana 10 (y la 5 de COMBO)")
        self.engine  = engine
        self.desde   = desde
        self.hasta   = hasta
        self.logica  = logica
        self.historia = historia
        self.ventana  = ventana
        self._desde_carga = desde - timedelta(days=BUFFER_DIAS)

        # DataFrames (None hasta llamar a cargar())
        self._pd   = None   # precios_diarios
        self._ind  = None   # indicadores_tecnicos
        self._fpa  = None   # features_precio_accion
        self._fms  = None   # features_market_structure
        self._act  = None   # activos (ticker -> sector)

        # Lista de dias habiles del periodo de simulacion
        self._trading_days: list[date] = []

    # ── Carga principal ────────────────────────────────────────────────────────

    def cargar(self) -> None:
        """Ejecuta todas las queries y prepara los DataFrames."""
        _log(f"Cargando datos [{self._desde_carga} -> {self.hasta}] logica={self.logica}...")

        self._cargar_precios()

        if self.logica in self.LOGICAS_TECH:
            self._cargar_indicadores()

        if self.logica in self.LOGICAS_FMS:
            self._cargar_fms()

        if self.logica in self.LOGICAS_FPA:
            self._cargar_fpa()

        if self.logica in self.LOGICAS_SECTOR:
            self._cargar_activos()

        self._trading_days = self._calcular_trading_days()
        _log(f"Carga completa. Dias habiles a simular: {len(self._trading_days)}")

    def _cargar_precios(self):
        with self.engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT ticker, fecha, open, high, low, close, volume
                FROM precios_diarios
                WHERE fecha BETWEEN :desde AND :hasta
                ORDER BY ticker, fecha
            """), conn, params={"desde": self._desde_carga, "hasta": self.hasta})
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        self._pd = df.set_index(["ticker", "fecha"])
        _log(f"  precios_diarios: {len(df):,} filas, {df['ticker'].nunique()} tickers")

    def _cargar_indicadores(self):
        with self.engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT ticker, fecha,
                       sma21, sma50, sma200,
                       rsi14, macd, macd_signal,
                       (macd - macd_signal) AS macd_hist,
                       atr14, adx, vol_relativo
                FROM indicadores_tecnicos
                WHERE fecha BETWEEN :desde AND :hasta
                ORDER BY ticker, fecha
            """), conn, params={"desde": self._desde_carga, "hasta": self.hasta})
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        self._ind = df.set_index(["ticker", "fecha"])
        _log(f"  indicadores_tecnicos: {len(df):,} filas")

    def _cargar_fms(self):
        if self.historia == "nueva":
            return self._cargar_estructura_nueva()
        with self.engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT ticker, fecha,
                       estructura_10, choch_bull_10, choch_bear_10, bos_bull_10,
                       dist_sl_10_pct, dist_sh_10_pct,
                       bos_bull_5, choch_bull_5, bos_bear_5, choch_bear_5
                FROM features_market_structure
                WHERE fecha BETWEEN :desde AND :hasta
                ORDER BY ticker, fecha
            """), conn, params={"desde": self._desde_carga, "hasta": self.hasta})
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        self._fms = df.set_index(["ticker", "fecha"])
        _log(f"  features_market_structure: {len(df):,} filas")

    def _cargar_estructura_nueva(self):
        """Estructura con swings confirmados, expuesta con los nombres que leen las estrategias."""
        n = self.ventana
        propias = [f"{b}_{n}_pct" if b.startswith("dist_") else f"{b}_{n}" for b in _COLS_SMC]
        combo = ["bos_bull_5", "choch_bull_5", "bos_bear_5", "choch_bear_5"]
        if n in (5, 10):
            cols = sorted(set(propias + combo))
            with self.engine.connect() as conn:
                df = pd.read_sql(text(f"""
                    SELECT ticker, fecha, {", ".join(cols)}
                    FROM features_estructura
                    WHERE fecha BETWEEN :desde AND :hasta
                    ORDER BY ticker, fecha
                """), conn, params={"desde": self._desde_carga, "hasta": self.hasta})
            origen = "features_estructura"
        else:
            # Ventana sin tabla: al vuelo sobre TODA la historia (la invariancia vale
            # para el mismo inicio de datos; cortar en desde cambiaria los swings).
            from src.indicators import estructura
            with self.engine.connect() as conn:
                px = pd.read_sql(text("""
                    SELECT ticker, fecha, high, low, close FROM precios_diarios
                    WHERE fecha <= :hasta AND close > 0 AND high > 0 AND low > 0 AND open > 0
                    ORDER BY ticker, fecha
                """), conn, params={"hasta": self.hasta})
            partes = [estructura.calcular_estructura(g, ventanas=sorted({n, 5}))
                      for _, g in px.groupby("ticker", sort=True)]
            df = pd.concat(partes, ignore_index=True)
            df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
            df = df[(df["fecha"] >= self._desde_carga) & (df["fecha"] <= self.hasta)]
            df = df[["ticker", "fecha"] + sorted(set(propias + combo))]
            origen = f"estructura.py al vuelo (N={n})"

        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        if n != 10:
            # Copia, no renombre: con N=5 los *_5 los sigue necesitando COMBO.
            for p in propias:
                df[p.replace(f"_{n}", "_10")] = df[p]
        self._fms = df.set_index(["ticker", "fecha"])
        _log(f"  estructura NUEVA ({origen}): {len(df):,} filas")

    def _cargar_fpa(self):
        if self.historia == "nueva":
            return self._cargar_velas_nuevas()
        with self.engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT ticker, fecha,
                       es_alcista,
                       patron_engulfing_bull, patron_engulfing_bear,
                       patron_hammer, patron_shooting_star, patron_marubozu,
                       vol_price_confirm, vol_price_diverge, vol_spike, up_vol_5d
                FROM features_precio_accion
                WHERE fecha BETWEEN :desde AND :hasta
                ORDER BY ticker, fecha
            """), conn, params={"desde": self._desde_carga, "hasta": self.hasta})
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        self._fpa = df.set_index(["ticker", "fecha"])
        _log(f"  features_precio_accion: {len(df):,} filas")

    def _cargar_velas_nuevas(self):
        """Patrones de features_velas + volumen de features_precio_accion + vela alcista del precio."""
        with self.engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT v.ticker, v.fecha,
                       CASE WHEN p.close > p.open THEN 1 ELSE 0 END AS es_alcista,
                       v.patron_engulfing_bull, v.patron_engulfing_bear,
                       v.patron_hammer, v.patron_shooting_star,
                       GREATEST(v.patron_marubozu_bull, v.patron_marubozu_bear) AS patron_marubozu,
                       f.vol_price_confirm, f.vol_price_diverge, f.vol_spike, f.up_vol_5d
                FROM features_velas v
                JOIN precios_diarios p ON p.ticker = v.ticker AND p.fecha = v.fecha
                LEFT JOIN features_precio_accion f ON f.ticker = v.ticker AND f.fecha = v.fecha
                WHERE v.fecha BETWEEN :desde AND :hasta
                ORDER BY v.ticker, v.fecha
            """), conn, params={"desde": self._desde_carga, "hasta": self.hasta})
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        self._fpa = df.set_index(["ticker", "fecha"])
        _log(f"  velas NUEVAS (features_velas + volumen de features_precio_accion): {len(df):,} filas")

    def retornos_universo(self) -> pd.Series:
        """
        Retorno diario del universo equal-weight (media de los retornos de los tickers
        con precio en ambas ruedas), sobre los dias simulados. Referencia "comprar todo".
        """
        close = self._pd["close"].unstack("ticker").sort_index()
        close = close[close.index <= self.hasta]
        ret = close.pct_change(fill_method=None).mean(axis=1)
        dias = set(self._trading_days)
        return ret[[d in dias for d in ret.index]].dropna()

    def _cargar_activos(self):
        with self.engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT ticker, sector
                FROM activos
                WHERE activo = TRUE AND sector IS NOT NULL
            """), conn)
        self._act = df.set_index("ticker")["sector"].to_dict()
        _log(f"  activos: {len(self._act)} tickers con sector")

    def _calcular_trading_days(self) -> list[date]:
        """
        Dias habiles NYSE en [desde, hasta] CON precios cargados.

        trading_calendar solo tiene feriados 2025-2027: antes de 2025 un feriado
        pasaba como dia habil sin precios y la equity de ese dia valuaba las
        posiciones al precio de entrada (saltos falsos de drawdown).
        """
        con_precio = set(self._pd.index.get_level_values("fecha")) if self._pd is not None else set()
        dias = []
        d = self.desde
        while d <= self.hasta:
            if is_trading_day(d) and d in con_precio:
                dias.append(d)
            d += timedelta(days=1)
        return dias

    # ── Propiedades publicas ───────────────────────────────────────────────────

    @property
    def trading_days(self) -> list[date]:
        return self._trading_days

    @property
    def sector_map(self) -> dict:
        return self._act or {}

    # ── Lookups por fecha ──────────────────────────────────────────────────────

    def get_close(self, fecha: date) -> dict[str, float]:
        """Precio de cierre de todos los tickers en fecha."""
        if self._pd is None:
            return {}
        try:
            sub = self._pd.xs(fecha, level="fecha")["close"]
            return sub.dropna().to_dict()
        except KeyError:
            return {}

    def get_indicadores_fecha(self, fecha: date) -> list[dict]:
        """
        Indicadores tecnicos + close de precios_diarios para todos los tickers en fecha.
        Retorna lista de dicts con las columnas que calcular_score_tecnico() necesita.
        """
        if self._ind is None or self._pd is None:
            return []
        try:
            ind_sub = self._ind.xs(fecha, level="fecha")
        except KeyError:
            return []

        try:
            pd_sub = self._pd.xs(fecha, level="fecha")[["close"]]
        except KeyError:
            pd_sub = pd.DataFrame()

        df = ind_sub.join(pd_sub, how="inner")

        if self._act:
            df["sector"] = df.index.map(self._act)

        rows = []
        for ticker, row in df.iterrows():
            d = row.to_dict()
            d["ticker"] = ticker
            d.setdefault("close", None)
            rows.append(d)
        return rows

    def get_fms_estado(self, fecha: date) -> dict[str, dict]:
        """
        Ultimo estado de features_market_structure por ticker en o antes de fecha.
        Enriquece con close de precios_diarios.
        """
        if self._fms is None:
            return {}

        fechas_disponibles = self._fms.index.get_level_values("fecha")
        mask = fechas_disponibles <= fecha
        if not mask.any():
            return {}

        sub = self._fms[mask]
        # Ultimo registro por ticker
        latest = sub.groupby("ticker").last()

        precios = self.get_close(fecha)
        result = {}
        for ticker, row in latest.iterrows():
            d = row.to_dict()
            d["ticker"] = ticker
            d["close"]  = precios.get(ticker)
            result[ticker] = d
        return result

    def get_fpa_estado(self, fecha: date) -> dict[str, dict]:
        """Ultimo estado de features_precio_accion por ticker en o antes de fecha."""
        if self._fpa is None:
            return {}
        try:
            sub = self._fpa.xs(fecha, level="fecha")
        except KeyError:
            fechas_disp = self._fpa.index.get_level_values("fecha")
            mask = fechas_disp <= fecha
            if not mask.any():
                return {}
            sub = self._fpa[mask].groupby("ticker").last()

        result = {}
        for ticker, row in sub.iterrows():
            d = row.to_dict()
            d["ticker"] = ticker
            result[ticker] = d
        return result

    def get_fms_eventos_lookback(self, fecha: date, lookback_dias: int = 12) -> dict[str, dict]:
        """
        Para SMC entrada: detecta tickers con CHoCH o BOS bull en los ultimos
        `lookback_dias` calendario hasta `fecha`.

        Retorna {ticker: {tuvo_choch_bull: bool, tuvo_bos_bull: bool}}.
        Solo incluye tickers que tuvieron al menos uno de los dos eventos.
        """
        if self._fms is None:
            return {}

        cutoff = fecha - timedelta(days=lookback_dias)
        fechas = self._fms.index.get_level_values("fecha")
        mask   = (fechas >= cutoff) & (fechas <= fecha)
        if not mask.any():
            return {}

        window = self._fms[mask][["choch_bull_10", "bos_bull_10"]]
        agg = window.fillna(0).groupby("ticker").max()

        result = {}
        for ticker, row in agg.iterrows():
            choch = _int(row.get("choch_bull_10"))
            bos   = _int(row.get("bos_bull_10"))
            if choch or bos:
                result[ticker] = {"tuvo_choch_bull": bool(choch), "tuvo_bos_bull": bool(bos)}
        return result

    def get_features_smc_entrada(self, fecha: date, lookback_dias: int = 12) -> list[dict]:
        """
        Construye la lista de candidatos para SMC_v1:
        - Tickers con CHoCH/BOS bull en el lookback
        - Enriquecidos con estado actual de fms, fpa, indicadores y close
        Equivale al output de obtener_features_hoy() pero sobre datos historicos.
        """
        eventos = self.get_fms_eventos_lookback(fecha, lookback_dias)
        if not eventos:
            return []

        fms_estado = self.get_fms_estado(fecha)
        fpa_estado = self.get_fpa_estado(fecha)
        precios    = self.get_close(fecha)

        rows = []
        for ticker, ev in eventos.items():
            fms = fms_estado.get(ticker, {})
            fpa = fpa_estado.get(ticker, {})
            close = precios.get(ticker)

            if close is None:
                continue

            row = {
                "ticker":              ticker,
                "close":               close,
                "tuvo_choch_bull":     int(ev["tuvo_choch_bull"]),
                "tuvo_bos_bull":       int(ev["tuvo_bos_bull"]),
                "estructura_10":       _int(fms.get("estructura_10")),
                "choch_bear_10":       _int(fms.get("choch_bear_10")),
                "bos_bull_10":         _int(fms.get("bos_bull_10")),
                "dist_sl_10_pct":      _float(fms.get("dist_sl_10_pct")),
                "dist_sh_10_pct":      _float(fms.get("dist_sh_10_pct")),
                "es_alcista":          _int(fpa.get("es_alcista")),
                "patron_engulfing_bull": _int(fpa.get("patron_engulfing_bull")),
                "patron_hammer":       _int(fpa.get("patron_hammer")),
                "vol_spike":           _int(fpa.get("vol_spike")),
                "up_vol_5d":           _int(fpa.get("up_vol_5d")),
            }
            rows.append(row)
        return rows

    def get_candle_score_5d(self, fecha: date) -> dict[str, float]:
        """
        Candle score acumulado de los ultimos 5 dias habiles hasta `fecha`.
        Equivale a obtener_candle_score_5d() pero sobre datos historicos.
        Retorna {ticker: float}.
        """
        if self._fpa is None or self._fms is None:
            return {}

        # 5 dias habiles <= fecha
        dias_previos = [d for d in self._trading_days if d <= fecha][-5:]
        if not dias_previos:
            return {}

        fechas_fpa = self._fpa.index.get_level_values("fecha")
        mask_5d    = fechas_fpa.isin(dias_previos)
        if not mask_5d.any():
            return {}

        window = self._fpa[mask_5d].copy()

        # Score base por fila (mismo algoritmo que obtener_candle_score_5d)
        es_alc = window.get("es_alcista", 0).fillna(0)
        window["_base"] = es_alc.map(lambda x: 0.5 if x == 1 else -0.5)

        window["_base"] += (
            window.get("patron_engulfing_bull", 0).fillna(0) *  1.5
            + window.get("patron_hammer",       0).fillna(0) *  1.0
            + window.get("patron_engulfing_bear",0).fillna(0) * -1.5
            + window.get("patron_shooting_star", 0).fillna(0) * -1.0
            + window.get("vol_price_confirm",    0).fillna(0) *  1.0
            + window.get("vol_price_diverge",    0).fillna(0) * -0.5
        )

        # Marubozu: alcista +1.0, bajista -0.5
        maru = window.get("patron_marubozu", 0).fillna(0)
        window["_base"] += (maru * es_alc).map(lambda x: 1.0 if x == 1 else 0)
        window["_base"] += (maru * (1 - es_alc)).map(lambda x: -0.5 if x == 1 else 0)

        # Vol spike sin confirm/diverge
        vs = window.get("vol_spike", 0).fillna(0)
        vc = window.get("vol_price_confirm", 0).fillna(0)
        vd = window.get("vol_price_diverge", 0).fillna(0)
        window["_base"] += ((vs == 1) & (vc == 0) & (vd == 0)).map(lambda x: 0.5 if x else 0)

        base_scores = window.groupby("ticker")["_base"].sum()

        # Score estructural del ultimo dia disponible por ticker (columnas _5)
        try:
            fms_latest = self._fms.xs(max(dias_previos), level="fecha")
        except KeyError:
            fms_latest = pd.DataFrame()

        struct_scores = {}
        if not fms_latest.empty:
            for ticker, row in fms_latest.iterrows():
                s = (
                    _float(row.get("bos_bull_5"))   *  2.0
                    + _float(row.get("choch_bull_5")) *  1.5
                    + _float(row.get("bos_bear_5"))   * -2.0
                    + _float(row.get("choch_bear_5")) * -3.0
                )
                struct_scores[ticker] = s

        result = {}
        for ticker, base in base_scores.items():
            result[ticker] = round(float(base) + struct_scores.get(ticker, 0.0), 2)
        return result
