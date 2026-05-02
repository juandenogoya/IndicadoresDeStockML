"""
iol_client.py
Cliente autenticado para la API REST de InvertirOnline (IOL).
Maneja autenticacion OAuth2 con refresh automatico de token.

Credenciales en .env:
    IOL_USERNAME=...
    IOL_PASSWORD=...

Endpoints clave:
    Historico  : /api/v2/bCBA/Titulos/{symbol}/Cotizacion/serieHistorica/...
    Quote live : /api/v2/bCBA/Titulos/{symbol}/Cotizacion
    Opciones   : /api/v2/bCBA/Titulos/{symbol}/Opciones   <-- lista de opciones con metadata
"""

import os
import re
import time
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional

IOL_BASE = "https://api.invertironline.com"

# Mapeo de nombre corto a codigo de mercado IOL
MARKET_MAP = {
    "BCBA":   "bCBA",
    "NYSE":   "nYSE",
    "NASDAQ": "nASDAQ",
    "AMEX":   "aMEX",
    "bCBA":   "bCBA",
    "nYSE":   "nYSE",
}

# Tickers BCBA con opciones activas (simbolos IOL correctos)
TICKERS_CON_OPCIONES = {
    "GGAL", "BMA", "YPFD", "PAMP", "TGSU2", "EDN",
    "CEPU", "TXAR", "ALUA", "VALO", "MIRG", "BYMA",
}


# ── Helpers de calculo BSM ────────────────────────────────────────────────────

def _extraer_strike(descripcion: str) -> float:
    """
    Extrae el strike de la descripcion IOL.
    Ej: "Call GGAL 4,348.70 Vencimiento: 17/04/2026" -> 4348.70
    """
    if not descripcion:
        return 0.0
    matches = re.findall(r"[\d]{1,3}(?:[,\.][\d]{3})*(?:[.,]\d+)?", descripcion)
    for m in matches:
        try:
            limpio = m.replace(",", "")
            val = float(limpio)
            if val > 0:
                return val
        except ValueError:
            continue
    return 0.0


def _bsm_calcular(S: float, K: float, T: float, r: float,
                  prima: float, tipo: str) -> dict:
    """
    Calcula IV y Greeks (Delta, Gamma, Theta, Vega, Rho) via Black-Scholes-Merton.

    tipo : "call" o "put"
    Retorna dict con claves: iv, delta, gamma, theta, vega, rho.
    Todos None si falla (scipy no disponible, T<=0, etc.).
    """
    resultado = {
        "iv": None, "delta": None, "gamma": None,
        "theta": None, "vega": None, "rho": None,
    }
    if S <= 0 or K <= 0 or T <= 0 or prima <= 0:
        return resultado

    tipo = tipo.lower()

    try:
        import numpy as np
        from scipy.stats import norm
        from scipy.optimize import brentq
    except ImportError:
        return resultado

    # Valor intrinseco: si prima <= intrinseco, IV no calculable
    intrinseco = max(S - K, 0.0) if tipo == "call" else max(K - S, 0.0)
    if prima <= intrinseco:
        return resultado

    def _d1_d2(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def precio_bsm(sigma):
        d1, d2 = _d1_d2(sigma)
        if tipo == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # IV via Brent's method
    try:
        iv = brentq(lambda s: precio_bsm(s) - prima, 1e-6, 20.0, xtol=1e-6, maxiter=200)
    except Exception:
        return resultado

    if iv <= 0:
        return resultado

    resultado["iv"] = iv

    # Greeks
    d1, d2 = _d1_d2(iv)
    if tipo == "call":
        delta = float(norm.cdf(d1))
        rho   = float(K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01)
    else:
        delta = float(norm.cdf(d1) - 1)
        rho   = float(-K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01)

    gamma = float(norm.pdf(d1) / (S * iv * np.sqrt(T)))

    theta_base = (
        -(S * norm.pdf(d1) * iv) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if tipo == "call" else -d2)
    )
    if tipo == "put":
        theta_base += r * K * np.exp(-r * T)
    theta = float(theta_base / 365)  # por dia calendario

    vega = float(S * norm.pdf(d1) * np.sqrt(T) * 0.01)  # por 1% cambio en vol

    resultado.update({
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
        "rho":   round(rho, 4),
    })
    return resultado


def _calcular_prob_otm(S: float, K: float, T: float, r: float,
                       iv: float, tipo: str) -> float:
    """
    Probabilidad de que la opcion expire OTM (sin valor) al vencimiento.
    Para el VENDEDOR es la probabilidad de quedarse con la prima completa.
      PUT  OTM = P(S_T > K) = N(d2)
      CALL OTM = P(S_T < K) = N(-d2)
    """
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        import numpy as np
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        if tipo.lower() == "put":
            return float(norm.cdf(d2))
        return float(norm.cdf(-d2))
    except Exception:
        return 0.0


def _calcular_tna(prima: float, base: float, dias: int) -> float:
    """
    TNA bruta = (prima / base) x (365 / dias) x 100
    Para PUT: base = strike. Para CALL cubierto: base = spot.
    """
    if base <= 0 or dias <= 0 or prima <= 0:
        return 0.0
    return (prima / base) * (365.0 / dias) * 100.0


def calcular_hv_iol(df_precios: "pd.DataFrame",
                    ventana: int = 20,
                    dias_anio: int = 245) -> float:
    """
    HV rolling usando precios de cierre de IOL.
    dias_anio=245 para BYMA (dias habiles Argentina).
    Retorna el ultimo valor de HV (decimal, ej: 0.45 = 45%).
    """
    try:
        import numpy as np
        import pandas as pd
        closes = pd.to_numeric(df_precios["close"], errors="coerce").dropna()
        if len(closes) < ventana + 1:
            return 0.0
        log_ret = np.log(closes / closes.shift(1)).dropna()
        hv_series = log_ret.rolling(ventana).std(ddof=1) * np.sqrt(dias_anio)
        return float(hv_series.dropna().iloc[-1])
    except Exception:
        return 0.0


class IOLClient:
    """
    Cliente para la API REST de InvertirOnline.
    Un solo objeto por proceso; renueva el token automaticamente.
    """

    def __init__(self):
        self.username = os.getenv("IOL_USERNAME")
        self.password = os.getenv("IOL_PASSWORD")
        if not self.username or not self.password:
            raise ValueError("IOL_USERNAME e IOL_PASSWORD requeridos en .env")
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: float = 0.0

    # ── Autenticacion ────────────────────────────────────────────

    def _autenticar(self):
        resp = requests.post(
            f"{IOL_BASE}/token",
            data={
                "username":   self.username,
                "password":   self.password,
                "grant_type": "password",
            },
            timeout=15,
        )
        resp.raise_for_status()
        self._guardar_token(resp.json())

    def _refrescar(self):
        try:
            resp = requests.post(
                f"{IOL_BASE}/token",
                data={
                    "refresh_token": self._refresh_token,
                    "grant_type":    "refresh_token",
                },
                timeout=15,
            )
            resp.raise_for_status()
            self._guardar_token(resp.json())
        except Exception:
            self._autenticar()

    def _guardar_token(self, data: dict):
        self._access_token  = data["access_token"]
        self._refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in", 1800)
        self._expires_at = time.time() + expires_in - 60

    def _headers(self) -> dict:
        if not self._access_token or time.time() >= self._expires_at:
            if self._refresh_token:
                self._refrescar()
            else:
                self._autenticar()
        return {"Authorization": f"Bearer {self._access_token}"}

    # ── Request base ─────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> any:
        url = f"{IOL_BASE}{path}"
        resp = requests.get(url, headers=self._headers(), params=params, timeout=20)
        if resp.status_code == 401:
            self._autenticar()
            resp = requests.get(url, headers=self._headers(), params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    # ── Precios historicos ───────────────────────────────────────

    def get_price_history(
        self,
        symbol: str,
        market: str = "BCBA",
        from_date: str = None,
        to_date: str = None,
    ) -> pd.DataFrame:
        """
        OHLCV diario desde IOL. Retorna DataFrame con columnas:
        ticker, fecha, open, high, low, close, volume
        """
        mercado = MARKET_MAP.get(market, market)
        hoy = date.today()
        fecha_hasta = to_date or hoy.strftime("%Y-%m-%d")
        fecha_desde = from_date or (hoy - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

        path = (
            f"/api/v2/{mercado}/Titulos/{symbol}/Cotizacion"
            f"/serieHistorica/{fecha_desde}/{fecha_hasta}/ajustada"
        )
        data = self._get(path)

        if not data:
            return pd.DataFrame()

        # IOL retorna lista de objetos con nombres en espanol
        rows = []
        for bar in data:
            fecha_raw = bar.get("fechaHora") or bar.get("fecha") or ""
            fecha = fecha_raw[:10]  # "YYYY-MM-DD"
            rows.append({
                "ticker":  symbol,
                "fecha":   fecha,
                "open":    bar.get("apertura"),
                "high":    bar.get("maximo"),
                "low":     bar.get("minimo"),
                "close":   bar.get("ultimoPrecio"),
                "volume":  bar.get("volumen"),
            })

        df = pd.DataFrame(rows)
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        df = df.sort_values("fecha").reset_index(drop=True)
        return df

    # ── Quote en tiempo real ─────────────────────────────────────

    def get_quote(
        self,
        symbol: str,
        market: str = "BCBA",
        term: str = "t1",
    ) -> dict:
        """
        Precio actual + OHLC intraday + order book (5 niveles).
        Solo significativo en horario BCBA: L-V 10:30-17:00 ART.
        """
        mercado = MARKET_MAP.get(market, market)
        path = f"/api/v2/{mercado}/Titulos/{symbol}/Cotizacion"
        return self._get(path, params={"term": term})

    # ── Opciones ─────────────────────────────────────────────────

    def _get_opciones_quotes_bulk(self) -> dict:
        """
        Cotizaciones en tiempo real para TODAS las opciones del mercado AR.
        Endpoint: /api/v2/argentina/Titulos/cotizaciones/Todos/opciones

        Retorna dict: { simbolo -> {ultimoPrecio, volumen, puntas, ...} }
        Necesario para obtener bid/ask y volumen en tiempo real.
        Devuelve {} si falla (el chain seguira mostrando datos de cotizacion).
        """
        try:
            data = self._get("/api/v2/argentina/Titulos/cotizaciones/Todos/opciones")
            if not isinstance(data, list):
                return {}
            return {
                item["simbolo"]: item
                for item in data
                if isinstance(item, dict) and "simbolo" in item
            }
        except Exception:
            # 500 fuera de horario BCBA es esperado — se usa cotizacion como fallback
            return {}

    def get_options_chain_raw(self, symbol: str) -> dict:
        """
        Chain de opciones desde IOL.

        Retorna un dict normalizado para consumo por iol_tab.py:
            precioSubyacente : float | None
            tasaLibreDeRiesgo: float  (default 0.155 = 15.5% referencia AR)
            opciones         : list   (opciones crudas de la API)

        Endpoint correcto: /api/v2/bCBA/Titulos/{symbol}/Opciones
        Retorna lista de opciones; este metodo la envuelve en el dict esperado.
        """
        options_list = self._get(f"/api/v2/bCBA/Titulos/{symbol}/Opciones")
        if not isinstance(options_list, list):
            options_list = []

        # Spot price via cotizacion
        spot = None
        try:
            quote = self.get_quote(symbol)
            # El endpoint /Cotizacion puede retornar ultimoPrecio directo o
            # anidado segun el tipo de respuesta
            spot = (
                quote.get("ultimoPrecio")
                or (quote.get("trade") or {}).get("lot_price", {}).get("value")
                or (quote.get("trade") or {}).get("unit_price")
            )
            spot = float(spot) if spot else None
        except Exception:
            pass

        return {
            "precioSubyacente":  spot,
            "tasaLibreDeRiesgo": 0.155,   # tasa referencia Argentina ~15.5%
            "opciones":          options_list,
        }

    def get_options_chain_df(
        self,
        symbol: str,
        tasa_libre_riesgo: float = 0.155,
    ) -> pd.DataFrame:
        """
        Chain de opciones normalizado con IV y Greeks calculados via BSM.

        Listo para insertar en opciones_ar_gregas o mostrar en el tab.
        Filtra opciones sin precio de mercado (bid=ask=ultimo=0).

        tasa_libre_riesgo: tasa anualizada decimal (default 15.5% Argentina)
        """
        raw = self.get_options_chain_raw(symbol)
        options_list = raw.get("opciones") or []
        spot = raw.get("precioSubyacente")
        r = tasa_libre_riesgo
        hoy = date.today()

        # Cotizaciones en tiempo real (bid/ask/volumen) via endpoint bulk
        # Devuelve {} si falla; en ese caso se usa el fallback de cotizacion
        quotes_rt = self._get_opciones_quotes_bulk()

        rows = []
        for opt in options_list:
            if not isinstance(opt, dict):
                continue

            simbolo    = opt.get("simbolo") or ""
            cotizacion = opt.get("cotizacion") or {}

            # Quote en tiempo real (bid/ask/volumen actualizados)
            # Si no disponible, cae en cotizacion del endpoint /Opciones
            qt = quotes_rt.get(simbolo, {})

            # ── Tipo (CALL / PUT) ─────────────────────────────────
            tipo_raw = str(opt.get("tipoOpcion") or "").upper()
            if "CALL" in tipo_raw or tipo_raw == "C":
                option_type = "C"
                tipo_bsm    = "call"
            else:
                option_type = "V"
                tipo_bsm    = "put"

            # ── Vencimiento ───────────────────────────────────────
            exp_raw    = opt.get("fechaVencimiento") or ""
            expiration = str(exp_raw)[:10]

            # ── Strike desde descripcion ──────────────────────────
            descripcion = opt.get("descripcion") or ""
            strike = _extraer_strike(descripcion)
            if strike <= 0:
                continue   # sin strike no sirve

            # ── Bid / Ask: RT primero, fallback a cotizacion ──────
            def _bid_ask_from(src: dict):
                puntas = src.get("puntas") or []
                if isinstance(puntas, list) and puntas:
                    puntas = puntas[0]
                elif not isinstance(puntas, dict):
                    return None, None
                b = puntas.get("precioCompra")
                a = puntas.get("precioVenta")
                return (float(b) if b else None), (float(a) if a else None)

            bid, ask = _bid_ask_from(qt)
            if bid is None and ask is None:
                bid, ask = _bid_ask_from(cotizacion)

            # ── Ultimo precio: RT primero, fallback a cotizacion ──
            ultimo_raw = qt.get("ultimoPrecio") or cotizacion.get("ultimoPrecio")
            ultimo = float(ultimo_raw) if ultimo_raw else None

            # ── Volumen: RT primero, fallback a cotizacion ────────
            # Campo en cotizacion es "volumenNominal" (no "volumen")
            volumen = int(
                qt.get("volumen")
                or qt.get("volumenNominal")
                or cotizacion.get("volumenNominal")
                or cotizacion.get("volumen")
                or 0
            )

            # Filtrar sin ningun precio de mercado
            if not bid and not ask and not ultimo:
                continue

            # ── Prima para BSM: mid si hay bid+ask, si no ultimo ─
            mid = None
            if bid and ask:
                mid = (bid + ask) / 2.0
            prima = mid or ultimo or 0.0

            # ── Dias hasta vencimiento ────────────────────────────
            dias = 0
            try:
                vto  = datetime.strptime(expiration, "%Y-%m-%d").date()
                dias = max((vto - hoy).days, 0)
            except Exception:
                pass
            T = dias / 365.0

            # ── IV + Greeks via BSM ───────────────────────────────
            bsm = {}
            if spot and spot > 0 and prima > 0 and T > 0:
                bsm = _bsm_calcular(spot, strike, T, r, prima, tipo_bsm)

            iv   = bsm.get("iv")

            # ── Metricas adicionales AR ───────────────────────────
            prob_otm = None
            tna_pct  = None
            tna_neta = None
            if iv and prima > 0 and dias > 0:
                prob_otm = round(_calcular_prob_otm(spot, strike, T, r, iv, tipo_bsm) * 100, 1)
                # TNA: para put base=strike, para call base=spot
                base_tna = strike if tipo_bsm == "put" else spot
                tna_pct  = round(_calcular_tna(prima, base_tna, dias), 1)
                tna_neta = round(tna_pct - (r * 100), 1)  # neta de tasa libre

            rows.append({
                "ticker_subyacente":  symbol,
                "symbol":             opt.get("simbolo") or "",
                "option_type":        option_type,
                "strike_price":       strike,
                "expiration":         expiration,
                "bid_price":          bid,
                "ask_price":          ask,
                "theoretical_price":  None,
                "implied_volatility": round(iv, 4) if iv else None,
                "delta":              bsm.get("delta"),
                "gamma":              bsm.get("gamma"),
                "theta":              bsm.get("theta"),
                "vega":               bsm.get("vega"),
                "rho":                bsm.get("rho"),
                "volume":             volumen if volumen > 0 else None,
                "spot_price":         spot,
                "risk_free_rate":     r,
                # Metricas AR adicionales (no van a DB, solo para UI)
                "prob_otm_pct":       prob_otm,
                "tna_pct":            tna_pct,
                "tna_neta_pct":       tna_neta,
                "prima_mercado":      round(prima, 2) if prima > 0 else None,
                "dias_vto":           dias,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Mantener solo filas con algun dato util
        has_data = (
            df["implied_volatility"].notna()
            | df["bid_price"].notna()
            | df["ask_price"].notna()
        )
        return df[has_data].reset_index(drop=True)

    # ── Info del activo ──────────────────────────────────────────

    def get_asset_info(self, symbol: str, market: str = "BCBA") -> dict:
        """Tipo, moneda, lot size y pares ARS/MEP/CCL si existen."""
        mercado = MARKET_MAP.get(market, market)
        return self._get(f"/api/v2/{mercado}/Titulos/{symbol}")
