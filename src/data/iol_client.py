"""
iol_client.py
Cliente autenticado para la API REST de InvertirOnline (IOL).
Maneja autenticacion OAuth2 con refresh automatico de token.

Credenciales en .env:
    IOL_USERNAME=...
    IOL_PASSWORD=...
"""

import os
import time
import requests
import pandas as pd
from datetime import date, timedelta
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

# Tickers BCBA con opciones activas en el panel
TICKERS_CON_OPCIONES = {
    "GGAL", "BMA", "YPFD", "PAM", "TGS", "EDN",
    "CEPU", "TXAR", "ALU", "VALO", "MIRG", "BYMA",
}


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

    def get_options_chain_raw(self, symbol: str) -> dict:
        """Chain completo de opciones incluyendo Greeks. Solo util en horario BCBA."""
        return self._get(f"/api/v2/Titulos/{symbol}/opciones")

    def get_options_chain_df(self, symbol: str) -> pd.DataFrame:
        """
        Chain filtrado: solo strikes liquidos (is_stale=False, IV no nula).
        Listo para insertar en opciones_ar_gregas.
        """
        data = self.get_options_chain_raw(symbol)

        spot        = data.get("precioSubyacente") or data.get("spot_price")
        risk_free   = data.get("tasaLibreDeRiesgo") or data.get("risk_free_rate")
        options_raw = data.get("opciones") or data.get("options") or []

        rows = []
        for opt in options_raw:
            # Filtrar sin datos reales
            iv = opt.get("volatilidad") or opt.get("implied_volatility")
            stale = opt.get("estaVencida") if "estaVencida" in opt else opt.get("is_stale")
            if iv is None or stale is True:
                continue

            tipo_raw = opt.get("tipo") or opt.get("option_type") or ""
            option_type = "C" if str(tipo_raw).lower() in ("call", "c") else "V"

            exp_raw = opt.get("vencimiento") or opt.get("expiration") or ""
            expiration = str(exp_raw)[:10]

            rows.append({
                "ticker_subyacente": symbol,
                "symbol":            opt.get("simbolo") or opt.get("symbol"),
                "option_type":       option_type,
                "strike_price":      opt.get("strike") or opt.get("strike_price"),
                "expiration":        expiration,
                "bid_price":         opt.get("bid") or opt.get("bid_price"),
                "ask_price":         opt.get("ask") or opt.get("ask_price"),
                "theoretical_price": opt.get("precioTeorico") or opt.get("theoretical_price"),
                "implied_volatility": iv,
                "delta":             opt.get("delta"),
                "gamma":             opt.get("gamma"),
                "theta":             opt.get("theta"),
                "vega":              opt.get("vega"),
                "rho":               opt.get("rho"),
                "volume":            opt.get("volumenNominal") or opt.get("volume"),
                "spot_price":        spot,
                "risk_free_rate":    risk_free,
            })

        return pd.DataFrame(rows)

    # ── Info del activo ──────────────────────────────────────────

    def get_asset_info(self, symbol: str, market: str = "BCBA") -> dict:
        """Tipo, moneda, lot size y pares ARS/MEP/CCL si existen."""
        mercado = MARKET_MAP.get(market, market)
        return self._get(f"/api/v2/{mercado}/Titulos/{symbol}")
