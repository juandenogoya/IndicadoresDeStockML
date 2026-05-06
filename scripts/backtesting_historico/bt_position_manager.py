"""
bt_position_manager.py
Gestiona el estado de posiciones y capital en memoria durante la simulacion.

Sin queries a DB. Todo en dicts de Python.
Una instancia por corrida de backtesting.

Terminologia:
    bt_id       : ID de la instancia en bt_hist_estrategias
    logica      : "tecnico_sectorial" | "combo_tech_candle" | "smc_estructura"
    posicion    : dict con campos equivalentes a bt_hist_operaciones
"""

from datetime import date
from typing import Optional

import bt_scoring as sc


class BtPositionManager:
    """
    Estado completo de una corrida de backtesting en memoria.

    Tres modos de capital segun la logica:
      tecnico_sectorial / combo_tech_candle:
        - 9 sectores x $11,111 fijo
        - Max 5 posiciones por sector; ~$2,222 por trade (20% del budget)
        - cash global = suma de budgets libres (no usado en practica; el control es por sector)

      smc_estructura:
        - Capital global $100,000
        - Max 5 posiciones totales
        - Nunca deployar mas del 80% del capital total
        - 15% del capital disponible por trade
    """

    def __init__(self, bt_id: int, logica: str, capital_inicial: float = 100_000.0):
        self.bt_id          = bt_id
        self.logica         = logica
        self.capital_inicial = capital_inicial

        # Posiciones abiertas: {ticker: dict}
        self._abiertas: dict[str, dict] = {}

        # Historial de trades cerrados (para escritura final)
        self._cerradas: list[dict] = []

        # Metricas diarias (para bt_hist_metricas_diarias)
        self._metricas_diarias: list[dict] = []

        # Candidatos evaluados (para bt_hist_candidatos)
        self._candidatos: list[dict] = []

        # Pico de capital (para drawdown)
        self._peak_capital = capital_inicial

        # Estado de capital segun logica
        if logica in ("tecnico_sectorial", "combo_tech_candle"):
            # Capital distribuido por sector; sector_budgets[sector] = capital libre
            self._sector_budgets: dict[str, float] = {}
            self._usar_sectores = True
        else:
            # Capital global SMC
            self._cash = capital_inicial
            self._usar_sectores = False

    # ── Capital sectorial ──────────────────────────────────────────────────────

    def init_sectores(self, sectores: list[str]):
        """Inicializa los budgets sectoriales. Llamar una vez al inicio."""
        for s in sectores:
            self._sector_budgets[s] = sc.SECTOR_BUDGET

    def capital_disponible_sector(self, sector: str) -> float:
        if not self._usar_sectores:
            return 0.0
        return self._sector_budgets.get(sector, 0.0)

    def posiciones_abiertas_sector(self, sector: str) -> int:
        return sum(1 for p in self._abiertas.values() if p.get("sector") == sector)

    def puede_entrar_sector(self, sector: str, capital_trade: float) -> tuple[bool, str]:
        """Verifica si hay cupo de posicion y capital en el sector."""
        n_pos = self.posiciones_abiertas_sector(sector)
        if n_pos >= sc.MAX_POS_SECTOR:
            return False, f"max_pos_sector ({n_pos}/{sc.MAX_POS_SECTOR})"
        cap_libre = self.capital_disponible_sector(sector)
        if cap_libre < capital_trade:
            return False, f"sin_capital_sector (libre={cap_libre:.0f} < trade={capital_trade:.0f})"
        return True, ""

    # ── Capital SMC ────────────────────────────────────────────────────────────

    def puede_entrar_smc(self, capital_trade: float) -> tuple[bool, str]:
        """Verifica si hay cupo global y capital para una posicion SMC."""
        n_pos = len(self._abiertas)
        if n_pos >= sc.MAX_POSICIONES_SMC:
            return False, f"max_posiciones ({n_pos}/{sc.MAX_POSICIONES_SMC})"
        capital_total = self.capital_total()
        if (self.capital_invertido() + capital_trade) / capital_total > sc.MAX_DEPLOY_PCT:
            return False, f"max_deploy ({sc.MAX_DEPLOY_PCT*100:.0f}%)"
        if self._cash < capital_trade:
            return False, f"sin_cash ({self._cash:.0f} < {capital_trade:.0f})"
        return True, ""

    # ── Abrir posicion ─────────────────────────────────────────────────────────

    def abrir(self, ticker: str, fecha: date, precio: float,
              stop_loss: float, score: float, detalle: dict,
              atr14: float = 0.0, take_profit: float = 0.0,
              sector: Optional[str] = None) -> dict:
        """
        Registra una nueva posicion abierta.
        Retorna el dict de la posicion creada.
        """
        if self._usar_sectores:
            budget_sector = self._sector_budgets.get(sector or "", 0.0)
            capital_trade = round(budget_sector * sc.POSITION_PCT, 2)
            cantidad      = max(1, int(capital_trade / precio))
            capital_real  = round(cantidad * precio, 2)
            # Descontar del budget sectorial
            self._sector_budgets[sector] = round(budget_sector - capital_real, 2)
        else:
            capital_trade = round(self._cash * sc.RIESGO_POR_TRADE_SMC, 2)
            cantidad      = max(1, int(capital_trade / precio))
            capital_real  = round(cantidad * precio, 2)
            self._cash    = round(self._cash - capital_real, 2)

        pos = {
            "bt_id":          self.bt_id,
            "ticker":         ticker,
            "lado":           "long",
            "fecha_entrada":  fecha,
            "precio_entrada": precio,
            "cantidad":       cantidad,
            "capital_entrada": capital_real,
            "stop_loss":      stop_loss,
            "take_profit":    take_profit or None,
            "score_entrada":  score,
            "detalle_entrada": detalle,
            "sector":         sector,
            "atr14":          atr14,
            "fecha_salida":   None,
            "precio_salida":  None,
            "pnl":            None,
            "pnl_pct":        None,
            "motivo_salida":  None,
            "dias_abierta":   None,
        }
        self._abiertas[ticker] = pos
        return pos

    # ── Cerrar posicion ────────────────────────────────────────────────────────

    def cerrar(self, ticker: str, fecha: date, precio_salida: float,
               motivo: str, dias_habiles: int) -> Optional[dict]:
        """
        Cierra la posicion del ticker. Retorna el dict cerrado o None si no existe.
        """
        pos = self._abiertas.pop(ticker, None)
        if pos is None:
            return None

        cantidad      = pos["cantidad"]
        precio_entrada = pos["precio_entrada"]
        capital_real  = pos["capital_entrada"]

        pnl     = round((precio_salida - precio_entrada) * cantidad, 2)
        pnl_pct = round((precio_salida - precio_entrada) / precio_entrada * 100, 4)

        pos.update({
            "fecha_salida":  fecha,
            "precio_salida": precio_salida,
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "motivo_salida": motivo,
            "dias_abierta":  dias_habiles,
        })

        # Devolver capital al pool correspondiente
        capital_recuperado = round(capital_real + pnl, 2)
        if self._usar_sectores:
            sector = pos.get("sector") or ""
            self._sector_budgets[sector] = round(
                self._sector_budgets.get(sector, 0) + capital_recuperado, 2
            )
        else:
            self._cash = round(self._cash + capital_recuperado, 2)

        self._cerradas.append(pos)
        return pos

    # ── Actualizar trailing SL (SMC) ───────────────────────────────────────────

    def actualizar_sl(self, ticker: str, nuevo_sl: float) -> bool:
        """Actualiza el stop_loss solo si nuevo_sl > sl_actual. Retorna True si cambio."""
        pos = self._abiertas.get(ticker)
        if pos is None:
            return False
        sl_actual = pos.get("stop_loss") or 0.0
        if nuevo_sl > sl_actual:
            pos["stop_loss"] = nuevo_sl
            return True
        return False

    # ── Metricas ───────────────────────────────────────────────────────────────

    def capital_invertido(self) -> float:
        """Capital actual en posiciones abiertas (al precio de entrada)."""
        return sum(p["capital_entrada"] for p in self._abiertas.values())

    def capital_total(self, precios_hoy: Optional[dict] = None) -> float:
        """
        Capital total = cash + valor de mercado de posiciones abiertas.
        Si precios_hoy es None, usa precio de entrada (conservador).
        """
        if self._usar_sectores:
            cash = sum(self._sector_budgets.values())
        else:
            cash = self._cash

        if not self._abiertas:
            return cash

        valor_posiciones = 0.0
        for ticker, pos in self._abiertas.items():
            if precios_hoy and ticker in precios_hoy:
                precio = precios_hoy[ticker]
            else:
                precio = pos["precio_entrada"]
            valor_posiciones += precio * pos["cantidad"]

        return round(cash + valor_posiciones, 2)

    def registrar_dia(self, fecha: date, precios_hoy: dict, ops_cerradas_dia: int) -> dict:
        """Calcula y guarda la metrica diaria. Retorna el dict para bt_hist_metricas_diarias."""
        capital_total   = self.capital_total(precios_hoy)
        capital_inv     = sum(
            precios_hoy.get(t, p["precio_entrada"]) * p["cantidad"]
            for t, p in self._abiertas.items()
        )

        if self._usar_sectores:
            cash_disp = sum(self._sector_budgets.values())
        else:
            cash_disp = self._cash

        # Retorno acumulado
        ret_acum = round((capital_total - self.capital_inicial) / self.capital_inicial * 100, 4)

        # Drawdown
        if capital_total > self._peak_capital:
            self._peak_capital = capital_total
        dd_pct = round((capital_total - self._peak_capital) / self._peak_capital * 100, 4)

        # PnL del dia (vs dia anterior)
        if self._metricas_diarias:
            pnl_dia = round(capital_total - self._metricas_diarias[-1]["capital_total"], 2)
        else:
            pnl_dia = 0.0

        metrica = {
            "bt_id":                    self.bt_id,
            "fecha":                    fecha,
            "capital_total":            capital_total,
            "cash_disponible":          round(cash_disp, 2),
            "capital_inmovilizado":     round(capital_inv, 2),
            "posiciones_abiertas":      len(self._abiertas),
            "operaciones_cerradas_dia": ops_cerradas_dia,
            "pnl_dia":                  pnl_dia,
            "retorno_acumulado_pct":    ret_acum,
            "drawdown_pct":             dd_pct,
        }
        self._metricas_diarias.append(metrica)
        return metrica

    # ── Registro de candidatos ─────────────────────────────────────────────────

    def registrar_candidato(self, bt_id: int, fecha: date, ticker: str,
                            score: float, entro: bool, motivo_skip: str = "",
                            precio: float = 0.0):
        self._candidatos.append({
            "bt_id":          bt_id,
            "fecha":          fecha,
            "ticker":         ticker,
            "score":          score,
            "entro":          entro,
            "motivo_skip":    motivo_skip or None,
            "precio_apertura": precio or None,
        })

    # ── Cierre de posiciones al fin del periodo ────────────────────────────────

    def cerrar_todas_fin_periodo(self, fecha_fin: date, precios: dict,
                                 trading_days: list) -> list[dict]:
        """Cierra todas las posiciones abiertas al final del periodo (FIN_PERIODO)."""
        cerradas = []
        for ticker in list(self._abiertas.keys()):
            pos = self._abiertas[ticker]
            precio = precios.get(ticker, pos["precio_entrada"])
            fecha_entrada = pos["fecha_entrada"]
            dias_hab = sum(
                1 for d in trading_days
                if fecha_entrada <= d <= fecha_fin
            )
            c = self.cerrar(ticker, fecha_fin, precio, "FIN_PERIODO", dias_hab)
            if c:
                cerradas.append(c)
        return cerradas

    # ── Acceso a resultados ────────────────────────────────────────────────────

    @property
    def posiciones_abiertas(self) -> dict[str, dict]:
        return self._abiertas

    @property
    def operaciones_cerradas(self) -> list[dict]:
        return self._cerradas

    @property
    def metricas_diarias(self) -> list[dict]:
        return self._metricas_diarias

    @property
    def candidatos(self) -> list[dict]:
        return self._candidatos

    def resumen_metricas(self) -> dict:
        """Calcula metricas resumen al finalizar el BT (para bt_hist_estrategias)."""
        todas_ops = self._cerradas
        if not todas_ops:
            return {
                "retorno_total_pct": 0.0,
                "drawdown_max_pct":  0.0,
                "win_rate_pct":      0.0,
                "profit_factor":     0.0,
                "total_operaciones": 0,
                "dias_promedio_pos": 0.0,
                "sharpe_simplificado": 0.0,
            }

        retorno_total = round(
            (self.capital_total() - self.capital_inicial) / self.capital_inicial * 100, 4
        )

        dds = [m["drawdown_pct"] for m in self._metricas_diarias if m["drawdown_pct"] is not None]
        drawdown_max = round(min(dds), 4) if dds else 0.0

        ganadores  = [o["pnl"] for o in todas_ops if (o["pnl"] or 0) > 0]
        perdedores = [o["pnl"] for o in todas_ops if (o["pnl"] or 0) <= 0]

        win_rate = round(len(ganadores) / len(todas_ops) * 100, 2)

        suma_gan  = sum(ganadores)
        suma_perd = abs(sum(perdedores))
        profit_factor = round(suma_gan / suma_perd, 4) if suma_perd > 0 else 0.0

        dias_pos = [o["dias_abierta"] for o in todas_ops if o.get("dias_abierta")]
        dias_prom = round(sum(dias_pos) / len(dias_pos), 2) if dias_pos else 0.0

        retornos_diarios = [m["pnl_dia"] / (m["capital_total"] - m["pnl_dia"])
                            for m in self._metricas_diarias
                            if m.get("pnl_dia") and m.get("capital_total") and
                            (m["capital_total"] - m["pnl_dia"]) > 0]

        if len(retornos_diarios) > 1:
            import statistics
            avg_ret = statistics.mean(retornos_diarios)
            std_ret = statistics.stdev(retornos_diarios)
            sharpe  = round(avg_ret / std_ret * (252 ** 0.5), 4) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "retorno_total_pct":   retorno_total,
            "drawdown_max_pct":    drawdown_max,
            "win_rate_pct":        win_rate,
            "profit_factor":       profit_factor,
            "total_operaciones":   len(todas_ops),
            "dias_promedio_pos":   dias_prom,
            "sharpe_simplificado": sharpe,
        }
