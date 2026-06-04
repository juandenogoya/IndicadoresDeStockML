"""
sectorial.py -- Decision COMPARTIDA de las estrategias tecnico-sectoriales.

Cubre dos variantes (Plan B, Tarea 16), seleccionadas por ConfigSectorial:
  - v1  (usar_opciones=False): solo score tecnico. FT_TECH_SECTOR_v1 / Alpaca Bot 2.
  - v2  (usar_opciones=True) : score tecnico + gate de opciones PCR_VOL.
                               FT_TECH_SECTOR_OPTIONS_v2 / Alpaca Bot 3.

Modulo PURO: sin DB, sin broker. Recibe data (dicts con la forma de
senales_bot_diaria) y devuelve decisiones (a_abrir / a_cerrar / retenidos).
El logging se inyecta (`log`) para que cada lado (FT, Alpaca) use el suyo y
la salida sea fiel.

OBSERVACION (preservada, no "corregida"): el ORDEN de prioridad de cierres
difiere entre v1 y v2.
  v1: earnings -> score degradado -> SL -> TP   (score pesa mas que SL/TP)
  v2: earnings -> SL -> TP -> score+PCR         (SL/TP de emergencia primero)
Se reproduce tal cual cada uno para no alterar el comportamiento historico.

Contrato del dict de indicadores (`ind`) -- claves de senales_bot_diaria:
    ticker, sector, close, sma21, sma50, sma200, rsi14, macd, macd_signal, atr14
Contrato del dict de PCR (`pcr_map[ticker]`):
    corto, medio, largo ('A'/'B'/None), pcr_score (0-3), valido (bool)
"""

from dataclasses import dataclass, field

from src.strategies.scoring import calcular_score_tecnico, SCORE_MAXIMO_TECH


@dataclass
class ConfigSectorial:
    """Parametros de una estrategia sectorial (v1 o v2)."""
    nombre:           str
    sectores_activos: list
    sector_budget:    float
    max_pos_sector:   int
    position_size:    float
    score_entrada:    float = 4.0
    score_salida:     float = 3.5
    atr_mult_sl:      float = 2.0
    atr_mult_tp:      float = 4.0
    score_maximo:     float = SCORE_MAXIMO_TECH
    # Gate de opciones (solo v2)
    usar_opciones:         bool = False
    pcr_score_entrada_min: int  = 2   # >= N/3 ventanas alcistas para entrar
    pcr_score_salida_max:  int  = 0   # <= N -> cerrar (con score degradado)


def _noop(*_a, **_k):
    pass


# ── Cierres ───────────────────────────────────────────────────────────────────

def evaluar_cierres_sectorial(
    posiciones:      list,
    precios:         dict,
    earnings_map:    dict,
    indicadores_map: dict,
    cfg:             ConfigSectorial,
    pcr_map:         dict = None,
    log=_noop,
) -> tuple:
    """
    Evalua que posiciones cerrar. Retorna (a_cerrar, retenidos).
    `retenidos` solo se puebla en v2 (retencion por opciones); en v1 es [].
    """
    pcr_map   = pcr_map or {}
    a_cerrar  = []
    retenidos = []

    for pos in posiciones:
        ticker  = pos["ticker"]
        precio  = precios.get(ticker)
        sl      = float(pos["stop_loss"])   if pos.get("stop_loss")   else None
        tp      = float(pos["take_profit"]) if pos.get("take_profit") else None
        ind     = indicadores_map.get(ticker)

        if not precio:
            log(f"  [WARN] Sin precio para {ticker}, se omite.")
            continue

        motivo = None

        if not cfg.usar_opciones:
            # ── v1: earnings -> score -> SL -> TP ──
            if ticker in earnings_map:
                motivo = "EARNINGS_MANANA"
            elif ind:
                score_actual, _ = calcular_score_tecnico(ind)
                if score_actual <= cfg.score_salida:
                    motivo = f"SCORE_DEGRADADO_{score_actual:.1f}"
            if not motivo and sl and precio <= sl:
                motivo = "STOP_LOSS_ATR"
            if not motivo and tp and precio >= tp:
                motivo = "TAKE_PROFIT_ATR"
        else:
            # ── v2: earnings -> SL -> TP -> score+PCR ──
            pcr_info = pcr_map.get(ticker, {})
            if ticker in earnings_map:
                motivo = "EARNINGS_MANANA"
            elif sl and precio <= sl:
                motivo = "STOP_LOSS_ATR"
            elif tp and precio >= tp:
                motivo = "TAKE_PROFIT_ATR"
            elif ind:
                score_actual, _ = calcular_score_tecnico(ind)
                if score_actual <= cfg.score_salida:
                    pcr_score  = pcr_info.get("pcr_score", 0)
                    pcr_valido = pcr_info.get("valido", False)
                    if not pcr_valido or pcr_score <= cfg.pcr_score_salida_max:
                        motivo = "SCORE_DEGRADADO_OPCIONES"
                    else:
                        v_corto = pcr_info.get("corto", "?")
                        v_medio = pcr_info.get("medio", "?")
                        v_largo = pcr_info.get("largo", "?")
                        log(f"  [RETENCION_OPCIONES] {ticker}: "
                            f"score={score_actual:.1f} pcr_score={pcr_score}/3 "
                            f"({v_corto}/{v_medio}/{v_largo})")
                        retenidos.append(ticker)

        if motivo:
            a_cerrar.append({**pos, "precio_cierre": precio, "motivo": motivo})

    return a_cerrar, retenidos


# ── Entradas por sector ───────────────────────────────────────────────────────

def evaluar_entradas_sectorial(
    posiciones:         list,
    indicadores:        list,
    precios:            dict,
    tickers_bloqueados: set,
    tickers_cerrados:   set,
    cfg:                ConfigSectorial,
    pcr_map:            dict = None,
    log=_noop,
) -> tuple:
    """
    Para cada sector con slots/capital: aplica gate(s), rankea y abre.
        Gate 1 (siempre): tech_score >= cfg.score_entrada
        Gate 2 (v2):      pcr_valido Y pcr_score >= cfg.pcr_score_entrada_min
        Ranking: v1 -> score DESC;  v2 -> (score, pcr_score) DESC
    Retorna (a_abrir, candidatos_log).
    """
    pcr_map          = pcr_map or {}
    tickers_abiertos = {p["ticker"] for p in posiciones}
    excluidos        = tickers_bloqueados | tickers_abiertos | tickers_cerrados

    sector_deployed = {}
    sector_n_open   = {}
    for pos in posiciones:
        s = pos.get("sector", "Unknown")
        sector_deployed[s] = sector_deployed.get(s, 0.0) + float(pos["capital_entrada"])
        sector_n_open[s]   = sector_n_open.get(s, 0) + 1

    candidatos_por_sector = {s: [] for s in cfg.sectores_activos}
    candidatos_log = []

    for ind in indicadores:
        t = ind["ticker"]
        s = ind.get("sector", "Unknown")
        if t in excluidos or s not in cfg.sectores_activos:
            continue

        score, detalle = calcular_score_tecnico(ind)
        if score < cfg.score_entrada:
            continue

        pcr_info  = pcr_map.get(t, {})
        pcr_score = pcr_info.get("pcr_score", 0)

        if cfg.usar_opciones:
            if not pcr_info.get("valido", False):
                candidatos_log.append({
                    "ticker": t, "score": float(score), "entro": False,
                    "motivo_skip": "SIN_LIQUIDEZ_OPCIONES",
                    "precio_apertura": precios.get(t),
                })
                continue
            if pcr_score < cfg.pcr_score_entrada_min:
                v = (pcr_info.get("corto", "?"), pcr_info.get("medio", "?"),
                     pcr_info.get("largo", "?"))
                candidatos_log.append({
                    "ticker": t, "score": float(score), "entro": False,
                    "motivo_skip": f"PCR_CONSENSO_{v[0]}{v[1]}{v[2]}",
                    "precio_apertura": precios.get(t),
                })
                continue

        candidatos_por_sector[s].append({
            "ticker": t, "sector": s, "score": score, "pcr_score": pcr_score,
            "detalle": detalle, "pcr_info": pcr_info, "ind": ind,
        })

    # Ranking: v2 desempata por pcr_score; v1 solo score
    for s in candidatos_por_sector:
        if cfg.usar_opciones:
            candidatos_por_sector[s].sort(key=lambda x: (x["score"], x["pcr_score"]), reverse=True)
        else:
            candidatos_por_sector[s].sort(key=lambda x: x["score"], reverse=True)

    a_abrir = []

    for sector in cfg.sectores_activos:
        candidatos = candidatos_por_sector[sector]
        deployed   = sector_deployed.get(sector, 0.0)
        n_open     = sector_n_open.get(sector, 0)
        available  = round(cfg.sector_budget - deployed, 2)
        slots      = cfg.max_pos_sector - n_open

        if not candidatos:
            continue
        if slots <= 0:
            log(f"  [{sector}] Max posiciones ({cfg.max_pos_sector}) alcanzado.")
            continue
        if available <= 0:
            log(f"  [{sector}] Sin capital disponible ({deployed:,.2f}/{cfg.sector_budget:,.2f}).")
            continue

        log(f"  [{sector}] {len(candidatos)} qualifying | slots={slots} | "
            f"disponible=${available:,.2f}")

        avail_local = available
        for c in candidatos:
            if slots <= 0 or avail_local <= 0:
                break
            ticker = c["ticker"]
            precio = precios.get(ticker)
            if not precio:
                continue
            atr = float(c["ind"].get("atr14") or 0)
            if not atr:
                log(f"    [SKIP] {ticker}: sin ATR14.")
                continue
            qty = int(cfg.position_size / precio)
            if qty < 1:
                log(f"    [SKIP] {ticker}: precio ${precio:.2f} > position_size "
                    f"${cfg.position_size:,.2f}.")
                continue
            capital_trade = round(precio * qty, 2)
            if capital_trade > avail_local:
                log(f"    [SKIP] {ticker}: trade ${capital_trade:,.2f} > "
                    f"disponible ${avail_local:,.2f}.")
                continue

            sl = round(precio - cfg.atr_mult_sl * atr, 4)
            tp = round(precio + cfg.atr_mult_tp * atr, 4)

            a_abrir.append({
                "ticker": ticker, "sector": sector, "precio": precio, "qty": qty,
                "capital": capital_trade, "sl": sl, "tp": tp,
                "score": c["score"], "pcr_score": c["pcr_score"],
                "pcr_info": c["pcr_info"], "detalle": c["detalle"],
                "atr": round(atr, 4),
            })
            candidatos_log.append({
                "ticker": ticker, "score": float(c["score"]), "entro": True,
                "motivo_skip": None, "precio_apertura": precio,
            })
            avail_local -= capital_trade
            slots       -= 1

        # Qualifying que no entraron por slots/capital
        abiertos = {e["ticker"] for e in a_abrir}
        for c in candidatos:
            if c["ticker"] not in abiertos and not any(
                cl["ticker"] == c["ticker"] for cl in candidatos_log
            ):
                candidatos_log.append({
                    "ticker": c["ticker"], "score": float(c["score"]), "entro": False,
                    "motivo_skip": f"CAPITAL_O_SLOTS_{sector}",
                    "precio_apertura": precios.get(c["ticker"]),
                })

    return a_abrir, candidatos_log
