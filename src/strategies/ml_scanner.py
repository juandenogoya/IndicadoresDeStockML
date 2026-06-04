"""
ml_scanner.py -- Decision COMPARTIDA de la estrategia ML Scanner.

FT_ML_SCANNER_v1 / Alpaca Bot 1 (Plan B, Tarea 16). Modulo PURO: sin DB,
sin broker. El ACCESO A DATOS (alertas del scanner) y el ESTADO DE CAPITAL
(cash, capital) los provee cada lado; la DECISION (que abrir/cerrar, cuanto)
vive aca.

ENTRADA: senales con nivel=COMPRA_FUERTE y score >= score_min, top-N por score
         que entren en cash. Sizing = min(capital*riesgo, cash) / precio.
SALIDA : earnings manana / score degradado / SL fijo / TP fijo.

Contrato de `senal` (dict): ticker, nivel, score, scan_fecha.
Contrato de `scanner_map[ticker]` (dict): nivel, score, scan_fecha.
"""

from dataclasses import dataclass


@dataclass
class ConfigML:
    """Parametros de la estrategia ML Scanner."""
    nombre:           str
    score_min:        float = 65
    nivel_min:        str   = "COMPRA_FUERTE"
    max_posiciones:   int   = 5
    max_deploy_pct:   float = 0.80   # techo de despliegue sobre capital_actual
    riesgo_por_trade: float = 0.15   # 15% del capital por trade
    sl_pct:           float = 0.05   # stop loss 5%
    tp_pct:           float = 0.10   # take profit 10%


def _noop(*_a, **_k):
    pass


def _qty(cash: float, precio: float, riesgo: float, capital_actual: float) -> int:
    """Shares para una operacion: min(capital*riesgo, cash) / precio. 0 si no alcanza."""
    base          = capital_actual if capital_actual else cash
    capital_trade = min(base * riesgo, cash)
    return max(int(capital_trade / precio), 0)


# ── Cierres ───────────────────────────────────────────────────────────────────

def evaluar_cierres_ml(
    posiciones:   list,
    precios:      dict,
    earnings_map: dict,
    scanner_map:  dict,
    cfg:          ConfigML,
    log=_noop,
) -> list:
    """
    Prioridades:
        P1. Earnings manana   (absoluta)
        P2. Score degradado   (sin senal, o nivel != COMPRA_FUERTE, o score < min)
        P3. Stop loss fijo
        P4. Take profit fijo
    """
    if not posiciones:
        return []

    a_cerrar = []
    for pos in posiciones:
        ticker         = pos["ticker"]
        precio_entrada = float(pos["precio_entrada"])
        precio_actual  = precios.get(ticker)
        sl             = round(precio_entrada * (1 - cfg.sl_pct), 4)
        tp             = round(precio_entrada * (1 + cfg.tp_pct), 4)

        if not precio_actual:
            log(f"  [WARN] Sin precio para {ticker}, se omite evaluacion.")
            continue

        motivo = None

        # P1: Earnings
        if ticker in earnings_map:
            motivo = "EARNINGS_MANANA"
        # P2: Score degradado
        elif ticker not in scanner_map:
            motivo = "SCORE_DEGRADADO_sin_senal"
        else:
            scan = scanner_map[ticker]
            if scan["nivel"] != cfg.nivel_min or float(scan["score"]) < cfg.score_min:
                motivo = f"SCORE_DEGRADADO_{scan['nivel']}_{scan['score']:.0f}"

        # P3: Stop loss
        if not motivo and precio_actual <= sl:
            motivo = "STOP_LOSS"
        # P4: Take profit
        if not motivo and precio_actual >= tp:
            motivo = "TAKE_PROFIT"

        if motivo:
            a_cerrar.append({**pos, "precio_cierre": precio_actual, "motivo": motivo})

    return a_cerrar


# ── Entradas ──────────────────────────────────────────────────────────────────

def evaluar_entradas_ml(
    posiciones:          list,
    senales:             list,
    precios:             dict,
    tickers_bloqueados:  set,
    cash:                float,
    capital_actual:      float,
    capital_inmovilizado: float,
    cfg:                 ConfigML,
    log=_noop,
) -> list:
    """
    Selecciona y dimensiona entradas a partir de las senales (ya filtradas a
    COMPRA_FUERTE + score >= min, ordenadas por score DESC).

    `cash` es el cash desplegable (respetando el techo de despliegue); el lado
    que llama lo provee desde su mundo (FT: estrategia; Alpaca: cuenta).
    """
    tickers_abiertos = {p["ticker"] for p in posiciones}
    bloqueados       = tickers_bloqueados | tickers_abiertos
    slots_libres     = cfg.max_posiciones - len(posiciones)

    if slots_libres <= 0:
        log("  Maximo de posiciones alcanzado, sin entradas.")
        return []

    if cash <= 0:
        pct = (capital_inmovilizado / capital_actual * 100) if capital_actual else 0.0
        log(f"  Techo de despliegue alcanzado ({pct:.1f}% desplegado, "
            f"max={cfg.max_deploy_pct*100:.0f}%). Sin entradas.")
        return []

    a_abrir = []
    for senal in senales:
        if len(a_abrir) >= slots_libres:
            break
        ticker = senal["ticker"]
        if ticker in bloqueados:
            continue
        precio = precios.get(ticker)
        if not precio:
            continue

        qty = _qty(cash, precio, cfg.riesgo_por_trade, capital_actual)
        if qty < 1:
            log(f"  [SKIP] {ticker}: capital insuficiente (precio={precio:.2f})")
            continue

        capital_trade = round(precio * qty, 2)
        if capital_trade > cash:
            log(f"  [SKIP] {ticker}: trade ({capital_trade:.2f}) supera cash ({cash:.2f})")
            continue

        sl = round(precio * (1 - cfg.sl_pct), 4)
        tp = round(precio * (1 + cfg.tp_pct), 4)

        a_abrir.append({
            "ticker":     ticker,
            "precio":     precio,
            "qty":        qty,
            "capital":    capital_trade,
            "sl":         sl,
            "tp":         tp,
            "score":      float(senal["score"]),
            "nivel":      senal["nivel"],
            "scan_fecha": str(senal["scan_fecha"]),
        })
        cash -= capital_trade

    return a_abrir
