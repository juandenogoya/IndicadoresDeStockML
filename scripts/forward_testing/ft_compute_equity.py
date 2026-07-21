"""
ft_compute_equity.py
Reconstruye la equity curve MARCADA A MERCADO de cada estrategia FT
y la persiste en ft_equity_diaria.

Diseno completo: docs/forward_testing/METRICAS.md

Por que existe:
    ft_metricas_diarias.capital_total esta a COSTO de entrada y solo se
    mueve cuando cierra una operacion -> es una curva de PnL realizado.
    Sobre ella el max drawdown de posiciones abiertas es invisible y toda
    metrica de riesgo subestima el riesgo.

Principio:
    La equity diaria es una FUNCION PURA de ft_operaciones + precios_diarios.
    Por lo tanto es recomputable hacia atras sin datos nuevos, y no depende
    de que el bot haya corrido ese dia (la rutina nocturna es manual y le
    faltaba el 34% de los dias habiles).

    Este script SOLO LEE ft_operaciones/ft_estrategias/precios_diarios y
    SOLO ESCRIBE ft_equity_diaria. No toca la logica de trading.

Definiciones (posicion abierta al cierre del dia d):
    fecha_entrada <= d AND (fecha_salida IS NULL OR fecha_salida > d)

    cash(d)          = capital_inicial
                       - SUM(capital_entrada)          [fecha_entrada <= d]
                       + SUM(precio_salida * cantidad) [fecha_salida  <= d]
    valor_mercado(d) = SUM(close(ticker, d) * cantidad) sobre las abiertas
    equity(d)        = cash(d) + valor_mercado(d)

Uso:
    python scripts/forward_testing/ft_compute_equity.py               # todas, incremental
    python scripts/forward_testing/ft_compute_equity.py --rebuild     # recalcula todo
    python scripts/forward_testing/ft_compute_equity.py --estrategia 4
    python scripts/forward_testing/ft_compute_equity.py --desde 2026-06-01
    python scripts/forward_testing/ft_compute_equity.py --check       # solo cuadre
    python scripts/forward_testing/ft_compute_equity.py --dry-run
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ft_env import configurar_entorno_local  # noqa: E402
configurar_entorno_local()

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402
from src.utils.trading_calendar import is_trading_day  # noqa: E402

# Tolerancia del control de cuadre, en dolares. El cash se arma de sumas de
# NUMERIC(12,2) redondeados, asi que un centavo de deriva es esperable.
TOLERANCIA_CUADRE = 0.05


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Carga ─────────────────────────────────────────────────────────────────────

def cargar_estrategias(engine, estrategia_id=None):
    q = """
        SELECT id, nombre, capital_inicial, cash_disponible, fecha_inicio
        FROM ft_estrategias
    """
    params = {}
    if estrategia_id:
        q += " WHERE id = :eid"
        params["eid"] = estrategia_id
    q += " ORDER BY id"
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn, params=params)


def cargar_operaciones(engine, ids):
    """
    Carga las operaciones usando la FECHA DEL DATO, no la de registro.

    `fecha_entrada` / `fecha_salida` son cuando se REGISTRO la operacion;
    `fecha_datos` / `fecha_datos_salida` son la fecha del OHLCV con el que se
    decidio y al que se ejecuto. El sistema es asincronico y el desfase NO es
    fijo: 18% mismo dia, 69% un dia, 8% entre 2 y 6 dias habiles (depende de
    cuan rancia estaba la data cuando corrio el bot).

    Usar la fecha de registro haria que una posicion entrada con datos de hace
    5 dias se marcara por primera vez recien el dia del registro, metiendo
    CINCO dias de movimiento de mercado como un salto de un solo dia -> infla
    la volatilidad y distorsiona el drawdown.

    COALESCE por si alguna quedara sin resolver: se degrada a la fecha de
    registro en vez de perder la operacion.
    """
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT estrategia_id, ticker,
                   COALESCE(fecha_datos, fecha_entrada)       AS fecha_entrada,
                   COALESCE(fecha_datos_salida, fecha_salida) AS fecha_salida,
                   fecha_entrada AS fecha_registro,
                   precio_entrada, cantidad, capital_entrada,
                   precio_salida, pnl
            FROM ft_operaciones
            WHERE estrategia_id = ANY(:ids)
        """), conn, params={"ids": list(ids)})

    for col in ("fecha_entrada", "fecha_salida", "fecha_registro"):
        df[col] = pd.to_datetime(df[col])
    for col in ("precio_entrada", "cantidad", "capital_entrada", "precio_salida", "pnl"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def cargar_precios(engine, tickers, desde, hasta):
    """
    Matriz precio[fecha][ticker] reindexada a dias habiles con forward-fill.

    Devuelve (pivot_ffill, mask_faltante). mask_faltante marca las celdas que
    NO tenian close propio ese dia (se marcaron con un precio arrastrado) para
    poder contarlas en ft_equity_diaria.precios_stale en vez de esconderlas.
    """
    with engine.connect() as conn:
        px = pd.read_sql(text("""
            SELECT ticker, fecha, close
            FROM precios_diarios
            WHERE ticker = ANY(:tickers) AND fecha BETWEEN :desde AND :hasta
        """), conn, params={"tickers": list(tickers), "desde": desde, "hasta": hasta})

    px["fecha"] = pd.to_datetime(px["fecha"])
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    pivot = px.pivot(index="fecha", columns="ticker", values="close").sort_index()

    dias = [d for d in pd.date_range(desde, hasta, freq="D") if is_trading_day(d.date())]
    pivot = pivot.reindex(dias)
    mask_faltante = pivot.isna()
    return pivot.ffill(), mask_faltante


# ── Construccion de la serie ──────────────────────────────────────────────────

def construir_serie(est, ops, dias, pivot, mask_faltante):
    """
    Serie diaria de una estrategia. Devuelve lista de dicts (una por dia habil).

    Coherencia en los bordes: el precio de ejecucion del FT es el close del
    dia, entonces el dia de ENTRADA close*cantidad == capital_entrada (la
    equity no salta) y el dia de SALIDA la posicion ya no cuenta pero el
    efectivo ya entro por precio_salida. Sin doble conteo.
    """
    capital_inicial = float(est["capital_inicial"])
    filas = []
    equity_prev = None

    for d in dias:
        entradas = ops[ops["fecha_entrada"] <= d]
        cerradas = ops[ops["fecha_salida"].notna() & (ops["fecha_salida"] <= d)]

        cash = (capital_inicial
                - float(entradas["capital_entrada"].sum())
                + float((cerradas["precio_salida"] * cerradas["cantidad"]).sum()))

        abiertas = ops[(ops["fecha_entrada"] <= d)
                       & (ops["fecha_salida"].isna() | (ops["fecha_salida"] > d))]

        valor_mercado = 0.0
        stale = 0
        for _, op in abiertas.iterrows():
            tk, qty = op["ticker"], float(op["cantidad"])
            precio = pivot.at[d, tk] if tk in pivot.columns else None

            if precio is None or pd.isna(precio):
                # Sin close propio ni arrastrable: se usa el precio de entrada.
                # Conservador (PnL no realizado 0) y queda contado como stale.
                precio = float(op["precio_entrada"])
                stale += 1
            elif tk in mask_faltante.columns and bool(mask_faltante.at[d, tk]):
                stale += 1

            valor_mercado += float(precio) * qty

        costo = float(abiertas["capital_entrada"].sum())
        equity = cash + valor_mercado
        pnl_dia = float(ops[ops["fecha_salida"] == d]["pnl"].sum())

        filas.append({
            "fecha":             d.date(),
            "equity":            round(equity, 2),
            "cash":              round(cash, 2),
            "valor_mercado":     round(valor_mercado, 2),
            "costo_posiciones":  round(costo, 2),
            "n_posiciones":      int(len(abiertas)),
            "exposicion_pct":    round(valor_mercado / equity, 6) if equity else None,
            "pnl_realizado_dia": round(pnl_dia, 2),
            "pnl_no_realizado":  round(valor_mercado - costo, 2),
            "retorno_dia_pct":   (round((equity / equity_prev - 1) * 100, 6)
                                  if equity_prev else None),
            "retorno_acum_pct":  round((equity / capital_inicial - 1) * 100, 4),
            "precios_stale":     stale,
        })
        equity_prev = equity

    return filas


# ── Persistencia ──────────────────────────────────────────────────────────────

UPSERT = """
INSERT INTO ft_equity_diaria (
    estrategia_id, fecha, equity, cash, valor_mercado, costo_posiciones,
    n_posiciones, exposicion_pct, pnl_realizado_dia, pnl_no_realizado,
    retorno_dia_pct, retorno_acum_pct, precios_stale, calculado_en
) VALUES (
    :eid, :fecha, :equity, :cash, :valor_mercado, :costo_posiciones,
    :n_posiciones, :exposicion_pct, :pnl_realizado_dia, :pnl_no_realizado,
    :retorno_dia_pct, :retorno_acum_pct, :precios_stale, NOW()
)
ON CONFLICT (estrategia_id, fecha) DO UPDATE SET
    equity            = EXCLUDED.equity,
    cash              = EXCLUDED.cash,
    valor_mercado     = EXCLUDED.valor_mercado,
    costo_posiciones  = EXCLUDED.costo_posiciones,
    n_posiciones      = EXCLUDED.n_posiciones,
    exposicion_pct    = EXCLUDED.exposicion_pct,
    pnl_realizado_dia = EXCLUDED.pnl_realizado_dia,
    pnl_no_realizado  = EXCLUDED.pnl_no_realizado,
    retorno_dia_pct   = EXCLUDED.retorno_dia_pct,
    retorno_acum_pct  = EXCLUDED.retorno_acum_pct,
    precios_stale     = EXCLUDED.precios_stale,
    calculado_en      = NOW()
"""


def persistir(engine, estrategia_id, filas):
    with engine.connect() as conn:
        for f in filas:
            conn.execute(text(UPSERT), {"eid": estrategia_id, **f})
        conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Reconstruye ft_equity_diaria (equity a mercado) desde "
                    "ft_operaciones + precios_diarios.")
    ap.add_argument("--estrategia", type=int, help="solo esta estrategia (id)")
    ap.add_argument("--desde", help="recalcular desde esta fecha (YYYY-MM-DD)")
    ap.add_argument("--rebuild", action="store_true",
                    help="recalcula la historia completa de cada estrategia")
    ap.add_argument("--check", action="store_true",
                    help="solo control de cuadre, no escribe")
    ap.add_argument("--dry-run", action="store_true", help="calcula y muestra, no escribe")
    args = ap.parse_args()

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")

    estrategias = cargar_estrategias(engine, args.estrategia)
    if estrategias.empty:
        log("[ERROR] No hay estrategias para procesar.")
        return 1

    ops_all = cargar_operaciones(engine, estrategias["id"].tolist())
    if ops_all.empty:
        log("[ERROR] No hay operaciones registradas.")
        return 1

    with engine.connect() as conn:
        ultimo_cierre = conn.execute(
            text("SELECT MAX(fecha) FROM precios_diarios")).scalar()
    hasta = pd.Timestamp(ultimo_cierre)
    log(f"Ultimo cierre disponible en precios_diarios: {hasta.date()}")

    inicio_global = ops_all["fecha_entrada"].min()
    tickers = sorted(ops_all["ticker"].unique())
    log(f"Cargando precios de {len(tickers)} tickers "
        f"({inicio_global.date()} -> {hasta.date()})...")
    pivot, mask_faltante = cargar_precios(engine, tickers, inicio_global.date(), hasta.date())
    log(f"Matriz de precios: {pivot.shape[0]} dias habiles x {pivot.shape[1]} tickers")

    desde_arg = pd.Timestamp(args.desde) if args.desde else None
    total_filas = 0
    problemas = []

    for _, est in estrategias.iterrows():
        eid = int(est["id"])
        ops = ops_all[ops_all["estrategia_id"] == eid]
        if ops.empty:
            log(f"  [{eid}] {est['nombre']}: sin operaciones, se omite.")
            continue

        inicio = ops["fecha_entrada"].min()
        if est["fecha_inicio"] is not None:
            inicio = min(inicio, pd.Timestamp(est["fecha_inicio"]))

        # Incremental: por defecto recalcula solo los ultimos dias (barato y
        # suficiente). --rebuild o --desde fuerzan una ventana mas amplia.
        if desde_arg is not None:
            inicio = max(inicio, desde_arg)
        elif not args.rebuild and not args.check:
            with engine.connect() as conn:
                ultima = conn.execute(text(
                    "SELECT MAX(fecha) FROM ft_equity_diaria WHERE estrategia_id = :eid"
                ), {"eid": eid}).scalar()
            if ultima is not None:
                # Se rehace el ultimo dia ya calculado por si cambio algo
                # (idempotente, el UPSERT lo pisa).
                inicio = max(inicio, pd.Timestamp(ultima))

        dias = [d for d in pivot.index if inicio <= d <= hasta]
        if not dias:
            log(f"  [{eid}] {est['nombre']}: al dia, nada que calcular.")
            continue

        filas = construir_serie(est, ops, dias, pivot, mask_faltante)

        # ── Control de cuadre ────────────────────────────────────────────
        # El cash es puro flujo realizado, asi que el reconstruido debe
        # coincidir exacto con ft_estrategias.cash_disponible, existan o no
        # posiciones abiertas. Es el invariante mas fuerte disponible.
        #
        # OJO: se calcula sobre TODAS las operaciones, sin corte de fecha, y
        # NO sobre el ultimo dia de la serie. La serie termina en el ultimo
        # cierre de precios_diarios, pero el sistema es asincronico: los bots
        # corren con el OHLCV del dia habil anterior, asi que suele haber
        # operaciones fechadas DESPUES del ultimo precio disponible. Comparar
        # contra el ultimo dia de la serie daria un falso descuadre.
        cerradas_all = ops[ops["fecha_salida"].notna()]
        cash_total = (float(est["capital_inicial"])
                      - float(ops["capital_entrada"].sum())
                      + float((cerradas_all["precio_salida"]
                               * cerradas_all["cantidad"]).sum()))
        cash_db = float(est["cash_disponible"])
        diff = abs(cash_total - cash_db)
        if diff > TOLERANCIA_CUADRE:
            cuadra = (f"  [!] CUADRE: cash calc {cash_total:,.2f} vs "
                      f"DB {cash_db:,.2f} (dif {diff:,.2f})")
            problemas.append((eid, est["nombre"], diff))
        else:
            cuadra = "  cuadre OK"

        ult = filas[-1]
        stale_total = sum(f["precios_stale"] for f in filas)
        log(f"  [{eid}] {est['nombre']:<28} {len(filas):>3} dias  "
            f"equity {ult['equity']:>11,.2f}  ret {ult['retorno_acum_pct']:>+7.2f}%  "
            f"pos {ult['n_posiciones']:>2}{cuadra}")
        if stale_total:
            log(f"       precios arrastrados (ffill): {stale_total} marcas")

        if not args.dry_run and not args.check:
            persistir(engine, eid, filas)
        total_filas += len(filas)

    modo = "calculadas (sin escribir)" if (args.dry_run or args.check) else "escritas"
    log(f"Listo: {total_filas} filas {modo}.")

    if problemas:
        log("[ATENCION] Estrategias que no cuadran:")
        for eid, nombre, diff in problemas:
            log(f"   {eid} {nombre}: diferencia {diff:,.2f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
