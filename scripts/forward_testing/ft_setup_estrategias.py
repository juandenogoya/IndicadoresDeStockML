"""
ft_setup_estrategias.py
Inserta las instancias de estrategia iniciales en ft_estrategias.

Uso:
    python scripts/forward_testing/ft_setup_estrategias.py           # inserta
    python scripts/forward_testing/ft_setup_estrategias.py --status  # muestra estado
"""

import sys
import os
import argparse
import json
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from sqlalchemy import text
from src.data.database import get_engine
from scripts.forward_testing.ft_utils import log

CAPITAL_INICIAL = 100_000.00

SECTORES_ACTIVOS = [
    "Technology", "Consumer Cyclical", "Financial Services",
    "Industrials", "Basic Materials", "Healthcare",
    "Energy", "Communication Services", "Consumer Defensive",
]

ESTRATEGIAS = [
    {
        "nombre":      "FT_TECH_SECTOR_v1",
        "descripcion": "AT Tecnico sectorial. 9 sectores x $11.111. "
                       "Max 5 pos/sector al 20% del presupuesto. "
                       "Scoring SMA/MACD/RSI. SL/TP ATR-based.",
        "logica":      "tecnico_sectorial",
        "parametros":  {
            "sectores":           SECTORES_ACTIVOS,
            "n_sectores":         len(SECTORES_ACTIVOS),
            "score_entrada":      4.0,
            "score_salida":       3.5,
            "atr_mult_sl":        2.0,
            "atr_mult_tp":        4.0,
            "max_pos_sector":     5,
            "position_pct":       0.20,
            "max_deploy_pct":     1.0,   # 100% — sector partition ya diversifica
        },
    },
    {
        "nombre":      "FT_ML_SCANNER_v1",
        "descripcion": "Replica Bot1 Alpaca. Entrada: COMPRA_FUERTE score>=65. "
                       "Salida: score degradado, SL 5%, TP 10%, earnings.",
        "logica":      "ml_scanner",
        "parametros":  {
            "nivel_min":        "COMPRA_FUERTE",
            "score_min":        65,
            "max_posiciones":   5,
            "riesgo_por_trade": 0.15,
            "sl_pct":           0.05,
            "tp_pct":           0.10,
        },
    },
    {
        "nombre":      "FT_TECH_v1",
        "descripcion": "Replica Bot2 Alpaca. Scoring SMA/MACD/RSI 3 capas (max 5.5pts). "
                       "Entrada >= 4.0, salida <= 3.5. SL/TP ATR-based (2x/4x).",
        "logica":      "tecnico",
        "parametros":  {
            "score_entrada":    4.0,
            "score_salida":     3.5,
            "score_maximo":     5.5,
            "rsi_min":          45.0,
            "rsi_max":          68.0,
            "atr_mult_sl":      2.0,
            "atr_mult_tp":      4.0,
            "max_posiciones":   5,
            "riesgo_por_trade": 0.15,
        },
    },
    {
        "nombre":      "FT_COMBO_v1",
        "descripcion": "AT Tecnico sectorial + scoring de velas 5d como desempate. "
                       "9 sectores x $11.111. Ranking: tech_score DESC, candle_score_5d DESC. "
                       "Excluye si candle_score < -3 (distribucion activa). SL/TP ATR-based.",
        "logica":      "combo_tech_candle",
        "parametros":  {
            "sectores":         SECTORES_ACTIVOS,
            "n_sectores":       len(SECTORES_ACTIVOS),
            "score_entrada":    4.0,
            "score_salida":     3.5,
            "candle_score_min": -3.0,
            "candle_ventana_d": 5,
            "atr_mult_sl":      2.0,
            "atr_mult_tp":      4.0,
            "max_pos_sector":   5,
            "position_pct":     0.20,
        },
    },
    {
        "nombre":      "FT_SMC_v1",
        "descripcion": "Replica Bot3 Alpaca. CHoCH/BOS + estructura + vela alcista. "
                       "Trailing SL estructural. Sin TP. Time stop 20d.",
        "logica":      "smc_estructura",
        "parametros":  {
            "score_entrada_min":  1,
            "score_maximo":       3,
            "lookback_dias":      12,
            "min_sl_dist_pct":    1.0,
            "max_sl_dist_pct":    8.0,
            "dias_max_pos":       20,
            "max_posiciones":     5,
            "riesgo_por_trade":   0.15,
        },
    },
    {
        "nombre":      "FT_TECH_SECTOR_v2",
        "descripcion": "TECH_SECTOR_v1 con retencion condicional y rotacion intrasectorial. "
                       "Cierre solo si score=0 AND candle<0 AND up_vol<2. "
                       "Rotacion si delta_score >= 1.0 en sector lleno.",
        "logica":      "tecnico_sectorial_v2",
        "parametros":  {
            "sectores":                 SECTORES_ACTIVOS,
            "n_sectores":               len(SECTORES_ACTIVOS),
            "score_entrada":            4.0,
            "atr_mult_sl":              2.0,
            "atr_mult_tp":              4.0,
            "max_pos_sector":           5,
            "position_pct":             0.20,
            "retencion_candle_min":     0,
            "retencion_up_vol_min":     2,
            "retencion_logica":         "OR",
            "rotacion_delta_score":     1.0,
        },
    },
    {
        "nombre":      "FT_SMC_v2",
        "descripcion": "SMC_v1 con filtro de contexto en entrada (OR: lateral>1 OR candle>0) "
                       "y salida por agotamiento (AND: up_vol=0 AND candle<-2 AND lateral<0.5). "
                       "Time stop eliminado.",
        "logica":      "smc_estructura_v2",
        "parametros":  {
            "score_entrada_min":            1,
            "score_maximo":                 3,
            "lookback_dias":                12,
            "min_sl_dist_pct":              1.0,
            "max_sl_dist_pct":              8.0,
            "max_posiciones":               5,
            "riesgo_por_trade":             0.15,
            "filtro_lateral_ratio_min":     1.0,
            "filtro_candle_score_min":      0,
            "filtro_logica":                "OR",
            "agotamiento_up_vol_max":       0,
            "agotamiento_candle_max":       -2,
            "agotamiento_lateral_max":      0.5,
            "agotamiento_logica":           "AND",
        },
    },
]


def cmd_status():
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT e.id, e.nombre, e.logica, e.activa,
                   e.capital_actual, e.cash_disponible, e.capital_inmovilizado,
                   e.fecha_inicio,
                   COUNT(o.id) FILTER (WHERE o.fecha_salida IS NULL) AS pos_abiertas,
                   COUNT(o.id) AS ops_total
            FROM ft_estrategias e
            LEFT JOIN ft_operaciones o ON o.estrategia_id = e.id
            GROUP BY e.id
            ORDER BY e.id
        """)).fetchall()

    print()
    print("  === ft_estrategias ===")
    print(f"  {'ID':<4} {'NOMBRE':<22} {'LOGICA':<15} "
          f"{'CAPITAL':>12} {'CASH':>12} {'INMOV':>10} "
          f"{'ABIERTAS':>8} {'OPS':>5} {'ACTIVA'}")
    print("  " + "-" * 110)
    for r in rows:
        print(f"  {r.id:<4} {r.nombre:<22} {r.logica:<15} "
              f"{float(r.capital_actual):>12,.2f} "
              f"{float(r.cash_disponible):>12,.2f} "
              f"{float(r.capital_inmovilizado):>10,.2f} "
              f"{r.pos_abiertas:>8} {r.ops_total:>5} {'SI' if r.activa else 'NO'}")
    print()


def cmd_insert():
    engine   = get_engine()
    hoy      = date.today()
    insertados = 0
    omitidos   = 0

    with engine.connect() as conn:
        for est in ESTRATEGIAS:
            existe = conn.execute(text(
                "SELECT COUNT(*) FROM ft_estrategias WHERE nombre = :nombre"
            ), {"nombre": est["nombre"]}).scalar()

            if existe:
                log(f"  [SKIP] '{est['nombre']}' ya existe.")
                omitidos += 1
                continue

            conn.execute(text("""
                INSERT INTO ft_estrategias (
                    nombre, descripcion, logica, parametros,
                    capital_inicial, capital_actual, cash_disponible,
                    capital_inmovilizado, activa, fecha_inicio
                ) VALUES (
                    :nombre, :descripcion, :logica, :parametros,
                    :capital, :capital, :capital,
                    0, TRUE, :hoy
                )
            """), {
                "nombre":      est["nombre"],
                "descripcion": est["descripcion"],
                "logica":      est["logica"],
                "parametros":  json.dumps(est["parametros"]),
                "capital":     CAPITAL_INICIAL,
                "hoy":         hoy,
            })
            log(f"  [OK]  '{est['nombre']}' insertada.")
            insertados += 1

        conn.commit()

    log(f"Insertadas: {insertados} | Omitidas (ya existian): {omitidos}")
    print()
    cmd_status()


def main():
    parser = argparse.ArgumentParser(description="Setup estrategias FT")
    parser.add_argument("--status", action="store_true",
                        help="Muestra estado actual de ft_estrategias")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    else:
        cmd_insert()


if __name__ == "__main__":
    main()
