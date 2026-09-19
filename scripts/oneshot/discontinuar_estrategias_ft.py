"""
discontinuar_estrategias_ft.py
Baja de una estrategia de Forward Testing: liquida sus posiciones abiertas, la marca
inactiva y deja el corte fechado.

Se uso el 17/9/2026 para FT_COMBO_v1 y FT_SMC_v2 (Tarea 23). Sirve igual para la
proxima baja: es generico y idempotente.

QUE HACE
    1. Cierra cada posicion abierta al ULTIMO CLOSE disponible del ticker
       (precios_diarios), con motivo_salida = ESTRATEGIA_DISCONTINUADA. Usa
       ft_utils.cerrar_operacion, el unico lugar que escribe ft_operaciones: el PnL
       se realiza en el cash como en cualquier cierre y queda fecha_datos_salida.
    2. activa = FALSE en ft_estrategias -> cargar_estrategia() devuelve None y el bot
       no puede volver a operar aunque alguien lo corra suelto.
    3. Imprime el cierre en numeros (operaciones, PnL, aciertos, equity, periodo) para
       pegarlo en la ficha de la estrategia.

POR QUE LIQUIDAR Y NO DEJARLAS ABIERTAS
    Sin bot que las gestione, las posiciones seguirian marcandose a mercado en
    ft_equity_diaria para siempre: la curva se movería sin que nadie decida nada y
    la estrategia no tendria fecha de cierre. El cierre es una salida ARTIFICIAL y
    esta etiquetada como tal (ESTRATEGIA_DISCONTINUADA), asi que cualquier analisis
    puede excluirla o cortar la serie en esta fecha.

DESPUES DE CORRER
    - scripts/forward_testing/ft_compute_equity.py --rebuild  (la equity se recalcula
      desde ft_operaciones; este script no la toca)
    - ft_cambios.py add  (registro del cambio)
    - Sacar el bot de scripts/manual/ft_run_diario.bat

Uso:
    python scripts/oneshot/discontinuar_estrategias_ft.py --dry-run
    python scripts/oneshot/discontinuar_estrategias_ft.py
    python scripts/oneshot/discontinuar_estrategias_ft.py --estrategias FT_X_v1
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Entorno FT: forzar conexion a la DB LOCAL (ver scripts/forward_testing/ft_env.py)
from scripts.forward_testing.ft_env import configurar_entorno_local  # noqa: E402
configurar_entorno_local()

from sqlalchemy import text  # noqa: E402

from src.data.database import get_engine  # noqa: E402
from scripts.forward_testing.ft_utils import (  # noqa: E402
    log, obtener_precios_cierre_todos, cerrar_operacion, registrar_metricas_diarias,
)

MOTIVO = "ESTRATEGIA_DISCONTINUADA"
POR_DEFECTO = ["FT_COMBO_v1", "FT_SMC_v2"]


def resumen(nombre: str) -> dict:
    """Cierre en numeros de una estrategia (para la ficha de baja)."""
    engine = get_engine()
    with engine.connect() as conn:
        est = conn.execute(text("""
            SELECT id, nombre, activa, fecha_inicio, capital_inicial,
                   capital_actual, cash_disponible, capital_inmovilizado
            FROM ft_estrategias WHERE nombre = :n
        """), {"n": nombre}).fetchone()
        if not est:
            return {}

        ops = conn.execute(text("""
            SELECT COUNT(*)                                        AS total,
                   COUNT(*) FILTER (WHERE fecha_salida IS NULL)    AS abiertas,
                   COUNT(*) FILTER (WHERE fecha_salida IS NOT NULL) AS cerradas,
                   COALESCE(SUM(pnl), 0)                           AS pnl,
                   AVG(pnl_pct)                                    AS pct_medio,
                   COUNT(*) FILTER (WHERE pnl > 0)                 AS ganadoras,
                   MIN(fecha_datos)                                AS primer_dato,
                   MAX(COALESCE(fecha_datos_salida, fecha_datos))  AS ultimo_dato,
                   COUNT(DISTINCT ticker)                          AS tickers
            FROM ft_operaciones WHERE estrategia_id = :id
        """), {"id": est.id}).fetchone()

        eq = conn.execute(text("""
            SELECT fecha, equity, retorno_acum_pct, exposicion_pct
            FROM ft_equity_diaria WHERE estrategia_id = :id
            ORDER BY fecha DESC LIMIT 1
        """), {"id": est.id}).fetchone()

        dd = conn.execute(text("""
            SELECT MIN(equity) AS min_eq, MAX(equity) AS max_eq, COUNT(*) AS ruedas
            FROM ft_equity_diaria WHERE estrategia_id = :id
        """), {"id": est.id}).fetchone()

    d = {"id": est.id, "nombre": est.nombre, "activa": est.activa,
         "inicio": est.fecha_inicio, "capital_inicial": float(est.capital_inicial),
         "capital_actual": float(est.capital_actual),
         "ops_total": ops.total, "abiertas": ops.abiertas, "cerradas": ops.cerradas,
         "pnl": float(ops.pnl or 0),
         "pct_medio": float(ops.pct_medio) if ops.pct_medio is not None else None,
         "ganadoras": ops.ganadoras, "tickers": ops.tickers,
         "primer_dato": ops.primer_dato, "ultimo_dato": ops.ultimo_dato,
         "equity": float(eq.equity) if eq else None,
         "equity_fecha": eq.fecha if eq else None,
         "retorno_acum_pct": float(eq.retorno_acum_pct) if eq else None,
         "ruedas_equity": dd.ruedas if dd else 0,
         "equity_min": float(dd.min_eq) if dd and dd.min_eq else None,
         "equity_max": float(dd.max_eq) if dd and dd.max_eq else None}
    if d["cerradas"]:
        d["aciertos_pct"] = round(d["ganadoras"] * 100.0 / d["cerradas"], 1)
    return d


def imprimir_resumen(d: dict) -> None:
    if not d:
        return
    log(f"  {d['nombre']} (id={d['id']}) | activa={d['activa']}")
    log(f"    periodo de datos   : {d['primer_dato']} -> {d['ultimo_dato']} "
        f"({d['ruedas_equity']} ruedas de equity)")
    log(f"    operaciones        : {d['ops_total']} ({d['cerradas']} cerradas, "
        f"{d['abiertas']} abiertas) sobre {d['tickers']} tickers")
    log(f"    PnL realizado      : {d['pnl']:+,.2f} USD | medio por op "
        f"{d['pct_medio']:+.2f}% | aciertos {d.get('aciertos_pct')}%")
    log(f"    equity a mercado   : {d['equity']:,.2f} "
        f"({d['retorno_acum_pct']:+.2f}%) al {d['equity_fecha']}")
    log(f"    equity min / max   : {d['equity_min']:,.2f} / {d['equity_max']:,.2f}")


def posiciones_abiertas(estrategia_id: int) -> list:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT o.id, o.ticker, o.cantidad, o.precio_entrada, o.capital_entrada,
                   o.fecha_datos
            FROM ft_operaciones o
            WHERE o.estrategia_id = :id AND o.fecha_salida IS NULL
            ORDER BY o.fecha_datos, o.ticker
        """), {"id": estrategia_id}).fetchall()
    return [dict(r._mapping) for r in rows]


def discontinuar(nombre: str, dry_run: bool, hoy: date, precios: dict) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        est = conn.execute(text(
            "SELECT id, nombre, activa FROM ft_estrategias WHERE nombre = :n"
        ), {"n": nombre}).fetchone()

    if not est:
        log(f"[ERROR] '{nombre}' no existe en ft_estrategias.")
        return {}

    log("")
    log("=" * 60)
    log(f"BAJA de {nombre} (id={est.id})")
    log("=" * 60)
    log("ANTES:")
    imprimir_resumen(resumen(nombre))

    abiertas = posiciones_abiertas(est.id)
    log("")
    log(f"LIQUIDACION: {len(abiertas)} posiciones abiertas")
    pnl_liquidado = 0.0
    sin_precio = []

    for pos in abiertas:
        precio = precios.get(pos["ticker"])
        if not precio:
            sin_precio.append(pos["ticker"])
            log(f"  [WARN] {pos['ticker']}: sin precio de cierre, NO se liquida.")
            continue
        pnl = (float(precio) - float(pos["precio_entrada"])) * int(pos["cantidad"])
        pnl_liquidado += pnl
        log(f"  {pos['ticker']:<6} qty={pos['cantidad']:>5} "
            f"entrada={float(pos['precio_entrada']):>9.2f} "
            f"salida={float(precio):>9.2f} pnl={pnl:>+10.2f}")
        if not dry_run:
            cerrar_operacion(pos["id"], est.id, float(precio), MOTIVO, hoy)

    log(f"  PnL de la liquidacion: {pnl_liquidado:+,.2f} USD")
    if sin_precio:
        log(f"  [WARN] quedan abiertas sin precio: {sin_precio}")

    if not dry_run:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE ft_estrategias SET activa = FALSE WHERE id = :id"
            ), {"id": est.id})
            conn.commit()
        log(f"  activa = FALSE ({nombre} ya no puede operar)")
        registrar_metricas_diarias(est.id, hoy)
        log("")
        log("DESPUES:")
        d = resumen(nombre)
        imprimir_resumen(d)
        return d

    log("  [DRY RUN] no se escribio nada.")
    return resumen(nombre)


def main() -> int:
    ap = argparse.ArgumentParser(description="Baja de estrategias de Forward Testing")
    ap.add_argument("--estrategias", default=",".join(POR_DEFECTO),
                    help="nombres separados por coma")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nombres = [n.strip() for n in args.estrategias.split(",") if n.strip()]
    hoy = date.today()
    precios = obtener_precios_cierre_todos()
    log(f"Precios de cierre cargados: {len(precios)} tickers")

    for nombre in nombres:
        discontinuar(nombre, args.dry_run, hoy, precios)

    log("")
    if args.dry_run:
        log("DRY RUN terminado. Sin --dry-run liquida y desactiva.")
    else:
        log("Pendiente: ft_compute_equity.py --rebuild, ft_cambios.py add y "
            "sacar los bots de ft_run_diario.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
