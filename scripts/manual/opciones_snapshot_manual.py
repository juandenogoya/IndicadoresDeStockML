"""
opciones_snapshot_manual.py
Ejecucion manual del snapshot de opciones con confirmacion interactiva.

Uso:
    python scripts/manual/opciones_snapshot_manual.py
    python scripts/manual/opciones_snapshot_manual.py --fecha 2026-04-25

Diferencias vs cron automatico:
    - Muestra estado actual de la DB antes de ejecutar
    - Pide confirmacion antes de correr
    - Permite ingresar fecha manualmente (backfill)
    - No envia mensajes a Telegram
"""

import sys
import os
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from dotenv import load_dotenv
    for f in [".env", ".env.local"]:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            load_dotenv(p, override=True)
except ImportError:
    pass


# ── Helpers ──────────────────────────────────────────────────────────────────

SEP  = "=" * 62
SEP2 = "-" * 62

def titulo(texto):
    print()
    print(SEP)
    print(f"  {texto}")
    print(SEP)

def seccion(texto):
    print()
    print(f"  {texto}")
    print(SEP2)


def _sugerir_fecha() -> date:
    """
    Detecta la fecha objetivo segun la hora UTC actual.
    - 20:00-23:59 UTC: mercado ya cerro hoy -> fecha = hoy
    - 00:00-19:59 UTC: todavia no cerro -> fecha = prev_trading_day(hoy)
    """
    from src.utils.trading_calendar import is_trading_day, prev_trading_day
    ahora_utc = datetime.now(timezone.utc)
    hoy       = ahora_utc.date()
    hora_utc  = ahora_utc.hour

    if hora_utc >= 20 and is_trading_day(hoy):
        return hoy
    else:
        return prev_trading_day(hoy)


def _estado_db() -> dict:
    """Consulta el estado actual de las 4 tablas relevantes."""
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    estado = {}

    with engine.connect() as conn:
        # precios_diarios
        r = conn.execute(text("""
            SELECT MAX(fecha), COUNT(DISTINCT ticker)
            FROM precios_diarios
        """)).fetchone()
        estado["precios_ultima_fecha"]  = r[0]
        estado["precios_tickers"]       = r[1] or 0

        # alertas_scanner
        r = conn.execute(text("""
            SELECT MAX(scan_fecha)::date, MAX(precio_fecha), COUNT(DISTINCT ticker)
            FROM alertas_scanner
            WHERE scan_fecha = (SELECT MAX(scan_fecha) FROM alertas_scanner)
        """)).fetchone()
        estado["scanner_ultima_scan"]   = r[0]
        estado["scanner_ultima_precio"] = r[1]
        estado["scanner_tickers"]       = r[2] or 0

        # opciones_snapshot
        r = conn.execute(text("""
            SELECT MAX(fecha_snapshot), COUNT(DISTINCT ticker), COUNT(*)
            FROM opciones_snapshot
            WHERE fecha_snapshot = (SELECT MAX(fecha_snapshot) FROM opciones_snapshot)
        """)).fetchone()
        estado["opciones_ultima_fecha"]   = r[0]
        estado["opciones_tickers"]        = r[1] or 0
        estado["opciones_contratos"]      = r[2] or 0

        # opciones_resumen_diario
        r = conn.execute(text("""
            SELECT MAX(fecha), COUNT(DISTINCT ticker)
            FROM opciones_resumen_diario
            WHERE fecha = (SELECT MAX(fecha) FROM opciones_resumen_diario)
        """)).fetchone()
        estado["resumen_ultima_fecha"] = r[0]
        estado["resumen_tickers"]      = r[1] or 0

    return estado


def _verificar_carga(fecha: date) -> dict:
    """Verifica cuantos datos hay para la fecha dada post-carga."""
    from src.data.database import get_engine
    from sqlalchemy import text
    engine = get_engine()

    with engine.connect() as conn:
        snap = conn.execute(text("""
            SELECT COUNT(DISTINCT ticker), COUNT(*)
            FROM opciones_snapshot
            WHERE fecha_snapshot = :f
        """), {"f": fecha}).fetchone()

        res = conn.execute(text("""
            SELECT COUNT(DISTINCT ticker)
            FROM opciones_resumen_diario
            WHERE fecha = :f
        """), {"f": fecha}).fetchone()

    return {
        "snap_tickers":   snap[0] or 0,
        "snap_contratos": snap[1] or 0,
        "res_tickers":    res[0]  or 0,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Snapshot manual de opciones")
    parser.add_argument("--fecha", default=None,
                        help="Fecha objetivo YYYY-MM-DD (default: auto-detectada)")
    args = parser.parse_args()

    titulo("OPCIONES SNAPSHOT -- EJECUCION MANUAL")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')} hora local")

    # ── 1. Estado DB ─────────────────────────────────────────────────────────
    seccion("Estado actual de la DB")
    try:
        estado = _estado_db()
        print(f"  precios_diarios   : ultimo EOD  = {estado['precios_ultima_fecha']}"
              f"  ({estado['precios_tickers']} tickers)")
        print(f"  alertas_scanner   : ultimo scan = {estado['scanner_ultima_scan']}"
              f"  (precio_fecha={estado['scanner_ultima_precio']}"
              f", {estado['scanner_tickers']} tickers)")
        print(f"  opciones_snapshot : ultima fecha= {estado['opciones_ultima_fecha']}"
              f"  ({estado['opciones_tickers']} tickers"
              f", {estado['opciones_contratos']:,} contratos)")
        print(f"  opciones_resumen  : ultima fecha= {estado['resumen_ultima_fecha']}"
              f"  ({estado['resumen_tickers']} tickers)")
    except Exception as e:
        print(f"  ERROR al consultar DB: {e}")
        sys.exit(1)

    # ── 2. Fecha objetivo ─────────────────────────────────────────────────────
    seccion("Fecha objetivo")
    if args.fecha:
        try:
            fecha_objetivo = date.fromisoformat(args.fecha)
            print(f"  Fecha indicada por argumento: {fecha_objetivo}")
        except ValueError:
            print(f"  ERROR: fecha invalida '{args.fecha}'. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        sugerida = _sugerir_fecha()
        print(f"  Fecha sugerida (auto): {sugerida}")
        print()
        override = input(f"  Ingrese fecha YYYY-MM-DD o Enter para usar {sugerida}: ").strip()
        if override:
            try:
                fecha_objetivo = date.fromisoformat(override)
                print(f"  Usando fecha: {fecha_objetivo}")
            except ValueError:
                print(f"  Fecha invalida '{override}'. Usando fecha sugerida: {sugerida}")
                fecha_objetivo = sugerida
        else:
            fecha_objetivo = sugerida
            print(f"  Usando fecha: {fecha_objetivo}")

    # ── 3. Check datos existentes ─────────────────────────────────────────────
    print()
    verificacion_previa = _verificar_carga(fecha_objetivo)
    tiene_datos = verificacion_previa["snap_contratos"] >= 10_000

    if tiene_datos:
        print(f"  AVISO: ya existen datos para {fecha_objetivo}:")
        print(f"    opciones_snapshot : {verificacion_previa['snap_contratos']:,} contratos,"
              f" {verificacion_previa['snap_tickers']} tickers")
        print(f"    opciones_resumen  : {verificacion_previa['res_tickers']} tickers")
        print()
        resp = input("  Desea sobreescribir los datos existentes? (S/N): ").strip().upper()
        if resp != "S":
            print()
            print("  Operacion cancelada.")
            print(SEP)
            sys.exit(0)
    else:
        print(f"  Sin datos para {fecha_objetivo} en DB.")

    # ── 4. Confirmacion ───────────────────────────────────────────────────────
    print()
    resp = input(f"  Ejecutar snapshot de opciones para {fecha_objetivo}? (S/N): ").strip().upper()
    if resp != "S":
        print()
        print("  Operacion cancelada.")
        print(SEP)
        sys.exit(0)

    # ── 5. Ejecucion ──────────────────────────────────────────────────────────
    seccion("Ejecutando extraccion yfinance")
    print(f"  Fecha: {fecha_objetivo}  |  Sin Telegram (modo manual)")
    print()

    # Importar y correr directamente (sin Telegram)
    try:
        import importlib.util, types

        # Cargar el modulo sin ejecutar main()
        spec = importlib.util.spec_from_file_location(
            "opciones_snapshot",
            os.path.join(ROOT, "scripts", "33_opciones_snapshot.py")
        )
        mod = importlib.util.load_from_spec(spec) if hasattr(importlib.util, 'load_from_spec') else None

        # Usar subprocess para mantener el output en tiempo real
        import subprocess
        cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "33_opciones_snapshot.py"),
            "--fecha", str(fecha_objetivo),
            "--intento", "1",
        ]
        # Parchear el notifier temporalmente via env para suprimir Telegram
        env = os.environ.copy()
        env["OPCIONES_SKIP_TELEGRAM"] = "1"

        result = subprocess.run(cmd, env=env, cwd=ROOT)
        exit_code = result.returncode

    except Exception as e:
        print(f"  ERROR al ejecutar snapshot: {e}")
        sys.exit(1)

    # ── 6. Verificacion post-carga ────────────────────────────────────────────
    seccion("Verificacion post-carga")
    try:
        verificacion = _verificar_carga(fecha_objetivo)
        snap_ok = verificacion["snap_contratos"] >= 10_000
        res_ok  = verificacion["res_tickers"] >= 100

        print(f"  opciones_snapshot  {fecha_objetivo}:")
        print(f"    Contratos : {verificacion['snap_contratos']:,}")
        print(f"    Tickers   : {verificacion['snap_tickers']}")
        print(f"    Estado    : {'OK' if snap_ok else 'INCOMPLETO -- menos de 10.000 contratos'}")

        print()
        print(f"  opciones_resumen_diario  {fecha_objetivo}:")
        print(f"    Tickers   : {verificacion['res_tickers']}")
        print(f"    Estado    : {'OK' if res_ok else 'INCOMPLETO -- menos de 100 tickers'}")

        print()
        if snap_ok and res_ok:
            print("  RESULTADO FINAL: OK -- datos cargados correctamente.")
        elif exit_code != 0:
            print("  RESULTADO FINAL: FALLO -- revisar errores arriba.")
        else:
            print("  RESULTADO FINAL: PARCIAL -- verificar manualmente.")

    except Exception as e:
        print(f"  ERROR al verificar DB: {e}")

    print(SEP)
    print()


if __name__ == "__main__":
    main()
