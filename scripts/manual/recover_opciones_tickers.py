"""
recover_opciones_tickers.py
Recovery quirurgico para tickers especificos del snapshot de opciones US
cuando fallaron por connection drop a Railway (no por rate limit yfinance).

Usa las funciones internas de scripts/33_opciones_snapshot.py para procesar
solo la lista que pases, sin el check upfront que ve "ya hay 88k filas en
DB" y haria skip.

Uso:
    python scripts/manual/recover_opciones_tickers.py --fecha 2026-05-13 --tickers DAL,XP,STNE

NOTA: escribe a Railway (carga .env.local con DATABASE_URL).
"""
import os
import sys
import argparse
import importlib.util
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
load_dotenv(os.path.join(ROOT, ".env.local"), override=True)


def load_snapshot_module():
    """Carga el modulo del snapshot sin invocar su main()."""
    path = os.path.join(ROOT, "scripts", "33_opciones_snapshot.py")
    spec = importlib.util.spec_from_file_location("snapshot_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fecha", required=True, help="YYYY-MM-DD del snapshot a completar")
    parser.add_argument("--tickers", required=True,
                        help="Lista separada por comas, ej: DAL,XP,STNE")
    args = parser.parse_args()

    fecha = date.fromisoformat(args.fecha)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    log("=" * 60)
    log(f"  RECOVERY QUIRURGICO OPCIONES")
    log(f"  Fecha   : {fecha}")
    log(f"  Tickers : {tickers}")
    log("=" * 60)

    snap = load_snapshot_module()

    log("Cargando precios subyacentes y HV_20d...")
    precios = snap._get_precios_subyacentes(tickers)
    hvs     = snap._get_hv_20d(tickers)
    log(f"  Precios cargados: {len(precios)}/{len(tickers)}")
    log(f"  HV_20d cargados : {len(hvs)}/{len(tickers)}")

    total_filas = 0
    sin_opc     = 0
    errores     = 0
    resumenes   = []

    for i, t in enumerate(tickers, 1):
        p = precios.get(t)
        h = hvs.get(t)
        try:
            filas = snap.recolectar_ticker(t, fecha, p, h)
        except Exception as e:
            log(f"  [{i}/{len(tickers)}] {t:<8s}  ERROR yfinance: {str(e)[:80]}")
            errores += 1
            continue

        if not filas:
            log(f"  [{i}/{len(tickers)}] {t:<8s}  sin opciones activas")
            sin_opc += 1
            continue

        try:
            n = snap.persistir_filas(filas)
            r = snap._computar_resumen(t, fecha, filas, p)
            resumenes.append(r)
            pcr = f"{r['pcr_vol']:.2f}" if r.get("pcr_vol") else "N/A"
            log(f"  [{i}/{len(tickers)}] {t:<8s}  {n:5d} filas  PCR_vol={pcr}")
            total_filas += n
        except Exception as e:
            log(f"  [{i}/{len(tickers)}] {t:<8s}  ERROR persistir: {str(e)[:80]}")
            errores += 1

    log("")
    if resumenes:
        try:
            n_res = snap.persistir_resumenes(resumenes)
            log(f"  Resumenes diarios persistidos: {n_res}")
        except Exception as e:
            log(f"  [WARN] No se pudo persistir resumenes: {str(e)[:80]}")

        # Recalcular Z-scores de opciones para esta fecha (idempotente)
        try:
            from src.utils.zscore_pipeline import calcular_zscore_opciones, init_tablas
            from src.data.database import get_engine
            engine = get_engine()
            init_tablas(engine)
            n_z = calcular_zscore_opciones(fecha, engine)
            log(f"  Z-scores opciones recalculados: {n_z} tickers")
        except Exception as e:
            log(f"  [WARN] Z-scores no recalculados: {str(e)[:80]}")

    log("")
    log(f"  Filas insertadas : {total_filas:,}")
    log(f"  Sin opciones     : {sin_opc}")
    log(f"  Errores          : {errores}")
    log("  Completado.")


if __name__ == "__main__":
    main()
