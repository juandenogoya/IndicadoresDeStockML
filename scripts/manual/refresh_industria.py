"""
refresh_industria.py
Rellena / actualiza activos.industry desde yahooquery assetProfile.

Por que existe: la columna industry venia poblada desde yfinance (poco fiable ->
62% NULL). yahooquery assetProfile la trae completa (199/200; FISV es un ticker
delistado sin perfil) y es la MISMA fuente ya verificada correcta para el sector.

Que hace:
  - Baja sector + industry de yahooquery para todo el universo activo.
  - AVISA si algun SECTOR difiere del de la DB (posible reclasificacion) pero
    NO lo cambia: cambiar un sector regrupa la historia point-in-time y es una
    decision manual (ver docs/gestion_universo.md).
  - ACTUALIZA activos.industry donde yahooquery trae dato y difiere del actual.
  - DUAL-WRITE local + Railway (activos vive sincronizada en ambas).

No destructivo: solo toca la columna industry. No reescribe sector ni historia.

Uso:
    python scripts/manual/refresh_industria.py --status    # cobertura actual
    python scripts/manual/refresh_industria.py --dry-run   # muestra sin escribir
    python scripts/manual/refresh_industria.py             # escribe local + Railway
    python scripts/manual/refresh_industria.py --no-railway # solo local

Respeta el lock de yfinance (rate limit por IP, regla #9). No baja OHLCV -> no
aplica la regla #10 (pre-mercado).
"""

import sys
import os
import argparse
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sqlalchemy import text

from src.utils.yfinance_lock import acquire as acquire_yf_lock
from scripts.push_senales_bot import _engine_local, _engine_railway

CHUNK = 25
PAUSA_SEG = 0.5


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_profiles(tickers):
    """ticker -> (sector, industry) via yahooquery assetProfile (chunks + pausa)."""
    from yahooquery import Ticker
    out = {}
    for ch in _chunks(tickers, CHUNK):
        prof = Ticker(ch, asynchronous=True, max_workers=6).asset_profile
        for tk in ch:
            d = prof.get(tk)
            if isinstance(d, dict):
                out[tk] = (d.get("sector"), d.get("industry"))
            else:
                out[tk] = (None, None)
        time.sleep(PAUSA_SEG)
    return out


def cargar_activos(eng):
    """ticker -> (sector, industry) del universo activo."""
    with eng.connect() as c:
        rows = c.execute(text(
            "SELECT ticker, sector, industry FROM activos "
            "WHERE activo = TRUE ORDER BY ticker")).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def cmd_status(eng):
    acts = cargar_activos(eng)
    n = len(acts)
    sin = [tk for tk, (s, i) in acts.items() if not i or not str(i).strip()]
    log(f"Universo activo: {n}")
    log(f"Con industria:   {n - len(sin)}")
    log(f"Sin industria:   {len(sin)}")
    if sin:
        log(f"  -> {', '.join(sorted(sin))}")


def _aplicar(eng, updates, destino):
    """UPDATE activos.industry para cada (ticker, industria)."""
    with eng.begin() as c:
        for tk, ind in updates:
            c.execute(text("UPDATE activos SET industry = :i WHERE ticker = :t"),
                      {"i": ind, "t": tk})
    log(f"   {len(updates)} filas actualizadas en {destino}.")


def main():
    ap = argparse.ArgumentParser(description="Refresh de activos.industry desde yahooquery")
    ap.add_argument("--status", action="store_true", help="muestra cobertura y sale")
    ap.add_argument("--dry-run", action="store_true", help="muestra cambios sin escribir")
    ap.add_argument("--no-railway", action="store_true", help="escribe solo local")
    args = ap.parse_args()

    eng_l = _engine_local()

    if args.status:
        cmd_status(eng_l)
        return

    acts = cargar_activos(eng_l)
    tickers = sorted(acts)

    acquire_yf_lock(f"refresh_industria (n={len(tickers)})")
    log(f"Bajando assetProfile de {len(tickers)} tickers via yahooquery...")
    yq = fetch_profiles(tickers)

    updates = []          # (ticker, industria_nueva)
    sector_warn = []      # (ticker, db_sector, yq_sector)
    sin_dato = []         # yahooquery no trae industria
    for tk in tickers:
        db_sec, db_ind = acts[tk]
        yq_sec, yq_ind = yq.get(tk, (None, None))

        if yq_sec and db_sec and yq_sec.strip() != db_sec.strip():
            sector_warn.append((tk, db_sec, yq_sec))

        if yq_ind and str(yq_ind).strip():
            nueva = str(yq_ind).strip()
            if (db_ind or "").strip() != nueva:
                updates.append((tk, nueva))
        else:
            sin_dato.append(tk)

    if sector_warn:
        log(f"[AVISO] {len(sector_warn)} sector(es) difieren de yahooquery (NO se tocan):")
        for tk, dbs, yqs in sector_warn:
            log(f"   {tk}: DB='{dbs}' yahooquery='{yqs}'  -> revisar manual")
    else:
        log("Sectores: coinciden con yahooquery (0 discrepancias).")

    if sin_dato:
        log(f"Sin industria en yahooquery ({len(sin_dato)}): {', '.join(sin_dato)} (se dejan como estan)")

    log(f"Industrias a actualizar: {len(updates)}")
    for tk, ind in updates:
        log(f"   {tk:6} -> {ind}")

    if args.dry_run:
        log("[DRY RUN] no se escribe nada.")
        return
    if not updates:
        log("Nada para actualizar.")
        return

    log("Escribiendo activos.industry...")
    _aplicar(eng_l, updates, "LOCAL")
    if not args.no_railway:
        try:
            _aplicar(_engine_railway(), updates, "RAILWAY")
        except Exception as e:
            log(f"[WARN] no se pudo escribir en Railway ({e}). Re-correr para sincronizar.")
    log("Listo.")


if __name__ == "__main__":
    main()
