"""
snapshot_intradia.py -- captura de barras de 30 min en los cortes intradiarios.

Corre en la VM de Oracle (cron de L-V, varios horarios UTC) y escribe en Railway
la tabla precios_intradia_30m. Cada disparo pregunta que corte vencio
(src/utils/sesion_nyse.cortes_vencidos); si no hay ninguno, o ya esta
capturado, sale sin hacer nada. Asi un unico horario de cron cubre horario de
verano, invierno y medias ruedas, y el disparo de los :20 es el reintento del de
los :05. Diseno y decisiones: docs/intradia.md.

    apertura_90 = apertura + 90 min (11:00 ET)
    cierre_60   = cierre   - 60 min (15:00 ET; 12:00 ET en media rueda)

Etapa 1 (en prueba en la rama main-y9tr06): SOLO captura. Telegram avisa
unicamente fallas de la captura, no senales de mercado.

Orden de una captura (mismo principio que 33_opciones_snapshot):
    1. reinyecta spools pendientes (DB caida en un disparo anterior)
    2. lock de Yahoo, descarga, normaliza (descarta la barra en curso)
    3. spool en disco ANTES de la DB
    4. upsert; si la DB confirmo, borra el spool; si no, avisa: dato a salvo
    5. cobertura < 90% del universo -> aviso

A diferencia de opciones, un corte perdido NO es irrecuperable: Yahoo guarda
barras de 30 min unos 60 dias. Reconstruir: --fecha AAAA-MM-DD --corte X.

Uso:
    python scripts/intradia/snapshot_intradia.py                 # lo que dispara el cron
    python scripts/intradia/snapshot_intradia.py --dry-run --fecha 2026-11-02 --corte apertura_90
    python scripts/intradia/snapshot_intradia.py --fecha 2026-11-02 --corte cierre_60 [--forzar]
    python scripts/intradia/snapshot_intradia.py --replay        # reinyecta spools pendientes
    python scripts/intradia/snapshot_intradia.py --status        # ultimas capturas + spools

Salida: 0 OK (o nada que hacer) / 2 aviso (cobertura baja, dato en spool) / 1 error.
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # el spool y el lock usan rutas relativas al repo

try:
    from dotenv import load_dotenv
    if os.path.exists(os.path.join(ROOT, ".env")):
        load_dotenv(os.path.join(ROOT, ".env"))
    if os.path.exists(os.path.join(ROOT, ".env.local")):
        load_dotenv(os.path.join(ROOT, ".env.local"), override=True)
except ImportError:
    pass

from sqlalchemy import text

from src.utils import intradia, sesion_nyse

COBERTURA_MIN = 0.90
_skip_telegram = os.getenv("INTRADIA_SKIP_TELEGRAM", "0") == "1"


def _peor(a: int, b: int) -> int:
    """Combina codigos de salida: el error (1) manda sobre el aviso (2)."""
    return 1 if 1 in (a, b) else max(a, b)


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)


def _telegram(texto: str):
    if _skip_telegram:
        return
    try:
        from src.pipeline.telegram_notifier import _send
        _send(texto)
    except Exception as e:
        log(f"  [WARN] Telegram no disponible: {e}")


# ── DB ────────────────────────────────────────────────────────────────────────

def persistir(filas: list[dict]) -> int:
    import psycopg2.extras
    from src.data.database import get_connection
    if not filas:
        return 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, intradia.SQL_UPSERT, intradia.valores_db(filas), page_size=500)
    return len(filas)


def tickers_capturados(fecha: date, corte: str, hora_corte: datetime) -> int:
    """Tickers que ya tienen en la DB la barra que termina en el corte."""
    from src.data.database import get_engine
    ultimo_inicio = hora_corte - timedelta(minutes=sesion_nyse.BARRA_MIN)
    with get_engine().connect() as c:
        return c.execute(text(
            f"SELECT COUNT(DISTINCT ticker) FROM {intradia.TABLA} "
            "WHERE fecha = :f AND corte = :c AND inicio_barra = :u"
        ), {"f": fecha, "c": corte, "u": ultimo_inicio}).scalar() or 0


def replay_spools() -> tuple[int, int]:
    """Reinyecta los spools pendientes. Devuelve (reinyectados, que_siguen)."""
    pendientes = intradia.listar_spools()
    ok = 0
    for path in pendientes:
        try:
            filas = intradia.leer_spool(path)
            persistir(filas)
            os.remove(path)
            ok += 1
            log(f"  spool reinyectado: {os.path.basename(path)} ({len(filas)} filas)")
        except Exception as e:
            log(f"  [WARN] spool sigue pendiente {os.path.basename(path)}: {e}")
    return ok, len(pendientes) - ok


# ── Descarga ──────────────────────────────────────────────────────────────────

def descargar(tickers: list[str], fecha: date):
    """
    Barras de 30 min de la rueda `fecha` via yahooquery. Pide start/end y, si
    vuelve vacio, cae a period (Yahoo a veces no respeta start/end en intradiario).
    El recorte a la sesion y al corte lo hace intradia.normalizar_barras.
    """
    from yahooquery import Ticker
    from src.utils.yahooquery_loader import RateLimitDetected, _is_ratelimit_msg

    def _pedir(**kw):
        df = Ticker(tickers, asynchronous=True).history(interval="30m", **kw)
        if isinstance(df, dict):
            msgs = " | ".join(str(v) for v in df.values())
            if _is_ratelimit_msg(msgs):
                raise RateLimitDetected(msgs[:200])
            return None
        return df

    df = _pedir(start=fecha.isoformat(), end=(fecha + timedelta(days=1)).isoformat())
    if df is None or df.empty:
        dias = (datetime.now(timezone.utc).date() - fecha).days
        df = _pedir(period="5d" if dias <= 4 else "60d")
    return df


# ── Captura de un corte ───────────────────────────────────────────────────────

def capturar(fecha: date, corte: str, hora: datetime, dry_run: bool, forzar: bool) -> int:
    inicio_sesion, _ = sesion_nyse.sesion(fecha)
    from src.data.universo import get_universo
    universo = get_universo()

    log("=" * 60)
    log(f"  SNAPSHOT INTRADIA  |  rueda {fecha}  |  {corte}")
    log(f"  Hora del corte : {hora:%Y-%m-%d %H:%M} UTC")
    log(f"  Tickers        : {len(universo)}")
    log(f"  Modo           : {'DRY RUN' if dry_run else 'REAL'}")
    log("=" * 60)

    if not dry_run and not forzar:
        # Con la DB caida NO se aborta: se captura igual y el dato queda en el
        # spool. Si este chequeo frenara la corrida, el spool no serviria para
        # lo unico que existe (incidente Railway 20/7/2026 en opciones).
        try:
            ya = tickers_capturados(fecha, corte, hora)
        except Exception as e:
            log(f"  [WARN] no se pudo consultar la DB ({type(e).__name__}); se captura igual.")
            ya = 0
        if ya >= COBERTURA_MIN * len(universo):
            log(f"  Ya capturado: {ya}/{len(universo)} tickers con la barra del corte -> skip.")
            return 0
        if ya:
            log(f"  Captura previa incompleta ({ya}/{len(universo)}): se reintenta.")

    if not dry_run:
        from src.utils.yfinance_lock import acquire
        acquire(f"snapshot_intradia {fecha} {corte}")

    capturado_en = datetime.now(timezone.utc)
    try:
        df = descargar(universo, fecha)
    except Exception as e:
        log(f"  [ERROR] descarga: {type(e).__name__}: {e}")
        nl = "\n"
        _telegram(f"ERROR Intradia snapshot -- descarga fallida{nl}"
                  f"<i>rueda {fecha} | {corte} | {capturado_en:%H:%M} UTC</i>{nl}"
                  f"{type(e).__name__}: {str(e)[:200]}")
        return 1

    filas = intradia.normalizar_barras(df, universo, fecha, corte,
                                      inicio_sesion, hora, capturado_en)
    cob = intradia.cobertura(filas, universo, hora)
    esperadas = intradia.barras_esperadas(inicio_sesion, hora)
    log(f"  Filas          : {len(filas)} (esperadas por ticker: {esperadas})")
    log(f"  Cobertura      : {len(cob['completos'])}/{cob['esperados']} "
        f"({cob['pct']:.1%}) con la barra que termina en el corte")
    if cob["sin_ultima_barra"]:
        log(f"  Sin la ultima barra: {', '.join(cob['sin_ultima_barra'][:20])}")
    if cob["sin_datos"]:
        log(f"  Sin datos      : {', '.join(cob['sin_datos'][:20])}"
            f"{' ...' if len(cob['sin_datos']) > 20 else ''}")

    if dry_run:
        # Muestra cruda para verificar el SUPUESTO de zona horaria (docs/intradia.md):
        # el indice que devolvio yahooquery y como quedo normalizado.
        try:
            log(f"  Indice crudo (primeras 3): {list(df.index[:3])}")
        except Exception:
            pass
        for f in filas[:esperadas]:
            log(f"    {f['ticker']:<6} {f['inicio_barra']:%Y-%m-%d %H:%M} UTC  "
                f"o={f['open']} h={f['high']} l={f['low']} c={f['close']} v={f['volumen']}")
        log("  DRY RUN: no se escribio nada.")
        return 0

    if not filas:
        log("  [ERROR] la descarga no trajo ninguna barra valida.")
        nl = "\n"
        _telegram(f"ERROR Intradia snapshot -- sin datos{nl}"
                  f"<i>rueda {fecha} | {corte} | {capturado_en:%H:%M} UTC</i>{nl}"
                  f"yahooquery no devolvio barras para ninguno de {len(universo)} tickers.")
        return 1

    path = intradia.escribir_spool(filas, fecha, corte)
    rc = 0
    try:
        persistir(filas)
        os.remove(path)
        log(f"  DB OK: {len(filas)} filas. Spool borrado.")
    except Exception as e:
        rc = 2
        log(f"  [ERROR] DB: {type(e).__name__}: {e}")
        log(f"  Dato a salvo en disco: {path}")
        nl = "\n"
        _telegram(f"ALERTA Intradia snapshot -- DB inaccesible{nl}"
                  f"<i>rueda {fecha} | {corte} | {capturado_en:%H:%M} UTC</i>{nl}"
                  f"DATO A SALVO en disco: {len(filas)} filas{nl}"
                  f"<code>{os.path.basename(path)}</code>{nl}"
                  f"El proximo disparo lo reinyecta solo; manual: --replay.")

    if cob["pct"] < COBERTURA_MIN:
        rc = max(rc, 2)
        nl = "\n"
        faltan = cob["sin_datos"] + cob["sin_ultima_barra"]
        _telegram(f"WARN Intradia snapshot -- cobertura baja{nl}"
                  f"<i>rueda {fecha} | {corte} | {capturado_en:%H:%M} UTC</i>{nl}"
                  f"{len(cob['completos'])} de {cob['esperados']} tickers con la barra del corte "
                  f"({cob['pct']:.0%}).{nl}"
                  f"Faltan: {', '.join(faltan[:15])}{' ...' if len(faltan) > 15 else ''}{nl}"
                  f"Si queda un disparo en la ventana, reintenta solo.")
    return rc


# ── Status ────────────────────────────────────────────────────────────────────

def status() -> int:
    from src.data.database import get_engine
    with get_engine().connect() as c:
        rows = c.execute(text(
            f"SELECT fecha, corte, COUNT(DISTINCT ticker) AS tickers, COUNT(*) AS filas, "
            f"       MAX(hora_corte) AS hora_corte, MAX(capturado_en) AS capturado_en "
            f"FROM {intradia.TABLA} GROUP BY fecha, corte "
            f"ORDER BY fecha DESC, corte LIMIT 20"
        )).fetchall()
    print(f"\n  {intradia.TABLA} -- ultimas capturas")
    print(f"  {'rueda':<11} {'corte':<12} {'tickers':>7} {'filas':>6}  {'demora':>7}")
    for r in rows:
        demora = (r.capturado_en - r.hora_corte).total_seconds() / 60.0
        print(f"  {r.fecha!s:<11} {r.corte:<12} {r.tickers:>7} {r.filas:>6}  {demora:>5.1f}m")
    pend = intradia.listar_spools()
    print(f"\n  Spools pendientes: {len(pend)}")
    for p in pend:
        print(f"    {os.path.basename(p)}")
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot intradiario (barras de 30 min)")
    ap.add_argument("--fecha", help="rueda AAAA-MM-DD (reconstruccion manual)")
    ap.add_argument("--corte", choices=sesion_nyse.CORTES)
    ap.add_argument("--dry-run", action="store_true", help="descarga y muestra, no escribe")
    ap.add_argument("--forzar", action="store_true", help="recaptura aunque ya este en la DB")
    ap.add_argument("--replay", action="store_true", help="reinyecta spools pendientes")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--sin-telegram", action="store_true")
    args = ap.parse_args()

    global _skip_telegram
    if args.sin_telegram:
        _skip_telegram = True

    if args.status:
        return status()
    if args.replay:
        ok, quedan = replay_spools()
        log(f"  Replay: {ok} reinyectados, {quedan} pendientes.")
        return 2 if quedan else 0

    ahora = datetime.now(timezone.utc)

    if args.fecha or args.corte:
        if not (args.fecha and args.corte):
            ap.error("--fecha y --corte van juntos")
        fecha = date.fromisoformat(args.fecha)
        if not sesion_nyse.con_cobertura(fecha):
            log(f"  [ERROR] {fecha} fuera de la cobertura del calendario "
                f"(hasta {sesion_nyse.COBERTURA_HASTA}): cargar los feriados.")
            return 1
        hora = sesion_nyse.hora_corte(fecha, args.corte)
        if hora is None:
            log(f"  [ERROR] {fecha} no es rueda NYSE.")
            return 1
        if ahora < hora + timedelta(minutes=sesion_nyse.DEMORA_MIN):
            log(f"  [ERROR] el corte {args.corte} de {fecha} todavia no cerro "
                f"({hora:%H:%M} UTC + {sesion_nyse.DEMORA_MIN} min).")
            return 1
        objetivos = [(fecha, args.corte, hora)]
    else:
        hoy = sesion_nyse.fecha_ny(ahora)
        if not sesion_nyse.con_cobertura(hoy):
            log(f"  [ERROR] {hoy} fuera de la cobertura del calendario "
                f"(hasta {sesion_nyse.COBERTURA_HASTA}).")
            # Una vez por dia, no en cada disparo.
            if (ahora.hour, ahora.minute) < (sesion_nyse.CRON_HORAS_UTC[0], 10):
                _telegram("ERROR Intradia snapshot -- el calendario de feriados NYSE "
                          f"termina en {sesion_nyse.COBERTURA_HASTA}: cargar el anio "
                          "nuevo en src/utils/trading_calendar.py. No se captura.")
            return 1
        objetivos = sesion_nyse.cortes_vencidos(ahora)
        if not objetivos:
            log(f"  Sin corte pendiente a las {ahora:%H:%M} UTC ({hoy}). Nada que hacer.")
            return 0

    rc = 0
    if not args.dry_run:
        _, quedan = replay_spools()
        if quedan:
            rc = 2

    for fecha, corte, hora in objetivos:
        rc = _peor(rc, capturar(fecha, corte, hora, args.dry_run, args.forzar))
    return rc


if __name__ == "__main__":
    sys.exit(main())
