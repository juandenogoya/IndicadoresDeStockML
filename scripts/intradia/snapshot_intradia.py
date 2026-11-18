"""
snapshot_intradia.py -- captura de barras de 30 min en los cortes intradiarios
y alerta de la lista de vigilancia.

Corre en la VM de Oracle (cron de L-V, varios horarios UTC) y escribe en Railway
la tabla precios_intradia_30m. Cada disparo pregunta que corte vencio
(src/utils/sesion_nyse.cortes_vencidos); si no hay ninguno, sale sin hacer nada.
Asi un unico horario de cron cubre horario de verano, invierno y medias ruedas,
y el disparo de los :20 es el reintento del de los :05. Diseno: docs/intradia.md.

    apertura_90 = apertura + 90 min (11:00 ET)
    cierre_60   = cierre   - 60 min (15:00 ET; 12:00 ET en media rueda)

En prueba en la rama main-y9tr06.

Orden de un corte:
    1. reinyecta spools pendientes (DB caida en un disparo anterior)
    2. si ya esta capturado, no descarga; si no: lock de Yahoo, descarga,
       normaliza (descarta la barra en curso), spool en disco ANTES de la DB,
       upsert, borra el spool. Con la DB caida el dato queda en disco y avisa.
    3. cobertura < 90% del universo -> aviso
    4. ALERTA (etapa 2): lee la lista de vigilancia de la rueda anterior (la sube
       push_vigilancia.py despues de la rutina), evalua los eventos y manda UN
       mensaje por corte, registrado en vigilancia_envios para no repetirlo en el
       reintento. Sin lista de la rueda anterior, lo dice en vez de callar.

A diferencia de opciones, un corte perdido NO es irrecuperable: Yahoo guarda
barras de 30 min unos 60 dias. Reconstruir uno: --fecha AAAA-MM-DD --corte X;
las ultimas N ruedas de una vez: --reconstruir N (sin alertas).

Uso:
    python scripts/intradia/snapshot_intradia.py                 # lo que dispara el cron
    python scripts/intradia/snapshot_intradia.py --dry-run --fecha 2026-11-02 --corte apertura_90
    python scripts/intradia/snapshot_intradia.py --fecha 2026-11-02 --corte cierre_60 [--forzar]
    python scripts/intradia/snapshot_intradia.py --solo-alertas --fecha 2026-11-02 --corte cierre_60
    python scripts/intradia/snapshot_intradia.py --reconstruir 40
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
from src.utils.trading_calendar import prev_trading_day, trading_days_between

import intradia_db as idb   # mismo directorio (scripts/intradia)

COBERTURA_MIN = 0.90
MAX_RECONSTRUIR = 40          # ruedas: Yahoo sirve barras de 30 min ~60 dias calendario
_skip_telegram = os.getenv("INTRADIA_SKIP_TELEGRAM", "0") == "1"


def _peor(a: int, b: int) -> int:
    """Combina codigos de salida: el error (1) manda sobre el aviso (2)."""
    return 1 if 1 in (a, b) else max(a, b)


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)


def _telegram(texto: str) -> bool:
    if _skip_telegram:
        log("  (sin Telegram) mensaje no enviado:")
        print(texto, flush=True)
        return True
    try:
        from src.pipeline.telegram_notifier import _send
        return bool(_send(texto))
    except Exception as e:
        log(f"  [WARN] Telegram no disponible: {e}")
        return False


def _engine():
    from src.data.database import get_engine
    return get_engine()


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
    ultimo_inicio = hora_corte - timedelta(minutes=sesion_nyse.BARRA_MIN)
    with _engine().connect() as c:
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

def _pedir(tickers: list[str], **kw):
    from yahooquery import Ticker
    from src.utils.yahooquery_loader import RateLimitDetected, _is_ratelimit_msg
    df = Ticker(tickers, asynchronous=True).history(interval="30m", **kw)
    if isinstance(df, dict):
        msgs = " | ".join(str(v) for v in df.values())
        if _is_ratelimit_msg(msgs):
            raise RateLimitDetected(msgs[:200])
        return None
    return df


def descargar(tickers: list[str], desde: date, hasta: date):
    """
    Barras de 30 min de las ruedas desde..hasta via yahooquery. Pide start/end y,
    si vuelve vacio, cae a period (Yahoo a veces no respeta start/end en
    intradiario). El recorte a la sesion y al corte lo hace
    intradia.normalizar_barras.
    """
    df = _pedir(tickers, start=desde.isoformat(), end=(hasta + timedelta(days=1)).isoformat())
    if df is None or df.empty:
        dias = (datetime.now(timezone.utc).date() - desde).days
        df = _pedir(tickers, period="5d" if dias <= 4 else "60d")
    return df


# ── Captura de un corte ───────────────────────────────────────────────────────

def capturar(fecha: date, corte: str, hora: datetime, dry_run: bool, forzar: bool):
    """Devuelve (codigo, filas): filas = barras del corte en memoria, o None."""
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
            return 0, None
        if ya:
            log(f"  Captura previa incompleta ({ya}/{len(universo)}): se reintenta.")

    if not dry_run:
        from src.utils.yfinance_lock import acquire
        acquire(f"snapshot_intradia {fecha} {corte}")

    capturado_en = datetime.now(timezone.utc)
    try:
        df = descargar(universo, fecha, fecha)
    except Exception as e:
        log(f"  [ERROR] descarga: {type(e).__name__}: {e}")
        nl = "\n"
        _telegram(f"ERROR Intradia snapshot -- descarga fallida{nl}"
                  f"<i>rueda {fecha} | {corte} | {capturado_en:%H:%M} UTC</i>{nl}"
                  f"{type(e).__name__}: {str(e)[:200]}")
        return 1, None

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
        return 0, filas

    if not filas:
        log("  [ERROR] la descarga no trajo ninguna barra valida.")
        nl = "\n"
        _telegram(f"ERROR Intradia snapshot -- sin datos{nl}"
                  f"<i>rueda {fecha} | {corte} | {capturado_en:%H:%M} UTC</i>{nl}"
                  f"yahooquery no devolvio barras para ninguno de {len(universo)} tickers.")
        return 1, None

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
    return rc, filas


# ── Alerta de la lista de vigilancia (etapa 2) ────────────────────────────────

def alertar(fecha: date, corte: str, hora: datetime, filas, dry_run: bool, forzar: bool) -> int:
    """
    Evalua la lista de vigilancia en el corte y manda UN mensaje. No repite un
    mensaje ya enviado, salvo que el anterior haya sido "sin lista" y la lista
    haya llegado despues (la rutina corrio tarde).
    """
    eng = _engine()
    try:
        if not dry_run and not forzar:
            prev = idb.envio_previo(eng, fecha, corte)
            if prev and (prev["fecha_lista"] is not None or
                         idb.ultima_lista(eng, fecha) != prev_trading_day(fecha)):
                log(f"  Alerta {corte} de {fecha} ya enviada a las "
                    f"{prev['enviado_en']:%H:%M} UTC -> no se repite.")
                return 0
        r = idb.evaluar_corte(eng, fecha, corte, hora, filas)
    except Exception as e:
        # DB caida: la captura ya aviso y el dato esta en el spool. Sin lista no
        # hay alerta; no se manda un segundo mensaje por la misma falla.
        log(f"  [WARN] alerta no evaluada ({type(e).__name__}: {e}).")
        return 2

    n_ev = len(r["nuevos"]) + len(r["deshechos"])
    log(f"  Alerta: lista {r['fecha_lista'] or 'NO DISPONIBLE'} | vigilados {len(r['tickers'])} "
        f"| eventos {len(r['eventos'])} | en el mensaje {n_ev} | sin dato {len(r['sin_dato'])}")
    if dry_run:
        log("  DRY RUN: mensaje que se enviaria:")
        print(r["texto"], flush=True)
        return 0
    if not _telegram(r["texto"]):
        log("  [ERROR] no se pudo enviar la alerta: el proximo disparo reintenta.")
        return 1
    try:
        idb.registrar_envio(eng, {
            "fecha": fecha, "corte": corte, "fecha_lista": r["fecha_lista"],
            "n_vigilados": len(r["tickers"]), "n_eventos": n_ev, "texto": r["texto"],
            "enviado_en": datetime.now(timezone.utc)})
    except Exception as e:
        log(f"  [WARN] alerta enviada pero no registrada ({e}): el reintento la repetiria.")
        return 2
    return 0 if r["ok"] else 2


# ── Reconstruccion de las ultimas N ruedas ────────────────────────────────────

def reconstruir(n: int, forzar: bool) -> int:
    """
    Baja de una vez las barras de las ultimas `n` ruedas y guarda ambos cortes
    de cada una (sin alertas: son ruedas pasadas). Sirve para la evaluacion de
    la etapa 2 sin esperar semanas. OJO: las filas quedan con capturado_en de
    HOY, asi que no son point-in-time (Yahoo pudo revisar barras) y la
    evaluacion de la etapa 1 tiene que empezar desde el arranque del cron.
    """
    if not 1 <= n <= MAX_RECONSTRUIR:
        log(f"  [ERROR] --reconstruir entre 1 y {MAX_RECONSTRUIR} ruedas.")
        return 1
    ahora = datetime.now(timezone.utc)
    hoy = sesion_nyse.fecha_ny(ahora)
    ruedas = [d for d in trading_days_between(hoy - timedelta(days=3 * n + 10), hoy)
              if sesion_nyse.con_cobertura(d)]
    objetivos = [(d, c, h) for d in ruedas for c, h in sesion_nyse.cortes(d).items()
                 if ahora >= h + timedelta(minutes=sesion_nyse.DEMORA_MIN)]
    dias = sorted({d for d, _, _ in objetivos})[-n:]
    objetivos = [o for o in objetivos if o[0] in dias]
    if not objetivos:
        log("  Nada para reconstruir.")
        return 0

    from src.data.universo import get_universo
    from src.utils.yfinance_lock import acquire
    universo = get_universo()
    acquire(f"snapshot_intradia --reconstruir {n}")
    log(f"  RECONSTRUIR {len(dias)} ruedas ({dias[0]} -> {dias[-1]}), {len(universo)} tickers")
    try:
        df = descargar(universo, dias[0], dias[-1])
    except Exception as e:
        log(f"  [ERROR] descarga: {type(e).__name__}: {e}")
        return 1

    rc = 0
    capturado_en = datetime.now(timezone.utc)
    for d, corte, hora in objetivos:
        if not forzar and tickers_capturados(d, corte, hora) >= COBERTURA_MIN * len(universo):
            log(f"  {d} {corte}: ya capturado -> skip")
            continue
        inicio, _ = sesion_nyse.sesion(d)
        filas = intradia.normalizar_barras(df, universo, d, corte, inicio, hora, capturado_en)
        cob = intradia.cobertura(filas, universo, hora)
        persistir(filas)
        if cob["pct"] < COBERTURA_MIN:
            rc = 2
        log(f"  {d} {corte}: {len(filas)} filas | cobertura {cob['pct']:.1%}")
    return rc


# ── Status ────────────────────────────────────────────────────────────────────

def status() -> int:
    from src.utils import vigilancia_intradia as vi
    eng = _engine()
    with eng.connect() as c:
        rows = c.execute(text(
            f"SELECT fecha, corte, COUNT(DISTINCT ticker) AS tickers, COUNT(*) AS filas, "
            f"       MAX(hora_corte) AS hora_corte, MAX(capturado_en) AS capturado_en "
            f"FROM {intradia.TABLA} GROUP BY fecha, corte "
            f"ORDER BY fecha DESC, corte LIMIT 20"
        )).fetchall()
        try:
            envios = c.execute(text(
                f"SELECT fecha, corte, fecha_lista, n_vigilados, n_eventos, enviado_en "
                f"FROM {vi.TABLA_ENVIOS} ORDER BY fecha DESC, corte LIMIT 10")).fetchall()
            listas = c.execute(text(
                f"SELECT fecha, n_tickers AS n, origen FROM {vi.TABLA_LISTAS} "
                f"ORDER BY fecha DESC LIMIT 5")).fetchall()
        except Exception:
            envios, listas = [], []
    print(f"\n  {intradia.TABLA} -- ultimas capturas")
    print(f"  {'rueda':<11} {'corte':<12} {'tickers':>7} {'filas':>6}  {'demora':>7}")
    for r in rows:
        demora = (r.capturado_en - r.hora_corte).total_seconds() / 60.0
        print(f"  {r.fecha!s:<11} {r.corte:<12} {r.tickers:>7} {r.filas:>6}  {demora:>5.1f}m")
    print("\n  Listas de vigilancia (ultimas 5)")
    for r in listas:
        print(f"  {r.fecha!s:<11} {r.n:>4} tickers  ({r.origen})")
    print("\n  Alertas enviadas (ultimas 10)")
    for r in envios:
        print(f"  {r.fecha!s:<11} {r.corte:<12} lista {r.fecha_lista!s:<11} "
              f"vigilados {r.n_vigilados:>3}  lineas {r.n_eventos:>3}  {r.enviado_en:%H:%M} UTC")
    pend = intradia.listar_spools()
    print(f"\n  Spools pendientes: {len(pend)}")
    for p in pend:
        print(f"    {os.path.basename(p)}")
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot intradiario (barras de 30 min) + alerta")
    ap.add_argument("--fecha", help="rueda AAAA-MM-DD (reconstruccion manual)")
    ap.add_argument("--corte", choices=sesion_nyse.CORTES)
    ap.add_argument("--dry-run", action="store_true", help="descarga y muestra, no escribe ni envia")
    ap.add_argument("--forzar", action="store_true", help="recaptura / reenvia aunque ya este hecho")
    ap.add_argument("--sin-alertas", action="store_true", help="solo captura")
    ap.add_argument("--solo-alertas", action="store_true", help="no captura: evalua y envia")
    ap.add_argument("--reconstruir", type=int, metavar="N", help="ultimas N ruedas, sin alertas")
    ap.add_argument("--replay", action="store_true", help="reinyecta spools pendientes")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--sin-telegram", action="store_true", help="imprime en vez de enviar")
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
    if args.reconstruir is not None:
        return reconstruir(args.reconstruir, args.forzar)

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
        # Un corte de otro dia reconstruido a mano no dispara alerta (seria vieja),
        # salvo pedido explicito con --solo-alertas.
        manual_viejo = fecha != sesion_nyse.fecha_ny(ahora)
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
        manual_viejo = False
        if not objetivos:
            log(f"  Sin corte pendiente a las {ahora:%H:%M} UTC ({hoy}). Nada que hacer.")
            return 0

    rc = 0
    if not args.dry_run and not args.solo_alertas:
        _, quedan = replay_spools()
        if quedan:
            rc = 2

    for fecha, corte, hora in objetivos:
        filas = None
        if not args.solo_alertas:
            rc_c, filas = capturar(fecha, corte, hora, args.dry_run, args.forzar)
            rc = _peor(rc, rc_c)
        if args.sin_alertas or (manual_viejo and not args.solo_alertas):
            continue
        rc = _peor(rc, alertar(fecha, corte, hora, filas, args.dry_run, args.forzar))
    return rc


if __name__ == "__main__":
    sys.exit(main())
