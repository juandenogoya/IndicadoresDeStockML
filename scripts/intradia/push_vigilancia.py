"""
push_vigilancia.py -- arma la LISTA DE VIGILANCIA de la rueda D y la sube.

Corre en LOCAL despues de la rutina diaria (doble clic en
scripts/manual/intradia_post_rutina.bat), que hoy se corre antes de la apertura:
la lista de D tiene que estar en Railway antes del corte de las 11:00 ET de la
rueda siguiente, que es cuando el snapshot de Oracle la lee. Etapa 2 de
docs/intradia.md.

QUE TICKERS (decision 3/11/2026, reemplaza a la del 30/10): SOLO los de la
FOTO -- 2+ familias de FT y top por familia (foto_ticker.seleccionar, la misma
seleccion que telegram_foto.py, de la misma corrida de FT). Son ~14 por rueda.

Antes entraban tambien los del CIERRE por sector, y la lista daba 111 tickers.
Medido sobre 38 ruedas: el cierre aportaba 104 de esos 111 y el 88% de las
lineas del mensaje, con la MISMA tasa de persistencia al cierre que los de la
foto (84% y 84%) -- o sea, no peores, pero sobre tickers que no se miran.
El cierre SIGUE usandose para dos cosas: saber QUE se cruzo en D (los muros de
D-1 y la marca en las SMA) y como CONTEXTO en los motivos de un ticker de foto.
Las definiciones NO se repiten aca: se importan de telegram_foto.py y
telegram_cierre.py, asi la lista dice lo mismo que los mensajes de la noche.

DESTINO. Escribe la lista en LOCAL y en RAILWAY (dual-write, como `activos`):
Oracle la lee de Railway y local guarda la historia para evaluar. Una lista
`push` de D REEMPLAZA a la anterior de D (la rutina pudo correr dos veces).

RECONSTRUCCION (--rueda / --desde --hasta): listas de ruedas pasadas para la
evaluacion de la etapa 2. Van solo a LOCAL con origen 'reconstruida' y nunca
pisan una lista 'push'. Aproximacion declarada: la corrida de FT de la rueda D
se toma como la ULTIMA corrida entre D y la rueda habil siguiente. La rutina se
corre antes de la apertura de D+1 (y entonces decide con los datos de D); si no
hubo corrida en D+1, la de la noche de D. `ft_candidatos_diarios.fecha` es la de
CORRIDA y no la de datos, asi que no hay mejor criterio. Y la foto solo ve las
estrategias activas HOY (las dadas de baja no aparecen en ruedas pasadas).

Uso:
    python scripts/intradia/push_vigilancia.py                  # rueda D de hoy, local + Railway
    python scripts/intradia/push_vigilancia.py --dry-run
    python scripts/intradia/push_vigilancia.py --desde 2026-09-01 --hasta 2026-10-29

Salida: 0 OK / 2 lista parcial (falto la foto o Railway) / 1 error.
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "manual"))

# telegram_foto / telegram_cierre cargan .env y borran DATABASE_URL al importarse:
# todas las lecturas son LOCAL (como la foto y el cierre). Railway se abre aparte.
import telegram_cierre as tc     # noqa: E402
import telegram_foto as tf       # noqa: E402

from sqlalchemy import create_engine, text   # noqa: E402

from src.data.database import get_engine, query_df          # noqa: E402
from src.utils import foto_ticker as ft                     # noqa: E402
from src.utils import vigilancia_intradia as vi             # noqa: E402
from src.utils.trading_calendar import next_trading_day     # noqa: E402

DIAS_SIGMA = 400     # calendario: cubre las 252 ruedas del desvio


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _leer_database_url_local() -> str:
    """Lee DATABASE_URL de .env.local SIN cargarla en os.environ."""
    env_path = os.path.join(ROOT, ".env.local")
    if not os.path.exists(env_path):
        raise RuntimeError(".env.local no encontrado (DATABASE_URL de Railway).")
    with open(env_path, "r", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if linea.startswith("DATABASE_URL"):
                return linea.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DATABASE_URL no esta definida en .env.local.")


def engine_railway():
    url = _leer_database_url_local().replace("postgres://", "postgresql://", 1)
    if "postgresql+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url, connect_args={"sslmode": "require"}, pool_pre_ping=True)


# ── Insumos ───────────────────────────────────────────────────────────────────

def corrida_para(rueda: date):
    """Ultima corrida de FT entre la rueda y la habil siguiente (reconstruccion)."""
    df = query_df("""
        SELECT MAX(c.fecha) AS f FROM ft_candidatos_diarios c
        JOIN ft_estrategias e ON e.id = c.estrategia_id
        WHERE e.activa AND c.fecha BETWEEN :d AND :d1
    """, {"d": rueda, "d1": next_trading_day(rueda)})
    f = df["f"].iloc[0] if not df.empty else None
    return None if f is None or str(f) == "NaT" else tc._d(f)


def motivos_de_foto(rueda: date, reconstruida: bool) -> tuple[dict, str]:
    """({ticker: [motivo]}, nota). Sin corrida valida para D, la foto no entra."""
    if reconstruida:
        fecha = corrida_para(rueda)
        if fecha is None:
            return {}, f"sin corrida de FT entre {rueda} y la rueda siguiente"
        fecha_cand, cands, _ = tf.cargar_candidatos(fecha)
    else:
        fecha_cand, cands, _ = tf.cargar_candidatos()
        fecha_cand = tc._d(fecha_cand) if fecha_cand is not None else None
        # Guard de frescura: una corrida anterior a D es de otra rueda.
        if fecha_cand is None or fecha_cand < rueda:
            return {}, f"ultima corrida de FT {fecha_cand} anterior a la rueda {rueda}"
    if not cands:
        return {}, f"corrida {fecha_cand} sin candidatos"
    return vi.motivos_foto(ft.seleccionar(cands)), f"corrida de FT {fecha_cand}"


def cargar_extra(tickers: list, rueda: date) -> tuple[dict, dict, dict]:
    """(strikes de OI, sma21, sigmas) de la rueda D para los tickers de la lista.

    La SMA21 se trae aca y no de telegram_cierre: ese modulo solo consulta sma50
    y sma200, y no se toca para no cambiar el mensaje de la noche.
    """
    p = {"t": list(tickers), "d": rueda}
    muros = query_df("""
        SELECT ticker, soporte_strike, resistencia_strike FROM opciones_pcr_plazo_diario
        WHERE ventana = 'corto' AND fecha = :d AND ticker = ANY(:t)
    """, p)
    smas = query_df("""
        SELECT ticker, sma21 FROM indicadores_tecnicos
        WHERE fecha = :d AND ticker = ANY(:t)
    """, p)
    precios = query_df("""
        SELECT ticker, fecha, close FROM precios_diarios
        WHERE ticker = ANY(:t) AND fecha <= :d AND fecha > :desde
        ORDER BY ticker, fecha
    """, dict(p, desde=rueda - timedelta(days=DIAS_SIGMA)))
    strikes = {r.ticker: (tc._nn(r.soporte_strike), tc._nn(r.resistencia_strike))
               for r in muros.itertuples()}
    sma21 = {r.ticker: tc._nn(r.sma21) for r in smas.itertuples()}
    sigmas = {tk: vi.sigma_log(list(g["close"])) for tk, g in precios.groupby("ticker")}
    return strikes, sma21, sigmas


def armar(rueda: date, reconstruida: bool) -> tuple[list, list, dict]:
    """(tickers, niveles, info) de la lista de vigilancia de la rueda D."""
    resultados = tc.cargar_resultados(1, hasta=rueda)
    if not resultados or resultados[0][0]["rueda"] != rueda:
        raise RuntimeError(f"{rueda} no es una rueda del universo en precios_diarios")
    res, filas = resultados[0]
    m_foto, nota_foto = motivos_de_foto(rueda, reconstruida)
    m_cierre = vi.motivos_cierre(res)
    # La lista son SOLO los tickers de la foto. Del cierre se conserva el motivo
    # como CONTEXTO de esos tickers: no agrega tickers nuevos.
    motivos = {tk: list(ms) + m_cierre.get(tk, []) for tk, ms in m_foto.items()}
    strikes, sma21, sigmas = cargar_extra(motivos, rueda)
    filas = [dict(f, sma21=sma21.get(f["ticker"])) for f in filas]
    origen = vi.ORIGEN_RECONSTRUIDA if reconstruida else vi.ORIGEN_PUSH
    tickers, niveles = vi.armar_lista(rueda, filas, res, motivos, strikes, sigmas,
                                      datetime.now(timezone.utc), origen)
    info = {"nota_foto": nota_foto, "n_foto": len(m_foto),
            "n_contexto": sum(1 for tk in motivos if tk in m_cierre),
            "rueda_ant": res["rueda_ant"]}
    return tickers, niveles, info


# ── Escritura ─────────────────────────────────────────────────────────────────

def _insert(tabla, cols):
    return text(f"INSERT INTO {tabla} ({', '.join(cols)}) "
                f"VALUES ({', '.join(':' + c for c in cols)})")


def escribir(eng, rueda: date, tickers: list, niveles: list, info: dict,
             origen: str, reemplazar: bool) -> str:
    """
    Una transaccion: la lista de D entra entera o no entra. `reemplazar`
    (push) borra la de D antes; si no (reconstruida), no toca una rueda que ya
    tiene lista. La cabecera se escribe SIEMPRE, aunque la lista este vacia:
    sin ella, Oracle diria "lista no subida" cuando en realidad no habia nada
    que vigilar.
    """
    with eng.begin() as c:
        existe = c.execute(text(f"SELECT COUNT(*) FROM {vi.TABLA_LISTAS} WHERE fecha = :f"),
                           {"f": rueda}).scalar()
        if existe and not reemplazar:
            return "ya tenia lista, no se toca"
        for tabla in (vi.TABLA_NIVELES, vi.TABLA_TICKERS, vi.TABLA_LISTAS):
            c.execute(text(f"DELETE FROM {tabla} WHERE fecha = :f"), {"f": rueda})
        c.execute(_insert(vi.TABLA_LISTAS, vi.COLS_LISTAS), {
            "fecha": rueda, "n_tickers": len(tickers), "origen": origen,
            "nota": info["nota_foto"], "generado_en": datetime.now(timezone.utc)})
        if tickers:
            c.execute(_insert(vi.TABLA_TICKERS, vi.COLS_TICKERS), tickers)
        if niveles:
            c.execute(_insert(vi.TABLA_NIVELES, vi.COLS_NIVELES), niveles)
    return f"{len(tickers)} tickers, {len(niveles)} niveles"


def resumen(rueda, tickers, niveles, info):
    log(f"  Rueda {rueda} (anterior {info['rueda_ant']}) | {info['nota_foto']}")
    log(f"  Vigilados: {len(tickers)} (todos de la foto; {info['n_contexto']} con "
        f"contexto del cierre) | niveles: {len(niveles)}")
    cruzados = sum(1 for n in niveles if n["cruzado_en_d"])
    sin_sigma = sum(1 for t in tickers if t["sigma_d"] is None)
    log(f"  Niveles cruzados en D: {cruzados} | sin desvio (historia corta): {sin_sigma}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Lista de vigilancia intradiaria")
    ap.add_argument("--rueda", help="AAAA-MM-DD: reconstruye UNA rueda (solo local)")
    ap.add_argument("--desde", help="AAAA-MM-DD: reconstruye un rango (solo local)")
    ap.add_argument("--hasta", help="AAAA-MM-DD (con --desde; default la ultima rueda)")
    ap.add_argument("--target", choices=["ambos", "local", "railway"], default="ambos",
                    help="destino del push de hoy (la reconstruccion va siempre a local)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eng_loc = get_engine()
    if args.rueda or args.desde:
        if args.rueda:
            ruedas = [date.fromisoformat(args.rueda)]
        else:
            from src.data.universo import get_universo_sectores
            todas = tc.ruedas_universo(get_universo_sectores().keys(),
                                       date.fromisoformat(args.hasta) if args.hasta else None)
            ruedas = [r for r in todas if r >= date.fromisoformat(args.desde)]
        rc = 0
        for r in ruedas:
            try:
                tickers, niveles, info = armar(r, reconstruida=True)
            except Exception as e:
                log(f"  [ERROR] {r}: {e}")
                rc = 1
                continue
            resumen(r, tickers, niveles, info)
            if not args.dry_run:
                txt = escribir(eng_loc, r, tickers, niveles, info, vi.ORIGEN_RECONSTRUIDA,
                               reemplazar=False)
                log(f"  LOCAL: {txt}")
        return rc

    # Push de hoy: la rueda D es la ultima rueda del universo.
    try:
        resultados = tc.cargar_resultados(1)
        if not resultados:
            log("  [ERROR] sin rueda del universo en precios_diarios.")
            return 1
        rueda = resultados[0][0]["rueda"]
        tickers, niveles, info = armar(rueda, reconstruida=False)
    except Exception as e:
        log(f"  [ERROR] armando la lista: {type(e).__name__}: {e}")
        return 1
    log("=" * 60)
    log(f"  LISTA DE VIGILANCIA  |  rueda {rueda}  |  {'DRY RUN' if args.dry_run else args.target}")
    log("=" * 60)
    resumen(rueda, tickers, niveles, info)
    for t in tickers[:60]:
        log(f"    {t['ticker']:<6} {t['motivos']}")
    if args.dry_run:
        return 0
    if not tickers:
        log("  [WARN] lista VACIA: se sube igual (Oracle informa 0 vigilados, no 'no subida').")

    rc = 0 if info["n_foto"] else 2
    if args.target in ("ambos", "local"):
        try:
            log(f"  LOCAL  : {escribir(eng_loc, rueda, tickers, niveles, info, vi.ORIGEN_PUSH, reemplazar=True)}")
        except Exception as e:
            log(f"  [ERROR] LOCAL: {e}")
            rc = 1
    if args.target in ("ambos", "railway"):
        try:
            log(f"  RAILWAY: {escribir(engine_railway(), rueda, tickers, niveles, info, vi.ORIGEN_PUSH, reemplazar=True)}")
        except Exception as e:
            log(f"  [ERROR] RAILWAY: {e} -- Oracle no va a tener la lista de {rueda}.")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
