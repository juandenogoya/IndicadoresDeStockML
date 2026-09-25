"""
telegram_foto.py
"Foto al cierre" por Telegram de los tickers que marcaron las estrategias FT.

QUE MANDA
    Para cada ticker elegido: cierre y variacion, RSI (Sobreventa/Neutro/
    Sobrecompra), MACD (Compra/Venta), volumen contra la MEDIANA de 52 semanas,
    posicion contra SMA50/SMA200, ruedas hasta el proximo balance y una tabla
    de las 5 ruedas previas con los cambios de estado (cruces de MACD, cambios
    de zona de RSI, cruces de SMA). Es la foto del estado, no un pronostico:
    sirve para decidir si abrir el chart. Reglas y razones:
    docs/telegram_foto.md.

QUE TICKERS
    Los candidatos de la ULTIMA corrida de las estrategias FT activas
    (`ft_candidatos_diarios`): los que aparecen en 2+ FAMILIAS y el top N por
    score dentro de cada familia. `--tickers` saltea la seleccion (foto a
    demanda).

FECHAS
    `ft_candidatos_diarios.fecha` es la fecha de CORRIDA del bot, no la de los
    datos. La foto se arma sobre la ultima rueda de `precios_diarios` de cada
    ticker; si algun ticker quedo en otra rueda que el resto, el bloque lo
    avisa.

LOCAL-only (como FT): borra DATABASE_URL para que get_engine caiga a local.

Uso:
    python scripts/manual/telegram_foto.py --dry-run          (imprime, no envia)
    python scripts/manual/telegram_foto.py                    (envia)
    python scripts/manual/telegram_foto.py --tickers NVDA,AMD --dry-run
    python scripts/manual/telegram_foto.py --top 2

Codigo de salida:
    0 = enviado (o impreso con --dry-run)
    2 = nada para mandar (sin candidatos / sin datos)
    1 = error de DB o de envio
"""

import os
import sys
import argparse
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# LOCAL-only: ft_candidatos_diarios, precios_diarios e indicadores viven en local.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass
os.environ.pop("DATABASE_URL", None)

from src.data.database import query_df                  # noqa: E402
from src.utils import foto_ticker as ft                 # noqa: E402

DIAS_HISTORIA = 400     # calendario: cubre 252 ruedas + las 6 de la tabla


def cargar_candidatos(fecha=None):
    """Candidatos de la ultima corrida (o de `fecha`) de las estrategias activas."""
    if fecha is None:
        df = query_df("""
            SELECT MAX(c.fecha) AS f
            FROM ft_candidatos_diarios c
            JOIN ft_estrategias e ON e.id = c.estrategia_id
            WHERE e.activa
        """)
        if df.empty or df["f"].iloc[0] is None:
            return None, [], 0
        fecha = df["f"].iloc[0]
    df = query_df("""
        SELECT c.ticker, c.score, e.nombre AS estrategia, e.logica
        FROM ft_candidatos_diarios c
        JOIN ft_estrategias e ON e.id = c.estrategia_id
        WHERE e.activa AND c.fecha = :f
    """, {"f": fecha})
    cands = [{
        "ticker": r.ticker,
        "estrategia": r.estrategia,
        "familia": ft.familia_de(r.logica),
        "score": None if r.score is None else float(r.score),
    } for r in df.itertuples()]
    n_estr = df["estrategia"].nunique() if not df.empty else 0
    return fecha, cands, n_estr


def cargar_historia(tickers):
    """Historia diaria por ticker (precio + indicadores), orden ascendente."""
    df = query_df("""
        SELECT p.ticker, p.fecha, p.close, p.volume,
               i.rsi14, i.macd, i.macd_signal, i.sma50, i.sma200
        FROM precios_diarios p
        LEFT JOIN indicadores_tecnicos i
               ON i.ticker = p.ticker AND i.fecha = p.fecha
        WHERE p.ticker = ANY(:t)
          AND p.fecha >= :desde
        ORDER BY p.ticker, p.fecha
    """, {"t": list(tickers), "desde": date.today() - timedelta(days=DIAS_HISTORIA)})
    out = {}
    for tk, g in df.groupby("ticker", sort=False):
        out[tk] = g.drop(columns="ticker").to_dict("records")
    return out


def cargar_contexto(tickers):
    """Proxima fecha de balance y sector de cada ticker."""
    earn = query_df("""
        SELECT ticker, earnings_date FROM earnings_calendar WHERE ticker = ANY(:t)
    """, {"t": list(tickers)})
    sect = query_df("""
        SELECT ticker, sector FROM activos WHERE ticker = ANY(:t)
    """, {"t": list(tickers)})
    earnings = {r.ticker: r.earnings_date for r in earn.itertuples()
                if r.earnings_date is not None and str(r.earnings_date) != "NaT"}
    sectores = {r.ticker: r.sector for r in sect.itertuples()}
    return earnings, sectores


def _a_date(x):
    if x is None:
        return None
    return x.date() if hasattr(x, "date") and not isinstance(x, date) else x


def armar_mensajes(seleccion, fecha_cand, n_cand, n_estr, historia,
                   earnings, sectores, familias_por_ticker):
    fotos = {}
    for tk in seleccion["orden"]:
        filas = historia.get(tk, [])
        for r in filas:
            r["fecha"] = _a_date(r["fecha"])
        foto = ft.construir_foto(filas)
        if foto is not None:
            fotos[tk] = foto
    if not fotos:
        return [], None

    # Rueda de referencia: la mas reciente entre los fotografiados
    rueda = max(f["fecha"] for f in fotos.values())
    bloques = [ft.encabezado(rueda, fecha_cand, seleccion, n_cand, n_estr)]
    for tk in seleccion["orden"]:
        if tk not in fotos:
            continue
        bloques.append(ft.bloque_ticker(
            tk, fotos[tk],
            familias=familias_por_ticker.get(tk),
            sector=sectores.get(tk),
            earnings_fecha=_a_date(earnings.get(tk)),
            rueda_ref=rueda,
        ))
    sin_datos = [tk for tk in seleccion["orden"] if tk not in fotos]
    if sin_datos:
        bloques.append("Sin historia suficiente: " + ", ".join(sin_datos))
    return ft.empaquetar(bloques), rueda


def main():
    ap = argparse.ArgumentParser(description="Foto al cierre por Telegram")
    ap.add_argument("--dry-run", action="store_true", help="imprime y no envia")
    ap.add_argument("--tickers", help="lista separada por coma (saltea la seleccion FT)")
    ap.add_argument("--top", type=int, default=ft.TOP_POR_FAMILIA,
                    help=f"top por familia (default {ft.TOP_POR_FAMILIA})")
    ap.add_argument("--fecha-candidatos", help="YYYY-MM-DD (default: ultima corrida)")
    args = ap.parse_args()

    try:
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            seleccion = {"confluencia": [], "top": {}, "orden": tickers}
            fecha_cand, n_cand, n_estr, fam_tk = None, 0, 0, {}
        else:
            f = date.fromisoformat(args.fecha_candidatos) if args.fecha_candidatos else None
            fecha_cand, cands, n_estr = cargar_candidatos(f)
            if not cands:
                print("Sin candidatos FT para la fecha pedida: nada para mandar.")
                return 2
            n_cand = len({c["ticker"] for c in cands})
            seleccion = ft.seleccionar(cands, top_n=args.top)
            grupos = ft.agrupar(cands)
            fam_tk = {tk: sorted(g["familias"], key=ft.orden_familia)
                      for tk, g in grupos.items()}

        historia = cargar_historia(seleccion["orden"])
        earnings, sectores = cargar_contexto(seleccion["orden"])
    except Exception as e:
        print(f"[ERROR] DB: {e}")
        return 1

    mensajes, rueda = armar_mensajes(seleccion, fecha_cand, n_cand, n_estr,
                                     historia, earnings, sectores, fam_tk)
    if not mensajes:
        print("Ningun ticker con historia suficiente: nada para mandar.")
        return 2

    print(f"Rueda de datos: {rueda} | tickers: {len(seleccion['orden'])} | "
          f"mensajes: {len(mensajes)}")
    for i, m in enumerate(mensajes, 1):
        print(f"\n----- mensaje {i}/{len(mensajes)} ({len(m)} chars) -----")
        print(m)

    if args.dry_run:
        return 0

    from src.pipeline.telegram_notifier import _send
    ok = all([_send(m) for m in mensajes])
    print("Telegram: enviado" if ok else "Telegram: FALLO el envio")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
