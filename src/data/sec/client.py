"""
client.py -- descarga de data.sec.gov con cache en disco.

Sin DB y sin imports del proyecto (ver la regla en src/data/sec/__init__.py):
solo stdlib + requests. El estado que necesita para decidir que re-bajar se lo
pasa el llamador; este modulo no consulta la base.

--------------------------------------------------------------------------
DOS COSAS QUE SE APRENDIERON A LOS GOLPES
--------------------------------------------------------------------------
1. SEC CORTA LAS DESCARGAS HECHAS CON UNA CONEXION NUEVA POR PEDIDO.
   Un loop de `curl` fallo los 147 tickers seguidos (codigo 000, conexion
   rechazada) mientras un pedido suelto andaba perfecto. Con
   requests.Session (keep-alive) y ~0.55 s de pausa: 147 de 147, cero fallos.
   Por eso la sesion es obligatoria y se reutiliza.

2. `submissions` PESA 23 VECES MENOS QUE `companyfacts`.
   AAPL: 164 KB contra 3.8 MB. Y trae el accession del ultimo 10-Q/10-K.
   Ese es el disparador del incremental: si el accession no cambio respecto
   de lo ya ingestado, no hace falta bajar el archivo pesado. Un refresh sin
   balances nuevos pasa de ~522 MB a ~24 MB para los 147 tickers.

SEC exige un User-Agent identificable (mail de contacto). Se toma de la env
var SEC_USER_AGENT; sin ella se usa un valor generico que SEC puede rechazar.
"""

import json
import os
import time

BASE = "https://data.sec.gov"
MAPA_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# SEC admite hasta 10 pedidos/segundo. Se usa bastante menos: la descarga
# completa es de una sola vez y no hay apuro; el incremental son 147 pedidos
# chicos. Ir al limite fue lo que hizo que nos cortaran.
PAUSA = 0.55
REINTENTOS = 3
TIMEOUT = 60

FORMS_PERIODICOS = ("10-Q", "10-K")


def user_agent():
    return os.environ.get(
        "SEC_USER_AGENT",
        "IndicadoresDeStockML research script (configurar SEC_USER_AGENT)")


def sesion(ua=None):
    """
    Sesion HTTP con keep-alive. IMPRESCINDIBLE: ver nota 1 del encabezado.
    """
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": ua or user_agent(),
                      "Accept-Encoding": "gzip, deflate"})
    return s


class AccesoDenegado(RuntimeError):
    """SEC rechazo el pedido por User-Agent. No se reintenta: no es transitorio."""


def _get(s, url):
    """
    GET con reintentos y backoff. Devuelve el response, o None si no existe.

    El 403 NO se reintenta: SEC lo devuelve cuando el User-Agent no identifica
    a un responsable (exige un mail de contacto). Reintentarlo son 12 segundos
    perdidos y, peor, esconde la causa detras de un "no disponible".
    """
    for intento in range(REINTENTOS):
        try:
            r = s.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 403:
                raise AccesoDenegado(
                    "SEC devolvio 403. Exige un User-Agent identificable con un "
                    "mail de contacto. Configurar SEC_USER_AGENT "
                    "(por ejemplo: 'tu-mail@dominio.com nombre-del-script').")
        except AccesoDenegado:
            raise
        except Exception:
            pass
        time.sleep(2 * (intento + 1))
    return None


def verificar_acceso(s):
    """
    Un pedido chico para fallar temprano y con un mensaje util si el
    User-Agent no sirve, en vez de descubrirlo a mitad de una corrida larga.
    """
    r = s.get(MAPA_TICKERS, timeout=TIMEOUT, stream=True)
    r.close()
    if r.status_code == 403:
        raise AccesoDenegado(
            "SEC devolvio 403. Exige un User-Agent identificable con un mail de "
            "contacto. Configurar SEC_USER_AGENT "
            "(por ejemplo: 'tu-mail@dominio.com nombre-del-script').")
    return r.status_code == 200


def mapa_cik(s):
    """{ticker: cik_int} desde el listado oficial de SEC."""
    r = _get(s, MAPA_TICKERS)
    if r is None:
        return {}
    return {v["ticker"]: int(v["cik_str"]) for v in r.json().values()}


def ultimo_filing(s, cik):
    """
    Ultimo 10-Q/10-K de una empresa, leyendo `submissions` (~164 KB).

    Devuelve {accn, form, filed, report_date} o None. Es lo que se compara
    contra lo ya ingestado para decidir si vale la pena bajar companyfacts.
    """
    r = _get(s, f"{BASE}/submissions/CIK{int(cik):010d}.json")
    if r is None:
        return None
    rec = r.json().get("filings", {}).get("recent", {})
    formas = rec.get("form", [])
    for i, forma in enumerate(formas):
        if forma in FORMS_PERIODICOS:
            # 'recent' viene ordenado del mas nuevo al mas viejo.
            return {"accn": rec["accessionNumber"][i], "form": forma,
                    "filed": rec["filingDate"][i],
                    "report_date": rec["reportDate"][i] or None}
    return None


def ruta_cache(destino, ticker):
    return os.path.join(destino, f"{ticker}.json")


def leer_cache(destino, ticker):
    """companyfacts del cache, o None si no esta."""
    p = ruta_cache(destino, ticker)
    if not os.path.exists(p) or os.path.getsize(p) < 1000:
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


def bajar_companyfacts(s, cik, destino, ticker):
    """
    Baja companyfacts y lo deja en el cache. Devuelve (dict, bytes) o
    (None, 0). El archivo se escribe ENTERO o no se escribe: se guarda en un
    temporal y recien despues se renombra, para que una descarga cortada no
    deje un JSON invalido que despues parezca cache valido.
    """
    r = _get(s, f"{BASE}/api/xbrl/companyfacts/CIK{int(cik):010d}.json")
    if r is None:
        return None, 0
    os.makedirs(destino, exist_ok=True)
    final = ruta_cache(destino, ticker)
    tmp = final + ".parcial"
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, final)
    try:
        return json.loads(r.content.decode("utf-8")), len(r.content)
    except ValueError:
        return None, len(r.content)


def sincronizar(pares, destino, accn_conocidos=None, forzar=False,
               s=None, pausa=PAUSA, log=None):
    """
    Deja el cache al dia para `pares` = [(ticker, cik), ...].

    accn_conocidos : {ticker: accession_del_ultimo_10Q/10K_ya_ingestado}.
                     Lo provee el llamador desde la tabla de control; este
                     modulo no consulta la base.
    forzar         : ignora el chequeo incremental y re-baja todo.

    Devuelve {ticker: {estado, accn, form, filed, bytes, error}} con estado en
    {"actualizado", "sin_cambios", "sin_cache", "error"}.

    "sin_cambios" significa: el ultimo 10-Q/10-K es el mismo que ya se
    ingesto Y el cache existe -> no se bajo nada pesado.
    """
    s = s or sesion()
    accn_conocidos = accn_conocidos or {}
    out = {}
    for ticker, cik in pares:
        info = {"estado": "error", "accn": None, "form": None, "filed": None,
                "bytes": 0, "error": None}
        try:
            filing = ultimo_filing(s, cik)
            time.sleep(pausa)
            if filing:
                info.update(accn=filing["accn"], form=filing["form"],
                            filed=filing["filed"])

            hay_cache = leer_cache(destino, ticker) is not None
            sin_novedad = (filing is not None
                           and accn_conocidos.get(ticker) == filing["accn"])
            if not forzar and sin_novedad and hay_cache:
                info["estado"] = "sin_cambios"
                out[ticker] = info
                if log:
                    log(f"  {ticker}: sin cambios ({filing['form']} {filing['filed']})")
                continue

            datos, n = bajar_companyfacts(s, cik, destino, ticker)
            time.sleep(pausa)
            info["bytes"] = n
            if datos is None:
                info["error"] = "companyfacts no disponible"
                info["estado"] = "sin_cache" if not hay_cache else "error"
            else:
                info["estado"] = "actualizado"
            if log:
                log(f"  {ticker}: {info['estado']} ({n/1e6:.1f} MB)")
        except Exception as e:                      # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"[:200]
            if log:
                log(f"  {ticker}: ERROR {info['error']}")
        out[ticker] = info
    return out
