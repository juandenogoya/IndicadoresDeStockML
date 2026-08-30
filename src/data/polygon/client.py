"""
client.py -- acceso a api.polygon.io con control de caudal.

Sin DB y sin imports del proyecto (ver la regla en __init__.py): stdlib +
requests. El llamador decide QUE pedir; este modulo se ocupa de pedirlo sin
que nos corten.

--------------------------------------------------------------------------
EL CAUDAL ES EL PROBLEMA DE DISENO, NO UN DETALLE
--------------------------------------------------------------------------
Medido en este plan: una rafaga de pedidos sin pausa recibe 429 en el SEXTO,
a los 1,9 segundos. O sea 5 por minuto. No es una pausa fija entre pedidos:
es una VENTANA. Cinco pedidos instantaneos pasan; el sexto no, aunque hayan
pasado 20 segundos desde el primero.

Por eso el limitador es un registro de marcas de tiempo y no un `sleep(12)`:
antes de cada pedido mira cuantos hubo en los ultimos 60 segundos y, si ya
son 5, espera hasta que el mas viejo salga de la ventana. Con `sleep` fijo se
desperdicia caudal al principio y se pega contra el limite igual despues.

CONSECUENCIA OPERATIVA: todo lo que use este cliente tarda horas, no minutos.
Los 200 tickers de splits son ~40 minutos. Por eso el llamador tiene que ser
REANUDABLE -- que lo maten a la mitad no puede costar la mitad del trabajo.

La env var se llama POLYGON_APY_KEY (sic, con el typo). Se acepta tambien
POLYGON_API_KEY para que renombrarla algun dia no rompa nada.
"""

import os
import time

BASE = "https://api.polygon.io"

# Pedidos permitidos por ventana, y el largo de la ventana en segundos. Se
# deja un pedido de margen (4 y no 5): el reloj del servidor no es el nuestro
# y rozar el limite es como nos cortaron la primera vez.
CUPO = 4
VENTANA = 60.0

REINTENTOS = 4
TIMEOUT = 45


def api_key():
    return os.getenv("POLYGON_APY_KEY") or os.getenv("POLYGON_API_KEY")


class SinCredencial(RuntimeError):
    pass


class Caudal:
    """
    Ventana deslizante de pedidos. No es un `sleep` fijo: ver el encabezado.

    `espera_total` acumula los segundos perdidos esperando, para que el
    llamador pueda informar cuanto del tiempo de corrida fue caudal y cuanto
    fue trabajo real.
    """

    def __init__(self, cupo=CUPO, ventana=VENTANA):
        self.cupo, self.ventana = cupo, ventana
        self.marcas = []
        self.espera_total = 0.0

    def esperar(self):
        ahora = time.time()
        self.marcas = [m for m in self.marcas if ahora - m < self.ventana]
        if len(self.marcas) >= self.cupo:
            dormir = self.ventana - (ahora - self.marcas[0]) + 0.25
            if dormir > 0:
                time.sleep(dormir)
                self.espera_total += dormir
        self.marcas.append(time.time())


class Cliente:
    """
    Sesion + caudal. La sesion se reutiliza a proposito (keep-alive): abrir
    una conexion nueva por pedido es lo que nos hizo cortar con SEC, y no hay
    razon para repetir el experimento aca.
    """

    def __init__(self, key=None, cupo=CUPO, ventana=VENTANA):
        import requests
        self.key = key or api_key()
        if not self.key:
            raise SinCredencial(
                "falta POLYGON_APY_KEY en el .env (o POLYGON_API_KEY)")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": "Bearer " + self.key,
                               "Accept": "application/json"})
        self.caudal = Caudal(cupo, ventana)
        self.n_pedidos = 0
        self.n_429 = 0

    def get(self, ruta, **params):
        """
        GET con control de caudal y reintentos. Devuelve el JSON ya parseado.

        Un 404 devuelve None y NO es un error: Polygon contesta 404 cuando no
        conoce el ticker, que para nosotros es informacion (ver la pregunta de
        los 53 sin SEC), no una falla.
        """
        url = ruta if ruta.startswith("http") else BASE + ruta
        demora = 5.0
        for intento in range(1, REINTENTOS + 1):
            self.caudal.esperar()
            self.n_pedidos += 1
            try:
                r = self.s.get(url, params=params, timeout=TIMEOUT)
            except Exception:
                if intento == REINTENTOS:
                    raise
                time.sleep(demora); demora *= 2
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                # El cupo se nos escapo igual. Se vacia la ventana para que el
                # proximo pedido arranque de cero en vez de insistir.
                self.n_429 += 1
                self.caudal.marcas = [time.time()] * self.caudal.cupo
                time.sleep(demora); demora *= 2
                continue
            if r.status_code >= 500:
                if intento == REINTENTOS:
                    r.raise_for_status()
                time.sleep(demora); demora *= 2
                continue
            r.raise_for_status()
            return r.json()
        return None


# ------------------------------------------------------------- endpoints --
def detalles(cli, ticker, fecha=None):
    """
    /v3/reference/tickers/{ticker}. Con `fecha` devuelve la foto de ESE dia.

    Dos campos de acciones, y confundirlos arruina el dato:
      share_class_shares_outstanding -- SOLO la clase de ese ticker
      weighted_shares_outstanding    -- el TOTAL de todas las clases

    En un filer de una sola clase son casi iguales (MSFT); en uno multi-clase
    la brecha ES la clase B/C: HSY da 145.430.000 contra 206.003.728, y V da
    1.635.020.000 contra 2.107.412.746. Para el market cap va el TOTAL.

    OJO: el conteo viene en la BASE DE SU MOMENTO, no en la de hoy. GOOG da
    663.763.994 en 2021-06-30 contra 12.229.934.831 hoy, que es su split 20:1.
    Quien lo empalme con precios_diarios (que si esta en base de hoy) tiene
    que rebasarlo con el historial de splits.
    """
    d = cli.get("/v3/reference/tickers/%s" % ticker,
                **({"date": fecha} if fecha else {}))
    return (d or {}).get("results")


def splits(cli, ticker, desde=None):
    """
    /v3/reference/splits de UN ticker.

    El filtro `ticker.in` de la API NO funciona -- probado: devuelve tickers
    sin relacion (fondos mutuos). Hay que pedirlos de a uno.

    `execution_date` es la fecha de EJECUCION, que puede diferir uno o dos
    dias de la ex-date que registra el proyecto: KLAC figura 2026-06-12 y en
    CLAUDE.md quedo anotado el 11/6; CRWD 2026-07-02 contra el 30/6. En un
    rebase el dia del corte define de que lado cae cada rueda, asi que la
    diferencia no es cosmetica.
    """
    p = {"ticker": ticker, "limit": 100}
    if desde:
        p["execution_date.gte"] = desde
    d = cli.get("/v3/reference/splits", **p)
    return (d or {}).get("results") or []
