"""
Cobertura de `earnings_historico`: que tickers DEBEN un balance y cuan al dia
esta la tabla. Modulo PURO (stdlib): sin DB, sin red, sin pandas.

POR QUE EXISTE (medido 19/9/2026). La deteccion original preguntaba
`earnings_calendar.earnings_date <= CURRENT_DATE`, pero esa tabla guarda SOLO LA
PROXIMA fecha de cada ticker: el dia que la empresa reporta, el refresh semanal
la reemplaza por la del trimestre siguiente y el ticker no vuelve a aparecer
como desactualizado NUNCA. La ventana para detectarlo era el hueco entre el
anuncio y el proximo refresh del calendario. Resultado: `--status` informaba
"Desactualizados: 0" con 108 de 200 tickers debiendo un balance y la tabla
frenada en el 3/8/2026. El incremental no es que no se corriera: corriendolo
todas las noches tampoco habria traido nada.

LA REGLA: una tabla de EVENTOS no se vigila por antiguedad absoluta (entre
temporadas de balances "vieja" es lo correcto) sino por la CADENCIA PROPIA de
cada ticker -- la mediana de dias entre sus anuncios, que es un hecho suyo y no
un supuesto nuestro. Si paso mas de esa cadencia mas un margen, debe un balance.
Mismo metodo que uso docs/fuentes_fundamentales.md para encontrar los 97 tickers
sin su ultimo balance.

FUENTE UNICA: lo importan `scripts/refresh_earnings_historico.py` (a quien
llamar) y `dashboard/earnings_reaccion.py` (que mostrarle al usuario), para que
no haya dos definiciones de "la tabla esta al dia".

Ver docs/earnings_reaccion.md.
"""

from datetime import date
from statistics import median
from typing import NamedTuple

# Tolerancia sobre la cadencia propia antes de decir "debe un balance". 15% de
# 91 dias son ~14: una empresa puede correr el anuncio un par de semanas sin que
# eso sea un dato faltante.
MARGEN = 1.15

# Cadencia supuesta cuando el ticker no tiene suficientes anuncios para medir la
# suya (recien dado de alta). 91 = mediana del universo, medida 19/9/2026.
CADENCIA_DEFAULT = 91

# Llamadas por corrida que tolera la key free de Alpha Vantage (25/dia).
MAX_CALLS_DIA = 20

# Anuncios necesarios para medir cadencia propia (2 anuncios = 1 intervalo).
MIN_ANUNCIOS = 3


class Estado(NamedTuple):
    """El estado de UN ticker."""
    ticker: str
    ultimo: date | None      # su ultimo announcement_date (None = sin historia)
    cadencia: float          # dias entre anuncios (propia, o CADENCIA_DEFAULT)
    propia: bool             # True si la cadencia se midio, False si es el default
    dias: int | None         # dias desde el ultimo anuncio
    debe: bool               # debe un balance que no tenemos


class Cobertura(NamedTuple):
    """El estado del universo entero."""
    hoy: date
    total: int
    sin_historia: list[str]
    deben: list[Estado]      # ordenados por atraso, el mas atrasado primero
    al_dia_hasta: date | None   # el anuncio mas reciente que si tenemos
    estados: dict[str, Estado]

    @property
    def pendientes(self) -> int:
        """Llamadas a la API que hacen falta para ponerse al dia."""
        return len(self.sin_historia) + len(self.deben)

    @property
    def corridas(self) -> int:
        """Corridas de MAX_CALLS_DIA que hacen falta (la cuota es diaria)."""
        return -(-self.pendientes // MAX_CALLS_DIA)

    @property
    def al_dia(self) -> bool:
        return self.pendientes == 0


def cadencia(anuncios) -> tuple[float, bool]:
    """
    Cadencia de anuncios de un ticker: mediana de dias entre anuncios
    consecutivos. Devuelve (dias, propia). `propia` es False cuando no hay
    suficiente historia y se cae al default del universo.

    Mediana y no promedio: un cambio de cierre fiscal o un anuncio adelantado
    dejan un intervalo raro, y el promedio se lo lleva puesto.
    """
    fechas = sorted(set(anuncios or ()))
    if len(fechas) < MIN_ANUNCIOS:
        return float(CADENCIA_DEFAULT), False
    gaps = [(b - a).days for a, b in zip(fechas, fechas[1:])]
    m = median(gaps)
    if m <= 0:
        return float(CADENCIA_DEFAULT), False
    return float(m), True


def estado(ticker: str, anuncios, hoy: date, margen: float = MARGEN) -> Estado:
    """El estado de un ticker a la fecha `hoy`."""
    fechas = sorted(set(anuncios or ()))
    cad, propia = cadencia(fechas)
    if not fechas:
        return Estado(ticker, None, cad, propia, None, True)
    ultimo = fechas[-1]
    dias = (hoy - ultimo).days
    return Estado(ticker, ultimo, cad, propia, dias, dias > cad * margen)


def cobertura(historia: dict, hoy: date, margen: float = MARGEN) -> Cobertura:
    """
    Estado del universo. `historia` = {ticker: [announcement_date, ...]}; un
    ticker sin filas entra igual con la lista vacia (asi el llamador no tiene
    que acordarse de sumarlos aparte).
    """
    estados = {t: estado(t, a, hoy, margen) for t, a in historia.items()}
    sin = sorted(t for t, e in estados.items() if e.ultimo is None)
    deben = sorted((e for e in estados.values() if e.debe and e.ultimo is not None),
                   key=lambda e: (-e.dias, e.ticker))
    ultimos = [e.ultimo for e in estados.values() if e.ultimo is not None]
    return Cobertura(hoy, len(estados), sin, deben,
                     max(ultimos) if ultimos else None, estados)


def a_traer(cob: Cobertura) -> list[str]:
    """
    A quien llamar, en orden: primero los que no tienen NADA (sin ellos la vista
    no dibuja nada), despues los atrasados, del mas atrasado al menos.
    """
    return list(cob.sin_historia) + [e.ticker for e in cob.deben]


def resumen(cob: Cobertura, detalle: int = 8) -> list[str]:
    """Lineas de texto para el script, el dashboard y el resumen de la rutina."""
    hasta = f"{cob.al_dia_hasta:%Y-%m-%d}" if cob.al_dia_hasta else "sin datos"
    out = [f"Ultimo anuncio en la tabla: {hasta} ({cob.total} tickers)"]
    if cob.al_dia:
        out.append("Al dia: ningun ticker debe un balance segun su propia cadencia.")
        return out
    if cob.sin_historia:
        out.append(f"Sin historia: {len(cob.sin_historia)} "
                   f"{cob.sin_historia[:detalle]}")
    if cob.deben:
        ej = ", ".join(f"{e.ticker} ({e.dias}d, cadencia {e.cadencia:.0f})"
                       for e in cob.deben[:detalle])
        out.append(f"Deben un balance: {len(cob.deben)} de {cob.total} -> {ej}"
                   + (" ..." if len(cob.deben) > detalle else ""))
    out.append(f"Faltan {cob.pendientes} llamadas = ~{cob.corridas} corrida(s) "
               f"de {MAX_CALLS_DIA} (key free: 25/dia).")
    return out
