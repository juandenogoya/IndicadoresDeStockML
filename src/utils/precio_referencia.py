"""
precio_referencia.py
Precio de REFERENCIA del subyacente para los calculos de opciones en LOCAL.

Modulo PURO (stdlib): sin DB, sin config, sin side effects.

Un solo duenio por dato (docs/arquitectura_fuentes.md sec. 2): el cierre del dia D
de un ticker es `precios_diarios.close`. `opciones_snapshot.precio_subyacente` es
lo que devolvio yahooquery al CAPTURAR la chain en la nube, donde precios_diarios
no existe. Queda como metadato crudo de la captura y solo TAPA EL HUECO cuando el
duenio no tiene esa rueda (dia no cargado, captura fuera de horario).

Por que (medido el 10/9/2026):
  - 2026-09-09: yahooquery .price devolvio 0 de 200 en la captura ->
    precio_subyacente NULL en el 100% del crudo -> sin zona de busqueda -> muros
    de OI vacios en todo el universo, aunque el close del dia estaba en
    precios_diarios.
  - Antes del 11/5 precio_subyacente salia de un precios_diarios CONGELADO en
    Railway: 1.658 pares ticker-fecha difieren mas de 1% del close real.
  - Cuando los dos existen y estan en la misma escala coinciden (max 0,055% en 4
    ruedas x 200 tickers).

ESCALA DE SPLIT
  El close de precios_diarios esta en la escala de HOY: ante un split, splits.py
  corrige la historia hacia atras por divisor. Los strikes de una cadena vieja,
  en cambio, estan en la escala de SU dia. Contra esos strikes el close se usa
  multiplicado por los splits REALES ejecutados DESPUES de la rueda:
      precio = close(D) x producto(ratio de splits con fecha > D)
  con la fuente FUENTE_DIARIO_ESCALA. El valor sigue saliendo del duenio; solo
  cambia la escala.

  El factor sale del REGISTRO `splits_aplicados`, no de la captura. Primer
  intento, descartado: deducir la escala del ratio captura/close cuando es un
  split exacto. Funcionaba con captura fresca y fallaba con captura RANCIA (antes
  del 11/5 KLAC daba x9,17..x10,70 y quedaba sin reconocer). Validado con el
  registro: captura / (close x factor) = 1,0000 exacto en 81 ruedas frescas de
  KLAC (10:1, 12/6) y 81 de CRWD (4:1, 2/7).

  Una fila del registro = "la historia de ese ticker en precios_diarios ya esta en
  la escala posterior a ese split". La escribe `scripts/manual/splits.py corregir`
  en la MISMA transaccion que la correccion, con la fecha de EJECUCION que informa
  Yahoo (elegir_evento_split), no la del corte en la DB. Por eso el factor es
  coherente en todo momento: antes de corregir, la historia sigue en la escala
  vieja (igual que los strikes) y no lleva factor; despues lo aporta el registro;
  y no puede aplicarse dos veces. Hasta el 12/9/2026 salia de `polygon_splits`,
  congelado desde el 30/8: el proximo split no habria entrado, y un split listado
  pero todavia sin corregir se habria escalado dos veces.

  Solo cuentan ratios >= SPLIT_RATIO_MIN (o su inverso). Las listas de splits
  mezclan ajustes chicos que precios_diarios NO refleja: con los dividendos en
  acciones de SCCO (1,01 y 1,012) de polygon aplicados, el cruce empeora a 0,988;
  y los ajustes por spinoff (HON 1,061) no son splits de precio.

  El ratio captura/close sigue sirviendo como VALIDADOR: si difieren por un split
  exacto que el registro no explica (escalas_sin_registro), falta corregir el
  split con splits.py, o se corrigio por fuera sin registrarlo.

LIMITE CONOCIDO: una correccion de precios_diarios hecha por fuera de splits.py
(SQL a mano) no queda en el registro; el validador la avisa solo en las ruedas
cuya captura trae precio.

La regla de la CAPTURA no cambia: 33_opciones_snapshot.py sigue tomando el precio
de yahooquery (en la nube no hay precios_diarios). Lo que cambia es QUIEN MANDA al
calcular en local. La fuente usada viaja en la columna `precio_fuente` de las
tablas derivadas (regla 1 de convivencia: cada tabla declara su fuente).
"""

import math

FUENTE_DIARIO = "precios_diarios"          # close del dia (sin splits posteriores)
FUENTE_DIARIO_ESCALA = "precios_x_split"   # close del dia llevado a la escala de ese dia
FUENTE_SNAPSHOT = "snapshot"               # la captura tapa el hueco del close

# Split REAL: ratio >= 1,5 o <= 1/1,5. Por debajo son ajustes (dividendo en
# acciones, spinoff) que precios_diarios no refleja -- ver docstring del modulo.
SPLIT_RATIO_MIN = 1.5

# Por encima de esta diferencia, el close (en escala) y el precio de la captura
# no hablan de lo mismo (captura rancia, ajuste no reflejado): se avisa.
TOL_DIVERGENCIA = 0.01

# Validador: ratios de split plausibles y su tolerancia (el mismo cierre en dos
# escalas da un ratio exacto, no aproximado).
RATIOS_SPLIT = (2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50)
TOL_RATIO_SPLIT = 0.01

# Una captura que trae precio para menos de esta fraccion de los tickers es una
# anomalia de la fuente (el 09-09 fue 0 de 200), no un par de tickers raros.
COBERTURA_MIN = 0.90

# Registro de splits: el evento de Yahoo que corresponde a una correccion cae
# entre la fecha de corte de la DB y esta cantidad de dias corridos despues. El
# corte se adelanta a la ejecucion tantas ruedas como tardo en bajarse la data
# (rutina manual: hasta 6 ruedas de atraso medidas en FT). Ver elegir_evento_split.
VENTANA_EVENTO_DIAS = 15


def _valido(valor):
    """float > 0 o None. Acepta Decimal (lo que devuelve la DB) y descarta NaN."""
    if valor is None:
        return None
    try:
        f = float(valor)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f <= 0:
        return None
    return f


# ── Escala ────────────────────────────────────────────────────────────────────

def es_split_real(ratio):
    """True si el ratio es un split de precio (>= SPLIT_RATIO_MIN o su inverso)."""
    r = _valido(ratio)
    return r is not None and (r >= SPLIT_RATIO_MIN or r <= 1.0 / SPLIT_RATIO_MIN)


def factor_escala(splits, fecha, hasta=None):
    """
    Factor para llevar el close de `fecha` (escala de hoy) a la escala de ESE dia.

    Args:
        splits: iterable de (execution_date, ratio) de UN ticker.
        fecha:  rueda a expresar. Un split con execution_date == fecha ya rige
                ese dia (se aplica a la apertura), asi que no cuenta.
        hasta:  si se da, ignora splits posteriores (no ejecutados / no reflejados).

    Returns:
        producto de los ratios de splits reales con fecha < execution_date [<= hasta].
    """
    f = 1.0
    for ejecucion, ratio in splits:
        if ejecucion <= fecha or (hasta is not None and ejecucion > hasta):
            continue
        if es_split_real(ratio):
            f *= float(ratio)
    return f


def factores_por_ticker(filas, fecha, hasta=None):
    """
    Args:
        filas: iterable de (ticker, execution_date, ratio).

    Returns:
        {ticker: factor} solo para los tickers con factor distinto de 1.
    """
    por_ticker = {}
    for ticker, ejecucion, ratio in filas:
        por_ticker.setdefault(ticker, []).append((ejecucion, ratio))
    out = {}
    for ticker, splits in por_ticker.items():
        f = factor_escala(splits, fecha, hasta)
        if f != 1.0:
            out[ticker] = f
    return out


def elegir_evento_split(eventos, ratio, fecha_corte, ventana_dias=VENTANA_EVENTO_DIAS,
                        tol=TOL_RATIO_SPLIT):
    """
    Evento de split (de Yahoo) que corresponde a una correccion de splits.py.

    `fecha_corte` es la primera rueda que YA estaba en la escala nueva en
    precios_diarios. NO es la fecha de mercado: depende de cuando se bajo cada
    rueda. Una rueda anterior al split bajada DESPUES de el llega ajustada, asi
    que el corte cae en la ejecucion o antes, nunca despues:
        KLAC  corte 2026-06-11, ejecucion 2026-06-12
        CRWD  corte 2026-06-30, ejecucion 2026-07-02
    Registrar el corte como fecha del split dejaria mal escaladas las cadenas de
    opciones de esas ruedas: su captura seguia en la escala vieja.

    Args:
        eventos:     iterable de (fecha, ratio) del ticker.
        ratio:       ratio de la correccion (10 = split 10:1, 0.5 = inverso 1:2).
        fecha_corte: primera rueda en escala nueva segun precios_diarios.

    Returns:
        (fecha, ratio) del evento con ese ratio (dentro de `tol`) y fecha en
        [fecha_corte, fecha_corte + ventana_dias]; el mas cercano al corte si hay
        varios. None si ninguno encaja: sin evento no se registra ni se corrige.
    """
    r = _valido(ratio)
    if r is None or fecha_corte is None:
        return None
    candidatos = []
    for fecha, ev_ratio in eventos:
        e = _valido(ev_ratio)
        if e is None or abs(e / r - 1.0) > tol:
            continue
        dias = (fecha - fecha_corte).days
        if 0 <= dias <= ventana_dias:
            candidatos.append((dias, fecha, e))
    if not candidatos:
        return None
    _, fecha, e = min(candidatos)
    return fecha, e


# ── Resolucion ────────────────────────────────────────────────────────────────

def resolver_precio(close_diario, precio_snapshot, factor=1.0):
    """
    Precio de referencia de UN ticker en una rueda.

    Returns:
        (precio, fuente):
          - hay close -> close x factor (FUENTE_DIARIO, o FUENTE_DIARIO_ESCALA si
            el factor no es 1);
          - solo hay captura -> la captura (FUENTE_SNAPSHOT), que ya esta en la
            escala de su dia;
          - ninguno valido -> (None, None).
    """
    c = _valido(close_diario)
    if c is not None:
        f = _valido(factor) or 1.0
        if f != 1.0:
            return c * f, FUENTE_DIARIO_ESCALA
        return c, FUENTE_DIARIO
    s = _valido(precio_snapshot)
    if s is not None:
        return s, FUENTE_SNAPSHOT
    return None, None


def resolver_mapa(closes, snapshots, tickers=None, factores=None):
    """
    Precio de referencia para varios tickers.

    Args:
        closes:    {ticker: close de precios_diarios en la rueda}
        snapshots: {ticker: precio_subyacente de la captura}
        tickers:   iterable a resolver (default: la union de ambos mapas)
        factores:  {ticker: factor de escala} (default: ninguno)

    Returns:
        {ticker: (precio, fuente)} -- incluye (None, None) para los sin precio.
    """
    factores = factores or {}
    if tickers is None:
        tickers = set(closes) | set(snapshots)
    return {t: resolver_precio(closes.get(t), snapshots.get(t), factores.get(t, 1.0))
            for t in tickers}


def contar_fuentes(resueltos):
    """Reparto por fuente de un resolver_mapa(), con 'sin_precio' para los vacios."""
    out = {FUENTE_DIARIO: 0, FUENTE_DIARIO_ESCALA: 0, FUENTE_SNAPSHOT: 0, "sin_precio": 0}
    for _, fuente in resueltos.values():
        out[fuente if fuente else "sin_precio"] += 1
    return out


# ── Validacion ────────────────────────────────────────────────────────────────

def ratio_de_split(base, otro, tol=TOL_RATIO_SPLIT):
    """
    Ratio otro/base si coincide con un split plausible dentro de `tol`.

    Returns:
        el ratio plausible (>1 forward, <1 reverso) o None.
    """
    b, o = _valido(base), _valido(otro)
    if b is None or o is None:
        return None
    r = o / b
    for k in RATIOS_SPLIT:
        for candidato in (float(k), 1.0 / k):
            if abs(r / candidato - 1.0) <= tol:
                return candidato
    return None


def _pares_en_escala(closes, snapshots, factores):
    """(ticker, close x factor, captura) donde existen los dos."""
    factores = factores or {}
    for t, c_raw in closes.items():
        c = _valido(c_raw)
        s = _valido(snapshots.get(t))
        if c is None or s is None:
            continue
        yield t, c * (_valido(factores.get(t, 1.0)) or 1.0), s


def escalas_sin_registro(closes, snapshots, factores=None):
    """
    Tickers donde la captura y el close (ya en escala) difieren por un split
    EXACTO: el registro de splits no lo explica (split sin registrar, o
    registrado pero sin corregir en precios_diarios).

    Returns:
        [(ticker, close_en_escala, captura, ratio)] ordenado por ticker.
    """
    out = []
    for t, c, s in _pares_en_escala(closes, snapshots, factores):
        k = ratio_de_split(c, s)
        if k is not None:
            out.append((t, c, s, k))
    return sorted(out)


def medir_divergencias(closes, snapshots, factores=None, tol=TOL_DIVERGENCIA):
    """
    Tickers donde la captura difiere del close EN ESCALA mas de `tol`, sin contar
    las escalas sin registro (esas las reporta escalas_sin_registro).

    Returns:
        [(ticker, close_en_escala, captura, dif)] ordenado por dif descendente,
        con dif = |captura / close_en_escala - 1| (fraccion).
    """
    out = []
    for t, c, s in _pares_en_escala(closes, snapshots, factores):
        if ratio_de_split(c, s) is not None:
            continue
        dif = abs(s / c - 1.0)
        if dif > tol:
            out.append((t, c, s, dif))
    out.sort(key=lambda x: x[3], reverse=True)
    return out


def cobertura_baja(n_con_precio, n_tickers, umbral=COBERTURA_MIN):
    """True si la captura trajo precio para menos de `umbral` de los tickers."""
    if not n_tickers:
        return False
    return (n_con_precio / n_tickers) < umbral
