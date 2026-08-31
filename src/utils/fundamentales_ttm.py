"""
fundamentales_ttm.py
Capa DERIVADA de la serie trimestral SEC: TTM rodante, magnitudes derivadas y
busqueda as-of point-in-time.

Funcion PURA: sin DB, sin red, sin config. Solo stdlib. ASCII en strings.
Consume la salida de src/utils/sec_xbrl.normalizar()["periodos"] (o las filas
equivalentes leidas de fundamentales_sec_q) y no sabe de donde vinieron.

Las tres cosas que hace, y por que cada una es su propio problema:

1. TTM RODANTE (`serie_ttm`)
   Suma los ultimos 4 trimestres de cada concepto de flujo. La parte no trivial
   NO es la suma: es negarse a sumar cuando la ventana no es un ano.
   `rolling(4).sum()` sobre una serie con un hueco suma 4 trimestres que abarcan
   5 o 6 -- el resultado es plausible y esta mal, en silencio. Aca la ventana
   exige que los 4 periodos sean CONSECUTIVOS en la serie y que el tramo
   period_end[i-3] -> period_end[i] mida ~9 meses (VENTANA_DIAS). Un hueco
   estira ese tramo a ~12 meses y la ventana se descarta.
   Ademas: si a UN concepto le falta el dato en UNO de los 4 trimestres, el TTM
   de ESE concepto es None. No se suman 3 de 4 (seria un ano corto disfrazado
   de ano completo, que es el mismo error de escala pero mas dificil de ver).

2. QUE SE SUMA Y QUE NO
   - Aditivos (revenue, cfo, net_income...): se suman.
   - eps_diluted / eps_basic: SE SUMAN. Contraintuitivo, pero el EPS acumulado
     que publica la empresa es NI_acum / acciones_promedio_acum, que en la
     practica coincide con la suma de los EPS trimestrales (verificado sobre
     AAPL FY2025: anual 7.46 vs 5.62 de 9M + 1.85 del Q4). Tratarlo como
     promedio ponderado da numeros absurdos (12.98 en ese mismo caso).
   - shares_diluted / shares_basic: NO se suman (son promedios del trimestre).
     Se toma el del ultimo trimestre de la ventana.
   - Instantes (assets, equity, cash, deuda, shares_out): NO se suman. Son una
     foto; se toma la del ultimo trimestre.

3. AS-OF POINT-IN-TIME (`indice_asof` / `asof`)
   Para una fecha D devuelve el TTM que estaba PUBLICO en D, usando
   `filed_primero` (no `period_end`). Un trimestre cerrado el 27/9 no estuvo
   disponible el 27/9: AAPL publica a los ~34 dias, JPM entre 31 y 44. Indexar
   por period_end adelantaria cada trimestre mas de un mes y meteria lookahead
   en toda serie historica de multiplos.
"""

from bisect import bisect_right

from src.utils.sec_xbrl import FLUJO_ADITIVO, INSTANTE

# Conceptos que se ACUMULAN en la ventana TTM.
# eps_* entra aca a proposito (ver nota 2 del docstring).
TTM_SUMA = tuple(FLUJO_ADITIVO) + ("eps_diluted", "eps_basic")

# Conceptos que NO se acumulan: se toma el valor del ultimo Q de la ventana.
# Los promedios ponderados de acciones y todos los instantes del balance.
TTM_ULTIMO = ("shares_diluted", "shares_basic") + tuple(INSTANTE)

# Cuantos trimestres tiene un ano.
VENTANA_Q = 4

# Distancia en dias entre el cierre del PRIMER y el ULTIMO trimestre de una
# ventana de 4: son 3 saltos trimestrales, ~273 dias. El rango tolera
# calendarios fiscales de 52/53 semanas y cierres moviles. Un hueco de un
# trimestre lleva el tramo a ~365 y queda afuera, que es el punto.
VENTANA_DIAS = (240, 310)


def _f(v):
    """float o None. Descarta NaN (NaN != NaN)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _dias(desde, hasta):
    """Dias entre dos fechas ISO 'YYYY-MM-DD' (o date). None si alguna falta."""
    if desde is None or hasta is None:
        return None
    a, b = _fecha(desde), _fecha(hasta)
    if a is None or b is None:
        return None
    return (b - a).days


def _fecha(v):
    """Normaliza a datetime.date. Acepta date o str ISO."""
    if v is None:
        return None
    if hasattr(v, "year") and hasattr(v, "month"):
        return v
    try:
        from datetime import date as _d
        p = str(v)[:10].split("-")
        return _d(int(p[0]), int(p[1]), int(p[2]))
    except (ValueError, IndexError):
        return None


def _iso(v):
    """Normaliza a str ISO 'YYYY-MM-DD'. None si no se puede."""
    f = _fecha(v)
    return f.isoformat() if f is not None else None


# ---------------------------------------------------------------------------
# TTM rodante
# ---------------------------------------------------------------------------

def serie_ttm(periodos, conceptos=None):
    """
    periodos : lista de filas con al menos period_end y los conceptos. Se
               ordena por period_end (no se asume orden de entrada).
    conceptos: subconjunto a calcular. None = todos.

    Devuelve una lista de filas nuevas (no muta la entrada), una por periodo,
    con:
      period_end, fiscal_year, fiscal_quarter, filed_primero, filed_ultimo
      <concepto>_ttm  para los de TTM_SUMA   (None si la ventana no cierra)
      <concepto>      para los de TTM_ULTIMO (valor del Q, sin acumular)
      ventana_ok      bool: si los 4 trimestres formaban un ano valido
      ventana_desde   period_end del primer Q de la ventana (None si no cierra)
      n_periodos      cuantos periodos previos habia (diagnostico)

    Las filas sin ventana completa SE DEVUELVEN igual, con los *_ttm en None:
    el consumidor tiene que poder distinguir "no hay dato" de "no hay fila".
    """
    filas = sorted((p for p in periodos if p.get("period_end")),
                   key=lambda p: _iso(p["period_end"]))
    suma = [c for c in TTM_SUMA if conceptos is None or c in conceptos]
    ultimo = [c for c in TTM_ULTIMO if conceptos is None or c in conceptos]

    salida = []
    for i, fila in enumerate(filas):
        out = {
            "period_end": _iso(fila["period_end"]),
            "fiscal_year": fila.get("fiscal_year"),
            "fiscal_quarter": fila.get("fiscal_quarter"),
            "filed_primero": _iso(fila.get("filed_primero")),
            "filed_ultimo": _iso(fila.get("filed_ultimo")),
            "n_periodos": i + 1,
        }
        # Los no acumulables salen siempre: son del trimestre, no de la ventana.
        for c in ultimo:
            out[c] = _f(fila.get(c))

        ventana = filas[i - VENTANA_Q + 1:i + 1] if i + 1 >= VENTANA_Q else []
        ok = bool(ventana) and _ventana_valida(ventana)
        out["ventana_ok"] = ok
        out["ventana_desde"] = _iso(ventana[0]["period_end"]) if ok else None

        for c in suma:
            out[c + "_ttm"] = _sumar(ventana, c) if ok else None
        salida.append(out)
    return salida


def _ventana_valida(ventana):
    """
    Los 4 periodos consecutivos de la serie tienen que abarcar un ano. Se mide
    del cierre del primero al cierre del ultimo: 3 saltos trimestrales.
    """
    if len(ventana) != VENTANA_Q:
        return False
    d = _dias(ventana[0]["period_end"], ventana[-1]["period_end"])
    if d is None:
        return False
    return VENTANA_DIAS[0] <= d <= VENTANA_DIAS[1]


def _sumar(ventana, concepto):
    """
    Suma el concepto sobre la ventana. None si FALTA en alguno de los 4:
    sumar 3 de 4 devuelve un ano corto con cara de ano completo.
    """
    total = 0.0
    for p in ventana:
        v = _f(p.get(concepto))
        if v is None:
            return None
        total += v
    return total


# ---------------------------------------------------------------------------
# Magnitudes derivadas
# ---------------------------------------------------------------------------

# Cuantos trimestres se puede arrastrar un componente de deuda ausente antes de
# considerarlo vencido. La deuda es un STOCK, no un flujo: el ultimo valor
# conocido sigue siendo la mejor estimacion disponible durante un tiempo. Pero
# solo durante un tiempo.
#
# El 4 (un anio) sale de medir, no de la intuicion. Sobre 2.631 pares de
# trimestres consecutivos, la deuda total se mueve una mediana de 2,73% de un
# trimestre al siguiente, con p90 de 20%. Eso sonaria prohibitivo, pero lo que
# importa no es el error de la deuda sino cuanto llega al EV, y eso lo escala el
# PESO de la deuda. Midiendo el error del EV al arrastrar SOLO el componente
# ausente, en los 15 tickers afectados: mediana 0,05% y 11 de 15 por debajo de
# 0,30%. En seis de ellos (KLAC, LEVI, NEM, ORCL, RIVN, RKLB) el componente que
# falta vale CERO en su ultimo dato: arrastrarlo no aproxima nada.
#
# Los expuestos de verdad son pocos y grandes: TMUS 6,2% (arrastra 86.282 MM de
# deuda larga), DE 3,1%, CVX 1,9%. Con el tope de 4, DE (17 trimestres de
# atraso) y ORCL (16) quedan AFUERA, que es lo correcto: un valor de hace
# cuatro anios no es una estimacion, es una suposicion.
MAX_ARRASTRE_Q = 4


def derivados(fila, deudas_conocidas=None, arrastre=None):
    """
    Magnitudes que no vienen tageadas y hay que armar. Sobre una fila de
    serie_ttm(). Devuelve un dict; cada clave None si le falta un insumo.

      ebitda_ttm  = operating_income_ttm + d_and_a_ttm
                    EBIT + D&A. Es la definicion directa. No excluye one-offs
                    (yahooquery si, con NormalizedEBITDA) -> un Q con
                    impairment lo deforma. Queda declarado, no corregido.
      fcf_ttm     = cfo_ttm - capex_ttm
                    capex viene POSITIVO en XBRL (es un 'Payments...'), asi
                    que se resta.
      net_debt    = debt_short + debt_long - cash
                    Si NINGUNA de las dos deudas esta tageada -> None, no 0.
                    Ausencia en XBRL no es prueba de que no haya deuda; darla
                    por cero convierte "no se" en "cero" y lo propaga al EV.
                    Si falta UNA sola, depende de `deudas_conocidas`: cuenta
                    como 0 solo si la empresa NO la tagea en NINGUN periodo de
                    su serie. Ver la nota de enriquecer().
      shares      = shares_out, y si falta, shares_diluted.
                    shares_out sale de la PORTADA del filing (acciones en
                    circulacion a esa fecha) -- es lo correcto para market cap.
                    shares_diluted es un promedio del trimestre: sirve de
                    respaldo, no es lo mismo.
      bvps        = equity / shares
      eps_ttm     = eps_diluted_ttm, y si falta, eps_basic_ttm
    """
    oi = _f(fila.get("operating_income_ttm"))
    da = _f(fila.get("d_and_a_ttm"))
    ebitda = oi + da if (oi is not None and da is not None) else None

    cfo = _f(fila.get("cfo_ttm"))
    capex = _f(fila.get("capex_ttm"))
    fcf = cfo - capex if (cfo is not None and capex is not None) else None

    ds, dl, cash = _f(fila.get("debt_short")), _f(fila.get("debt_long")), _f(fila.get("cash"))
    conocidas = deudas_conocidas or ()

    # ARRASTRE del componente ausente. Antes, si la empresa tageaba una deuda
    # en algun periodo pero no en ESTE, el EV se apagaba entero. La guarda es
    # correcta -- sumar deuda parcial fue el defecto que dio a AT&T caja neta
    # -- pero era mas dura de lo necesario: en 14 de los 15 tickers afectados
    # el trimestre actual SI trae uno de los dos componentes, y el que falta
    # esta publicado uno o dos trimestres antes. Se arrastra ese, no se asume
    # cero, y la edad viaja en el resultado para que el consumidor la vea.
    arr = arrastre or {}
    edad_q = 0
    for nombre, actual in (("debt_short", ds), ("debt_long", dl)):
        if actual is not None or nombre not in conocidas:
            continue
        prev = arr.get(nombre)
        if prev is None:
            continue
        valor, edad = prev
        if edad > MAX_ARRASTRE_Q:
            continue
        if nombre == "debt_short":
            ds = valor
        else:
            dl = valor
        edad_q = max(edad_q, edad)

    falta_esperada = ((ds is None and "debt_short" in conocidas) or
                      (dl is None and "debt_long" in conocidas))
    net_debt = None
    if cash is not None and (ds is not None or dl is not None) and not falta_esperada:
        net_debt = (ds or 0.0) + (dl or 0.0) - cash
    else:
        edad_q = None

    shares = _f(fila.get("shares_out"))
    fuente_shares = "shares_out"
    if shares is None or shares <= 0:
        shares = _f(fila.get("shares_diluted"))
        fuente_shares = "shares_diluted" if shares else None

    equity = _f(fila.get("equity"))
    bvps = (equity / shares) if (equity is not None and shares and shares > 0) else None

    eps = _f(fila.get("eps_diluted_ttm"))
    if eps is None:
        eps = _f(fila.get("eps_basic_ttm"))

    return {
        "ebitda_ttm": ebitda,
        "fcf_ttm": fcf,
        "net_debt": net_debt,
        # Trimestres de antiguedad del componente de deuda mas viejo que se uso.
        # 0 = todo del periodo; None = no hay net_debt. Se expone en vez de
        # esconderse: un dato arrastrado sigue siendo un dato, uno arrastrado
        # EN SILENCIO es una trampa.
        "net_debt_q": edad_q if net_debt is not None else None,
        "shares": shares,
        "shares_fuente": fuente_shares,
        "book_value_per_share": bvps,
        "eps_ttm": eps,
    }


def enriquecer(filas):
    """
    serie_ttm() + derivados() en una pasada. Devuelve filas nuevas.

    Antes de derivar mira la serie ENTERA para saber que deudas tagea la
    empresa. Hace falta porque "no esta en este trimestre" y "esta empresa no
    tiene esa deuda" son cosas distintas y se ven iguales en una fila sola.

    El caso real: la API de companyfacts DESCARTA los hechos dimensionados, y
    varias empresas grandes pasaron a declarar su deuda de largo plazo solo con
    dimensiones. Verizon tiene 8 hechos de LongTermDebt y ninguno desde 2025;
    AT&T igual. Con la regla vieja -- "si falta una, cuenta 0" -- el net_debt
    de VZ salia 19.479 MM en vez de ~150.000 MM, y el de T salia NEGATIVO, como
    si AT&T tuviera caja neta. Eso se propagaba al EV y al EV/EBITDA sin que
    nada avisara: 52 de 147 tickers y 41.820 filas con deuda neta negativa.

    Ahora, si la empresa tagea esa deuda en ALGUN periodo pero no en este, no
    se asume cero: net_debt queda en None y el EV desaparece. Es la misma regla
    que usa sec_xbrl para los minoritarios del resultado neto. Se pierde
    cobertura de EV/EBITDA a cambio de que lo que quede sea cierto.

    Una empresa realmente sin deuda de largo plazo nunca tagea el concepto, asi
    que sigue contando como cero y conserva su EV.
    """
    conocidas = {c for c in ("debt_short", "debt_long")
                 if any(f.get(c) is not None for f in filas)}

    # Se recorre en orden de period_end para poder arrastrar hacia ADELANTE el
    # ultimo valor conocido de cada componente. Nunca hacia atras: eso seria
    # lookahead, meterle a un trimestre un dato que todavia no existia.
    orden = sorted(range(len(filas)),
                   key=lambda i: _iso(filas[i].get("period_end")) or "")
    ultimo = {}                      # componente -> (valor, indice en `orden`)
    derivado_por_i = {}
    for pos, i in enumerate(orden):
        f = filas[i]
        arrastre = {c: (v, pos - p) for c, (v, p) in ultimo.items()}
        derivado_por_i[i] = derivados(f, conocidas, arrastre)
        for c in ("debt_short", "debt_long"):
            if f.get(c) is not None:
                ultimo[c] = (_f(f[c]), pos)

    out = []
    for i, f in enumerate(filas):
        nueva = dict(f)
        nueva.update(derivado_por_i[i])
        out.append(nueva)
    return out


# ---------------------------------------------------------------------------
# As-of point-in-time
# ---------------------------------------------------------------------------

def indice_asof(filas):
    """
    Prepara la busqueda as-of. Devuelve (claves, filas) ordenados por
    filed_primero ascendente.

    Descarta las filas sin filed_primero: sin fecha de publicacion no se puede
    afirmar que ese dato estuviera disponible, y meterlo con period_end seria
    justo el lookahead que esto viene a evitar.

    Empates (dos trimestres publicados el mismo dia, tipico del 10-K que sale
    junto al Q4): gana el de period_end mayor, que es el orden natural del
    desempate por la clave compuesta.
    """
    ordenadas = sorted(
        (f for f in filas if f.get("filed_primero")),
        key=lambda f: (_iso(f["filed_primero"]), _iso(f.get("period_end")) or ""))
    return [_iso(f["filed_primero"]) for f in ordenadas], ordenadas


def asof(indice, fecha):
    """
    Ultima fila publicada en o antes de `fecha`. None si en esa fecha no habia
    ninguna todavia.

    indice: la tupla que devuelve indice_asof().
    """
    claves, filas = indice
    f = _iso(fecha)
    if f is None or not claves:
        return None
    i = bisect_right(claves, f)
    return filas[i - 1] if i > 0 else None


def recorrer_asof(filas, fechas):
    """
    Aplica asof() a una lista de fechas ORDENADA, en una sola pasada.
    Devuelve [(fecha, fila_o_None), ...]. Equivalente a llamar asof() por
    fecha, pero lineal en vez de N*log(M).
    """
    claves, orden = indice_asof(filas)
    salida, j, actual = [], 0, None
    for fecha in fechas:
        f = _iso(fecha)
        while j < len(claves) and claves[j] <= f:
            actual = orden[j]
            j += 1
        salida.append((f, actual))
    return salida
