"""
tags_curados.py -- mapeo CURADO de ticker -> tag XBRL, por concepto.

Modulo de DATOS. Sin logica, sin DB, sin red. Lo consume
`src/utils/sec_xbrl.normalizar(..., tags_curados=...)`.

Por que hace falta curar a mano
-------------------------------
"Revenue" no es un renglon en XBRL. La SEC no publica estados contables:
publica hechos sueltos, y una misma empresa puede tagear su top line con dos
conceptos que NO valen lo mismo. El normalizador elige por prioridad de
sinonimo, y esa eleccion puede resolver distinto en los tres 10-Q que en el
10-K -- de donde sale el Q4. El resultado es una MEZCLA DE TAGS dentro del
mismo ejercicio: los 4 trimestres dejan de medir la misma magnitud y su suma
no es el anual de nada.

Medido sobre los 147 tickers USA, con la suma de los 4 trimestres contra el
anual que publico la propia empresa:

    net_income        99,6%        cfo               97,0%
    operating_income  98,6%        revenue           87,8%

23 tickers tienen ambiguedad real. Ningun algoritmo la resuelve: hay CINCO
salidas automaticas ya probadas y descartadas con sus numeros en
docs/fuentes_fundamentales.md seccion 15 -- entre ellas la mas tentadora,
Q4 = FY - (Q1+Q2+Q3), que haria pasar el control POR CONSTRUCCION y
convertiria un error visible en uno invisible.

Elegir entre dos tags es una decision CONTABLE, y por eso vive en un mapeo
curado y no en una heuristica. Es el mismo patron que el override de perfil
banco/no-banco de docs/fundamentales_calculo.md.

Como se decidio cada entrada
----------------------------
  arbitraje  el anual de ese tag coincide con la suma de los 4 trimestres de
             yahooquery, que es una fuente independiente. Es la evidencia mas
             fuerte disponible y gana cuando existe.
  criterio   yahooquery no tiene los 4 trimestres del ejercicio (sus filas
             stub con total_revenue en NULL, que es justamente uno de los
             defectos que motivaron traer SEC). Se elige por convencion del
             estado contable, explicada caso por caso.

Alpha Vantage se probo como tercer arbitro y NO sirve: para WFC da 125.397 MM
en 2024 y 77.198 MM en 2023 -- cambia de base a mitad de la serie, que es el
mismo problema semantico que se esta tratando de resolver.

Como mantenerlo
---------------
El aviso `mezcla_en_ejercicio` de sec_xbrl.py detecta el defecto solo. Si
entra un ticker nuevo al universo o una empresa cambia de taxonomia, el aviso
aparece y esta tabla se actualiza. NO hace falta re-auditar a mano: el
diagnostico se regenera con

    python scripts/oneshot/revenue_tags_reporte.py

Un valor puede ser un tag (str) o una lista ordenada de tags, para las
empresas que migraron de taxonomia de verdad y ninguna sola cubre la ventana.
"""

# Los cuatro tags que aparecen en las decisiones de abajo, por legibilidad.
_RFCC_EX = "RevenueFromContractWithCustomerExcludingAssessedTax"
_RFCC_IN = "RevenueFromContractWithCustomerIncludingAssessedTax"
_NETO_INT = "RevenuesNetOfInterestExpense"
_TOTAL = "Revenues"

# ---------------------------------------------------------------------------
# revenue -- los 23 tickers con ambiguedad real (universo de 147 USA)
# ---------------------------------------------------------------------------
# Las cifras entre parentesis son el ejercicio 2025 en millones, para que se
# vea CONTRA QUE se decidio y se pueda revisar sin volver a correr nada.
REVENUE = {
    # -- arbitrado contra yahooquery ---------------------------------------
    # Revenues 10.645 es el total; el otro tag es un renglon chico de 936.
    "AMT":  _TOTAL,
    # Ingresos netos de intereses 72.229, que es el titular de una emisora de
    # tarjetas. El bruto de intereses y dividendos (25.598) es solo una parte.
    "AXP":  _NETO_INT,
    # BAC llama Revenues a su ingreso NETO (113.097). InterestAndDividend...
    # es el bruto de intereses (138.566) y no es la linea de resultados.
    "BAC":  _TOTAL,
    # Ventas totales de la comercializadora 70.329; el otro tag cubre solo la
    # porcion con contratos con clientes (16.944).
    "BG":   _TOTAL,
    "BLK":  _RFCC_EX,
    # Ventas y otros ingresos operativos 184.432, NO "total revenues and other
    # income" (189.031), que suma resultados de inversiones y ventas de
    # activos. Para un P/S el denominador tiene que ser ventas.
    "CVX":  _RFCC_EX,
    "FCX":  _TOTAL,
    # Ingresos netos de intereses 58.283, el titular de un banco de inversion.
    "GS":   _NETO_INT,
    # Neto de impuestos internos: el tabaco cobra y remite impuestos que no
    # son ingreso propio. 40.648 excluyendolos.
    "PM":   _RFCC_EX,
    "UPST": _TOTAL,

    # -- por criterio, sin arbitro externo ----------------------------------
    # BANCOS Y SEGUROS. El titular de un banco es el ingreso NETO de
    # intereses, no el bruto: el bruto cuenta como ingreso plata que se paga
    # como costo de fondeo. Usarlo infla el denominador de un P/S entre 40% y
    # 70% y hace incomparable al sector consigo mismo.
    "C":    _TOTAL,        # Citi tagea su ingreso neto como Revenues (85.225)
    "MS":   _NETO_INT,     # 70.645 vs 59.063 de bruto de intereses
    "WFC":  _NETO_INT,     # 83.699 vs 87.314 de bruto
    "PGR":  _TOTAL,        # 87.671 total; el otro tag es solo la renta de
                           # inversiones (3.583), una linea del estado
    "MA":   _TOTAL,        # 32.791, unico candidato en los ejercicios nuevos

    # INDUSTRIA Y CONSUMO CON BRAZO FINANCIERO. Aca el tag mayor SI es el
    # titular: la actividad de financiacion de GM Financial y de HDFS es un
    # segmento operativo, no un resultado no operativo. Es la diferencia con
    # el caso de las petroleras de abajo.
    "GM":   _TOTAL,        # 185.019 incluye GM Financial; 167.971 lo excluye
    "HOG":  _TOTAL,        # 4.473 incluye HDFS; 3.604 lo excluye

    # ENERGIA. Mismo criterio que CVX, que si tiene arbitraje: la linea de
    # ventas, no "total revenues and other income", que suma resultados por
    # participaciones y ventas de activos.
    "COP":  _RFCC_EX,      # 51.824 ventas vs 58.944 con otros ingresos
    "OXY":  _RFCC_EX,      # 21.569 vs 21.593, dos decimas de diferencia
    "VST":  _TOTAL,        # 17.738 vs 17.586; la utility no tiene el corte
                           # ventas/otros que tienen las petroleras

    # RESTO
    "LYFT": _TOTAL,        # 6.316 total vs 5.895
    "PFE":  _TOTAL,        # 62.579, unico candidato en los ejercicios nuevos
    "RBLX": _RFCC_EX,      # 4.891, unico candidato
}


# Un solo diccionario por ahora. La estructura admite mas conceptos sin tocar
# a los consumidores: si aparece ambiguedad en operating_income o en cfo, se
# agrega su propio dict y se suma aca.
POR_CONCEPTO = {"revenue": REVENUE}


def para(ticker):
    """
    {concepto: tag} para un ticker, en el formato que espera
    sec_xbrl.normalizar(tags_curados=...). Devuelve {} si no hay nada curado,
    que es el caso de 124 de los 147 tickers.
    """
    if not ticker:
        return {}
    t = ticker.upper()
    return {concepto: mapa[t] for concepto, mapa in POR_CONCEPTO.items()
            if t in mapa}
