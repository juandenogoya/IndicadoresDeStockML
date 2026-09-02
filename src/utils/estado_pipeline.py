"""
src/utils/estado_pipeline.py
Diagnostico del estado de la rutina diaria: que tablas estan al dia, cuales
quedaron atras, y que .bat hay que correr para cada caso.

PURO: sin DB, sin Streamlit, sin config. Recibe un dict {tabla: fecha} ya
consultado y devuelve el diagnostico. Los dos consumidores (la banda del
dashboard y scripts/manual/chequeo_rutina.py) arman ese dict con su propia
conexion y comparten ESTE razonamiento -- que es el punto: dos
implementaciones de "estan alineados los datos?" se desincronizan y una de las
dos empieza a mentir sin que nadie lo note.

POR QUE EXISTE
    La rutina diaria es MANUAL (4 pasos, ver docs/checklist_recovery_manual.md)
    y hasta ahora nada avisaba si faltaba uno. El 2/9/2026 no se corrio
    ft_run_diario: el crudo de opciones estaba en local pero sin derivar, y
    todo el sistema siguio andando cruzando tecnico del 1/9 con opciones del
    31/8, sin una sola queja.

LOS DOS EJES, QUE SON FALLAS DISTINTAS
    1. ANTIGUEDAD -- precios_diarios contra el ultimo dia habil cerrado.
       Que los datos sean del cierre anterior es la CONVENCION del proyecto,
       no un error: los bots FT operan asi el 73% de las veces (CLAUDE.md,
       "FT asincronico"). Por eso la antiguedad se informa, pero NO frena.

    2. MEZCLA -- cada tabla contra precios_diarios (el ancla).
       Esto si es un error: significa que una decision se computa con tecnico
       de una rueda y opciones de otra. Es el caso traicionero, porque cada
       numero por separado parece correcto.

    Un solo eje no distingue "todo viejo pero coherente" (esta bien) de
    "mezcla de ruedas" (esta mal). De ahi que sean dos.

EL CASO GRAVE, APARTE
    Si el que quedo atras es `opciones_snapshot` (el CRUDO), el problema no es
    que falte procesar: es que el snapshot no se capturo. Yahoo solo sirve la
    chain vigente, asi que apenas abre el mercado del dia siguiente ese dato es
    IRRECUPERABLE. Es el unico caso de esta lista con reloj, y necesita otro
    comando (poblar_opciones_yq.bat) antes de las 13:30 UTC. Por eso se reporta
    como una condicion propia y no mezclado con el resto.
"""

from datetime import date, datetime
from typing import NamedTuple, Optional


class Tabla(NamedTuple):
    nombre: str
    columna: str      # columna de fecha (no todas se llaman igual)
    etiqueta: str     # como se muestra
    critica: bool     # una desalineacion aca produce decisiones con datos mezclados
    arreglo: str      # que correr para ponerla al dia


# El orden es el de la rutina, para que la salida se lea como el circuito.
# `critica` = es INSUMO de una decision. Los veredictos y la equity son SALIDAS:
# que esten atrasadas es un sintoma, no una causa, y no deben frenar nada.
TABLAS = (
    Tabla("precios_diarios", "fecha", "Precios",
          True, "cron_paso1_precios_yq.bat"),
    Tabla("indicadores_tecnicos", "fecha", "Indicadores",
          True, "cron_paso1_precios_yq.bat"),
    Tabla("features_precio_accion", "fecha", "Features PA",
          True, "cron_paso2_features.bat"),
    Tabla("features_market_structure", "fecha", "Features SMC",
          True, "cron_paso2_features.bat"),
    Tabla("alertas_scanner", "scan_fecha", "Scanner ML",
          True, "cron_paso3_scanner.bat"),
    Tabla("opciones_snapshot", "fecha_snapshot", "Opciones (crudo)",
          True, "sync_opciones_railway_to_local.bat"),
    Tabla("opciones_pcr_plazo_diario", "fecha", "Opciones (PCR plazo)",
          True, "ft_run_diario.bat  [paso 0b]"),
    Tabla("opciones_sector_pcr_plazo_diario", "fecha", "Opciones (sector)",
          True, "ft_run_diario.bat  [paso 0b]"),
    Tabla("ft_equity_diaria", "fecha", "Equity FT",
          False, "ft_run_diario.bat"),
    Tabla("veredictos_universo_diario", "fecha", "Veredictos",
          False, "ft_run_diario.bat  [paso final]"),
)

ANCLA = "precios_diarios"
CRUDO_OPCIONES = "opciones_snapshot"


def _a_fecha(v) -> Optional[date]:
    """Acepta date, datetime o 'YYYY-MM-DD...' (alertas_scanner es timestamp)."""
    if v is None:
        return None
    # datetime es SUBCLASE de date, asi que el orden de los isinstance importa:
    # al reves, un timestamp caeria en la rama de date y se devolveria entero.
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        y, m, d = str(v)[:10].split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def diagnosticar(fechas: dict, esperado: date, tablas=TABLAS) -> dict:
    """
    fechas   : {nombre_tabla: fecha_maxima}. Las tablas ausentes del dict se
               omiten del diagnostico (una instalacion puede no tenerlas todas).
    esperado : ultimo dia habil NYSE ya cerrado. Lo calcula el llamador con
               src.utils.trading_calendar -- aca no se deduce, porque adivinar
               si un dia es habil es la regla que el proyecto prohibe romper.

    Returns:
        {
          ancla, esperado, atraso_ancla, ancla_al_dia,
          tablas: [{tabla, etiqueta, fecha, dias_vs_ancla, al_dia, critica, arreglo}],
          desalineadas: [etiqueta, ...],   # solo CRITICAS fuera del ancla
          mezcla: bool,                    # hay al menos una critica desalineada
          snapshot_ausente: bool,          # el crudo de opciones quedo atras
          arreglos: [comando, ...],        # sin repetir, en orden de rutina
        }
    """
    ancla = _a_fecha(fechas.get(ANCLA))

    filas = []
    desalineadas = []
    arreglos = []
    snapshot_ausente = False

    for t in tablas:
        if t.nombre not in fechas:
            continue
        f = _a_fecha(fechas.get(t.nombre))
        dias = (ancla - f).days if (f is not None and ancla is not None) else None
        # "al dia" incluye ir ADELANTE del ancla: una tabla mas fresca que los
        # precios no es una mezcla peligrosa, es un paso que ya se corrio.
        al_dia = dias is not None and dias <= 0
        filas.append({
            "tabla": t.nombre, "etiqueta": t.etiqueta, "fecha": f,
            "dias_vs_ancla": dias, "al_dia": al_dia,
            "critica": t.critica, "arreglo": t.arreglo,
        })
        if t.critica and not al_dia and t.nombre != ANCLA:
            desalineadas.append(t.etiqueta)
            if t.arreglo not in arreglos:
                arreglos.append(t.arreglo)
            if t.nombre == CRUDO_OPCIONES:
                snapshot_ausente = True

    atraso = (esperado - ancla).days if ancla is not None else None
    if ancla is not None and atraso is not None and atraso > 0:
        # El ancla atrasada no es "mezcla", pero igual hay que decir que correr.
        cmd = tablas[0].arreglo
        if cmd not in arreglos:
            arreglos.insert(0, cmd)

    return {
        "ancla": ancla,
        "esperado": esperado,
        "atraso_ancla": atraso,
        "ancla_al_dia": bool(ancla is not None and ancla >= esperado),
        "tablas": filas,
        "desalineadas": desalineadas,
        "mezcla": bool(desalineadas),
        "snapshot_ausente": snapshot_ausente,
        "arreglos": arreglos,
    }


def enumerar(items) -> str:
    """'A' | 'A y B' | 'A, B y C'. Un mensaje de estado lo lee alguien apurado:
    una lista separada por comas se lee como enumeracion truncada."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} y {items[-1]}"


def resumen(diag: dict) -> str:
    """Una linea con el veredicto. Es lo que se muestra en la banda del
    dashboard y lo que imprime el chequeo de la rutina."""
    if diag["ancla"] is None:
        return "precios_diarios esta vacia: no hay datos que diagnosticar."

    if diag["snapshot_ausente"]:
        return (f"El snapshot de opciones no llego a local (ultimo "
                f"{_fecha_de(diag, CRUDO_OPCIONES)}, precios al {diag['ancla']}). "
                f"OJO: la chain de opciones es IRRECUPERABLE una vez que abre el "
                f"mercado siguiente.")

    if diag["mezcla"]:
        return (f"Cierre {diag['ancla']}, pero {enumerar(diag['desalineadas'])} "
                f"{'quedo' if len(diag['desalineadas']) == 1 else 'quedaron'} "
                f"atras: lo que se calcule ahora cruza ruedas distintas.")

    if not diag["ancla_al_dia"]:
        n = diag["atraso_ancla"]
        return (f"Todo coherente al {diag['ancla']}, pero son {n} "
                f"{'rueda' if n == 1 else 'ruedas'} atras del ultimo cierre "
                f"({diag['esperado']}).")

    return f"Todo al dia y alineado al {diag['ancla']}."


def _fecha_de(diag: dict, tabla: str):
    for f in diag["tablas"]:
        if f["tabla"] == tabla:
            return f["fecha"]
    return None
