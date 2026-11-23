"""
exportar_catalogo.py
Arma el Excel con TODO lo que el proyecto calcula y como estan programadas las
estrategias, para poder experimentar con parametros, scores y umbrales.

    python scripts/manual/exportar_catalogo.py
    python scripts/manual/exportar_catalogo.py --salida ruta/archivo.xlsx
    python scripts/manual/exportar_catalogo.py --check   # no escribe nada

HOJAS
    Indicadores   las ~77 columnas que se calculan: sigla, nombre, que mide, de
                  que tabla sale. Incluye las que tienen DEFECTO conocido y las
                  que se calculan y NO se usan (candidatas a experimentar).
    Estrategias   una fila por instancia de ft_estrategias, con la pregunta que
                  vino a contestar, el concepto, entrada, salida y sizing.
    Scores        como se arma cada numero que decide (tecnico, SMC, scanner,
                  ponderado, pcr_score, EV1-EV4), condicion por condicion.
    Metodo        las reglas MEDIDAS para no engancharse con una mejora que es
                  ruido. Leer antes de la primera grilla.
    <ESTRATEGIA>  una hoja por instancia: un renglon por insumo de la decision,
                  con los parametros que lo gobiernan y su VALOR real.

DE DONDE SALEN LOS DATOS
    Los PARAMETROS salen de `ft_estrategias` (LOCAL), que es lo que de verdad
    corrio -- no de la lista del .py que las sembro. El script compara las dos y
    AVISA si difieren: una deriva ahi significa que la tabla y el codigo no
    cuentan lo mismo.
    Las DESCRIPCIONES salen de src/utils/catalogo_tecnico.py (modulo de datos).

El .xlsx es DERIVADO y gitignoreado (reportes/ y *.xlsx): se regenera cuando
haga falta. Lo que se versiona es el generador y el catalogo.
"""

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# FT corre 100% en LOCAL: sin esto, get_engine() levanta DATABASE_URL de
# .env.local y leeria Railway, donde ft_estrategias no existe.
from scripts.forward_testing.ft_env import configurar_entorno_local
configurar_entorno_local()

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sqlalchemy import text

from src.data.database import get_engine
from src.utils import catalogo_tecnico as cat

SALIDA_DEFAULT = os.path.join(ROOT, "reportes", "catalogo_indicadores_estrategias.xlsx")

# -- Estilos -------------------------------------------------------------------
AZUL = "1F4E79"
GRIS = "D9D9D9"
AMBAR = "FFF2CC"
VERDE = "E2EFDA"
ROJO = "FCE4E4"

F_TITULO = Font(bold=True, size=14, color="FFFFFF")
F_CABECERA = Font(bold=True, size=10, color="FFFFFF")
F_GRUPO = Font(bold=True, size=10)
F_ETIQUETA = Font(bold=True, size=10)
F_NORMAL = Font(size=10)
F_MONO = Font(name="Consolas", size=9)

R_TITULO = PatternFill("solid", fgColor=AZUL)
R_CABECERA = PatternFill("solid", fgColor=AZUL)
R_GRUPO = PatternFill("solid", fgColor=GRIS)
R_NOTA = PatternFill("solid", fgColor=AMBAR)
R_ALTA = PatternFill("solid", fgColor=VERDE)
R_BAJA = PatternFill("solid", fgColor=ROJO)

ARRIBA = Alignment(vertical="top", wrap_text=True)
ARRIBA_IZQ = Alignment(vertical="top", horizontal="left", wrap_text=True)
BORDE = Border(bottom=Side(style="thin", color="BFBFBF"))


def _cabecera(ws, fila, titulos, anchos):
    """Escribe la fila de encabezado de una tabla y fija los anchos."""
    for i, (titulo, ancho) in enumerate(zip(titulos, anchos), start=1):
        c = ws.cell(row=fila, column=i, value=titulo)
        c.font = F_CABECERA
        c.fill = R_CABECERA
        c.alignment = ARRIBA_IZQ
        ws.column_dimensions[get_column_letter(i)].width = ancho
    ws.row_dimensions[fila].height = 28
    ws.freeze_panes = ws.cell(row=fila + 1, column=1)


def _titulo(ws, texto, ancho_merge):
    """Banda de titulo en la fila 1."""
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ancho_merge)
    c = ws.cell(row=1, column=1, value=texto)
    c.font = F_TITULO
    c.fill = R_TITULO
    c.alignment = Alignment(vertical="center", horizontal="left")
    ws.row_dimensions[1].height = 24


def _fila(ws, fila, valores, fuente=F_NORMAL):
    for i, v in enumerate(valores, start=1):
        c = ws.cell(row=fila, column=i, value=v)
        c.font = fuente
        c.alignment = ARRIBA
        c.border = BORDE


def _banda_grupo(ws, fila, texto, ancho):
    ws.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=ancho)
    c = ws.cell(row=fila, column=1, value=texto)
    c.font = F_GRUPO
    c.fill = R_GRUPO
    c.alignment = ARRIBA_IZQ


def _fmt(valor):
    """Un valor de parametro, legible en una celda."""
    if isinstance(valor, bool):
        return "SI" if valor else "NO"
    if isinstance(valor, (list, tuple)):
        return ", ".join(str(v) for v in valor)
    if isinstance(valor, float):
        # Las fracciones se leen mal con %g (0.10 -> "0,1"), asi que las de
        # abajo de 1 van con 2 decimales. 4.0 -> "4", 2.5 -> "2,5".
        if 0 < abs(valor) < 1 and round(abs(valor), 2) > 0:
            return ("%.2f" % valor).replace(".", ",")
        return ("%g" % valor).replace(".", ",")
    return str(valor)


# -- Lectura de la DB ----------------------------------------------------------

def leer_estrategias():
    """ft_estrategias (LOCAL) + el motivo de baja y la deriva contra el seed."""
    from scripts.forward_testing.ft_setup_estrategias import ESTRATEGIAS as SEED

    seed = {e["nombre"]: e for e in SEED}

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, nombre, descripcion, logica, activa, fecha_inicio, parametros
            FROM ft_estrategias
            ORDER BY id
        """)).fetchall()

    out = []
    for r in rows:
        params = r.parametros if isinstance(r.parametros, dict) else json.loads(r.parametros or "{}")
        s = seed.get(r.nombre, {})
        p_seed = s.get("parametros", {})
        difs = sorted(k for k in set(params) | set(p_seed)
                      if _fmt(params.get(k, "<falta>")) != _fmt(p_seed.get(k, "<falta>")))
        out.append({
            "id": r.id,
            "nombre": r.nombre,
            "descripcion": r.descripcion or "",
            "logica": r.logica,
            "activa": bool(r.activa),
            "fecha_inicio": r.fecha_inicio,
            "parametros": params,
            "baja": s.get("discontinuada", ""),
            "en_seed": bool(s),
            "deriva": difs,
        })
    return out


# -- Hojas ---------------------------------------------------------------------

def hoja_indicadores(wb):
    ws = wb.create_sheet("Indicadores")
    _titulo(ws, "INDICADORES, OSCILADORES Y MEDIAS QUE EL PROYECTO CALCULA", 4)
    _cabecera(ws, 2, ["Sigla / columna", "Nombre completo",
                      "Funcion: que mide y donde decide",
                      "Donde vive (tabla / modulo)"],
              [30, 38, 92, 46])

    fila = 3
    i = 0
    for titulo, n in cat.GRUPOS:
        _banda_grupo(ws, fila, titulo, 4)
        fila += 1
        for ind in cat.INDICADORES[i:i + n]:
            _fila(ws, fila, [ind.sigla, ind.nombre, ind.funcion, ind.origen])
            fila += 1
        i += n

    fila += 1
    _banda_grupo(ws, fila, "COMO LEER ESTA HOJA", 4)
    fila += 1
    for nota in (
        "Las que dicen 'se calcula y NO se usa' (ADX, Bandas de Bollinger, OBV, "
        "clv, ad_flow, chaikin_mf_20, vol_ratio_5d, rango_rel_atr, gap_apertura_pct, "
        "impulso_N) son las candidatas mas limpias para experimentar: ya estan "
        "calculadas y ninguna decision depende de ellas hoy.",
        "Las que dicen MAL DEFINIDA o DEFECTUOSA estan medidas asi en "
        "docs/estructura_velas.md sec. 12. Siguen en uso a proposito donde se "
        "aclara (FT_SMC_v1 es el control), pero no son base para nada nuevo.",
        "features_market_structure es la excepcion importante: PROHIBIDA para "
        "entrenar, validar o backtestear. Su equivalente honesto es "
        "features_estructura.",
        "Los periodos de calculo viven en src/utils/config.py. Cambiarlos ahi "
        "cambia la tabla entera para todos los consumidores a la vez: no es una "
        "palanca de experimento, es una migracion.",
    ):
        ws.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=4)
        c = ws.cell(row=fila, column=1, value=nota)
        c.font = F_NORMAL
        c.fill = R_NOTA
        c.alignment = ARRIBA
        ws.row_dimensions[fila].height = 30
        fila += 1
    return ws


def hoja_estrategias(wb, estrategias):
    ws = wb.create_sheet("Estrategias")
    _titulo(ws, "ESTRATEGIAS: QUE PREGUNTA VINO A CONTESTAR CADA UNA", 10)
    _cabecera(ws, 2, ["Estrategia", "Estado", "Logica (familia)",
                      "La pregunta que contesta",
                      "Concepto", "Entrada", "Salida", "Sizing", "Insumos",
                      "Desde"],
              [26, 10, 26, 46, 60, 46, 46, 32, 36, 12])

    fila = 3
    for e in estrategias:
        lg = cat.LOGICAS.get(e["logica"])
        estado = "ACTIVA" if e["activa"] else "BAJA"
        _fila(ws, fila, [
            e["nombre"], estado, lg.titulo if lg else e["logica"],
            lg.pregunta if lg else "", lg.concepto if lg else e["descripcion"],
            lg.entrada if lg else "", lg.salida if lg else "",
            lg.sizing if lg else "", lg.insumos if lg else "",
            str(e["fecha_inicio"]),
        ])
        ws.cell(row=fila, column=2).fill = R_ALTA if e["activa"] else R_BAJA
        ws.cell(row=fila, column=2).font = F_ETIQUETA
        fila += 1

    fila += 1
    _banda_grupo(ws, fila, "ESTADO Y CIERRES", 10)
    fila += 1
    for e in estrategias:
        lg = cat.LOGICAS.get(e["logica"])
        txt = e["baja"] or (lg.estado if lg else "")
        ws.cell(row=fila, column=1, value=e["nombre"]).font = F_ETIQUETA
        ws.merge_cells(start_row=fila, start_column=2, end_row=fila, end_column=10)
        c = ws.cell(row=fila, column=2, value=txt)
        c.font = F_NORMAL
        c.alignment = ARRIBA
        if not e["activa"]:
            c.fill = R_BAJA
        fila += 1

    fila += 1
    _banda_grupo(ws, fila, "VARIAS INSTANCIAS COMPARTEN LOGICA: AHI ESTA EL CONTROL", 10)
    fila += 1
    for nota in (
        "Las parejas que comparten logica son experimentos con control: "
        "FT_ML_SCANNER_v1 y v2 solo difieren en el modelo; FT_SMC_v3_N5 y N3 solo "
        "en la ventana de confirmacion; OPTIONS_v1 y v2 solo en OI contra VOL. "
        "Cualquier diferencia de resultado entre ellas es atribuible a ESA unica "
        "cosa, y por eso la vieja nunca se apaga.",
        "FT_TECH_SECTOR_v1 es la referencia del proyecto: TECH_SECTOR_v2, "
        "OPTIONS_v1/v2, OIEXIT_v1 y COMBO_v1 son todas esa misma estrategia con "
        "una pata cambiada. Si se toca su entrada, se mueve el piso de "
        "comparacion de cinco experimentos a la vez.",
    ):
        ws.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=10)
        c = ws.cell(row=fila, column=1, value=nota)
        c.font = F_NORMAL
        c.fill = R_NOTA
        c.alignment = ARRIBA
        ws.row_dimensions[fila].height = 42
        fila += 1
    return ws


def hoja_scores(wb):
    ws = wb.create_sheet("Scores")
    _titulo(ws, "COMO SE ARMA CADA NUMERO QUE DECIDE", 4)
    _cabecera(ws, 2, ["Capa / componente", "Condicion", "Suma",
                      "Nota: por que importa"],
              [24, 62, 14, 86])

    fila = 3
    for nombre, filas in cat.SCORES.items():
        _banda_grupo(ws, fila, nombre, 4)
        fila += 1
        for f in filas:
            _fila(ws, fila, [f.capa, f.condicion, f.puntos, f.nota])
            if f.capa == "TOTAL" or f.capa == "NIVELES" or f.capa == "USO":
                for col in range(1, 5):
                    ws.cell(row=fila, column=col).font = F_ETIQUETA
            fila += 1
        fila += 1
    return ws


def hoja_metodo(wb):
    ws = wb.create_sheet("Metodo")
    _titulo(ws, "REGLAS MEDIDAS PARA EXPERIMENTAR SIN ENGANCHARSE", 2)
    _cabecera(ws, 2, ["Regla", "De donde salio (medicion de este proyecto)"],
              [52, 124])
    fila = 3
    for regla, detalle in cat.METODO:
        _fila(ws, fila, [regla, detalle])
        ws.cell(row=fila, column=1).font = F_ETIQUETA
        fila += 1
    return ws


def hoja_estrategia(wb, e):
    """Una hoja por instancia: un renglon por insumo, con sus parametros."""
    nombre_hoja = e["nombre"][:31]
    ws = wb.create_sheet(nombre_hoja)
    lg = cat.LOGICAS.get(e["logica"])
    filas_uso = cat.USO.get(e["logica"], ())
    params = e["parametros"]

    # Cuantas columnas de parametro hacen falta
    max_p = max([len(f.params) for f in filas_uso] or [1])
    n_col = 1 + max_p + 1                      # indicador + parametros + rol
    col_rol = n_col

    estado = "ACTIVA" if e["activa"] else "DADA DE BAJA"
    _titulo(ws, f"{e['nombre']}  --  {estado}  --  "
                f"{lg.titulo if lg else e['logica']}", n_col)

    # -- Encabezado descriptivo --
    fila = 2
    datos = [
        ("Logica (ft_estrategias.logica)", e["logica"]),
        ("Instancia", f"id {e['id']}, desde {e['fecha_inicio']}"),
        ("DONDE SE CAMBIA UN PARAMETRO",
         cat.BOTS.get(e["nombre"], "(sin mapeo de bot en el catalogo)")),
        ("Logica COMPARTIDA (toca a varias)",
         cat.CEREBRO.get(e["logica"], "")),
        ("La pregunta que contesta", lg.pregunta if lg else ""),
        ("Concepto", lg.concepto if lg else e["descripcion"]),
        ("Entrada", lg.entrada if lg else ""),
        ("Salida", lg.salida if lg else ""),
        ("Sizing", lg.sizing if lg else ""),
        ("Insumos", lg.insumos if lg else ""),
        ("Estado", e["baja"] or (lg.estado if lg else "")),
    ]
    for etiqueta, valor in datos:
        ws.cell(row=fila, column=1, value=etiqueta).font = F_ETIQUETA
        ws.cell(row=fila, column=1).alignment = ARRIBA
        ws.merge_cells(start_row=fila, start_column=2, end_row=fila, end_column=n_col)
        c = ws.cell(row=fila, column=2, value=valor)
        c.font = F_NORMAL
        c.alignment = ARRIBA
        if etiqueta == "Estado" and not e["activa"]:
            c.fill = R_BAJA
        if etiqueta == "DONDE SE CAMBIA UN PARAMETRO":
            c.font = F_MONO
        fila += 1

    # La advertencia que vale para las 13: la tabla es un registro.
    ws.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=n_col)
    c = ws.cell(row=fila, column=1, value=cat.ADVERTENCIA_PARAMETROS)
    c.font = F_ETIQUETA
    c.fill = R_NOTA
    c.alignment = ARRIBA
    ws.row_dimensions[fila].height = 58
    fila += 1

    if e["deriva"]:
        ws.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=n_col)
        c = ws.cell(row=fila, column=1,
                    value="AVISO: estos parametros DIFIEREN entre ft_estrategias "
                          "(lo que corrio) y ft_setup_estrategias.py (lo que "
                          "declara el codigo): " + ", ".join(e["deriva"]) +
                          ". Los valores de abajo son los de la TABLA.")
        c.font = F_ETIQUETA
        c.fill = R_NOTA
        c.alignment = ARRIBA
        fila += 1

    fila += 1

    # -- La tabla: un renglon por insumo --
    titulos = (["Indicador / insumo que forma parte"]
               + [f"Parametro {i}" for i in range(1, max_p + 1)]
               + ["Rol en la decision y por que importa"])
    anchos = [38] + [30] * max_p + [86]
    _cabecera(ws, fila, titulos, anchos)
    ws.freeze_panes = ws.cell(row=fila + 1, column=2)
    fila += 1

    usados = set()
    for f in filas_uso:
        valores = [f.indicador]
        for clave in f.params:
            usados.add(clave)
            if clave in params:
                valores.append(f"{clave} = {_fmt(params[clave])}")
            else:
                # No lo declara la instancia: el umbral existe en otro lado, y
                # decir "no definido" seria falso. El motivo va en el glosario.
                valores.append(f"{clave} = NO PARAMETRIZADO (*)")
        valores += [""] * (max_p - len(f.params))
        valores.append(f.rol)
        _fila(ws, fila, valores)
        for col in range(2, 2 + len(f.params)):
            ws.cell(row=fila, column=col).font = F_MONO
        fila += 1

    # -- Nada se pierde: parametros de la tabla que no mapeo ninguna fila --
    sobrantes = [k for k in params if k not in usados]
    if sobrantes:
        fila += 1
        _banda_grupo(ws, fila, "OTROS PARAMETROS DE LA INSTANCIA "
                               "(no gobiernan un indicador en particular)", n_col)
        fila += 1
        for clave in sobrantes:
            ws.cell(row=fila, column=1,
                    value=f"{clave} = {_fmt(params[clave])}").font = F_MONO
            ws.cell(row=fila, column=1).alignment = ARRIBA
            ws.merge_cells(start_row=fila, start_column=2, end_row=fila,
                           end_column=n_col)
            c = ws.cell(row=fila, column=2,
                        value=cat.PARAMETROS.get(clave, "(sin descripcion en el "
                                                        "catalogo)"))
            c.font = F_NORMAL
            c.alignment = ARRIBA
            fila += 1

    # -- Glosario de los parametros que si aparecen arriba --
    fila += 1
    _banda_grupo(ws, fila, "QUE SIGNIFICA CADA PARAMETRO DE ESTA HOJA", n_col)
    fila += 1
    for clave in sorted(usados):
        ws.cell(row=fila, column=1, value=clave).font = F_MONO
        ws.cell(row=fila, column=1).alignment = ARRIBA
        ws.merge_cells(start_row=fila, start_column=2, end_row=fila, end_column=n_col)
        if clave in params:
            txt = cat.PARAMETROS.get(clave, "(sin descripcion en el catalogo)")
        else:
            txt = ("(*) NO PARAMETRIZADO EN ESTA INSTANCIA. "
                   + cat.NO_PARAMETRIZADO.get(
                       clave, "No lo declara y el catalogo no dice por que: "
                              "revisar."))
        c = ws.cell(row=fila, column=2, value=txt)
        c.font = F_NORMAL
        c.alignment = ARRIBA
        if clave not in params:
            c.fill = R_NOTA
        fila += 1

    ws.sheet_properties.tabColor = VERDE if e["activa"] else ROJO
    return ws


def hoja_portada(wb, estrategias, salida):
    ws = wb.create_sheet("Leeme", 0)
    _titulo(ws, "CATALOGO DE INDICADORES Y ESTRATEGIAS", 2)
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 130

    activas = sum(1 for e in estrategias if e["activa"])
    deriva = [e["nombre"] for e in estrategias if e["deriva"]]

    bloques = [
        ("Generado", f"{date.today().isoformat()} por "
                     f"scripts/manual/exportar_catalogo.py"),
        ("Para que", "Saber que se esta calculando y como estan programadas las "
                     "estrategias, para empezar a experimentar con parametros, "
                     "scores y umbrales."),
        ("Contenido", f"{len(cat.INDICADORES)} indicadores en "
                      f"{len(cat.GRUPOS)} grupos; {len(estrategias)} instancias "
                      f"de estrategia ({activas} activas, "
                      f"{len(estrategias) - activas} dadas de baja); "
                      f"{len(cat.SCORES)} scores desarmados condicion por "
                      f"condicion."),
        ("Hoja Indicadores", "Que mide cada columna, con que umbral y de que "
                             "tabla sale. Dice tambien cuales tienen defecto "
                             "conocido y cuales se calculan y NO se usan."),
        ("Hoja Estrategias", "Una fila por instancia: la pregunta que vino a "
                             "contestar, el concepto, entrada, salida y sizing. "
                             "Abajo, el estado y el motivo de cada baja."),
        ("Hoja Scores", "Como se arma cada numero que decide, condicion por "
                        "condicion y punto por punto."),
        ("Hoja Metodo", "LEER ANTES DE LA PRIMERA GRILLA. Las reglas que salieron "
                        "de mediciones de este proyecto y que evitan confundir "
                        "ruido con mejora."),
        ("Una hoja por estrategia", "Un renglon por insumo de la decision, los "
                                    "parametros que lo gobiernan con su VALOR "
                                    "real, y para que sirve cada uno."),
        ("Fuente de los parametros", "ft_estrategias de la DB LOCAL: lo que de "
                                     "verdad corrio, no lo que declara el "
                                     "codigo. El script compara las dos cosas y "
                                     "avisa si difieren."),
        ("ANTES DE TOCAR UN PARAMETRO", cat.ADVERTENCIA_PARAMETROS),
        ("Deriva detectada",
         ", ".join(deriva) if deriva else "ninguna: la tabla y el codigo dicen "
                                          "lo mismo en las 13 instancias."),
        ("Este archivo es DERIVADO", "Esta gitignoreado y se regenera con el "
                                     "script. Si un valor de aca no coincide con "
                                     "el codigo, manda el codigo: avisar y "
                                     "corregir src/utils/catalogo_tecnico.py."),
        ("Lo que NO esta", "Fundamentales, perfiles de riesgo y el detalle de las "
                           "53 features del ML: ninguna estrategia FT los "
                           "consume hoy. Ver docs/features_ml.md y "
                           "docs/fundamentales_calculo.md."),
    ]
    fila = 2
    for etiqueta, valor in bloques:
        ws.cell(row=fila, column=1, value=etiqueta).font = F_ETIQUETA
        ws.cell(row=fila, column=1).alignment = ARRIBA
        c = ws.cell(row=fila, column=2, value=valor)
        c.font = F_NORMAL
        c.alignment = ARRIBA
        if etiqueta in ("Hoja Metodo", "ANTES DE TOCAR UN PARAMETRO"):
            c.fill = R_NOTA
        if etiqueta == "Deriva detectada" and deriva:
            c.fill = R_NOTA
        fila += 1
    ws.sheet_view.showGridLines = False
    return ws


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default=SALIDA_DEFAULT,
                    help="ruta del .xlsx (default: reportes/)")
    ap.add_argument("--check", action="store_true",
                    help="valida el catalogo contra la DB y NO escribe el archivo")
    args = ap.parse_args()

    print("  Leyendo ft_estrategias (LOCAL)...")
    estrategias = leer_estrategias()
    print(f"  {len(estrategias)} instancias "
          f"({sum(1 for e in estrategias if e['activa'])} activas).")

    # -- Avisos de coherencia: que el catalogo no se quede atras --
    problemas = 0
    sin_logica = sorted({e["logica"] for e in estrategias} - set(cat.LOGICAS))
    if sin_logica:
        print(f"  [AVISO] logicas en la DB sin entrada en el catalogo: {sin_logica}")
        problemas += 1
    sin_uso = sorted({e["logica"] for e in estrategias} - set(cat.USO))
    if sin_uso:
        print(f"  [AVISO] logicas sin mapa de insumos: {sin_uso}")
        problemas += 1
    for e in estrategias:
        faltan = sorted(set(e["parametros"]) - set(cat.PARAMETROS))
        if faltan:
            print(f"  [AVISO] {e['nombre']}: parametros sin descripcion: {faltan}")
            problemas += 1
        if e["nombre"] not in cat.BOTS:
            print(f"  [AVISO] {e['nombre']}: sin mapeo al .py que lo implementa")
            problemas += 1
        # Claves que una fila pide y la instancia no declara: tienen que tener
        # explicacion, o la planilla miente diciendo NO PARAMETRIZADO sin motivo.
        pedidas = {p for f in cat.USO.get(e["logica"], ()) for p in f.params}
        sin_motivo = sorted(pedidas - set(e["parametros"]) - set(cat.NO_PARAMETRIZADO))
        if sin_motivo:
            print(f"  [AVISO] {e['nombre']}: no declara {sin_motivo} y el "
                  f"catalogo no explica por que")
            problemas += 1
        if not e["en_seed"]:
            print(f"  [AVISO] {e['nombre']} esta en la DB y NO en "
                  f"ft_setup_estrategias.py")
            problemas += 1
        if e["deriva"]:
            print(f"  [DERIVA] {e['nombre']}: la tabla y el seed difieren en "
                  f"{e['deriva']}")
            problemas += 1

    if args.check:
        print(f"\n  CHECK: {problemas} aviso(s). No se escribio ningun archivo.")
        return 2 if problemas else 0

    print("  Armando el libro...")
    wb = Workbook()
    wb.remove(wb.active)
    hoja_indicadores(wb)
    hoja_estrategias(wb, estrategias)
    hoja_scores(wb)
    hoja_metodo(wb)
    for e in estrategias:
        hoja_estrategia(wb, e)
    hoja_portada(wb, estrategias, args.salida)
    wb.active = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.salida)), exist_ok=True)
    wb.save(args.salida)
    print(f"\n  OK -> {args.salida}")
    print(f"  {len(wb.sheetnames)} hojas: {', '.join(wb.sheetnames)}")
    if problemas:
        print(f"  ({problemas} aviso(s) arriba: el archivo se escribio igual.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
