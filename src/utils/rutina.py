"""
src/utils/rutina.py
La rutina diaria como UNA secuencia: orden de los pasos, que hacer si falla
cada uno y como se resume el resultado. Modulo PURO: sin DB, red, archivos ni
subprocesos. Lo ejecuta scripts/manual/rutina_diaria.py.

POR QUE EXISTE (Etapa 2, 13/9/2026)
    La rutina eran 5 .bat corridos a mano, en orden, y los pasos 1-3 no dejaban
    log ni registro de cuando corrieron. Al revisarla aparecieron tres fallas:
      - cron_paso2_features.bat abortaba apenas terminaba el calculo: un
        parentesis sin escapar dentro de un bloque ELSE ("No se esperaba a en
        este momento"). Sin estado final, sin pausa, codigo de error perdido.
      - recovery_incremental.bat decia "RECOVERY COMPLETO" siempre: leia el
        ERRORLEVEL del `tee`, no el de Python (medido: Python sale 3, lee 0).
      - Para fechar los cambios de FT (Etapa 1) hubo que reconstruir el orden de
        las corridas cruzando timestamps de filas, archivos y commits.
    Y el incidente de fondo: el 2/9/2026 se salteo un paso y el sistema cruzo
    ruedas distintas sin quejarse.

POLITICA ANTE UNA FALLA (decidida con el usuario, 13/9/2026)
    sync   SEGUIR    ft_run_diario vuelve a sincronizar; se saltea la purga
    paso1  FRENAR    salvo PARCIAL: hasta MAX_PENDIENTES_PASO1 tickers sin la
                     rueda. El umbral es arbitrario (10 = 5% del universo);
                     precedente: el 14/5 SE no tenia datos en Yahoo y el Paso 1
                     salia con error con 199 de 200 al dia.
    paso2  FRENAR
    paso3  FRENAR
    ft     INFORMAR  es el ultimo; el guard de mezcla de ft_run_diario sigue
                     decidiendo si operan los bots.
"""

from html import escape
from typing import NamedTuple

from src.utils.estado_pipeline import resumen_huecos

# Resultados de un paso (tambien los valores validos de rutina_corridas.resultado)
OK = "OK"
PARCIAL = "PARCIAL"
ERROR = "ERROR"
SALTEADO = "SALTEADO"
INTERRUMPIDO = "INTERRUMPIDO"
EN_CURSO = "EN_CURSO"
RESULTADOS = (OK, PARCIAL, ERROR, SALTEADO, INTERRUMPIDO, EN_CURSO)

# Que hacer con el resto de la rutina si un paso falla
SEGUIR = "SEGUIR"
FRENAR = "FRENAR"
INFORMAR = "INFORMAR"

MAX_PENDIENTES_PASO1 = 10

# Codigos de salida de ft_run_diario.bat
FT_GUARD = 1        # el guard de coherencia freno: los bots no corrieron
FT_BOT_FALLO = 2    # los bots corrieron, al menos uno termino con error


class Paso(NamedTuple):
    clave: str
    titulo: str
    si_falla: str
    tabla: str       # tabla cuya fecha de DATOS se mide antes y despues del paso
    columna: str     # su columna de fecha de DATOS (ver estado_pipeline: no la de registro)
    bat: str         # .bat para rehacer solo este paso


PASOS = (
    Paso("sync", "Sync opciones + purga Railway", SEGUIR,
         "opciones_snapshot", "fecha_snapshot", "sync_opciones_railway_to_local.bat"),
    Paso("paso1", "Paso 1 - precios e indicadores", FRENAR,
         "precios_diarios", "fecha", "cron_paso1_precios_yq.bat"),
    Paso("paso2", "Paso 2 - features", FRENAR,
         "features_market_structure", "fecha", "cron_paso2_features.bat"),
    Paso("paso3", "Paso 3 - scanner ML", FRENAR,
         "alertas_scanner", "precio_fecha", "cron_paso3_scanner.bat"),
    Paso("ft", "Forward testing", INFORMAR,
         "ft_equity_diaria", "fecha", "ft_run_diario.bat"),
)

# Pasos que se ejecutan sueltos pero no forman parte de la secuencia
EXTRA = {
    "recovery_incremental": Paso("recovery_incremental", "Recovery incremental", FRENAR,
                                 "precios_diarios", "fecha", "recovery_incremental.bat"),
}


def paso(clave):
    for p in PASOS:
        if p.clave == clave:
            return p
    if clave in EXTRA:
        return EXTRA[clave]
    raise ValueError(f"paso desconocido: {clave}")


def pasos_desde(clave=None):
    """La secuencia completa, o desde un paso (para retomar despues de arreglar)."""
    if clave is None:
        return PASOS
    claves = [p.clave for p in PASOS]
    if clave not in claves:
        raise ValueError(f"paso desconocido: {clave} (validos: {', '.join(claves)})")
    return PASOS[claves.index(clave):]


# ── Clasificacion del resultado de cada paso ──────────────────────────────────

def clasificar_generico(exit_code):
    if exit_code == 0:
        return OK, []
    return ERROR, [f"termino con codigo {exit_code}"]


def clasificar_sync(exit_sync, exit_purga=None):
    """`exit_purga` None = la purga no corrio."""
    if exit_sync != 0:
        return ERROR, ["el sync fallo; ft_run_diario lo vuelve a intentar",
                       "purga de Railway salteada: sin sync verificado no se borra"]
    if exit_purga not in (0, None):
        return PARCIAL, [f"la purga de Railway termino con codigo {exit_purga}"]
    return OK, []


def clasificar_paso1(exit_code, resumen):
    """
    El Paso 1 sale con 1 tanto si falta UN ticker como si se cayo: desde el
    codigo no se distinguen. Por eso decide el resumen que escribe
    recovery_incremental.py --resumen-json al terminar. Sin resumen, no llego
    al final.

    Devuelve (resultado, notas, pendientes) con
    pendientes = {"rueda": str, "precios": {ticker: ultima_fecha}, "futuros": {...}}.
    """
    if resumen is None:
        if exit_code == 0:
            return OK, ["el recovery no dejo resumen"], None
        return ERROR, [f"termino con codigo {exit_code} y sin resumen: se cayo antes de terminar"], None

    rueda = resumen.get("target_date")
    precios = resumen.get("pend_precios") or {}
    futuros = resumen.get("pend_futuros") or {}
    pend = {"rueda": rueda, "precios": precios, "futuros": futuros}

    if exit_code not in (0, 1):
        return ERROR, [f"termino con codigo {exit_code}"], pend
    if len(precios) > MAX_PENDIENTES_PASO1:
        return ERROR, [f"{len(precios)} tickers sin la rueda {rueda}: mas de "
                       f"{MAX_PENDIENTES_PASO1}, la rutina frena"], pend

    notas = []
    if precios:
        notas.append(f"{len(precios)} {'ticker' if len(precios) == 1 else 'tickers'} sin la "
                     f"rueda {rueda} (hasta {MAX_PENDIENTES_PASO1} se sigue)")
    if futuros:
        notas.append(f"{len(futuros)} futuros sin la rueda {rueda}")
    return (PARCIAL if notas else OK), notas, pend


def clasificar_ft(exit_code):
    if exit_code == 0:
        return OK, []
    if exit_code == FT_GUARD:
        return ERROR, ["el guard de coherencia freno: datos no alineados, los bots NO corrieron"]
    if exit_code == FT_BOT_FALLO:
        return PARCIAL, ["al menos un bot termino con error (ver el log de FT)"]
    return ERROR, [f"termino con codigo {exit_code}"]


def debe_seguir(p, resultado):
    if resultado in (OK, PARCIAL):
        return True
    if resultado == INTERRUMPIDO:
        return False
    return p.si_falla != FRENAR


# ── Resumen ───────────────────────────────────────────────────────────────────

def fmt_duracion(seg):
    if seg is None:
        return "-"
    seg = int(round(seg))
    h, r = divmod(seg, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _hay_huecos(p):
    return any((h or {}).get("n") for h in (p.get("huecos") or {}).values())


def estado_general(corrida):
    pasos = corrida["pasos"]
    if any(p["resultado"] == INTERRUMPIDO for p in pasos):
        return "INTERRUMPIDA"
    if corrida.get("frenada_en"):
        return f"FRENADA en {paso(corrida['frenada_en']).titulo}"
    if any(p["resultado"] == ERROR for p in pasos):
        return "CON ERRORES"
    if any(p["resultado"] == PARCIAL or _hay_huecos(p) for p in pasos):
        return "CON AVISOS"
    return "OK"


def codigo_salida(corrida):
    """0 = OK, 2 = con avisos, 1 = frenada, con errores o interrumpida."""
    estado = estado_general(corrida)
    if estado == "OK":
        return 0
    if estado == "CON AVISOS":
        return 2
    return 1


def lineas_pendientes(pend, max_items=50):
    """El detalle de lo que no se pudo bajar, ticker por ticker."""
    if not pend:
        return []
    out = []
    for clave, titulo in (("precios", "Tickers pendientes"), ("futuros", "Futuros pendientes")):
        items = sorted((pend.get(clave) or {}).items())
        if not items:
            continue
        out.append(f"{titulo} de la rueda {pend.get('rueda')} ({len(items)}):")
        for t, ultima in items[:max_items]:
            out.append(f"  {t} (ultimo dato {ultima or 'sin datos'})")
        if len(items) > max_items:
            out.append(f"  ... y {len(items) - max_items} mas (ver log)")
    return out


def lineas_huecos(huecos_por_tabla):
    """Detalle de los huecos de precios_diarios y un conteo de las demas tablas."""
    con = {t: h for t, h in (huecos_por_tabla or {}).items() if h and h.get("n")}
    if not con:
        return []
    base = "precios_diarios" if "precios_diarios" in con else sorted(con)[0]
    out = [f"Huecos en el medio de {base} ({con[base]['n']}), se avisa y no frena:"]
    out += ["  " + x for x in resumen_huecos(con[base])]
    otras = [f"{t} ({h['n']})" for t, h in sorted(con.items()) if t != base]
    if otras:
        out.append("  Tambien en: " + ", ".join(otras))
    return out


def _detalle(corrida):
    lineas = []
    for p in corrida["pasos"]:
        for n in p.get("notas") or []:
            lineas.append(f"- {p['titulo']}: {n}")
    for p in corrida["pasos"]:
        lineas += lineas_pendientes(p.get("pendientes"))
        lineas += lineas_huecos(p.get("huecos"))
    if corrida.get("frenada_en"):
        pf = paso(corrida["frenada_en"])
        lineas.append(f"Arreglar y retomar desde ese paso: rutina_diaria.bat --desde {pf.clave} "
                      f"(o correr solo el paso: {pf.bat})")
    return lineas


def _fila(p):
    ruedas = ""
    if p.get("rueda_antes") or p.get("rueda_despues"):
        ruedas = f"{p.get('tabla')} {p.get('rueda_antes') or '-'} -> {p.get('rueda_despues') or '-'}"
    return (f"{p['resultado']:<13}{p['titulo']:<34}"
            f"{fmt_duracion(p.get('duracion_s')):>8}   {ruedas}")


def resumen_texto(corrida):
    """Resumen para consola y resumen.txt. ASCII puro."""
    sep = "=" * 76
    out = [sep, f"  RUTINA DIARIA {corrida['rutina_id']}  --  {estado_general(corrida)}", sep]
    if corrida.get("desde"):
        out.append(f"  (retomada desde {corrida['desde']})")
    out += ["  " + _fila(p) for p in corrida["pasos"]]
    det = _detalle(corrida)
    if det:
        out.append("")
        out += ["  " + x for x in det]
    total = sum(p.get("duracion_s") or 0 for p in corrida["pasos"])
    out += ["", f"  Duracion total: {fmt_duracion(total)}",
            f"  Logs: {corrida.get('log_dir')}", sep]
    return "\n".join(out)


def mensaje_telegram(corrida):
    """Mensaje HTML para telegram_notifier (parse_mode HTML): todo texto variable escapado."""
    inicio = str(corrida.get("inicio") or "")[:16]
    filas = [f"{p['resultado']:<9} {p['titulo'][:30]:<30} {fmt_duracion(p.get('duracion_s'))}"
             for p in corrida["pasos"]]
    partes = [f"<b>Rutina diaria</b> {escape(inicio)} -- <b>{escape(estado_general(corrida))}</b>",
              "<pre>" + escape("\n".join(filas)) + "</pre>"]
    det = _detalle(corrida)
    if det:
        partes.append(escape("\n".join(det)))
    partes.append(escape(f"Logs: {corrida.get('log_dir')}"))
    return "\n".join(partes)
