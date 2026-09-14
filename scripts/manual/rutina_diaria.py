"""
rutina_diaria.py
La rutina diaria completa en UNA corrida, y el ejecutor de cada paso suelto.

Politica (que frena y que no) y resumen: src/utils/rutina.py (modulo PURO).
Flujo documentado: docs/checklist_recovery_manual.md.

QUE HACE CADA CORRIDA
    - Muestra la salida de cada paso en vivo y la guarda en un log:
        rutina completa -> logs/rutina/AAAAMMDD_HHMM/NN_<paso>.log + resumen.txt
        paso suelto     -> logs/rutina/pasos/<paso>_AAAAMMDD_HHMM.log
    - Devuelve el codigo de salida REAL de cada proceso. recovery_incremental.bat
      leia el del `tee` (siempre 0) y decia "RECOVERY COMPLETO" con pendientes.
    - Anota cada paso en rutina_corridas (LOCAL) al arrancar y al terminar. Si la
      DB no responde, el paso corre igual y se avisa.
    - Al terminar el Paso 1 busca huecos en el medio de la serie: avisa, no frena.
    - La rutina completa manda el resumen por Telegram, con el detalle de los
      tickers que quedaron pendientes.

Uso:
    python scripts/manual/rutina_diaria.py todo                 sync, paso1, paso2, paso3, ft
    python scripts/manual/rutina_diaria.py todo --desde paso2   retomar despues de arreglar
    python scripts/manual/rutina_diaria.py paso paso1           un solo paso (lo usan los .bat)
    python scripts/manual/rutina_diaria.py correr --nombre recovery_incremental -- --dry-run
    python scripts/manual/rutina_diaria.py registrar --paso ft --inicio 20260914_0930 --exit 0 --log RUTA

Codigo de salida: 0 OK | 2 con avisos (PARCIAL o huecos) | 1 frenada, con errores o interrumpida.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import rutina as R  # noqa: E402

PYTHON = sys.executable
DIR_LOGS = os.path.join(ROOT, "logs", "rutina")
BAT_FT = os.path.join(ROOT, "scripts", "manual", "ft_run_diario.bat")
SEP = "=" * 76


# ── Log a consola y archivo ───────────────────────────────────────────────────

class Log:
    """Escribe a la vez en la consola y en el archivo de log."""

    def __init__(self, ruta):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        self.ruta = ruta
        self._fh = open(ruta, "a", encoding="utf-8")

    def escribir(self, texto):
        try:
            sys.stdout.write(texto)
            sys.stdout.flush()
        except Exception:
            pass   # una consola que no acepta un caracter no puede tirar el paso
        self._fh.write(texto)
        self._fh.flush()

    def linea(self, texto=""):
        self.escribir(texto + "\n")

    def cerrar(self):
        self._fh.close()


def ejecutar(argv, log, env=None, al_leer=None):
    """
    Corre `argv` mostrando la salida en vivo y guardandola en el log, y devuelve
    el codigo de salida DEL PROCESO. Es lo que el `tee` de los .bat perdia.
    """
    entorno = dict(os.environ if env is None else env)
    entorno["PYTHONUNBUFFERED"] = "1"     # sin esto la salida llega de a bloques
    entorno["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(argv, cwd=ROOT, env=entorno,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for bruto in iter(proc.stdout.readline, b""):
            texto = bruto.decode("utf-8", errors="replace").replace("\r\n", "\n")
            log.escribir(texto)
            if al_leer:
                al_leer(texto)
        return proc.wait()
    except KeyboardInterrupt:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


# ── Registro en rutina_corridas ───────────────────────────────────────────────

_ENGINE = None
_DB_AVISADA = False

_SQL_INICIO = """
    INSERT INTO rutina_corridas (rutina_id, paso, origen, inicio, resultado,
                                 rueda_antes, log_path, git_commit)
    VALUES (:rutina_id, :paso, :origen, :inicio, :resultado,
            :rueda_antes, :log_path, :git_commit)
    RETURNING id
"""
_SQL_FIN = """
    UPDATE rutina_corridas
    SET fin = :fin, duracion_s = :duracion_s, exit_code = :exit_code,
        resultado = :resultado, rueda_despues = :rueda_despues,
        detalle = CAST(:detalle AS JSONB)
    WHERE id = :id
"""
_SQL_COMPLETO = """
    INSERT INTO rutina_corridas (rutina_id, paso, origen, inicio, fin, duracion_s,
                                 exit_code, resultado, rueda_antes, rueda_despues,
                                 detalle, log_path, git_commit)
    VALUES (:rutina_id, :paso, :origen, :inicio, :fin, :duracion_s,
            :exit_code, :resultado, :rueda_antes, :rueda_despues,
            CAST(:detalle AS JSONB), :log_path, :git_commit)
    RETURNING id
"""


def _engine():
    global _ENGINE
    if _ENGINE is None:
        # LOCAL-only: con DATABASE_URL seteada get_engine cae a Railway.
        os.environ.pop("DATABASE_URL", None)
        from src.data.database import get_engine
        _ENGINE = get_engine()
    return _ENGINE


def _db(sql, params, log=None):
    """Ejecuta contra la DB local. Nunca levanta: el registro no puede tirar un paso."""
    global _DB_AVISADA
    try:
        from sqlalchemy import text
        with _engine().connect() as c:
            r = c.execute(text(sql), params)
            valor = r.scalar() if r.returns_rows else None
            c.commit()
        return valor
    except Exception as e:
        if not _DB_AVISADA:
            msg = (f"[WARN] rutina_corridas: no se pudo registrar ({str(e).splitlines()[0][:120]}). "
                   f"El paso sigue igual.")
            (log.linea if log else print)(msg)
            _DB_AVISADA = True
        return None


def rueda_de(p):
    """Fecha de DATOS de la tabla del paso (MAX de su columna de datos)."""
    try:
        from sqlalchemy import text
        with _engine().connect() as c:
            v = c.execute(text(f"SELECT MAX({p.columna}) FROM {p.tabla}")).scalar()
        return v.date() if isinstance(v, datetime) else v
    except Exception:
        return None


def _commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        pass
    return None


# ── Pasos ─────────────────────────────────────────────────────────────────────

def buscar_huecos(log):
    """Huecos en el medio de la serie, con la misma logica que chequeo_rutina."""
    try:
        aqui = os.path.dirname(os.path.abspath(__file__))
        if aqui not in sys.path:
            sys.path.insert(0, aqui)
        import chequeo_rutina as ch
        h = ch.leer_huecos()
        for x in ch.lineas_huecos(h):
            log.linea(x)
        return {t: {"n": r["n"],
                    "huecos": {str(f): tks for f, tks in r["huecos"].items()},
                    "conocidos": [[tk, str(f), m] for tk, f, m in r["conocidos"]]}
                for t, r in h.items()}
    except Exception as e:
        log.linea(f"[WARN] no se pudo buscar huecos: {str(e).splitlines()[0][:150]}")
        return None


def _script(*partes):
    return os.path.join(ROOT, "scripts", *partes)


def correr_paso(p, log, origen, rutina_id=None, dir_trabajo=None, extra=None):
    """Corre un paso con encabezado, pie, clasificacion y registro. Devuelve su resultado."""
    extra = [a for a in (extra or []) if a != "--"]
    dir_trabajo = dir_trabajo or os.path.dirname(log.ruta)
    dry_run = "--dry-run" in extra
    medir = not ("--target" in extra and "railway" in extra)

    inicio = datetime.now()
    antes = rueda_de(p) if medir else None
    commit = _commit()
    log.linea(SEP)
    log.linea(f"  {p.titulo.upper()}")
    log.linea(f"  inicio {inicio:%Y-%m-%d %H:%M:%S}  |  commit {commit or '-'}  |  "
              f"{p.tabla} con datos al {antes or '-'}")
    log.linea(f"  log {log.ruta}")
    log.linea(SEP)

    rid = None
    if not dry_run:
        rid = _db(_SQL_INICIO, {"rutina_id": rutina_id, "paso": p.clave, "origen": origen,
                                "inicio": inicio, "resultado": R.EN_CURSO,
                                "rueda_antes": antes, "log_path": log.ruta,
                                "git_commit": commit}, log)

    res = {"clave": p.clave, "titulo": p.titulo, "tabla": p.tabla, "resultado": R.EN_CURSO,
           "notas": [], "pendientes": None, "huecos": None, "log": log.ruta,
           "rueda_antes": str(antes) if antes else None}
    exit_code = None
    try:
        if p.clave == "sync":
            e_sync = ejecutar([PYTHON, _script("migrations", "sync_railway_to_local.py"),
                               "--tabla", "opciones"], log)
            e_purga = None
            if e_sync == 0:
                log.linea("")
                log.linea("--- Purga de opciones_snapshot en Railway (deja 10 dias) ---")
                e_purga = ejecutar([PYTHON, _script("manual", "retencion_opciones_railway.py"),
                                    "--quiet-skip"], log)
            exit_code = e_sync if e_sync != 0 else (e_purga or 0)
            res["resultado"], res["notas"] = R.clasificar_sync(e_sync, e_purga)

        elif p.clave in ("paso1", "recovery_incremental"):
            args = (["--target", "local", "--engine", "yahooquery"] if p.clave == "paso1"
                    else extra)
            ruta_json = os.path.join(dir_trabajo, f"{p.clave}_{inicio:%Y%m%d_%H%M%S}_resumen.json")
            exit_code = ejecutar([PYTHON, _script("recovery_incremental.py"), *args,
                                  "--resumen-json", ruta_json], log)
            resumen = None
            if os.path.exists(ruta_json):
                with open(ruta_json, encoding="utf-8") as fh:
                    resumen = json.load(fh)
            if dry_run:
                # En dry-run el 1 significa "hay pendientes para bajar", no una falla.
                res["resultado"], res["notas"] = R.clasificar_generico(
                    0 if exit_code in (0, 1) and resumen is not None else exit_code)
                res["notas"].append("dry-run: no se escribio nada")
            else:
                res["resultado"], res["notas"], res["pendientes"] = R.clasificar_paso1(exit_code, resumen)
                if medir:
                    log.linea("")
                    res["huecos"] = buscar_huecos(log)

        elif p.clave in ("paso2", "paso3"):
            step = "features" if p.clave == "paso2" else "scanner"
            exit_code = ejecutar([PYTHON, _script("cron_diario.py"), "--step", step], log)
            res["resultado"], res["notas"] = R.clasificar_generico(exit_code)

        elif p.clave == "ft":
            def _captar_log_ft(texto):
                t = texto.strip()
                if "log_ft" not in res and t.startswith("Log") and ":" in t:
                    res["log_ft"] = t.split(":", 1)[1].strip()
            # RUTINA_ORQUESTADA: ft_run_diario no pausa ni se registra por su cuenta.
            exit_code = ejecutar(["cmd", "/c", BAT_FT], log,
                                 env=dict(os.environ, RUTINA_ORQUESTADA="1"),
                                 al_leer=_captar_log_ft)
            res["resultado"], res["notas"] = R.clasificar_ft(exit_code)

        else:
            raise ValueError(f"paso sin ejecutor: {p.clave}")
    except KeyboardInterrupt:
        res["resultado"], res["notas"] = R.INTERRUMPIDO, ["corrida interrumpida con Ctrl+C"]

    fin = datetime.now()
    duracion = (fin - inicio).total_seconds()
    despues = rueda_de(p) if medir else None
    res.update(duracion_s=round(duracion), exit_code=exit_code,
               rueda_despues=str(despues) if despues else None)

    log.linea("")
    log.linea(SEP)
    log.linea(f"  {p.titulo}: {res['resultado']}  |  codigo {exit_code}  |  "
              f"{R.fmt_duracion(duracion)}  |  {p.tabla} {antes or '-'} -> {despues or '-'}")
    for n in res["notas"]:
        log.linea(f"  - {n}")
    for x in R.lineas_pendientes(res["pendientes"]):
        log.linea(f"  {x}")
    if res.get("log_ft"):
        log.linea(f"  Log detallado de FT: {res['log_ft']}")
    log.linea(SEP)

    if rid is not None:
        detalle = json.dumps({"notas": res["notas"], "pendientes": res["pendientes"],
                              "huecos": res["huecos"], "log_ft": res.get("log_ft")}, default=str)
        _db(_SQL_FIN, {"id": rid, "fin": fin, "duracion_s": round(duracion),
                       "exit_code": exit_code, "resultado": res["resultado"],
                       "rueda_despues": despues, "detalle": detalle}, log)
    return res


def _telegram(mensaje):
    try:
        from src.pipeline.telegram_notifier import _send_long
        if _send_long(mensaje):
            print("[OK] Resumen enviado por Telegram.")
        else:
            print("[WARN] No se pudo enviar el resumen por Telegram.")
    except Exception as e:
        print(f"[WARN] Telegram: {str(e)[:120]}")


def _codigo_de(res):
    return R.codigo_salida({"pasos": [res], "frenada_en": None})


# ── Comandos ──────────────────────────────────────────────────────────────────

def cmd_todo(args):
    inicio = datetime.now()
    rutina_id = f"{inicio:%Y%m%d_%H%M}"
    dir_corrida = os.path.join(DIR_LOGS, rutina_id)
    pasos = R.pasos_desde(args.desde)
    corrida = {"rutina_id": rutina_id, "inicio": f"{inicio:%Y-%m-%d %H:%M}",
               "desde": args.desde, "log_dir": dir_corrida, "pasos": [], "frenada_en": None}

    print(SEP)
    print(f"  RUTINA DIARIA {rutina_id}")
    print(f"  Pasos: {', '.join(p.titulo for p in pasos)}")
    print(f"  Logs : {dir_corrida}")
    print(SEP)

    seguir = True
    for i, p in enumerate(pasos, 1):
        if not seguir:
            corrida["pasos"].append({"clave": p.clave, "titulo": p.titulo, "tabla": p.tabla,
                                     "resultado": R.SALTEADO})
            continue
        print(f"\n[{i}/{len(pasos)}] {p.titulo}")
        log = Log(os.path.join(dir_corrida, f"{i:02d}_{p.clave}.log"))
        try:
            res = correr_paso(p, log, "rutina", rutina_id, dir_corrida)
        finally:
            log.cerrar()
        corrida["pasos"].append(res)
        if not R.debe_seguir(p, res["resultado"]):
            seguir = False
            if res["resultado"] != R.INTERRUMPIDO:
                corrida["frenada_en"] = p.clave

    texto = R.resumen_texto(corrida)
    print("\n" + texto)
    os.makedirs(dir_corrida, exist_ok=True)
    with open(os.path.join(dir_corrida, "resumen.txt"), "w", encoding="utf-8") as fh:
        fh.write(texto + "\n")
    if not args.sin_telegram:
        _telegram(R.mensaje_telegram(corrida))
    return R.codigo_salida(corrida)


def cmd_paso(args):
    p = R.paso(args.clave)
    log = Log(os.path.join(DIR_LOGS, "pasos", f"{p.clave}_{datetime.now():%Y%m%d_%H%M}.log"))
    try:
        res = correr_paso(p, log, "suelto")
    finally:
        log.cerrar()
    return _codigo_de(res)


def cmd_correr(args):
    p = R.paso(args.nombre)
    log = Log(os.path.join(DIR_LOGS, "pasos", f"{p.clave}_{datetime.now():%Y%m%d_%H%M}.log"))
    try:
        res = correr_paso(p, log, "suelto", extra=args.resto)
    finally:
        log.cerrar()
    return _codigo_de(res)


def cmd_registrar(args):
    """Para ft_run_diario.bat corrido por su cuenta. Nunca devuelve error."""
    try:
        p = R.paso(args.paso)
        fin = datetime.now()
        try:
            inicio = datetime.strptime(args.inicio or "", "%Y%m%d_%H%M")
        except ValueError:
            inicio = fin
        resultado, notas = (R.clasificar_ft(args.exit) if p.clave == "ft"
                            else R.clasificar_generico(args.exit))
        despues = rueda_de(p)
        rid = _db(_SQL_COMPLETO, {
            "rutina_id": None, "paso": p.clave, "origen": "suelto", "inicio": inicio,
            "fin": fin, "duracion_s": round((fin - inicio).total_seconds()),
            "exit_code": args.exit, "resultado": resultado, "rueda_antes": None,
            "rueda_despues": despues,
            "detalle": json.dumps({"notas": notas, "log_ft": args.log}),
            "log_path": args.log, "git_commit": _commit()})
        if rid is not None:
            print(f"[rutina_corridas] {p.clave}: {resultado} registrado.")
    except Exception as e:
        print(f"[WARN] rutina_corridas: {str(e)[:120]}")
    return 0


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Rutina diaria orquestada (LOCAL).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    claves = [p.clave for p in R.PASOS]

    t = sub.add_parser("todo", help="la rutina completa, en orden")
    t.add_argument("--desde", choices=claves, help="retomar desde este paso")
    t.add_argument("--sin-telegram", action="store_true", help="no mandar el resumen")

    s = sub.add_parser("paso", help="un solo paso, con log y registro")
    s.add_argument("clave", choices=claves)

    c = sub.add_parser("correr", help="un script por fuera de la secuencia (recovery_incremental)")
    c.add_argument("--nombre", required=True, choices=sorted(R.EXTRA))
    c.add_argument("resto", nargs=argparse.REMAINDER, help="argumentos para el script, despues de --")

    g = sub.add_parser("registrar", help="anota una corrida hecha por fuera (ft_run_diario suelto)")
    g.add_argument("--paso", required=True)
    g.add_argument("--inicio", help="AAAAMMDD_HHMM")
    g.add_argument("--exit", type=int, required=True)
    g.add_argument("--log")

    args = ap.parse_args()
    return {"todo": cmd_todo, "paso": cmd_paso, "correr": cmd_correr,
            "registrar": cmd_registrar}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
