"""
create_ft_cambios.py
Crea la tabla ft_cambios (registro de los cambios que afectan a las estrategias
de Forward Testing) y la carga con la historia ya identificada. LOCAL-only.

Contexto (13/9/2026, docs/forward_testing/METRICAS.md seccion 12):
    Para medir si un cambio mejoro o empeoro una estrategia hay que saber CUANDO
    entro en sus decisiones. Hasta hoy eso vivia disperso en el JOURNAL, commits
    y memoria. El reporte HTML lee esta tabla y corta la historia de cada
    estrategia en tramos por los cambios que cambian decisiones.

FECHAS DE LA CARGA: cada fecha efectiva se fecho con EVIDENCIA y no con la fecha
del commit. Se cruzo la hora del commit (o del archivo, o de las filas escritas)
contra ft_operaciones.creado_en de cada corrida y se tomo el fecha_datos de la
primera corrida que ya tenia el cambio. El detalle de cada fila dice como.

Altas posteriores: scripts/forward_testing/ft_cambios.py add

Uso:
    python scripts/oneshot/create_ft_cambios.py            # dry-run: DDL y carga
    python scripts/oneshot/create_ft_cambios.py --apply    # crea y carga (idempotente)
    python scripts/oneshot/create_ft_cambios.py --status
"""

import sys
import os
import argparse
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.forward_testing.ft_env import configurar_entorno_local  # noqa: E402
configurar_entorno_local()

from sqlalchemy import text  # noqa: E402
from src.data.database import get_engine  # noqa: E402
from src.utils.ft_tramos import TIPOS_CAMBIO  # noqa: E402
from src.utils.trading_calendar import is_trading_day  # noqa: E402


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


_TIPOS_SQL = ", ".join(f"'{t}'" for t in TIPOS_CAMBIO)

# Sin dos puntos en los COMMENT: text() de SQLAlchemy los toma como parametros.
DDL = f"""
CREATE TABLE IF NOT EXISTS ft_cambios (
    id                 SERIAL PRIMARY KEY,
    clave              VARCHAR(60) NOT NULL UNIQUE,
    fecha_efectiva     DATE        NOT NULL,
    tipo               VARCHAR(20) NOT NULL,
    estrategias        INTEGER[]   NOT NULL,
    cambia_decisiones  BOOLEAN     NOT NULL,
    titulo             TEXT        NOT NULL,
    detalle            TEXT,
    ref                TEXT,
    registrado_en      TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT ft_cambios_tipo_chk CHECK (tipo IN ({_TIPOS_SQL})),
    CONSTRAINT ft_cambios_estrategias_chk CHECK (cardinality(estrategias) > 0)
);

CREATE INDEX IF NOT EXISTS idx_ft_cambios_fecha ON ft_cambios(fecha_efectiva);

COMMENT ON TABLE ft_cambios IS
    'Cambios que afectan a las estrategias de FT. El reporte corta la historia de '
    'cada estrategia en tramos por los que cambian decisiones. Ver '
    'docs/forward_testing/METRICAS.md seccion 12';
COMMENT ON COLUMN ft_cambios.fecha_efectiva IS
    'Primera rueda de DATOS con la que la estrategia decidio ya con el cambio '
    '(fecha_datos de la primera corrida que lo tuvo). NO es la fecha del commit '
    'ni la de la corrida.';
COMMENT ON COLUMN ft_cambios.estrategias IS
    'ids de ft_estrategias afectados. Sin FK (array); lo valida ft_cambios.py.';
COMMENT ON COLUMN ft_cambios.cambia_decisiones IS
    'TRUE solo si cambia la LOGICA, los PARAMETROS o el MODELO de la estrategia '
    '(corta tramos). Una correccion de datos puntual va en FALSE aunque mueva '
    'alguna decision (queda como marca).';
"""

TODAS = list(range(1, 11))

SEMILLA = [
    {
        "clave": "earnings_calendar_nasdaq",
        "fecha_efectiva": date(2026, 5, 19),
        "tipo": "DATOS",
        "estrategias": list(range(1, 10)),
        "cambia_decisiones": False,
        "titulo": "El filtro de earnings pasa a leer earnings_calendar (Nasdaq) en vez de yfinance",
        "detalle": (
            "Antes cada bot consultaba yfinance por ticker y el rate limit cortaba la consulta "
            "a mitad de corrida; sin fecha el filtro no actua (fail-safe). Mueve el bloqueo de "
            "entradas y el cierre EARNINGS_MANANA de las 9 estrategias vivas. Commit 15.19 del "
            "18/5, despues de la corrida de la madrugada del 18/5 (dato 8/5); la siguiente fue "
            "el 20/5 con el dato del 19/5. Afecta a todas: no tiene grupo de control."
        ),
        "ref": "JOURNAL 2026-05-18 DISENO; commit 8652db5",
    },
    {
        "clave": "precio_subyacente_yahooquery",
        "fecha_efectiva": date(2026, 5, 26),
        "tipo": "DATOS",
        "estrategias": [10],
        "cambia_decisiones": False,
        "titulo": "precio_subyacente de la captura de opciones desde yahooquery, no desde Railway congelado",
        "detalle": (
            "Mueve la zona donde OIEXIT_v1 busca el put wall del SL inicial. Fecha APROXIMADA: "
            "la captura corre en Oracle, que no hacia git pull automatico hasta el 6/6/2026. "
            "La estrategia nacio el 23/5: no hay historia previa con que comparar."
        ),
        "ref": "commit bd0642f",
    },
    {
        "clave": "fix_score_cero_salida",
        "fecha_efectiva": date(2026, 5, 29),
        "tipo": "BUG_FIX",
        "estrategias": [4, 6, 8, 9],
        "cambia_decisiones": True,
        "titulo": "La query de salida de 4 bots sectoriales devolvia score tecnico 0.0 siempre",
        "detalle": (
            "Faltaba el JOIN con precios_diarios (close=0), asi que las salidas se decidian con "
            "score 0 todos los dias: churn SCORE_DEGRADADO_0.0 en v1, SIN_MOMENTUM ficticio en "
            "v2 y decisiones 100% PCR en OPTIONS v1/v2. Commit 11.08 del 30/5; la corrida de las "
            "11.25 (dato 29/5) ya uso el fix. Es el corte de la ventana comparable del reporte."
        ),
        "ref": "JOURNAL 2026-05-30 BUG FIX y DECISION; commit 4394f2d",
    },
    {
        "clave": "cerebro_compartido_src_strategies",
        "fecha_efectiva": date(2026, 6, 4),
        "tipo": "REFACTOR",
        "estrategias": [1, 4, 9],
        "cambia_decisiones": False,
        "titulo": "La logica de decision de 3 bots pasa al cerebro compartido src/strategies/",
        "detalle": (
            "Refactor sin cambio de comportamiento buscado: ML_SCANNER_v1, TECH_SECTOR_v1 y "
            "TECH_SECTOR_OPTIONS_v2 deciden con el modulo que comparten con Alpaca. Se registra "
            "porque mover logica de decision es donde un cambio silencioso pasa desapercibido. "
            "Commit 17.59 del 4/6, despues de la corrida de las 11.57; la del 5/6 (dato 4/6) ya "
            "lo tenia."
        ),
        "ref": "commit 66a84f9",
    },
    {
        "clave": "splits_klac_crwd_corregidos",
        "fecha_efectiva": date(2026, 7, 21),
        "tipo": "DATOS",
        "estrategias": TODAS,
        "cambia_decisiones": False,
        "titulo": "Splits KLAC 10:1 y CRWD 4:1 aplicados a precios_diarios",
        "detalle": (
            "Hasta aca la serie de los dos tickers estaba partida en dos escalas: indicadores, "
            "features y stops rotos para ellos en cualquier estrategia. Backups de la correccion "
            "12.40 del 21/7, despues de la corrida de las 11.25; la del 22/7 (dato 21/7) ya "
            "estaba corregida. Las 8 operaciones cerradas por el falso derrumbe llevan _SPLIT_FIX "
            "y quedan fuera de las metricas de operacion por tramo."
        ),
        "ref": "JOURNAL 2026-07-21 BUG FIX; commit 0724f5d",
    },
    {
        "clave": "equity_a_mercado_y_fecha_datos",
        "fecha_efectiva": date(2026, 7, 21),
        "tipo": "MEDICION",
        "estrategias": TODAS,
        "cambia_decisiones": False,
        "titulo": "Equity marcada a mercado (ft_equity_diaria) y fecha_datos en ft_operaciones",
        "detalle": (
            "No cambia ninguna decision: cambia como se MIDE y aplica hacia atras sobre toda la "
            "historia, porque ambas son capas recomputables."
        ),
        "ref": "docs/forward_testing/METRICAS.md; commits 0724f5d, 3248528",
    },
    {
        "clave": "sin_contexto_sectorial",
        "fecha_efectiva": date(2026, 9, 2),
        "tipo": "DATOS",
        "estrategias": [1],
        "cambia_decisiones": False,
        "titulo": "Real Estate y Utilities fuera del calculo de features sectoriales (AMT, EQIX, PLD, VST)",
        "detalle": (
            "Esos 4 tickers pasan a recibir 11 de las 53 features del modelo en NaN; el modelo "
            "sigue siendo el global. Afecta solo al ML, via alertas_scanner. Merge 17.56 del 2/9, "
            "despues de la corrida de las 17.25; las features de la rueda del 2/9 ya se "
            "calcularon sin ellos (corrida del 3/9)."
        ),
        "ref": "CLAUDE.md 4 tickers sin contexto sectorial; commits c35a0a9, 2f8833e",
    },
    {
        "clave": "guard_coherencia_rutina",
        "fecha_efectiva": date(2026, 9, 2),
        "tipo": "INFRA",
        "estrategias": TODAS,
        "cambia_decisiones": False,
        "titulo": "Guard de coherencia de la rutina antes de que operen los bots",
        "detalle": (
            "chequeo_rutina.py frena ft_run_diario si las tablas de insumo tienen fechas de DATOS "
            "distintas entre si. No cambia la logica: evita corridas con senal ML de una rueda y "
            "tecnico de otra (incidente 2/9). Primera corrida protegida, la del 3/9 (dato 2/9)."
        ),
        "ref": "commits f23982e, 850b981, 1d549c6",
    },
    {
        "clave": "precio_referencia_put_wall",
        "fecha_efectiva": date(2026, 9, 11),
        "tipo": "DATOS",
        "estrategias": [10],
        "cambia_decisiones": False,
        "titulo": "El put wall de OIEXIT_v1 usa el precio de referencia (close de precios_diarios)",
        "detalle": (
            "El SL inicial por put wall deja de caer al fallback de ATR cuando la captura trae el "
            "precio vacio. Verificado en las operaciones: la corrida del 10/9 (dato 9/9, captura "
            "sin precio) abrio 5 con SL por ATR y el bot se modifico a las 17.26 de ese dia; la "
            "del 12/9 (dato 11/9) ya abrio 2 por put wall. Las posiciones abiertas antes no se "
            "tocan: el SL inicial se fija al abrir."
        ),
        "ref": "JOURNAL 2026-09-10 BUG FIX; commits 8f28ba2, 8eb1179",
    },
    {
        "clave": "relleno_hueco_20260828",
        "fecha_efectiva": date(2026, 9, 14),
        "tipo": "DATOS",
        "estrategias": TODAS,
        "cambia_decisiones": False,
        "titulo": "Relleno de la rueda 2026-08-28 (157 tickers) y de SCCO 2026-08-11 en precios_diarios",
        "detalle": (
            "Durante dos semanas indicadores, features y equity se calcularon sin esa rueda. Las "
            "filas se insertaron a las 21.47 del 12/9, despues de la corrida de las 19.22 (dato "
            "11/9): la primera corrida con la serie completa es la de la rueda del 14/9. Los "
            "registros historicos (alertas, operaciones, posiciones) no se tocan."
        ),
        "ref": "relleno manual 12/9/2026, operacion de datos sin commit",
    },
]

UPSERT = """
    INSERT INTO ft_cambios (clave, fecha_efectiva, tipo, estrategias,
                            cambia_decisiones, titulo, detalle, ref)
    VALUES (:clave, :fecha_efectiva, :tipo, :estrategias,
            :cambia_decisiones, :titulo, :detalle, :ref)
    ON CONFLICT (clave) DO UPDATE SET
        fecha_efectiva = EXCLUDED.fecha_efectiva,
        tipo = EXCLUDED.tipo,
        estrategias = EXCLUDED.estrategias,
        cambia_decisiones = EXCLUDED.cambia_decisiones,
        titulo = EXCLUDED.titulo,
        detalle = EXCLUDED.detalle,
        ref = EXCLUDED.ref
"""


def validar(ids_existentes):
    errores = []
    for s in SEMILLA:
        if not is_trading_day(s["fecha_efectiva"]):
            errores.append(f"{s['clave']}: {s['fecha_efectiva']} no es dia habil")
        if s["tipo"] not in TIPOS_CAMBIO:
            errores.append(f"{s['clave']}: tipo {s['tipo']} invalido")
        faltan = set(s["estrategias"]) - ids_existentes
        if faltan:
            errores.append(f"{s['clave']}: estrategias inexistentes {sorted(faltan)}")
    return errores


def estado(engine):
    with engine.connect() as conn:
        if conn.execute(text("SELECT to_regclass('public.ft_cambios')")).scalar() is None:
            log("ft_cambios NO existe.")
            return
        rows = conn.execute(text("""
            SELECT fecha_efectiva, cambia_decisiones, tipo, clave, estrategias
            FROM ft_cambios ORDER BY fecha_efectiva, id
        """)).fetchall()
    log(f"ft_cambios: {len(rows)} filas")
    for r in rows:
        print(f"   {r.fecha_efectiva}  {'CORTA' if r.cambia_decisiones else '-    '}  "
              f"{r.tipo:<9} {r.clave:<36} {r.estrategias}")


def main():
    ap = argparse.ArgumentParser(description="Crea y carga ft_cambios (LOCAL).")
    ap.add_argument("--apply", action="store_true", help="crea la tabla y carga la semilla")
    ap.add_argument("--status", action="store_true", help="estado actual")
    args = ap.parse_args()

    engine = get_engine()
    log(f"Target: {engine.url.host}/{engine.url.database}")
    if args.status:
        estado(engine)
        return 0

    with engine.connect() as conn:
        ids = {r[0] for r in conn.execute(text("SELECT id FROM ft_estrategias"))}
    errores = validar(ids)
    if errores:
        for e in errores:
            log(f"[ERROR] {e}")
        return 1

    if not args.apply:
        print(DDL)
        for s in SEMILLA:
            print(f"  {s['fecha_efectiva']}  {'CORTA' if s['cambia_decisiones'] else '-    '}  "
                  f"{s['tipo']:<9} {s['clave']:<36} {s['estrategias']}")
        log("DRY-RUN: nada escrito. Correr con --apply.")
        return 0

    with engine.connect() as conn:
        conn.execute(text(DDL))
        for s in SEMILLA:
            conn.execute(text(UPSERT), s)
        conn.commit()
    log(f"ft_cambios creada/actualizada con {len(SEMILLA)} cambios.")
    estado(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
