"""
sheets_export.py
Exporta datos del scanner ML a Google Sheets.

Se ejecuta diariamente via GitHub Actions despues del cron_diario.py.

Requiere (GitHub Secrets):
    GOOGLE_SHEETS_KEY  -- JSON del Service Account (string completo)
    GOOGLE_SHEETS_ID   -- ID del spreadsheet (de la URL de Google Sheets)
    DATABASE_URL       -- ya existente en el workflow

Tabs que actualiza:
    Dashboard         -- ultima alerta por ticker, coloreada por nivel
    Analisis Tecnico  -- ultimos indicadores por ticker
    Historial         -- ultimas 90 dias de alertas con retornos reales

Uso local (para probar):
    python scripts/sheets_export.py
"""

import os
import sys
import json
import time
import traceback
import pathlib
import pandas as pd
from datetime import date, timedelta, datetime

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Cargar .env local si existe (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.data.database import query_df

# Ruta al archivo de credenciales local (para desarrollo)
_LOCAL_KEY_FILE = ROOT / "secrets" / "google_sheets_key.json"


# ─────────────────────────────────────────────────────────────
# Logging con timestamps
# ─────────────────────────────────────────────────────────────

_t0_global = time.time()

def _log(msg: str, nivel: str = "INFO"):
    """Imprime mensaje con timestamp relativo al inicio del script."""
    elapsed = time.time() - _t0_global
    ts = datetime.now().strftime("%H:%M:%S")
    prefijos = {
        "INFO":  "  ",
        "OK":    "  [OK]  ",
        "STEP":  "  [-->] ",
        "WARN":  "  [WARN]",
        "ERROR": "  [ERR] ",
        "SKIP":  "  [SKIP]",
    }
    prefijo = prefijos.get(nivel, "  ")
    print(f"  {ts} +{elapsed:5.1f}s {prefijo} {msg}", flush=True)


def _separador(titulo: str = ""):
    ancho = 60
    if titulo:
        print(f"\n  --- {titulo} {'-'*(ancho - len(titulo) - 5)}", flush=True)
    else:
        print(f"  {'-'*ancho}", flush=True)


# ─────────────────────────────────────────────────────────────
# Colores RGB por nivel de alerta
# ─────────────────────────────────────────────────────────────

NIVEL_COLOR = {
    "COMPRA_FUERTE": {"red": 0.13, "green": 0.55, "blue": 0.13},  # verde oscuro
    "COMPRA":        {"red": 0.72, "green": 0.96, "blue": 0.72},  # verde claro
    "NEUTRAL":       {"red": 1.00, "green": 1.00, "blue": 1.00},  # blanco
    "VENTA":         {"red": 1.00, "green": 0.71, "blue": 0.71},  # rojo claro
    "VENTA_FUERTE":  {"red": 0.86, "green": 0.20, "blue": 0.20},  # rojo oscuro
}

_COLOR_HEADER     = {"red": 0.20, "green": 0.29, "blue": 0.49}   # azul oscuro
_COLOR_HEADER_TXT = {"red": 1.0,  "green": 1.0,  "blue": 1.0}    # blanco


def _nivel_color(nivel):
    return NIVEL_COLOR.get(
        str(nivel).strip() if nivel else "",
        {"red": 1.0, "green": 1.0, "blue": 1.0}
    )


# ─────────────────────────────────────────────────────────────
# Conexion a Google Sheets
# ─────────────────────────────────────────────────────────────

def _get_spreadsheet():
    _log("Importando gspread y google-auth...", "STEP")
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        _log(f"gspread {gspread.__version__} importado OK", "OK")
    except ImportError as e:
        raise ImportError(f"Instalar: pip install gspread google-auth  ({e})")

    # Sheet ID
    sheet_id = os.environ.get("GOOGLE_SHEETS_ID", "").strip()
    if not sheet_id:
        raise ValueError(
            "GOOGLE_SHEETS_ID no configurado.\n"
            "         Local:  agregar GOOGLE_SHEETS_ID=... en .env\n"
            "         GitHub: agregar Secret GOOGLE_SHEETS_ID"
        )
    _log(f"GOOGLE_SHEETS_ID: ...{sheet_id[-12:]}", "INFO")

    # Credenciales
    creds_json = os.environ.get("GOOGLE_SHEETS_KEY", "").strip()
    if creds_json:
        _log("Credenciales: desde variable de entorno GOOGLE_SHEETS_KEY", "INFO")
        creds_dict = json.loads(creds_json)
    elif _LOCAL_KEY_FILE.exists():
        _log(f"Credenciales: desde archivo local {_LOCAL_KEY_FILE.name}", "INFO")
        with open(_LOCAL_KEY_FILE, "r", encoding="utf-8") as f:
            creds_dict = json.load(f)
    else:
        raise ValueError(
            "Credenciales de Google no encontradas.\n"
            "         Opcion A (GitHub): configurar Secret GOOGLE_SHEETS_KEY\n"
            f"         Opcion B (local):  guardar JSON en {_LOCAL_KEY_FILE}"
        )

    # Mostrar service account sin exponer datos sensibles
    sa_email = creds_dict.get("client_email", "desconocido")
    _log(f"Service account: {sa_email}", "INFO")

    # Autenticar
    _log("Autenticando con Google API...", "STEP")
    t = time.time()
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    _log(f"Autenticacion OK ({time.time()-t:.1f}s)", "OK")

    # Abrir spreadsheet
    _log("Abriendo spreadsheet...", "STEP")
    t = time.time()
    spreadsheet = gc.open_by_key(sheet_id)
    _log(f"Spreadsheet '{spreadsheet.title}' abierto ({time.time()-t:.1f}s)", "OK")
    return spreadsheet


# ─────────────────────────────────────────────────────────────
# Escribir un DataFrame en un tab
# ─────────────────────────────────────────────────────────────

def _escribir_tab(spreadsheet, nombre: str, df: pd.DataFrame,
                  col_nivel: str = None):
    """
    Limpia el tab y escribe el DataFrame completo.
    Aplica formato al header y colores por nivel si se indica.
    """
    # Obtener o crear worksheet
    _log(f"  Buscando tab '{nombre}'...", "STEP")
    t = time.time()
    try:
        ws = spreadsheet.worksheet(nombre)
        _log(f"  Tab '{nombre}' encontrado, limpiando...", "INFO")
        ws.clear()
        _log(f"  Tab limpiado ({time.time()-t:.1f}s)", "OK")
    except Exception:
        _log(f"  Tab '{nombre}' no existe, creando...", "INFO")
        ws = spreadsheet.add_worksheet(title=nombre, rows=600, cols=30)
        _log(f"  Tab '{nombre}' creado ({time.time()-t:.1f}s)", "OK")

    if df.empty:
        _log(f"  DataFrame vacio — escribiendo placeholder", "WARN")
        ws.update("A1", [["Sin datos"]])
        return ws

    n_filas = len(df)
    n_cols  = len(df.columns)
    _log(f"  Datos: {n_filas} filas x {n_cols} columnas", "INFO")
    _log(f"  Columnas: {', '.join(df.columns.tolist())}", "INFO")

    # Serializar a string
    _log(f"  Serializando datos...", "STEP")
    df_str = df.copy().fillna("")
    for col in df_str.columns:
        df_str[col] = df_str[col].astype(str).str.replace("nan", "")
    valores = [df_str.columns.tolist()] + df_str.values.tolist()

    # Escribir datos
    _log(f"  Escribiendo {len(valores)} filas en Sheets API...", "STEP")
    t = time.time()
    ws.update(valores, "A1")
    _log(f"  Datos escritos ({time.time()-t:.1f}s)", "OK")

    ultima_col = _col_letra(n_cols)

    # Formato header
    _log(f"  Aplicando formato al header (A1:{ultima_col}1)...", "STEP")
    t = time.time()
    ws.format(f"A1:{ultima_col}1", {
        "textFormat": {
            "bold": True,
            "foregroundColor": _COLOR_HEADER_TXT,
        },
        "backgroundColor": _COLOR_HEADER,
        "horizontalAlignment": "CENTER",
    })
    _log(f"  Header formateado ({time.time()-t:.1f}s)", "OK")

    # Colorear filas por nivel
    if col_nivel and col_nivel in df_str.columns:
        _log(f"  Coloreando filas por columna '{col_nivel}'...", "STEP")
        t = time.time()
        requests = []
        conteo = {}
        for i, nivel_val in enumerate(df_str[col_nivel]):
            color = _nivel_color(nivel_val)
            conteo[str(nivel_val)] = conteo.get(str(nivel_val), 0) + 1
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": i + 1,
                        "endRowIndex":   i + 2,
                        "startColumnIndex": 0,
                        "endColumnIndex": n_cols,
                    },
                    "cell": {
                        "userEnteredFormat": {"backgroundColor": color}
                    },
                    "fields": "userEnteredFormat.backgroundColor",
                }
            })
        if requests:
            spreadsheet.batch_update({"requests": requests})
        resumen_colores = ", ".join(f"{k}:{v}" for k, v in sorted(conteo.items()))
        _log(f"  Filas coloreadas ({time.time()-t:.1f}s) — {resumen_colores}", "OK")

    # Congelar fila header
    _log(f"  Congelando fila 1...", "STEP")
    t = time.time()
    spreadsheet.batch_update({"requests": [{
        "updateSheetProperties": {
            "properties": {
                "sheetId": ws.id,
                "gridProperties": {"frozenRowCount": 1},
            },
            "fields": "gridProperties.frozenRowCount",
        }
    }]})
    _log(f"  Fila congelada ({time.time()-t:.1f}s)", "OK")

    return ws


def _col_letra(n: int) -> str:
    """Convierte numero de columna (1-based) a letra: 1=A, 26=Z, 27=AA..."""
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


# ─────────────────────────────────────────────────────────────
# Queries a Railway PostgreSQL
# ─────────────────────────────────────────────────────────────

def _query_dashboard():
    sql = """
        SELECT
            a.ticker,
            COALESCE(act.sector, 'Sin sector')       AS sector,
            a.alert_nivel                            AS nivel,
            a.alert_score                            AS score,
            ROUND(a.ml_prob_ganancia::numeric*100,1) AS ml_pct,
            a.ml_modelo_usado                        AS modelo,
            ROUND(a.precio_cierre::numeric, 2)       AS precio,
            a.pa_ev1,
            a.pa_ev2,
            a.pa_ev3,
            a.pa_ev4,
            a.scan_fecha                             AS fecha_scan
        FROM (
            SELECT DISTINCT ON (ticker) *
            FROM alertas_scanner
            ORDER BY ticker, scan_fecha DESC
        ) a
        LEFT JOIN activos act ON act.ticker = a.ticker
        ORDER BY
            CASE a.alert_nivel
                WHEN 'COMPRA_FUERTE' THEN 1
                WHEN 'COMPRA'        THEN 2
                WHEN 'NEUTRAL'       THEN 3
                WHEN 'VENTA'         THEN 4
                WHEN 'VENTA_FUERTE'  THEN 5
                ELSE 6
            END,
            a.alert_score DESC
    """
    return query_df(sql)


def _query_analisis():
    sql = """
        SELECT
            i.ticker,
            COALESCE(act.sector, 'Sin sector')        AS sector,
            ROUND(pd.close::numeric, 2)               AS precio,
            ROUND(i.rsi14::numeric, 1)                AS rsi14,
            ROUND(i.macd::numeric, 4)                 AS macd,
            ROUND(i.adx::numeric, 1)                  AS adx,
            ROUND(i.dist_sma21::numeric, 2)           AS dist_sma21_pct,
            ROUND(i.dist_sma50::numeric, 2)           AS dist_sma50_pct,
            ROUND(i.dist_sma200::numeric, 2)          AS dist_sma200_pct,
            ROUND(i.vol_relativo::numeric, 2)         AS vol_relativo,
            ROUND(i.atr14::numeric, 2)                AS atr14,
            i.fecha                                   AS fecha_indicador
        FROM (
            SELECT DISTINCT ON (ticker) ticker, fecha,
                   rsi14, macd, adx, dist_sma21, dist_sma50, dist_sma200,
                   vol_relativo, atr14
            FROM indicadores_tecnicos
            ORDER BY ticker, fecha DESC
        ) i
        LEFT JOIN activos act ON act.ticker = i.ticker
        JOIN (
            SELECT DISTINCT ON (ticker) ticker, close
            FROM precios_diarios
            ORDER BY ticker, fecha DESC
        ) pd ON pd.ticker = i.ticker
        ORDER BY i.ticker
    """
    return query_df(sql)


def _query_historial(dias: int = 90):
    fecha_desde = (date.today() - timedelta(days=dias)).isoformat()
    sql = """
        SELECT
            scan_fecha                                       AS fecha,
            ticker,
            alert_nivel                                      AS nivel,
            alert_score                                      AS score,
            ROUND(ml_prob_ganancia::numeric*100, 1)         AS ml_pct,
            ROUND(precio_cierre::numeric, 2)                AS precio_cierre,
            ROUND(retorno_1d_real::numeric*100,  2)         AS retorno_1d_pct,
            ROUND(retorno_5d_real::numeric*100,  2)         AS retorno_5d_pct,
            ROUND(retorno_20d_real::numeric*100, 2)         AS retorno_20d_pct,
            CASE WHEN verificado THEN 'si' ELSE 'no' END    AS verificado
        FROM alertas_scanner
        WHERE scan_fecha >= :fecha_desde
        ORDER BY scan_fecha DESC, alert_score DESC
    """
    return query_df(sql, params={"fecha_desde": fecha_desde})


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def exportar_a_sheets():
    print(flush=True)
    print("=" * 60, flush=True)
    print("  GOOGLE SHEETS EXPORT", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)

    # ── Pre-checks ────────────────────────────────────────────
    _separador("Pre-checks")

    sheet_id = os.environ.get("GOOGLE_SHEETS_ID", "").strip()
    if not sheet_id:
        _log("GOOGLE_SHEETS_ID no configurado.", "SKIP")
        _log("Agregar en .env: GOOGLE_SHEETS_ID=tu_id_del_sheet", "INFO")
        return

    tiene_creds = (
        bool(os.environ.get("GOOGLE_SHEETS_KEY", "").strip())
        or _LOCAL_KEY_FILE.exists()
    )
    if not tiene_creds:
        _log("Credenciales de Google no encontradas.", "SKIP")
        _log(f"Guardar JSON del Service Account en: {_LOCAL_KEY_FILE}", "INFO")
        return

    _log("GOOGLE_SHEETS_ID: configurado", "OK")
    _log(f"Credenciales:     {'env var' if os.environ.get('GOOGLE_SHEETS_KEY') else _LOCAL_KEY_FILE.name}", "OK")

    # ── Conexion a Google Sheets ──────────────────────────────
    _separador("Conexion a Google Sheets")
    try:
        spreadsheet = _get_spreadsheet()
    except Exception as e:
        _log(f"Error conectando a Google Sheets: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)

    resultados = {}

    # ── Tab 1: Dashboard ──────────────────────────────────────
    _separador("Tab 1/3: Dashboard")
    try:
        _log("Consultando PostgreSQL (alertas_scanner)...", "STEP")
        t = time.time()
        df1 = _query_dashboard()
        _log(f"Query OK: {len(df1)} tickers en {time.time()-t:.1f}s", "OK")

        if not df1.empty:
            niveles = df1["nivel"].value_counts().to_dict()
            resumen = ", ".join(f"{k}:{v}" for k, v in niveles.items())
            _log(f"Distribucion niveles: {resumen}", "INFO")

        _escribir_tab(spreadsheet, "Dashboard", df1, col_nivel="nivel")
        _log(f"Tab 'Dashboard' actualizado: {len(df1)} tickers", "OK")
        resultados["Dashboard"] = "OK"

    except Exception as e:
        _log(f"FALLO en tab Dashboard: {e}", "ERROR")
        traceback.print_exc()
        resultados["Dashboard"] = f"ERROR: {e}"

    # ── Tab 2: Analisis Tecnico ───────────────────────────────
    _separador("Tab 2/3: Analisis Tecnico")
    try:
        _log("Consultando PostgreSQL (indicadores_tecnicos)...", "STEP")
        t = time.time()
        df2 = _query_analisis()
        _log(f"Query OK: {len(df2)} tickers en {time.time()-t:.1f}s", "OK")

        _escribir_tab(spreadsheet, "Analisis Tecnico", df2)
        _log(f"Tab 'Analisis Tecnico' actualizado: {len(df2)} tickers", "OK")
        resultados["Analisis Tecnico"] = "OK"

    except Exception as e:
        _log(f"FALLO en tab Analisis Tecnico: {e}", "ERROR")
        traceback.print_exc()
        resultados["Analisis Tecnico"] = f"ERROR: {e}"

    # ── Tab 3: Historial ──────────────────────────────────────
    _separador("Tab 3/3: Historial")
    try:
        _log("Consultando PostgreSQL (alertas_scanner 90 dias)...", "STEP")
        t = time.time()
        df3 = _query_historial(dias=90)
        _log(f"Query OK: {len(df3)} registros en {time.time()-t:.1f}s", "OK")

        if not df3.empty:
            verificados = (df3["verificado"] == "si").sum()
            _log(f"Alertas verificadas con retorno real: {verificados}/{len(df3)}", "INFO")

        _escribir_tab(spreadsheet, "Historial", df3, col_nivel="nivel")
        _log(f"Tab 'Historial' actualizado: {len(df3)} registros", "OK")
        resultados["Historial"] = "OK"

    except Exception as e:
        _log(f"FALLO en tab Historial: {e}", "ERROR")
        traceback.print_exc()
        resultados["Historial"] = f"ERROR: {e}"

    # ── Resumen final ─────────────────────────────────────────
    _separador("Resumen")
    total_seg = time.time() - _t0_global
    todos_ok = all(v == "OK" for v in resultados.values())

    for tab, estado in resultados.items():
        nivel_log = "OK" if estado == "OK" else "ERROR"
        _log(f"{tab:<20} {estado}", nivel_log)

    print(flush=True)
    _log(f"Tiempo total: {total_seg:.1f}s", "INFO")
    _log(f"Fecha:        {date.today().isoformat()}", "INFO")
    _log(f"URL:  https://docs.google.com/spreadsheets/d/{sheet_id}/edit", "INFO")
    print("=" * 60, flush=True)

    if not todos_ok:
        sys.exit(1)


if __name__ == "__main__":
    exportar_a_sheets()
