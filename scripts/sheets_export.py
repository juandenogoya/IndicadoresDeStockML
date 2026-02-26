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
import pathlib
import pandas as pd
from datetime import date, timedelta

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
# Colores RGB por nivel de alerta
# ─────────────────────────────────────────────────────────────

NIVEL_COLOR = {
    "COMPRA_FUERTE": {"red": 0.13, "green": 0.55, "blue": 0.13},  # verde oscuro
    "COMPRA":        {"red": 0.72, "green": 0.96, "blue": 0.72},  # verde claro
    "NEUTRAL":       {"red": 1.00, "green": 1.00, "blue": 1.00},  # blanco
    "VENTA":         {"red": 1.00, "green": 0.71, "blue": 0.71},  # rojo claro
    "VENTA_FUERTE":  {"red": 0.86, "green": 0.20, "blue": 0.20},  # rojo oscuro
}

_COLOR_HEADER = {"red": 0.20, "green": 0.29, "blue": 0.49}   # azul oscuro
_COLOR_HEADER_TXT = {"red": 1.0, "green": 1.0, "blue": 1.0}  # blanco


def _nivel_color(nivel):
    return NIVEL_COLOR.get(str(nivel).strip() if nivel else "", {"red": 1.0, "green": 1.0, "blue": 1.0})


# ─────────────────────────────────────────────────────────────
# Conexion a Google Sheets
# ─────────────────────────────────────────────────────────────

def _get_spreadsheet():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        raise ImportError("Instalar: pip install gspread google-auth")

    sheet_id = os.environ.get("GOOGLE_SHEETS_ID", "").strip()
    if not sheet_id:
        raise ValueError(
            "GOOGLE_SHEETS_ID no configurado.\n"
            "  Local: agregar GOOGLE_SHEETS_ID=... en el archivo .env\n"
            "  GitHub: agregar Secret GOOGLE_SHEETS_ID"
        )

    # Prioridad 1: variable de entorno GOOGLE_SHEETS_KEY (GitHub Actions)
    creds_json = os.environ.get("GOOGLE_SHEETS_KEY", "").strip()
    if creds_json:
        creds_dict = json.loads(creds_json)

    # Prioridad 2: archivo local secrets/google_sheets_key.json (desarrollo)
    elif _LOCAL_KEY_FILE.exists():
        print(f"  [local] Usando credenciales desde: {_LOCAL_KEY_FILE}")
        with open(_LOCAL_KEY_FILE, "r", encoding="utf-8") as f:
            creds_dict = json.load(f)

    else:
        raise ValueError(
            "Credenciales de Google no encontradas.\n"
            "  Opcion A (GitHub): configurar Secret GOOGLE_SHEETS_KEY\n"
            f"  Opcion B (local):  guardar el JSON en {_LOCAL_KEY_FILE}"
        )

    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)


# ─────────────────────────────────────────────────────────────
# Escribir un DataFrame en un tab
# ─────────────────────────────────────────────────────────────

def _escribir_tab(spreadsheet, nombre: str, df: pd.DataFrame,
                  col_nivel: str = None):
    """
    Limpia el tab y escribe el DataFrame completo.
    Aplica formato al header y (si se indica) colores por nivel.
    """
    # Obtener o crear worksheet
    try:
        ws = spreadsheet.worksheet(nombre)
        ws.clear()
    except Exception:
        ws = spreadsheet.add_worksheet(title=nombre, rows=600, cols=30)

    if df.empty:
        ws.update("A1", [["Sin datos"]])
        return ws

    # Serializar: todo a string para evitar errores de tipo
    df_str = df.copy().fillna("")
    for col in df_str.columns:
        df_str[col] = df_str[col].astype(str).str.replace("nan", "")

    valores = [df_str.columns.tolist()] + df_str.values.tolist()
    ws.update("A1", valores)

    n_cols  = len(df.columns)
    n_filas = len(df)
    ultima_col = _col_letra(n_cols)

    # Formato header: fondo azul oscuro, texto blanco, negrita
    ws.format(f"A1:{ultima_col}1", {
        "textFormat": {
            "bold": True,
            "foregroundColor": _COLOR_HEADER_TXT,
        },
        "backgroundColor": _COLOR_HEADER,
        "horizontalAlignment": "CENTER",
    })

    # Colorear filas por nivel si corresponde
    if col_nivel and col_nivel in df_str.columns:
        col_idx = df_str.columns.tolist().index(col_nivel)
        requests = []
        for i, nivel_val in enumerate(df_str[col_nivel]):
            color = _nivel_color(nivel_val)
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": i + 1,   # +1 por el header
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

    # Congelar primera fila
    spreadsheet.batch_update({"requests": [{
        "updateSheetProperties": {
            "properties": {
                "sheetId": ws.id,
                "gridProperties": {"frozenRowCount": 1},
            },
            "fields": "gridProperties.frozenRowCount",
        }
    }]})

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
            ROUND(a.precio_entrada::numeric, 2)      AS precio,
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
            ROUND(precio_entrada::numeric, 2)               AS precio_entrada,
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
    print("\n=== Google Sheets Export ===")

    # Verificar que al menos el Sheet ID esté configurado
    if not os.environ.get("GOOGLE_SHEETS_ID", "").strip():
        print("  [SKIP] GOOGLE_SHEETS_ID no configurado. Saltando export.")
        return

    # Verificar credenciales: env var o archivo local
    tiene_creds = (
        bool(os.environ.get("GOOGLE_SHEETS_KEY", "").strip())
        or _LOCAL_KEY_FILE.exists()
    )
    if not tiene_creds:
        print(f"  [SKIP] Credenciales no encontradas.")
        print(f"         Guardar JSON del Service Account en: {_LOCAL_KEY_FILE}")
        return

    print("  Conectando a Google Sheets...")
    spreadsheet = _get_spreadsheet()
    print(f"  OK — spreadsheet: '{spreadsheet.title}'")

    # 1. Tab Dashboard
    print("  [1/3] Escribiendo tab 'Dashboard'...")
    df1 = _query_dashboard()
    _escribir_tab(spreadsheet, "Dashboard", df1, col_nivel="nivel")
    print(f"        {len(df1)} tickers.")

    # 2. Tab Analisis Tecnico
    print("  [2/3] Escribiendo tab 'Analisis Tecnico'...")
    df2 = _query_analisis()
    _escribir_tab(spreadsheet, "Analisis Tecnico", df2)
    print(f"        {len(df2)} tickers.")

    # 3. Tab Historial
    print("  [3/3] Escribiendo tab 'Historial'...")
    df3 = _query_historial(dias=90)
    _escribir_tab(spreadsheet, "Historial", df3, col_nivel="nivel")
    print(f"        {len(df3)} registros (90 dias).")

    sheet_id = os.environ.get("GOOGLE_SHEETS_ID", "")
    print(f"\n  Actualizado: {date.today().isoformat()}")
    print(f"  URL: https://docs.google.com/spreadsheets/d/{sheet_id}/edit")
    print("=" * 35)


if __name__ == "__main__":
    exportar_a_sheets()
