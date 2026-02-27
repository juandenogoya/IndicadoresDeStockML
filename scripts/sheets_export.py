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
    # Niveles de alerta (Dashboard / Historial)
    "COMPRA_FUERTE": {"red": 0.13, "green": 0.55, "blue": 0.13},  # verde oscuro
    "COMPRA":        {"red": 0.72, "green": 0.96, "blue": 0.72},  # verde claro
    "NEUTRAL":       {"red": 1.00, "green": 1.00, "blue": 1.00},  # blanco
    "VENTA":         {"red": 1.00, "green": 0.71, "blue": 0.71},  # rojo claro
    "VENTA_FUERTE":  {"red": 0.86, "green": 0.20, "blue": 0.20},  # rojo oscuro
    # Tipo vela (tab Velas)
    "Alcista":       {"red": 0.72, "green": 0.96, "blue": 0.72},  # verde claro
    "Bajista":       {"red": 1.00, "green": 0.71, "blue": 0.71},  # rojo claro
    # Resultado operaciones (tab BT Trades)
    "GANANCIA":      {"red": 0.72, "green": 0.96, "blue": 0.72},  # verde claro
    "PERDIDA":       {"red": 1.00, "green": 0.71, "blue": 0.71},  # rojo claro
    "NEUTRO":        {"red": 1.00, "green": 1.00, "blue": 1.00},  # blanco
    # Accion recomendada (tab Conclusiones)
    "PRIORIDAD":     {"red": 0.13, "green": 0.55, "blue": 0.13},  # verde oscuro
    "ESTUDIAR":      {"red": 0.72, "green": 0.96, "blue": 0.72},  # verde claro
    "MONITOREAR":    {"red": 1.00, "green": 1.00, "blue": 0.75},  # amarillo claro
    "OBSERVAR":      {"red": 1.00, "green": 1.00, "blue": 1.00},  # blanco
    "EVITAR":        {"red": 1.00, "green": 0.71, "blue": 0.71},  # rojo claro
    # Tipo de paso (tab Nuevo Activo)
    "CMD":           {"red": 0.87, "green": 0.94, "blue": 1.00},  # azul claro
    "ARCHIVO":       {"red": 1.00, "green": 0.97, "blue": 0.82},  # amarillo claro
    "GIT":           {"red": 0.96, "green": 0.88, "blue": 0.77},  # naranja claro
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
    # DISTINCT ON (date(scan_fecha), ticker): un registro por dia por ticker,
    # el mas reciente (ya garantizado en DB por el DELETE previo al INSERT,
    # pero lo dejamos aqui como respaldo para datos historicos ya existentes).
    sql = """
        SELECT
            fecha, ticker, nivel, score, ml_pct,
            precio_cierre, retorno_1d_pct, retorno_5d_pct, retorno_20d_pct,
            verificado
        FROM (
            SELECT DISTINCT ON (date(scan_fecha), ticker)
                scan_fecha                                       AS fecha,
                ticker,
                alert_nivel                                      AS nivel,
                alert_score                                      AS score,
                ROUND(ml_prob_ganancia::numeric*100, 1)         AS ml_pct,
                ROUND(precio_cierre::numeric, 2)                AS precio_cierre,
                ROUND(retorno_1d_real::numeric,  2)             AS retorno_1d_pct,
                ROUND(retorno_5d_real::numeric,  2)             AS retorno_5d_pct,
                ROUND(retorno_20d_real::numeric, 2)             AS retorno_20d_pct,
                CASE WHEN verificado THEN 'si' ELSE 'no' END    AS verificado
            FROM alertas_scanner
            WHERE scan_fecha >= :fecha_desde
            ORDER BY date(scan_fecha) DESC, ticker, scan_fecha DESC
        ) sub
        ORDER BY fecha DESC, score DESC
    """
    return query_df(sql, params={"fecha_desde": fecha_desde})


def _query_velas():
    """
    Ultimos 5 dias de estructura de velas para todos los tickers.
    Calcula: tipo, cuerpo_pct, sombras, cierre_rango, patron.
    """
    sql_precios = """
        SELECT ticker, fecha, open, high, low, close, volume
        FROM (
            SELECT ticker, fecha, open, high, low, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
            FROM precios_diarios
        ) t
        WHERE rn <= 5
        ORDER BY ticker, fecha DESC
    """
    df_p = query_df(sql_precios)
    if df_p.empty:
        return df_p

    # Patrones de velas (tabla features_precio_accion)
    sql_pat = """
        SELECT ticker, fecha,
               COALESCE(patron_hammer,         0) AS patron_hammer,
               COALESCE(patron_engulfing_bull,  0) AS patron_engulfing_bull,
               COALESCE(patron_engulfing_bear,  0) AS patron_engulfing_bear,
               COALESCE(patron_shooting_star,   0) AS patron_shooting_star
        FROM (
            SELECT ticker, fecha,
                   patron_hammer, patron_engulfing_bull,
                   patron_engulfing_bear, patron_shooting_star,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
            FROM features_precio_accion
        ) t
        WHERE rn <= 5
    """
    try:
        df_pat = query_df(sql_pat)
    except Exception:
        df_pat = pd.DataFrame()

    # Volumen relativo
    sql_vol = """
        SELECT ticker, fecha, vol_relativo
        FROM (
            SELECT ticker, fecha, vol_relativo,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fecha DESC) AS rn
            FROM indicadores_tecnicos
        ) t
        WHERE rn <= 5
    """
    df_vol = query_df(sql_vol)

    # Merge
    df = df_p.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    if not df_pat.empty:
        df_pat["fecha"] = pd.to_datetime(df_pat["fecha"])
        df = df.merge(df_pat, on=["ticker", "fecha"], how="left")
    else:
        for c in ["patron_hammer", "patron_engulfing_bull",
                  "patron_engulfing_bear", "patron_shooting_star"]:
            df[c] = 0

    if not df_vol.empty:
        df_vol["fecha"] = pd.to_datetime(df_vol["fecha"])
        df = df.merge(df_vol[["ticker", "fecha", "vol_relativo"]],
                      on=["ticker", "fecha"], how="left")
    else:
        df["vol_relativo"] = float("nan")

    # Calculos numericos
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rango"]          = (df["high"] - df["low"]).replace(0, float("nan"))
    df["cuerpo"]         = (df["close"] - df["open"]).abs()
    df["sombra_sup"]     = df["high"] - df[["close", "open"]].max(axis=1)
    df["sombra_inf"]     = df[["close", "open"]].min(axis=1) - df["low"]
    df["cuerpo_pct"]     = (df["cuerpo"]     / df["rango"] * 100).round(1)
    df["sombra_sup_pct"] = (df["sombra_sup"] / df["rango"] * 100).round(1)
    df["sombra_inf_pct"] = (df["sombra_inf"] / df["rango"] * 100).round(1)
    df["cierre_rango"]   = ((df["close"] - df["low"]) / df["rango"] * 100).round(1)
    df["tipo"]           = df.apply(
        lambda r: "Alcista" if r["close"] >= r["open"] else "Bajista", axis=1
    )

    def _lbl_cuerpo(v):
        if pd.isna(v): return "n/d"
        if v > 70: return f"Grande ({v:.0f}%)"
        if v > 40: return f"Medio ({v:.0f}%)"
        if v > 20: return f"Pequeno ({v:.0f}%)"
        return f"Doji ({v:.0f}%)"

    def _patron(r):
        if r.get("patron_hammer",         0): return "Hammer"
        if r.get("patron_engulfing_bull",  0): return "Engulfing Bull"
        if r.get("patron_engulfing_bear",  0): return "Engulfing Bear"
        if r.get("patron_shooting_star",   0): return "Shooting Star"
        return "-"

    df["cuerpo_lbl"] = df["cuerpo_pct"].apply(_lbl_cuerpo)
    df["patron"]     = df.apply(_patron, axis=1)

    result = pd.DataFrame({
        "ticker":           df["ticker"],
        "fecha":            df["fecha"].dt.strftime("%Y-%m-%d"),
        "close":            df["close"].round(2),
        "tipo":             df["tipo"],
        "cuerpo":           df["cuerpo_lbl"],
        "sombra_sup_pct":   df["sombra_sup_pct"],
        "sombra_inf_pct":   df["sombra_inf_pct"],
        "cierre_rango_pct": df["cierre_rango"],
        "patron":           df["patron"],
        "vol_relativo":     df["vol_relativo"].round(2),
    })
    return result.sort_values(["ticker", "fecha"],
                              ascending=[True, False]).reset_index(drop=True)


def _query_conclusiones():
    """
    Resumen accionable para swing trading.
    Sintetiza: señal ML + indicadores + vela del dia -> accion recomendada.
    Acciones: PRIORIDAD > ESTUDIAR > MONITOREAR > OBSERVAR > EVITAR
    """
    # Señal ML (ultima por ticker)
    sql_alertas = """
        SELECT
            a.ticker,
            COALESCE(act.sector, 'Sin sector')        AS sector,
            a.alert_nivel                             AS nivel,
            a.alert_score                             AS score,
            ROUND(a.ml_prob_ganancia::numeric*100, 1) AS ml_pct,
            ROUND(a.precio_cierre::numeric, 2)        AS precio,
            a.scan_fecha                              AS fecha_scan
        FROM (
            SELECT DISTINCT ON (ticker) *
            FROM alertas_scanner
            ORDER BY ticker, scan_fecha DESC
        ) a
        LEFT JOIN activos act ON act.ticker = a.ticker
    """
    df_a = query_df(sql_alertas)

    # Indicadores tecnicos (ultimo por ticker)
    sql_ind = """
        SELECT DISTINCT ON (ticker)
            ticker,
            ROUND(rsi14::numeric, 1)        AS rsi14,
            ROUND(adx::numeric, 1)          AS adx,
            ROUND(dist_sma200::numeric, 2)  AS dist_sma200_pct,
            ROUND(vol_relativo::numeric, 2) AS vol_relativo
        FROM indicadores_tecnicos
        ORDER BY ticker, fecha DESC
    """
    df_ind = query_df(sql_ind)

    # Ultima vela + patron del dia
    sql_vela = """
        SELECT
            p.ticker,
            CASE WHEN p.close >= p.open THEN 'Alcista' ELSE 'Bajista' END AS tipo_vela,
            COALESCE(
                CASE
                    WHEN fp.patron_hammer         = 1 THEN 'Hammer'
                    WHEN fp.patron_engulfing_bull = 1 THEN 'Engulfing Bull'
                    WHEN fp.patron_engulfing_bear = 1 THEN 'Engulfing Bear'
                    WHEN fp.patron_shooting_star  = 1 THEN 'Shooting Star'
                    ELSE '-'
                END, '-'
            ) AS patron_vela
        FROM (
            SELECT DISTINCT ON (ticker) ticker, fecha, open, close
            FROM precios_diarios
            ORDER BY ticker, fecha DESC
        ) p
        LEFT JOIN (
            SELECT DISTINCT ON (ticker) ticker, fecha,
                   patron_hammer, patron_engulfing_bull,
                   patron_engulfing_bear, patron_shooting_star
            FROM features_precio_accion
            ORDER BY ticker, fecha DESC
        ) fp ON fp.ticker = p.ticker AND fp.fecha = p.fecha
    """
    try:
        df_v = query_df(sql_vela)
    except Exception:
        df_v = pd.DataFrame()

    # Merge
    df = df_a.copy()
    if not df_ind.empty:
        df = df.merge(df_ind, on="ticker", how="left")
    if not df_v.empty:
        df = df.merge(df_v, on="ticker", how="left")
    else:
        df["tipo_vela"]  = "-"
        df["patron_vela"] = "-"

    # Accion recomendada basada en señales
    def _accion(r):
        nivel  = str(r.get("nivel",  "") or "")
        ml_pct = float(r.get("ml_pct", 0) or 0)
        if nivel == "COMPRA_FUERTE" and ml_pct >= 60:
            return "PRIORIDAD"
        if nivel == "COMPRA_FUERTE" or (nivel == "COMPRA" and ml_pct >= 55):
            return "ESTUDIAR"
        if nivel == "COMPRA" or (nivel == "NEUTRAL" and ml_pct >= 60):
            return "MONITOREAR"
        if nivel in ("VENTA", "VENTA_FUERTE"):
            return "EVITAR"
        return "OBSERVAR"

    df["accion"] = df.apply(_accion, axis=1)

    # Orden por accion + score
    _orden = {"PRIORIDAD": 1, "ESTUDIAR": 2, "MONITOREAR": 3,
              "OBSERVAR": 4, "EVITAR": 5}
    df["_ord"] = df["accion"].map(_orden).fillna(6)
    df = df.sort_values(["_ord", "score"], ascending=[True, False])
    df = df.drop(columns=["_ord"]).reset_index(drop=True)

    cols = ["ticker", "sector", "accion", "nivel", "score", "ml_pct",
            "precio", "rsi14", "adx", "dist_sma200_pct", "vol_relativo",
            "tipo_vela", "patron_vela", "fecha_scan"]
    return df[[c for c in cols if c in df.columns]]


def _query_bt_resumen():
    """
    Resumen del backtesting PA por ticker para la estrategia champion EV4/SV1.
    Periodo: FECHA_INICIO_BT (2023-01-01) -> hoy, segmento FULL.
    Incluye todos los tickers activos aunque tengan 0 operaciones.
    """
    sql = """
        SELECT
            a.sector,
            a.ticker,
            COUNT(o.id)                                                       AS total_ops,
            COALESCE(SUM(CASE WHEN o.resultado='GANANCIA' THEN 1 ELSE 0 END), 0) AS ganadoras,
            COALESCE(SUM(CASE WHEN o.resultado='PERDIDA'  THEN 1 ELSE 0 END), 0) AS perdedoras,
            COALESCE(SUM(CASE WHEN o.resultado='NEUTRO'   THEN 1 ELSE 0 END), 0) AS neutros,
            ROUND(COALESCE(
                AVG(CASE WHEN o.resultado='GANANCIA' THEN 100.0 ELSE 0.0 END), 0
            )::numeric, 1)                                                    AS wr_pct,
            ROUND(COALESCE(AVG(o.retorno_pct), 0)::numeric, 2)               AS ret_prom,
            ROUND(COALESCE(
                SUM(CASE WHEN o.retorno_pct > 0 THEN o.retorno_pct ELSE 0 END) /
                NULLIF(ABS(SUM(CASE WHEN o.retorno_pct < 0 THEN o.retorno_pct ELSE 0 END)), 0)
            , 0)::numeric, 2)                                                 AS profit_factor
        FROM activos a
        LEFT JOIN operaciones_bt_pa o
               ON a.ticker              = o.ticker
              AND o.estrategia_entrada  = 'EV4'
              AND o.estrategia_salida   = 'SV1'
              AND o.segmento            = 'FULL'
        WHERE a.activo = TRUE
        GROUP BY a.sector, a.ticker
        ORDER BY a.sector, profit_factor DESC NULLS LAST, a.ticker
    """
    return query_df(sql)


def _query_bt_trades():
    """
    Log de operaciones individuales del champion EV4/SV1 FULL.
    Incluye sector del ticker via join con activos.
    """
    sql = """
        SELECT
            a.sector,
            o.ticker,
            o.fecha_entrada::text                        AS fecha_entrada,
            ROUND(o.precio_entrada::numeric, 2)          AS precio_entrada,
            o.fecha_salida::text                         AS fecha_salida,
            ROUND(o.precio_salida::numeric, 2)           AS precio_salida,
            o.dias_posicion                              AS dias,
            ROUND(o.retorno_pct::numeric, 2)             AS retorno_pct,
            o.resultado,
            o.motivo_salida                              AS motivo
        FROM operaciones_bt_pa o
        JOIN activos a ON a.ticker = o.ticker
        WHERE o.estrategia_entrada = 'EV4'
          AND o.estrategia_salida  = 'SV1'
          AND o.segmento           = 'FULL'
        ORDER BY a.sector, o.ticker, o.fecha_entrada
    """
    return query_df(sql)


# ─────────────────────────────────────────────────────────────
# Procedimiento: agregar nuevo activo (contenido estatico)
# ─────────────────────────────────────────────────────────────

def _datos_nuevo_activo() -> pd.DataFrame:
    """
    Tabla de referencia con los pasos para incorporar un nuevo ticker.
    Contenido estatico: no lee de la DB.
    Tipos de paso: CMD (azul), ARCHIVO (amarillo), GIT (naranja).
    """
    pasos = [
        {
            "paso":    "1",
            "tipo":    "CMD",
            "accion":  "Incorporar ticker en la DB",
            "comando": "python scripts/19_incorporar_ticker.py TICKER --forzar",
            "notas":   "Descarga 5 anios yfinance -> precios_diarios + indicadores_tecnicos + asigna modelo ML champion",
        },
        {
            "paso":    "2",
            "tipo":    "CMD",
            "accion":  "Calcular scoring tecnico",
            "comando": "python scripts/03_calcular_scoring.py",
            "notas":   "Prerequisito obligatorio del backtesting PA (script 15 verifica que scoring_tecnico no este vacio)",
        },
        {
            "paso":    "3",
            "tipo":    "ARCHIVO",
            "accion":  "Agregar ticker a config.py",
            "comando": "src/utils/config.py  ->  dict ACTIVOS",
            "notas":   "Sector: usar el mismo que tiene en tabla activos (auto-detectado por yfinance). Si es sector nuevo, agregar como nueva key",
        },
        {
            "paso":    "4",
            "tipo":    "CMD",
            "accion":  "Recalcular features precio/accion",
            "comando": "python scripts/09_calcular_precio_accion.py",
            "notas":   "Lee todos los tickers en precios_diarios, hace upsert automatico. No hay que filtrar por ticker",
        },
        {
            "paso":    "5",
            "tipo":    "CMD",
            "accion":  "Recalcular features market structure",
            "comando": "python scripts/12_calcular_market_structure.py",
            "notas":   "Lee todos los tickers en precios_diarios, hace upsert automatico. No hay que filtrar por ticker",
        },
        {
            "paso":    "6",
            "tipo":    "CMD",
            "accion":  "Re-ejecutar backtesting PA",
            "comando": "python scripts/15_backtesting_pa.py",
            "notas":   "TRUNCATE + re-simula EV1-4 x SV1-4 desde 2023-01-01 para todos los tickers de ALL_TICKERS en config.py",
        },
        {
            "paso":    "7",
            "tipo":    "CMD",
            "accion":  "Actualizar Google Sheets",
            "comando": "python scripts/sheets_export.py",
            "notas":   "Opcional localmente. El cron diario (GitHub Actions L-V 17:30 ET) lo ejecuta automaticamente",
        },
        {
            "paso":    "8",
            "tipo":    "GIT",
            "accion":  "Commit y push a GitHub",
            "comando": "git add src/utils/config.py && git commit -m 'feat: agregar TICKER (Sector)' && git push origin main",
            "notas":   "Obligatorio para que GitHub Actions (cron) y Streamlit Cloud usen el nuevo ticker",
        },
        {
            "paso":    "---",
            "tipo":    "ARCHIVO",
            "accion":  "CONSULTAR SECTOR en tabla activos",
            "comando": "SELECT ticker, sector FROM activos WHERE activo=TRUE ORDER BY sector, ticker",
            "notas":   "Ejecutar en psql o DBeaver para ver el sector asignado antes de editar config.py (paso 3)",
        },
        {
            "paso":    "---",
            "tipo":    "ARCHIVO",
            "accion":  "VERIFICAR features cargadas",
            "comando": "SELECT ticker, COUNT(*) n_pa FROM features_precio_accion GROUP BY ticker ORDER BY ticker",
            "notas":   "Si n_pa=0 para algun ticker despues del paso 4, revisar que precios_diarios tiene datos validos (close>0)",
        },
    ]
    return pd.DataFrame(pasos)


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
    _separador("Tab 1/7: Dashboard")
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
    _separador("Tab 2/7: Analisis Tecnico")
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
    _separador("Tab 3/7: Historial")
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

    # ── Tab 4: Velas ─────────────────────────────────────────
    _separador("Tab 4/7: Velas")
    try:
        _log("Consultando PostgreSQL (precios + patrones + vol)...", "STEP")
        t = time.time()
        df4 = _query_velas()
        _log(f"Query OK: {len(df4)} filas en {time.time()-t:.1f}s", "OK")

        if not df4.empty:
            n_alc = (df4["tipo"] == "Alcista").sum()
            n_baj = (df4["tipo"] == "Bajista").sum()
            patrones = df4[df4["patron"] != "-"]["patron"].value_counts().to_dict()
            _log(f"Alcistas: {n_alc} | Bajistas: {n_baj} | Patrones: {patrones}", "INFO")

        _escribir_tab(spreadsheet, "Velas", df4, col_nivel="tipo")
        _log(f"Tab 'Velas' actualizado: {len(df4)} filas ({len(df4)//5} tickers x 5 dias)", "OK")
        resultados["Velas"] = "OK"

    except Exception as e:
        _log(f"FALLO en tab Velas: {e}", "ERROR")
        traceback.print_exc()
        resultados["Velas"] = f"ERROR: {e}"

    # ── Tab 5: Conclusiones ───────────────────────────────────
    _separador("Tab 5/7: Conclusiones")
    try:
        _log("Construyendo resumen accionable para swing trading...", "STEP")
        t = time.time()
        df5 = _query_conclusiones()
        _log(f"Query OK: {len(df5)} tickers en {time.time()-t:.1f}s", "OK")

        if not df5.empty:
            acciones = df5["accion"].value_counts().to_dict()
            resumen  = ", ".join(f"{k}:{v}" for k, v in acciones.items())
            _log(f"Distribucion acciones: {resumen}", "INFO")
            prioritarios = df5[df5["accion"] == "PRIORIDAD"]["ticker"].tolist()
            if prioritarios:
                _log(f"PRIORIDAD: {', '.join(prioritarios)}", "OK")

        _escribir_tab(spreadsheet, "Conclusiones", df5, col_nivel="accion")
        _log(f"Tab 'Conclusiones' actualizado: {len(df5)} tickers", "OK")
        resultados["Conclusiones"] = "OK"

    except Exception as e:
        _log(f"FALLO en tab Conclusiones: {e}", "ERROR")
        traceback.print_exc()
        resultados["Conclusiones"] = f"ERROR: {e}"

    # ── Tab 6: BT Resumen ─────────────────────────────────────
    _separador("Tab 6/8: BT Resumen")
    try:
        _log("Consultando PostgreSQL (backtesting EV4/SV1 por ticker)...", "STEP")
        t = time.time()
        df6 = _query_bt_resumen()
        _log(f"Query OK: {len(df6)} tickers en {time.time()-t:.1f}s", "OK")

        if not df6.empty:
            con_ops = (df6["total_ops"] > 0).sum()
            sin_ops = (df6["total_ops"] == 0).sum()
            avg_wr  = df6[df6["total_ops"] > 0]["wr_pct"].mean()
            _log(f"Tickers con ops: {con_ops} | Sin ops: {sin_ops} | WR% promedio: {avg_wr:.1f}%", "INFO")

        _escribir_tab(spreadsheet, "BT Resumen", df6)
        _log(f"Tab 'BT Resumen' actualizado: {len(df6)} tickers (EV4/SV1 FULL)", "OK")
        resultados["BT Resumen"] = "OK"

    except Exception as e:
        _log(f"FALLO en tab BT Resumen: {e}", "ERROR")
        traceback.print_exc()
        resultados["BT Resumen"] = f"ERROR: {e}"

    # ── Tab 7: BT Trades ──────────────────────────────────────
    _separador("Tab 7/8: BT Trades")
    try:
        _log("Consultando PostgreSQL (operaciones EV4/SV1 individuales)...", "STEP")
        t = time.time()
        df7 = _query_bt_trades()
        _log(f"Query OK: {len(df7)} operaciones en {time.time()-t:.1f}s", "OK")

        if not df7.empty:
            ganadoras = (df7["resultado"] == "GANANCIA").sum()
            perdedoras = (df7["resultado"] == "PERDIDA").sum()
            wr = ganadoras / len(df7) * 100
            ret_prom = df7["retorno_pct"].mean()
            _log(f"G:{ganadoras} | P:{perdedoras} | WR:{wr:.1f}% | RetProm:{ret_prom:.2f}%", "INFO")

        _escribir_tab(spreadsheet, "BT Trades", df7, col_nivel="resultado")
        _log(f"Tab 'BT Trades' actualizado: {len(df7)} operaciones (EV4/SV1 2023->hoy)", "OK")
        resultados["BT Trades"] = "OK"

    except Exception as e:
        _log(f"FALLO en tab BT Trades: {e}", "ERROR")
        traceback.print_exc()
        resultados["BT Trades"] = f"ERROR: {e}"

    # ── Tab 8: Nuevo Activo ───────────────────────────────────
    _separador("Tab 8/8: Nuevo Activo")
    try:
        df8 = _datos_nuevo_activo()
        _escribir_tab(spreadsheet, "Nuevo Activo", df8, col_nivel="tipo")
        _log(f"Tab 'Nuevo Activo' actualizado: {len(df8)} pasos", "OK")
        resultados["Nuevo Activo"] = "OK"

    except Exception as e:
        _log(f"FALLO en tab Nuevo Activo: {e}", "ERROR")
        traceback.print_exc()
        resultados["Nuevo Activo"] = f"ERROR: {e}"

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
