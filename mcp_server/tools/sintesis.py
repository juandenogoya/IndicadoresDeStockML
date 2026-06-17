"""
mcp_server/tools/sintesis.py
Tool de sintesis: une lectura tecnica (RSI/MACD diario + semanal) con
posicionamiento de opciones por plazo (PCR_vol, muros de OI = S/R) y el
sentimiento del sector, y aplica reglas de interpretacion.

Tool unica: get_ticker_sintesis(ticker)

Diferencia con get_ticker_overview:
  - overview = snapshot tecnico amplio (1 timeframe, opciones agregadas).
  - sintesis = lectura cruzada tecnico D+W x opciones por plazo x sector,
    con REGLAS que detectan confluencias y divergencias.

Plazos (homogeneo con config / estrategias FT):
  corto 1-14d, medio 15-45d, largo 46-90d.

Reglas: son heuristicas interpretativas (no señales de trading validadas).
Se computan en Python para no volcar numeros crudos al LLM.

Imports: solo funciones puras de src/utils (clasificacion_tecnica + weekly_tf,
ambos autonomos sin config ni DB). Cumple la regla del MCP de no importar
config/.env del proyecto. El semanal se computa AL VUELO desde precios_diarios
(weekly_tf), no se lee de indicadores_tecnicos_1w (congelada bajo Plan C).
"""

from datetime import date, timedelta
from decimal import Decimal

from mcp_server.db.pool import get_pool
from mcp_server.db.queries import (
    SQL_OVERVIEW_PERFIL,
    SQL_OVERVIEW_PRECIO,
    SQL_OVERVIEW_TECNICOS,
    SQL_SINTESIS_PRECIOS_SEMANAL,
    SQL_SINTESIS_PCR_PLAZO,
    SQL_SINTESIS_PCR_SECTOR_PLAZO,
)
from src.utils.clasificacion_tecnica import clasificar_rsi, clasificar_macd
from src.utils.weekly_tf import tecnico_semanal

SINTESIS_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

# Umbral de z-score sectorial para considerar el sentimiento "inusual"
_Z_SECTOR_INUSUAL = 1.5
# Distancia maxima (%) para considerar un muro de OI "cercano" al precio
_MURO_CERCANO_PCT = 3.0
# Dias de historia diaria a pedir para el semanal al vuelo. ~760d cubre los ~2
# anos que guarda precios_diarios -> el RSI/MACD semanal converge al mismo valor
# que el dashboard (que resamplea la historia completa); el warmup de MACD 26+9
# queda holgado.
_DIAS_HISTORIA_1W = 760


def _row_to_dict(row) -> dict:
    """asyncpg.Record -> dict JSON-serializable."""
    result = {}
    for key, val in row.items():
        if isinstance(val, Decimal):
            result[key] = float(val)
        elif isinstance(val, date):
            result[key] = val.isoformat()
        else:
            result[key] = val
    return result


def _bloque_tecnico(rsi, macd, macd_signal) -> dict:
    """Arma el bloque de un timeframe con valores + clasificacion."""
    return {
        "rsi":         float(rsi) if rsi is not None else None,
        "rsi_estado":  clasificar_rsi(rsi),
        "macd":        float(macd) if macd is not None else None,
        "macd_signal": float(macd_signal) if macd_signal is not None else None,
        "macd_estado": clasificar_macd(macd, macd_signal),
    }


def _muro_recalc(strike, oi, close, fuerza=None) -> dict | None:
    """
    Recalcula el muro de OI con el CLOSE diario real (no con el precio_sub del
    snapshot, que es el cierre del dia anterior y queda desfasado).

    dist_pct: negativo = muro DEBAJO del precio (actua como soporte),
              positivo = muro ARRIBA del precio (actua como resistencia).
    fuerza:   % de concentracion del OI del lado en este muro (muros v2). Alto =
              muro DOMINANTE; bajo = OI DISTRIBUIDO entre strikes. None si falta.
    """
    if strike is None or close is None:
        return None
    try:
        s = float(strike)
        c = float(close)
    except (TypeError, ValueError):
        return None
    if c == 0:
        return None
    dist = (s - c) / c * 100
    try:
        fz = round(float(fuerza), 1) if fuerza is not None else None
    except (TypeError, ValueError):
        fz = None
    return {
        "strike":   round(s, 2),
        "oi":       oi,
        "dist_pct": round(dist, 2),
        "posicion": "debajo" if dist < 0 else "arriba",
        "fuerza":   fz,
    }


def _bloque_plazo(row: dict, close) -> dict:
    """
    Una ventana de opciones del ticker: PCR + muros de OI.

    Muros v2 (OI COMBINADO call+put por strike):
      put_wall  = mayor OI total DEBAJO del precio (soporte).
      call_wall = mayor OI total ARRIBA del precio (resistencia).
    Cada muro trae `fuerza` (% de concentracion del OI del lado): alta = muro
    dominante; baja = OI distribuido. La distancia/posicion se recalcula con el
    close diario real para no arrastrar el desfase del precio_sub del snapshot.
    """
    return {
        "pcr_vol":      row.get("pcr_vol"),
        "pcr_oi":       row.get("pcr_oi"),
        "veredicto_oi": {"A": "Alcista", "B": "Bajista"}.get(row.get("veredicto_oi")),
        "put_wall":     _muro_recalc(row.get("soporte_strike"), row.get("soporte_oi"),
                                     close, row.get("soporte_fuerza")),
        "call_wall":    _muro_recalc(row.get("resistencia_strike"), row.get("resistencia_oi"),
                                     close, row.get("resistencia_fuerza")),
    }


def _generar_reglas(tecnico: dict, opciones: dict, sector: dict) -> list:
    """
    Heuristicas interpretativas que cruzan tecnico + opciones + sector.
    Devuelve lista de {tipo, mensaje}. NO son señales validadas.
    """
    reglas = []

    diario   = tecnico.get("diario") or {}
    semanal  = tecnico.get("semanal") or {}
    macd_d   = diario.get("macd_estado")
    macd_w   = semanal.get("macd_estado")
    rsi_d    = diario.get("rsi_estado")

    # 1. Confluencia / divergencia tecnica entre timeframes
    if macd_d and macd_w:
        if macd_d == macd_w == "Compra":
            reglas.append({"tipo": "tecnico", "mensaje": "MACD en Compra en diario Y semanal: momentum alcista alineado."})
        elif macd_d == macd_w == "Venta":
            reglas.append({"tipo": "tecnico", "mensaje": "MACD en Venta en diario Y semanal: momentum bajista alineado."})
        else:
            reglas.append({"tipo": "tecnico", "mensaje": f"MACD mixto entre timeframes (diario={macd_d}, semanal={macd_w}): señal poco confiable."})

    # 2. Sobrecompra/sobreventa con muro de OI cercano (posicion real vs close)
    medio = opciones.get("medio") or {}
    call_w = medio.get("call_wall")
    put_w  = medio.get("put_wall")
    # Resistencia real: call wall ARRIBA del precio (dist_pct > 0) y cercano
    if rsi_d == "Sobrecompra" and call_w and call_w.get("posicion") == "arriba" \
            and abs(call_w["dist_pct"]) <= _MURO_CERCANO_PCT:
        reglas.append({"tipo": "riesgo", "mensaje": f"RSI sobrecompra + call wall de OI a +{call_w['dist_pct']}% (strike {call_w['strike']}): poco recorrido / posible techo."})
    # Soporte real: put wall DEBAJO del precio (dist_pct < 0) y cercano
    if rsi_d == "Sobreventa" and put_w and put_w.get("posicion") == "debajo" \
            and abs(put_w["dist_pct"]) <= _MURO_CERCANO_PCT:
        reglas.append({"tipo": "oportunidad", "mensaje": f"RSI sobreventa + put wall de OI a {put_w['dist_pct']}% (strike {put_w['strike']}): zona de rebote potencial."})

    # 3. Divergencia ticker (tecnico alcista) vs sector (PCR sectorial inusual)
    sec_medio = sector.get("medio") or {}
    z_sec = sec_medio.get("pcr_vol_sector_zscore")
    if macd_d == "Compra" and z_sec is not None and z_sec >= _Z_SECTOR_INUSUAL:
        reglas.append({"tipo": "divergencia", "mensaje": f"Ticker alcista (MACD compra) PERO el sector muestra PCR_vol z-score {z_sec} (cobertura/bajismo inusual): el sector no acompaña, cautela."})

    # 4. Confluencia bajista total
    if macd_d == "Venta" and (medio.get("veredicto_oi") == "Bajista"):
        reglas.append({"tipo": "tecnico", "mensaje": "MACD diario en Venta + PCR_OI del plazo medio Bajista: presion bajista confirmada por opciones."})

    if not reglas:
        reglas.append({"tipo": "info", "mensaje": "Sin confluencias ni divergencias destacadas con las reglas actuales."})
    return reglas


async def get_ticker_sintesis(ticker: str) -> dict:
    """
    Sintesis cruzada de un ticker: tecnico (RSI/MACD diario y semanal) +
    opciones por plazo (PCR_vol, muros de OI como soporte/resistencia) +
    sentimiento del sector por plazo, con reglas de interpretacion.

    Diseñada para responder:
      "Como se ve AAPL juntando tecnico, opciones y sector?"
      "El sector acompaña la tendencia de NVDA?"
      "Hay un techo de opciones cerca en MSFT?"

    Plazos: corto (1-14d), medio (15-45d), largo (46-90d).

    Args:
        ticker: Simbolo (ej "AAPL"). Case-sensitive.

    Returns:
        {ticker, fecha_analisis, perfil, precio, tecnico{diario,semanal},
         opciones_plazo{corto,medio,largo}, sector_plazo{...}, reglas[]}
        Secciones con {"disponible": False, ...} si faltan datos.
    """
    result: dict = {
        "ticker":         ticker,
        "fecha_analisis": date.today().isoformat(),
    }

    try:
        pool = await get_pool()
    except Exception as exc:
        return {**result, "error": f"Error de conexion para '{ticker}': {exc}"}

    try:
        async with pool.acquire() as conn:
            # 1. Perfil (requerido) -> sector
            perfil_row = await conn.fetchrow(SQL_OVERVIEW_PERFIL, ticker)
            if perfil_row is None:
                return {**result, "error": f"Ticker '{ticker}' no encontrado. Verificar con list_tickers()."}
            sector = perfil_row["sector"]
            result["perfil"] = {"sector": sector, "industry": perfil_row["industry"]}

            # 2. Precio (ultimo cierre)
            try:
                precio_raw = await conn.fetch(SQL_OVERVIEW_PRECIO, ticker)
                if precio_raw:
                    hoy = _row_to_dict(precio_raw[0])
                    result["precio"] = {
                        "fecha": hoy.get("fecha"),
                        "close": hoy.get("close") or hoy.get("adj_close"),
                    }
                else:
                    result["precio"] = {"disponible": False, "razon": "sin datos de precio"}
            except Exception as exc:
                result["precio"] = {"error": str(exc)}

            # close real para recalcular distancias de muros (no usar precio_sub)
            close_real = result.get("precio", {}).get("close")

            # 3. Tecnico diario + semanal
            tecnico: dict = {}
            try:
                d_raw = await conn.fetchrow(SQL_OVERVIEW_TECNICOS, ticker)
                if d_raw:
                    d = _row_to_dict(d_raw)
                    tecnico["diario"] = {"fecha": d.get("fecha"), **_bloque_tecnico(d.get("rsi14"), d.get("macd"), d.get("macd_signal"))}
                else:
                    tecnico["diario"] = {"disponible": False}
                # Semanal AL VUELO: resample W-FRI de los precios diarios + RSI/MACD
                # (src.utils.weekly_tf). Reemplaza la lectura de indicadores_tecnicos_1w
                # (tabla congelada, pipeline semanal deprecado bajo Plan C).
                desde_1w = date.today() - timedelta(days=_DIAS_HISTORIA_1W)
                precios_1w = await conn.fetch(SQL_SINTESIS_PRECIOS_SEMANAL, ticker, desde_1w)
                sem = tecnico_semanal(precios_1w) if precios_1w else {}
                if sem:
                    fecha_sem = sem.get("fecha")
                    tecnico["semanal"] = {
                        "fecha": fecha_sem.isoformat() if isinstance(fecha_sem, date) else fecha_sem,
                        **_bloque_tecnico(sem.get("rsi"), sem.get("macd"), sem.get("macd_signal")),
                    }
                else:
                    tecnico["semanal"] = {
                        "disponible": False,
                        "razon": "historia diaria insuficiente para el semanal al vuelo",
                    }
            except Exception as exc:
                tecnico = {"error": str(exc)}
            result["tecnico"] = tecnico

            # 4. Opciones por plazo (PCR + muros S/R)
            opciones: dict = {}
            try:
                plazo_raw = await conn.fetch(SQL_SINTESIS_PCR_PLAZO, ticker)
                if plazo_raw:
                    primera = _row_to_dict(plazo_raw[0])
                    opciones["precio_sub"] = primera.get("precio_sub")
                    opciones["close_usado"] = close_real
                    for r in plazo_raw:
                        rd = _row_to_dict(r)
                        opciones[rd["ventana"]] = _bloque_plazo(rd, close_real)
                else:
                    opciones = {"disponible": False, "razon": "sin datos de opciones por plazo"}
            except Exception as exc:
                opciones = {"error": str(exc)}
            result["opciones_plazo"] = opciones

            # 5. Sector por plazo
            sector_plazo: dict = {}
            try:
                if sector:
                    sec_raw = await conn.fetch(SQL_SINTESIS_PCR_SECTOR_PLAZO, sector)
                    if sec_raw:
                        for r in sec_raw:
                            rd = _row_to_dict(r)
                            sector_plazo[rd["ventana"]] = {
                                "pcr_vol_sector":        rd.get("pcr_vol_sector"),
                                "pcr_oi_sector":         rd.get("pcr_oi_sector"),
                                "veredicto_oi":          {"A": "Alcista", "B": "Bajista"}.get(rd.get("veredicto_oi")),
                                "pcr_vol_sector_zscore": rd.get("pcr_vol_sector_zscore"),
                                "n_tickers":             rd.get("n_tickers"),
                            }
                    else:
                        sector_plazo = {"disponible": False, "razon": "sin datos sectoriales"}
                else:
                    sector_plazo = {"disponible": False, "razon": "ticker sin sector"}
            except Exception as exc:
                sector_plazo = {"error": str(exc)}
            result["sector_plazo"] = sector_plazo

            # 6. Reglas de interpretacion
            try:
                result["reglas"] = _generar_reglas(
                    tecnico if isinstance(tecnico, dict) else {},
                    opciones if isinstance(opciones, dict) else {},
                    sector_plazo if isinstance(sector_plazo, dict) else {},
                )
            except Exception as exc:
                result["reglas"] = [{"tipo": "error", "mensaje": str(exc)}]

    except Exception as exc:
        result["error_critico"] = f"Error de conexion: {exc}"

    return result
