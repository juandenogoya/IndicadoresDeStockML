"""
explore_narrativo.py
Script exploratorio: dado un ticker, consulta indicadores de la DB
y llama a Claude Haiku para generar un analisis narrativo en español.

Uso:
    python scripts/explore_narrativo.py AAPL
    python scripts/explore_narrativo.py RIVN
"""

import sys
import os
import anthropic

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.database import query_df


def obtener_datos_ticker(ticker: str) -> dict | None:
    """Extrae indicadores del ultimo dia disponible para un ticker."""
    df = query_df("""
        SELECT
            it.ticker,
            it.fecha,
            pd.close                            AS precio,
            pd.close - LAG(pd.close) OVER (
                PARTITION BY pd.ticker ORDER BY pd.fecha
            )                                   AS cambio_abs,
            ROUND(
                (pd.close - LAG(pd.close) OVER (
                    PARTITION BY pd.ticker ORDER BY pd.fecha
                )) / NULLIF(LAG(pd.close) OVER (
                    PARTITION BY pd.ticker ORDER BY pd.fecha
                ), 0) * 100, 2
            )                                   AS cambio_pct,
            it.rsi14,
            it.macd,
            it.macd_signal,
            it.macd_hist,
            it.sma21,
            it.sma50,
            it.sma200,
            it.dist_sma21,
            it.dist_sma50,
            it.dist_sma200,
            it.atr14,
            it.bb_upper,
            it.bb_lower,
            it.adx,
            it.vol_relativo,
            a.sector
        FROM indicadores_tecnicos it
        JOIN precios_diarios pd
            ON pd.ticker = it.ticker AND pd.fecha = it.fecha
        JOIN activos a
            ON a.ticker = it.ticker
        WHERE it.ticker = :ticker
        ORDER BY it.fecha DESC
        LIMIT 2
    """, {"ticker": ticker.upper()})

    if df.empty:
        return None

    row = df.iloc[0]

    # Rango 52 semanas desde precios_diarios
    rango = query_df("""
        SELECT
            MIN(low)  AS min_52w,
            MAX(high) AS max_52w
        FROM precios_diarios
        WHERE ticker = :ticker
          AND fecha >= CURRENT_DATE - INTERVAL '52 weeks'
    """, {"ticker": ticker.upper()})

    # Tendencia semanal
    semanal = query_df("""
        SELECT
            it1w.rsi14    AS rsi_1w,
            it1w.macd_hist AS macd_hist_1w,
            it1w.sma21    AS sma21_1w,
            it1w.sma50    AS sma50_1w
        FROM indicadores_tecnicos_1w it1w
        WHERE it1w.ticker = :ticker
        ORDER BY it1w.fecha DESC
        LIMIT 1
    """, {"ticker": ticker.upper()})

    data = {
        "ticker":       row["ticker"],
        "fecha":        str(row["fecha"]),
        "precio":       float(row["precio"]),
        "cambio_pct":   float(row["cambio_pct"]) if row["cambio_pct"] is not None else 0.0,
        "sector":       row["sector"] or "N/A",
        # Diario
        "rsi14":        float(row["rsi14"]) if row["rsi14"] is not None else None,
        "macd":         float(row["macd"]) if row["macd"] is not None else None,
        "macd_signal":  float(row["macd_signal"]) if row["macd_signal"] is not None else None,
        "macd_hist":    float(row["macd_hist"]) if row["macd_hist"] is not None else None,
        "sma21":        float(row["sma21"]) if row["sma21"] is not None else None,
        "sma50":        float(row["sma50"]) if row["sma50"] is not None else None,
        "sma200":       float(row["sma200"]) if row["sma200"] is not None else None,
        "dist_sma50":   float(row["dist_sma50"]) if row["dist_sma50"] is not None else None,
        "dist_sma200":  float(row["dist_sma200"]) if row["dist_sma200"] is not None else None,
        "atr14":        float(row["atr14"]) if row["atr14"] is not None else None,
        "bb_upper":     float(row["bb_upper"]) if row["bb_upper"] is not None else None,
        "bb_lower":     float(row["bb_lower"]) if row["bb_lower"] is not None else None,
        "adx":          float(row["adx"]) if row["adx"] is not None else None,
        "vol_relativo": float(row["vol_relativo"]) if row["vol_relativo"] is not None else None,
    }

    if not rango.empty:
        data["min_52w"] = float(rango.iloc[0]["min_52w"]) if rango.iloc[0]["min_52w"] is not None else None
        data["max_52w"] = float(rango.iloc[0]["max_52w"]) if rango.iloc[0]["max_52w"] is not None else None

    if not semanal.empty:
        r1w = semanal.iloc[0]
        data["rsi_1w"]       = float(r1w["rsi_1w"]) if r1w["rsi_1w"] is not None else None
        data["macd_hist_1w"] = float(r1w["macd_hist_1w"]) if r1w["macd_hist_1w"] is not None else None

    return data


def generar_narrativo(data: dict) -> str:
    """Llama a Claude Haiku con los indicadores y devuelve el analisis narrativo."""

    # Descripcion de posicion vs SMAs
    sma_desc = []
    if data.get("sma21") and data["precio"] > data["sma21"]:
        sma_desc.append("sobre SMA21")
    elif data.get("sma21"):
        sma_desc.append("bajo SMA21")
    if data.get("sma50") and data["precio"] > data["sma50"]:
        sma_desc.append("sobre SMA50")
    elif data.get("sma50"):
        sma_desc.append("bajo SMA50")
    if data.get("sma200") and data["precio"] > data["sma200"]:
        sma_desc.append("sobre SMA200")
    elif data.get("sma200"):
        sma_desc.append("bajo SMA200")

    macd_dir = "alcista" if (data.get("macd_hist") or 0) > 0 else "bajista"
    macd_dir_1w = "alcista" if (data.get("macd_hist_1w") or 0) > 0 else "bajista"

    contexto = f"""Ticker: {data['ticker']} | Sector: {data['sector']}
Fecha: {data['fecha']}
Precio: ${data['precio']:.2f} ({data['cambio_pct']:+.2f}%)

Indicadores diarios:
- RSI14: {data.get('rsi14', 'N/A')}
- MACD histograma: {data.get('macd_hist', 'N/A'):.4f} ({macd_dir})
- Posicion vs medias: {', '.join(sma_desc) if sma_desc else 'N/A'}
- Distancia SMA50: {data.get('dist_sma50', 'N/A'):.2f}%
- Distancia SMA200: {data.get('dist_sma200', 'N/A'):.2f}%
- ATR14: ${data.get('atr14', 'N/A'):.2f}
- BB superior: ${data.get('bb_upper', 'N/A'):.2f} | BB inferior: ${data.get('bb_lower', 'N/A'):.2f}
- ADX: {data.get('adx', 'N/A')}
- Volumen relativo (vs media 20d): {data.get('vol_relativo', 'N/A'):.2f}x

Rango 52 semanas: ${data.get('min_52w', 'N/A'):.2f} - ${data.get('max_52w', 'N/A'):.2f}

Indicadores semanales:
- RSI14 semanal: {data.get('rsi_1w', 'N/A')}
- MACD histograma semanal: {macd_dir_1w}
"""

    prompt = f"""Eres un analista tecnico. Con base en los siguientes indicadores, escribe un parrafo conciso de 3-4 oraciones en espanol (sin tildes, solo ASCII) que describa:
1. El estado tecnico actual (momentum, tendencia)
2. Niveles clave a vigilar
3. Confluencia entre timeframes (diario vs semanal)

No inventes datos. Basa el analisis solo en los indicadores proporcionados. Tono directo y profesional.

{contexto}"""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY no configurada"

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def formatear_senal(data: dict) -> str:
    """Genera la linea de señal basada en RSI + MACD + posicion vs SMAs."""
    score = 0
    if data.get("rsi14", 50) > 50:
        score += 1
    if (data.get("macd_hist") or 0) > 0:
        score += 1
    if data.get("sma50") and data["precio"] > data["sma50"]:
        score += 1
    if (data.get("macd_hist_1w") or 0) > 0:
        score += 1

    if score >= 4:
        return "Señal de Compra Fuerte"
    elif score >= 3:
        return "Señal de Compra"
    elif score <= 1:
        return "Señal de Venta"
    else:
        return "Neutral"


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/explore_narrativo.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    print(f"\nConsultando datos para {ticker}...")

    data = obtener_datos_ticker(ticker)
    if data is None:
        print(f"No se encontraron datos para {ticker} en la DB.")
        sys.exit(1)

    senal = formatear_senal(data)
    cambio_signo = "+" if data["cambio_pct"] >= 0 else ""

    print(f"\n{'='*60}")
    print(f"{ticker}  ${data['precio']:.2f} ({cambio_signo}{data['cambio_pct']:.2f}%)  ---  {senal}")
    print(f"{'='*60}")
    print(f"RSI14: {data.get('rsi14', 'N/A')} | MACD: {'alcista' if (data.get('macd_hist') or 0) > 0 else 'bajista'}")
    print(f"ATR14: ${data.get('atr14', 'N/A'):.2f} | Vol relativo: {data.get('vol_relativo', 'N/A'):.2f}x")
    if data.get("min_52w") and data.get("max_52w"):
        print(f"Rango 52w: ${data['min_52w']:.2f} - ${data['max_52w']:.2f}")
    print(f"\nGenerando analisis narrativo con Claude Haiku...")

    narrativo = generar_narrativo(data)

    print(f"\nConfluencia tecnica:")
    print(narrativo)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
