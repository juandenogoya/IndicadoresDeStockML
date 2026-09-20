"""
ft_analisis_entradas.py
Analisis de las ENTRADAS de TECH_SECTOR_v1 (docs/forward_testing/ANALISIS_ENTRADAS.md).
Solo LEE (DB local). Guarda cada corrida en un directorio con fecha:
reportes/analisis_entradas/AAAAMMDD_<etiqueta>/  (log + CSV + parametros.json).

POR QUE EXISTE
    Con el reparto (5 por sector) y la salida de TECH_SECTOR_v1 fijos, la entrada queda
    como unica variable. Este es el PASO 0: antes de armar ninguna grilla, saber en que
    estados cae el universo y con que frecuencia. Un peso sobre un estado que ocurre 50
    veces en cinco anios no es una palanca.

SECCIONES (--seccion; por defecto todas)
    fidelidad   guard: la regla escrita sin pesos tiene que ser IDENTICA a
                `calcular_score_tecnico(...) >= 4,0` fila por fila, y las condiciones
                derivadas de las zonas tienen que reproducir las de scoring.py.
                Ademas audita `dist_sma*` guardada contra la recomputada.
    estados     distribucion de los estados por zonas, concentracion de la masa, y la
                INFLUENCIA de cada condicion (en que fraccion de filas voltearla cambia
                la decision) -- la medida directa del aporte propio de SMA21.
    cruces      tablas cruzadas SMA50 x SMA21 (cuanta masa queda fuera de la diagonal)
                y distancia x RSI (si las dos condiciones se pelean).
    seleccion   los juegos de pesos pre-declarados, ?eligen tickers distintos? Top-5 por
                sector y rueda contra la v1 (PROXY: arma el top de cero cada rueda).
                Ademas agrupa los juegos que producen la MISMA regla booleana.

LO QUE ESTE PASO NO RESPONDE
    Nada sobre rendimiento. No mide retorno, no simula cartera y no compara reglas: eso
    es el paso 1, con pre-registro escrito antes de correrlo. Aca solo se cuenta.

Uso:
    python scripts/forward_testing/ft_analisis_entradas.py --etiqueta paso0
    python scripts/forward_testing/ft_analisis_entradas.py --seccion estados
"""

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# LOCAL es la fuente de verdad: con DATABASE_URL seteada get_engine cae a Railway.
os.environ.pop("DATABASE_URL", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.database import query_df  # noqa: E402
from src.utils import ft_entradas as fe  # noqa: E402

# Mismo periodo que el backtest pre-registrado de docs/estructura_velas.md sec. 9.3.
DESDE_DEFAULT = "2021-09-01"
DIR_BASE = os.path.join(ROOT, "reportes", "analisis_entradas")
MIN_CASOS_ESTADO = 200          # por debajo, un estado no sostiene un ponderador propio


class Salida:
    """print + archivo. Un log por corrida."""

    def __init__(self, ruta):
        self.f = open(ruta, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg)
        self.f.write(str(msg) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _pct(parte, total):
    return 0.0 if not total else round(100.0 * parte / total, 2)


# --- datos --------------------------------------------------------------------

def cargar_datos(desde: str, hasta: str) -> pd.DataFrame:
    """OHLCV + indicadores del universo activo. Una fila por (ticker, rueda)."""
    sql = """
        SELECT i.ticker, i.fecha, a.sector,
               p.close,
               i.sma21, i.sma50, i.sma200,
               i.dist_sma21 AS dist_sma21_db,
               i.dist_sma50 AS dist_sma50_db,
               i.dist_sma200 AS dist_sma200_db,
               i.rsi14, i.macd, i.macd_signal
        FROM indicadores_tecnicos i
        JOIN precios_diarios p ON p.ticker = i.ticker AND p.fecha = i.fecha
        JOIN activos a ON a.ticker = i.ticker AND a.activo = TRUE
        WHERE i.fecha BETWEEN :desde AND :hasta
        ORDER BY i.ticker, i.fecha
    """
    df = query_df(sql, params={"desde": desde, "hasta": hasta})
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    for col in ("close", "sma21", "sma50", "sma200", "rsi14", "macd", "macd_signal",
                "dist_sma21_db", "dist_sma50_db", "dist_sma200_db"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Distancias recomputadas (sin el redondeo de la tabla), zonas y condiciones."""
    df = df.copy()
    for n in (21, 50, 200):
        sma = df[f"sma{n}"]
        df[f"dist_sma{n}"] = np.where(sma > 0, (df["close"] - sma) / sma * 100.0, np.nan)

    df["z_s50"] = fe.clasificar_serie(df["dist_sma50"], fe.CORTES_S50, fe.ZONAS_S50)
    df["z_s21"] = fe.clasificar_serie(df["dist_sma21"], fe.CORTES_S21, fe.ZONAS_S21)
    df["z_macd"] = np.where(df["macd"].notna() & df["macd_signal"].notna(),
                            np.where(df["macd"] > df["macd_signal"], "MACD_UP",
                                     "MACD_DOWN"), None)
    df["z_rsi"] = fe.clasificar_serie(df["rsi14"], fe.CORTES_RSI, fe.ZONAS_RSI)

    df["c_sma200"] = df["dist_sma200"] > 0
    df["c_sma50"] = df["dist_sma50"] > 0
    df["c_sma21"] = df["dist_sma21"] > 0
    df["c_macd"] = df["z_macd"] == "MACD_UP"
    df["c_rsi"] = df["z_rsi"] == "RSI_IN"

    dos_de_tres = (df["c_sma21"].astype(int) + df["c_macd"].astype(int)
                   + df["c_rsi"].astype(int)) >= 2
    df["entra"] = df["c_sma200"] & df["c_sma50"] & dos_de_tres
    df["clave"] = (df["z_s50"].astype(str) + "|" + df["z_s21"].astype(str) + "|"
                   + df["z_macd"].astype(str) + "|" + df["z_rsi"].astype(str))
    return df


# --- seccion: fidelidad -------------------------------------------------------

def seccion_fidelidad(df: pd.DataFrame, out: Salida) -> dict:
    out("")
    out("=" * 78)
    out("FIDELIDAD -- la maquinaria nueva tiene que reproducir la regla del bot")
    out("=" * 78)

    completas = df.dropna(subset=["close", "sma21", "sma50", "sma200", "rsi14",
                                  "macd", "macd_signal"])
    out(f"Filas con todos los insumos: {len(completas):,} de {len(df):,}")

    dif_regla = dif_cond = dif_vector = 0
    for r in completas.itertuples(index=False):
        fila = {"close": r.close, "sma21": r.sma21, "sma50": r.sma50,
                "sma200": r.sma200, "rsi14": r.rsi14, "macd": r.macd,
                "macd_signal": r.macd_signal}
        cond = fe.condiciones_desde_fila(fila)
        if fe.cumple_entrada_por_score(cond) != fe.regla_v1_booleana(cond):
            dif_regla += 1
        estado = fe.estado_desde_fila(fila)
        if fe.condiciones_desde_estado(estado) != cond:
            dif_cond += 1
        if bool(r.entra) != fe.cumple_entrada_por_score(cond):
            dif_vector += 1

    out("")
    out(f"  score >= 4,0  vs  SMA200 y SMA50 y 2 de 3 : {dif_regla} diferencias")
    out(f"  condiciones desde ZONAS vs desde scoring  : {dif_cond} diferencias")
    out(f"  columna 'entra' (vectorizada) vs escalar  : {dif_vector} diferencias")

    # Auditoria de dist_sma* guardada contra la recomputada.
    audit = OrderedDict()
    for n, cortes, etiquetas in ((50, fe.CORTES_S50, fe.ZONAS_S50),
                                 (21, fe.CORTES_S21, fe.ZONAS_S21)):
        calc, guard = completas[f"dist_sma{n}"], completas[f"dist_sma{n}_db"]
        material = int((np.abs(calc - guard) > 1e-4).sum())
        z_calc = fe.clasificar_serie(calc, cortes, etiquetas)
        z_guard = fe.clasificar_serie(guard, cortes, etiquetas)
        cambia = int((z_calc != z_guard).sum())
        audit[f"sma{n}"] = {"material": material, "cambia_zona": cambia}
        out(f"  dist_sma{n}: {material} diferencias materiales (>1e-4), "
            f"{cambia} filas cambian de zona por el redondeo")

    ok = (dif_regla == 0 and dif_cond == 0 and dif_vector == 0)
    out("")
    out("  VEREDICTO: " + ("OK, la regla se reproduce exacto"
                           if ok else "FALLA -- no seguir sin entender la diferencia"))
    return {"dif_regla": dif_regla, "dif_cond": dif_cond, "dif_vector": dif_vector,
            "audit_dist": audit, "ok": ok}


# --- seccion: estados ---------------------------------------------------------

def _influencia(df: pd.DataFrame) -> "OrderedDict[str, float]":
    """Fraccion de filas donde voltear una condicion cambia la decision de entrada.

    Es la medida directa del aporte de cada condicion a la regla booleana: una
    condicion que nunca es pivotal no decide nada, por mas peso que tenga.
    """
    c200, c50 = df["c_sma200"], df["c_sma50"]
    c21, cmacd, crsi = df["c_sma21"], df["c_macd"], df["c_rsi"]
    n_otras = c21.astype(int) + cmacd.astype(int) + crsi.astype(int)

    # Una de las tres es pivotal cuando exactamente una de las OTRAS dos se cumple.
    piv = OrderedDict()
    piv["sma200"] = (c50 & (n_otras >= 2)).mean()
    piv["sma50"] = (c200 & (n_otras >= 2)).mean()
    piv["sma21"] = (c200 & c50 & ((cmacd.astype(int) + crsi.astype(int)) == 1)).mean()
    piv["macd"] = (c200 & c50 & ((c21.astype(int) + crsi.astype(int)) == 1)).mean()
    piv["rsi"] = (c200 & c50 & ((c21.astype(int) + cmacd.astype(int)) == 1)).mean()
    return OrderedDict((k, round(100.0 * float(v), 2)) for k, v in piv.items())


def seccion_estados(df: pd.DataFrame, out: Salida, dir_corrida: str) -> dict:
    out("")
    out("=" * 78)
    out("ESTADOS -- donde cae el universo")
    out("=" * 78)

    total = len(df)
    pasa = df[df["c_sma200"]]
    out(f"Filas totales            : {total:,}")
    out(f"Pasan el filtro SMA200   : {len(pasa):,} ({_pct(len(pasa), total)}%)")
    out(f"Cumplen la entrada actual: {int(df['entra'].sum()):,} "
        f"({_pct(int(df['entra'].sum()), total)}% del total, "
        f"{_pct(int(df['entra'].sum()), len(pasa))}% de las que pasan el filtro)")
    out(f"Tickers                  : {df['ticker'].nunique()}")
    out(f"Ruedas                   : {df['fecha'].nunique()}")

    # Influencia de cada condicion.
    out("")
    out("INFLUENCIA -- en que % de las filas voltear la condicion cambia la decision")
    inf = _influencia(df)
    for cond, valor in inf.items():
        out(f"  {cond:<8}: {valor:>6.2f}%")
    out("")
    out("  (una condicion que casi nunca es pivotal no decide, tenga el peso que tenga)")

    # Distribucion de estados sobre las filas que pasan el filtro.
    g = (pasa.groupby("clave")
         .agg(casos=("clave", "size"), tickers=("ticker", "nunique"),
              entra=("entra", "sum"))
         .sort_values("casos", ascending=False))
    g["pct"] = (100.0 * g["casos"] / len(pasa)).round(3)
    g["pct_acum"] = g["pct"].cumsum().round(2)
    g["entra"] = g["entra"].astype(int)

    n_90 = int((g["pct_acum"] <= 90).sum()) + 1
    n_99 = int((g["pct_acum"] <= 99).sum()) + 1
    chicos = g[g["casos"] < MIN_CASOS_ESTADO]

    out("")
    out(f"Estados observados        : {len(g)} de {len(fe.estados_posibles())} posibles")
    out(f"Concentran el 90% de masa : {n_90} estados")
    out(f"Concentran el 99% de masa : {n_99} estados")
    out(f"Con menos de {MIN_CASOS_ESTADO} casos   : {len(chicos)} estados "
        f"({_pct(int(chicos['casos'].sum()), len(pasa))}% de la masa)")

    out("")
    out("TOP 15 estados (filas que pasan el filtro SMA200):")
    out(f"  {'estado':<42} {'casos':>8} {'%':>6} {'%acum':>7} {'tick':>5} {'entra':>7}")
    for clave, r in g.head(15).iterrows():
        out(f"  {clave:<42} {int(r['casos']):>8,} {r['pct']:>6.2f} "
            f"{r['pct_acum']:>7.2f} {int(r['tickers']):>5} {int(r['entra']):>7,}")

    g.to_csv(os.path.join(dir_corrida, "estados.csv"), encoding="utf-8")
    return {"filas": total, "pasan_filtro": len(pasa), "entran": int(df["entra"].sum()),
            "estados_observados": len(g), "estados_90pct": n_90, "estados_99pct": n_99,
            "estados_chicos": len(chicos), "influencia_pct": dict(inf)}


# --- seccion: cruces ----------------------------------------------------------

def _imprimir_cruce(tabla: pd.DataFrame, out: Salida, titulo: str):
    out("")
    out(titulo)
    ancho = max(14, max(len(str(c)) for c in tabla.columns) + 1)
    out("  " + " " * 16 + "".join(f"{str(c):>{ancho}}" for c in tabla.columns))
    for idx, fila in tabla.iterrows():
        out(f"  {str(idx):<16}" + "".join(f"{v:>{ancho},.0f}" for v in fila.values))


def seccion_cruces(df: pd.DataFrame, out: Salida, dir_corrida: str) -> dict:
    out("")
    out("=" * 78)
    out("CRUCES -- cuanto se solapan las condiciones entre si")
    out("=" * 78)
    pasa = df[df["c_sma200"]].copy()

    # 1. SMA50 x SMA21: cuanta masa queda fuera de la diagonal.
    orden50, orden21 = list(fe.ZONAS_S50), list(fe.ZONAS_S21)
    cruce = (pd.crosstab(pasa["z_s50"], pasa["z_s21"])
             .reindex(index=orden50, columns=orden21).fillna(0))
    _imprimir_cruce(cruce, out, "SMA50 (filas) x SMA21 (columnas), casos:")
    diag = sum(cruce.iloc[i, i] for i in range(len(orden50)))
    out("")
    out(f"  En la misma posicion relativa (diagonal): {_pct(diag, len(pasa))}%")

    # Concordancia binaria y casos donde SMA21 aporta algo distinto de SMA50.
    ac = (pasa["c_sma50"] == pasa["c_sma21"]).mean()
    solo21 = (pasa["c_sma21"] & ~pasa["c_sma50"]).mean()
    solo50 = (pasa["c_sma50"] & ~pasa["c_sma21"]).mean()
    out(f"  Concordancia binaria SMA50/SMA21        : {round(100 * float(ac), 2)}%")
    out(f"  Solo SMA21 (sobre 21, bajo 50)          : {round(100 * float(solo21), 2)}%")
    out(f"  Solo SMA50 (sobre 50, bajo 21)          : {round(100 * float(solo50), 2)}%")
    cruce.to_csv(os.path.join(dir_corrida, "cruce_s50_s21.csv"), encoding="utf-8")

    # 2. Distancia x RSI: se pelean?
    resumen_rsi = OrderedDict()
    for eje, orden, nombre in (("z_s21", orden21, "SMA21"), ("z_s50", orden50, "SMA50")):
        cruce_rsi = (pd.crosstab(pasa[eje], pasa["z_rsi"])
                     .reindex(index=orden, columns=list(fe.ZONAS_RSI)).fillna(0))
        _imprimir_cruce(cruce_rsi, out, f"{nombre} (filas) x RSI (columnas), casos:")
        pct_in = (100.0 * cruce_rsi["RSI_IN"] / cruce_rsi.sum(axis=1)).round(2)
        out("")
        out(f"  % con RSI en banda (45-68) por zona de {nombre}:")
        for zona_nombre, valor in pct_in.items():
            marca = "  <-- se pelean" if (zona_nombre.endswith("_ALTA")
                                          and valor < 50) else ""
            out(f"    {zona_nombre:<16}: {valor:>6.2f}%{marca}")
        resumen_rsi[nombre] = {k: (None if pd.isna(v) else float(v))
                               for k, v in pct_in.items()}
        cruce_rsi.to_csv(os.path.join(dir_corrida, f"cruce_{eje}_rsi.csv"),
                         encoding="utf-8")

    # 3. Por que lado falla el RSI.
    fuera = pasa[pasa["z_rsi"] != "RSI_IN"]
    low = int((fuera["z_rsi"] == "RSI_LOW").sum())
    high = int((fuera["z_rsi"] == "RSI_HIGH").sum())
    out("")
    out("RSI: por que lado falla (filas que pasan el filtro SMA200)")
    out(f"  Por debilidad (<45): {low:,} ({_pct(low, len(fuera))}% de las que fallan)")
    out(f"  Por fuerza   (>68): {high:,} ({_pct(high, len(fuera))}% de las que fallan)")

    return {"diagonal_s50_s21_pct": _pct(diag, len(pasa)),
            "concordancia_binaria_pct": round(100 * float(ac), 2),
            "solo_sma21_pct": round(100 * float(solo21), 2),
            "solo_sma50_pct": round(100 * float(solo50), 2),
            "rsi_in_por_zona": resumen_rsi,
            "rsi_falla_por_debilidad": low, "rsi_falla_por_fuerza": high}


# --- seccion: seleccion -------------------------------------------------------

TOPE_SECTOR = 5                 # lugares por sector de TECH_SECTOR_v1


def _seleccion(df: pd.DataFrame, pesos, umbral):
    """Top-N por (rueda, sector) con el ranking del bot: score DESC y, entre
    empatados, el orden en que vienen los candidatos -- que es alfabetico, porque la
    consulta del bot trae ORDER BY sector, ticker y el sort de Python es estable.

    PROXY, no simulacion: arma el top de cero cada rueda. La estrategia real mantiene
    posiciones y solo llena lugares vacios, asi que esto sobreestima cuanto cambia la
    operacion. Responde "?hay elecciones distintas?", no "?cuanto rinde?".
    """
    s = np.where(df["c_sma200"],
                 pesos[0] * df["c_sma50"] + pesos[1] * df["c_sma21"]
                 + pesos[2] * df["c_macd"] + pesos[3] * df["c_rsi"], 0.0)
    mask = df["c_sma200"].values & (s >= umbral) & (s > 0)
    cal = df.loc[mask].copy()
    cal["score"] = s[mask]
    cal = cal.sort_values(["fecha", "sector", "score"], ascending=[True, True, False],
                          kind="mergesort")
    top = cal.groupby(["fecha", "sector"], sort=False).head(TOPE_SECTOR)
    return set(zip(top["fecha"], top["ticker"])), cal


def seccion_seleccion(df: pd.DataFrame, out: Salida, dir_corrida: str) -> dict:
    out("")
    out("=" * 78)
    out("SELECCION -- los ponderadores, ?eligen tickers distintos?")
    out("=" * 78)
    out("Top-5 por sector y rueda. PROXY: arma el top de cero cada rueda (la estrategia")
    out("real mantiene posiciones). Responde si hay elecciones distintas, no cuanto rinden.")

    # Duplicados: dos juegos con el mismo conjunto de estados son la misma estrategia.
    grupos = fe.reglas_distintas()
    out("")
    out(f"Juegos pre-declarados: {len(fe.JUEGOS_PESOS)}  ->  "
        f"reglas booleanas distintas: {len(grupos)}")
    for estados, nombres in grupos.items():
        if len(nombres) > 1:
            out(f"  DUPLICADOS (misma regla): {', '.join(nombres)}")

    df = df.dropna(subset=["close", "sma21", "sma50", "sma200", "rsi14",
                           "macd", "macd_signal"])
    df = df.sort_values(["fecha", "sector", "ticker"]).reset_index(drop=True)
    ruedas = df["fecha"].nunique()

    base, _ = _seleccion(df, *fe.JUEGOS_PESOS["v1"])
    out("")
    out(f"Base v1: {len(base):,} selecciones ({len(base) / ruedas:.1f} por rueda)")
    out("")
    out(f"  {'juego':<16} {'califican':>10} {'elegidos':>9} {'distintos':>10} "
        f"{'Jaccard':>8} {'niveles':>8} {'%tope':>7}")

    filas = []
    for nombre, (pesos, umbral) in fe.JUEGOS_PESOS.items():
        sel, cal = _seleccion(df, pesos, umbral)
        inter, union = len(sel & base), len(sel | base)
        vc = cal["score"].value_counts(normalize=True).sort_index()
        fila = {"juego": nombre, "pesos": list(pesos), "umbral": umbral,
                "califican": len(cal), "elegidos": len(sel),
                "distintos_de_v1": len(sel ^ base),
                "jaccard": round(inter / union, 4) if union else 1.0,
                "niveles_score": len(vc),
                "pct_en_el_tope": round(100 * float(vc.iloc[-1]), 2) if len(vc) else 0.0}
        filas.append(fila)
        out(f"  {nombre:<16} {fila['califican']:>10,} {fila['elegidos']:>9,} "
            f"{fila['distintos_de_v1']:>10,} {fila['jaccard']:>8.3f} "
            f"{fila['niveles_score']:>8} {fila['pct_en_el_tope']:>6.1f}%")

    out("")
    out("Lectura: 'distintos' mezcla DOS efectos -- entra mas gente (laxitud) y entra")
    out("otra gente (reordenamiento). Separarlos es condicion del paso 1: sin eso se")
    out("compara exposicion y se lo llama calidad de entrada.")
    out("'%tope' es la fraccion de candidatos empatados en el score maximo: es la parte")
    out("de la decision que hoy toma el orden ALFABETICO.")

    pd.DataFrame(filas).to_csv(os.path.join(dir_corrida, "seleccion.csv"),
                               index=False, encoding="utf-8")
    return {"ruedas": ruedas, "base_v1": len(base), "juegos": filas,
            "reglas_distintas": len(grupos)}


# --- main ---------------------------------------------------------------------

def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", default=DESDE_DEFAULT)
    ap.add_argument("--hasta", default=None, help="default: la ultima rueda con datos")
    ap.add_argument("--etiqueta", default="paso0")
    ap.add_argument("--seccion", default="todas",
                    choices=["todas", "fidelidad", "estados", "cruces",
                             "seleccion"])
    args = ap.parse_args()

    hasta = args.hasta or str(query_df(
        "SELECT MAX(fecha) AS f FROM indicadores_tecnicos")["f"].iloc[0])

    dir_corrida = os.path.join(DIR_BASE,
                               f"{date.today().strftime('%Y%m%d')}_{args.etiqueta}")
    os.makedirs(dir_corrida, exist_ok=True)
    out = Salida(os.path.join(dir_corrida, "log.txt"))

    out("ANALISIS DE ENTRADAS -- PASO 0 (solo cuenta; no mide rendimiento)")
    out(f"Periodo: {args.desde} -> {hasta}   Seccion: {args.seccion}")
    out(f"Cortes SMA50: {fe.CORTES_S50}   SMA21: {fe.CORTES_S21} "
        f"(factor {fe.FACTOR_S21:.3f})")

    df = preparar(cargar_datos(args.desde, hasta))
    out(f"Filas cargadas: {len(df):,}")

    resultados = {}
    if args.seccion in ("todas", "fidelidad"):
        resultados["fidelidad"] = seccion_fidelidad(df, out)
    if args.seccion in ("todas", "estados"):
        resultados["estados"] = seccion_estados(df, out, dir_corrida)
    if args.seccion in ("todas", "cruces"):
        resultados["cruces"] = seccion_cruces(df, out, dir_corrida)
    if args.seccion in ("todas", "seleccion"):
        resultados["seleccion"] = seccion_seleccion(df, out, dir_corrida)

    params = {"desde": args.desde, "hasta": hasta, "seccion": args.seccion,
              "cortes_s50": list(fe.CORTES_S50), "cortes_s21": list(fe.CORTES_S21),
              "factor_s21": round(fe.FACTOR_S21, 4),
              "min_casos_estado": MIN_CASOS_ESTADO, "tope_sector": TOPE_SECTOR,
              "juegos_pesos": {k: {"pesos": list(v[0]), "umbral": v[1]}
                               for k, v in fe.JUEGOS_PESOS.items()},
              "rsi_min": fe.RSI_MIN, "rsi_max": fe.RSI_MAX,
              "umbral_entrada": fe.SCORE_ENTRADA_V1,
              "git_commit": _git_commit(), "corrida": date.today().isoformat(),
              "resultados": resultados}
    with open(os.path.join(dir_corrida, "parametros.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False, default=str)

    out("")
    out(f"Resultados en {dir_corrida}")
    out.close()


if __name__ == "__main__":
    main()
