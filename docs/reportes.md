# Reportes / Infografías de análisis técnico
# Ultima actualizacion: 2026-05-22

## Que es

Modulo en `scripts/reports/` que genera material visual para compartir
análisis técnico (principalmente en X). Consume los datos del MCP server
(`get_ticker_overview` + `get_options_analysis`) y produce dos productos:

| Producto | Archivo | Para que |
|----------|---------|----------|
| **Infografía** | `<TICKER>_<FECHA>_ig.png` | Imagen unica para subir a X. Solo datos + reglas, SIN LLM |
| **Reporte PDF** | `<TICKER>_<FECHA>.pdf` + PNGs por pagina | Reporte detallado con narrativa del LLM. Archivo de referencia |

Ambos comparten el mismo origen de datos (el YAML) pero distinto destino.

## Decisiones de diseño (el "por que")

1. **X NO soporta PDF como adjunto.** Por eso todo termina en PNG. El PDF
   se conserva como archivo de referencia / para mandar por otros medios.
   La conversion PDF -> PNG usa **PyMuPDF** (pure Python, sin deps de sistema,
   a diferencia de pdf2image que necesita Poppler).

2. **WeasyPrint necesita GTK en Windows.** HTML+CSS -> PDF se hace con
   WeasyPrint. En Windows requiere el GTK3 runtime instalado (ver README del
   modulo). Es la unica dependencia de sistema.

3. **Datos y narrativa separados.** El reporte PDF mezcla datos (del MCP) con
   narrativa (interpretacion humana/LLM). Para no pelear con el escapado de
   YAML, la narrativa vive en un `.md` hermano del YAML (mismo nombre base).
   `make_report.py` lo lee automaticamente. Asi se edita markdown puro.

4. **La infografía NO usa LLM.** Es 100% datos + reglas determinísticas. Por
   eso `make_infografia.bat <TICKER>` corre de una sola vez (no necesita que
   un humano escriba la narrativa). Cero tokens, reproducible, consistente
   entre tickers.

5. **Preprocesador de markdown del LLM.** El output de Gemini CLI no es
   markdown estandar (prefijo `✦`, indentacion, bullets con `*`, secciones
   "1. Titulo"). `make_report.py` lo limpia antes de renderizar.

## App local (Streamlit) -- la forma amigable

`scripts/reports/app.py` (lanzar con `app.bat`) es una UI local que envuelve
todo el modulo: escribís un ticker, "Traer datos", pegás la narrativa del LLM
(opcional) y generás infografía o PDF con un boton, con preview y descarga.
Abre en `http://localhost:8501`.

Decision tecnica clave: la app obtiene los datos corriendo `build_yaml.py`
como **subproceso**, no importando las tools del MCP en proceso. Razon:
asyncpg no se lleva bien con el modelo de event-loop/threads de Streamlit
(da WinError 64). Ademas, el subproceso se lanza con `DATABASE_URL` removida
del env para forzar el DSN LOCAL (si Streamlit heredo una DATABASE_URL
remota del shell, sin esto intentaria conectar a Railway y falla).

## Estructura del modulo

```
scripts/reports/
|-- app.py / .bat              # UI local Streamlit (envuelve todo el modulo)
|-- build_yaml.py / .bat       # ticker -> YAML (datos del MCP) + .md placeholder
|-- make_report.py / .bat      # YAML (+ .md) -> PDF + PNGs por pagina
|-- make_infografia.py / .bat  # ticker o YAML -> infografía PNG (single image)
|-- templates/
|   |-- report.html            # template PDF (Jinja2)
|   |-- styles.css             # estilos PDF
|   |-- infografia.html        # template infografía (Jinja2)
|   `-- infografia.css         # estilos infografía (layout 1080x1500)
|-- examples/                  # ejemplos de YAML
|-- output/                    # YAMLs/MDs/PDFs/PNGs generados (gitignored)
|-- requirements.txt           # weasyprint, jinja2, pyyaml, dotenv, markdown, pymupdf
`-- README.md
```

Aislado del resto: importa funciones puras de `mcp_server/tools/` para
reusar la logica de datos, pero no toca el MCP server ni la DB en escritura.

## Flujos de uso

### Infografía (rapido, para X, sin LLM)

```
scripts\reports\make_infografia.bat BAC
  -> scripts\reports\output\BAC_<FECHA>_ig.png
```

Un solo comando. Acepta ticker (consulta el MCP) o una ruta a un YAML ya
generado. Cero edicion manual.

### Reporte PDF (detallado, con narrativa)

```
1. scripts\reports\build_yaml.bat BAC
     -> BAC_<FECHA>.yaml (datos) + BAC_<FECHA>.md (placeholder narrativa)
2. [editar el .md: pegar la respuesta del LLM tal cual]
3. scripts\reports\make_report.bat <ruta_al_yaml>
     -> BAC_<FECHA>.pdf + BAC_<FECHA>_p1.png, _p2.png, ...
```

## Reglas de las frases interpretativas (infografía, sin LLM)

La infografía genera dos conclusiones de lectura rapida por regla
determinística. Viven en `make_infografia.py`.

### frase_combinada (bloque "Sesgo general")

Combina sesgo de indicadores (tendencia_sma) + opciones (PCR OI corto) en
una matriz fija. Toda salida es una conclusion cerrada, nunca "mirar X".

| Indicadores | Opciones | Conclusión |
|---|---|---|
| alcista | alcista | Confluencia alcista |
| alcista | bajista | Divergencia: técnico alcista, opciones defensivas |
| alcista | neutro | Sesgo alcista en técnico, opciones sin confirmar |
| bajista | alcista | Divergencia: técnico débil, opciones optimistas |
| bajista | bajista | Confluencia bajista |
| bajista | neutro | Sesgo bajista en técnico, opciones sin confirmar |
| neutro | alcista | Consolidación con opciones alcistas |
| neutro | bajista | Consolidación con opciones defensivas |
| neutro | neutro | Sin convicción direccional |

Matiz: si confluencia alcista pero IV skew bajista -> agrega nota de cobertura.

### frase_price_action (bloque "Price Action")

Combina patrón de vela + estructura SMC (estructura_5). El caso mas util es
patrón de reversion contra la estructura (giro no confirmado).

| Patrón | Estructura | Conclusión |
|---|---|---|
| alcista (engulfing_bull/hammer) | bajista | Reversión alcista sin confirmar |
| bajista (shooting_star/engulfing_bear) | alcista | Posible techo en formación |
| alcista | alcista | Continuación al alza |
| bajista | bajista | Continuación a la baja |
| alcista | rango | Posible salida al alza |
| doji | - | Indecisión |
| marubozu | - | Fuerte momentum |
| sin patrón | (segun estructura) | Estructura X vigente |

## Notas de presentacion (datos vs interpretacion)

- **variacion_5d_pct** (cierre hoy vs cierre 5 ruedas atras) NO es lo mismo
  que **estructura_5** (geometria de swings SMC). El precio puede subir sin
  cambiar la estructura (se necesita un Break of Structure). El contraste
  entre ambos es informativo, no un error -- la frase_price_action lo explica.
- Las 3 SMAs (21/50/200) se muestran con su distancia % (positivo = precio
  sobre la media). Sobre = verde, bajo = rojo.

## Branding

- Paleta navy `#1B2A4E` + blanco + grafito `#2C3E50`.
- Verde alcista, rojo bajista, gris neutro/lateral.
- Handle e identidad configurables (default `@juan_de_nogoya`).
- Para cambiar look: editar `templates/*.css`.
