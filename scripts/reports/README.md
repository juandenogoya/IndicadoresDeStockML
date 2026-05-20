# Reportes PDF de análisis técnico

Script que genera un PDF de 2 páginas (cover + detalle) con el formato
estándar que usamos para compartir análisis en X.

## Instalación

Las dependencias están aisladas en `requirements.txt` local — no tocan el
`requirements.txt` global del proyecto.

```bash
pip install -r scripts/reports/requirements.txt
```

### Nota para Windows

WeasyPrint en Windows puede requerir GTK runtime. Si el `pip install` no
alcanza, descargar el runtime desde:
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer

(La mayoría de las instalaciones recientes de WeasyPrint funcionan sin
GTK separado, así que probá primero solo con pip.)

## Uso

```bash
python scripts/reports/make_report.py scripts/reports/examples/ul_2026-05-16.yaml
```

Genera `scripts/reports/examples/ul_2026-05-16.pdf` al lado del YAML.

Salida personalizada:

```bash
python scripts/reports/make_report.py mi_analisis.yaml -o /ruta/al/reporte.pdf
```

## Estructura del input YAML

Ver `examples/ul_2026-05-16.yaml` como plantilla. Campos requeridos:

- `ticker`, `nombre`, `fecha_legible`, `handle`
- `precio`, `var_5d` (opcional)
- `veredicto` — texto del TL;DR (página 1)
- `niveles` — tabla resumen S/R por ventana
- `sesgo` — sesgo general (indicadores / opciones / IV skew)
- `indicadores` — RSI, MACD, SMA, ADX (página 2)
- `opciones_ventanas` — detalle por ventana corto/medio/largo
- `strikes_calientes` — top contratos por Δ OI (opcional)
- `conclusion` — setup operativo (cierre página 2)

## Flujo de trabajo sugerido

1. Pedir análisis al LLM vía MCP en Gemini CLI.
2. Copiar los datos relevantes a un YAML con la estructura del ejemplo.
3. Correr el script → PDF listo.
4. Subir el PDF a X como adjunto del tweet (con el chart como imagen).

## Diseño

- **A4**, 2 páginas
- Header navy `#1B2A4E` + texto blanco
- Cuerpo grafito `#2C3E50` sobre blanco
- Tipografía: Inter / Helvetica Neue / sans-serif
- Veredicto y conclusión en callout box (fondo gris claro, borde navy)

Si querés cambiar colores/tipografía, editar `templates/styles.css`.
