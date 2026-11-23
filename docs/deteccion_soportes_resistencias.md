> **DOCUMENTO DE ENTRADA, NO ES UN DOC DEL PROYECTO.** Lo trajo el usuario el
> 19/11/2026 como disparador; se guarda tal cual, sin editar su contenido, para
> que quede el origen de la linea de trabajo. **Lo que el proyecto MIDIO a partir
> de el esta en otro lado**: `docs/niveles_precio.md` (el detector que si se
> construyo) y `docs/perfil_precio.md` (el POC contra el azar).
>
> **Tres cosas de este documento ya estan contestadas y conviene no rehacerlas:**
>
> - **El SQL del Paso 1 (sec. 2) mira al futuro.** `ROWS BETWEEN 5 PRECEDING AND
>   5 FOLLOWING` es la ventana centrada que en este repo costo 11 puntos de AUC
>   (`features_market_structure`: 0,65 guardada contra 0,52 real). Para dibujar el
>   grafico de ayer sirve; para decidir hoy o backtestear, no. La version sin
>   futuro es `src/indicators/estructura.py`.
> - **El Paso 3 (sec. 2) y los Modelos 4 y 7 son la misma pregunta, y ya tiene
>   respuesta**: si un nivel vale por como reacciono el precio ahi antes. Medido
>   el 6/9/2026 fuera de muestra: el 85% de rechazo historico cae a 56% y el
>   Spearman entre tasa pasada y futura es -0,10 / +0,06
>   (`memory/zonas_precio_descartado.md`).
> - **El placebo del sec. 5.2.5 tiene un agujero MEDIDO.** Una banda de control
>   "del mismo ancho en otra ubicacion" no es un control limpio para un evento
>   DIRECCIONAL: la ubicacion determina cuanto tuvo que moverse el precio para
>   llegar, asi que cambia dos cosas a la vez. Los eventos del control llegaban
>   de una caida 3 pp mayor (`docs/perfil_precio.md` sec. 5.2).
>
> Lo que si quedo del documento: el Modelo 6 (prominencia) como idea de encontrar
> TODOS los picos y no el maximo, la separacion temporal minima entre toques y el
> cambio de polaridad. Los tres estan en el detector.

---

# Arquitectura para Detección Algorítmica de Soportes y Resistencias Estáticos

## 1. Contexto y Objetivo del Problema

El objetivo es identificar de forma automatizada **zonas estáticas de soporte y resistencia** a partir de una base de datos con registros diarios **OHLCV** (Open, High, Low, Close, Volume) actualizados desde 2021 hasta la fecha.

Un nivel relevante en el precio no es un valor exacto, sino una **zona de tolerancia** donde el activo ha manifestado repetidamente alguno de los siguientes comportamientos:

1. **Rechazo al alza (Resistencia):** El precio alcanza la zona, no logra superarla y corrige a la baja.
2. **Ruptura y Retesteo (Pullback):** El precio atraviesa la zona y continúa su camino, pero en el corto o mediano plazo regresa a testear ese mismo nivel.
3. **Rebote desde arriba (Soporte):** El precio cae hacia la zona, hace contacto y rebota al alza.

A continuación se detallan dos grandes enfoques para abordar el problema: un **pipeline clásico (geométrico)** y **siete modelos disruptivos** basados en física, teoría de grafos, finanzas conductuales, probabilidad, visión por computadora, topografía y estadística condicional.

---

## 2. Enfoque 1: Pipeline Clásico (Geométrico y Estadístico)

Este enfoque busca replicar matemáticamente cómo el ojo humano identifica zonas horizontales en un gráfico de velas. Se divide en tres etapas secuenciales.

### Paso 1: Detección de Pivotes (Swing Highs y Swing Lows)

En lugar de evaluar todas las velas de la serie, se filtran únicamente los puntos de inflexión locales (fractales). Una vela en el día $t$ se considera un pivote si su extremo es el mayor o menor dentro de una ventana simétrica de $N$ ruedas hacia atrás y $N$ ruedas hacia adelante (típicamente $N = 5$ o $N = 10$).

* **Swing High (Resistencia potencial):** $\text{High}_t = \max(\text{High}_{t-N}, \dots, \text{High}_{t+N})$
* **Swing Low (Soporte potencial):** $\text{Low}_t = \min(\text{Low}_{t-N}, \dots, \text{Low}_{t+N})$

> **Nota técnica:** En activos con mucha volatilidad intradiaria (mechas largas), conviene evaluar los pivotes usando los extremos del cuerpo de la vela ($\max(\text{Open}, \text{Close})$ y $\min(\text{Open}, \text{Close})$) o almacenar ambos para comparar resultados.

**Ejemplo de consulta base en SQL (Window Functions):**

```sql
WITH pivotes AS (
    SELECT 
        fecha,
        high,
        low,
        volume,
        CASE 
            WHEN high = MAX(high) OVER (ORDER BY fecha ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING) 
            THEN high 
        END AS swing_high,
        CASE 
            WHEN low = MIN(low) OVER (ORDER BY fecha ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING) 
            THEN low 
        END AS swing_low
    FROM ohlcv_diario
    WHERE ticker = 'GS' AND fecha >= '2021-01-01'
)
SELECT fecha, swing_high AS precio, 'RESISTENCIA' AS tipo, volume 
FROM pivotes 
WHERE swing_high IS NOT NULL
UNION ALL
SELECT fecha, swing_low AS precio, 'SOPORTE' AS tipo, volume 
FROM pivotes 
WHERE swing_low IS NOT NULL;
```

### Paso 2: Agrupación en Zonas (Clustering)

Los pivotes detectados tendrán precios cercanos pero raramente idénticos (por ejemplo: $781.5$, $784.0$, $786.2$). Para consolidarlos en una única zona se requiere definir una **tolerancia**.

* **Definición de la tolerancia:**
  * *Porcentaje fijo:* Por ejemplo, $\pm 1.5\%$ alrededor del precio central.
  * *Basada en volatilidad (Recomendada):* Un múltiplo del *Average True Range*, como $0.5 \times \text{ATR}_{14}$. Esto permite que el algoritmo se adapte automáticamente a tickers lentos o muy volátiles sin recalibrar parámetros manualmente.
* **Implementación del agrupamiento:**
  * *En SQL puro:* Mediante un *Self-Join* sobre la tabla de pivotes contando cuántos registros caen dentro de la condición $\frac{|p_1 - p_2|}{p_1} \le \text{tolerancia}$.
  * *En Python:* Utilizando algoritmos no supervisados como **DBSCAN** (agrupa puntos por densidad sin exigir un número fijo de niveles) o **KDE (Kernel Density Estimation)**, donde los máximos locales de la curva de densidad representan los niveles horizontales más significativos.

### Paso 3: Reglas de Validación y Scoring

Para eliminar el ruido y jerarquizar las zonas encontradas, se aplica un sistema de puntuación basado en la calidad de los toques:

1. **Separación temporal mínima:** Exigir que dos pivotes dentro de una misma zona estén separados por al menos $X$ ruedas (ej. $15$ o $20$ días). Esto evita que una consolidación lateral corta infle artificialmente el conteo de toques.
2. **Cambio de polaridad (Role Reversal):** Asignar un multiplicador de puntaje (ej. $\times 2$) a las zonas que contengan tanto etiquetas de `SOPORTE` como de `RESISTENCIA` en su historial. Este criterio captura de manera directa los escenarios de ruptura y retesteo.
3. **Confirmación por volumen:** Ponderar positivamente cada toque cuyo volumen diario supere la media móvil de volumen de 20 días ($\text{Vol}_t > \text{SMA}_{20}(\text{Vol})$).
4. **Umbral de relevancia:** Descartar cualquier zona que no alcance un mínimo de toques independientes validados (por ejemplo, $\ge 3$ toques).

---

## 3. Enfoque 2: Paradigmas Disruptivos

El enfoque clásico analiza la geometría del gráfico, pero ignora la microestructura, la probabilidad de transición y la dinámica real que genera los frenos en el precio. Los siguientes siete modelos atacan el problema desde perspectivas alternativas utilizando exactamente los mismos datos OHLCV.

### Modelo 1: Liquidez Atrapada y Decaimiento (Finanzas Conductuales + Física)

**Premisa:** Los soportes y resistencias existen por la **memoria psicológica del mercado**. Cuando se opera gran volumen en un nivel y el precio se mueve bruscamente en contra, los participantes atrapados esperan el regreso a ese precio para salir en punto de equilibrio (*breakeven*), generando la reacción.

* **Mecánica del algoritmo:**
  1. Se discretiza el rango histórico de precios en contenedores (*bins*) uniformes (por ejemplo, saltos del $0.5\%$).
  2. Se recorre la serie temporal cronológicamente desde 2021. Cada vela diaria "inyecta" energía (proporcional a su volumen) en los *bins* que atraviesa.
* **Reglas de evolución de energía:**
  * **Decaimiento exponencial:** En cada nueva rueda, la energía acumulada en todos los *bins* disminuye según un factor de vida media $\lambda$:
    $$E_{t} = E_{t-1} \cdot e^{-\lambda}$$
    De esta manera, los niveles recientes pesan más que los de hace varios años.
  * **Efecto de Borrado (Consumo de liquidez):** Si una vela atraviesa un *bin* con un cuerpo sólido amplio y cierra lejos del nivel, se considera que la liquidez fue absorbida y se reduce drásticamente (o se reinicia) la energía de ese *bin*.
  * **Efecto de Rechazo (Refuerzo):** Si una vela ingresa al *bin* únicamente con su mecha y cierra fuera de él, se multiplica la energía acumulada en esa zona.
* **Salida:** Un perfil continuo de "energía remanente" por nivel de precio. Los picos más altos representan las zonas activas que el mercado aún recuerda y no ha limpiado.

---

### Modelo 2: Grafos de Visibilidad Horizontal (Teoría de Grafos)

**Premisa:** Eliminar por completo el parámetro arbitrario de la ventana $N$ (como los 5 días hacia atrás y adelante del método clásico), detectando barreras estructurales a múltiples escalas de tiempo de forma simultánea.

* **Mecánica del algoritmo:**
  1. Cada vela diaria se modela como un **nodo** $t$ en un grafo, con altura dada por su `High` (para resistencias) o su `-Low` (para soportes).
  2. Se traza una **arista** (conexión horizontal) entre dos días $t_1$ y $t_2$ si y solo si ningún día intermedio $t_k$ ($t_1 < t_k < t_2$) supera la altura de los extremos:
     $$\text{Precio}(t_k) < \min(\text{Precio}(t_1), \text{Precio}(t_2)) \quad \forall t_k \in (t_1, t_2)$$
* **Por qué funciona:** Los niveles que actúan como verdaderas barreras estructurales se convierten naturalmente en **nodos *hub*** con altísima conectividad, uniendo puntos separados por semanas o meses sin necesidad de predefinir el horizonte temporal.
* **Salida:** Al calcular métricas de centralidad del grafo (como el grado del nodo o *Degree Centrality* y la distancia temporal máxima de sus conexiones), emergen automáticamente los niveles estructurales más fuertes del activo.

---

### Modelo 3: Matriz de Fricción y Cinemática del Precio

**Premisa:** Inspirado en la mecánica clásica. Si se aplica una gran fuerza (**Volumen**) sobre un objeto (**Precio**) y su desplazamiento neto (**Cuerpo de la vela**) es mínimo, significa que el objeto impactó contra una pared invisible (absorción institucional).

* **Mecánica del algoritmo:**
  1. Para cada día $t$, se calcula un **Índice de Fricción**:
     $$\text{Fricción}_t = \frac{\text{Volumen}_t}{|\text{Close}_t - \text{Open}_t| + \epsilon}$$
     *(donde $\epsilon$ es una constante muy pequeña para evitar divisiones por cero en velas Doji).*
  2. Opcionalmente, se normaliza el volumen dividiéndolo por su media móvil de 20 días y el cuerpo dividiéndolo por el $\text{ATR}_{14}$ para que la métrica sea adimensional y comparable a lo largo de los años.
  3. Se proyecta el valor de fricción sobre el eje de precios (asignándolo al precio medio de la vela o ponderado hacia la zona de la mecha de rechazo) y se acumula por zonas.
* **Salida:** Un mapa de calor de "desaceleración del precio". A diferencia de los pivotes tradicionales, este modelo es capaz de detectar zonas de fuerte acumulación o distribución institucional incluso antes de que se forme un giro evidente en el gráfico.

---

### Modelo 4: Cadenas de Markov (Barreras de Probabilidad Asimétrica)

**Premisa:** En lugar de buscar líneas geométricas, se buscan zonas de precio donde se rompe la simetría aleatoria del mercado (*random walk*). Un nivel es real si altera significativamente las probabilidades de movimiento en los días posteriores.

* **Mecánica del algoritmo:**
  1. Se discretiza el eje de precios en $K$ estados (bloques o *bins* porcentuales o múltiplos de ATR).
  2. Se construye la **matriz de transición empírica** $P$ evaluando a un horizonte de $h$ días (ej. $h = 3$ o $h = 5$ ruedas):
     $$P_{i \to j} = P(S_{t+h} = j \mid S_t = i)$$
  3. Para cada estado $i$, se calcula la probabilidad acumulada de subir ($P_{\uparrow}(i) = \sum_{j > i} P_{i \to j}$) frente a la probabilidad acumulada de bajar ($P_{\downarrow}(i) = \sum_{j < i} P_{i \to j}$).
* **La señal:**
  * **Soporte fuerte:** Un bloque $i$ donde $P_{\uparrow}(i) \gg P_{\downarrow}(i)$ (por ejemplo, $80\%$ vs $20\%$) con un tamaño muestral mínimo de visitas.
  * **Resistencia fuerte:** Un bloque $i$ donde $P_{\downarrow}(i) \gg P_{\uparrow}(i)$.
* **Salida:** Un mapa probabilístico del eje de precios que mide el *efecto real* de cada zona sobre el futuro inmediato, ignorando por completo la forma de las velas o el volumen.

---

### Modelo 5: Transformada de Hough (Visión por Computadora)

**Premisa:** El cerebro humano detecta soportes y resistencias al instante porque procesa patrones espaciales. Este modelo replica exactamente esa capacidad tratando la serie OHLCV como una imagen bidimensional.

* **Mecánica del algoritmo:**
  1. Se proyecta la serie histórica en una matriz binaria 2D (imagen en blanco y negro), donde el eje $X$ es el tiempo (índice de rueda) y el eje $Y$ es el precio cuantizado en píxeles.
  2. Se "encienden" únicamente los píxeles correspondientes a los extremos de cada vela (`High`, `Low`, o los bordes del cuerpo `Open`/`Close`).
  3. Se aplica la **Transformada de Hough** (clásica en *Computer Vision* para detectar rectas en imágenes, disponible en `OpenCV` o `scikit-image`), restringiendo el espacio de parámetros únicamente al ángulo $\theta = 0^\circ$ (líneas estrictamente horizontales).
* **La señal:** Cada píxel encendido "vota" en una matriz acumuladora por la coordenada horizontal $y$ que atraviesa. Las filas con mayor acumulación de votos son las alineaciones horizontales más fuertes a lo largo de toda la historia, tolerando grandes brechas temporales en el medio.
* **Salida:** Un ranking de niveles horizontales puros, inmunes al ruido de corto plazo y capaces de conectar toques separados por años.

---

### Modelo 6: Topografía e "Inundación" (Prominencia / Homología Persistente)

**Premisa:** Resolver definitivamente el problema de elegir la ventana temporal $N$. En geografía y análisis topológico de datos (*Topological Data Analysis - TDA*), la importancia de una montaña no se mide por su ancho, sino por su **prominencia topográfica**: cuánto hay que descender desde su cima antes de poder subir a una cima aún más alta.

* **Mecánica del algoritmo:**
  1. Se interpreta la serie de precios como un relieve montañoso unidimensional.
  2. Se simula una "inundación" progresiva desde abajo hacia arriba (para detectar valles/soportes) y de forma invertida (para detectar picos/resistencias).
  3. Se mide el **tiempo de vida (persistencia)** o la **prominencia** de cada extremo local antes de fusionarse con una estructura mayor. En Python, una aproximación directa es `scipy.signal.find_peaks` parametrizado por `prominence` (ej. un múltiplo del ATR o un porcentaje del precio), o librerías de homología persistente como `ripser` o `giotto-tda`.
* **La señal:** Solo sobreviven los pisos y techos verdaderamente estructurales; las fluctuaciones menores quedan filtradas automáticamente sin importar si duraron 3 días o 3 meses.
* **Salida:** Un conjunto limpio y jerarquizado de puntos críticos estructurales listos para agruparse en zonas o usarse directamente como niveles clave.

---

### Modelo 7: Mapa de Retornos Condicionales (Fuerza de Reversión)

**Premisa:** Invertir el planteo tradicional. En lugar de preguntar *"¿dónde rebotó visualmente el precio?"*, se pregunta *"si se hubiera operado cada vez que el precio tocó el nivel $P$, ¿qué retorno promedio habría generado?"*.

* **Mecánica del algoritmo:**
  1. Se barre el rango histórico de precios en pasos discretos $P_1, P_2, \dots, P_m$.
  2. Cada vez que el precio alcanza el nivel $P_k$ (dentro de una tolerancia $\delta$), se registra la **dirección de llegada** (desde arriba o desde abajo) y se calcula el retorno futuro a $h$ días (ej. $h = 5, 10, 20$):
     $$R_{t, h} = \frac{\text{Close}_{t+h} - P_k}{P_k}$$
  3. Para cada nivel $P_k$ se calcula el retorno medio esperado y su significancia estadística (ej. *t-statistic* o *Sharpe* condicional) separado por dirección de llegada.
* **La señal:**
  * Un **soporte operable** presenta un retorno promedio significativamente positivo cuando el precio llega desde arriba ($E[R_{t,h} \mid \text{llegada desde arriba}] > 0$).
  * Una **resistencia operable** presenta un retorno promedio significativamente negativo cuando el precio llega desde abajo ($E[R_{t,h} \mid \text{llegada desde abajo}] < 0$).
* **Salida:** Un perfil de expectativa matemática por nivel de precio. Permite descartar niveles que "se ven lindos en el gráfico" pero que históricamente no ofrecen ventaja estadística real al operarlos.

---

## 4. Comparativa Rápida para Implementación

| Enfoque / Modelo | Complejidad Técnica | Dependencia de Parámetros | Fortaleza Principal | Ideal para... |
| :--- | :--- | :--- | :--- | :--- |
| **Clásico (3 Pasos)** | Baja / Media (SQL + Python) | Alta (ventana $N$, tolerancia) | Muy interpretable y fácil de validar visualmente. | Construir una línea base (*baseline*) rápida y confiable. |
| **1. Liquidez y Decaimiento** | Media (Python / NumPy) | Media ($\lambda$, tamaño de *bin*) | Descarta automáticamente niveles viejos o ya rotos. | Trading activo enfocado en niveles "frescos". |
| **2. Grafos de Visibilidad** | Media / Alta (`networkx`) | Muy Baja (sin ventana $N$) | Detecta estructuras multiescala de forma pura. | Análisis estructural sin sesgo de temporalidad. |
| **3. Matriz de Fricción** | Baja / Media (SQL / Python) | Baja | Detecta absorción institucional oculta. | Capa de confirmación junto a otros modelos. |
| **4. Cadenas de Markov** | Media (Python / Pandas) | Media (tamaño de estado, horizonte $h$) | Mide la asimetría probabilística real del precio. | Sistemas cuantitativos basados en probabilidad. |
| **5. Transformada de Hough** | Media (`OpenCV` / `skimage`) | Baja / Media (resolución de grilla) | Replica la visión humana; tolera huecos largos. | Detección de alineaciones históricas de largo plazo. |
| **6. Topografía / Prominencia** | Baja / Media (`scipy` / TDA) | Muy Baja (solo umbral de prominencia) | Elimina el sesgo temporal de las ventanas fijas. | Reemplazar el Paso 1 del enfoque clásico. |
| **7. Retornos Condicionales** | Media (Python / NumPy) | Media (horizonte $h$, tolerancia $\delta$) | Valida directamente si el nivel genera ventaja económica. | Filtrado final orientado a estrategias operables. |

---

## 5. Diseño de Pruebas A/B y Evaluación Objetiva

Para evitar el sesgo visual (*"esta línea parece buena en el gráfico"*), todos los modelos deben someterse a un protocolo uniforme de **Pruebas A/B** con validación *Out-of-Sample* (fuera de muestra).

### 5.1. Esquema de División Temporal (*Walk-Forward*)
No se deben evaluar los niveles usando los mismos datos con los que se calcularon.
* **Ventana de Entrenamiento (*In-Sample*):** Por ejemplo, datos desde `2021-01-01` hasta `2024-12-31`. Cada modelo (A, B, C...) genera su lista fija de los Top $K$ niveles (ej. $K = 5$ niveles por ticker).
* **Ventana de Prueba (*Out-of-Sample*):** Datos posteriores (ej. `2025-01-01` hasta la fecha actual). Se mide cómo reaccionó el precio futuro al tocar esos niveles previamente fijados.
* *(Opcional)* **Walk-Forward deslizante:** Recalcular los niveles cada trimestre usando todo el historial previo y evaluar el comportamiento en el trimestre siguiente.

### 5.2. Métricas Clave para el Benchmark A/B
Cada vez que el precio en el período de prueba ingresa a la zona de tolerancia ($\pm 0.5 \times \text{ATR}_{14}$) de un nivel predicho por el modelo, se registra un **Evento de Toque** y se miden:

1. **Tasa de Reacción (*Bounce Rate*):** Porcentaje de toques en los que el precio revierte al menos $1.5 \times \text{ATR}_{14}$ en la dirección esperada antes de romper el nivel en contra por $1.0 \times \text{ATR}_{14}$.
2. **Ratio Riesgo/Beneficio Empírico (*MFE / MAE*):** Relación entre la Máxima Excursión Favorable (*Maximum Favorable Excursion*) y la Máxima Excursión Adversa (*Maximum Adverse Excursion*) en las $10$ ruedas posteriores al toque.
3. **Retorno Promedio Condicional ($h = 5, 10$ días):** Rentabilidad media a favor del rebote tras el contacto.
4. **Tasa de Ruptura Limpia (*False Signal Rate*):** Porcentaje de veces que el precio atraviesa el nivel sin mostrar fricción ni retroceso.
5. **Eficiencia (Niveles vs. Ruido):** Comparar siempre contra un **Modelo Placebo (Grupo de Control)** que genera $K$ líneas horizontales al azar dentro del rango de precios. Un modelo solo aporta valor real si supera de forma estadísticamente significativa al grupo de control.

### 5.3. Estrategia de Combinación (Ensemble)
Una vez corridas las pruebas A/B sobre un universo representativo de tickers (por ejemplo, 20 activos de distintos sectores y volatilidades), es altamente probable que ningún modelo sea perfecto por sí solo. El paso final natural es construir un **sistema híbrido**, por ejemplo:
* Detectar candidatos con **Topografía (Modelo 6)** o **Transformada de Hough (Modelo 5)**.
* Ponderar su vigencia actual con **Liquidez y Decaimiento (Modelo 1)** y **Fricción (Modelo 3)**.
* Filtrar únicamente los que superen el umbral de ventaja en **Cadenas de Markov (Modelo 4)** o **Retornos Condicionales (Modelo 7)**.