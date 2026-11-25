# dice-computer-vision

TP3 de Procesamiento de Imágenes — **Problema: Cinco dados**.

A partir de videos de tiradas de cinco dados rojos, el programa:

- **A)** Detecta los frames en los que los dados están quietos (usando solo los píxeles rojos).
- **B)** Informa por consola el valor de cada dado.
- **C)** Genera un video con la bounding box, el id y el valor de cada dado (ej.: `D1-5, D2-2, D3-2, D4-2, D5-3`) desde que quedan en reposo. La caja de cada dado se mantiene hasta que ya no queda ningún píxel rojo dentro de ella.

El enunciado completo está en [Enunciado.pdf](Enunciado.pdf).

## Estructura

```
dice-computer-vision/
├── trabajo_practico.py     # Script principal: procesa las tiradas y genera los videos
├── ejemplo.py              # Muestra paso a paso (con plots) cómo se procesa una tirada
├── notebooks/
│   └── pruebas.ipynb       # Notebook de experimentación
├── videos_input/           # Videos de entrada (tirada_1.mp4 ... tirada_4.mp4)
├── videos_output/          # Videos resultantes (resultado_tirada_<id>.mp4)
├── Enunciado.pdf
└── requirements.txt
```

## Instalación

Requiere Python 3.8+.

```bash
python -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencias: `opencv-python`, `numpy` y `matplotlib` (esta última solo para `ejemplo.py`).

## Uso

### Procesar las tiradas

```bash
python trabajo_practico.py
```

Busca los videos `videos_input/tirada_*.mp4`, imprime por consola el rango de frames en reposo y el valor de cada dado, y escribe `videos_output/resultado_tirada_<id>.mp4`. Opcionalmente se puede pasar otra carpeta base como argumento:

```bash
python trabajo_practico.py <carpeta>
```

Formato de la salida por consola (por cada video):

```
tirada_<id>: dados quietos en los frames <inicio>-<fin>
   D1-<valor>, D2-<valor>, D3-<valor>, D4-<valor>, D5-<valor>
```

### Ver el proceso paso a paso

```bash
python ejemplo.py 3
```

Abre una serie de plots que muestran cada etapa del pipeline sobre la tirada indicada (por defecto la 1): detección de reposo, máscara roja, morfología, componentes conexas, binarización de cada dado, puntos detectados y las cajas en el video. También acepta la ruta a cualquier video.

## Cómo funciona

### A) Detección de reposo

1. Cada frame se reduce a 1/3 y se convierte a HSV; se arma una máscara con los píxeles rojos (los dos extremos del canal H).
2. Se calcula el cambio entre frames consecutivos (XOR de máscaras) y se elige como **frame de referencia** el de menor cambio promedio en una ventana de ±2 frames, exigiendo que haya dados visibles.
3. Se compara cada frame contra el de referencia con una diferencia tolerante (dilatación de 5 px) y se extiende el rango hacia atrás y adelante mientras la diferencia sea ≤ 5 % del área roja.

### B) Lectura de los dados

Sobre el frame de referencia:

```
HSV → píxeles rojos → clausura (rellena los puntos) → apertura (saca ruido)
    → componentes conexas → filtro por área y forma cuadrada
    → Otsu usando solo los píxeles de cada dado → apertura
    → componentes conexas → filtro por área y redondez (ejes por momentos)
    → cantidad de puntos = valor del dado
```

- **Forma cuadrada:** `área / lado_mayor²` del rectángulo mínimo rotado (cuadrado perfecto = 1, umbral 0.70). Descarta, por ejemplo, dos dados pegados.
- **Redondez de los puntos:** relación eje menor / eje mayor calculada con momentos centrales (círculo = 1, umbral 0.80).
- Los ids se asignan en orden de lectura (arriba → abajo, izquierda → derecha).

### C) Video resultante

Se dibujan las cajas y etiquetas desde el primer frame en reposo. Después del reposo, la caja de cada dado se mantiene mientras quede al menos un píxel rojo dentro de ella; en el primer frame sin rojo desaparece.

## Parámetros

Todos los umbrales están al principio de [trabajo_practico.py](trabajo_practico.py) (sección `PARÁMETROS`): rango HSV del rojo, tamaños de los elementos estructurantes, áreas mínimas/máximas de dados y puntos, índices de forma, etc.
