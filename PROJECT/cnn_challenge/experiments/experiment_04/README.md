# Experimento propio 04 · CNN ampliada a tamaño original

El [notebook](cnn_experiment_04.ipynb) está preparado para entrenar desde cero a
**224×224**, tamaño de los recortes originales del benchmark. Combina una CNN
mayor, Batch Normalization, volteos horizontales/verticales y Dropout 0.2.
Está pendiente de entrenar.

## Motivación

El experimento 02 alcanzó 73,92 % de accuracy a 64×64. El experimento 03, con
la misma arquitectura a 128×128, bajó a 70,83 % y tardó más. Ambos ajustaron
casi todo train: añadir capacidad no responde a una incapacidad demostrada de
memorizar, sino a la hipótesis de mejorar la representación espacial a mayor
resolución, acompañada de medidas para generalizar.

Se prioriza buscar una configuración de mayor accuracy. Es un **ensayo combinado**:
si mejora, no permitirá atribuir la ganancia individualmente a resolución,
arquitectura, BN, aumentación o Dropout. El 80–85 % es un objetivo de búsqueda,
no un resultado garantizado.

## Arquitectura

Cada bloque contiene dos secuencias `Conv3×3 → BN → ReLU`, seguidas de MaxPool2×2.

| Etapa | Filtros por convolución | Salida tras pooling |
|---|---:|---|
| Entrada | RGB | 224×224×3 |
| Bloque 1 | 32 | 112×112×32 |
| Bloque 2 | 64 | 56×56×64 |
| Bloque 3 | 128 | 28×28×128 |
| Bloque 4 | 256 | 14×14×256 |
| Bloque 5 | 256 | 7×7×256 |

La cabeza es `GAP → Dropout(0.2) → Dense(13, Softmax)`.
Las convoluciones usan padding SAME, stride 1, inicialización He y `use_bias=False`
al incorporar BN. GAP entrega 256 valores; el clasificador tiene 3.341 parámetros.

Total: **2.360.173 parámetros**, de los cuales **2.357.229 son entrenables** y
**2.944 corresponden a estadísticas BN**. El campo receptivo teórico del último
mapa pasa de 36×36 a 156×156 píxeles con los dos bloques adicionales. Esto facilita
combinar información antes de promediar los mapas, sin garantizar una mejora.

## Configuración

| Parámetro | Valor |
|---|---|
| Entrada | RGB 224×224, float32 en [0,1] |
| Aumentación | Volteos horizontales y verticales solo durante entrenamiento |
| Rotaciones / recortes / Cutout | Desactivados |
| Normalización | BN después de cada convolución y antes de ReLU |
| Dropout | 0.2, después de GAP |
| L2 añadida | 0 |
| Optimizador | Adam, LR 1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, AMSGrad y clipnorm 1 |
| Batch | 32, conservado |
| Semillas | Entrenamiento **42**, reparto **1**, sin cambios |
| Reparto | Histórico 90/10, 16.871 train y 1.875 validación, sin estratificación |
| Máximo de épocas / monitor | 200 / `val_accuracy` |
| Reducción de LR | Factor 0.2, paciencia 10 y mínimo 1e-7 |
| EarlyStopping | Paciencia 20 |

Aquí «Dropout» significa desactivar aleatoriamente componentes del vector de la
cabeza durante entrenamiento. No se introduce Cutout ni se borran regiones de la
imagen. Los volteos cambian entre pasadas, preservan las etiquetas y se desactivan
en inferencia. La caché guarda imágenes sin aumentación.

Se conserva la semilla, pero los pesos tienen formas y cantidades distintas de
los experimentos 02/03. Se registra su nuevo hash para reproducir este diseño;
no se espera que coincida con el de la arquitectura anterior. No se cargan pesos
entrenados ni se utiliza transfer learning.

## Coste y ejecución

La caché compartida de 224×224 ocupa aproximadamente **10,51 GiB en disco**.
Se crea imagen a imagen, separadamente de las cachés de 64/128, y se abre con
`CACHE_MODE='mmap'` para leer lotes. Los mapas de activación mayores también
requieren más memoria de entrenamiento; el coste real se medirá en esta ejecución.

`NOTEBOOK_PATH` ya apunta a esta copia. Al ejecutar todo, se guardan `model.keras`
y siempre `prediction.json` junto al notebook, auxiliares en `artifacts/` y
figuras PDF en `plots/`. No se ha creado la caché grande ni se ha iniciado el
entrenamiento al preparar este experimento.

## Qué evaluar

- Mejor accuracy, F1 y recall macro, CE y comportamiento por clase frente a las
  CNN propias 02 y 03.
- Train y validación **en inferencia** para medir la brecha. Las curvas de train
  incluyen ahora aumentación, BN y Dropout activos, y no equivalen a esa evaluación.
- Errores de Truck, Bus, Dump truck y Storage tank, junto al soporte de cada clase.
- Fit completo, segundos por época y coste respecto a los 4,87 minutos del
  experimento 02. No valorar únicamente el máximo de accuracy.

Si no mejora, revisar primero las curvas y la estabilidad de entrenamiento:
al combinar varios cambios no se podrá identificar una causa única solo a partir
del resultado final. Se conservan los experimentos anteriores como referencias.

## Comparaciones opcionales entre CNN propias

No se comparan arquitecturas populares. `SELECT_RUNS` contiene únicamente
`experiments/experiment_02`, `experiments/experiment_03` y
`experiments/experiment_04`, con las llamadas finales **comentadas**.
«Ejecutar todo» no busca otros resultados ni genera comparaciones.

Si después interesa compararlos, ejecutar la primera celda de código si el kernel
se ha reiniciado y descomentar en el apartado 9:

```python
runs = load_runs(CHALLENGE_DIR, SELECT_RUNS)
compare_runs(runs, COMPARISON_DIR)
```

Los archivos se guardarán en
`PROJECT/cnn_challenge/comparisons/experiment_04_vs_custom/`.
