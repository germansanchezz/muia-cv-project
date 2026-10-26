# Round 1 · referencias con arquitecturas populares

Seis experimentos iniciales con el mismo protocolo, para comparar después con
las CNN propias. Cada notebook entrena una red desde cero y se puede ejecutar
completo, en cualquier orden. Los entrenamientos quedan pendientes de ejecución.

## Notebooks

| Experimento | Notebook | Referencia |
|---|---|---|
| 01 | [VGG16](01_vgg16/vgg16_round1.ipynb) | Convoluciones apiladas; cabeza GAP adaptada |
| 02 | [GoogLeNet](02_googlenet/googlenet_round1.ipynb) | Inception-v1; sin cabezas auxiliares |
| 03 | [ResNet18](03_resnet18/resnet18_round1.ipynb) | Conexiones residuales |
| 04 | [DenseNet121](04_densenet121/densenet121_round1.ipynb) | Conexiones densas |
| 05 | [MobileNetV2](05_mobilenet_v2/mobilenet_v2_round1.ipynb) | Referencia de eficiencia; anchura alpha=1 |
| 06 | [EfficientNetB0](06_efficientnet_b0/efficientnet_b0_round1.ipynb) | Variante B0; ajuste de escala interno de la plantilla |

Las adaptaciones están descritas en el apartado 4 de los notebooks y en el
[README general](../../README.md).

## Protocolo común

| Parámetro | Valor |
|---|---|
| Entrada | RGB 64×64, float32 en [0,1] |
| Reparto | Histórico 90/10, semilla 1, sin estratificación |
| Inicialización | Aleatoria; todas las capas entrenables |
| Semilla de entrenamiento | 42, operaciones deterministas |
| Optimizador | Adam, LR 0.001, AMSGrad, clipnorm 1 |
| Batch | 32 |
| Máximo de épocas | 200 |
| Selección | Primer máximo de `val_accuracy` |
| Reducción de LR | Factor 0.2 tras 10 épocas sin mejora; mínimo 1e-7 |
| EarlyStopping | 20 épocas sin mejora |
| Dropout de la cabeza | 0.1 |
| L2 añadida / aumentación | Desactivadas |
| Normalización del modelo | La propia de cada arquitectura |
| Caché | Compartida en `cnn_challenge/.cache/`, modo `mmap` |

Es una primera comparación con configuración común. Si una arquitectura no
converge, se revisarán sus curvas antes de interpretar su resultado como una
limitación del modelo. Los ajustes posteriores pueden guardarse en `round2/`
para conservar esta ronda.

## Ejecutar y conservar resultados

`NOTEBOOK_PATH`, `experiment_id` y `architecture` ya están configurados en cada
copia. Abre el notebook y utiliza «Ejecutar todo». Cada carpeta recibe su propio
`model.keras`, `prediction.json`, `artifacts/` y `plots/` con figuras PDF.
El JSON de test se genera siempre; no se crea ZIP ni se envía a Codabench.

La caché se crea con la primera preparación y se reutiliza en los siguientes
experimentos. Reejecutar un entrenamiento en la misma carpeta sobrescribe sus
resultados. Para repetir con otra semilla o configuración conservando esta prueba,
crea otra copia y actualiza `NOTEBOOK_PATH` e `experiment_id`.

## Comparar cuando terminen las seis redes

Las comparaciones están **desactivadas en los seis notebooks**: las dos llamadas
finales del apartado 9 están comentadas. «Ejecutar todo» no intenta leer otros
experimentos ni modifica las comparaciones existentes.

En cualquiera de las seis copias, `SELECT_RUNS` ya contiene las carpetas de toda
la ronda. Cuando todas tengan resultados:

1. Si has reiniciado el kernel, ejecuta la primera celda de código.
2. En el apartado 9, descomenta las dos llamadas finales:

   ```python
   runs = load_runs(CHALLENGE_DIR, SELECT_RUNS)
   compare_runs(runs, CHALLENGE_DIR / 'comparisons')
   ```

3. Ejecuta solo esa celda. Los resultados se guardan en
   `PROJECT/cnn_challenge/comparisons/`.

Puedes cambiar `SELECT_RUNS` para comparar un subconjunto o incorporar una CNN
propia. No hace falta ejecutar los otros apartados ni volver a entrenar.
