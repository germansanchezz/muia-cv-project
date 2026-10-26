# Round 1 · referencias con arquitecturas populares

Seis arquitecturas populares como referencias para las futuras CNN propias.
Cada notebook entrena una red desde cero y puede ejecutarse completo.
Los ajustes menores se guardan como variantes dentro de esta misma ronda.

## Notebooks

| Experimento | Notebook | Referencia |
|---|---|---|
| 01-a | [VGG16 · variante a](01_vgg16-a/vgg16_round1_a.ipynb) | LR 1e-3; ejecución original conservada, estancada en 19,36 % |
| 01-b | [VGG16 · variante b](01_vgg16-b/vgg16_round1_b.ipynb) | LR 1e-4; pendiente de entrenar, resto igual a la variante a |
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

Este es el protocolo inicial. La variante VGG16-b cambia únicamente el LR inicial
a **0.0001**; mantiene semilla, arquitectura, resolución, batch, regularización y
callbacks. Si una arquitectura no converge, se revisarán sus curvas antes de
interpretar su resultado como una limitación del modelo.

## Variantes dentro de la ronda

Los ajustes menores de entrenamiento (LR, callbacks o regularización) permanecen
en `round1/`, con sufijos `-a`, `-b`, etc. Cada variante conserva sus propios
resultados y documenta qué cambia. `round2/` se reserva para versiones superiores
de las arquitecturas o un cambio general de resolución.

La primera VGG16 se ha movido de `01_vgg16/` a `01_vgg16-a/`, conservando el modelo,
las predicciones, los artefactos, las figuras y las salidas del notebook. Su mejor
checkpoint es el de la época 2; terminó en la 22 y predice Building para todas las
imágenes de validación. Su accuracy es **19,36 %**. Se conserva como intento
fallido, no como referencia definitiva de rendimiento.

El identificador histórico `round1_01_vgg16` se mantiene en la variante a y sus
artefactos. La variante b usa `round1_01_vgg16-b`, arranca con pesos aleatorios y
no contiene resultados heredados de a.

## Ejecutar y conservar resultados

`NOTEBOOK_PATH`, `experiment_id` y `architecture` ya están configurados en cada
copia. Abre el notebook y utiliza «Ejecutar todo». Cada carpeta recibe su propio
`model.keras`, `prediction.json`, `artifacts/` y `plots/` con figuras PDF.
El JSON de test se genera siempre; no se crea ZIP ni se envía a Codabench.

La caché se crea con la primera preparación y se reutiliza en los siguientes
experimentos. Reejecutar un entrenamiento en la misma carpeta sobrescribe sus
resultados. Para repetir con otra semilla o configuración conservando esta prueba,
crea otra copia y actualiza `NOTEBOOK_PATH` e `experiment_id`.

## Comparaciones pendientes

Las llamadas de comparación permanecen comentadas y sus secciones no se han
modificado al crear las variantes. «Ejecutar todo» no lee otros experimentos ni
modifica las comparaciones existentes. La selección final se preparará en el
último cuaderno cuando terminen los entrenamientos; por ahora puede contener
rutas anteriores al renombrado y no debe activarse.
