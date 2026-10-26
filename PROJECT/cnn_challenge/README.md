# CNN challenge

La [plantilla](baseline/cnn_challenge_baseline.ipynb) permite entrenar CNN propias
y arquitecturas populares para clasificar las 13 categorías de xView. Se entrena
**una red por ejecución, siempre desde cero**, y se guardan los resultados para
compararlos más tarde sin volver a entrenar.

## 1. Entorno y datos

Dependencias: Python, TensorFlow/Keras, NumPy, scikit-learn, Rasterio, Matplotlib
y Jupyter. El notebook no instala paquetes automáticamente.

El dataset se busca en `PROJECT/xview_recognition/`. Para usar otra ubicación,
configura `DATA_DIR_OVERRIDE` en la primera celda de código. Si abres Jupyter
fuera del repositorio y no se localiza `PROJECT/`, indica `PROJECT_DIR_OVERRIDE`.

Se conserva el reparto de las fases anteriores: **16.871 imágenes para entrenar
y 1.875 para validar**, con `random_state=1`, sin estratificación y manteniendo
el orden original de las imágenes. La semilla del entrenamiento se fija por
separado. El test es un conjunto independiente, sin etiquetas públicas.

## 2. Preparar y ejecutar un experimento

1. Copia la plantilla en una carpeta propia, por ejemplo
   `experiments/cnn_001/` o `popular_architectures/resnet50_001/`.
2. En la primera celda de código, cambia `NOTEBOOK_PATH` para que apunte a la
   copia que vas a ejecutar. La ruta es relativa a `PROJECT/` o absoluta:

   ```python
   NOTEBOOK_PATH = Path('cnn_challenge/experiments/cnn_001/mi_experimento.ipynb')
   ```

   Si ejecutas la plantilla en su ubicación actual, utiliza:

   ```python
   NOTEBOOK_PATH = Path('cnn_challenge/baseline/cnn_challenge_baseline.ipynb')
   ```

3. Completa los autores, la hipótesis y la ficha del experimento. Ajusta `CFG`
   según la arquitectura y los parámetros que quieras probar.
4. Ejecuta el notebook de arriba abajo. Si no quieres comparar todavía, omite
   la celda del apartado 9 o comenta sus dos llamadas finales, como se explica
   en la sección 7 de este README.
5. Comprueba los archivos generados y completa la ficha de resultados del
   apartado 10 del notebook.

**`NOTEBOOK_PATH` determina dónde se guardan los resultados.** Repetir un
entrenamiento en esa carpeta sobrescribe sus archivos anteriores. Para conservar
otra prueba, utiliza otra carpeta y actualiza esa ruta. `CFG['experiment_id']`
identifica el experimento en la configuración, pero no cambia el destino.

Al ejecutar el notebook completo se genera siempre `prediction.json` con el
mejor checkpoint. Conserva las categorías como texto y las dimensiones y cajas
originales, aunque la entrada de la red se reduzca. No se crea un ZIP ni se
realiza ningún envío automático a Codabench.

## 3. Configuración y arquitecturas

La configuración inicial es una CNN propia de tres bloques, entrada 64×64, SGD,
Dropout 0.1 y sin L2, Batch Normalization ni aumentación. Es un punto de partida
editable, no un modelo optimizado.

| Parámetro | Qué controla |
|---|---|
| `architecture` | CNN propia o variante popular del catálogo inferior |
| `image_size`, `batch_size` | Resolución de entrada y tamaño del lote |
| `optimizer`, `learning_rate`, `momentum`, `clipnorm` | Optimización; momentum se usa con SGD |
| `max_epochs`, `monitor`, `monitor_mode` y parámetros de los callbacks | Presupuesto, selección del checkpoint y parada |
| `seed`, `deterministic` | Reproducibilidad del entrenamiento |
| `dropout`, `augmentation`, `flip`, `rotation` | Regularización y aumentación |
| `filters`, `convs_per_block`, `kernel_size`, `batch_norm`, `l2` | Arquitectura y regularización de la CNN propia |
| `mobilenet_alpha` | Anchura de MobileNet y MobileNetV2 |

El catálogo sigue el tema [1.CNN-Architectures.pdf](1.CNN-Architectures.pdf):

| Familia | Valores de `CFG['architecture']` |
|---|---|
| Propia | `custom` |
| LeNet | `lenet5` |
| AlexNet | `alexnet` |
| VGG | `vgg11`, `vgg13`, `vgg16`, `vgg19` |
| GoogLeNet / Inception | `googlenet`, `inception_v2`, `inception_v3`, `inception_v4` |
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| Inception-ResNet | `inception_resnet_v2` |
| ResNeXt | `resnext50`, `resnext101` (32×4d) |
| DenseNet | `densenet121`, `densenet169`, `densenet201` |
| MobileNet | `mobilenet`, `mobilenet_v2` |
| EfficientNet | `efficientnet_b0` … `efficientnet_b7` |

Las aplicaciones de Keras usan `weights=None` y todas sus capas se entrenan.
Las demás se implementan en el notebook. Las populares conservan las capas BN
que pertenecen a su arquitectura y requieren `l2=0` en esta plantilla;
`batch_norm` y `l2` solo controlan la CNN propia.

### Adaptaciones y resolución mínima

Todas las redes se adaptan a RGB y 13 clases. LeNet-5 usa convoluciones densas
y pooling sin pesos aprendidos. AlexNet conserva FC4096×2 y usa padding SAME en
la primera convolución. GoogLeNet e Inception-v4 omiten las cabezas auxiliares;
VGG sustituye sus FC originales por GAP. Inception-v2 y v4 siguen las topologías
de TF-Slim enlazadas en el apartado 4 del notebook, implementadas con capas Keras.

La tasa de Dropout se configura explícitamente: el valor inicial 0.1 no reproduce
la regularización histórica de todas las arquitecturas. EfficientNet incluye
un ajuste de escala para compensar su preprocesamiento interno de Keras.

AlexNet requiere al menos **64×64**; InceptionV3, InceptionV4 e Inception-ResNetV2,
**75×75**; las demás populares, **32×32**. En la CNN propia, la resolución debe
permitir todos sus poolings. Estos mínimos técnicos no garantizan buen rendimiento.
El apartado 4 del notebook detalla las implementaciones y sus adaptaciones.

## 4. Caché y uso de memoria

Las imágenes se decodifican, reducen y normalizan una vez para crear una caché
compartida en `PROJECT/cnn_challenge/.cache/`. Se escribe imagen a imagen, sin
reservar el dataset completo en RAM, y se reutiliza entre experimentos con los
mismos datos, resolución y preprocesamiento.

`CACHE_MODE` se configura en la primera celda, fuera de `CFG`:

| Modo | Comportamiento |
|---|---|
| `'mmap'` — predeterminado | Mapea el archivo y entrega lotes sin cargar explícitamente todo el array en RAM |
| `'ram'` | Carga el array completo en memoria desde la caché |

El sistema operativo puede mantener páginas del archivo en RAM y recuperarlas
cuando necesita memoria. A 64×64, la caché float32 ocupa unos **0,86 GiB** en disco;
a 224×224, unos **10,5 GiB**. El modelo y los lotes necesitan memoria adicional.

La primera preparación añade el coste de escribir la caché. Con mapeo, el
entrenamiento puede ser más lento que con todo en RAM, especialmente con disco
lento; con SSD y la caché del sistema caliente puede ser similar. Si la precarga
provocaba paginación por falta de RAM, el mapeo puede mejorar el rendimiento.
Ambos modos entregan los mismos píxeles; no se ha medido su diferencia de
velocidad en cada equipo.

Cambiar la resolución requiere volver a ejecutar la preparación de datos y las
celdas siguientes; se crea o reutiliza su caché correspondiente. Si modificas el
contenido de los TIFF sin cambiar las anotaciones, elimina la caché afectada.
El `.gitignore` general del repositorio excluye esta carpeta.

## 5. Archivos generados

Cada experimento guarda el modelo y las predicciones junto al notebook. Los
archivos auxiliares se agrupan en `artifacts/` y sus figuras en `plots/`:

```text
carpeta_del_experimento/
├── mi_experimento.ipynb
├── model.keras
├── prediction.json
├── artifacts/
│   ├── architecture.json
│   ├── config.json
│   ├── split.json
│   ├── model_summary.txt
│   ├── history.csv
│   ├── history.json
│   ├── timing.json
│   ├── metrics.json
│   ├── class_metrics.csv
│   └── validation_predictions.npz
└── plots/
    └── figuras en PDF
```

| Archivo auxiliar | Contenido |
|---|---|
| `architecture.json`, `model_summary.txt` | Estructura de la red y resumen de capas y parámetros |
| `config.json` | Hiperparámetros, semilla, entorno e identificación del protocolo |
| `split.json` | Imágenes y etiquetas del reparto de entrenamiento y validación |
| `history.csv`, `history.json` | Evolución por época; el CSV se actualiza durante el entrenamiento |
| `timing.json` | Tiempo total de entrenamiento |
| `metrics.json`, `class_metrics.csv` | Evaluación del mejor checkpoint y métricas por clase |
| `validation_predictions.npz` | Etiquetas, probabilidades, nombres de archivo y matriz de confusión de validación |

Las comparaciones se guardan en `PROJECT/cnn_challenge/comparisons/`: tablas,
resumen de semillas y figuras PDF. Se actualizan al volver a ejecutar la
comparación. Las funciones de lectura también aceptan el formato antiguo con
auxiliares junto al notebook.

## 6. Gráficas y métricas

**Todas las figuras se exportan únicamente en PDF.**

| Por experimento, en `plots/` | Contenido |
|---|---|
| `training_curves.pdf` | Accuracy, pérdida total y entropía cruzada de train y validación; marca el mejor checkpoint |
| `learning_rate_time.pdf` | LR utilizado y segundos por época |
| `confusion_matrix.pdf` | Matriz de confusión normalizada por clase real |
| `precision_recall.pdf` | Curvas por clase, AP y prevalencia como referencia |

| Entre experimentos, en `comparisons/` | Contenido |
|---|---|
| `comparison_curves.pdf` | Curvas superpuestas de accuracy, CE de validación y LR |
| `comparison_metrics.pdf` | Accuracy, precisión macro, recall macro y F1 macro |
| `comparison_class_recall.pdf` | Mapa de calor del recall por clase y modelo |
| `comparison_cost.pdf` | Accuracy frente al número de parámetros y al tiempo de entrenamiento |

La pérdida total incluye L2 cuando se utiliza; la entropía cruzada (`ce`) se
registra aparte para poder comparar. Las curvas de train se calculan con la
regularización y aumentación activas; la brecha final se calcula con train y
validación en modo inferencia. Las métricas finales corresponden al mejor
checkpoint, no necesariamente a la última época ejecutada.

Si dos entrenamientos duran distinto, cada curva termina en su última época;
no se rellenan ni prolongan valores. El resumen `seed_summary.json` agrupa
configuraciones iguales salvo semilla e identidad de ejecución, y registra media
y desviación muestral cuando hay varias semillas distintas. Con una sola semilla
no se estima dispersión.

## 7. Comparar resultados sin volver a entrenar

Las comparaciones se controlan en el apartado **9. Comparar CNN propias y
populares** del notebook. `SELECT_RUNS` contiene las carpetas que se van a leer,
con rutas relativas a `PROJECT/cnn_challenge/`:

```python
SELECT_RUNS = [
    'baseline',
    'popular_architectures/resnet50_001',
    'experiments/cnn_001',
]
```

Con `SELECT_RUNS = []` se buscan todas las ejecuciones completas bajo
`cnn_challenge/`, incluida la actual si tiene los resultados necesarios.
Las comparaciones exigen el mismo dataset, reparto y orden de clases.
Los cambios de resolución, optimización y regularización deben tenerse en cuenta
al interpretar las métricas y los tiempos.

### Ejecutar las comparaciones más tarde

Si conservas el kernel, ajusta `SELECT_RUNS` y vuelve a ejecutar la celda del
apartado 9.

Si has reiniciado el kernel o vuelves a abrir el notebook:

1. Ejecuta la primera celda de código, con los imports, la configuración y las
   utilidades compartidas, incluida `save_figure`. Comprueba `NOTEBOOK_PATH`.
2. Ajusta `SELECT_RUNS` y ejecuta la celda del apartado 9.

No hace falta cargar las imágenes, reconstruir la red, entrenar ni ejecutar el
apartado 8. Para regenerar solo las gráficas individuales, ejecuta la primera
celda de código y el apartado **8. Figuras del experimento**.

### Omitir las comparaciones al entrenar

Puedes saltarte la celda del apartado 9. Si utilizas «Ejecutar todo», comenta
sus dos llamadas finales:

```python
# runs = load_runs(CHALLENGE_DIR, SELECT_RUNS)
# compare_runs(runs, CHALLENGE_DIR / 'comparisons')
```

El entrenamiento, las predicciones de test y las gráficas individuales siguen
funcionando. Mantén los resultados de validación y los de Codabench separados
en la ficha de cada experimento.
