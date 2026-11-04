# Baseline CNN alternativo

[Notebook](cnn_challenge_baseline.ipynb) independiente del baseline original, con un catálogo limitado a `custom`, `resnet18`, `densenet121`, `efficientnet_b0` y `mobilenet_v2`. Todas las redes se entrenan desde cero, sin descargar pesos.

El baseline mantiene la CNN pequeña original: entrada 64×64, tres bloques de dos convoluciones (32/64/128 filtros), SGD, LR 0.001, batch 32, Dropout 0.1, sin BN, L2 ni aumentación. Su pérdida es Categorical Cross Entropy con etiquetas one-hot.

## Ejecutar o copiar

1. Abre el notebook y ejecuta sus celdas en orden.
2. Para un experimento nuevo, copia el notebook a otra carpeta dentro de `cnn_challenge_alt/` y actualiza `NOTEBOOK_PATH`, relativo a `PROJECT/`. Esa ruta determina dónde se guardan los resultados.
3. Ajusta `CFG`, identidad e hipótesis. Para una CNN propia, edita `build_custom_body`.

Dependencias: TensorFlow/Keras con AdamW, NumPy, scikit-learn, Rasterio, Matplotlib y Jupyter. Los datos se buscan en `PROJECT/xview_recognition/`; hay overrides para las rutas de datos y proyecto.

## Rutas al compartir el repositorio

No hay rutas personales ni dependencias de la carpeta `cnn_challenge/` original ni de otros notebooks. Se localiza `PROJECT/` buscando la carpeta `cnn_challenge_alt/` desde el directorio de trabajo y sus padres. Funciona si Jupyter se inicia desde la raíz del clon, `PROJECT/`, el baseline o una carpeta de experimentos.

`NOTEBOOK_PATH` se configura explícitamente porque Jupyter no proporciona de forma fiable la ruta del notebook al kernel. **Al copiar o renombrar el notebook, actualiza esta variable**, aunque el notebook anterior siga existiendo: de lo contrario los resultados se guardarían en la carpeta anterior. Usa rutas relativas a `PROJECT/`, por ejemplo:

```python
NOTEBOOK_PATH = Path('cnn_challenge_alt/experiments/experiment_01/cnn_experiment_01.ipynb')
```

Tanto `NOTEBOOK_PATH` como `DATA_DIR_OVERRIDE` aceptan rutas absolutas o relativas a `PROJECT/`, con independencia del directorio de trabajo. Si se inicia Jupyter fuera del clon, configura `PROJECT_DIR_OVERRIDE` con la ruta local a `PROJECT/` o utiliza un `NOTEBOOK_PATH` absoluto. Las rutas también admiten `~`.

Cada compañero debe tener las dependencias instaladas y los datos en `PROJECT/xview_recognition/`, o configurar `DATA_DIR_OVERRIDE` con su ubicación. Los datos, la caché y `model.keras` están excluidos de Git; cada clon genera su propia caché. La configuración inicial y las comparaciones desde artefactos no requieren que el dataset esté disponible.

## Protocolo y caché

Se conserva el orden de las imágenes y el reparto histórico 90/10, `random_state=1`, sin estratificación. La semilla del entrenamiento es 42 por defecto. Los hashes de datos, reparto y clases permiten detectar comparaciones incompatibles.

La caché propia de esta versión está en `PROJECT/cnn_challenge_alt/.cache/`: RGB float32 [0,1], reducción `Resampling.average` y lectura mediante `mmap`. Se escribe imagen a imagen en un temporal y se publica al terminar; el entrenamiento solo copia lotes y anticipa uno. Otros experimentos con los mismos datos y resolución reutilizan sus píxeles. A 128×128 ocupa unos 3,43 GiB. Si cambias el contenido de los TIFF, elimina su caché: el hash de anotaciones no detecta esos cambios. La carpeta ya está excluida por el `.gitignore` del repositorio.

Con `augmentation=True`, cada imagen de entrenamiento recibe una rotación uniforme entre 0°, 90°, 180° y 270°, seguida de flips horizontal y vertical independientes con probabilidad 0.5. No hay interpolación de la rotación. Se requieren imágenes cuadradas. La evaluación de train, validación y test no usan aumentación.

## Arquitecturas

ResNet18 se implementa con bloques básicos [2,2,2,2] y stem 7×7/stride 2. Las otras tres redes usan aplicaciones de Keras con `weights=None`, sin cabeza original, y GAP + Dense13 softmax. EfficientNet recibe un factor ×255 para compensar su reescalado interno; las demás conservan la entrada [0,1] del protocolo original. `batch_norm` y `l2` solo controlan la CNN propia. AdamW permite weight decay en todas.

## Resultados y comparaciones

Cada ejecución escribe junto al notebook:

- `model.keras`: mejor checkpoint.
- `prediction.json`: generación obligatoria inmediatamente después del entrenamiento, antes de métricas y gráficas. Se recarga el checkpoint; se conservan categorías como texto, dimensiones y cajas originales. Un error de datos o entrenamiento debe resolverse antes de poder exportar predicciones válidas.
- `artifacts/`: configuración y entorno, arquitectura, resumen, reparto, historial CSV/JSON, tiempos, métricas agregadas y por clase, probabilidades de validación y matriz de confusión.
- `plots/`: curvas de entrenamiento, LR/tiempo, matriz de confusión y precision–recall, en PDF.

Reejecutar en la misma carpeta sobrescribe sus resultados. Usa otra carpeta para conservar un experimento. Al empezar el entrenamiento se eliminan resultados anteriores que podrían confundirse con la nueva ejecución.

Para comparar después de reiniciar el kernel, ejecuta **solo la primera celda de código y la sección 9**. No requiere cargar imágenes ni entrenar. `SELECT_RUNS=[]` descubre ejecuciones completas dentro de `cnn_challenge_alt/`; también puedes indicar carpetas relativas a esa raíz. Para regenerar únicamente las figuras individuales, ejecuta la primera celda y la sección 8.

Las comparaciones escriben en `cnn_challenge_alt/comparisons/` una tabla CSV, curvas, métricas, recall por clase, costes y resumen de semillas. Se comprueba el mismo dataset, reparto, orden de clases y entrenamiento desde cero. Las columnas incluyen weight decay, label smoothing, aumentación y parámetros de los callbacks. `ce` mide entropía cruzada sin suavizado ni L2, aunque la pérdida de entrenamiento pueda incluirlos. Se conserva el esquema de artefactos del challenge original.

No se incluyen entrenamientos ni resultados nuevos con esta plantilla.
