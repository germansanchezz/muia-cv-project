# Transfer learning · protocolo y orden de ejecución

Se comparan **EfficientNetB0 y MobileNetV2** con las mismas arquitecturas entrenadas desde cero en el challenge anterior. Cada red tiene dos etapas independientes: entrenar una nueva cabeza con el backbone ImageNet congelado y después hacer fine-tuning parcial desde su mejor checkpoint. No se crean nuevos entrenamientos desde cero.

## Ejecutar en este orden

**Reinicia el kernel al abrir cada notebook y ejecuta todas sus celdas en orden.** No basta ejecutar solo la celda de entrenamiento para reiniciar una prueba.

| Orden | Notebook | Qué entrena | LR inicial | Máximo de épocas |
|---|---|---|---:|---:|
| 1 | [EfficientNetB0 experiment01](efficientnet_b0/experiment01/efficientnet_b0_tl_experiment01.ipynb) | Cabeza GAP + Dense13; backbone congelado | 1e-3 | 20 |
| 2 | [EfficientNetB0 experiment02](efficientnet_b0/experiment02/efficientnet_b0_tl_experiment02.ipynb) | Últimos bloques y cabeza, desde el mejor experiment01 | 1e-5 | 180 adicionales |
| 3 | [MobileNetV2 experiment01](mobilenet_v2/experiment01/mobilenet_v2_tl_experiment01.ipynb) | Cabeza GAP + Dense13; backbone congelado | 1e-3 | 20 |
| 4 | [MobileNetV2 experiment02](mobilenet_v2/experiment02/mobilenet_v2_tl_experiment02.ipynb) | Últimos bloques y cabeza, desde el mejor experiment01 | 1e-5 | 180 adicionales |

**El máximo es 200 épocas entre ambas etapas de cada arquitectura**, no 200 por etapa. EarlyStopping puede terminar antes. Los historiales de cada notebook empiezan en época local 1; los artefactos también registran épocas globales y totales.

Cada experiment02 necesita **el experiment01 de su arquitectura completamente ejecutado en ese clon**, con `model.keras` y `artifacts/`. No requiere el experiment01 de la otra arquitectura. Los modelos están excluidos de Git: disponer solo de las métricas en otro equipo no permite continuar el fine-tuning.

## Entorno y rutas

Dependencias: Python, TensorFlow/Keras con AdamW y `ModelCheckpoint.initial_value_threshold`, NumPy, scikit-learn, Rasterio, Matplotlib y Jupyter. El código se verifica con TensorFlow 2.15.1; no instala paquetes.

El dataset se busca en `PROJECT/xview_recognition/`. `PROJECT_DIR_OVERRIDE` permite indicar la ruta local a `PROJECT/`; `DATA_DIR_OVERRIDE` admite una ruta absoluta o relativa a `PROJECT/`. La localización funciona desde la raíz del clon, `PROJECT/` o las carpetas de experimentos, sin depender de los notebooks antiguos ni de rutas personales.

Los cuatro `NOTEBOOK_PATH` ya apuntan a su ubicación. Si se copian o renombran, hay que actualizarlos para que sus resultados no se escriban en la carpeta anterior. Una ruta absoluta del notebook también permite localizar el proyecto desde fuera del clon.

**La primera construcción de cada experiment01 descarga los pesos oficiales ImageNet**. La caché de Keras se configura antes de importar TensorFlow, mediante `KERAS_HOME=PROJECT/tl_challenge/.weights/`. Se requiere acceso a Internet en esa primera descarga o tener la caché correspondiente preparada en el equipo. Experiment02 carga el checkpoint local y no descarga otro backbone. Reiniciar el kernel asegura que Keras toma esta ubicación de caché.

## Protocolo común

| Parámetro | Valor |
|---|---|
| Entrada | RGB, 128×128 |
| Batch size | 32 |
| Cabeza | Global Average Pooling + Dense13 softmax, sin Dropout adicional |
| Inicialización del backbone | Pesos oficiales ImageNet de Keras, `include_top=False` |
| Optimizador | AdamW nuevo al comenzar cada etapa |
| Weight decay | 1e-4 |
| Pérdida | Categorical Cross Entropy con etiquetas one-hot |
| Label smoothing / MixUp / L2 adicional | 0.0 / desactivado / 0.0 |
| Aumentación de train | Rotación uniforme 0°, 90°, 180°, 270°; flips H/V independientes p=0.5 |
| Validación, evaluación de train y test | Sin aumentación |
| Semilla | 42, determinismo activado |
| Reparto | Histórico 90/10, `random_state=1`, sin estratificación, orden original |
| ModelCheckpoint | `val_accuracy`, modo max, guardar solo el mejor |
| ReduceLROnPlateau | `val_accuracy`, max, paciencia 10, factor 0.5, mínimo 1e-6 |
| EarlyStopping | `val_accuracy`, max, paciencia 20, restaurar mejores pesos |

Se conserva la partición de **16.871 imágenes de train y 1.875 de validación**, las trece clases y sus hashes, para leer las referencias anteriores sin cambiar el protocolo de datos. El test independiente se utiliza para generar predicciones, sin etiquetas públicas ni ajuste de hiperparámetros sobre él.

### Capas y preprocesamiento

En experiment01 el backbone completo se congela y solo se entrenan los **16.653 parámetros** de Dense13. En experiment02 se conserva la cabeza entrenada y se habilitan:

- **EfficientNetB0:** capas con prefijos `block6`, `block7`, `top_`.
- **MobileNetV2:** prefijos `block_15_`, `block_16_`, `Conv_1`, `out_relu`.

**Todas las BatchNorm quedan congeladas**, incluidos sus parámetros afines. El backbone se llama con `training=False` tanto al entrenar la cabeza como durante fine-tuning: esto fija las estadísticas BN y desactiva también las operaciones estocásticas internas del backbone. Las convoluciones seleccionadas sí reciben gradientes durante fine-tuning. Las capas con pesos entrenables y el número de parámetros se imprimen y guardan en `config.json`.

La caché entrega RGB float32 [0,1]. **El preprocesamiento ImageNet forma parte del modelo guardado**, para aplicarse igual en entrenamiento, evaluación y exportación:

- EfficientNetB0: multiplicar por 255 y usar su reescalado/normalización internos de Keras.
- MobileNetV2: transformar `x → 2x−1`, rango [-1,1].

No se cargan pesos preentrenados sobre la antigua entrada sin adaptar. El histórico desde cero no utilizaba exactamente todos estos pasos ImageNet; la diferencia se documenta en los artefactos y limita una interpretación donde solo hubiese cambiado la inicialización. Referencias: [Keras transfer learning](https://keras.io/guides/transfer_learning/), [EfficientNet](https://keras.io/api/applications/efficientnet/efficientnet_models/) y [MobileNetV2](https://keras.io/api/applications/mobilenet/).

### Conservar el mejor modelo entre ambas etapas

Experiment02 copia inicialmente el mejor `model.keras` de experiment01 a su propia carpeta. ModelCheckpoint usa su accuracy como umbral y **solo sustituye esa copia si una época de fine-tuning la supera estrictamente**. Los empates conservan el extractor. El cambio de capas entrenables se acompaña de recompilación y optimizador nuevo; no se reutiliza el estado de AdamW de la cabeza.

Por tanto, si el fine-tuning no ayuda, el modelo final y `prediction.json` siguen correspondiendo al extractor. El historial de experiment02 muestra el intento realizado, incluso si se descartó. Sus artefactos distinguen claramente:

- `stage`: entrenamiento realizado.
- `checkpoint_stage` y `selected_from_source`: origen del modelo seleccionado.
- `best_epoch`: época local de la etapa de origen del checkpoint.
- `best_epoch_this_stage`: mejor intento de la etapa actual.
- `best_epoch_global` y `epochs_total`: posición del checkpoint y coste en épocas dentro del procedimiento completo.
- `fit_seconds`: entrenamiento de la etapa actual; `fit_seconds_total`: cabeza + fine-tuning.

El gráfico individual no marca una época del extractor sobre el historial de fine-tuning como si fuese un checkpoint de esa curva. La evaluación y el test siempre recargan el archivo seleccionado en disco.

## Caché y archivos

Caché de píxeles propia compartida entre los cuatro notebooks en `PROJECT/tl_challenge/.cache/`, con lectura `mmap`, escritura imagen a imagen y publicación atómica. A 128×128 ocupa unos 3,43 GiB. No se reserva el dataset completo en RAM. Si se cambia el contenido de los TIFF sin cambiar las anotaciones, se debe eliminar su caché; la clave no incluye un hash del contenido de cada imagen.

`.cache/`, `.weights/` y `model.keras` están excluidos de Git. Cada experimento genera:

```text
experiment01/ o experiment02/
├── ...ipynb
├── model.keras
├── prediction.json
├── artifacts/
│   ├── config.json, split.json, architecture.json, model_summary.txt
│   ├── history.csv, history.json, timing.json
│   └── metrics.json, class_metrics.csv, validation_predictions.npz
└── plots/
    └── training_curves.pdf, learning_rate_time.pdf, confusion_matrix.pdf, precision_recall.pdf
```

**`prediction.json` se genera obligatoriamente después de entrenar**, antes de las métricas y los PDF. Conserva el esquema, categorías como texto, cajas y dimensiones originales. No se crea un ZIP ni se realiza un envío automático. Se mantiene el formato de artefactos de CNN y se añaden los campos TL, parámetros entrenables, preprocesamiento y hash del checkpoint de origen.

## Comparaciones preparadas

| Arquitectura | Referencia desde cero | Accuracy histórica |
|---|---|---:|
| EfficientNetB0 | [CNN experiment01](../cnn_challenge_alt/popular_architectures/efficientnet_b0/experiment01/efficientnet_b0_experiment01.ipynb) | 77,97 % |
| MobileNetV2 | [CNN experiment01](../cnn_challenge_alt/popular_architectures/mobilenet_v2/experiment01/mobilenet_v2_experiment01.ipynb) | 77,60 % |

La sección 9 de cada experiment01 compara **misma arquitectura desde cero + extractor congelado**. La de experiment02 compara **desde cero + extractor congelado + fine-tuning**. Las rutas de `SELECT_RUNS` son relativas a `PROJECT/`, por ejemplo:

```python
SELECT_RUNS = [
    'cnn_challenge_alt/popular_architectures/efficientnet_b0/experiment01',
    'tl_challenge/efficientnet_b0/experiment01',
    'tl_challenge/efficientnet_b0/experiment02',
]
```

No se utiliza ResNet18 ni la custom como referencia principal de esta fase: el enunciado pide comparar **la misma arquitectura**. Tampoco se confunde EfficientNetB0 con variantes superiores. Se comprueban hashes de datos, reparto, clases y arquitectura; se admiten capas congeladas y pesos ImageNet.

Las tablas y PDF se guardan en `tl_challenge/comparisons/<arquitectura>_experiment01/` o `..._experiment02/`. Incluyen etapa, pesos, checkpoint elegido, preprocesamiento, épocas locales/totales, parámetros entrenables y tiempos de fit locales/acumulados. **El gráfico de coste usa el tiempo acumulado para fine-tuning**, incluso si finalmente se conserva el extractor. Las curvas son locales a cada etapa y no se concatenan ni se prolongan.

Tras reiniciar el kernel, **primera celda de código + sección 9** bastan para comparar artefactos completos: no se cargan imágenes ni se reconstruyen modelos. Para regenerar solo los PDF individuales, primera celda + sección 8. Es necesario conservar el árbol del repositorio con los artefactos de CNN o ajustar las rutas seleccionadas.

El LR de fine-tuning, la política de congelación y el preprocesamiento requerido por ImageNet difieren del procedimiento desde cero. Las conclusiones se formularán como comparación de procedimientos de entrenamiento, y se documentará ese límite. Una sola semilla/partición no demuestra superioridad general ni sustituye la evaluación de test.

## Registro de resultados

| Arquitectura | Experimento | Objetivo | Estado |
|---|---|---|---|
| EfficientNetB0 | [experiment01](efficientnet_b0/experiment01/efficientnet_b0_tl_experiment01.ipynb) | Extractor ImageNet congelado | Completado: accuracy 70,72 %, F1 macro 71,66 %, época 20/20 |
| EfficientNetB0 | [experiment02](efficientnet_b0/experiment02/efficientnet_b0_tl_experiment02.ipynb) | Fine-tuning parcial desde experiment01 | Completado: accuracy 75,20 %, F1 macro 76,76 %, brecha 21,47 puntos |
| EfficientNetB0 | [experiment03](efficientnet_b0/experiment03/efficientnet_b0_tl_experiment03.ipynb) | Mismo fine-tuning desde experiment01, con MixUp alpha=0.2 | Completado: accuracy 76,96 %, F1 macro 79,25 %, brecha 14,98 puntos |
| EfficientNetB0 | [experiment04](efficientnet_b0/experiment04/efficientnet_b0_tl_experiment04.ipynb) | Todo el backbone salvo BatchNorm, con MixUp alpha=0.2 | Completado: accuracy 77,97 %, F1 macro 79,34 %, brecha 13,40 puntos |
| EfficientNetB0 | [experiment05](efficientnet_b0/experiment05/efficientnet_b0_tl_experiment05.ipynb) | Mismo fine-tuning completo y MixUp, entrada 192×192 | Completado: accuracy 80,16 %, F1 macro 82,02 %, brecha 12,16 puntos |
| EfficientNetB0 | [experiment06](efficientnet_b0/experiment06/efficientnet_b0_tl_experiment06.ipynb) | Mismo procedimiento a 192×192, LR inicial 3e-5 | Completado: accuracy 80,91 %, F1 macro 83,18 %, brecha 18,69 puntos |
| EfficientNetB0 | [experiment07](efficientnet_b0/experiment07/efficientnet_b0_tl_experiment07.ipynb) | Mismo procedimiento y LR 3e-5, MixUp alpha=0.4 | Completado: accuracy 80,80 %, F1 macro 82,62 %, brecha 9,37 puntos; opción equilibrada |
| MobileNetV2 | [experiment01](mobilenet_v2/experiment01/mobilenet_v2_tl_experiment01.ipynb) | Extractor ImageNet congelado | Preparado; entrenamiento pendiente |
| MobileNetV2 | [experiment02](mobilenet_v2/experiment02/mobilenet_v2_tl_experiment02.ipynb) | Fine-tuning parcial desde experiment01 | Preparado; requiere completar el anterior |

Después de entrenar, completar accuracy/F1/CE/AP, resultados por clase, brecha en inferencia, coste acumulado y si el fine-tuning mejora realmente al extractor. Mantener Codabench separado de validación.

Las comprobaciones automáticas de preparación usan muestras pequeñas y topologías reales con pesos aleatorios en un entorno temporal para evitar descargas y entrenamientos completos. **No equivalen a resultados de transfer learning ni validan el rendimiento de los pesos ImageNet**; los notebooks de producción sí solicitan obligatoriamente `weights='imagenet'` en experiment01.

## Revisión de EfficientNetB0 y compatibilidad GPU

Experiment01 alcanza **70,72 % de accuracy y 71,66 % de F1 macro**; train evaluado en inferencia da **77,83 %** y la brecha es de **7,11 puntos**. El entrenamiento tardó 2,58 minutos. La mejor época fue la última, **20/20**: se agotó el presupuesto de esta etapa, no se alcanzó la parada anticipada. No se puede afirmar que la cabeza haya convergido por completo.

Este resultado es plausible para un extractor congelado: solo se entrenaron los **16.653 parámetros** de Dense13, mientras las características ImageNet permanecieron fijas. La comparación con 77,97 % de EfficientNetB0 desde cero corresponde a otro procedimiento donde todas las capas se adaptaban a xView. El 70,72 % no demuestra por sí solo que el transfer learning esté mal configurado ni predice el resultado del fine-tuning. El preprocesamiento guardado es el previsto para ImageNet: entrada RGB [0,1], factor 255 y normalización interna de EfficientNet.

Experiment02 falló al calcular el primer gradiente con TensorFlow 2.15.1, GPU y determinismo activado:

```text
A deterministic GPU implementation of fused batch-norm backprop,
when training is disabled, is not currently available.
... top_bn/FusedBatchNormGradV3
```

Las BatchNorm estaban correctamente congeladas, pero el gradiente respecto a sus entradas todavía debe atravesarlas para actualizar las convoluciones seleccionadas. La limitación corresponde al kernel GPU de esa operación, no al resultado del extractor, al LR ni a la falta de memoria. Las pruebas iniciales en CPU no detectaron esta incompatibilidad.

**Corrección en los experiment02 de ambas arquitecturas:** se clona el modelo de origen conservando sus pesos y se sustituyen las BN por `FrozenBatchNormalization`, que calcula explícitamente la normalización de inferencia con operaciones elementales de TensorFlow. Gamma, beta, media y varianza permanecen congeladas. Se conserva el determinismo y se evita `FusedBatchNormGradV3`. El resto del protocolo, capas seleccionadas, LR y presupuesto no cambia. La implementación queda registrada en `config.json` como `batch_norm_implementation=explicit_frozen_inference`.

La clase está registrada en la primera celda de los notebooks de fine-tuning para guardar y recargar su `.keras`. **Para cargar esos checkpoints fuera del notebook, hay que registrar también esa clase o proporcionarla como custom object.** Los modelos originales de experiment01 siguen siendo modelos estándar y no se han modificado ni reentrenado.

Tras aplicar la corrección, se reinició el kernel y se completó EfficientNetB0 experiment02 desde el principio, sin repetir experiment01. La misma corrección está incluida para MobileNetV2 experiment02 y EfficientNetB0 experiment03. Un `model.keras` en una carpeta de fine-tuning fallido puede ser simplemente la copia inicial del extractor; no implica que la etapa se haya entrenado ni que tenga métricas válidas.

Verificación de la corrección: en CPU se comprueban pesos copiados exactamente, equivalencia numérica de predicciones antes de entrenar, ausencia de operaciones BN fusionadas en el grafo de gradientes, un paso finito de optimización, invariancia de todos los pesos BN y guardado/recarga. EfficientNet se prueba con el checkpoint ImageNet real de experiment01; MobileNet con su topología y pesos aleatorios. Estas comprobaciones no ejecutaron un entrenamiento completo en GPU; posteriormente se completó experiment02 y sus resultados se registran a continuación.

## EfficientNetB0: fine-tuning completado y prueba de MixUp

| Procedimiento | Accuracy train sin aumentación | Accuracy validación | F1 macro | CE validación | Brecha | Fit acumulado |
|---|---:|---:|---:|---:|---:|---:|
| Desde cero, CNN experiment01 | 88,95 % | **77,97 %** | 78,40 % | 0,7691 | 10,98 puntos | 48,41 min |
| TL experiment01, extractor | 77,83 % | 70,72 % | 71,66 % | 0,8613 | 7,11 puntos | 2,58 min |
| TL experiment02, fine-tuning | 96,67 % | 75,20 % | 76,76 % | 1,1581 | 21,47 puntos | 19,45 min |
| TL experiment03, fine-tuning + MixUp 0.2 | 91,94 % | 76,96 % | **79,25 %** | **0,7359** | 14,98 puntos | 21,35 min |

El fine-tuning mejora al extractor en **4,48 puntos**, pero queda **2,77 puntos por debajo de scratch**. El checkpoint elegido procede realmente de fine-tuning (`selected_from_source=false`), en la época local 56, global 76; se ejecutaron 76 épocas adicionales y 96 totales. La pérdida de validación alcanza su mínimo de 0,8103 en la época local 7 y después empeora, aunque la accuracy aún mejora. La brecha y esa evolución indican sobreajuste y son compatibles con errores de mayor confianza. No demuestran que falten épocas ni que ImageNet sea inútil; el procedimiento TL es más rápido. La adaptación limitada por las capas iniciales y BatchNorm congeladas es una hipótesis adicional, no una causa demostrada.

**Prueba de regularización completada: [experiment03](efficientnet_b0/experiment03/efficientnet_b0_tl_experiment03.ipynb), MixUp alpha=0.2.** Cambia únicamente la mezcla de imágenes y etiquetas de train, después de las transformaciones geométricas. Conserva 128×128, batch 32, AdamW LR 1e-5, weight decay 1e-4, capas entrenables, BN congeladas, semilla y callbacks de experiment02. Vuelve a partir del mejor experiment01, con optimizador nuevo; no continúa desde experiment02 ni requiere repetir la cabeza. Son dos ramas independientes: cada una dispone de hasta 180 épocas adicionales a las hasta 20 de la cabeza.

La comparación de experiment03 incluye scratch y experiment01/02/03 de TL. El coste acumulado suma experiment01 y esta nueva rama, sin añadir experiment02. Se conservan caché en disco, selección del extractor si no se supera y generación obligatoria de `prediction.json`. Las métricas finales de train y validación se calculan sin mezclas ni aumentación; la accuracy de train del historial con MixUp no se interpreta como la accuracy limpia. Se buscará mejorar accuracy/F1 y revisar CE y brecha, sin asumir que reducir solo la brecha supone mejorar.


## Experiment04: adaptar todos los bloques conservando MixUp

Experiment03 mejora a experiment02 en **1,76 puntos de accuracy**, reduce la brecha de 21,47 a **14,98 puntos** y la CE de 1,1581 a **0,7359**. Supera al EfficientNetB0 desde cero en F1 macro (79,25 % frente a 78,40 %) y CE, aunque su accuracy sigue **1,01 puntos por debajo**. El checkpoint corresponde a la época local 60, global 80, de 80 épocas de fine-tuning y 100 totales. La mejora es consistente con una regularización útil, pero aún queda sobreajuste; las métricas de una sola semilla no demuestran superioridad general.

La siguiente prueba es [EfficientNetB0 experiment04](efficientnet_b0/experiment04/efficientnet_b0_tl_experiment04.ipynb): **habilitar todos los pesos del backbone salvo BatchNorm**, manteniendo MixUp alpha=0.2 y todos los hiperparámetros de experiment03. Se comprueba así la hipótesis de que las características iniciales congeladas limitan la adaptación a imágenes aéreas. No se presupone que mejore; también podría aumentar el sobreajuste.

Se parte de nuevo del mejor experiment01, con AdamW nuevo, entrada 128×128, batch 32, LR 1e-5, weight decay 1e-4, mismos callbacks y hasta 180 épocas adicionales (200 contando la cabeza en esta rama). Las BN mantienen congelados parámetros y estadísticas mediante la implementación explícita; el backbone continúa en modo de inferencia para BN y operaciones estocásticas, con gradientes habilitados en todas las convoluciones.

**Ejecución:** reiniciar el kernel y ejecutar experiment04 completo. No repetir experiment01 ni continuar desde experiment03. Se conservan caché propia en disco, preprocesamiento ImageNet, artefactos y `prediction.json` obligatorio. La comparación incluye scratch y experiment01/02/03/04 de TL, con salida en `tl_challenge/comparisons/efficientnet_b0_experiment04/`. El coste acumulado suma solo la cabeza y esta rama de fine-tuning. Entrenar todos los bloques puede aumentar tiempo por época y memoria respecto al ajuste parcial.


## Experiment04 completado y prueba de resolución en experiment05

Experiment04 alcanza **77,97 % de accuracy**, **79,34 % de F1 macro**, **CE 0,6711** y **brecha 13,40 puntos**, con 91,38 % de accuracy sobre train limpio. El checkpoint está en la época local 41 (global 61); se ejecutaron 61 épocas adicionales y 81 totales, con 31,63 minutos de fit acumulado. Frente a experiment03 gana **1,01 puntos de accuracy** y reduce CE y brecha. Iguala exactamente la accuracy del EfficientNetB0 desde cero y mejora su F1 macro (78,40 %) y CE (0,7691). Esto apoya la utilidad de adaptar también los primeros bloques; no demuestra superioridad general con una sola semilla.

**Prueba preparada: [experiment05](efficientnet_b0/experiment05/efficientnet_b0_tl_experiment05.ipynb), entrada 192×192.** Mantiene todos los demás ajustes de experiment04: MixUp alpha=0.2, todo el backbone salvo BN, batch 32, AdamW LR 1e-5, weight decay 1e-4, callbacks y hasta 180 épocas adicionales. Se reconstruyen la entrada y el backbone del mejor experiment01 a 192, copiando exactamente sus pesos, incluida la cabeza. No se continúa desde experiment04 ni se repite la cabeza. Cada rama tiene máximo 200 épocas contando el extractor de origen.

La nueva caché mmap lee los TIFF originales a 192×192 y ocupa aproximadamente 7,72 GiB, conservando la de 128. Se espera más coste por lote. Antes del fit se evalúa el extractor adaptado a la nueva resolución para establecer el umbral de selección: si no se mejora, se conserva ese mismo modelo de entrada 192 (`checkpoint_stage=feature_extraction_resized`), no el archivo original de entrada 128 ni sus métricas históricas. Se registran ambas accuracies de referencia en config.

Reiniciar el kernel y ejecutar experiment05 completo. Se conservan `prediction.json` obligatorio, artefactos, PDF y comparación con scratch y experiment01/02/03/04/05 de TL. El coste acumulado suma solo la cabeza original y esta rama. Scratch usa 128×128: la comparación también incorpora una diferencia de resolución, no solamente de inicialización.


## Experiment05 completado y prueba de LR en experiment06

Experiment05 obtiene **80,16 % de accuracy**, **82,02 % de F1 macro**, **CE 0,6236** y **brecha 12,16 puntos**, con train limpio al 92,32 %. El checkpoint seleccionado es de fine-tuning (`selected_from_source=false`), en la época local 43, global 63; se ejecutaron 63 épocas adicionales y 83 totales. El fit acumulado fue **71,80 minutos**. Frente a experiment04 gana **2,19 puntos de accuracy** (41 aciertos en validación), mejora F1/CE y reduce la brecha; el coste pasa de 31,63 a 71,80 minutos. Supera a EfficientNetB0 scratch en accuracy, F1 y CE, aunque la comparación incluye diferencias de resolución, regularización y entrenamiento. ResNet18 experiment04 alcanzó 80,11 % y la mejor custom 80,69 %: la diferencia de un acierto con ResNet no demuestra superioridad general.

**Siguiente prueba: [experiment06](efficientnet_b0/experiment06/efficientnet_b0_tl_experiment06.ipynb), LR inicial 3e-5 frente a 1e-5.** Es el único cambio respecto a experiment05. Se conservan entrada 192×192, batch 32, MixUp alpha=0.2, AdamW, weight decay 1e-4, todo el backbone salvo BN, semilla, reparto, aumentación y callbacks, incluido mínimo de LR 1e-6. Se comprueba otra velocidad de adaptación, sin afirmar que las curvas demuestren un LR insuficiente.

Vuelve a partir del mismo experiment01 de 128×128, con entrada reconstruida a 192 y todos los pesos copiados, incluido el clasificador. No continúa desde experiment05 ni requiere repetir la cabeza. Se reevalúa el extractor adaptado antes del fit para seleccionar checkpoints a la resolución correcta. Máximo 180 épocas adicionales y 200 contando la cabeza de esta rama; se reutiliza la caché mmap de 192. Reiniciar el kernel y ejecutar el notebook completo.

Cada ejecución de experiment06 genera **su propio `prediction.json` nuevo**, calculado sobre test con su checkpoint seleccionado; no copia predicciones anteriores. Los artefactos y PDF se escriben en su carpeta. La comparación incluye scratch y experiment01/02/03/04/05/06 de TL, en `tl_challenge/comparisons/efficientnet_b0_experiment06/`. El coste acumulado suma solo la cabeza y esta rama.


## Experiment06 completado y última prueba: MixUp 0.4

Experiment06 alcanza **80,91 % de accuracy**, **83,18 % de F1 macro**, **CE 0,6962** y **brecha 18,69 puntos**, con **99,60 %** en train limpio. El checkpoint es de fine-tuning, en la época local 52, global 72; se ejecutaron 72 épocas adicionales y 92 totales, con **87,30 minutos** de fit acumulado. Frente a experiment05 gana **0,75 puntos de accuracy** (14 aciertos) y 1,17 puntos de F1 macro, pero empeora CE y brecha. La CE mínima de validación fue 0,6139 en la época 20: el máximo de accuracy posterior no supone mejorar la calidad de probabilidades. El mayor LR permitió ajustar mucho más train y vino acompañado de mayor sobreajuste. El resultado supera ligeramente a la mejor custom en accuracy (80,69 %), con solo cuatro aciertos de diferencia; no demuestra superioridad general.

**Última prueba prevista: [experiment07](efficientnet_b0/experiment07/efficientnet_b0_tl_experiment07.ipynb), MixUp alpha=0.4 frente a 0.2.** Es el único cambio respecto a experiment06. Se conservan 192×192, batch 32, AdamW LR inicial 3e-5, weight decay 1e-4, todo el backbone salvo BN, callbacks, reparto, semilla y aumentación geométrica. Se comprobará si reforzar la regularización conserva la mejora de accuracy/F1 reduciendo CE y brecha; no se presupone que funcione. Reducir solo la brecha no basta para elegir la configuración final.

Se vuelve a partir del mismo checkpoint de experiment01 con la entrada adaptada a 192; no se continúa desde experiment06 ni se repite la cabeza. Se reutiliza la caché mmap de 192 y se mantienen hasta 180 épocas adicionales, máximo 200 contando la cabeza en esta rama independiente. Se evalúa el extractor adaptado antes del fit para fijar el umbral de selección y conservarlo si no se mejora. Reiniciar el kernel y ejecutar experiment07 completo.

Se genera **un `prediction.json` nuevo** desde el checkpoint seleccionado de esta ejecución, además de artefactos y PDF en su propia carpeta. La comparación incluye scratch y experiment01/02/03/04/05/06/07 de TL, con salida en `tl_challenge/comparisons/efficientnet_b0_experiment07/`. El coste acumulado suma solo la cabeza original y esta rama.


## Experiment07 completado: balance final de EfficientNetB0

[Resultados detallados y análisis por clase](efficientnet_b0/experiment07/README.md#resultados-y-conclusión).

| Procedimiento | Accuracy validación | F1 macro | CE validación | AP macro | Brecha limpia |
|---|---:|---:|---:|---:|---:|
| EfficientNetB0 desde cero, 128×128 | 77,97 % | 78,40 % | 0,7691 | 84,59 % | 10,98 puntos |
| TL experiment05, 192×192, LR 1e-5, MixUp 0.2 | 80,16 % | 82,02 % | 0,6236 | 86,71 % | 12,16 puntos |
| TL experiment06, 192×192, LR 3e-5, MixUp 0.2 | **80,91 %** | **83,18 %** | 0,6962 | 86,25 % | 18,69 puntos |
| TL experiment07, 192×192, LR 3e-5, MixUp 0.4 | 80,80 % | 82,62 % | **0,6037** | **87,11 %** | **9,37 puntos** |

MixUp 0.4 sacrifica solo **dos aciertos** respecto a experiment06, reduce la brecha en **9,32 puntos** y mejora CE/AP. El F1 macro baja **0,57 puntos**: no es una mejora uniforme, y empeoran Truck y Fishing vessel, entre otras clases. Train limpio es 90,17 %, frente al 99,60 % anterior, con validación casi igual. La regularización parece útil; esta única semilla y partición no permite demostrar superioridad general ni una mejor calibración a partir de la CE por sí sola.

Se seleccionó un checkpoint real de fine-tuning, época local **18**, global **38**, y se ejecutaron **38 épocas adicionales / 58 totales**. Los pesos iniciales y de origen coinciden con experiment06. El fit acumulado registrado es **26,56 minutos**. Aunque hay menos épocas que antes, también se observa menor tiempo por época (aproximadamente 38 frente a 71 segundos); el mismo entorno registrado no explica ese cambio. No se atribuye toda la diferencia de tiempo a MixUp. `prediction.json` corresponde al checkpoint seleccionado de esta ejecución y se genera en su propia carpeta.

**Recomendación final para EfficientNetB0:** conservar experiment07 como opción equilibrada y experiment06 como mejor accuracy/F1 de validación. Experiment07 supera al B0 scratch en accuracy (+2,83 puntos), F1, CE, AP y brecha; la comparación incluye también cambios de resolución y regularización. Supera por solo dos aciertos a la mejor custom (80,69 %), por lo que no se afirma superioridad general frente a ella.

**Se recomienda cerrar el afinamiento de EfficientNetB0 aquí.** Ya se ha estudiado una secuencia suficiente de factores y no hay necesidad de abrir otra búsqueda con B1/B2 para completar esta comparación. Si se invierte más entrenamiento, repetir las configuraciones finalistas con otras semillas aportaría más evidencia que seguir ajustando sobre la misma validación. Esta recomendación se refiere a EfficientNetB0, no declara completados los experimentos pendientes de otras arquitecturas de TL. No se han creado nuevos notebooks.
