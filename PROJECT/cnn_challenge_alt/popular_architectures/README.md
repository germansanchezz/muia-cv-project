# Arquitecturas populares · protocolo y registro de experimentos

**Fase de entrenamiento de redes populares finalizada.** Se documentan nueve ejecuciones de cuatro arquitecturas. La referencia final por accuracy y F1 macro de validación es **ResNet18 experiment04, a 192×192 y con label smoothing 0.1** (80,11 % / 82,24 %). ResNet18 experiment03 a 128×128 se conserva como alternativa de menor coste. No se planifican más entrenamientos en esta parte.

Comparamos ResNet18, DenseNet121, EfficientNetB0 y MobileNetV2 para las 13 clases de xView. **El diseño inicial establece igualdad de condiciones:** los `experiment01` comparten datos, reparto, semilla, resolución, aumentación, optimizador y criterio de selección. La comparación estudia el cambio de arquitectura; la ejecución de ResNet18 tiene una diferencia en el máximo de épocas, documentada abajo.

**El máximo común es de 200 épocas para todos los entrenamientos**, con EarlyStopping. La ejecución histórica de ResNet18 experiment01 con máximo 100 fue una excepción por error; se conserva porque terminó antes de ese límite.

**Todos los entrenamientos empiezan desde cero**, con inicialización aleatoria, sin pesos preentrenados ni reutilización de checkpoints de otra ejecución. La misma semilla hace reproducible cada configuración; los pesos iniciales y las operaciones internas dependen de la arquitectura.

## Condiciones comunes previstas para experiment01

| Parámetro | Valor |
|---|---|
| Entrada | RGB, 128×128, float32 [0,1] |
| Batch size | 32 |
| Optimizador | AdamW |
| Learning rate inicial | 0.001 |
| Weight decay | 0.0001 |
| Pérdida | Categorical Cross Entropy, etiquetas one-hot |
| Label smoothing | 0.0 |
| Máximo de épocas | 200 |
| Dropout adicional | Ninguno |
| L2 adicional | 0.0 |
| Clipping de gradientes | Ninguno |
| Rotación de train | Por imagen, uniforme entre 0°, 90°, 180° y 270° |
| Flip horizontal de train | Probabilidad 0.5 |
| Flip vertical de train | Probabilidad 0.5, independiente del horizontal |
| Semilla de entrenamiento | 42, operaciones deterministas activadas |
| Reparto | Histórico 90/10, `random_state=1`, sin estratificación, orden original de imágenes |
| Capas entrenables | Todas |
| Cabeza | Global Average Pooling + Dense13 softmax |

La aumentación se aplica solo al entrenamiento. Se conservan las capas y regularizaciones internas de cada arquitectura. ResNet18 usa bloques básicos [2,2,2,2]; las otras tres usan aplicaciones de Keras con `weights=None`. EfficientNetB0 incluye el factor ×255 que compensa su reescalado interno de Keras; todas reciben los mismos píxeles de la caché.

Los tres callbacks comparten monitor `val_accuracy` y modo `max`:

| Callback | Configuración |
|---|---|
| ReduceLROnPlateau | Paciencia 10, factor 0.5, LR mínimo 1e-6, `min_delta=0` |
| EarlyStopping | Paciencia 20, `restore_best_weights=True`, `min_delta=0` |
| ModelCheckpoint | `save_best_only=True`, guarda el mejor `model.keras` |

Se selecciona el primer máximo de accuracy de validación y se recarga ese checkpoint para test y evaluación final. El diseño previsto fija el mismo presupuesto máximo y las mismas reglas; las épocas realmente ejecutadas pueden variar por EarlyStopping. La diferencia de presupuesto registrada se detalla a continuación. Las diferencias de parámetros, regularización interna y tiempo forman parte de cada arquitectura y deben tenerse en cuenta al comparar.

### Comprobación de las ejecuciones

Los cuatro `config.json` y `split.json` confirman el mismo dataset, orden de clases y reparto: **16.871 imágenes de train y 1.875 de validación**, con idénticos hashes de datos y partición. Coinciden el resto de parámetros del protocolo y el entorno registrado: Python 3.10.21, TensorFlow 2.15.1 y NVIDIA GeForce RTX 4070 Ti. Las configuraciones guardadas coinciden con las celdas de sus respectivos notebooks.

**Excepción histórica por error: ResNet18 experiment01 se ejecutó con `max_epochs=100`; las otras tres, con 200. El máximo fijado para todos los entrenamientos es 200.** Las ejecuciones terminaron en las épocas 71, 87, 62 y 80, respectivamente, exactamente 20 épocas después de su mejor accuracy. Por tanto, ninguna alcanzó su límite y la diferencia de máximo no recortó estas ejecuciones. No se ha modificado la configuración histórica para ocultar esa diferencia.

Los historiales, métricas y probabilidades guardadas son coherentes: el mejor checkpoint corresponde al primer máximo de `val_accuracy`, y accuracy, F1 macro y matrices de confusión se han comprobado a partir de `validation_predictions.npz`. Los cuatro notebooks no contienen salidas de error y tienen `prediction.json` con **2.365 predicciones de test** cada uno. No se dispone aquí de etiquetas de test ni de resultados de Codabench.

## Registro

| Experimento | Arquitectura | Objetivo / cambio | Estado | Resultados |
|---|---|---|---|---|
| [experiment01](resnet18/experiment01/resnet18_experiment01.ipynb) | ResNet18 | Referencia inicial; máximo 100 épocas | Completado | [Métricas](resnet18/experiment01/artifacts/metrics.json): accuracy 78,03 %, F1 macro 79,99 %, mejor época 51 |
| [experiment01](densenet121/experiment01/densenet121_experiment01.ipynb) | DenseNet121 | Referencia inicial; máximo 200 épocas | Completado | [Métricas](densenet121/experiment01/artifacts/metrics.json): accuracy 79,31 %, F1 macro 79,62 %, mejor época 67 |
| [experiment01](efficientnet_b0/experiment01/efficientnet_b0_experiment01.ipynb) | EfficientNetB0 | Referencia inicial; máximo 200 épocas | Completado | [Métricas](efficientnet_b0/experiment01/artifacts/metrics.json): accuracy 77,97 %, F1 macro 78,40 %, mejor época 42 |
| [experiment01](mobilenet_v2/experiment01/mobilenet_v2_experiment01.ipynb) | MobileNetV2 | Referencia inicial; máximo 200 épocas | Completado | [Métricas](mobilenet_v2/experiment01/artifacts/metrics.json): accuracy 77,60 %, F1 macro 78,82 %, mejor época 60 |
| [experiment02](resnet18/experiment02/resnet18_experiment02.ipynb) | ResNet18 | Weight decay 1e-4 → 1e-3 para reducir sobreajuste | Completado | [Métricas](resnet18/experiment02/artifacts/metrics.json): accuracy 78,03 %, F1 macro 79,48 %, mejor época 63 |
| [experiment02](efficientnet_b0/experiment02/efficientnet_b0_experiment02.ipynb) | EfficientNetB0 | Learning rate 1e-3 → 3e-4 para mejorar optimización | Completado | [Métricas](efficientnet_b0/experiment02/artifacts/metrics.json): accuracy 76,75 %, F1 macro 77,90 %, mejor época 79 |
| [experiment03](resnet18/experiment03/resnet18_experiment03.ipynb) | ResNet18 | Referencia experiment01 + label smoothing 0.1; máximo 200 épocas | Completado | [Métricas](resnet18/experiment03/artifacts/metrics.json): accuracy 79,73 %, F1 macro 81,68 %, mejor época 93 |
| [experiment03](efficientnet_b0/experiment03/efficientnet_b0_experiment03.ipynb) | EfficientNetB0 | Referencia experiment01 + label smoothing 0.1; máximo 200 épocas | Completado | [Métricas](efficientnet_b0/experiment03/artifacts/metrics.json): accuracy 79,09 %, F1 macro 79,94 %, mejor época 85 |
| [experiment04](resnet18/experiment04/resnet18_experiment04.ipynb) | ResNet18 | Resolución 128×128 → 192×192 sobre experiment03; última prueba de esta parte | Completado | [Métricas](resnet18/experiment04/artifacts/metrics.json): accuracy 80,11 %, F1 macro 82,24 %, mejor época 115 |

Al terminar los próximos entrenamientos, actualizar esta tabla con el estado y enlaces a sus métricas, e indicar la mejor época, accuracy y F1 macro de validación. Registrar los resultados de Codabench por separado.

Para las siguientes pruebas, crear `experiment02`, `experiment03`, etc. dentro de la arquitectura correspondiente, copiando el notebook y actualizando `NOTEBOOK_PATH`, `experiment_id` e hipótesis. Partir de esta referencia y documentar aquí cada cambio deliberado. Para comparar una nueva ronda entre arquitecturas con igualdad de condiciones, aplicar el mismo cambio de protocolo a las cuatro. No sobrescribir un experimento cuyos resultados se quieran conservar.

## Resultados de experiment01

**DenseNet121 gana según el criterio principal, accuracy de validación: 79,31 %. ResNet18 obtiene el mejor F1 macro, 79,99 %, y el menor tiempo de entrenamiento registrado.** Las métricas siguientes corresponden al mejor checkpoint por accuracy, no a la última época ni al checkpoint de menor CE.

| Arquitectura | Accuracy (%) | F1 macro (%) | Recall macro (%) | CE validación ↓ | Mejor época / ejecutadas | Parámetros (millones) | Entrenamiento (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 78,03 | 79,99 | 79,30 | 0,973 | 51 / 71 | 11,193 | 14,06 |
| DenseNet121 | 79,31 | 79,62 | 81,05 | 1,463 | 67 / 87 | 7,051 | 96,23 |
| EfficientNetB0 | 77,97 | 78,40 | 79,21 | 0,769 | 42 / 62 | 4,066 | 48,41 |
| MobileNetV2 | 77,60 | 78,82 | 79,32 | 0,783 | 60 / 80 | 2,275 | 29,85 |

Fuentes: los enlaces de métricas del registro y la [tabla de comparación](../comparisons/popular_architectures_e01/comparison.csv). Figuras disponibles: [curvas](../comparisons/popular_architectures_e01/comparison_curves.pdf), [métricas](../comparisons/popular_architectures_e01/comparison_metrics.pdf), [recall por clase](../comparisons/popular_architectures_e01/comparison_class_recall.pdf) y [coste](../comparisons/popular_architectures_e01/comparison_cost.pdf).

### Qué explica las diferencias observadas

**DenseNet121 acierta más imágenes:** 1.487/1.875 frente a 1.463 de ResNet18, una ventaja de **24 imágenes y 1,28 puntos porcentuales**. Frente a ResNet18, gana 13 aciertos en Shipping container, 10 en Bus y 8 en Storage tank, entre otras diferencias, pero pierde 15 en Truck. Su recall macro también es el mayor (81,05 %). Esa distribución explica por qué lidera accuracy sin ganar necesariamente en todas las clases ni en F1 macro. Los resultados muestran dónde mejora; una sola ejecución no permite atribuir causalmente la mejora a un mecanismo concreto de la arquitectura.

**ResNet18 equilibra mejor precisión y recall al promediar las 13 clases:** F1 macro 79,99 % frente a 79,62 % de DenseNet121, y precisión macro 81,10 % frente a 78,95 %. Destaca en Truck, Helipad y Pylon respecto a DenseNet121. Sin embargo, Helipad solo tiene 9 imágenes y Pylon 32: su peso en el promedio macro es el mismo que el de clases mucho mayores. Como análisis de sensibilidad, al excluir ambas clases del promedio, DenseNet121 alcanza F1 78,71 % frente a 77,17 % de ResNet18. Este cálculo no sustituye la métrica de 13 clases; muestra que la pequeña ventaja macro de ResNet18 depende de cómo se ponderan las clases escasas.

**DenseNet121 presenta más sobreajuste y errores de alta confianza.** Su accuracy de train en inferencia es 99,90 %, frente al 79,31 % de validación: brecha de 20,59 puntos. ResNet18 tiene 97,10 % / 78,03 % y brecha de 19,07 puntos. En DenseNet121, la CE de validación mínima aparece en la época 30 (0,794), mientras que el checkpoint elegido por accuracy es el de la época 67 (CE 1,463). Entre sus 388 errores de validación, 115 asignan más del 99 % de probabilidad a la clase equivocada; en ResNet18 son 54 de 412 errores. Esto ayuda a explicar que DenseNet121 tenga mayor accuracy y, al mismo tiempo, peor CE: la entropía cruzada penaliza mucho esos errores confiados. No equivale a una evaluación completa de calibración.

**EfficientNetB0 consigue la menor CE del checkpoint seleccionado (0,769)** y la menor brecha train–validación (10,98 puntos). Su accuracy, 77,97 %, queda a una sola imagen de ResNet18, aunque su F1 macro es menor (78,40 %). Los errores con probabilidad superior al 99 % son 9 de 413. Estas medidas favorecen sus probabilidades bajo este protocolo, pero la menor brecha por sí sola no demuestra que sea superior ni descarta capacidad de ajuste pendiente.

**MobileNetV2 es la opción más pequeña:** 2,275 millones de parámetros frente a 11,193 millones de ResNet18, aproximadamente un 80 % menos. Su accuracy es 77,60 %, ocho aciertos menos que ResNet18, y su F1 macro es 78,82 %. En estas ejecuciones no fue la más rápida: tardó 29,85 minutos frente a 14,06 de ResNet18. DenseNet121 tardó 96,23 minutos, unas 6,84 veces lo de ResNet18. Los promedios por época también difieren: 11,89 s en ResNet18, 66,37 s en DenseNet121, 46,85 s en EfficientNetB0 y 22,38 s en MobileNetV2. Menos parámetros no implica menor tiempo; estos tiempos incluyen el pipeline y guardado y proceden de una ejecución por modelo, por lo que no constituyen un benchmark controlado de velocidad.

### Rendimiento por clase

F1 por clase (%) calculado con las predicciones del mejor checkpoint. El soporte es común a las cuatro redes.

| Clase | Soporte validación | ResNet18 | DenseNet121 | EfficientNetB0 | MobileNetV2 |
|---|---:|---:|---:|---:|---:|
| Cargo plane | 59 | 94,02 | 94,31 | 95,65 | 94,02 |
| Small car | 329 | 85,02 | 86,33 | 85,01 | 85,21 |
| Bus | 174 | 64,72 | 68,17 | 69,52 | 63,98 |
| Truck | 241 | 55,48 | 51,42 | 55,50 | 52,85 |
| Motorboat | 109 | 75,73 | 77,32 | 72,38 | 74,63 |
| Fishing vessel | 73 | 76,25 | 76,65 | 72,00 | 77,78 |
| Dump truck | 118 | 63,90 | 65,85 | 68,72 | 68,35 |
| Excavator | 81 | 85,53 | 84,66 | 85,37 | 84,88 |
| Building | 363 | 89,42 | 90,44 | 88,56 | 87,62 |
| Helipad | 9 | 94,12 | 80,00 | 81,82 | 84,21 |
| Storage tank | 152 | 86,71 | 89,80 | 83,99 | 86,51 |
| Shipping container | 135 | 72,12 | 80,88 | 74,02 | 73,68 |
| Pylon | 32 | 96,88 | 89,23 | 86,67 | 90,91 |

Truck es la clase con menor F1 en las cuatro redes (51,42–55,50 %). Las matrices de confusión muestran errores recurrentes entre Truck, Small car, Bus y Dump truck. Es un problema compartido del reconocimiento de esas categorías en esta partición; no queda resuelto por cambiar únicamente la arquitectura. En Helipad, EfficientNetB0 recupera 9/9 positivos, pero su precisión es 69,23 % por los falsos positivos; ResNet18 recupera 8/9 con precisión 100 %. Esto ilustra por qué recall y F1 pueden ordenar los modelos de forma distinta.

### Conclusión y siguientes comparaciones

Para el objetivo fijado de **maximizar accuracy**, la referencia ganadora de esta ronda es **DenseNet121**. Para combinar **F1 macro y tiempo de entrenamiento observado**, ResNet18 ofrece el mejor compromiso en estas ejecuciones. MobileNetV2 conserva interés si se prioriza el número de parámetros; EfficientNetB0, si se valora la CE de validación.

Las cuatro accuracies están en un intervalo de solo 1,71 puntos porcentuales. Hay una única semilla de entrenamiento y una única partición, y se seleccionan checkpoints sobre esa misma validación: no se ha demostrado una superioridad estadística ni una ventaja en test. Antes de generalizar la clasificación conviene repetir el protocolo con varias semillas manteniendo el reparto fijo. Cualquier cambio de regularización, resolución o criterio de selección debe registrarse como otro experimento; los resultados actuales se conservan tal como se obtuvieron.

## Segunda ronda: EfficientNetB0 y ResNet18

Se seleccionaron **EfficientNetB0 por su menor brecha train–validación en experiment01 (10,98 puntos)** y **ResNet18 por su velocidad de entrenamiento observada (14,06 minutos)**. Las accuracies de las cuatro arquitecturas en la primera ronda eran similares (77,60–79,31 %), por lo que se priorizaron estas dos líneas de mejora. DenseNet121 conserva la mayor accuracy de aquella ronda, pero sus 96,23 minutos de entrenamiento no compensan para esta fase de experimentación. DenseNet121 y MobileNetV2 mantienen sus referencias; no se han preparado nuevas pruebas para ellas en esta ronda.

| Modelo | Modificación prevista en experiment02 | Motivo |
|---|---|---|
| ResNet18 | Weight decay `1e-3` (antes `1e-4`); LR se mantiene en `1e-3` | Intentar reducir su sobreajuste |
| EfficientNetB0 | Learning rate inicial `3e-4` (antes `1e-3`); weight decay se mantiene en `1e-4` | Intentar mejorar la optimización |

### Protocolo realmente ejecutado

Los artefactos confirman 128×128, batch 32, AdamW, semilla 42, mismo reparto, clases, aumentación, callbacks y ausencia de Dropout adicional y label smoothing. Dentro de cada arquitectura coinciden los hashes de pesos iniciales y el entorno registrado. Ambas pruebas se entrenaron desde cero y generaron 2.365 predicciones de test cada una. Se han comprobado accuracy, F1 macro, matriz de confusión y selección de la mejor época contra las probabilidades e historiales guardados; los resultados son coherentes.

**ResNet18 experiment02 también cambió `max_epochs` de 100 a 200**, además del weight decay. Terminó en la época 83, por debajo de ambos límites: la ampliación no fue necesaria para completar esta ejecución. EfficientNetB0 mantuvo 200 épocas y el único hiperparámetro cambiado fue el LR inicial. Los dos entrenamientos terminaron 20 épocas después de su mejor accuracy, conforme a EarlyStopping. Esta ronda compara ajustes distintos por modelo; no es otra comparación de arquitecturas bajo hiperparámetros idénticos.

### Resultados frente a experiment01

Métricas del mejor checkpoint por `val_accuracy`. Accuracy y F1 se expresan en %, la brecha en puntos porcentuales; train se evalúa en modo inferencia, sin aumentación.

| Modelo | Experimento | Accuracy train | Accuracy validación | F1 macro | CE validación ↓ | Brecha train–validación ↓ |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 | experiment01 | 97,10 | 78,03 | 79,99 | 0,973 | 19,07 |
| ResNet18 | experiment02 | 98,90 | 78,03 | 79,48 | 1,212 | 20,87 |
| EfficientNetB0 | experiment01 | 88,95 | 77,97 | 78,40 | 0,769 | 10,98 |
| EfficientNetB0 | experiment02 | 98,42 | 76,75 | 77,90 | 1,044 | 21,67 |

| Modelo | Experimento | Mejor época / ejecutadas | Entrenamiento (min) | Tiempo medio por época (s) |
|---|---|---:|---:|---:|
| ResNet18 | experiment01 | 51 / 71 | 14,06 | 11,89 |
| ResNet18 | experiment02 | 63 / 83 | 31,11 | 22,49 |
| EfficientNetB0 | experiment01 | 42 / 62 | 48,41 | 46,85 |
| EfficientNetB0 | experiment02 | 79 / 99 | 74,05 | 44,88 |

Fuentes: métricas individuales enlazadas en el registro, [comparación ResNet18](../comparisons/resnet18_experiment02/comparison.csv) y [comparación EfficientNetB0](../comparisons/efficientnet_b0_experiment02/comparison.csv). Curvas guardadas: [ResNet18](../comparisons/resnet18_experiment02/comparison_curves.pdf) y [EfficientNetB0](../comparisons/efficientnet_b0_experiment02/comparison_curves.pdf).

### ResNet18: aumentar weight decay no redujo el sobreajuste observado

La accuracy de validación se mantiene exactamente en **78,03 %: 1.463 aciertos de 1.875**, pero el F1 macro baja de **79,99 % a 79,48 %** (−0,51 puntos). La CE empeora de **0,973 a 1,212** y la brecha aumenta de **19,07 a 20,87 puntos** (+1,80). Mientras la validación no mejora, la accuracy de train sube de 97,10 % a 98,90 %. **La hipótesis de reducir sobreajuste no queda respaldada por el checkpoint seleccionado de esta ejecución.**

Igual accuracy no significa iguales predicciones: experiment02 corrige 129 errores de experiment01, pero pierde otros 129 aciertos. Mejora el F1 de Bus (+2,50 puntos), Dump truck (+1,92) y Shipping container (+1,84); empeora Small car (−1,30), Fishing vessel (−1,89) y especialmente Pylon (−9,00). Pylon solo tiene 32 muestras: se pierden dos verdaderos positivos y también cambia la precisión. Helipad conserva F1 94,12 % con solo nueve positivos. Las variaciones por clase explican la caída macro aunque el número total de aciertos se mantenga.

El mejor checkpoint llega más tarde, época 63 frente a 51. En experiment02, la CE mínima se alcanza en la época 25 (0,737), pero se guarda la época 63 porque el criterio es accuracy; su CE es 1,212. Entre los errores de validación, los que asignan más del 99 % de probabilidad a una clase equivocada aumentan de 54 a 88. Por tanto, mejorar alguna parte temprana de la curva de CE no se traduce en mejores probabilidades en el checkpoint seleccionado.

El tiempo sube de 14,06 a 31,11 minutos, pero no solo por las 12 épocas adicionales: también aumenta el tiempo medio por época de 11,89 a 22,49 segundos. El entorno declarado coincide, pero estos registros no permiten atribuir esa diferencia de velocidad al weight decay; los tiempos son observaciones de ejecución, no una medición controlada del coste del hiperparámetro.

### EfficientNetB0: reducir el learning rate tampoco mejoró el resultado

La accuracy baja de **77,97 % a 76,75 %** (−1,23 puntos): **1.462 → 1.439 aciertos**, 23 menos. El F1 macro pasa de **78,40 % a 77,90 %** (−0,50 puntos), el recall macro de **79,21 % a 76,42 %** y la CE de **0,769 a 1,044**. Aunque la precisión macro aumenta (78,42 % → 80,33 %), no compensa la caída del recall.

La accuracy de train sube de **88,95 % a 98,42 %** y la brecha casi se duplica: **10,98 → 21,67 puntos**, un aumento de 10,69. Con el LR menor se observa un ajuste mucho mayor del train al llegar al checkpoint elegido, pero ese ajuste no se trasladó a validación. No hay evidencia aquí de que el problema sea falta de ajuste al entrenamiento. El mejor checkpoint se retrasa de la época 42 a la 79, el total pasa de 62 a 99 épocas y el tiempo aumenta de **48,41 a 74,05 minutos**. **La hipótesis de mejorar la optimización, entendida como mejorar el resultado de validación con este protocolo, no queda respaldada.**

La pérdida de aciertos se concentra especialmente en Bus: 21 menos y F1 **69,52 % → 60,72 %**. También empeoran Truck (−3,25 puntos de F1), Excavator (−5,37) y Building (−1,78). Hay ganancias en Fishing vessel (+8,00), Dump truck (+1,87), Small car (+1,16) y Pylon (+6,88), pero no compensan el deterioro global. Helipad pasa de 9/9 a 6/9 positivos recuperados; su escaso soporte obliga a interpretar con cautela esta variación.

La CE mínima de experiment02 es 0,818 en la época 30, peor que la mínima de experiment01 (0,708, época 39). Los errores con confianza superior al 99 % pasan de 9 a 57. El deterioro de CE no se limita, por tanto, a haber elegido una época tardía por accuracy: tampoco mejora el mínimo de CE alcanzado en esta prueba.

### Decisión tras la segunda ronda

**Se conservan los experiment01 como referencias preferidas en las dos arquitecturas.** Para ResNet18, el weight decay `1e-4` mantiene la misma accuracy y ofrece mejor F1 macro, menor CE y menor brecha que `1e-3`. Para EfficientNetB0, el LR inicial `1e-3` supera a `3e-4` en accuracy, F1 macro, recall macro, CE y brecha, y requiere menos épocas.

Los dos ajustes ensayados no aportaron la mejora buscada; esto no demuestra que aumentar regularización o reducir LR sea siempre perjudicial. Son observaciones de una ejecución por configuración, seleccionadas sobre una única validación, sin resultados etiquetados de test. Cualquier siguiente prueba debe partir explícitamente de la referencia elegida y registrar su cambio; los resultados de experiment03 se analizan en la siguiente sección.

Las comparaciones están guardadas por separado en `cnn_challenge_alt/comparisons/resnet18_experiment02/` y `cnn_challenge_alt/comparisons/efficientnet_b0_experiment02/`. Tras reiniciar el kernel, la primera celda de código y la sección 9 de los notebooks de experiment02 permiten regenerarlas desde artefactos, sin entrenar.

## Tercera ronda: label smoothing 0.1 en ambas redes

Se recuperó la configuración de **experiment01** para **ResNet18 y EfficientNetB0**: learning rate inicial **`1e-3`** y weight decay **`1e-4`**. No se acumularon los ajustes de experiment02, que no aportaron mejora. El factor probado fue **label smoothing `0.0 → 0.1`**, con máximo común de **200 épocas**.

### Protocolo y comprobación de artefactos

Se mantienen 128×128, batch 32, AdamW, semilla 42, reparto, aumentación, callbacks y ausencia de Dropout adicional y L2. Coinciden los hashes de datos, reparto y pesos iniciales respecto a la referencia de cada arquitectura, así como el entorno registrado. Se entrenó desde cero en los dos casos y se generaron **2.365 predicciones de test** por ejecución.

Se han comprobado accuracy, F1 macro, AP macro, matriz de confusión, orden de validación y selección del mejor checkpoint contra `validation_predictions.npz`, `split.json` e historiales. Las configuraciones guardadas coinciden con las de los notebooks y sus salidas no contienen errores. No se ha reentrenado ni modificado ningún notebook para hacer este análisis.

La ejecución histórica de ResNet18 experiment01 tuvo máximo 100 por error y se conserva porque terminó en la época 71. Experiment03 usa 200 y termina en la 113; su mejor checkpoint es el de la **época 93**, antes del límite antiguo. Se mantiene la norma de 200 épocas para todos los entrenamientos futuros.

La pérdida `loss` usa etiquetas suavizadas en experiment03 y no representa el mismo objetivo que en experiment01. Para comparar probabilidades se usa **`ce`, sin suavizado**, cuya configuración se conserva en las métricas. [Keras documenta el parámetro de suavizado de la pérdida](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) y [el de la métrica de CE](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalCrossentropy).

### Resultados frente a experiment01

Métricas del mejor checkpoint por `val_accuracy`, no de la última época. Accuracy y F1 se expresan en %, y las diferencias en puntos porcentuales.

| Modelo | Experimento | Accuracy validación | F1 macro | CE sin suavizado ↓ | AP macro (%) |
|---|---|---:|---:|---:|---:|
| ResNet18 | experiment01 | 78,03 | 79,99 | 0,973 | 85,47 |
| ResNet18 | experiment03 | 79,73 | 81,68 | 0,777 | 84,27 |
| EfficientNetB0 | experiment01 | 77,97 | 78,40 | 0,769 | 84,59 |
| EfficientNetB0 | experiment03 | 79,09 | 79,94 | 0,796 | 81,25 |

| Modelo | Experimento | Accuracy train (%) | Brecha train–validación (puntos) | Mejor época / ejecutadas | Entrenamiento (min) |
|---|---|---:|---:|---:|---:|
| ResNet18 | experiment01 | 97,10 | 19,07 | 51 / 71 | 14,06 |
| ResNet18 | experiment03 | 99,96 | 20,23 | 93 / 113 | 23,44 |
| EfficientNetB0 | experiment01 | 88,95 | 10,98 | 42 / 62 | 48,41 |
| EfficientNetB0 | experiment03 | 98,30 | 19,21 | 85 / 105 | 80,79 |

Fuentes: métricas individuales del registro, [comparación ResNet18](../comparisons/resnet18_experiment03/comparison.csv) y [comparación EfficientNetB0](../comparisons/efficientnet_b0_experiment03/comparison.csv). Curvas: [ResNet18](../comparisons/resnet18_experiment03/comparison_curves.pdf) y [EfficientNetB0](../comparisons/efficientnet_b0_experiment03/comparison_curves.pdf).

### ResNet18: mejora clara en las métricas de clasificación de esta ejecución

La accuracy sube de **78,03 % a 79,73 %** (+1,71 puntos), con **1.463 → 1.495 aciertos**, 32 más. El F1 macro sube de **79,99 % a 81,68 %** (+1,69 puntos) y la CE baja de **0,973 a 0,777**. Es la primera modificación que mejora conjuntamente estas tres medidas frente a experiment01. Experiment03 corrige 138 errores de la referencia, aunque pierde otros 106 aciertos: la mejora neta no implica que gane en cada imagen o clase.

El F1 mejora en clases difíciles: Dump truck **63,90 → 69,96 %**, Truck **55,48 → 59,52 %**, Fishing vessel **76,25 → 80,27 %** y Shipping container **72,12 → 75,62 %**. Truck aporta 12 aciertos adicionales, Shipping container 10, Dump truck 8 y Excavator 6, entre las ganancias y pérdidas del resto de categorías. Pylon empeora de 96,88 % a 93,55 % de F1 con solo 32 muestras, y Helipad conserva 94,12 % con nueve positivos.

Los errores con probabilidad asignada a la clase equivocada superior al 99 % pasan de **54 a 0**; los superiores al 90 %, de **150 a 35**. La confianza media en la clase predicha entre los errores baja de 76,72 % a 64,23 %. Estos datos respaldan la reducción de errores extremadamente confiados, pero no sustituyen un estudio de calibración.

**No se reduce la brecha de accuracy:** pasa de 19,07 a **20,23 puntos**, porque train sube de 97,10 % a 99,96 %. Tampoco mejora todo: AP macro baja de **85,47 % a 84,27 %**. Mejoran el argmax y la CE del checkpoint seleccionado, pero no el ranking resumido por AP. Además, la CE mínima del historial (0,749, época 42) es inferior a la CE de la época 93 seleccionada por accuracy (0,777): los dos criterios siguen eligiendo épocas diferentes.

El entrenamiento pasa de 71 a 113 épocas y de 14,06 a **23,44 minutos**. El tiempo medio por época es parecido (11,89 → 12,45 segundos); el aumento total se explica principalmente por la mayor duración del entrenamiento, sin convertir esta comparación en un benchmark controlado.

### EfficientNetB0: mejor accuracy y F1, con contrapartidas

La accuracy sube de **77,97 % a 79,09 %** (+1,12 puntos): **1.462 → 1.483 aciertos**, 21 más. El F1 macro aumenta de **78,40 % a 79,94 %** (+1,54 puntos). Destacan Fishing vessel (**72,00 → 83,33 %** de F1), Motorboat (**72,38 → 78,34 %**) y Storage tank (**83,99 → 86,81 %**). Truck empeora de 55,50 % a 53,57 % y Helipad de 81,82 % a 77,78 %, por lo que tampoco es una mejora uniforme.

La **CE empeora ligeramente, de 0,769 a 0,796**, y AP macro baja de **84,59 % a 81,25 %**. Los errores con confianza superior al 99 % desaparecen (9 → 0), pero la confianza media de los errores apenas cambia (68,88 % → 68,32 %). Eliminar esos errores extremos no basta para garantizar una CE o calibración mejores en el conjunto completo.

La accuracy de train sube de 88,95 % a 98,30 %, y la brecha aumenta de **10,98 a 19,21 puntos** (+8,23). El mejor checkpoint llega en la época 85, frente a la 42; el entrenamiento pasa de 62 a 105 épocas y de 48,41 a **80,79 minutos**. Su CE mínima también queda algo por encima de la referencia: 0,727 frente a 0,708. El label smoothing mejora la clasificación observada, pero no cumple una interpretación de «reducir sobreajuste» basada en la brecha ni mejora todas las medidas de las probabilidades.

### Decisión tras la tercera ronda

**Al cerrar la tercera ronda, las modificaciones fueron útiles para accuracy y F1 macro y ambos experiment03 se tomaron como nuevas referencias para ese objetivo. ResNet18 experiment03 se eligió para la última prueba:** obtuvo la mayor accuracy (**79,73 %**) y F1 macro (**81,68 %**) de las configuraciones revisadas hasta esa ronda, con 23,44 minutos de entrenamiento frente a 80,79 de EfficientNetB0 experiment03.

La ventaja sobre DenseNet121 experiment01 es pequeña en accuracy: **1.495 frente a 1.487 aciertos**, ocho imágenes. Frente a EfficientNetB0 experiment03 son 12 aciertos más. Estas diferencias no demuestran una superioridad general con una sola semilla y una única partición. Si se priorizan la CE o AP en lugar de accuracy/F1, conviene conservar las referencias sin suavizado para valorar ese compromiso; EfficientNetB0 experiment01 aún tiene mejor CE y AP que su experiment03.

## Cuarta ronda: ResNet18 a 192×192 · última prueba completada

**Experiment04 cambia únicamente la resolución, de 128×128 a 192×192, respecto a experiment03.** Se mantienen label smoothing `0.1`, LR `1e-3`, weight decay `1e-4`, AdamW, batch 32, máximo 200 épocas, semilla 42, reparto, aumentación y callbacks. Las configuraciones guardadas confirman el mismo entorno y los mismos hashes de datos, reparto y pesos iniciales. Se entrenó desde cero, sin reutilizar pesos anteriores.

Se han comprobado accuracy, F1 macro, AP macro, matriz de confusión, orden de las imágenes de validación y selección del checkpoint a partir de los artefactos. El mejor checkpoint corresponde a la época **115**, y el entrenamiento terminó en la **135**, 20 épocas después, conforme a EarlyStopping y sin alcanzar el máximo. El notebook no contiene salidas de error y `prediction.json` tiene **2.365 predicciones de test**. La comparación CSV coincide con las métricas individuales.

### Resultado frente a 128×128

Todas las métricas corresponden al mejor checkpoint por accuracy. CE se calcula sin label smoothing; los porcentajes de train se evalúan en modo inferencia, sin aumentación.

| Métrica | Experiment03 · 128×128 | Experiment04 · 192×192 |
|---|---:|---:|
| Accuracy validación (%) | 79,73 | 80,11 |
| F1 macro (%) | 81,68 | 82,24 |
| Recall macro (%) | 80,76 | 81,22 |
| CE validación ↓ | 0,777 | 0,756 |
| AP macro (%) | 84,27 | 85,10 |
| Accuracy train (%) | 99,96 | 99,99 |
| Brecha train–validación (puntos) | 20,23 | 19,88 |
| Entrenamiento (min) | 23,44 | 42,59 |
| Tiempo medio por época (s) | 12,45 | 18,93 |
| Aciertos / imágenes de validación | 1.495 / 1.875 | 1.502 / 1.875 |
| Mejor época / ejecutadas | 93 / 113 | 115 / 135 |
| Parámetros | 11.192.781 | 11.192.781 |

Fuentes: [métricas experiment04](resnet18/experiment04/artifacts/metrics.json), [tabla comparativa](../comparisons/resnet18_experiment04/comparison.csv), [curvas](../comparisons/resnet18_experiment04/comparison_curves.pdf), [métricas agregadas](../comparisons/resnet18_experiment04/comparison_metrics.pdf), [recall por clase](../comparisons/resnet18_experiment04/comparison_class_recall.pdf) y [coste](../comparisons/resnet18_experiment04/comparison_cost.pdf).

### Qué ha aportado aumentar la resolución

La accuracy mejora **0,37 puntos porcentuales**, con **siete aciertos netos adicionales**, y el F1 macro mejora **0,56 puntos**. También mejoran CE (0,777 → 0,756) y AP macro (84,27 % → 85,10 %). Experiment04 corrige 122 errores de experiment03, pero pierde 115 aciertos anteriores: la ganancia neta es pequeña y no beneficia a todas las imágenes.

La mejora por clase es desigual. La tabla muestra F1 (%) y el cambio en verdaderos positivos, usando el mismo conjunto de validación:

| Clase | F1 a 128×128 | F1 a 192×192 | Cambio de aciertos |
|---|---:|---:|---:|
| Bus | 65,48 | 72,19 | +12 |
| Shipping container | 75,62 | 79,86 | +4 |
| Storage tank | 87,63 | 88,97 | +5 |
| Fishing vessel | 80,27 | 83,44 | +4 |
| Truck | 59,52 | 55,46 | -4 |
| Dump truck | 69,96 | 63,76 | -12 |
| Motorboat | 78,30 | 78,00 | -5 |

La hipótesis de mejorar vehículos queda **respaldada parcialmente**: Bus mejora claramente, pero Truck y Dump truck empeoran. No se puede concluir que la mayor resolución resuelva de forma general la confusión entre vehículos. Helipad mantiene F1 94,12 % con solo nueve positivos; Pylon baja de 93,55 % a 91,80 % con 32 muestras.

La brecha disminuye ligeramente (**20,23 → 19,88 puntos**), pero train continúa prácticamente al 100 % y el sobreajuste sigue presente. No desaparece por aumentar la resolución. En ambas configuraciones no hay errores con confianza superior al 99 %; los superiores al 90 % pasan de 35 a 25. Estos datos y la menor CE no sustituyen una evaluación completa de calibración.

### Coste y elección final

La mejora de validación requiere **42,59 minutos frente a 23,44**: aproximadamente **un 82 % más de tiempo**, para siete aciertos netos adicionales. Se ejecutan 22 épocas más y cada época tarda de media 18,93 s frente a 12,45 s. El tiempo de preparación de datos registrado para 192×192 fue de unos 69 s adicionales al `fit`; la caché de esta resolución se mantiene separada de la de 128×128. Son tiempos de estas ejecuciones, no un benchmark controlado.

**Se conserva ResNet18 experiment04 como referencia final por accuracy y F1 macro de validación.** Es la mejor configuración observada en las dos métricas principales. **Experiment03 se conserva como alternativa de menor coste**, porque queda muy cerca y tarda sustancialmente menos. Los modelos, predicciones y artefactos de ambas configuraciones se mantienen en sus respectivas carpetas.

## Cierre del entrenamiento de redes populares

**Se da por finalizada esta fase, sin nuevos experimentos previstos.** Se han revisado cuatro arquitecturas y nueve entrenamientos. El mejor resultado de validación por arquitectura es:

| Arquitectura | Experimento elegido por accuracy | Accuracy (%) | F1 macro (%) | Entrenamiento (min) |
|---|---|---:|---:|---:|
| ResNet18 | [experiment04](resnet18/experiment04/resnet18_experiment04.ipynb) | 80,11 | 82,24 | 42,59 |
| DenseNet121 | [experiment01](densenet121/experiment01/densenet121_experiment01.ipynb) | 79,31 | 79,62 | 96,23 |
| EfficientNetB0 | [experiment03](efficientnet_b0/experiment03/efficientnet_b0_experiment03.ipynb) | 79,09 | 79,94 | 80,79 |
| MobileNetV2 | [experiment01](mobilenet_v2/experiment01/mobilenet_v2_experiment01.ipynb) | 77,60 | 78,82 | 29,85 |

Estas filas corresponden a las mejores variantes observadas de cada arquitectura después de ajustes distintos; no representan otra ronda con hiperparámetros idénticos. La comparación controlada inicial sigue documentada en experiment01.

En ResNet18, el label smoothing de experiment03 aportó una mejora mayor de accuracy que el salto posterior de resolución: +1,71 frente a +0,37 puntos respecto a sus respectivas referencias. El conjunto de ajustes lleva de **78,03 % / 79,99 %** de accuracy/F1 macro en experiment01 a **80,11 % / 82,24 %** en experiment04. El aumento de weight decay de ResNet18 y la reducción de LR de EfficientNetB0 de experiment02 no aportaron mejoras y no se incorporan a la configuración final.

La elección se basa en una única semilla y partición, con selección de checkpoints sobre la misma validación; la pequeña ventaja de la última prueba no demuestra superioridad estadística ni un mejor resultado en test. No se documentan resultados de Codabench porque no están disponibles aquí. Se conservan las predicciones para su evaluación externa y todos los artefactos para la memoria y comparaciones posteriores.

## Ejecutar y comparar

Las rutas de los cuatro notebooks ya apuntan a sus carpetas. Con las dependencias y los datos preparados según el [README del baseline](../baseline/README.md), se pueden ejecutar de arriba abajo desde cualquier clon del repositorio. No requieren la carpeta `cnn_challenge/` original.

La caché propia se comparte en `cnn_challenge_alt/.cache/` y está excluida de Git. Cada experimento guarda `model.keras` y genera obligatoriamente `prediction.json` tras entrenar; la configuración, entorno, reparto, historial, tiempos, métricas y probabilidades de validación quedan en `artifacts/`, y los PDF en `plots/`.

Para comparar los cuatro resultados sin entrenar de nuevo, ejecutar la primera celda de código y la sección 9 de cualquiera de estos notebooks, con:

```python
SELECT_RUNS = [
    'popular_architectures/resnet18/experiment01',
    'popular_architectures/densenet121/experiment01',
    'popular_architectures/efficientnet_b0/experiment01',
    'popular_architectures/mobilenet_v2/experiment01',
]
```

Las cuatro ejecuciones deben estar completas. Las comparaciones comprueban dataset, reparto y clases, y generan tablas y PDF en `cnn_challenge_alt/comparisons/`. Usar el destino `CHALLENGE_DIR / 'comparisons' / 'popular_architectures_e01'` en la llamada a `compare_runs` si se quiere conservar esa comparación separada de las posteriores.
