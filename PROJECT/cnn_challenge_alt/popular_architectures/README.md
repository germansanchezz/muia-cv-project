# Arquitecturas populares · protocolo y registro de experimentos

Comparamos ResNet18, DenseNet121, EfficientNetB0 y MobileNetV2 para las 13 clases de xView. **El diseño inicial establece igualdad de condiciones:** los `experiment01` comparten datos, reparto, semilla, resolución, aumentación, optimizador y criterio de selección. La comparación estudia el cambio de arquitectura; la ejecución de ResNet18 tiene una diferencia en el máximo de épocas, documentada abajo.

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

**Diferencia respecto al diseño: ResNet18 se ejecutó con `max_epochs=100`; las otras tres, con 200.** Las ejecuciones terminaron en las épocas 71, 87, 62 y 80, respectivamente, exactamente 20 épocas después de su mejor accuracy. Por tanto, ninguna alcanzó su límite y la diferencia de máximo no recortó estas ejecuciones. No se ha modificado la configuración histórica para ocultar esa diferencia.

Los historiales, métricas y probabilidades guardadas son coherentes: el mejor checkpoint corresponde al primer máximo de `val_accuracy`, y accuracy, F1 macro y matrices de confusión se han comprobado a partir de `validation_predictions.npz`. Los cuatro notebooks no contienen salidas de error y tienen `prediction.json` con **2.365 predicciones de test** cada uno. No se dispone aquí de etiquetas de test ni de resultados de Codabench.

## Registro

| Experimento | Arquitectura | Objetivo / cambio | Estado | Resultados |
|---|---|---|---|---|
| [experiment01](resnet18/experiment01/resnet18_experiment01.ipynb) | ResNet18 | Referencia inicial; máximo 100 épocas | Completado | [Métricas](resnet18/experiment01/artifacts/metrics.json): accuracy 78,03 %, F1 macro 79,99 %, mejor época 51 |
| [experiment01](densenet121/experiment01/densenet121_experiment01.ipynb) | DenseNet121 | Referencia inicial; máximo 200 épocas | Completado | [Métricas](densenet121/experiment01/artifacts/metrics.json): accuracy 79,31 %, F1 macro 79,62 %, mejor época 67 |
| [experiment01](efficientnet_b0/experiment01/efficientnet_b0_experiment01.ipynb) | EfficientNetB0 | Referencia inicial; máximo 200 épocas | Completado | [Métricas](efficientnet_b0/experiment01/artifacts/metrics.json): accuracy 77,97 %, F1 macro 78,40 %, mejor época 42 |
| [experiment01](mobilenet_v2/experiment01/mobilenet_v2_experiment01.ipynb) | MobileNetV2 | Referencia inicial; máximo 200 épocas | Completado | [Métricas](mobilenet_v2/experiment01/artifacts/metrics.json): accuracy 77,60 %, F1 macro 78,82 %, mejor época 60 |

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
