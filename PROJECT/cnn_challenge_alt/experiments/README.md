# Arquitecturas propias · registro de experimentos

Cada experimento se guarda en su propia carpeta `experiment01/`, `experiment02/`, etc., con el notebook, modelo, predicciones, artefactos y figuras. Las arquitecturas populares quedan como referencias; esta fase empieza con el mismo protocolo de su primera ronda.

## Protocolo inicial

| Parámetro | Valor |
|---|---|
| Entrada | RGB 128×128, float32 [0,1] |
| Batch size | 32 |
| Optimizador | AdamW |
| LR inicial / weight decay | 0.001 / 0.0001 |
| Pérdida | Categorical Cross Entropy, etiquetas one-hot |
| Label smoothing | 0.0 |
| Dropout adicional / L2 / clipping de gradientes | Ninguno |
| Máximo de épocas | 200 |
| ReduceLROnPlateau | `val_accuracy`, `max`, paciencia 10, factor 0.5, mínimo 1e-6 |
| EarlyStopping | `val_accuracy`, `max`, paciencia 20, restaurar mejores pesos |
| ModelCheckpoint | `val_accuracy`, `max`, guardar solo el mejor |
| Rotaciones | 0°, 90°, 180°, 270°, elección uniforme por imagen de train |
| Flips de train | Horizontal y vertical independientes, p=0.5 cada uno |
| Semilla | 42, operaciones deterministas activadas |
| Reparto | Histórico 90/10, `random_state=1`, sin estratificación |
| Inicialización | Aleatoria; todas las capas se entrenan desde cero |

En el protocolo inicial de experiment01, label smoothing es cero y la entrada es 128×128; experiment02 prueba explícitamente suavizado 0.1. Las capas BN y SE forman parte de la arquitectura propia, no de un cambio del protocolo de entrenamiento. Validación, evaluación de train y test no usan aumentación.

## Registro

| Experimento | Arquitectura | Objetivo | Estado | Resultados |
|---|---|---|---|---|
| [experiment01](experiment01/cnn_experiment01.ipynb) | CNN residual compacta con atención de canales SE, `residual_se_v1` | Competir con las populares bajo el protocolo inicial, conservando detalle espacial con menos canales | Completado | [Métricas](experiment01/artifacts/metrics.json): accuracy 79,15 %, F1 macro 80,18 %, mejor época 44 |
| [experiment02](experiment02/cnn_experiment02.ipynb) | La misma CNN `residual_se_v1` de experiment01 | Probar únicamente label smoothing 0.1 | Completado | [Métricas](experiment02/artifacts/metrics.json): accuracy 79,04 %, F1 macro 80,73 %, mejor época 66 |
| [experiment03](experiment03/cnn_experiment03.ipynb) | La misma CNN `residual_se_v1` de experiment01 | MixUp alpha 0.2, 128×128 y sin label smoothing | Completado | [Métricas](experiment03/artifacts/metrics.json): accuracy 80,69 %, F1 macro 82,27 %, mejor época 97 |
| [experiment04](experiment04/cnn_experiment04.ipynb) | La misma CNN `residual_se_v1` de experiment03 | MixUp alpha 0.2 → 0.4 como único cambio | Completado | [Métricas](experiment04/artifacts/metrics.json): accuracy 80,16 %, F1 macro 81,38 %, mejor época 77 |
| [experiment05](experiment05/cnn_experiment05.ipynb) | La misma CNN `residual_se_v1` de experiment03 | Entrada 128×128 → 192×192, manteniendo MixUp alpha 0.2 | Completado | [Métricas](experiment05/artifacts/metrics.json): accuracy 79,47 %, F1 macro 81,02 %, mejor época 95 |

## Experiment01: diseño

La red se construye directamente con capas estándar de Keras. Es una combinación propia inspirada en bloques residuales y atención SE, no una arquitectura presentada como inédita ni un backbone predefinido.

| Etapa | Diseño | Salida |
|---|---|---|
| Entrada | RGB | 128×128×3 |
| Stem | Dos Conv3×3 de 32 canales, cada una con BN + ReLU; la segunda con stride 2 | 64×64×32 |
| Etapa 1 | 2 bloques residuales SE, 32 canales | 64×64×32 |
| Etapa 2 | 2 bloques residuales SE, 64 canales; primero stride 2 | 32×32×64 |
| Etapa 3 | 2 bloques residuales SE, 128 canales; primero stride 2 | 16×16×128 |
| Etapa 4 | 2 bloques residuales SE, 256 canales; primero stride 2 | 8×8×256 |
| Clasificador | Global Average Pooling + Dense13 softmax | 13 probabilidades |

La red tiene **2.857.125 parámetros totales**, de los cuales **2.852.261 son entrenables**, frente a los 11.192.781 de ResNet18 de referencia (aproximadamente un 74 % menos de parámetros totales). El tamaño no permite asegurar por sí solo una mayor velocidad.

Cada bloque tiene dos Conv3×3, BN, activaciones ReLU, atención SE y suma con el atajo. Al cambiar resolución o canales, el atajo usa Conv1×1 + BN. SE resume cada canal con GAP y aprende sus pesos mediante dos capas densas, con reducción de canales por un factor 8 y salida sigmoid. No utiliza Dropout.

Las conexiones residuales se usan para facilitar la optimización; SE aprende a ponderar los canales de características. Referencias: [Deep Residual Learning](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) y [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html).

La adaptación a este problema está en las convoluciones de entrada pequeñas, la ausencia de MaxPool inicial, las anchuras 32/64/128/256 y la reducción total de 16×. ResNet18 de referencia usa un stem 7×7 con stride 2 seguido de MaxPool y llega a reducción 32×. Nuestra hipótesis es conservar más detalle espacial para objetos pequeños sin aumentar tanto los canales; no se garantiza una mejor accuracy ni una mayor velocidad antes de medirlas.

## Experiment01: resultados y conclusiones

**La custom es competitiva como primera referencia, pero aún no supera a ResNet18 experiment04.** Alcanza **79,15 % de accuracy y 80,18 % de F1 macro**, frente a **80,11 % y 82,24 %** de la popular, con aproximadamente **un 74 % menos de parámetros**.

Los artefactos confirman el mismo dataset, orden de clases y reparto (**16.871 imágenes de train y 1.875 de validación**), con hashes idénticos y el mismo entorno registrado. Accuracy, F1 macro, AP macro, matrices de confusión, orden de validación y mejor época se han comprobado contra las probabilidades, reparto e historiales. Los notebooks no contienen salidas de error; las configuraciones guardadas coinciden con sus celdas, y la custom genera **2.365 predicciones de test**.

**Esta es una comparación de resultados finales, no una prueba que aísle la arquitectura:** la custom usa **128×128 y label smoothing 0.0**, mientras que ResNet18 experiment04 usa **192×192 y label smoothing 0.1**. No se puede atribuir toda la diferencia al diseño de la red.

| Métrica | Custom experiment01 | ResNet18 experiment04 |
|---|---:|---:|
| Accuracy validación (%) | 79,15 | 80,11 |
| F1 macro (%) | 80,18 | 82,24 |
| Precisión macro (%) | 79,92 | 83,56 |
| Recall macro (%) | 80,95 | 81,22 |
| CE sin suavizado ↓ | 0,809 | 0,756 |
| AP macro (%) | 86,21 | 85,10 |
| Accuracy train en inferencia (%) | 96,67 | 99,99 |
| Brecha train–validación (puntos) | 17,53 | 19,88 |
| Entrenamiento (min) | 21,60 | 42,59 |
| Tiempo medio por época (s) | 20,25 | 18,93 |
| Aciertos / imágenes de validación | 1.484 / 1.875 | 1.502 / 1.875 |
| Mejor época / ejecutadas | 44 / 64 | 115 / 135 |
| Parámetros | 2.857.125 | 11.192.781 |

Fuentes: [métricas custom](experiment01/artifacts/metrics.json), [métricas ResNet18](../popular_architectures/resnet18/experiment04/artifacts/metrics.json) y [tabla de comparación](../comparisons/custom_experiment01_vs_popular_experiment01/comparison.csv). Figuras: [curvas](../comparisons/custom_experiment01_vs_popular_experiment01/comparison_curves.pdf), [métricas](../comparisons/custom_experiment01_vs_popular_experiment01/comparison_metrics.pdf), [recall por clase](../comparisons/custom_experiment01_vs_popular_experiment01/comparison_class_recall.pdf) y [coste](../comparisons/custom_experiment01_vs_popular_experiment01/comparison_cost.pdf).

### Qué se deduce

**La reducción de tamaño no ha impedido alcanzar un resultado cercano a la referencia:** la diferencia es de **18 aciertos, 0,96 puntos de accuracy y 2,06 puntos de F1 macro**. La custom incluso obtiene mayor AP macro (86,21 % frente a 85,10 %), aunque su CE es peor. AP mide el ranking de probabilidades por clase y no demuestra mejor calibración ni más aciertos del argmax. No se puede atribuir la competitividad específicamente a SE o al stem sin hacer pruebas que aíslen esos componentes.

**Hay sobreajuste, no una señal clara de falta de capacidad para ajustar train.** La accuracy de train en inferencia es 96,67 %, frente al 79,15 % de validación: brecha de **17,53 puntos**. La CE mínima de validación se alcanza en la época 30 (**0,700**), pero el checkpoint elegido por accuracy es el de la época 44 (**0,809**). En la última época, la accuracy de validación baja a 77,33 % y la CE sube a 0,972. EarlyStopping termina en la 64, 20 épocas después del mejor checkpoint. Estas curvas no justifican como primer paso aumentar la anchura o añadir más épocas; conviene probar antes un cambio de regularización y comprobar su efecto en validación.

Entre los **391 errores** de la custom, **31** asignan más del 99 % de probabilidad a la clase equivocada, y **116** superan el 90 %. En ResNet18 experiment04 son 0 y 25, respectivamente. Esto señala errores de alta confianza que merece la pena investigar, aunque las diferencias de resolución y label smoothing impiden adjudicarlas solo a la arquitectura.

**El comportamiento por clase es desigual.** La custom destaca en Dump truck, con F1 **69,73 % frente a 63,76 %** de ResNet18 y 91/118 positivos recuperados frente a 73/118. También obtiene F1 mayor en Truck (57,33 % frente a 55,46 %), aunque recupera tres positivos menos: la mejora del F1 proviene del equilibrio entre precisión y recall. Motorboat (72,28 % frente a 78,00 %), Excavator (84,15 % frente a 89,16 %) y Small car (84,90 % frente a 86,34 %) ofrecen margen de mejora.

Las confusiones más frecuentes de la custom son Truck → Dump truck (32), Truck → Small car (30), Small car → Truck (24) y Truck → Bus (23). El problema de vehículos similares sigue presente. Helipad tiene F1 80,00 % frente a 94,12 %, pero ambas redes recuperan 8/9 positivos: la custom añade tres falsos positivos y ResNet18 ninguno. Dado su escaso soporte y el peso igual de cada clase en la media macro, esa diferencia por sí sola aporta aproximadamente **1,09 puntos** a los 2,06 puntos de distancia global de F1. No debe interpretarse como una inferioridad sólida en toda la distribución de clases.

**Menos tiempo total no equivale aquí a épocas más rápidas.** La custom tarda 21,60 minutos frente a 42,59, pero ejecuta 64 épocas frente a 135. Su tiempo medio por época es ligeramente mayor (20,25 s frente a 18,93 s), pese a tener menos parámetros y menor entrada. Mantener mapas espaciales mayores y otras diferencias del grafo pueden influir en el coste; estos registros no aíslan la causa. No se debe prometer una aceleración proporcional al número de parámetros.

## Experiment02: efecto de label smoothing 0.1

**La modificación reduce los errores extremadamente confiados, pero no aporta una mejora general clara de clasificación.** La accuracy queda prácticamente igual (dos aciertos menos), el F1 macro mejora ligeramente y AP macro empeora. La mejor custom por el criterio principal de accuracy sigue siendo **experiment01**, aunque una diferencia de dos imágenes no demuestra superioridad estadística.

### Verificación y resultados

Se ha comprobado que el **único hiperparámetro cambiado respecto a experiment01 es `label_smoothing`, de 0.0 a 0.1**. Coinciden arquitectura, parámetros, pesos iniciales, semilla, resolución, datos, clases, reparto y entorno registrado. Se entrenó desde cero. El historial, accuracy, F1 macro, AP macro y matriz de confusión se han contrastado con las probabilidades guardadas; el orden de validación coincide con `split.json` y los hashes son correctos. La configuración coincide con la celda del notebook, no hay salidas de error y se generaron **2.365 predicciones de test**.

| Métrica | Custom experiment01 | Custom experiment02 | ResNet18 experiment04 |
|---|---:|---:|---:|
| Accuracy validación (%) | 79,15 | 79,04 | 80,11 |
| F1 macro (%) | 80,18 | 80,73 | 82,24 |
| Precisión macro (%) | 79,92 | 81,16 | 83,56 |
| Recall macro (%) | 80,95 | 80,45 | 81,22 |
| CE sin suavizado ↓ | 0,809 | 0,800 | 0,756 |
| AP macro (%) | 86,21 | 83,47 | 85,10 |
| Accuracy train en inferencia (%) | 96,67 | 99,21 | 99,99 |
| Brecha train–validación (puntos) | 17,53 | 20,17 | 19,88 |
| Entrenamiento (min) | 21,60 | 29,88 | 42,59 |
| Tiempo medio por época (s) | 20,25 | 20,85 | 18,93 |
| Aciertos / validación | 1.484 / 1.875 | 1.482 / 1.875 | 1.502 / 1.875 |
| Mejor época / ejecutadas | 44 / 64 | 66 / 86 | 115 / 135 |
| Resolución | 128×128 | 128×128 | 192×192 |
| Label smoothing | 0.0 | 0.1 | 0.1 |
| Parámetros | 2.857.125 | 2.857.125 | 11.192.781 |

CE se mide siempre sin suavizado. `loss` usa objetivos distintos en las dos custom y no se compara como una misma pérdida. Fuentes: [métricas experiment02](experiment02/artifacts/metrics.json), [comparación CSV](../comparisons/custom_experiment02/comparison.csv), [curvas](../comparisons/custom_experiment02/comparison_curves.pdf), [métricas](../comparisons/custom_experiment02/comparison_metrics.pdf), [recall por clase](../comparisons/custom_experiment02/comparison_class_recall.pdf) y [coste](../comparisons/custom_experiment02/comparison_cost.pdf).

### Qué ha mejorado y qué no

**Accuracy prácticamente empatada:** 79,15 % → **79,04 %**, una diferencia de −0,11 puntos. Experiment02 corrige 124 errores de experiment01, pero pierde otros 126 aciertos. No se ha conseguido cerrar la distancia a ResNet18 experiment04: queda a **20 aciertos y 1,07 puntos de accuracy**, con las diferencias de arquitectura y resolución indicadas en la tabla.

**El F1 macro sube 0,54 puntos, de 80,18 % a 80,73 %, pero hay que interpretar su composición.** Helipad pasa de F1 80,00 % a 88,89 % manteniendo los mismos **8/9 positivos recuperados**; los falsos positivos bajan de tres a uno. Esa mejora aporta aproximadamente **0,68 puntos al promedio macro**, más que la ganancia global. Como análisis de sensibilidad, al excluir Helipad, el F1 medio de las otras doce clases baja de **80,20 % a 80,05 %**. Esto no sustituye la métrica oficial de trece clases; muestra que la mejora global no es uniforme y depende de una clase con muy poco soporte.

Hay mejoras útiles en Motorboat (F1 **72,28 → 77,83 %**), Excavator (**84,15 → 89,44 %**) y Fishing vessel (**78,98 → 81,05 %**). Empeoran Dump truck (**69,73 → 66,96 %**), Bus (**69,34 → 67,06 %**), Truck (**57,33 → 55,88 %**), Shipping container (**79,00 → 77,37 %**) y Pylon (**93,55 → 86,15 %**). Dump truck pierde 16 verdaderos positivos, aunque mejora su precisión. Las confusiones entre vehículos continúan: Truck → Small car (35), Bus → Truck (30), Truck → Bus (24) y Dump truck → Truck (24).

**La reducción de confianza extrema sí está observada.** Entre las predicciones equivocadas, las que superan el 99 % de confianza pasan de **31 a 0** y las superiores al 90 %, de **116 a 56**; la confianza media en la clase errónea baja de **74,16 % a 66,01 %**. La CE del checkpoint seleccionado mejora ligeramente, **0,809 → 0,800**, pero AP macro baja de **86,21 % a 83,47 %**. Menor confianza extrema no garantiza mejor ranking, más aciertos ni una mejor calibración global.

**La brecha de accuracy no se reduce:** aumenta de **17,53 a 20,17 puntos** (+2,64), porque la accuracy de train sube de 96,67 % a 99,21 % sin una ganancia de validación. La CE mínima del historial es **0,748 en la época 42**, frente a 0,700 en la época 30 de experiment01. La mejor accuracy se selecciona más tarde, en la época 66; al terminar en la 86, la accuracy de validación es 78,67 % y la CE 0,824. No hay evidencia de que prolongar más el entrenamiento sea la solución.

El tiempo total sube de **21,60 a 29,88 minutos** (+8,28), con 22 épocas adicionales; el tiempo medio por época cambia poco (20,25 → 20,85 s). El ajuste no ha sido gratuito y sus beneficios dependen de qué métrica se priorice.

### Decisión tras experiment02

**Se conserva custom experiment01 como referencia por accuracy**, y experiment02 como alternativa de mejor F1 macro global y menor confianza extrema. No se descarta su utilidad, pero no se adopta label smoothing 0.1 como una mejora general de la custom. La diferencia de accuracy es pequeña y no permite una conclusión fuerte con una sola semilla y partición.

## Experiment03 · MixUp alpha 0.2 · completado

Se parte de **experiment01**, la mejor custom anterior por accuracy, y se añade **solo MixUp con `alpha=0.2`**. Se mantienen arquitectura `residual_se_v1` (2.857.125 parámetros), entrada **128×128**, label smoothing **0.0**, batch 32, AdamW, LR `1e-3`, weight decay `1e-4`, máximo 200 épocas, semilla 42, reparto, aumentación geométrica y callbacks. No se acumula el ajuste de experiment02. La subida de resolución se reserva para una posible prueba final; experiment03 no la modifica.

La hipótesis es mejorar validación y reducir errores de alta confianza regularizando las imágenes y sus etiquetas conjuntamente. [MixUp](https://arxiv.org/abs/1710.09412) interpola pares de ejemplos y sus objetivos. No se presupone que supere el label smoothing ni que reduzca necesariamente la brecha de accuracy.

### Implementación de la mezcla

1. Leer un lote de imágenes originales de la caché mmap y construir etiquetas one-hot.
2. Aplicar los mismos giros 0°/90°/180°/270° y flips H/V de experiment01 a cada imagen.
3. Elegir una permutación de parejas dentro del lote y una lambda por imagen, muestreada de **Beta(0.2, 0.2)**.
4. Usar esa misma pareja y lambda para imagen y etiqueta: `x_mix = lambda*x + (1-lambda)*x_pareja`, `y_mix = lambda*y + (1-lambda)*y_pareja`.

La mezcla se aplica en todos los lotes de train; algunas lambdas son próximas a 0 o 1 y la permutación puede emparejar una imagen consigo misma. Un lote final de una imagen queda sin cambio. El RNG de MixUp es independiente del barajado (`seed+1000`) y avanza entre épocas. El protocolo de mezcla se guarda en `config.json`, incluido su alpha, orden, emparejamiento y relación de semillas. No se cachean imágenes mezcladas ni se copia el dataset completo a RAM.

**Validación, evaluación final de train y test no usan MixUp ni aumentación.** Los callbacks siguen monitorizando `val_accuracy` de imágenes originales. La pérdida es la misma Categorical Cross Entropy, ahora con los objetivos blandos de MixUp en train; no se añade label smoothing. El modelo y el preprocesamiento de inferencia no cambian.

### Cómo interpretar y comparar las métricas

`accuracy`, `loss` y `ce` del train durante `fit` usan imágenes y etiquetas mezcladas. La accuracy utiliza el argmax de la etiqueta blanda: **no mide lo mismo que la accuracy de train de experiment01**. Las métricas `val_*` sí conservan el significado previo. La brecha final, `train_inference` y `valid_inference` se calculan sobre datos originales y siguen siendo comparables.

Las figuras individuales identifican **Train (MixUp)**. En la figura de comparación se omite esa curva de accuracy de train, manteniendo las de validación y las de train sin mezcla de las referencias. La tabla añade `mixup_alpha`, con 0 para las ejecuciones anteriores. Los resultados siguientes comparan accuracy/F1, CE/AP de validación, confianza de los errores, vehículos, brecha en inferencia y coste.

El [notebook](experiment03/cnn_experiment03.ipynb) inicializa la red desde cero y guarda el mejor `model.keras`, obligatoriamente `prediction.json`, artefactos y PDF en su carpeta. Su `SELECT_RUNS` contiene **experiment03 + custom experiment01 + ResNet18 experiment04**. La comparación principal aísla MixUp frente a experiment01; la popular mantiene su entrada 192×192 y label smoothing 0.1. Destino: `cnn_challenge_alt/comparisons/custom_experiment03/`.

### Resultados y comprobación

**MixUp aporta la primera mejora conjunta clara de accuracy, F1 macro y CE de la custom respecto a experiment01. Experiment03 pasa a ser la mejor custom por accuracy y la referencia para las siguientes pruebas.** Alcanza **80,69 % de accuracy y 82,27 % de F1 macro** a 128×128, con 2.857.125 parámetros.

Los artefactos confirman la misma arquitectura, parámetros, pesos iniciales, semilla, datos, reparto y entorno que experiment01; el factor añadido es MixUp, además de los metadatos que lo describen. Las métricas, AP, matrices de confusión, orden de validación, hashes y mejor época se han comprobado contra las probabilidades, reparto e historiales. Las configuraciones guardadas coinciden con los notebooks; no hay salidas de error y se generaron **2.365 predicciones de test**. La tabla CSV concuerda con las métricas individuales.

| Métrica | Custom exp01 | Custom exp02 · suavizado | Custom exp03 · MixUp | ResNet18 exp04 |
|---|---:|---:|---:|---:|
| Accuracy validación (%) | 79,15 | 79,04 | 80,69 | 80,11 |
| F1 macro (%) | 80,18 | 80,73 | 82,27 | 82,24 |
| CE sin suavizado ↓ | 0,809 | 0,800 | 0,695 | 0,756 |
| AP macro (%) | 86,21 | 83,47 | 86,38 | 85,10 |
| Accuracy train original en inferencia (%) | 96,67 | 99,21 | 98,30 | 99,99 |
| Brecha train–validación (puntos) | 17,53 | 20,17 | 17,61 | 19,88 |
| Entrenamiento (min) | 21,60 | 29,88 | 39,18 | 42,59 |
| Tiempo medio por época (s) | 20,25 | 20,85 | 20,09 | 18,93 |
| Aciertos / validación | 1.484 / 1.875 | 1.482 / 1.875 | 1.513 / 1.875 | 1.502 / 1.875 |
| Mejor época / ejecutadas | 44 / 64 | 66 / 86 | 97 / 117 | 115 / 135 |
| Parámetros | 2.857.125 | 2.857.125 | 2.857.125 | 11.192.781 |

Fuentes: [métricas experiment03](experiment03/artifacts/metrics.json), [tabla de comparación](../comparisons/custom_experiment03/comparison.csv), [curvas](../comparisons/custom_experiment03/comparison_curves.pdf), [métricas](../comparisons/custom_experiment03/comparison_metrics.pdf), [recall por clase](../comparisons/custom_experiment03/comparison_class_recall.pdf) y [coste](../comparisons/custom_experiment03/comparison_cost.pdf). Experiment02 se incluye como contexto a partir de sus métricas; la selección del notebook sigue siendo experiment03, experiment01 y ResNet18 experiment04.

### Brecha final: 17,61 puntos, calculada sin mezcla

**La brecha comparable es `98,30 % − 80,69 % = 17,61 puntos porcentuales`.** Se usa el mejor checkpoint, época 97, evaluando train y validación en modo inferencia, sin MixUp ni aumentación. No se resta la accuracy del historial de entrenamiento con etiquetas mezcladas: esa mide otro problema.

La brecha de experiment01 era **17,53 puntos**; con MixUp sube solo **0,08 puntos**. Frente a experiment02 mejora (20,17 → 17,61), pero **no se ha eliminado el sobreajuste ni reducido la brecha respecto a la referencia original**. La accuracy de train aumenta de 96,67 % a 98,30 %, mientras validación mejora de 79,15 % a 80,69 %. El resultado útil es que la red generaliza mejor en esta partición, aunque la distancia entre ambos conjuntos siga siendo amplia. Reducir la brecha por sí solo no es el objetivo: podría hacerse empeorando train sin mejorar validación.

### Mejoras frente a la custom anterior

Respecto a experiment01, la accuracy gana **1,55 puntos y 29 aciertos netos**; el F1 macro gana **2,09 puntos** y la CE baja de **0,809 a 0,695**. AP macro también mejora ligeramente (**86,21 → 86,38 %**). Experiment03 corrige 148 errores de experiment01, pero pierde 119 aciertos anteriores: las ganancias no son uniformes.

A diferencia de experiment02, la mejora macro no depende solo de Helipad. Excluyendo esa clase como análisis de sensibilidad, el F1 medio de las otras doce pasa de **80,20 % a 81,72 %**. Algunos cambios de F1:

| Clase | Experiment01 | Experiment03 |
|---|---:|---:|
| Motorboat | 72,28 % | 79,21 % |
| Dump truck | 69,73 % | 73,59 % |
| Excavator | 84,15 % | 91,14 % |
| Truck | 57,33 % | 58,39 % |
| Storage tank | 87,72 % | 89,42 % |
| Shipping container | 79,00 % | 78,10 % |
| Pylon | 93,55 % | 87,50 % |
| Helipad | 80,00 % | 88,89 % |

Small car aporta 13 verdaderos positivos adicionales; Motorboat, siete; Storage tank, seis; y Truck, cinco. Dump truck recupera seis positivos menos, pero reduce sus falsos positivos y mejora su precisión (63,64 → 75,22 %), elevando su F1. Shipping container y Pylon empeoran; Helipad sigue teniendo solo nueve positivos. Permanecen confusiones como Truck → Small car (37), Truck → Bus (28) y Bus → Truck (23).

Los errores con confianza superior al 99 % pasan de **31 a 12**, y los superiores al 90 %, de **116 a 87**. La confianza media en la clase equivocada baja de **74,16 % a 71,17 %**. MixUp reduce los extremos menos que label smoothing, que los eliminó en experiment02, pero consigue mejor accuracy, CE y AP. Por tanto, hacer las probabilidades menos extremas no basta como criterio de selección, ni estos recuentos sustituyen una evaluación completa de calibración.

La CE mínima de validación es **0,657 en la época 47**, mejor que la mínima de experiment01 (0,700). El checkpoint elegido por accuracy llega en la época 97, con CE 0,695; se entrena hasta la 117 y se detiene conforme a la paciencia 20. La regularización permite una trayectoria útil más larga en esta ejecución, pero continúa existiendo separación entre la época de mejor CE y la de mejor accuracy.

### Comparación con la referencia popular y coste

La custom con MixUp supera la accuracy de **ResNet18 experiment04 en 0,59 puntos: 1.513 frente a 1.502 aciertos**, once más. El F1 macro está **prácticamente empatado**: 82,27 % frente a 82,24 %, una diferencia de solo 0,03 puntos. La custom tiene mejor CE (0,695 frente a 0,756), mayor AP macro (86,38 % frente a 85,10 %) y aproximadamente un **74 % menos de parámetros**.

No es una comparación que aísle arquitectura: la custom utiliza 128×128 con MixUp y sin label smoothing, mientras ResNet18 usa 192×192 con label smoothing y sin MixUp. La ventaja de once imágenes no demuestra superioridad estadística ni un mejor resultado en test.

El entrenamiento dura **39,18 minutos**, frente a 21,60 de experiment01 y 42,59 de ResNet18. La subida frente a la custom inicial se debe principalmente a pasar de 64 a 117 épocas; los tiempos medios por época son similares (**20,25 → 20,09 s**). El coste por época no muestra una penalización apreciable de MixUp en estos registros, pero son ejecuciones individuales, no un benchmark controlado. Con más épocas, el ahorro de tiempo total frente a ResNet18 ya es pequeño, aunque se mantiene la ventaja en parámetros.

### Decisión tras experiment03

**Se adopta experiment03 como nueva referencia custom.** Para los siguientes notebooks, la selección será **custom actual + custom experiment03 + ResNet18 experiment04**, mientras experiment03 siga siendo la mejor custom anterior por accuracy. Los resultados de todas las pruebas se conservan.

En experiment04 se eligió **probar únicamente `mixup_alpha=0.4` sobre experiment03**, sin cambiar arquitectura, resolución, label smoothing, LR, weight decay ni callbacks. Alpha controla la distribución Beta de la mezcla; aumentar de 0.2 a 0.4 hace las proporciones menos extremas. La hipótesis es comprobar si una mezcla algo más intensa mejora validación y el compromiso entre CE y errores de alta confianza, pero podría regularizar demasiado. Se conservaría alpha 0.2 si 0.4 solo baja train y no mejora validación. El [trabajo original de MixUp](https://arxiv.org/abs/1710.09412) plantea la interpolación como regularización; el valor óptimo debe comprobarse en este dataset.

Antes de generalizar la pequeña ventaja frente a ResNet18, conviene confirmar su estabilidad con otras semillas de entrenamiento y el mismo reparto. La resolución **192×192 sigue reservada para una posible última prueba**, partiendo de la mejor custom disponible entonces y cambiando solo ese factor. No se propone acumular MixUp, label smoothing y resolución de una vez.

## Experiment04 · MixUp alpha 0.4 · completado

El [notebook de experiment04](experiment04/cnn_experiment04.ipynb) se ejecutó a partir de **experiment03** y el **único hiperparámetro cambiado fue `mixup_alpha`, de 0.2 a 0.4**. No se añade Dropout ni label smoothing y no se aumenta la resolución.

Se mantienen la arquitectura `residual_se_v1` con 2.857.125 parámetros, entrada 128×128, batch 32, AdamW, LR 0.001, weight decay 0.0001, label smoothing 0.0, máximo 200 épocas, semilla 42, reparto, giros/flips y callbacks. La red se entrena desde cero y reutiliza la misma caché de píxeles mediante mmap.

MixUp conserva emparejamiento dentro del lote, lambda por imagen y el mismo RNG independiente; solo cambia su distribución a **Beta(0.4, 0.4)**. Validación, evaluación final de train y test siguen sin mezcla ni aumentación. La brecha se calcula sobre datos originales y las gráficas mantienen identificadas las métricas de train con MixUp.

La referencia a superar es custom experiment03: **80,69 % de accuracy, 82,27 % de F1 macro, CE 0,695 y brecha 17,61 puntos**. El criterio principal es accuracy de validación, valorando también F1, CE/AP, confianza de los errores y tiempo. Una menor accuracy de train o brecha por sí sola no demuestra mejora; una mezcla más intensa también puede perjudicar la clasificación.

`SELECT_RUNS` contiene **custom experiment04 + custom experiment03 + ResNet18 experiment04**. Las comparaciones se guardarán en `cnn_challenge_alt/comparisons/custom_experiment04/`. Se mantienen la exportación obligatoria de `prediction.json`, el checkpoint y los artefactos en la carpeta del nuevo experimento.

### Resultados: elección entre alpha 0.2 y 0.4

**Se conserva alpha 0.2 de custom experiment03 para maximizar accuracy y F1 macro.** Alpha 0.4 mejora CE, AP y brecha, pero pierde aciertos y recall. La decisión depende del objetivo: no es que 0.4 sea peor en todas las medidas.

Se han comprobado métricas, AP, matriz de confusión, orden de validación, reparto, hashes y mejor época contra las probabilidades e historiales guardados. Ambos experimentos tienen los mismos pesos iniciales y entorno; solo cambia alpha y su descripción en los metadatos. Las configuraciones coinciden con los notebooks, no hay salidas de error y experiment04 generó **2.365 predicciones de test**. El CSV de comparación concuerda con los artefactos.

| Métrica | Experiment03 · alpha 0.2 | Experiment04 · alpha 0.4 |
|---|---:|---:|
| Accuracy validación (%) | 80,69 | 80,16 |
| F1 macro (%) | 82,27 | 81,38 |
| Precisión macro (%) | 82,88 | 83,64 |
| Recall macro (%) | 81,85 | 79,65 |
| CE sin suavizado ↓ | 0,695 | 0,646 |
| AP macro (%) | 86,38 | 86,51 |
| Accuracy train original en inferencia (%) | 98,30 | 94,54 |
| Brecha train–validación (puntos) | 17,61 | 14,38 |
| Entrenamiento (min) | 39,18 | 65,24 |
| Tiempo medio por época (s) | 20,09 | 40,36 |
| Aciertos / validación | 1.513 / 1.875 | 1.503 / 1.875 |
| Mejor época / ejecutadas | 97 / 117 | 77 / 97 |

Fuentes: [métricas experiment04](experiment04/artifacts/metrics.json), [tabla comparativa](../comparisons/custom_experiment04/comparison.csv), [curvas](../comparisons/custom_experiment04/comparison_curves.pdf), [métricas](../comparisons/custom_experiment04/comparison_metrics.pdf), [recall por clase](../comparisons/custom_experiment04/comparison_class_recall.pdf) y [coste](../comparisons/custom_experiment04/comparison_cost.pdf).

Con alpha 0.4, la accuracy baja **0,53 puntos** (diez aciertos menos) y el F1 macro baja **0,89 puntos**. Experiment04 corrige 102 errores de experiment03, pero pierde otros 112 aciertos. El recall macro disminuye de 81,85 % a 79,65 %, aunque la precisión macro sube de 82,88 % a 83,64 %. El mayor suavizado de la mezcla no se traduce en mejor clasificación al seleccionar por accuracy.

Por clase, Bus pierde diez verdaderos positivos, Dump truck siete y Excavator cinco; sus F1 pasan de **69,66 → 68,47 %**, **73,59 → 68,72 %** y **91,14 → 86,45 %**. Hay ganancias en Small car (ocho positivos más) y Truck (siete más), y mejoras de F1 en Motorboat y Fishing vessel. El balance global sigue favoreciendo alpha 0.2.

**La brecha sí mejora:** **17,61 → 14,38 puntos**, pero la accuracy de train original también baja de **98,30 % a 94,54 %**, mientras validación empeora. Esto es compatible con una regularización más intensa y no justifica elegir alpha 0.4 si el criterio principal son los aciertos. Sí resulta interesante para las probabilidades: CE **0,695 → 0,646**, AP **86,38 → 86,51 %**, errores con confianza superior al 99 % **12 → 2**, y superiores al 90 % **87 → 43**. Estos recuentos no constituyen por sí solos una evaluación completa de calibración.

La CE mínima de experiment04 es **0,640 en la época 82**, ligeramente posterior al checkpoint elegido por accuracy (época 77, CE 0,646). El entrenamiento termina en la 97, conforme a la paciencia 20. El tiempo registrado es mayor, **65,24 frente a 39,18 minutos**, pese a ejecutar menos épocas: el promedio por época casi se duplica (40,36 frente a 20,09 s). El entorno declarado coincide, pero los registros no permiten atribuir esa diferencia al cambio de alpha; no se usa como prueba de que una mezcla 0.4 sea intrínsecamente más lenta.

**Decisión:** experiment03 con alpha **0.2** continúa como mejor custom por accuracy/F1, aunque las diferencias de diez imágenes no demuestran superioridad general con una sola semilla y partición. Experiment04 se conserva como alternativa de mejor CE y menor confianza extrema; sus resultados no se descartan ni se sobrescriben.

## Experiment05 · 192×192 con MixUp alpha 0.2 · completado

Se ejecutó el [notebook de experiment05](experiment05/cnn_experiment05.ipynb) a partir de **custom experiment03**, no de experiment04. **Único hiperparámetro cambiado: resolución `128×128 → 192×192`.** Se mantiene la configuración elegida: **MixUp alpha 0.2**, label smoothing **0.0**, sin Dropout adicional ni L2.

La arquitectura sigue siendo `residual_se_v1`, canales 32/64/128/256, bloques 2/2/2/2 con SE y **2.857.125 parámetros**. Los mapas finales pasan de 8×8×256 a **12×12×256**; GAP permite conservar el mismo número de parámetros. Se mantienen batch 32, AdamW, LR 0.001, weight decay 0.0001, máximo 200 épocas, semilla 42, reparto, aumentación y callbacks.

La hipótesis es que el detalle adicional mejore la clasificación, especialmente entre vehículos, y compense el mayor coste. Se evaluará frente a **80,69 % de accuracy y 82,27 % de F1 macro** de experiment03. No se presupone una ganancia: se compararán CE/AP, brecha de datos originales, clases y tiempo, además de accuracy/F1.

Se entrenó desde cero, sin cargar checkpoints anteriores. La caché mmap de 192×192 se mantiene separada de la de 128×128. El notebook guarda el mejor `model.keras`, obligatoriamente `prediction.json`, artefactos y PDF en `experiment05/`.

`SELECT_RUNS` contiene **custom experiment05 + custom experiment03 + ResNet18 experiment04**. La referencia custom aísla resolución; la popular también usa 192×192, pero label smoothing 0.1 en lugar de MixUp, por lo que no aísla arquitectura. Destino: `cnn_challenge_alt/comparisons/custom_experiment05/`.

### Resultados: subir a 192×192 no mejora esta custom

**Experiment05 empeora frente a experiment03. Se conserva la configuración de 128×128 con MixUp alpha 0.2 como mejor modelo custom individual.** La resolución fue el único hiperparámetro cambiado: coinciden arquitectura, número de parámetros, pesos iniciales, resto de configuración y entorno registrado.

Los artefactos son coherentes: accuracy, F1, AP, matrices de confusión, orden de validación, reparto, hashes y mejor época se han comprobado a partir de las probabilidades e historiales. Las configuraciones guardadas coinciden con las celdas de los notebooks, no hay salidas de error y experiment05 generó **2.365 predicciones de test**. La comparación CSV concuerda con las métricas individuales.

| Métrica | Experiment03 · 128×128 | Experiment05 · 192×192 |
|---|---:|---:|
| Accuracy validación (%) | 80,69 | 79,47 |
| F1 macro (%) | 82,27 | 81,02 |
| CE sin suavizado ↓ | 0,695 | 0,704 |
| AP macro (%) | 86,38 | 85,82 |
| Accuracy train original (%) | 98,30 | 97,41 |
| Brecha train–validación (puntos) | 17,61 | 17,94 |
| Entrenamiento (min) | 39,18 | 58,57 |
| Tiempo medio por época (s) | 20,09 | 30,56 |
| Aciertos / validación | 1.513 / 1.875 | 1.490 / 1.875 |
| Mejor época / ejecutadas | 97 / 117 | 95 / 115 |
| Parámetros | 2.857.125 | 2.857.125 |

Fuentes: [métricas experiment05](experiment05/artifacts/metrics.json), [tabla comparativa](../comparisons/custom_experiment05/comparison.csv), [curvas](../comparisons/custom_experiment05/comparison_curves.pdf), [métricas](../comparisons/custom_experiment05/comparison_metrics.pdf), [recall por clase](../comparisons/custom_experiment05/comparison_class_recall.pdf) y [coste](../comparisons/custom_experiment05/comparison_cost.pdf).

La accuracy pierde **1,23 puntos y 23 aciertos**; el F1 macro pierde **1,25 puntos**. CE empeora ligeramente (**0,695 → 0,704**) y AP baja **0,56 puntos**. La brecha final, calculada sin MixUp ni aumentación, sube de **17,61 a 17,94 puntos**: **97,41 % de train original − 79,47 % de validación**. El mayor tamaño de entrada no resuelve el sobreajuste ni la clasificación de esta ejecución.

### Dónde se pierde rendimiento y cuánto cuesta

La pérdida se concentra en vehículos: **Bus pierde 23 verdaderos positivos**, y su F1 baja de **69,66 % a 65,58 %**; **Dump truck pierde diez**, con F1 **73,59 % → 64,66 %**. Excavator baja de **91,14 % a 85,89 %** de F1. Hay ganancias en Small car (seis positivos adicionales), Truck (cuatro), Shipping container (cuatro) y Storage tank (dos), pero no compensan. La hipótesis de que mayor resolución mejorase el reconocimiento de vehículos no queda respaldada globalmente.

Experiment05 corrige 119 errores de experiment03, pero pierde 142 aciertos. Entre las confusiones más frecuentes aparecen Truck → Small car (39), Dump truck → Truck (28) y Bus → Truck (28). Los errores con confianza superior al 99 % pasan de 12 a 16; los superiores al 90 %, de 87 a 88. No se observa una mejora de esos extremos.

La CE mínima del historial es **0,659 en la época 76**, prácticamente igual y ligeramente peor que la mínima de experiment03 (0,657). La mejor accuracy se alcanza en la época 95 y EarlyStopping termina en la 115, 20 después y sin alcanzar el máximo 200. No hay evidencia en esta curva de que simplemente añadir más épocas sea la solución.

El coste aumenta de **39,18 a 58,57 minutos**, aproximadamente un **49 % más**, pese a ejecutar dos épocas menos. El promedio por época pasa de **20,09 a 30,56 segundos**. El tiempo de inferencia de validación registrado también sube (0,341 → 0,526 ms/imagen), aunque una única pasada no es un benchmark controlado. **En estas ejecuciones, aumentar resolución no compensa: exige más cómputo y empeora las métricas principales.**

### Balance tras los cinco entrenamientos custom

**El mejor modelo individual sigue siendo experiment03:** CNN `residual_se_v1`, 128×128, MixUp alpha **0.2**, sin label smoothing ni Dropout adicional, AdamW con LR 0.001 y weight decay 0.0001. Mantiene **80,69 % de accuracy y 82,27 % de F1 macro** con 2.857.125 parámetros.

Label smoothing 0.1 no aportó una mejora general en experiment02; MixUp 0.2 sí mejoró la referencia inicial; MixUp 0.4 mejoró CE y brecha a costa de accuracy/F1; 192×192 no mejoró el resultado individual. Esta es la conclusión de las ejecuciones observadas, no una regla universal sobre las técnicas. Se conservan todos los resultados, incluida la custom de 192×192, porque comete errores diferentes.

### Diagnóstico adicional sin entrenar: combinar probabilidades

Se calculó una **media aritmética fija 50/50 de probabilidades de validación** de dos modelos, alineando nombres de imagen y clases. No se ajustaron pesos ni se generaron predicciones de test para estas combinaciones. Se examinaron estas dos parejas, como análisis exploratorio sobre la misma validación ya usada para seleccionar modelos:

| Combinación 50/50 | Aciertos / 1.875 | Accuracy (%) | F1 macro (%) | CE ↓ | AP macro (%) |
|---|---:|---:|---:|---:|---:|
| Custom experiment03 + custom experiment05 | 1.533 | 81,76 | 83,27 | 0,596 | 88,48 |
| Custom experiment03 + ResNet18 experiment04 | 1.541 | 82,19 | 83,80 | 0,613 | 88,87 |

Fuentes: [probabilidades custom03](experiment03/artifacts/validation_predictions.npz), [probabilidades custom05](experiment05/artifacts/validation_predictions.npz) y [probabilidades ResNet18](../popular_architectures/resnet18/experiment04/artifacts/validation_predictions.npz). Las medias se evaluaron directamente en memoria durante la revisión; no son métricas de nuevos entrenamientos ni se han añadido a los CSV de modelos individuales.

Las dos custom juntas ganan **20 aciertos** respecto a custom03 sola, aunque custom05 individual sea peor. Esto revela complementariedad de errores y ofrece una opción sin otro entrenamiento **si el objetivo permite un ensemble**. La combinación con ResNet18 mejora aún más, pero deja de ser una solución exclusivamente custom. En ambos casos hay que ejecutar dos modelos en inferencia: no se conserva el coste de un modelo individual. Estos resultados no demuestran una mejora en test ni una ventaja estadística; reutilizan la misma validación y requieren una evaluación independiente antes de generalizar.

### ¿Merece la pena otra prueba?

No seguiría aumentando la resolución ni el número máximo de épocas. Si se permite combinar modelos, la media 50/50 ya observada es una opción concreta antes de gastar tiempo en otro entrenamiento; el uso de ensembles tiene antecedentes como [Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), sin que sus resultados se puedan extrapolar automáticamente a este proyecto.

**Si se busca mejorar un único modelo, haría como máximo una prueba adicional de Dropout `0.2` después de GAP, partiendo de experiment03.** Se mantendrían 128×128, MixUp 0.2, LR, weight decay y demás protocolo. El único cambio sería Dropout: una regularización distinta de los ajustes ya probados, sin ampliar mapas de características ni parámetros aprendidos. [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) desactiva unidades durante entrenamiento; en nuestra red, su posible beneficio tendría que comprobarse, no deducirse solo de la brecha. Se conservaría únicamente si mejora validación o un compromiso de métricas relevante.

La recomendación no implica que otra prueba vaya a superar el 80,69 %. Si se da por terminada la fase, **experiment03 ya es una referencia defendible y competitiva**; confirmar su estabilidad con otra semilla aportaría más evidencia que seguir acumulando ajustes sobre la misma partición. La pequeña ventaja individual frente a ResNet18 tampoco demuestra superioridad general.

**No se ha creado experiment06 ni iniciado otro entrenamiento.** Los notebooks y artefactos de las ejecuciones se mantienen; esta revisión solo actualiza la documentación y calcula el diagnóstico de medias en validación.




## Ejecutar y comparar

El notebook ya tiene `NOTEBOOK_PATH` configurado para su carpeta. Ejecutarlo en orden crea `model.keras`, genera obligatoriamente `prediction.json`, y guarda configuración, arquitectura, reparto, historial, métricas y probabilidades en `artifacts/`, además de los PDF en `plots/`.

Se mantienen la caché en disco con `mmap` y las rutas portables del [baseline](../baseline/README.md). La caché de 128×128 se reutiliza en `cnn_challenge_alt/.cache/` sin duplicarla por arquitectura. Cada compañero necesita las dependencias y su copia local de los datos; no hay rutas personales en el notebook.

La selección de comparaciones sigue la decisión de esta fase: **custom actual + mejor custom anterior (cuando exista) + ResNet18 experiment04** como referencia popular fija. Para custom experiment01 no hay una custom anterior; su sección 9 selecciona únicamente esta ejecución y ResNet18 experiment04.

La comparación existente conserva el nombre de carpeta `cnn_challenge_alt/comparisons/custom_experiment01_vs_popular_experiment01/`, pero su CSV contiene **solo esas dos ejecuciones**: el nombre heredado no implica que se hayan comparado los cuatro experiment01 populares. Para regenerarla tras reiniciar el kernel, ejecutar solo la primera celda de código y la sección 9. Todas las ejecuciones seleccionadas deben tener artefactos completos.

El protocolo inicial de entrenamiento sigue siendo el de la primera ronda popular, pero **la referencia de resultados elegida es ResNet18 experiment04**, con las diferencias de resolución y suavizado indicadas arriba. El resto de ejecuciones populares queda como contexto en su [registro](../popular_architectures/README.md).

En los siguientes experimentos, completar accuracy, F1 macro, CE, brecha train–validación, métricas por clase, parámetros y tiempo. Revisar especialmente Truck, Bus y Dump truck, y no extraer conclusiones fuertes de Helipad con solo nueve muestras de validación. Los resultados de test/Codabench se documentan por separado. Una sola semilla y partición no demuestran superioridad general.
