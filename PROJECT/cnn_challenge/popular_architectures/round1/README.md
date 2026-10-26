# Round 1 · referencias con arquitecturas populares

Ronda completada: **seis arquitecturas entrenadas desde cero y siete ejecuciones**,
contando el primer intento fallido de VGG16. El objetivo es obtener referencias
funcionales para comparar después las CNN propias, explorar distintos diseños y
registrar tanto rendimiento como coste.

Los resultados documentados son de **validación local**, con una partición y una
semilla de entrenamiento. El mejor resultado observado es DenseNet121, con
**74,35 % de accuracy y 75,31 % de F1 macro**. Las seis ejecuciones seleccionadas
aprenden las 13 clases, aunque todas presentan sobreajuste.

## 1. Experimentos realizados

| Experimento | Notebook | Configuración y estado |
|---|---|---|
| 01-a | [VGG16-a](01_vgg16-a/vgg16_round1_a.ipynb) | Adam, LR 1e-3; colapso a Building, 19,36 % de accuracy |
| 01-b | [VGG16-b](01_vgg16-b/vgg16_round1_b.ipynb) | LR 1e-4; ejecución válida seleccionada para representar VGG16 |
| 02 | [GoogLeNet](02_googlenet/googlenet_round1.ipynb) | Inception-v1; LR 1e-3 |
| 03 | [ResNet18](03_resnet18/resnet18_round1.ipynb) | Conexiones residuales; LR 1e-3 |
| 04 | [DenseNet121](04_densenet121/densenet121_round1.ipynb) | Conexiones densas; LR 1e-3 |
| 05 | [MobileNetV2](05_mobilenet_v2/mobilenet_v2_round1.ipynb) | Bloques residuales invertidos; alpha=1, LR 1e-3 |
| 06 | [EfficientNetB0](06_efficientnet_b0/efficientnet_b0_round1.ipynb) | Variante B0; LR 1e-3 |

Las redes se adaptan a RGB y 13 clases. VGG16 usa una cabeza GAP en lugar de
sus FC originales. GoogLeNet conserva sus nueve módulos y omite las cabezas
auxiliares. EfficientNet compensa el reescalado interno de Keras para recibir
el mismo rango de píxeles ante sus convoluciones. Las implementaciones y otras
adaptaciones se describen en el apartado 4 de los notebooks y en el
[README general](../../README.md).

### Ajuste de VGG16: variantes a y b

VGG16-a terminó en la época 22, con el mejor checkpoint en la 2. Predijo
Building para las 1.875 imágenes de validación: acertó los 363 edificios y obtuvo
19,36 %. Se conserva como ensayo fallido, excluido de la comparación principal.

VGG16-b se entrenó de nuevo desde cero, cambiando únicamente el LR inicial de
Adam, de **1e-3 a 1e-4**. Conservó el reparto, la semilla, el hash de los pesos
iniciales, la arquitectura y los demás parámetros de entrenamiento. Alcanzó
72,37 % de accuracy y predijo las 13 clases. Esta comparación apoya la utilidad
del LR menor para esta ejecución; no demuestra que sea el LR óptimo ni que el
mismo cambio beneficie a todas las arquitecturas.

La variante a mantiene el identificador histórico `round1_01_vgg16` en sus
artefactos; la b utiliza `round1_01_vgg16-b`. Los resultados de a se conservan
íntegros en su carpeta, separadamente de los de b.

## 2. Protocolo de entrenamiento y evaluación

| Parámetro | Valor |
|---|---|
| Dataset | xView recognition, 13 clases |
| Reparto | 16.871 imágenes de train y 1.875 de validación; histórico 90/10, semilla 1, sin estratificación |
| Entrada | RGB 64×64, reducción por promedio y float32 en [0,1] |
| Inicialización | Aleatoria; sin transfer learning y con todas las capas entrenables |
| Semilla de entrenamiento | 42; operaciones deterministas |
| Optimizador | Adam, AMSGrad, clipnorm 1 |
| LR inicial | 1e-3, salvo VGG16-b con 1e-4 |
| Batch / máximo de épocas | 32 / 200 |
| Selección del modelo | Primer máximo de `val_accuracy`; se recarga el mejor checkpoint |
| Reducción de LR | Factor 0.2 tras 10 épocas sin mejora; mínimo 1e-7 |
| EarlyStopping | 20 épocas sin mejora en `val_accuracy` |
| Dropout de la cabeza | 0.1 |
| L2 añadida / aumentación | Desactivadas |
| Normalización y regularización internas | Las propias de cada arquitectura |
| Caché de entrada | Compartida en `cnn_challenge/.cache/`, modo `mmap` |
| Entorno registrado | NVIDIA GeForce RTX 4070 Ti, TensorFlow 2.15.1, Python 3.10.21 |

Todas las ejecuciones seleccionadas terminaron por EarlyStopping, 20 épocas
después de su mejor accuracy. Las métricas finales corresponden a ese checkpoint,
con Dropout y aumentación desactivados y las capas BN en modo inferencia.

El protocolo es común salvo el LR ajustado de VGG16-b. La misma semilla facilita
la reproducibilidad, pero arquitecturas distintas no parten de los mismos pesos.
El presupuesto máximo es compartido; el número real de épocas y el tiempo varían.

## 3. Resultados de validación

Fuente: [comparison.csv](../../comparisons/comparison.csv), contrastado con los
`artifacts/metrics.json` de cada ejecución. Las métricas se expresan en porcentaje;
precision, recall y F1 son medias macro sobre las 13 clases.

| Modelo | Accuracy (%) | Precision macro (%) | Recall macro (%) | F1 macro (%) | Mejor época / ejecutadas |
|---|---:|---:|---:|---:|---:|
| DenseNet121 | 74,35 | 76,18 | 74,89 | 75,31 | 69 / 89 |
| VGG16-b | 72,37 | 76,37 | 72,89 | 74,43 | 27 / 47 |
| ResNet18 | 71,89 | 74,25 | 72,52 | 73,18 | 53 / 73 |
| MobileNetV2 | 70,13 | 72,16 | 70,27 | 71,00 | 75 / 95 |
| EfficientNetB0 | 67,47 | 69,71 | 66,22 | 67,63 | 48 / 68 |
| GoogLeNet | 65,12 | 64,49 | 62,44 | 63,33 | 36 / 56 |

DenseNet121 encabeza accuracy, recall macro y F1 macro. VGG16-b obtiene la mayor
precision macro. DenseNet121 supera a VGG16-b en **1,97 puntos de accuracy**, es
decir, 37 aciertos más sobre las mismas 1.875 imágenes. VGG16-b y ResNet18 están
separadas por solo **9 aciertos**. Estas diferencias describen las ejecuciones
observadas; una semilla no permite establecer superioridad general.

### Coste y brecha de generalización

| Modelo | Parámetros (millones) | Fit (min) | Predicción (ms/imagen) | Accuracy train, inferencia (%) | Brecha train−validación (pp) |
|---|---:|---:|---:|---:|---:|
| DenseNet121 | 7,05 | 62,71 | 0,679 | 99,99 | 25,65 |
| VGG16-b | 14,72 | 10,41 | 0,228 | 99,92 | 27,54 |
| ResNet18 | 11,19 | 12,04 | 0,192 | 99,99 | 28,10 |
| MobileNetV2 | 2,27 | 28,44 | 0,305 | 99,99 | 29,85 |
| EfficientNetB0 | 4,07 | 36,77 | 0,506 | 99,93 | 32,47 |
| GoogLeNet | 5,99 | 14,60 | 0,261 | 99,23 | 34,11 |

- **DenseNet121** ofrece el mayor rendimiento observado y la menor brecha de las
  seis referencias, pero el mayor tiempo de entrenamiento: unos 63 minutos.
- **VGG16-b** obtiene la segunda accuracy y el menor tiempo total de fit, unos
  10 minutos, aunque tiene más parámetros que las demás redes seleccionadas.
- **ResNet18** queda cerca de VGG16-b y registra el menor tiempo por imagen en
  esta medición de predicción.
- **MobileNetV2** es la referencia más pequeña: 2,27 millones de parámetros para
  una accuracy de 70,13 %. Tener menos parámetros no la hizo la más rápida en
  este entorno.
- **EfficientNetB0 y GoogLeNet** completan la diversidad de diseños y muestran
  resultados inferiores bajo estas configuraciones, con margen de ajuste.

Los seis entrenamientos seleccionados suman unos **165 minutos de fit**, sin
contar VGG16-a, preparación de datos, evaluación y gráficas. El fit incluye
validación y callbacks, incluido el guardado de checkpoints. Los tiempos dependen
del hardware, las implementaciones, las épocas y el estado del sistema. La
predicción incluye el pipeline y el arranque de la llamada; no es un benchmark
estandarizado de latencia por imagen.

## 4. Lectura de las gráficas

Las comparaciones ya se han generado en
[`cnn_challenge/comparisons/`](../../comparisons/).

| Figura | Qué permite observar |
|---|---|
| [Métricas agregadas](../../comparisons/comparison_metrics.pdf) | Clasificación por accuracy, precision, recall y F1 macro; el ganador depende de la métrica |
| [Curvas de entrenamiento](../../comparisons/comparison_curves.pdf) | Separación entre train y validación, evolución de CE y ajustes de LR |
| [Recall por clase](../../comparisons/comparison_class_recall.pdf) | Fortalezas y debilidades de cada arquitectura |
| [Rendimiento frente a coste](../../comparisons/comparison_cost.pdf) | Compromiso entre accuracy, parámetros y tiempo de entrenamiento |

### Sobreajuste y selección por accuracy

Las curvas de train se acercan al 100 %, mientras las accuracies de validación
se estabilizan entre 65 % y 74 %. Las brechas de inferencia van de **25,65 a
34,11 puntos**. La capacidad de ajustar train es suficiente en estas ejecuciones;
mejorar la generalización es una prioridad para los siguientes experimentos.

La CE de validación alcanza su mínimo antes que la accuracy máxima en las seis
redes. Después aumenta, aunque la accuracy todavía mejora o permanece estable:

| Modelo | Época de CE mínima | CE mínima de validación | CE del checkpoint de mayor accuracy |
|---|---:|---:|---:|
| DenseNet121 | 8 | 0,957 | 1,729 |
| VGG16-b | 11 | 0,943 | 2,697 |
| ResNet18 | 7 | 1,112 | 2,001 |
| MobileNetV2 | 10 | 1,309 | 2,158 |
| EfficientNetB0 | 7 | 1,116 | 2,232 |
| GoogLeNet | 18 | 1,147 | 2,043 |

Esto es compatible con errores cada vez más confiados. No se ha calculado una
métrica explícita de calibración. El checkpoint usado en la comparación es el
de mayor accuracy, no el de menor CE; no se han guardado todos los checkpoints
intermedios. Por tanto, la tabla de mínimos procede del historial y no implica
que esos modelos anteriores estén disponibles para evaluación adicional.

Las curvas terminan en épocas diferentes porque cada red tuvo su propia parada.
Se comparan sus mejores checkpoints bajo el mismo criterio, con costes reales
distintos.

### Resultados por clase

- **Cargo plane, Small car y Building** mantienen recalls relativamente altos
  en las seis redes: aproximadamente 83–88 %, 81–90 % y 78–86 %, respectivamente.
- **Truck** es una dificultad común: recall entre 37,34 % y 49,38 %. Las matrices
  individuales muestran confusiones con Small car, Dump truck y Bus. VGG16-b
  tiene el mayor recall de Truck, aunque DenseNet121 gana en accuracy global.
- **Bus** también presenta margen de mejora, con recall entre 45,98 % y 62,64 %.
- **Helipad** solo tiene nueve ejemplos de validación. VGG16-b y ResNet18 aciertan
  ocho; DenseNet121, siete; MobileNetV2, seis; EfficientNetB0, cinco; GoogLeNet,
  tres. Un único acierto cambia su recall en 11,11 puntos; los porcentajes deben
  interpretarse junto a ese soporte reducido.

El mejor modelo agregado no es necesariamente el mejor para todas las clases.
Las matrices individuales y curvas precision–recall están en el `plots/` de cada
experimento; AP no equivale a la precision del argmax.

## 5. Alcance de las conclusiones

Esta ronda proporciona seis referencias funcionales y reproducibles, no una
búsqueda exhaustiva de los mejores hiperparámetros de cada arquitectura.

Solo se ha utilizado la semilla de entrenamiento 42 y una partición de validación.
[`seed_summary.json`](../../comparisons/seed_summary.json) registra `n=1` por
configuración y desviaciones `null`: no hay una estimación de variabilidad entre
semillas ni intervalos de confianza. La validación se usa también para seleccionar
época y ajustar VGG16, así que no es una evaluación independiente de esas decisiones.

Se comparan adaptaciones a 64×64 y 13 clases, con cabezas modificadas y un ajuste
de LR específico para VGG16. El orden observado puede cambiar al modificar la
resolución, la regularización, la optimización o la semilla. Estas seis ejecuciones
no prueban que una familia sea intrínsecamente mejor que otra.

Cada ejecución conserva su `model.keras`, `prediction.json`, `artifacts/` y
`plots/`. Generar el JSON de test no aporta una puntuación de test: no hay métricas
de Codabench documentadas en esta comparación. Las puntuaciones externas deberán
registrarse aparte cuando existan.

## 6. Siguientes pasos y posible Round 2

### Prioridad: CNN propia

Las referencias actuales permiten empezar a diseñar y entrenar la CNN propia con
el mismo reparto. La comparación debería considerar accuracy, F1 y recall macro,
comportamiento por clase, parámetros y tiempo. Dados los resultados de train,
conviene explorar una capacidad ajustada al problema y técnicas de regularización,
en lugar de aumentar únicamente la profundidad.

### Ajustes menores dentro de Round 1

Si se decide afinar alguna referencia, crear variantes `-b`, `-c`, etc., conservando
los resultados actuales. Cambiar una variable por vez permite interpretar mejor
el efecto: Dropout de la cabeza, aumentación, LR o criterio de los callbacks.
La L2 configurable de esta plantilla solo está implementada para la CNN propia;
aplicarla a los backbones populares requeriría implementar y documentar la variante.

También sería útil repetir las configuraciones más relevantes con otras semillas,
manteniendo el reparto. Esto permitiría estimar si las diferencias observadas son
estables. No es necesario repetir todas las familias para empezar a trabajar en
la CNN propia.

### Round 2: hipótesis pendientes, sin ejecuciones preparadas

Una segunda ronda se reservaría para cambios de resolución o variantes superiores
de arquitectura. Para contener el coste, seleccionar primero una o dos referencias:

| Prueba propuesta | Control | Pregunta |
|---|---|---|
| ResNet18 o DenseNet121 a 128×128 | Misma arquitectura a 64×64, mismo reparto y protocolo | ¿Conservar más detalle mejora la generalización y las clases difíciles? |
| ResNet50 | ResNet18 a la misma resolución y con el mismo protocolo | ¿Una variante de mayor capacidad aporta una mejora que compense su coste? |
| EfficientNetB1, solo si interesa esa familia | EfficientNetB0 a la misma resolución | ¿El aumento de escala mejora su referencia actual? |

No combinar de entrada el cambio de resolución y de versión: se perdería la
posibilidad de atribuir la diferencia. Tampoco hay evidencia todavía de que esas
variantes vayan a mejorar el resultado. El sobreajuste actual hace razonable probar
primero regularización o resolución antes de invertir en muchas redes mayores.

## 7. Conservar y regenerar las comparaciones

La comparación principal incluye **VGG16-b y las otras cinco arquitecturas**;
VGG16-a permanece fuera por su colapso. La selección se ha ejecutado en el apartado
9 del [último notebook, EfficientNetB0](06_efficientnet_b0/efficientnet_b0_round1.ipynb).
Las llamadas de comparación de ese cuaderno están activas; al usar «Ejecutar todo»
volverá a generar las comparaciones. Los restantes cuadernos conservan las llamadas
comentadas y algunas selecciones antiguas que deberán revisarse antes de activarlas.

Para regenerar solo la comparación, ejecutar la primera celda de código del último
notebook, comprobar `SELECT_RUNS` y ejecutar únicamente el apartado 9. No hace falta
cargar imágenes ni entrenar. Se actualizan `comparison.csv`, `seed_summary.json`
y los cuatro PDF en `cnn_challenge/comparisons/`.

Para conservar una selección distinta de resultados, copiar esa carpeta antes de
regenerarla. Para repetir un entrenamiento sin sobrescribir el anterior, crear
otra carpeta y actualizar `NOTEBOOK_PATH` e `experiment_id`. La caché puede
reutilizarse entre experimentos con el mismo dataset y resolución.
