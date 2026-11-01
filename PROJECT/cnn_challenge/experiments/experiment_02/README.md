# Experimento propio 02 · CNN compacta con Adam

[Notebook ejecutado](cnn_experiment_02.ipynb) · [Configuración](artifacts/config.json) ·
[Métricas](artifacts/metrics.json) · [Historia](artifacts/history.csv) ·
[Comparación con Round 1](../../comparisons/experiment_02_vs_round1/comparison.csv)

**Resultado:** accuracy de validación **73,92 %**, F1 macro **75,04 %** y
**288.685 parámetros**. Cambiar SGD por Adam mejora 3,79 puntos de accuracy
—71 aciertos más— y reduce el fit completo de 13,66 a **4,87 minutos**.
La accuracy de train sube a 99,95 % y la brecha de inferencia alcanza 26,03 puntos:
la mejora de optimización viene acompañada de sobreajuste.

La CNN queda a **ocho aciertos de DenseNet121**, la mejor referencia de Round 1,
con 24,4 veces menos parámetros. Es una comparación de configuraciones y una
sola semilla, no una demostración de equivalencia entre modelos.

## 1. Hipótesis y control del experimento

Se parte del [experimento 01](../experiment_01/README.md) y se cambia únicamente
el optimizador de entrenamiento: **SGD → Adam**, manteniendo el LR inicial
numérico de 1e-3. Se busca aprender más rápido y mejorar el ajuste y la validación
sin ampliar la arquitectura.

La red conserva seis convoluciones 3×3, bloques de 32/64/128 filtros, tres MaxPool,
GAP, Dropout 0.1 y Dense(13, Softmax). No se añaden BN, L2 ni aumentación.

| Parámetro | Valor |
|---|---|
| Optimizador | Adam, beta1=0.9, beta2=0.999, epsilon=1e-8, AMSGrad y clipnorm 1 |
| LR inicial | 1e-3; mismo valor numérico que en SGD |
| Entrada / batch | RGB 64×64 en [0,1] / 32 |
| Inicialización | He, desde cero, semilla 42 |
| Reparto | Histórico 90/10: 16.871 train y 1.875 validación; semilla 1, sin estratificación |
| Selección / máximo de épocas | Primer máximo de `val_accuracy` / 200 |
| Reducción de LR | Factor 0.2 tras 10 épocas sin mejora, mínimo 1e-7 |
| EarlyStopping | 20 épocas sin mejora en `val_accuracy` |
| Regularización | Dropout 0.1; sin BN, L2 ni aumentación |
| Caché / entorno | `mmap`; RTX 4070 Ti, TensorFlow 2.15.1, Python 3.10.21 |

**El hash de los pesos iniciales coincide con el del experimento 01**, al igual
que los hashes del dataset y del reparto. La salida del notebook confirma
`Pesos iniciales idénticos al experimento 01 guardado: True`. La duda surgida en
la preparación local con CPU queda resuelta para estas dos ejecuciones reales.
No se han cargado los pesos entrenados de SGD.

El campo heredado `momentum=0.9` no se utiliza con Adam. Se conservan las reglas
de los callbacks; sus acciones ocurren en épocas distintas porque cambia la
trayectoria de aprendizaje. El cambio prueba el régimen de optimización completo,
no implica que el mismo LR numérico produzca actualizaciones equivalentes.

## 2. Resultado frente a SGD

Todas las métricas finales corresponden al mejor checkpoint por accuracy de
validación, evaluado en inferencia. Las medias macro incluyen las 13 clases.

| Métrica | Experimento 01 · SGD | Experimento 02 · Adam |
|---|---:|---:|
| Accuracy validación (%) | 70,13 | 73,92 |
| Precision macro (%) | 73,36 | 77,38 |
| Recall macro (%) | 68,41 | 73,17 |
| F1 macro (%) | 70,17 | 75,04 |
| AP macro (%) | 74,50 | 80,07 |
| Train en inferencia (%) | 77,80 | 99,95 |
| Brecha train−validación (pp) | 7,66 | 26,03 |
| CE de validación del checkpoint | 0,942 | 1,773 |
| Mejor época / ejecutadas | 142 / 162 | 37 / 57 |
| Fit completo (min) | 13,66 | 4,87 |
| Mediana de segundos por época | 4,98 | 5,03 |

Adam consigue **1.386 aciertos de 1.875**, frente a 1.315 con SGD. El F1 macro
mejora **4,87 puntos**. El ajuste de train también cambia mucho: la misma red
pasa de 77,80 % a 99,95 % en inferencia, sin añadir parámetros.

Esto indica que la optimización limitaba el resultado anterior. La explicación
de su menor ajuste de train no puede atribuirse únicamente a tener una red
pequeña. Con Adam, esta misma capacidad permite ajustar casi todo el entrenamiento.
La menor brecha de SGD tampoco bastaba para concluir que fuese mejor: en estas
ejecuciones, Adam ofrece mayor accuracy y F1 de validación, aunque peor CE.

### Rapidez de aprendizaje

| Umbral de accuracy de validación | SGD: primera época / tiempo acumulado | Adam: primera época / tiempo acumulado |
|---|---|---|
| 65 % | 73 / 369,15 s | 6 / 33,53 s |
| 70 % | 142 / 717,03 s | 11 / 58,68 s |
| 72 % | No alcanzado | 22 / 113,98 s |

Se registra el primer cruce del umbral, no que se mantenga en todas las épocas
siguientes. Los tiempos acumulados suman `epoch_seconds` y no incluyen todos los
costes de callbacks; el fit completo se mide por separado.

El coste por época es parecido —aproximadamente cinco segundos—. La reducción
del tiempo total procede sobre todo de necesitar menos épocas: **57 frente a
162**, para un fit completo **2,8 veces más rápido**.

## 3. Lectura de las curvas y parada

Las [curvas individuales](plots/training_curves.pdf) muestran un aprendizaje
rápido al principio. En la época 11 ya se alcanza **70,67 % de validación**, con
CE **0,908**, el mínimo del historial. Después train sigue mejorando, pero la
CE de validación crece hasta **1,773** en el checkpoint de mayor accuracy,
seleccionado en la época 37.

En esa época, `fit` registra 99,07 % de accuracy en train con Dropout activo;
la evaluación del checkpoint sin Dropout obtiene 99,95 %. La brecha de 26,03
puntos se calcula con train y validación en inferencia.

La parada en la época **57** es coherente: transcurren 20 épocas sin superar la
accuracy de la 37. Los LR utilizados bajan a 2e-4 en la época 33 y a 4e-5 en la
48. El [gráfico de LR y tiempos](plots/learning_rate_time.pdf) y el historial
permiten revisar estos cambios.

El patrón indica **sobreajuste**: la red ya puede ajustar train y gran parte del
esfuerzo posterior no se traduce en mejora de validación. El crecimiento de CE
es compatible con errores cada vez más confiados; no se ha medido calibración
explícitamente. Elegir el checkpoint de menor CE no garantizaría mayor accuracy:
en este historial su accuracy es 70,67 %, inferior al 73,92 % seleccionado.

## 4. Comparación con mejor y peor referencia de Round 1

La comparación guardada contiene exclusivamente esta CNN, DenseNet121 y
GoogLeNet. No incluye una arquitectura popular intermedia.

| Modelo | Accuracy validación (%) | F1 macro (%) | Parámetros (M) | Fit completo (min) |
|---|---:|---:|---:|---:|
| CNN propia · Adam | 73,92 | 75,04 | 0,289 | 4,87 |
| DenseNet121 | 74,35 | 75,31 | 7,051 | 62,71 |
| GoogLeNet | 65,12 | 63,33 | 5,987 | 14,60 |

Frente a **DenseNet121**, la diferencia es de solo **0,43 puntos de accuracy**:
1.386 frente a 1.394 aciertos. La CNN propia usa **24,4 veces menos parámetros**
y su fit completo es **12,9 veces más rápido** en estas ejecuciones. El F1 macro
queda a unos 0,28 puntos. La diferencia pequeña exige prudencia con una única
semilla; no establece equivalencia ni superioridad general.

Frente a **GoogLeNet**, mejora 8,80 puntos de accuracy —165 aciertos— con menos
parámetros y menor tiempo total. La propia alcanza la mayor precision macro de
las tres —77,38 %—; DenseNet121 mantiene la mayor accuracy, recall y F1 macro.

Las figuras están disponibles en:

- [Métricas agregadas](../../comparisons/experiment_02_vs_round1/comparison_metrics.pdf).
- [Curvas superpuestas](../../comparisons/experiment_02_vs_round1/comparison_curves.pdf).
- [Recall por clase](../../comparisons/experiment_02_vs_round1/comparison_class_recall.pdf).
- [Accuracy frente a parámetros y tiempo](../../comparisons/experiment_02_vs_round1/comparison_cost.pdf).

Los tiempos incluyen validación y callbacks, dependen del entorno y no son un
benchmark aislado de cómputo de la arquitectura.

## 5. Resultados por clase

La [matriz de confusión](plots/confusion_matrix.pdf) y las
[curvas precision–recall](plots/precision_recall.pdf) complementan las métricas
agregadas. El recall cambia así respecto a SGD:

| Clase | Recall SGD (%) | Recall Adam (%) | Aciertos SGD → Adam / soporte |
|---|---:|---:|---|
| Cargo plane | 84,75 | 84,75 | 50 → 50 / 59 |
| Small car | 89,36 | 87,23 | 294 → 287 / 329 |
| Bus | 57,47 | 59,77 | 100 → 104 / 174 |
| Truck | 39,83 | 49,79 | 96 → 120 / 241 |
| Motorboat | 65,14 | 67,89 | 71 → 74 / 109 |
| Fishing vessel | 72,60 | 78,08 | 53 → 57 / 73 |
| Dump truck | 59,32 | 63,56 | 70 → 75 / 118 |
| Excavator | 75,31 | 70,37 | 61 → 57 / 81 |
| Building | 82,09 | 88,43 | 298 → 321 / 363 |
| Helipad | 55,56 | 77,78 | 5 → 7 / 9 |
| Storage tank | 60,53 | 75,00 | 92 → 114 / 152 |
| Shipping container | 75,56 | 70,37 | 102 → 95 / 135 |
| Pylon | 71,88 | 78,12 | 23 → 25 / 32 |

Los avances más relevantes incluyen:

- **Truck:** de 96 a 120 aciertos, +9,96 puntos de recall. Aun así, solo recupera
  el 49,79 %; se confunde con Small car —39—, Bus —28— y Dump truck —23—.
- **Storage tank:** de 92 a 114 aciertos, +14,47 puntos. Las confusiones con
  Building bajan de 31 a 14.
- **Building:** de 298 a 321 aciertos; su recall alcanza 88,43 %.
- **Helipad:** de cinco a siete aciertos de nueve; su precision es 100 %, con
  dos falsos negativos. El soporte reducido impide interpretar la mejora como
  una conclusión estable.

Hay retrocesos en Small car, Excavator y Shipping container. El siguiente cambio
se evaluará también por F1/recall macro y por clase, sin reducir la valoración a
accuracy global. AP resume la curva precision–recall y no equivale a la precision
del argmax.

## 6. Conclusiones y siguiente experimento propuesto

La hipótesis de mejorar optimización y rapidez queda apoyada en esta ejecución:
la CNN supera al ensayo con SGD sin cambiar capacidad, datos ni pesos iniciales.
El límite prioritario pasa a ser la **generalización**. No hay una razón empírica
para aumentar profundidad o filtros antes de probar regularización.

### Siguiente prueba elegida: resolución 128×128

El [experimento 03](../experiment_03/README.md) ya está preparado para cambiar
únicamente la entrada de **64×64 a 128×128**, conservando arquitectura, Adam, LR,
batch, Dropout y callbacks. Se mantienen las semillas de entrenamiento **42**
y reparto **1**; no se activan volteos ni otra aumentación.

La hipótesis es que el detalle adicional ayude a distinguir clases parecidas.
Con esta arquitectura y GAP se conservan los **288.685 parámetros**, pero crecen
los mapas de activación y el coste de las convoluciones. Deben medirse accuracy,
F1, errores por clase, brecha en inferencia y tiempo frente a este experimento 02.

Es una prueba pendiente de entrenamiento, sin garantía de mejora. La caché de
128×128 se guardará separadamente de la de 64×64. Las referencias populares de
Round 1 siguen a 64×64, por lo que el control directo para estudiar la resolución
es este experimento 02.

### Alternativas posteriores, en pruebas separadas

- Volteos horizontales/verticales mediante aumentación, si interesa estudiar su
  efecto sobre la generalización. No se incluyen en la prueba de resolución.
- Dropout 0.1 → 0.2 o L2=1e-5, una técnica por experimento.
- BN, para estudiar estabilidad y representaciones sin asumir una mejora automática.

Las comparaciones futuras seguirán usando solo DenseNet121 y GoogLeNet como
referencias populares, además de la CNN propia evaluada. Se mantienen las semillas
actuales en esta serie de experimentos.

## 7. Resultados conservados y regeneración

Se conservan `model.keras` y `prediction.json` junto al notebook, auxiliares en
`artifacts/` y figuras PDF en `plots/`. Las comparaciones están en
[`comparisons/experiment_02_vs_round1/`](../../comparisons/experiment_02_vs_round1/).

Las llamadas finales de comparación de esta copia están actualmente activas.
Para regenerar únicamente esas comparaciones, ejecutar la primera celda de código
y el apartado 9; no hace falta cargar imágenes ni volver a entrenar. Repetir un
entrenamiento en esta carpeta sobrescribe los resultados: crear otra carpeta para
conservar este ensayo.

Las conclusiones se limitan a una semilla y una partición usada para selección.
No se ha medido variabilidad entre semillas ni significación estadística.
Generar `prediction.json` no aporta una puntuación de test: las métricas de
Codabench, cuando existan, deben registrarse aparte.
