# Experimento propio 05 · LR más suave y L2 ligera

[Notebook ejecutado](cnn_experiment_05.ipynb) · [Configuración](artifacts/config.json) ·
[Métricas](artifacts/metrics.json) · [Historia](artifacts/history.csv) ·
[Referencia: experimento 04](../experiment_04/README.md)

**Resultado:** **78,40 % de accuracy y 79,60 % de F1 macro** en validación.
Son **12 aciertos más** que el experimento 04, sin cambiar arquitectura ni número
de parámetros. La CE predictiva baja de 0,843 a **0,709** y la brecha de inferencia
de 19,05 a **14,75 puntos**. La ejecución apoya una mejora moderada de accuracy y
un mejor compromiso de generalización, con validación menos errática.

El recall macro baja ligeramente y el fit completo sigue rondando 64 minutos.
El objetivo del 80 % aún no se alcanza: faltan 30 aciertos sobre esta validación.

## 1. Hipótesis y cambios realizados

Se conserva la CNN ampliada a 224×224 y se combinan una optimización más suave
con L2 ligera, para reducir oscilaciones y sobreajuste sin añadir capacidad.

| Parámetro | Experimento 04 | Experimento 05 |
|---|---|---|
| LR inicial de Adam | 1e-3 | **3e-4** |
| L2 en kernels Conv/Dense | 0 | **1e-5** |
| Monitor de reducción de LR | `val_accuracy`, max | **`val_ce`, min** |
| Paciencia de reducción | 10 | **5** |
| Factor de reducción | 0.2 | **0.5** |

Es un ensayo combinado. La diferencia observada no se puede atribuir
exclusivamente al LR inicial, al scheduler o a L2.

### Configuración conservada

- RGB **224×224**, batch **32**, caché compartida en `mmap`.
- Semilla de entrenamiento **42**, reparto **1**, sin cambios.
- Reparto histórico 90/10, sin estratificación: 16.871 train y 1.875 validación.
- Cinco bloques con dos convoluciones 3×3, filtros **32/64/128/256/256** y MaxPool.
- Conv sin bias → BN → ReLU; cabeza GAP → Dropout **0.2** → Dense(13, Softmax).
- Volteos horizontales/verticales, sin rotaciones ni recortes.
- Adam, beta1=0.9, beta2=0.999, epsilon=1e-8, AMSGrad y clipnorm 1.
- Máximo **200 épocas**, mínimo LR **1e-7**.
- Checkpoint por primer máximo estricto de `val_accuracy`; EarlyStopping por
  el mismo monitor con paciencia **20**.

La red conserva **2.360.173 parámetros**. L2 se aplica a los kernels de las diez
convoluciones y del clasificador, no a BN, y no añade pesos.
Se entrenó desde cero y **el hash de pesos iniciales coincide con el del
experimento 04**. También coinciden los hashes de dataset y reparto. La salida
confirma `Pesos iniciales idénticos al experimento 04 guardado: True`.

## 2. Resultados frente al experimento 04

Las métricas finales corresponden al checkpoint de mayor accuracy, evaluado en
inferencia. Las medias macro incluyen las 13 clases.

| Métrica | Experimento 04 | Experimento 05 |
|---|---:|---:|
| Accuracy de validación (%) | 77,76 | 78,40 |
| Precision macro (%) | 80,38 | 81,38 |
| Recall macro (%) | 78,57 | 78,25 |
| F1 macro (%) | 79,36 | 79,60 |
| AP macro (%) | 84,48 | 84,65 |
| Aciertos / validación | 1458 / 1.875 | 1470 / 1.875 |
| Accuracy train en inferencia (%) | 96,81 | 93,15 |
| Brecha train−validación en inferencia (pp) | 19,05 | 14,75 |
| CE de validación sin L2 | 0,843 | 0,709 |
| Pérdida total de validación | 0,843 | 0,756 |
| Mejor época / ejecutadas | 69 / 89 | 56 / 76 |
| Fit completo (min) | 64,18 | 63,99 |
| Mediana de segundos por época | 42,78 | 50,59 |

La accuracy mejora **0,64 puntos**, de 1.458 a **1.470 aciertos**. La precision
macro sube 0,99 puntos y el F1 macro, 0,24. El recall macro baja 0,32 puntos:
la mejora no es uniforme entre clases ni métricas.

La accuracy de train baja de 96,81 % a 93,15 %, al tiempo que aumenta ligeramente
la de validación. La brecha se reduce **4,30 puntos**. Este patrón es compatible
con un mejor equilibrio entre ajuste y generalización, aunque persiste sobreajuste.

**La comparación de pérdida usa CE sin L2.** En el checkpoint del experimento 05,
la pérdida total es 0,756 y la CE es 0,709: aproximadamente **0,0464** corresponden
a la penalización. No deben confundirse ambas magnitudes ni atribuir la mejora a
una calibración demostrada; no se ha medido una métrica explícita de calibración.

## 3. Curvas, oscilaciones y selección del checkpoint

Las [curvas de entrenamiento](plots/training_curves.pdf) muestran oscilaciones
al principio, pero se estabilizan antes que en el experimento 04. Train continúa
mejorando y validación se mantiene al final alrededor del 78 %. El resultado no
es únicamente un máximo aislado: las últimas épocas oscilan aproximadamente
entre 77,9 % y 78,35 %.

Como descripción de la variabilidad, se calcula el **salto medio absoluto de
accuracy entre épocas consecutivas**, en puntos porcentuales, sobre las mismas
ventanas de historial:

| Épocas, inclusive | Experimento 04 (pp) | Experimento 05 (pp) |
|---|---:|---:|
| 1–25 | 5,66 | 3,34 |
| 26–50 | 2,56 | 1,02 |
| 51–76 | 0,80 | 0,37 |

La medida es `100 * mean(abs(diff(val_accuracy)))` dentro de cada ventana.
Describe los historiales observados, incluye la tendencia de aprendizaje y no
es un test estadístico ni aísla el efecto de cada modificación. En las tres
ventanas el salto medio es menor en el experimento 05; no puede concluirse que
las oscilaciones anteriores fueran causadas exclusivamente por BN.

El mejor checkpoint aparece en la **época 56** y el entrenamiento termina en la
**76**, tras 20 épocas sin superar su accuracy. En la época 56, `fit` registra
89,55 % de train con Dropout, aumentación y BN activos. Al evaluar ese checkpoint
en inferencia se obtiene 93,15 %. La brecha final se calcula con train y validación
en inferencia, no restando directamente las curvas de modos distintos.

La **CE mínima de validación** se registra en la **época 60**, con **0,699** y
accuracy **78,19 %**. El checkpoint seleccionado sigue siendo el de la 56, con
accuracy 78,40 % y CE 0,709. La diferencia entre mínimos y máximos es pequeña,
pero responde a criterios distintos. El mínimo de CE se conoce por el historial;
no implica que su checkpoint intermedio se haya guardado.

## 4. Reducción del LR y rapidez de aprendizaje

Los monitores están separados:

- `monitor='val_accuracy'`, modo max: checkpoint, EarlyStopping y mejor época.
- `lr_monitor='val_ce'`, modo min: ReduceLROnPlateau, paciencia 5, factor 0.5,
  `min_delta=0` y mínimo 1e-7.

Una mejora de accuracy no reinicia la paciencia del LR si la CE no mejora. La
selección del modelo sigue respondiendo a accuracy aunque mejore la CE.

El [gráfico de LR y tiempos](plots/learning_rate_time.pdf) y `lr_used` registran
el LR utilizado al inicio de cada época. Las nuevas etapas empiezan así:

| Primera época con ese LR | LR utilizado |
|---|---:|
| 1 | 0.0003 |
| 26 | 0.00015 |
| 35 | 7.5e-05 |
| 44 | 3.75e-05 |
| 51 | 1.875e-05 |
| 56 | 9.375e-06 |
| 66 | 4.6875e-06 |
| 71 | 2.34375e-06 |
| 76 | 1.17188e-06 |

La primera reducción se utiliza en la época 26, frente a la 53 del experimento 04.
El mejor checkpoint utiliza aproximadamente **9,375e-6**. El mínimo permitido de
1e-7 no llega a alcanzarse durante las épocas ejecutadas.

| Umbral de validación | Exp. 04: primera época / tiempo acumulado | Exp. 05: primera época / tiempo acumulado |
|---|---|---|
| 70 % | 26 / 19,48 min | 27 / 22,29 min |
| 75 % | 53 / 38,76 min | 35 / 29,00 min |
| 77 % | 55 / 40,18 min | 40 / 33,24 min |
| 78 % | No alcanzado | 56 / 46,76 min |
| 80 % | No alcanzado | No alcanzado |

Los tiempos acumulados suman `epoch_seconds` y no incluyen todos los costes de
callbacks; se registra el primer cruce, no la permanencia sobre el umbral.
El 05 llega antes a 75 % y 77 %, aunque no cruza antes el 70 %.

**Menos épocas no han supuesto un fit completo apreciablemente más corto:**
76 frente a 89 épocas, pero unos **63,99 frente a 64,18 minutos**. La mediana por
época observada sube de 42,78 a 50,59 segundos. No se ha aislado el coste añadido
de L2 de posibles diferencias en el estado del sistema; no se atribuye toda esa
variación a la penalización ni se presenta como benchmark controlado.

## 5. Cambios por clase

La [matriz de confusión](plots/confusion_matrix.pdf), las
[curvas precision–recall](plots/precision_recall.pdf) y las
[métricas por clase](artifacts/class_metrics.csv) muestran el siguiente recall:

| Clase | Recall exp. 04 (%) | Recall exp. 05 (%) | Aciertos 04 → 05 / soporte |
|---|---:|---:|---|
| Cargo plane | 96,61 | 91,53 | 57 → 54 / 59 |
| Small car | 88,75 | 91,19 | 292 → 300 / 329 |
| Bus | 63,22 | 70,11 | 110 → 122 / 174 |
| Truck | 49,79 | 53,53 | 120 → 129 / 241 |
| Motorboat | 68,81 | 68,81 | 75 → 75 / 109 |
| Fishing vessel | 80,82 | 83,56 | 59 → 61 / 73 |
| Dump truck | 69,49 | 66,10 | 82 → 78 / 118 |
| Excavator | 86,42 | 83,95 | 70 → 68 / 81 |
| Building | 89,26 | 88,98 | 324 → 323 / 363 |
| Helipad | 77,78 | 77,78 | 7 → 7 / 9 |
| Storage tank | 82,24 | 80,26 | 125 → 122 / 152 |
| Shipping container | 80,74 | 77,04 | 109 → 104 / 135 |
| Pylon | 87,50 | 84,38 | 28 → 27 / 32 |

- **Bus** mejora 6,90 puntos de recall —12 aciertos más—. De sus 174 ejemplos,
  122 se recuperan y 25 se confunden con Truck, frente a 39 en el experimento 04.
- **Truck** mejora 3,73 puntos —nueve aciertos—. Recupera 129/241, pero sigue
  confundiendo 40 con Small car, 27 con Bus y 24 con Dump truck. La precision sube
  de 51,50 % a 56,83 % y AP de 0,535 a 0,596; sigue siendo una dificultad central.
- **Small car** gana ocho aciertos y **Fishing vessel**, dos.
- Hay pérdidas en Cargo plane, Dump truck, Excavator, Building, Storage tank,
  Shipping container y Pylon. Esto explica que el recall macro no mejore aunque
  aumente la accuracy global.
- **Helipad** conserva siete aciertos de nueve y alcanza precision 100 %, sin
  falsos positivos. Su soporte reducido exige interpretar los porcentajes con
  cautela; AP no equivale a la precision del argmax.

El avance agregado es modesto y se concentra en parte en clases frecuentes y
vehículos difíciles. La configuración no mejora de manera uniforme las 13 clases.

## 6. Conclusiones

La hipótesis queda **parcialmente apoyada** en esta ejecución: se observa una
validación menos errática, menor CE y menor brecha, junto a una mejora pequeña
de accuracy/F1. Se conserva el tamaño y se parte de los mismos pesos iniciales.
No se puede separar cuánto aporta el LR inicial, el nuevo scheduler o L2.

El experimento 05 pasa a ser la referencia propia con mayor accuracy observada,
**78,40 %**. La diferencia de 12 aciertos frente al 04, sobre una sola semilla y
partición utilizada para ajustar hiperparámetros, no demuestra una superioridad
general. El 80 % requiere **30 aciertos adicionales** en esta validación; el
85 %, **124**. No se han alcanzado esos objetivos ni se ha preparado otro ensayo
en esta actualización.

Los errores de vehículos y las pérdidas de recall en otras clases permiten
priorizar los siguientes cambios. No basta con suavizar las curvas o reducir
más la accuracy de train: la decisión deberá apoyarse en validación, métricas
macro y análisis por clase, manteniendo las semillas acordadas.

## 7. Archivos conservados y comparación opcional

`model.keras` y `prediction.json` están junto al notebook; los auxiliares están
en `artifacts/` y las cuatro figuras PDF en `plots/`. El JSON generado no aporta
una puntuación de test: no hay resultados de Codabench documentados aquí.

`SELECT_RUNS` contiene solo los experimentos propios **04 y 05**. Las llamadas
finales del apartado 9 **siguen comentadas** y no se ha generado la carpeta de
comparación de este par. Las tablas de este README se han calculado directamente
a partir de los artefactos guardados, sin entrenar ni modificar los modelos.

Para generar la comparación, descomentar y ejecutar el apartado 9:

```python
runs = load_runs(CHALLENGE_DIR, SELECT_RUNS)
compare_runs(runs, COMPARISON_DIR)
```

Si se ha reiniciado el kernel, ejecutar primero la celda inicial de imports y
utilidades. Los archivos se guardarán en
`PROJECT/cnn_challenge/comparisons/experiment_05_vs_04/`, sin populares.
Reentrenar en esta carpeta sobrescribe los resultados del 05; usar otra carpeta
para conservarlos al preparar la siguiente prueba.
