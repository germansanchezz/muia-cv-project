# Experimento propio 01 · CNN compacta

[Notebook ejecutado](cnn_experiment_01.ipynb) ·
[Configuración](artifacts/config.json) · [Métricas](artifacts/metrics.json) ·
[Comparación con Round 1](../../comparisons/experiment_01_vs_round1/comparison.csv)

**Resultado principal:** una CNN de **288.685 parámetros** alcanza **70,13 % de
accuracy de validación y 70,17 % de F1 macro**, con **7,66 puntos de brecha
train/validación**. Es una referencia competitiva en relación con su tamaño,
aunque queda 4,21 puntos por debajo de DenseNet121 y aprende más lentamente en
número de épocas. La hipótesis de obtener una base útil con una red pequeña
queda apoyada en esta ejecución; la capacidad óptima sigue pendiente de explorar.

## 1. Hipótesis y diseño

Comprobar si una CNN compacta puede acercarse a las arquitecturas populares con
muchos menos parámetros y un coste reducido por época. El diseño procede de la
arquitectura custom del baseline, sin modificaciones:

```text
RGB 64×64×3
→ [Conv3×3(32) → ReLU] ×2 → MaxPool2×2
→ [Conv3×3(64) → ReLU] ×2 → MaxPool2×2
→ [Conv3×3(128) → ReLU] ×2 → MaxPool2×2
→ GAP → Dropout(0.1) → Dense(13, Softmax)
```

Los bloques reducen los mapas a 32×32×32, 16×16×64 y 8×8×128. Las convoluciones
usan stride 1, padding SAME e inicialización He. GAP produce un vector de 128
valores y la última Dense añade solo 1.677 parámetros.

La profundidad, la anchura y el uso de GAP son decisiones iniciales de diseño.
GAP y los poolings limitan el clasificador, pero también pueden perder detalles
o información de posición útil. Una red pequeña no garantiza mejor generalización.

## 2. Protocolo de la ejecución

| Parámetro | Valor |
|---|---|
| Entrada | RGB 64×64, reducción por promedio, float32 en [0,1] |
| Reparto | Histórico 90/10: 16.871 train y 1.875 validación, semilla 1, sin estratificación |
| Inicialización | Desde cero; semilla 42, operaciones deterministas |
| Optimizador | SGD, LR inicial 1e-3, momentum 0.9, Nesterov y clipnorm 1 |
| Batch / máximo de épocas | 32 / 200 |
| Regularización | Dropout 0.1; sin BN, L2 ni aumentación |
| Checkpoint | Primer máximo de `val_accuracy` |
| Reducción de LR | Factor 0.2 tras 10 épocas sin mejora; mínimo 1e-7 |
| EarlyStopping | 20 épocas sin mejora en `val_accuracy` |
| Entrada de datos | Caché compartida, modo `mmap` |
| Entorno | RTX 4070 Ti, TensorFlow 2.15.1, Python 3.10.21 |

Se conserva el mismo dataset, reparto, resolución, batch y semilla que en las
referencias seleccionadas de Round 1. **El optimizador es distinto: nuestra CNN
usa SGD y las populares Adam.** También difieren profundidad, conexiones y capas
de normalización internas. Se comparan configuraciones completas; no se aísla
el efecto del número de parámetros ni el de la arquitectura.

## 3. Resultado y evolución

| Métrica | Valor |
|---|---:|
| Accuracy de validación | 70,13 % — 1.315 aciertos de 1.875 |
| Precision macro | 73,36 % |
| Recall macro | 68,41 % |
| F1 macro | 70,17 % |
| AP macro | 74,50 % |
| Accuracy de train, en inferencia | 77,80 % |
| Brecha train−validación, en inferencia | 7,66 pp |
| CE de validación del checkpoint | 0,942 |
| Mejor época / épocas ejecutadas | 142 / 162 |
| Tiempo total de fit | 819,31 s — 13,66 min |

Las [curvas de entrenamiento](plots/training_curves.pdf) muestran una mejora
prolongada y progresiva, con oscilaciones de validación especialmente al principio.
La validación supera el 60 % alrededor de la parte media del entrenamiento y se
estabiliza cerca del 70 % al final. No aparece el crecimiento sostenido de la CE
de validación que se observa en las populares seleccionadas.

Los LR utilizados pasan de 1e-3 a 2e-4 en la época 128, a 4e-5 en la 141 y a 8e-6
en la 153. El checkpoint de mayor accuracy aparece en la 142; la parada en la
162 corresponde a las 20 épocas de paciencia. La CE mínima se observa en la
144, con **0,940**, muy próxima a la del checkpoint seleccionado. Puede consultarse
el [historial](artifacts/history.csv) y la [figura de LR y tiempos](plots/learning_rate_time.pdf).

En la época seleccionada, `fit` registra 74,27 % de accuracy en train, con
Dropout activo. La evaluación del mismo checkpoint sin Dropout alcanza 77,80 %.
La brecha de 7,66 puntos se calcula con ambos conjuntos en inferencia, no restando
las curvas medidas en modos distintos.

## 4. Comparación con las referencias populares

Las figuras existentes incluyen la CNN propia y las referencias mejor, intermedia
y peor por accuracy de Round 1: DenseNet121, ResNet18 y GoogLeNet. VGG16-a queda
fuera del ranking porque su entrenamiento colapsó.

| Modelo | Accuracy validación (%) | F1 macro (%) | Parámetros (M) | Fit total (min) | Mediana s/época |
|---|---:|---:|---:|---:|---:|
| CNN propia | 70,13 | 70,17 | 0,289 | 13,66 | 4,98 |
| DenseNet121 | 74,35 | 75,31 | 7,051 | 62,71 | 41,81 |
| ResNet18 | 71,89 | 73,18 | 11,193 | 12,04 | 9,64 |
| GoogLeNet | 65,12 | 63,33 | 5,987 | 14,60 | 15,25 |

Las [métricas comparadas](../../comparisons/experiment_01_vs_round1/comparison_metrics.pdf)
y la [figura de coste](../../comparisons/experiment_01_vs_round1/comparison_cost.pdf)
permiten concretar el compromiso:

- Frente a **DenseNet121**, la CNN propia usa **24,4 veces menos parámetros** y
  tarda **4,6 veces menos en el fit completo**, con 4,21 puntos menos de accuracy
  —79 aciertos menos en validación—.
- Frente a **ResNet18**, usa **38,8 veces menos parámetros** y pierde 1,76 puntos
  —33 aciertos—, pero su fit completo tarda algo más: 13,66 frente a 12,04 minutos.
- Frente a **GoogLeNet**, usa **20,7 veces menos parámetros** y gana 5,01 puntos
  —94 aciertos—. El tiempo total es cercano: 13,66 frente a 14,60 minutos.

**La ventaja de velocidad es clara por época; no es igual de grande en el tiempo
total.** La CNN necesita 162 épocas, frente a 56–89 de estas populares. Una época
más barata y una convergencia más lenta se compensan en parte. Los tiempos incluyen
validación y callbacks; no constituyen un benchmark aislado del cálculo de la red.

### Ajuste de train y generalización

| Modelo | Primera época con train ≥99 % durante `fit` | Train en inferencia, mejor checkpoint (%) | Brecha de inferencia (pp) | CE validación, mejor checkpoint |
|---|---:|---:|---:|---:|
| CNN propia | No alcanzado | 77,80 | 7,66 | 0,942 |
| DenseNet121 | 32 | 99,99 | 25,65 | 1,729 |
| ResNet18 | 23 | 99,99 | 28,10 | 2,001 |
| GoogLeNet | 37 | 99,23 | 34,11 | 2,043 |

Las [curvas superpuestas](../../comparisons/experiment_01_vs_round1/comparison_curves.pdf)
confirman que las populares ajustan train mucho antes. La mayor capacidad puede
facilitar ese ajuste, pero **estas ejecuciones no demuestran que sea la única
causa**: cambian también SGD por Adam y el diseño de los bloques, conexiones y
normalización. Alcanzar casi el 100 % de train demuestra un ajuste muy alto; por
sí solo no identifica qué parte corresponde a memorización frente a características
útiles para generalizar.

Nuestra CNN obtiene una accuracy de validación cercana a ResNet18 con una brecha
mucho menor y la menor CE de las cuatro configuraciones seleccionadas. Esto apoya
su utilidad como referencia compacta. **Una brecha pequeña no basta para afirmar
que sea el mejor modelo:** DenseNet121 obtiene más aciertos y mayor F1 macro.
Tampoco la menor CE demuestra por sí sola una mejor calibración; esta no se ha
medido explícitamente.

La accuracy de train de 77,80 % deja margen para investigar optimización y capacidad.
No permite decidir todavía si el límite procede de SGD y su LR, de la arquitectura,
de la resolución o de su combinación. Aumentar regularización o filtros sin aislar
estos factores haría más difícil interpretar el siguiente resultado.

## 5. Errores por clase y oportunidades de mejora

La [matriz de confusión](plots/confusion_matrix.pdf), las
[curvas precision–recall](plots/precision_recall.pdf) y el
[recall comparado](../../comparisons/experiment_01_vs_round1/comparison_class_recall.pdf)
señalan objetivos concretos:

- **Small car:** recall 89,36 %. **Cargo plane:** 84,75 %, con AP 0,946.
  Son clases con resultados fuertes en esta ejecución.
- **Truck:** recall 39,83 % —96/241—. Se confunde especialmente con Small car
  —45—, Bus —32— y Dump truck —26—. Su AP es 0,447: la dificultad también aparece
  al examinar las probabilidades, no solo al aplicar argmax.
- **Storage tank:** recall 60,53 % —92/152—; 31 ejemplos se predicen como Building.
  DenseNet121 y ResNet18 recuperan 115 y 119 ejemplos, respectivamente.
- **Bus:** recall 57,47 % y AP 0,573; sigue siendo una clase con margen de mejora.
- **Fishing vessel:** recall 72,60 %, superior al observado en las tres referencias
  seleccionadas. La clasificación global no resume todas las diferencias por clase.
- **Helipad:** cinco aciertos de nueve, recall 55,56 % y precision 100 %. No hay
  falsos positivos de esa clase, pero se pierden cuatro ejemplos. Su soporte es
  demasiado pequeño para interpretar un porcentaje alto como un resultado estable.

Las confusiones sugieren investigar cómo representar detalles y formas de objetos
parecidos, además de mejorar la accuracy agregada. No prueban todavía que subir
la resolución o añadir una capa concreta vaya a resolver esos errores.

## 6. Qué probar en el siguiente experimento

### Primera propuesta: aislar el efecto del optimizador

Para `experiment_02`, mantener la misma CNN y cambiar **solo SGD por Adam** con LR
inicial 1e-3, usando la implementación ya disponible en el notebook —AMSGrad y
clipnorm 1—. Conservar los pesos iniciales mediante la misma semilla, resolución,
batch, Dropout, datos, criterio de selección y callbacks; comprobar el hash de los
pesos iniciales antes de entrenar.

La pregunta sería: **¿una optimización distinta permite ajustar mejor train y
alcanzar antes una validación igual o superior, sin aumentar parámetros?**
Adam utiliza estimaciones adaptativas de momentos, por lo que compartir el valor
numérico del LR no implica compartir la misma dinámica de actualización
([documentación de Keras](https://keras.io/api/optimizers/adam/)).
La mejora es una hipótesis, no una garantía; también podría aumentar el sobreajuste
u ocasionar inestabilidad y exigir un ajuste posterior de LR.

Además de la accuracy máxima, registrar el tiempo o época para alcanzar un umbral
común de validación, por ejemplo 65 % o 70 %, y el fit completo. Esto permitiría
separar velocidad de aprendizaje, coste por época y rendimiento final.

### Cambios posteriores, condicionados al resultado

1. **BN como experimento independiente:** valorar si estabiliza el entrenamiento
   y ayuda a aprender representaciones más útiles, conservando los demás factores.
2. **Capacidad o conservación de información espacial:** si persiste un ajuste
   bajo de train con optimización funcional, probar una ampliación moderada de
   filtros o modificar un pooling, una decisión por experimento.
3. **Regularización o aumentación:** si al mejorar train crece la brecha, probar
   Dropout, L2 o volteos por separado. Un menor sobreajuste aparente en este primer
   ensayo no justifica aumentar automáticamente Dropout.
4. **Resolución:** explorar 128×128 en una prueba separada si los errores de detalle
   lo justifican, midiendo también el incremento de coste.

**Referencias para los siguientes experimentos:** comparar cada nueva CNN propia
solo con **DenseNet121 —mejor de Round 1— y GoogLeNet —peor—**, además de la propia
CNN evaluada. La comparación histórica del experimento 01 conserva ResNet18;
no se reescriben sus resultados para adaptarlos a la selección futura.

Estas propuestas están documentadas, pero el experimento 02 todavía no se ha creado
ni entrenado como parte de esta actualización.

## 7. Resultados conservados y límites de interpretación

El modelo y `prediction.json` se guardan junto al notebook; configuración, historia,
métricas y probabilidades están en `artifacts/`, y las figuras individuales en
`plots/`. Las comparaciones existentes están en
[`comparisons/experiment_01_vs_round1/`](../../comparisons/experiment_01_vs_round1/).

Las llamadas finales de comparación de este notebook están actualmente activas.
Para regenerar únicamente esas figuras, ejecutar la primera celda de código y el
apartado 9. Para repetir un entrenamiento conservando este resultado, crear una
carpeta nueva y actualizar `NOTEBOOK_PATH` e `experiment_id`.

Las conclusiones se limitan a una semilla y una partición usada también para ajustar
hiperparámetros y seleccionar el checkpoint. No se ha estimado variabilidad entre
semillas ni significación estadística. El tiempo de inferencia guardado incluye el
pipeline y no es una medida aislada de latencia. Los JSON de test generados no
aportan por sí mismos puntuaciones de Codabench.
