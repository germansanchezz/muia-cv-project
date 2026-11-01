# Experimento propio 03 · Resolución 128×128

El [notebook](cnn_experiment_03.ipynb) parte del
[experimento 02](../experiment_02/README.md) y cambia **únicamente la resolución
de entrada: 64×64 → 128×128**. La ejecución ya está completada.

Resultado: **70,83 % de accuracy**, **72,14 % de F1 macro**, mejor época **62**
y parada en la **82**. Train alcanza **99,84 %** en inferencia, con una brecha de
**29,01 puntos**. El fit tarda **13,64 minutos**, frente a 4,87 del experimento 02.
Los hashes iniciales y del reparto coinciden. Subir la resolución sin modificar
la arquitectura no mejoró esta configuración: son **58 aciertos menos** que a 64×64.

El [experimento 04](../experiment_04/README.md) prueba una arquitectura ampliada
a 224×224 con BN, volteos y Dropout 0.2. Se mantienen las semillas. Desde ese
experimento se dejan de utilizar referencias populares y se prioriza mejorar
la CNN propia. La selección de comparación del experimento 03 se conserva como
configuración histórica, con sus llamadas desactivadas.

Se mantienen la semilla de entrenamiento **42** y la del reparto **1**.
No se activa aumentación ni se modifica el optimizador.

## Hipótesis

Conservar más detalle de los recortes originales puede ayudar a distinguir clases
parecidas, como Truck, Bus y Storage tank. Se contrastará con el experimento 02,
que obtuvo 73,92 % de accuracy y 75,04 % de F1 macro a 64×64.

La mejora no está garantizada: con los mismos bloques y kernels, cada característica
local cubre una fracción menor de la imagen. La evaluación debe considerar también
la brecha de generalización, los errores por clase y el incremento de coste.

## Configuración

| Parámetro | Valor |
|---|---|
| Entrada — único cambio | RGB 128×128, reducción por promedio, float32 en [0,1] |
| Semillas | Entrenamiento 42; reparto 1 |
| Reparto | Histórico 90/10, 16.871 train y 1.875 validación, sin estratificación |
| Arquitectura | Seis Conv3×3, filtros 32/64/128, tres MaxPool, GAP y Dense13 |
| Parámetros | 288.685, igual que en los experimentos 01 y 02 |
| Optimizador | Adam, LR 1e-3, AMSGrad, beta1=0.9, beta2=0.999, epsilon=1e-8 y clipnorm 1 |
| Batch | 32, sin cambios |
| Regularización | Dropout 0.1; sin BN, L2 ni aumentación |
| Máximo de épocas / monitor | 200 / `val_accuracy` |
| Reducción de LR / parada | Factor 0.2, paciencia 10 y mínimo 1e-7 / paciencia 20 |

Tras los poolings, los mapas son 64×64×32, 32×32×64 y 16×16×128. GAP sigue
produciendo 128 valores, por lo que las formas de los pesos y el número de
parámetros permanecen iguales. Se entrena desde cero; no se cargan los pesos
entrenados del experimento 02.

## Caché y coste

Se utiliza la caché compartida en modo `mmap`. La resolución forma parte de su
clave: la primera preparación crea un archivo nuevo para 128×128 y conserva el
de 64×64. El archivo float32 ocupa aproximadamente **3,43 GiB en disco** y se
escribe imagen a imagen; no se precarga explícitamente entero en RAM.

Hay cuatro veces más píxeles por imagen. Aumentan el cálculo convolucional y la
memoria de activaciones, aunque los parámetros no cambian. El incremento real de
tiempo debe medirse: no tiene por qué ser exactamente cuatro veces. El batch se
mantiene en 32 para aislar el cambio de resolución.

## Ejecución y evaluación

`NOTEBOOK_PATH` ya apunta a esta copia. «Ejecutar todo» genera `model.keras` y
siempre `prediction.json` junto al notebook, auxiliares en `artifacts/` y figuras
PDF en `plots/`. La comprobación del hash inicial se hace frente al experimento
02; revisar también los hashes de dataset y reparto al analizar los resultados.

Registrar accuracy, F1/recall macro, CE, brecha en inferencia, mejor época,
épocas ejecutadas, fit completo y segundos por época. Contrastar principalmente
con el experimento 02, que mantiene la misma configuración a 64×64.

## Comparaciones preparadas

`SELECT_RUNS` contiene solo **este experimento, DenseNet121 y GoogLeNet**.
Las llamadas finales están **comentadas** para ejecutar el entrenamiento completo
sin comparar. Al terminar, pueden activarse en el apartado 9:

```python
runs = load_runs(CHALLENGE_DIR, SELECT_RUNS)
compare_runs(runs, COMPARISON_DIR)
```

Tras reiniciar el kernel, ejecutar antes la primera celda de código. Los archivos
se guardarán en `PROJECT/cnn_challenge/comparisons/experiment_03_vs_round1/`.
Las populares seleccionadas usan 64×64: la comparación con ellas incluye una
diferencia de resolución y no permite atribuir el resultado solo a la arquitectura.
