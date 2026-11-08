# CNN propia · experiment01

Ejecutar [cnn_experiment01.ipynb](cnn_experiment01.ipynb) de arriba abajo. Las rutas ya apuntan a esta carpeta.

La arquitectura es una CNN residual compacta con ocho bloques SE y canales 32/64/128/256. El protocolo de entrenamiento es el inicial de las populares: 128×128, batch 32, AdamW, LR 0.001, weight decay 0.0001, label smoothing 0, sin Dropout y hasta 200 épocas. Se entrena desde cero.

El [registro de arquitecturas propias](../README.md) describe la estructura, la hipótesis y las comparaciones preparadas. La caché de 128×128 se comparte en `cnn_challenge_alt/.cache/`. El mejor modelo y `prediction.json` están guardados aquí, junto con `artifacts/` y `plots/`.

Estado: **completado**. Mejor checkpoint en la época **44**, de **64** ejecutadas:

- Accuracy de validación: **79,15 %** (1.484/1.875 aciertos).
- F1 macro: **80,18 %**; CE: **0,809**; AP macro: **86,21 %**.
- Accuracy de train en inferencia: **96,67 %**; brecha: **17,53 puntos**.
- Parámetros: **2.857.125**; entrenamiento: **21,60 minutos**.

Frente a ResNet18 experiment04, queda 0,96 puntos por debajo en accuracy y 2,06 en F1 macro, con aproximadamente un 74 % menos de parámetros. La comparación utiliza protocolos distintos: la custom tiene entrada 128×128 sin label smoothing y la popular usa 192×192 con label smoothing 0.1.

La sección 9 selecciona únicamente esta ejecución y ResNet18 experiment04. El [análisis completo](../README.md) documenta los resultados, sus límites y la prueba de label smoothing 0.1 como único cambio en [experiment02](../experiment02/cnn_experiment02.ipynb), ya completado, con resultados analizados en el registro general. [Métricas guardadas](artifacts/metrics.json).
