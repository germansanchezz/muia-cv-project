# CNN propia · experiment02

Ejecutar [cnn_experiment02.ipynb](cnn_experiment02.ipynb) de arriba abajo; las rutas ya apuntan a esta carpeta.

Se copia la arquitectura residual compacta SE de experiment01 sin cambios (2.857.125 parámetros). **Único cambio de hiperparámetro: label smoothing 0.0 → 0.1.** Se mantienen entrada 128×128, batch 32, AdamW, LR 0.001, weight decay 0.0001, máximo 200 épocas, semilla 42, reparto, aumentación y callbacks. No se añade Dropout ni L2.

Se entrena desde cero y se reutiliza la caché de 128×128 mediante mmap. Se guardan el mejor `model.keras`, obligatoriamente `prediction.json`, los auxiliares en `artifacts/` y los PDF en `plots/`.

La sección 9 compara experiment02 con **custom experiment01**, la mejor custom anterior, y **ResNet18 experiment04**, referencia popular fija. El destino es `cnn_challenge_alt/comparisons/custom_experiment02/`. Para regenerarla desde artefactos, bastan la primera celda de código y la sección 9, con las tres ejecuciones completas.

Evaluar cambios de accuracy y F1 macro, CE sin suavizado, AP, brecha y errores de alta confianza. `loss` incluye label smoothing y no debe compararse como el mismo objetivo que la pérdida de experiment01. La referencia popular usa una resolución diferente (192×192).

Estado: **completado**. Mejor checkpoint en la época **66**, de **86** ejecutadas:

- Accuracy de validación: **79,04 %**, 1.482/1.875 aciertos (dos menos que experiment01).
- F1 macro: **80,73 %**, +0,54 puntos; la mejora depende en gran medida de Helipad, con nueve positivos.
- CE sin suavizado: **0,800** (antes 0,809); AP macro: **83,47 %** (antes 86,21 %).
- Train en inferencia: **99,21 %**; brecha: **20,17 puntos** (antes 17,53).
- Errores con confianza superior al 99 %: **0**, frente a 31 en experiment01.
- Entrenamiento: **29,88 minutos**, frente a 21,60.

Conclusión: el suavizado reduce errores extremadamente confiados y mejora el F1 macro global, pero no supone una mejora general de accuracy, AP o brecha. **Experiment01 sigue siendo la referencia custom por accuracy.** El [registro general](../README.md) analiza las clases y documenta la elección de probar MixUp alpha 0.2 en experiment03 sobre experiment01, a 128×128 y sin label smoothing; la resolución se reserva para una posible prueba final. [Métricas guardadas](artifacts/metrics.json).
