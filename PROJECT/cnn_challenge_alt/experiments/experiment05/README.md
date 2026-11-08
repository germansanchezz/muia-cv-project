# CNN propia · experiment05 · 192×192 con MixUp alpha 0.2

Ejecutar [cnn_experiment05.ipynb](cnn_experiment05.ipynb) de arriba abajo; las rutas están preparadas.

Se parte de **custom experiment03**, elegida frente a experiment04 porque MixUp 0.2 conserva mejor accuracy y F1 que 0.4. **Único cambio frente a experiment03: entrada 128×128 → 192×192.**

Se mantienen arquitectura `residual_se_v1` (2.857.125 parámetros), MixUp alpha **0.2**, label smoothing **0.0**, batch 32, AdamW, LR 0.001, weight decay 0.0001, máximo 200 épocas, semilla 42, reparto, aumentación geométrica y callbacks. Sin Dropout adicional ni L2. Los mapas finales pasan de 8×8 a 12×12 antes de GAP.

Entrenamiento desde cero. Se crea o reutiliza la caché mmap de 192×192 conservando la de 128×128. MixUp solo afecta a train; evaluación de train, validación y test usan originales. La brecha final no usa la accuracy mezclada de fit.

La sección 9 selecciona custom experiment05, custom experiment03 y ResNet18 experiment04. Destino: `comparisons/custom_experiment05/`. El mejor modelo, `prediction.json` obligatorio, artefactos y PDF se guardan aquí.

Estado: **completado**. Mejor checkpoint en la época **95**, de **115** ejecutadas:

- Accuracy de validación: **79,47 %** (1.490/1.875 aciertos), **23 aciertos menos** que experiment03.
- F1 macro: **81,02 %**, frente a 82,27 %; CE: **0,704**, frente a 0,695.
- AP macro: **85,82 %**, frente a 86,38 %.
- Train original en inferencia: **97,41 %**; **brecha: 17,94 puntos**, frente a 17,61.
- Entrenamiento: **58,57 minutos**, frente a 39,18.

**La mayor resolución no compensa en esta ejecución. Se conserva custom experiment03 a 128×128 y MixUp 0.2 como mejor modelo individual.** Bus y Dump truck explican buena parte de la pérdida de aciertos. La media 50/50 de probabilidades de ambas custom sí mejora de forma exploratoria en validación, pero exige dos modelos y no representa un entrenamiento individual mejor.

[Métricas guardadas](artifacts/metrics.json). Análisis completo, combinación de probabilidades y posibles siguientes pruebas en el [registro general](../README.md). No se ha preparado otro experimento.
