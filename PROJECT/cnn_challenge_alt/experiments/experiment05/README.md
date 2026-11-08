# CNN propia · experiment05 · 192×192 con MixUp alpha 0.2

Ejecutar [cnn_experiment05.ipynb](cnn_experiment05.ipynb) de arriba abajo; las rutas están preparadas.

Se parte de **custom experiment03**, elegida frente a experiment04 porque MixUp 0.2 conserva mejor accuracy y F1 que 0.4. **Único cambio frente a experiment03: entrada 128×128 → 192×192.**

Se mantienen arquitectura `residual_se_v1` (2.857.125 parámetros), MixUp alpha **0.2**, label smoothing **0.0**, batch 32, AdamW, LR 0.001, weight decay 0.0001, máximo 200 épocas, semilla 42, reparto, aumentación geométrica y callbacks. Sin Dropout adicional ni L2. Los mapas finales pasan de 8×8 a 12×12 antes de GAP.

Entrenamiento desde cero. Se crea o reutiliza la caché mmap de 192×192 conservando la de 128×128. MixUp solo afecta a train; evaluación de train, validación y test usan originales. La brecha final no usa la accuracy mezclada de fit.

La sección 9 selecciona custom experiment05, custom experiment03 y ResNet18 experiment04. Destino: `comparisons/custom_experiment05/`. El mejor modelo, `prediction.json` obligatorio, artefactos y PDF se guardan aquí.

La referencia tiene **80,69 % de accuracy y 82,27 % de F1 macro**. Se comprobará si 192×192 mejora validación y si compensa el coste. Estado: **preparado; entrenamiento pendiente**. Detalles y elección de alpha en el [registro general](../README.md).
