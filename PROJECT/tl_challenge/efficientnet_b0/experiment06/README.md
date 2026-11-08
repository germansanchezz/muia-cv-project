# EfficientNetB0 · transfer learning · experiment06

[Notebook](efficientnet_b0_tl_experiment06.ipynb). **Preparado; entrenamiento pendiente.** Reiniciar el kernel y ejecutar todas las celdas en orden.

**Único cambio frente a experiment05: LR inicial `3e-5` en lugar de `1e-5`.** Se prueba si una adaptación más rápida mejora el resultado; las curvas anteriores no demuestran que el LR fuese insuficiente.

Se mantienen 192×192, batch 32, AdamW nuevo, weight decay 1e-4, MixUp alpha=0.2, Categorical Cross Entropy, sin label smoothing ni Dropout adicional, todo el backbone entrenable salvo BatchNorm, semilla 42, reparto y aumentación geométrica. Se conservan ModelCheckpoint y EarlyStopping sobre val_accuracy, y ReduceLROnPlateau con paciencia 10, factor 0.5 y mínimo 1e-6; EarlyStopping tiene paciencia 20. Máximo 180 épocas adicionales, 200 contando la cabeza en esta rama.

Se parte del mejor **experiment01**, no de experiment05. Se adapta la entrada y el backbone del checkpoint de 128 a 192, copiando todos sus pesos, incluida la cabeza. Conservar su `model.keras` y `artifacts/`; no repetir la cabeza. La caché mmap de 192 de TL se reutiliza.

Antes del fit se evalúa el extractor adaptado a 192 para fijar el umbral del checkpoint. Si no se mejora se conserva ese modelo adaptado (`feature_extraction_resized`), con métricas recalculadas a 192. Las BatchNorm permanecen congeladas mediante la implementación explícita compatible con GPU determinista. La brecha se calcula sobre train y validación sin MixUp ni aumentación.

Referencia principal: experiment05 obtuvo **80,16 % de accuracy**, **82,02 % de F1 macro**, **CE 0,6236**, **brecha 12,16 puntos** y **71,80 minutos de fit acumulado**. Se compararán calidad y coste, sin asumir que un LR mayor mejore.

Esta ejecución genera **un `prediction.json` nuevo** a partir de su checkpoint seleccionado, junto a `model.keras`, artefactos y PDF, en su propia carpeta. No copia las predicciones de otra ejecución. La comparación incluye scratch y experiment01/02/03/04/05/06 de TL, en `tl_challenge/comparisons/efficientnet_b0_experiment06/`. El tiempo acumulado suma la cabeza y esta rama, sin añadir otros fine-tunings.

[Protocolo y registro general](../../README.md).
