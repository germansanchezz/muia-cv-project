# EfficientNetB0 · transfer learning · experiment05

[Notebook](efficientnet_b0_tl_experiment05.ipynb). **Preparado; entrenamiento pendiente.** Reiniciar el kernel y ejecutar todo en orden.

**Único cambio frente a experiment04: resolución 192×192 en lugar de 128×128.** Se mantienen todo el backbone entrenable salvo BatchNorm, MixUp alpha=0.2, batch 32, AdamW nuevo, LR 1e-5, weight decay 1e-4, sin label smoothing ni Dropout adicional, mismos callbacks, giros/flips, semilla 42 y reparto. Hasta 180 épocas adicionales, máximo 200 contando la cabeza de esta rama.

Se vuelve a cargar el mejor **experiment01** de 128×128 y se adapta la entrada exterior y el backbone a 192×192 conservando exactamente todos sus pesos, incluida la cabeza. No se continúa desde experiment04. Se necesitan `model.keras` y `artifacts/` de experiment01; no hay que repetir la cabeza. Su preprocesamiento ImageNet permanece integrado.

La caché se genera leyendo los TIFF originales a 192×192, con mmap y lotes. Ocupa aproximadamente 7,72 GiB y utiliza una clave propia en `tl_challenge/.cache/`, sin sobrescribir la de 128. Se espera más coste y memoria por lote; se conserva batch 32 para aislar la resolución.

Antes de entrenar se evalúa el extractor adaptado a 192 para fijar el umbral del checkpoint. Se guarda ese modelo como respaldo: si el fine-tuning no lo mejora, se conserva la versión adaptada (`checkpoint_stage=feature_extraction_resized`). La accuracy histórica de 128 no se utiliza como umbral a otra resolución. El origen de los pesos y la época global siguen refiriéndose al extractor original si se conserva.

Referencia principal: experiment04 obtuvo **77,97 % de accuracy**, **79,34 % de F1 macro**, **CE 0,6711** y **brecha 13,40 puntos**. Iguala la accuracy del EfficientNetB0 scratch (77,97 %) y supera su F1 (78,40 %). Se comprobará si conservar más detalle mejora la clasificación; no se presupone una ganancia.

Se generan `model.keras`, `prediction.json` obligatorio, artefactos, métricas limpias y PDF en esta carpeta. MixUp solo se aplica a train. Las comparaciones incluyen scratch y experiment01/02/03/04/05 de TL, en `tl_challenge/comparisons/efficientnet_b0_experiment05/`. El tiempo acumulado suma la cabeza original y esta rama, no los anteriores fine-tunings. La referencia scratch usa 128×128; el efecto de resolución se documenta explícitamente.

[Protocolo y registro general](../../README.md).
