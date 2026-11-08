# EfficientNetB0 · transfer learning · experiment04

[Notebook](efficientnet_b0_tl_experiment04.ipynb). **Preparado; entrenamiento pendiente.** Reiniciar el kernel y ejecutar todas sus celdas en orden.

**Único cambio respecto a experiment03: fine-tuning de todo el backbone salvo BatchNorm**, en lugar de limitarse a `block6`, `block7` y `top_`. Las BatchNorm mantienen congelados gamma, beta y estadísticas, con `FrozenBatchNormalization` para GPU determinista. Se conserva la llamada al backbone con `training=False`, que permite gradientes en las convoluciones pero mantiene desactivadas sus operaciones estocásticas internas.

Se vuelve a cargar el mejor **experiment01**, con optimizador nuevo. Conservar su `model.keras` y `artifacts/`; no hay que repetir la cabeza ni continuar desde experiment03. Mismos pesos iniciales que las otras ramas de fine-tuning.

Se mantienen 128×128, batch 32, AdamW, LR 1e-5, weight decay 1e-4, MixUp alpha=0.2, Categorical Cross Entropy, sin label smoothing ni Dropout adicional, semilla 42, giros/flips, reparto y callbacks. Hasta 180 épocas de fine-tuning, máximo 200 contando la cabeza en esta rama independiente. Entrenar todos los bloques puede aumentar el tiempo por época y el uso de memoria; se conserva batch 32 para aislar el cambio.

MixUp solo se aplica a train, después de giros y flips. La brecha final se calcula sobre train y validación sin mezclas ni aumentación. La accuracy de train del historial con etiquetas mezcladas no equivale a la accuracy limpia.

Referencias: experiment03 alcanzó **76,96 % de accuracy**, **79,25 % de F1 macro**, **CE 0,7359** y **brecha 14,98 puntos**. EfficientNetB0 desde cero obtuvo **77,97 % de accuracy** y **78,40 % de F1 macro**. La hipótesis de que las capas iniciales congeladas limitan la adaptación sigue pendiente de comprobar; descongelarlas también podría aumentar el sobreajuste.

Se conservan caché propia en disco, preprocesamiento ImageNet, mejor checkpoint, artefactos, PDF y generación obligatoria de `prediction.json`. Si no se supera al extractor se conserva su checkpoint. Comparación con scratch y experiment01/02/03/04 de TL, en `tl_challenge/comparisons/efficientnet_b0_experiment04/`. El coste acumulado suma la cabeza y esta rama; no suma experiment02 ni experiment03.

[Protocolo y registro general](../../README.md).
