# EfficientNetB0 · transfer learning · experiment03

[Notebook](efficientnet_b0_tl_experiment03.ipynb). **Preparado; entrenamiento pendiente.** Reiniciar el kernel y ejecutar todas las celdas en orden.

Se prueba únicamente **MixUp `alpha=0.2`** frente al fine-tuning de experiment02. Se vuelve a cargar el mejor checkpoint de **experiment01**; no se continúa desde los pesos sobreajustados de experiment02. Conservar `model.keras` y sus artefactos de experiment01.

Se mantienen entrada 128×128, batch 32, AdamW nuevo, LR 1e-5, weight decay 1e-4, Categorical Cross Entropy, sin label smoothing ni Dropout adicional. Se entrenan la cabeza y los bloques `block6`, `block7`, `top_`, con todas las BatchNorm congeladas y la corrección de GPU determinista. Mismos callbacks y semilla 42. Hasta 180 épocas adicionales: máximo 200 contando la cabeza en esta rama independiente.

MixUp se aplica después de giros y flips, con un lambda Beta(0.2, 0.2) por ejemplo y una permutación dentro del lote. Se mezclan imágenes y etiquetas con los mismos coeficientes. No se aplica a validación, train de evaluación ni test. La accuracy del historial de train usa etiquetas mezcladas; la brecha final se calcula con ejemplos originales, sin aumentación.

Referencia a superar: experiment02 obtuvo **75,20 % de accuracy**, **76,76 % de F1 macro**, **CE 1,1581** y **brecha 21,47 puntos**. Scratch alcanzó **77,97 %**. Se evaluarán accuracy y F1 junto con CE y brecha; reducir solo la brecha no demuestra mejorar el modelo.

Se reutiliza la caché en disco de TL y se guardan los resultados en esta carpeta. `prediction.json` se genera siempre al ejecutar el notebook completo. Si no se mejora al extractor, se conserva su checkpoint, igual que en experiment02. La comparación incluye scratch y los tres experimentos TL, con salida en `tl_challenge/comparisons/efficientnet_b0_experiment03/`. El tiempo acumulado suma cabeza y esta rama de fine-tuning, no el experimento02.

[Protocolo y registro general](../../README.md).
