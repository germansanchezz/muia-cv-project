# Modelo final para la presentación

Copia del mejor checkpoint por accuracy: `experiment_10/dropout_01`, Dropout 0.1, época 92. Se copian checkpoint, configuración, historial y métricas; hashes y origen en `provenance.json`.

Ejecutar **graficas_modelo_final.ipynb** con un kernel nuevo. No ejecuta `fit` ni modifica los experimentos. Reproduce el split y preprocesamiento del baseline, evalúa y genera PNG/PDF en `presentaciones/reg_challenge/images/`. También exporta métricas, informe por clase, matriz de conteos y AP en esta carpeta. No se ha ejecutado aún la inferencia para generar las gráficas.

Dependencias: TensorFlow/Keras, Rasterio, NumPy, Matplotlib y scikit-learn, preferiblemente el mismo entorno de los entrenamientos (TensorFlow 2.15.1).
