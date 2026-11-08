# CNN propia · experiment04 · MixUp alpha 0.4

Ejecutar [cnn_experiment04.ipynb](cnn_experiment04.ipynb) de arriba abajo; las rutas están configuradas para esta carpeta.

Se parte de la mejor custom anterior, **experiment03**. **Único cambio: MixUp alpha 0.2 → 0.4.** Se mantienen la red residual SE (2.857.125 parámetros), 128×128, batch 32, AdamW, LR 0.001, weight decay 0.0001, label smoothing 0.0, máximo 200 épocas y semilla 42. Sin Dropout ni L2 adicionales.

Entrenamiento desde cero. MixUp usa Beta(0.4,0.4) por imagen y se aplica tras los giros/flips, solo en train. Evaluación de train, validación y test siguen con datos originales. La brecha final no usa la accuracy de fit con etiquetas blandas.

La sección 9 selecciona custom experiment04, custom experiment03 y ResNet18 experiment04. Destino: `comparisons/custom_experiment04/`. El mejor modelo, `prediction.json` obligatorio, artefactos y PDF se guardan aquí; se reutiliza la caché mmap de 128×128.

Referencia: **80,69 % de accuracy y 82,27 % de F1 macro** de custom experiment03. Se comprobará si una mezcla más intensa mejora validación, CE/AP y errores de alta confianza; no se presupone la mejora ni se elegirá solo por una brecha menor.

Estado: **completado**. Mejor checkpoint en la época **77**, de **97** ejecutadas.

- Accuracy: **80,16 %**, diez aciertos menos que alpha 0.2 (80,69 %).
- F1 macro: **81,38 %**, frente a 82,27 %.
- CE: **0,646**, mejor que 0,695; AP macro: **86,51 %**, frente a 86,38 %.
- Brecha de train original–validación: **14,38 puntos**, frente a 17,61.
- Entrenamiento registrado: **65,24 minutos**; la diferencia de tiempo por época no se puede atribuir al alpha con estos registros.

**Se elige alpha 0.2 de experiment03 para continuar por accuracy/F1.** Alpha 0.4 queda como alternativa de mejor CE y menor confianza extrema. El [registro general](../README.md) detalla la comparación y la prueba de 192×192 preparada en experiment05 sobre la configuración elegida. [Métricas](artifacts/metrics.json).
