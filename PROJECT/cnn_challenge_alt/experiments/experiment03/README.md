# CNN propia · experiment03 · MixUp

Ejecutar [cnn_experiment03.ipynb](cnn_experiment03.ipynb) de arriba abajo; las rutas ya están configuradas.

**Único factor añadido frente a experiment01: MixUp alpha 0.2**, después de giros y flips. Se interpolan imágenes y etiquetas con la misma pareja y lambda por imagen, usando Beta(0.2,0.2). Se mantienen la arquitectura, entrada 128×128, batch 32, AdamW, LR 0.001, weight decay 0.0001, label smoothing 0.0 y máximo 200 épocas. Se entrena desde cero y se reutiliza la caché mmap original.

MixUp solo afecta a train. La accuracy durante fit usa etiquetas blandas y no se compara directamente con train anterior. Validación y test usan imágenes originales; la brecha final se obtiene evaluando train y validación sin mezcla ni aumentación.

La sección 9 selecciona custom experiment03, custom experiment01 (mejor anterior por accuracy) y ResNet18 experiment04 (referencia popular fija). Guarda las tablas y PDF en `comparisons/custom_experiment03/`, incluye `mixup_alpha` en CSV y omite la curva mezclada de accuracy de train en la comparación.

Se generan `model.keras`, obligatoriamente `prediction.json`, `artifacts/` y `plots/`. La explicación completa y la hipótesis están en el [registro general](../README.md).

Estado: **completado**. Mejor checkpoint en la época **97**, de **117** ejecutadas:

- Accuracy de validación: **80,69 %** (1.513/1.875 aciertos), +1,55 puntos frente a custom experiment01.
- F1 macro: **82,27 %**; CE sin suavizado: **0,695**; AP macro: **86,38 %**.
- Accuracy de train sobre imágenes originales, sin MixUp ni aumentación: **98,30 %**.
- **Brecha final comparable: 17,61 puntos porcentuales**. No se usa la accuracy de fit con objetivos mezclados.
- Parámetros: **2.857.125**; entrenamiento: **39,18 minutos**.

Se convierte en la mejor custom por accuracy. Supera a ResNet18 experiment04 en once aciertos; el F1 macro está prácticamente empatado. La brecha frente a custom experiment01 apenas cambia (17,53 → 17,61), aunque mejoran accuracy, F1 y CE.

El [registro general](../README.md) recoge verificación, resultados por clase, costes y propuestas: probar alpha 0.4 como cambio aislado y reservar resolución para el final. El [experiment04](../experiment04/cnn_experiment04.ipynb) ya está preparado con MixUp alpha 0.4 como único cambio; ya se ha evaluado: no mejora accuracy/F1; experiment03 sigue como referencia por accuracy. La prueba de 192×192 está preparada en experiment05. [Métricas guardadas](artifacts/metrics.json).
