# EfficientNetB0 · transfer learning · experiment02

Ejecutar [efficientnet_b0_tl_experiment02.ipynb](efficientnet_b0_tl_experiment02.ipynb) desde un kernel nuevo, de arriba abajo.

Etapa: **fine-tuning de últimos bloques; BatchNorm congelada**. Entrada 128×128, batch 32, AdamW, weight decay 1e-4, sin MixUp, label smoothing ni Dropout adicional. LR 1e-5, máximo 180 adicionales épocas, dentro del máximo común de 200 entre ambas etapas.

**Ejecutar primero experiment01 de esta arquitectura y conservar su model.keras y artifacts/.** Si el fine-tuning no supera su accuracy, se conserva el checkpoint del extractor.

Las rutas ya apuntan aquí. Se guardan model.keras, prediction.json obligatorio, artefactos y PDF. La sección 9 compara con la misma arquitectura desde cero y el extractor previo; el coste de fine-tuning incluye la cabeza.

[Orden completo, protocolo y registro](../../README.md). Estado: **preparado; entrenamiento pendiente**.

Corrección de compatibilidad con TensorFlow 2.15 y GPU determinista: las BN del checkpoint se sustituyen por `FrozenBatchNormalization`, conservando los pesos y las estadísticas congeladas y evitando el kernel de gradiente fusionado no implementado. La clase se registra en la primera celda. Reiniciar el kernel y ejecutar todo el notebook después de actualizarlo. No es necesario repetir experiment01. Detalles y verificación en el [README general](../../README.md).
