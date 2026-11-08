# MobileNetV2 · transfer learning · experiment01

Ejecutar [mobilenet_v2_tl_experiment01.ipynb](mobilenet_v2_tl_experiment01.ipynb) desde un kernel nuevo, de arriba abajo.

Etapa: **backbone ImageNet congelado; solo cabeza GAP + Dense13**. Entrada 128×128, batch 32, AdamW, weight decay 1e-4, sin MixUp, label smoothing ni Dropout adicional. LR 1e-3, máximo 20 épocas, dentro del máximo común de 200 entre ambas etapas.

La primera ejecución descarga los pesos ImageNet en tl_challenge/.weights/.

Las rutas ya apuntan aquí. Se guardan model.keras, prediction.json obligatorio, artefactos y PDF. La sección 9 compara con la misma arquitectura desde cero; el coste de fine-tuning incluye la cabeza.

[Orden completo, protocolo y registro](../../README.md). Estado: **preparado; entrenamiento pendiente**.
