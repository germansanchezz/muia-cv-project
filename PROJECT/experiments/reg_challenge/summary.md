# Resumen de reg_challenge

Resultados guardados revisados el **17/10/2026**. No se han ejecutado nuevos entrenamientos para este resumen.

## Configuración de partida

Red densa **256 → 128 → 64 → 13**, ReLU en capas ocultas y Softmax en salida; imágenes RGB **32 × 32**, normalizadas a [0,1] y precargadas. Split fijo: 16.871 muestras de entrenamiento y 1.875 de validación, `random_state=1`. SGD con LR inicial `1e-3`, momentum `0.9`, Nesterov y `clipnorm=1`. Sin CNN ni transfer learning.

Los experimentos 01–07 mantienen batch 8 salvo el 05, máximo 100 épocas y paciencias **5/10** (reducción del LR / EarlyStopping). Desde el 08 se introduce semilla **42**, determinismo, historial y evaluación de ambos conjuntos en inferencia. Cada cambio adicional se indica en la tabla.

## Resultados de validación

Cada fila usa el **mejor checkpoint por accuracy de validación**, no necesariamente el de menor pérdida. Accuracy, recall y precisión se expresan en **%**; recall y precisión son medias macro entre las 13 clases. La última columna indica **mejor época / épocas ejecutadas**.

| Experimento | Cambio o configuración | Accuracy | Recall macro | Precisión macro | Val loss | Época / total |
|---|---|---:|---:|---:|---:|---:|
| [Baseline](reg_challenge_baseline.ipynb) | Sin regularización añadida; batch 8, paciencias 5/10, máximo 100. | 60,853 | 53,368 | 55,068 | 1,309 | 58 / 68 |
| [01](experiment_01/reg_experiment_01.ipynb) | L2 `1e-4` en las tres capas ocultas. | 59,200 | 52,135 | 53,786 | 1,378¹ | 59 / 69 |
| [02](experiment_02/reg_experiment_02.ipynb) | L2 `1e-5`: diez veces más suave que 01. | 59,947 | 52,368 | 55,729 | 1,356¹ | 60 / 70 |
| [03](experiment_03/reg_experiment_03.ipynb) | Dropout `0.2` tras cada capa oculta. | 60,160 | 53,134 | 56,264 | 1,228 | 85 / 95 |
| [04](experiment_04/reg_experiment_04.ipynb) | Dense → BN → ReLU en cada bloque; batch 8. | 55,680 | 51,671 | 50,081 | 1,346 | 45 / 55 |
| [05](experiment_05/reg_experiment_05.ipynb) | Misma red BN del 04; batch **32**. | 56,853 | 51,255 | 53,389 | 1,411 | 45 / 55 |
| [06](experiment_06/reg_experiment_06.ipynb) | Volteos horizontales y verticales antes de Flatten. | 57,600 | 50,085 | 51,623 | 1,316 | 49 / 59 |
| [07](experiment_07/reg_experiment_07.ipynb) | Dropout `0.1`, frente a `0.2` del 03. | 57,707 | 50,724 | 54,001 | 1,290 | 46 / 56 |
| [08 · control](experiment_08/reg_experiment_08.ipynb) | Dropout 0.1, semilla 42, paciencias 5/10; máximo 100. | 60,373 | — | — | 1,258 | 52 / 62 |
| [08 · propuesta](experiment_08/reg_experiment_08.ipynb) | Dropout 0.1, semilla 42; paciencias **10/20**; máximo 100. | 61,013 | 54,137 | 56,636 | 1,268 | 92 / 100 |
| [09 · control](experiment_09/reg_experiment_09.ipynb) | Sin Dropout efectivo; semilla 42, paciencias 10/20; máximo 100. | 59,573 | — | — | 1,292 | 60 / 80 |
| [09 · propuesta](experiment_09/reg_experiment_09.ipynb) | Dropout **0.2**; mismo protocolo que 08 propuesta. | 60,533 | 52,875 | 55,831 | 1,239 | 95 / 100 |
| [10 · control](experiment_10/reg_experiment_10.ipynb) | Sin Dropout efectivo; máximo **200**, resto igual al 09. | 59,573 | — | — | 1,292 | 60 / 80 |
| [10 · Dropout 0.1](experiment_10/reg_experiment_10.ipynb) | Dropout 0.1; máximo **200**, resto igual al 08 propuesta. | 61,013 | 54,137 | 56,636 | 1,268 | 92 / 112 |
| [10 · Dropout 0.2](experiment_10/reg_experiment_10.ipynb) | Dropout 0.2; máximo **200**, resto igual al 09 propuesta. | 60,907 | — | — | 1,231 | 107 / 127 |

¹ En 01 y 02 la pérdida incluye L2; no es directamente equivalente a la entropía cruzada de las otras configuraciones. **—**: métrica por clase no guardada para esa ejecución; no se ha sustituido por la de otro checkpoint. En el 10 las métricas por clase guardadas corresponden al modelo seleccionado, Dropout 0.1.

## Conclusiones breves

- **L2, BN y volteos no superaron al baseline histórico** en sus primeras ejecuciones. Sin semillas compartidas, esas diferencias no aíslan completamente el efecto de la técnica.
- Con semilla y protocolo común, **Dropout 0.1 y 0.2 mejoran al control sin Dropout**: 61,013 % y 60,907 % frente a 59,573 %. La diferencia entre las dos tasas es de **solo dos imágenes**; 0.2 tiene menor pérdida y menor brecha que 0.1.
- Ampliar el máximo de 100 a 200 épocas aporta **siete aciertos** a Dropout 0.2 (+0,373 puntos), ninguno a las otras configuraciones. Todas terminan por EarlyStopping antes de 200: aumentar más el límite sin cambiar el protocolo no aportaría entrenamiento adicional.
- **Helipad sigue con recall 0 %** en las evaluaciones por clase guardadas. La regularización no ha resuelto ese problema. Los resultados de test no se conocen únicamente por generar `prediction.json`.

En el experimento 10, la brecha train-validación **evaluada con Dropout desactivado** es 9,332 puntos sin Dropout, 13,529 con 0.1 y 11,087 con 0.2. Una brecha menor no equivale automáticamente a mejor validación. Las brechas de `fit` de pruebas anteriores no son directamente comparables cuando entrenamiento y validación usan modos diferentes.

## Posibles próximos experimentos

1. **Prioridad: varias semillas compartidas.** Repetir tasas 0 / 0.1 / 0.2 con semillas como 1, 7 y 123, conservando el protocolo del 10. Comparar medias, dispersión y diferencias por semilla; no elegir simplemente la mejor ejecución.
2. **Ubicación de Dropout.** Aplicarlo solo tras las capas de 256 y 128, dejando libre la de 64. Mantener tasa, semilla y entrenamiento; comprobar si mejora el equilibrio frente a aplicarlo en los tres bloques.
3. **Combinación moderada.** Si Dropout confirma su mejora, probar añadir L2 suave (`1e-5`) a esa configuración, cambiando únicamente ese factor. Las pruebas iniciales de L2 no usaban el protocolo controlado actual; la combinación no tiene una mejora garantizada.
4. **Diagnóstico de clases minoritarias.** Revisar errores y ejemplos de Helipad antes de introducir más técnicas. Si se prueban pesos de clase, confirmar que encajan en la entrega y estudiarlos como un cambio separado.

La primera opción aporta la evidencia más útil antes de otra búsqueda de hiperparámetros. Mantener siempre configuración, historial, checkpoint y resultados en carpetas propias; evitar ajustar indefinidamente sobre la misma validación.
