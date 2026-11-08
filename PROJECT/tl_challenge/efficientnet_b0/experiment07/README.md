# EfficientNetB0 · transfer learning · experiment07

[Notebook](efficientnet_b0_tl_experiment07.ipynb). **Completado. Última prueba del afinamiento de EfficientNetB0; se recomienda cerrar esta búsqueda.** Para reproducirlo, reiniciar el kernel y ejecutar todas las celdas en orden.

**Único cambio frente a experiment06: MixUp alpha=0.4 en lugar de 0.2.** Se busca regularizar el fine-tuning con LR 3e-5, que mejoró accuracy/F1 pero aumentó la brecha y empeoró la CE. Una mezcla mayor no garantiza mejorar: podría reducir la accuracy.

Se mantienen 192×192, batch 32, AdamW nuevo, LR inicial 3e-5, weight decay 1e-4, Categorical Cross Entropy, sin label smoothing ni Dropout adicional, todo el backbone entrenable salvo BatchNorm, semilla 42, reparto y giros/flips. Mismos callbacks sobre val_accuracy: checkpoint del mejor, ReduceLROnPlateau paciencia 10/factor 0.5/mínimo 1e-6 y EarlyStopping paciencia 20/restaurar mejores pesos. Hasta 180 épocas adicionales, máximo 200 contando la cabeza de esta rama.

Se parte nuevamente del mejor **experiment01**, no de experiment06. Se reconstruyen la entrada y el backbone a 192×192 copiando exactamente todos los pesos, incluida la cabeza. Se requieren su `model.keras` y `artifacts/`; no repetir la cabeza. Se reutiliza la caché mmap de 192 de TL y la implementación de BN congelada compatible con GPU determinista.

MixUp se aplica solo a train, después de giros/flips: un lambda Beta(0.4, 0.4) por ejemplo y una pareja por permutación dentro del lote, con los mismos coeficientes para imágenes y etiquetas. Validación, train de evaluación y test no reciben mezclas ni aumentación. La brecha final usa métricas limpias, no la accuracy de train del historial con etiquetas mezcladas.

Antes del fit se evalúa el extractor adaptado a 192 para fijar el umbral del checkpoint. Si no se supera, se conserva ese modelo adaptado (`feature_extraction_resized`). Cada ejecución genera **un `prediction.json` nuevo** desde su checkpoint seleccionado, junto a `model.keras`, artefactos y PDF, en esta carpeta.

Referencia principal: experiment06 obtuvo **80,91 % de accuracy**, **83,18 % de F1 macro**, **CE 0,6962**, **brecha 18,69 puntos** y **87,30 minutos de fit acumulado**. Experiment05 obtuvo 80,16 %, F1 82,02 %, CE 0,6236 y brecha 12,16 puntos. Se valorará si la regularización conserva o mejora accuracy/F1 y reduce CE/brecha; reducir solo la brecha no basta para elegir el modelo.

La comparación incluye scratch y experiment01/02/03/04/05/06/07 de TL, en `tl_challenge/comparisons/efficientnet_b0_experiment07/`. El tiempo acumulado suma la cabeza original y esta rama, sin añadir otros fine-tunings.

[Protocolo y registro general](../../README.md).


## Resultados y conclusión

| Métrica del checkpoint seleccionado | Exp05: LR 1e-5, MixUp 0.2 | Exp06: LR 3e-5, MixUp 0.2 | Exp07: LR 3e-5, MixUp 0.4 |
|---|---:|---:|---:|
| Accuracy validación | 80,16 % | **80,91 %** | 80,80 % |
| F1 macro | 82,02 % | **83,18 %** | 82,62 % |
| CE validación | 0,6236 | 0,6962 | **0,6037** |
| AP macro | 86,71 % | 86,25 % | **87,11 %** |
| Accuracy train limpio | 92,32 % | 99,60 % | 90,17 % |
| Brecha train–validación | 12,16 puntos | 18,69 puntos | **9,37 puntos** |
| Mejor época local / global | 43 / 63 | 52 / 72 | 18 / 38 |
| Épocas ejecutadas de fine-tuning / totales | 63 / 83 | 72 / 92 | 38 / 58 |
| Fit acumulado registrado | 71,80 min | 87,30 min | **26,56 min** |

El checkpoint procede de fine-tuning (`selected_from_source=false`); no se ha conservado el extractor como fallback. Los hashes del checkpoint de origen y de los pesos iniciales coinciden con experiment06. La configuración registrada mantiene todos sus ajustes salvo MixUp alpha.

Respecto a experiment06, experiment07 pierde **0,11 puntos de accuracy (dos aciertos de 1.875)** y **0,57 puntos de F1 macro**, pero mejora CE de 0,6962 a **0,6037**, aumenta AP macro en **0,87 puntos** y reduce la brecha en **9,32 puntos**. El train limpio baja de 99,60 % a 90,17 % mientras validación prácticamente se mantiene. Esto es consistente con una regularización útil: se reduce el ajuste excesivo a train sin una pérdida material de accuracy en esta partición. No basta por sí solo para demostrar mejor generalización en datos nuevos, y una CE menor no demuestra por sí sola mejor calibración.

La CE mínima del historial es **0,6007 en la época 31**; la selección por accuracy conserva la **época 18**, con CE 0,6037. Se detiene en la época 38, veinte épocas después del máximo de accuracy. El modelo no termina mejorando sostenidamente: no hay evidencia que justifique alargar este entrenamiento.

Por clase, frente a experiment06 mejoran especialmente Small car (F1 **84,33 → 86,70 %**) y Dump truck (**68,62 → 71,07 %**). Empeoran Truck (**57,27 → 53,24 %**), Fishing vessel (**85,52 → 80,77 %**) y Storage tank (**91,61 → 89,56 %**). Por tanto, la mezcla más intensa no mejora todas las clases. Helipad conserva F1 94,12 %, pero solo hay nueve ejemplos de validación.

El tiempo acumulado registrado baja a **26,56 minutos** (23,98 de fine-tuning más 2,58 de la cabeza), frente a 87,30 en experiment06. Hay menos épocas, pero también cambia el tiempo observado por época, aproximadamente 38 frente a 71 segundos de media. El hardware y las versiones registrados coinciden; los artefactos no explican esa diferencia de ejecución. No se atribuye toda la reducción de tiempo a MixUp ni se presenta como una aceleración intrínseca del método.

Frente a EfficientNetB0 desde cero, experiment07 mejora accuracy **77,97 → 80,80 %** (+2,83 puntos), F1 **78,40 → 82,62 %**, CE **0,7691 → 0,6037** y brecha **10,98 → 9,37 puntos**. Los procedimientos difieren también en resolución, regularización y política de BN, por lo que no se aísla únicamente el efecto de los pesos ImageNet. La mejor custom obtuvo 80,69 % y ResNet18 experiment04 80,11 %; la ventaja en accuracy frente a la custom es de solo dos aciertos.

**Elección y cierre:** experiment07 es la opción preferida por equilibrio entre accuracy, CE, AP y brecha; experiment06 sigue siendo el ganador si el criterio prioritario es accuracy o F1 macro de validación. Se conservan ambos modelos y sus predicciones. No se propone otra configuración de B0/B1/B2: ya existe una secuencia suficiente para explicar el efecto del fine-tuning, MixUp, resolución y LR. Si se dispone de más presupuesto, sería más informativo repetir las configuraciones finalistas con otras semillas que seguir eligiendo hiperparámetros sobre esta misma validación. No se han preparado nuevos entrenamientos.
