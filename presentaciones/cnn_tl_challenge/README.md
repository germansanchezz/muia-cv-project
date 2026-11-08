# Presentación · CNN y Transfer Learning

**12 diapositivas para aproximadamente 10 minutos**, Beamer **Madrid / whale**, formato panorámico **16:9** y autoría de las presentaciones anteriores.

Cubre las CNN populares desde cero, la arquitectura propia residual-SE y los siete experimentos TL de EfficientNetB0. MobileNetV2 aparece únicamente en la fase CNN desde cero; no se incluyen sus notebooks TL sin entrenar.

- [PDF listo para presentar](presentacion.pdf).
- [Fuente LaTeX](presentacion.tex).
- [Guion con tiempos](guion.md).
- [Generador de figuras](generar_figuras.py).
- [Procedencia de las métricas](fuentes_resultados.json).

## Hilo de la presentación

Se comienza directamente con los experimentos: no se repite la introducción al proyecto ni al dataset. La exposición se divide mediante secciones y títulos en **CNN Challenge** y **TL Challenge**. En CNN se termina primero el trabajo con populares y después se pasa a la custom. En TL se explica por qué se eligió B0 y cómo cambió el entrenamiento.

Cada grupo muestra **configuración de partida → cambio por experimento → accuracy → decisión**, explicitando qué ajustes se mantienen y cuáles se descartan. Accuracy es la medida principal; F1 y CE la complementan. La brecha no se presenta como métrica central; solo se conserva la mención histórica de menor sobreajuste como uno de los motivos para seleccionar B0 inicialmente.

Las ocho figuras mostradas son curvas reales de las populares iniciales y sus ajustes, curvas de custom01--03 y custom03--05, curvas de TL02--04 y TL05--07, la matriz de confusión de TL06 y una síntesis de accuracy/parámetros entre fases. No se suavizan, rellenan ni concatenan historiales. Las gráficas antiguas se conservan como material auxiliar y no todas se usan en pantalla.

Para TL se precisa que Keras Applications ofrece EfficientNetB0 con pesos ImageNet, pero no **ResNet18**. No se afirma que carezca de todas las variantes ResNet: sí incluye otras, como ResNet50. La disponibilidad se ha comprobado en el entorno TensorFlow/Keras 2.15.1 del proyecto.

## Compilación

Desde esta carpeta:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error presentacion.tex
```

Requiere TeX Live con Beamer, TikZ, Latin Modern y soporte de español. Las figuras PDF están incluidas; para compilar no hacen falta dataset, pesos ni TensorFlow.

Para regenerar las figuras con Python, NumPy y Matplotlib:

```bash
python generar_figuras.py
latexmk -pdf -interaction=nonstopmode -halt-on-error presentacion.tex
```

El generador resuelve las rutas desde su propia ubicación y utiliza solo los artefactos versionados del repositorio. No entrena ni carga modelos o TIFF. Comprueba el máximo de los historiales contra las métricas, y recalcula la matriz de confusión desde las probabilidades. También genera figuras auxiliares que no se muestran todas en el PDF.

## Duración orientativa

| Diapositiva | Contenido | Tiempo |
|---|---|---:|
| 1 | Portada | 15 s |
| 2 | CNN: populares iniciales y selección | 55 s |
| 3 | CNN: experimentos populares 02--04 | 60 s |
| 4 | CNN: arquitectura custom y punto de partida | 45 s |
| 5 | CNN: custom02/03, smoothing y MixUp | 50 s |
| 6 | CNN: custom04/05 y configuración elegida | 45 s |
| 7 | TL: elección de B0 y exp01 congelado | 55 s |
| 8 | TL: exp02--04, capas y MixUp | 65 s |
| 9 | TL: exp05--07, resolución, LR y MixUp | 65 s |
| 10 | TL: matriz del mejor por accuracy | 45 s |
| 11 | Balance entre fases | 55 s |
| 12 | Configuraciones conservadas y candidatos | 45 s |
| **Total** | | **600 s** |

## Interpretación de resultados

Todas las cifras son de validación local. No hay resultados externos de Codabench aportados para estas fases; exp06 y exp07 se presentan como candidatos, sin afirmar que ya se hayan enviado. Se excluyen ensembles exploratorios.

Los detalles del protocolo, la excepción del máximo 100 en ResNet18 inicial (paró antes) y los límites de la comparación están en el guion. La comparación TL final incorpora también resolución y regularización: no aísla únicamente ImageNet. Las diferencias de pocos aciertos entre finalistas no demuestran superioridad general con una sola semilla/partición. No se usa el tiempo variable entre ejecuciones para atribuir una aceleración intrínseca a MixUp.
