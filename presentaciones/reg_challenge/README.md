# Presentación — Regularization Challenge

`presentacion.tex` reutiliza el tema Madrid/whale y la autoría de `../fnn_challenge/presentacion.tex`, con contenido específico de regularización. `presentacion.pdf` contiene **9 diapositivas** para una exposición de 10 minutos.

Las diapositivas compilan con paneles de «gráfica pendiente» mientras no existan los PNG. Para completarlas:

1. Ejecutar `PROJECT/experiments/reg_challenge/entrenamiento final/graficas_modelo_final.ipynb` desde un kernel nuevo. Usa el checkpoint de Dropout 0.1 del experimento 10 y sus historiales; **no reentrena**.
2. El notebook escribe imágenes PNG/PDF directamente en `presentaciones/reg_challenge/images/`, y métricas auxiliares en `entrenamiento final/`.
3. Recompilar desde esta carpeta:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error presentacion.tex
```

Gráficas usadas en las diapositivas: `training_curves.png` y `confusion_matrix.png`. El resto de imágenes generadas por el notebook queda como material de apoyo, fuera de las 9 diapositivas.

La presentación distingue los resultados históricos sin semilla global de las comparaciones controladas con semilla 42. Las métricas y conclusiones proceden de `PROJECT/experiments/reg_challenge/summary.md` y los JSON de resultados. Las curvas AP quedan como material de apoyo; no se inventan valores.

## Codabench y modelo presentado

Los resultados externos proporcionados por el usuario están conservados en `codabench_results.json`. Se muestran a dos decimales en la presentación. **El experimento 11 (Dropout 0.1 + L2 1e-5) es el publicado en la clasificación**, con 52,39 % de accuracy. El mejor checkpoint por validación local sigue siendo Dropout 0.1 del experimento 10; las curvas y matriz de confusión pertenecen a este, no al 11. No se mezclan los conjuntos de evaluación.

Distribución orientativa de los 10 minutos: portada 15 s; objetivo/modelo 60 s; exploración 75 s; comparación controlada 90 s; curvas 60 s; matriz 60 s; combinación 60 s; Codabench 90 s; conclusiones 60 s. Total aproximado: 9 min 30 s.

Actualización: se añade el envío del challenge ffNN anterior (47,91 % de accuracy, 44,84 % de precisión y 41,15 % de recall). Frente al 11 publicado: +4,48, +11,55 y +6,06 puntos, respectivamente. Se explicita que la comparación entre fases no aísla la regularización. Tablas de las diapositivas 3, 4, 7 y 8 centradas; conclusiones originales restauradas, sin el apartado de hipótesis. Se conservan 9 diapositivas.
