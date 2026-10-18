# Presentación — Regularization Challenge

`presentacion.tex` reutiliza el tema Madrid/whale y la autoría de `../fnn_challenge/presentacion.tex`, con contenido específico de regularización. `presentacion.pdf` es una primera versión revisable.

Las diapositivas compilan con paneles de «gráfica pendiente» mientras no existan los PNG. Para completarlas:

1. Ejecutar `PROJECT/experiments/reg_challenge/entrenamiento final/graficas_modelo_final.ipynb` desde un kernel nuevo. Usa el checkpoint de Dropout 0.1 del experimento 10 y sus historiales; **no reentrena**.
2. El notebook escribe imágenes PNG/PDF directamente en `presentaciones/reg_challenge/images/`, y métricas auxiliares en `entrenamiento final/`.
3. Recompilar desde esta carpeta:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error presentacion.tex
```

Gráficas esperadas en las diapositivas: `training_curves.png`, `controlled_comparison.png`, `confusion_matrix.png`, `precision_recall.png`. También se generan `learning_rate`, `validation_loss` y `class_recall` como material de apoyo.

La presentación distingue los resultados históricos sin semilla global de las comparaciones controladas con semilla 42. Las métricas y conclusiones proceden de `PROJECT/experiments/reg_challenge/summary.md` y los JSON de resultados. Las curvas AP están pendientes de evaluación; no se inventan valores.
