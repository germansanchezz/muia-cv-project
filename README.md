# MUIA Computer Vision Project

## Datasets

Los datasets no se incluyen en este repositorio debido al tamaño de sus
archivos. Descárgalos manualmente desde los siguientes enlaces:

- [xview_recognition](https://drive.upm.es/s/2DDPE2zHw5dbM3G)
- [xview_detection](https://drive.upm.es/s/P7nEf3Bygns7tbM)

Después de descargarlos, descomprime ambos datasets y coloca sus carpetas
dentro de `PROJECT/`. La estructura mínima esperada es:

```text
.
├── PROJECT/
│   ├── common/
│   │   └── ffnn_challenge_utils.py
│   ├── experiments/
│   │   └── fnn_challenge/
│   │       ├── ffnn_challenge_baseline.ipynb
│   │       └── experiment_01/
│   │           ├── ffnn_01.ipynb
│   │           ├── model.keras
│   │           ├── prediction.json
│   │           └── description.txt
│   └── xview_recognition/
└── models/
```

Los notebooks de `PROJECT/experiments/fnn_challenge/` importan las funciones
comunes desde `PROJECT/common/` y localizan el repositorio automáticamente.
No dependen de ejecutar el notebook desde una carpeta concreta.

## Ejecución y resultados

Cada experimento tiene su propio notebook y su propia carpeta de resultados.
Por ejemplo, `ffnn_01.ipynb` guarda `model.keras` y `prediction.json` dentro de
`PROJECT/experiments/fnn_challenge/experiment_01/`.

> **Warning:** Es muy importante cambiar `EXPERIMENT_DIR` en la primera celda
> del notebook antes de ejecutar cada experimento. La ruta debe apuntar a la
> carpeta del experimento actual para no sobrescribir los resultados de otra
> prueba.

### Preparar y ejecutar un experimento

1. Crea una carpeta dentro de `PROJECT/experiments/fnn_challenge/` con un
   nombre que identifique el experimento, por ejemplo `experiment_02/`.
2. Copia `ffnn_challenge_baseline.ipynb` dentro de esa carpeta y renómbralo.
3. Cambia `EXPERIMENT_DIR` en la primera celda para que apunte a la nueva
   carpeta, ajusta la configuración del experimento y ejecuta el notebook.

### Después de ejecutar

1. Comprueba que se han generado `model.keras` y `prediction.json` dentro de
   la carpeta del experimento.
2. Añade o completa `description.txt` con la arquitectura, los parámetros de
   entrenamiento y los resultados obtenidos.

De esta forma, cada entrenamiento queda almacenado de manera independiente y
los archivos generados por el siguiente experimento no sobrescriben los
resultados anteriores. La carpeta `models/` conserva los resultados históricos
que ya existían antes de esta reorganización.
