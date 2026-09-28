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
│   ├── xview_recognition/
│   ├── xview_detection/
│   └── ffNN_example.ipynb
└── models/
```

El notebook `PROJECT/ffNN_example.ipynb` utiliza rutas relativas, por lo que
debe ejecutarse desde la carpeta `PROJECT/`.

## Ejecución y resultados

Al ejecutar `PROJECT/ffNN_example.ipynb`, el experimento genera los archivos
`model.keras` y `prediction.json` dentro de `PROJECT/`.

Cuando termines un experimento:

1. Crea una carpeta dentro de `models/` con un nombre que identifique el
   experimento, por ejemplo `models/01_2026-09-28/`.
2. Mueve `PROJECT/model.keras` y `PROJECT/prediction.json` a esa carpeta.
3. Añade, si es necesario, una descripción de la configuración y los
   resultados del experimento en un archivo `description.txt`.

De esta forma, cada entrenamiento queda almacenado de manera independiente y
los archivos generados por el siguiente experimento no sobrescriben los
resultados anteriores.
