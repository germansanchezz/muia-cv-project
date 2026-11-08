# Guion · CNN Challenge y TL Challenge · 10 minutos

La exposición empieza directamente por el trabajo realizado, sin reintroducir xView ni el objetivo general del proyecto. Se distinguen los dos challenges. Accuracy de validación es el criterio principal y F1 macro lo complementa; CE sirve para leer las curvas. Las referencias a experimentos se interpretan dentro de cada familia: populares, custom o TL.

## 1 · Portada — 15 s

Vamos a contar el recorrido de los experimentos: primero las redes populares desde cero, después la arquitectura propia y, por último, transfer learning con EfficientNetB0. En cada paso veremos qué cambiamos, si mejoró la accuracy y qué configuración conservamos.

## 2 · CNN: punto de partida y selección — 55 s

Empezamos entrenando ResNet18, DenseNet121, EfficientNetB0 y MobileNetV2 desde cero. Usamos inicialmente 128 píxeles, batch 32, AdamW, learning rate 0,001 y weight decay 0,0001. Conservamos el reparto, la semilla, las rotaciones y los volteos. Las curvas son los historiales originales, con el checkpoint seleccionado marcado por un punto.

Las cuatro quedan cerca en accuracy: DenseNet alcanza 79,31 por ciento; ResNet, 78,03; B0, 77,97, y MobileNet, 77,60. No seguimos simplemente la clasificación de accuracy. DenseNet tarda unos 96 minutos; ResNet, catorce. Elegimos ResNet por eficiencia y B0 por sus resultados competitivos y menor sobreajuste inicial. Ese diagnóstico explica la selección histórica, pero no sustituye a accuracy como objetivo. En ResNet se configuraron cien épocas por error, pero se detuvo en la 71 y el límite no recortó la ejecución.

## 3 · CNN: ajustes de populares — 60 s

En el experimento 02 probamos dos cambios independientes: aumentar weight decay a 0,001 en ResNet y bajar el learning rate a 0,0003 en B0. No mejoraron: ResNet mantuvo 78,03 por ciento y B0 bajó a 76,75. Descartamos esos cambios.

Para el experimento 03 volvimos a las configuraciones del 01 y añadimos únicamente label smoothing de 0,1. Esta vez mejoraron ambas: ResNet subió a 79,73 y B0 a 79,09. No estamos combinando el mayor weight decay ni el learning rate reducido del experimento anterior.

La última prueba popular fue ResNet a 192 píxeles, conservando smoothing 0,1 y los demás ajustes. Alcanzó 80,11 y se convirtió en la mejor popular. Cerramos esa parte con ResNet exp04 como referencia. Cada una de estas ejecuciones empieza desde pesos aleatorios; cambiar la configuración no significa continuar el entrenamiento anterior.

## 4 · CNN: pasamos a la custom — 45 s

Después de terminar las populares construimos una CNN propia. Es una residual compacta con ocho bloques distribuidos en canales 32, 64, 128 y 256. Cada bloque incorpora Squeeze-and-Excitation, que pondera canales a partir de información global. Terminamos con global average pooling y trece salidas.

Tiene 2,86 millones de parámetros, frente a los 11,19 de ResNet18. El punto de partida usa los mismos hiperparámetros iniciales de las populares: 128 píxeles, AdamW y sin suavizado, Dropout adicional ni preentrenamiento. El primer entrenamiento obtiene 79,15 por ciento. A partir de esa arquitectura mantenemos los bloques y exploramos cambios de entrenamiento.

## 5 · CNN custom: smoothing o MixUp — 50 s

Las dos primeras modificaciones parten de la configuración de la custom inicial. En el experimento 02 añadimos smoothing de 0,1. La accuracy queda en 79,04, ligeramente por debajo de 79,15, y no lo conservamos como configuración ganadora.

En el experimento 03 volvemos a quitar smoothing y probamos MixUp de 0,2, que mezcla imágenes y etiquetas solo durante entrenamiento. Mantenemos arquitectura, resolución, optimizador, learning rate y weight decay. La accuracy sube a 80,69: gana 1,55 puntos a la referencia y supera ligeramente a la mejor ResNet de esta validación. Conservamos MixUp 0,2 como base de las siguientes pruebas. Las curvas que mostramos son de validación, por lo que las etiquetas no están mezcladas.

## 6 · CNN custom: últimas pruebas y cierre — 45 s

Sobre la configuración ganadora probamos dos cosas. En el experimento 04 aumentamos MixUp a 0,4, manteniendo 128 píxeles. La pérdida mejora, pero la accuracy baja a 80,16, diez aciertos menos. Si el objetivo principal son los aciertos, mantenemos 0,2.

En el experimento 05 volvemos a MixUp 0,2 y cambiamos solo la entrada a 192 píxeles. La accuracy baja a 79,47, de modo que tampoco conservamos la resolución mayor. Cerramos la fase custom con el experimento 03: arquitectura residual-SE, 128 píxeles y MixUp 0,2. El aumento de resolución que ayudó a ResNet no ha ayudado a esta custom.

## 7 · TL: por qué B0 y cuál es la referencia — 55 s

Pasamos al challenge de transfer learning y nos centramos en EfficientNetB0. Había dado buenos resultados en la fase CNN y Keras Applications ofrece esa arquitectura con pesos ImageNet. En cambio, nuestra ResNet18 no está incluida en ese catálogo; esto no significa que Keras no tenga otras variantes ResNet.

La referencia que queremos superar es el B0 inicial desde cero, con 77,97 por ciento. Comenzamos con un extractor ImageNet congelado y una cabeza GAP más Dense13. En el experimento 01 solo entrenamos esa cabeza durante veinte épocas, a 128 píxeles y LR 0,001. Alcanza 70,72: el extractor fijo todavía no supera al entrenamiento desde cero.

Las siguientes pruebas arrancan del mismo mejor checkpoint de este experimento 01, con optimizador nuevo y hasta 180 épocas adicionales. Las BatchNorm se mantienen congeladas. No vamos encadenando los pesos de cada experimento con el siguiente.

## 8 · TL: adaptar más capas y añadir MixUp — 65 s

En el experimento 02 entrenamos la cabeza y los últimos bloques de B0, con LR 0,00001. Conservamos entrada 128, batch 32 y weight decay 0,0001. La accuracy sube de 70,72 a 75,20, pero sigue por debajo de scratch.

En el experimento 03 mantenemos ese ajuste parcial y añadimos MixUp 0,2. Pasamos a 76,96. Como mejora, conservamos la mezcla. Para el experimento 04 permitimos entrenar todo el backbone salvo las BatchNorm. Se mantienen MixUp, resolución, learning rate y demás hiperparámetros. Llegamos a 77,97: igualamos la accuracy de B0 desde cero y mejoramos su F1 macro.

Las curvas permiten ver las trayectorias de las tres configuraciones. Sus épocas son locales de cada prueba y comienzan en uno. Las mejoras apoyan adaptar más la representación y conservar MixUp; no se explican por haber continuado acumulando épocas entre experimentos.

## 9 · TL: resolución, LR y última regularización — 65 s

En el experimento 05 cambiamos la resolución a 192, manteniendo el fine-tuning completo salvo BN, MixUp 0,2 y LR 0,00001. Se adapta la entrada del mismo checkpoint inicial sin cambiar sus pesos de partida. Subimos a 80,16: ahora sí superamos claramente al B0 desde cero.

El experimento 06 conserva esa configuración y aumenta únicamente el LR inicial a 0,00003. La accuracy mejora a 80,91 y F1 macro a 83,18. Es nuestro mejor resultado por esas medidas.

Como última prueba, el experimento 07 mantiene ese LR y 192 píxeles, pero sube MixUp a 0,4. La accuracy queda en 80,80: son dos aciertos menos que en el 06. Tiene menor pérdida, pero no lo presentamos como una mejora de accuracy. Conservamos ambos finalistas: el 06 por accuracy y F1 y el 07 como alternativa cercana con menor CE. Esta última comparación cierra la búsqueda.

## 10 · TL: errores del mejor modelo — 45 s

La matriz corresponde al checkpoint del experimento 06, el de mayor accuracy. Las filas son clases reales y las columnas predichas; mostramos porcentajes por fila. Las celdas fuera de la diagonal indican las confusiones.

Truck sigue siendo difícil: 35 ejemplos se confunden con Small car y 30 con Dump truck. También hay 26 Bus clasificados como Truck. Son categorías visualmente parecidas en los recortes y el TL no elimina esa dificultad. El gráfico se calcula directamente desde las probabilidades guardadas del modelo. Para Helipad debemos ser prudentes: solo hay nueve ejemplos de validación.

## 11 · Balance de las fases — 55 s

Ahora reunimos las configuraciones elegidas. La mejor popular desde cero es ResNet18 a 192 con smoothing 0,1. La mejor custom usa 128 y MixUp 0,2, con bastantes menos parámetros. En TL, B0 exp06 alcanza 80,91 frente a 77,97 del mismo B0 desde cero: una mejora de 2,93 puntos.

Es una comparación de procedimientos: al final también han cambiado la resolución y la regularización. No atribuimos toda la ganancia únicamente a los pesos ImageNet. A 128, el TL del experimento 04 ya igualaba la accuracy de scratch. Frente a la mejor custom, la ventaja de exp06 es de cuatro aciertos; sirve para describir esta validación, no para asegurar superioridad universal. Aquí se ve tanto el rendimiento como el tamaño del modelo.

## 12 · Configuraciones que conservamos — 45 s

El recorrido termina con decisiones concretas. En CNN populares conservamos ResNet18 exp04 a 192 con smoothing 0,1. En custom, exp03 a 128 con MixUp 0,2. Los cambios que no mejoraron la accuracy no se acumularon en la siguiente configuración.

Para TL proponemos dos candidatos para Codabench: B0 exp06 y exp07. El primero gana por accuracy y F1; el segundo queda muy cerca con menor pérdida. Cada uno genera un archivo de predicción propio con su mejor checkpoint. Aún no tenemos resultados externos de estos candidatos.

Cerramos los ajustes con una sola semilla y partición. Si se dedicara más entrenamiento, confirmar la estabilidad de los finalistas sería más informativo que seguir acumulando cambios sobre la misma validación.
