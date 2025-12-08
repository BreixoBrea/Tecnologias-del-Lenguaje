El entregable contiene:
- Una carpeta archivos_cesga, con el script que compara los embeddings por rasgo con los posts originales (limpios y reestructurados), así como el .slurm utilizado para lanzar el trabajo.
- Una carpeta baseline_media con el baseline inicial que imputa la media de cada rasgo de authors_train a los autores de authors_test.
- Una carpeta embeddings_neo_ffi con el notebook con el que se calculan los embeddings del cuestionario Neo-FFI con los que se compararán los posts en el CESGA.
- Una carpeta predicciones, con los archivos csv que contienen las predicciones entregadas.
- Un notebook ensemble_regresores, que detalla el procedimiento para ajustar varios modelos de regresión y seleccionar los mejores por rasgo, basándonos en los resultados MSE de validación para crear el ensemble.
- Un notebook finetuneT5_mejorado, con las pruebas y el ajuste fino del modelo de Google T5, aunque no fue satisfactoria.
- Un notebook limpar_y_reestructurar, que contiene la primera limpieza del archivo posts.csv original, para prepararlo de manera que sea posible realizar las comparaciones de cada post con los embeddings por rasgo del cuestionario Neo-FFI.
- Un notebook segunda_limpieza, que se realiza tras la ejecución del script reduccion_csv_por_rasgo en el CESGA, y que hace una segunda limpieza basada en un umbral de similitud (0.3 y 0.4), por lo que obtiene los dos archivos con los que se entrenarán los modelos regresores.